/*
 * Pure AVX2 INT4 Matmul - No Scalar Fallback in Hot Loop
 * 
 * Strategy:
 * 1. Pre-expand INT4 weights to INT8 in blocks (amortize cost)
 * 2. Use AVX2 for everything
 * 3. Process 8 rows at a time
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

/* Pre-expanded INT8 representation of INT4 weights */
typedef struct {
    int8_t* weights;       /* Expanded int8 weights */
    float* scales;         /* Per-block scales */
    int rows;
    int cols;
    int block_size;
} int4_expanded_t;

/* Expand INT4 to INT8 using AVX2 */
static void expand_int4_to_int8_avx2(const uint8_t* int4_weights, int8_t* int8_weights,
                                      int num_packed, float scale) {
    const __m256i low_mask = _mm256_set1_epi8(0x0F);
    const __m256i offset = _mm256_set1_epi8(8);
    
    int i = 0;
    /* Process 32 packed bytes = 64 int4 values at a time */
    for (; i + 31 < num_packed; i += 32) {
        /* Load 32 packed bytes */
        __m256i packed = _mm256_loadu_si256((const __m256i*)(int4_weights + i));
        
        /* Extract low nibbles: packed & 0x0F */
        __m256i lo = _mm256_and_si256(packed, low_mask);
        /* Extract high nibbles: (packed >> 4) & 0x0F */
        __m256i hi = _mm256_srli_epi16(packed, 4);
        hi = _mm256_and_si256(hi, low_mask);
        
        /* Subtract 8 to get signed values: -8 to 7 */
        lo = _mm256_sub_epi8(lo, offset);
        hi = _mm256_sub_epi8(hi, offset);
        
        /* Store interleaved or separate - store as 64 consecutive int8 values */
        /* We need to interleave lo and hi */
        __m256i interleaved_lo = _mm256_unpacklo_epi8(lo, hi);
        __m256i interleaved_hi = _mm256_unpackhi_epi8(lo, hi);
        
        /* Store */
        _mm256_storeu_si256((__m256i*)(int8_weights + i * 2), interleaved_lo);
        _mm256_storeu_si256((__m256i*)(int8_weights + i * 2 + 32), interleaved_hi);
    }
    
    /* Scalar fallback for remaining */
    for (; i < num_packed; i++) {
        uint8_t packed = int4_weights[i];
        int8_weights[i * 2] = (int8_t)((packed & 0x0F) - 8);
        int8_weights[i * 2 + 1] = (int8_t)(((packed >> 4) & 0x0F) - 8);
    }
}

/* Create expanded INT4 matrix */
int4_expanded_t* create_int4_expanded(const float* weights, int rows, int cols, int block_size) {
    int4_expanded_t* mat = (int4_expanded_t*)malloc(sizeof(int4_expanded_t));
    mat->rows = rows;
    mat->cols = cols;
    mat->block_size = block_size;
    
    /* Allocate expanded weights (2x size for int8 vs int4) */
    mat->weights = (int8_t*)aligned_malloc(rows * cols, 64);
    
    int num_blocks = (rows * cols + block_size - 1) / block_size;
    mat->scales = (float*)aligned_malloc(num_blocks * sizeof(float), 64);
    
    /* First pass: compute scales and quantize to temporary INT4 */
    uint8_t* temp_int4 = (uint8_t*)malloc((rows * cols + 1) / 2);
    
    for (int block = 0; block < num_blocks; block++) {
        int start = block * block_size;
        int end = start + block_size;
        if (end > rows * cols) end = rows * cols;
        
        /* Find max abs in block */
        float max_abs = 0.0f;
        for (int i = start; i < end; i++) {
            float abs_val = fabsf(weights[i]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        
        mat->scales[block] = (max_abs > 0) ? (max_abs / 7.0f) : 1.0f;
        
        /* Quantize this block to INT4 */
        for (int i = start; i < end; i += 2) {
            float w0 = weights[i];
            float w1 = (i + 1 < end) ? weights[i + 1] : 0.0f;
            
            int q0 = (int)roundf(w0 / mat->scales[block]) + 8;
            int q1 = (int)roundf(w1 / mat->scales[block]) + 8;
            
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            
            temp_int4[i / 2] = (uint8_t)((q1 << 4) | q0);
        }
    }
    
    /* Second pass: expand to INT8 using AVX2 */
    for (int block = 0; block < num_blocks; block++) {
        int start = block * block_size;
        int end = start + block_size;
        if (end > rows * cols) end = rows * cols;
        int packed_start = start / 2;
        int packed_count = (end - start + 1) / 2;
        
        expand_int4_to_int8_avx2(temp_int4 + packed_start, 
                                  mat->weights + start,
                                  packed_count,
                                  mat->scales[block]);
    }
    
    free(temp_int4);
    return mat;
}

void free_int4_expanded(int4_expanded_t* mat) {
    if (!mat) return;
    aligned_free(mat->weights);
    aligned_free(mat->scales);
    free(mat);
}

/* Pure AVX2 INT8 matmul for expanded INT4 */
void matmul_int4_expanded_avx2(const float* A, const int4_expanded_t* B, float* C,
                                int M, int N, int K) {
    const int8_t* w = B->weights;
    const float* s = B->scales;
    const int block_size = B->block_size;
    
    for (int i = 0; i < M; i++) {
        const float* a_row = A + i * K;
        
        /* Process 8 columns at a time */
        for (int j = 0; j < N; j += 8) {
            int j_end = (j + 8 <= N) ? j + 8 : N;
            int j_count = j_end - j;
            
            __m256 sum_vec[8];
            for (int jj = 0; jj < j_count; jj++) {
                sum_vec[jj] = _mm256_setzero_ps();
            }
            
            /* Process K in blocks */
            for (int k_block = 0; k_block < K; k_block += block_size) {
                int k_end = k_block + block_size;
                if (k_end > K) k_end = K;
                
                int block_idx = (j * K + k_block) / block_size;
                __m256 scale_vec = _mm256_set1_ps(s[block_idx]);
                
                /* Process block */
                int k = k_block;
                for (; k + 7 < k_end; k += 8) {
                    __m256 a_vec = _mm256_loadu_ps(a_row + k);
                    
                    for (int jj = 0; jj < j_count; jj++) {
                        /* Load 8 int8 weights */
                        __m128i w8 = _mm_loadl_epi64(
                            (const __m128i*)(w + (j + jj) * K + k));
                        
                        /* Expand int8 to int32 */
                        __m256i w32 = _mm256_cvtepi8_epi32(w8);
                        
                        /* Convert to float */
                        __m256 w_vec = _mm256_cvtepi32_ps(w32);
                        
                        /* Apply scale and multiply-accumulate */
                        w_vec = _mm256_mul_ps(w_vec, scale_vec);
                        sum_vec[jj] = _mm256_fmadd_ps(a_vec, w_vec, sum_vec[jj]);
                    }
                }
                
                /* Handle remainder of block */
                for (; k < k_end; k++) {
                    float a_val = a_row[k];
                    for (int jj = 0; jj < j_count; jj++) {
                        /* Scalar for remainder */
                        int8_t w_val = w[(j + jj) * K + k];
                        float w_float = (float)w_val * s[block_idx];
                        
                        /* Add to sum manually */
                        float current_sum = 0.0f;
                        _mm256_storeu_ps((float*)&current_sum, sum_vec[jj]);
                        current_sum += a_val * w_float;
                        sum_vec[jj] = _mm256_set1_ps(current_sum);
                    }
                }
            }
            
            /* Horizontal sum and store */
            for (int jj = 0; jj < j_count; jj++) {
                float sum_arr[8];
                _mm256_storeu_ps(sum_arr, sum_vec[jj]);
                float sum = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                           sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
                C[i * N + j + jj] = sum;
            }
        }
    }
}
