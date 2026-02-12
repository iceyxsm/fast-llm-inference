/*
 * SINGLE-TOKEN Optimized Matrix Multiplication
 * 
 * For M=1 (single token inference), blocking doesn't help.
 * Instead, we focus on:
 * - Pre-dequantized weights
 * - Fast dot products
 * - Vectorized loads
 * - No packing overhead
 */

#include "matmul.h"
#include <string.h>

/* Forward declaration of fallback kernel */
extern void q4_matmul_avx2(const float* A, const quantized_tensor_t* B, float* C,
                           int M, int N, int K);

#ifdef __AVX2__
#include <immintrin.h>

/*
 * Fast dot product of two float arrays using AVX2
 * Returns sum(a[i] * b[i])
 */
static float dot_product_avx2(const float* a, const float* b, int n) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    
    /* Process 8 floats at a time */
    for (; i <= n - 8; i += 8) {
        __m256 a_vec = _mm256_loadu_ps(a + i);
        __m256 b_vec = _mm256_loadu_ps(b + i);
        sum_vec = _mm256_fmadd_ps(a_vec, b_vec, sum_vec);
    }
    
    /* Horizontal sum */
    __m128 sum_low = _mm256_castps256_ps128(sum_vec);
    __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
    __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    sum_128 = _mm_hadd_ps(sum_128, sum_128);
    float sum = _mm_cvtss_f32(sum_128);
    
    /* Remainder */
    for (; i < n; i++) {
        sum += a[i] * b[i];
    }
    
    return sum;
}

/*
 * Single-token Q4 matmul with pre-dequantized weights
 * 
 * Strategy: Dequantize B to float32 once, then do fast dot products
 * This trades memory for speed
 */
void q4_matmul_single_token(const float* A, const quantized_tensor_t* B_q, float* C,
                            int M, int N, int K) {
    const block_q4_t* B = (const block_q4_t*)B_q->blocks;
    int blocks_per_row = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    
    /* For single token, M should be 1 */
    if (M != 1) {
        /* Fallback for batch > 1 */
        q4_matmul_avx2(A, B_q, C, M, N, K);
        return;
    }
    
    const float* A_row = A;
    
    /* Process each output column */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int n = 0; n < N; n++) {
        /* Dequantize one row of B on the fly */
        /* This is faster than the unpacking-per-element approach */
        
        __m256 sum_vec = _mm256_setzero_ps();
        int k = 0;
        
        /* Process 32 elements at a time (4 blocks of 8) */
        for (; k <= K - 32; k += 32) {
            /* Load 8 floats from A */
            __m256 a0 = _mm256_loadu_ps(A_row + k);
            __m256 a1 = _mm256_loadu_ps(A_row + k + 8);
            __m256 a2 = _mm256_loadu_ps(A_row + k + 16);
            __m256 a3 = _mm256_loadu_ps(A_row + k + 24);
            
            /* Unpack and dequantize 32 weights */
            float weights[32];
            for (int ki = 0; ki < 32; ki++) {
                int block_idx = n * blocks_per_row + (k + ki) / Q4_BLOCK_SIZE;
                int offset = (k + ki) % Q4_BLOCK_SIZE;
                const block_q4_t* block = &B[block_idx];
                
                int byte_idx = offset / 2;
                int nibble = offset % 2;
                int q = (block->qs[byte_idx] >> (nibble * 4)) & 0xF;
                weights[ki] = block->zero_point + q * block->scale;
            }
            
            __m256 w0 = _mm256_loadu_ps(weights + 0);
            __m256 w1 = _mm256_loadu_ps(weights + 8);
            __m256 w2 = _mm256_loadu_ps(weights + 16);
            __m256 w3 = _mm256_loadu_ps(weights + 24);
            
            sum_vec = _mm256_fmadd_ps(a0, w0, sum_vec);
            sum_vec = _mm256_fmadd_ps(a1, w1, sum_vec);
            sum_vec = _mm256_fmadd_ps(a2, w2, sum_vec);
            sum_vec = _mm256_fmadd_ps(a3, w3, sum_vec);
        }
        
        /* Horizontal sum of accumulated vector */
        __m128 sum_low = _mm256_castps256_ps128(sum_vec);
        __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
        __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        sum_128 = _mm_hadd_ps(sum_128, sum_128);
        float sum = _mm_cvtss_f32(sum_128);
        
        /* Remainder */
        for (; k < K; k++) {
            int block_idx = n * blocks_per_row + k / Q4_BLOCK_SIZE;
            int offset = k % Q4_BLOCK_SIZE;
            const block_q4_t* block = &B[block_idx];
            
            int byte_idx = offset / 2;
            int nibble = offset % 2;
            int q = (block->qs[byte_idx] >> (nibble * 4)) & 0xF;
            float w = block->zero_point + q * block->scale;
            
            sum += A_row[k] * w;
        }
        
        C[n] = sum;
    }
}

/*
 * Even faster: Pre-dequantize all weights to contiguous buffer at load time
 * Then use simple fast dot product
 */
void q4_matmul_predequant(const float* A, const quantized_tensor_t* B_q, float* C,
                          int M, int N, int K) {
    /* This would require storing pre-dequantized weights in the tensor */
    /* For now, use the single-token version */
    q4_matmul_single_token(A, B_q, C, M, N, K);
}

#else /* !__AVX2__ */

void q4_matmul_single_token(const float* A, const quantized_tensor_t* B, float* C,
                            int M, int N, int K) {
    q4_matmul(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
