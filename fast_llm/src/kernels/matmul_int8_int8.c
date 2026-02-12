/*
 * True INT8 x INT8 Matrix Multiplication
 * Uses _mm256_maddubs_epi16 for 2x throughput on AVX2
 * 
 * Precondition: Weights pre-packed and quantized to int8 at load time
 * Input activations quantized on-the-fly or cached
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "dequantized_tensor.h"

#ifdef __AVX2__

/* 
 * Quantize float input to int8
 * Uses per-channel scaling for better accuracy
 */
static inline void quantize_activations_int8(const float* input, int8_t* output, 
                                              int K, float scale) {
    /* Block-wise quantization for better accuracy */
    int blocks = (K + 31) / 32;  /* 32 values per block */
    
    for (int b = 0; b < blocks; b++) {
        int k_start = b * 32;
        int k_end = (k_start + 32 < K) ? k_start + 32 : K;
        
        /* Find max in block for per-block quantization */
        float max_val = 0.0f;
        for (int k = k_start; k < k_end; k++) {
            float abs_val = fabsf(input[k]);
            if (abs_val > max_val) max_val = abs_val;
        }
        
        float block_scale = max_val / 127.0f;
        if (block_scale < 1e-8f) block_scale = 1.0f;
        
        /* Quantize block */
        for (int k = k_start; k < k_end; k++) {
            int q = (int)roundf(input[k] / block_scale);
            if (q > 127) q = 127;
            if (q < -128) q = -128;
            output[k] = (int8_t)q;
        }
    }
    
    (void)scale; /* Not using global scale for now */
}

/*
 * 4x16 INT8 micro-kernel
 * Processes 4 rows x 16 K values using maddubs
 * 
 * Using _mm256_maddubs_epi16:
 * - Input A: uint8 (but we use int8 with offset)
 * - Input B: int8 
 * - Output: 16-bit intermediate, accumulated to 32-bit
 * 
 * For signed int8 x int8, we use trick:
 * a*b = ((a+128)-128)*b = (a+128)*b - 128*b
 */
static inline void int8_kernel_4x16(const int8_t* A_q, 
                                     const int8_t* B0, const int8_t* B1, 
                                     const int8_t* B2, const int8_t* B3,
                                     int K,
                                     float* sums,
                                     const float* scales_B) {
    /* 4 accumulators (32-bit integers) */
    __m256i acc0 = _mm256_setzero_si256();
    __m256i acc1 = _mm256_setzero_si256();
    __m256i acc2 = _mm256_setzero_si256();
    __m256i acc3 = _mm256_setzero_si256();
    
    /* Correction terms for signed multiplication */
    __m256i corr0 = _mm256_setzero_si256();
    __m256i corr1 = _mm256_setzero_si256();
    __m256i corr2 = _mm256_setzero_si256();
    __m256i corr3 = _mm256_setzero_si256();
    
    /* Process K in chunks of 32 (2x16 to fill 256-bit register) */
    for (int k = 0; k < K; k += 32) {
        /* Load 32 int8 values from A (activations) */
        __m256i a = _mm256_loadu_si256((__m256i*)(A_q + k));
        /* Convert to uint8 by adding 128 (for maddubs) */
        __m256i a_u8 = _mm256_add_epi8(a, _mm256_set1_epi8(128));
        
        /* Process each row */
        #define PROCESS_ROW(n, acc, corr) do { \
            __m256i b = _mm256_loadu_si256((__m256i*)(B##n + k)); \
            /* maddubs: (a_u8 * b) pairwise multiply and add to 16-bit */ \
            __m256i prod = _mm256_maddubs_epi16(a_u8, b); \
            /* Expand to 32-bit and accumulate */ \
            __m256i prod_lo = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod)); \
            __m256i prod_hi = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod, 1)); \
            acc = _mm256_add_epi32(acc, prod_lo); \
            acc = _mm256_add_epi32(acc, prod_hi); \
            /* Correction: subtract 128 * b for each element */ \
            __m256i b_lo = _mm256_cvtepi8_epi32(_mm256_castsi256_si128(b)); \
            __m256i b_hi = _mm256_cvtepi8_epi32(_mm256_extracti128_si256(b, 1)); \
            corr = _mm256_add_epi32(corr, b_lo); \
            corr = _mm256_add_epi32(corr, b_hi); \
        } while(0)
        
        PROCESS_ROW(0, acc0, corr0);
        PROCESS_ROW(1, acc1, corr1);
        PROCESS_ROW(2, acc2, corr2);
        PROCESS_ROW(3, acc3, corr3);
        
        #undef PROCESS_ROW
    }
    
    /* Apply correction: result = acc - 128 * corr */
    __m256i corr_val = _mm256_set1_epi32(128);
    acc0 = _mm256_sub_epi32(acc0, _mm256_mullo_epi32(corr0, corr_val));
    acc1 = _mm256_sub_epi32(acc1, _mm256_mullo_epi32(corr1, corr_val));
    acc2 = _mm256_sub_epi32(acc2, _mm256_mullo_epi32(corr2, corr_val));
    acc3 = _mm256_sub_epi32(acc3, _mm256_mullo_epi32(corr3, corr_val));
    
    /* Horizontal sum of 8 int32 values */
    /* Actually compute sums properly */
    int sums_i[4];
    {
        __m128i lo = _mm256_castsi256_si128(acc0);
        __m128i hi = _mm256_extracti128_si256(acc0, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_hadd_epi32(lo, lo);
        lo = _mm_hadd_epi32(lo, lo);
        sums_i[0] = _mm_cvtsi128_si32(lo);
    }
    {
        __m128i lo = _mm256_castsi256_si128(acc1);
        __m128i hi = _mm256_extracti128_si256(acc1, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_hadd_epi32(lo, lo);
        lo = _mm_hadd_epi32(lo, lo);
        sums_i[1] = _mm_cvtsi128_si32(lo);
    }
    {
        __m128i lo = _mm256_castsi256_si128(acc2);
        __m128i hi = _mm256_extracti128_si256(acc2, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_hadd_epi32(lo, lo);
        lo = _mm_hadd_epi32(lo, lo);
        sums_i[2] = _mm_cvtsi128_si32(lo);
    }
    {
        __m128i lo = _mm256_castsi256_si128(acc3);
        __m128i hi = _mm256_extracti128_si256(acc3, 1);
        lo = _mm_add_epi32(lo, hi);
        lo = _mm_hadd_epi32(lo, lo);
        lo = _mm_hadd_epi32(lo, lo);
        sums_i[3] = _mm_cvtsi128_si32(lo);
    }
    
    /* Convert to float with scales */
    for (int i = 0; i < 4; i++) {
        sums[i] = (float)sums_i[i] * scales_B[i] * (1.0f / 127.0f);
    }
}

/* 
 * INT8 x INT8 matmul with pre-quantized weights
 */
void matmul_int8_int8(const int8_t* A_q,  /* [K] quantized activations */
                      const int8_t* B_q,  /* [N*K] quantized weights */
                      const float* scales_B, /* [N] weight scales */
                      float* C,           /* [N] output */
                      int M, int N, int K) {
    (void)M;
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int n = 0; n <= N - 4; n += 4) {
        float sums[4] = {0};
        
        int8_kernel_4x16(A_q,
                         B_q + (n+0)*K,
                         B_q + (n+1)*K,
                         B_q + (n+2)*K,
                         B_q + (n+3)*K,
                         K,
                         sums,
                         scales_B + n);
        
        for (int i = 0; i < 4; i++) {
            C[n + i] = sums[i];
        }
    }
    
    /* Handle remaining rows */
    int n_rem = (N / 4) * 4;
    for (int n = n_rem; n < N; n++) {
        const int8_t* B_row = B_q + n * K;
        float scale = scales_B[n] * (1.0f / 127.0f);
        int sum = 0;
        for (int k = 0; k < K; k++) {
            sum += (int)A_q[k] * (int)B_row[k];
        }
        C[n] = (float)sum * scale;
    }
}

/* 
 * Wrapper: Float input -> INT8 matmul -> Float output
 */
void matmul_dequantized_int8xint8(const float* A, const dequantized_tensor_t* B,
                                   float* C, int M, int N, int K) {
    (void)M;
    
    /* Quantize activations */
    int8_t* A_q = (int8_t*)aligned_malloc(K + 32, 32);
    quantize_activations_int8(A, A_q, K, 0.01f);
    
    /* INT8 matmul */
    matmul_int8_int8(A_q, B->weights, B->scales, C, 1, N, K);
    
    aligned_free(A_q);
}

#else /* No AVX2 */

void matmul_dequantized_int8xint8(const float* A, const dequantized_tensor_t* B,
                                   float* C, int M, int N, int K) {
    /* Fallback to standard */
    extern void matmul_dequantized(const float*, const dequantized_tensor_t*, float*, int, int, int);
    matmul_dequantized(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
