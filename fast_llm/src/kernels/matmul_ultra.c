/*
 * Ultra-Optimized Matmul Kernel
 * Based on llama.cpp/llamafile techniques
 * 
 * Key optimizations:
 * 1. 4x2 register blocking (4 accumulators x 2 K blocks)
 * 2. Aggressive prefetching
 * 3. Minimize data movement
 * 4. Process 2 output rows at once
 * 5. Interleave loads and compute (software pipelining)
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
 * Process 2 output rows at once
 * Each row processes 32 K values (1 cache line)
 * Total: 2 accumulators, each with 4 partial sums for ILP
 * 
 * Layout:
 * - acc0, acc1: accumulators for row 0
 * - acc2, acc3: accumulators for row 1
 */
static inline void matmul_2x32(const float* A, 
                                const int8_t* B0, const int8_t* B1,
                                float* sum0, float* sum1,
                                int K, float scale_b0, float scale_b1) {
    
    /* 4 accumulators for ILP (2 per row) */
    __m256 acc00 = _mm256_setzero_ps();  /* Row 0, part 0 */
    __m256 acc01 = _mm256_setzero_ps();  /* Row 0, part 1 */
    __m256 acc10 = _mm256_setzero_ps();  /* Row 1, part 0 */
    __m256 acc11 = _mm256_setzero_ps();  /* Row 1, part 1 */
    
    /* Process 64 K values at a time (interleaved) */
    int k = 0;
    for (; k <= K - 64; k += 64) {
        /* Prefetch next iteration */
        _mm_prefetch((const char*)(A + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(B0 + k + 128), _MM_HINT_T0);
        _mm_prefetch((const char*)(B1 + k + 128), _MM_HINT_T0);
        
        /* Load 16 floats from A (will be reused for both rows) */
        __m256 a0 = _mm256_loadu_ps(A + k);
        __m256 a1 = _mm256_loadu_ps(A + k + 8);
        __m256 a2 = _mm256_loadu_ps(A + k + 16);
        __m256 a3 = _mm256_loadu_ps(A + k + 24);
        __m256 a4 = _mm256_loadu_ps(A + k + 32);
        __m256 a5 = _mm256_loadu_ps(A + k + 40);
        __m256 a6 = _mm256_loadu_ps(A + k + 48);
        __m256 a7 = _mm256_loadu_ps(A + k + 56);
        
        /* Row 0: Load 64 int8s and dequantize */
        /* First 32 */
        __m256i b0_0 = _mm256_loadu_si256((__m256i*)(B0 + k));
        __m256i b0_1 = _mm256_loadu_si256((__m256i*)(B0 + k + 32));
        
        /* Dequantize first 16 of first 32 */
        __m256i b0_00 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b0_0));
        __m256i b0_01 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b0_0, 1));
        __m256 b0_f00 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_00)));
        __m256 b0_f01 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_00, 1)));
        __m256 b0_f02 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_01)));
        __m256 b0_f03 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_01, 1)));
        
        acc00 = _mm256_fmadd_ps(a0, b0_f00, acc00);
        acc01 = _mm256_fmadd_ps(a1, b0_f01, acc01);
        acc00 = _mm256_fmadd_ps(a2, b0_f02, acc00);
        acc01 = _mm256_fmadd_ps(a3, b0_f03, acc01);
        
        /* Dequantize second 16 of first 32 */
        __m256i b0_10 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b0_1));
        __m256i b0_11 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b0_1, 1));
        __m256 b0_f10 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_10)));
        __m256 b0_f11 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_10, 1)));
        __m256 b0_f12 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_11)));
        __m256 b0_f13 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_11, 1)));
        
        acc00 = _mm256_fmadd_ps(a4, b0_f10, acc00);
        acc01 = _mm256_fmadd_ps(a5, b0_f11, acc01);
        acc00 = _mm256_fmadd_ps(a6, b0_f12, acc00);
        acc01 = _mm256_fmadd_ps(a7, b0_f13, acc01);
        
        /* Row 1: Same for B1 */
        __m256i b1_0 = _mm256_loadu_si256((__m256i*)(B1 + k));
        __m256i b1_1 = _mm256_loadu_si256((__m256i*)(B1 + k + 32));
        
        __m256i b1_00 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b1_0));
        __m256i b1_01 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b1_0, 1));
        __m256 b1_f00 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_00)));
        __m256 b1_f01 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_00, 1)));
        __m256 b1_f02 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_01)));
        __m256 b1_f03 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_01, 1)));
        
        acc10 = _mm256_fmadd_ps(a0, b1_f00, acc10);
        acc11 = _mm256_fmadd_ps(a1, b1_f01, acc11);
        acc10 = _mm256_fmadd_ps(a2, b1_f02, acc10);
        acc11 = _mm256_fmadd_ps(a3, b1_f03, acc11);
        
        __m256i b1_10 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b1_1));
        __m256i b1_11 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b1_1, 1));
        __m256 b1_f10 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_10)));
        __m256 b1_f11 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_10, 1)));
        __m256 b1_f12 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_11)));
        __m256 b1_f13 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_11, 1)));
        
        acc10 = _mm256_fmadd_ps(a4, b1_f10, acc10);
        acc11 = _mm256_fmadd_ps(a5, b1_f11, acc11);
        acc10 = _mm256_fmadd_ps(a6, b1_f12, acc10);
        acc11 = _mm256_fmadd_ps(a7, b1_f13, acc11);
    }
    
    /* Horizontal sum for row 0 */
    __m256 acc0 = _mm256_add_ps(acc00, acc01);
    __m128 sum0_low = _mm256_castps256_ps128(acc0);
    __m128 sum0_high = _mm256_extractf128_ps(acc0, 1);
    sum0_low = _mm_add_ps(sum0_low, sum0_high);
    sum0_low = _mm_hadd_ps(sum0_low, sum0_low);
    sum0_low = _mm_hadd_ps(sum0_low, sum0_low);
    *sum0 = _mm_cvtss_f32(sum0_low) * scale_b0 * 0.0625f;
    
    /* Horizontal sum for row 1 */
    __m256 acc1 = _mm256_add_ps(acc10, acc11);
    __m128 sum1_low = _mm256_castps256_ps128(acc1);
    __m128 sum1_high = _mm256_extractf128_ps(acc1, 1);
    sum1_low = _mm_add_ps(sum1_low, sum1_high);
    sum1_low = _mm_hadd_ps(sum1_low, sum1_low);
    sum1_low = _mm_hadd_ps(sum1_low, sum1_low);
    *sum1 = _mm_cvtss_f32(sum1_low) * scale_b1 * 0.0625f;
    
    /* Scalar remainder */
    for (; k < K; k++) {
        *sum0 += A[k] * B0[k] * scale_b0 * 0.0625f;
        *sum1 += A[k] * B1[k] * scale_b1 * 0.0625f;
    }
}

/* Ultra-fast matmul - processes 2 rows at a time */
void matmul_dequantized_ultra(const float* A, const dequantized_tensor_t* B,
                               float* C, int M, int N, int K) {
    (void)M;  /* Must be 1 */
    
    /* Process 2 rows at a time */
    int n = 0;
    for (; n <= N - 2; n += 2) {
        const int8_t* B0 = B->weights + n * B->cols;
        const int8_t* B1 = B->weights + (n + 1) * B->cols;
        float scale0 = B->scales[n];
        float scale1 = B->scales[n + 1];
        
        matmul_2x32(A, B0, B1, &C[n], &C[n + 1], K, scale0, scale1);
    }
    
    /* Handle remaining row */
    if (n < N) {
        /* Scalar fallback for last row */
        const int8_t* B_row = B->weights + n * B->cols;
        float scale = B->scales[n];
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k] * B_row[k] * scale * 0.0625f;
        }
        C[n] = sum;
    }
}

#else /* No AVX2 */

void matmul_dequantized_ultra(const float* A, const dequantized_tensor_t* B,
                               float* C, int M, int N, int K) {
    matmul_dequantized(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
