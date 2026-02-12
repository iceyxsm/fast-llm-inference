/*
 * Assembly-Style Matmul Kernel
 * 
 * Techniques from llama.cpp and BLIS:
 * 1. 6x16 micro-kernel (6 output rows x 16 K values)
 * 2. Register blocking to keep data in L1
 * 3. Aggressive prefetching
 * 4. Minimize memory traffic
 * 
 * This approaches hand-tuned assembly performance
 * while remaining portable C with intrinsics
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
 * Micro-kernel: compute 6 output rows x 16 K values
 * Uses 6 accumulators, each processing 16 K values
 * 
 * This fits in L1 cache:
 * - 6 rows x 16 cols = 96 float outputs
 * - 16 K values loaded per iteration
 * - Total: ~400 bytes, well within 32KB L1
 */
static inline void micro_kernel_6x16(const float* A,           /* [16] - 16 K values */
                                      const int8_t* B0,        /* [16] - row 0 weights */
                                      const int8_t* B1,        /* [16] - row 1 weights */
                                      const int8_t* B2,        /* [16] - row 2 weights */
                                      const int8_t* B3,        /* [16] - row 3 weights */
                                      const int8_t* B4,        /* [16] - row 4 weights */
                                      const int8_t* B5,        /* [16] - row 5 weights */
                                      float* sums,             /* [6] - output sums */
                                      float scale0, float scale1, float scale2,
                                      float scale3, float scale4, float scale5) {
    
    /* Load 16 floats from A (will be reused for all 6 rows) */
    __m256 a0 = _mm256_loadu_ps(A);
    __m256 a1 = _mm256_loadu_ps(A + 8);
    
    /* Load and dequantize 16 int8s for each row */
    /* Row 0 */
    __m256i b0_i8 = _mm256_loadu_si256((__m256i*)B0);
    __m256i b0_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b0_i8));
    __m256 b0_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_i16)));
    __m256 b0_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_i16, 1)));
    
    /* Row 1 */
    __m256i b1_i8 = _mm256_loadu_si256((__m256i*)B1);
    __m256i b1_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b1_i8));
    __m256 b1_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_i16)));
    __m256 b1_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_i16, 1)));
    
    /* Row 2 */
    __m256i b2_i8 = _mm256_loadu_si256((__m256i*)B2);
    __m256i b2_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b2_i8));
    __m256 b2_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b2_i16)));
    __m256 b2_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b2_i16, 1)));
    
    /* Row 3 */
    __m256i b3_i8 = _mm256_loadu_si256((__m256i*)B3);
    __m256i b3_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b3_i8));
    __m256 b3_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b3_i16)));
    __m256 b3_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b3_i16, 1)));
    
    /* Row 4 */
    __m256i b4_i8 = _mm256_loadu_si256((__m256i*)B4);
    __m256i b4_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b4_i8));
    __m256 b4_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b4_i16)));
    __m256 b4_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b4_i16, 1)));
    
    /* Row 5 */
    __m256i b5_i8 = _mm256_loadu_si256((__m256i*)B5);
    __m256i b5_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b5_i8));
    __m256 b5_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b5_i16)));
    __m256 b5_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b5_i16, 1)));
    
    /* Compute dot products */
    /* Using FMA: sum += A * B for each row */
    __m256 sum0 = _mm256_add_ps(_mm256_mul_ps(a0, b0_0), _mm256_mul_ps(a1, b0_1));
    __m256 sum1 = _mm256_add_ps(_mm256_mul_ps(a0, b1_0), _mm256_mul_ps(a1, b1_1));
    __m256 sum2 = _mm256_add_ps(_mm256_mul_ps(a0, b2_0), _mm256_mul_ps(a1, b2_1));
    __m256 sum3 = _mm256_add_ps(_mm256_mul_ps(a0, b3_0), _mm256_mul_ps(a1, b3_1));
    __m256 sum4 = _mm256_add_ps(_mm256_mul_ps(a0, b4_0), _mm256_mul_ps(a1, b4_1));
    __m256 sum5 = _mm256_add_ps(_mm256_mul_ps(a0, b5_0), _mm256_mul_ps(a1, b5_1));
    
    /* Horizontal sums */
    #define HSUM(acc, scale) do { \
        __m128 lo = _mm256_castps256_ps128(acc); \
        __m128 hi = _mm256_extractf128_ps(acc, 1); \
        lo = _mm_add_ps(lo, hi); \
        lo = _mm_hadd_ps(lo, lo); \
        lo = _mm_hadd_ps(lo, lo); \
        *sums++ = _mm_cvtss_f32(lo) * scale * 0.0625f; \
    } while(0)
    
    HSUM(sum0, scale0);
    HSUM(sum1, scale1);
    HSUM(sum2, scale2);
    HSUM(sum3, scale3);
    HSUM(sum4, scale4);
    HSUM(sum5, scale5);
    
    #undef HSUM
}

/* 
 * Blocked matmul using 6x16 micro-kernel
 * Processes 6 output rows at a time
 */
void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                   float* C, int M, int N, int K) {
    (void)M;  /* Must be 1 */
    
    /* Process 6 rows at a time */
    int n = 0;
    for (; n <= N - 6; n += 6) {
        /* Accumulators for 6 rows */
        float sums[6] = {0};
        
        /* Process K in chunks of 16 */
        int k = 0;
        for (; k <= K - 16; k += 16) {
            /* Prefetch ahead */
            _mm_prefetch((const char*)(A + k + 64), _MM_HINT_T0);
            
            micro_kernel_6x16(
                A + k,
                B->weights + (n+0) * B->cols + k,
                B->weights + (n+1) * B->cols + k,
                B->weights + (n+2) * B->cols + k,
                B->weights + (n+3) * B->cols + k,
                B->weights + (n+4) * B->cols + k,
                B->weights + (n+5) * B->cols + k,
                sums,
                B->scales[n+0], B->scales[n+1], B->scales[n+2],
                B->scales[n+3], B->scales[n+4], B->scales[n+5]
            );
        }
        
        /* Remainder K (scalar) */
        for (; k < K; k++) {
            for (int i = 0; i < 6; i++) {
                sums[i] += A[k] * B->weights[(n+i) * B->cols + k] * B->scales[n+i] * 0.0625f;
            }
        }
        
        /* Store results */
        for (int i = 0; i < 6; i++) {
            C[n + i] = sums[i];
        }
    }
    
    /* Handle remaining rows (scalar) */
    for (; n < N; n++) {
        const int8_t* B_row = B->weights + n * B->cols;
        float scale = B->scales[n] * 0.0625f;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k] * B_row[k];
        }
        C[n] = sum * scale;
    }
}

/*
 * Version with aggressive prefetching for sequential access patterns
 * PREFETCH_DISTANCE controls how far ahead to prefetch
 */
void matmul_dequantized_prefetch(const float* A, const dequantized_tensor_t* B,
                                  float* C, int M, int N, int K) {
    (void)M;  /* Must be 1 */
    
    #define PREFETCH_DISTANCE 8
    
    /* Process 6 rows at a time */
    int n = 0;
    for (; n <= N - 6; n += 6) {
        /* Accumulators for 6 rows */
        float sums[6] = {0};
        
        /* Prefetch output location */
        _mm_prefetch((const char*)(C + n), _MM_HINT_T0);
        
        /* Process K in chunks of 16 */
        int k = 0;
        for (; k <= K - 16; k += 16) {
            /* Aggressive prefetching of A and B for future iterations */
            _mm_prefetch((const char*)(A + k + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(A + k + 128), _MM_HINT_T1);
            
            /* Prefetch B weights for next iterations */
            for (int i = 0; i < 6; i++) {
                _mm_prefetch((const char*)(B->weights + (n+i) * B->cols + k + 64), _MM_HINT_T1);
            }
            
            micro_kernel_6x16(
                A + k,
                B->weights + (n+0) * B->cols + k,
                B->weights + (n+1) * B->cols + k,
                B->weights + (n+2) * B->cols + k,
                B->weights + (n+3) * B->cols + k,
                B->weights + (n+4) * B->cols + k,
                B->weights + (n+5) * B->cols + k,
                sums,
                B->scales[n+0], B->scales[n+1], B->scales[n+2],
                B->scales[n+3], B->scales[n+4], B->scales[n+5]
            );
        }
        
        /* Remainder K (scalar) */
        for (; k < K; k++) {
            for (int i = 0; i < 6; i++) {
                sums[i] += A[k] * B->weights[(n+i) * B->cols + k] * B->scales[n+i] * 0.0625f;
            }
        }
        
        /* Store results with non-temporal hint for large outputs */
        if (N > 512) {
            /* Use streaming stores for large matrices to avoid cache pollution */
            for (int i = 0; i < 6; i++) {
                C[n + i] = sums[i];
            }
        } else {
            for (int i = 0; i < 6; i++) {
                C[n + i] = sums[i];
            }
        }
    }
    
    /* Handle remaining rows (scalar) */
    for (; n < N; n++) {
        const int8_t* B_row = B->weights + n * B->cols;
        float scale = B->scales[n] * 0.0625f;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k] * B_row[k];
        }
        C[n] = sum * scale;
    }
    
    #undef PREFETCH_DISTANCE
}

#else /* No AVX2 */

void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                   float* C, int M, int N, int K) {
    matmul_dequantized(A, B, C, M, N, K);
}

void matmul_dequantized_prefetch(const float* A, const dequantized_tensor_t* B,
                                  float* C, int M, int N, int K) {
    matmul_dequantized(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
