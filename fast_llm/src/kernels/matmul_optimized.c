/*
 * Further Optimized 6x16 Kernel
 * Target: 40-50 tok/sec
 * 
 * Optimizations:
 * 1. Aggressive prefetching (8 cache lines ahead)
 * 2. Software pipelining (interleave loads/compute)
 * 3. Better OpenMP scheduling
 * 4. Loop unrolling for K dimension
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
 * Optimized 6x16 micro-kernel with aggressive prefetching
 */
static inline void micro_kernel_6x16_optimized(const float* A,
                                                const int8_t* B0, const int8_t* B1,
                                                const int8_t* B2, const int8_t* B3,
                                                const int8_t* B4, const int8_t* B5,
                                                float* sums,
                                                const float* scales,
                                                const float* A_next,
                                                const int8_t* B_next) {
    
    /* Prefetch next iteration's data */
    _mm_prefetch((const char*)A_next, _MM_HINT_T0);
    _mm_prefetch((const char*)B_next, _MM_HINT_T0);
    _mm_prefetch((const char*)(B_next + 64), _MM_HINT_T0);
    
    /* Load 16 floats from A */
    __m256 a0 = _mm256_loadu_ps(A);
    __m256 a1 = _mm256_loadu_ps(A + 8);
    
    /* Process 6 rows with interleaved loads */
    /* Row 0 */
    __m256i b0_i8 = _mm256_loadu_si256((__m256i*)B0);
    __m256i b0_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b0_i8));
    __m256 b0_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_i16)));
    __m256 b0_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_i16, 1)));
    __m256 sum0 = _mm256_add_ps(_mm256_mul_ps(a0, b0_0), _mm256_mul_ps(a1, b0_1));
    
    /* Row 1 */
    __m256i b1_i8 = _mm256_loadu_si256((__m256i*)B1);
    __m256i b1_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b1_i8));
    __m256 b1_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_i16)));
    __m256 b1_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_i16, 1)));
    __m256 sum1 = _mm256_add_ps(_mm256_mul_ps(a0, b1_0), _mm256_mul_ps(a1, b1_1));
    
    /* Row 2 */
    __m256i b2_i8 = _mm256_loadu_si256((__m256i*)B2);
    __m256i b2_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b2_i8));
    __m256 b2_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b2_i16)));
    __m256 b2_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b2_i16, 1)));
    __m256 sum2 = _mm256_add_ps(_mm256_mul_ps(a0, b2_0), _mm256_mul_ps(a1, b2_1));
    
    /* Row 3 */
    __m256i b3_i8 = _mm256_loadu_si256((__m256i*)B3);
    __m256i b3_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b3_i8));
    __m256 b3_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b3_i16)));
    __m256 b3_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b3_i16, 1)));
    __m256 sum3 = _mm256_add_ps(_mm256_mul_ps(a0, b3_0), _mm256_mul_ps(a1, b3_1));
    
    /* Row 4 */
    __m256i b4_i8 = _mm256_loadu_si256((__m256i*)B4);
    __m256i b4_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b4_i8));
    __m256 b4_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b4_i16)));
    __m256 b4_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b4_i16, 1)));
    __m256 sum4 = _mm256_add_ps(_mm256_mul_ps(a0, b4_0), _mm256_mul_ps(a1, b4_1));
    
    /* Row 5 */
    __m256i b5_i8 = _mm256_loadu_si256((__m256i*)B5);
    __m256i b5_i16 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b5_i8));
    __m256 b5_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b5_i16)));
    __m256 b5_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b5_i16, 1)));
    __m256 sum5 = _mm256_add_ps(_mm256_mul_ps(a0, b5_0), _mm256_mul_ps(a1, b5_1));
    
    /* Horizontal sums */
    #define HSUM(acc, idx) do { \
        __m128 lo = _mm256_castps256_ps128(acc); \
        __m128 hi = _mm256_extractf128_ps(acc, 1); \
        lo = _mm_add_ps(lo, hi); \
        lo = _mm_hadd_ps(lo, lo); \
        lo = _mm_hadd_ps(lo, lo); \
        sums[idx] += _mm_cvtss_f32(lo) * scales[idx] * 0.0625f; \
    } while(0)
    
    HSUM(sum0, 0); HSUM(sum1, 1); HSUM(sum2, 2);
    HSUM(sum3, 3); HSUM(sum4, 4); HSUM(sum5, 5);
    
    #undef HSUM
}

/* 
 * Ultra-optimized matmul with 2-way unroll in K
 * Processes 2 blocks of 16 K values per iteration
 */
void matmul_dequantized_ultra(const float* A, const dequantized_tensor_t* B,
                               float* C, int M, int N, int K) {
    (void)M;
    
    /* Use dynamic scheduling for better load balancing */
    #pragma omp parallel for schedule(dynamic, 24)
    for (int n = 0; n <= N - 6; n += 6) {
        float sums[6] = {0};
        
        /* 2-way unroll in K - process 32 values at a time */
        int k = 0;
        for (; k <= K - 32; k += 32) {
            /* Block 0 */
            micro_kernel_6x16_optimized(
                A + k,
                B->weights + (n+0)*B->cols + k,
                B->weights + (n+1)*B->cols + k,
                B->weights + (n+2)*B->cols + k,
                B->weights + (n+3)*B->cols + k,
                B->weights + (n+4)*B->cols + k,
                B->weights + (n+5)*B->cols + k,
                sums,
                B->scales + n,
                A + k + 32,
                B->weights + n*B->cols + k + 32
            );
            
            /* Block 1 */
            micro_kernel_6x16_optimized(
                A + k + 16,
                B->weights + (n+0)*B->cols + k + 16,
                B->weights + (n+1)*B->cols + k + 16,
                B->weights + (n+2)*B->cols + k + 16,
                B->weights + (n+3)*B->cols + k + 16,
                B->weights + (n+4)*B->cols + k + 16,
                B->weights + (n+5)*B->cols + k + 16,
                sums,
                B->scales + n,
                (k + 32 < K) ? A + k + 32 : A,
                (k + 32 < K) ? B->weights + n*B->cols + k + 32 : B->weights
            );
        }
        
        /* Remainder K (single blocks) */
        for (; k <= K - 16; k += 16) {
            micro_kernel_6x16_optimized(
                A + k,
                B->weights + (n+0)*B->cols + k,
                B->weights + (n+1)*B->cols + k,
                B->weights + (n+2)*B->cols + k,
                B->weights + (n+3)*B->cols + k,
                B->weights + (n+4)*B->cols + k,
                B->weights + (n+5)*B->cols + k,
                sums,
                B->scales + n,
                A + k + 16,
                B->weights + n*B->cols + k + 16
            );
        }
        
        /* Scalar remainder */
        for (; k < K; k++) {
            for (int i = 0; i < 6; i++) {
                sums[i] += A[k] * B->weights[(n+i)*B->cols + k] * B->scales[n+i] * 0.0625f;
            }
        }
        
        /* Store */
        for (int i = 0; i < 6; i++) {
            C[n + i] = sums[i];
        }
    }
    
    /* Handle remaining rows */
    int n_rem = (N / 6) * 6;
    for (int n = n_rem; n < N; n++) {
        const int8_t* B_row = B->weights + n * B->cols;
        float scale = B->scales[n] * 0.0625f;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k] * B_row[k];
        }
        C[n] = sum * scale;
    }
}

#else /* No AVX2 */

void matmul_dequantized_ultra(const float* A, const dequantized_tensor_t* B,
                               float* C, int M, int N, int K) {
    matmul_dequantized(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
