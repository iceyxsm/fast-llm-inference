/*
 * Optimized Matmul for Large N (Gate+Up projection)
 * N = 16384, K = 3072
 * 
 * Strategy for large output dimensions:
 * 1. Process 16 rows at a time to maximize parallelism
 * 2. Use tiled approach for better cache utilization
 * 3. Transpose weights for better access pattern
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
 * 16x8 micro-kernel for large N
 * Processes 16 output rows x 8 K values
 * Better for large N where we want more parallelism
 */
static inline void micro_kernel_16x8(const float* A,  /* [8] */
                                      const int8_t* B0, const int8_t* B1,
                                      const int8_t* B2, const int8_t* B3,
                                      const int8_t* B4, const int8_t* B5,
                                      const int8_t* B6, const int8_t* B7,
                                      const int8_t* B8, const int8_t* B9,
                                      const int8_t* B10, const int8_t* B11,
                                      const int8_t* B12, const int8_t* B13,
                                      const int8_t* B14, const int8_t* B15,
                                      float* sums,
                                      const float* scales) {
    
    /* Load 8 floats from A */
    __m256 a = _mm256_loadu_ps(A);
    
    /* Process 16 rows */
    #define PROCESS_ROW(n) do { \
        __m128i b_i8 = _mm_loadl_epi64((__m128i*)(B##n)); \
        __m256i b_i32 = _mm256_cvtepi8_epi32(b_i8); \
        __m256 b_f = _mm256_cvtepi32_ps(b_i32); \
        __m256 prod = _mm256_mul_ps(a, b_f); \
        __m128 lo = _mm256_castps256_ps128(prod); \
        __m128 hi = _mm256_extractf128_ps(prod, 1); \
        lo = _mm_add_ps(lo, hi); \
        lo = _mm_hadd_ps(lo, lo); \
        lo = _mm_hadd_ps(lo, lo); \
        sums[n] += _mm_cvtss_f32(lo) * scales[n] * 0.0625f; \
    } while(0)
    
    PROCESS_ROW(0);  PROCESS_ROW(1);  PROCESS_ROW(2);  PROCESS_ROW(3);
    PROCESS_ROW(4);  PROCESS_ROW(5);  PROCESS_ROW(6);  PROCESS_ROW(7);
    PROCESS_ROW(8);  PROCESS_ROW(9);  PROCESS_ROW(10); PROCESS_ROW(11);
    PROCESS_ROW(12); PROCESS_ROW(13); PROCESS_ROW(14); PROCESS_ROW(15);
    
    #undef PROCESS_ROW
}

/* 
 * Large-N optimized matmul
 * Uses 16x8 micro-kernel for better parallelism on large output dims
 */
void matmul_dequantized_large_n(const float* A, const dequantized_tensor_t* B,
                                 float* C, int M, int N, int K) {
    (void)M;
    
    /* For large N, use 16-row blocking */
    #pragma omp parallel for schedule(dynamic, 32)
    for (int n = 0; n <= N - 16; n += 16) {
        float sums[16] = {0};
        
        /* Process K in chunks of 8 */
        int k = 0;
        for (; k <= K - 8; k += 8) {
            /* Prefetch */
            _mm_prefetch((const char*)(A + k + 64), _MM_HINT_T0);
            
            micro_kernel_16x8(
                A + k,
                B->weights + (n+0)*B->cols + k,
                B->weights + (n+1)*B->cols + k,
                B->weights + (n+2)*B->cols + k,
                B->weights + (n+3)*B->cols + k,
                B->weights + (n+4)*B->cols + k,
                B->weights + (n+5)*B->cols + k,
                B->weights + (n+6)*B->cols + k,
                B->weights + (n+7)*B->cols + k,
                B->weights + (n+8)*B->cols + k,
                B->weights + (n+9)*B->cols + k,
                B->weights + (n+10)*B->cols + k,
                B->weights + (n+11)*B->cols + k,
                B->weights + (n+12)*B->cols + k,
                B->weights + (n+13)*B->cols + k,
                B->weights + (n+14)*B->cols + k,
                B->weights + (n+15)*B->cols + k,
                sums,
                B->scales + n
            );
        }
        
        /* Remainder */
        for (; k < K; k++) {
            for (int i = 0; i < 16; i++) {
                sums[i] += A[k] * B->weights[(n+i)*B->cols + k] * B->scales[n+i] * 0.0625f;
            }
        }
        
        for (int i = 0; i < 16; i++) {
            C[n + i] = sums[i];
        }
    }
    
    /* Handle remaining rows */
    int n_rem = (N / 16) * 16;
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

/* 
 * Adaptive matmul - chooses best kernel based on dimensions
 */
void matmul_dequantized_adaptive(const float* A, const dequantized_tensor_t* B,
                                  float* C, int M, int N, int K) {
    
    /* For large N (Gate+Up), use 16x8 kernel */
    if (N >= 8192 && K <= 4096) {
        matmul_dequantized_large_n(A, B, C, M, N, K);
    } else {
        /* For smaller N (Down), use 6x16 kernel */
        /* Fall back to original implementation */
        extern void matmul_dequantized_asm_style(const float*, const dequantized_tensor_t*, float*, int, int, int);
        matmul_dequantized_asm_style(A, B, C, M, N, K);
    }
}

#else /* No AVX2 */

void matmul_dequantized_adaptive(const float* A, const dequantized_tensor_t* B,
                                  float* C, int M, int N, int K) {
    matmul_dequantized(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
