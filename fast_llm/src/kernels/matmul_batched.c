/*
 * Batched Matrix Multiplication
 * Process multiple tokens simultaneously to amortize overhead
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
 * Batched 6x16 micro-kernel
 * Processes 2 tokens x 6 rows x 16 K values
 * 
 * Key optimization: reuse B weights across multiple A vectors
 */
void matmul_batched_6x16(const float** A_batch,  /* [batch_size][K] */
                          const dequantized_tensor_t* B,
                          float** C_batch,        /* [batch_size][N] */
                          int batch_size, int M, int N, int K) {
    (void)M;
    
    /* Process batch in parallel */
    #pragma omp parallel for schedule(dynamic, 4)
    for (int b = 0; b < batch_size; b++) {
        const float* A = A_batch[b];
        float* C = C_batch[b];
        
        /* Use the optimized 6x16 kernel for each batch element */
        extern void matmul_dequantized_asm_style(const float*, const dequantized_tensor_t*, float*, int, int, int);
        matmul_dequantized_asm_style(A, B, C, 1, N, K);
    }
}

/*
 * True batched matmul - better cache utilization
 * Process 2 tokens simultaneously within each thread
 */
void matmul_batched_2x(const float* A0, const float* A1,  /* 2 input vectors */
                        const dequantized_tensor_t* B,
                        float* C0, float* C1,              /* 2 output vectors */
                        int M, int N, int K) {
    (void)M;
    
    /* Process N in blocks of 6 rows */
    #pragma omp parallel for schedule(dynamic, 32)
    for (int n = 0; n <= N - 6; n += 6) {
        /* 12 accumulators - 6 for each output */
        __m256 c00 = _mm256_setzero_ps(), c01 = _mm256_setzero_ps();
        __m256 c02 = _mm256_setzero_ps(), c03 = _mm256_setzero_ps();
        __m256 c04 = _mm256_setzero_ps(), c05 = _mm256_setzero_ps();
        
        __m256 c10 = _mm256_setzero_ps(), c11 = _mm256_setzero_ps();
        __m256 c12 = _mm256_setzero_ps(), c13 = _mm256_setzero_ps();
        __m256 c14 = _mm256_setzero_ps(), c15 = _mm256_setzero_ps();
        
        /* Process K in blocks of 16 */
        for (int k = 0; k <= K - 16; k += 16) {
            /* Load 16 floats from each input */
            __m256 a0 = _mm256_loadu_ps(A0 + k);
            __m256 a1 = _mm256_loadu_ps(A1 + k);
            
            /* Prefetch next K block */
            _mm_prefetch((const char*)(A0 + k + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(A1 + k + 64), _MM_HINT_T0);
            
            /* Process 6 rows */
            for (int i = 0; i < 6; i++) {
                /* Load B weights */
                __m128i b_i8_lo = _mm_loadu_si128((__m128i*)(B->weights + (n+i) * K + k));
                __m256i b_i32_0 = _mm256_cvtepi8_epi32(b_i8_lo);
                __m256i b_i32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(b_i8_lo, 8));
                __m256 b_f_0 = _mm256_cvtepi32_ps(b_i32_0);
                __m256 b_f_1 = _mm256_cvtepi32_ps(b_i32_1);
                
                __m256 scale = _mm256_set1_ps(B->scales[n+i] * 0.0625f);
                b_f_0 = _mm256_mul_ps(b_f_0, scale);
                b_f_1 = _mm256_mul_ps(b_f_1, scale);
                
                /* FMA for A0 */
                switch(i) {
                    case 0: c00 = _mm256_fmadd_ps(a0, b_f_0, c00); c00 = _mm256_fmadd_ps(a0, b_f_1, c00); break;
                    case 1: c01 = _mm256_fmadd_ps(a0, b_f_0, c01); c01 = _mm256_fmadd_ps(a0, b_f_1, c01); break;
                    case 2: c02 = _mm256_fmadd_ps(a0, b_f_0, c02); c02 = _mm256_fmadd_ps(a0, b_f_1, c02); break;
                    case 3: c03 = _mm256_fmadd_ps(a0, b_f_0, c03); c03 = _mm256_fmadd_ps(a0, b_f_1, c03); break;
                    case 4: c04 = _mm256_fmadd_ps(a0, b_f_0, c04); c04 = _mm256_fmadd_ps(a0, b_f_1, c04); break;
                    case 5: c05 = _mm256_fmadd_ps(a0, b_f_0, c05); c05 = _mm256_fmadd_ps(a0, b_f_1, c05); break;
                }
                
                /* FMA for A1 */
                switch(i) {
                    case 0: c10 = _mm256_fmadd_ps(a1, b_f_0, c10); c10 = _mm256_fmadd_ps(a1, b_f_1, c10); break;
                    case 1: c11 = _mm256_fmadd_ps(a1, b_f_0, c11); c11 = _mm256_fmadd_ps(a1, b_f_1, c11); break;
                    case 2: c12 = _mm256_fmadd_ps(a1, b_f_0, c12); c12 = _mm256_fmadd_ps(a1, b_f_1, c12); break;
                    case 3: c13 = _mm256_fmadd_ps(a1, b_f_0, c13); c13 = _mm256_fmadd_ps(a1, b_f_1, c13); break;
                    case 4: c14 = _mm256_fmadd_ps(a1, b_f_0, c14); c14 = _mm256_fmadd_ps(a1, b_f_1, c14); break;
                    case 5: c15 = _mm256_fmadd_ps(a1, b_f_0, c15); c15 = _mm256_fmadd_ps(a1, b_f_1, c15); break;
                }
            }
        }
        
        /* Horizontal sum and store for A0 */
        float sum00 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c00), _mm256_extractf128_ps(c00, 1)));
        float sum01 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c01), _mm256_extractf128_ps(c01, 1)));
        float sum02 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c02), _mm256_extractf128_ps(c02, 1)));
        float sum03 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c03), _mm256_extractf128_ps(c03, 1)));
        float sum04 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c04), _mm256_extractf128_ps(c04, 1)));
        float sum05 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c05), _mm256_extractf128_ps(c05, 1)));
        
        C0[n+0] = sum00; C0[n+1] = sum01; C0[n+2] = sum02;
        C0[n+3] = sum03; C0[n+4] = sum04; C0[n+5] = sum05;
        
        /* Horizontal sum and store for A1 */
        float sum10 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c10), _mm256_extractf128_ps(c10, 1)));
        float sum11 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c11), _mm256_extractf128_ps(c11, 1)));
        float sum12 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c12), _mm256_extractf128_ps(c12, 1)));
        float sum13 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c13), _mm256_extractf128_ps(c13, 1)));
        float sum14 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c14), _mm256_extractf128_ps(c14, 1)));
        float sum15 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c15), _mm256_extractf128_ps(c15, 1)));
        
        C1[n+0] = sum10; C1[n+1] = sum11; C1[n+2] = sum12;
        C1[n+3] = sum13; C1[n+4] = sum14; C1[n+5] = sum15;
    }
    
    /* Handle remaining rows */
    int n_rem = (N / 6) * 6;
    for (int n = n_rem; n < N; n++) {
        const int8_t* B_row = B->weights + n * K;
        float scale = B->scales[n] * 0.0625f;
        float sum0 = 0.0f, sum1 = 0.0f;
        for (int k = 0; k < K; k++) {
            sum0 += A0[k] * B_row[k] * scale;
            sum1 += A1[k] * B_row[k] * scale;
        }
        C0[n] = sum0;
        C1[n] = sum1;
    }
}

#else /* No AVX2 */

void matmul_batched_2x(const float* A0, const float* A1,
                        const dequantized_tensor_t* B,
                        float* C0, float* C1,
                        int M, int N, int K) {
    extern void matmul_dequantized(const float*, const dequantized_tensor_t*, float*, int, int, int);
    matmul_dequantized(A0, B, C0, M, N, K);
    matmul_dequantized(A1, B, C1, M, N, K);
}

#endif /* __AVX2__ */
