/*
 * Matmul with pre-transposed weights
 * Weights stored as [K, N] instead of [N, K] for sequential access
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

/* Pre-transpose weights from [N, K] to [K/16, N, 16] block format */
void transpose_weights_16xN(const int8_t* src, int N, int K,
                             int8_t* dst, const float* src_scales, float* dst_scales) {
    /* Copy scales (unchanged) */
    memcpy(dst_scales, src_scales, N * sizeof(float));
    
    /* Transpose blocks of 16xN */
    for (int k = 0; k < K; k += 16) {
        for (int n = 0; n < N; n++) {
            for (int kk = 0; kk < 16; kk++) {
                int src_idx = n * K + k + kk;
                int dst_idx = (k / 16) * N * 16 + n * 16 + kk;
                dst[dst_idx] = src[src_idx];
            }
        }
    }
}

/* 
 * 6x16 micro-kernel with transposed weights
 * Sequential access pattern for weights
 */
void matmul_dequantized_transposed(const float* A,  /* [K] */
                                    const int8_t* B_t, /* [K/16, N, 16] transposed */
                                    const float* scales_B,
                                    float* C,        /* [N] */
                                    int M, int N, int K) {
    (void)M;
    
    int num_k_blocks = K / 16;
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int n = 0; n <= N - 6; n += 6) {
        __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
        __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
        
        /* Process K blocks sequentially - better cache behavior */
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const float* A_ptr = A + kb * 16;
            __m256 a0 = _mm256_loadu_ps(A_ptr);
            __m256 a1 = _mm256_loadu_ps(A_ptr + 8);
            
            const int8_t* B_ptr = B_t + kb * N * 16 + n * 16;
            
            for (int i = 0; i < 6; i++) {
                __m128i b_i8 = _mm_loadu_si128((__m128i*)(B_ptr + i * 16));
                __m256i b_i32_0 = _mm256_cvtepi8_epi32(b_i8);
                __m256i b_i32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(b_i8, 8));
                __m256 b_f_0 = _mm256_cvtepi32_ps(b_i32_0);
                __m256 b_f_1 = _mm256_cvtepi32_ps(b_i32_1);
                
                float scale = scales_B[n + i] * 0.0625f;
                __m256 s = _mm256_set1_ps(scale);
                b_f_0 = _mm256_mul_ps(b_f_0, s);
                b_f_1 = _mm256_mul_ps(b_f_1, s);
                
                switch(i) {
                    case 0: c0 = _mm256_fmadd_ps(a0, b_f_0, c0); c0 = _mm256_fmadd_ps(a1, b_f_1, c0); break;
                    case 1: c1 = _mm256_fmadd_ps(a0, b_f_0, c1); c1 = _mm256_fmadd_ps(a1, b_f_1, c1); break;
                    case 2: c2 = _mm256_fmadd_ps(a0, b_f_0, c2); c2 = _mm256_fmadd_ps(a1, b_f_1, c2); break;
                    case 3: c3 = _mm256_fmadd_ps(a0, b_f_0, c3); c3 = _mm256_fmadd_ps(a1, b_f_1, c3); break;
                    case 4: c4 = _mm256_fmadd_ps(a0, b_f_0, c4); c4 = _mm256_fmadd_ps(a1, b_f_1, c4); break;
                    case 5: c5 = _mm256_fmadd_ps(a0, b_f_0, c5); c5 = _mm256_fmadd_ps(a1, b_f_1, c5); break;
                }
            }
        }
        
        float sum0 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c0), _mm256_extractf128_ps(c0, 1)));
        float sum1 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c1), _mm256_extractf128_ps(c1, 1)));
        float sum2 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c2), _mm256_extractf128_ps(c2, 1)));
        float sum3 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c3), _mm256_extractf128_ps(c3, 1)));
        float sum4 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c4), _mm256_extractf128_ps(c4, 1)));
        float sum5 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c5), _mm256_extractf128_ps(c5, 1)));
        
        C[n+0] = sum0; C[n+1] = sum1; C[n+2] = sum2;
        C[n+3] = sum3; C[n+4] = sum4; C[n+5] = sum5;
    }
    
    /* Handle remainder */
    int n_rem = (N / 6) * 6;
    for (int n = n_rem; n < N; n++) {
        float sum = 0.0f;
        for (int kb = 0; kb < num_k_blocks; kb++) {
            const float* A_ptr = A + kb * 16;
            const int8_t* B_ptr = B_t + kb * N * 16 + n * 16;
            float scale = scales_B[n] * 0.0625f;
            for (int kk = 0; kk < 16; kk++) {
                sum += A_ptr[kk] * B_ptr[kk] * scale;
            }
        }
        C[n] = sum;
    }
}

#else /* No AVX2 */

void transpose_weights_16xN(const int8_t* src, int N, int K,
                             int8_t* dst, const float* src_scales, float* dst_scales) {
    /* No-op fallback */
    (void)src; (void)N; (void)K; (void)dst; (void)src_scales; (void)dst_scales;
}

void matmul_dequantized_transposed(const float* A, const int8_t* B_t,
                                    const float* scales_B,
                                    float* C, int M, int N, int K) {
    (void)A; (void)B_t; (void)scales_B; (void)C; (void)M; (void)N; (void)K;
}

#endif /* __AVX2__ */
