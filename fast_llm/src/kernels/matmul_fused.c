/*
 * Fused Matmul + SwiGLU Kernels
 * Combines gate/up projection with SwiGLU activation
 * Eliminates separate memory passes
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

/* Fast sigmoid approximation */
static inline float fast_sigmoid_scalar(float x) {
    /* sigmoid(x) = 1 / (1 + exp(-x)) */
    /* For small x, use approximation */
    if (x >= 0) {
        float z = expf(-x);
        return 1.0f / (1.0f + z);
    } else {
        float z = expf(x);
        return z / (1.0f + z);
    }
}

/* 
 * Fused Gate+Up matmul + SwiGLU
 * Processes 16 output columns at a time
 * Computes: output[i] = sigmoid(gate[i]) * gate[i] * up[i]
 */
void matmul_gate_up_swiglu_fused(const float* A,  /* [K] input */
                                  const dequantized_tensor_t* B_gate, /* [inter, K] */
                                  const dequantized_tensor_t* B_up,   /* [inter, K] */
                                  float* C,        /* [inter] output */
                                  int M, int N, int K) {
    (void)M;
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int n = 0; n < N; n++) {
        float sum_gate = 0.0f;
        float sum_up = 0.0f;
        
        const int8_t* Bg_row = B_gate->weights + n * K;
        const int8_t* Bu_row = B_up->weights + n * K;
        float scale_gate = B_gate->scales[n] * 0.0625f;
        float scale_up = B_up->scales[n] * 0.0625f;
        
        /* Unrolled dot product for gate and up */
        int k = 0;
        
        /* Main loop - 16 values at a time */
        for (; k <= K - 16; k += 16) {
            __m256 a0 = _mm256_loadu_ps(A + k);
            __m256 a1 = _mm256_loadu_ps(A + k + 8);
            
            /* Process gate */
            __m128i bg_i8_0 = _mm_loadu_si128((__m128i*)(Bg_row + k));
            __m128i bg_i8_1 = _mm_loadu_si128((__m128i*)(Bg_row + k + 8));
            __m256 bg_f_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bg_i8_0));
            __m256 bg_f_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bg_i8_0, 8)));
            __m256 bg_f_2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bg_i8_1));
            __m256 bg_f_3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bg_i8_1, 8)));
            
            __m256 prod_g0 = _mm256_mul_ps(a0, bg_f_0);
            __m256 prod_g1 = _mm256_mul_ps(a0, bg_f_1);
            __m256 prod_g2 = _mm256_mul_ps(a1, bg_f_2);
            __m256 prod_g3 = _mm256_mul_ps(a1, bg_f_3);
            
            /* Horizontal sum for gate */
            __m256 g01 = _mm256_hadd_ps(prod_g0, prod_g1);
            __m256 g23 = _mm256_hadd_ps(prod_g2, prod_g3);
            __m256 g0123 = _mm256_hadd_ps(g01, g23);
            sum_gate += _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(g0123), 
                                                  _mm256_extractf128_ps(g0123, 1)));
            
            /* Process up */
            __m128i bu_i8_0 = _mm_loadu_si128((__m128i*)(Bu_row + k));
            __m128i bu_i8_1 = _mm_loadu_si128((__m128i*)(Bu_row + k + 8));
            __m256 bu_f_0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bu_i8_0));
            __m256 bu_f_1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bu_i8_0, 8)));
            __m256 bu_f_2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(bu_i8_1));
            __m256 bu_f_3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(bu_i8_1, 8)));
            
            __m256 prod_u0 = _mm256_mul_ps(a0, bu_f_0);
            __m256 prod_u1 = _mm256_mul_ps(a0, bu_f_1);
            __m256 prod_u2 = _mm256_mul_ps(a1, bu_f_2);
            __m256 prod_u3 = _mm256_mul_ps(a1, bu_f_3);
            
            /* Horizontal sum for up */
            __m256 u01 = _mm256_hadd_ps(prod_u0, prod_u1);
            __m256 u23 = _mm256_hadd_ps(prod_u2, prod_u3);
            __m256 u0123 = _mm256_hadd_ps(u01, u23);
            sum_up += _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(u0123), 
                                               _mm256_extractf128_ps(u0123, 1)));
        }
        
        /* Remainder */
        for (; k < K; k++) {
            sum_gate += A[k] * Bg_row[k];
            sum_up += A[k] * Bu_row[k];
        }
        
        /* Apply scales */
        sum_gate *= scale_gate;
        sum_up *= scale_up;
        
        /* SwiGLU: sigmoid(gate) * gate * up */
        float sig = 1.0f / (1.0f + expf(-sum_gate));
        C[n] = sig * sum_gate * sum_up;
    }
}

/* Simple scalar version for comparison */
void matmul_gate_up_swiglu_scalar(const float* A,
                                   const dequantized_tensor_t* B_gate,
                                   const dequantized_tensor_t* B_up,
                                   float* C,
                                   int M, int N, int K) {
    (void)M;
    
    #pragma omp parallel for schedule(dynamic, 64)
    for (int n = 0; n < N; n++) {
        float sum_gate = 0.0f;
        float sum_up = 0.0f;
        
        const int8_t* Bg_row = B_gate->weights + n * K;
        const int8_t* Bu_row = B_up->weights + n * K;
        float scale_gate = B_gate->scales[n] * 0.0625f;
        float scale_up = B_up->scales[n] * 0.0625f;
        
        for (int k = 0; k < K; k++) {
            sum_gate += A[k] * Bg_row[k];
            sum_up += A[k] * Bu_row[k];
        }
        
        sum_gate *= scale_gate;
        sum_up *= scale_up;
        
        float sig = 1.0f / (1.0f + expf(-sum_gate));
        C[n] = sig * sum_gate * sum_up;
    }
}

#else /* No AVX2 */

void matmul_gate_up_swiglu_fused(const float* A,
                                  const dequantized_tensor_t* B_gate,
                                  const dequantized_tensor_t* B_up,
                                  float* C,
                                  int M, int N, int K) {
    matmul_gate_up_swiglu_scalar(A, B_gate, B_up, C, M, N, K);
}

void matmul_gate_up_swiglu_scalar(const float* A,
                                   const dequantized_tensor_t* B_gate,
                                   const dequantized_tensor_t* B_up,
                                   float* C,
                                   int M, int N, int K) {
    /* Fallback to simple implementation */
    for (int n = 0; n < N; n++) {
        float sum_gate = 0.0f, sum_up = 0.0f;
        for (int k = 0; k < K; k++) {
            sum_gate += A[k] * B_gate->weights[n * K + k];
            sum_up += A[k] * B_up->weights[n * K + k];
        }
        sum_gate *= B_gate->scales[n] * 0.0625f;
        sum_up *= B_up->scales[n] * 0.0625f;
        float sig = 1.0f / (1.0f + expf(-sum_gate));
        C[n] = sig * sum_gate * sum_up;
    }
}

#endif /* __AVX2__ */
