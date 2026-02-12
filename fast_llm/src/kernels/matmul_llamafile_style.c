/*
 * Llamafile-style Matmul Kernel
 * Heavy unrolling (8x) like llama.cpp
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
 * Llamafile-style matmul with 8-way unrolling
 * Based on ggml_vec_dot_f32 approach
 */
void matmul_dequantized_llamafile(const float* A, const dequantized_tensor_t* B,
                                   float* C, int M, int N, int K) {
    (void)M;  /* Must be 1 */
    
    const int STEP = 64;  /* Process 64 K values at a time (8 * 8 floats) */
    
    #pragma omp parallel for schedule(static)
    for (int n = 0; n < N; n++) {
        const int8_t* B_row = B->weights + n * B->cols;
        float scale = B->scales[n] * 0.0625f;
        
        /* 8 accumulators for heavy unrolling */
        __m256 sum0 = _mm256_setzero_ps();
        __m256 sum1 = _mm256_setzero_ps();
        __m256 sum2 = _mm256_setzero_ps();
        __m256 sum3 = _mm256_setzero_ps();
        __m256 sum4 = _mm256_setzero_ps();
        __m256 sum5 = _mm256_setzero_ps();
        __m256 sum6 = _mm256_setzero_ps();
        __m256 sum7 = _mm256_setzero_ps();
        
        int k = 0;
        /* Main loop: 64 values per iteration, 8 separate accumulators */
        for (; k <= K - STEP; k += STEP) {
            /* Prefetch far ahead */
            _mm_prefetch((const char*)(A + k + 256), _MM_HINT_T0);
            _mm_prefetch((const char*)(B_row + k + 256), _MM_HINT_T0);
            
            /* Load 8 floats from A, each reused 8 times */
            __m256 a[8];
            for (int i = 0; i < 8; i++) {
                a[i] = _mm256_loadu_ps(A + k + i * 8);
            }
            
            /* Load and dequantize 64 int8s from B */
            __m256 b[8];
            for (int i = 0; i < 8; i++) {
                __m256i b_i8 = _mm256_loadu_si256((__m256i*)(B_row + k + i * 32));
                __m256i b_i16_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b_i8));
                __m256i b_i16_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b_i8, 1));
                
                /* Dequantize both halves */
                __m256 b_lo = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b_i16_lo)));
                __m256 b_hi = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b_i16_lo, 1)));
                
                /* We need 8 separate b values, each is 8 floats */
                /* Rearrange: first 4 from lo, last 4 from hi */
                /* Actually we need to load 64 int8s as 8 chunks of 8 floats each */
                /* Each chunk: 8 int8s sign-extended to 8 floats */
                
                /* Simpler: just load 8 at a time using scalar extraction */
                /* But that's slow... */
                
                /* Better: rearrange the 64 int8s into 8 groups of 8 */
                /* Each group becomes one __m256 after sign-extension */
                
                /* For now, use simple approach: load 32 at a time */
                b[i] = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b_i16_lo)));
            }
            
            /* Actually, let's do this properly */
            /* Load 64 int8s as 2 x 256-bit registers */
            __m256i b0 = _mm256_loadu_si256((__m256i*)(B_row + k));
            __m256i b1 = _mm256_loadu_si256((__m256i*)(B_row + k + 32));
            
            /* Dequantize 8 groups of 8 values */
            /* Group 0: bytes 0-7 */
            __m256i b0_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b0));
            __m256 b_val0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_0)));
            sum0 = _mm256_fmadd_ps(a[0], b_val0, sum0);
            
            __m256 b_val1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_0, 1)));
            sum1 = _mm256_fmadd_ps(a[1], b_val1, sum1);
            
            /* Group 2: bytes 16-23 */
            __m256i b0_2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b0, 1));
            __m256 b_val2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_2)));
            sum2 = _mm256_fmadd_ps(a[2], b_val2, sum2);
            
            __m256 b_val3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_2, 1)));
            sum3 = _mm256_fmadd_ps(a[3], b_val3, sum3);
            
            /* Group 4: bytes 32-39 */
            __m256i b1_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b1));
            __m256 b_val4 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_0)));
            sum4 = _mm256_fmadd_ps(a[4], b_val4, sum4);
            
            __m256 b_val5 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_0, 1)));
            sum5 = _mm256_fmadd_ps(a[5], b_val5, sum5);
            
            /* Group 6: bytes 48-55 */
            __m256i b1_2 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b1, 1));
            __m256 b_val6 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_2)));
            sum6 = _mm256_fmadd_ps(a[6], b_val6, sum6);
            
            __m256 b_val7 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_2, 1)));
            sum7 = _mm256_fmadd_ps(a[7], b_val7, sum7);
        }
        
        /* Reduce 8 sums to 1 */
        sum0 = _mm256_add_ps(sum0, sum1);
        sum2 = _mm256_add_ps(sum2, sum3);
        sum4 = _mm256_add_ps(sum4, sum5);
        sum6 = _mm256_add_ps(sum6, sum7);
        sum0 = _mm256_add_ps(sum0, sum2);
        sum4 = _mm256_add_ps(sum4, sum6);
        sum0 = _mm256_add_ps(sum0, sum4);
        
        /* Horizontal sum */
        __m128 s = _mm_add_ps(_mm256_castps256_ps128(sum0),
                              _mm256_extractf128_ps(sum0, 1));
        s = _mm_hadd_ps(s, s);
        s = _mm_hadd_ps(s, s);
        float total = _mm_cvtss_f32(s);
        
        /* Remainder */
        for (; k < K; k++) {
            total += A[k] * B_row[k];
        }
        
        C[n] = total * scale;
    }
}

#else /* No AVX2 */

void matmul_dequantized_llamafile(const float* A, const dequantized_tensor_t* B,
                                   float* C, int M, int N, int K) {
    matmul_dequantized(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
