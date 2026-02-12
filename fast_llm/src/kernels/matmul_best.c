/*
 * Best Matmul Kernel - Final Optimized Version
 * Combines lessons from all attempts
 * 
 * Key optimizations:
 * 1. Simple dequantization: int8 -> float on-the-fly with _mm256_cvtepi32_ps
 * 2. Aggressive prefetching (4 cache lines ahead)
 * 3. Unroll 4x for ILP
 * 4. No bias correction overhead
 * 5. Process 32 K values per iteration
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
 * Best matmul kernel
 * Simple, fast, no frills
 */
void matmul_dequantized_best(const float* A, const dequantized_tensor_t* B,
                              float* C, int M, int N, int K) {
    (void)M;  /* Must be 1 */
    
    /* Tile size for parallelization */
    const int TILE_N = 256;
    
    #pragma omp parallel for schedule(static)
    for (int n_tile = 0; n_tile < N; n_tile += TILE_N) {
        int n_end = (n_tile + TILE_N < N) ? n_tile + TILE_N : N;
        
        for (int n = n_tile; n < n_end; n++) {
            const int8_t* B_row = B->weights + n * B->cols;
            float scale = B->scales[n] * 0.0625f;
            
            /* 4 accumulators for ILP */
            __m256 acc0 = _mm256_setzero_ps();
            __m256 acc1 = _mm256_setzero_ps();
            __m256 acc2 = _mm256_setzero_ps();
            __m256 acc3 = _mm256_setzero_ps();
            
            int k = 0;
            /* Process 128 K values at a time (4x32) */
            for (; k <= K - 128; k += 128) {
                /* Prefetch 4 cache lines ahead */
                _mm_prefetch((const char*)(A + k + 256), _MM_HINT_T0);
                _mm_prefetch((const char*)(B_row + k + 256), _MM_HINT_T0);
                
                /* Block 0 (k+0 to k+31) */
                __m256 a00 = _mm256_loadu_ps(A + k + 0);
                __m256 a01 = _mm256_loadu_ps(A + k + 8);
                __m256 a02 = _mm256_loadu_ps(A + k + 16);
                __m256 a03 = _mm256_loadu_ps(A + k + 24);
                
                __m256i b0 = _mm256_loadu_si256((__m256i*)(B_row + k));
                __m256i b0_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b0));
                __m256i b0_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b0, 1));
                
                acc0 = _mm256_fmadd_ps(a00, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_lo))), acc0);
                acc1 = _mm256_fmadd_ps(a01, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_lo, 1))), acc1);
                acc2 = _mm256_fmadd_ps(a02, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b0_hi))), acc2);
                acc3 = _mm256_fmadd_ps(a03, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b0_hi, 1))), acc3);
                
                /* Block 1 (k+32 to k+63) */
                __m256 a10 = _mm256_loadu_ps(A + k + 32);
                __m256 a11 = _mm256_loadu_ps(A + k + 40);
                __m256 a12 = _mm256_loadu_ps(A + k + 48);
                __m256 a13 = _mm256_loadu_ps(A + k + 56);
                
                __m256i b1 = _mm256_loadu_si256((__m256i*)(B_row + k + 32));
                __m256i b1_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b1));
                __m256i b1_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b1, 1));
                
                acc0 = _mm256_fmadd_ps(a10, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_lo))), acc0);
                acc1 = _mm256_fmadd_ps(a11, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_lo, 1))), acc1);
                acc2 = _mm256_fmadd_ps(a12, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b1_hi))), acc2);
                acc3 = _mm256_fmadd_ps(a13, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b1_hi, 1))), acc3);
                
                /* Block 2 (k+64 to k+95) */
                __m256 a20 = _mm256_loadu_ps(A + k + 64);
                __m256 a21 = _mm256_loadu_ps(A + k + 72);
                __m256 a22 = _mm256_loadu_ps(A + k + 80);
                __m256 a23 = _mm256_loadu_ps(A + k + 88);
                
                __m256i b2 = _mm256_loadu_si256((__m256i*)(B_row + k + 64));
                __m256i b2_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b2));
                __m256i b2_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b2, 1));
                
                acc0 = _mm256_fmadd_ps(a20, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b2_lo))), acc0);
                acc1 = _mm256_fmadd_ps(a21, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b2_lo, 1))), acc1);
                acc2 = _mm256_fmadd_ps(a22, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b2_hi))), acc2);
                acc3 = _mm256_fmadd_ps(a23, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b2_hi, 1))), acc3);
                
                /* Block 3 (k+96 to k+127) */
                __m256 a30 = _mm256_loadu_ps(A + k + 96);
                __m256 a31 = _mm256_loadu_ps(A + k + 104);
                __m256 a32 = _mm256_loadu_ps(A + k + 112);
                __m256 a33 = _mm256_loadu_ps(A + k + 120);
                
                __m256i b3 = _mm256_loadu_si256((__m256i*)(B_row + k + 96));
                __m256i b3_lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b3));
                __m256i b3_hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b3, 1));
                
                acc0 = _mm256_fmadd_ps(a30, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b3_lo))), acc0);
                acc1 = _mm256_fmadd_ps(a31, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b3_lo, 1))), acc1);
                acc2 = _mm256_fmadd_ps(a32, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(b3_hi))), acc2);
                acc3 = _mm256_fmadd_ps(a33, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(b3_hi, 1))), acc3);
            }
            
            /* Process remaining 32-value blocks */
            for (; k <= K - 32; k += 32) {
                __m256 a0 = _mm256_loadu_ps(A + k);
                __m256 a1 = _mm256_loadu_ps(A + k + 8);
                __m256 a2 = _mm256_loadu_ps(A + k + 16);
                __m256 a3 = _mm256_loadu_ps(A + k + 24);
                
                __m256i b = _mm256_loadu_si256((__m256i*)(B_row + k));
                __m256i blo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(b));
                __m256i bhi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(b, 1));
                
                acc0 = _mm256_fmadd_ps(a0, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(blo))), acc0);
                acc1 = _mm256_fmadd_ps(a1, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(blo, 1))), acc1);
                acc2 = _mm256_fmadd_ps(a2, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(bhi))), acc2);
                acc3 = _mm256_fmadd_ps(a3, _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(bhi, 1))), acc3);
            }
            
            /* Sum accumulators */
            __m256 acc = _mm256_add_ps(acc0, acc1);
            acc = _mm256_add_ps(acc, acc2);
            acc = _mm256_add_ps(acc, acc3);
            
            /* Horizontal sum */
            __m128 sum_low = _mm256_castps256_ps128(acc);
            __m128 sum_high = _mm256_extractf128_ps(acc, 1);
            sum_low = _mm_add_ps(sum_low, sum_high);
            sum_low = _mm_hadd_ps(sum_low, sum_low);
            sum_low = _mm_hadd_ps(sum_low, sum_low);
            float sum = _mm_cvtss_f32(sum_low);
            
            /* Scalar remainder */
            for (; k < K; k++) {
                sum += A[k] * B_row[k];
            }
            
            C[n] = sum * scale;
        }
    }
}

#else /* No AVX2 */

void matmul_dequantized_best(const float* A, const dequantized_tensor_t* B,
                              float* C, int M, int N, int K) {
    matmul_dequantized(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
