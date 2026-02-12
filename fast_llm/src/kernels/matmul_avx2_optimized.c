/*
 * Highly Optimized AVX2 Kernels
 * Pre-dequantize weights or use lookup tables
 */

#include "matmul.h"

#ifdef __AVX2__
#include <immintrin.h>
#include <string.h>

/* 
 * Optimized Q4 matmul with weight pre-expansion
 * Expand 4-bit weights to float32 in blocks
 */
void q4_matmul_avx2_optimized(const float* A, const quantized_tensor_t* B_q, float* C,
                              int M, int N, int K) {
    const block_q4_t* B = (const block_q4_t*)B_q->blocks;
    int blocks_per_row = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    
    /* Temporary buffer for dequantized weights */
    /* In production, this should be thread-local and reused */
    float* w_dequant = (float*)malloc(K * sizeof(float));
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int n = 0; n < N; n++) {
        /* Pre-dequantize entire row */
        for (int k = 0; k < K; k++) {
            int block_idx = n * blocks_per_row + k / Q4_BLOCK_SIZE;
            int offset = k % Q4_BLOCK_SIZE;
            const block_q4_t* block = &B[block_idx];
            
            int byte_idx = offset / 2;
            int nibble = offset % 2;
            int q = (block->qs[byte_idx] >> (nibble * 4)) & 0xF;
            
            w_dequant[k] = block->zero_point + q * block->scale;
        }
        
        /* Now do fast matmul with dequantized weights */
        for (int m = 0; m < M; m++) {
            const float* A_row = A + m * K;
            
            __m256 sum_vec = _mm256_setzero_ps();
            int k = 0;
            
            /* Main loop - 8 floats at a time */
            for (; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(A_row + k);
                __m256 w_vec = _mm256_loadu_ps(w_dequant + k);
                sum_vec = _mm256_fmadd_ps(a_vec, w_vec, sum_vec);
            }
            
            /* Horizontal sum */
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            float sum = _mm_cvtss_f32(sum_128);
            
            /* Remainder */
            for (; k < K; k++) {
                sum += A_row[k] * w_dequant[k];
            }
            
            C[m * N + n] = sum;
        }
    }
    
    free(w_dequant);
}

/* 
 * Even more optimized: Process multiple N values at once
 * This improves cache utilization
 */
void q4_matmul_avx2_blocked(const float* A, const quantized_tensor_t* B_q, float* C,
                            int M, int N, int K) {
    const block_q4_t* B = (const block_q4_t*)B_q->blocks;
    int blocks_per_row = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    
    /* Block size for N dimension */
    const int N_BLOCK = 4;
    
    memset(C, 0, M * N * sizeof(float));
    
    /* Temporary buffer for dequantized weights */
    float* w_block = (float*)malloc(N_BLOCK * K * sizeof(float));
    
    for (int n_block = 0; n_block < N; n_block += N_BLOCK) {
        int n_end = (n_block + N_BLOCK < N) ? n_block + N_BLOCK : N;
        int n_count = n_end - n_block;
        
        /* Pre-dequantize block of N rows */
        for (int n = 0; n < n_count; n++) {
            for (int k = 0; k < K; k++) {
                int block_idx = (n_block + n) * blocks_per_row + k / Q4_BLOCK_SIZE;
                int offset = k % Q4_BLOCK_SIZE;
                const block_q4_t* block = &B[block_idx];
                
                int byte_idx = offset / 2;
                int nibble = offset % 2;
                int q = (block->qs[byte_idx] >> (nibble * 4)) & 0xF;
                
                w_block[n * K + k] = block->zero_point + q * block->scale;
            }
        }
        
        /* Compute matmul for this block */
        for (int m = 0; m < M; m++) {
            const float* A_row = A + m * K;
            
            for (int n = 0; n < n_count; n++) {
                float* w_row = w_block + n * K;
                
                __m256 sum_vec = _mm256_setzero_ps();
                int k = 0;
                
                for (; k <= K - 8; k += 8) {
                    __m256 a_vec = _mm256_loadu_ps(A_row + k);
                    __m256 w_vec = _mm256_loadu_ps(w_row + k);
                    sum_vec = _mm256_fmadd_ps(a_vec, w_vec, sum_vec);
                }
                
                __m128 sum_low = _mm256_castps256_ps128(sum_vec);
                __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
                __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                sum_128 = _mm_hadd_ps(sum_128, sum_128);
                float sum = _mm_cvtss_f32(sum_128);
                
                for (; k < K; k++) {
                    sum += A_row[k] * w_row[k];
                }
                
                C[m * N + (n_block + n)] = sum;
            }
        }
    }
    
    free(w_block);
}

#endif /* __AVX2__ */
