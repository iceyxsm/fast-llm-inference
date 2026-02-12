/*
 * AVX-512 Optimized Matrix Multiplication Kernels
 * Requires AVX-512F, AVX-512BW, AVX-512VL
 */

#include "matmul.h"

#ifdef __AVX512F__
#include <immintrin.h>
#include <string.h>

/* 
 * Q8 matmul with AVX-512
 * Process 16 floats at a time (512-bit vectors)
 */
void q8_matmul_avx512(const float* A, const quantized_tensor_t* B_q, float* C,
                      int M, int N, int K) {
    const block_q8_t* B = (const block_q8_t*)B_q->blocks;
    int blocks_per_row = (K + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        const float* A_row = A + m * K;
        
        for (int n = 0; n < N; n++) {
            __m512 sum_vec = _mm512_setzero_ps();
            
            int k = 0;
            /* Process 16 elements at a time */
            for (; k <= K - 16; k += 16) {
                /* Load 16 floats from A */
                __m512 a_vec = _mm512_loadu_ps(A_row + k);
                
                /* Load and dequantize 16 int8 weights from B */
                int8_t weights[16];
                float scales[16];
                
                for (int i = 0; i < 16; i++) {
                    int block_idx = n * blocks_per_row + (k + i) / Q8_BLOCK_SIZE;
                    int offset = (k + i) % Q8_BLOCK_SIZE;
                    weights[i] = B[block_idx].qs[offset];
                    scales[i] = B[block_idx].scale;
                }
                
                /* Convert int8 to int32, then to float */
                __m128i w_i8 = _mm_loadu_si128((__m128i*)weights);
                __m256i w_i16 = _mm256_cvtepi8_epi16(w_i8);
                __m512i w_i32 = _mm512_cvtepi16_epi32(w_i16);
                __m512 w_f32 = _mm512_cvtepi32_ps(w_i32);
                
                /* Apply scales (per-element scale) */
                __m512 scale_vec = _mm512_loadu_ps(scales);
                w_f32 = _mm512_mul_ps(w_f32, scale_vec);
                
                /* Multiply-accumulate: sum += A * W */
                sum_vec = _mm512_fmadd_ps(a_vec, w_f32, sum_vec);
            }
            
            /* Horizontal sum of the 512-bit vector */
            float sum = _mm512_reduce_add_ps(sum_vec);
            
            /* Handle remaining elements */
            for (; k < K; k++) {
                int block_idx = n * blocks_per_row + k / Q8_BLOCK_SIZE;
                int offset = k % Q8_BLOCK_SIZE;
                float w = B[block_idx].qs[offset] * B[block_idx].scale;
                sum += A_row[k] * w;
            }
            
            C[m * N + n] = sum;
        }
    }
}

/* 
 * Q4 matmul with AVX-512
 * Unpack 4-bit weights on the fly
 */
void q4_matmul_avx512(const float* A, const quantized_tensor_t* B_q, float* C,
                      int M, int N, int K) {
    const block_q4_t* B = (const block_q4_t*)B_q->blocks;
    int blocks_per_row = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        const float* A_row = A + m * K;
        
        for (int n = 0; n < N; n++) {
            __m512 sum_vec = _mm512_setzero_ps();
            
            int k = 0;
            /* Process 16 elements at a time */
            for (; k <= K - 16; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(A_row + k);
                
                /* Unpack 16 4-bit weights */
                float weights[16];
                for (int i = 0; i < 16; i++) {
                    int block_idx = n * blocks_per_row + (k + i) / Q4_BLOCK_SIZE;
                    int offset = (k + i) % Q4_BLOCK_SIZE;
                    const block_q4_t* block = &B[block_idx];
                    
                    int byte_idx = offset / 2;
                    int nibble = offset % 2;
                    int q = (block->qs[byte_idx] >> (nibble * 4)) & 0xF;
                    
                    weights[i] = block->zero_point + q * block->scale;
                }
                
                __m512 w_vec = _mm512_loadu_ps(weights);
                sum_vec = _mm512_fmadd_ps(a_vec, w_vec, sum_vec);
            }
            
            float sum = _mm512_reduce_add_ps(sum_vec);
            
            /* Handle remaining elements */
            for (; k < K; k++) {
                int block_idx = n * blocks_per_row + k / Q4_BLOCK_SIZE;
                int offset = k % Q4_BLOCK_SIZE;
                const block_q4_t* block = &B[block_idx];
                
                int byte_idx = offset / 2;
                int nibble = offset % 2;
                int q = (block->qs[byte_idx] >> (nibble * 4)) & 0xF;
                float w = block->zero_point + q * block->scale;
                
                sum += A_row[k] * w;
            }
            
            C[m * N + n] = sum;
        }
    }
}

/* 
 * Q2 matmul with AVX-512
 * Unpack 2-bit weights on the fly
 */
void q2_matmul_avx512(const float* A, const quantized_tensor_t* B_q, float* C,
                      int M, int N, int K) {
    const block_q2_t* B = (const block_q2_t*)B_q->blocks;
    int blocks_per_row = (K + Q2_BLOCK_SIZE - 1) / Q2_BLOCK_SIZE;
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        const float* A_row = A + m * K;
        
        for (int n = 0; n < N; n++) {
            __m512 sum_vec = _mm512_setzero_ps();
            
            int k = 0;
            /* Process 16 elements at a time */
            for (; k <= K - 16; k += 16) {
                __m512 a_vec = _mm512_loadu_ps(A_row + k);
                
                /* Unpack 16 2-bit weights */
                float weights[16];
                for (int i = 0; i < 16; i++) {
                    int block_idx = n * blocks_per_row + (k + i) / Q2_BLOCK_SIZE;
                    int offset = (k + i) % Q2_BLOCK_SIZE;
                    const block_q2_t* block = &B[block_idx];
                    
                    int byte_idx = offset / 4;
                    int shift = (offset % 4) * 2;
                    int q = (block->qs[byte_idx] >> shift) & 0x3;
                    
                    weights[i] = block->zero_point + q * block->scale;
                }
                
                __m512 w_vec = _mm512_loadu_ps(weights);
                sum_vec = _mm512_fmadd_ps(a_vec, w_vec, sum_vec);
            }
            
            float sum = _mm512_reduce_add_ps(sum_vec);
            
            /* Handle remaining elements */
            for (; k < K; k++) {
                int block_idx = n * blocks_per_row + k / Q2_BLOCK_SIZE;
                int offset = k % Q2_BLOCK_SIZE;
                const block_q2_t* block = &B[block_idx];
                
                int byte_idx = offset / 4;
                int shift = (offset % 4) * 2;
                int q = (block->qs[byte_idx] >> shift) & 0x3;
                float w = block->zero_point + q * block->scale;
                
                sum += A_row[k] * w;
            }
            
            C[m * N + n] = sum;
        }
    }
}

#else /* !__AVX512F__ */

/* Fallback to scalar if AVX-512 not available at compile time */
void q8_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K) {
    q8_matmul(A, B, C, M, N, K);
}

void q4_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K) {
    q4_matmul(A, B, C, M, N, K);
}

void q2_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K) {
    q2_matmul(A, B, C, M, N, K);
}

#endif /* __AVX512F__ */
