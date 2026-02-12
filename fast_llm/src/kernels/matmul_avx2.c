/*
 * AVX2 Optimized Matrix Multiplication Kernels
 * Available on most modern CPUs (Intel Haswell+, AMD Ryzen+)
 * 256-bit vectors, 2x faster than AVX, 8x faster than SSE
 */

#include "matmul.h"

#ifdef __AVX2__
#include <immintrin.h>
#include <string.h>

/*
 * Q8 matmul with AVX2
 * Process 8 floats at a time (256-bit vectors)
 */
void q8_matmul_avx2(const float* A, const quantized_tensor_t* B_q, float* C,
                    int M, int N, int K) {
    const block_q8_t* B = (const block_q8_t*)B_q->blocks;
    int blocks_per_row = (K + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        const float* A_row = A + m * K;
        
        for (int n = 0; n < N; n++) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            int k = 0;
            /* Process 8 elements at a time */
            for (; k <= K - 8; k += 8) {
                /* Load 8 floats from A */
                __m256 a_vec = _mm256_loadu_ps(A_row + k);
                
                /* Load and dequantize 8 int8 weights */
                float weights[8];
                for (int i = 0; i < 8; i++) {
                    int block_idx = n * blocks_per_row + (k + i) / Q8_BLOCK_SIZE;
                    int offset = (k + i) % Q8_BLOCK_SIZE;
                    weights[i] = B[block_idx].qs[offset] * B[block_idx].scale;
                }
                
                __m256 w_vec = _mm256_loadu_ps(weights);
                sum_vec = _mm256_fmadd_ps(a_vec, w_vec, sum_vec);
            }
            
            /* Horizontal sum of the 256-bit vector */
            /* Extract high and low 128-bit halves */
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            
            /* Add them together */
            __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
            
            /* Horizontal sum within 128-bit register */
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            
            float sum = _mm_cvtss_f32(sum_128);
            
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
 * Q4 matmul with AVX2
 */
void q4_matmul_avx2(const float* A, const quantized_tensor_t* B_q, float* C,
                    int M, int N, int K) {
    const block_q4_t* B = (const block_q4_t*)B_q->blocks;
    int blocks_per_row = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        const float* A_row = A + m * K;
        
        for (int n = 0; n < N; n++) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(A_row + k);
                
                /* Unpack 8 4-bit weights */
                float weights[8];
                for (int i = 0; i < 8; i++) {
                    int block_idx = n * blocks_per_row + (k + i) / Q4_BLOCK_SIZE;
                    int offset = (k + i) % Q4_BLOCK_SIZE;
                    const block_q4_t* block = &B[block_idx];
                    
                    int byte_idx = offset / 2;
                    int nibble = offset % 2;
                    int q = (block->qs[byte_idx] >> (nibble * 4)) & 0xF;
                    
                    weights[i] = block->zero_point + q * block->scale;
                }
                
                __m256 w_vec = _mm256_loadu_ps(weights);
                sum_vec = _mm256_fmadd_ps(a_vec, w_vec, sum_vec);
            }
            
            /* Horizontal sum */
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            float sum = _mm_cvtss_f32(sum_128);
            
            /* Handle remaining */
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
 * Q2 matmul with AVX2
 */
void q2_matmul_avx2(const float* A, const quantized_tensor_t* B_q, float* C,
                    int M, int N, int K) {
    const block_q2_t* B = (const block_q2_t*)B_q->blocks;
    int blocks_per_row = (K + Q2_BLOCK_SIZE - 1) / Q2_BLOCK_SIZE;
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        const float* A_row = A + m * K;
        
        for (int n = 0; n < N; n++) {
            __m256 sum_vec = _mm256_setzero_ps();
            
            int k = 0;
            for (; k <= K - 8; k += 8) {
                __m256 a_vec = _mm256_loadu_ps(A_row + k);
                
                /* Unpack 8 2-bit weights */
                float weights[8];
                for (int i = 0; i < 8; i++) {
                    int block_idx = n * blocks_per_row + (k + i) / Q2_BLOCK_SIZE;
                    int offset = (k + i) % Q2_BLOCK_SIZE;
                    const block_q2_t* block = &B[block_idx];
                    
                    int byte_idx = offset / 4;
                    int shift = (offset % 4) * 2;
                    int q = (block->qs[byte_idx] >> shift) & 0x3;
                    
                    weights[i] = block->zero_point + q * block->scale;
                }
                
                __m256 w_vec = _mm256_loadu_ps(weights);
                sum_vec = _mm256_fmadd_ps(a_vec, w_vec, sum_vec);
            }
            
            /* Horizontal sum */
            __m128 sum_low = _mm256_castps256_ps128(sum_vec);
            __m128 sum_high = _mm256_extractf128_ps(sum_vec, 1);
            __m128 sum_128 = _mm_add_ps(sum_low, sum_high);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            sum_128 = _mm_hadd_ps(sum_128, sum_128);
            float sum = _mm_cvtss_f32(sum_128);
            
            /* Handle remaining */
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

/* Alias for single-token inference (optimized for M=1) */
void q4_matmul_single_token(const float* A, const quantized_tensor_t* B, float* C,
                            int M, int N, int K) {
    /* For now, use the standard AVX2 implementation */
    q4_matmul_avx2(A, B, C, M, N, K);
}

/* Stubs for AVX-512 functions (fallback to AVX2) */
void q2_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K) {
    q2_matmul_avx2(A, B, C, M, N, K);
}

void q4_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K) {
    q4_matmul_avx2(A, B, C, M, N, K);
}

void q8_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K) {
    q8_matmul_avx2(A, B, C, M, N, K);
}

#else /* !__AVX2__ */

/* Fallback to scalar if AVX2 not available at compile time */
void q8_matmul_avx2(const float* A, const quantized_tensor_t* B, float* C,
                    int M, int N, int K) {
    q8_matmul(A, B, C, M, N, K);
}

void q4_matmul_avx2(const float* A, const quantized_tensor_t* B, float* C,
                    int M, int N, int K) {
    q4_matmul(A, B, C, M, N, K);
}

void q2_matmul_avx2(const float* A, const quantized_tensor_t* B, float* C,
                    int M, int N, int K) {
    q2_matmul(A, B, C, M, N, K);
}

/* Stubs for non-AVX2 builds */
void q4_matmul_single_token(const float* A, const quantized_tensor_t* B, float* C,
                            int M, int N, int K) {
    q4_matmul(A, B, C, M, N, K);
}

void q2_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K) {
    q2_matmul(A, B, C, M, N, K);
}

void q4_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K) {
    q4_matmul(A, B, C, M, N, K);
}

void q8_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K) {
    q8_matmul(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
