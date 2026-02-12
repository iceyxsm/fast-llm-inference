/*
 * Scalar (Fallback) Matrix Multiplication Kernels
 * Works on any CPU, not optimized
 */

#include "matmul.h"
#include <string.h>

/* Scalar Q2 matmul - reference implementation */
void q2_matmul(const float* A, const quantized_tensor_t* B_q, float* C,
               int M, int N, int K) {
    const block_q2_t* B = (const block_q2_t*)B_q->blocks;
    int blocks_per_row = (K + Q2_BLOCK_SIZE - 1) / Q2_BLOCK_SIZE;
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            
            for (int k = 0; k < K; k++) {
                int block_idx = n * blocks_per_row + k / Q2_BLOCK_SIZE;
                int offset_in_block = k % Q2_BLOCK_SIZE;
                
                const block_q2_t* block = &B[block_idx];
                int byte_idx = offset_in_block / 4;
                int bit_offset = (offset_in_block % 4) * 2;
                int q = (block->qs[byte_idx] >> bit_offset) & 0x3;
                
                float w = block->zero_point + q * block->scale;
                sum += A[m * K + k] * w;
            }
            
            C[m * N + n] = sum;
        }
    }
}

/* Scalar Q4 matmul */
void q4_matmul(const float* A, const quantized_tensor_t* B_q, float* C,
               int M, int N, int K) {
    const block_q4_t* B = (const block_q4_t*)B_q->blocks;
    int blocks_per_row = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            
            for (int k = 0; k < K; k++) {
                int block_idx = n * blocks_per_row + k / Q4_BLOCK_SIZE;
                int offset_in_block = k % Q4_BLOCK_SIZE;
                
                const block_q4_t* block = &B[block_idx];
                int byte_idx = offset_in_block / 2;
                int nibble_offset = (offset_in_block % 2) * 4;
                int q = (block->qs[byte_idx] >> nibble_offset) & 0xF;
                
                float w = block->zero_point + q * block->scale;
                sum += A[m * K + k] * w;
            }
            
            C[m * N + n] = sum;
        }
    }
}

/* Scalar Q8 matmul */
void q8_matmul(const float* A, const quantized_tensor_t* B_q, float* C,
               int M, int N, int K) {
    const block_q8_t* B = (const block_q8_t*)B_q->blocks;
    int blocks_per_row = (K + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
    
    memset(C, 0, M * N * sizeof(float));
    
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            float sum = 0.0f;
            
            for (int k = 0; k < K; k++) {
                int block_idx = n * blocks_per_row + k / Q8_BLOCK_SIZE;
                int offset_in_block = k % Q8_BLOCK_SIZE;
                
                const block_q8_t* block = &B[block_idx];
                float w = block->qs[offset_in_block] * block->scale;
                sum += A[m * K + k] * w;
            }
            
            C[m * N + n] = sum;
        }
    }
}
