/*
 * Dequantized Tensor - Pre-dequantized weights for fast inference
 * 
 * Research-based implementation:
 * - Pre-dequantize Q4 -> INT8 at model load time (llama.cpp approach)
 * - Use _mm256_maddubs_epi16 for fast int8 dot products (AVX2)
 * - Reference: https://stackoverflow.com/questions/51382276/
 * - Reference: ggml-quants.c from llama.cpp
 */

#ifndef DEQUANTIZED_TENSOR_H
#define DEQUANTIZED_TENSOR_H

#include <stdint.h>
#include <stddef.h>
#include "quant_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Pre-dequantized INT8 tensor
 * Stores weights as int8 for fast dot products
 * Scale is applied after accumulation
 */
typedef struct {
    int8_t* weights;       /* [rows, cols] int8 weights, row-major */
    float*  scales;        /* [rows] per-row scales */
    int     rows;
    int     cols;
    /* Original quantization info for reference */
    int     original_bits;
} dequantized_tensor_t;

/* Create dequantized tensor from Q4 */
dequantized_tensor_t* dequantized_from_q4(const quantized_tensor_t* q4_tensor);

/* Create dequantized tensor from Q8 */
dequantized_tensor_t* dequantized_from_q8(const quantized_tensor_t* q8_tensor);

/* Free dequantized tensor */
void dequantized_tensor_free(dequantized_tensor_t* tensor);

/* 
 * Fast matmul with dequantized int8 weights
 * C[M, N] = A[M, K] @ B[N, K]^T where B is pre-dequantized int8
 */
void matmul_dequantized(const float* A, const dequantized_tensor_t* B, 
                        float* C, int M, int N, int K);

/* AVX2 optimized version */
void matmul_dequantized_avx2(const float* A, const dequantized_tensor_t* B,
                             float* C, int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif /* DEQUANTIZED_TENSOR_H */
