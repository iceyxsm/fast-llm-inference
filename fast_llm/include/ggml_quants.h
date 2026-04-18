/*
 * GGML Quantization Kernels
 * Native Q4_K, Q5_K, Q6_K dot products (no dequantization)
 * Based on llama.cpp ggml-quants.c
 */

#ifndef GGML_QUANTS_H
#define GGML_QUANTS_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Q4_K block - 256 weights per block */
/* Based on llama.cpp ggml-common.h */
typedef struct {
    uint16_t d;            /* super-block scale (f16) */
    uint16_t dmin;         /* super-block min (f16) */
    uint8_t scales[12];    /* 6-bit scales and mins packed */
    uint8_t qs[128];       /* 256 4-bit values */
} block_q4_K;

/* Q5_K block */
typedef struct {
    uint16_t d;            /* super-block scale (f16) */
    uint16_t dmin;         /* super-block min (f16) */
    uint8_t scales[12];    /* 6-bit scales and mins packed */
    uint8_t qh[32];        /* 256 high bits */
    uint8_t qs[128];       /* 256 low 4 bits */
} block_q5_K;

/* Q6_K block */
typedef struct {
    uint8_t ql[128];       /* quants, lower 4 bits */
    uint8_t qh[64];        /* quants, upper 2 bits */
    int8_t  scales[16];    /* scales, quantized with 8 bits */
    uint16_t d;            /* super-block scale (f16) */
} block_q6_K;

/* 
 * Native Q4_K dot product with float input
 * Computes: sum(input[i] * dequant(q4_k_weights[i]))
 * No intermediate dequantization - scales applied on-the-fly
 * 
 * This is 2-3x faster than dequantize + int8 matmul because:
 * - 4 bits per weight vs 8 bits (2x memory bandwidth)
 * - Fused dequantization + multiply (no separate pass)
 * - Better cache locality
 */
void ggml_vec_dot_q4_K(const int n, float* restrict s,
                       const void* restrict vx,
                       const float* restrict y);

/* Q6_K dot product */
void ggml_vec_dot_q6_K(const int n, float* restrict s,
                       const void* restrict vx,
                       const float* restrict y);

/* Q4_K matrix-vector multiplication */
void ggml_gemv_q4_K(int n, int m, float* s, const void* vx, const float* y);

/* Q6_K matrix-vector multiplication */
void ggml_gemv_q6_K(int n, int m, float* s, const void* vx, const float* y);

/* Quantize float to Q4_K format (for reference) */
void quantize_row_q4_K(const float* x, void* vy, int k);

#ifdef __cplusplus
}
#endif

#endif /* GGML_QUANTS_H */
