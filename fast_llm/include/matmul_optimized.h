/*
 * Optimized Matrix Multiplication Kernels
 * High-performance implementations for 50+ tok/sec target
 */

#ifndef MATMUL_OPTIMIZED_H
#define MATMUL_OPTIMIZED_H

#include <stdint.h>
#include <stddef.h>
#include "dequantized_tensor.h"
#include "ggml_quants.h"

#ifdef __cplusplus
extern "C" {
#endif

/* ============================================================================
 * Q4_K 4-bit Quantized Matmul (50% memory bandwidth reduction)
 * ============================================================================ */

/* 
 * Optimized Q4_K matrix-vector multiplication
 * Uses on-the-fly dequantization with aggressive prefetching
 * 
 * Parameters:
 *   n - input dimension (K)
 *   m - output dimension (N)
 *   s - output vector [m]
 *   vx - Q4_K quantized weights [m, n/2] as block_q4_K array
 *   y - input vector [n]
 */
void matmul_q4_K_optimized(int n, int m, float* s, const void* vx, const float* y);
void matmul_q4_K_avx2(int n, int m, float* s, const void* vx, const float* y);
void matmul_q4_K_avx2_v2(int n, int m, float* s, const void* vx, const float* y);

/* ============================================================================
 * Advanced Prefetching Matmul
 * ============================================================================ */

/*
 * Matmul with multi-level cache prefetching (L1/L2/L3)
 * Optimized for memory-bound inference on DDR4-3200
 * 
 * Expected improvement: 20-40% over basic implementation
 */
void matmul_dequantized_prefetch_optimized(
    const float* A,
    const dequantized_tensor_t* B,
    float* C,
    int M, int N, int K
);

/*
 * Streaming matmul for large matrices
 * Uses non-temporal prefetching to avoid cache pollution
 */
void matmul_dequantized_streaming(
    const float* A,
    const dequantized_tensor_t* B,
    float* C,
    int M, int N, int K
);

/*
 * VNNI-style INT8 matmul (2x throughput vs FP32)
 */
void matmul_int8_vnni(
    const float* A,
    const dequantized_tensor_t* B,
    float* C,
    int M, int N, int K
);

void matmul_int8_vnni_prequantized(
    const int8_t* A_i8,
    float A_scale,
    const dequantized_tensor_t* B,
    float* C,
    int M, int N, int K
);

/*
 * Ultra-optimized kernels for bandwidth saturation
 */
void matmul_int8_ultra(
    const float* A,
    const dequantized_tensor_t* B,
    float* C,
    int M, int N, int K
);

void matmul_int8_super(
    const float* A,
    const dequantized_tensor_t* B,
    float* C,
    int M, int N, int K
);

/* ============================================================================
 * Fused Operations (SwiGLU + RMSNorm)
 * ============================================================================ */

/*
 * Fused RMSNorm + SwiGLU forward pass
 * Eliminates intermediate memory round-trips
 * 
 * Formula: SwiGLU(RMSNorm(x)) = Swish(RMSNorm(x) @ W_gate) * (RMSNorm(x) @ W_value)
 */
void fused_rmsnorm_swiglu_forward(
    const float* input,          /* [batch, hidden] */
    const float* w_gate,         /* [hidden, intermediate] */
    const float* w_value,        /* [hidden, intermediate] */
    const float* gamma,          /* [hidden] RMSNorm weights */
    float* output,               /* [batch, intermediate] */
    int batch_size,
    int hidden_size,
    int intermediate_size
);

/*
 * Single-token fused RMSNorm + SwiGLU
 * Optimized for batch=1 (token-by-token generation)
 */
void fused_rmsnorm_swiglu_single(
    const float* x,              /* [hidden] */
    const void* w_gate_q4,       /* Q4_K quantized */
    const void* w_value_q4,      /* Q4_K quantized */
    const float* gamma,          /* [hidden] */
    float* out,                  /* [intermediate] */
    int hidden_size,
    int intermediate_size
);

/*
 * Standalone optimized RMSNorm
 */
void rmsnorm_forward_optimized(
    const float* input,
    const float* gamma,
    float* output,
    int batch_size,
    int hidden_size
);

/*
 * Standalone optimized SwiGLU
 */
void swiglu_forward_optimized(
    const float* input,
    const float* w_gate,
    const float* w_value,
    float* output,
    int batch_size,
    int hidden_size,
    int intermediate_size
);

/* ============================================================================
 * Dispatch Functions (Auto-select best implementation)
 * ============================================================================ */

/*
 * Auto-dispatch to best matmul implementation based on:
 * - Matrix dimensions
 * - Hardware capabilities (AVX2, etc.)
 * - Memory pressure
 */
void matmul_optimized_dispatch(
    const float* A,
    const dequantized_tensor_t* B,
    float* C,
    int M, int N, int K
);

/*
 * Auto-dispatch for Q4_K quantized matmul
 */
void matmul_q4_K_dispatch(
    int n, int m, float* s,
    const void* vx, const float* y
);

/* ============================================================================
 * Configuration and Tuning
 * ============================================================================ */

typedef struct {
    int use_prefetching;         /* Enable software prefetching */
    int prefetch_distance_l1;    /* L1 prefetch distance (cache lines) */
    int prefetch_distance_l2;    /* L2 prefetch distance */
    int prefetch_distance_l3;    /* L3 prefetch distance */
    int use_fused_ops;           /* Enable fused kernels */
    int use_q4_k;                /* Enable Q4_K quantization */
    int num_threads;             /* OpenMP threads */
} matmul_config_t;

/* Get default configuration for current hardware */
matmul_config_t matmul_get_default_config(void);

/* Set global configuration */
void matmul_set_config(const matmul_config_t* config);

/* Get current configuration */
const matmul_config_t* matmul_get_config(void);

#ifdef __cplusplus
}
#endif

#endif /* MATMUL_OPTIMIZED_H */
