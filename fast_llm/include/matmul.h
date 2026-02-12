/*
 * Matrix Multiplication Kernels
 * Optimized for quantized weights
 */

#ifndef MATMUL_H
#define MATMUL_H

#include "quant_types.h"
#include "cpu_features.h"

/* Portable aligned allocation */
#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#include <stdlib.h>
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* 
 * Q2 quantized matrix multiplication
 * C = A @ B^T where B is Q2 quantized
 * A: [M, K] float32
 * B: [N, K] Q2 quantized (stored as block_q2_t[N * K / Q2_BLOCK_SIZE])
 * C: [M, N] float32
 */
void q2_matmul(const float* A, const quantized_tensor_t* B, float* C,
               int M, int N, int K);

/* Q4 quantized matrix multiplication */
void q4_matmul(const float* A, const quantized_tensor_t* B, float* C,
               int M, int N, int K);

/* Q8 quantized matrix multiplication */
void q8_matmul(const float* A, const quantized_tensor_t* B, float* C,
               int M, int N, int K);

/* 
 * AVX-512 optimized versions
 */
void q2_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K);

void q4_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K);

void q8_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K);

/* 
 * AVX-512 VNNI (Vector Neural Network Instructions)
 * Faster for int8 operations
 */
void q8_matmul_avx512vnni(const float* A, const quantized_tensor_t* B, float* C,
                          int M, int N, int K);

/*
 * Intel AMX (Advanced Matrix Extensions)
 * Only on 4th+ Gen Xeon
 */
void q2_matmul_amx(const float* A, const quantized_tensor_t* B, float* C,
                   int M, int N, int K);

void q8_matmul_amx(const float* A, const quantized_tensor_t* B, float* C,
                   int M, int N, int K);

/* 
 * Dispatch to best kernel based on CPU features
 */
typedef void (*matmul_fn_t)(const float*, const quantized_tensor_t*, float*, int, int, int);

matmul_fn_t select_q2_kernel(const cpu_features_t* features);
matmul_fn_t select_q4_kernel(const cpu_features_t* features);
matmul_fn_t select_q8_kernel(const cpu_features_t* features);

/* 
 * AVX2 optimized kernels
 */
void q2_matmul_avx2(const float* A, const quantized_tensor_t* B, float* C,
                    int M, int N, int K);
void q4_matmul_avx2(const float* A, const quantized_tensor_t* B, float* C,
                    int M, int N, int K);
void q8_matmul_avx2(const float* A, const quantized_tensor_t* B, float* C,
                    int M, int N, int K);
void q4_matmul_single_token(const float* A, const quantized_tensor_t* B, float* C,
                            int M, int N, int K);

/* 
 * AVX-512 stubs (fallback to AVX2)
 */
void q2_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K);
void q4_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K);
void q8_matmul_avx512(const float* A, const quantized_tensor_t* B, float* C,
                      int M, int N, int K);

#ifdef __cplusplus
}
#endif

#endif /* MATMUL_H */
