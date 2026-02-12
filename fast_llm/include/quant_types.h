/*
 * Quantization Types and Structures
 * Q2, Q4, Q8 quantization formats
 */

#ifndef QUANT_TYPES_H
#define QUANT_TYPES_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Block sizes for quantization */
#define Q2_BLOCK_SIZE 256
#define Q4_BLOCK_SIZE 32
#define Q8_BLOCK_SIZE 32

/* Q2 quantized block (2-bit weights)
 * 256 weights = 64 bytes (packed 4 per byte)
 * 1 scale per block (float)
 * Total: 64 bytes + 4 bytes = 68 bytes for 256 weights
 */
typedef struct {
    uint8_t qs[64];        /* Packed 2-bit weights: 4 values per byte */
    float scale;           /* Quantization scale */
    float zero_point;      /* Zero point (for asymmetric) */
} block_q2_t;

/* Q4 quantized block (4-bit weights)
 * 32 weights = 16 bytes
 * 1 scale per block
 */
typedef struct {
    uint8_t qs[16];        /* Packed 4-bit weights: 2 values per byte */
    float scale;
    float zero_point;
} block_q4_t;

/* Q8 quantized block (8-bit weights)
 * 32 weights = 32 bytes
 * 1 scale per block
 */
typedef struct {
    int8_t qs[32];         /* 8-bit weights */
    float scale;
} block_q8_t;

/* Quantized tensor */
typedef struct {
    void* blocks;          /* Pointer to quantized blocks */
    int n_blocks;          /* Number of blocks */
    int rows;              /* Original matrix rows */
    int cols;              /* Original matrix cols */
    int bits;              /* Quantization bits (2, 4, or 8) */
} quantized_tensor_t;

/* Matrix dimensions */
typedef struct {
    int m;                 /* Rows of A, Rows of C */
    int n;                 /* Cols of B, Cols of C */
    int k;                 /* Cols of A, Rows of B */
} matmul_dims_t;

/* Create quantized tensor */
quantized_tensor_t* create_q2_tensor(int rows, int cols);
quantized_tensor_t* create_q4_tensor(int rows, int cols);
quantized_tensor_t* create_q8_tensor(int rows, int cols);

/* Free quantized tensor */
void free_quantized_tensor(quantized_tensor_t* tensor);

/* Quantize float32 to Q2/Q4/Q8 */
void quantize_f32_to_q2(const float* src, block_q2_t* dst, int n);
void quantize_f32_to_q4(const float* src, block_q4_t* dst, int n);
void quantize_f32_to_q8(const float* src, block_q8_t* dst, int n);

/* Dequantize Q2/Q4/Q8 to float32 */
void dequantize_q2_to_f32(const block_q2_t* src, float* dst, int n);
void dequantize_q4_to_f32(const block_q4_t* src, float* dst, int n);
void dequantize_q8_to_f32(const block_q8_t* src, float* dst, int n);

#ifdef __cplusplus
}
#endif

#endif /* QUANT_TYPES_H */
