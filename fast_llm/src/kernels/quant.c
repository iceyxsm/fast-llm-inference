/*
 * Quantization / Dequantization Implementation
 */

#include "quant_types.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* Create Q2 tensor */
quantized_tensor_t* create_q2_tensor(int rows, int cols) {
    quantized_tensor_t* t = (quantized_tensor_t*)malloc(sizeof(quantized_tensor_t));
    if (!t) return NULL;
    
    int n_blocks = (rows * cols + Q2_BLOCK_SIZE - 1) / Q2_BLOCK_SIZE;
    t->blocks = calloc(n_blocks, sizeof(block_q2_t));
    if (!t->blocks) {
        free(t);
        return NULL;
    }
    
    t->n_blocks = n_blocks;
    t->rows = rows;
    t->cols = cols;
    t->bits = 2;
    
    return t;
}

/* Create Q4 tensor */
quantized_tensor_t* create_q4_tensor(int rows, int cols) {
    quantized_tensor_t* t = (quantized_tensor_t*)malloc(sizeof(quantized_tensor_t));
    if (!t) return NULL;
    
    int n_blocks = (rows * cols + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    t->blocks = calloc(n_blocks, sizeof(block_q4_t));
    if (!t->blocks) {
        free(t);
        return NULL;
    }
    
    t->n_blocks = n_blocks;
    t->rows = rows;
    t->cols = cols;
    t->bits = 4;
    
    return t;
}

/* Create Q8 tensor */
quantized_tensor_t* create_q8_tensor(int rows, int cols) {
    quantized_tensor_t* t = (quantized_tensor_t*)malloc(sizeof(quantized_tensor_t));
    if (!t) return NULL;
    
    int n_blocks = (rows * cols + Q8_BLOCK_SIZE - 1) / Q8_BLOCK_SIZE;
    t->blocks = calloc(n_blocks, sizeof(block_q8_t));
    if (!t->blocks) {
        free(t);
        return NULL;
    }
    
    t->n_blocks = n_blocks;
    t->rows = rows;
    t->cols = cols;
    t->bits = 8;
    
    return t;
}

void free_quantized_tensor(quantized_tensor_t* tensor) {
    if (tensor) {
        free(tensor->blocks);
        free(tensor);
    }
}

/* Quantize float32 to Q2 (2-bit)
 * Values are mapped to: 0, 1, 2, 3
 * With scale and zero_point for asymmetric quantization
 */
void quantize_f32_to_q2(const float* src, block_q2_t* dst, int n) {
    /* Find min/max for this block */
    float min_val = src[0];
    float max_val = src[0];
    for (int i = 1; i < n && i < Q2_BLOCK_SIZE; i++) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }
    
    /* Compute scale and zero_point */
    /* 2 bits = 4 values: 0, 1, 2, 3 */
    float scale = (max_val - min_val) / 3.0f;
    if (scale == 0) scale = 1.0f;  /* Avoid division by zero */
    
    dst->scale = scale;
    dst->zero_point = min_val;
    
    /* Quantize and pack */
    memset(dst->qs, 0, sizeof(dst->qs));
    
    for (int i = 0; i < n && i < Q2_BLOCK_SIZE; i++) {
        /* Map to 0-3 range */
        float normalized = (src[i] - min_val) / scale;
        int q = (int)(normalized + 0.5f);
        if (q > 3) q = 3;
        if (q < 0) q = 0;
        
        /* Pack 4 values per byte (2 bits each) */
        int byte_idx = i / 4;
        int bit_offset = (i % 4) * 2;
        dst->qs[byte_idx] |= (q << bit_offset);
    }
}

/* Quantize float32 to Q4 (4-bit) */
void quantize_f32_to_q4(const float* src, block_q4_t* dst, int n) {
    float min_val = src[0];
    float max_val = src[0];
    for (int i = 1; i < n && i < Q4_BLOCK_SIZE; i++) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }
    
    /* 4 bits = 16 values */
    float scale = (max_val - min_val) / 15.0f;
    if (scale == 0) scale = 1.0f;
    
    dst->scale = scale;
    dst->zero_point = min_val;
    
    memset(dst->qs, 0, sizeof(dst->qs));
    
    for (int i = 0; i < n && i < Q4_BLOCK_SIZE; i++) {
        float normalized = (src[i] - min_val) / scale;
        int q = (int)(normalized + 0.5f);
        if (q > 15) q = 15;
        if (q < 0) q = 0;
        
        /* Pack 2 values per byte (4 bits each) */
        int byte_idx = i / 2;
        int nibble_offset = (i % 2) * 4;
        dst->qs[byte_idx] |= (q << nibble_offset);
    }
}

/* Quantize float32 to Q8 (8-bit) */
void quantize_f32_to_q8(const float* src, block_q8_t* dst, int n) {
    float min_val = src[0];
    float max_val = src[0];
    for (int i = 1; i < n && i < Q8_BLOCK_SIZE; i++) {
        if (src[i] < min_val) min_val = src[i];
        if (src[i] > max_val) max_val = src[i];
    }
    
    /* 8 bits = 256 values, use symmetric quantization around 0 */
    float max_abs = fmaxf(fabsf(min_val), fabsf(max_val));
    float scale = max_abs / 127.0f;
    if (scale == 0) scale = 1.0f;
    
    dst->scale = scale;
    
    for (int i = 0; i < n && i < Q8_BLOCK_SIZE; i++) {
        float normalized = src[i] / scale;
        int q = (int)(normalized);
        if (q > 127) q = 127;
        if (q < -128) q = -128;
        dst->qs[i] = (int8_t)q;
    }
}

/* Dequantize Q2 to float32 */
void dequantize_q2_to_f32(const block_q2_t* src, float* dst, int n) {
    for (int i = 0; i < n && i < Q2_BLOCK_SIZE; i++) {
        int byte_idx = i / 4;
        int bit_offset = (i % 4) * 2;
        int q = (src->qs[byte_idx] >> bit_offset) & 0x3;
        
        dst[i] = src->zero_point + q * src->scale;
    }
}

/* Dequantize Q4 to float32 */
void dequantize_q4_to_f32(const block_q4_t* src, float* dst, int n) {
    for (int i = 0; i < n && i < Q4_BLOCK_SIZE; i++) {
        int byte_idx = i / 2;
        int nibble_offset = (i % 2) * 4;
        int q = (src->qs[byte_idx] >> nibble_offset) & 0xF;
        
        dst[i] = src->zero_point + q * src->scale;
    }
}

/* Dequantize Q8 to float32 */
void dequantize_q8_to_f32(const block_q8_t* src, float* dst, int n) {
    for (int i = 0; i < n && i < Q8_BLOCK_SIZE; i++) {
        dst[i] = src->qs[i] * src->scale;
    }
}
