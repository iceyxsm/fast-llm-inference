/*
 * Fast INT4 Quantization v2
 * Optimized for speed over flexibility
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <immintrin.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

typedef struct {
    uint8_t* weights;   /* Packed: 2 int4 per byte */
    float* scales;      /* Per-row scales */
    int rows;
    int cols;
} int4_matrix_t;

/* Simple scalar INT4 matmul - optimized for clarity and compiler optimization */
void matmul_int4_simple(const float* A, const int4_matrix_t* B, float* C,
                         int M, int N, int K) {
    const uint8_t* w = B->weights;
    const float* s = B->scales;
    
    for (int i = 0; i < M; i++) {
        const float* a_row = A + i * K;
        
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            float scale = s[j];
            int row_offset = j * ((K + 1) / 2);  /* Packed row offset */
            
            /* Unroll by 4 for better performance */
            int k = 0;
            for (; k + 7 < K; k += 8) {
                /* Process 8 elements = 4 packed bytes */
                uint8_t p0 = w[row_offset + k/2];
                uint8_t p1 = w[row_offset + k/2 + 1];
                uint8_t p2 = w[row_offset + k/2 + 2];
                uint8_t p3 = w[row_offset + k/2 + 3];
                
                /* Unpack and accumulate */
                sum += a_row[k+0] * ((float)((p0 >> 4) & 0xF) - 8.0f) * scale;
                sum += a_row[k+1] * ((float)((p0 >> 0) & 0xF) - 8.0f) * scale;
                sum += a_row[k+2] * ((float)((p1 >> 4) & 0xF) - 8.0f) * scale;
                sum += a_row[k+3] * ((float)((p1 >> 0) & 0xF) - 8.0f) * scale;
                sum += a_row[k+4] * ((float)((p2 >> 4) & 0xF) - 8.0f) * scale;
                sum += a_row[k+5] * ((float)((p2 >> 0) & 0xF) - 8.0f) * scale;
                sum += a_row[k+6] * ((float)((p3 >> 4) & 0xF) - 8.0f) * scale;
                sum += a_row[k+7] * ((float)((p3 >> 0) & 0xF) - 8.0f) * scale;
            }
            
            /* Remaining elements */
            for (; k + 1 < K; k += 2) {
                uint8_t packed = w[row_offset + k/2];
                sum += a_row[k] * ((float)((packed >> 4) & 0xF) - 8.0f) * scale;
                sum += a_row[k+1] * ((float)((packed >> 0) & 0xF) - 8.0f) * scale;
            }
            if (k < K) {
                uint8_t packed = w[row_offset + k/2];
                sum += a_row[k] * ((float)((packed >> 4) & 0xF) - 8.0f) * scale;
            }
            
            C[i * N + j] = sum;
        }
    }
}

/* Create INT4 matrix from float weights (per-row quantization) */
int4_matrix_t* create_int4_matrix_simple(const float* weights, int rows, int cols) {
    int4_matrix_t* mat = (int4_matrix_t*)malloc(sizeof(int4_matrix_t));
    mat->rows = rows;
    mat->cols = cols;
    
    /* Allocate packed weights */
    int packed_cols = (cols + 1) / 2;
    mat->weights = (uint8_t*)aligned_malloc(rows * packed_cols, 64);
    mat->scales = (float*)aligned_malloc(rows * sizeof(float), 64);
    
    /* Quantize each row */
    for (int r = 0; r < rows; r++) {
        /* Find max abs in row for scale */
        float max_abs = 0.0f;
        for (int c = 0; c < cols; c++) {
            float abs_val = fabsf(weights[r * cols + c]);
            if (abs_val > max_abs) max_abs = abs_val;
        }
        
        /* Scale to fit -8 to 7 range */
        mat->scales[r] = (max_abs > 0) ? (max_abs / 7.0f) : 1.0f;
        
        /* Quantize and pack */
        for (int c = 0; c < cols; c += 2) {
            float w0 = weights[r * cols + c];
            float w1 = (c + 1 < cols) ? weights[r * cols + c + 1] : 0.0f;
            
            int q0 = (int)roundf(w0 / mat->scales[r]) + 8;
            int q1 = (int)roundf(w1 / mat->scales[r]) + 8;
            
            /* Clamp */
            if (q0 < 0) q0 = 0; if (q0 > 15) q0 = 15;
            if (q1 < 0) q1 = 0; if (q1 > 15) q1 = 15;
            
            mat->weights[r * packed_cols + c/2] = (uint8_t)((q0 << 4) | q1);
        }
    }
    
    return mat;
}

void free_int4_matrix_simple(int4_matrix_t* mat) {
    if (!mat) return;
    aligned_free(mat->weights);
    aligned_free(mat->scales);
    free(mat);
}
