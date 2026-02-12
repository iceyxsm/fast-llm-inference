/*
 * Simple test for dequantized tensor implementation
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "quant_types.h"
#include "dequantized_tensor.h"

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

int main(void) {
    printf("Testing Dequantized Tensor Implementation\n");
    printf("=========================================\n\n");
    
    /* Small test dimensions */
    int rows = 4;
    int cols = 32;  /* Multiple of Q4_BLOCK_SIZE (32) */
    
    /* Create Q4 tensor */
    quantized_tensor_t* q4 = create_q4_tensor(rows, cols);
    if (!q4) {
        fprintf(stderr, "Failed to create Q4 tensor\n");
        return 1;
    }
    
    /* Fill with known values */
    block_q4_t* blocks = (block_q4_t*)q4->blocks;
    for (int i = 0; i < q4->n_blocks; i++) {
        blocks[i].scale = 0.5f;
        blocks[i].zero_point = -4.0f;
        for (int j = 0; j < 16; j++) {  /* 16 bytes per block, 32 weights */
            /* Set all nibbles to 8 -> should give 0 after dequant */
            blocks[i].qs[j] = 0x88;
        }
    }
    
    /* Convert to dequantized */
    printf("Converting Q4 -> INT8...\n");
    dequantized_tensor_t* dq = dequantized_from_q4(q4);
    if (!dq) {
        fprintf(stderr, "Failed to create dequantized tensor\n");
        return 1;
    }
    
    printf("  Q4 blocks: %d\n", q4->n_blocks);
    printf("  INT8 weights: %d x %d\n", dq->rows, dq->cols);
    printf("  Memory: Q4=%zu bytes, INT8=%zu bytes\n\n",
           q4->n_blocks * sizeof(block_q4_t),
           dq->rows * dq->cols * sizeof(int8_t));
    
    /* Test matmul */
    float* A = (float*)aligned_malloc(cols * sizeof(float), 64);
    float* C = (float*)aligned_malloc(rows * sizeof(float), 64);
    
    /* Set A to all 1.0f */
    for (int i = 0; i < cols; i++) A[i] = 1.0f;
    
    printf("Running matmul_dequantized...\n");
    matmul_dequantized(A, dq, C, 1, rows, cols);
    
    /* Check results */
    printf("  Results (should be near 0):\n");
    for (int i = 0; i < rows; i++) {
        printf("    C[%d] = %.3f\n", i, C[i]);
    }
    
    /* Verify correctness */
    int passed = 1;
    for (int i = 0; i < rows; i++) {
        if (fabs(C[i]) > 1.0f) {  /* Should be close to 0 */
            passed = 0;
        }
    }
    
    printf("\n");
    if (passed) {
        printf("TEST PASSED!\n");
    } else {
        printf("TEST FAILED!\n");
    }
    
    /* Cleanup */
    aligned_free(A);
    aligned_free(C);
    free_quantized_tensor(q4);
    dequantized_tensor_free(dq);
    
    return passed ? 0 : 1;
}
