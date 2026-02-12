/*
 * Simple benchmark - test basic functionality
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#define HIDDEN_SIZE 3072
#define INTERMEDIATE_SIZE 8192

int main() {
    printf("Simple Benchmark Test\n");
    printf("=====================\n\n");
    
    int n = HIDDEN_SIZE;
    int m = INTERMEDIATE_SIZE;
    
    printf("Allocating memory...\n");
    
    /* Allocate */
    float* input = (float*)aligned_malloc(n * sizeof(float), 32);
    float* output = (float*)aligned_malloc(m * sizeof(float), 32);
    int8_t* weights = (int8_t*)aligned_malloc((size_t)m * n, 32);
    float* scales = (float*)aligned_malloc(m * sizeof(float), 32);
    
    printf("Memory allocated:\n");
    printf("  Input:   %p (%d bytes)\n", (void*)input, n * sizeof(float));
    printf("  Output:  %p (%d bytes)\n", (void*)output, m * sizeof(float));
    printf("  Weights: %p (%zu bytes)\n", (void*)weights, (size_t)m * n);
    printf("  Scales:  %p (%d bytes)\n", (void*)scales, m * sizeof(float));
    
    /* Initialize */
    printf("\nInitializing data...\n");
    for (int i = 0; i < n; i++) input[i] = 0.01f;
    for (int i = 0; i < m * n; i++) weights[i] = 1;
    for (int i = 0; i < m; i++) scales[i] = 0.01f;
    
    /* Simple matmul */
    printf("Running simple matmul...\n");
    
    for (int i = 0; i < m; i++) {
        float sum = 0.0f;
        for (int k = 0; k < n; k++) {
            sum += input[k] * weights[(size_t)i * n + k] * scales[i];
        }
        output[i] = sum;
    }
    
    printf("Result[0] = %f\n", output[0]);
    printf("Result[%d] = %f\n", m-1, output[m-1]);
    
    /* Cleanup */
    aligned_free(input);
    aligned_free(output);
    aligned_free(weights);
    aligned_free(scales);
    
    printf("\nDone!\n");
    return 0;
}
