/*
 * Simple test with real weights
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include <time.h>
#include "dequantized_tensor.h"

int main() {
    printf("Simple test: Loading just 1 FFN layer from real model\n\n");
    
    /* Just test the matmul with pre-loaded weights */
    int hidden = 3072;
    int intermediate = 8192;
    
    /* Create test weights similar to real model */
    dequantized_tensor_t* gate = malloc(sizeof(dequantized_tensor_t));
    gate->rows = intermediate;
    gate->cols = hidden;
    gate->weights = aligned_malloc(intermediate * hidden, 64);
    gate->scales = aligned_malloc(intermediate * sizeof(float), 64);
    
    /* Fill with random values */
    for (int r = 0; r < intermediate; r++) {
        gate->scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            gate->weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    /* Test input/output */
    float* input = aligned_malloc(hidden * sizeof(float), 32);
    float* output = aligned_malloc(intermediate * sizeof(float), 32);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Warmup */
    for (int i = 0; i < 10; i++) {
        matmul_dequantized(input, gate, output, 1, intermediate, hidden);
    }
    
    /* Benchmark */
    printf("Benchmarking matmul: [%d] @ [%d, %d]\n", hidden, intermediate, hidden);
    printf("Running 100 iterations...\n");
    
    clock_t start = clock();
    for (int i = 0; i < 100; i++) {
        matmul_dequantized(input, gate, output, 1, intermediate, hidden);
    }
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double flops = (2.0 * 100 * intermediate * hidden) / (elapsed * 1e9);
    double time_per = elapsed / 100.0 * 1000.0;  /* ms */
    
    printf("\nResults:\n");
    printf("  Time: %.3f sec\n", elapsed);
    printf("  GFLOPS: %.2f\n", flops);
    printf("  Ms per matmul: %.3f ms\n", time_per);
    
    /* Estimate full model speed */
    /* 32 layers * 3 matmuls (gate, up, down) = 96 matmuls per token */
    double tok_per_sec = 1000.0 / (time_per * 96);
    printf("\nEstimated full model speed:\n");
    printf("  %.2f tokens/second (32 layers)\n", tok_per_sec);
    printf("  (Real speed will be lower due to attention)\n");
    
    /* Cleanup */
    aligned_free(gate->weights);
    aligned_free(gate->scales);
    free(gate);
    aligned_free(input);
    aligned_free(output);
    
    return 0;
}
