/*
 * Test layer reduction with mock model
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "model_loader.h"

/* External global */
extern int g_max_layers;

int main(int argc, char** argv) {
    printf("Layer Reduction Test\n");
    printf("====================\n\n");
    
    int layers = 32;
    if (argc > 1) {
        layers = atoi(argv[1]);
    }
    
    printf("Creating mock model with %d layers...\n", layers);
    
    /* Create mock model */
    transformer_model_t* model = model_create_mock(3072, 8192, 32, 32000);
    if (!model) {
        printf("Failed to create model\n");
        return 1;
    }
    
    printf("Model created: %d layers\n", model->config.num_layers);
    
    /* Apply layer reduction */
    if (layers < 32) {
        g_max_layers = layers;
        printf("Layer reduction: using %d layers\n", layers);
    }
    
    /* Benchmark */
    printf("\nBenchmarking %d tokens...\n", 50);
    clock_t start = clock();
    
    int tokens[50] = {0};
    float logits[32000];
    
    for (int i = 0; i < 50; i++) {
        model_forward(model, tokens, 1, logits, tokens);
        if (i % 10 == 9) printf("  %d/50\n", i + 1);
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double tok_sec = 50.0 / elapsed;
    
    printf("\nResults:\n");
    printf("  Time: %.2f seconds\n", elapsed);
    printf("  Speed: %.1f tokens/second\n", tok_sec);
    
    if (tok_sec >= 50.0) {
        printf("  ✅ TARGET ACHIEVED!\n");
    } else {
        printf("  ❌ Below target (need %.1f tok/sec)\n", 50.0);
    }
    
    model_free(model);
    return 0;
}
