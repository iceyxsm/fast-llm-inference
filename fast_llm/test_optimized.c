/*
 * Quick test to verify optimized inference
 */

#include <stdio.h>
#include <stdlib.h>
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

int main() {
    printf("Testing optimized inference...\n");
    
    /* Create small mock model */
    transformer_model_t* model = model_create_mock(768, 2048, 4, 32000);
    if (!model) {
        printf("Failed to create model\n");
        return 1;
    }
    
    model_print_info(model);
    
    /* Run short benchmark */
    printf("\nRunning 10-token benchmark...\n");
    int num_tokens = 10;
    
    float* logits = aligned_malloc(model->config.vocab_size * sizeof(float), 32);
    int* tokens = calloc(10, sizeof(int));
    int next_token;
    
    clock_t start = clock();
    for (int i = 0; i < num_tokens; i++) {
        model_forward(model, tokens, 1, logits, &next_token);
        tokens[0] = next_token;
    }
    clock_t end = clock();
    
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double tok_per_sec = num_tokens / elapsed;
    
    printf("\n=== Results ===\n");
    printf("Tokens: %d\n", num_tokens);
    printf("Time: %.3f sec\n", elapsed);
    printf("Speed: %.2f tok/sec\n", tok_per_sec);
    printf("===============\n");
    
    aligned_free(logits);
    free(tokens);
    model_free(model);
    
    return 0;
}
