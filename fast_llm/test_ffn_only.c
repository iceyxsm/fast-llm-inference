/*
 * Test FFN-only optimized inference
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

/* External function from inference_optimized.c */
extern double model_forward_ffn_only(transformer_model_t* model, int num_tokens);

int main() {
    printf("Testing FFN-only optimized inference...\n\n");
    
    /* Create Phi-3 mini sized mock model */
    int hidden = 3072;
    int intermediate = 8192;
    int layers = 32;
    int vocab = 32064;
    
    printf("Model config: hidden=%d, intermediate=%d, layers=%d\n", hidden, intermediate, layers);
    
    transformer_model_t* model = model_create_mock(hidden, intermediate, layers, vocab);
    if (!model) {
        printf("Failed to create model\n");
        return 1;
    }
    
    /* Run FFN-only benchmark */
    double tok_per_sec = model_forward_ffn_only(model, 20);
    
    printf("\nFFN-only Speed: %.2f tok/sec\n", tok_per_sec);
    printf("\nNote: Full model would be ~20-30%% slower due to attention\n");
    
    model_free(model);
    
    return 0;
}
