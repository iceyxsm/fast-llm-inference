/*
 * Benchmark for Speculative Decoding
 * 
 * Compares:
 * - Standard autoregressive (target only)
 * - Speculative decoding with draft model
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#include "cpu_features.h"
#include "quant_types.h"
#include "dequantized_tensor.h"
#include "speculative.h"

/* Portable aligned allocation */
#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

/* Get time in seconds */
double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Target model forward pass (simulated slow model) */
void target_forward_slow(void* model, const float* input, float* output, int len) {
    /* Simulate 32-layer model: ~50ms per token */
    /* In real implementation, this would be actual model forward */
    
    /* Busy-wait to simulate computation */
    double start = get_time();
    while (get_time() - start < 0.050) {
        /* Spin */
    }
    
    /* Generate random output */
    for (int i = 0; i < len; i++) {
        output[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f;
    }
}

int main(void) {
    printf("============================================\n");
    printf("Speculative Decoding Benchmark\n");
    printf("============================================\n\n");
    
    srand((unsigned int)time(NULL));
    
    /* Detect CPU */
    cpu_features_t features = detect_cpu_features();
    print_cpu_info(&features);
    printf("\n");
    
    /* Model dimensions */
    int hidden_size = 3072;
    int intermediate_size = 8192;
    int vocab_size = 32064;
    int num_tokens = 20;  /* Generate 20 tokens */
    
    printf("Model: hidden=%d, intermediate=%d, vocab=%d\n", 
           hidden_size, intermediate_size, vocab_size);
    printf("Generating %d tokens\n\n", num_tokens);
    
    /* Create draft model (4 layers vs 32) */
    printf("Creating draft model (4 layers)...\n");
    draft_model_t* draft = draft_model_create(4, hidden_size, intermediate_size, vocab_size);
    if (!draft) {
        fprintf(stderr, "Failed to create draft model\n");
        return 1;
    }
    
    /* Allocate dummy weights for draft model */
    /* In real implementation, would load pre-trained weights */
    printf("Note: Using random weights (no pre-trained draft model)\n");
    printf("      Real speedup requires trained draft model\n\n");
    
    /* Allocate buffers */
    float* prompt_hidden = (float*)aligned_malloc(hidden_size * sizeof(float), 64);
    int* output_ar = (int*)malloc(num_tokens * sizeof(int));
    int* output_spec = (int*)malloc(num_tokens * sizeof(int));
    
    /* Initialize prompt */
    for (int i = 0; i < hidden_size; i++) {
        prompt_hidden[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Benchmark 1: Autoregressive (baseline) */
    printf("Benchmarking AUTOREGRESSIVE (target only)...\n");
    double start = get_time();
    
    for (int i = 0; i < num_tokens; i++) {
        /* Simulate target model forward */
        target_forward_slow(NULL, prompt_hidden, prompt_hidden, hidden_size);
        output_ar[i] = rand() % vocab_size;
    }
    
    double time_ar = get_time() - start;
    double tok_sec_ar = num_tokens / time_ar;
    
    printf("  Time: %.3f sec\n", time_ar);
    printf("  Speed: %.2f tok/sec\n\n", tok_sec_ar);
    
    /* Benchmark 2: Speculative decoding */
    printf("Benchmarking SPECULATIVE DECODING...\n");
    
    speculative_config_t config = speculative_default_config();
    config.num_draft_tokens = 4;  /* Draft 4, hope to accept 3 */
    config.temperature = 0.8f;
    
    start = get_time();
    
    int generated = speculative_generate(
        draft, NULL, target_forward_slow,
        prompt_hidden, 1, output_spec, num_tokens, &config
    );
    
    double time_spec = get_time() - start;
    double tok_sec_spec = generated / time_spec;
    
    printf("  Time: %.3f sec\n", time_spec);
    printf("  Speed: %.2f tok/sec\n", tok_sec_spec);
    printf("  Generated: %d tokens\n\n", generated);
    
    /* Calculate speedup */
    double speedup = tok_sec_spec / tok_sec_ar;
    
    printf("============================================\n");
    printf("RESULTS\n");
    printf("============================================\n");
    printf("Autoregressive:  %.2f tok/sec\n", tok_sec_ar);
    printf("Speculative:     %.2f tok/sec\n", tok_sec_spec);
    printf("Speedup:         %.2fx\n", speedup);
    printf("\n");
    
    if (speedup >= 2.0) {
        printf("*** EXCELLENT SPEEDUP! ***\n");
    } else if (speedup >= 1.5) {
        printf("*** GOOD SPEEDUP! ***\n");
    } else {
        printf("Note: Limited speedup due to:\n");
        printf("  - Random draft weights (no training)\n");
        printf("  - Simulated target model\n");
        printf("  - With trained draft: expect 2-3x speedup\n");
    }
    
    /* Theoretical analysis */
    printf("\n");
    printf("Theoretical Analysis:\n");
    printf("  Draft cost: ~%.1f ms/token (4 layers)\n", 5.0);
    printf("  Target cost: ~%.1f ms/token (32 layers)\n", 50.0);
    printf("  Draft K=4 tokens: %.1f ms\n", 4 * 5.0);
    printf("  Verify K=4 (parallel): %.1f ms\n", 50.0);
    printf("  Accept M=3 tokens: %.1f ms for 3 tokens\n", 4 * 5.0 + 50.0);
    printf("  Effective speedup: %.1fx\n", (3 * 50.0) / (4 * 5.0 + 50.0));
    
    /* Cleanup */
    aligned_free(prompt_hidden);
    free(output_ar);
    free(output_spec);
    draft_model_free(draft);
    
    return 0;
}
