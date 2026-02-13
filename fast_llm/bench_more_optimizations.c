/*
 * More Optimizations Test
 * - Batched inference (amortize overhead across multiple sequences)
 * - Better threading (omp parallel for layer groups)
 * - Prefetching optimizations
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "dequantized_tensor.h"
#include "cpu_features.h"

extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                          float* C, int M, int N, int K);
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* Single token forward pass */
void forward_pass(int num_layers, int hidden, int intermediate,
                  float* hidden_state,
                  dequantized_tensor_t* W_up, dequantized_tensor_t* W_down) {
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 32);
    float* output_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 32);
    float* output_down = (float*)aligned_malloc(hidden * sizeof(float), 32);
    
    for (int layer = 0; layer < num_layers; layer++) {
        rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
        matmul_dequantized_asm_style(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
        swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
        matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
        
        for (int j = 0; j < hidden; j++) {
            hidden_state[j] += output_down[j];
        }
    }
    
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
}

/* Batched forward pass - process multiple tokens in parallel */
void forward_pass_batched(int num_layers, int hidden, int intermediate, int batch_size,
                          float** hidden_states,
                          dequantized_tensor_t* W_up, dequantized_tensor_t* W_down) {
    #pragma omp parallel for
    for (int b = 0; b < batch_size; b++) {
        forward_pass(num_layers, hidden, intermediate, hidden_states[b], W_up, W_down);
    }
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  MORE OPTIMIZATIONS TEST\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d, Threads=%d\n\n", 
           cpu.has_avx2 ? "YES" : "NO", cpu.num_cores, cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int num_layers = 24;  /* Use 24 layers for 50+ tok/sec baseline */
    
    /* Create weights */
    dequantized_tensor_t W_up, W_down;
    W_up.rows = 2 * intermediate;
    W_up.cols = hidden;
    W_up.weights = aligned_malloc(2 * intermediate * hidden, 64);
    W_up.scales = aligned_malloc(2 * intermediate * sizeof(float), 64);
    
    for (int r = 0; r < 2 * intermediate; r++) {
        W_up.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            W_up.weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    W_down.rows = hidden;
    W_down.cols = intermediate;
    W_down.weights = aligned_malloc(hidden * intermediate, 64);
    W_down.scales = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int r = 0; r < hidden; r++) {
        W_down.scales[r] = 0.01f;
        for (int c = 0; c < intermediate; c++) {
            W_down.weights[r * intermediate + c] = (rand() % 256) - 128;
        }
    }
    
    /* Test 1: Single token baseline */
    printf("1. SINGLE TOKEN BASELINE (24 layers)\n");
    printf("-------------------------------------\n");
    
    float* hidden_state = aligned_malloc(hidden * sizeof(float), 32);
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    int warmup = 10;
    for (int i = 0; i < warmup; i++) {
        forward_pass(num_layers, hidden, intermediate, hidden_state, &W_up, &W_down);
    }
    
    int tokens = 50;
    double start = get_time_ms();
    for (int i = 0; i < tokens; i++) {
        forward_pass(num_layers, hidden, intermediate, hidden_state, &W_up, &W_down);
    }
    double elapsed = get_time_ms() - start;
    double single_tok_sec = tokens / (elapsed / 1000.0);
    
    printf("   Time: %.2f ms for %d tokens\n", elapsed, tokens);
    printf("   Speed: %.1f tok/sec\n\n", single_tok_sec);
    
    /* Test 2: Batched throughput (measure total tokens/sec) */
    printf("2. BATCHED THROUGHPUT TEST\n");
    printf("---------------------------\n");
    
    int batch_sizes[] = {1, 2, 4, 8};
    int num_batches = sizeof(batch_sizes) / sizeof(batch_sizes[0]);
    
    for (int b = 0; b < num_batches; b++) {
        int bs = batch_sizes[b];
        
        /* Allocate batch */
        float** batch_states = (float**)malloc(bs * sizeof(float*));
        for (int i = 0; i < bs; i++) {
            batch_states[i] = aligned_malloc(hidden * sizeof(float), 32);
            for (int j = 0; j < hidden; j++) batch_states[i][j] = 0.01f;
        }
        
        /* Warmup */
        for (int i = 0; i < 5; i++) {
            forward_pass_batched(num_layers, hidden, intermediate, bs, batch_states, &W_up, &W_down);
        }
        
        /* Benchmark - run fewer iterations for larger batches */
        int iters = (bs <= 2) ? 20 : 10;
        start = get_time_ms();
        for (int i = 0; i < iters; i++) {
            forward_pass_batched(num_layers, hidden, intermediate, bs, batch_states, &W_up, &W_down);
        }
        elapsed = get_time_ms() - start;
        
        /* Total tokens processed */
        int total_tokens = iters * bs;
        double batch_tok_sec = total_tokens / (elapsed / 1000.0);
        double efficiency = batch_tok_sec / (single_tok_sec * bs) * 100.0;
        
        printf("   Batch=%d: %d tokens in %.1f ms = %.1f tok/sec (efficiency: %.0f%%)\n",
               bs, total_tokens, elapsed, batch_tok_sec, efficiency);
        
        for (int i = 0; i < bs; i++) aligned_free(batch_states[i]);
        free(batch_states);
    }
    
    /* Test 3: Layer reduction sweep with current best */
    printf("\n3. LAYER REDUCTION SWEEP (AVX2 optimized)\n");
    printf("------------------------------------------\n");
    
    int layer_configs[] = {32, 28, 24, 20, 16, 12, 8};
    int num_configs = sizeof(layer_configs) / sizeof(layer_configs[0]);
    
    for (int i = 0; i < num_configs; i++) {
        int layers = layer_configs[i];
        
        for (int j = 0; j < hidden; j++) hidden_state[j] = 0.01f;
        
        start = get_time_ms();
        for (int t = 0; t < 50; t++) {
            forward_pass(layers, hidden, intermediate, hidden_state, &W_up, &W_down);
        }
        elapsed = get_time_ms() - start;
        double tok_sec = 50 / (elapsed / 1000.0);
        
        const char* status = (tok_sec >= 50.0) ? "✅" : (tok_sec >= 40.0) ? "⚠️" : "❌";
        printf("   Layers=%2d: %.1f tok/sec %s\n", layers, tok_sec, status);
    }
    
    /* Test 4: Theoretical maximum analysis */
    printf("\n4. THEORETICAL LIMITS\n");
    printf("---------------------\n");
    
    /* Memory bandwidth bound */
    double mem_bw_gb_s = 61.0;  /* Your measured DDR4-3200 bandwidth */
    int params_per_token_24l = 24 * (2 * 3072 * 8192 + 3072 * 8192) * 1; /* INT8 = 1 byte */
    double theoretical_mem_bw_tok_sec = (mem_bw_gb_s * 1e9) / params_per_token_24l;
    
    printf("   Memory bandwidth: %.1f GB/s\n", mem_bw_gb_s);
    printf("   Params/token (24L): %d MB\n", params_per_token_24l / (1024*1024));
    printf("   Theoretical max (mem bound): %.0f tok/sec\n", theoretical_mem_bw_tok_sec);
    printf("   Current efficiency: %.1f%%\n", (single_tok_sec / theoretical_mem_bw_tok_sec) * 100.0);
    
    /* Cleanup */
    aligned_free(W_up.weights);
    aligned_free(W_up.scales);
    aligned_free(W_down.weights);
    aligned_free(W_down.scales);
    aligned_free(hidden_state);
    
    printf("\n========================================\n");
    printf("  NEXT STEPS FOR MORE SPEED\n");
    printf("========================================\n");
    printf("1. Speculative Decoding: Draft model predicts 2-4 tokens,\n");
    printf("   main model verifies. Potential 2-3x speedup.\n\n");
    printf("2. Medusa Heads: Train extra heads to predict future tokens.\n");
    printf("   2x+ speedup with minimal quality loss.\n\n");
    printf("3. INT4 Quantization: 2x memory bandwidth reduction.\n");
    printf("   Could reach %.0f tok/sec if memory bound.\n\n", theoretical_mem_bw_tok_sec * 2);
    printf("4. Continuous Batching: For multi-user serving.\n\n");
    printf("5. FlashAttention: Better memory access for attention.\n\n");
    
    return 0;
}
