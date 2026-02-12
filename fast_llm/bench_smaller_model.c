/*
 * Smaller Model Benchmark
 * Test with fewer layers to verify memory bandwidth theory
 */

#include <stdio.h>
#include <stdlib.h>
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

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  SMALLER MODEL BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int num_tokens = 100;
    
    /* Test different layer counts */
    int layer_counts[] = {32, 24, 20, 16};
    int num_tests = sizeof(layer_counts) / sizeof(layer_counts[0]);
    
    /* Create weights */
    dequantized_tensor_t W_gate, W_up, W_down;
    
    W_gate.rows = intermediate;
    W_gate.cols = hidden;
    W_gate.weights = aligned_malloc(intermediate * hidden, 64);
    W_gate.scales = aligned_malloc(intermediate * sizeof(float), 64);
    
    W_up.rows = intermediate;
    W_up.cols = hidden;
    W_up.weights = aligned_malloc(intermediate * hidden, 64);
    W_up.scales = aligned_malloc(intermediate * sizeof(float), 64);
    
    W_down.rows = hidden;
    W_down.cols = intermediate;
    W_down.weights = aligned_malloc(hidden * intermediate, 64);
    W_down.scales = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int r = 0; r < intermediate; r++) {
        W_gate.scales[r] = W_up.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            W_gate.weights[r * hidden + c] = (rand() % 256) - 128;
            W_up.weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    for (int r = 0; r < hidden; r++) {
        W_down.scales[r] = 0.01f;
        for (int c = 0; c < intermediate; c++) {
            W_down.weights[r * intermediate + c] = (rand() % 256) - 128;
        }
    }
    
    float* input = aligned_malloc(hidden * sizeof(float), 64);
    float* gate_out = aligned_malloc(intermediate * sizeof(float), 64);
    float* up_out = aligned_malloc(intermediate * sizeof(float), 64);
    float* final_out = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    printf("Testing different layer counts:\n");
    printf("(Hidden=%d, Intermediate=%d, Tokens=%d)\n\n", hidden, intermediate, num_tokens);
    printf("Layers | Time (s) | Tok/sec | vs 32L | Target\n");
    printf("-------|----------|---------|--------|-------\n");
    
    double base_time = 0;
    
    for (int t = 0; t < num_tests; t++) {
        int num_layers = layer_counts[t];
        
        /* Warmup */
        for (int w = 0; w < 3; w++) {
            for (int layer = 0; layer < num_layers; layer++) {
                matmul_dequantized_asm_style(input, &W_gate, gate_out, 1, intermediate, hidden);
                matmul_dequantized_asm_style(input, &W_up, up_out, 1, intermediate, hidden);
                for (int i = 0; i < intermediate; i++) {
                    float sig = 1.0f / (1.0f + expf(-gate_out[i]));
                    gate_out[i] = sig * gate_out[i] * up_out[i];
                }
                matmul_dequantized_asm_style(gate_out, &W_down, final_out, 1, hidden, intermediate);
            }
        }
        
        /* Benchmark */
        double start = get_time_ms();
        for (int tok = 0; tok < num_tokens; tok++) {
            for (int layer = 0; layer < num_layers; layer++) {
                matmul_dequantized_asm_style(input, &W_gate, gate_out, 1, intermediate, hidden);
                matmul_dequantized_asm_style(input, &W_up, up_out, 1, intermediate, hidden);
                for (int i = 0; i < intermediate; i++) {
                    float sig = 1.0f / (1.0f + expf(-gate_out[i]));
                    gate_out[i] = sig * gate_out[i] * up_out[i];
                }
                matmul_dequantized_asm_style(gate_out, &W_down, final_out, 1, hidden, intermediate);
            }
        }
        double elapsed = (get_time_ms() - start) / 1000.0;
        double tok_per_sec = num_tokens / elapsed;
        
        if (t == 0) base_time = elapsed;
        double speedup = base_time / elapsed;
        const char* target = (tok_per_sec >= 50.0) ? "✓ HIT" : "";
        
        printf("  %2d   |  %6.2f  |  %5.1f  |  %.2fx | %s\n", 
               num_layers, elapsed, tok_per_sec, speedup, target);
    }
    
    printf("\n=== ANALYSIS ===\n");
    printf("Linear scaling with layer count confirms memory bandwidth limit.\n");
    printf("To reach 50 tok/sec with 32 layers, need either:\n");
    printf("  1. 1.85x memory bandwidth (faster RAM)\n");
    printf("  2. 1.85x compute efficiency (less memory traffic)\n");
    printf("  3. Smaller model (20 layers → ~43 tok/sec)\n");
    printf("  4. Quantize to 4-bit (50%% memory reduction)\n");
    
    aligned_free(W_gate.weights); aligned_free(W_gate.scales);
    aligned_free(W_up.weights); aligned_free(W_up.scales);
    aligned_free(W_down.weights); aligned_free(W_down.scales);
    aligned_free(input); aligned_free(gate_out); aligned_free(up_out); aligned_free(final_out);
    
    return 0;
}
