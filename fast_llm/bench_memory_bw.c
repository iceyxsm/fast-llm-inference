/*
 * Memory Bandwidth Analysis
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

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  MEMORY BANDWIDTH ANALYSIS\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int iterations = 100;
    
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
    
    /* Calculate memory traffic per FFN layer */
    /*
     * Gate matmul: Read input (3072*4) + Read weights (8192*3072*1) + Write output (8192*4)
     * Up matmul:   Read input (3072*4) + Read weights (8192*3072*1) + Write output (8192*4)
     * SwiGLU:      Read gate (8192*4) + Read up (8192*4) + Write gate (8192*4)
     * Down matmul: Read gate (8192*4) + Read weights (3072*8192*1) + Write output (3072*4)
     */
    
    size_t gate_weights = (size_t)intermediate * hidden;
    size_t up_weights = (size_t)intermediate * hidden;
    size_t down_weights = (size_t)hidden * intermediate;
    
    size_t bytes_gate = hidden * 4 + gate_weights * 1 + intermediate * 4;
    size_t bytes_up = hidden * 4 + up_weights * 1 + intermediate * 4;
    size_t bytes_swiglu = intermediate * 4 * 3;
    size_t bytes_down = intermediate * 4 + down_weights * 1 + hidden * 4;
    
    size_t total_bytes_per_layer = bytes_gate + bytes_up + bytes_swiglu + bytes_down;
    
    printf("Memory traffic per layer:\n");
    printf("  Gate matmul:  %6.2f MB\n", bytes_gate / (1024.0 * 1024.0));
    printf("  Up matmul:    %6.2f MB\n", bytes_up / (1024.0 * 1024.0));
    printf("  SwiGLU:       %6.2f MB\n", bytes_swiglu / (1024.0 * 1024.0));
    printf("  Down matmul:  %6.2f MB\n", bytes_down / (1024.0 * 1024.0));
    printf("  Total:        %6.2f MB\n\n", total_bytes_per_layer / (1024.0 * 1024.0));
    
    /* Benchmark */
    printf("Benchmarking FFN layer...\n");
    
    /* Warmup */
    for (int w = 0; w < 10; w++) {
        matmul_dequantized_asm_style(input, &W_gate, gate_out, 1, intermediate, hidden);
        matmul_dequantized_asm_style(input, &W_up, up_out, 1, intermediate, hidden);
        for (int i = 0; i < intermediate; i++) {
            float sig = 1.0f / (1.0f + expf(-gate_out[i]));
            gate_out[i] = sig * gate_out[i] * up_out[i];
        }
        matmul_dequantized_asm_style(gate_out, &W_down, final_out, 1, hidden, intermediate);
    }
    
    /* Measure */
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) {
        matmul_dequantized_asm_style(input, &W_gate, gate_out, 1, intermediate, hidden);
        matmul_dequantized_asm_style(input, &W_up, up_out, 1, intermediate, hidden);
        for (int i = 0; i < intermediate; i++) {
            float sig = 1.0f / (1.0f + expf(-gate_out[i]));
            gate_out[i] = sig * gate_out[i] * up_out[i];
        }
        matmul_dequantized_asm_style(gate_out, &W_down, final_out, 1, hidden, intermediate);
    }
    double elapsed = get_time_ms() - start;
    double ms_per_layer = elapsed / iterations;
    
    double bytes_per_sec = (total_bytes_per_layer * iterations * 1000.0) / (elapsed * 1024.0 * 1024.0 * 1024.0);
    double tok_per_sec = 1000.0 / ms_per_layer / 32.0;  /* 32 layers */
    
    printf("\nResults:\n");
    printf("  Time per layer: %.3f ms\n", ms_per_layer);
    printf("  Memory bandwidth: %.2f GB/s\n", bytes_per_sec);
    printf("  Estimated tok/sec: %.2f\n", tok_per_sec);
    printf("\nTypical DDR4-3200 bandwidth: ~50 GB/s (theoretical)\n");
    printf("Typical achievable: ~40 GB/s\n");
    printf("\nIf memory bound, max speed with 40 GB/s: %.2f tok/sec\n", 
           tok_per_sec * (40.0 / bytes_per_sec));
    
    /* Calculate arithmetic intensity */
    size_t total_flops = 2ULL * intermediate * hidden * 2 + 2ULL * hidden * intermediate;  /* Gate + Up + Down */
    double arithmetic_intensity = (double)total_flops / (double)total_bytes_per_layer;
    printf("\nArithmetic intensity: %.2f FLOPs/byte\n", arithmetic_intensity);
    printf("(Higher = more compute bound, Lower = more memory bound)\n");
    
    aligned_free(W_gate.weights); aligned_free(W_gate.scales);
    aligned_free(W_up.weights); aligned_free(W_up.scales);
    aligned_free(W_down.weights); aligned_free(W_down.scales);
    aligned_free(input); aligned_free(gate_out); aligned_free(up_out); aligned_free(final_out);
    
    return 0;
}
