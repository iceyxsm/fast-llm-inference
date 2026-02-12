/*
 * Fused Operations Benchmark
 * Compares separate matmuls vs fused FFN kernel
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

/* Declare fused function */
extern void fused_ffn(const float* input,
                      const dequantized_tensor_t* gate_up,
                      const dequantized_tensor_t* down,
                      float* output,
                      int hidden, int intermediate);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* Separate matmuls FFN */
void ffn_separate(const float* input,
                  const dequantized_tensor_t* gate,
                  const dequantized_tensor_t* up,
                  const dequantized_tensor_t* down,
                  float* output,
                  int hidden, int intermediate) {
    
    float* gate_out = aligned_malloc(intermediate * sizeof(float), 32);
    float* up_out = aligned_malloc(intermediate * sizeof(float), 32);
    float* down_out = aligned_malloc(hidden * sizeof(float), 32);
    
    /* Separate matmuls */
    matmul_dequantized(input, gate, gate_out, 1, intermediate, hidden);
    matmul_dequantized(input, up, up_out, 1, intermediate, hidden);
    
    /* Activation */
    for (int i = 0; i < intermediate; i++) {
        float sig = 1.0f / (1.0f + expf(-gate_out[i]));
        gate_out[i] = gate_out[i] * sig * up_out[i];
    }
    
    /* Down projection */
    matmul_dequantized(gate_out, down, down_out, 1, hidden, intermediate);
    
    /* Residual */
    for (int i = 0; i < hidden; i++) {
        output[i] = input[i] + down_out[i];
    }
    
    aligned_free(gate_out);
    aligned_free(up_out);
    aligned_free(down_out);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  FUSED vs SEPARATE FFN BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    
    /* Create test weights */
    dequantized_tensor_t gate_up, gate_sep, up_sep, down;
    
    /* Fused gate_up: [2*intermediate, hidden] */
    gate_up.rows = 2 * intermediate;
    gate_up.cols = hidden;
    gate_up.weights = aligned_malloc(2 * intermediate * hidden, 64);
    gate_up.scales = aligned_malloc(2 * intermediate * sizeof(float), 64);
    
    for (int r = 0; r < 2 * intermediate; r++) {
        gate_up.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            gate_up.weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    /* Separate gate and up */
    gate_sep.rows = intermediate;
    gate_sep.cols = hidden;
    gate_sep.weights = aligned_malloc(intermediate * hidden, 64);
    gate_sep.scales = aligned_malloc(intermediate * sizeof(float), 64);
    memcpy(gate_sep.weights, gate_up.weights, intermediate * hidden);
    memcpy(gate_sep.scales, gate_up.scales, intermediate * sizeof(float));
    
    up_sep.rows = intermediate;
    up_sep.cols = hidden;
    up_sep.weights = aligned_malloc(intermediate * hidden, 64);
    up_sep.scales = aligned_malloc(intermediate * sizeof(float), 64);
    memcpy(up_sep.weights, gate_up.weights + intermediate * hidden, intermediate * hidden);
    memcpy(up_sep.scales, gate_up.scales + intermediate, intermediate * sizeof(float));
    
    /* Down projection */
    down.rows = hidden;
    down.cols = intermediate;
    down.weights = aligned_malloc(hidden * intermediate, 64);
    down.scales = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int r = 0; r < hidden; r++) {
        down.scales[r] = 0.01f;
        for (int c = 0; c < intermediate; c++) {
            down.weights[r * intermediate + c] = (rand() % 256) - 128;
        }
    }
    
    /* Input/output */
    float* input = aligned_malloc(hidden * sizeof(float), 32);
    float* output_fused = aligned_malloc(hidden * sizeof(float), 32);
    float* output_sep = aligned_malloc(hidden * sizeof(float), 32);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Warmup */
    printf("Warmup...\n");
    for (int i = 0; i < 10; i++) {
        fused_ffn(input, &gate_up, &down, output_fused, hidden, intermediate);
        ffn_separate(input, &gate_sep, &up_sep, &down, output_sep, hidden, intermediate);
    }
    
    /* Benchmark fused */
    printf("\nBenchmarking fused FFN...\n");
    double start = get_time_ms();
    for (int i = 0; i < 1000; i++) {
        fused_ffn(input, &gate_up, &down, output_fused, hidden, intermediate);
    }
    double fused_time = get_time_ms() - start;
    
    /* Benchmark separate */
    printf("Benchmarking separate matmuls...\n");
    start = get_time_ms();
    for (int i = 0; i < 1000; i++) {
        ffn_separate(input, &gate_sep, &up_sep, &down, output_sep, hidden, intermediate);
    }
    double sep_time = get_time_ms() - start;
    
    /* Results */
    printf("\n=== RESULTS ===\n");
    printf("Fused:   %.3f ms (%.2f us per call)\n", fused_time, fused_time);
    printf("Separate: %.3f ms (%.2f us per call)\n", sep_time, sep_time);
    printf("Speedup: %.2fx\n", sep_time / fused_time);
    
    /* Estimate full model */
    double ms_per_layer_fused = fused_time;  /* Already per 1000 calls, so us per call */
    double ms_per_layer_sep = sep_time;
    
    double tok_per_sec_fused = 1000000.0 / (ms_per_layer_fused * 32 * 2);  /* 2 matmuls per layer, 32 layers */
    double tok_per_sec_sep = 1000000.0 / (ms_per_layer_sep * 32 * 2);
    
    printf("\nEstimated 32-layer FFN-only speed:\n");
    printf("  Fused:   %.2f tok/sec\n", tok_per_sec_fused);
    printf("  Separate: %.2f tok/sec\n", tok_per_sec_sep);
    printf("\n");
    
    /* Cleanup */
    aligned_free(gate_up.weights);
    aligned_free(gate_up.scales);
    aligned_free(gate_sep.weights);
    aligned_free(gate_sep.scales);
    aligned_free(up_sep.weights);
    aligned_free(up_sep.scales);
    aligned_free(down.weights);
    aligned_free(down.scales);
    aligned_free(input);
    aligned_free(output_fused);
    aligned_free(output_sep);
    
    return 0;
}
