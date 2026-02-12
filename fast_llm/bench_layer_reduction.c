/*
 * Approach 1: Layer Reduction Benchmark
 * Tests performance with 32 vs 20 vs 16 layers
 * Expected: 37% speedup for 20 layers vs 32
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <stdint.h>

#ifdef _OPENMP
#include <omp.h>
#endif

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
#define NUM_HEADS 32

/* External functions */
extern void matmul_dequantized_asm_style(const float* A, void* B, float* C,
                                          int M, int N, int K);

/* Simulated dequantized tensor */
typedef struct {
    int8_t* weights;
    float* scales;
} sim_tensor_t;

static double get_time(void) {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

/* Simulate one transformer layer with matmuls only (no attention) */
void simulate_layer_ffn_only(
    float* hidden,
    sim_tensor_t* gate_proj,
    sim_tensor_t* up_proj,
    sim_tensor_t* down_proj,
    int hidden_size,
    int intermediate_size
) {
    float* gate_out = (float*)aligned_malloc(intermediate_size * sizeof(float), 32);
    float* up_out = (float*)aligned_malloc(intermediate_size * sizeof(float), 32);
    float* down_out = (float*)aligned_malloc(hidden_size * sizeof(float), 32);
    
    /* Gate projection */
    matmul_dequantized_asm_style(hidden, (void*)gate_proj, gate_out, 
                                  1, intermediate_size, hidden_size);
    
    /* Up projection */
    matmul_dequantized_asm_style(hidden, (void*)up_proj, up_out,
                                  1, intermediate_size, hidden_size);
    
    /* Element-wise multiply (SwiGLU) */
    for (int i = 0; i < intermediate_size; i++) {
        gate_out[i] = gate_out[i] * up_out[i] / (1.0f + expf(-gate_out[i]));
    }
    
    /* Down projection */
    matmul_dequantized_asm_style(gate_out, (void*)down_proj, down_out,
                                  1, hidden_size, intermediate_size);
    
    /* Add residual */
    for (int i = 0; i < hidden_size; i++) {
        hidden[i] = hidden[i] + down_out[i];
    }
    
    aligned_free(gate_out);
    aligned_free(up_out);
    aligned_free(down_out);
}

/* Benchmark with specified number of layers */
double benchmark_layers(int num_layers, int iterations) {
    int hidden_size = HIDDEN_SIZE;
    int intermediate_size = INTERMEDIATE_SIZE;
    
    /* Setup weights */
    sim_tensor_t gate_proj, up_proj, down_proj;
    gate_proj.weights = (int8_t*)aligned_malloc((size_t)intermediate_size * hidden_size, 32);
    up_proj.weights = (int8_t*)aligned_malloc((size_t)intermediate_size * hidden_size, 32);
    down_proj.weights = (int8_t*)aligned_malloc((size_t)hidden_size * intermediate_size, 32);
    gate_proj.scales = (float*)aligned_malloc(intermediate_size * sizeof(float), 32);
    up_proj.scales = (float*)aligned_malloc(intermediate_size * sizeof(float), 32);
    down_proj.scales = (float*)aligned_malloc(hidden_size * sizeof(float), 32);
    
    /* Initialize */
    for (int i = 0; i < intermediate_size * hidden_size; i++) {
        gate_proj.weights[i] = (int8_t)(i % 256 - 128);
        up_proj.weights[i] = (int8_t)(i % 256 - 128);
    }
    for (int i = 0; i < hidden_size * intermediate_size; i++) {
        down_proj.weights[i] = (int8_t)(i % 256 - 128);
    }
    for (int i = 0; i < intermediate_size; i++) {
        gate_proj.scales[i] = 0.01f;
        up_proj.scales[i] = 0.01f;
    }
    for (int i = 0; i < hidden_size; i++) {
        down_proj.scales[i] = 0.01f;
    }
    
    float* hidden = (float*)aligned_malloc(hidden_size * sizeof(float), 32);
    for (int i = 0; i < hidden_size; i++) hidden[i] = 0.01f;
    
    /* Warmup */
    for (int iter = 0; iter < 5; iter++) {
        for (int layer = 0; layer < num_layers; layer++) {
            simulate_layer_ffn_only(hidden, &gate_proj, &up_proj, &down_proj,
                                     hidden_size, intermediate_size);
        }
    }
    
    /* Benchmark */
    double start = get_time();
    for (int iter = 0; iter < iterations; iter++) {
        for (int layer = 0; layer < num_layers; layer++) {
            simulate_layer_ffn_only(hidden, &gate_proj, &up_proj, &down_proj,
                                     hidden_size, intermediate_size);
        }
    }
    double elapsed = get_time() - start;
    
    /* Cleanup */
    aligned_free(gate_proj.weights);
    aligned_free(up_proj.weights);
    aligned_free(down_proj.weights);
    aligned_free(gate_proj.scales);
    aligned_free(up_proj.scales);
    aligned_free(down_proj.scales);
    aligned_free(hidden);
    
    return elapsed / iterations; /* Time per forward pass */
}

int main() {
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║  APPROACH 1: LAYER REDUCTION BENCHMARK                           ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n\n");
    
    printf("Testing different layer counts...\n\n");
    
    int iterations = 20;
    int layer_counts[] = {32, 28, 24, 20, 16};
    int num_tests = sizeof(layer_counts) / sizeof(layer_counts[0]);
    
    double times[5];
    double tok_per_sec[5];
    double speedup[5];
    
    for (int i = 0; i < num_tests; i++) {
        int layers = layer_counts[i];
        printf("Testing %d layers... ", layers);
        fflush(stdout);
        
        times[i] = benchmark_layers(layers, iterations);
        tok_per_sec[i] = 1.0 / times[i];
        speedup[i] = times[0] / times[i];
        
        printf("%.1f tok/sec (%.2fx vs 32-layer)\n", tok_per_sec[i], speedup[i]);
    }
    
    /* Print results table */
    printf("\n");
    printf("╔════════════════╦════════════╦════════════╦══════════╗\n");
    printf("║ Layers         ║ Time (ms)  ║ Tok/sec    ║ Speedup  ║\n");
    printf("╠════════════════╬════════════╬════════════╬══════════╣\n");
    
    for (int i = 0; i < num_tests; i++) {
        printf("║ %2d layers      ║ %8.2f   ║ %8.1f   ║ %6.2fx ║\n",
               layer_counts[i], times[i] * 1000, tok_per_sec[i], speedup[i]);
    }
    
    printf("╚════════════════╩════════════╩════════════╩══════════╝\n");
    
    /* Analysis */
    printf("\n");
    printf("ANALYSIS:\n");
    printf("---------\n");
    
    int best_idx = 0;
    for (int i = 1; i < num_tests; i++) {
        if (tok_per_sec[i] > tok_per_sec[best_idx]) {
            best_idx = i;
        }
    }
    
    printf("Best configuration: %d layers @ %.1f tok/sec\n",
           layer_counts[best_idx], tok_per_sec[best_idx]);
    
    if (tok_per_sec[best_idx] >= 50.0) {
        printf("✅ TARGET ACHIEVED with %d layers!\n", layer_counts[best_idx]);
    } else {
        double needed = 50.0 / tok_per_sec[best_idx];
        printf("⚠️  Need %.2fx more speedup to reach 50 tok/sec\n", needed);
        printf("    Layer reduction alone is insufficient.\n");
    }
    
    /* Compare to theoretical */
    printf("\nTheoretical vs Actual:\n");
    for (int i = 0; i < num_tests; i++) {
        double theoretical = 32.0 / layer_counts[i];
        printf("  %d layers: theoretical %.2fx, actual %.2fx\n",
               layer_counts[i], theoretical, speedup[i]);
    }
    
    return 0;
}
