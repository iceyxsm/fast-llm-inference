/*
 * Ultimate Benchmark - All Optimizations Combined
 * 1. Fast assembly-style matmul
 * 2. Thread pool
 * 3. INT8 weights
 * 
 * Target: 25+ tok/sec
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
#include "thread_pool.h"
#include "cpu_features.h"

/* Declare fast matmul */
extern void matmul_dequantized_fast(const float* A, const dequantized_tensor_t* B,
                                    float* C, int M, int N, int K);

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* Task context for thread pool */
typedef struct {
    const float* input;
    dequantized_tensor_t* up;
    dequantized_tensor_t* down;
    float* output;
    float* temp;  /* For gate+up output */
    int hidden;
    int inter;
} ffn_task_t;

/* FFN layer task */
void ffn_layer_task(void* arg, int worker_id) {
    (void)worker_id;
    ffn_task_t* task = (ffn_task_t*)arg;
    
    /* Gate + Up projection (fused in one tensor) */
    matmul_dequantized_fast(task->input, task->up, task->temp, 1, 2 * task->inter, task->hidden);
    
    /* SwiGLU activation: gate * sigmoid(gate) * up */
    float* gate = task->temp;
    float* up = task->temp + task->inter;
    
    for (int i = 0; i < task->inter; i++) {
        float g = gate[i];
        float u = up[i];
        float sig = 1.0f / (1.0f + expf(-g));
        task->temp[i] = g * sig * u;
    }
    
    /* Down projection */
    matmul_dequantized_fast(task->temp, task->down, task->output, 1, task->hidden, task->inter);
    
    /* Residual */
    for (int i = 0; i < task->hidden; i++) {
        task->output[i] += task->input[i];
    }
}

/* Run single layer */
void run_layer(ffn_task_t* task) {
    ffn_layer_task(task, 0);
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    
    int hidden = 3072;
    int inter = 8192;
    int num_layers = (argc > 1) ? atoi(argv[1]) : 32;
    int num_tokens = (argc > 2) ? atoi(argv[2]) : 100;
    
    printf("\n========================================\n");
    printf("  ULTIMATE OPTIMIZED BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    printf("Configuration:\n");
    printf("  Layers: %d\n", num_layers);
    printf("  Hidden: %d\n", hidden);
    printf("  Intermediate: %d\n", inter);
    printf("  Tokens: %d\n\n", num_tokens);
    
    /* Create thread pool */
    int num_threads = cpu.num_cores;
    printf("Creating thread pool with %d workers...\n", num_threads);
    thread_pool_t* pool = thread_pool_create(num_threads);
    
    /* Create mock weights (simulate loaded GGUF) */
    printf("Creating weights...\n");
    dequantized_tensor_t** ups = calloc(num_layers, sizeof(dequantized_tensor_t*));
    dequantized_tensor_t** downs = calloc(num_layers, sizeof(dequantized_tensor_t*));
    
    for (int l = 0; l < num_layers; l++) {
        /* Fused gate+up: [2*inter, hidden] */
        ups[l] = malloc(sizeof(dequantized_tensor_t));
        ups[l]->rows = 2 * inter;
        ups[l]->cols = hidden;
        ups[l]->weights = aligned_malloc(2 * inter * hidden, 64);
        ups[l]->scales = aligned_malloc(2 * inter * sizeof(float), 64);
        
        for (int r = 0; r < 2 * inter; r++) {
            ups[l]->scales[r] = 0.01f;
            for (int c = 0; c < hidden; c++) {
                ups[l]->weights[r * hidden + c] = (rand() % 256) - 128;
            }
        }
        
        /* Down: [hidden, inter] */
        downs[l] = malloc(sizeof(dequantized_tensor_t));
        downs[l]->rows = hidden;
        downs[l]->cols = inter;
        downs[l]->weights = aligned_malloc(hidden * inter, 64);
        downs[l]->scales = aligned_malloc(hidden * sizeof(float), 64);
        
        for (int r = 0; r < hidden; r++) {
            downs[l]->scales[r] = 0.01f;
            for (int c = 0; c < inter; c++) {
                downs[l]->weights[r * inter + c] = (rand() % 256) - 128;
            }
        }
    }
    printf("Weights created.\n\n");
    
    /* Allocate buffers */
    float* input = aligned_malloc(hidden * sizeof(float), 32);
    float* output = aligned_malloc(hidden * sizeof(float), 32);
    float* temp = aligned_malloc(inter * 2 * sizeof(float), 32);
    
    for (int i = 0; i < hidden; i++) {
        input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Warmup */
    printf("Warmup...\n");
    ffn_task_t warmup_task = {
        .input = input,
        .up = ups[0],
        .down = downs[0],
        .output = output,
        .temp = temp,
        .hidden = hidden,
        .inter = inter
    };
    
    for (int w = 0; w < 10; w++) {
        run_layer(&warmup_task);
        float* t = input; input = output; output = t;
    }
    
    /* Benchmark */
    printf("\nBenchmarking %d tokens with %d layers...\n", num_tokens, num_layers);
    
    double start = get_time_ms();
    
    for (int tok = 0; tok < num_tokens; tok++) {
        /* Option 1: Sequential (single-threaded) */
        for (int l = 0; l < num_layers; l++) {
            ffn_task_t task = {
                .input = input,
                .up = ups[l],
                .down = downs[l],
                .output = output,
                .temp = temp,
                .hidden = hidden,
                .inter = inter
            };
            run_layer(&task);
            float* t = input; input = output; output = t;
        }
        
        if ((tok + 1) % 10 == 0) {
            printf("  %d/%d tokens...\r", tok + 1, num_tokens);
            fflush(stdout);
        }
    }
    
    double elapsed = get_time_ms() - start;
    double tok_per_sec = num_tokens / (elapsed / 1000.0);
    
    printf("\n\n=== RESULTS ===\n");
    printf("Time: %.2f ms\n", elapsed);
    printf("Tokens/sec: %.2f\n", tok_per_sec);
    printf("Ms/token: %.2f\n", elapsed / num_tokens);
    printf("\n");
    printf("vs llama.cpp (~25 tok/sec): %.2fx\n", tok_per_sec / 25.0);
    printf("\n");
    
    /* Cleanup */
    thread_pool_destroy(pool);
    
    for (int l = 0; l < num_layers; l++) {
        aligned_free(ups[l]->weights);
        aligned_free(ups[l]->scales);
        free(ups[l]);
        aligned_free(downs[l]->weights);
        aligned_free(downs[l]->scales);
        free(downs[l]);
    }
    free(ups);
    free(downs);
    aligned_free(input);
    aligned_free(output);
    aligned_free(temp);
    
    return 0;
}
