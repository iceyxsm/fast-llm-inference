/*
 * Quick INT4 vs INT8 Benchmark
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

typedef struct {
    uint8_t* weights;
    float* scales;
    int rows;
    int cols;
} int4_matrix_t;

extern void matmul_int4_simple(const float* A, const int4_matrix_t* B, float* C,
                                int M, int N, int K);
extern int4_matrix_t* create_int4_matrix_simple(const float* weights, int rows, int cols);
extern void free_int4_matrix_simple(int4_matrix_t* mat);

typedef struct {
    int8_t* weights;
    float* scales;
    int rows;
    int cols;
} int8_matrix_t;

/* Simple INT8 matmul for comparison */
void matmul_int8_simple(const float* A, const int8_matrix_t* B, float* C,
                         int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            float scale = B->scales[j];
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B->weights[j * K + k] * scale;
            }
            C[i * N + j] = sum;
        }
    }
}

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  QUICK INT4 vs INT8 TEST\n");
    printf("========================================\n\n");
    
    int hidden = 3072;
    int intermediate = 8192;
    int num_layers = 32;
    
    /* Create weights */
    int up_size = 2 * intermediate * hidden;
    int down_size = hidden * intermediate;
    
    float* W_up_float = (float*)aligned_malloc(up_size * sizeof(float), 64);
    float* W_down_float = (float*)aligned_malloc(down_size * sizeof(float), 64);
    
    for (int i = 0; i < up_size; i++) W_up_float[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    for (int i = 0; i < down_size; i++) W_down_float[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    /* Create INT4 matrices */
    printf("Creating INT4 matrices...\n");
    int4_matrix_t* W_up_int4 = create_int4_matrix_simple(W_up_float, 2*intermediate, hidden);
    int4_matrix_t* W_down_int4 = create_int4_matrix_simple(W_down_float, hidden, intermediate);
    
    /* Create INT8 matrices */
    printf("Creating INT8 matrices...\n");
    int8_matrix_t W_up_int8, W_down_int8;
    W_up_int8.rows = 2 * intermediate;
    W_up_int8.cols = hidden;
    W_up_int8.weights = (int8_t*)aligned_malloc(up_size, 64);
    W_up_int8.scales = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    for (int r = 0; r < 2 * intermediate; r++) {
        W_up_int8.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            int q = (int)roundf(W_up_float[r * hidden + c] / 0.01f);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            W_up_int8.weights[r * hidden + c] = (int8_t)q;
        }
    }
    W_down_int8.rows = hidden;
    W_down_int8.cols = intermediate;
    W_down_int8.weights = (int8_t*)aligned_malloc(down_size, 64);
    W_down_int8.scales = (float*)aligned_malloc(hidden * sizeof(float), 64);
    for (int r = 0; r < hidden; r++) {
        W_down_int8.scales[r] = 0.01f;
        for (int c = 0; c < intermediate; c++) {
            int q = (int)roundf(W_down_float[r * intermediate + c] / 0.01f);
            if (q < -128) q = -128;
            if (q > 127) q = 127;
            W_down_int8.weights[r * intermediate + c] = (int8_t)q;
        }
    }
    
    /* Allocate buffers */
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    /* Benchmark INT8 */
    printf("\n1. Benchmarking INT8...\n");
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    int tokens = 10;
    double start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        for (int layer = 0; layer < num_layers; layer++) {
            matmul_int8_simple(hidden_state, &W_up_int8, output_up, 1, 2*intermediate, hidden);
            /* Simple SwiGLU */
            for (int i = 0; i < intermediate; i++) {
                float g = output_up[i];
                float u = output_up[i + intermediate];
                float sig = 1.0f / (1.0f + expf(-g));
                output_up[i] = g * sig * u;
            }
            matmul_int8_simple(output_up, &W_down_int8, output_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
        }
    }
    double int8_time = get_time_ms() - start;
    double int8_tok_sec = tokens / (int8_time / 1000.0);
    printf("   INT8: %.1f ms, %.1f tok/sec\n", int8_time, int8_tok_sec);
    
    /* Benchmark INT4 */
    printf("\n2. Benchmarking INT4...\n");
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        for (int layer = 0; layer < num_layers; layer++) {
            matmul_int4_simple(hidden_state, W_up_int4, output_up, 1, 2*intermediate, hidden);
            /* Simple SwiGLU */
            for (int i = 0; i < intermediate; i++) {
                float g = output_up[i];
                float u = output_up[i + intermediate];
                float sig = 1.0f / (1.0f + expf(-g));
                output_up[i] = g * sig * u;
            }
            matmul_int4_simple(output_up, W_down_int4, output_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
        }
    }
    double int4_time = get_time_ms() - start;
    double int4_tok_sec = tokens / (int4_time / 1000.0);
    printf("   INT4: %.1f ms, %.1f tok/sec\n", int4_time, int4_tok_sec);
    
    /* Results */
    printf("\n========================================\n");
    printf("  RESULTS (32 LAYERS)\n");
    printf("========================================\n");
    printf("INT8: %.1f tok/sec\n", int8_tok_sec);
    printf("INT4: %.1f tok/sec\n", int4_tok_sec);
    printf("Speedup: %.2fx\n", int4_tok_sec / int8_tok_sec);
    printf("Target: 50 tok/sec\n\n");
    
    if (int4_tok_sec >= 50.0) {
        printf("✅ 50 TOK/SEC ACHIEVED!\n");
    } else if (int4_tok_sec > int8_tok_sec) {
        printf("⚠️ INT4 faster but need %.1f more tok/sec\n", 50.0 - int4_tok_sec);
    } else {
        printf("❌ INT4 unpacking too slow\n");
        printf("   Need highly optimized AVX2 unpacking\n");
    }
    
    /* Cleanup */
    aligned_free(W_up_float);
    aligned_free(W_down_float);
    free_int4_matrix_simple(W_up_int4);
    free_int4_matrix_simple(W_down_int4);
    aligned_free(W_up_int8.weights);
    aligned_free(W_up_int8.scales);
    aligned_free(W_down_int8.weights);
    aligned_free(W_down_int8.scales);
    aligned_free(hidden_state);
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
    
    return 0;
}
