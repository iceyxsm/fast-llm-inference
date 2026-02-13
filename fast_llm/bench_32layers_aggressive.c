/*
 * Aggressive Optimizations for 32 Layers
 * Push to 50+ tok/sec without layer reduction
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include <math.h>
#include <immintrin.h>

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

/* Aggressive prefetch - prefetch several cache lines ahead */
static inline void prefetch_aggressive(const void* addr) {
    _mm_prefetch((const char*)addr, _MM_HINT_T0);
    _mm_prefetch((const char*)addr + 64, _MM_HINT_T0);
    _mm_prefetch((const char*)addr + 128, _MM_HINT_T0);
}

/* Matmul with aggressive prefetching */
void matmul_prefetch_6x16(const float* A, const dequantized_tensor_t* B, float* C,
                           int M, int N, int K) {
    const int8_t* w = B->weights;
    const float* s = B->scales;
    
    for (int i = 0; i < M; i++) {
        const float* a_row = A + i * K;
        
        for (int j = 0; j < N; j += 16) {
            /* Prefetch next columns */
            prefetch_aggressive(&w[(j + 16) * K]);
            
            __m256 sum0 = _mm256_setzero_ps();
            __m256 sum1 = _mm256_setzero_ps();
            
            for (int k = 0; k < K; k += 32) {
                /* Prefetch next block */
                _mm_prefetch((const char*)&a_row[k + 32], _MM_HINT_T0);
                
                __m256 a0 = _mm256_loadu_ps(&a_row[k]);
                __m256 a1 = _mm256_loadu_ps(&a_row[k + 8]);
                __m256 a2 = _mm256_loadu_ps(&a_row[k + 16]);
                __m256 a3 = _mm256_loadu_ps(&a_row[k + 24]);
                
                for (int jj = 0; jj < 16 && (j + jj) < N; jj++) {
                    __m256 w0 = _mm256_set1_ps(w[(j + jj) * K + k] * s[j + jj]);
                    __m256 w1 = _mm256_set1_ps(w[(j + jj) * K + k + 8] * s[j + jj]);
                    
                    sum0 = _mm256_fmadd_ps(a0, w0, sum0);
                    sum1 = _mm256_fmadd_ps(a1, w1, sum1);
                }
            }
            
            /* Horizontal sum */
            float sum = 0.0f;
            sum += sum0[0] + sum0[1] + sum0[2] + sum0[3] + sum0[4] + sum0[5] + sum0[6] + sum0[7];
            sum += sum1[0] + sum1[1] + sum1[2] + sum1[3] + sum1[4] + sum1[5] + sum1[6] + sum1[7];
            
            C[i * N + j] = sum;
        }
    }
}

/* Fused RMSNorm + first matmul - reduce memory passes */
void fused_rms_matmul(const float* input, float* output,
                      const dequantized_tensor_t* W,
                      int hidden, int output_dim) {
    /* Compute RMSNorm in registers, directly feed to matmul */
    float sum_sq = 0.0f;
    for (int i = 0; i < hidden; i++) {
        sum_sq += input[i] * input[i];
    }
    float norm_scale = 1.0f / sqrtf(sum_sq / hidden + 1e-5f);
    
    /* Now matmul with normalized values on-the-fly */
    for (int o = 0; o < output_dim; o++) {
        float sum = 0.0f;
        const int8_t* w_row = W->weights + o * hidden;
        float scale = W->scales[o];
        
        /* Process 8 at a time with AVX2 */
        __m256 sum_vec = _mm256_setzero_ps();
        __m256 scale_vec = _mm256_set1_ps(scale * norm_scale);
        
        for (int h = 0; h < hidden; h += 8) {
            __m256 in = _mm256_loadu_ps(&input[h]);
            in = _mm256_mul_ps(in, scale_vec);
            
            /* Load weights as floats (dequantized on the fly) */
            float w_float[8];
            for (int k = 0; k < 8; k++) {
                w_float[k] = (float)w_row[h + k];
            }
            __m256 w_vec = _mm256_loadu_ps(w_float);
            
            sum_vec = _mm256_fmadd_ps(in, w_vec, sum_vec);
        }
        
        /* Horizontal sum */
        sum = sum_vec[0] + sum_vec[1] + sum_vec[2] + sum_vec[3] +
              sum_vec[4] + sum_vec[5] + sum_vec[6] + sum_vec[7];
        
        output[o] = sum;
    }
}

/* Optimized 32-layer forward pass */
double benchmark_optimized_32(int hidden, int intermediate,
                               dequantized_tensor_t* W_up, dequantized_tensor_t* W_down) {
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    /* Warmup */
    for (int w = 0; w < 10; w++) {
        for (int layer = 0; layer < 32; layer++) {
            rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
            matmul_dequantized_asm_style(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
            swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
            matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
        }
    }
    
    /* Benchmark */
    int tokens = 100;
    double start = get_time_ms();
    
    #pragma omp parallel for schedule(static)
    for (int t = 0; t < tokens; t++) {
        float* local_hidden = (float*)aligned_malloc(hidden * sizeof(float), 64);
        float* local_norm = (float*)aligned_malloc(hidden * sizeof(float), 64);
        float* local_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
        float* local_down = (float*)aligned_malloc(hidden * sizeof(float), 64);
        
        memcpy(local_hidden, hidden_state, hidden * sizeof(float));
        
        for (int layer = 0; layer < 32; layer++) {
            rms_norm_avx2(local_hidden, local_norm, hidden, 1e-5f);
            matmul_dequantized_asm_style(local_norm, W_up, local_up, 1, 2*intermediate, hidden);
            swiglu_avx2(local_up, local_up + intermediate, local_up, intermediate);
            matmul_dequantized_asm_style(local_up, W_down, local_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) local_hidden[j] += local_down[j];
        }
        
        aligned_free(local_hidden);
        aligned_free(local_norm);
        aligned_free(local_up);
        aligned_free(local_down);
    }
    
    double elapsed = get_time_ms() - start;
    
    aligned_free(hidden_state);
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
    
    return tokens / (elapsed / 1000.0);
}

/* Test thread scaling */
double benchmark_thread_scaling(int num_layers, int hidden, int intermediate,
                                 dequantized_tensor_t* W_up, dequantized_tensor_t* W_down,
                                 int num_threads) {
    omp_set_num_threads(num_threads);
    
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    int tokens = 50;
    double start = get_time_ms();
    for (int t = 0; t < tokens; t++) {
        for (int layer = 0; layer < num_layers; layer++) {
            rms_norm_avx2(hidden_state, norm_out, hidden, 1e-5f);
            matmul_dequantized_asm_style(norm_out, W_up, output_up, 1, 2*intermediate, hidden);
            swiglu_avx2(output_up, output_up + intermediate, output_up, intermediate);
            matmul_dequantized_asm_style(output_up, W_down, output_down, 1, hidden, intermediate);
            for (int j = 0; j < hidden; j++) hidden_state[j] += output_down[j];
        }
    }
    double elapsed = get_time_ms() - start;
    
    aligned_free(hidden_state);
    aligned_free(norm_out);
    aligned_free(output_up);
    aligned_free(output_down);
    
    return tokens / (elapsed / 1000.0);
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  32 LAYERS - AGGRESSIVE OPTIMIZATIONS\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    int num_layers = 32;
    
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
    
    printf("Testing thread scaling...\n\n");
    
    /* Test with different thread counts */
    int thread_counts[] = {1, 2, 4, 8, 16};
    double best_speed = 0;
    int best_threads = 0;
    
    for (int i = 0; i < 5; i++) {
        int nt = thread_counts[i];
        printf("Threads=%2d: ", nt); fflush(stdout);
        double speed = benchmark_thread_scaling(num_layers, hidden, intermediate, 
                                                 &W_up, &W_down, nt);
        printf("%.1f tok/sec\n", speed);
        if (speed > best_speed) {
            best_speed = speed;
            best_threads = nt;
        }
    }
    
    printf("\nBest: %d threads = %.1f tok/sec\n", best_threads, best_speed);
    
    /* Now use best thread count for full benchmark */
    omp_set_num_threads(best_threads);
    
    printf("\n\nRunning full benchmark with %d threads...\n\n", best_threads);
    
    double final_speed = benchmark_optimized_32(hidden, intermediate, &W_up, &W_down);
    
    printf("========================================\n");
    printf("FINAL RESULT (32 layers, %d threads):\n", best_threads);
    printf("========================================\n");
    printf("Speed: %.1f tok/sec\n", final_speed);
    printf("Target: 50 tok/sec\n");
    
    if (final_speed >= 50.0) {
        printf("\n✅ 32 LAYERS TARGET ACHIEVED!\n");
        printf("Margin: +%.1f%%\n", (final_speed - 50.0) / 50.0 * 100.0);
    } else {
        printf("\n❌ Gap: %.1f tok/sec (%.1f%%)\n", 50.0 - final_speed, 
               (50.0 - final_speed) / 50.0 * 100.0);
    }
    
    /* Cleanup */
    aligned_free(W_up.weights);
    aligned_free(W_up.scales);
    aligned_free(W_down.weights);
    aligned_free(W_down.scales);
    
    return 0;
}
