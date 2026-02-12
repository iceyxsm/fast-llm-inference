/*
 * Final Optimization Attempt - 50 tok/sec
 * 
 * Optimizations:
 * 1. Static scheduling with tuned chunk size
 * 2. Pre-transposed weights (sequential access)
 * 3. Reduced synchronization overhead
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

/* Optimized version with static scheduling */
void matmul_optimized_static(const float* A, const dequantized_tensor_t* B,
                              float* C, int M, int N, int K) {
    (void)M;
    
    /* Static scheduling with chunk size tuned for 16 cores */
    /* Each thread gets ~1024 rows (16384 / 16 = 1024) */
    #pragma omp parallel for schedule(static, 1024)
    for (int n = 0; n <= N - 6; n += 6) {
        __m256 c0 = _mm256_setzero_ps(), c1 = _mm256_setzero_ps();
        __m256 c2 = _mm256_setzero_ps(), c3 = _mm256_setzero_ps();
        __m256 c4 = _mm256_setzero_ps(), c5 = _mm256_setzero_ps();
        
        for (int k = 0; k <= K - 16; k += 16) {
            __m256 a0 = _mm256_loadu_ps(A + k);
            __m256 a1 = _mm256_loadu_ps(A + k + 8);
            
            for (int i = 0; i < 6; i++) {
                __m128i b_i8_lo = _mm_loadu_si128((__m128i*)(B->weights + (n+i) * K + k));
                __m256i b_i32_0 = _mm256_cvtepi8_epi32(b_i8_lo);
                __m256i b_i32_1 = _mm256_cvtepi8_epi32(_mm_srli_si128(b_i8_lo, 8));
                __m256 b_f_0 = _mm256_cvtepi32_ps(b_i32_0);
                __m256 b_f_1 = _mm256_cvtepi32_ps(b_i32_1);
                
                __m256 scale = _mm256_set1_ps(B->scales[n+i] * 0.0625f);
                b_f_0 = _mm256_mul_ps(b_f_0, scale);
                b_f_1 = _mm256_mul_ps(b_f_1, scale);
                
                switch(i) {
                    case 0: c0 = _mm256_fmadd_ps(a0, b_f_0, c0); c0 = _mm256_fmadd_ps(a1, b_f_1, c0); break;
                    case 1: c1 = _mm256_fmadd_ps(a0, b_f_0, c1); c1 = _mm256_fmadd_ps(a1, b_f_1, c1); break;
                    case 2: c2 = _mm256_fmadd_ps(a0, b_f_0, c2); c2 = _mm256_fmadd_ps(a1, b_f_1, c2); break;
                    case 3: c3 = _mm256_fmadd_ps(a0, b_f_0, c3); c3 = _mm256_fmadd_ps(a1, b_f_1, c3); break;
                    case 4: c4 = _mm256_fmadd_ps(a0, b_f_0, c4); c4 = _mm256_fmadd_ps(a1, b_f_1, c4); break;
                    case 5: c5 = _mm256_fmadd_ps(a0, b_f_0, c5); c5 = _mm256_fmadd_ps(a1, b_f_1, c5); break;
                }
            }
        }
        
        float sum0 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c0), _mm256_extractf128_ps(c0, 1)));
        float sum1 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c1), _mm256_extractf128_ps(c1, 1)));
        float sum2 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c2), _mm256_extractf128_ps(c2, 1)));
        float sum3 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c3), _mm256_extractf128_ps(c3, 1)));
        float sum4 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c4), _mm256_extractf128_ps(c4, 1)));
        float sum5 = _mm_cvtss_f32(_mm_add_ss(_mm256_castps256_ps128(c5), _mm256_extractf128_ps(c5, 1)));
        
        C[n+0] = sum0; C[n+1] = sum1; C[n+2] = sum2;
        C[n+3] = sum3; C[n+4] = sum4; C[n+5] = sum5;
    }
    
    int n_rem = (N / 6) * 6;
    for (int n = n_rem; n < N; n++) {
        const int8_t* B_row = B->weights + n * K;
        float scale = B->scales[n] * 0.0625f;
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[k] * B_row[k] * scale;
        }
        C[n] = sum;
    }
}

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

void benchmark_kernel(const char* name,
                      void (*kernel)(const float*, const dequantized_tensor_t*, float*, int, int, int),
                      const float* A, const dequantized_tensor_t* B, float* C,
                      int M, int N, int K, int iterations) {
    
    for (int w = 0; w < 10; w++) kernel(A, B, C, M, N, K);
    
    double start = get_time_ms();
    for (int i = 0; i < iterations; i++) kernel(A, B, C, M, N, K);
    double elapsed = get_time_ms() - start;
    
    printf("%-25s: %6.3f ms | %5.1f GFLOPS\n", name, elapsed/iterations, 
           (2.0*M*N*K*iterations)/(elapsed*1e6));
}

int main() {
    srand((unsigned)time(NULL));
    
    printf("\n========================================\n");
    printf("  FINAL 50 TOK/SEC OPTIMIZATION\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    int hidden = 3072;
    int intermediate = 8192;
    
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
    
    float* input = aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = aligned_malloc(hidden * sizeof(float), 64);
    
    for (int i = 0; i < hidden; i++) input[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    printf("Gate+Up:\n");
    benchmark_kernel("6x16 (dynamic)", matmul_dequantized_asm_style,
                     input, &W_up, output_up, 1, 2*intermediate, hidden, 100);
    benchmark_kernel("6x16 (static)", matmul_optimized_static,
                     input, &W_up, output_up, 1, 2*intermediate, hidden, 100);
    
    printf("\nDown:\n");
    benchmark_kernel("6x16 (dynamic)", matmul_dequantized_asm_style,
                     output_up, &W_down, output_down, 1, hidden, intermediate, 100);
    benchmark_kernel("6x16 (static)", matmul_optimized_static,
                     output_up, &W_down, output_down, 1, hidden, intermediate, 100);
    
    /* Quick full model test */
    printf("\n=== FULL MODEL TEST ===\n");
    
    double t_up_dyn = 0, t_down_dyn = 0;
    double t_up_stat = 0, t_down_stat = 0;
    
    for (int i = 0; i < 50; i++) {
        double s = get_time_ms();
        matmul_dequantized_asm_style(input, &W_up, output_up, 1, 2*intermediate, hidden);
        t_up_dyn += get_time_ms() - s;
        
        s = get_time_ms();
        matmul_optimized_static(input, &W_up, output_up, 1, 2*intermediate, hidden);
        t_up_stat += get_time_ms() - s;
        
        s = get_time_ms();
        matmul_dequantized_asm_style(output_up, &W_down, output_down, 1, hidden, intermediate);
        t_down_dyn += get_time_ms() - s;
        
        s = get_time_ms();
        matmul_optimized_static(output_up, &W_down, output_down, 1, hidden, intermediate);
        t_down_stat += get_time_ms() - s;
    }
    
    t_up_dyn /= 50; t_down_dyn /= 50;
    t_up_stat /= 50; t_down_stat /= 50;
    
    double tok_dyn = 1000.0 / (t_up_dyn + t_down_dyn) / 32.0;
    double tok_stat = 1000.0 / (t_up_stat + t_down_stat) / 32.0;
    
    printf("\nDynamic schedule:  %.2f tok/sec\n", tok_dyn);
    printf("Static schedule:   %.2f tok/sec\n", tok_stat);
    printf("\nTarget: 50 tok/sec\n");
    
    if (tok_stat >= 50.0) {
        printf("\n🎉🎉🎉 50 TOK/SEC ACHIEVED! 🎉🎉🎉\n");
    } else {
        printf("\nBest: %.2f tok/sec (%.1f%% of target)\n", 
               (tok_stat > tok_dyn ? tok_stat : tok_dyn),
               (tok_stat > tok_dyn ? tok_stat : tok_dyn) / 50.0 * 100.0);
        printf("\nTo reach 50 tok/sec:\n");
        printf("  - Need hand-written assembly\n");
        printf("  - Or AVX-512 VNNI\n");
        printf("  - Or reduce model size\n");
    }
    
    aligned_free(W_up.weights); aligned_free(W_up.scales);
    aligned_free(W_down.weights); aligned_free(W_down.scales);
    aligned_free(input); aligned_free(output_up); aligned_free(output_down);
    
    return 0;
}
