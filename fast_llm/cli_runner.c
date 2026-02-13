/*
 * Fast LLM CLI Runner
 * Run models with full statistics display
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>
#include <psapi.h>
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

/* External kernels */
extern void matmul_dequantized_asm_style(const float* A, const dequantized_tensor_t* B,
                                          float* C, int M, int N, int K);
extern void swiglu_avx2(const float* gate, const float* up, float* output, int n);
extern void rms_norm_avx2(const float* input, float* output, int n, float eps);

/* Stats structure */
typedef struct {
    double start_time;
    double end_time;
    double total_time;
    double tokens_per_sec;
    double avg_ms_per_token;
    double peak_ram_mb;
    double current_ram_mb;
    int tokens_generated;
} run_stats_t;

/* Get memory usage */
void get_memory_stats(double* current_mb, double* peak_mb) {
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        *current_mb = (double)pmc.WorkingSetSize / (1024.0 * 1024.0);
        *peak_mb = (double)pmc.PeakWorkingSetSize / (1024.0 * 1024.0);
    } else {
        *current_mb = 0.0;
        *peak_mb = 0.0;
    }
}

/* Get high-res time */
double get_time_sec(void) {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart / (double)freq.QuadPart;
}

/* Print colored text */
void color_print(const char* text, int color) {
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, color);
    printf("%s", text);
    SetConsoleTextAttribute(hConsole, 7);
}

/* Print header */
void print_banner(void) {
    printf("\n");
    color_print("========================================\n", 11);
    color_print("       FAST LLM - MODEL RUNNER         \n", 11);
    color_print("========================================\n", 11);
    printf("\n");
}

/* Forward pass simulation with stats */
void run_forward_pass(int num_layers, int hidden, int intermediate,
                      float* hidden_state,
                      dequantized_tensor_t* W_up, dequantized_tensor_t* W_down,
                      run_stats_t* stats) {
    float* norm_out = (float*)aligned_malloc(hidden * sizeof(float), 64);
    float* output_up = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    float* output_down = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
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

/* Run benchmark with full stats */
void run_benchmark(int num_layers, int hidden, int intermediate, int num_tokens) {
    run_stats_t stats = {0};
    
    /* Create weights */
    printf("Creating test weights (simulating model)...\n");
    dequantized_tensor_t W_up, W_down;
    
    W_up.rows = 2 * intermediate;
    W_up.cols = hidden;
    W_up.weights = (int8_t*)aligned_malloc(2 * intermediate * hidden, 64);
    W_up.scales = (float*)aligned_malloc(2 * intermediate * sizeof(float), 64);
    
    for (int r = 0; r < 2 * intermediate; r++) {
        W_up.scales[r] = 0.01f;
        for (int c = 0; c < hidden; c++) {
            W_up.weights[r * hidden + c] = (rand() % 256) - 128;
        }
    }
    
    W_down.rows = hidden;
    W_down.cols = intermediate;
    W_down.weights = (int8_t*)aligned_malloc(hidden * intermediate, 64);
    W_down.scales = (float*)aligned_malloc(hidden * sizeof(float), 64);
    
    for (int r = 0; r < hidden; r++) {
        W_down.scales[r] = 0.01f;
        for (int c = 0; c < intermediate; c++) {
            W_down.weights[r * intermediate + c] = (rand() % 256) - 128;
        }
    }
    
    float* hidden_state = (float*)aligned_malloc(hidden * sizeof(float), 64);
    for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
    
    /* Warmup */
    printf("Warming up...\n");
    for (int w = 0; w < 3; w++) {
        run_forward_pass(num_layers, hidden, intermediate, hidden_state, &W_up, &W_down, &stats);
    }
    
    /* Benchmark */
    printf("\nRunning benchmark: %d layers, %d tokens\n", num_layers, num_tokens);
    printf("Configuration: hidden=%d, intermediate=%d\n\n", hidden, intermediate);
    
    get_memory_stats(&stats.current_ram_mb, &stats.peak_ram_mb);
    double initial_ram = stats.current_ram_mb;
    
    stats.start_time = get_time_sec();
    
    printf("Progress: [");
    fflush(stdout);
    
    int progress_interval = num_tokens / 20;
    if (progress_interval < 1) progress_interval = 1;
    
    for (int t = 0; t < num_tokens; t++) {
        for (int i = 0; i < hidden; i++) hidden_state[i] = 0.01f;
        
        run_forward_pass(num_layers, hidden, intermediate, hidden_state, &W_up, &W_down, &stats);
        
        stats.tokens_generated++;
        
        /* Update peak memory */
        double current, peak;
        get_memory_stats(&current, &peak);
        if (peak > stats.peak_ram_mb) stats.peak_ram_mb = peak;
        
        /* Progress bar */
        if ((t + 1) % progress_interval == 0) {
            color_print("=", 10);
            fflush(stdout);
        }
    }
    
    printf("] Done!\n\n");
    
    stats.end_time = get_time_sec();
    stats.total_time = stats.end_time - stats.start_time;
    stats.tokens_per_sec = stats.tokens_generated / stats.total_time;
    stats.avg_ms_per_token = (stats.total_time / stats.tokens_generated) * 1000.0;
    
    /* Print results */
    printf("========================================\n");
    color_print("           BENCHMARK RESULTS           \n", 10);
    printf("========================================\n\n");
    
    printf("Model Configuration:\n");
    printf("  Layers:      %d\n", num_layers);
    printf("  Hidden:      %d\n", hidden);
    printf("  Intermediate:%d\n", intermediate);
    printf("  Parameters:  ~%.1fB\n\n", (2.0 * hidden * intermediate + hidden * intermediate) * num_layers / 1e9);
    
    printf("Performance:\n");
    printf("  Tokens:      %d\n", stats.tokens_generated);
    printf("  Total time:  %.2f sec\n", stats.total_time);
    printf("  Speed:       ");
    
    if (stats.tokens_per_sec >= 50.0) {
        color_print("✓ ", 10);
        printf("%.2f tok/sec ", stats.tokens_per_sec);
        color_print("(TARGET MET!)\n", 10);
    } else if (stats.tokens_per_sec >= 40.0) {
        color_print("~ ", 14);
        printf("%.2f tok/sec ", stats.tokens_per_sec);
        color_print("(Close)\n", 14);
    } else {
        color_print("✗ ", 12);
        printf("%.2f tok/sec ", stats.tokens_per_sec);
        color_print("(Below target)\n", 12);
    }
    
    printf("  Avg/token:   %.2f ms\n\n", stats.avg_ms_per_token);
    
    printf("Memory Usage:\n");
    printf("  Initial:     %.2f MB\n", initial_ram);
    printf("  Peak:        %.2f MB\n", stats.peak_ram_mb);
    printf("  Delta:       +%.2f MB\n\n", stats.peak_ram_mb - initial_ram);
    
    /* Performance rating */
    printf("Rating: ");
    if (stats.tokens_per_sec >= 55.0) {
        color_print("★★★★★ EXCELLENT\n", 10);
    } else if (stats.tokens_per_sec >= 45.0) {
        color_print("★★★★☆ VERY GOOD\n", 10);
    } else if (stats.tokens_per_sec >= 35.0) {
        color_print("★★★☆☆ GOOD\n", 14);
    } else if (stats.tokens_per_sec >= 25.0) {
        color_print("★★☆☆☆ FAIR\n", 14);
    } else {
        color_print("★☆☆☆☆ SLOW\n", 12);
    }
    
    printf("\n");
    
    /* Cleanup */
    aligned_free(W_up.weights);
    aligned_free(W_up.scales);
    aligned_free(W_down.weights);
    aligned_free(W_down.scales);
    aligned_free(hidden_state);
}

/* Print help */
void print_help(void) {
    printf("Usage: cli_runner [options]\n\n");
    printf("Options:\n");
    printf("  --layers N       Number of layers (default: 24)\n");
    printf("  --tokens N       Number of tokens to generate (default: 50)\n");
    printf("  --hidden N       Hidden size (default: 3072)\n");
    printf("  --intermediate N Intermediate size (default: 8192)\n");
    printf("  --quick          Quick test (10 tokens, 24 layers)\n");
    printf("  --full           Full test (50 tokens, 32 layers)\n");
    printf("  --fast           Fast config (20 layers for speed)\n");
    printf("  --help           Show this help\n\n");
    printf("Examples:\n");
    printf("  cli_runner                    # Default: 24 layers, 50 tokens\n");
    printf("  cli_runner --quick            # Quick 10 token test\n");
    printf("  cli_runner --layers 32        # 32 layers (full model)\n");
    printf("  cli_runner --layers 20 --tokens 100  # 20 layers, 100 tokens\n");
}

int main(int argc, char* argv[]) {
    /* Default settings */
    int num_layers = 24;
    int num_tokens = 50;
    int hidden = 3072;
    int intermediate = 8192;
    
    print_banner();
    
    /* Parse arguments */
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            print_help();
            return 0;
        } else if (strcmp(argv[i], "--quick") == 0) {
            num_layers = 24;
            num_tokens = 10;
            printf("Mode: Quick test (24 layers, 10 tokens)\n\n");
        } else if (strcmp(argv[i], "--full") == 0) {
            num_layers = 32;
            num_tokens = 50;
            printf("Mode: Full test (32 layers, 50 tokens)\n\n");
        } else if (strcmp(argv[i], "--fast") == 0) {
            num_layers = 20;
            num_tokens = 50;
            printf("Mode: Fast test (20 layers, 50 tokens)\n\n");
        } else if (strcmp(argv[i], "--layers") == 0 && i + 1 < argc) {
            num_layers = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tokens") == 0 && i + 1 < argc) {
            num_tokens = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--hidden") == 0 && i + 1 < argc) {
            hidden = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--intermediate") == 0 && i + 1 < argc) {
            intermediate = atoi(argv[++i]);
        }
    }
    
    /* Show CPU info */
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU Features:\n");
    printf("  AVX2:  %s\n", cpu.has_avx2 ? "Yes" : "No");
    printf("  Cores: %d\n\n", cpu.num_cores);
    
    /* Run benchmark */
    run_benchmark(num_layers, hidden, intermediate, num_tokens);
    
    printf("Done!\n\n");
    return 0;
}
