/*
 * Fast LLM - Pure C CLI
 * No Python, no dependencies, single binary
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

/* Portable aligned allocation */
#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "cpu_features.h"
#include "quant_types.h"
#include "matmul.h"
#include "dequantized_tensor.h"
#include "model_loader.h"

/* CLI Arguments */
typedef struct {
    const char* model_path;
    const char* prompt;
    int max_tokens;
    float temperature;
    int num_threads;
    int num_layers;      /* Number of layers to use (0 = all) */
    bool benchmark;
    int quant_bits;
    bool use_mock;
    bool verbose;
} cli_args_t;

/* Print usage */
void print_usage(const char* prog) {
    printf("Fast LLM - Pure C Inference Engine\n");
    printf("Usage: %s [options]\n\n", prog);
    printf("Options:\n");
    printf("  -m, --model <path>      Model file (GGUF format)\n");
    printf("  -p, --prompt <text>     Prompt for generation\n");
    printf("  -n, --max-tokens <n>    Max tokens to generate (default: 100)\n");
    printf("  -t, --temp <float>      Temperature (default: 0.8)\n");
    printf("  --threads <n>           Number of threads (default: auto)\n");
    printf("  -q, --quant <bits>      Quantization: 2, 4, or 8 (default: 4)\n");
    printf("  --layers <n>            Number of layers to use (default: 32, try 20 for speed)\n");
    printf("  --mock                  Use mock model for testing\n");
    printf("  -b, --benchmark         Run benchmark\n");
    printf("  -v, --verbose           Verbose output\n");
    printf("  -h, --help              Show this help\n");
}

/* Parse arguments */
cli_args_t parse_args(int argc, char** argv) {
    cli_args_t args = {
        .model_path = NULL,
        .prompt = "Hello, I am an AI assistant",
        .max_tokens = 100,
        .temperature = 0.8f,
        .num_threads = 0,
        .num_layers = 0,  /* 0 = use all layers in model */
        .benchmark = false,
        .quant_bits = 4,
        .use_mock = false,
        .verbose = false
    };
    
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 || strcmp(argv[i], "--model") == 0) {
            if (++i < argc) args.model_path = argv[i];
        } else if (strcmp(argv[i], "-p") == 0 || strcmp(argv[i], "--prompt") == 0) {
            if (++i < argc) args.prompt = argv[i];
        } else if (strcmp(argv[i], "-n") == 0 || strcmp(argv[i], "--max-tokens") == 0) {
            if (++i < argc) args.max_tokens = atoi(argv[i]);
        } else if (strcmp(argv[i], "-t") == 0 || strcmp(argv[i], "--temp") == 0) {
            if (++i < argc) args.temperature = atof(argv[i]);
        } else if (strcmp(argv[i], "--threads") == 0) {
            if (++i < argc) args.num_threads = atoi(argv[i]);
        } else if (strcmp(argv[i], "--layers") == 0) {
            if (++i < argc) args.num_layers = atoi(argv[i]);
        } else if (strcmp(argv[i], "-q") == 0 || strcmp(argv[i], "--quant") == 0) {
            if (++i < argc) args.quant_bits = atoi(argv[i]);
        } else if (strcmp(argv[i], "--mock") == 0) {
            args.use_mock = true;
        } else if (strcmp(argv[i], "-b") == 0 || strcmp(argv[i], "--benchmark") == 0) {
            args.benchmark = true;
        } else if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
            args.verbose = true;
        } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
            print_usage(argv[0]);
            exit(0);
        }
    }
    
    return args;
}

/* Get time in seconds */
double get_time(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* Run comprehensive benchmark */
void run_benchmark(cli_args_t* args) {
    (void)args;
    
    printf("\n========================================\n");
    printf("    Fast LLM - Performance Benchmark    \n");
    printf("========================================\n\n");
    
    /* CPU info */
    cpu_features_t cpu = detect_cpu_features();
    print_cpu_info(&cpu);
    printf("\n");
    
    /* Create mock model matching Phi-3 Mini dimensions */
    printf("Loading mock model (Phi-3 Mini architecture)...\n");
    transformer_model_t* model = model_create_mock(
        3072,    /* hidden_size */
        8192,    /* intermediate_size */
        32,      /* num_layers */
        32064    /* vocab_size */
    );
    
    if (!model) {
        printf("Failed to create model\n");
        return;
    }
    
    model_print_info(model);
    
    /* Benchmark 1: Raw matmul performance */
    printf("\n--- Test 1: INT8 Matrix Multiplication ---\n");
    {
        int M = 1, N = 3072, K = 3072;
        int iterations = 10000;
        
        /* Create test matrices */
        float* A = aligned_malloc(M * K * sizeof(float), 32);
        float* C = aligned_malloc(M * N * sizeof(float), 32);
        
        for (int i = 0; i < M * K; i++) A[i] = ((float)rand() / RAND_MAX) - 0.5f;
        
        dequantized_tensor_t B;
        B.rows = N;
        B.cols = K;
        B.weights = aligned_malloc(N * K, 32);
        B.scales = aligned_malloc(N * sizeof(float), 32);
        
        for (int i = 0; i < N * K; i++) {
            B.weights[i] = (int8_t)(((float)rand() / RAND_MAX - 0.5f) * 127);
        }
        for (int i = 0; i < N; i++) B.scales[i] = 0.01f;
        
        /* Warmup */
        for (int i = 0; i < 100; i++) {
            matmul_dequantized(A, &B, C, M, N, K);
        }
        
        /* Benchmark */
        double start = get_time();
        for (int i = 0; i < iterations; i++) {
            matmul_dequantized(A, &B, C, M, N, K);
        }
        double elapsed = get_time() - start;
        
        double gflops = (2.0 * M * N * K * iterations) / (elapsed * 1e9);
        printf("Matrix: %dx%d @ %dx%d, %d iterations\n", M, K, K, N, iterations);
        printf("Time: %.3f seconds\n", elapsed);
        printf("Speed: %.2f GFLOPS\n", gflops);
        printf("Latency: %.3f ms/matmul\n", (elapsed / iterations) * 1000);
        
        aligned_free(A);
        aligned_free(C);
        aligned_free(B.weights);
        aligned_free(B.scales);
    }
    
    /* Benchmark 2: Full forward pass */
    printf("\n--- Test 2: Full Forward Pass ---\n");
    double speed = model_benchmark(model, 50, true);
    
    /* Benchmark 3: Extended run with memory tracking */
    printf("\n--- Test 3: Extended Generation ---\n");
    {
        int num_tokens = 100;
        int* output = calloc(num_tokens, sizeof(int));
        
        model_generate(model, "Test prompt", output, num_tokens, 0.8f, false, false);
        
        free(output);
    }
    
    /* Summary */
    printf("\n========================================\n");
    printf("           Benchmark Complete           \n");
    printf("========================================\n");
    printf("\nTarget: llama.cpp baseline ~25 tok/sec\n");
    printf("Achieved: %.2f tok/sec\n", speed);
    printf("Speedup: %.2fx\n", speed / 25.0);
    printf("\n");
    
    model_free(model);
}

/* Run actual model inference */
void run_generation(cli_args_t* args) {
    printf("\n========================================\n");
    printf("    Fast LLM - Generation Mode         \n");
    printf("========================================\n\n");
    
    /* CPU info */
    cpu_features_t cpu = detect_cpu_features();
    print_cpu_info(&cpu);
    printf("\n");
    
    /* Load or create model */
    transformer_model_t* model = NULL;
    
    if (args->use_mock) {
        printf("Using mock model...\n");
        model = model_create_mock(
            3072,    /* hidden_size */
            8192,    /* intermediate_size */
            32,      /* num_layers */
            32064    /* vocab_size */
        );
    } else if (args->model_path) {
        printf("Loading model from: %s\n", args->model_path);
        model = model_load_gguf(args->model_path, 1);
        
        /* Fall back to mock if GGUF loading fails */
        if (!model) {
            printf("GGUF loading failed, falling back to mock model...\n");
            model = model_create_mock(3072, 8192, 32, 32064);
        }
    } else {
        printf("No model specified, using mock model...\n");
        model = model_create_mock(3072, 8192, 32, 32064);
    }
    
    if (!model) {
        printf("Failed to load/create model\n");
        return;
    }
    
    model_print_info(model);
    
    /* Generate */
    printf("\nPrompt: \"%s\"\n", args->prompt);
    printf("Max tokens: %d\n", args->max_tokens);
    printf("Temperature: %.2f\n\n", args->temperature);
    
    int* output_tokens = calloc(args->max_tokens, sizeof(int));
    
    double start = get_time();
    int generated = model_generate(model, args->prompt, output_tokens, 
                                    args->max_tokens, args->temperature, false, false);
    double elapsed = get_time() - start;
    
    /* Print "output" (token IDs for now) */
    printf("\nGenerated %d tokens in %.2f seconds\n", generated, elapsed);
    printf("Speed: %.2f tok/sec\n\n", generated / elapsed);
    
    /* Clean up */
    free(output_tokens);
    model_free(model);
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    
    cli_args_t args = parse_args(argc, argv);
    
    printf("\n");
    printf("  _____         _     _      __  __ _                 \n");
    printf(" |  ___|_ _ ___| |_  | |    |  \\/  (_)_ __   __ _ ___ \n");
    printf(" | |_ / _` / __| __| | |    | |\\/| | | '_ \\ / _` / __|\n");
    printf(" |  _| (_| \\__ \\ |_  | |___ | |  | | | | | | (_| \\__ \\\n");
    printf(" |_|  \\__,_|___/\\__| |_____||_|  |_|_|_| |_|\\__, |___/\n");
    printf("                                             |___/     \n");
    printf("        Pure C LLM Inference Engine\n");
    printf("        INT8 + AVX2 + Speculative + Medusa\n\n");
    
    /* Apply layer reduction if specified */
    if (args.num_layers > 0) {
        extern int g_max_layers;
        g_max_layers = args.num_layers;
        printf("Using %d layers (reduced from 32)\n\n", args.num_layers);
    }
    
    if (args.benchmark) {
        run_benchmark(&args);
    } else {
        run_generation(&args);
    }
    
    return 0;
}
