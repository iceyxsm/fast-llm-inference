/*
 * Fast LLM Engine - Main API
 */

#ifndef FAST_LLM_H
#define FAST_LLM_H

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque engine handle */
typedef struct fast_llm_engine fast_llm_engine_t;

/* Engine configuration */
typedef struct {
    int num_threads;           /* Number of threads to use (0 = auto) */
    int quantization_bits;     /* 2, 4, or 8 */
    bool use_speculative;      /* Enable speculative decoding */
    int draft_tokens;          /* Number of draft tokens for speculative */
    void* user_data;           /* User data pointer */
} fast_llm_config_t;

/* Generation parameters */
typedef struct {
    int max_tokens;            /* Maximum tokens to generate */
    float temperature;         /* Sampling temperature */
    float top_p;              /* Nucleus sampling */
    int top_k;                /* Top-k sampling */
    unsigned long seed;       /* Random seed */
} generation_params_t;

/* Create default config */
static inline fast_llm_config_t fast_llm_default_config(void) {
    fast_llm_config_t cfg = {
        .num_threads = 0,       /* Auto-detect */
        .quantization_bits = 4, /* Q4 default */
        .use_speculative = false,
        .draft_tokens = 4,
        .user_data = NULL
    };
    return cfg;
}

/* Create default generation params */
static inline generation_params_t fast_llm_default_params(void) {
    generation_params_t params = {
        .max_tokens = 100,
        .temperature = 0.8f,
        .top_p = 0.9f,
        .top_k = 40,
        .seed = 0
    };
    return params;
}

/* Engine lifecycle */
fast_llm_engine_t* fast_llm_create(const fast_llm_config_t* config);
void fast_llm_destroy(fast_llm_engine_t* engine);

/* Load model from GGUF file */
int fast_llm_load_gguf(fast_llm_engine_t* engine, const char* path);

/* Generate text */
int fast_llm_generate(fast_llm_engine_t* engine,
                      const char* prompt,
                      const generation_params_t* params,
                      char* output,
                      size_t output_size);

/* Get tokens per second from last generation */
double fast_llm_get_tokens_per_sec(fast_llm_engine_t* engine);

/* Benchmark the engine */
double fast_llm_benchmark(fast_llm_engine_t* engine, int num_tokens);

/* Get engine info */
void fast_llm_print_info(fast_llm_engine_t* engine);

#ifdef __cplusplus
}
#endif

#endif /* FAST_LLM_H */
