/*
 * Single Layer Real Weights Benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <windows.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "dequantized_tensor.h"
#include <math.h>
#include "cpu_features.h"

#define GGUF_MAGIC 0x46554747

typedef struct __attribute__((packed)) { uint8_t scales[12]; uint8_t qs[128]; } block_q4_K;
typedef struct __attribute__((packed)) { uint8_t scales[16]; uint8_t qs[192]; } block_q6_K;

static uint64_t read_u64(FILE* f) { uint64_t v; fread(&v, 8, 1, f); return v; }
static uint32_t read_u32(FILE* f) { uint32_t v; fread(&v, 4, 1, f); return v; }
static int32_t read_i32(FILE* f) { int32_t v; fread(&v, 4, 1, f); return v; }
static char* read_string(FILE* f) { uint64_t len = read_u64(f); char* s = malloc(len+1); fread(s,1,len,f); s[len]=0; return s; }
static void skip_value(FILE* f, int type) {
    switch(type) {
        case 0: case 1: case 7: {uint8_t v; fread(&v,1,1,f); break;}
        case 2: case 3: {uint16_t v; fread(&v,2,1,f); break;}
        case 4: case 5: case 6: read_u32(f); break;
        case 8: free(read_string(f)); break;
        case 9: {int at=read_i32(f); uint64_t al=read_u64(f); for(uint64_t i=0;i<al;i++)skip_value(f,at); break;}
        case 10: case 11: case 12: read_u64(f); break;
    }
}

static void deq_q4_K(const block_q4_K* src, float* dst, int n) {
    int nb = n / 256;
    for (int b = 0; b < nb; b++) {
        float d = (src[b].scales[0] & 0x3F) / 32.0f + 0.001f;
        for (int i = 0; i < 128; i++) {
            uint8_t q = src[b].qs[i];
            dst[b*256+i] = (q & 0x0F) * d - 8*d;
            dst[b*256+i+128] = (q >> 4) * d - 8*d;
        }
    }
}

static void deq_q6_K(const block_q6_K* src, float* dst, int n) {
    int nb = n / 256;
    for (int b = 0; b < nb; b++) {
        for (int g = 0; g < 16; g++) {
            float d = src[b].scales[g] / 127.0f;
            for (int i = 0; i < 16; i++) {
                dst[b*256+g*16+i] = (src[b].qs[g*12+i/2] & 0x3F) * d;
            }
        }
    }
}

dequantized_tensor_t* load_single_tensor(const char* path, const char* pattern, int* rows, int* cols) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    
    if (read_u32(f) != GGUF_MAGIC) { fclose(f); return NULL; }
    read_u32(f);
    uint64_t num_tensors = read_u64(f);
    uint64_t num_metadata = read_u64(f);
    
    for (uint64_t i = 0; i < num_metadata; i++) {
        char* key = read_string(f);
        int type = read_i32(f);
        skip_value(f, type);
        free(key);
    }
    
    typedef struct { char* name; int type; uint64_t offset; int rows, cols; size_t size; } tinfo_t;
    tinfo_t* tensors = calloc(num_tensors, sizeof(tinfo_t));
    size_t max_offset = 0;
    
    for (uint64_t i = 0; i < num_tensors; i++) {
        tensors[i].name = read_string(f);
        int nd = read_i32(f);
        uint64_t dims[4] = {0}, ne = 1;
        for (int d = 0; d < nd; d++) { dims[d] = read_u64(f); ne *= dims[d]; }
        tensors[i].type = read_i32(f);
        tensors[i].offset = read_u64(f);
        if (tensors[i].offset > max_offset) max_offset = tensors[i].offset;
        
        if (tensors[i].type == 12) tensors[i].size = (ne / 256) * sizeof(block_q4_K);
        else if (tensors[i].type == 14) tensors[i].size = (ne / 256) * sizeof(block_q6_K);
        else tensors[i].size = ne * 4;
        tensors[i].size = (tensors[i].size + 31) & ~31;
        
        if (nd >= 2) { tensors[i].rows = dims[0]; tensors[i].cols = dims[1]; }
    }
    
    size_t data_off = (max_offset + 31) & ~31;
    
    /* Find and load tensor */
    dequantized_tensor_t* result = NULL;
    for (uint64_t i = 0; i < num_tensors; i++) {
        if (!strstr(tensors[i].name, pattern)) continue;
        
        printf("Loading %s [%d, %d] type=%d...\n", tensors[i].name, tensors[i].rows, tensors[i].cols, tensors[i].type);
        
        fseek(f, data_off + tensors[i].offset, SEEK_SET);
        void* data = malloc(tensors[i].size);
        if (!data) { printf("Malloc failed for data\n"); break; }
        fread(data, 1, tensors[i].size, f);
        
        int n = tensors[i].rows * tensors[i].cols;
        float* f32 = aligned_malloc(n * sizeof(float), 64);
        if (!f32) { printf("Malloc failed for f32\n"); free(data); break; }
        
        if (tensors[i].type == 12) deq_q4_K((const block_q4_K*)data, f32, n);
        else if (tensors[i].type == 14) deq_q6_K((const block_q6_K*)data, f32, n);
        else memcpy(f32, data, n * sizeof(float));
        
        result = malloc(sizeof(dequantized_tensor_t));
        if (!result) { printf("Malloc failed for result\n"); aligned_free(f32); free(data); break; }
        
        result->rows = tensors[i].rows;
        result->cols = tensors[i].cols;
        result->weights = aligned_malloc(tensors[i].rows * tensors[i].cols, 64);
        result->scales = aligned_malloc(tensors[i].rows * sizeof(float), 64);
        
        if (!result->weights || !result->scales) {
            printf("Malloc failed for weights/scales\n");
            free(result); aligned_free(f32); free(data); break;
        }
        
        for (int r = 0; r < tensors[i].rows; r++) {
            float max_abs = 0.0f;
            for (int c = 0; c < tensors[i].cols; c++) {
                float v = fabsf(f32[r * tensors[i].cols + c]);
                if (v > max_abs) max_abs = v;
            }
            result->scales[r] = max_abs > 0 ? max_abs / 127.0f : 1.0f;
            float inv = 1.0f / result->scales[r];
            for (int c = 0; c < tensors[i].cols; c++) {
                int val = (int)(f32[r * tensors[i].cols + c] * inv);
                if (val > 127) val = 127;
                if (val < -128) val = -128;
                result->weights[r * tensors[i].cols + c] = (int8_t)val;
            }
        }
        
        *rows = tensors[i].rows;
        *cols = tensors[i].cols;
        
        aligned_free(f32);
        free(data);
        break;  /* Just load first match */
    }
    
    for (uint64_t i = 0; i < num_tensors; i++) free(tensors[i].name);
    free(tensors);
    fclose(f);
    return result;
}

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    const char* path = (argc > 1) ? argv[1] : "..\\models\\Phi-3-mini-4k-instruct-q4.gguf";
    
    printf("\n=== Single Layer Real Weights Test ===\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    /* Load just ffn_up from layer 0 */
    int rows, cols;
    printf("Loading ffn_up weights...\n");
    dequantized_tensor_t* up = load_single_tensor(path, "blk.0.ffn_up", &rows, &cols);
    
    if (!up) {
        printf("Failed to load weights\n");
        return 1;
    }
    
    printf("Loaded: rows=%d, cols=%d\n\n", rows, cols);
    
    /* Benchmark matmul */
    int hidden = rows;
    int inter = cols / 2;  /* Fused */
    
    float* in = aligned_malloc(hidden * sizeof(float), 32);
    float* out = aligned_malloc(cols * sizeof(float), 32);
    for (int i = 0; i < hidden; i++) in[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    
    /* Warmup */
    printf("Warmup...\n");
    for (int i = 0; i < 100; i++) {
        matmul_dequantized(in, up, out, 1, cols, hidden);
    }
    
    /* Benchmark */
    printf("Benchmarking 10000 matmuls...\n");
    double start = get_time_ms();
    for (int i = 0; i < 10000; i++) {
        matmul_dequantized(in, up, out, 1, cols, hidden);
    }
    double elapsed = get_time_ms() - start;
    
    double ms_per = elapsed / 10000.0;
    double gflops = (2.0 * hidden * cols * 10000) / (elapsed * 1e6);
    
    printf("\n=== RESULTS ===\n");
    printf("Total time: %.2f ms\n", elapsed);
    printf("Time per matmul: %.4f ms\n", ms_per);
    printf("GFLOPS: %.2f\n", gflops);
    
    /* Estimate full model (32 layers, 2 matmuls per layer for FFN) */
    double ms_per_token = ms_per * 32 * 2;
    double tok_per_sec = 1000.0 / ms_per_token;
    printf("\nEstimated 32-layer speed: %.2f tok/sec\n", tok_per_sec);
    printf("vs llama.cpp (~25): %.2fx\n", tok_per_sec / 25.0);
    printf("\n");
    
    aligned_free(up->weights);
    aligned_free(up->scales);
    free(up);
    aligned_free(in);
    aligned_free(out);
    
    return 0;
}
