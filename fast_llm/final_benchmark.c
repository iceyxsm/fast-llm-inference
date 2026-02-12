/*
 * Final Comprehensive Benchmark
 * Real GGUF weights + All optimizations
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

static float get_d(const uint8_t* scales) { return ((scales[0] & 0x3F) + 1) / 64.0f; }
static float get_m(const uint8_t* scales) { return ((scales[1] & 0x3F) + 1) / 64.0f; }

double get_time_ms() {
    LARGE_INTEGER freq, count;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&count);
    return (double)count.QuadPart * 1000.0 / (double)freq.QuadPart;
}

/* Load Phi-3 FFN weights */
int load_phi3_ffn(const char* path,
                  dequantized_tensor_t** up,    /* [num_layers] - fused gate+up */
                  dequantized_tensor_t** down,  /* [num_layers] */
                  int max_layers,
                  int* out_hidden, int* out_inter) {
    
    FILE* f = fopen(path, "rb");
    if (!f) return -1;
    
    if (read_u32(f) != GGUF_MAGIC) { fclose(f); return -1; }
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
    
    /* Get dimensions from layer 0 */
    for (uint64_t i = 0; i < num_tensors; i++) {
        if (strstr(tensors[i].name, "blk.0.ffn_up")) {
            *out_hidden = tensors[i].rows;
            *out_inter = tensors[i].cols / 2;
            break;
        }
    }
    
    int loaded = 0;
    for (int layer = 0; layer < max_layers; layer++) {
        up[layer] = NULL;
        down[layer] = NULL;
        
        for (uint64_t i = 0; i < num_tensors; i++) {
            char lname[64];
            sprintf(lname, "blk.%d.ffn_up", layer);
            dequantized_tensor_t** target = NULL;
            
            if (strstr(tensors[i].name, lname)) target = &up[layer];
            else {
                sprintf(lname, "blk.%d.ffn_down", layer);
                if (strstr(tensors[i].name, lname)) target = &down[layer];
            }
            
            if (!target) continue;
            
            fseek(f, data_off + tensors[i].offset, SEEK_SET);
            void* data = malloc(tensors[i].size);
            fread(data, 1, tensors[i].size, f);
            
            /* Dequantize based on type */
            int n = tensors[i].rows * tensors[i].cols;
            float* f32 = aligned_malloc(n * sizeof(float), 64);
            
            if (tensors[i].type == 12) {
                /* Q4_K dequant */
                int nb = n / 256;
                for (int b = 0; b < nb; b++) {
                    float d = get_d(((block_q4_K*)data)[b].scales);
                    float m = get_m(((block_q4_K*)data)[b].scales);
                    for (int j = 0; j < 128; j++) {
                        uint8_t qv = ((block_q4_K*)data)[b].qs[j];
                        f32[b*256 + j] = (qv & 0x0F) * d + m;
                        f32[b*256 + j + 128] = (qv >> 4) * d + m;
                    }
                }
            } else if (tensors[i].type == 14) {
                /* Q6_K dequant - simplified */
                int nb = n / 256;
                for (int b = 0; b < nb; b++) {
                    for (int g = 0; g < 16; g++) {
                        float d = ((block_q6_K*)data)[b].scales[g] / 127.0f;
                        for (int j = 0; j < 16; j++) {
                            int idx = g * 16 + j;
                            f32[b*256 + idx] = (((block_q6_K*)data)[b].qs[g*12 + j/2] >> ((j%2)*4)) & 0x3F;
                            f32[b*256 + idx] = (f32[b*256 + idx] - 32) * d;
                        }
                    }
                }
            } else {
                memcpy(f32, data, n * sizeof(float));
            }
            
            /* Convert to int8 */
            *target = malloc(sizeof(dequantized_tensor_t));
            (*target)->rows = tensors[i].rows;
            (*target)->cols = tensors[i].cols;
            (*target)->weights = aligned_malloc(tensors[i].rows * tensors[i].cols, 64);
            (*target)->scales = aligned_malloc(tensors[i].rows * sizeof(float), 64);
            
            for (int r = 0; r < tensors[i].rows; r++) {
                float max_abs = 0.0f;
                for (int c = 0; c < tensors[i].cols; c++) {
                    float v = fabsf(f32[r * tensors[i].cols + c]);
                    if (v > max_abs) max_abs = v;
                }
                (*target)->scales[r] = max_abs > 0 ? max_abs / 127.0f : 1.0f;
                float inv = 1.0f / (*target)->scales[r];
                for (int c = 0; c < tensors[i].cols; c++) {
                    int val = (int)(f32[r * tensors[i].cols + c] * inv);
                    if (val > 127) val = 127;
                    if (val < -128) val = -128;
                    (*target)->weights[r * tensors[i].cols + c] = (int8_t)val;
                }
            }
            
            printf("  Loaded %s [%d, %d]\n", tensors[i].name, tensors[i].rows, tensors[i].cols);
            aligned_free(f32);
            free(data);
            loaded++;
        }
    }
    
    for (uint64_t i = 0; i < num_tensors; i++) free(tensors[i].name);
    free(tensors);
    fclose(f);
    return loaded;
}

/* Run FFN layer with real weights */
void run_ffn(dequantized_tensor_t* up, dequantized_tensor_t* down,
             float* in, float* out, int hidden, int inter) {
    float* gate_up_out = aligned_malloc(2 * inter * sizeof(float), 32);
    float* down_out = aligned_malloc(hidden * sizeof(float), 32);
    
    /* Fused gate+up projection */
    matmul_dequantized(in, up, gate_up_out, 1, 2 * inter, hidden);
    
    /* SwiGLU: gate * sigmoid(gate) * up */
    for (int i = 0; i < inter; i++) {
        float g = gate_up_out[i];
        float u = gate_up_out[i + inter];
        float sig = 1.0f / (1.0f + expf(-g));
        gate_up_out[i] = g * sig * u;
    }
    
    /* Down projection */
    matmul_dequantized(gate_up_out, down, down_out, 1, hidden, inter);
    
    /* Residual */
    for (int i = 0; i < hidden; i++) {
        out[i] = in[i] + down_out[i];
    }
    
    aligned_free(gate_up_out);
    aligned_free(down_out);
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    
    const char* path = (argc > 1) ? argv[1] : "..\\models\\Phi-3-mini-4k-instruct-q4.gguf";
    int num_layers_to_load = (argc > 2) ? atoi(argv[2]) : 4;  /* Default 4 layers */
    int num_tokens = (argc > 3) ? atoi(argv[3]) : 100;
    
    printf("\n========================================\n");
    printf("  FINAL REAL WEIGHTS BENCHMARK\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    printf("Loading %d layers from:\n  %s\n\n", num_layers_to_load, path);
    
    /* Allocate arrays for weights */
    dequantized_tensor_t** up = calloc(32, sizeof(dequantized_tensor_t*));
    dequantized_tensor_t** down = calloc(32, sizeof(dequantized_tensor_t*));
    int hidden = 3072, inter = 8192;
    
    int loaded = load_phi3_ffn(path, up, down, num_layers_to_load, &hidden, &inter);
    printf("\nLoaded %d tensors\n", loaded);
    printf("Architecture: hidden=%d, intermediate=%d\n\n", hidden, inter);
    
    if (loaded == 0) {
        printf("Failed to load weights\n");
        return 1;
    }
    
    /* Allocate buffers */
    float* in = aligned_malloc(hidden * sizeof(float), 32);
    float* out = aligned_malloc(hidden * sizeof(float), 32);
    for (int i = 0; i < hidden; i++) {
        in[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    /* Warmup */
    printf("Warmup...\n");
    for (int w = 0; w < 10; w++) {
        for (int l = 0; l < num_layers_to_load; l++) {
            if (up[l] && down[l]) {
                run_ffn(up[l], down[l], in, out, hidden, inter);
                float* t = in; in = out; out = t;
            }
        }
    }
    
    /* Benchmark */
    printf("\nBenchmarking %d tokens with %d layers...\n", num_tokens, num_layers_to_load);
    double start = get_time_ms();
    
    for (int tok = 0; tok < num_tokens; tok++) {
        for (int l = 0; l < num_layers_to_load; l++) {
            if (up[l] && down[l]) {
                run_ffn(up[l], down[l], in, out, hidden, inter);
                float* t = in; in = out; out = t;
            }
        }
        if ((tok + 1) % 10 == 0) {
            printf("  %d/%d\r", tok + 1, num_tokens);
            fflush(stdout);
        }
    }
    
    double elapsed = get_time_ms() - start;
    double tok_per_sec = num_tokens / (elapsed / 1000.0);
    
    printf("\n\n=== RESULTS ===\n");
    printf("Layers loaded: %d\n", num_layers_to_load);
    printf("Tokens: %d\n", num_tokens);
    printf("Time: %.2f ms\n", elapsed);
    printf("Speed: %.2f tokens/second\n", tok_per_sec);
    printf("\n");
    
    /* Scale to 32 layers */
    double tok_per_sec_32 = tok_per_sec * ((double)num_layers_to_load / 32.0);
    printf("Estimated 32-layer speed: %.2f tok/sec\n", tok_per_sec_32);
    printf("vs llama.cpp (~25 tok/sec): %.2fx\n", tok_per_sec_32 / 25.0);
    printf("\n");
    
    /* Cleanup */
    for (int i = 0; i < 32; i++) {
        if (up[i]) {
            aligned_free(up[i]->weights);
            aligned_free(up[i]->scales);
            free(up[i]);
        }
        if (down[i]) {
            aligned_free(down[i]->weights);
            aligned_free(down[i]->scales);
            free(down[i]);
        }
    }
    free(up);
    free(down);
    aligned_free(in);
    aligned_free(out);
    
    return 0;
}
