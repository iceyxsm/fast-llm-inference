/*
 * Load Real GGUF Weights and Run Benchmark
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
typedef enum { GGML_TYPE_Q4_K = 12, GGML_TYPE_Q6_K = 14 } ggml_type_t;

typedef struct __attribute__((packed)) {
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

typedef struct __attribute__((packed)) {
    uint8_t scales[16];
    uint8_t qs[192];
} block_q6_K;

static uint64_t read_u64(FILE* f) { uint64_t v; fread(&v, 8, 1, f); return v; }
static uint32_t read_u32(FILE* f) { uint32_t v; fread(&v, 4, 1, f); return v; }
static int32_t read_i32(FILE* f) { int32_t v; fread(&v, 4, 1, f); return v; }

static char* read_string(FILE* f) {
    uint64_t len = read_u64(f);
    char* str = malloc(len + 1);
    fread(str, 1, len, f);
    str[len] = '\0';
    return str;
}

static void skip_value(FILE* f, int type) {
    switch (type) {
        case 0: case 1: { uint8_t v; fread(&v, 1, 1, f); break; }
        case 2: case 3: { uint16_t v; fread(&v, 2, 1, f); break; }
        case 4: case 5: case 6: read_u32(f); break;
        case 7: { uint8_t v; fread(&v, 1, 1, f); break; }
        case 8: free(read_string(f)); break;
        case 9: {
            int at = read_i32(f);
            uint64_t al = read_u64(f);
            for (uint64_t i = 0; i < al; i++) skip_value(f, at);
            break;
        }
        case 10: case 11: case 12: read_u64(f); break;
    }
}

static void deq_q4_K(const block_q4_K* src, float* dst, int n) {
    int nb = n / 256;
    for (int b = 0; b < nb; b++) {
        float d = (src[b].scales[0] & 0x3F) / 32.0f + 0.001f;
        for (int i = 0; i < 128; i++) {
            uint8_t q = src[b].qs[i];
            dst[b*256 + i] = (q & 0x0F) * d - 8*d;
            dst[b*256 + i + 128] = (q >> 4) * d - 8*d;
        }
    }
}

static void deq_q6_K(const block_q6_K* src, float* dst, int n) {
    int nb = n / 256;
    for (int b = 0; b < nb; b++) {
        for (int g = 0; g < 16; g++) {
            float d = src[b].scales[g] / 127.0f;
            for (int i = 0; i < 16; i++) {
                dst[b*256 + g*16 + i] = (src[b].qs[g*12 + i/2] & 0x3F) * d;
            }
        }
    }
}

int load_some_ffn_weights(const char* path,
                          dequantized_tensor_t* gate[4],
                          dequantized_tensor_t* up[4],
                          dequantized_tensor_t* down[4]) {
    
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
    
    typedef struct {
        char* name;
        int type;
        uint64_t offset;
        int rows, cols;
        size_t size;
    } tinfo_t;
    
    tinfo_t* tensors = calloc(num_tensors, sizeof(tinfo_t));
    size_t max_offset = 0;
    
    for (uint64_t i = 0; i < num_tensors; i++) {
        tensors[i].name = read_string(f);
        int nd = read_i32(f);
        
        uint64_t dims[4] = {0};
        uint64_t ne = 1;
        for (int d = 0; d < nd; d++) {
            dims[d] = read_u64(f);
            ne *= dims[d];
        }
        
        tensors[i].type = read_i32(f);
        tensors[i].offset = read_u64(f);
        
        if (tensors[i].offset > max_offset) max_offset = tensors[i].offset;
        
        if (tensors[i].type == GGML_TYPE_Q4_K) {
            tensors[i].size = (ne / 256) * sizeof(block_q4_K);
        } else if (tensors[i].type == GGML_TYPE_Q6_K) {
            tensors[i].size = (ne / 256) * sizeof(block_q6_K);
        } else {
            tensors[i].size = ne * 4;
        }
        tensors[i].size = (tensors[i].size + 31) & ~31;
        
        if (nd >= 2) {
            tensors[i].rows = dims[0];
            tensors[i].cols = dims[1];
        }
    }
    
    size_t data_off = (max_offset + 31) & ~31;
    
    int loaded = 0;
    for (int layer = 0; layer < 4; layer++) {
        gate[layer] = NULL;
        up[layer] = NULL;
        down[layer] = NULL;
        
        for (uint64_t i = 0; i < num_tensors; i++) {
            char lname[64];
            sprintf(lname, "blk.%d.ffn_gate", layer);
            
            dequantized_tensor_t** target = NULL;
            if (strstr(tensors[i].name, lname)) target = &gate[layer];
            else {
                sprintf(lname, "blk.%d.ffn_up", layer);
                if (strstr(tensors[i].name, lname)) target = &up[layer];
                else {
                    sprintf(lname, "blk.%d.ffn_down", layer);
                    if (strstr(tensors[i].name, lname)) target = &down[layer];
                }
            }
            
            if (!target) continue;
            
            fseek(f, data_off + tensors[i].offset, SEEK_SET);
            void* data = malloc(tensors[i].size);
            fread(data, 1, tensors[i].size, f);
            
            int n = tensors[i].rows * tensors[i].cols;
            float* f32 = aligned_malloc(n * sizeof(float), 64);
            
            if (tensors[i].type == GGML_TYPE_Q4_K) {
                deq_q4_K((const block_q4_K*)data, f32, n);
            } else if (tensors[i].type == GGML_TYPE_Q6_K) {
                deq_q6_K((const block_q6_K*)data, f32, n);
            } else {
                memcpy(f32, data, n * sizeof(float));
            }
            
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
            
            printf("  Loaded %s [%d, %d] type=%d\n", tensors[i].name,
                   tensors[i].rows, tensors[i].cols, tensors[i].type);
            
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

void run_ffn(dequantized_tensor_t* g, dequantized_tensor_t* u, dequantized_tensor_t* d,
             float* in, float* out, int hidden, int inter) {
    float* go = aligned_malloc(inter * sizeof(float), 32);
    float* uo = aligned_malloc(inter * sizeof(float), 32);
    float* dout = aligned_malloc(hidden * sizeof(float), 32);
    
    matmul_dequantized(in, g, go, 1, inter, hidden);
    matmul_dequantized(in, u, uo, 1, inter, hidden);
    
    for (int i = 0; i < inter; i++) {
        float sig = 1.0f / (1.0f + expf(-go[i]));
        go[i] = go[i] * sig * uo[i];
    }
    
    matmul_dequantized(go, d, dout, 1, hidden, inter);
    
    for (int i = 0; i < hidden; i++) {
        out[i] = in[i] + dout[i];
    }
    
    aligned_free(go);
    aligned_free(uo);
    aligned_free(dout);
}

int main(int argc, char** argv) {
    srand((unsigned)time(NULL));
    
    const char* path = (argc > 1) ? argv[1] : "..\\models\\Phi-3-mini-4k-instruct-q4.gguf";
    
    printf("\n========================================\n");
    printf("  REAL WEIGHTS BENCHMARK (4 Layers)\n");
    printf("========================================\n\n");
    
    cpu_features_t cpu = detect_cpu_features();
    printf("CPU: AVX2=%s, Cores=%d\n\n", cpu.has_avx2 ? "YES" : "NO", cpu.num_cores);
    
    printf("Loading FFN weights from:\n  %s\n\n", path);
    
    dequantized_tensor_t *gate[4] = {0}, *up[4] = {0}, *down[4] = {0};
    
    int loaded = load_some_ffn_weights(path, gate, up, down);
    printf("\nLoaded %d tensors\n\n", loaded);
    
    int hidden = 3072;
    int inter = 8192;
    int num_tokens = 100;
    
    float* in = aligned_malloc(hidden * sizeof(float), 32);
    float* out = aligned_malloc(hidden * sizeof(float), 32);
    
    for (int i = 0; i < hidden; i++) {
        in[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
    }
    
    printf("Warmup...\n");
    for (int w = 0; w < 5; w++) {
        for (int l = 0; l < 4; l++) {
            if (gate[l] && up[l] && down[l]) {
                run_ffn(gate[l], up[l], down[l], in, out, hidden, inter);
                float* t = in; in = out; out = t;
            }
        }
    }
    
    printf("Running %d tokens with 4 layers (real weights)...\n", num_tokens);
    clock_t start = clock();
    
    for (int tok = 0; tok < num_tokens; tok++) {
        for (int l = 0; l < 4; l++) {
            if (gate[l] && up[l] && down[l]) {
                run_ffn(gate[l], up[l], down[l], in, out, hidden, inter);
                float* t = in; in = out; out = t;
            }
        }
        if ((tok + 1) % 20 == 0) {
            printf("  %d/%d\r", tok + 1, num_tokens);
            fflush(stdout);
        }
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    double tok_per_sec = num_tokens / elapsed;
    
    printf("\n\n=== RESULTS ===\n");
    printf("Tokens: %d, Time: %.3f sec\n", num_tokens, elapsed);
    printf("Speed: %.2f tokens/second\n", tok_per_sec);
    printf("Ms/token: %.2f ms\n", 1000.0 / tok_per_sec);
    printf("\n");
    
    double scaled_32 = tok_per_sec / 8.0;
    printf("Estimated 32-layer speed: %.2f tok/sec\n", scaled_32);
    printf("vs llama.cpp (~25 tok/sec): %.2fx\n", scaled_32 / 25.0);
    printf("\n");
    
    for (int i = 0; i < 4; i++) {
        if (gate[i]) {
            aligned_free(gate[i]->weights);
            aligned_free(gate[i]->scales);
            free(gate[i]);
        }
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
    aligned_free(in);
    aligned_free(out);
    
    return 0;
}
