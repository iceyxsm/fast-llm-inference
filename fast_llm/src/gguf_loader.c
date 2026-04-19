/*
 * GGUF Format Loader - Complete Implementation
 * Loads real weights from Phi-3 Mini GGUF model
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <math.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "model_loader.h"

/* GGUF magic number */
#define GGUF_MAGIC 0x46554747  /* "GGUF" in little endian */

/* GGML tensor types */
typedef enum {
    GGML_TYPE_F32  = 0,
    GGML_TYPE_F16  = 1,
    GGML_TYPE_Q4_0 = 2,
    GGML_TYPE_Q4_1 = 3,
    GGML_TYPE_Q5_0 = 6,
    GGML_TYPE_Q5_1 = 7,
    GGML_TYPE_Q8_0 = 8,
    GGML_TYPE_Q8_1 = 9,
    GGML_TYPE_Q4_K = 12,
    GGML_TYPE_Q5_K = 13,
    GGML_TYPE_Q6_K = 14,
    GGML_TYPE_IQ4_XS = 29,
} ggml_type_t;

/* Q4_0 block: 32 4-bit weights + 1 float16 scale */
typedef struct {
    float d;           /* delta/scale */
    uint8_t qs[16];    /* 32 4-bit values packed */
} block_q4_0;

/* Q4_K block: 256 weights, based on llama.cpp ggml-common.h */
typedef struct {
    uint16_t d;            /* super-block scale (f16) */
    uint16_t dmin;         /* super-block min (f16) */
    uint8_t scales[12];    /* 6-bit scales and mins packed */
    uint8_t qs[128];       /* 256 4-bit values */
} block_q4_K;

/* Q5_K block: 256 weights, 5-bit */
typedef struct {
    uint16_t d;            /* super-block scale (f16) */
    uint16_t dmin;         /* super-block min (f16) */
    uint8_t scales[12];    /* 6-bit scales and mins packed */
    uint8_t qh[32];        /* 256 high bits */
    uint8_t qs[128];       /* 256 low 4 bits */
} block_q5_K;

/* Q6_K block: 256 weights, 6-bit */
typedef struct {
    uint8_t ql[128];       /* quants, lower 4 bits */
    uint8_t qh[64];        /* quants, upper 2 bits */
    int8_t  scales[16];    /* scales, quantized with 8 bits */
    uint16_t d;            /* super-block scale (f16) */
} block_q6_K;

/* Tensor info */
typedef struct {
    char* name;
    int n_dims;
    uint64_t dims[4];
    int type;
    uint64_t offset;
    uint64_t size;
} tensor_info_t;

static uint64_t read_u64(FILE* f) {
    uint64_t v;
    fread(&v, 8, 1, f);
    return v;
}

static uint32_t read_u32(FILE* f) {
    uint32_t v;
    fread(&v, 4, 1, f);
    return v;
}

static int32_t read_i32(FILE* f) {
    int32_t v;
    fread(&v, 4, 1, f);
    return v;
}

static uint8_t read_u8(FILE* f) {
    uint8_t v;
    fread(&v, 1, 1, f);
    return v;
}

static char* read_string(FILE* f) {
    uint64_t len = read_u64(f);
    char* str = malloc(len + 1);
    fread(str, 1, len, f);
    str[len] = '\0';
    return str;
}

static void skip_value(FILE* f, int type) {
    switch (type) {
        case 0: case 1: read_u8(f); break;
        case 2: case 3: fread(&(uint16_t){0}, 2, 1, f); break;
        case 4: case 5: case 6: read_u32(f); break;
        case 7: read_u8(f); break;
        case 8: free(read_string(f)); break;
        case 9: {
            int arr_type = read_i32(f);
            uint64_t arr_len = read_u64(f);
            for (uint64_t i = 0; i < arr_len; i++) skip_value(f, arr_type);
            break;
        }
        case 10: case 11: case 12: read_u64(f); break;
    }
}

/* Get block size and type size for GGML types */
static void get_type_info(int type, int* block_size, int* type_size) {
    switch (type) {
        case GGML_TYPE_F32:  *block_size = 1; *type_size = 4; break;
        case GGML_TYPE_F16:  *block_size = 1; *type_size = 2; break;
        case GGML_TYPE_Q4_0: *block_size = 32; *type_size = sizeof(block_q4_0); break;
        case GGML_TYPE_Q4_K: *block_size = 256; *type_size = sizeof(block_q4_K); break;
        case GGML_TYPE_Q5_K: *block_size = 256; *type_size = sizeof(block_q5_K); break;
        case GGML_TYPE_Q6_K: *block_size = 256; *type_size = sizeof(block_q6_K); break;
        default: *block_size = 1; *type_size = 4; break;
    }
}

/* Dequantize Q4_0 to float */
static void dequantize_q4_0(const void* src, float* dst, int n) {
    const block_q4_0* blocks = (const block_q4_0*)src;
    int num_blocks = n / 32;
    
    for (int b = 0; b < num_blocks; b++) {
        float d = blocks[b].d;
        for (int i = 0; i < 16; i++) {
            uint8_t qs = blocks[b].qs[i];
            int x0 = (qs & 0x0F) - 8;  /* Lower nibble */
            int x1 = (qs >> 4) - 8;    /* Upper nibble */
            dst[b * 32 + i] = x0 * d;
            dst[b * 32 + i + 16] = x1 * d;
        }
    }
}

/* Helper: convert f16 (uint16_t) to float */
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp  = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x3FF;
    uint32_t f;
    if (exp == 0) {
        if (mant == 0) f = sign;
        else { exp = 1; while (!(mant & 0x400)) { mant <<= 1; exp--; } mant &= 0x3FF; f = sign | ((exp + 127 - 15) << 23) | (mant << 13); }
    } else if (exp == 31) {
        f = sign | 0x7F800000 | (mant << 13);
    } else {
        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
    }
    float result;
    memcpy(&result, &f, 4);
    return result;
}

/* Helper: unpack scale and min for Q4_K/Q5_K (matches llama.cpp get_scale_min_k4) */
static inline void get_scale_min_k4(int j, const uint8_t *q, uint8_t *d, uint8_t *m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}

/* 
 * Dequantize Q4_K to float — matches llama.cpp dequantize_row_q4_K exactly
 * 256 weights per super-block, 8 sub-blocks of 32 weights
 */
static void dequantize_q4_K(const void* src, float* dst, int n) {
    const block_q4_K* x = (const block_q4_K*)src;
    int num_blocks = n / 256;

    for (int i = 0; i < num_blocks; i++) {
        const uint8_t* q = x[i].qs;
        float d   = f16_to_f32(x[i].d);
        float min = f16_to_f32(x[i].dmin);
        int is = 0;
        float* y = dst + i * 256;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}

/*
 * Dequantize Q5_K to float — matches llama.cpp dequantize_row_q5_K exactly
 */
static void dequantize_q5_K(const void* src, float* dst, int n) {
    const block_q5_K* x = (const block_q5_K*)src;
    int num_blocks = n / 256;

    for (int i = 0; i < num_blocks; i++) {
        const uint8_t* ql = x[i].qs;
        const uint8_t* qh = x[i].qh;
        float d   = f16_to_f32(x[i].d);
        float min = f16_to_f32(x[i].dmin);
        int is = 0;
        uint8_t u1 = 1, u2 = 2;
        float* y = dst + i * 256;
        for (int j = 0; j < 256; j += 64) {
            uint8_t sc, m;
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            float d1 = d * sc; float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            float d2 = d * sc; float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * ((ql[l] & 0xF) + (qh[l] & u1 ? 16 : 0)) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * ((ql[l]  >> 4) + (qh[l] & u2 ? 16 : 0)) - m2;
            ql += 32; is += 2;
            u1 <<= 2; u2 <<= 2;
        }
    }
}

/*
 * Dequantize Q6_K to float — matches llama.cpp dequantize_row_q6_K exactly
 */
static void dequantize_q6_K(const void* src, float* dst, int n) {
    const block_q6_K* x = (const block_q6_K*)src;
    int num_blocks = n / 256;

    for (int i = 0; i < num_blocks; i++) {
        float d = f16_to_f32(x[i].d);
        const uint8_t* ql = x[i].ql;
        const uint8_t* qh = x[i].qh;
        const int8_t*  sc = x[i].scales;
        float* y = dst + i * 256;
        for (int nn = 0; nn < 256; nn += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}

/* Parse tensor name to identify layer and projection type */
typedef enum {
    PROJ_UNKNOWN = 0,
    PROJ_Q,
    PROJ_K,
    PROJ_V,
    PROJ_O,
    PROJ_GATE,
    PROJ_UP,
    PROJ_DOWN,
    PROJ_EMBED,
    PROJ_LM_HEAD,
    PROJ_NORM,
} proj_type_t;

static void parse_tensor_name(const char* name, int* layer, proj_type_t* proj) {
    *layer = -1;
    *proj = PROJ_UNKNOWN;

    /* Extract layer number FIRST */
    const char* blk = strstr(name, "blk.");
    if (blk) *layer = atoi(blk + 4);

    /* Skip bias tensors — we only handle weights */
    if (strstr(name, ".bias")) return;

    /* Token embeddings */
    if (strstr(name, "token_embd") || strstr(name, "tok_embeddings")) {
        *proj = PROJ_EMBED; *layer = -1; return;
    }

    /* LM head */
    if (strstr(name, "lm_head")) { *proj = PROJ_LM_HEAD; *layer = -1; return; }
    if (strstr(name, "output.weight") && *layer < 0) {
        *proj = PROJ_LM_HEAD; return;
    }

    /* Final output norm (not in a block) */
    if (strstr(name, "output_norm") && *layer < 0) {
        *proj = PROJ_NORM; return;
    }

    /* Layer norms (only if inside a block) */
    if (*layer >= 0 && strstr(name, "norm")) { *proj = PROJ_NORM; return; }

    /* Attention projections */
    if (strstr(name, "attn_q") || strstr(name, "q_proj")) { *proj = PROJ_Q; return; }
    if (strstr(name, "attn_k") || strstr(name, "k_proj")) { *proj = PROJ_K; return; }
    if (strstr(name, "attn_v") || strstr(name, "v_proj")) { *proj = PROJ_V; return; }
    if (strstr(name, "attn_output") || strstr(name, "attn_o") || strstr(name, "o_proj")) { *proj = PROJ_O; return; }
    if (strstr(name, "attn_qkv")) { *proj = PROJ_Q; return; }

    /* FFN projections */
    if (strstr(name, "ffn_gate") || strstr(name, "gate_proj")) { *proj = PROJ_GATE; return; }
    if (strstr(name, "ffn_up") || strstr(name, "up_proj")) { *proj = PROJ_UP; return; }
    if (strstr(name, "ffn_down") || strstr(name, "down_proj")) { *proj = PROJ_DOWN; return; }
}

/* Load GGUF file and create model with real weights */
transformer_model_t* model_load_gguf(const char* path, int use_int8) {
    (void)use_int8;
    
    printf("Loading GGUF model: %s\n", path);
    
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Error: Cannot open file %s\n", path);
        return NULL;
    }
    
    /* Read header */
    uint32_t magic = read_u32(f);
    if (magic != GGUF_MAGIC) {
        printf("Error: Invalid GGUF magic (expected 0x%08X, got 0x%08X)\n", GGUF_MAGIC, magic);
        fclose(f);
        return NULL;
    }
    
    uint32_t version = read_u32(f);
    uint64_t num_tensors = read_u64(f);
    uint64_t num_metadata = read_u64(f);
    
    printf("  Version: %d\n", version);
    printf("  Tensors: %llu\n", (unsigned long long)num_tensors);
    printf("  Metadata: %llu\n", (unsigned long long)num_metadata);
    
    /* Default architecture */
    model_config_t config = {
        .vocab_size = 32064,
        .hidden_size = 3072,
        .intermediate_size = 8192,
        .num_layers = 32,
        .num_heads = 32,
        .num_kv_heads = 32,
        .head_dim = 96,
        .max_seq_len = 4096
    };
    
    /* Read metadata */
    char** temp_vocab = NULL;
    int temp_vocab_size = 0;
    for (uint64_t i = 0; i < num_metadata; i++) {
        char* key = read_string(f);
        int type = read_i32(f);
        
        if (strcmp(key, "phi3.embedding_length") == 0 ||
            strcmp(key, "llama.embedding_length") == 0) {
            config.hidden_size = read_i32(f);
        } else if (strcmp(key, "phi3.feed_forward_length") == 0 ||
                   strcmp(key, "llama.feed_forward_length") == 0) {
            config.intermediate_size = read_i32(f);
        } else if (strcmp(key, "phi3.block_count") == 0 ||
                   strcmp(key, "llama.block_count") == 0) {
            config.num_layers = read_i32(f);
        } else if (strcmp(key, "phi3.attention.head_count") == 0 ||
                   strcmp(key, "llama.attention.head_count") == 0) {
            config.num_heads = read_i32(f);
        } else if (strcmp(key, "phi3.attention.head_count_kv") == 0 ||
                   strcmp(key, "llama.attention.head_count_kv") == 0) {
            config.num_kv_heads = read_i32(f);
        } else if (strcmp(key, "phi3.context_length") == 0 ||
                   strcmp(key, "llama.context_length") == 0) {
            config.max_seq_len = read_i32(f);
        } else if (strcmp(key, "tokenizer.ggml.tokens") == 0) {
            /* Read vocabulary: array of strings */
            if (type == 9) { /* array type */
                int arr_type = read_i32(f);
                uint64_t arr_len = read_u64(f);
                if (arr_type == 8 && arr_len > 0 && arr_len <= 200000) {
                    /* Store temporarily — will attach to model later */
                    temp_vocab = (char**)malloc(arr_len * sizeof(char*));
                    temp_vocab_size = (int)arr_len;
                    for (uint64_t i = 0; i < arr_len; i++) {
                        temp_vocab[i] = read_string(f);
                    }
                } else {
                    for (uint64_t i = 0; i < arr_len; i++) skip_value(f, arr_type);
                }
            } else {
                skip_value(f, type);
            }
        } else {
            skip_value(f, type);
        }
        
        free(key);
    }
    
    config.head_dim = config.hidden_size / config.num_heads;

    /* Update vocab size from tokenizer if loaded AND no metadata vocab_size was set */
    if (temp_vocab_size > 0 && config.vocab_size <= 32064) {
        /* Cap at a reasonable size to avoid huge allocations */
        /* The actual embedding tensor will tell us the real size */
        config.vocab_size = temp_vocab_size;
        if (config.vocab_size > 200000) config.vocab_size = 200000;
    }
    
    printf("\nArchitecture:\n");
    printf("  Hidden size: %d\n", config.hidden_size);
    printf("  Intermediate: %d\n", config.intermediate_size);
    printf("  Layers: %d\n", config.num_layers);
    printf("  Heads: %d / %d\n", config.num_heads, config.num_kv_heads);
    printf("  Max seq len: %d\n", config.max_seq_len);
    
    /* Read tensor info */
    tensor_info_t* tensors = calloc(num_tensors, sizeof(tensor_info_t));
    
    for (uint64_t i = 0; i < num_tensors; i++) {
        tensors[i].name = read_string(f);
        tensors[i].n_dims = read_i32(f);
        
        uint64_t num_elements = 1;
        for (int d = 0; d < tensors[i].n_dims; d++) {
            tensors[i].dims[d] = read_u64(f);
            num_elements *= tensors[i].dims[d];
        }
        
        tensors[i].type = read_i32(f);
        tensors[i].offset = read_u64(f);
        
        /* Calculate size */
        int block_size, type_size;
        get_type_info(tensors[i].type, &block_size, &type_size);
        tensors[i].size = (num_elements / block_size) * type_size;
        
        /* Align to 32 bytes */
        tensors[i].size = (tensors[i].size + 31) & ~31;
    }
    
    /* The data section starts right after tensor info, aligned to 32 bytes.
       Tensor offsets are RELATIVE to the data section start. */
    long current_pos = ftell(f);
    size_t data_start = (current_pos + 31) & ~(size_t)31;

    printf("\nData section at file offset: %zu\n", data_start);
    printf("Loading %llu tensors...\n\n", (unsigned long long)num_tensors);

    /* Read each tensor by seeking to data_start + tensor.offset */
    
    /* Allocate model */
    transformer_model_t* model = calloc(1, sizeof(transformer_model_t));
    model->config = config;
    strcpy(model->config.model_name, "Phi-3-Mini-Real");

    /* Attach tokenizer vocabulary if loaded */
    if (temp_vocab && temp_vocab_size > 0) {
        model->vocab_tokens = temp_vocab;
        model->vocab_loaded = 1;
        printf("  Tokenizer: %d tokens loaded\n", temp_vocab_size);
    }
    
    /* Allocate weight arrays */
    model->q_proj = calloc(config.num_layers, sizeof(dequantized_tensor_t*));
    model->k_proj = calloc(config.num_layers, sizeof(dequantized_tensor_t*));
    model->v_proj = calloc(config.num_layers, sizeof(dequantized_tensor_t*));
    model->o_proj = calloc(config.num_layers, sizeof(dequantized_tensor_t*));
    model->gate_proj = calloc(config.num_layers, sizeof(dequantized_tensor_t*));
    model->up_proj = calloc(config.num_layers, sizeof(dequantized_tensor_t*));
    model->down_proj = calloc(config.num_layers, sizeof(dequantized_tensor_t*));
    model->input_layernorm = calloc(config.num_layers, sizeof(float*));
    model->post_attn_layernorm = calloc(config.num_layers, sizeof(float*));
    
    /* Allocate token embeddings */
    model->token_embeddings = aligned_malloc(config.vocab_size * config.hidden_size * sizeof(float), 64);
    
    /* Allocate LM head (minimal — will be filled during tensor loading or tied) */
    model->lm_head = calloc(1, sizeof(dequantized_tensor_t));
    model->lm_head->rows = config.vocab_size;
    model->lm_head->cols = config.hidden_size;
    /* Don't pre-allocate weights — they'll be set during loading or tied */
    
    /* KV cache — use reasonable max context, not the model's theoretical max */
    int actual_max_seq = config.max_seq_len;
    if (actual_max_seq > 4096) actual_max_seq = 4096; /* Cap at 4K for memory */
    size_t kv_size = (size_t)config.num_layers * actual_max_seq * 
                     config.num_kv_heads * config.head_dim * sizeof(float);
    model->k_cache = aligned_malloc(kv_size, 64);
    model->v_cache = aligned_malloc(kv_size, 64);
    if (model->k_cache) memset(model->k_cache, 0, kv_size);
    if (model->v_cache) memset(model->v_cache, 0, kv_size);
    memset(model->v_cache, 0, kv_size);
    
    /* Track loading stats */
    int loaded_tensors = 0;
    int skipped_tensors = 0;
    int unmapped_tensors = 0;
    size_t total_bytes = 0;
    
    /* Read and process each tensor */
    for (uint64_t i = 0; i < num_tensors; i++) {
        /* Seek to this tensor's data: data_start + tensor offset */
        fseek(f, (long)(data_start + tensors[i].offset), SEEK_SET);

        /* Read tensor data */
        void* tensor_data = malloc(tensors[i].size);
        if (!tensor_data) {
            printf("Warning: Failed to allocate tensor %llu (%s)\n", 
                   (unsigned long long)i, tensors[i].name);
            continue;
        }
        
        size_t read = fread(tensor_data, 1, tensors[i].size, f);
        if (read != tensors[i].size) {
            printf("Warning: Short read on %s (%zu < %llu)\n", 
                   tensors[i].name, read, (unsigned long long)tensors[i].size);
        }
        
        /* Parse tensor name */
        int layer_idx;
        proj_type_t proj_type;
        parse_tensor_name(tensors[i].name, &layer_idx, &proj_type);
        
        /* Debug: print first few tensor names */
        if (i < 10 || strstr(tensors[i].name, "output")) {
            printf("  Tensor %d: %s -> layer=%d proj=%d type=%d dims=%llux%llu size=%llu\n", 
                   (int)i, tensors[i].name, layer_idx, proj_type, tensors[i].type,
                   (unsigned long long)tensors[i].dims[0], 
                   (unsigned long long)(tensors[i].n_dims > 1 ? tensors[i].dims[1] : 1),
                   (unsigned long long)tensors[i].size);
        }
        
        /* Get dimensions — GGUF convention:
         *   dims[0] = ne[0] = contiguous dimension = number of elements per row
         *   dims[1] = ne[1] = number of rows
         * For weight matrices: ne[0] = in_features, ne[1] = out_features
         * In memory: ne[1] rows of ne[0] elements = [out_features, in_features]
         * For 1D tensors (norms): dims[0] = vector length
         */
        int ne0 = (tensors[i].n_dims > 0) ? tensors[i].dims[0] : 1;  /* contiguous dim */
        int ne1 = (tensors[i].n_dims > 1) ? tensors[i].dims[1] : 1;  /* rows */
        int rows = ne0;  /* legacy name — actually ne[0] */
        int cols = ne1;  /* legacy name — actually ne[1] */
        int num_elems_1d = ne0;  /* for 1D tensors */
        
        /* Dequantize and store based on type */
        if ((tensors[i].type == GGML_TYPE_Q4_0 || 
            tensors[i].type == GGML_TYPE_Q4_K ||
            tensors[i].type == GGML_TYPE_Q4_1 ||
            tensors[i].type == GGML_TYPE_Q5_K ||
            tensors[i].type == GGML_TYPE_Q6_K) && proj_type == PROJ_EMBED) {
            /* Special handling for embeddings: dequantize row by row
             * GGUF: ne[0]=hidden_size (contiguous), ne[1]=vocab_size
             * In memory: vocab_size rows of hidden_size elements = [vocab_size, hidden_size]
             * We need [vocab_size, hidden_size] for embedding lookup — direct copy! */
            int emb_hidden = ne0;  /* hidden_size */
            int emb_vocab = ne1;   /* vocab_size */
            if (emb_hidden > config.hidden_size) emb_hidden = config.hidden_size;
            if (emb_vocab > config.vocab_size) emb_vocab = config.vocab_size;

            /* Dequantize one row at a time (one row = one token's embedding) */
            float* row_buf = aligned_malloc(emb_hidden * sizeof(float), 64);

            for (int v = 0; v < emb_vocab; v++) {
                /* Dequantize row v (one token embedding of hidden_size elements) */
                int row_bytes = tensors[i].size / ne1;
                const uint8_t* row_data = (const uint8_t*)tensor_data + v * row_bytes;
                if (tensors[i].type == GGML_TYPE_Q6_K) {
                    dequantize_q6_K(row_data, row_buf, emb_hidden);
                } else if (tensors[i].type == GGML_TYPE_Q5_K) {
                    dequantize_q5_K(row_data, row_buf, emb_hidden);
                } else if (tensors[i].type == GGML_TYPE_Q4_K) {
                    dequantize_q4_K(row_data, row_buf, emb_hidden);
                } else if (tensors[i].type == GGML_TYPE_Q4_0) {
                    dequantize_q4_0(row_data, row_buf, emb_hidden);
                }
                /* Direct copy — already [vocab, hidden] layout */
                memcpy(&model->token_embeddings[v * config.hidden_size], row_buf, emb_hidden * sizeof(float));
            }
            aligned_free(row_buf);
            loaded_tensors++;
            total_bytes += rows * cols * 2;
            printf("  [Embed loaded via row-by-row dequant+transpose]\n");
        }
        else if (tensors[i].type == GGML_TYPE_Q4_0 || 
            tensors[i].type == GGML_TYPE_Q4_K ||
            tensors[i].type == GGML_TYPE_Q4_1 ||
            tensors[i].type == GGML_TYPE_Q5_K ||
            tensors[i].type == GGML_TYPE_Q6_K) {
            
            int num_elements = rows * cols;
            float* f32_data = aligned_malloc(num_elements * sizeof(float), 64);
            
            if (tensors[i].type == GGML_TYPE_Q4_0) {
                dequantize_q4_0(tensor_data, f32_data, num_elements);
            } else if (tensors[i].type == GGML_TYPE_Q4_K) {
                dequantize_q4_K(tensor_data, f32_data, num_elements);
            } else if (tensors[i].type == GGML_TYPE_Q5_K) {
                dequantize_q5_K(tensor_data, f32_data, num_elements);
            } else if (tensors[i].type == GGML_TYPE_Q6_K) {
                dequantize_q6_K(tensor_data, f32_data, num_elements);
            }
            
            /* GGUF weight layout (researched from llama.cpp/ggml):
             *   ne[0] (=rows var) = in_features (contiguous in memory)
             *   ne[1] (=cols var) = out_features (number of rows in memory)
             * Data in memory: ne[1] rows × ne[0] elements = [out_features, in_features]
             * Our matvec expects [out_dim, in_dim] — matches directly, no transpose. */
            int out_dim = ne1;  /* ne[1] = out_features */
            int in_dim = ne0;   /* ne[0] = in_features */

            dequantized_tensor_t* dt = malloc(sizeof(dequantized_tensor_t));
            dt->rows = out_dim;
            dt->cols = in_dim;
            dt->weights = NULL;  /* not using int8 */
            dt->scales = NULL;
            dt->f32_weights = aligned_malloc(out_dim * in_dim * sizeof(float), 64);
            dt->original_bits = 0;

            /* Store float32 weights directly — no int8 re-quantization */
            memcpy(dt->f32_weights, f32_data, out_dim * in_dim * sizeof(float));
            
            aligned_free(f32_data);
            
            /* Handle fused QKV - split into Q, K, V */
            if (strstr(tensors[i].name, "attn_qkv")) {
                /* QKV fused: out_dim = 3 * hidden_size, split into 3 parts */
                int hidden_size = config.hidden_size;
                int kv_dim = config.num_kv_heads * config.head_dim;
                /* For GQA: Q is hidden_size, K and V are kv_dim each */
                int q_dim = hidden_size;
                int k_dim = kv_dim;
                int v_dim = kv_dim;
                /* If total doesn't match, fall back to equal split */
                if (q_dim + k_dim + v_dim != out_dim) {
                    q_dim = out_dim / 3;
                    k_dim = out_dim / 3;
                    v_dim = out_dim - 2 * (out_dim / 3);
                }

                if (dt->f32_weights) {
                    /* f32 path */
                    dequantized_tensor_t* dt_q = malloc(sizeof(dequantized_tensor_t));
                    memset(dt_q, 0, sizeof(*dt_q));
                    dt_q->rows = q_dim;
                    dt_q->cols = in_dim;
                    dt_q->f32_weights = aligned_malloc(q_dim * in_dim * sizeof(float), 64);
                    memcpy(dt_q->f32_weights, dt->f32_weights, q_dim * in_dim * sizeof(float));
                    model->q_proj[layer_idx] = dt_q;

                    dequantized_tensor_t* dt_k = malloc(sizeof(dequantized_tensor_t));
                    memset(dt_k, 0, sizeof(*dt_k));
                    dt_k->rows = k_dim;
                    dt_k->cols = in_dim;
                    dt_k->f32_weights = aligned_malloc(k_dim * in_dim * sizeof(float), 64);
                    memcpy(dt_k->f32_weights, dt->f32_weights + q_dim * in_dim, k_dim * in_dim * sizeof(float));
                    model->k_proj[layer_idx] = dt_k;

                    dequantized_tensor_t* dt_v = malloc(sizeof(dequantized_tensor_t));
                    memset(dt_v, 0, sizeof(*dt_v));
                    dt_v->rows = v_dim;
                    dt_v->cols = in_dim;
                    dt_v->f32_weights = aligned_malloc(v_dim * in_dim * sizeof(float), 64);
                    memcpy(dt_v->f32_weights, dt->f32_weights + (q_dim + k_dim) * in_dim, v_dim * in_dim * sizeof(float));
                    model->v_proj[layer_idx] = dt_v;
                } else if (dt->weights && dt->scales) {
                    /* int8 path */
                    dequantized_tensor_t* dt_q = malloc(sizeof(dequantized_tensor_t));
                    memset(dt_q, 0, sizeof(*dt_q));
                    dt_q->rows = q_dim;
                    dt_q->cols = in_dim;
                    dt_q->weights = aligned_malloc(q_dim * in_dim, 64);
                    dt_q->scales = aligned_malloc(q_dim * sizeof(float), 64);
                    memcpy(dt_q->weights, dt->weights, q_dim * in_dim);
                    memcpy(dt_q->scales, dt->scales, q_dim * sizeof(float));
                    model->q_proj[layer_idx] = dt_q;

                    dequantized_tensor_t* dt_k = malloc(sizeof(dequantized_tensor_t));
                    memset(dt_k, 0, sizeof(*dt_k));
                    dt_k->rows = k_dim;
                    dt_k->cols = in_dim;
                    dt_k->weights = aligned_malloc(k_dim * in_dim, 64);
                    dt_k->scales = aligned_malloc(k_dim * sizeof(float), 64);
                    memcpy(dt_k->weights, dt->weights + q_dim * in_dim, k_dim * in_dim);
                    memcpy(dt_k->scales, dt->scales + q_dim, k_dim * sizeof(float));
                    model->k_proj[layer_idx] = dt_k;

                    dequantized_tensor_t* dt_v = malloc(sizeof(dequantized_tensor_t));
                    memset(dt_v, 0, sizeof(*dt_v));
                    dt_v->rows = v_dim;
                    dt_v->cols = in_dim;
                    dt_v->weights = aligned_malloc(v_dim * in_dim, 64);
                    dt_v->scales = aligned_malloc(v_dim * sizeof(float), 64);
                    memcpy(dt_v->weights, dt->weights + (q_dim + k_dim) * in_dim, v_dim * in_dim);
                    memcpy(dt_v->scales, dt->scales + q_dim + k_dim, v_dim * sizeof(float));
                    model->v_proj[layer_idx] = dt_v;
                }
                
                /* Free original fused tensor */
                dequantized_tensor_free(dt);
                
                loaded_tensors += 3;
                total_bytes += rows * cols;
            }
            /* Store in appropriate slot */
            else if (layer_idx >= 0 && layer_idx < config.num_layers) {
                switch (proj_type) {
                    case PROJ_Q: model->q_proj[layer_idx] = dt; break;
                    case PROJ_K: model->k_proj[layer_idx] = dt; break;
                    case PROJ_V: model->v_proj[layer_idx] = dt; break;
                    case PROJ_O: model->o_proj[layer_idx] = dt; break;
                    case PROJ_GATE: model->gate_proj[layer_idx] = dt; break;
                    case PROJ_UP: model->up_proj[layer_idx] = dt; break;
                    case PROJ_DOWN: model->down_proj[layer_idx] = dt; break;
                    default: 
                        dequantized_tensor_free(dt);
                        dt = NULL;
                        break;
                }
                if (dt) {
                    loaded_tensors++;
                    total_bytes += rows * cols;
                }
            } else if (proj_type == PROJ_LM_HEAD) {
                /* Replace LM head */
                if (model->lm_head) {
                    dequantized_tensor_free(model->lm_head);
                }
                model->lm_head = dt;
                loaded_tensors++;
            } else if (proj_type == PROJ_EMBED) {
                /* GGUF stores embeddings as [hidden_size, vocab_size], we need [vocab_size, hidden_size] */
                /* Transpose during copy */
                int emb_hidden = (rows < cols) ? rows : cols;
                int emb_vocab = (rows < cols) ? cols : rows;
                if (emb_hidden > config.hidden_size) emb_hidden = config.hidden_size;
                if (emb_vocab > config.vocab_size) emb_vocab = config.vocab_size;
                printf("  [Embed: tensor %dx%d -> transposing to [%d, %d]]\n", rows, cols, emb_vocab, emb_hidden);

                /* Use f32_weights if available, otherwise reconstruct from int8 */
                float* emb_f32 = NULL;
                int need_free_emb = 0;
                if (dt->f32_weights) {
                    emb_f32 = dt->f32_weights;
                } else if (dt->weights && dt->scales) {
                    emb_f32 = aligned_malloc(rows * cols * sizeof(float), 64);
                    need_free_emb = 1;
                    for (int r = 0; r < rows; r++) {
                        for (int c = 0; c < cols; c++) {
                            emb_f32[r * cols + c] = dt->weights[r * cols + c] * dt->scales[r];
                        }
                    }
                } else {
                    printf("  [Embed: no weights available, skipping]\n");
                    dequantized_tensor_free(dt);
                    free(tensor_data);
                    free(tensors[i].name);
                    continue;
                }

                /* GGUF embeds: ne[0]=hidden, ne[1]=vocab. Data is [vocab, hidden] in memory.
                 * dt->f32_weights is [out_dim=ne1, in_dim=ne0] = [vocab, hidden].
                 * Direct copy to token_embeddings[vocab, hidden]. */
                for (int v = 0; v < emb_vocab; v++) {
                    for (int h = 0; h < emb_hidden; h++) {
                        model->token_embeddings[v * config.hidden_size + h] = emb_f32[v * ne0 + h];
                    }
                }
                if (need_free_emb) aligned_free(emb_f32);
                dequantized_tensor_free(dt);
                loaded_tensors++;
                printf("  [Embed loaded: sample token 1 = %.6f]\n", model->token_embeddings[1 * config.hidden_size]);
            } else {
                skipped_tensors++;
                /* Debug: print first few unmapped tensors */
                if (unmapped_tensors < 10) {
                    printf("  [Unmapped: %s type=%d layer=%d proj=%d]\n", 
                           tensors[i].name, tensors[i].type, layer_idx, proj_type);
                    unmapped_tensors++;
                }
                dequantized_tensor_free(dt);
            }
        } else if (tensors[i].type == GGML_TYPE_F16) {
            /* Handle float16 tensors — convert to f32 then to int8 dequantized */
            int num_elements = rows * cols;
            float* f32_data = aligned_malloc(num_elements * sizeof(float), 64);

            /* Convert F16 to F32 */
            const uint16_t* f16 = (const uint16_t*)tensor_data;
            for (int e = 0; e < num_elements; e++) {
                /* IEEE 754 half-precision to single-precision */
                uint16_t h = f16[e];
                uint32_t sign = (h & 0x8000) << 16;
                uint32_t exp  = (h >> 10) & 0x1F;
                uint32_t mant = h & 0x3FF;
                uint32_t f;
                if (exp == 0) {
                    if (mant == 0) { f = sign; }
                    else { /* subnormal */
                        exp = 1;
                        while (!(mant & 0x400)) { mant <<= 1; exp--; }
                        mant &= 0x3FF;
                        f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                    }
                } else if (exp == 31) {
                    f = sign | 0x7F800000 | (mant << 13); /* inf/nan */
                } else {
                    f = sign | ((exp + 127 - 15) << 23) | (mant << 13);
                }
                memcpy(&f32_data[e], &f, 4);
            }

            /* Handle embeddings — GGUF: ne[0]=hidden, ne[1]=vocab.
             * F16 data is [ne1 rows × ne0 elements] = [vocab, hidden] in memory.
             * Direct copy to token_embeddings[vocab, hidden]. */
            if (proj_type == PROJ_EMBED) {
                int ev = ne1 < config.vocab_size ? ne1 : config.vocab_size;
                int eh = ne0 < config.hidden_size ? ne0 : config.hidden_size;
                for (int v = 0; v < ev; v++)
                    for (int h = 0; h < eh; h++)
                        model->token_embeddings[v * config.hidden_size + h] = f32_data[v * ne0 + h];
                aligned_free(f32_data);
                loaded_tensors++;
                total_bytes += num_elements * 2;
            }
            /* Handle norms (store as float) */
            else if (proj_type == PROJ_NORM && layer_idx >= 0 && layer_idx < config.num_layers) {
                if (strstr(tensors[i].name, "attn_norm") || strstr(tensors[i].name, "input_layernorm")) {
                    model->input_layernorm[layer_idx] = aligned_malloc(num_elems_1d * sizeof(float), 64);
                    memcpy(model->input_layernorm[layer_idx], f32_data, num_elems_1d * sizeof(float));
                    loaded_tensors++;
                } else if (strstr(tensors[i].name, "ffn_norm") || strstr(tensors[i].name, "post_attention_layernorm")) {
                    model->post_attn_layernorm[layer_idx] = aligned_malloc(num_elems_1d * sizeof(float), 64);
                    memcpy(model->post_attn_layernorm[layer_idx], f32_data, num_elems_1d * sizeof(float));
                    loaded_tensors++;
                }
                aligned_free(f32_data);
            }
            /* Handle weight matrices — GGUF: ne[0]=in_features, ne[1]=out_features */
            else {
                int out_dim = ne1;
                int in_dim = ne0;

                dequantized_tensor_t* dt = malloc(sizeof(dequantized_tensor_t));
                dt->rows = out_dim;
                dt->cols = in_dim;
                dt->weights = NULL;
                dt->scales = NULL;
                dt->f32_weights = aligned_malloc(out_dim * in_dim * sizeof(float), 64);
                dt->original_bits = 0;
                memcpy(dt->f32_weights, f32_data, out_dim * in_dim * sizeof(float));

                aligned_free(f32_data);

                /* Store in appropriate slot */
                if (layer_idx >= 0 && layer_idx < config.num_layers) {
                    switch (proj_type) {
                        case PROJ_Q: model->q_proj[layer_idx] = dt; break;
                        case PROJ_K: model->k_proj[layer_idx] = dt; break;
                        case PROJ_V: model->v_proj[layer_idx] = dt; break;
                        case PROJ_O: model->o_proj[layer_idx] = dt; break;
                        case PROJ_GATE: model->gate_proj[layer_idx] = dt; break;
                        case PROJ_UP: model->up_proj[layer_idx] = dt; break;
                        case PROJ_DOWN: model->down_proj[layer_idx] = dt; break;
                        default: dequantized_tensor_free(dt); dt = NULL; break;
                    }
                    if (dt) { loaded_tensors++; total_bytes += rows * cols * 2; }
                } else if (proj_type == PROJ_LM_HEAD) {
                    if (model->lm_head) {
                        dequantized_tensor_free(model->lm_head);
                    }
                    model->lm_head = dt;
                    loaded_tensors++;
                } else {
                    if (unmapped_tensors < 10) {
                        printf("  [Unmapped F16: %s layer=%d proj=%d]\n", tensors[i].name, layer_idx, proj_type);
                        unmapped_tensors++;
                    }
                    dequantized_tensor_free(dt);
                }
            }
        } else if (tensors[i].type == GGML_TYPE_F32) {
            /* Handle float32 tensors (norms, biases) */
            if (proj_type == PROJ_NORM && layer_idx >= 0 && layer_idx < config.num_layers) {
                if (strstr(tensors[i].name, "attn_norm") || strstr(tensors[i].name, "input_layernorm") || strstr(tensors[i].name, "ln1")) {
                    model->input_layernorm[layer_idx] = aligned_malloc(num_elems_1d * sizeof(float), 64);
                    memcpy(model->input_layernorm[layer_idx], tensor_data, num_elems_1d * sizeof(float));
                    loaded_tensors++;
                } else if (strstr(tensors[i].name, "ffn_norm") || strstr(tensors[i].name, "post_attention_layernorm") || strstr(tensors[i].name, "ln2")) {
                    model->post_attn_layernorm[layer_idx] = aligned_malloc(num_elems_1d * sizeof(float), 64);
                    memcpy(model->post_attn_layernorm[layer_idx], tensor_data, num_elems_1d * sizeof(float));
                    loaded_tensors++;
                }
            } else if (proj_type == PROJ_NORM && layer_idx < 0) {
                /* Final output norm */
                model->output_norm = aligned_malloc(num_elems_1d * sizeof(float), 64);
                memcpy(model->output_norm, tensor_data, num_elems_1d * sizeof(float));
                loaded_tensors++;
            }
        }
        
        free(tensor_data);
        free(tensors[i].name);
    }
    
    free(tensors);
    fclose(f);
    
    printf("\nLoaded: %d tensors (%zu MB)\n", loaded_tensors, total_bytes / (1024*1024));
    printf("Skipped: %d tensors\n", skipped_tensors);

    /* Validate loaded weights */
    int loaded_projs = 0;
    int missing_projs = 0;
    for (int l = 0; l < config.num_layers; l++) {
        if (model->q_proj[l]) loaded_projs++; else missing_projs++;
        if (model->k_proj[l]) loaded_projs++; else missing_projs++;
        if (model->v_proj[l]) loaded_projs++; else missing_projs++;
        if (model->o_proj[l]) loaded_projs++; else missing_projs++;
        if (model->gate_proj[l]) loaded_projs++; else missing_projs++;
        if (model->up_proj[l]) loaded_projs++; else missing_projs++;
        if (model->down_proj[l]) loaded_projs++; else missing_projs++;
    }
    printf("Weight projections: %d loaded, %d missing (of %d total)\n",
           loaded_projs, missing_projs, config.num_layers * 7);
    if (model->output_norm) printf("Output norm: loaded\n");
    else printf("Output norm: MISSING\n");

    /* Handle tied embeddings: if LM head has no weights, use token embeddings */
    if (model->lm_head && !model->lm_head->f32_weights && !model->lm_head->weights) {
        printf("  LM head empty — using tied embeddings\n");
        model->lm_head->f32_weights = model->token_embeddings; /* shared pointer */
        model->lm_head->original_bits = -1; /* marker: don't free f32_weights */
        model->lm_head->rows = config.vocab_size;
        model->lm_head->cols = config.hidden_size;
    }
    if (skipped_tensors > 100) {
        printf("WARNING: Most tensors were skipped!\n");
    }
    printf("Model ready for inference!\n\n");
    
    return model;
}

/* Create a mock model with random weights */
transformer_model_t* model_create_mock(int hidden_size, int intermediate_size, 
                                        int num_layers, int vocab_size) {
    printf("Creating mock model:\n");
    printf("  Hidden: %d, Intermediate: %d, Layers: %d, Vocab: %d\n",
           hidden_size, intermediate_size, num_layers, vocab_size);
    
    transformer_model_t* model = calloc(1, sizeof(transformer_model_t));
    model->config.vocab_size = vocab_size;
    model->config.hidden_size = hidden_size;
    model->config.intermediate_size = intermediate_size;
    model->config.num_layers = num_layers;
    model->config.num_heads = hidden_size / 128;
    model->config.num_kv_heads = model->config.num_heads;
    model->config.head_dim = 128;
    model->config.max_seq_len = 4096;
    strcpy(model->config.model_name, "Mock-Model");
    
    int heads = model->config.num_heads;
    int head_dim = model->config.head_dim;
    
    /* Allocate arrays */
    model->q_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    model->k_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    model->v_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    model->o_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    model->gate_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    model->up_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    model->down_proj = calloc(num_layers, sizeof(dequantized_tensor_t*));
    
    /* Initialize weights */
    for (int l = 0; l < num_layers; l++) {
        for (int proj = 0; proj < 3; proj++) {
            dequantized_tensor_t* dt = malloc(sizeof(dequantized_tensor_t));
            dt->rows = hidden_size;
            dt->cols = hidden_size;
            dt->weights = aligned_malloc(hidden_size * hidden_size, 32);
            dt->scales = aligned_malloc(hidden_size * sizeof(float), 32);
            
            for (int r = 0; r < hidden_size; r++) {
                float max_abs = 0.0f;
                for (int c = 0; c < hidden_size; c++) {
                    float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                    dt->weights[r * hidden_size + c] = (int8_t)(v * 100);
                    if (fabsf(v) > max_abs) max_abs = fabsf(v);
                }
                dt->scales[r] = max_abs > 0 ? max_abs / 127.0f : 1.0f;
            }
            
            if (proj == 0) model->q_proj[l] = dt;
            else if (proj == 1) model->k_proj[l] = dt;
            else model->v_proj[l] = dt;
        }
        
        model->o_proj[l] = malloc(sizeof(dequantized_tensor_t));
        model->o_proj[l]->rows = hidden_size;
        model->o_proj[l]->cols = hidden_size;
        model->o_proj[l]->weights = aligned_malloc(hidden_size * hidden_size, 32);
        model->o_proj[l]->scales = aligned_malloc(hidden_size * sizeof(float), 32);
        for (int r = 0; r < hidden_size; r++) {
            float max_abs = 0.0f;
            for (int c = 0; c < hidden_size; c++) {
                float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                model->o_proj[l]->weights[r * hidden_size + c] = (int8_t)(v * 100);
                if (fabsf(v) > max_abs) max_abs = fabsf(v);
            }
            model->o_proj[l]->scales[r] = max_abs > 0 ? max_abs / 127.0f : 1.0f;
        }
        
        for (int proj = 0; proj < 2; proj++) {
            dequantized_tensor_t* dt = malloc(sizeof(dequantized_tensor_t));
            dt->rows = intermediate_size;
            dt->cols = hidden_size;
            dt->weights = aligned_malloc(intermediate_size * hidden_size, 32);
            dt->scales = aligned_malloc(intermediate_size * sizeof(float), 32);
            
            for (int r = 0; r < intermediate_size; r++) {
                float max_abs = 0.0f;
                for (int c = 0; c < hidden_size; c++) {
                    float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                    dt->weights[r * hidden_size + c] = (int8_t)(v * 100);
                    if (fabsf(v) > max_abs) max_abs = fabsf(v);
                }
                dt->scales[r] = max_abs > 0 ? max_abs / 127.0f : 1.0f;
            }
            
            if (proj == 0) model->gate_proj[l] = dt;
            else model->up_proj[l] = dt;
        }
        
        model->down_proj[l] = malloc(sizeof(dequantized_tensor_t));
        model->down_proj[l]->rows = hidden_size;
        model->down_proj[l]->cols = intermediate_size;
        model->down_proj[l]->weights = aligned_malloc(hidden_size * intermediate_size, 32);
        model->down_proj[l]->scales = aligned_malloc(hidden_size * sizeof(float), 32);
        for (int r = 0; r < hidden_size; r++) {
            float max_abs = 0.0f;
            for (int c = 0; c < intermediate_size; c++) {
                float v = ((float)rand() / RAND_MAX - 0.5f) * 0.1f;
                model->down_proj[l]->weights[r * intermediate_size + c] = (int8_t)(v * 100);
                if (fabsf(v) > max_abs) max_abs = fabsf(v);
            }
            model->down_proj[l]->scales[r] = max_abs > 0 ? max_abs / 127.0f : 1.0f;
        }
    }
    
    model->lm_head = malloc(sizeof(dequantized_tensor_t));
    model->lm_head->rows = vocab_size;
    model->lm_head->cols = hidden_size;
    model->lm_head->weights = aligned_malloc(vocab_size * hidden_size, 32);
    model->lm_head->scales = aligned_malloc(vocab_size * sizeof(float), 32);
    for (int r = 0; r < vocab_size; r++) {
        float max_abs = 0.0f;
        for (int c = 0; c < hidden_size; c++) {
            float v = ((float)rand() / RAND_MAX - 0.5f) * 0.05f;
            model->lm_head->weights[r * hidden_size + c] = (int8_t)(v * 100);
            if (fabsf(v) > max_abs) max_abs = fabsf(v);
        }
        model->lm_head->scales[r] = max_abs > 0 ? max_abs / 127.0f : 1.0f;
    }
    
    model->token_embeddings = aligned_malloc(vocab_size * hidden_size * sizeof(float), 32);
    for (int i = 0; i < vocab_size * hidden_size; i++) {
        model->token_embeddings[i] = ((float)rand() / RAND_MAX - 0.5f) * 0.02f;
    }
    
    size_t kv_cache_size = (size_t)num_layers * 4096 * heads * head_dim * sizeof(float);
    model->k_cache = aligned_malloc(kv_cache_size, 32);
    model->v_cache = aligned_malloc(kv_cache_size, 32);
    memset(model->k_cache, 0, kv_cache_size);
    memset(model->v_cache, 0, kv_cache_size);
    
    printf("Mock model created with %d layers\n", num_layers);
    return model;
}

void model_free(transformer_model_t* model) {
    if (!model) return;
    
    for (int l = 0; l < model->config.num_layers; l++) {
        if (model->q_proj[l]) dequantized_tensor_free(model->q_proj[l]);
        if (model->k_proj[l]) dequantized_tensor_free(model->k_proj[l]);
        if (model->v_proj[l]) dequantized_tensor_free(model->v_proj[l]);
        if (model->o_proj[l]) dequantized_tensor_free(model->o_proj[l]);
        if (model->gate_proj[l]) dequantized_tensor_free(model->gate_proj[l]);
        if (model->up_proj[l]) dequantized_tensor_free(model->up_proj[l]);
        if (model->down_proj[l]) dequantized_tensor_free(model->down_proj[l]);
        if (model->input_layernorm[l]) aligned_free(model->input_layernorm[l]);
        if (model->post_attn_layernorm[l]) aligned_free(model->post_attn_layernorm[l]);
    }
    
    free(model->q_proj);
    free(model->k_proj);
    free(model->v_proj);
    free(model->o_proj);
    free(model->gate_proj);
    free(model->up_proj);
    free(model->down_proj);
    free(model->input_layernorm);
    free(model->post_attn_layernorm);
    
    if (model->lm_head) dequantized_tensor_free(model->lm_head);
    if (model->token_embeddings) aligned_free(model->token_embeddings);
    if (model->output_norm) aligned_free(model->output_norm);
    if (model->k_cache) aligned_free(model->k_cache);
    if (model->v_cache) aligned_free(model->v_cache);

    /* Free tokenizer vocab */
    if (model->vocab_tokens) {
        for (int i = 0; i < model->config.vocab_size; i++)
            if (model->vocab_tokens[i]) free(model->vocab_tokens[i]);
        free(model->vocab_tokens);
    }

    free(model);
}

void model_print_info(const transformer_model_t* model) {
    if (!model) {
        printf("Model is NULL\n");
        return;
    }
    
    printf("\n=== Model Info ===\n");
    printf("Name: %s\n", model->config.model_name);
    printf("Vocab size: %d\n", model->config.vocab_size);
    printf("Hidden size: %d\n", model->config.hidden_size);
    printf("Intermediate: %d\n", model->config.intermediate_size);
    printf("Layers: %d\n", model->config.num_layers);
    printf("Heads: %d (KV: %d)\n", model->config.num_heads, model->config.num_kv_heads);
    printf("Head dim: %d\n", model->config.head_dim);
    printf("Max seq len: %d\n", model->config.max_seq_len);
    
    /* Count loaded weights */
    int loaded = 0;
    for (int l = 0; l < model->config.num_layers; l++) {
        if (model->q_proj[l]) loaded++;
        if (model->k_proj[l]) loaded++;
        if (model->v_proj[l]) loaded++;
        if (model->o_proj[l]) loaded++;
        if (model->gate_proj[l]) loaded++;
        if (model->up_proj[l]) loaded++;
        if (model->down_proj[l]) loaded++;
    }
    printf("Loaded weights: %d/%d projections\n", loaded, model->config.num_layers * 7);
    printf("==================\n\n");
}
