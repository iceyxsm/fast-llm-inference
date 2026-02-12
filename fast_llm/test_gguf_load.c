/*
 * Test GGUF loading step by step
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define GGUF_MAGIC 0x46554747

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
            int arr_type = read_i32(f);
            uint64_t arr_len = read_u64(f);
            for (uint64_t i = 0; i < arr_len; i++) skip_value(f, arr_type);
            break;
        }
        case 10: case 11: case 12: read_u64(f); break;
    }
}

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "..\\models\\Phi-3-mini-4k-instruct-q4.gguf";
    
    printf("Testing GGUF load: %s\n\n", path);
    
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("ERROR: Cannot open file\n");
        return 1;
    }
    
    /* Read header */
    uint32_t magic = read_u32(f);
    printf("Magic: 0x%08X (expected 0x%08X) %s\n", magic, GGUF_MAGIC, 
           magic == GGUF_MAGIC ? "OK" : "FAIL");
    
    if (magic != GGUF_MAGIC) {
        fclose(f);
        return 1;
    }
    
    uint32_t version = read_u32(f);
    uint64_t num_tensors = read_u64(f);
    uint64_t num_metadata = read_u64(f);
    
    printf("Version: %d\n", version);
    printf("Tensors: %llu\n", (unsigned long long)num_tensors);
    printf("Metadata: %llu\n\n", (unsigned long long)num_metadata);
    
    /* Parse metadata */
    printf("=== Metadata ===\n");
    int hidden_size = 3072;
    int intermediate_size = 8192;
    int num_layers = 32;
    
    for (uint64_t i = 0; i < num_metadata && i < 30; i++) {  /* Limit to first 30 */
        char* key = read_string(f);
        int type = read_i32(f);
        
        if (strstr(key, "embedding_length") || strstr(key, "hidden_size")) {
            hidden_size = read_i32(f);
            printf("  %s = %d\n", key, hidden_size);
        } else if (strstr(key, "feed_forward_length") || strstr(key, "intermediate_size")) {
            intermediate_size = read_i32(f);
            printf("  %s = %d\n", key, intermediate_size);
        } else if (strstr(key, "block_count") || strstr(key, "num_hidden_layers")) {
            num_layers = read_i32(f);
            printf("  %s = %d\n", key, num_layers);
        } else if (strstr(key, "attention.head_count")) {
            int heads = read_i32(f);
            printf("  %s = %d\n", key, heads);
        } else {
            skip_value(f, type);
        }
        
        free(key);
    }
    
    printf("\nArchitecture: %d layers, %d hidden, %d intermediate\n",
           num_layers, hidden_size, intermediate_size);
    
    /* Parse tensor info */
    printf("\n=== Tensor Info (first 10) ===\n");
    size_t max_offset = 0;
    
    for (uint64_t i = 0; i < num_tensors && i < 10; i++) {
        char* name = read_string(f);
        int n_dims = read_i32(f);
        
        uint64_t dims[4] = {0};
        uint64_t num_elements = 1;
        for (int d = 0; d < n_dims; d++) {
            dims[d] = read_u64(f);
            num_elements *= dims[d];
        }
        
        int type = read_i32(f);
        uint64_t offset = read_u64(f);
        
        if (offset > max_offset) max_offset = offset;
        
        printf("  %s: dims=[", name);
        for (int d = 0; d < n_dims; d++) {
            printf("%llu%s", (unsigned long long)dims[d], d < n_dims - 1 ? "," : "");
        }
        printf("], type=%d, offset=%llu\n", type, (unsigned long long)offset);
        
        free(name);
    }
    
    /* Skip remaining tensors */
    for (uint64_t i = 10; i < num_tensors; i++) {
        char* name = read_string(f);
        int n_dims = read_i32(f);
        for (int d = 0; d < n_dims; d++) read_u64(f);
        read_i32(f);  /* type */
        uint64_t offset = read_u64(f);
        if (offset > max_offset) max_offset = offset;
        free(name);
    }
    
    /* Calculate data offset */
    size_t data_offset = (max_offset + 31) & ~31;
    printf("\nData offset: %zu\n", data_offset);
    
    fclose(f);
    printf("\nGGUF parsing successful!\n");
    
    return 0;
}
