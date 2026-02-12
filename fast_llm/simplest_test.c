/*
 * Simplest test - just load GGUF header
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>

#define GGUF_MAGIC 0x46554747

int main(int argc, char** argv) {
    const char* path = (argc > 1) ? argv[1] : "..\\models\\Phi-3-mini-4k-instruct-q4.gguf";
    
    printf("Opening: %s\n", path);
    
    FILE* f = fopen(path, "rb");
    if (!f) {
        printf("Failed to open file\n");
        return 1;
    }
    
    uint32_t magic;
    fread(&magic, 4, 1, f);
    printf("Magic: 0x%08X (expected 0x%08X) %s\n", 
           magic, GGUF_MAGIC, magic == GGUF_MAGIC ? "OK" : "FAIL");
    
    if (magic != GGUF_MAGIC) {
        fclose(f);
        return 1;
    }
    
    uint32_t version;
    fread(&version, 4, 1, f);
    printf("Version: %d\n", version);
    
    uint64_t num_tensors, num_metadata;
    fread(&num_tensors, 8, 1, f);
    fread(&num_metadata, 8, 1, f);
    
    printf("Tensors: %llu\n", (unsigned long long)num_tensors);
    printf("Metadata: %llu\n", (unsigned long long)num_metadata);
    
    fclose(f);
    printf("\nSuccess!\n");
    return 0;
}
