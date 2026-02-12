#include <stdio.h>
#include <stdint.h>

typedef struct {
    uint8_t scales[12];
    uint8_t qs[128];
} block_q4_K;

typedef struct {
    uint8_t scales[16];
    uint8_t qs[192];
} block_q6_K;

int main() {
    printf("sizeof(block_q4_K) = %zu\n", sizeof(block_q4_K));
    printf("sizeof(block_q6_K) = %zu\n", sizeof(block_q6_K));
    printf("Expected Q4_K: 144 bytes (12 + 128 + 4 padding)\n");
    printf("Expected Q6_K: 208 bytes (16 + 192)\n");
    return 0;
}
