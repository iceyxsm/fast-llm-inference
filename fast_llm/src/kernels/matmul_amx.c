/*
 * Intel AMX (Advanced Matrix Extensions) Kernels
 * Only available on 4th+ Gen Intel Xeon (Sapphire Rapids+)
 * 
 * For now, these are stubs that fallback to AVX-512
 * Full AMX implementation requires tile register configuration
 */

#include "matmul.h"

/* AMX detection at runtime would go here */
/* For now, fallback to AVX-512 */

void q2_matmul_amx(const float* A, const quantized_tensor_t* B, float* C,
                   int M, int N, int K) {
    /* Fallback to AVX-512 for now */
    q2_matmul_avx512(A, B, C, M, N, K);
}

void q8_matmul_amx(const float* A, const quantized_tensor_t* B, float* C,
                   int M, int N, int K) {
    /* Fallback to AVX-512 for now */
    q8_matmul_avx512(A, B, C, M, N, K);
}
