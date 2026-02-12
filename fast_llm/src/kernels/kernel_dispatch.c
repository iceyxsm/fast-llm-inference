/*
 * Kernel Dispatcher
 * Selects optimal kernel based on CPU features
 */

#include "matmul.h"
#include <stdio.h>

/* Declare AVX2 kernels */
extern void q2_matmul_avx2(const float*, const quantized_tensor_t*, float*, int, int, int);
extern void q4_matmul_avx2(const float*, const quantized_tensor_t*, float*, int, int, int);
extern void q8_matmul_avx2(const float*, const quantized_tensor_t*, float*, int, int, int);
extern void q4_matmul_avx2_optimized(const float*, const quantized_tensor_t*, float*, int, int, int);
extern void q4_matmul_blocked(const float*, const quantized_tensor_t*, float*, int, int, int);
extern void q4_matmul_single_token(const float*, const quantized_tensor_t*, float*, int, int, int);

matmul_fn_t select_q2_kernel(const cpu_features_t* features) {
    if (features->has_amx) {
        printf("Using Q2 AMX kernel\n");
        return q2_matmul_amx;
    }
    if (features->has_avx512f) {
        printf("Using Q2 AVX-512 kernel\n");
        return q2_matmul_avx512;
    }
    if (features->has_avx2) {
        printf("Using Q2 AVX2 kernel\n");
        return q2_matmul_avx2;
    }
    printf("Using Q2 scalar kernel (fallback)\n");
    return q2_matmul;
}

matmul_fn_t select_q4_kernel(const cpu_features_t* features) {
    if (features->has_amx) {
        printf("Using Q4 AMX kernel (fallback to AVX-512)\n");
        return q4_matmul_avx512;
    }
    if (features->has_avx512f) {
        printf("Using Q4 AVX-512 kernel\n");
        return q4_matmul_avx512;
    }
    if (features->has_avx2) {
        printf("Using Q4 AVX2 SINGLE-TOKEN kernel\n");
        return q4_matmul_single_token;
    }
    printf("Using Q4 scalar kernel (fallback)\n");
    return q4_matmul;
}

matmul_fn_t select_q8_kernel(const cpu_features_t* features) {
    if (features->has_amx) {
        printf("Using Q8 AMX kernel\n");
        return q8_matmul_amx;
    }
    if (features->has_avx512vnni) {
        printf("Using Q8 AVX-512 VNNI kernel\n");
        return q8_matmul_avx512vnni;
    }
    if (features->has_avx512f) {
        printf("Using Q8 AVX-512 kernel\n");
        return q8_matmul_avx512;
    }
    if (features->has_avx2) {
        printf("Using Q8 AVX2 kernel\n");
        return q8_matmul_avx2;
    }
    printf("Using Q8 scalar kernel (fallback)\n");
    return q8_matmul;
}
