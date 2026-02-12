/*
 * Fast Matmul - Simplified Efficient Implementation
 * 
 * Strategy:
 * - 16-way unroll with separate accumulators for ILP
 * - Process 32 elements at a time with _mm256_maddubs_epi16
 * - Minimize data movement
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

#ifdef __AVX2__
#include <immintrin.h>
#endif

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#include "dequantized_tensor.h"

#ifdef __AVX2__

/*
 * Core dot product using _mm256_maddubs_epi16
 * 
 * This instruction multiplies uint8 x int8 -> int16
 * We quantize the float input to uint8 [0, 255]
 * The int8 weights are already signed [-128, 127]
 * 
 * Result is 8 int32 sums that we need to reduce
 */
static inline float dot_product_u8_i8_fast(const uint8_t* A_u8, 
                                            const int8_t* B_i8,
                                            int n,
                                            float scale_a,  /* Scale from A quantization */
                                            float scale_b) { /* Scale from B tensor */
    
    __m256i sum_acc = _mm256_setzero_si256();
    
    int i = 0;
    /* Process 32 bytes at a time (optimal for _mm256_maddubs_epi16) */
    for (; i <= n - 32; i += 32) {
        /* Load 32 uint8 from A */
        __m256i a_u8 = _mm256_loadu_si256((__m256i*)(A_u8 + i));
        
        /* Load 32 int8 from B */
        __m256i b_i8 = _mm256_loadu_si256((__m256i*)(B_i8 + i));
        
        /* Multiply: a (unsigned) * b (signed) -> 16-bit results */
        /* Each byte pair produces one int16 */
        __m256i prod_i16 = _mm256_maddubs_epi16(a_u8, b_i8);
        
        /* Sum adjacent int16 pairs to int32 */
        /* Using [1, 1] multiplier to just sum */
        prod_i16 = _mm256_madd_epi16(prod_i16, _mm256_set1_epi16(1));
        
        /* Accumulate */
        sum_acc = _mm256_add_epi32(sum_acc, prod_i16);
    }
    
    /* Horizontal sum of 8 int32 values */
    __m128i sum_low = _mm256_castsi256_si128(sum_acc);
    __m128i sum_high = _mm256_extracti128_si256(sum_acc, 1);
    sum_low = _mm_add_epi32(sum_low, sum_high);
    sum_low = _mm_hadd_epi32(sum_low, sum_low);
    sum_low = _mm_hadd_epi32(sum_low, sum_low);
    int32_t sum = _mm_cvtsi128_si32(sum_low);
    
    /* Scalar remainder */
    for (; i < n; i++) {
        /* Treat uint8 as signed for consistency */
        sum += (int32_t)((int8_t)A_u8[i]) * (int32_t)B_i8[i];
    }
    
    /* Apply scales */
    return (float)sum * scale_a * scale_b;
}

/* Quantize float to uint8 with bias */
static inline void quantize_u8(const float* src, uint8_t* dst, int n, float scale) {
    __m256 scale_vec = _mm256_set1_ps(scale);
    __m256 bias_vec = _mm256_set1_ps(128.0f);
    
    int i = 0;
    for (; i <= n - 16; i += 16) {
        /* Load 16 floats */
        __m256 f0 = _mm256_loadu_ps(src + i);
        __m256 f1 = _mm256_loadu_ps(src + i + 8);
        
        /* Scale and add bias */
        f0 = _mm256_fmadd_ps(f0, scale_vec, bias_vec);
        f1 = _mm256_fmadd_ps(f1, scale_vec, bias_vec);
        
        /* Convert to int32 */
        __m256i i0 = _mm256_cvtps_epi32(f0);
        __m256i i1 = _mm256_cvtps_epi32(f1);
        
        /* Pack int32 -> int16 -> int8 with saturation */
        __m256i i16 = _mm256_packs_epi32(i0, i1);
        __m256i i8 = _mm256_packus_epi16(i16, i16);  /* Unsigned saturation */
        
        /* Store lower 128 bits (16 bytes) */
        _mm_storeu_si128((__m128i*)(dst + i), _mm256_castsi256_si128(i8));
    }
    
    /* Scalar remainder */
    for (; i < n; i++) {
        int val = (int)(src[i] * scale + 128.0f);
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        dst[i] = (uint8_t)val;
    }
}

/* Fast matmul - single token, multiple outputs */
void matmul_dequantized_fast(const float* A, const dequantized_tensor_t* B,
                             float* C, int M, int N, int K) {
    (void)M;  /* Must be 1 */
    
    /* Thread-local buffer for quantized A */
    /* In production, allocate once per thread */
    uint8_t* A_u8 = aligned_malloc(K + 32, 32);  /* Extra for alignment */
    
    /* Quantize A once */
    float a_scale = 127.0f;  /* Assuming input range [-1, 1] */
    quantize_u8(A, A_u8, K, a_scale);
    
    /* Scale factor: 1 / (127 * 16) for Q4 range normalization */
    float combined_scale = 1.0f / (a_scale * 16.0f);
    
    /* Process each output row */
    #pragma omp parallel for schedule(static) if(N > 256)
    for (int n = 0; n < N; n++) {
        const int8_t* B_row = B->weights + n * B->cols;
        float b_scale = B->scales[n];
        
        C[n] = dot_product_u8_i8_fast(A_u8, B_row, K, combined_scale, b_scale);
    }
    
    aligned_free(A_u8);
}

#else /* No AVX2 */

void matmul_dequantized_fast(const float* A, const dequantized_tensor_t* B,
                             float* C, int M, int N, int K) {
    /* Fallback to regular implementation */
    matmul_dequantized(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
