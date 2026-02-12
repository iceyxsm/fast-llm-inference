/*
 * Super-Optimized Matmul using _mm256_maddubs_epi16
 * 
 * Key insight from llamafile: use the dot product instruction directly
 * - Quantize activations to uint8: x' = x * scale + 128
 * - Weights are int8: w
 * - _mm256_maddubs_epi16 does: sum(x' * w) for adjacent pairs
 * - Then adjust for the 128 bias
 * 
 * This gives true 8-bit dot products at full AVX2 throughput
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
 * The trick: we need to handle the bias from uint8 quantization
 * If A is quantized as uint8: A_u8 = A_f32 * scale + 128
 * Then: dot(A_u8, B_i8) = dot(A_f32 * scale + 128, B_i8)
 *                       = scale * dot(A_f32, B_i8) + 128 * sum(B_i8)
 * 
 * So: dot(A_f32, B_i8) = (dot(A_u8, B_i8) - 128 * sum(B_i8)) / scale
 * 
 * We precompute sum(B_i8) per row to save time
 */

/* Precompute row sums for bias correction */
void precompute_row_sums(const dequantized_tensor_t* B, int32_t* row_sums) {
    for (int r = 0; r < B->rows; r++) {
        const int8_t* row = B->weights + r * B->cols;
        int32_t sum = 0;
        int c = 0;
        
        /* SIMD sum */
        __m256i sum_vec = _mm256_setzero_si256();
        for (; c <= B->cols - 32; c += 32) {
            __m256i vals = _mm256_loadu_si256((__m256i*)(row + c));
            /* Sign-extend and sum */
            __m256i lo = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(vals));
            __m256i hi = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(vals, 1));
            __m256i lo32 = _mm256_cvtepi16_epi32(_mm256_castsi256_si128(lo));
            __m256i hi32 = _mm256_cvtepi16_epi32(_mm256_extracti128_si256(lo, 1));
            sum_vec = _mm256_add_epi32(sum_vec, lo32);
            sum_vec = _mm256_add_epi32(sum_vec, hi32);
        }
        
        /* Horizontal sum */
        __m128i sum_low = _mm256_castsi256_si128(sum_vec);
        __m128i sum_high = _mm256_extracti128_si256(sum_vec, 1);
        sum_low = _mm_add_epi32(sum_low, sum_high);
        sum_low = _mm_hadd_epi32(sum_low, sum_low);
        sum_low = _mm_hadd_epi32(sum_low, sum_low);
        sum = _mm_cvtsi128_si32(sum_low);
        
        /* Scalar remainder */
        for (; c < B->cols; c++) {
            sum += row[c];
        }
        
        row_sums[r] = sum;
    }
}

/* 
 * Super fast matmul with _mm256_maddubs_epi16
 * A is float input (will be quantized to uint8 on the fly)
 * B is int8 weights
 */
void matmul_dequantized_super(const float* A, const dequantized_tensor_t* B,
                               float* C, int M, int N, int K,
                               const int32_t* row_sums) {
    (void)M;  /* Must be 1 */
    
    /* Quantize A to uint8 with scale */
    /* Scale chosen so typical input range [-1, 1] maps to [0, 255] */
    const float a_scale_f = 127.0f;
    const __m256 a_scale = _mm256_set1_ps(a_scale_f);
    const __m256 a_bias = _mm256_set1_ps(128.0f);
    
    /* Thread-local buffer for quantized A */
    uint8_t* A_u8 = aligned_malloc(K + 64, 64);
    
    /* Quantize A */
    int k = 0;
    for (; k <= K - 32; k += 32) {
        __m256 f0 = _mm256_loadu_ps(A + k);
        __m256 f1 = _mm256_loadu_ps(A + k + 8);
        __m256 f2 = _mm256_loadu_ps(A + k + 16);
        __m256 f3 = _mm256_loadu_ps(A + k + 24);
        
        /* Scale and bias */
        f0 = _mm256_fmadd_ps(f0, a_scale, a_bias);
        f1 = _mm256_fmadd_ps(f1, a_scale, a_bias);
        f2 = _mm256_fmadd_ps(f2, a_scale, a_bias);
        f3 = _mm256_fmadd_ps(f3, a_scale, a_bias);
        
        /* Convert to int32 */
        __m256i i0 = _mm256_cvtps_epi32(f0);
        __m256i i1 = _mm256_cvtps_epi32(f1);
        __m256i i2 = _mm256_cvtps_epi32(f2);
        __m256i i3 = _mm256_cvtps_epi32(f3);
        
        /* Pack to int16 */
        __m256i i01 = _mm256_packs_epi32(i0, i1);
        __m256i i23 = _mm256_packs_epi32(i2, i3);
        
        /* Pack to uint8 */
        __m256i i8 = _mm256_packus_epi16(i01, i23);
        
        /* Store */
        _mm256_storeu_si256((__m256i*)(A_u8 + k), i8);
    }
    
    /* Scalar remainder */
    for (; k < K; k++) {
        int val = (int)(A[k] * a_scale_f + 128.0f);
        if (val < 0) val = 0;
        if (val > 255) val = 255;
        A_u8[k] = (uint8_t)val;
    }
    
    /* Now compute dot products using _mm256_maddubs_epi16 */
    #pragma omp parallel for schedule(static) if(N > 512)
    for (int n = 0; n < N; n++) {
        const int8_t* B_row = B->weights + n * B->cols;
        float b_scale = B->scales[n];
        
        __m256i sum_acc = _mm256_setzero_si256();
        
        /* Process 32 bytes at a time */
        int k = 0;
        for (; k <= K - 32; k += 32) {
            /* Prefetch next iteration */
            _mm_prefetch((const char*)(A_u8 + k + 64), _MM_HINT_T0);
            _mm_prefetch((const char*)(B_row + k + 64), _MM_HINT_T0);
            
            /* Load 32 uint8 from A */
            __m256i a_u8 = _mm256_loadu_si256((__m256i*)(A_u8 + k));
            
            /* Load 32 int8 from B */
            __m256i b_i8 = _mm256_loadu_si256((__m256i*)(B_row + k));
            
            /* Multiply: uint8 * int8 -> int16 (with saturation) */
            /* This is the key instruction! 32 multiplications at once */
            __m256i prod_i16 = _mm256_maddubs_epi16(a_u8, b_i8);
            
            /* Sum adjacent pairs to int32 */
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
        for (; k < K; k++) {
            /* Treat uint8 as signed for consistency with SIMD path */
            sum += (int32_t)((int8_t)A_u8[k]) * (int32_t)B_row[k];
        }
        
        /* Apply bias correction and scales */
        /* C[n] = (sum - 128 * row_sum[n]) / a_scale * b_scale / 16 */
        float result = (float)(sum - 128 * row_sums[n]) / a_scale_f * b_scale / 16.0f;
        C[n] = result;
    }
    
    aligned_free(A_u8);
}

#else /* No AVX2 */

void precompute_row_sums(const dequantized_tensor_t* B, int32_t* row_sums) {
    (void)B; (void)row_sums;
}

void matmul_dequantized_super(const float* A, const dequantized_tensor_t* B,
                               float* C, int M, int N, int K,
                               const int32_t* row_sums) {
    (void)row_sums;
    matmul_dequantized(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
