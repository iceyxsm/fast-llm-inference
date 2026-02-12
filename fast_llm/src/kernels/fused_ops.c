/*
 * Fused Operations Kernels
 * Combines matmul + activation + residual into single kernel
 * Reduces memory bandwidth by 2-3x
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

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

/* 
 * Fused Matmul + SiLU + Multiply + Down + Residual
 * This is the core of the FFN layer
 * 
 * Input: [hidden]
 * Gate/Up weights: [2*intermediate, hidden] (fused)
 * Down weights: [hidden, intermediate]
 * Output: [hidden] (with residual)
 * 
 * Memory bandwidth reduction:
 * - Naive: load input (H) + store gate (I) + store up (I) + load gate+up (2I) + store down (H) + add residual (H)
 * - Fused: load input (H) + store temp (I) + load temp (I) + store output (H)
 * - Savings: ~2x reduction in memory traffic
 */

#ifdef __AVX2__

/* 
 * AVX2 Fused FFN kernel
 * Processes 32 output features at a time
 */
void fused_ffn_avx2(const float* input,
                    const dequantized_tensor_t* gate_up,
                    const dequantized_tensor_t* down,
                    float* output,
                    int hidden, int intermediate) {
    
    /* Temporary buffer for SwiGLU output */
    float* swiglu_out = aligned_malloc(intermediate * sizeof(float), 32);
    
    /* ========== Step 1: Gate + Up projection (fused) ========== */
    /* gate_up has shape [2*intermediate, hidden] */
    /* First half is gate, second half is up */
    
    const int tile = 64;  /* Process 64 intermediate features at a time */
    
    #pragma omp parallel for schedule(static)
    for (int i_tile = 0; i_tile < intermediate; i_tile += tile) {
        int i_end = (i_tile + tile < intermediate) ? i_tile + tile : intermediate;
        
        for (int i = i_tile; i < i_end; i++) {
            /* Compute gate[i] and up[i] simultaneously */
            const int8_t* gate_row = gate_up->weights + i * gate_up->cols;
            const int8_t* up_row = gate_up->weights + (i + intermediate) * gate_up->cols;
            float gate_scale = gate_up->scales[i];
            float up_scale = gate_up->scales[i + intermediate];
            
            __m256 gate_acc = _mm256_setzero_ps();
            __m256 up_acc = _mm256_setzero_ps();
            
            int j = 0;
            /* Process 32 elements at a time with SIMD */
            for (; j <= hidden - 32; j += 32) {
                /* Load 32 floats from input */
                __m256 in0 = _mm256_loadu_ps(input + j);
                __m256 in1 = _mm256_loadu_ps(input + j + 8);
                __m256 in2 = _mm256_loadu_ps(input + j + 16);
                __m256 in3 = _mm256_loadu_ps(input + j + 24);
                
                /* Load and dequantize 32 int8s from gate */
                __m256i gate_i8 = _mm256_loadu_si256((__m256i*)(gate_row + j));
                __m256i gate_i16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(gate_i8));
                __m256i gate_i16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(gate_i8, 1));
                __m256 gate_f32_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(gate_i16_0)));
                __m256 gate_f32_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(gate_i16_0, 1)));
                __m256 gate_f32_2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(gate_i16_1)));
                __m256 gate_f32_3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(gate_i16_1, 1)));
                
                /* Multiply by scale */
                __m256 gate_s = _mm256_set1_ps(gate_scale * 0.0625f);
                gate_f32_0 = _mm256_mul_ps(gate_f32_0, gate_s);
                gate_f32_1 = _mm256_mul_ps(gate_f32_1, gate_s);
                gate_f32_2 = _mm256_mul_ps(gate_f32_2, gate_s);
                gate_f32_3 = _mm256_mul_ps(gate_f32_3, gate_s);
                
                /* FMA: gate += input * weight */
                gate_acc = _mm256_fmadd_ps(in0, gate_f32_0, gate_acc);
                gate_acc = _mm256_fmadd_ps(in1, gate_f32_1, gate_acc);
                gate_acc = _mm256_fmadd_ps(in2, gate_f32_2, gate_acc);
                gate_acc = _mm256_fmadd_ps(in3, gate_f32_3, gate_acc);
                
                /* Same for up */
                __m256i up_i8 = _mm256_loadu_si256((__m256i*)(up_row + j));
                __m256i up_i16_0 = _mm256_cvtepi8_epi16(_mm256_castsi256_si128(up_i8));
                __m256i up_i16_1 = _mm256_cvtepi8_epi16(_mm256_extracti128_si256(up_i8, 1));
                __m256 up_f32_0 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(up_i16_0)));
                __m256 up_f32_1 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(up_i16_0, 1)));
                __m256 up_f32_2 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_castsi256_si128(up_i16_1)));
                __m256 up_f32_3 = _mm256_cvtepi32_ps(_mm256_cvtepi16_epi32(_mm256_extracti128_si256(up_i16_1, 1)));
                
                __m256 up_s = _mm256_set1_ps(up_scale * 0.0625f);
                up_f32_0 = _mm256_mul_ps(up_f32_0, up_s);
                up_f32_1 = _mm256_mul_ps(up_f32_1, up_s);
                up_f32_2 = _mm256_mul_ps(up_f32_2, up_s);
                up_f32_3 = _mm256_mul_ps(up_f32_3, up_s);
                
                up_acc = _mm256_fmadd_ps(in0, up_f32_0, up_acc);
                up_acc = _mm256_fmadd_ps(in1, up_f32_1, up_acc);
                up_acc = _mm256_fmadd_ps(in2, up_f32_2, up_acc);
                up_acc = _mm256_fmadd_ps(in3, up_f32_3, up_acc);
            }
            
            /* Horizontal sum of gate accumulators */
            __m128 gate_low = _mm256_castps256_ps128(gate_acc);
            __m128 gate_high = _mm256_extractf128_ps(gate_acc, 1);
            gate_low = _mm_add_ps(gate_low, gate_high);
            gate_low = _mm_hadd_ps(gate_low, gate_low);
            gate_low = _mm_hadd_ps(gate_low, gate_low);
            float gate_val = _mm_cvtss_f32(gate_low);
            
            /* Horizontal sum of up accumulators */
            __m128 up_low = _mm256_castps256_ps128(up_acc);
            __m128 up_high = _mm256_extractf128_ps(up_acc, 1);
            up_low = _mm_add_ps(up_low, up_high);
            up_low = _mm_hadd_ps(up_low, up_low);
            up_low = _mm_hadd_ps(up_low, up_low);
            float up_val = _mm_cvtss_f32(up_low);
            
            /* Scalar remainder for gate and up */
            for (; j < hidden; j++) {
                gate_val += input[j] * gate_row[j] * gate_scale * 0.0625f;
                up_val += input[j] * up_row[j] * up_scale * 0.0625f;
            }
            
            /* SiLU + multiply: gate * sigmoid(gate) * up */
            float sigmoid = 1.0f / (1.0f + expf(-gate_val));
            swiglu_out[i] = gate_val * sigmoid * up_val;
        }
    }
    
    /* ========== Step 2: Down projection + Residual ========== */
    #pragma omp parallel for schedule(static)
    for (int h_tile = 0; h_tile < hidden; h_tile += 32) {
        int h_end = (h_tile + 32 < hidden) ? h_tile + 32 : hidden;
        
        for (int h = h_tile; h < h_end; h++) {
            const int8_t* down_row = down->weights + h * down->cols;
            float down_scale = down->scales[h];
            
            __m256 acc = _mm256_setzero_ps();
            
            int j = 0;
            for (; j <= intermediate - 8; j += 8) {
                __m256 swiglu = _mm256_loadu_ps(swiglu_out + j);
                
                /* Load and dequantize 8 int8 weights */
                __m128i w_i8 = _mm_loadl_epi64((__m128i*)(down_row + j));
                __m256i w_i32 = _mm256_cvtepi8_epi32(w_i8);
                __m256 w_f32 = _mm256_cvtepi32_ps(w_i32);
                
                __m256 w_scale = _mm256_set1_ps(down_scale * 0.0625f);
                w_f32 = _mm256_mul_ps(w_f32, w_scale);
                
                acc = _mm256_fmadd_ps(swiglu, w_f32, acc);
            }
            
            /* Horizontal sum */
            __m128 sum_low = _mm256_castps256_ps128(acc);
            __m128 sum_high = _mm256_extractf128_ps(acc, 1);
            sum_low = _mm_add_ps(sum_low, sum_high);
            sum_low = _mm_hadd_ps(sum_low, sum_low);
            sum_low = _mm_hadd_ps(sum_low, sum_low);
            float sum = _mm_cvtss_f32(sum_low);
            
            /* Scalar remainder */
            for (; j < intermediate; j++) {
                sum += swiglu_out[j] * down_row[j] * down_scale * 0.0625f;
            }
            
            /* Residual connection */
            output[h] = input[h] + sum;
        }
    }
    
    aligned_free(swiglu_out);
}

#else /* No AVX2 */

void fused_ffn_avx2(const float* input,
                    const dequantized_tensor_t* gate_up,
                    const dequantized_tensor_t* down,
                    float* output,
                    int hidden, int intermediate) {
    /* Fallback - should not be called without AVX2 */
    (void)input; (void)gate_up; (void)down; (void)output;
    (void)hidden; (void)intermediate;
}

#endif /* __AVX2__ */

/* 
 * Simple fused FFN (scalar fallback)
 * Still benefits from reduced memory traffic
 */
void fused_ffn_scalar(const float* input,
                      const dequantized_tensor_t* gate_up,
                      const dequantized_tensor_t* down,
                      float* output,
                      int hidden, int intermediate) {
    
    float* swiglu_out = aligned_malloc(intermediate * sizeof(float), 32);
    
    /* Gate + Up projection */
    for (int i = 0; i < intermediate; i++) {
        const int8_t* gate_row = gate_up->weights + i * gate_up->cols;
        const int8_t* up_row = gate_up->weights + (i + intermediate) * gate_up->cols;
        float gate_scale = gate_up->scales[i] * 0.0625f;
        float up_scale = gate_up->scales[i + intermediate] * 0.0625f;
        
        float gate_val = 0.0f;
        float up_val = 0.0f;
        
        for (int j = 0; j < hidden; j++) {
            gate_val += input[j] * gate_row[j] * gate_scale;
            up_val += input[j] * up_row[j] * up_scale;
        }
        
        float sigmoid = 1.0f / (1.0f + expf(-gate_val));
        swiglu_out[i] = gate_val * sigmoid * up_val;
    }
    
    /* Down projection + Residual */
    for (int h = 0; h < hidden; h++) {
        const int8_t* down_row = down->weights + h * down->cols;
        float down_scale = down->scales[h] * 0.0625f;
        
        float sum = 0.0f;
        for (int j = 0; j < intermediate; j++) {
            sum += swiglu_out[j] * down_row[j] * down_scale;
        }
        
        output[h] = input[h] + sum;
    }
    
    aligned_free(swiglu_out);
}

/* Main entry point */
void fused_ffn(const float* input,
               const dequantized_tensor_t* gate_up,
               const dequantized_tensor_t* down,
               float* output,
               int hidden, int intermediate) {
    
#ifdef __AVX2__
    fused_ffn_avx2(input, gate_up, down, output, hidden, intermediate);
#else
    fused_ffn_scalar(input, gate_up, down, output, hidden, intermediate);
#endif
}
