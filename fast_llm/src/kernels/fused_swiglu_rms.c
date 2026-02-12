/*
 * Fused SwiGLU + RMSNorm Kernel
 * 
 * Based on research from:
 * - Liger Kernel (LinkedIn): https://github.com/linkedin/Liger-Kernel
 * - FlashAttention-2 optimizations
 * 
 * Fusing these operations eliminates intermediate memory round-trips:
 * Standard: Input -> RMSNorm -> write to memory -> SwiGLU -> write to memory
 * Fused:   Input -> RMSNorm + SwiGLU -> write to memory (1 trip instead of 2)
 * 
 * This improves:
 * - Memory bandwidth by 2x for these operations
 * - Cache locality
 * - Arithmetic intensity
 * 
 * SwiGLU formula:
 *   SwiGLU(x) = Swish(xW) * (xV)
 *   where Swish(x) = x * sigmoid(beta * x)
 * 
 * RMSNorm formula:
 *   RMSNorm(x) = x / sqrt(mean(x^2) + eps) * gamma
 */

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
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

#define SWISH_BETA 1.0f  /* Standard Swish beta parameter */
#define RMS_EPS 1e-6f

#ifdef __AVX2__

/* Fast sigmoid approximation using AVX2 */
static inline __m256 fast_sigmoid_ps(__m256 x) {
    /* 
     * sigmoid(x) = 1 / (1 + exp(-x))
     * Approximation: sigmoid(x) ≈ 0.5 * (1 + tanh(x/2))
     * Or use fast exp approximation
     */
    
    /* Clamp x to avoid overflow in exp */
    __m256 xmin = _mm256_set1_ps(-10.0f);
    __m256 xmax = _mm256_set1_ps(10.0f);
    x = _mm256_max_ps(xmin, _mm256_min_ps(xmax, x));
    
    /* Polynomial approximation of sigmoid */
    /* sigmoid(x) ≈ 0.5 + 0.5 * x / (1 + |x|) for rough approx */
    /* Better: use fast exp */
    
    /* For now, use direct computation with fast exp approximation */
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    
    /* Fast exp2 approximation: exp2(x) = 2^x */
    /* exp(x) = exp2(x * log2(e)) */
    __m256 log2e = _mm256_set1_ps(1.44269504f);
    __m256 x_scaled = _mm256_mul_ps(neg_x, log2e);
    
    /* exp2 polynomial approximation */
    __m256 c0 = _mm256_set1_ps(1.0f);
    __m256 c1 = _mm256_set1_ps(0.693147f);
    __m256 c2 = _mm256_set1_ps(0.240227f);
    __m256 c3 = _mm256_set1_ps(0.055504f);
    
    __m256 exp2_x = c0;
    exp2_x = _mm256_fmadd_ps(c1, x_scaled, exp2_x);
    __m256 x2 = _mm256_mul_ps(x_scaled, x_scaled);
    exp2_x = _mm256_fmadd_ps(c2, x2, exp2_x);
    __m256 x3 = _mm256_mul_ps(x2, x_scaled);
    exp2_x = _mm256_fmadd_ps(c3, x3, exp2_x);
    
    /* sigmoid = 1 / (1 + exp(-x)) */
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 denom = _mm256_add_ps(one, exp2_x);
    
    /* Approximate division: 1/denom */
    return _mm256_div_ps(one, denom);
}

/* Fast sigmoid - scalar version for verification */
static inline float fast_sigmoid_f(float x) {
    if (x < -10.0f) return 0.0f;
    if (x > 10.0f) return 1.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* Swish activation: x * sigmoid(beta * x) */
static inline __m256 swish_ps(__m256 x) {
    __m256 beta = _mm256_set1_ps(SWISH_BETA);
    __m256 beta_x = _mm256_mul_ps(x, beta);
    __m256 sig = fast_sigmoid_ps(beta_x);
    return _mm256_mul_ps(x, sig);
}

/* 
 * Fused RMSNorm + SwiGLU forward pass
 * 
 * Input: x [batch, hidden]
 * Gate projection weights: W_gate [hidden, intermediate]  
 * Value projection weights: W_value [hidden, intermediate]
 * RMSNorm weights: gamma [hidden]
 * Output: out [batch, intermediate]
 * 
 * Computes: SwiGLU(RMSNorm(x)) = Swish(RMSNorm(x) @ W_gate) * (RMSNorm(x) @ W_value)
 */
void fused_rmsnorm_swiglu_forward(
    const float* input,          /* [batch, hidden] */
    const float* w_gate,         /* [hidden, intermediate] - quantized or float */
    const float* w_value,        /* [hidden, intermediate] */
    const float* gamma,          /* [hidden] RMSNorm weights */
    float* output,               /* [batch, intermediate] */
    int batch_size,
    int hidden_size,
    int intermediate_size
) {
    const __m256 eps_vec = _mm256_set1_ps(RMS_EPS);
    const __m256 one_vec = _mm256_set1_ps(1.0f);
    
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch_size; b++) {
        const float* x = input + b * hidden_size;
        float* out = output + b * intermediate_size;
        
        /* 
         * Step 1: Compute RMSNorm 
         * RMS = sqrt(mean(x^2) + eps)
         * normalized = x / RMS * gamma
         */
        
        /* Compute sum of squares */
        __m256 sum_sq = _mm256_setzero_ps();
        
        for (int h = 0; h <= hidden_size - 8; h += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + h);
            sum_sq = _mm256_fmadd_ps(x_vec, x_vec, sum_sq);
        }
        
        /* Horizontal sum */
        float sum_sq_f = 0.0f;
        float sum_sq_arr[8];
        _mm256_storeu_ps(sum_sq_arr, sum_sq);
        for (int i = 0; i < 8; i++) sum_sq_f += sum_sq_arr[i];
        
        /* Handle remainder */
        for (int h = (hidden_size / 8) * 8; h < hidden_size; h++) {
            sum_sq_f += x[h] * x[h];
        }
        
        /* Compute RMS */
        float rms = sqrtf(sum_sq_f / hidden_size + RMS_EPS);
        __m256 rms_vec = _mm256_set1_ps(rms);
        
        /* 
         * Step 2: For each intermediate output, compute:
         * gate = sum(normalized[h] * w_gate[h, i])
         * value = sum(normalized[h] * w_value[h, i])
         * out[i] = Swish(gate) * value
         * 
         * We fuse the RMSNorm multiplication into the dot product
         */
        
        /* Precompute normalized input with gamma */
        /* We'll do this on-the-fly to save memory */
        
        for (int i = 0; i < intermediate_size; i++) {
            /* Compute dot products for gate and value */
            __m256 dot_gate = _mm256_setzero_ps();
            __m256 dot_value = _mm256_setzero_ps();
            
            const float* wg = w_gate + i * hidden_size;
            const float* wv = w_value + i * hidden_size;
            
            for (int h = 0; h <= hidden_size - 8; h += 8) {
                /* Load input and weights */
                __m256 x_vec = _mm256_loadu_ps(x + h);
                __m256 wg_vec = _mm256_loadu_ps(wg + h);
                __m256 wv_vec = _mm256_loadu_ps(wv + h);
                __m256 gamma_vec = _mm256_loadu_ps(gamma + h);
                
                /* Normalize: x / rms */
                __m256 x_norm = _mm256_div_ps(x_vec, rms_vec);
                
                /* Apply gamma */
                x_norm = _mm256_mul_ps(x_norm, gamma_vec);
                
                /* Accumulate dot products */
                dot_gate = _mm256_fmadd_ps(x_norm, wg_vec, dot_gate);
                dot_value = _mm256_fmadd_ps(x_norm, wv_vec, dot_value);
                
                /* Prefetch next cache line */
                _mm_prefetch((const char*)(wg + h + 64), _MM_HINT_T0);
                _mm_prefetch((const char*)(wv + h + 64), _MM_HINT_T0);
            }
            
            /* Horizontal sum for gate */
            float gate_arr[8];
            _mm256_storeu_ps(gate_arr, dot_gate);
            float gate_f = 0.0f;
            for (int j = 0; j < 8; j++) gate_f += gate_arr[j];
            
            /* Horizontal sum for value */
            float value_arr[8];
            _mm256_storeu_ps(value_arr, dot_value);
            float value_f = 0.0f;
            for (int j = 0; j < 8; j++) value_f += value_arr[j];
            
            /* Handle remainder */
            for (int h = (hidden_size / 8) * 8; h < hidden_size; h++) {
                float x_norm = x[h] / rms * gamma[h];
                gate_f += x_norm * wg[h];
                value_f += x_norm * wv[h];
            }
            
            /* Apply Swish to gate and multiply with value */
            float swish_gate = gate_f * fast_sigmoid_f(SWISH_BETA * gate_f);
            out[i] = swish_gate * value_f;
        }
    }
}

/* 
 * Simpler fused kernel for single vector (batch=1)
 * This is the common case for token-by-token generation
 */
void fused_rmsnorm_swiglu_single(
    const float* x,              /* [hidden] input */
    const void* w_gate_q4,       /* Q4_K quantized gate weights */
    const void* w_value_q4,      /* Q4_K quantized value weights */
    const float* gamma,          /* [hidden] RMSNorm weights */
    float* out,                  /* [intermediate] output */
    int hidden_size,
    int intermediate_size
) {
    (void)w_gate_q4;
    (void)w_value_q4;
    
    /* Compute RMS */
    __m256 sum_sq = _mm256_setzero_ps();
    
    for (int h = 0; h <= hidden_size - 8; h += 8) {
        __m256 x_vec = _mm256_loadu_ps(x + h);
        sum_sq = _mm256_fmadd_ps(x_vec, x_vec, sum_sq);
    }
    
    float sum_sq_f = 0.0f;
    float sum_sq_arr[8];
    _mm256_storeu_ps(sum_sq_arr, sum_sq);
    for (int i = 0; i < 8; i++) sum_sq_f += sum_sq_arr[i];
    
    for (int h = (hidden_size / 8) * 8; h < hidden_size; h++) {
        sum_sq_f += x[h] * x[h];
    }
    
    float rms = sqrtf(sum_sq_f / hidden_size + RMS_EPS);
    
    /* For now, placeholder - would need Q4_K matmul integrated */
    /* This demonstrates the fused structure */
    (void)gamma;
    (void)out;
    (void)intermediate_size;
    (void)rms;
}

/* 
 * Optimized RMSNorm only (for cases where we don't fuse)
 */
void rmsnorm_forward_optimized(
    const float* input,
    const float* gamma,
    float* output,
    int batch_size,
    int hidden_size
) {
    const __m256 eps_vec = _mm256_set1_ps(RMS_EPS);
    
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch_size; b++) {
        const float* x = input + b * hidden_size;
        float* out = output + b * hidden_size;
        
        /* Compute sum of squares */
        __m256 sum_sq = _mm256_setzero_ps();
        
        for (int h = 0; h <= hidden_size - 8; h += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + h);
            sum_sq = _mm256_fmadd_ps(x_vec, x_vec, sum_sq);
        }
        
        float sum_sq_f = 0.0f;
        float sum_sq_arr[8];
        _mm256_storeu_ps(sum_sq_arr, sum_sq);
        for (int i = 0; i < 8; i++) sum_sq_f += sum_sq_arr[i];
        
        for (int h = (hidden_size / 8) * 8; h < hidden_size; h++) {
            sum_sq_f += x[h] * x[h];
        }
        
        float rms = sqrtf(sum_sq_f / hidden_size + RMS_EPS);
        __m256 rms_vec = _mm256_set1_ps(rms);
        
        /* Normalize and apply gamma */
        for (int h = 0; h <= hidden_size - 8; h += 8) {
            __m256 x_vec = _mm256_loadu_ps(x + h);
            __m256 gamma_vec = _mm256_loadu_ps(gamma + h);
            
            __m256 x_norm = _mm256_div_ps(x_vec, rms_vec);
            x_norm = _mm256_mul_ps(x_norm, gamma_vec);
            
            _mm256_storeu_ps(out + h, x_norm);
        }
        
        /* Handle remainder */
        for (int h = (hidden_size / 8) * 8; h < hidden_size; h++) {
            out[h] = (x[h] / rms) * gamma[h];
        }
    }
}

/*
 * Optimized SwiGLU only (for cases where we don't fuse)
 */
void swiglu_forward_optimized(
    const float* input,
    const float* w_gate,
    const float* w_value,
    float* output,
    int batch_size,
    int hidden_size,
    int intermediate_size
) {
    #pragma omp parallel for schedule(static)
    for (int b = 0; b < batch_size; b++) {
        const float* x = input + b * hidden_size;
        float* out = output + b * intermediate_size;
        
        for (int i = 0; i < intermediate_size; i++) {
            __m256 dot_gate = _mm256_setzero_ps();
            __m256 dot_value = _mm256_setzero_ps();
            
            const float* wg = w_gate + i * hidden_size;
            const float* wv = w_value + i * hidden_size;
            
            for (int h = 0; h <= hidden_size - 8; h += 8) {
                __m256 x_vec = _mm256_loadu_ps(x + h);
                __m256 wg_vec = _mm256_loadu_ps(wg + h);
                __m256 wv_vec = _mm256_loadu_ps(wv + h);
                
                dot_gate = _mm256_fmadd_ps(x_vec, wg_vec, dot_gate);
                dot_value = _mm256_fmadd_ps(x_vec, wv_vec, dot_value);
            }
            
            float gate_f = 0.0f, value_f = 0.0f;
            float gate_arr[8], value_arr[8];
            _mm256_storeu_ps(gate_arr, dot_gate);
            _mm256_storeu_ps(value_arr, dot_value);
            for (int j = 0; j < 8; j++) {
                gate_f += gate_arr[j];
                value_f += value_arr[j];
            }
            
            for (int h = (hidden_size / 8) * 8; h < hidden_size; h++) {
                gate_f += x[h] * wg[h];
                value_f += x[h] * wv[h];
            }
            
            float swish_gate = gate_f * fast_sigmoid_f(SWISH_BETA * gate_f);
            out[i] = swish_gate * value_f;
        }
    }
}

#else /* No AVX2 */

void fused_rmsnorm_swiglu_forward(
    const float* input,
    const float* w_gate,
    const float* w_value,
    const float* gamma,
    float* output,
    int batch_size,
    int hidden_size,
    int intermediate_size
) {
    /* Scalar fallback */
    for (int b = 0; b < batch_size; b++) {
        const float* x = input + b * hidden_size;
        float* out = output + b * intermediate_size;
        
        /* Compute RMS */
        float sum_sq = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            sum_sq += x[h] * x[h];
        }
        float rms = sqrtf(sum_sq / hidden_size + RMS_EPS);
        
        /* For each intermediate output */
        for (int i = 0; i < intermediate_size; i++) {
            float gate_f = 0.0f, value_f = 0.0f;
            
            for (int h = 0; h < hidden_size; h++) {
                float x_norm = (x[h] / rms) * gamma[h];
                gate_f += x_norm * w_gate[i * hidden_size + h];
                value_f += x_norm * w_value[i * hidden_size + h];
            }
            
            float swish_gate = gate_f / (1.0f + expf(-SWISH_BETA * gate_f));
            out[i] = swish_gate * value_f;
        }
    }
}

void rmsnorm_forward_optimized(
    const float* input,
    const float* gamma,
    float* output,
    int batch_size,
    int hidden_size
) {
    for (int b = 0; b < batch_size; b++) {
        const float* x = input + b * hidden_size;
        float* out = output + b * hidden_size;
        
        float sum_sq = 0.0f;
        for (int h = 0; h < hidden_size; h++) {
            sum_sq += x[h] * x[h];
        }
        float rms = sqrtf(sum_sq / hidden_size + RMS_EPS);
        
        for (int h = 0; h < hidden_size; h++) {
            out[h] = (x[h] / rms) * gamma[h];
        }
    }
}

void swiglu_forward_optimized(
    const float* input,
    const float* w_gate,
    const float* w_value,
    float* output,
    int batch_size,
    int hidden_size,
    int intermediate_size
) {
    for (int b = 0; b < batch_size; b++) {
        const float* x = input + b * hidden_size;
        float* out = output + b * intermediate_size;
        
        for (int i = 0; i < intermediate_size; i++) {
            float gate_f = 0.0f, value_f = 0.0f;
            
            for (int h = 0; h < hidden_size; h++) {
                gate_f += x[h] * w_gate[i * hidden_size + h];
                value_f += x[h] * w_value[i * hidden_size + h];
            }
            
            float swish_gate = gate_f / (1.0f + expf(-SWISH_BETA * gate_f));
            out[i] = swish_gate * value_f;
        }
    }
}

#endif /* __AVX2__ */
