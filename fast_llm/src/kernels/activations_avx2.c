/*
 * AVX2-Optimized Activation Functions
 * 
 * Optimizes:
 * - SiLU (Swish): x * sigmoid(x)
 * - SwiGLU: SiLU(gate) * up
 * - RMSNorm
 * 
 * These were taking 53% of total time - more than matmul!
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

#ifdef __AVX2__

/* Fast sigmoid approximation using polynomial */
static inline __m256 fast_sigmoid_ps(__m256 x) {
    /* Clamp to avoid overflow */
    __m256 xmin = _mm256_set1_ps(-10.0f);
    __m256 xmax = _mm256_set1_ps(10.0f);
    x = _mm256_max_ps(xmin, _mm256_min_ps(xmax, x));
    
    /* Polynomial approximation of sigmoid */
    /* sigmoid(x) ≈ 0.5 + 0.5 * tanh(x/2) */
    /* tanh(x) ≈ x * (1 - x^2/3) for small x */
    
    /* Actually use a better approximation: */
    __m256 one = _mm256_set1_ps(1.0f);
    __m256 half = _mm256_set1_ps(0.5f);
    __m256 neg_x = _mm256_sub_ps(_mm256_setzero_ps(), x);
    
    /* Approximate exp(-x) with polynomial */
    /* exp(x) ≈ 1 + x + x^2/2 + x^3/6 + x^4/24 */
    __m256 x2 = _mm256_mul_ps(x, x);
    __m256 x3 = _mm256_mul_ps(x2, x);
    __m256 x4 = _mm256_mul_ps(x2, x2);
    
    __m256 exp_approx = _mm256_add_ps(one, 
        _mm256_add_ps(neg_x,
            _mm256_add_ps(_mm256_mul_ps(x2, _mm256_set1_ps(0.5f)),
                _mm256_add_ps(_mm256_mul_ps(x3, _mm256_set1_ps(-0.166667f)),
                    _mm256_mul_ps(x4, _mm256_set1_ps(0.041667f))))));
    
    /* sigmoid = 1 / (1 + exp(-x)) */
    __m256 denom = _mm256_add_ps(one, exp_approx);
    return _mm256_div_ps(one, denom);
}

/* SiLU: x * sigmoid(x) - AVX2 optimized */
void silu_avx2(const float* input, float* output, int n) {
    int i = 0;
    
    /* Process 8 floats at a time */
    for (; i <= n - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 sig = fast_sigmoid_ps(x);
        __m256 result = _mm256_mul_ps(x, sig);
        _mm256_storeu_ps(output + i, result);
    }
    
    /* Handle remainder */
    for (; i < n; i++) {
        float x = input[i];
        float sig = 1.0f / (1.0f + expf(-x));
        output[i] = x * sig;
    }
}

/* SwiGLU: gate * sigmoid(gate) * up - AVX2 optimized */
void swiglu_avx2(const float* gate, const float* up, float* output, int n) {
    int i = 0;
    
    /* Process 8 floats at a time */
    for (; i <= n - 8; i += 8) {
        __m256 g = _mm256_loadu_ps(gate + i);
        __m256 u = _mm256_loadu_ps(up + i);
        __m256 sig = fast_sigmoid_ps(g);
        __m256 silu = _mm256_mul_ps(g, sig);
        __m256 result = _mm256_mul_ps(silu, u);
        _mm256_storeu_ps(output + i, result);
    }
    
    /* Handle remainder */
    for (; i < n; i++) {
        float g = gate[i];
        float sig = 1.0f / (1.0f + expf(-g));
        output[i] = g * sig * up[i];
    }
}

/* RMSNorm: x / sqrt(mean(x^2) + eps) - AVX2 optimized */
void rms_norm_avx2(const float* input, float* output, int n, float eps) {
    /* Compute sum of squares */
    __m256 sum_vec = _mm256_setzero_ps();
    int i = 0;
    
    for (; i <= n - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        sum_vec = _mm256_fmadd_ps(x, x, sum_vec);
    }
    
    /* Horizontal sum */
    float sum_arr[8];
    _mm256_storeu_ps(sum_arr, sum_vec);
    float sum_sq = sum_arr[0] + sum_arr[1] + sum_arr[2] + sum_arr[3] +
                   sum_arr[4] + sum_arr[5] + sum_arr[6] + sum_arr[7];
    
    /* Handle remainder */
    for (; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    
    float scale = 1.0f / sqrtf(sum_sq / n + eps);
    __m256 scale_vec = _mm256_set1_ps(scale);
    
    /* Normalize */
    i = 0;
    for (; i <= n - 8; i += 8) {
        __m256 x = _mm256_loadu_ps(input + i);
        __m256 result = _mm256_mul_ps(x, scale_vec);
        _mm256_storeu_ps(output + i, result);
    }
    
    for (; i < n; i++) {
        output[i] = input[i] * scale;
    }
}

#else /* No AVX2 */

void silu_avx2(const float* input, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float x = input[i];
        output[i] = x / (1.0f + expf(-x));
    }
}

void swiglu_avx2(const float* gate, const float* up, float* output, int n) {
    for (int i = 0; i < n; i++) {
        float g = gate[i];
        output[i] = g / (1.0f + expf(-g)) * up[i];
    }
}

void rms_norm_avx2(const float* input, float* output, int n, float eps) {
    float sum_sq = 0.0f;
    for (int i = 0; i < n; i++) {
        sum_sq += input[i] * input[i];
    }
    float scale = 1.0f / sqrtf(sum_sq / n + eps);
    for (int i = 0; i < n; i++) {
        output[i] = input[i] * scale;
    }
}

#endif /* __AVX2__ */
