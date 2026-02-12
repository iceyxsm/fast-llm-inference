/*
 * HIGHLY OPTIMIZED Matrix Multiplication Kernels
 * 
 * Techniques from research:
 * - Cache blocking/tiling (fit in L1/L2 cache)
 * - FMA (Fused Multiply-Add) instructions
 * - Multi-threading with OpenMP
 * - Column-major storage
 * - Weight pre-packing
 * 
 * Reference: https://salykova.github.io/gemm-cpu
 * Reference: Intel LLM Inference Paper
 */

#include "matmul.h"
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

#ifdef _OPENMP
#include <omp.h>
#endif

/* 
 * Cache block sizes (tuned for L1/L2 cache)
 * L1 cache typically 32-64 KB
 * L2 cache typically 256-512 KB per core
 */
#define MC 64   /* Block size in M dimension */
#define NC 512  /* Block size in N dimension */
#define KC 256  /* Block size in K dimension (fits in L1) */
#define NR 8    /* Micro-kernel size in N (AVX2 = 8 floats) */
#define MR 4    /* Micro-kernel size in M */

/*
 * Micro-kernel: 4x8 matrix multiply using AVX2 FMA
 * Computes C[4x8] += A[4xKC] * B[KCx8]
 * 
 * This is the innermost loop - must be hand-optimized
 */
#ifdef __AVX2__
#include <immintrin.h>

static void micro_kernel_4x8(int kc, const float* A, const float* B, float* C, 
                             int lda, int ldb, int ldc) {
    /* Load C accumulators */
    __m256 c0 = _mm256_loadu_ps(C + 0 * ldc);
    __m256 c1 = _mm256_loadu_ps(C + 1 * ldc);
    __m256 c2 = _mm256_loadu_ps(C + 2 * ldc);
    __m256 c3 = _mm256_loadu_ps(C + 3 * ldc);
    
    /* Main loop - process KC elements */
    for (int k = 0; k < kc; k++) {
        /* Load B row (8 elements) */
        __m256 b = _mm256_loadu_ps(B + k * ldb);
        
        /* Load A column and broadcast for each row */
        __m256 a0 = _mm256_broadcast_ss(A + 0 * lda + k);
        __m256 a1 = _mm256_broadcast_ss(A + 1 * lda + k);
        __m256 a2 = _mm256_broadcast_ss(A + 2 * lda + k);
        __m256 a3 = _mm256_broadcast_ss(A + 3 * lda + k);
        
        /* FMA: C += A * B */
        c0 = _mm256_fmadd_ps(a0, b, c0);
        c1 = _mm256_fmadd_ps(a1, b, c1);
        c2 = _mm256_fmadd_ps(a2, b, c2);
        c3 = _mm256_fmadd_ps(a3, b, c3);
    }
    
    /* Store results */
    _mm256_storeu_ps(C + 0 * ldc, c0);
    _mm256_storeu_ps(C + 1 * ldc, c1);
    _mm256_storeu_ps(C + 2 * ldc, c2);
    _mm256_storeu_ps(C + 3 * ldc, c3);
}

/*
 * Macro-kernel: Process MCxNC block
 * Breaks into micro-kernels
 */
static void macro_kernel(int mc, int nc, int kc,
                         const float* A_packed, const float* B_packed,
                         float* C, int ldc) {
    for (int j = 0; j < nc; j += NR) {
        int nr = (j + NR <= nc) ? NR : (nc - j);
        
        for (int i = 0; i < mc; i += MR) {
            int mr = (i + MR <= mc) ? MR : (mc - i);
            
            /* Call micro-kernel */
            if (mr == MR && nr == NR) {
                /* Full micro-kernel */
                micro_kernel_4x8(kc, 
                    A_packed + i * kc, 
                    B_packed + j * kc,
                    C + i * ldc + j,
                    kc, NR, ldc);
            } else {
                /* Edge case - partial block */
                float C_temp[MR * NR] = {0};
                micro_kernel_4x8(kc,
                    A_packed + i * kc,
                    B_packed + j * kc,
                    C_temp,
                    kc, NR, NR);
                
                /* Copy back */
                for (int ii = 0; ii < mr; ii++) {
                    for (int jj = 0; jj < nr; jj++) {
                        C[(i + ii) * ldc + (j + jj)] += C_temp[ii * NR + jj];
                    }
                }
            }
        }
    }
}

/*
 * Pack A matrix (MC x KC) into contiguous memory
 * This improves cache locality
 */
static void pack_a(int mc, int kc, const float* A, int lda, float* A_packed) {
    for (int i = 0; i < mc; i += MR) {
        int mr = (i + MR <= mc) ? MR : (mc - i);
        
        for (int k = 0; k < kc; k++) {
            for (int ii = 0; ii < mr; ii++) {
                A_packed[i * kc + ii * kc + k] = A[(i + ii) * lda + k];
            }
            /* Padding for incomplete blocks */
            for (int ii = mr; ii < MR; ii++) {
                A_packed[i * kc + ii * kc + k] = 0.0f;
            }
        }
    }
}

/*
 * Pack B matrix (KC x NC) into contiguous memory
 * Transpose during pack for better access pattern
 */
static void pack_b(int kc, int nc, const float* B, int ldb, float* B_packed) {
    for (int j = 0; j < nc; j += NR) {
        int nr = (j + NR <= nc) ? NR : (nc - j);
        
        for (int k = 0; k < kc; k++) {
            for (int jj = 0; jj < nr; jj++) {
                B_packed[j * kc + jj * kc + k] = B[k * ldb + (j + jj)];
            }
            for (int jj = nr; jj < NR; jj++) {
                B_packed[j * kc + jj * kc + k] = 0.0f;
            }
        }
    }
}

/*
 * Pre-dequantize Q4 weights to float32
 */
static void dequantize_q4_to_f32_row(const quantized_tensor_t* B_q, int n, 
                                     float* B_f32, int K) {
    const block_q4_t* B = (const block_q4_t*)B_q->blocks;
    int blocks_per_row = (K + Q4_BLOCK_SIZE - 1) / Q4_BLOCK_SIZE;
    
    for (int k = 0; k < K; k++) {
        int block_idx = n * blocks_per_row + k / Q4_BLOCK_SIZE;
        int offset = k % Q4_BLOCK_SIZE;
        const block_q4_t* block = &B[block_idx];
        
        int byte_idx = offset / 2;
        int nibble = offset % 2;
        int q = (block->qs[byte_idx] >> (nibble * 4)) & 0xF;
        
        B_f32[k] = block->zero_point + q * block->scale;
    }
}

/*
 * Optimized Q4 matmul with cache blocking
 * C[M x N] = A[M x K] @ B[N x K]^T
 */
void q4_matmul_blocked(const float* A, const quantized_tensor_t* B_q, float* C,
                       int M, int N, int K) {
    /* Allocate packed buffers */
    /* A_packed: MC x KC */
    /* B_packed: KC x NC but stored as NC x KC (transposed for access) */
    float* A_packed = (float*)aligned_malloc(MC * KC * sizeof(float), 64);
    float* B_packed = (float*)aligned_malloc(KC * NC * sizeof(float), 64);
    float* B_row = (float*)aligned_malloc(K * sizeof(float), 64);
    
    if (!A_packed || !B_packed || !B_row) {
        fprintf(stderr, "Failed to allocate packed buffers\n");
        return;
    }
    
    memset(C, 0, M * N * sizeof(float));
    
    /* Loop order: M -> N -> K (cache friendly) */
    #ifdef _OPENMP
    #pragma omp parallel for schedule(static) private(B_row, B_packed)
    #endif
    for (int j = 0; j < N; j += NC) {
        int nc = (j + NC <= N) ? NC : (N - j);
        
        /* Allocate thread-local buffers */
        float* B_packed_local = (float*)aligned_malloc(KC * NC * sizeof(float), 64);
        float* B_row_local = (float*)aligned_malloc(K * sizeof(float), 64);
        
        for (int k = 0; k < K; k += KC) {
            int kc = (k + KC <= K) ? KC : (K - k);
            
            /* Pack B block */
            for (int jj = 0; jj < nc; jj++) {
                /* Dequantize one row of B */
                dequantize_q4_to_f32_row(B_q, j + jj, B_row_local, K);
                
                /* Copy to packed buffer */
                for (int kk = 0; kk < kc; kk++) {
                    B_packed_local[jj * KC + kk] = B_row_local[k + kk];
                }
            }
            
            for (int i = 0; i < M; i += MC) {
                int mc = (i + MC <= M) ? MC : (M - i);
                
                /* Pack A block (single row or small batch) */
                for (int ii = 0; ii < mc; ii++) {
                    for (int kk = 0; kk < kc; kk++) {
                        A_packed[ii * KC + kk] = A[(i + ii) * K + k + kk];
                    }
                }
                
                /* Compute block */
                macro_kernel(mc, nc, kc, A_packed, B_packed_local, 
                           C + i * N + j, N);
            }
        }
        
        aligned_free(B_packed_local);
        aligned_free(B_row_local);
    }
    
    aligned_free(A_packed);
    aligned_free(B_packed);
    aligned_free(B_row);
}

#else /* !__AVX2__ */

void q4_matmul_blocked(const float* A, const quantized_tensor_t* B, float* C,
                       int M, int N, int K) {
    /* Fallback to scalar */
    q4_matmul(A, B, C, M, N, K);
}

#endif /* __AVX2__ */
