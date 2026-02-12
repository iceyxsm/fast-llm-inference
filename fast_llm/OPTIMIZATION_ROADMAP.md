# Optimization Roadmap - Extracting Max Performance

## Current Status (as of research)

| Kernel | Time/Layer | Tok/Sec | Improvement |
|--------|-----------|---------|-------------|
| Scalar | 174 ms | 0.2 | baseline |
| AVX2 basic | 92 ms | 0.3 | 1.9x |
| AVX2 single-token | 42 ms | 0.7 | **4.1x** |
| **Target: llama.cpp** | **~4 ms** | **25** | **~40x faster** |

## Why We're Still Slow

### The Real Bottleneck: Weight Unpacking

Every single multiply requires:
```c
/* 4-bit weight unpacking */
int byte_idx = k / 2;
int nibble = k % 2;
int q = (packed[byte_idx] >> (nibble * 4)) & 0xF;
float w = zero + q * scale;
```

**That's ~10 instructions per weight!**

### llama.cpp's Secret

They pre-convert Q4 → int8 at **model load time**:
```c
// Load time (once)
int8_t w_int8 = unpack_q4_to_int8(packed_weight);

// Runtime (millions of times)
sum += a * w_int8;  // Direct multiply, no unpacking!
```

**Result: 10x fewer instructions per multiply**

## Research Findings - Key Optimization Techniques

### 1. Cache Blocking (From GEMM Research)
```
Paper: "Anatomy of High-Performance Matrix Multiplication"
Technique: Tiling to fit in L1/L2 cache
Block sizes: MC=64, NC=512, KC=256
```

**When to use:** Large batch sizes (M > 8)
**Our case:** M=1 (single token) → Doesn't help

### 2. FMA Instructions (Fused Multiply-Add)
```c
// Instead of:
c = c + a * b;  // 2 instructions

// Use:
c = fma(a, b, c);  // 1 instruction, same latency
```

**Speedup: 2x peak FLOPS**

### 3. Multi-threading with OpenMP
```c
#pragma omp parallel for schedule(static)
for (int n = 0; n < N; n++) {
    // Each thread processes some columns
}
```

**Speedup: Near-linear with cores (for large N)**

### 4. Weight Pre-dequantization (INT8)
```c
// Intel's approach from "Efficient LLM Inference on CPUs"
// Store weights as INT8 instead of Q4
// Use VNNI instructions for dot product
```

**Speedup: 4-8x**

### 5. Memory Layout Optimization
```
Current: Row-major
Optimal: Column-major (for B matrix)
Reason: Sequential memory access
```

## The Winning Strategy (Based on Research)

### Phase 1: Weight Pre-dequantization (Biggest Win)
```c
/* At model load time */
typedef struct {
    int8_t* weights_int8;  // Pre-unpacked, size [N x K]
    float*  scales;        // Per-channel scales [N]
} dequantized_q4_t;

/* Runtime kernel */
void matmul_dequantized(const float* A, 
                        const dequantized_q4_t* B,
                        float* C, int N, int K) {
    #pragma omp parallel for
    for (int n = 0; n < N; n++) {
        float sum = 0;
        for (int k = 0; k < K; k++) {
            // Direct int8 multiply, no unpacking!
            sum += A[k] * (B->weights_int8[n*K + k] * B->scales[n]);
        }
        C[n] = sum;
    }
}
```

**Expected speedup: 5-10x**

### Phase 2: Int8 Dot Product with AVX2
```c
// Process 32 int8 values at once
__m256i a_vec = _mm256_loadu_si256((__m256i*)(A + k));
__m256i b_vec = _mm256_loadu_si256((__m256i*)(B + k));

// Convert to int16, then int32, multiply and add
// ... (AVX2 int8 dot product)
```

**Expected speedup: Additional 2-3x**

### Phase 3: Speculative Decoding (EAGLE-3)
```c
// Draft 4 tokens with tiny model
// Verify in parallel
// Accept ~3 tokens on average
```

**Expected speedup: 2.5x**

### Combined Target
```
Current:   0.7 tok/sec
After Phase 1+2:  7-20 tok/sec (10-30x)
After Phase 3:    17-50 tok/sec (25-70x)
llama.cpp:        25 tok/sec
```

**We can beat llama.cpp!**

## Implementation Priority

1. **Weight pre-dequantization** (5-10x speedup)
2. **Int8 AVX2 kernels** (2-3x additional)
3. **EAGLE-3 speculative** (2.5x additional)
4. **Multi-threading optimization** (1.5x additional)

## Next Steps

1. Implement `dequantized_q4_t` structure
2. Pre-convert all Q4 weights at model load
3. Write int8 dot product kernels
4. Benchmark vs llama.cpp

**Estimated final performance: 20-50 tok/sec**
**vs llama.cpp: 25 tok/sec**

**Ready to implement Phase 1 (weight pre-dequantization)?**
