/*
 * Hardware Utilization Analysis - Accurate Version
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main() {
    printf("========================================\n");
    printf("  HARDWARE MAX-OUT ANALYSIS\n");
    printf("========================================\n\n");
    
    /* Phi-3-mini specs */
    int hidden = 3072;
    int intermediate = 8192;
    int layers = 32;
    int head_dim = 96;  /* 3072 / 32 heads */
    int num_heads = 32;
    
    /* Hardware specs */
    double memory_bw_gb_s = 61.0;  /* Measured from bench_memory_bandwidth.c */
    int num_cores = 16;
    double avx2_gflops_per_core = 140.0;  /* Theoretical AVX2 FMA peak per core */
    double total_compute_gflops = avx2_gflops_per_core * num_cores;
    
    printf("Hardware Specs:\n");
    printf("  CPU: 16 cores with AVX2\n");
    printf("  Memory: DDR4-3200 (measured: %.1f GB/s)\n", memory_bw_gb_s);
    printf("  Compute peak: %.0f GFLOPS\n", total_compute_gflops);
    printf("\n");
    
    /* Memory traffic per token */
    printf("Memory Traffic Per Token:\n");
    printf("  FFN Gate: %d x %d = %.1f MB weights\n", intermediate, hidden, 
           (intermediate * hidden * 1.0) / (1024*1024));
    printf("  FFN Up:   %d x %d = %.1f MB weights\n", intermediate, hidden,
           (intermediate * hidden * 1.0) / (1024*1024));
    printf("  FFN Down: %d x %d = %.1f MB weights\n", hidden, intermediate,
           (hidden * intermediate * 1.0) / (1024*1024));
    
    double ffn_weights_mb = (2.0 * intermediate * hidden + hidden * intermediate) * 1.0 / (1024*1024);
    double total_ffn_mb = ffn_weights_mb * layers;
    printf("  FFN total per token: %.2f MB\n", total_ffn_mb);
    
    /* Attention memory traffic (rough estimate) */
    double attn_qkv_mb = 3.0 * hidden * hidden * layers * 1.0 / (1024*1024);
    double attn_o_mb = hidden * hidden * layers * 1.0 / (1024*1024);
    double total_attn_mb = (attn_qkv_mb + attn_o_mb) * 1.0 / (1024*1024);
    printf("  Attention per token: %.2f MB\n", total_attn_mb);
    
    double total_memory_mb = total_ffn_mb + total_attn_mb;
    printf("  TOTAL per token: %.2f MB\n\n", total_memory_mb);
    
    /* Theoretical limits */
    printf("Performance Limits:\n");
    double max_tok_sec_memory = memory_bw_gb_s * 1024.0 / total_memory_mb;
    printf("  Memory bound: %.1f tok/sec (at %.1f GB/s)\n", max_tok_sec_memory, memory_bw_gb_s);
    
    /* FLOPs per token */
    long long ffn_flops = 2LL * layers * (2LL * intermediate * hidden + hidden * intermediate);
    long long attn_flops = 2LL * layers * (4LL * hidden * hidden);  /* Q,K,V,O */
    long long total_flops = ffn_flops + attn_flops;
    
    printf("  FFN FLOPs: %lld per token\n", ffn_flops);
    printf("  Attn FLOPs: %lld per token\n", attn_flops);
    printf("  Total FLOPs: %lld per token (%.2f GFLOPs)\n", total_flops, total_flops / 1e9);
    
    double max_tok_sec_compute = total_compute_gflops / (total_flops / 1e9);
    printf("  Compute bound: %.1f tok/sec (at %.0f GFLOPS)\n\n", max_tok_sec_compute, total_compute_gflops);
    
    /* Which bound are we? */
    printf("Bottleneck Analysis:\n");
    if (max_tok_sec_memory < max_tok_sec_compute) {
        printf("  → MEMORY BOUND\n");
        printf("    Memory limit: %.1f tok/sec\n", max_tok_sec_memory);
        printf("    Compute limit: %.1f tok/sec (%.1fx headroom)\n", 
               max_tok_sec_compute, max_tok_sec_compute / max_tok_sec_memory);
    } else {
        printf("  → COMPUTE BOUND\n");
    }
    
    /* Actual performance */
    double actual_tok_sec = 30.74;
    printf("\nActual Performance:\n");
    printf("  Measured: %.2f tok/sec\n", actual_tok_sec);
    printf("  Memory utilization: %.1f%%\n", 100.0 * actual_tok_sec / max_tok_sec_memory);
    printf("  Compute utilization: %.1f%%\n", 100.0 * actual_tok_sec / max_tok_sec_compute);
    
    /* Are we maxing out? */
    printf("\n========================================\n");
    printf("  ANSWER: Are we maxing out hardware?\n");
    printf("========================================\n\n");
    
    double memory_efficiency = 100.0 * actual_tok_sec / max_tok_sec_memory;
    
    if (memory_efficiency > 90.0) {
        printf("✅ YES - We are at %.1f%% of memory bandwidth limit!\n", memory_efficiency);
        printf("\nThe 6x16 kernel is near-optimal for this hardware.\n");
        printf("To go faster, we MUST reduce memory traffic.\n");
    } else if (memory_efficiency > 70.0) {
        printf("⚠️  MOSTLY - We are at %.1f%% of memory bandwidth limit.\n", memory_efficiency);
        printf("\nThere's some room for improvement (~%.0f%%), but\n", 100.0 - memory_efficiency);
        printf("we're fundamentally memory bound.\n");
    } else {
        printf("❌ NO - We are only at %.1f%% of memory bandwidth.\n", memory_efficiency);
        printf("\nThere may be implementation inefficiencies.\n");
    }
    
    printf("\nKey Insight:\n");
    printf("  Even with PERFECT code, max speed is %.1f tok/sec\n", max_tok_sec_memory);
    printf("  Current: %.1f tok/sec\n", actual_tok_sec);
    printf("  Gap to theoretical max: %.1f%%\n", 100.0 * actual_tok_sec / max_tok_sec_memory);
    
    printf("\nTo reach 50 tok/sec:\n");
    double needed_bw = 50.0 * total_memory_mb / 1024.0;
    printf("  Option 1: Need %.1f GB/s bandwidth (vs current %.1f GB/s)\n", 
           needed_bw, memory_bw_gb_s);
    printf("     → Requires DDR5 or HBM\n\n");
    
    double needed_quant = total_memory_mb * actual_tok_sec / 50.0;
    printf("  Option 2: Need %.2f MB per token (vs current %.2f MB)\n",
           needed_quant, total_memory_mb);
    printf("     → Requires 4-bit quantization (50%% traffic reduction)\n\n");
    
    printf("  Option 3: Reduce model size\n");
    printf("     → Use 20 layers instead of 32 (37%% speedup)\n");
    printf("     → Use smaller hidden size\n");
    
    return 0;
}
