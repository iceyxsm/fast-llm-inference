/*
 * CPU Feature Detection Implementation
 */

#include "cpu_features.h"
#include <stdio.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#include <intrin.h>
#else
#include <cpuid.h>
#endif

/* CPUID leaf values */
#define CPUID_LEAF_1 0x00000001
#define CPUID_LEAF_7 0x00000007
#define CPUID_LEAF_AMX 0x0000001E

/* Extended feature flags */
#define CPUID_ECX_SSE2 (1 << 26)
#define CPUID_ECX_AVX (1 << 28)

/* Feature flags from leaf 7 (ebx) */
#define CPUID_EBX_AVX2 (1 << 5)
#define CPUID_EBX_AVX512F (1 << 16)
#define CPUID_EBX_AVX512DQ (1 << 17)
#define CPUID_EBX_AVX512BW (1 << 30)
#define CPUID_EBX_AVX512VL (1 << 31)

/* Feature flags from leaf 7 (ecx) */
#define CPUID_ECX_AVX512VNNI (1 << 11)
#define CPUID_ECX_AVX512AMX (1 << 24)  /* AMX-BF16 tile support */

#ifdef _WIN32
static void cpuid(int info[4], int leaf) {
    __cpuid(info, leaf);
}

static void cpuidex(int info[4], int leaf, int subleaf) {
    __cpuidex(info, leaf, subleaf);
}
#else
static void cpuid(int info[4], int leaf) {
    __cpuid_count(leaf, 0, info[0], info[1], info[2], info[3]);
}

static void cpuidex(int info[4], int leaf, int subleaf) {
    __cpuid_count(leaf, subleaf, info[0], info[1], info[2], info[3]);
}
#endif

cpu_features_t detect_cpu_features(void) {
    cpu_features_t features = {0};
    int info[4] = {0};
    
    /* Get basic info */
    cpuid(info, 0);
    int max_leaf = info[0];
    
    /* Leaf 1: Basic features */
    if (max_leaf >= 1) {
        cpuid(info, CPUID_LEAF_1);
        features.has_sse2 = (info[2] & CPUID_ECX_SSE2) != 0;
        features.has_avx = (info[2] & CPUID_ECX_AVX) != 0;
    }
    
    /* Leaf 7: Extended features */
    if (max_leaf >= 7) {
        cpuidex(info, CPUID_LEAF_7, 0);
        
        features.has_avx2 = (info[1] & CPUID_EBX_AVX2) != 0;
        features.has_avx512f = (info[1] & CPUID_EBX_AVX512F) != 0;
        features.has_avx512bw = (info[1] & CPUID_EBX_AVX512BW) != 0;
        features.has_avx512vl = (info[1] & CPUID_EBX_AVX512VL) != 0;
        features.has_avx512vnni = (info[2] & CPUID_ECX_AVX512VNNI) != 0;
        
        /* Check for AMX */
        features.has_amx = (info[2] & CPUID_ECX_AVX512AMX) != 0;
    }
    
    /* Get core count */
#ifdef _WIN32
    SYSTEM_INFO sysinfo;
    GetSystemInfo(&sysinfo);
    features.num_threads = sysinfo.dwNumberOfProcessors;
#else
    /* Try to read from /proc/cpuinfo or use sysconf */
    features.num_threads = 1;
    FILE* f = fopen("/proc/cpuinfo", "r");
    if (f) {
        char line[256];
        features.num_threads = 0;
        while (fgets(line, sizeof(line), f)) {
            if (strncmp(line, "processor", 9) == 0) {
                features.num_threads++;
            }
        }
        fclose(f);
    }
    if (features.num_threads == 0) {
        features.num_threads = 1;
    }
#endif
    features.num_cores = features.num_threads;  /* Approximation */
    
    return features;
}

void print_cpu_info(const cpu_features_t* f) {
    printf("CPU Features:\n");
    printf("  SSE2:      %s\n", f->has_sse2 ? "YES" : "NO");
    printf("  AVX:       %s\n", f->has_avx ? "YES" : "NO");
    printf("  AVX2:      %s\n", f->has_avx2 ? "YES" : "NO");
    printf("  AVX-512F:  %s\n", f->has_avx512f ? "YES" : "NO");
    printf("  AVX-512BW: %s\n", f->has_avx512bw ? "YES" : "NO");
    printf("  AVX-512VL: %s\n", f->has_avx512vl ? "YES" : "NO");
    printf("  AVX-512VNNI: %s\n", f->has_avx512vnni ? "YES" : "NO");
    printf("  AMX:       %s\n", f->has_amx ? "YES" : "NO");
    printf("  Cores:     %d\n", f->num_cores);
    printf("  Threads:   %d\n", f->num_threads);
    
    /* Recommendations */
    printf("\nRecommended kernel:\n");
    if (f->has_amx) {
        printf("  -> Intel AMX (fastest)\n");
    } else if (f->has_avx512vnni) {
        printf("  -> AVX-512 VNNI\n");
    } else if (f->has_avx512f) {
        printf("  -> AVX-512\n");
    } else if (f->has_avx2) {
        printf("  -> AVX2\n");
    } else {
        printf("  -> SSE2 (fallback)\n");
    }
}
