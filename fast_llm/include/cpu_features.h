/*
 * CPU Feature Detection
 * Detects AVX-512, AMX, VNNI support
 */

#ifndef CPU_FEATURES_H
#define CPU_FEATURES_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

/* CPU Feature Flags */
typedef struct {
    bool has_sse2;
    bool has_avx;
    bool has_avx2;
    bool has_avx512f;
    bool has_avx512bw;
    bool has_avx512vl;
    bool has_avx512vnni;
    bool has_amx;
    int num_cores;
    int num_threads;
} cpu_features_t;

/* Detect CPU features */
cpu_features_t detect_cpu_features(void);

/* Print CPU info */
void print_cpu_info(const cpu_features_t* features);

/* Check specific features */
static inline bool has_fast_int8(const cpu_features_t* f) {
    return f->has_avx512vnni || f->has_avx512f;
}

static inline bool has_amx_support(const cpu_features_t* f) {
    return f->has_amx;
}

#ifdef __cplusplus
}
#endif

#endif /* CPU_FEATURES_H */
