/*
 * fllm CLI - Shared header
 */
#ifndef CLI_H
#define CLI_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <errno.h>

#ifdef _WIN32
  #define WIN32_LEAN_AND_MEAN
  #include <windows.h>
  #include <winsock2.h>
  #include <ws2tcpip.h>
  #include <psapi.h>
  #include <io.h>
  #include <process.h>
  #include <direct.h>
  #include <malloc.h>
  typedef SOCKET sock_t;
  #define SOCK_INVALID INVALID_SOCKET
  #define sock_close(s) closesocket(s)
  #define aligned_malloc(sz, al) _aligned_malloc(sz, al)
  #define aligned_free(p) _aligned_free(p)
  #define PATH_SEP '\\'
  #define PATH_SEP_S "\\"
  #define MKDIR_P(p) _mkdir(p)
  #define GETPID() _getpid()
  #define SLEEP_MS(ms) Sleep(ms)
#else
  #include <unistd.h>
  #include <sys/socket.h>
  #include <sys/stat.h>
  #include <sys/wait.h>
  #include <sys/resource.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <fcntl.h>
  typedef int sock_t;
  #define SOCK_INVALID (-1)
  #define sock_close(s) close(s)
  #define aligned_malloc(sz, al) aligned_alloc(al, sz)
  #define aligned_free(p) free(p)
  #define PATH_SEP '/'
  #define PATH_SEP_S "/"
  #define MKDIR_P(p) mkdir(p, 0755)
  #define GETPID() getpid()
  #define SLEEP_MS(ms) usleep((ms)*1000)
#endif

/* ── Constants ────────────────────────────────────────────────────── */
#define DAEMON_PORT       8199
#define PROTO_END         "<<FLLM_END>>\n"
#define FLLM_DIR_NAME     ".fllm"
#define MODELS_DIR_NAME   "models"
#define SPECS_FILE_NAME   "specs.dat"
#define CATALOG_CACHE     "catalog.json"
#define PID_FILE_NAME     "fllm.pid"
#define LOG_FILE_NAME     "fllm.log"

#define CLR_RESET   0
#define CLR_GREEN   10
#define CLR_CYAN    11
#define CLR_RED     12
#define CLR_YELLOW  14
#define CLR_WHITE   15

/* ── Hardware tier ────────────────────────────────────────────────── */
typedef enum {
    TIER_LOW = 0,
    TIER_MEDIUM,
    TIER_HIGH,
    TIER_ULTRA
} hw_tier_t;

/* ── Saved specs ──────────────────────────────────────────────────── */
/* ── Per-GPU info ─────────────────────────────────────────────────── */
#define MAX_GPUS 4

typedef struct {
    char   name[128];
    double vram_gb;
    char   driver[64];
    int    is_discrete;     /* 1 = dedicated, 0 = integrated */
    int    has_cuda;
    int    has_rocm;
    int    has_metal;
} gpu_info_t;

typedef struct {
    int       valid;
    int       cpu_cores;
    int       cpu_threads;
    int       has_avx2;
    int       has_avx512;
    double    ram_gb;
    int       stars;        /* 1-5 */
    hw_tier_t tier;
    char      cpu_name[128];
    /* GPUs */
    int       gpu_count;
    gpu_info_t gpus[MAX_GPUS];
} hw_specs_t;

/* ── Model type ────────────────────────────────────────────────────── */
typedef enum {
    MTYPE_LLM = 0,       /* Text generation / chat */
    MTYPE_VISION,         /* Vision-language (image input) */
    MTYPE_CODE,           /* Code-specialized */
    MTYPE_VOICE,          /* Text-to-speech */
    MTYPE_STT,            /* Speech-to-text */
    MTYPE_COUNT
} model_type_t;

/* ── Dynamic model catalog entry ───────────────────────────────────── */
#define MAX_CATALOG 64

typedef struct {
    char   id[128];          /* HF repo id */
    char   short_name[32];   /* display: "Llama 3.2" */
    char   full_name[128];   /* full: "Llama-3.2-3B-Instruct" */
    char   family[32];       /* "Llama", "Mistral", "Qwen", etc */
    char   filename[256];
    char   url[512];
    char   quant[16];
    double size_mb;
    double ram_needed;
    double params_b;         /* 1.1, 3.0, 7.0, etc */
    int    context_len;
    int    has_vision;       /* supports image input */
    int    has_tools;        /* supports function calling */
    int    has_code;         /* optimized for code */
    int    has_voice;        /* TTS model */
    int    has_stt;          /* speech-to-text model */
    model_type_t type;       /* primary type */
    char   license[32];
    int    downloads;
    hw_tier_t min_tier;
} model_entry_t;

/* ── Path helpers (cli_ui.c) ──────────────────────────────────────── */
void   cli_get_fllm_dir(char* buf, int sz);
void   cli_get_models_dir(char* buf, int sz);
void   cli_get_pid_path(char* buf, int sz);
void   cli_get_specs_path(char* buf, int sz);
void   cli_get_log_path(char* buf, int sz);
void   cli_get_catalog_path(char* buf, int sz);
void   cli_ensure_dirs(void);
int    cli_file_exists(const char* path);

/* ── UI helpers (cli_ui.c) ────────────────────────────────────────── */
void   cli_color(int color);
void   cli_reset(void);
void   cli_cprint(const char* text, int color);
void   cli_banner(void);
void   cli_separator(void);
void   cli_clear(void);
void   cli_stars(int n);
const char* cli_tier_name(hw_tier_t t);
int    cli_prompt_yn(const char* msg);
int    cli_prompt_choice(int max);
double cli_time_sec(void);
void   cli_delay(int ms);
void   cli_print_slow(const char* text, int delay_ms);

/* ── Specs (cli_specs.c) ──────────────────────────────────────────── */
void   specs_detect(hw_specs_t* s);
void   specs_rate(hw_specs_t* s);
void   specs_save(const hw_specs_t* s);
int    specs_load(hw_specs_t* s);
void   specs_print(const hw_specs_t* s);
void   specs_run_interactive(void);

/* ── Catalog (cli_catalog.c) ──────────────────────────────────────── */
int    catalog_fetch(model_entry_t* out, int max, model_type_t type);
int    catalog_load_cache(model_entry_t* out, int max);
void   catalog_save_cache(const model_entry_t* entries, int count);
int    catalog_refresh(model_entry_t* out, int max, model_type_t type);
void catalog_browse(const model_entry_t* entries, int count, const hw_specs_t* specs, model_type_t type);
int  catalog_filter(const model_entry_t* entries, int count, model_type_t type,
                    double fmin, double fmax, int* idx_out, int max_idx);
int    catalog_select_and_download(model_entry_t* entries, int count, const hw_specs_t* specs);
const char* catalog_type_name(model_type_t t);

/* ── Download (cli_download.c) ────────────────────────────────────── */
int    download_file(const char* url, const char* dest);
int    download_model(const model_entry_t* m);
int    find_local_model(char* out, int sz);
int    list_local_models(char out[][512], int max);

/* ── Daemon (cli_daemon.c) ────────────────────────────────────────── */
void   net_init(void);
void   net_cleanup(void);
void   daemon_write_pid(void);
void   daemon_remove_pid(void);
int    daemon_read_pid(int* pid, int* port);
int    daemon_is_running(int* out_port);
int    daemon_send(int port, const char* cmd, char* resp, int rsz);
void   daemon_serve(const char* model_path);
void   daemon_start_detached(const char* model_path);
void   daemon_interactive(int port);
void   daemon_stop(void);

#endif /* CLI_H */
