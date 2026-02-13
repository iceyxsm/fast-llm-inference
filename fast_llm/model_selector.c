/*
 * Model Selector CLI
 * Interactive model selection and benchmarking
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <windows.h>
#include <dirent.h>
#include <sys/stat.h>

#ifdef _WIN32
#include <malloc.h>
#define aligned_malloc(size, alignment) _aligned_malloc(size, alignment)
#define aligned_free(ptr) _aligned_free(ptr)
#else
#define aligned_malloc(size, alignment) aligned_alloc(alignment, size)
#define aligned_free(ptr) free(ptr)
#endif

#define MAX_MODELS 20
#define PATH_MAX_LEN 512

typedef struct {
    char name[256];
    char path[PATH_MAX_LEN];
    double size_mb;
    int is_q4;
    int is_q8;
    int layers;
} model_info_t;

/* Get file size in MB */
double get_file_size_mb(const char* path) {
    struct _stat64 st;
    if (_stat64(path, &st) == 0) {
        return (double)st.st_size / (1024.0 * 1024.0);
    }
    return 0.0;
}

/* Print colored text */
void color_printf(int color, const char* fmt, ...) {
    va_list args;
    HANDLE hConsole = GetStdHandle(STD_OUTPUT_HANDLE);
    SetConsoleTextAttribute(hConsole, color);
    
    va_start(args, fmt);
    vprintf(fmt, args);
    va_end(args);
    
    SetConsoleTextAttribute(hConsole, 7);
}

/* Clear screen */
void clear_screen(void) {
    system("cls");
}

/* Print banner */
void print_banner(void) {
    color_printf(11, "\n");
    color_printf(11, "========================================\n");
    color_printf(11, "       FAST LLM - MODEL SELECTOR        \n");
    color_printf(11, "========================================\n");
    color_printf(7, "\n");
}

/* Scan models directory */
int scan_models(const char* dir_path, model_info_t* models, int max_models) {
    DIR* dir = opendir(dir_path);
    if (!dir) return 0;
    
    struct dirent* entry;
    int count = 0;
    
    while ((entry = readdir(dir)) != NULL && count < max_models) {
        if (strstr(entry->d_name, ".gguf")) {
            strcpy(models[count].name, entry->d_name);
            snprintf(models[count].path, PATH_MAX_LEN, "%s/%s", dir_path, entry->d_name);
            models[count].size_mb = get_file_size_mb(models[count].path);
            
            /* Detect quantization type */
            models[count].is_q4 = strstr(entry->d_name, "q4") || strstr(entry->d_name, "Q4");
            models[count].is_q8 = strstr(entry->d_name, "q8") || strstr(entry->d_name, "Q8");
            
            /* Try to detect layers from filename */
            models[count].layers = 32; /* Default */
            
            count++;
        }
    }
    
    closedir(dir);
    return count;
}

/* Display model list */
void display_models(model_info_t* models, int count) {
    clear_screen();
    print_banner();
    
    color_printf(14, "Available Models:\n");
    color_printf(7, "-----------------\n\n");
    
    if (count == 0) {
        color_printf(12, "No .gguf models found!\n");
        color_printf(7, "\nPlease place models in the 'models/' folder.\n");
        return;
    }
    
    printf("%-4s %-40s %10s %8s\n", "No.", "Model Name", "Size", "Type");
    printf("%-4s %-40s %10s %8s\n", "---", "----------", "----", "----");
    
    for (int i = 0; i < count; i++) {
        const char* type = models[i].is_q4 ? "INT4" : models[i].is_q8 ? "INT8" : "FP16";
        int color = models[i].is_q4 ? 10 : models[i].is_q8 ? 14 : 7;
        
        color_printf(7, "[%2d] ", i + 1);
        color_printf(7, "%-40s ", models[i].name);
        color_printf(7, "%8.1f MB ", models[i].size_mb);
        color_printf(color, "%6s\n", type);
    }
    
    printf("\n");
}

/* Show model details */
void show_model_details(model_info_t* model) {
    clear_screen();
    print_banner();
    
    color_printf(14, "Model Details:\n");
    color_printf(7, "-------------\n\n");
    
    printf("Name:        %s\n", model->name);
    printf("Path:        %s\n", model->path);
    printf("Size:        %.2f MB (%.2f GB)\n", model->size_mb, model->size_mb / 1024.0);
    printf("Type:        %s\n", model->is_q4 ? "INT4 Quantized" : 
                              model->is_q8 ? "INT8 Quantized" : "Full Precision");
    printf("Layers:      %d (estimated)\n", model->layers);
    
    /* Estimate RAM usage */
    double ram_needed = model->size_mb * 1.5; /* 1.5x for working memory */
    printf("\nRAM Needed:  ");
    if (ram_needed < 4096) {
        color_printf(10, "%.0f MB ✓\n", ram_needed);
    } else if (ram_needed < 8192) {
        color_printf(14, "%.0f MB ~\n", ram_needed);
    } else {
        color_printf(12, "%.0f MB ⚠\n", ram_needed);
    }
    
    printf("\n");
}

/* Run selected model */
void run_model(model_info_t* model) {
    show_model_details(model);
    
    color_printf(11, "Starting benchmark...\n\n");
    
    /* Build command */
    char cmd[1024];
    
    /* Determine best layer count based on model type */
    int layers = 24; /* Default for speed */
    if (strstr(model->name, "mini") || strstr(model->name, "small")) {
        layers = 24;
    } else if (model->is_q4) {
        layers = 32; /* Can handle full with Q4 */
    }
    
    snprintf(cmd, sizeof(cmd), ".\\cli_runner.exe --layers %d --tokens 50", layers);
    
    color_printf(7, "Running: %s\n\n", cmd);
    system(cmd);
    
    printf("\n");
    color_printf(7, "Press Enter to continue...");
    getchar();
}

int main(int argc, char* argv[]) {
    model_info_t models[MAX_MODELS];
    int num_models = 0;
    
    /* Try to find models */
    num_models = scan_models("../models", models, MAX_MODELS);
    if (num_models == 0) {
        num_models = scan_models("models", models, MAX_MODELS);
    }
    
    if (argc > 1) {
        /* Direct mode - run specific model */
        int idx = atoi(argv[1]) - 1;
        if (idx >= 0 && idx < num_models) {
            run_model(&models[idx]);
            return 0;
        }
    }
    
    /* Interactive mode */
    while (1) {
        display_models(models, num_models);
        
        if (num_models == 0) {
            color_printf(7, "\nPress Enter to exit...");
            getchar();
            return 1;
        }
        
        color_printf(14, "Options:\n");
        color_printf(7, "  [1-%d] Select model to benchmark\n", num_models);
        color_printf(7, "  [r] Refresh model list\n");
        color_printf(7, "  [q] Quit\n");
        printf("\nChoice: ");
        fflush(stdout);
        
        char choice[10];
        if (!fgets(choice, sizeof(choice), stdin)) break;
        
        /* Remove newline */
        choice[strcspn(choice, "\n")] = 0;
        
        if (choice[0] == 'q' || choice[0] == 'Q') {
            break;
        } else if (choice[0] == 'r' || choice[0] == 'R') {
            num_models = scan_models("../models", models, MAX_MODELS);
            if (num_models == 0) {
                num_models = scan_models("models", models, MAX_MODELS);
            }
            continue;
        }
        
        int idx = atoi(choice) - 1;
        if (idx >= 0 && idx < num_models) {
            run_model(&models[idx]);
        } else {
            color_printf(12, "\nInvalid selection!\n");
            Sleep(1000);
        }
    }
    
    clear_screen();
    color_printf(10, "\nGoodbye!\n\n");
    return 0;
}
