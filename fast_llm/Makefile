# Fast LLM - Pure C Makefile
# Minimal build - only working kernels

# Compiler
CC = gcc
CFLAGS = -O3 -Wall -Wextra -fPIC -fopenmp -ffast-math -mavx2 -mfma -Iinclude
LDFLAGS = -lm -fopenmp

# Detect OS
ifeq ($(OS),Windows_NT)
    EXE = .exe
    RM = del /Q
    LDFLAGS_EXTRA = -lws2_32 -lpsapi
else
    EXE =
    RM = rm -f
    LDFLAGS_EXTRA =
endif

# Core source files - only working kernels
SRCS = src/main.c \
       src/inference.c \
       src/gguf_loader.c \
       src/kernels/cpu_features.c \
       src/kernels/quant.c \
       src/kernels/matmul_asm_style.c \
       src/kernels/dequantized_tensor.c \
       src/kernels/matmul_avx2.c \
       src/kernels/activations_avx2.c

# Object files
OBJS = $(SRCS:.c=.o)

# Target binary
TARGET = fast_llm$(EXE)

# Default target
all: $(TARGET) cli_runner model_selector chat

# Main binary
$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
	@echo "Built: $(TARGET)"

# Pattern rule for object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# CLI tools - shared deps
CLI_DEPS = src/inference.o src/gguf_loader.o src/kernels/cpu_features.o src/kernels/quant.o src/kernels/matmul_asm_style.o src/kernels/dequantized_tensor.o src/kernels/matmul_avx2.o src/kernels/activations_avx2.o
CLI_FLAGS = -O3 -mavx2 -mfma -fopenmp -ffast-math -Iinclude -Icli

# fllm modular CLI sources
CLI_SRCS = cli/cli_main.c cli/cli_ui.c cli/cli_specs.c cli/cli_catalog.c cli/cli_download.c cli/cli_daemon.c cli/cli_chat.c

cli_runner: cli_runner.c $(CLI_DEPS)
	$(CC) $(CLI_FLAGS) -o cli_runner$(EXE) cli_runner.c $(CLI_DEPS) $(LDFLAGS) $(LDFLAGS_EXTRA)
	@echo "Built: cli_runner$(EXE)"

# fllm = modular CLI build
fllm: $(CLI_SRCS) $(CLI_DEPS)
	$(CC) $(CLI_FLAGS) -o fllm$(EXE) $(CLI_SRCS) $(CLI_DEPS) $(LDFLAGS) $(LDFLAGS_EXTRA)
	@echo "Built: fllm$(EXE)"

model_selector: model_selector.c
	$(CC) -O3 -Iinclude -o model_selector$(EXE) model_selector.c $(LDFLAGS)
	@echo "Built: model_selector$(EXE)"

chat: chat.c src/inference.o src/gguf_loader.o src/kernels/cpu_features.o src/kernels/quant.o src/kernels/matmul_asm_style.o src/kernels/dequantized_tensor.o src/kernels/matmul_avx2.o src/kernels/activations_avx2.o
	$(CC) -O3 -mavx2 -mfma -fopenmp -ffast-math -Iinclude -o chat$(EXE) chat.c src/inference.o src/gguf_loader.o src/kernels/cpu_features.o src/kernels/quant.o src/kernels/matmul_asm_style.o src/kernels/dequantized_tensor.o src/kernels/matmul_avx2.o src/kernels/activations_avx2.o $(LDFLAGS)
	@echo "Built: chat$(EXE) (with REAL inference)"

# Quick benchmark
bench_quick: bench_quick.c src/kernels/matmul_asm_style.o src/kernels/cpu_features.o src/kernels/dequantized_tensor.o src/kernels/activations_avx2.o
	$(CC) -O3 -mavx2 -mfma -fopenmp -ffast-math -Iinclude -o bench_quick$(EXE) bench_quick.c src/kernels/matmul_asm_style.o src/kernels/cpu_features.o src/kernels/dequantized_tensor.o src/kernels/activations_avx2.o $(LDFLAGS)

# Clean
clean:
	$(RM) $(OBJS) $(TARGET) cli_runner$(EXE) model_selector$(EXE) chat$(EXE)

# Test
test: $(TARGET)
	./$(TARGET) --benchmark --quant 2

# Print info
info:
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "Sources: $(SRCS)"

.PHONY: all clean test info cli_runner fllm model_selector chat
