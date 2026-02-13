# Fast LLM - Pure C Makefile
# No Python, no dependencies, single binary

# Compiler
CC = gcc
CFLAGS = -O3 -Wall -Wextra -fPIC -fopenmp -ffast-math -mavx2 -mfma -Iinclude
LDFLAGS = -lm -fopenmp

# Detect OS
ifeq ($(OS),Windows_NT)
    EXE = .exe
    RM = del /Q
else
    EXE =
    RM = rm -f
endif

# Feature detection
# Check for AVX-512 support at compile time
HAVE_AVX512 := $(shell echo | $(CC) -mavx512f -E - > /dev/null 2>&1 && echo 1 || echo 0)

# Source files - PURE C, NO PYTHON
SRCS = src/main.c \
       src/inference.c \
       src/gguf_loader.c \
       src/kernels/cpu_features.c \
       src/kernels/quant.c \
       src/kernels/matmul_scalar.c \
       src/kernels/matmul_best.c \
       src/kernels/matmul_asm_style.c \
       src/kernels/matmul_large_n.c \
       src/kernels/kernel_dispatch.c \
       src/kernels/matmul_amx.c \
       src/kernels/matmul_vnni.c \
       src/kernels/dequantized_tensor.c \
       src/kernels/matmul_avx2.c \
       src/kernels/ggml_quants.c \
       src/kernels/matmul_q4.c \
       src/kernels/matmul_q4_k_optimized.c \
       src/kernels/matmul_q4_k_avx2.c \
       src/kernels/matmul_prefetch_optimized.c \
       src/kernels/matmul_int8_vnni.c \
       src/kernels/fused_swiglu_rms.c \
       src/kernels/activations_avx2.c

# Add AVX-512 kernels if supported
ifeq ($(HAVE_AVX512),1)
    SRCS += src/kernels/matmul_avx512.c
    CFLAGS += -mavx512f -mavx512bw -mavx512vl -D__AVX512F__
    $(info AVX-512 support detected and enabled)
else
    $(info AVX-512 not supported by compiler, using scalar fallback)
endif

# Object files
OBJS = $(SRCS:.c=.o)

# Target binary
TARGET = fast_llm$(EXE)

# Default target
all: $(TARGET)

# Main binary - PURE C, NO PYTHON
$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LDFLAGS)
	@echo "Built: $(TARGET)"
	@echo "Size: $$(ls -lh $(TARGET) | awk '{print $$5}')"

# Pattern rule for object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Clean
clean:
	$(RM) $(OBJS) $(TARGET)

# Run benchmark
test: $(TARGET)
	./$(TARGET) --benchmark --quant 2

# Run benchmark with Q4
test-q4: $(TARGET)
	./$(TARGET) --benchmark --quant 4

# Run benchmark with Q8
test-q8: $(TARGET)
	./$(TARGET) --benchmark --quant 8

# 50 tok/sec benchmark
bench-50tok: src/kernels/matmul_asm_style.o src/kernels/matmul_large_n.o src/kernels/cpu_features.o src/kernels/quant.o src/kernels/dequantized_tensor.o src/kernels/matmul_avx2.o src/kernels/matmul_scalar.o
	$(CC) -O3 -mavx2 -mfma -fopenmp -ffast-math -Iinclude \
	    bench_50tok_final.c \
	    src/kernels/matmul_asm_style.o src/kernels/matmul_large_n.o \
	    src/kernels/cpu_features.o src/kernels/quant.o \
	    src/kernels/dequantized_tensor.o src/kernels/matmul_avx2.o \
	    src/kernels/matmul_scalar.o \
	    -o bench_50tok.exe -lm -fopenmp
	@echo "Built: bench_50tok.exe"

# Print info
info:
	@echo "Compiler: $(CC)"
	@echo "CFLAGS: $(CFLAGS)"
	@echo "HAVE_AVX512: $(HAVE_AVX512)"
	@echo "Target: $(TARGET)"
	@echo "Sources: $(SRCS)"

# Help
help:
	@echo "Fast LLM - Pure C Build System"
	@echo ""
	@echo "Targets:"
	@echo "  make          - Build the binary"
	@echo "  make test     - Run Q2 benchmark"
	@echo "  make test-q4  - Run Q4 benchmark"
	@echo "  make clean    - Remove build files"
	@echo "  make info     - Show build configuration"
	@echo ""
	@echo "Usage:"
	@echo "  ./fast_llm --benchmark --quant 2"
	@echo "  ./fast_llm --model model.gguf --prompt \"Hello\""

.PHONY: all clean test test-q4 test-q8 info help
