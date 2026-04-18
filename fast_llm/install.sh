#!/usr/bin/env bash
# fllm installer - works on Linux, macOS, and Windows (Git Bash / MSYS2)
set -e

CYAN='\033[36m'
GREEN='\033[32m'
RED='\033[31m'
RESET='\033[0m'

echo ""
echo -e "${CYAN}╔══════════════════════════════════╗${RESET}"
echo -e "${CYAN}║     fllm installer               ║${RESET}"
echo -e "${CYAN}╚══════════════════════════════════╝${RESET}"
echo ""

# Find project root (where this script lives)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check for gcc
if ! command -v gcc &>/dev/null; then
    # Try mingw on Windows
    if command -v x86_64-w64-mingw32-gcc &>/dev/null; then
        CC=x86_64-w64-mingw32-gcc
    else
        echo -e "${RED}Error: gcc not found. Install gcc or mingw-w64.${RESET}"
        exit 1
    fi
else
    CC=gcc
fi

echo "  Compiler: $CC"

# Detect OS
EXE=""
INSTALL_DIR="/usr/local/bin"
LDEXTRA=""
case "$(uname -s 2>/dev/null || echo Windows)" in
    Linux*)
        OS="linux"
        LDEXTRA="-lm"
        ;;
    Darwin*)
        OS="macos"
        LDEXTRA="-lm"
        ;;
    MINGW*|MSYS*|CYGWIN*|Windows*)
        OS="windows"
        EXE=".exe"
        LDEXTRA="-lws2_32 -lm -lpsapi"
        INSTALL_DIR="$HOME/bin"
        ;;
esac

echo "  OS: $OS"
echo ""

# Build object files
echo "  Compiling kernels..."
CFLAGS="-O3 -Wall -fPIC -fopenmp -ffast-math -mavx2 -mfma -Iinclude"

$CC $CFLAGS -c src/inference.c -o src/inference.o
$CC $CFLAGS -c src/gguf_loader.c -o src/gguf_loader.o
$CC $CFLAGS -c src/kernels/cpu_features.c -o src/kernels/cpu_features.o
$CC $CFLAGS -c src/kernels/quant.c -o src/kernels/quant.o
$CC $CFLAGS -c src/kernels/matmul_asm_style.c -o src/kernels/matmul_asm_style.o
$CC $CFLAGS -c src/kernels/dequantized_tensor.c -o src/kernels/dequantized_tensor.o
$CC $CFLAGS -c src/kernels/matmul_avx2.c -o src/kernels/matmul_avx2.o
$CC $CFLAGS -c src/kernels/activations_avx2.c -o src/kernels/activations_avx2.o

echo "  Linking fllm..."
OBJ="src/inference.o src/gguf_loader.o src/kernels/cpu_features.o src/kernels/quant.o"
OBJ="$OBJ src/kernels/matmul_asm_style.o src/kernels/dequantized_tensor.o"
OBJ="$OBJ src/kernels/matmul_avx2.o src/kernels/activations_avx2.o"
CLI_SRCS="cli/cli_main.c cli/cli_ui.c cli/cli_specs.c cli/cli_catalog.c cli/cli_download.c cli/cli_daemon.c cli/cli_chat.c"

$CC $CFLAGS -Icli -o "fllm${EXE}" $CLI_SRCS $OBJ -fopenmp $LDEXTRA

echo -e "  ${GREEN}Build successful!${RESET}"
echo ""

# Install to PATH
if [ "$OS" = "windows" ]; then
    mkdir -p "$INSTALL_DIR"
    cp "fllm${EXE}" "$INSTALL_DIR/fllm${EXE}"
    echo "  Installed to: $INSTALL_DIR/fllm${EXE}"
    # Check if ~/bin is in PATH
    if [[ ":$PATH:" != *":$HOME/bin:"* ]]; then
        echo ""
        echo -e "  ${RED}Note:${RESET} Add $HOME/bin to your PATH:"
        echo "    export PATH=\"\$HOME/bin:\$PATH\""
        echo "  Or add to your shell profile (~/.bashrc or ~/.bash_profile)"
    fi
else
    if [ -w "$INSTALL_DIR" ]; then
        cp "fllm${EXE}" "$INSTALL_DIR/fllm"
        chmod +x "$INSTALL_DIR/fllm"
        echo "  Installed to: $INSTALL_DIR/fllm"
    else
        echo "  Installing to $INSTALL_DIR (needs sudo)..."
        sudo cp "fllm${EXE}" "$INSTALL_DIR/fllm"
        sudo chmod +x "$INSTALL_DIR/fllm"
        echo "  Installed to: $INSTALL_DIR/fllm"
    fi
fi

# Create ~/.fllm directory
mkdir -p "$HOME/.fllm/models"

echo ""
echo -e "  ${GREEN}Done! Type 'fllm' to start.${RESET}"
echo ""
echo "  Quick reference:"
echo "    fllm          Start interactive CLI"
echo "    fllm -off     Stop background daemon"
echo "    fllm --status Check daemon status"
echo "    fllm -h       Help"
echo ""
