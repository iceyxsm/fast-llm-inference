@echo off
REM fllm installer for Windows (CMD / PowerShell)
setlocal enabledelayedexpansion

echo.
echo   ======================================
echo        fllm installer (Windows)
echo   ======================================
echo.

REM Find script directory
set "SCRIPT_DIR=%~dp0"
cd /d "%SCRIPT_DIR%"

REM Check for gcc
where gcc >nul 2>&1
if %errorlevel% neq 0 (
    echo   Error: gcc not found.
    echo   Install MinGW-w64 or WinLibs: https://winlibs.com
    exit /b 1
)

echo   Compiler: gcc
echo   OS: Windows
echo.

set CFLAGS=-O3 -Wall -Wextra -fPIC -fopenmp -ffast-math -mavx2 -mfma -Iinclude
set LDFLAGS=-lws2_32 -lm -lpsapi -fopenmp

echo   Compiling kernels...
gcc %CFLAGS% -c src/inference.c -o src/inference.o
gcc %CFLAGS% -c src/gguf_loader.c -o src/gguf_loader.o
gcc %CFLAGS% -c src/kernels/cpu_features.c -o src/kernels/cpu_features.o
gcc %CFLAGS% -c src/kernels/quant.c -o src/kernels/quant.o
gcc %CFLAGS% -c src/kernels/matmul_asm_style.c -o src/kernels/matmul_asm_style.o
gcc %CFLAGS% -c src/kernels/dequantized_tensor.c -o src/kernels/dequantized_tensor.o
gcc %CFLAGS% -c src/kernels/matmul_avx2.c -o src/kernels/matmul_avx2.o
gcc %CFLAGS% -c src/kernels/activations_avx2.c -o src/kernels/activations_avx2.o

echo   Linking fllm.exe...
gcc %CFLAGS% -Icli -o fllm.exe ^
    cli/cli_main.c cli/cli_ui.c cli/cli_specs.c ^
    cli/cli_catalog.c cli/cli_download.c cli/cli_daemon.c ^
    cli/cli_chat.c ^
    src/inference.o src/gguf_loader.o src/kernels/cpu_features.o ^
    src/kernels/quant.o src/kernels/matmul_asm_style.o ^
    src/kernels/dequantized_tensor.o src/kernels/matmul_avx2.o ^
    src/kernels/activations_avx2.o %LDFLAGS%

if %errorlevel% neq 0 (
    echo   Build failed!
    exit /b 1
)

echo   Build successful!
echo.

REM Install to user bin directory
set "INSTALL_DIR=%USERPROFILE%\bin"
if not exist "%INSTALL_DIR%" mkdir "%INSTALL_DIR%"
copy /y fllm.exe "%INSTALL_DIR%\fllm.exe" >nul
echo   Installed to: %INSTALL_DIR%\fllm.exe

REM Create data directory
if not exist "%USERPROFILE%\.fllm\models" mkdir "%USERPROFILE%\.fllm\models"

REM Check PATH
echo %PATH% | findstr /i "%USERPROFILE%\bin" >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo   Adding %INSTALL_DIR% to your user PATH...
    setx PATH "%PATH%;%INSTALL_DIR%" >nul 2>&1
    echo   PATH updated. Restart your terminal for it to take effect.
)

echo.
echo   Done! Type 'fllm' to start.
echo.
echo   Quick reference:
echo     fllm          Start (downloads model on first run)
echo     fllm -off     Stop background daemon
echo     fllm --bench  Synthetic benchmark
echo     fllm -h       Help
echo.
