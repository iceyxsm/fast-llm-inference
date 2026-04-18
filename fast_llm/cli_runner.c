/*
 * fllm - Fast LLM CLI
 *
 * Interactive CLI with:
 *   - Hardware spec detection & rating
 *   - Model catalog with smart recommendations
 *   - One-click model download
 *   - Background daemon with model in RAM
 *
 * Usage:
 *   fllm              Interactive CLI (starts daemon if needed)
 *   fllm -off         Stop background daemon
 *   fllm --status     Check daemon status
 *   fllm --bench      Synthetic benchmark
 *   fllm -h           Help
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <errno.h>
#include <ctype.h>

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
  #define MKDIR(p) _mkdir(p)
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
  #define MKDIR(p) mkdir(p, 0755)
  #define GETPID() getpid()
  #define SLEEP_MS(ms) usleep((ms)*1000)
#endif

#include "model_loader.h"
#include "dequantized_tensor.h"
#include "cpu_features.h"
