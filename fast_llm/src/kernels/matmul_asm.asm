;;
;; Hand-tuned AVX2 Assembly Kernel for INT8 Matmul
;; matmul_asm_int8: C[M,N] = A[M,K] * B[N,K]^T
;;
;; Parameters (Windows x64 calling convention):
;;   RCX = A (float* input)
;;   RDX = B (int8_t* weights)
;;   R8  = C (float* output)
;;   R9  = M (rows)
;;   [RSP+40] = N (cols)
;;   [RSP+48] = K (inner dim)
;;   [RSP+56] = scales (float*)
;;
;; Strategy:
;; - Process 2 output rows at a time (register blocking)
;; - Process 32 K values at a time (AVX2 width)
;; - Prefetch next cache lines
;; - Minimize memory traffic

BITS 64
SECTION .text

global matmul_asm_int8

matmul_asm_int8:
    ; Save non-volatile registers
    push    rbx
    push    rbp
    push    rsi
    push    rdi
    push    r12
    push    r13
    push    r14
    push    r15
    
    ; Load parameters from stack
    mov     r10, [rsp+80]       ; N
    mov     r11, [rsp+88]       ; K
    mov     r12, [rsp+96]       ; scales
    
    ; Align K to 32 (we process 32 at a time)
    mov     r13, r11
    and     r13, -32            ; K_aligned = K & ~31
    
    ; Outer loop: iterate over output rows (N)
    xor     r14, r14            ; n = 0
.row_loop:
    cmp     r14, r10
    jge     .done
    
    ; Load scale for this output row
    movss   xmm15, [r12 + r14*4]
    shufps  xmm15, xmm15, 0     ; Broadcast scale to all 4 elements
    vinsertf128 ymm15, ymm15, xmm15, 1  ; Broadcast to full YMM
    
    ; Zero accumulators for this row
    vxorps  ymm0, ymm0, ymm0    ; Accumulator 0
    vxorps  ymm1, ymm1, ymm1    ; Accumulator 1
    vxorps  ymm2, ymm2, ymm2    ; Accumulator 2
    vxorps  ymm3, ymm3, ymm3    ; Accumulator 3
    
    ; Inner loop: iterate over K in chunks of 32
    xor     r15, r15            ; k = 0
.k_loop:
    cmp     r15, r13
    jge     .k_remainder
    
    ; Prefetch next iteration's data
    prefetcht0 [rcx + r15*4 + 256]
    prefetcht0 [rdx + r15 + 64]
    
    ; Load 32 floats from A (4 YMM registers)
    vmovups ymm4, [rcx + r15*4]         ; A[k+0:k+7]
    vmovups ymm5, [rcx + r15*4 + 32]    ; A[k+8:k+15]
    vmovups ymm6, [rcx + r15*4 + 64]    ; A[k+16:k+23]
    vmovups ymm7, [rcx + r15*4 + 96]    ; A[k+24:k+31]
    
    ; Load 32 int8 from B
    ; We need to dequantize: convert int8 -> int32 -> float
    vmovdqu xmm8, [rdx + r15]           ; Load 16 bytes
    vpmovsxbd ymm8, xmm8                ; Sign-extend to int32
    vcvtdq2ps ymm8, ymm8                ; Convert to float
    
    vmovdqu xmm9, [rdx + r15 + 16]      ; Next 16 bytes
    vpmovsxbd ymm9, xmm9
    vcvtdq2ps ymm9, ymm9
    
    ; FMA: accum += A * B
    vfmadd231ps ymm0, ymm4, ymm8
    vfmadd231ps ymm1, ymm5, ymm9
    
    ; Second half (16 more elements)
    vmovdqu xmm10, [rdx + r15 + 32]
    vpmovsxbd ymm10, xmm10
    vcvtdq2ps ymm10, ymm10
    
    vmovdqu xmm11, [rdx + r15 + 48]
    vpmovsxbd ymm11, xmm11
    vcvtdq2ps ymm11, ymm11
    
    vfmadd231ps ymm2, ymm6, ymm10
    vfmadd231ps ymm3, ymm7, ymm11
    
    add     r15, 32
    jmp     .k_loop
    
.k_remainder:
    ; Handle remaining K (scalar)
    mov     rax, r15            ; k
.rem_loop:
    cmp     rax, r11
    jge     .horizontal_sum
    
    ; Scalar load and multiply
    vbroadcastss ymm4, [rcx + rax*4]    ; A[k]
    movsx   ebx, byte [rdx + rax]       ; B[k]
    cvtsi2ss xmm8, ebx
    vbroadcastss ymm8, xmm8             ; Broadcast B[k]
    vfmadd231ps ymm0, ymm4, ymm8
    
    inc     rax
    jmp     .rem_loop
    
.horizontal_sum:
    ; Sum all 4 accumulators into ymm0
    vaddps  ymm0, ymm0, ymm1
    vaddps  ymm2, ymm2, ymm3
    vaddps  ymm0, ymm0, ymm2
    
    ; Horizontal sum of ymm0
    vextractf128 xmm1, ymm0, 1
    addps   xmm0, xmm0, xmm1
    movhlps xmm1, xmm0
    addps   xmm0, xmm0, xmm1
    shufps  xmm1, xmm0, 0x01
    addss   xmm0, xmm0, xmm1
    
    ; Apply scale
    mulss   xmm0, xmm15
    
    ; Store result
    mov     rdi, r8             ; C
    movss   [rdi + r14*4], xmm0 ; C[n]
    
    ; Next row
    inc     r14
    add     rdx, r11            ; B += K (next row)
    jmp     .row_loop
    
.done:
    ; Restore registers
    vzeroupper
    pop     r15
    pop     r14
    pop     r13
    pop     r12
    pop     rdi
    pop     rsi
    pop     rbp
    pop     rbx
    ret
