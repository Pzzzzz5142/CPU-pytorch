#pragma once

#include <memory.h>
#include <iostream>
#define __AVX256__ 1
#include "simd.h"
#include <new>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <math.h>

#define A(i, j) calA[(i) + (j)*lda]
#define B(i, j) calB[(i)*ldb + (j)]
#define C(i, j) calC[(i)*newN + (j)]

#define oldA(i, j) a[(i)*K + (j)]
#define oldB(i, j) b[(i)*N + (j)]
#define oldC(i, j) c[(i)*N + (j)]

const size_t M_KERNEL_SIZE = 6;
const size_t N_KERNEL_SIZE = 16;
const size_t K_BLOCK_SIZE = 128;
const size_t M_BLOCK_SIZE = 384;
const size_t N_BLOCK_SIZE = 1024;
const size_t packA_SIZE = sizeof(float) * M_BLOCK_SIZE * K_BLOCK_SIZE / (1024.0);
const auto packB_SIZE = sizeof(float) * N_BLOCK_SIZE * K_BLOCK_SIZE / (1024);

const char *gemm_desc = "my mmul";

#include <stdio.h>
#include <immintrin.h> // AVX

typedef unsigned long long inc_t;
typedef unsigned long long dim_t;

auto addr_align = std::align_val_t(64);

void addDot_asm_6x16(
    size_t K, size_t newN, float *calA, int lda, float *calB, int ldb, float *calC, float *nextB)
{
    float *pointA = &A(0, 0), *pointB = &B(0, 0), *pointC = &C(0, 0);

    auto kc = K / 4;
    auto kl = K % 4;

    __asm__ volatile(
        "movq      %0,        %%rsi                \n\t" // kc (64 bit) stored in %rsi
        "movq      %1,        %%r9                 \n\t" // kl (64 bit) stored in %r9
        "movq      %2,        %%rax                \n\t" // Address of A stored in %rax
        "movq      %3,        %%rbx                \n\t" // Address of B stored in %rbx
        "movq      %4,        %%rcx                \n\t" // Address of C(0, 0) stored in %rcx
        "movq      %5,        %%rdx                \n\t" // newN stored in %rdx
        "movq      %6,        %%r10                \n\t" // Address of sum stored in %r10

        "leaq        (%%rcx, %%rdx, 8),  %%r8      \n\t"
        "leaq         (%%r8, %%rdx, 4),  %%r8      \n\t"

        //"prefetchnta  (%%rbx)                      \n\t"
        //"prefetchnta  512(%%rbx)                   \n\t"
        "vmovaps               (%%rcx),  %%ymm0    \n\t" // loading data from c to avx regs
        "vmovaps             32(%%rcx),  %%ymm1    \n\t"
        "vmovaps     (%%rcx, %%rdx, 4),  %%ymm2    \n\t"
        "vmovaps   32(%%rcx, %%rdx, 4),  %%ymm3    \n\t"
        "vmovaps     (%%rcx, %%rdx, 8),  %%ymm4    \n\t"
        "vmovaps   32(%%rcx, %%rdx, 8),  %%ymm5    \n\t"
        "vmovaps                (%%r8),  %%ymm6    \n\t"
        "vmovaps              32(%%r8),  %%ymm7    \n\t"
        "vmovaps      (%%r8, %%rdx, 4),  %%ymm8    \n\t"
        "vmovaps    32(%%r8, %%rdx, 4),  %%ymm9    \n\t"
        "vmovaps      (%%r8, %%rdx, 8),  %%ymm10   \n\t"
        "vmovaps    32(%%r8, %%rdx, 8),  %%ymm11   \n\t"

        //"vzeroall                                  \n\t"

        "testq     %%rsi,    %%rsi                 \n\t" // if kc==0 start kl loop
        "je        .DKLEFT%=                       \n\t"

        ".DLOOP%=:                                 \n\t"

        // update 1.
        "vmovaps    (%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps  32(%%rbx), %%ymm13               \n\t"

        "vbroadcastss   (%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss  4(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss  8(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 12(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 16(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 20(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        // update 2.
        "vmovaps  64(%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps  96(%%rbx), %%ymm13               \n\t"

        "vbroadcastss 24(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 28(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss 32(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 36(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 40(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 44(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        // update 3.
        //"prefetcht0 72*4(%%rax)                    \n\t"

        "vmovaps 128(%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps 160(%%rbx), %%ymm13               \n\t"

        "vbroadcastss 48(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 52(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss 56(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 60(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 64(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 68(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        // update 4.
        "vmovaps 192(%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps 224(%%rbx), %%ymm13               \n\t"

        "vbroadcastss 72(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 76(%%rax),  %%ymm15          \n\t"
        "subq            $-256,   %%rbx            \n\t" // pointB += 16*4
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss 80(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 84(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 88(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 92(%%rax),  %%ymm15           \n\t"
        "subq             $-96,   %%rax            \n\t" // pointA += 6*4
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        "decq      %%rsi                           \n\t"
        "jne       .DLOOP%=                        \n\t"

        ".DKLEFT%=:                                \n\t"

        "testq     %%r9,    %%r9                   \n\t" // if kl==0 start writeback to c
        "je        .DWRITEBACK%=                   \n\t"

        ".DLEFTLOOP%=:                             \n\t"

        //"prefetcht0 64*4(%%rax)                    \n\t"
        "vmovaps    (%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps  32(%%rbx), %%ymm13               \n\t"

        "vbroadcastss   (%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss  4(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss  8(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 12(%%rax),  %%ymm15          \n\t"
        "addq               $64,  %%rbx            \n\t" // pointB += 16
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 16(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 20(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        "addq     $24,       %%rax                 \n\t" // pointA += 6

        "decq      %%r9                            \n\t"
        "jne       .DLEFTLOOP%=                    \n\t"

        ".DWRITEBACK%=:                            \n\t"

        "vmovaps   %%ymm0,               (%%rcx)   \n\t" // storing data from avx regs to c
        "vmovaps   %%ymm1,             32(%%rcx)   \n\t" // I have tried vmovntps to directly save data to memory but not cache.
        "vmovaps   %%ymm2,     (%%rcx, %%rdx, 4)   \n\t" // However, since there are multiple adds and writes across the kernel, save data directly to memory will hurt the performance.
        "vmovaps   %%ymm3,   32(%%rcx, %%rdx, 4)   \n\t"
        "vmovaps   %%ymm4,     (%%rcx, %%rdx, 8)   \n\t"
        "vmovaps   %%ymm5,   32(%%rcx, %%rdx, 8)   \n\t"
        "vmovaps   %%ymm6,                (%%r8)   \n\t"
        "vmovaps   %%ymm7,              32(%%r8)   \n\t"
        "vmovaps   %%ymm8,      (%%r8, %%rdx, 4)   \n\t"
        "vmovaps   %%ymm9,    32(%%r8, %%rdx, 4)   \n\t"
        "vmovaps  %%ymm10,      (%%r8, %%rdx, 8)   \n\t"
        "vmovaps  %%ymm11,    32(%%r8, %%rdx, 8)   \n\t"
        //"prefetchnta        (%%r10)                \n\t"

        :            // output
        :            // input
        "m"(kc),     // 0
        "m"(kl),     // 1
        "m"(pointA), // 2
        "m"(pointB), // 3
        "m"(pointC), // 4
        "m"(newN),   // 5
        "m"(nextB)
        : // register clobber list
        "rax", "rbx", "rcx", "rdx", "rsi", "r8", "r9", "r10",
        "ymm0", "ymm1", "ymm2", "ymm3",
        "ymm4", "ymm5", "ymm6", "ymm7",
        "ymm8", "ymm9", "ymm10", "ymm11",
        "ymm12", "ymm13", "ymm14", "ymm15");
}

float *addDotReduce_asm_6x16(
    size_t K, size_t newN, float *calA, int lda, float *calB, int ldb, float *calC, float *nextB)
{
    float *pointA = &A(0, 0), *pointB = &B(0, 0), *pointC = &C(0, 0);

    auto kc = K / 4;
    auto kl = K % 4;
    static float total_sum[12] __attribute__((aligned(64)));
    float *p = total_sum;

    __asm__ volatile(
        "movq      %0,        %%rsi                \n\t" // kc (64 bit) stored in %rsi
        "movq      %1,        %%r9                 \n\t" // kl (64 bit) stored in %r9
        "movq      %2,        %%rax                \n\t" // Address of A stored in %rax
        "movq      %3,        %%rbx                \n\t" // Address of B stored in %rbx
        "movq      %4,        %%rcx                \n\t" // Address of C(0, 0) stored in %rcx
        "movq      %5,        %%rdx                \n\t" // newN stored in %rdx
        "movq      %6,        %%r10                \n\t" // Address of total_sum stored in %r10

        "leaq        (%%rcx, %%rdx, 8),  %%r8      \n\t"
        "leaq         (%%r8, %%rdx, 4),  %%r8      \n\t"

        "vmovaps               (%%rcx),  %%ymm0    \n\t" // loading data from c to avx regs
        "vmovaps             32(%%rcx),  %%ymm1    \n\t"
        "vmovaps     (%%rcx, %%rdx, 4),  %%ymm2    \n\t"
        "vmovaps   32(%%rcx, %%rdx, 4),  %%ymm3    \n\t"
        "vmovaps     (%%rcx, %%rdx, 8),  %%ymm4    \n\t"
        "vmovaps   32(%%rcx, %%rdx, 8),  %%ymm5    \n\t"
        "vmovaps                (%%r8),  %%ymm6    \n\t"
        "vmovaps              32(%%r8),  %%ymm7    \n\t"
        "vmovaps      (%%r8, %%rdx, 4),  %%ymm8    \n\t"
        "vmovaps    32(%%r8, %%rdx, 4),  %%ymm9    \n\t"
        "vmovaps      (%%r8, %%rdx, 8),  %%ymm10   \n\t"
        "vmovaps    32(%%r8, %%rdx, 8),  %%ymm11   \n\t"

        //"vzeroall                                  \n\t"

        "testq     %%rsi,    %%rsi                 \n\t" // if kc==0 start kl loop
        "je        .DKLEFT%=                       \n\t"

        ".DLOOP%=:                                 \n\t"

        // update 1.
        //"prefetcht0 64*4(%%rax)                    \n\t"
        "vmovaps    (%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps  32(%%rbx), %%ymm13               \n\t"

        "vbroadcastss   (%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss  4(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss  8(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 12(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 16(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 20(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        // update 2.
        "vmovaps  64(%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps  96(%%rbx), %%ymm13               \n\t"

        "vbroadcastss 24(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 28(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss 32(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 36(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 40(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 44(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        // update 3.
        //"prefetcht0 72*4(%%rax)                    \n\t"

        "vmovaps 128(%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps 160(%%rbx), %%ymm13               \n\t"

        "vbroadcastss 48(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 52(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss 56(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 60(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 64(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 68(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        // update 4.
        "vmovaps 192(%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps 224(%%rbx), %%ymm13               \n\t"

        "vbroadcastss 72(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 76(%%rax),  %%ymm15          \n\t"
        "subq            $-256,   %%rbx            \n\t" // pointB += 16*4
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss 80(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 84(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 88(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 92(%%rax),  %%ymm15           \n\t"
        "subq             $-96,   %%rax            \n\t" // pointA += 6*4
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        "decq      %%rsi                           \n\t"
        "jne       .DLOOP%=                        \n\t"

        ".DKLEFT%=:                                \n\t"

        "testq     %%r9,    %%r9                   \n\t" // if kl==0 start writeback to c
        "je        .DWRITEBACK%=                   \n\t"

        ".DLEFTLOOP%=:                             \n\t"

        //"prefetcht0 64*4(%%rax)                    \n\t"
        "vmovaps    (%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps  32(%%rbx), %%ymm13               \n\t"

        "vbroadcastss   (%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss  4(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm0   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm1   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm2   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm3   \n\t"

        "vbroadcastss  8(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 12(%%rax),  %%ymm15          \n\t"
        "addq               $64,  %%rbx            \n\t" // pointB += 16
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm4   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm5   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm6   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm7   \n\t"

        "vbroadcastss 16(%%rax),  %%ymm14          \n\t" // loading data from a to avx reg
        "vbroadcastss 20(%%rax),  %%ymm15          \n\t"
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        "addq     $24,       %%rax                 \n\t" // pointA += 6

        "decq      %%r9                            \n\t"
        "jne       .DLEFTLOOP%=                    \n\t"

        ".DWRITEBACK%=:                            \n\t"

        "vmovaps   %%ymm0,               (%%rcx)   \n\t" // storing data from avx regs to c
        "vmovaps   %%ymm1,             32(%%rcx)   \n\t" // I have tried vmovntps to directly save data to memory but not cache.
        "vmovaps   %%ymm2,     (%%rcx, %%rdx, 4)   \n\t" // However, since there are multiple adds and writes across the kernel, save data directly to memory will hurt the performance.
        "vmovaps   %%ymm3,   32(%%rcx, %%rdx, 4)   \n\t"
        "vmovaps   %%ymm4,     (%%rcx, %%rdx, 8)   \n\t"
        "vmovaps   %%ymm5,   32(%%rcx, %%rdx, 8)   \n\t"
        "vmovaps   %%ymm6,                (%%r8)   \n\t"
        "vmovaps   %%ymm7,              32(%%r8)   \n\t"
        "vmovaps   %%ymm8,      (%%r8, %%rdx, 4)   \n\t"
        "vmovaps   %%ymm9,    32(%%r8, %%rdx, 4)   \n\t"
        "vmovaps  %%ymm10,      (%%r8, %%rdx, 8)   \n\t"
        "vmovaps  %%ymm11,    32(%%r8, %%rdx, 8)   \n\t"

        "vmulps    %%ymm0,    %%ymm0,   %%ymm12    \n\t"
        "vmulps    %%ymm1,    %%ymm1,   %%ymm13    \n\t"
        "vmulps    %%ymm2,    %%ymm2,   %%ymm14    \n\t"
        "vmulps    %%ymm3,    %%ymm3,   %%ymm15    \n\t"
        "vaddps    %%ymm8,    %%ymm9,    %%ymm8    \n\t"
        "vaddps   %%ymm10,   %%ymm11,   %%ymm10    \n\t"
        "vaddps    %%ymm0,    %%ymm1,    %%ymm0    \n\t"
        "vaddps    %%ymm2,    %%ymm3,    %%ymm2    \n\t"
        "vaddps    %%ymm4,    %%ymm5,    %%ymm4    \n\t"
        "vaddps    %%ymm6,    %%ymm7,    %%ymm6    \n\t"
        "vaddps   %%ymm12,   %%ymm13,   %%ymm12    \n\t"
        "vaddps   %%ymm14,   %%ymm15,   %%ymm14    \n\t"

        "vextractf128   $1,    %%ymm0,   %%xmm1    \n\t"
        "vextractf128   $1,    %%ymm2,   %%xmm3    \n\t"
        "vextractf128   $1,    %%ymm4,   %%xmm5    \n\t"
        "vextractf128   $1,    %%ymm6,   %%xmm7    \n\t"
        "vextractf128   $1,    %%ymm8,   %%xmm9    \n\t"
        "vextractf128   $1,   %%ymm10,  %%xmm11    \n\t"
        "vextractf128   $1,   %%ymm12,  %%xmm13    \n\t"
        "vextractf128   $1,   %%ymm14,  %%xmm15    \n\t" // no free mm

        "addps      %%xmm1,    %%xmm0              \n\t"
        "addps      %%xmm3,    %%xmm2              \n\t"
        "addps      %%xmm5,    %%xmm4              \n\t"
        "addps      %%xmm7,    %%xmm6              \n\t"
        "addps      %%xmm9,    %%xmm8              \n\t"
        "addps     %%xmm11,   %%xmm10              \n\t"
        "addps     %%xmm13,   %%xmm12              \n\t"
        "addps     %%xmm15,   %%xmm14              \n\t"

        "vshufps  $0x4e,  %%xmm2,  %%xmm0,  %%xmm1 \n\t"
        "vshufps  $0xe4,  %%xmm2,  %%xmm0,  %%xmm0 \n\t"
        "vshufps  $0x4e,  %%xmm6,  %%xmm4,  %%xmm5 \n\t"
        "vshufps  $0xe4,  %%xmm6,  %%xmm4,  %%xmm4 \n\t"
        "vshufps  $0x4e, %%xmm10,  %%xmm8,  %%xmm9 \n\t"
        "vshufps  $0xe4, %%xmm10,  %%xmm8,  %%xmm8 \n\t"
        "vshufps  $0x4e, %%xmm14, %%xmm12, %%xmm13 \n\t"
        "vshufps  $0xe4, %%xmm14, %%xmm12, %%xmm12 \n\t" // free mm: 2 3 6 7 10 11 14 15

        "vmovaps    (%%rcx, %%rdx, 8),  %%ymm2     \n\t"
        "vmovaps  32(%%rcx, %%rdx, 8),  %%ymm3     \n\t"
        "vmovaps               (%%r8),  %%ymm6     \n\t"
        "vmovaps             32(%%r8),  %%ymm7     \n\t"
        "vmovaps     (%%r8, %%rdx, 4), %%ymm10     \n\t"
        "vmovaps   32(%%r8, %%rdx, 4), %%ymm11     \n\t"
        "vmovaps     (%%r8, %%rdx, 8), %%ymm14     \n\t"
        "vmovaps   32(%%r8, %%rdx, 8), %%ymm15     \n\t"

        "vmulps    %%ymm2,    %%ymm2,    %%ymm2    \n\t"
        "vmulps    %%ymm3,    %%ymm3,    %%ymm3    \n\t"
        "vmulps    %%ymm6,    %%ymm6,    %%ymm6    \n\t"
        "vmulps    %%ymm7,    %%ymm7,    %%ymm7    \n\t"
        "vmulps   %%ymm10,   %%ymm10,   %%ymm10    \n\t"
        "vmulps   %%ymm11,   %%ymm11,   %%ymm11    \n\t"
        "vmulps   %%ymm14,   %%ymm14,   %%ymm14    \n\t"
        "vmulps   %%ymm15,   %%ymm15,   %%ymm15    \n\t"

        "addps      %%xmm1,    %%xmm0              \n\t"
        "addps      %%xmm5,    %%xmm4              \n\t"
        "vaddps     %%ymm3,    %%ymm2,   %%ymm2    \n\t"
        "vaddps     %%ymm7,    %%ymm6,   %%ymm6    \n\t"
        "vaddps    %%ymm11,   %%ymm10,  %%ymm10    \n\t"
        "vaddps    %%ymm15,   %%ymm14,  %%ymm14    \n\t"
        "addps      %%xmm9,    %%xmm8              \n\t"
        "addps     %%xmm13,   %%xmm12              \n\t"

        "vshufps  $0x88,  %%xmm4,  %%xmm0,  %%xmm1 \n\t"
        "vshufps  $0xdd,  %%xmm4,  %%xmm0,  %%xmm0 \n\t"
        "vshufps  $0x88, %%xmm12,  %%xmm8,  %%xmm9 \n\t"
        "vshufps  $0xdd, %%xmm12,  %%xmm8,  %%xmm8 \n\t"

        "vextractf128   $1,    %%ymm2,   %%xmm3    \n\t"
        "vextractf128   $1,    %%ymm6,   %%xmm7    \n\t"
        "vextractf128   $1,   %%ymm10,  %%xmm11    \n\t"
        "vextractf128   $1,   %%ymm14,  %%xmm15    \n\t"

        "addps      %%xmm3,    %%xmm2              \n\t"
        "addps      %%xmm7,    %%xmm6              \n\t"
        "addps     %%xmm11,   %%xmm10              \n\t"
        "addps     %%xmm15,   %%xmm14              \n\t"

        "addps      %%xmm1,    %%xmm0              \n\t"
        "addps      %%xmm9,    %%xmm8              \n\t"

        "vshufps  $0x4e,  %%xmm6,  %%xmm2,  %%xmm3 \n\t"
        "vshufps  $0xe4,  %%xmm6,  %%xmm2,  %%xmm2 \n\t"
        "vshufps  $0x4e, %%xmm14, %%xmm10, %%xmm11 \n\t"
        "vshufps  $0xe4, %%xmm14, %%xmm10, %%xmm10 \n\t"

        "addps      %%xmm3,    %%xmm2              \n\t"
        "addps     %%xmm11,   %%xmm10              \n\t"

        "vshufps  $0x88, %%xmm10,  %%xmm2, %%xmm11 \n\t"
        "vshufps  $0xdd, %%xmm10,  %%xmm2,  %%xmm2 \n\t"

        "addps     %%xmm11,    %%xmm2              \n\t"

        "movaps     %%xmm0,    (%%r10)             \n\t"
        "movaps     %%xmm8,  16(%%r10)             \n\t"
        "movaps     %%xmm2,  32(%%r10)             \n\t"

        :            // output
        :            // input
        "m"(kc),     // 0
        "m"(kl),     // 1
        "m"(pointA), // 2
        "m"(pointB), // 3
        "m"(pointC), // 4
        "m"(newN),   // 5
        "m"(p)       // 6
        :            // register clobber list
        "rax", "rbx", "rcx", "rdx", "rsi", "r8", "r9", "r10",
        "ymm0", "ymm1", "ymm2", "ymm3",
        "ymm4", "ymm5", "ymm6", "ymm7",
        "ymm8", "ymm9", "ymm10", "ymm11",
        "ymm12", "ymm13", "ymm14", "ymm15");

    return p;
}

void addDot(int K, int newN, float *calA, int lda, float *calB, int ldb, float *calC)
{
    AVX_Data avx_c[M_KERNEL_SIZE * 2];
    float *point = &C(0, 0);
#pragma unroll
    for (size_t i = 0; i < M_KERNEL_SIZE; i++)
        simd_load<2>(avx_c + i * 2, point + i * newN, false);
    float *pointB = &B(0, 0);

#pragma unroll(4)
    for (int k = 0; k < K; k++)
    {
        AVX_Data avx_b[2];
        simd_load<2>(avx_b, pointB + k * ldb, false);
#pragma unroll
        for (size_t i = 0; i < M_KERNEL_SIZE; i++)
        {
            AVX_Data avx_a;
            avx_a.data = SIMD_SET(A(i, k));
            simd_fma<2>(avx_c + i * 2, avx_a, avx_b, avx_c + i * 2);
        }
    }
#pragma unroll
    for (size_t i = 0; i < M_KERNEL_SIZE; i++)
        simd_store<2>(&C(i, 0), avx_c + i * 2, false);
}

inline void packCol(float *calA, int lda, size_t I, size_t J, float *to)
{
    for (int j = 0; j < J; j++)
#pragma unroll
        for (int i = 0; i < I; i++)
        {
            *to = A(i, j);
            to++;
        }
}

inline void packRow(float *calB, int ldb, size_t I, size_t J, float *to)
{
    for (int i = 0; i < I; i++)
#pragma unroll
        for (int j = 0; j < J; j++)
        {
            *to = B(i, j);
            to++;
        }
}

inline void packCol(float *calB, int ldb, size_t I, size_t IBound, size_t J, size_t JBound, float *to)
{
    IBound = std::min(IBound, I);
    JBound = std::min(JBound, J);
    _mm_prefetch(&B(0, 0), _MM_HINT_T0);
    for (int i = 0; i < IBound - 1; i++)
    {
        if (i % 32 == 0)
            _mm_prefetch(&B(i + 32, 0), _MM_HINT_T0);
#pragma unroll
        for (int j = 0; j < JBound; j++)
        {
            to[i + j * I] = B(i, j);
        }
    }
    for (int j = 0; j < JBound; j++)
    {
        to[IBound - 1 + j * I] = B(IBound - 1, j);
    }
}

inline void packRow(float *calB, int ldb, size_t I, size_t IBound, size_t J, size_t JBound, float *to)
{
    IBound = std::min(IBound, I);
    JBound = std::min(JBound, J);
    for (int i = 0; i < IBound; i++)
    {
        for (int j = 0; j < JBound; j += 32)
        {
            _mm_prefetch(&B(i, j), _MM_HINT_NTA);
        }
    }
    for (int i = 0; i < IBound; i++)
    {
        for (int j = 0; j < JBound; j++)
            *(to + i * J + j) = B(i, j);
        for (int j = JBound; j < J; j++)
            *(to + i * J + j) = 0;
    }
}

void inner_kernal(int m, int n, int k, int newN, float *pointA, float *pointB, float *calC)
{
    for (int i = 0; i < m; i += M_KERNEL_SIZE)
    {
        //#pragma omp parallel for
        for (int j = 0; j < n; j += N_KERNEL_SIZE)
        {
            // for (int p = 0; p < N_KERNEL_SIZE * k; p += 8)
            //{
            //     _mm_prefetch(pointB + ((j + N_KERNEL_SIZE) * k) + p, _MM_HINT_NTA);
            // }
            addDot_asm_6x16(k, newN, pointA, M_KERNEL_SIZE, pointB + (j * k), N_KERNEL_SIZE, &C(i, j), pointB + ((j + N_KERNEL_SIZE) * k));
        }
        //_mm_prefetch(pointB, _MM_HINT_NTA);
        pointA += M_KERNEL_SIZE * k;
    }
}

float *inner_reduce_kernal(int m, int n, int k, int newN, float *pointA, float *pointB, float *calC)
{
    static float res[2 * M_BLOCK_SIZE] __attribute__((aligned(64)));
    memset(res, 0, sizeof(res));
    float *p = res;
    for (int i = 0; i < m; i += M_KERNEL_SIZE)
    {
        //#pragma omp parallel for
        for (int j = 0; j < n; j += N_KERNEL_SIZE)
        {
            auto tmp = addDotReduce_asm_6x16(k, newN, pointA, M_KERNEL_SIZE, pointB + (j * k), N_KERNEL_SIZE, &C(i, j), pointB + ((j + N_KERNEL_SIZE) * k));
            for (int kk = 0; kk < 6; kk++)
            {
                p[kk] += tmp[kk];
                p[M_BLOCK_SIZE + kk] += tmp[kk + 6];
            }
        }
        p += M_KERNEL_SIZE;
        pointA += M_KERNEL_SIZE * k;
    }
    return res;
}

void globalPackingA(size_t m, size_t k, size_t K, size_t MBound, float *a, size_t lda, float *newA)
{
    for (int i = 0; i < m; i += M_KERNEL_SIZE)
    {
        packCol(&oldA(i, 0), lda, M_KERNEL_SIZE, MBound - i, k, k, newA);
        newA += M_KERNEL_SIZE * k;
    }
}

void globalPackingB(size_t n, size_t k, size_t N, size_t NBound, float *b, size_t ldb, float *newB)
{
    for (int i = 0; i < n; i += N_KERNEL_SIZE)
    {
        packRow(&oldB(0, i), ldb, k, k, N_KERNEL_SIZE, NBound - i, newB);
        newB += N_KERNEL_SIZE * k;
    }
}

void square_gemm(int M, int N, int K, float *a, float *b, float *c, bool bias = false)
{
    float *calA = a, *calB = b, *calC = c;

    size_t padM = (M_KERNEL_SIZE - (M % M_KERNEL_SIZE)) % M_KERNEL_SIZE, padN = (N_KERNEL_SIZE - (N % N_KERNEL_SIZE)) % N_KERNEL_SIZE;
    size_t newM = M + padM, newN = N + padN, newK = K;

    calA = new (addr_align) float[newM * newK];
    calB = new (addr_align) float[newN * newK];
    calC = new (addr_align) float[newM * newN];
    memset(calA, 0, sizeof(float) * newM * newK);
    memset(calB, 0, sizeof(float) * newN * newK);
    memset(calC, 0, sizeof(float) * newM * newN);

    auto pointA = calA, pointB = calB;

    for (int k = 0; k < K; k += K_BLOCK_SIZE)
    {
        auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
        for (int m = 0; m < newM; m += M_BLOCK_SIZE)
        {
            auto im = M_BLOCK_SIZE < newM - m ? M_BLOCK_SIZE : newM - m;
            globalPackingA(im, ik, K, M - m, &oldA(m, k), K, pointA);
            pointA += im * ik;
        }
        for (int n = 0; n < newN; n += N_BLOCK_SIZE)
        {
            auto in = N_BLOCK_SIZE < newN - n ? N_BLOCK_SIZE : newN - n;
            globalPackingB(in, ik, N, N - n, &oldB(k, n), N, pointB);
            pointB += in * ik;
        }
    }

    for (int k = 0; k < K; k += K_BLOCK_SIZE)
    {
        // auto calC = new (addr_align) float[newM * newN];
        // memset(calC, 0, sizeof(float) * newM * newN);
        auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
        for (int n = 0; n < newN; n += N_BLOCK_SIZE)
        {
            auto in = N_BLOCK_SIZE < newN - n ? N_BLOCK_SIZE : newN - n;
            for (int m = 0; m < newM; m += M_BLOCK_SIZE)
            {
                auto im = M_BLOCK_SIZE < newM - m ? M_BLOCK_SIZE : newM - m;
                inner_reduce_kernal(im, in, ik, newN, calA + k * newM + m * ik, calB + k * newN + n * ik, &C(m, n));
            }
        }
    }

    float *tmp = c;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // printf("%f", C(i, j));
            if (bias)
                *tmp += C(i, j);
            else
                *tmp = C(i, j);
            tmp++;
        }
    }
    ::operator delete[](calA, addr_align);
    ::operator delete[](calB, addr_align);
    ::operator delete[](calC, addr_align);
}

void gemm_compute(int M, int N, int K, float *a, float *b, float *c, bool bias = false)
{

    float *calA = a, *calB = b, *calC = c;

    size_t padM = (M_KERNEL_SIZE - (M % M_KERNEL_SIZE)) % M_KERNEL_SIZE, padN = (N_KERNEL_SIZE - (N % N_KERNEL_SIZE)) % N_KERNEL_SIZE;
    size_t newM = M + padM, newN = N + padN;

    calA = new (addr_align) float[newM * K];
    calC = new (addr_align) float[newM * newN];
    memset(calA, 0, sizeof(float) * newM * K);
    memset(calC, 0, sizeof(float) * newM * newN);
    auto pointA = calA, pointB = calB;

    for (int k = 0; k < K; k += K_BLOCK_SIZE)
    {
        auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
        for (int m = 0; m < newM; m += M_BLOCK_SIZE)
        {
            auto im = M_BLOCK_SIZE < newM - m ? M_BLOCK_SIZE : newM - m;
            globalPackingA(im, ik, K, M, &oldA(m, k), K, pointA);
            pointA += im * ik;
        }
    }
    //#pragma omp parallel for
    for (int k = 0; k < K; k += K_BLOCK_SIZE)
    {
        auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
        for (int n = 0; n < newN; n += N_BLOCK_SIZE)
        {
            auto in = N_BLOCK_SIZE < newN - n ? N_BLOCK_SIZE : newN - n;
            for (int m = 0; m < newM; m += M_BLOCK_SIZE)
            {
                auto im = M_BLOCK_SIZE < newM - m ? M_BLOCK_SIZE : newM - m;
                inner_kernal(im, in, ik, newN, calA + k * newM + m * ik, calB + k * newN + n * ik, &C(m, n));
            }
        }
    }

    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            if (bias)
                c[i * N + j] += C(i, j);
            else
                c[i * N + j] = C(i, j);
        }
    }

    ::operator delete[](calC, addr_align);
    ::operator delete[](calA, addr_align);
}

float *packing(int N, int K, float *b, int ldb)
{
    size_t padN = (N_KERNEL_SIZE - (N % N_KERNEL_SIZE)) % N_KERNEL_SIZE;
    size_t newN = N + padN, newK = K;
    auto res = new (addr_align) float[newN * newK];
    memset(res, 0, sizeof(float) * newN * newK);
    auto pointB = res;

    for (int k = 0; k < K; k += K_BLOCK_SIZE)
    {
        auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
        for (int n = 0; n < newN; n += N_BLOCK_SIZE)
        {
            auto in = N_BLOCK_SIZE < newN - n ? N_BLOCK_SIZE : newN - n;
            globalPackingB(in, ik, N, N - n, &oldB(k, n), N, pointB);
            pointB += in * ik;
        }
    }
    return res;
}

void gemm_layernorm_compute_sum(int M, int N, int K, float *a, float *b, float *c, float *gamma, float *beta, const bool &bias = false, float eps = 1e-5)
{
    float *calA = a, *calB = b, *calC = c;

    size_t padM = (M_KERNEL_SIZE - (M % M_KERNEL_SIZE)) % M_KERNEL_SIZE, padN = (N_KERNEL_SIZE - (N % N_KERNEL_SIZE)) % N_KERNEL_SIZE;
    size_t newM = M + padM, newN = N + padN;

    calA = new (addr_align) float[newM * K];
    calC = new (addr_align) float[newM * newN];
    memset(calA, 0, sizeof(float) * newM * K);
    memset(calC, 0, sizeof(float) * newM * newN);
    auto pointA = calA, pointB = calB;

    for (int k = 0; k < K; k += K_BLOCK_SIZE)
    {
        auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
        for (int m = 0; m < newM; m += M_BLOCK_SIZE)
        {
            auto im = M_BLOCK_SIZE < newM - m ? M_BLOCK_SIZE : newM - m;
            globalPackingA(im, ik, K, M, &oldA(m, k), K, pointA);
            pointA += im * ik;
        }
    }
    float sum[2 * newM];
    memset(sum, 0, sizeof(sum));
    if (bias)
        for (int i = 0; i < M; i++)
            for (int j = 0; j < N; j++)
                C(i, j) = oldC(i, j);
    int k = 0;
    //#pragma omp parallel for
    if (K > K_BLOCK_SIZE)
        for (k = 0; k < K - K_BLOCK_SIZE; k += K_BLOCK_SIZE)
        {
            auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
            for (int n = 0; n < newN; n += N_BLOCK_SIZE)
            {
                auto in = N_BLOCK_SIZE < newN - n ? N_BLOCK_SIZE : newN - n;
                for (int m = 0; m < newM; m += M_BLOCK_SIZE)
                {
                    auto im = M_BLOCK_SIZE < newM - m ? M_BLOCK_SIZE : newM - m;
                    inner_kernal(im, in, ik, newN, calA + k * newM + m * ik, calB + k * newN + n * ik, &C(m, n));
                }
            }
        }

    auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
    for (int n = 0; n < newN; n += N_BLOCK_SIZE)
    {
        auto in = N_BLOCK_SIZE < newN - n ? N_BLOCK_SIZE : newN - n;
        for (int m = 0; m < newM; m += M_BLOCK_SIZE)
        {
            auto im = M_BLOCK_SIZE < newM - m ? M_BLOCK_SIZE : newM - m;
            auto s = inner_reduce_kernal(im, in, ik, newN, calA + k * newM + m * ik, calB + k * newN + n * ik, &C(m, n));
            for (int kk = 0; kk < im; kk++)
            {
                sum[m + kk] += s[kk];
                sum[M + m + kk] += s[kk + M_BLOCK_SIZE];
            }
        }
    }

    for (int i = 0; i < M; i++)
    {
        sum[i] /= N;
        sum[i + M] /= N;
        for (int j = 0; j < N; j++)
        {
            c[i * N + j] = gamma[j] * ((C(i, j) - sum[i]) / sqrt(sum[i + M] - sum[i] * sum[i] + eps)) + beta[j];
        }
    }

    ::operator delete[](calC, addr_align);
    ::operator delete[](calA, addr_align);
}

void free_packing(float *Bp)
{
    ::operator delete[](Bp, addr_align);
}

std::vector<size_t> get_packed_size(size_t M, size_t N, size_t K)
{
    size_t padM = (M_KERNEL_SIZE - (M % M_KERNEL_SIZE)) % M_KERNEL_SIZE, padN = (N_KERNEL_SIZE - (N % N_KERNEL_SIZE)) % N_KERNEL_SIZE;
    size_t newM = M + padM, newN = N + padN, newK = K;
    return {newM, newN, newK};
}