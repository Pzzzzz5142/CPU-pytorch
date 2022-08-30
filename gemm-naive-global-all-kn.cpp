#include <memory.h>
#include <iostream>
#define __AVX256__ 1
#include <simd.h>
#include <new>
#include <assert.h>

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
const size_t N_BLOCK_SIZE = 256;
const size_t packA_SIZE = sizeof(float) * M_BLOCK_SIZE * K_BLOCK_SIZE / (1024.0);
const auto packB_SIZE = sizeof(float) * N_BLOCK_SIZE * K_BLOCK_SIZE / (1024.0);
const auto packC_SIZE = sizeof(float) * N_BLOCK_SIZE * M_KERNEL_SIZE / (1024.0);

const char *gemm_desc = "my mmul";

#include <stdio.h>
#include <immintrin.h> // AVX

typedef unsigned long long inc_t;
typedef unsigned long long dim_t;

auto addr_align = std::align_val_t(32);

void addDot_asm_6x16(
    size_t K, size_t newN, float *calA, int lda, float *calB, int ldb, float *calC)
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

        "vmovaps    (%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps  32(%%rbx), %%ymm13               \n\t"

        "testq     %%rsi,    %%rsi                 \n\t" // if kc==0 start kl loop
        "je        .DKLEFT%=                       \n\t"

        ".DLOOP%=:                                 \n\t"

        // update 1.
        //"prefetcht0 64*4(%%rax)                    \n\t"

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
        "vbroadcastss 92(%%rax),  %%ymm15          \n\t"
        "subq             $-96,   %%rax            \n\t" // pointA += 6*4
        "vfmadd231ps   %%ymm14,  %%ymm12, %%ymm8   \n\t" // cal fma
        "vfmadd231ps   %%ymm14,  %%ymm13, %%ymm9   \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm12, %%ymm10  \n\t"
        "vfmadd231ps   %%ymm15,  %%ymm13, %%ymm11  \n\t"

        "vmovaps    (%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps  32(%%rbx), %%ymm13               \n\t"

        "decq      %%rsi                           \n\t"
        "jne       .DLOOP%=                        \n\t"

        ".DKLEFT%=:                                \n\t"

        "testq     %%r9,    %%r9                   \n\t" // if kl==0 start writeback to c
        "je        .DWRITEBACK%=                   \n\t"

        ".DLEFTLOOP%=:                             \n\t"

        //"prefetcht0 64*4(%%rax)                    \n\t"

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

        "vmovaps    (%%rbx), %%ymm12               \n\t" // loading data from b to avx regs
        "vmovaps  32(%%rbx), %%ymm13               \n\t"

        "decq      %%r9                            \n\t"
        "jne       .DLEFTLOOP%=                    \n\t"

        ".DWRITEBACK%=:                            \n\t"

        "vmovaps   %%ymm0,               (%%rcx)   \n\t" // storing data from avx regs to c
        "vmovaps   %%ymm1,             32(%%rcx)   \n\t" // I have tried vmovntps to directly save data to memory but not cache.
        "vmovaps   %%ymm2,     (%%rcx, %%rdx, 4)   \n\t" // However, since there is multiple add and write across the kernel, save data directly to memory will hurt the performance.
        "vmovaps   %%ymm3,   32(%%rcx, %%rdx, 4)   \n\t"
        "vmovaps   %%ymm4,     (%%rcx, %%rdx, 8)   \n\t"
        "vmovaps   %%ymm5,   32(%%rcx, %%rdx, 8)   \n\t"
        "vmovaps   %%ymm6,                (%%r8)   \n\t"
        "vmovaps   %%ymm7,              32(%%r8)   \n\t"
        "vmovaps   %%ymm8,      (%%r8, %%rdx, 4)   \n\t"
        "vmovaps   %%ymm9,    32(%%r8, %%rdx, 4)   \n\t"
        "vmovaps  %%ymm10,      (%%r8, %%rdx, 8)   \n\t"
        "vmovaps  %%ymm11,    32(%%r8, %%rdx, 8)   \n\t"

        :            // output
        :            // input
        "m"(kc),     // 0
        "m"(kl),     // 1
        "m"(pointA), // 2
        "m"(pointB), // 3
        "m"(pointC), // 4
        "m"(newN)    // 5
        :            // register clobber list
        "rax", "rbx", "rcx", "rdx", "rsi", "r8", "r9",
        "ymm0", "ymm1", "ymm2", "ymm3",
        "ymm4", "ymm5", "ymm6", "ymm7",
        "ymm8", "ymm9", "ymm10", "ymm11",
        "ymm12", "ymm13", "ymm14", "ymm15");
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
            // addDot(k, newN, pointA, M_KERNEL_SIZE, pointB + (j * k), N_KERNEL_SIZE, &C(i, j));
            addDot_asm_6x16(k, newN, pointA, M_KERNEL_SIZE, pointB + (j * k), N_KERNEL_SIZE, &C(i, j));
        }
        pointA += M_KERNEL_SIZE * k;
    }
    // printf("using space: %d, N_KERNEL_SIZE * k: %d * %d = %d.\n", cal, (n + N_KERNEL_SIZE - 1) / N_KERNEL_SIZE * N_KERNEL_SIZE, k, (n + N_KERNEL_SIZE - 1) / N_KERNEL_SIZE * N_KERNEL_SIZE * k);
}

void globalPackingA(size_t m, size_t k, size_t K, float *a, size_t lda, float *newA)
{
    assert(m % M_KERNEL_SIZE == 0);
    for (int i = 0; i < m; i += M_KERNEL_SIZE)
    {
        packCol(&oldA(i, 0), lda, M_KERNEL_SIZE, K - i, k, k, newA);
        newA += M_KERNEL_SIZE * k;
    }
}

void globalPackingB(size_t n, size_t k, size_t N, float *b, size_t ldb, float *newB)
{
    assert(n % N_KERNEL_SIZE == 0);
    for (int i = 0; i < n; i += N_KERNEL_SIZE)
    {
        packRow(&oldB(0, i), ldb, k, k, N_KERNEL_SIZE, N - i, newB);
        newB += N_KERNEL_SIZE * k;
    }
}

void square_gemm(int M, int N, int K, float *a, float *b, float *c)
{
    float *calA = a, *calB = b, *calC = c;

    const size_t paddingTarget = 48;
    size_t padM = M_KERNEL_SIZE - (M % M_KERNEL_SIZE), padN = N_KERNEL_SIZE - (N % N_KERNEL_SIZE), padK = 48 - (K % 48);
    size_t newM = M + padM, newN = N + padN, newK = K + padK;

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
        for (int m = 0; m < M; m += M_BLOCK_SIZE)
        {
            auto im = M_BLOCK_SIZE < M - m ? M_BLOCK_SIZE : M - m;
            if (im % M_KERNEL_SIZE)
                im += M_KERNEL_SIZE - (im % M_KERNEL_SIZE);
            globalPackingA(im, ik, K, &oldA(m, k), K, pointA);
            pointA += im * ik;
        }
        for (int n = 0; n < N; n += N_BLOCK_SIZE)
        {
            auto in = N_BLOCK_SIZE < N - n ? N_BLOCK_SIZE : N - n;
            if (in % N_KERNEL_SIZE)
                in += N_KERNEL_SIZE - (in % N_KERNEL_SIZE);
            globalPackingB(in, ik, N, &oldB(k, n), N, pointB);
            pointB += in * ik;
        }
    }
    pointA = calA;
    pointB = calB;
    for (int k = 0; k < K; k += K_BLOCK_SIZE)
    {
        auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
#ifdef __OMP__
#pragma omp parallel for
#endif
        auto tmp = pointA;
        for (int n = 0; n < N; n += N_BLOCK_SIZE)
        {
            auto in = N_BLOCK_SIZE < N - n ? N_BLOCK_SIZE : N - n;
            if (in % N_KERNEL_SIZE)
                in += N_KERNEL_SIZE - (in % N_KERNEL_SIZE);
            tmp = pointA;
            for (int m = 0; m < M; m += M_BLOCK_SIZE)
            {
                auto im = M_BLOCK_SIZE < M - m ? M_BLOCK_SIZE : M - m;
                if (im % M_KERNEL_SIZE)
                    im += M_KERNEL_SIZE - (im % M_KERNEL_SIZE);
                inner_kernal(im, in, ik, newN, tmp, pointB, &C(m, n));
                tmp += im * ik;
            }
            pointB += in * ik;
        }
        pointA = tmp;
    }

    float *tmp = c;
    for (int i = 0; i < M; i++)
    {
        for (int j = 0; j < N; j++)
        {
            // printf("%f", C(i, j));
            *tmp = C(i, j);
            tmp++;
        }
    }
    ::operator delete[](calA, addr_align);
    ::operator delete[](calB, addr_align);
    ::operator delete[](calC, addr_align);
}

void gemm_compute(int M, int N, int K, float *a, float *b, float *c)
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
            globalPackingA(im, ik, K, &oldA(m, k), K, pointA);
            pointA += im * ik;
        }
    }
    //#pragma omp parallel for
    for (int n = 0; n < newN; n += N_BLOCK_SIZE)
    {
        auto in = N_BLOCK_SIZE < newN - n ? N_BLOCK_SIZE : newN - n;
        for (int k = 0; k < K; k += K_BLOCK_SIZE)
        {
            // auto calC = new (addr_align) float[newM * newN];
            // memset(calC, 0, sizeof(float) * newM * newN);
            auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;

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
            // printf("%f", C(i, j));
            c[i * N + j] += C(i, j);
        }
    }
    ::operator delete[](calC, addr_align);
    ::operator delete[](calA, addr_align);
}

float *packing(int M, int N, int K, float *b, int ldb)
{
    size_t padN = (N_KERNEL_SIZE - (N % N_KERNEL_SIZE)) % N_KERNEL_SIZE, padK = 64 - (K % 64);
    size_t newN = N + padN, newK = K + padK;
    auto res = new (addr_align) float[newN * newK];
    memset(res, 0, sizeof(float) * newN * newK);
    auto pointB = res;

    for (int n = 0; n < newN; n += N_BLOCK_SIZE)
    {
        auto in = N_BLOCK_SIZE < newN - n ? N_BLOCK_SIZE : newN - n;
        for (int k = 0; k < K; k += K_BLOCK_SIZE)
        {
            auto ik = K_BLOCK_SIZE < K - k ? K_BLOCK_SIZE : K - k;
            globalPackingB(in, ik, N, &oldB(k, n), N, pointB);
            pointB += in * ik;
        }
    }
    return res;
}

void free_packing(float *Bp)
{
    ::operator delete[](Bp, addr_align);
}