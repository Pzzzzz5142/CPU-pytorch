#include <memory.h>
#include <iostream>
#define __AVX256__ 1
#include <simd.h>

#define A(i, j) calA[(i) + (j)*lda]
#define B(i, j) calB[(i)*ldb + (j)]
#define C(i, j) calC[(i)*newN + (j)]

#define oldA(i, j) a[(i)*N + (j)]
#define oldB(i, j) b[(i)*N + (j)]
#define oldC(i, j) c[(i)*N + (j)]

const size_t M_KERNEL_SIZE = 6;
const size_t N_KERNEL_SIZE = 16;
const size_t K_BLOCK_SIZE = 256;
const size_t M_BLOCK_SIZE = 384;
const size_t N_BLOCK_SIZE = 2000;

const char *gemm_desc = "my mmul";

#include <stdio.h>
#include <immintrin.h> // AVX

typedef unsigned long long inc_t;
typedef unsigned long long dim_t;

void sgemm_asm_8x8_addDot(
    size_t N,
    int newN,
    float *calA,
    int lda,
    float *calB,
    int ldb,
    float *calC,
    unsigned long long ldc)
{
    sizeof(calB);
    float *pointA = &A(0, 0), *pointB = &B(0, 0), *pointC = &C(0, 0);
    //__asm__ volatile(
    //    "movq      %0,      %%rsi    \n\t" // N (32 bit) stored in %rsi
    //    "movq      %1,      %%rax    \n\t" // Address of A stored in %rax
    //    "movq      %2,      %%rbx    \n\t" // Address of B stored in %rbx
    //    "movq      %3,      %%rcx    \n\t" // Address of C stored in %rcx

    //    : // output
    //    : // input
    //    "N"(N),
    //    "m"(pointA),
    //    "m"(pointB),
    //    "m"(pointC)
    //    : // register clobber list
    //    "rax", "rbx", "rcx", "rsi",
    //    "xmm0", "xmm1", "xmm2", "xmm3",
    //    "xmm4", "xmm5", "xmm6", "xmm7",
    //    "xmm8", "xmm9", "xmm10", "xmm11",
    //    "xmm12", "xmm13", "xmm14", "xmm15");
}

void addDot(int N, int newN, float *calA, int lda, float *calB, int ldb, float *calC)
{
    AVX_Data avx_c[M_KERNEL_SIZE * 2];
    float *point = &C(0, 0);
#pragma unroll
    for (size_t i = 0; i < M_KERNEL_SIZE; i++)
        simd_load<2>(avx_c + i * 2, point + i * newN, false);
    float *pointB = &B(0, 0);

    for (int k = 0; k < N; k++)
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

#define XRow(i, j) x[(i)*I + (j)]
#define XCol(i, j) x[(i) + (j)*J]
void packCol(float *calA, int lda, size_t I, size_t J, float *to)
{
    for (int j = 0; j < J; j++)
        for (int i = 0; i < I; i++)
        {
            *to = A(i, j);
            to++;
        }
}

void packRow(float *calB, int ldb, size_t I, size_t J, float *to)
{
    for (int i = 0; i < I; i++)
        for (int j = 0; j < J; j++)
        {
            *to = B(i, j);
            to++;
        }
}

void inner_kernal(int m, int n, int k, int newN, float *calA, int lda, float *calB, int ldb, float *calC)
{
    float packA[M_KERNEL_SIZE * k];
    float packB[(n + N_KERNEL_SIZE - 1) / N_KERNEL_SIZE * N_KERNEL_SIZE * k];
    for (int i = 0; i < m; i += M_KERNEL_SIZE)
    {
        packCol(&A(i, 0), newN, M_KERNEL_SIZE, k, packA);
        for (int j = 0; j < n; j += N_KERNEL_SIZE)
        {
            if (i == 0)
                packRow(&B(0, j), newN, k, N_KERNEL_SIZE, packB + (j * k));
            addDot(k, newN, packA, M_KERNEL_SIZE, packB + (j * k), N_KERNEL_SIZE, &C(i, j));
            // bl_sgemm_asm_8x8(k, packA, packB, &C(i, j), newN);
        }
    }
    // printf("using space: %d, N_KERNEL_SIZE * k: %d * %d = %d.\n", cal, (n + N_KERNEL_SIZE - 1) / N_KERNEL_SIZE * N_KERNEL_SIZE, k, (n + N_KERNEL_SIZE - 1) / N_KERNEL_SIZE * N_KERNEL_SIZE * k);
}

void square_gemm(int N, float *a, float *b, float *c)
{
    float *calA = a, *calB = b, *calC = c;
    int newN = N;
    if (N % 48 != 0)
    {
        int pad = 48 - (N % 48);
        newN += pad;
        calA = new float[newN * newN];
        calB = new float[newN * newN];
        calC = new float[newN * newN];
        memset(calA, 0, sizeof(float) * newN * newN);
        memset(calB, 0, sizeof(float) * newN * newN);
        memset(calC, 0, sizeof(float) * newN * newN);

        int lda = newN;
        int ldb = newN;

        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A(i, j) = oldA(i, j);
                B(i, j) = oldB(i, j);
                C(i, j) = oldC(i, j);
            }
        }
    }
    else
    {
        calA = new float[newN * newN];
        int lda = newN;
        memset(calA, 0, sizeof(float) * newN * newN);
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                A(i, j) = oldA(i, j);
            }
        }
    }
    int lda = newN;
    int ldb = newN;
    for (int k = 0; k < N; k += K_BLOCK_SIZE)
    {
        auto ik = K_BLOCK_SIZE < N - k ? K_BLOCK_SIZE : N - k;
#ifdef __OMP__
#pragma omp parallel for
#endif
        for (int m = 0; m < N; m += M_BLOCK_SIZE)
        {
            auto im = M_BLOCK_SIZE < N - m ? M_BLOCK_SIZE : N - m;
            for (int n = 0; n < N; n += N_BLOCK_SIZE)
            {
                auto in = N_BLOCK_SIZE < N - n ? N_BLOCK_SIZE : N - n;
                inner_kernal(im, in, ik, newN, &A(m, k), lda, &B(k, n), ldb, &C(m, n));
            }
        }
    }

    if (newN != N)
    {
        float *tmp = c;
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                // printf("%f", C(i, j));
                *tmp = C(i, j);
                tmp++;
            }
        }
        delete calB;
        delete calC;
    }

    delete calA;
}