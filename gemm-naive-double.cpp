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

const size_t M_KERNEL_SIZE = 4;
const size_t K_BLOCK_SIZE = 256;
const size_t M_BLOCK_SIZE = 256;

const char *gemm_desc = "my mmul";

#include <stdio.h>
#include <immintrin.h> // AVX

typedef unsigned long long inc_t;
typedef unsigned long long dim_t;

void sgemm_asm_8x8_addDot(
    const size_t N,
    int newN,
    const double *calA,
    int lda,
    const double *calB,
    int ldb,
    const double *calC,
    unsigned long long ldc)
{
    const double *pointA = &A(0, 0), *pointB = &B(0, 0), *pointC = &C(0, 0);
    //__asm__ volatile(
    //    //"movq      %0,      %%rsi    \n\t" // N (32 bit) stored in %rsi
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

void addDot(int N, int newN, double *calA, int lda, double *calB, int ldb, double *calC)
{
    __m128d avx_c[M_KERNEL_SIZE];
    double *point = &C(0, 0);
#pragma unroll
    for (size_t i = 0; i < M_KERNEL_SIZE; i++)
        *(avx_c + i) = _mm_loadu_pd(point + i * newN);
    double *pointB = &B(0, 0);

    for (int k = 0; k < N; k++)
    {
        __m128d avx_b;
        avx_b = _mm_loadu_pd(pointB + k * ldb);
        // simd_load<1>(&avx_b, pointB + k * ldb, false);
#pragma unroll
        for (size_t i = 0; i < M_KERNEL_SIZE; i++)
        {
            __m128d avx_a;
            // avx_a = SIMD_SET(A(i, k));
            avx_a = _mm_set1_pd(A(i, k));
            // simd_fma<1>(avx_c + i, &avx_a, avx_b, avx_c + i);
            avx_c[i] = _mm_fmadd_pd(avx_a, avx_b, avx_c[i]);
        }
    }
#pragma unroll
    for (size_t i = 0; i < M_KERNEL_SIZE; i++)
        // simd_store<1>(&C(i, 0), avx_c + i, false);
        _mm_storeu_pd(&C(i, 0), avx_c[i]);
}

#define XRow(i, j) x[(i)*I + (j)]
#define XCol(i, j) x[(i) + (j)*J]
void packCol(double *calA, int lda, size_t I, size_t J, double *to)
{
    for (int j = 0; j < J; j++)
        for (int i = 0; i < I; i++)
        {
            *to = A(i, j);
            to++;
        }
}

void packRow(double *calB, int ldb, size_t I, size_t J, double *to)
{
    for (int i = 0; i < I; i++)
        for (int j = 0; j < J; j++)
        {
            *to = B(i, j);
            to++;
        }
}

void inner_kernal(int n, int m, int k, int newN, double *calA, int lda, double *calB, int ldb, double *calC)
{
    double packA[M_KERNEL_SIZE * k];
    double packB[newN * k];
    for (int i = 0; i < n; i += M_KERNEL_SIZE)
    {
        packCol(&A(i, 0), newN, M_KERNEL_SIZE, k, packA);
        for (int j = 0; j < m; j += M_KERNEL_SIZE)
        {
            if (i == 0)
                packRow(&B(0, j), newN, k, M_KERNEL_SIZE, packB + (j * k));
            addDot(k, newN, packA, M_KERNEL_SIZE, packB + (j * k), M_KERNEL_SIZE, &C(i, j));
            // bl_sgemm_asm_8x8(k, packA, packB, &C(i, j), newN);
        }
    }
}

void square_gemm(int N, float *a, float *b, float *c)
{
    double *calA, *calB, *calC;
    int newN = N;
    if (true || N % M_KERNEL_SIZE != 0)
    {
        int pad = M_KERNEL_SIZE - (N % M_KERNEL_SIZE);
        newN += pad;
        calA = new double[newN * newN];
        calB = new double[newN * newN];
        calC = new double[newN * newN];
        memset(calA, 0, sizeof(double) * newN * newN);
        memset(calB, 0, sizeof(double) * newN * newN);
        memset(calC, 0, sizeof(double) * newN * newN);

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
        calA = new double[newN * newN];
        int lda = newN;
        memset(calA, 0, sizeof(double) * newN * newN);
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
        for (int n = 0; n < N; n += M_BLOCK_SIZE)
        {
            auto in = M_BLOCK_SIZE < N - n ? M_BLOCK_SIZE : N - n;
            inner_kernal(in, N, ik, newN, &A(n, k), newN, &B(k, 0), ldb, &C(n, 0));
        }
    }

    if (true || newN != N)
    {
        for (int i = 0; i < N; i++)
        {
            for (int j = 0; j < N; j++)
            {
                // printf("%f", C(i, j));
                oldC(i, j) = C(i, j);
            }
        }
        delete calB;
        delete calC;
    }

    delete calA;
}