#include <mkl.h>
#include <iostream>

const char *gemm_desc;
void square_gemm(int N, float *a, float *b, float *c)
{
    // smkl_set_num_threads(1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, a, N, b, N, 0, c, N);
}

void square_gemm(int M, int N, int K, float *a, float *b, float *c, bool aa = false)
{
    // smkl_set_num_threads(1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, a, K, b, N, 0, c, N);
}

void gemm_compute(int M, int N, const int K, float *a, float *b, float *c, bool bias = false)
{
    // smkl_set_num_threads(1);
    float beta = 0;
    cblas_sgemm_compute(CblasRowMajor, CblasNoTrans, CblasPacked, M, N, K, a, K, b, N, 0, c, N);
    // sgemm_compute("N", "P", &M, &N, &K, a, &M, b, &N, &beta, c, &M);
}

float *packing(int N, int K, float *a, int lda)
{
    // smkl_set_num_threads(1);
    int M = N;
    auto size = cblas_sgemm_pack_get_size(CblasBMatrix, M, N, K);
    float *res = (float *)mkl_malloc(size, 64);
    cblas_sgemm_pack(CblasRowMajor, CblasBMatrix, CblasNoTrans, M, N, K, 1, a, lda, res);
    return res;
}

void free_packing(float *a)
{
    mkl_free(a);
}

void gemm_layernorm_compute_sum(int, int, int, float *, float *, float *, float *, float *, const bool &beta = false, float eps = 1e-5)
{
}