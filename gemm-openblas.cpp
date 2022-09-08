#include <cblas.h>
#include <new>

const char *gemm_desc = "Open blas";
void square_gemm(int N, float *a, float *b, float *c)
{

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, a, N, b, N, 0, c, N);
}
void square_gemm(int M, int N, int K, float *a, float *b, float *c, bool aa = false)
{
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, a, K, b, N, 0, c, N);
}