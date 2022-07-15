#include <cblas.h>

const char *gemm_desc = "Open blas";
void square_gemm(int N, float *a, float *b, float *c)
{

    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, a, N, b, N, 0, c, N);
}