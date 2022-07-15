#include <mkl.h>

const char *gemm_desc;
void square_gemm(int N, float *a, float *b, float *c)
{
    mkl_set_num_threads(1);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, 1, a, N, b, N, 0, c, N);
}