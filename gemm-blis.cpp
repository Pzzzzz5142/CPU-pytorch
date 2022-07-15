#include <blis/blis.h>

const char *gemm_desc;
void square_gemm(int N, float *a, float *b, float *c)
{
    const float aa = 1, bb = 1;
    bli_sgemm(BLIS_NO_TRANSPOSE, BLIS_NO_TRANSPOSE, N, N, N, &aa, a, 1, N, b, 1, N, &bb, c, 1, N);
}