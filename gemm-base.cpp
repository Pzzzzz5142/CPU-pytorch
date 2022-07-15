const char *gemm_desc;
void square_gemm(int N, float *a, float *b, float *c)
{
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < N; k++)
                c[i * N + j] += a[i * N + k] + b[k * N + j];
}