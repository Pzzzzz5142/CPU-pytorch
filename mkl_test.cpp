#include <mkl.h>
#include <iostream>
extern void square_gemm(int, float *, float *, float *);
using namespace std;

void fill(float *a, int size)
{
    for (int i = 0; i < size; i++)
    {
        a[i] = i;
    }
}

void show_matrix(float *a, int m, int n)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
            cout << a[i * n + j] << ' ';
        cout << endl;
    }
    cout << endl;
}

int main()
{
    int m = 8, n = 8, k = 8;
    int lda = k, ldb = n, ldc = n;
    float a[m * k], b[n * k], c[m * n];
    fill(a, m * k);
    fill(b, n * k);
    show_matrix(a, m, k);
    show_matrix(b, k, n);
    square_gemm(m, a, b, c);
    show_matrix(c, m, n);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, a, lda, b, ldb, 0, c, ldc);
    show_matrix(c, m, n);
}