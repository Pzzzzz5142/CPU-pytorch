#include <stdio.h>
#include <emmintrin.h>
#define MC 384
#define KC 384
#define NC 4096

#define MR 4
#define NR 4

//
//  Local buffers for storing panels from A, B and C
//
static double _A[MC * KC];
static double _B[KC * NC];
static double _C[MR * NR];

//
//  Packing complete panels from A (i.e. without padding)
//
static void
pack_MRxk(int k, const double *A, int incRowA, int incColA,
          double *buffer)
{
    int i, j;

    for (j = 0; j < k; ++j)
    {
        for (i = 0; i < MR; ++i)
        {
            buffer[i] = A[i * incRowA];
        }
        buffer += MR;
        A += incColA;
    }
}

//
//  Packing panels from A with padding if required
//
static void
pack_A(int mc, int kc, const double *A, int incRowA, int incColA,
       double *buffer)
{
    int mp = mc / MR;
    int _mr = mc % MR;

    int i, j;

    for (i = 0; i < mp; ++i)
    {
        pack_MRxk(kc, A, incRowA, incColA, buffer);
        buffer += kc * MR;
        A += MR * incRowA;
    }
    if (_mr > 0)
    {
        for (j = 0; j < kc; ++j)
        {
            for (i = 0; i < _mr; ++i)
            {
                buffer[i] = A[i * incRowA];
            }
            for (i = _mr; i < MR; ++i)
            {
                buffer[i] = 0.0;
            }
            buffer += MR;
            A += incColA;
        }
    }
}

//
//  Packing complete panels from B (i.e. without padding)
//
static void
pack_kxNR(int k, const double *B, int incRowB, int incColB,
          double *buffer)
{
    int i, j;

    for (i = 0; i < k; ++i)
    {
        for (j = 0; j < NR; ++j)
        {
            buffer[j] = B[j * incColB];
        }
        buffer += NR;
        B += incRowB;
    }
}

//
//  Packing panels from B with padding if required
//
static void
pack_B(int kc, int nc, const double *B, int incRowB, int incColB,
       double *buffer)
{
    int np = nc / NR;
    int _nr = nc % NR;

    int i, j;

    for (j = 0; j < np; ++j)
    {
        pack_kxNR(kc, B, incRowB, incColB, buffer);
        buffer += kc * NR;
        B += NR * incColB;
    }
    if (_nr > 0)
    {
        for (i = 0; i < kc; ++i)
        {
            for (j = 0; j < _nr; ++j)
            {
                buffer[j] = B[j * incColB];
            }
            for (j = _nr; j < NR; ++j)
            {
                buffer[j] = 0.0;
            }
            buffer += NR;
            B += incRowB;
        }
    }
}

//
//  Micro kernel for multiplying panels from A and B.
//
static void
dgemm_micro_kernel(int kc,
                   double alpha, const double *A, const double *B,
                   double beta,
                   double *C, int incRowC, int incColC)
{
    double AB[MR * NR] __attribute__((aligned(16)));

    // Cols of AB in SSE registers
    __m128d ab_00_10, ab_20_30;
    __m128d ab_01_11, ab_21_31;
    __m128d ab_02_12, ab_22_32;
    __m128d ab_03_13, ab_23_33;

    __m128d a_01, a_23;
    __m128d b_00, b_11, b_22, b_33;
    __m128d tmp1, tmp2;

    int i, j, l;

    ab_00_10 = _mm_setzero_pd();
    ab_20_30 = _mm_setzero_pd();
    ab_01_11 = _mm_setzero_pd();
    ab_21_31 = _mm_setzero_pd();
    ab_02_12 = _mm_setzero_pd();
    ab_22_32 = _mm_setzero_pd();
    ab_03_13 = _mm_setzero_pd();
    ab_23_33 = _mm_setzero_pd();

    //
    //  Compute AB = A*B
    //
    for (l = 0; l < kc / 4; ++l)
    {
        a_01 = _mm_load_pd(A);
        a_23 = _mm_load_pd(A + 2);

        b_00 = _mm_load_pd1(B);
        b_11 = _mm_load_pd1(B + 1);
        b_22 = _mm_load_pd1(B + 2);
        b_33 = _mm_load_pd1(B + 3);

        tmp1 = a_01;
        tmp2 = a_23;

        // col 0 of AB
        tmp1 = _mm_mul_pd(tmp1, b_00);
        tmp2 = _mm_mul_pd(tmp2, b_00);
        ab_00_10 = _mm_add_pd(tmp1, ab_00_10);
        ab_20_30 = _mm_add_pd(tmp2, ab_20_30);

        // col 1 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_11);
        tmp2 = _mm_mul_pd(tmp2, b_11);
        ab_01_11 = _mm_add_pd(tmp1, ab_01_11);
        ab_21_31 = _mm_add_pd(tmp2, ab_21_31);

        // col 2 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_22);
        tmp2 = _mm_mul_pd(tmp2, b_22);
        ab_02_12 = _mm_add_pd(tmp1, ab_02_12);
        ab_22_32 = _mm_add_pd(tmp2, ab_22_32);

        // col 3 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_33);
        tmp2 = _mm_mul_pd(tmp2, b_33);
        ab_03_13 = _mm_add_pd(tmp1, ab_03_13);
        ab_23_33 = _mm_add_pd(tmp2, ab_23_33);

        A += 4;
        B += 4;

        a_01 = _mm_load_pd(A);
        a_23 = _mm_load_pd(A + 2);

        b_00 = _mm_load_pd1(B);
        b_11 = _mm_load_pd1(B + 1);
        b_22 = _mm_load_pd1(B + 2);
        b_33 = _mm_load_pd1(B + 3);

        tmp1 = a_01;
        tmp2 = a_23;

        // col 0 of AB
        tmp1 = _mm_mul_pd(tmp1, b_00);
        tmp2 = _mm_mul_pd(tmp2, b_00);
        ab_00_10 = _mm_add_pd(tmp1, ab_00_10);
        ab_20_30 = _mm_add_pd(tmp2, ab_20_30);

        // col 1 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_11);
        tmp2 = _mm_mul_pd(tmp2, b_11);
        ab_01_11 = _mm_add_pd(tmp1, ab_01_11);
        ab_21_31 = _mm_add_pd(tmp2, ab_21_31);

        // col 2 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_22);
        tmp2 = _mm_mul_pd(tmp2, b_22);
        ab_02_12 = _mm_add_pd(tmp1, ab_02_12);
        ab_22_32 = _mm_add_pd(tmp2, ab_22_32);

        // col 3 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_33);
        tmp2 = _mm_mul_pd(tmp2, b_33);
        ab_03_13 = _mm_add_pd(tmp1, ab_03_13);
        ab_23_33 = _mm_add_pd(tmp2, ab_23_33);

        A += 4;
        B += 4;

        a_01 = _mm_load_pd(A);
        a_23 = _mm_load_pd(A + 2);

        b_00 = _mm_load_pd1(B);
        b_11 = _mm_load_pd1(B + 1);
        b_22 = _mm_load_pd1(B + 2);
        b_33 = _mm_load_pd1(B + 3);

        tmp1 = a_01;
        tmp2 = a_23;

        // col 0 of AB
        tmp1 = _mm_mul_pd(tmp1, b_00);
        tmp2 = _mm_mul_pd(tmp2, b_00);
        ab_00_10 = _mm_add_pd(tmp1, ab_00_10);
        ab_20_30 = _mm_add_pd(tmp2, ab_20_30);

        // col 1 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_11);
        tmp2 = _mm_mul_pd(tmp2, b_11);
        ab_01_11 = _mm_add_pd(tmp1, ab_01_11);
        ab_21_31 = _mm_add_pd(tmp2, ab_21_31);

        // col 2 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_22);
        tmp2 = _mm_mul_pd(tmp2, b_22);
        ab_02_12 = _mm_add_pd(tmp1, ab_02_12);
        ab_22_32 = _mm_add_pd(tmp2, ab_22_32);

        // col 3 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_33);
        tmp2 = _mm_mul_pd(tmp2, b_33);
        ab_03_13 = _mm_add_pd(tmp1, ab_03_13);
        ab_23_33 = _mm_add_pd(tmp2, ab_23_33);

        A += 4;
        B += 4;

        a_01 = _mm_load_pd(A);
        a_23 = _mm_load_pd(A + 2);

        b_00 = _mm_load_pd1(B);
        b_11 = _mm_load_pd1(B + 1);
        b_22 = _mm_load_pd1(B + 2);
        b_33 = _mm_load_pd1(B + 3);

        tmp1 = a_01;
        tmp2 = a_23;

        // col 0 of AB
        tmp1 = _mm_mul_pd(tmp1, b_00);
        tmp2 = _mm_mul_pd(tmp2, b_00);
        ab_00_10 = _mm_add_pd(tmp1, ab_00_10);
        ab_20_30 = _mm_add_pd(tmp2, ab_20_30);

        // col 1 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_11);
        tmp2 = _mm_mul_pd(tmp2, b_11);
        ab_01_11 = _mm_add_pd(tmp1, ab_01_11);
        ab_21_31 = _mm_add_pd(tmp2, ab_21_31);

        // col 2 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_22);
        tmp2 = _mm_mul_pd(tmp2, b_22);
        ab_02_12 = _mm_add_pd(tmp1, ab_02_12);
        ab_22_32 = _mm_add_pd(tmp2, ab_22_32);

        // col 3 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_33);
        tmp2 = _mm_mul_pd(tmp2, b_33);
        ab_03_13 = _mm_add_pd(tmp1, ab_03_13);
        ab_23_33 = _mm_add_pd(tmp2, ab_23_33);

        A += 4;
        B += 4;
    }

    for (l = 0; l < kc % 4; ++l)
    {
        a_01 = _mm_load_pd(A);
        a_23 = _mm_load_pd(A + 2);

        b_00 = _mm_load_pd1(B);
        b_11 = _mm_load_pd1(B + 1);
        b_22 = _mm_load_pd1(B + 2);
        b_33 = _mm_load_pd1(B + 3);

        tmp1 = a_01;
        tmp2 = a_23;

        // col 0 of AB
        tmp1 = _mm_mul_pd(tmp1, b_00);
        tmp2 = _mm_mul_pd(tmp2, b_00);
        ab_00_10 = _mm_add_pd(tmp1, ab_00_10);
        ab_20_30 = _mm_add_pd(tmp2, ab_20_30);

        // col 1 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_11);
        tmp2 = _mm_mul_pd(tmp2, b_11);
        ab_01_11 = _mm_add_pd(tmp1, ab_01_11);
        ab_21_31 = _mm_add_pd(tmp2, ab_21_31);

        // col 2 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_22);
        tmp2 = _mm_mul_pd(tmp2, b_22);
        ab_02_12 = _mm_add_pd(tmp1, ab_02_12);
        ab_22_32 = _mm_add_pd(tmp2, ab_22_32);

        // col 3 of AB
        tmp1 = a_01;
        tmp2 = a_23;
        tmp1 = _mm_mul_pd(tmp1, b_33);
        tmp2 = _mm_mul_pd(tmp2, b_33);
        ab_03_13 = _mm_add_pd(tmp1, ab_03_13);
        ab_23_33 = _mm_add_pd(tmp2, ab_23_33);

        A += 4;
        B += 4;
    }

    _mm_store_pd(AB + 0, ab_00_10);
    _mm_store_pd(AB + 2, ab_20_30);

    _mm_store_pd(AB + 4, ab_01_11);
    _mm_store_pd(AB + 6, ab_21_31);

    _mm_store_pd(AB + 8, ab_02_12);
    _mm_store_pd(AB + 10, ab_22_32);

    _mm_store_pd(AB + 12, ab_03_13);
    _mm_store_pd(AB + 14, ab_23_33);

    //
    //  Update C <- beta*C
    //
    if (beta == 0.0)
    {
        for (j = 0; j < NR; ++j)
        {
            for (i = 0; i < MR; ++i)
            {
                C[i * incRowC + j * incColC] = 0.0;
            }
        }
    }
    else if (beta != 1.0)
    {
        for (j = 0; j < NR; ++j)
        {
            for (i = 0; i < MR; ++i)
            {
                C[i * incRowC + j * incColC] *= beta;
            }
        }
    }

    //
    //  Update C <- C + alpha*AB (note: the case alpha==0.0 was already treated in
    //                                  the above layer dgemm_nn)
    //
    if (alpha == 1.0)
    {
        for (j = 0; j < NR; ++j)
        {
            for (i = 0; i < MR; ++i)
            {
                C[i * incRowC + j * incColC] += AB[i + j * MR];
            }
        }
    }
    else
    {
        for (j = 0; j < NR; ++j)
        {
            for (i = 0; i < MR; ++i)
            {
                C[i * incRowC + j * incColC] += alpha * AB[i + j * MR];
            }
        }
    }
}

//
//  Compute Y += alpha*X
//
static void
dgeaxpy(int m,
        int n,
        double alpha,
        const double *X,
        int incRowX,
        int incColX,
        double *Y,
        int incRowY,
        int incColY)
{
    int i, j;

    if (alpha != 1.0)
    {
        for (j = 0; j < n; ++j)
        {
            for (i = 0; i < m; ++i)
            {
                Y[i * incRowY + j * incColY] += alpha * X[i * incRowX + j * incColX];
            }
        }
    }
    else
    {
        for (j = 0; j < n; ++j)
        {
            for (i = 0; i < m; ++i)
            {
                Y[i * incRowY + j * incColY] += X[i * incRowX + j * incColX];
            }
        }
    }
}

//
//  Compute X *= alpha
//
static void
dgescal(int m,
        int n,
        double alpha,
        double *X,
        int incRowX,
        int incColX)
{
    int i, j;

    if (alpha != 0.0)
    {
        for (j = 0; j < n; ++j)
        {
            for (i = 0; i < m; ++i)
            {
                X[i * incRowX + j * incColX] *= alpha;
            }
        }
    }
    else
    {
        for (j = 0; j < n; ++j)
        {
            for (i = 0; i < m; ++i)
            {
                X[i * incRowX + j * incColX] = 0.0;
            }
        }
    }
}

//
//  Macro Kernel for the multiplication of blocks of A and B.  We assume that
//  these blocks were previously packed to buffers _A and _B.
//
static void
dgemm_macro_kernel(int mc,
                   int nc,
                   int kc,
                   double alpha,
                   double beta,
                   double *C,
                   int incRowC,
                   int incColC)
{
    int mp = (mc + MR - 1) / MR;
    int np = (nc + NR - 1) / NR;

    int _mr = mc % MR;
    int _nr = nc % NR;

    int mr, nr;
    int i, j;

    for (j = 0; j < np; ++j)
    {
        nr = (j != np - 1 || _nr == 0) ? NR : _nr;

        for (i = 0; i < mp; ++i)
        {
            mr = (i != mp - 1 || _mr == 0) ? MR : _mr;

            if (mr == MR && nr == NR)
            {
                dgemm_micro_kernel(kc, alpha, &_A[i * kc * MR], &_B[j * kc * NR],
                                   beta,
                                   &C[i * MR * incRowC + j * NR * incColC],
                                   incRowC, incColC);
            }
            else
            {
                dgemm_micro_kernel(kc, alpha, &_A[i * kc * MR], &_B[j * kc * NR],
                                   0.0,
                                   _C, 1, MR);
                dgescal(mr, nr, beta,
                        &C[i * MR * incRowC + j * NR * incColC], incRowC, incColC);
                dgeaxpy(mr, nr, 1.0, _C, 1, MR,
                        &C[i * MR * incRowC + j * NR * incColC], incRowC, incColC);
            }
        }
    }
}

//
//  Compute C <- beta*C + alpha*A*B
//
void dgemm_nn(int m,
              int n,
              int k,
              double alpha,
              const double *A,
              int incRowA,
              int incColA,
              const double *B,
              int incRowB,
              int incColB,
              double beta,
              double *C,
              int incRowC,
              int incColC)
{
    int mb = (m + MC - 1) / MC;
    int nb = (n + NC - 1) / NC;
    int kb = (k + KC - 1) / KC;

    int _mc = m % MC;
    int _nc = n % NC;
    int _kc = k % KC;

    int mc, nc, kc;
    int i, j, l;

    double _beta;

    if (alpha == 0.0 || k == 0)
    {
        dgescal(m, n, beta, C, incRowC, incColC);
        return;
    }

    for (j = 0; j < nb; ++j)
    {
        nc = (j != nb - 1 || _nc == 0) ? NC : _nc;

        for (l = 0; l < kb; ++l)
        {
            kc = (l != kb - 1 || _kc == 0) ? KC : _kc;
            _beta = (l == 0) ? beta : 1.0;

            pack_B(kc, nc,
                   &B[l * KC * incRowB + j * NC * incColB], incRowB, incColB,
                   _B);

            for (i = 0; i < mb; ++i)
            {
                mc = (i != mb - 1 || _mc == 0) ? MC : _mc;

                pack_A(mc, kc,
                       &A[i * MC * incRowA + l * KC * incColA], incRowA, incColA,
                       _A);

                dgemm_macro_kernel(mc, nc, kc, alpha, _beta,
                                   &C[i * MC * incRowC + j * NC * incColC],
                                   incRowC, incColC);
            }
        }
    }
}
const char *gemm_desc;
void square_gemm(int N, float *a, float *b, float *c)
{
    double AA[N * N], BB[N * N], CC[N * N];
    for (int i = 0; i < N * N; i++)
        AA[i] = a[i], BB[i] = b[i], CC[i] = c[i];

    dgemm_nn(N, N, N, 1, AA, 1, N, BB, 1, N, 0, CC, 1, N);
    for (int i = 0; i < N * N; i++)
        a[i] = AA[i], b[i] = BB[i], c[i] = CC[i];
}