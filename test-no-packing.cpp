

extern void gemm_compute(int, int, int, float *, float *, float *, float beta = 0);
extern float *packing(int, int, float *, int);
extern void free_packing(float *);
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>

void ref_gemm(int M, int N, int K, float *A, float *B, float *C)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                C[i * N + j] += A[i * K + k] * B[k * N + j];
}

#define min(x, y) (x) < (y) ? (x) : (y)

static void print_matrix(int col, int row, float *A)
{
    for (int i = 0; i < col; ++i)
    {
        for (int j = 0; j < row; ++j)
        {
            printf("%0.2f ", A[j + i * row]);
        }
        printf("\n");
    }
    printf("\n");
}
void test()
{
#define TEST_N_LEN 4
#define TEST_M_LEN 6
#define TEST_K_LEN 3
    auto *A = new float[TEST_M_LEN * TEST_K_LEN], *B = new float[TEST_N_LEN * TEST_K_LEN];
    for (int i = 0; i < TEST_K_LEN * TEST_M_LEN; i++)
        A[i] = i;
    for (int i = 0; i < TEST_K_LEN * TEST_N_LEN; i++)
        B[i] = i;
    print_matrix(TEST_M_LEN, TEST_K_LEN, A);
    print_matrix(TEST_K_LEN, TEST_N_LEN, B);
    float *C = new float[TEST_M_LEN * TEST_N_LEN];
    auto packedB = packing(TEST_N_LEN, TEST_K_LEN, B, TEST_K_LEN);
    gemm_compute(TEST_M_LEN, TEST_N_LEN, TEST_K_LEN, A, packedB, C, 0);
    print_matrix(TEST_M_LEN, TEST_N_LEN, C);
    memset(C, 0, sizeof(float) * TEST_N_LEN * TEST_M_LEN);
    ref_gemm(TEST_M_LEN, TEST_N_LEN, TEST_K_LEN, A, B, C);
    print_matrix(TEST_M_LEN, TEST_N_LEN, C);
    delete[] A;
    delete[] B;
    delete[] C;
}

static void transpose_small_blk(int lda, int M, int N, float *A)
{
    for (int i = 1; i < M; ++i)
    {
        for (int j = 0; j < i && j < N; ++j)
        {
            int ij_index = i + j * lda, ji_index = j + i * lda;
            float tmp = A[ij_index];
            A[ij_index] = A[ji_index];
            A[ji_index] = tmp;
        }
    }
}

#define BLOCK_SIZE 2

static void transpose_m_blk(int lda, float *A)
{
    for (int i = 0; i < lda; i += BLOCK_SIZE)
    {
        for (int j = 0; j < lda; j += BLOCK_SIZE)
        {
            int M = min(BLOCK_SIZE, lda - i);
            int N = min(BLOCK_SIZE, lda - j);
            transpose_small_blk(lda, M, N, A + i + j * lda);
        }
    }
}

static float *pack_L_small_blk(int lda, int M, int N, float *A, float *temp_P)
{
    for (int i = 0; i < M; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            int ij_index = i + j * lda;
            *temp_P = A[ij_index];
            temp_P++;
        }
    }
    return temp_P;
}

static float *pack_L_block(int lda, float *A)
{
    float *temp_A = (float *)malloc(sizeof(float) * lda * lda);
    float *temp_P = temp_A;
    for (int i = 0; i < lda; i += BLOCK_SIZE)
    {
        for (int j = 0; j < lda; j += BLOCK_SIZE)
        {
            int M = min(BLOCK_SIZE, lda - i);
            int N = min(BLOCK_SIZE, lda - j);
            temp_P = pack_L_small_blk(lda, M, N, A + i + j * lda, temp_P);
        }
    }
    return temp_A;
}

static float *pack_R_small_blk(int lda, int M, int N, float *A, float *temp_P)
{
    for (int j = 0; j < N; ++j)
    {
        for (int i = 0; i < M; ++i)
        {
            int ij_index = i + j * lda;
            *temp_P = A[ij_index];
            temp_P++;
        }
    }
    return temp_P;
}

static float *pack_R_block(int lda, float *A)
{
    float *temp_A = (float *)malloc(sizeof(float) * lda * lda);
    float *temp_P = temp_A;
    for (int j = 0; j < lda; j += BLOCK_SIZE)
    {
        for (int i = 0; i < lda; i += BLOCK_SIZE)
        {
            int M = min(BLOCK_SIZE, lda - i);
            int N = min(BLOCK_SIZE, lda - j);
            temp_P = pack_R_small_blk(lda, M, N, A + i + j * lda, temp_P);
        }
    }
    return temp_A;
}

void test_blk()
{
    float A[16] = {1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15, 4, 8, 12, 16};
    print_matrix(4, 4, A);
    // transpose_m_blk(4, A);
    float *B = pack_L_block(4, A);
    // float *B = pack_R_block(4, A);
    print_matrix(4, 4, B);
}

int main()
{
    // test_blk();
    test();
}