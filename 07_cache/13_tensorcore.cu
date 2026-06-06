#include <iostream>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
using namespace std;
using namespace nvcuda;

#define TILE_M 128
#define TILE_N 128
#define TILE_K 32
#define WARPS_M 2
#define WARPS_N 2
#define NUM_WARPS (WARPS_M * WARPS_N)
#define BLOCK_THREADS (NUM_WARPS * 32)
#define WARP_TILE_M (TILE_M / WARPS_M)
#define WARP_TILE_N (TILE_N / WARPS_N)
#define FRAG_M (WARP_TILE_M / 16)
#define FRAG_N (WARP_TILE_N / 16)
#define PAD 8

__global__ void kernel_opt(int dim_m, int dim_n, int dim_k,
                           const float * __restrict__ d_a,
                           const float * __restrict__ d_b,
                           float * __restrict__ d_c) {
    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;
    int warp_id  = threadIdx.x / 32;
    int warp_row = warp_id / WARPS_N;
    int warp_col = warp_id % WARPS_N;
    int warp_m = warp_row * WARP_TILE_M;
    int warp_n = warp_col * WARP_TILE_N;

    __shared__ half smem_a[2][TILE_K][TILE_M + PAD];
    __shared__ half smem_b[2][TILE_K][TILE_N + PAD];

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[FRAG_M][FRAG_N];
    #pragma unroll
    for (int fm = 0; fm < FRAG_M; fm++)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; fn++)
            wmma::fill_fragment(acc[fm][fn], 0.0f);

    int num_k_tiles = (dim_k + TILE_K - 1) / TILE_K;

    {
        int k_base = 0;
        for (int idx = threadIdx.x; idx < TILE_K * TILE_M; idx += BLOCK_THREADS) {
            int tk = idx / TILE_M, tm = idx % TILE_M;
            int g_k = k_base + tk, g_m = block_m + tm;
            smem_a[0][tk][tm] = (g_k < dim_k && g_m < dim_m)
                                 ? __float2half(d_a[g_k * dim_m + g_m]) : __float2half(0.f);
        }
        for (int idx = threadIdx.x; idx < TILE_K * TILE_N; idx += BLOCK_THREADS) {
            int tk = idx / TILE_N, tn = idx % TILE_N;
            int g_k = k_base + tk, g_n = block_n + tn;
            smem_b[0][tk][tn] = (g_k < dim_k && g_n < dim_n)
                                 ? __float2half(d_b[g_n * dim_k + g_k]) : __float2half(0.f);
        }
    }
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int cur  = kt & 1;
        int next = 1 - cur;

        if (kt + 1 < num_k_tiles) {
            int k_base = (kt + 1) * TILE_K;
            for (int idx = threadIdx.x; idx < TILE_K * TILE_M; idx += BLOCK_THREADS) {
                int tk = idx / TILE_M, tm = idx % TILE_M;
                int g_k = k_base + tk, g_m = block_m + tm;
                smem_a[next][tk][tm] = (g_k < dim_k && g_m < dim_m)
                                       ? __float2half(d_a[g_k * dim_m + g_m]) : __float2half(0.f);
            }
            for (int idx = threadIdx.x; idx < TILE_K * TILE_N; idx += BLOCK_THREADS) {
                int tk = idx / TILE_N, tn = idx % TILE_N;
                int g_k = k_base + tk, g_n = block_n + tn;
                smem_b[next][tk][tn] = (g_k < dim_k && g_n < dim_n)
                                       ? __float2half(d_b[g_n * dim_k + g_k]) : __float2half(0.f);
            }
        }

        #pragma unroll
        for (int ki = 0; ki < TILE_K / 16; ki++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag[FRAG_M];
            wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag[FRAG_N];
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; fm++)
                wmma::load_matrix_sync(a_frag[fm], &smem_a[cur][ki*16][warp_m + fm*16], TILE_M + PAD);
            #pragma unroll
            for (int fn = 0; fn < FRAG_N; fn++)
                wmma::load_matrix_sync(b_frag[fn], &smem_b[cur][ki*16][warp_n + fn*16], TILE_N + PAD);
            #pragma unroll
            for (int fm = 0; fm < FRAG_M; fm++)
                #pragma unroll
                for (int fn = 0; fn < FRAG_N; fn++)
                    wmma::mma_sync(acc[fm][fn], a_frag[fm], b_frag[fn], acc[fm][fn]);
        }
        __syncthreads();
    }

    #pragma unroll
    for (int fm = 0; fm < FRAG_M; fm++)
        #pragma unroll
        for (int fn = 0; fn < FRAG_N; fn++) {
            int g_m = block_m + warp_m + fm * 16;
            int g_n = block_n + warp_n + fn * 16;
            if (g_m < dim_m && g_n < dim_n)
                wmma::store_matrix_sync(&d_c[g_n * dim_m + g_m], acc[fm][fn], dim_m, wmma::mem_col_major);
        }
}

int main(int argc, const char **argv) {
    int m = 10240, k = 4096, n = 8192;
    float alpha = 1.0f, beta = 0.0f;
    int Nt = 10;
    float *A, *B, *C, *C2;
    cudaMallocManaged(&A,  m * k * sizeof(float));
    cudaMallocManaged(&B,  k * n * sizeof(float));
    cudaMallocManaged(&C,  m * n * sizeof(float));
    cudaMallocManaged(&C2, m * n * sizeof(float));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < k; j++)
            A[k*i+j] = drand48();
    for (int i = 0; i < k; i++)
        for (int j = 0; j < n; j++)
            B[n*i+j] = drand48();
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C[m*i+j] = C2[m*i+j] = 0.0f;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k,
                     &alpha, A, CUDA_R_32F, m, B, CUDA_R_32F, k,
                     &beta,  C, CUDA_R_32F, m,
                     CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();
    int64_t num_flops = 2LL*m*n*k + 2LL*m*n;
    double tcublas = chrono::duration<double>(toc-tic).count() / Nt;
    double cublas_gflops = (double)num_flops / tcublas / 1e9;

    dim3 block(BLOCK_THREADS);
    dim3 grid((m+TILE_M-1)/TILE_M, (n+TILE_N-1)/TILE_N);
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_opt<<<grid, block>>>(m, n, k, A, B, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tkernel = chrono::duration<double>(toc-tic).count() / Nt;
    double kernel_gflops = (double)num_flops / tkernel / 1e9;

    printf("CUBLAS: %.2f Gflops, OPT_KERNEL: %.2f Gflops (%.1f%% of cuBLAS)\n",
           cublas_gflops, kernel_gflops, 100.0 * kernel_gflops / cublas_gflops);

    double err = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            err += fabs(C[m*i+j] - C2[m*i+j]);
    printf("error: %lf\n", err/n/m);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C2);
    cublasDestroy(cublas_handle);
    return 0;
}