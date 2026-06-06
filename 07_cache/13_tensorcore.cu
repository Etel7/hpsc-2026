#include <iostream>
#include <stdint.h>
#include <cublas_v2.h>
#include <chrono>
#include <cuda_fp16.h>
using namespace std;

#define WGMMA_M 64
#define WGMMA_N 64
#define WGMMA_K 16
#define TILE_M 128
#define TILE_N 128
#define TILE_K 64
#define PAD 8
#define BLOCK_THREADS 128

__device__ __forceinline__ uint64_t make_smem_desc(half* ptr, int ld_bytes) {
    uint64_t desc = 0;
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    desc |= ((uint64_t)(addr >> 4) & 0x3FFF) << 0;
    desc |= ((uint64_t)(ld_bytes >> 4) & 0x3FFF) << 16;
    desc |= ((uint64_t)(ld_bytes >> 4) & 0x3FFF) << 32;
    return desc;
}

__global__ void __launch_bounds__(BLOCK_THREADS, 1)
kernel_wgmma(int dim_m, int dim_n, int dim_k,
             const float* __restrict__ d_a,
             const float* __restrict__ d_b,
             float* __restrict__ d_c) {

    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;
    int tid = threadIdx.x;

    __shared__ __align__(128) half smem_a[2][TILE_K][TILE_M + PAD];
    __shared__ __align__(128) half smem_b[2][TILE_K][TILE_N + PAD];

    int num_k_tiles = (dim_k + TILE_K - 1) / TILE_K;

    float acc[2][2][32];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 2; j++)
            #pragma unroll
            for (int k = 0; k < 32; k++)
                acc[i][j][k] = 0.f;

    auto load_tile = [&](int buf, int k_base) {
        for (int idx = tid; idx < TILE_K * TILE_M; idx += BLOCK_THREADS) {
            int tk = idx / TILE_M, tm = idx % TILE_M;
            int g_k = k_base + tk, g_m = block_m + tm;
            smem_a[buf][tk][tm] = (g_k < dim_k && g_m < dim_m)
                ? __float2half(d_a[g_k * dim_m + g_m]) : __float2half(0.f);
        }
        for (int idx = tid; idx < TILE_K * TILE_N; idx += BLOCK_THREADS) {
            int tk = idx / TILE_N, tn = idx % TILE_N;
            int g_k = k_base + tk, g_n = block_n + tn;
            smem_b[buf][tk][tn] = (g_k < dim_k && g_n < dim_n)
                ? __float2half(d_b[g_n * dim_k + g_k]) : __float2half(0.f);
        }
    };

    load_tile(0, 0);
    __syncthreads();

    for (int kt = 0; kt < num_k_tiles; kt++) {
        int cur  = kt & 1;
        int next = 1 - cur;

        if (kt + 1 < num_k_tiles)
            load_tile(next, (kt + 1) * TILE_K);

        asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");

        #pragma unroll
        for (int ki = 0; ki < TILE_K / WGMMA_K; ki++) {
            #pragma unroll
            for (int mi = 0; mi < 2; mi++) {
                #pragma unroll
                for (int ni = 0; ni < 2; ni++) {
                    uint64_t desc_a = make_smem_desc(
                        &smem_a[cur][ki * WGMMA_K][mi * WGMMA_M],
                        (TILE_M + PAD) * sizeof(half));
                    uint64_t desc_b = make_smem_desc(
                        &smem_b[cur][ki * WGMMA_K][ni * WGMMA_N],
                        (TILE_N + PAD) * sizeof(half));

                    asm volatile(
                        "{\n"
                        ".reg .pred p;\n"
                        "setp.ne.b32 p, 1, 0;\n"
                        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
                        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
                        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31},"
                        " %32, %33,"
                        " p, 1, 1, 0, 0;\n"
                        "}\n"
                        : "+f"(acc[mi][ni][0]),  "+f"(acc[mi][ni][1]),
                          "+f"(acc[mi][ni][2]),  "+f"(acc[mi][ni][3]),
                          "+f"(acc[mi][ni][4]),  "+f"(acc[mi][ni][5]),
                          "+f"(acc[mi][ni][6]),  "+f"(acc[mi][ni][7]),
                          "+f"(acc[mi][ni][8]),  "+f"(acc[mi][ni][9]),
                          "+f"(acc[mi][ni][10]), "+f"(acc[mi][ni][11]),
                          "+f"(acc[mi][ni][12]), "+f"(acc[mi][ni][13]),
                          "+f"(acc[mi][ni][14]), "+f"(acc[mi][ni][15]),
                          "+f"(acc[mi][ni][16]), "+f"(acc[mi][ni][17]),
                          "+f"(acc[mi][ni][18]), "+f"(acc[mi][ni][19]),
                          "+f"(acc[mi][ni][20]), "+f"(acc[mi][ni][21]),
                          "+f"(acc[mi][ni][22]), "+f"(acc[mi][ni][23]),
                          "+f"(acc[mi][ni][24]), "+f"(acc[mi][ni][25]),
                          "+f"(acc[mi][ni][26]), "+f"(acc[mi][ni][27]),
                          "+f"(acc[mi][ni][28]), "+f"(acc[mi][ni][29]),
                          "+f"(acc[mi][ni][30]), "+f"(acc[mi][ni][31])
                        : "l"(desc_a), "l"(desc_b)
                    );
                }
            }
        }

        asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
        asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
        __syncthreads();
    }

    // Exact register layout from CLayout_64x64:
    // flat = t0*128 + t1*1 + t2*16 + r0*64 + r1*8 + r2*512
    // row = flat % 64, col = flat / 64
    // where: t0=tid%4, t1=(tid/4)%8, t2=tid/32, r0=r%2, r1=(r/2)%2, r2=r/4
    int t0 = tid % 4;
    int t1 = (tid / 4) % 8;
    int t2 = tid / 32;

    #pragma unroll
    for (int mi = 0; mi < 2; mi++) {
        #pragma unroll
        for (int ni = 0; ni < 2; ni++) {
            int base_m = block_m + mi * WGMMA_M;
            int base_n = block_n + ni * WGMMA_N;

            #pragma unroll
            for (int r = 0; r < 32; r++) {
                int r0 = r % 2;
                int r1 = (r / 2) % 2;
                int r2 = r / 4;

                int flat = t0*128 + t1 + t2*16 + r0*64 + r1*8 + r2*512;
                int row  = flat % 64;
                int col  = flat / 64;

                int g_m = base_m + row;
                int g_n = base_n + col;

                if (g_m < dim_m && g_n < dim_n)
                    d_c[g_n * dim_m + g_m] = acc[mi][ni][r];
            }
        }
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

    kernel_wgmma<<<grid, block>>>(m, n, k, A, B, C2);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Kernel error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_wgmma<<<grid, block>>>(m, n, k, A, B, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tkernel = chrono::duration<double>(toc-tic).count() / Nt;
    double kernel_gflops = (double)num_flops / tkernel / 1e9;

    printf("CUBLAS: %.2f Gflops, WGMMA: %.2f Gflops (%.1f%% of cuBLAS)\n",
           cublas_gflops, kernel_gflops, 100.0 * kernel_gflops / cublas_gflops);

    double error = 0;
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            error += fabs(C[m*i+j] - C2[m*i+j]);
    printf("error: %lf\n", error/n/m);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C2);
    cublasDestroy(cublas_handle);
    return 0;
}