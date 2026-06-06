#include <iostream>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
#include <cuda_fp16.h>
using namespace std;

// WGMMA: 1 warpgroup = 4 warps = 128 threads
// Instruction: wgmma.mma_async m64n128k16
// Each warpgroup computes a 64x128 output tile per wgmma call

#define TILE_M 128
#define TILE_N 128  
#define TILE_K 64
#define WGMMA_M 64
#define WGMMA_N 128
#define WGMMA_K 16
#define PAD 8
#define BLOCK_THREADS 128  // 1 warpgroup

// Descriptor for wgmma shared memory matrix
__device__ __forceinline__ uint64_t make_smem_desc(void* ptr) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0;
    // Base address in bytes [13:0] → bits [13:0] of desc
    desc |= ((uint64_t)(addr >> 4) & 0x3FFF);
    // Stride: TILE_M * sizeof(half) / 16 bytes
    uint64_t stride = ((TILE_M + PAD) * sizeof(half)) >> 4;
    desc |= (stride & 0x3FFF) << 16;
    // Leading dim
    desc |= (stride & 0x3FFF) << 32;
    return desc;
}

__global__ void __launch_bounds__(BLOCK_THREADS, 2)
kernel_wgmma(int dim_m, int dim_n, int dim_k,
             const float* __restrict__ d_a,
             const float* __restrict__ d_b,
             float* __restrict__ d_c) {

    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;

    // Double-buffered shared memory
    __shared__ __align__(1024) half smem_a[2][TILE_K][TILE_M + PAD];
    __shared__ __align__(1024) half smem_b[2][TILE_K][TILE_N + PAD];

    int tid = threadIdx.x;
    int num_k_tiles = (dim_k + TILE_K - 1) / TILE_K;

    // Accumulator registers: wgmma m64n128k16 produces 64 float regs per warpgroup
    // For TILE_M=128, TILE_N=128: need 2x1 wgmma calls → 128 floats = 64 regs x2
    float acc[2][64];
    #pragma unroll
    for (int i = 0; i < 2; i++)
        #pragma unroll
        for (int j = 0; j < 64; j++)
            acc[i][j] = 0.f;

    // Load first tile
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

        // WGMMA over TILE_K/WGMMA_K steps
        // Two 64x128 tiles to cover 128x128 output
        #pragma unroll
        for (int ki = 0; ki < TILE_K / WGMMA_K; ki++) {
            uint64_t desc_a0 = make_smem_desc(&smem_a[cur][ki * WGMMA_K][0]);
            uint64_t desc_a1 = make_smem_desc(&smem_a[cur][ki * WGMMA_K][64]);
            uint64_t desc_b  = make_smem_desc(&smem_b[cur][ki * WGMMA_K][0]);

            // wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16
            // First 64 rows of M
            asm volatile(
                "{\n"
                ".reg .pred p;\n"
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,\n"
                " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,\n"
                " %32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,\n"
                " %48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
                " %64, %65, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(acc[0][0]),"+f"(acc[0][1]),"+f"(acc[0][2]),"+f"(acc[0][3]),
                  "+f"(acc[0][4]),"+f"(acc[0][5]),"+f"(acc[0][6]),"+f"(acc[0][7]),
                  "+f"(acc[0][8]),"+f"(acc[0][9]),"+f"(acc[0][10]),"+f"(acc[0][11]),
                  "+f"(acc[0][12]),"+f"(acc[0][13]),"+f"(acc[0][14]),"+f"(acc[0][15]),
                  "+f"(acc[0][16]),"+f"(acc[0][17]),"+f"(acc[0][18]),"+f"(acc[0][19]),
                  "+f"(acc[0][20]),"+f"(acc[0][21]),"+f"(acc[0][22]),"+f"(acc[0][23]),
                  "+f"(acc[0][24]),"+f"(acc[0][25]),"+f"(acc[0][26]),"+f"(acc[0][27]),
                  "+f"(acc[0][28]),"+f"(acc[0][29]),"+f"(acc[0][30]),"+f"(acc[0][31]),
                  "+f"(acc[0][32]),"+f"(acc[0][33]),"+f"(acc[0][34]),"+f"(acc[0][35]),
                  "+f"(acc[0][36]),"+f"(acc[0][37]),"+f"(acc[0][38]),"+f"(acc[0][39]),
                  "+f"(acc[0][40]),"+f"(acc[0][41]),"+f"(acc[0][42]),"+f"(acc[0][43]),
                  "+f"(acc[0][44]),"+f"(acc[0][45]),"+f"(acc[0][46]),"+f"(acc[0][47]),
                  "+f"(acc[0][48]),"+f"(acc[0][49]),"+f"(acc[0][50]),"+f"(acc[0][51]),
                  "+f"(acc[0][52]),"+f"(acc[0][53]),"+f"(acc[0][54]),"+f"(acc[0][55]),
                  "+f"(acc[0][56]),"+f"(acc[0][57]),"+f"(acc[0][58]),"+f"(acc[0][59]),
                  "+f"(acc[0][60]),"+f"(acc[0][61]),"+f"(acc[0][62]),"+f"(acc[0][63])
                : "l"(desc_a0), "l"(desc_b)
            );

            // Second 64 rows of M
            asm volatile(
                "{\n"
                "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16\n"
                "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,\n"
                " %16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,\n"
                " %32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,\n"
                " %48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63},\n"
                " %64, %65, 1, 1, 1, 0, 0;\n"
                "}\n"
                : "+f"(acc[1][0]),"+f"(acc[1][1]),"+f"(acc[1][2]),"+f"(acc[1][3]),
                  "+f"(acc[1][4]),"+f"(acc[1][5]),"+f"(acc[1][6]),"+f"(acc[1][7]),
                  "+f"(acc[1][8]),"+f"(acc[1][9]),"+f"(acc[1][10]),"+f"(acc[1][11]),
                  "+f"(acc[1][12]),"+f"(acc[1][13]),"+f"(acc[1][14]),"+f"(acc[1][15]),
                  "+f"(acc[1][16]),"+f"(acc[1][17]),"+f"(acc[1][18]),"+f"(acc[1][19]),
                  "+f"(acc[1][20]),"+f"(acc[1][21]),"+f"(acc[1][22]),"+f"(acc[1][23]),
                  "+f"(acc[1][24]),"+f"(acc[1][25]),"+f"(acc[1][26]),"+f"(acc[1][27]),
                  "+f"(acc[1][28]),"+f"(acc[1][29]),"+f"(acc[1][30]),"+f"(acc[1][31]),
                  "+f"(acc[1][32]),"+f"(acc[1][33]),"+f"(acc[1][34]),"+f"(acc[1][35]),
                  "+f"(acc[1][36]),"+f"(acc[1][37]),"+f"(acc[1][38]),"+f"(acc[1][39]),
                  "+f"(acc[1][40]),"+f"(acc[1][41]),"+f"(acc[1][42]),"+f"(acc[1][43]),
                  "+f"(acc[1][44]),"+f"(acc[1][45]),"+f"(acc[1][46]),"+f"(acc[1][47]),
                  "+f"(acc[1][48]),"+f"(acc[1][49]),"+f"(acc[1][50]),"+f"(acc[1][51]),
                  "+f"(acc[1][52]),"+f"(acc[1][53]),"+f"(acc[1][54]),"+f"(acc[1][55]),
                  "+f"(acc[1][56]),"+f"(acc[1][57]),"+f"(acc[1][58]),"+f"(acc[1][59]),
                  "+f"(acc[1][60]),"+f"(acc[1][61]),"+f"(acc[1][62]),"+f"(acc[1][63])
                : "l"(desc_a1), "l"(desc_b)
            );
        }
        asm volatile("wgmma.wait_group.sync.aligned 0;\n");
        __syncthreads();
    }

    // Store: each thread in warpgroup owns specific output elements
    // wgmma m64n128k16: thread t owns rows [t/4 + (t%4)*... ] — complex layout
    // Use wmma store via registers for simplicity
    int warp = tid / 32;
    int lane = tid % 32;

    // Each warp stores its portion: warp 0,1 → rows 0-63, warp 2,3 → rows 64-127
    int half_id = warp / 2;  // 0 = first 64 rows, 1 = second 64 rows
    // wgmma n128 layout: each thread owns 8 consecutive N elements per row group
    // rows: lane/4 + (warp%2)*8, cols: (lane%4)*2 + [0..7]*16
    int row_base = (lane / 4) + (warp % 2) * 8;
    
    #pragma unroll
    for (int n_outer = 0; n_outer < 8; n_outer++) {
        int col = (lane % 4) * 2 + n_outer * 16;
        int reg_base = n_outer * 8;
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int row = row_base + r * 8;  // stride within fragment
            int g_m = block_m + half_id * 64 + row;
            int g_n = block_n + col;
            if (g_m < dim_m && g_n + 1 < dim_n && g_m < dim_m) {
                d_c[g_n * dim_m + g_m]       = acc[half_id][reg_base + r*4];
                d_c[(g_n+1) * dim_m + g_m]   = acc[half_id][reg_base + r*4 + 1];
                d_c[(g_n+8) * dim_m + g_m]   = acc[half_id][reg_base + r*4 + 2];
                d_c[(g_n+9) * dim_m + g_m]   = acc[half_id][reg_base + r*4 + 3];
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
    // Reset and benchmark
    for (int i = 0; i < n; i++)
        for (int j = 0; j < m; j++)
            C2[m*i+j] = 0.0f;
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_wgmma<<<grid, block>>>(m, n, k, A, B, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tkernel = chrono::duration<double>(toc-tic).count() / Nt;
    double kernel_gflops = (double)num_flops / tkernel / 1e9;

    printf("CUBLAS: %.2f Gflops, WGMMA_KERNEL: %.2f Gflops (%.1f%% of cuBLAS)\n",
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