#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>
#include <cmath>

using namespace std;
using namespace nvcuda;

const int TILE_SIZE = 64;

__global__ void kernel_optimized_v6(int dim_m, int dim_n, int dim_k,
                                    const float * __restrict__ d_a, 
                                    const float * __restrict__ d_b, 
                                    float *d_c) {
    
    int offset_a_m = TILE_SIZE * blockIdx.x;
    int offset_b_n = TILE_SIZE * blockIdx.y;
    int warp_id = threadIdx.x / 32;

    // Mémoire partagée alignée
    __shared__ half block_a[16][TILE_SIZE];
    __shared__ half block_b[16][TILE_SIZE];

    // Accumulateurs de fragments WMMA
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            wmma::fill_fragment(acc[r][c], 0.0f);
        }
    }

    // Boucle principale le long de la dimension K
    for (int k = 0; k < dim_k; k += 16) {
        
        // --- CHARGEMENT VECTORISÉ 128-BIT (FLOAT4) DE LA MATRICE A ---
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int row_k = step * 4 + (threadIdx.x / 16); 
            int col_m_quad = threadIdx.x % 16;        
            int col_m = col_m_quad * 4;
            
            int global_m = offset_a_m + col_m;
            int global_k = k + row_k;
            
            if (global_m < dim_m && global_k < dim_k) {
                // Lecture coalescée de 16 octets d'un coup
                float4 tmp = reinterpret_cast<const float4*>(&d_a[global_k * dim_m + global_m])[0];
                block_a[row_k][col_m + 0] = __float2half(tmp.x);
                block_a[row_k][col_m + 1] = __float2half(tmp.y);
                block_a[row_k][col_m + 2] = __float2half(tmp.z);
                block_a[row_k][col_m + 3] = __float2half(tmp.w);
            } else {
                block_a[row_k][col_m + 0] = __float2half(0.0f);
                block_a[row_k][col_m + 1] = __float2half(0.0f);
                block_a[row_k][col_m + 2] = __float2half(0.0f);
                block_a[row_k][col_m + 3] = __float2half(0.0f);
            }
        }

        // --- CHARGEMENT VECTORISÉ 128-BIT (FLOAT4) DE LA MATRICE B ---
        #pragma unroll
        for (int step = 0; step < 4; ++step) {
            int global_f4_idx = threadIdx.x + step * 64;
            int col_n = global_f4_idx / 4; 
            int row_k_quad = global_f4_idx % 4; 
            int row_k = row_k_quad * 4; 
            
            int global_n = offset_b_n + col_n;
            int global_k = k + row_k;
            
            if (global_n < dim_n && global_k < dim_k) {
                // Alignement parfait respectant l'axe contigu de B
                float4 tmp = reinterpret_cast<const float4*>(&d_b[global_n * dim_k + global_k])[0];
                block_b[row_k + 0][col_n] = __float2half(tmp.x);
                block_b[row_k + 1][col_n] = __float2half(tmp.y);
                block_b[row_k + 2][col_n] = __float2half(tmp.z);
                block_b[row_k + 3][col_n] = __float2half(tmp.w);
            } else {
                block_b[row_k + 0][col_n] = __float2half(0.0f);
                block_b[row_k + 1][col_n] = __float2half(0.0f);
                block_b[row_k + 2][col_n] = __float2half(0.0f);
                block_b[row_k + 3][col_n] = __float2half(0.0f);
            }
        }

        // Synchronisation des warps après le chargement des tuiles
        __syncthreads();

        // --- CALCULS WMMA INTENSIFS ---
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int row_tile = warp_id * 2 + r;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
            wmma::load_matrix_sync(a_frag, &block_a[0][row_tile * 16], TILE_SIZE);

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &block_b[0][c * 16], TILE_SIZE);
                
                wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
            }
        }
        __syncthreads();
    }

    // --- ÉCRITURE FINALE ---
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int c_m = offset_a_m + (warp_id * 2 + r) * 16;
            int c_n = offset_b_n + c * 16;
            if (c_n < dim_n && c_m < dim_m) {
                wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
            }
        }
    }
}

int main(int argc, const char **argv) {
    int m = 10240;
    int k = 4096;
    int n = 8192;
    float alpha = 1.0;
    float beta = 0.0;
    int Nt = 10;
    
    float *A, *B, *C, *C2;
    cudaMallocManaged(&A, m * k * sizeof(float));
    cudaMallocManaged(&B, k * n * sizeof(float));
    cudaMallocManaged(&C, m * n * sizeof(float));
    cudaMallocManaged(&C2, m * n * sizeof(float));
    
    srand48(42); 
    for (int i=0; i<m; i++)
        for (int j=0; j<k; j++)
            A[k*i+j] = drand48();
    for (int i=0; i<k; i++)
        for (int j=0; j<n; j++)
            B[n*i+j] = drand48();
    for (int i=0; i<n; i++)
        for (int j=0; j<m; j++)
            C[m*i+j] = C2[m*i+j] = 0;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    // Benchmark cuBLAS
    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        cublasGemmEx(cublas_handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     m, n, k,
                     &alpha,
                     A, CUDA_R_32F, m,
                     B, CUDA_R_32F, k,
                     &beta,
                     C, CUDA_R_32F, m,
                     CUBLAS_COMPUTE_32F_FAST_16F,
                     CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();
    int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
    double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
    double cublas_flops = double(num_flops) / tcublas / 1.0e9;

    dim3 block = dim3(TILE_SIZE);
    dim3 grid = dim3((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // Benchmark Kernel Vectorisé
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_optimized_v6<<< grid, block >>>(m, n, k, A, B, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tcustom = chrono::duration<double>(toc - tic).count() / Nt;
    double custom_flops = double(num_flops) / tcustom / 1.0e9;

    double eff_percentage = (custom_flops / cublas_flops) * 100.0;

    printf("===================================================\n");
    printf("CUBLAS       : %.2f Gflops\n", cublas_flops);
    printf("CUSTOM KERNEL: %.2f Gflops\n", custom_flops);
    printf("PERFORMANCE  : %.2f%% de cuBLAS\n", eff_percentage);
    printf("===================================================\n");

    double err = 0;
    for (int i=0; i<n; i++) {
        for (int j=0; j<m; j++) {
            err += fabs(C[m*i+j] - C2[m*i+j]);
        }
    }
    printf("error: %lf\n", err/n/m);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaFree(C2);
    cublasDestroy(cublas_handle);
    return 0;
}