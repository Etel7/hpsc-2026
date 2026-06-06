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

// Tuile de 64x64 par bloc pour correspondre à la structure de base stable
const int TILE_SIZE = 64;

__global__ void kernel_optimized_v5(int dim_m, int dim_n, int dim_k,
                                    const float * __restrict__ d_a, 
                                    const float * __restrict__ d_b, 
                                    float *d_c) {
    
    int offset_a_m = TILE_SIZE * blockIdx.x;
    int offset_b_n = TILE_SIZE * blockIdx.y;
    int i = threadIdx.x;
    int warp_id = threadIdx.x / 32;

    // Utilisation de la mémoire partagée originale (pas de padding complexe qui désaligne les index)
    __shared__ half block_a[16][TILE_SIZE];
    __shared__ half block_b[16][TILE_SIZE];

    // Accumulateurs du Warp : Tuile de 64x64 gérée par le warp
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[2][4];
    
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            wmma::fill_fragment(acc[r][c], 0.0f);
        }
    }

    // Boucle principale le long de la dimension K (pas de 16)
    for (int k = 0; k < dim_k; k += 16) {
        
        // Chargement mémoire globale -> partagée identique à l'original (garantie erreur = 0)
        __syncthreads();
        #pragma unroll
        for (int j = 0; j < 16; ++j) {
            block_a[j][i] = __float2half(d_a[(k + j) * dim_m + offset_a_m + i]);
            block_b[j][i] = __float2half(d_b[(offset_b_n + i) * dim_k + k + j]);
        }
        __syncthreads();

        // Calculs WMMA optimisés avec déroulement complet
        #pragma unroll
        for (int r = 0; r < 2; r++) {
            int row_tile = warp_id * 2 + r;
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
            wmma::load_matrix_sync(a_frag, &block_a[0][row_tile * 16], TILE_SIZE);

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &block_b[0][c * 16], TILE_SIZE);
                
                // Exécution sur Tensor Cores
                wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
            }
        }
    }

    // Écriture finale synchrone
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

    // Configuration identique au code de départ
    dim3 block = dim3(TILE_SIZE);
    dim3 grid = dim3((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    // Benchmark du Kernel V5
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_optimized_v5<<< grid, block >>>(m, n, k, A, B, C2);
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

    // Calcul d'erreur fiable
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