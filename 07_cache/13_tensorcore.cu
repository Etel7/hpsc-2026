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

// Kernel CUDA optimisé avec chargement correct et alignement Shmem
__global__ void kernel_optimized(int dim_m, int dim_n, int dim_k,
                                 const float * __restrict__ d_a, 
                                 const float * __restrict__ d_b, 
                                 float *d_c) {
    
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    int block_m = blockIdx.x * 128;
    int block_n = blockIdx.y * 128;

    // Mémoire partagée avec Padding (+8) pour éliminer les conflits de bancs (Bank Conflicts)
    __shared__ half shmem_a[16][128 + 8]; 
    __shared__ half shmem_b[16][128 + 8];

    // Fragments d'accumulation (4x4 fragments de 16x16 = tuile de 64x64 par Warp)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
    
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            wmma::fill_fragment(acc[r][c], 0.0f);
        }
    }

    // Grille interne de 2x2 warps distribuant la tuile de 128x128
    int warp_m = (warp_id % 2) * 64;
    int warp_n = (warp_id / 2) * 64;

    // Boucle principale sur l'axe K
    for (int k = 0; k < dim_k; k += 16) {
        
        // Chargement coalescé pour la matrice A (Column-Major)
        // Chaque thread charge des éléments spécifiques pour éviter les divergences
        for (int row = threadIdx.x / 8; row < 16; row += 16) { 
            int col_offset = threadIdx.x % 8;
            for (int i = col_offset; i < 128; i += 8) {
                int global_m = block_m + i;
                int global_k = k + row;
                if (global_m < dim_m && global_k < dim_k) {
                    shmem_a[row][i] = __float2half(d_a[global_k * dim_m + global_m]);
                } else {
                    shmem_a[row][i] = __float2half(0.0f);
                }
            }
        }

        // Chargement coalescé pour la matrice B (Row-Major)
        for (int row = threadIdx.x / 8; row < 16; row += 16) {
            int col_offset = threadIdx.x % 8;
            for (int i = col_offset; i < 128; i += 8) {
                int global_n = block_n + i;
                int global_k = k + row;
                if (global_n < dim_n && global_k < dim_k) {
                    shmem_b[row][i] = __float2half(d_b[global_n * dim_k + global_k]);
                } else {
                    shmem_b[row][i] = __float2half(0.0f);
                }
            }
        }

        // Synchronisation requise : attente que la Shmem soit complètement remplie
        __syncthreads();

        // Multiplications Tensor Cores via WMMA
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
            wmma::load_matrix_sync(a_frag, &shmem_a[0][warp_m + r * 16], 128 + 8);

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &shmem_b[0][warp_n + c * 16], 128 + 8);

                // Calcul synchrone sur Tensor Cores
                wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
            }
        }
        
        // Synchronisation avant de charger la sous-tuile K suivante
        __syncthreads();
    }

    // Écriture finale des résultats dans la matrice globale C
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int final_m = block_m + warp_m + r * 16;
            int final_n = block_n + warp_n + c * 16;
            
            if (final_m < dim_m && final_n < dim_n) {
                wmma::store_matrix_sync(&d_c[final_n * dim_m + final_m], acc[r][c], dim_m, wmma::mem_col_major);
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
    
    // Initialisation des données
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

    // Configuration de notre Kernel
    int tile_size = 128;
    dim3 block(128); // 4 Warps
    dim3 grid((m + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size);

    // Benchmark du Kernel Custom
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_optimized<<< grid, block >>>(m, n, k, A, B, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tcustom = chrono::duration<double>(toc - tic).count() / Nt;
    double custom_flops = double(num_flops) / tcustom / 1.0e9;

    // Calcul du pourcentage d'efficacité par rapport à cuBLAS
    double eff_percentage = (custom_flops / cublas_flops) * 100.0;

    printf("===================================================\n");
    printf("CUBLAS       : %.2f Gflops\n", cublas_flops);
    printf("CUSTOM KERNEL: %.2f Gflops\n", custom_flops);
    printf("PERFORMANCE  : %.2f%% de cuBLAS\n", eff_percentage);
    printf("===================================================\n");

    // Calcul de l'erreur numérique moyenne
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