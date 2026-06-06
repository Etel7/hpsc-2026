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

// Taille de tuile agressive de 128x128 pour saturer le H100
const int TILE_M = 128;
const int TILE_N = 128;
const int TILE_K = 16;

__global__ void kernel_optimized_v2(int dim_m, int dim_n, int dim_k,
                                    const float * __restrict__ d_a, 
                                    const float * __restrict__ d_b, 
                                    float *d_c) {
    
    int warp_id = threadIdx.x / 32;

    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;

    // Mémoire partagée alignée avec Padding (+8) pour éliminer complètement les Bank Conflicts
    __shared__ half shmem_a[TILE_K][TILE_M + 8]; 
    __shared__ half shmem_b[TILE_K][TILE_N + 8];

    // Fragments d'accumulation (4x4 fragments de 16x16 = 64x64 par Warp)
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
    
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            wmma::fill_fragment(acc[r][c], 0.0f);
        }
    }

    // Coordonnées du Warp dans la tuile du bloc (Configuration 2x2 warps)
    int warp_m = (warp_id % 2) * 64;
    int warp_n = (warp_id / 2) * 64;

    // Variables pour le chargement vectorisé (128 threads chargent des lignes de la matrice)
    int tid = threadIdx.x;

    // Boucle principale sur la dimension K
    for (int k = 0; k < dim_k; k += TILE_K) {
        
        // 1. CHARGEMENT VECTORISÉ DE LA MATRICE A (Column-Major de base, traité pour la Shmem)
        // On configure les threads pour lire des paquets consécutifs
        int a_row_load = tid / 32; // 0 à 3
        int a_col_load = (tid % 32) * 4; // 0 à 124 (par pas de 4)

        if (block_m + a_col_load < dim_m) {
            #pragma unroll
            for (int r = 0; r < 4; r++) { // Multi-mapping pour couvrir les 16 lignes de K
                int cur_row = a_row_load + r * 4;
                if (k + cur_row < dim_k) {
                    // Chargement via float4 (Coalescence globale parfaite)
                    float4 tmp = reinterpret_cast<const float4*>(&d_a[(k + cur_row) * dim_m + block_m + a_col_load])[0];
                    
                    // Conversion directe et écriture en Shmem
                    shmem_a[cur_row][a_col_load + 0] = __float2half(tmp.x);
                    shmem_a[cur_row][a_col_load + 1] = __float2half(tmp.y);
                    shmem_a[cur_row][a_col_load + 2] = __float2half(tmp.z);
                    shmem_a[cur_row][a_col_load + 3] = __float2half(tmp.w);
                }
            }
        }

        // 2. CHARGEMENT VECTORISÉ DE LA MATRICE B (Row-Major)
        int b_row_load = tid / 32;
        int b_col_load = (tid % 32) * 4;

        if (block_n + b_col_load < dim_n) {
            #pragma unroll
            for (int r = 0; r < 4; r++) {
                int cur_row = b_row_load + r * 4;
                if (k + cur_row < dim_k) {
                    // Lecture mémoire ultra-rapide 128-bit
                    float4 tmp = reinterpret_cast<const float4*>(&d_b[(block_n + b_col_load) * dim_k + k + cur_row])[0];
                    
                    shmem_b[cur_row][b_col_load + 0] = __float2half(tmp.x);
                    shmem_b[cur_row][b_col_load + 1] = __float2half(tmp.y);
                    shmem_b[cur_row][b_col_load + 2] = __float2half(tmp.z);
                    shmem_b[cur_row][b_col_load + 3] = __float2half(tmp.w);
                }
            }
        }

        // Synchronisation : on attend que le pipeline mémoire globale -> Shmem soit fini
        __syncthreads();

        // 3. CALCULS TENSOR CORES (WMMA)
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
            wmma::load_matrix_sync(a_frag, &shmem_a[0][warp_m + r * 16], TILE_M + 8);

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &shmem_b[0][warp_n + c * 16], TILE_N + 8);

                // Multiplication-Accumulation synchrone matérielle
                wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
            }
        }
        
        // Synch avant de vider et recharger la Shmem pour l'itération K suivante
        __syncthreads();
    }

    // 4. ÉCRITURE DU RÉSULTAT FINAL DANS LA MATRICE C
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

    // Configuration optimale du Grid/Block pour la version V2
    dim3 block(128); // 4 Warps actifs
    dim3 grid((m + TILE_M - 1) / TILE_M, (n + TILE_N - 1) / TILE_N);

    // Benchmark de notre Kernel custom V2
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_optimized_v2<<< grid, block >>>(m, n, k, A, B, C2);
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

    // Recalcul de l'erreur
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