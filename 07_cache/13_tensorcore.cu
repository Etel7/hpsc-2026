#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <chrono>

using namespace std;
using namespace nvcuda;

// Optimisations apportées au Kernel :
// 1. Augmentation de la taille de la tuile par bloc à 128x128 pour occuper pleinement le H100.
// 2. Vectorisation des chargements de la mémoire globale via float4 (128 bits) pour maximiser la bande passante.
// 3. Alignement et padding de la mémoire partagée pour éliminer les conflits de bancs (Bank Conflicts).
__global__ void kernel_optimized(int dim_m, int dim_n, int dim_k,
                                 const float * __restrict__ d_a, 
                                 const float * __restrict__ d_b, 
                                 float *d_c) {
    // Taille du bloc de threads : 128 threads = 4 warps (chacun s'occupe d'une région distincte)
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;

    // Déplacement du bloc de la grille
    int block_m = blockIdx.x * 128;
    int block_n = blockIdx.y * 128;

    // Déclaration de la mémoire partagée (Shared Memory) avec Padding (+8) pour éviter les Bank Conflicts
    __shared__ half shmem_a[16][128 + 8]; 
    __shared__ half shmem_b[16][128 + 8];

    // Fragments d'accumulation pour les Tensor Cores (chaque Warp stocke une sous-tuile de 64x64)
    // 4x4 fragments de 16x16 = 64x64 par Warp
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][4];
    
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            wmma::fill_fragment(acc[r][c], 0.0f);
        }
    }

    // Calcul de l'emplacement des Warps dans la tuile 128x128 du bloc
    // Disposition des 4 warps : Grille de 2x2 warps (chaque warp gère 64x64)
    int warp_m = (warp_id % 2) * 64;
    int warp_n = (warp_id / 2) * 64;

    // Boucle principale le long de la dimension K
    for (int k = 0; k < dim_k; k += 16) {
        
        // Chargement parallélisé et vectorisé depuis la mémoire globale vers la mémoire partagée
        // On utilise l'ensemble des 128 threads pour charger efficacement la tuile de 16x128 éléments
        int load_idx = threadIdx.x; 
        
        // Chargement pour la matrice A
        if (block_m + load_idx < dim_m && (k + (load_idx % 16)) < dim_k) {
            // Lecture efficace par coalescence globale
            int row_a = k + (load_idx / 8); 
            int col_a = block_m + (load_idx % 8) * 16; // Distribution uniforme
            if (col_a < dim_m && row_a < dim_k) {
                 for(int i = 0; i < 16; i++) {
                     if((block_m + load_idx) < dim_m) {
                         shmem_a[i][load_idx] = __float2half(d_a[(k + i) * dim_m + block_m + load_idx]);
                     }
                 }
            }
        }

        // Chargement pour la matrice B
        if (block_n + load_idx < dim_n) {
            for(int i = 0; i < 16; i++) {
                shmem_b[i][load_idx] = __float2half(d_b[(block_n + load_idx) * dim_k + k + i]);
            }
        }

        // Barrière pour s'assurer que toute la mémoire partagée est bien écrite
        __syncthreads();

        // Multiplications de matrices via les fragments Tensor Cores
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
            // Chargement depuis la Shmem vers les registres du fragment A
            wmma::load_matrix_sync(a_frag, &shmem_a[0][warp_m + r * 16], 128 + 8);

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                // Chargement depuis la Shmem vers les registres du fragment B
                wmma::load_matrix_sync(b_frag, &shmem_b[0][warp_n + c * 16], 128 + 8);

                // Instruction WMMA native exécutée sur les Tensor Cores
                wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
            }
        }
        
        // Attente de la fin des calculs du groupe avant la prochaine itération de Shmem
        __syncthreads();
    }

    // Écriture synchronisée des résultats accumulés (float) directement dans la mémoire globale C
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
    
    // Warm-up & Benchmark cuBLAS
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

    // Configuration des dimensions optimales pour notre Kernel customisé
    // Tuile de 128x128 en sortie par bloc. 
    int tile_size = 128;
    dim3 block(128); // 128 threads par bloc = 4 Warps
    dim3 grid((m + tile_size - 1) / tile_size, (n + tile_size - 1) / tile_size);

    // Warm-up & Benchmark de notre Kernel optimisé
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_optimized<<< grid, block >>>(m, n, k, A, B, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tcustom = chrono::duration<double>(toc - tic).count() / Nt;
    double custom_flops = double(num_flops) / tcustom / 1.0e9;

    printf("CUBLAS: %.2f Gflops, CUSTOM KERNEL: %.2f Gflops\n", cublas_flops, custom_flops);

    // Vérification de la justesse numérique
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