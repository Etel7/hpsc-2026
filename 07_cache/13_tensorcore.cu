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

const int TILE_M = 128;
const int TILE_N = 128;
const int TILE_K = 16;

__global__ void kernel_optimized_v4(int dim_m, int dim_n, int dim_k,
                                    const float * __restrict__ d_a, 
                                    const float * __restrict__ d_b, 
                                    float *d_c) {
    
    int warp_id = threadIdx.x / 32;

    int block_m = blockIdx.x * TILE_M;
    int block_n = blockIdx.y * TILE_N;

    // Déclaration de la mémoire partagée
    // shmem_a est indexée en [TILE_M + 8][TILE_K] pour correspondre au chargement col-major originel
    __shared__ half shmem_a[TILE_M + 8][TILE_K]; 
    __shared__ half shmem_b[TILE_K][TILE_N + 8];

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

    // Itération sur la dimension K
    for (int k = 0; k < dim_k; k += TILE_K) {
        
        // --- CHARGEMENT STRICTEMENT COALESCÉ DE LA MATRICE A (Column-Major) ---
        // On calque exactement la logique de ton code original, mais parallélisée sur 128 threads
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int thread_m = threadIdx.x; // 0 à 127 -> couvre exactement TILE_M
            int global_m = block_m + thread_m;
            int global_k = k + i;
            
            if (global_m < dim_m && global_k < dim_k) {
                shmem_a[thread_m][i] = __float2half(d_a[global_k * dim_m + global_m]);
            } else {
                shmem_a[thread_m][i] = __float2half(0.0f);
            }
        }

        // --- CHARGEMENT STRICTEMENT COALESCÉ DE LA MATRICE B (Row-Major) ---
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            int thread_n = threadIdx.x; // 0 à 127 -> couvre exactement TILE_N
            int global_n = block_n + thread_n;
            int global_k = k + i;
            
            if (global_n < dim_n && global_k < dim_k) {
                shmem_b[i][thread_n] = __float2half(d_b[global_n * dim_k + global_k]);
            } else {
                shmem_b[i][thread_n] = __float2half(0.0f);
            }
        }

        // Synchronisation globale du bloc après remplissage de la Shmem
        __syncthreads();

        // --- CALCULS SUR LES TENSOR CORES VIA WMMA ---
        #pragma unroll
        for (int r = 0; r < 4; r++) {
            wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
            // On charge depuis shmem_a qui respecte le format col_major (stride = TILE_K)
            wmma::load_matrix_sync(a_frag, &shmem_a[warp_m + r * 16][0], TILE_K);

            #pragma unroll
            for (int c = 0; c < 4; c++) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::row_major> b_frag;
                wmma::load_matrix_sync(b_frag, &shmem_b[0][warp_n + c * 16], TILE_N + 8);

                // Multiplication matérielle
                wmma::mma_sync(acc[r][c], a_frag, b_frag, acc[r][c]);
            }
        }
        
        // Synchronisation avant l'étape K suivante
        __syncthreads();
    }

    // --- ÉCRITURE FINALE VERS LA MATRICE C ---
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

    // Configuration géométrique optimale
    dim3 block(128); 
    dim3 grid((m + TILE_M - 1) / TILE_M, (n + TILE_N - 1) / TILE_N);

    // Benchmark du Kernel V4
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_optimized_v4<<< grid, block >>>(m, n, k, A, B, C2);
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

    // Recalcul strict de l'erreur
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