#include <iostream>
#include <typeinfo>
#include <random>
#include <stdint.h>
#include <cublas_v2.h>
#include <mma.h>
#include <cuda_fp16.h> 
#include <chrono>
#include <cmath>

using namespace std;
using namespace nvcuda;

const int TILE_SIZE = 128;

__global__ void kernel_optimized_v12(int dim_m, int dim_n, int dim_k,
                                     const float * __restrict__ d_a, 
                                     const float * __restrict__ d_b, 
                                     float *d_c) {
    
    int offset_a_m = TILE_SIZE * blockIdx.x;
    int offset_b_n = TILE_SIZE * blockIdx.y;
    
    int warp_id = threadIdx.x / 32;
    int warp_m = warp_id / 4; 
    int warp_n = warp_id % 4; 

    // Augmentation de la taille K à 32 pour l'alignement des lignes de cache L2
    __shared__ half block_a[2][32][TILE_SIZE + 8];
    __shared__ half block_b[2][TILE_SIZE][32 + 8]; 

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc[4][2];
    
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            wmma::fill_fragment(acc[r][c], 0.0f);
        }
    }

    int write_stage = 0;

    // On avance de 32 en 32
    for (int k = 0; k < dim_k + 32; k += 32) {
        
        // 1. CHARGEMENT GLOBAL -> PARTAGÉ (4 étapes pour couvrir K=32 avec 256 threads)
        if (k < dim_k) {
            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                int total_f4 = threadIdx.x + step * 256; 
                int row_k = total_f4 / 32;               
                int col_m = (total_f4 % 32) * 4;         
                
                int global_m = offset_a_m + col_m;
                int global_k = k + row_k;
                
                if (global_m < dim_m && global_k < dim_k) {
                    float4 tmp = reinterpret_cast<const float4*>(&d_a[global_k * dim_m + global_m])[0];
                    block_a[write_stage][row_k][col_m + 0] = __float2half(tmp.x);
                    block_a[write_stage][row_k][col_m + 1] = __float2half(tmp.y);
                    block_a[write_stage][row_k][col_m + 2] = __float2half(tmp.z);
                    block_a[write_stage][row_k][col_m + 3] = __float2half(tmp.w);
                } else {
                    block_a[write_stage][row_k][col_m + 0] = __float2half(0.0f);
                    block_a[write_stage][row_k][col_m + 1] = __float2half(0.0f);
                    block_a[write_stage][row_k][col_m + 2] = __float2half(0.0f);
                    block_a[write_stage][row_k][col_m + 3] = __float2half(0.0f);
                }
            }

            #pragma unroll
            for (int step = 0; step < 4; ++step) {
                int total_f4 = threadIdx.x + step * 256; 
                int row_k = (total_f4 % 8) * 4;          // 8 threads consécutifs lisent 32 floats d'une colonne 
                int col_n = total_f4 / 8;                
                
                int global_n = offset_b_n + col_n;
                int global_k = k + row_k;
                
                if (global_n < dim_n && global_k < dim_k) {
                    float4 tmp = reinterpret_cast<const float4*>(&d_b[global_n * dim_k + global_k])[0];
                    // Transaction mémoire de 128 octets parfaite par colonne lue
                    block_b[write_stage][col_n][row_k + 0] = __float2half(tmp.x);
                    block_b[write_stage][col_n][row_k + 1] = __float2half(tmp.y);
                    block_b[write_stage][col_n][row_k + 2] = __float2half(tmp.z);
                    block_b[write_stage][col_n][row_k + 3] = __float2half(tmp.w);
                } else {
                    block_b[write_stage][col_n][row_k + 0] = __float2half(0.0f);
                    block_b[write_stage][col_n][row_k + 1] = __float2half(0.0f);
                    block_b[write_stage][col_n][row_k + 2] = __float2half(0.0f);
                    block_b[write_stage][col_n][row_k + 3] = __float2half(0.0f);
                }
            }
        }

        __syncthreads();

        // 2. CALCULS TENSOR CORES (Déroulés sur 2 sous-étapes de 16)
        if (k > 0) {
            int read_stage = 1 - write_stage;

            #pragma unroll
            for (int ki = 0; ki < 2; ki++) {
                int k_offset = ki * 16;

                wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag[2];
                #pragma unroll
                for (int c = 0; c < 2; c++) {
                    int col_tile = warp_n * 2 + c;
                    wmma::load_matrix_sync(b_frag[c], &block_b[read_stage][col_tile * 16][k_offset], 32 + 8);
                }

                #pragma unroll
                for (int r = 0; r < 4; r++) {
                    int row_tile = warp_m * 4 + r;
                    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::col_major> a_frag;
                    wmma::load_matrix_sync(a_frag, &block_a[read_stage][k_offset][row_tile * 16], TILE_SIZE + 8);

                    #pragma unroll
                    for (int c = 0; c < 2; c++) {
                        wmma::mma_sync(acc[r][c], a_frag, b_frag[c], acc[r][c]);
                    }
                }
            }
        }
        
        write_stage = 1 - write_stage;
    }

    // ÉCRITURE FINALE
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 2; c++) {
            int c_m = offset_a_m + (warp_m * 4 + r) * 16;
            int c_n = offset_b_n + (warp_n * 2 + c) * 16;
            if (c_n < dim_n && c_m < dim_m) {
                wmma::store_matrix_sync(&d_c[c_n * dim_m + c_m], acc[r][c], dim_m, wmma::mem_col_major);
            }
        }
    }
}

int main(int argc, const char **argv) {
    int m = 10240; int k = 4096; int n = 8192;
    float alpha = 1.0; float beta = 0.0; int Nt = 10;
    
    float *A, *B, *C, *C2;
    cudaMallocManaged(&A, m * k * sizeof(float));
    cudaMallocManaged(&B, k * n * sizeof(float));
    cudaMallocManaged(&C, m * n * sizeof(float));
    cudaMallocManaged(&C2, m * n * sizeof(float));
    
    srand48(42); 
    for (int i=0; i<m; i++) for (int j=0; j<k; j++) A[k*i+j] = drand48();
    for (int i=0; i<k; i++) for (int j=0; j<n; j++) B[n*i+j] = drand48();
    for (int i=0; i<n; i++) for (int j=0; j<m; j++) C[m*i+j] = C2[m*i+j] = 0;

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);
    
    auto tic = chrono::steady_clock::now();
    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        cublasGemmEx(cublas_handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha, A, CUDA_R_32F, m, B, CUDA_R_32F, k, &beta, C, CUDA_R_32F, m, CUBLAS_COMPUTE_32F_FAST_16F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        cudaDeviceSynchronize();
    }
    auto toc = chrono::steady_clock::now();
    int64_t num_flops = (2 * int64_t(m) * int64_t(n) * int64_t(k)) + (2 * int64_t(m) * int64_t(n));
    double tcublas = chrono::duration<double>(toc - tic).count() / Nt;
    double cublas_flops = double(num_flops) / tcublas / 1.0e9;

    dim3 block = dim3(256);
    dim3 grid = dim3((m + TILE_SIZE - 1) / TILE_SIZE, (n + TILE_SIZE - 1) / TILE_SIZE);

    for (int i = 0; i < Nt+2; i++) {
        if (i == 2) tic = chrono::steady_clock::now();
        kernel_optimized_v12<<< grid, block >>>(m, n, k, A, B, C2);
        cudaDeviceSynchronize();
    }
    toc = chrono::steady_clock::now();
    double tcustom = chrono::duration<double>(toc - tic).count() / Nt;
    double custom_flops = double(num_flops) / tcustom / 1.0e9;

    printf("===================================================\n");
    printf("CUBLAS       : %.2f Gflops\n", cublas_flops);
    printf("CUSTOM KERNEL: %.2f Gflops\n", custom_flops);
    printf("PERFORMANCE  : %.2f%% de cuBLAS\n", (custom_flops / cublas_flops) * 100.0);
    printf("===================================================\n");

    double err = 0;
    for (int i=0; i<n; i++) for (int j=0; j<m; j++) err += fabs(C[m*i+j] - C2[m*i+j]);
    printf("error: %lf\n", err/n/m);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(C2);
    cublasDestroy(cublas_handle);
    return 0;
}