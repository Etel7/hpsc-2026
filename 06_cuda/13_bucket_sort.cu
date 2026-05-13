#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cuda_runtime.h>

__global__ void count_kernel(int* key, int* bucket, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n)
    atomicAdd(&bucket[key[i]], 1);
}

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  int *d_key, *d_bucket;
  cudaMalloc(&d_key, n * sizeof(int));
  cudaMalloc(&d_bucket, range * sizeof(int));

  cudaMemcpy(d_key, key.data(), n * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemset(d_bucket, 0, range * sizeof(int));

  int threads = 256;
  int blocks = (n + threads - 1) / threads;
  count_kernel<<<blocks, threads>>>(d_key, d_bucket, n);

  std::vector<int> bucket(range);
  cudaMemcpy(bucket.data(), d_bucket, range * sizeof(int), cudaMemcpyDeviceToHost);

  cudaFree(d_key);
  cudaFree(d_bucket);

  for (int i=0, j=0; i<range; i++) {
    for (; bucket[i]>0; bucket[i]--) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");
}