#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 16;
  float x[N], y[N], m[N], fx[N], fy[N];
  for(int i=0; i<N; i++) {
    x[i] = (float)rand() / RAND_MAX;
    y[i] = (float)rand() / RAND_MAX;
    m[i] = (float)rand() / RAND_MAX;
    fx[i] = fy[i] = 0;
  }

  for(int i=0; i<N; i++) {
    __m256 xi = _mm256_set1_ps(x[i]);
    __m256 yi = _mm256_set1_ps(y[i]);

    float fxi = 0, fyi = 0;

    // 2 passes de 8 (N=16)
    for(int block=0; block<2; block++) {
      __m256 xj = _mm256_loadu_ps(x + block*8);
      __m256 yj = _mm256_loadu_ps(y + block*8);
      __m256 mj = _mm256_loadu_ps(m + block*8);

      __m256 rx = _mm256_sub_ps(xi, xj);
      __m256 ry = _mm256_sub_ps(yi, yj);
      __m256 r2 = _mm256_add_ps(_mm256_mul_ps(rx, rx), _mm256_mul_ps(ry, ry));

      // Evite division par 0 pour i==j
      __m256 eps = _mm256_set1_ps(1e-10f);
      r2 = _mm256_add_ps(r2, eps);

      __m256 rinv  = _mm256_rsqrt_ps(r2);
      __m256 rinv2 = _mm256_mul_ps(rinv, rinv);
      __m256 rinv3 = _mm256_mul_ps(rinv2, rinv);

      __m256 dfx = _mm256_mul_ps(rx, _mm256_mul_ps(mj, rinv3));
      __m256 dfy = _mm256_mul_ps(ry, _mm256_mul_ps(mj, rinv3));

      // Réduction horizontale
      __m256 sumx = _mm256_hadd_ps(dfx, dfx);
      sumx = _mm256_hadd_ps(sumx, sumx);
      fxi -= ((float*)&sumx)[0] + ((float*)&sumx)[4];

      __m256 sumy = _mm256_hadd_ps(dfy, dfy);
      sumy = _mm256_hadd_ps(sumy, sumy);
      fyi -= ((float*)&sumy)[0] + ((float*)&sumy)[4];
    }

    fx[i] = fxi;
    fy[i] = fyi;
  }

  for(int i=0; i<N; i++) {
    printf("%d %g %g\n", i, fx[i], fy[i]);
  }
}