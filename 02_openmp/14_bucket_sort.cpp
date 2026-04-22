#include <cstdio>
#include <cstdlib>
#include <vector>
#include <omp.h>

int main() {
  int n = 50;
  int range = 5;
  std::vector<int> key(n);
  for (int i=0; i<n; i++) {
    key[i] = rand() % range;
    printf("%d ", key[i]);
  }
  printf("\n");

  int num_threads;
  std::vector<int> bucket(range, 0);

  // Comptage parallèle avec reduction
  #pragma omp parallel
  {
    #pragma omp single
    num_threads = omp_get_num_threads();

    std::vector<int> local_bucket(range, 0);

    #pragma omp for
    for (int i=0; i<n; i++)
      local_bucket[key[i]]++;

    #pragma omp critical
    for (int i=0; i<range; i++)
      bucket[i] += local_bucket[i];
  }

  // Calcul des offsets (séquentiel, dépendance de données)
  std::vector<int> offset(range, 0);
  for (int i=1; i<range; i++)
    offset[i] = offset[i-1] + bucket[i-1];

  // Réécriture parallèle
  #pragma omp parallel for
  for (int i=0; i<range; i++) {
    int j = offset[i];
    for (int k=0; k<bucket[i]; k++) {
      key[j++] = i;
    }
  }

  for (int i=0; i<n; i++) {
    printf("%d ", key[i]);
  }
  printf("\n");
}