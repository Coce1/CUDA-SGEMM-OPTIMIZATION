#include "../include/sgemm.cuh"

void sgemmCPU(const float* A, const float* B, float* C, int N) {
    for (int y = 0; y < N; ++y) {
        for (int x = 0; x < N; ++x) {
            float sum = 0.0f;
            for (int k = 0; k < N; ++k) {
                // Produit scalaire : ligne y de A * colonne x de B
                sum += A[y * N + k] * B[k * N + x];
            }
            C[y * N + x] = sum;
        }
    }
}