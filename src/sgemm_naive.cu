#include "../include/sgemm.cuh"

__global__ void sgemmGPUNaive(const float* A, const float* B, float* C, int N) {
    // Calcul des coordonnées 2D du thread
    int x = blockIdx.x * blockDim.x + threadIdx.x; // Colonne
    int y = blockIdx.y * blockDim.y + threadIdx.y; // Ligne

    // Vérification des limites de la matrice
    if (x < N && y < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; ++k) {
            sum += A[y * N + k] * B[k * N + x];
        }
        C[y * N + x] = sum;
    }
}