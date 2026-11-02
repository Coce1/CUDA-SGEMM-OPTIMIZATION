#include "../include/sgemm.cuh"
#include "../include/cuda_utils.cuh"

__global__ void sgemmGPUNaive(const float* __restrict__ A, const float* __restrict__ B,
                              float* __restrict__ C, int N) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x; // colonne
    const int y = blockIdx.y * blockDim.y + threadIdx.y; // ligne

    if (x < N && y < N) {
        float sum = 0.0f;
        // threadIdx.x -> colonne : à k fixé, un warp lit 32 éléments contigus de B
        // (coalescé) et un seul élément de A (broadcast).
        for (int k = 0; k < N; ++k) {
            sum += A[static_cast<size_t>(y) * N + k] * B[static_cast<size_t>(k) * N + x];
        }
        C[static_cast<size_t>(y) * N + x] = sum;
    }
}

void launchSgemmNaive(const float* d_A, const float* d_B, float* d_C, int N) {
    // Bloc 32x8 : un warp couvre 32 colonnes d'une même ligne (128 o contigus),
    // alors qu'un bloc 16x16 répartit un warp sur 2 lignes (2 segments de 64 o).
    const dim3 block(32, 8);
    const dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
    sgemmGPUNaive<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK_KERNEL();
}
