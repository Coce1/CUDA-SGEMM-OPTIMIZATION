#ifndef SGEMM_CUH
#define SGEMM_CUH

// 1. Référence CPU (Visible par le compilateur C++ standard et nvcc)
void sgemmCPU(const float* A, const float* B, float* C, int N);

// 2. Kernels GPU (Visibles uniquement par nvcc)
#ifdef __CUDACC__
__global__ void sgemmGPUNaive(const float* A, const float* B, float* C, int N);
__global__ void sgemmGPUShared(const float* A, const float* B, float* C, int N);
#endif

#endif