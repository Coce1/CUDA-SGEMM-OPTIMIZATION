#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "../include/sgemm.cuh"

int main() {
    // Massive matrix size: 4096 x 4096 (~67 MB per matrix)
    int N = 4096; 
    size_t bytes = N * N * sizeof(float);
    
    std::cout << "=== SGEMM Benchmark Extreme (" << N << "x" << N << ") ===" << std::endl;

    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_gpu(N * N, 0.0f);

    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    // Calculate total floating-point operations (N:int to double for limit of integer overflow)
    double flops = 2.0 * static_cast<double>(N) * N * N;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEvent_t start_gpu, stop_gpu;
    cudaEventCreate(&start_gpu);
    cudaEventCreate(&stop_gpu);

    // 1. Naive GPU Benchmark (Global Memory)
    std::cout << "\n[GPU] Version 1: Naive (Global Memory)..." << std::endl;
    cudaEventRecord(start_gpu);
    sgemmGPUNaive<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float ms_naive = 0;
    cudaEventElapsedTime(&ms_naive, start_gpu, stop_gpu);
    double sec_naive = ms_naive / 1000.0;
    
    // Convert to Tera-FLOPS (1e12) for HPC scale
    double tflops_naive = (flops / sec_naive) / 1e12; 
    
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "- Time: " << sec_naive << " s | " << tflops_naive << " TFLOPS" << std::endl;

    // 2. Optimized GPU Benchmark (Shared Memory - Tiling)
    std::cout << "\n[GPU] Version 2: Optimized (Shared Memory)..." << std::endl;
    
    // Reset destination memory to avoid interference
    cudaMemset(d_C, 0, bytes);

    cudaEventRecord(start_gpu);
    // Ensure the function name matches the one defined in sgemm_shared.cu
    MatrixMulOptimized<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop_gpu);
    cudaEventSynchronize(stop_gpu);

    float ms_opt = 0;
    cudaEventElapsedTime(&ms_opt, start_gpu, stop_gpu);
    double sec_opt = ms_opt / 1000.0;
    double tflops_opt = (flops / sec_opt) / 1e12; 
    
    std::cout << "- Time: " << sec_opt << " s | " << tflops_opt << " TFLOPS" << std::endl;
    std::cout << "- Speedup vs Naive: " << (sec_naive / sec_opt) << "x" << std::endl;

    cudaMemcpy(h_C_gpu.data(), d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaEventDestroy(start_gpu);
    cudaEventDestroy(stop_gpu);

    return 0;
}