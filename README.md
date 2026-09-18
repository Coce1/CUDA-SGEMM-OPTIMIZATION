# CUDA SGEMM Optimization

## Overview
This repository contains a High-Performance Computing (HPC) project focused on optimizing the Single-precision General Matrix Multiply (SGEMM) algorithm using CUDA C++. 

Matrix multiplication is a foundational operation with high arithmetic intensity. This project explores GPU acceleration techniques essential for handling the massive computational workloads typically found in mechanical simulation, automatic control systems, and Computer-Aided Design (CAD). The implementation progresses from a sequential CPU baseline to a highly optimized shared-memory GPU kernel, ultimately benchmarking against NVIDIA's proprietary cuBLAS library.

## Project Structure
The repository is organized to separate interfaces, implementations, and benchmarking logic:

* **`include/sgemm.cuh`**: Header file containing function declarations and CUDA macros.
* **`src/sgemm_cpu.cpp`**: Baseline sequential C++ implementation ($O(N^3)$) for numerical validation and baseline performance metrics.
* **`src/sgemm_naive.cu`**: Unoptimized GPU kernel relying directly on global memory accesses.
* **`src/sgemm_shared.cu`**: Optimized GPU kernel utilizing 2D thread blocks, shared memory (`__shared__`), and loop tiling to drastically reduce global memory latency.
* **`src/benchmark.cu`**: The main execution pipeline. It handles memory allocation, asynchronous execution timing via `cudaEvent_t`, matrix scaling, and TFLOPS calculation.
* **`Makefile`**: Compilation script configured for `nvcc`, linking the cuBLAS library and targeting C++17.

## Optimization Stages & Benchmarking

The benchmark suite measures performance in **TFLOPS** (Tera Floating-Point Operations Per Second) using large-scale matrices (e.g., $4096 \times 4096$, requiring over 137 billion operations).

1. **CPU Reference (Single Core):** Establishes the baseline truth and standard GFLOPS throughput.
2. **Naive GPU (Global Memory):** Maps one thread to one output element. Performance is severely bottlenecked by redundant global VRAM reads.
3. **Optimized GPU (Shared Memory & Tiling):** Threads collaboratively load $16 \times 16$ data blocks into ultra-fast on-chip shared memory, synchronizing via `__syncthreads()`. This maximizes the computational arithmetic intensity and saturates the GPU cores.
4. **cuBLAS (NVIDIA Official):** The theoretical hardware limit, utilizing proprietary assembly-level optimizations and the `cublasHandle_t` context stream.

## Prerequisites
* **CUDA Toolkit**: Required for `nvcc` compiler and `cudaEvent_t` timing.
* **cuBLAS Library**: Included with the CUDA Toolkit.
* **C++ Compiler**: Compatible with C++17 (e.g., GCC or MSVC).
* **Make**: For automated building.

## Build and Execute
Clone the repository and compile the project using the provided Makefile:

```bash
# Clone the repository
git clone [https://github.com/Coce1/cuda-sgemm-optimization.git](https://github.com/Coce1/cuda-sgemm-optimization.git)
cd cuda-sgemm-optimization

# Compile the project
make

# Run the extreme benchmark
./sgemm_benchmark

```

## Performance Analysis

**Hardware Environment:** Google Colab (GPU: [e.g., NVIDIA Tesla T4])  
**Matrix Size:** 4096 x 4096 ($N \times N$)  

| Implementation | Execution Time (s) | Performance (TFLOPS) | Speedup (vs Naive) |
| :--- | :--- | :--- | :--- |
| **Naive (Global Memory)** | [0.000] | [0.000] | 1.0x |
| **Optimized (Shared Memory)** | [0.000] | [0.000] | **[X.X]x** |
| **cuBLAS (NVIDIA Official)** | [0.000] | [0.000] | **[Y.Y]x** |

### Key Takeaways
* **The Memory Wall:** The naive implementation is strictly memory-bound. Each thread reads directly from the slow global VRAM, starving the CUDA cores of data.
* **The Power of Tiling:** By utilizing a $16 \times 16$ shared memory tile, the optimized kernel reduces global memory reads by a factor of 16. This shifts the bottleneck from memory bandwidth to computational throughput, resulting in a massive **[X.X]x** speedup.
* **The Assembly Gap:** cuBLAS outperforms our custom shared-memory kernel by a factor of **[Z.Z]x**. This highlights the impact of hardware-specific optimizations (like register-level tuning, loop unrolling, and potentially Tensor Cores) that NVIDIA implements at the assembly level.


