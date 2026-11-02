# CUDA SGEMM Optimization

## Overview
This repository contains a High-Performance Computing (HPC) project focused on optimizing the Single-precision General Matrix Multiply (SGEMM) algorithm using CUDA C++.

Matrix multiplication is a foundational operation with high arithmetic intensity. This project explores GPU acceleration techniques essential for handling the massive computational workloads typically found in mechanical simulation, automatic control systems, and Computer-Aided Design (CAD). The implementation progresses from a sequential CPU baseline to a shared-memory kernel and a register-blocked kernel, and is benchmarked against NVIDIA's cuBLAS library. Every implementation is **numerically validated** before its performance is reported.

## Project Structure
The repository is organized to separate interfaces, implementations, and benchmarking logic:

* **`include/sgemm.cuh`**: Host-side API (CPU reference and kernel launchers). Kernels stay private to their `.cu` file.
* **`include/cuda_utils.cuh`**: `CUDA_CHECK` / `CUDA_CHECK_KERNEL` error macros, `DeviceBuffer<T>` (RAII GPU memory), `benchmarkMs` (warm-up + averaged `cudaEvent` timing) and validation helpers.
* **`src/sgemm_cpu.cpp`**: Baseline sequential C++ implementation ($O(N^3)$) for numerical validation. Loops are ordered `y-k-x` for contiguous memory access.
* **`src/sgemm_naive.cu`**: Unoptimized GPU kernel relying directly on global memory accesses (32×8 blocks for coalesced reads of `B`).
* **`src/sgemm_shared.cu`**: Two optimized kernels:
  * 16×16 shared-memory tiling (1 output element per thread);
  * 64×64 tiling with 4×4 register blocking (16 accumulators per thread, 256 threads per block).
* **`src/benchmark.cu`**: Main execution pipeline: correctness check, timing via `cudaEvent_t` with warm-up, TFLOPS calculation, cuBLAS comparison, summary table.
* **`Makefile`**: Compilation script configured for `nvcc`, linking the cuBLAS library and targeting C++17.

## Optimization Stages

The benchmark suite measures performance in **TFLOPS** (Tera Floating-Point Operations Per Second) using large-scale matrices ($4096 \times 4096$, requiring over 137 billion operations).

1. **CPU Reference (Single Core):** Establishes the baseline truth for numerical validation.
2. **Naive GPU (Global Memory):** Maps one thread to one output element. Performance is bottlenecked by redundant global VRAM reads.
3. **Shared Memory & Tiling (16×16):** Threads collaboratively load $16 \times 16$ data blocks into on-chip shared memory, reducing global reads by a factor of 16.
   * Loading **and** computing both live inside the loop over tiles.
   * Two `__syncthreads()` per iteration: one so the tile is complete before being read, one so all reads are finished before the tile is overwritten.
   * Sizes that are not a multiple of the tile are handled by zero-padding.
4. **Register Blocking (64×64 tile, 4×4 per thread):** each value read from shared memory feeds 4 FMA instead of 1, which raises the arithmetic intensity. `A` tile padded by one column to avoid shared-memory bank conflicts; columns interleaved so the final write to `C` is coalesced.
5. **cuBLAS (NVIDIA Official):** The reference for the hardware limit, called through `cublasSgemm` (row-major handled by swapping `A` and `B`).

### Correction of the previous version
The original shared-memory kernel closed its loading loop **before** the computation, so only the last tile of each output element was multiplied. This explained the previously reported 8.50 TFLOPS, which was above the FP32 peak of the Tesla T4 (about 8.1 TFLOPS) and therefore not physically possible. The previously reported cuBLAS figure (1.79 TFLOPS) was also wrong: it included the library's cold start because the measurement had no warm-up. Both issues are fixed in the current version.

## Prerequisites
* **NVIDIA GPU** with CUDA support (default target `sm_75`, Turing / Tesla T4). CUDA Toolkit 13 requires Compute Capability 7.5 or higher; with CUDA 11/12, older architectures (`sm_60` to `sm_70`) are still supported.
* **CUDA Toolkit** 11.0+ (tested with 13.0): required for the `nvcc` compiler; **cuBLAS** is included.
* **C++ Compiler**: Compatible with C++17 (e.g., GCC or MSVC).
* **Make**: For automated building (optional, see the direct `nvcc` command below).

## Build and Execute
Clone the repository and compile the project using the provided Makefile:

```bash
# Clone the repository
git clone https://github.com/Coce1/cuda-sgemm-optimization.git
cd cuda-sgemm-optimization

# Compile the project (set ARCH to your GPU: sm_75, sm_80, sm_86, sm_89...)
make ARCH=sm_75

# Run the benchmark
./sgemm_benchmark
```

Equivalent direct command (useful if the Makefile tabs were altered by a copy-paste):

```bash
nvcc -O3 -std=c++17 -arch=sm_75 -lineinfo -Iinclude \
     src/benchmark.cu src/sgemm_naive.cu src/sgemm_shared.cu src/sgemm_cpu.cpp \
     -o sgemm_benchmark -lcublas
./sgemm_benchmark
```

## Benchmark Protocol
1. **Correctness phase** ($N = 500$, not a multiple of 16 or 64, to exercise the boundaries): each implementation is compared with the CPU reference. The output buffer is pre-filled with NaN to detect unwritten elements. Results are always printed, with an `OK` / `ÉCHEC` status.
2. **Performance phase** ($N = 4096$): **2 warm-up runs**, then the average of several `cudaEvent`-timed runs (3 for the naive kernel, 10 for the others). The warm-up matters especially for cuBLAS, whose first call includes library initialization and lazy kernel loading.
3. **Validation of the timed result** against cuBLAS.
4. $\text{TFLOPS} = \frac{2N^3}{\text{time}} \times 10^{-12}$. A result above the theoretical FP32 peak of the GPU indicates an error.

### Memory and race checking (optional)
```bash
compute-sanitizer --tool memcheck ./sgemm_benchmark
compute-sanitizer --tool racecheck ./sgemm_benchmark
```

## Performance Analysis

**Hardware Environment:** Google Colab | GPU: Tesla T4 (sm_75, 40 SM, 15 GB) | CUDA Toolkit: 13.0 (V13.0.88) | Driver: 580.82.07
**Matrix Size:** 4096 x 4096 (137.4 GFLOP per product)

**Correctness ($N = 500$, vs CPU):** all four implementations: max relative error 3.04e-07, `OK`.

| Implementation | Execution Time (ms) | Performance (TFLOPS) | Speedup (vs Naive) | Error vs cuBLAS | Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Naive (Global Memory)** | 277.936 | 0.4945 | 1.0x | 0.00e+00 | OK |
| **Shared Memory (16×16 tiling)** | 202.614 | 0.6783 | 1.4x | 0.00e+00 | OK |
| **Register Blocking (64×64, 4×4)** | 54.949 | 2.5012 | 5.1x | 0.00e+00 | OK |
| **cuBLAS (NVIDIA Official)** | **31.960** | **4.3004** | **8.7x** | (reference) | OK |

Total program run time: 5.69 s.

| Comparison | Factor |
| :--- | :--- |
| Register blocking vs Shared 16×16 | 3.69x faster |
| cuBLAS vs Register blocking | 1.72x faster |
| cuBLAS vs Shared 16×16 | 6.34x faster |

### Key Takeaways

*   **Plain 16×16 tiling gives only 1.4x.** The naive kernel already uses 32×8 blocks, so its reads of `B` are coalesced and well served by the L1/L2 caches. The 16×16 tiled kernel still computes one output per thread: each FMA needs two shared-memory loads, so the bottleneck moves from global memory to shared-memory/instruction throughput (0.68 TFLOPS, far below the compute peak).
*   **Register blocking is the decisive step (3.7x over shared, 5.1x over naive).** With 4×4 outputs per thread, each pair of values loaded from shared memory feeds several FMA instead of one: the arithmetic intensity at the shared-memory level is multiplied accordingly, and 16 accumulators stay in registers. The kernel reaches 2.50 TFLOPS, about 58 % of cuBLAS.
*   **cuBLAS reaches 4.30 TFLOPS** once warmed up, more than twice the previously reported 1.79 TFLOPS (which measured its cold start). The remaining 1.7x gap relies on techniques not implemented here: larger tiles (128×128), vectorized `float4` loads, double-buffered shared-memory tiles and hand-tuned instruction scheduling.
*   **Numerical agreement.** All kernels give a result identical to cuBLAS (relative error exactly 0 at $N = 4096$, i.e. the same accumulation order along $k$) and differ from the CPU reference by about 3e-07, a floating-point rounding effect (FMA contraction on the GPU).
*   **Reading the TFLOPS figures.** The nominal FP32 peak of the T4 is about 8.1 TFLOPS, but the card is power-limited (70 W), so sustained performance is lower; the best measured value (4.30 TFLOPS) is below the nominal peak, as expected.
