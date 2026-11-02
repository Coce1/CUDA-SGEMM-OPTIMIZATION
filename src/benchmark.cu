#include <cublas_v2.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include "../include/cuda_utils.cuh"
#include "../include/sgemm.cuh"

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                       \
        cublasStatus_t cublas_check_st_ = (call);                              \
        if (cublas_check_st_ != CUBLAS_STATUS_SUCCESS) {                       \
            std::fflush(stdout);                                               \
            std::fprintf(stderr, "[CUBLAS ERROR] %s:%d : statut %d (%s)\n",    \
                         __FILE__, __LINE__, static_cast<int>(cublas_check_st_), #call); \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

struct Impl {
    const char* name;
    std::function<void(const float*, const float*, float*, int)> fn;
    int reps; // exécutions chronométrées (le naïf est lent)
};

struct Result {
    std::string name;
    double ms = 0.0;
    double tflops = 0.0;
    double speedup_naive = 0.0;
    double err = 0.0;
    bool ok = false;
};

static void fillRandom(std::vector<float>& v, unsigned seed) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (auto& x : v) x = dist(gen);
}

// Phase 1 : exactitude sur N non multiple de 16/64, comparée à la référence CPU.
// Un échec est AFFICHÉ mais n'empêche plus la phase de performance.
static void checkCorrectness(const std::vector<Impl>& impls, int N) {
    const size_t n = static_cast<size_t>(N) * N;
    std::vector<float> h_A(n), h_B(n), h_ref(n), h_C;
    fillRandom(h_A, 1);
    fillRandom(h_B, 2);
    sgemmCPU(h_A.data(), h_B.data(), h_ref.data(), N);

    DeviceBuffer<float> d_A(n), d_B(n), d_C(n);
    d_A.upload(h_A);
    d_B.upload(h_B);

    std::printf("--- Phase 1 : exactitude (N = %d, référence CPU) ---\n", N);
    for (const Impl& impl : impls) {
        CUDA_CHECK(cudaMemset(d_C.get(), 0xFF, n * sizeof(float))); // NaN partout
        impl.fn(d_A.get(), d_B.get(), d_C.get(), N);
        CUDA_CHECK(cudaDeviceSynchronize());
        d_C.download(h_C);

        const double diff = maxRelDiff(h_C, h_ref);
        std::printf("  %-32s : erreur relative max = %.3e | %s\n",
                    impl.name, diff, diff < 1e-4 ? "OK" : "ÉCHEC");
    }
    std::printf("\n");
}

int main() {
    // Sortie non bufferisée : tout ce qui est affiché reste visible même en cas de crash.
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    const auto t_program = std::chrono::steady_clock::now();

    std::printf("=== SGEMM Benchmark ===\n");
    printDeviceInfo();

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    // cuBLAS est column-major : en passant (B, A) on obtient C = A*B en row-major.
    auto cublasImpl = [&](const float* A, const float* B, float* C, int N) {
        const float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N,
                                 &alpha, B, N, A, N, &beta, C, N));
    };

    const std::vector<Impl> impls = {
        {"Naive (global memory)", launchSgemmNaive, 3},
        {"Shared 16x16", launchSgemmShared, 10},
        {"Register blocking 64x64 (4x4)", launchSgemmRegBlocked, 10},
        {"cuBLAS", cublasImpl, 10},
    };

    // ------------------------------ Phase 1 ------------------------------
    checkCorrectness(impls, 500);

    // ------------------------------ Phase 2 ------------------------------
    const int N = 4096;
    const size_t n = static_cast<size_t>(N) * N;
    const double flops = 2.0 * static_cast<double>(N) * N * N;
    std::printf("--- Phase 2 : performance (N = %d, %.1f GFLOP par produit) ---\n", N, flops / 1e9);

    std::vector<float> h_A(n), h_B(n), h_ref, h_C;
    fillRandom(h_A, 42);
    fillRandom(h_B, 43);

    DeviceBuffer<float> d_A(n), d_B(n), d_C(n);
    d_A.upload(h_A);
    d_B.upload(h_B);

    // Référence de validation = cuBLAS (la référence CPU en 4096^3 serait trop longue)
    CUDA_CHECK(cudaMemset(d_C.get(), 0, n * sizeof(float)));
    cublasImpl(d_A.get(), d_B.get(), d_C.get(), N);
    CUDA_CHECK(cudaDeviceSynchronize());
    d_C.download(h_ref);

    std::vector<Result> results;
    double sec_naive = 0.0;

    for (const Impl& impl : impls) {
        std::printf("  [en cours] %s ...\n", impl.name);

        // benchmarkMs : 2 warm-up puis moyenne sur impl.reps exécutions (cudaEvent)
        const float ms = benchmarkMs([] {}, [&] { impl.fn(d_A.get(), d_B.get(), d_C.get(), N); },
                                     2, impl.reps);
        const double sec = ms / 1000.0;

        Result r;
        r.name = impl.name;
        r.ms = ms;
        r.tflops = (flops / sec) / 1e12;
        if (sec_naive == 0.0) sec_naive = sec; // le premier élément est le naïf
        r.speedup_naive = sec_naive / sec;

        d_C.download(h_C);
        r.err = maxRelDiff(h_C, h_ref);
        r.ok = r.err < 1e-3;
        results.push_back(r);

        std::printf("    -> %.3f ms | %.3f TFLOPS | err vs cuBLAS %.2e | %s\n",
                    r.ms, r.tflops, r.err, r.ok ? "OK" : "ÉCHEC");
    }

    // ------------------------------ Récapitulatif ------------------------------
    std::printf("\n=== RÉCAPITULATIF SGEMM (N = %d) ===\n", N);
    std::printf("%-32s | %12s | %10s | %12s | %11s | %s\n",
                "Implémentation", "Temps (ms)", "TFLOPS", "Speedup/naïf", "Err/cuBLAS", "Statut");
    std::printf("---------------------------------+--------------+------------+--------------+-------------+-------\n");
    for (const Result& r : results) {
        std::printf("%-32s | %12.3f | %10.4f | %11.1fx | %11.2e | %s\n",
                    r.name.c_str(), r.ms, r.tflops, r.speedup_naive, r.err, r.ok ? "OK" : "ÉCHEC");
    }
    if (results.size() == 4) {
        std::printf("\nShared 16x16 vs cuBLAS : %.2fx plus lent\n", results[1].ms / results[3].ms);
        std::printf("Reg. blocking vs cuBLAS : %.2fx plus lent\n", results[2].ms / results[3].ms);
        std::printf("Reg. blocking vs Shared : %.2fx plus rapide\n", results[1].ms / results[2].ms);
    }

    CUBLAS_CHECK(cublasDestroy(handle));

    const double total_s =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - t_program).count();
    std::printf("\nTemps total d'exécution du programme : %.2f s\n", total_s);
    std::printf("Benchmark terminé.\n");
    return 0;
}
