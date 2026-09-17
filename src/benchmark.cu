#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include "../include/sgemm.cuh"

int main() {
    // Taille de la matrice : N x N
    // 512 est un bon point de départ pour le CPU (~268 millions d'opérations)
    int N = 512; 
    
    std::cout << "=== SGEMM Benchmark (" << N << "x" << N << ") ===" << std::endl;

    // Allocation de la mémoire côté hôte (CPU)
    std::vector<float> h_A(N * N);
    std::vector<float> h_B(N * N);
    std::vector<float> h_C_cpu(N * N, 0.0f);

    // Initialisation avec des valeurs aléatoires (graine fixe pour la reproductibilité)
    std::mt19937 gen(42); 
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < N * N; ++i) {
        h_A[i] = dist(gen);
        h_B[i] = dist(gen);
    }

    // Benchmark CPU
    std::cout << "\nLancement de l'implementation de reference CPU..." << std::endl;
    auto start_cpu = std::chrono::high_resolution_clock::now();
    
    sgemmCPU(h_A.data(), h_B.data(), h_C_cpu.data(), N);
    
    auto end_cpu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> cpu_duration = end_cpu - start_cpu;

    // Calcul des performances en GFLOPS
    // Formule : (2 * N^3 operations) / (Temps en secondes * 10^9)
    double flops = 2.0 * static_cast<double>(N) * N * N;
    double gflops = (flops / cpu_duration.count()) / 1e9;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "- Temps d'execution : " << cpu_duration.count() << " secondes" << std::endl;
    std::cout << "- Performance       : " << gflops << " GFLOPS" << std::endl;

    return 0;
}