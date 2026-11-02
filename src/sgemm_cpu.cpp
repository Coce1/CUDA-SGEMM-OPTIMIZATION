#include "../include/sgemm.cuh"

#include <cstddef>

void sgemmCPU(const float* A, const float* B, float* C, int N) {
    const std::size_t n = static_cast<std::size_t>(N);

    for (std::size_t i = 0; i < n * n; ++i) C[i] = 0.0f;

    // Ordre y-k-x au lieu de y-x-k : B et C sont parcourus en ligne (contigu),
    // contre un saut de N éléments par itération dans l'ordre d'origine.
    // L'ordre de sommation par élément (k croissant) est inchangé -> même résultat.
    for (std::size_t y = 0; y < n; ++y) {
        for (std::size_t k = 0; k < n; ++k) {
            const float a = A[y * n + k];
            for (std::size_t x = 0; x < n; ++x) {
                C[y * n + x] += a * B[k * n + x];
            }
        }
    }
}
