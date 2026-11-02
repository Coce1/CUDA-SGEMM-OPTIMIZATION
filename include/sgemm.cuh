#ifndef SGEMM_CUH
#define SGEMM_CUH

// Interface 100 % hôte : les kernels restent privés à leur fichier .cu.
// Toutes les matrices sont N x N, row-major. Les pointeurs d_* sont en mémoire GPU.
// Chaque launcher vérifie sa configuration de lancement (cudaGetLastError) mais
// ne synchronise pas : l'appelant synchronise (ou chronomètre avec des cudaEvent).

// Référence CPU O(N^3), pour la validation numérique.
void sgemmCPU(const float* A, const float* B, float* C, int N);

// Naïf : 1 thread = 1 élément de C, mémoire globale uniquement.
void launchSgemmNaive(const float* d_A, const float* d_B, float* d_C, int N);

// Tuiles 16x16 en shared memory (1 élément de C par thread).
void launchSgemmShared(const float* d_A, const float* d_B, float* d_C, int N);

// Tuiles 64x64, register blocking 4x4 par thread (16 accumulateurs).
void launchSgemmRegBlocked(const float* d_A, const float* d_B, float* d_C, int N);

#endif // SGEMM_CUH
