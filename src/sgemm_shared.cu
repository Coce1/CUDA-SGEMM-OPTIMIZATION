#include "../include/sgemm.cuh"
#include "../include/cuda_utils.cuh"

// ===========================================================================
// Kernel 1 : tuiles 16x16, 1 élément de C par thread
// ===========================================================================
constexpr int TILE = 16;

__global__ void sgemmGPUShared(const float* __restrict__ A, const float* __restrict__ B,
                               float* __restrict__ C, int N) {
    __shared__ float s_A[TILE][TILE]; // 1 Ko (l'ancien S_TILE valait 1024 floats : 4x trop)
    __shared__ float s_B[TILE][TILE];

    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;
    float val = 0.0f;

    // CORRECTION CRITIQUE : l'ancienne version fermait la boucle de chargement AVANT
    // le calcul, donc seule la dernière tuile était multipliée. Chargement ET calcul
    // sont maintenant dans la même boucle sur les tuiles.
    // (N + TILE - 1) / TILE : gère N non multiple de 16 (zero-padding des bords).
    for (int t = 0; t < (N + TILE - 1) / TILE; ++t) {
        const int a_col = t * TILE + threadIdx.x;
        const int b_row = t * TILE + threadIdx.y;

        s_A[threadIdx.y][threadIdx.x] =
            (row < N && a_col < N) ? A[static_cast<size_t>(row) * N + a_col] : 0.0f;
        s_B[threadIdx.y][threadIdx.x] =
            (b_row < N && col < N) ? B[static_cast<size_t>(b_row) * N + col] : 0.0f;

        // Barrière 1 : la tuile doit être entièrement chargée avant d'être lue.
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k) {
            val += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // Barrière 2 : tous les threads ont fini de LIRE avant que l'itération suivante
        // n'écrase les tuiles (sans elle : course lecture/écriture).
        __syncthreads();
    }

    if (row < N && col < N) {
        C[static_cast<size_t>(row) * N + col] = val;
    }
}

void launchSgemmShared(const float* d_A, const float* d_B, float* d_C, int N) {
    const dim3 block(TILE, TILE);
    const dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    sgemmGPUShared<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK_KERNEL();
}

// ===========================================================================
// Kernel 2 : register blocking. Tuile de C 64x64, 256 threads, 4x4 éléments/thread.
// Intensité arithmétique : chaque valeur lue en shared sert à 4 FMA (contre 1 au kernel 1),
// ce qui est nécessaire pour approcher le débit de cuBLAS.
// ===========================================================================
constexpr int RB_BM = 64;                       // lignes de C par bloc
constexpr int RB_BN = 64;                       // colonnes de C par bloc
constexpr int RB_BK = 16;                       // profondeur de tuile en k
constexpr int RB_TM = 4;                        // lignes de C par thread
constexpr int RB_TN = 4;                        // colonnes de C par thread
constexpr int RB_THREADS_X = RB_BN / RB_TN;     // 16
constexpr int RB_THREADS_Y = RB_BM / RB_TM;     // 16
constexpr int RB_THREADS = RB_THREADS_X * RB_THREADS_Y; // 256
static_assert((RB_BM * RB_BK) % RB_THREADS == 0, "chargement de A : répartition entière requise");
static_assert((RB_BK * RB_BN) % RB_THREADS == 0, "chargement de B : répartition entière requise");

__global__ void __launch_bounds__(RB_THREADS)
sgemmGPURegBlocked(const float* __restrict__ A, const float* __restrict__ B,
                   float* __restrict__ C, int N) {

    // +1 de padding sur A : écritures As[m][k] sans conflit de banque.
    __shared__ float As[RB_BM][RB_BK + 1];
    __shared__ float Bs[RB_BK][RB_BN];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int tid = ty * RB_THREADS_X + tx;
    const int rowBase = blockIdx.y * RB_BM;
    const int colBase = blockIdx.x * RB_BN;

    float acc[RB_TM][RB_TN];
    #pragma unroll
    for (int i = 0; i < RB_TM; ++i)
        #pragma unroll
        for (int j = 0; j < RB_TN; ++j) acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < N; k0 += RB_BK) {
        // Chargement coopératif de la tuile A (64x16) : 4 éléments/thread.
        // Un warp lit 2 segments de 16 floats contigus (64 o chacun).
        #pragma unroll
        for (int i = 0; i < (RB_BM * RB_BK) / RB_THREADS; ++i) {
            const int idx = tid + i * RB_THREADS;
            const int m = idx / RB_BK;
            const int k = idx % RB_BK;
            const int gr = rowBase + m;
            const int gc = k0 + k;
            As[m][k] = (gr < N && gc < N) ? A[static_cast<size_t>(gr) * N + gc] : 0.0f;
        }
        // Chargement coopératif de la tuile B (16x64) : un warp lit 32 floats
        // contigus d'une ligne de B (128 o, coalescé).
        #pragma unroll
        for (int i = 0; i < (RB_BK * RB_BN) / RB_THREADS; ++i) {
            const int idx = tid + i * RB_THREADS;
            const int k = idx / RB_BN;
            const int n = idx % RB_BN;
            const int gr = k0 + k;
            const int gc = colBase + n;
            Bs[k][n] = (gr < N && gc < N) ? B[static_cast<size_t>(gr) * N + gc] : 0.0f;
        }
        __syncthreads(); // tuiles complètes avant lecture

        #pragma unroll
        for (int k = 0; k < RB_BK; ++k) {
            float a_reg[RB_TM];
            float b_reg[RB_TN];
            #pragma unroll
            for (int i = 0; i < RB_TM; ++i) a_reg[i] = As[ty * RB_TM + i][k];  // broadcast
            // Colonnes entrelacées (j*16 + tx) : threads consécutifs -> banques consécutives.
            #pragma unroll
            for (int j = 0; j < RB_TN; ++j) b_reg[j] = Bs[k][j * RB_THREADS_X + tx];

            #pragma unroll
            for (int i = 0; i < RB_TM; ++i)
                #pragma unroll
                for (int j = 0; j < RB_TN; ++j) acc[i][j] += a_reg[i] * b_reg[j];
        }
        __syncthreads(); // lectures finies avant écrasement par la tuile suivante
    }

    // Écriture : à (i, j) fixés, threads consécutifs -> colonnes consécutives (coalescé).
    #pragma unroll
    for (int i = 0; i < RB_TM; ++i) {
        const int r = rowBase + ty * RB_TM + i;
        #pragma unroll
        for (int j = 0; j < RB_TN; ++j) {
            const int c = colBase + j * RB_THREADS_X + tx;
            if (r < N && c < N) C[static_cast<size_t>(r) * N + c] = acc[i][j];
        }
    }
}

void launchSgemmRegBlocked(const float* d_A, const float* d_B, float* d_C, int N) {
    const dim3 block(RB_THREADS_X, RB_THREADS_Y);
    const dim3 grid((N + RB_BN - 1) / RB_BN, (N + RB_BM - 1) / RB_BM);
    sgemmGPURegBlocked<<<grid, block>>>(d_A, d_B, d_C, N);
    CUDA_CHECK_KERNEL();
}
