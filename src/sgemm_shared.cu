
#define S_TILE 16*16*4
#define TILE 16	

__global__ void sgemmGPUShared(const float* A, const float* B, float* C, int N) {
	__shared__ float s_A[S_TILE];
	__shared__ float s_B[S_TILE];

	float val = 0.0f;
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;

	for (int i=0;i< N/TILE; i++){
		s_A[threadIdx.y * TILE + threadIdx.x] = A[row * N + (i * TILE + threadIdx.x)];
		s_B[threadIdx.y * TILE + threadIdx.x] = B[(i * TILE + threadIdx.y) * N + col];	
	}

	__syncthreads();


	for (int i=0;i<TILE; i++){
		val += s_A[threadIdx.y*TILE + i] * s_B[i*TILE + threadIdx.x];
	}
	__syncthreads();
	C[row * N + col] = val;
	
}