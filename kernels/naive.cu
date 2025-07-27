#define BLOCKSIZE 32

// naive
__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            // dot the x-th row of A and the y-th column of B
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}

// naive with coalesced memory access
__global__ void sgemm_naive_coalesced(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const uint x = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
    const uint y = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);

    if (x < M && y < N) {
        float tmp = 0.0;
        for (int i = 0; i < K; ++i) {
            // dot the x-th row of A and the y-th column of B
            tmp += A[x * K + i] * B[i * N + y];
        }
        C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }
}