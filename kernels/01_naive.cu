#define BLOCKSIZE 32
#define CEIL_DIV(M, N) (((M) + (N) - 1) / (N))
namespace k1 {

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

void launch_sgemm_naive(int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 block(32, 32);

    sgemm_naive<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

void launch_sgemm_naive_coalesced(int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    dim3 grid(CEIL_DIV(M, 32), CEIL_DIV(N, 32));
    dim3 block(32 * 32);

    sgemm_naive_coalesced<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

} // namespace k1