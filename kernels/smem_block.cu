#define BLOCKSIZE 32

// shared memory cache-blocking
__global__ void sgemm_smem_block(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;
    const uint thread_row = threadIdx.x / BLOCKSIZE;
    const uint thread_col = threadIdx.x % BLOCKSIZE;

    __shared__ float A_shared[BLOCKSIZE * BLOCKSIZE];
    __shared__ float B_shared[BLOCKSIZE * BLOCKSIZE];

    const float *A_ptr = A + block_row * BLOCKSIZE * K;
    const float *B_ptr = B + block_col * BLOCKSIZE;
    float *C_ptr = C + block_row * BLOCKSIZE * N + block_col * BLOCKSIZE;
    
    float tmp = 0.0f;
    for (int k = 0; k < K; k += BLOCKSIZE) {
        // load chunks into shared memory
        A_shared[thread_row * BLOCKSIZE + thread_col] = A_ptr[thread_row * K + thread_col];
        B_shared[thread_row * BLOCKSIZE + thread_col] = B_ptr[thread_row * N + thread_col];
        __syncthreads();

        A_ptr += BLOCKSIZE;
        B_ptr += BLOCKSIZE * N;

        for (int j = 0; j < BLOCKSIZE; j++) {
            tmp += A_shared[thread_row * BLOCKSIZE + j] * B_shared[thread_col + j * BLOCKSIZE];
        }
        __syncthreads();
    }
    C_ptr[thread_row * N + thread_col] = alpha * tmp + beta * C_ptr[thread_row * N + thread_col];
}