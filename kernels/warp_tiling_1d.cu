#include <assert.h>

// 1d warptiling
__global__ void sgemm_1d_warp_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    // block from A = (BM, BK)
    // block from B = (BK, BN)
    // each thread calculates TM rows of one thread_col col
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;

    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    const uint totalResultsBlocktile = BM * BN;
    const uint numThreadsBlocktile = totalResultsBlocktile / TM;
    assert(numThreadsBlocktile == blockDim.x);

    // thread row and col in the A block, B block, and output C block
    const uint thread_row_A = threadIdx.x / BK;
    const uint thread_col_A = threadIdx.x % BK;

    const uint thread_row_B = threadIdx.x / BN;
    const uint thread_col_B = threadIdx.x % BN;

    const uint thread_row_C = threadIdx.x / BN;
    const uint thread_col_C = threadIdx.x % BN;


    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    // set pointers to their starting positions
    const float *A_ptr = A + block_row * BM * K;
    const float *B_ptr = B + block_col * BN;
    float *C_ptr = C + block_row * BM * N + block_col * BN;
    
    float tmp[TM] = {0.0};

    for (int block_idx = 0; block_idx < K; block_idx += BK) {
        A_shared[thread_row_A * BK + thread_col_A] = A_ptr[thread_row_A * K + thread_col_A];
        B_shared[thread_row_B * BN + thread_col_B] = B_ptr[thread_row_B * N + thread_col_B];
        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;
        
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            float b_col_elem = B_shared[dot_idx * BN + thread_col_C];
            for (int out_row = 0; out_row < TM; out_row++) {
                tmp[out_row] += A_shared[(thread_row_C * TM + out_row) * BK + dot_idx] * b_col_elem;
            } 
        }
        __syncthreads();
    }

    for (int i = 0; i < TM; i++) {
        // we want the (thread_row_C * TM + i)th row of C.
        // This is because each thread handles TM rows so we have to multiply 
        C_ptr[(thread_row_C * TM + i) * N + thread_col_C] = alpha * tmp[i] + beta * C_ptr[(thread_row_C * TM + i) * N + thread_col_C];
    }
}

