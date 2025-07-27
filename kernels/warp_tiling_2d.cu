__global__ void sgemm_2d_warp_tiling(int M, int N, int K, float alpha, const float *A, const float *B, float beta, float *C) {
    const int BM = 64;
    const int BN = 64;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const uint total_results_blocktile = BM * BN;
    const uint num_thread_blocktile = total_results_blocktile / (TM * TN);
    assert(num_thread_blocktile == blockDim.x);
    
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    // thread row and col in the A block, B block.
    const uint thread_row_A = threadIdx.x / BK;
    const uint thread_col_A = threadIdx.x % BK;

    const uint thread_row_B = threadIdx.x / BN;
    const uint thread_col_B = threadIdx.x % BN;

    // row,col of the TM*TN tile that this thread handles in the C output block
    const uint thread_row_C = threadIdx.x / (BN / TN);
    const uint thread_col_C = threadIdx.x % (BN / TN);

    // max rows of A, B we can load at once (it takes BK thread to load a row of A in parallel)
    const uint stride_A = num_thread_blocktile / BK;
    const uint stride_B = num_thread_blocktile / BN;

    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    // set pointers to their starting positions
    const float *A_ptr = A + block_row * BM * K;
    const float *B_ptr = B + block_col * BN;
    float *C_ptr = C + block_row * BM * N + block_col * BN;

    float thread_results[TM * TN] = {0.0};
    float tmp_A[TM] = {0.0};
    float tmp_B[TN] = {0.0};
    
    for (int block_idx = 0; block_idx < K; block_idx += BK) {
        // ld_offset is what row we are starting loading at for loading into shared mem
        for (int ld_offset = 0; ld_offset < BM; ld_offset += stride_A) {
            A_shared[(ld_offset + thread_row_A) * BK + thread_col_A] = A_ptr[(ld_offset + thread_row_A) * K + thread_col_A];
        }
        for (int ld_offset = 0; ld_offset < BK; ld_offset += stride_B) {
            B_shared[(ld_offset + thread_row_B) * BN + thread_col_B] = B_ptr[(ld_offset + thread_row_B) * N + thread_col_B];
        }
        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;

        // Compute partial results for this tile
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            // Load column from A shared memory
            for (int i = 0; i < TM; i++) {
                tmp_A[i] = A_shared[(thread_row_C * TM + i) * BK + dot_idx];
            }
            // Load row from B shared memory
            for (int i = 0; i < TN; i++) {
                tmp_B[i] = B_shared[dot_idx * BN + (thread_col_C * TN + i)];
            }

            // Outer product accumulation
            for (int i = 0; i < TM; i++) {
                for (int j = 0; j < TN; j++) {
                    thread_results[i * TN + j] += tmp_A[i] * tmp_B[j];
                }
            }
        }
        __syncthreads();
    }

    // Write results back to global memory
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j++) {
            int row = thread_row_C * TM + i;
            int col = thread_col_C * TN + j;
            C_ptr[row * N + col] = alpha * thread_results[i * TN + j] + beta * C_ptr[row * N + col];
        }
    }

}