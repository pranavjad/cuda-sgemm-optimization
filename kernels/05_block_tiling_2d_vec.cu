namespace k5 {
__global__ void sgemm_2d_block_tiling_vec(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const int BM = 128;
    const int BN = 128;
    const int BK = 8;
    const int TM = 8;
    const int TN = 8;

    const uint total_results_blocktile = BM * BN;
    const uint num_thread_blocktile = total_results_blocktile / (TM * TN);
    assert(num_thread_blocktile == blockDim.x);
    
    const uint block_row = blockIdx.x;
    const uint block_col = blockIdx.y;

    // thread row and col in the A block, B block. (each thread loads 4 values)
    const uint thread_row_A = threadIdx.x / (BK / 4);
    const uint thread_col_A = threadIdx.x % (BK / 4);

    const uint thread_row_B = threadIdx.x / (BN / 4);
    const uint thread_col_B = threadIdx.x % (BN / 4);

    // row,col of the TM*TN tile that this thread handles in the C output block
    const uint thread_row_C = threadIdx.x / (BN / TN);
    const uint thread_col_C = threadIdx.x % (BN / TN);

    // max rows of A, B we can load at once (it takes BK thread to load a row of A in parallel)
    const uint stride_A = num_thread_blocktile / (BK / 4);
    const uint stride_B = num_thread_blocktile / (BN / 4);

    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    // set pointers to their starting positions
    float *A_ptr = A + block_row * BM * K;
    float *B_ptr = B + block_col * BN;
    float *C_ptr = C + block_row * BM * N + block_col * BN;

    float thread_results[TM * TN] = {0.0};
    float tmp_A[TM] = {0.0};
    float tmp_B[TN] = {0.0};
    
    for (int block_idx = 0; block_idx < K; block_idx += BK) {
        
        //// for block sizes
        // for (int ld_offset = 0; ld_offset < BM; ld_offset += stride_A) {
        //     float4 tmp = reinterpret_cast<float4 *>(&A_ptr[(thread_row_A + ld_offset) * K + thread_col_A * 4])[0];
        //     A_shared[(thread_col_A * 4 + 0) * BM + (thread_row_A + ld_offset)] = tmp.x;
        //     A_shared[(thread_col_A * 4 + 1) * BM + (thread_row_A + ld_offset)] = tmp.y;
        //     A_shared[(thread_col_A * 4 + 2) * BM + (thread_row_A + ld_offset)] = tmp.z;
        //     A_shared[(thread_col_A * 4 + 3) * BM + (thread_row_A + ld_offset)] = tmp.w;
        // }

        // for (int ld_offset = 0; ld_offset < BK; ld_offset += stride_B) {
        //     reinterpret_cast<float4 *>(&B_shared[(ld_offset + thread_row_B) * BN + thread_col_B * 4])[0] = 
        //         reinterpret_cast<float4 *>(&B_ptr[(ld_offset + thread_row_B) * N + thread_col_B * 4])[0];
        // }

        // in this case, we can be sure we have enough threads in the block to load the full shared tile without looping
        // transpose A as we load in shared memory (becomes useful later)
        float4 tmp = reinterpret_cast<float4 *>(&A_ptr[(thread_row_A) * K + thread_col_A * 4])[0];
        A_shared[(thread_col_A * 4 + 0) * BM + (thread_row_A)] = tmp.x;
        A_shared[(thread_col_A * 4 + 1) * BM + (thread_row_A)] = tmp.y;
        A_shared[(thread_col_A * 4 + 2) * BM + (thread_row_A)] = tmp.z;
        A_shared[(thread_col_A * 4 + 3) * BM + (thread_row_A)] = tmp.w;

        reinterpret_cast<float4 *>(&B_shared[(thread_row_B) * BN + thread_col_B * 4])[0] = 
            reinterpret_cast<float4 *>(&B_ptr[(thread_row_B) * N + thread_col_B * 4])[0];
        __syncthreads();

        A_ptr += BK;
        B_ptr += BK * N;

        // Compute partial results for this tile
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            // Load column from A shared memory (but A is transposed in smem)
            for (int i = 0; i < TM; i++) {
                tmp_A[i] = A_shared[dot_idx * BM + (thread_row_C * TM + i)];
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

    // Write results back to global memory (vectorized)
    for (int i = 0; i < TM; i++) {
        for (int j = 0; j < TN; j += 4) {
            int row = thread_row_C * TM + i;
            int col = thread_col_C * TN + j;
            float4 tmp = reinterpret_cast<float4 *>(&C_ptr[row * N + col])[0];
            tmp.x = alpha * thread_results[i * TN + j] + beta * tmp.x;
            tmp.y = alpha * thread_results[i * TN + j + 1] + beta * tmp.y;
            tmp.z = alpha * thread_results[i * TN + j + 2] + beta * tmp.z;
            tmp.w = alpha * thread_results[i * TN + j + 3] + beta * tmp.w;
            reinterpret_cast<float4 *>(&C_ptr[row * N + col])[0] = tmp;
        }
    }



}

void launch_sgemm_2d_block_tiling_vec(int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block((BM * BN) / (TM * TN));

    sgemm_2d_block_tiling_vec<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}

} // namespace k5