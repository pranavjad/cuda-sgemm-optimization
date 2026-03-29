namespace k6 {

const uint WARPSIZE = 32;
const uint NUM_THREADS = 128;
__global__ void __launch_bounds__(NUM_THREADS)
sgemm_warp_tiling(int M, int N, int K, float alpha, float *A, float *B, float beta, float *C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 16;
    const uint WM = 64;
    const uint WN = 64;
    const uint WNITER = 4; // # of warp subtiles along horizontal dim of warptile
    const uint TM = 8;
    const uint TN = 4;

    // block tile index
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    // warp tile index
    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warp_row = warp_idx / (BN / WN);
    const uint warp_col = warp_idx % (BN / WN);

    // size of warp subtile
    constexpr uint WMITER = (WM * WN) / (WNITER * TM * TN * WARPSIZE); // # of warp subtiles along vertical dim of warptile
    constexpr uint WSUBM = WM / WMITER;
    constexpr uint WSUBN = WN / WNITER;

    // placement of thread in warp subtile
    const uint thread_id_warp = threadIdx.x % WARPSIZE;
    const uint thread_row_subtile = thread_id_warp / (WSUBN / TN);
    const uint thread_col_subtile = thread_id_warp % (WSUBN / TN);

    __shared__ float A_shared[BM * BK];
    __shared__ float B_shared[BK * BN];

    A = A + block_row * BM * K;
    B = B + block_col * BN;
    // move C to start of this warp's warptile
    C = C + (block_row * BM + warp_row * WM) * N + (block_col * BN + warp_col * WN);


    // thread row and col in the A block, B block.
    const uint thread_row_A = threadIdx.x / (BK / 4);
    const uint thread_col_A = threadIdx.x % (BK / 4);

    const uint thread_row_B = threadIdx.x / (BN / 4);
    const uint thread_col_B = threadIdx.x % (BN / 4);

    // max rows of A, B we can load at once (it takes BK thread to load a row of A in parallel)
    const uint stride_A = NUM_THREADS / (BK / 4);
    const uint stride_B = NUM_THREADS / (BN / 4);

    float thread_results[TM * TN * WMITER * WNITER] = {0.0};
    float regM[WMITER * TM] = {0.0};
    float regN[WNITER * TN] = {0.0};

    for (int block_idx = 0; block_idx < K; block_idx += BK) {
        // load block tiles from GMEM to SMEM
        for (int offset = 0; offset + stride_A <= BM; offset += stride_A) {
            float4 tmp = reinterpret_cast<float4 *>(&A[(offset + thread_row_A) * K + thread_col_A * 4])[0];
            A_shared[(thread_col_A * 4 + 0) * BM + (offset + thread_row_A)] = tmp.x;
            A_shared[(thread_col_A * 4 + 1) * BM + (offset + thread_row_A)] = tmp.y;
            A_shared[(thread_col_A * 4 + 2) * BM + (offset + thread_row_A)] = tmp.z;
            A_shared[(thread_col_A * 4 + 3) * BM + (offset + thread_row_A)] = tmp.w;
        }
        for (int offset = 0; offset + stride_B <= BK; offset += stride_B) {
            reinterpret_cast<float4 *>(&B_shared[(offset + thread_row_B) * BN + thread_col_B * 4])[0] =
                reinterpret_cast<float4 *>(&B[(offset + thread_row_B) * N + thread_col_B * 4])[0];
        }
        __syncthreads();

        for (uint dot_idx = 0; dot_idx < BK; dot_idx++) {
            for (int subtile_row = 0; subtile_row < WMITER; subtile_row++) {
                for (int j = 0; j < TM; j++) {
                    regM[subtile_row * TM + j] = 
                        A_shared[(dot_idx) * BM + (warp_row * WM + subtile_row * WSUBM + thread_row_subtile * TM + j)];
                }
            }
            for (int subtile_col = 0; subtile_col < WNITER; subtile_col++) {
                for (int j = 0; j < TN; j++) {
                    regN[subtile_col * TN + j] = 
                        B_shared[(dot_idx) * BN + (warp_col * WN + subtile_col * WSUBN + thread_col_subtile * TN + j)];
                }
            }

            // matmul
            for (int subtile_row = 0; subtile_row < WMITER; subtile_row++) {
                for (int subtile_col = 0; subtile_col < WNITER; subtile_col++) {
                    for (int i = 0; i < TM; i++) {
                        for (int j = 0; j < TN; j++) {
                            thread_results[(subtile_row * TM + i) * (TN * WNITER) + (subtile_col * TN + j)] +=
                                regM[(subtile_row * TM) + i] * regN[(subtile_col * TN) + j];
                        }
                    }
                }
            }
        }

        A += BK;
        B += BK * N;
        __syncthreads();
    }

    // write the results
    for (uint subtile_row = 0; subtile_row < WMITER; subtile_row++) {
        for (uint subtile_col = 0; subtile_col < WNITER; subtile_col++) {
            float *C_interim = C + (subtile_row * WSUBM) * N + subtile_col * WSUBN;
            for (uint i = 0; i < TM; i++) {
                for (uint j = 0; j < TN; j += 4) {
                    float4 tmp = reinterpret_cast<float4 *>(
                        &C_interim[(thread_row_subtile * TM + i) * N + (thread_col_subtile * TN + j)]
                    )[0];
                    const uint idx = (subtile_row * TM + i) * (WNITER * TN) + (subtile_col * TN + j);
                    tmp.x = alpha * thread_results[idx + 0] + beta * tmp.x;
                    tmp.y = alpha * thread_results[idx + 1] + beta * tmp.y;
                    tmp.z = alpha * thread_results[idx + 2] + beta * tmp.z;
                    tmp.w = alpha * thread_results[idx + 3] + beta * tmp.w;
                    reinterpret_cast<float4 *>(
                        &C_interim[(thread_row_subtile * TM + i) * N + (thread_col_subtile * TN + j)]
                    )[0] = tmp;
                }
            }
        }
    }
}

void launch_sgemm_warp_tiling(int M, int N, int K, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 16;
    const uint TM = 8;
    const uint TN = 4;
    dim3 grid(CEIL_DIV(N, BN), CEIL_DIV(M, BM));
    dim3 block(128);

    sgemm_warp_tiling<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
}


} // namespace k6
