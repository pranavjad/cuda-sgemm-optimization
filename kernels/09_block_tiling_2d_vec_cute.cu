#include <cstdlib>
#include <cstdio>
#include <cassert>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

#include <cute/tensor.hpp>

// #include "cutlass/util/print_error.hpp"
// #include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

namespace k9 {



template <class ATiler, class BTiler, class CTiler,
          class AStride, class ASmemLayout, class AThreadLayout,
          class BStride, class BSmemLayout, class BThreadLayout,
          class CStride, class CThreadLayout
          >
__global__ void sgemm_2d_block_tiling_vec_cute(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    AStride A_strides, BStride B_strides, CStride C_strides,
    ATiler A_tiler, BTiler B_tiler, CTiler C_tiler,
    ASmemLayout A_shared_layout, BSmemLayout B_shared_layout,
    AThreadLayout A_thread_layout, BThreadLayout B_thread_layout, CThreadLayout C_thread_layout
) {
    using namespace cute;

    // create Tensors from data + Layout
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), A_strides);
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(K, N), B_strides);
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), C_strides);

    // global tiles
    Tensor gA = local_tile(mA, A_tiler, make_coord(blockIdx.y, _)); // (BM, BK, k)
    Tensor gB = local_tile(mB, B_tiler, make_coord(_, blockIdx.x)); // (BK, BN, k)
    Tensor gC = local_tile(mC, C_tiler, make_coord(blockIdx.y, blockIdx.x)); // (BM, BN)

    // shared tiles
    __shared__ float A_smem[cosize_v<ASmemLayout>];
    __shared__ float B_smem[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(A_smem), A_shared_layout);
    Tensor sB = make_tensor(make_smem_ptr(B_smem), B_shared_layout);

    // part of gA each thread loads to sA
    Tensor gA_to_r = local_partition(gA, A_thread_layout, threadIdx.x);
    Tensor sA_to_w = local_partition(sA, A_thread_layout, threadIdx.x);

    // part of gB each thread loads to sB
    Tensor gB_to_r = local_partition(gB, B_thread_layout, threadIdx.x);
    Tensor sB_to_w = local_partition(sB, B_thread_layout, threadIdx.x);

    // part of sA, sB each thread reads for computation
    auto BN = shape<1>(C_tiler);
    auto TM = shape<0>(C_thread_layout);
    auto TN = shape<1>(C_thread_layout);
    auto thread_row_C = threadIdx.x / (BN / TN);
    auto thread_col_C = threadIdx.x % (BN / TN);
    auto A_col_shape = make_shape(TM, 1);
    auto B_row_shape = make_shape(1, TN);
    Tensor sA_to_r = local_tile(sA, A_col_shape, make_coord(thread_row_C, _)); // (TM, 1, BK)
    Tensor sB_to_r = local_tile(sB, B_row_shape, make_coord(_, thread_col_C)); // (1, TN, BK)

    // if(thread0()) {
    //     print(sA_to_r);
    //     print(sB_to_r);
    // }

    // part of gC each thread writes results
    Tensor gC_to_w = local_tile(gC, shape(C_thread_layout), make_coord(thread_row_C, thread_col_C)); // (TM, TN)
    Tensor thread_results = make_tensor_like(gC_to_w);
    clear(thread_results);

    Tensor tmp_A = make_tensor_like<float>(make_layout(make_shape(TM)));
    Tensor tmp_B = make_tensor_like<float>(make_layout(make_shape(TN)));



    auto max_tile_idx = shape<2>(gA);
    for (int tile_idx = 0; tile_idx < max_tile_idx; tile_idx++) {
        // load tiles
        Tensor gA_tile = gA_to_r(_, _, tile_idx);
        CUTE_UNROLL
        for (int i = 0; i < size(gA_tile); i++) {
            sA_to_w(i) = gA_tile(i);
        }
        Tensor gB_tile = gB_to_r(_, _, tile_idx);
        CUTE_UNROLL
        for (int i = 0; i < size(gB_tile); i++) {
            sB_to_w(i) = gB_tile(i);
        }

        __syncthreads();

        // compute partial results for this tile
        CUTE_UNROLL
        for (int dot_idx = 0; dot_idx < shape<2>(sA_to_r); dot_idx++) {
            // load row/col from smem to rmem
            Tensor sA_col = sA_to_r(_, _, dot_idx);
            Tensor sB_row = sB_to_r(_, _, dot_idx);
            CUTE_UNROLL
            for (int i = 0; i < size(tmp_A); i++) {
                tmp_A(i) = sA_col(i);
            }
            CUTE_UNROLL
            for (int i = 0; i < size(tmp_B); i++) {
                tmp_B(i) = sB_row(i);
            }

            // outer product
            CUTE_UNROLL
            for (int i = 0; i < shape<0>(thread_results); i++) {
                CUTE_UNROLL
                for (int j = 0; j < shape<1>(thread_results); j++) {
                    thread_results(i, j) += tmp_A(i) * tmp_B(j);
                }
            }
        }
        __syncthreads();
    }

    // write results back to gmem
    CUTE_UNROLL
    for (int i = 0; i < shape<0>(thread_results); i++) {
        CUTE_UNROLL
        for (int j = 0; j < shape<1>(thread_results); j++) {
            gC_to_w(i, j) = alpha * thread_results(i, j) + beta * gC_to_w(i, j);
        }
    }
} 


void launch_sgemm_2d_block_tiling_vec_cute(
    int m, int n, int k,
    float alpha, float* d_A, float* d_B, float beta, float* d_C
) {
    using namespace cute;
    // define problem shape
    auto M = int(m);
    auto N = int(n);
    auto K = int(k);
    // define strides of A, B, C
    auto A_strides = make_stride(K, Int<1>{});
    auto B_strides = make_stride(N, Int<1>{});
    auto C_strides = make_stride(N, Int<1>{});

    // define blocktile size
    auto BM = Int<64>{};
    auto BN = Int<64>{};
    auto BK = Int<8>{};
    // auto block_tiler = make_shape(BM, BN, BK);
    auto A_tiler = make_shape(BM, BK);
    auto B_tiler = make_shape(BK, BN);
    auto C_tiler = make_shape(BM, BN);

    // define smem layouts
    // maps coords in (BM, BK) --> 1d offset in smem buffer
    auto A_shared_layout = make_layout(make_shape(BM, BK), LayoutRight{});
    auto B_shared_layout = make_layout(make_shape(BK, BN), LayoutRight{});

    // define thread layouts
    // maps coords in (BM, BK) --> thread index that loads it
    auto A_thread_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), LayoutRight{});
    auto B_thread_layout = make_layout(make_shape(Int<1>{}, Int<64>{}), LayoutRight{});
    auto C_thread_layout = make_layout(make_shape(Int<8>{}, Int<8>{}));

    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    dim3 block(8 * 8);

    sgemm_2d_block_tiling_cute<<<grid, block>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C,
        A_strides, B_strides, C_strides,
        A_tiler, B_tiler, C_tiler,
        A_shared_layout, B_shared_layout,
        A_thread_layout, B_thread_layout, C_thread_layout
    );
}


} // namespace k9
