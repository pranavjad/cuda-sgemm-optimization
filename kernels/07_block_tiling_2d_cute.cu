#include <cstdlib>
#include <cstdio>
#include <cassert>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

#include <cute/tensor.hpp>

// #include "cutlass/util/print_error.hpp"
// #include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

namespace k7 {
template <class BlockTiler,
          class AStride, class ASmemLayout, class AThreadLayout,
          class BStride, class BSmemLayout, class BThreadLayout,
          class CStride, class CThreadLayout
          >
__global__ void sgemm_2d_block_tiling_cute(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    AStride A_strides, BStride B_strides, CStride C_strides,
    BlockTiler block_tiler,
    ASmemLayout A_shared_layout, BSmemLayout B_shared_layout,
    AThreadLayout A_thread_layout, BThreadLayout B_thread_layout, CThreadLayout C_thread_layout
) {
    using namespace cute;
    // create Tensors from data + Layout
    Tensor mA = make_tensor(make_gmem_ptr(A), make_shape(M, K), A_strides);
    Tensor mB = make_tensor(make_gmem_ptr(B), make_shape(K, N), B_strides);
    Tensor mC = make_tensor(make_gmem_ptr(C), make_shape(M, N), C_strides);

    // create Tensors for the chunks of mA, mB, mC that this block is handling
    // local_tile(tensor, tiler, coord, proj)
    // tensor = the data to grab the tile from
    // tiler = shape of one tile
    // coord = logical coordinates of the tile
    // proj = which dimensions of the tile we care about
    auto block_coord = make_coord(blockIdx.x, blockIdx.y, _);
    Tensor gA = local_tile(mA, block_tiler, block_coord, Step<_1, X, _1>{}); // (BM, BK, K/BK) <-- this view contains all tiles
    Tensor gB = local_tile(mB, block_tiler, block_coord, Step<X, _1, _1>{});
    Tensor gC = local_tile(mC, block_tiler, block_coord, Step<_1, _1, X>{});

    // shared memory buffers
    __shared__ float A_smem[cosize_v<ASmemLayout>];
    __shared__ float B_smem[cosize_v<BSmemLayout>];
    Tensor sA = make_tensor(make_smem_ptr(A_smem), A_shared_layout);
    Tensor sB = make_tensor(make_smem_ptr(B_smem), B_shared_layout);


    // create Tensors for the chunk of A_global this thread reads,
    // and the chunk of A_shared it writes
    // local partition: split a Tensor according to some layout, and give thread x's part.
    Tensor gA_to_r = local_partition(gA, A_thread_layout, threadIdx.x);
    Tensor sA_to_w = local_partition(sA, A_thread_layout, threadIdx.x);

    Tensor gB_to_r = local_partition(gB, B_thread_layout, threadIdx.x);
    Tensor sB_to_w = local_partition(sB, B_thread_layout, threadIdx.x);

    // create Tensors for the chunk of A_shared this thread reads for computation
    // Step<_1, X>{} indicates that we want to split up sA by the num of rows in C_thread_layout
    Tensor sA_to_r = local_partition(sA, C_thread_layout, threadIdx.x, Step<_1, X>{});
    Tensor sB_to_r = local_partition(sB, C_thread_layout, threadIdx.x, Step<X,_1>{});

    // create Tensor for the chunk of C_global this thread writes
    Tensor gC_to_w = local_partition(gC, C_thread_layout, threadIdx.x, Step<_1, _1>{});

    // create register array for thread results and zero it
    Tensor thread_results = make_tensor_like(gC_to_w);
    clear(thread_results);

    // slide the blocktile over K
    auto K_TILE_MAX = size<2>(gA_to_r);
    for (int k_tile = 0; k_tile < K_TILE_MAX; k_tile++) {
        // load gmem --> smem
        Tensor gA_tile = gA_to_r(_, _, k_tile); // grab the tile
        CUTE_UNROLL
        for (int i = 0; i < size(sA_to_w); i++) {
            // 1D index because Layouts handles indexing calculation internally
            sA_to_w(i) = gA_tile(i);
        }

        Tensor gB_tile = gB_to_r(_, _, k_tile);
        CUTE_UNROLL
        for (int i = 0; i < size(sB_to_w); i++) {
            sB_to_w(i) = gB_tile(i);
        }

        cp_async_fence();
        cp_async_wait<0>();
        __syncthreads();

        // Compute partial results for this tile
        CUTE_UNROLL
        for (int k = 0; k < size<1>(sA_to_r); k++) {
            CUTE_UNROLL
            for (int i = 0; i < size<0>(thread_results); i++) {
                CUTE_UNROLL
                for (int j = 0; j < size<1>(thread_results); j++) {
                    thread_results(i, j) += sA_to_r(i, k) * sB_to_r(k, j);
                }
            }
        }
        __syncthreads();
    }

    // Write results back to global memory
    CUTE_UNROLL
    for (int i = 0; i < size(thread_results); i++) {
        gC_to_w(i) = alpha * thread_results(i) + beta * gC_to_w(i);
    }
}



void launch_sgemm_2d_block_tiling_cute(
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
    auto block_tiler = make_shape(BM, BN, BK);

    // define smem layouts
    // maps coords in (BM, BK) --> 1d offset in smem buffer
    auto A_shared_layout = make_layout(make_shape(BM, BK));
    auto B_shared_layout = make_layout(make_shape(BK, BN));

    // define thread layouts
    // maps coords in (BM, BK) --> thread index that loads it
    auto A_thread_layout = make_layout(make_shape(Int<8>{}, Int<8>{}));
    auto B_thread_layout = make_layout(make_shape(Int<8>{}, Int<8>{}));
    auto C_thread_layout = make_layout(make_shape(Int<8>{}, Int<8>{}));

    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    dim3 block(8 * 8);

    sgemm_2d_block_tiling_cute<<<grid, block>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C,
        A_strides, B_strides, C_strides,
        block_tiler,
        A_shared_layout, B_shared_layout,
        A_thread_layout, B_thread_layout, C_thread_layout
    );
}

} // namespace k7