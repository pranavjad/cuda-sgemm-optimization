
#include <cstdlib>
#include <cstdio>
#include <cassert>

// #include <thrust/host_vector.h>
// #include <thrust/device_vector.h>

#include <cute/tensor.hpp>

// #include "cutlass/util/print_error.hpp"
// #include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/helper_cuda.hpp"

namespace k10 {
using namespace cute;
using VecType = uint_bit_t<128>;
static constexpr auto WARPSIZE = Int<32>{};
static constexpr auto BM = Int<128>{};
static constexpr auto BN = Int<128>{};
static constexpr auto BK = Int<16>{};
static constexpr auto TM = Int<8>{};
static constexpr auto TN = Int<4>{};
static constexpr auto WM = Int<64>{};
static constexpr auto WN = Int<64>{};
static constexpr auto WNITER = Int<4>{};
static constexpr auto WMITER = (WM * WN) / (WNITER * TM * TN * WARPSIZE);
static constexpr auto WSUBM = WM / WMITER;
static constexpr auto WSUBN = WN / WNITER;
static constexpr auto NUM_THREADS = 128;


template <class ATiler, class BTiler, class CTiler,
          class AStride, class ASmemLayout, class AThreadLayout,
          class BStride, class BSmemLayout, class BThreadLayout,
          class CStride, class CThreadLayout
          >
__global__ void sgemm_warp_tiling_cute(
    int M, int N, int K,
    float alpha, const float *A, const float *B, float beta, float *C,
    AStride A_strides, BStride B_strides, CStride C_strides,
    ATiler A_tiler, BTiler B_tiler, CTiler C_tiler,
    ASmemLayout A_shared_layout, BSmemLayout B_shared_layout,
    AThreadLayout A_thread_layout, BThreadLayout B_thread_layout, CThreadLayout C_thread_layout
) {

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

    // set up gA to sA copy
    using Element = float;
    using CopyOp = UniversalCopy<uint_byte_t<16>>;
    using CopyAtom = Copy_Atom<CopyOp, Element>;
    auto A_thr_layout = make_layout(make_shape(Int<32>{}, Int<4>{}), LayoutRight{});
    auto A_val_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));
    auto A_tiled_copy = make_tiled_copy(CopyAtom{}, A_thr_layout, A_val_layout);
    auto A_thr_copy = A_tiled_copy.get_thread_slice(threadIdx.x);
    auto A_thr_coord = A_thr_layout.get_flat_coord(threadIdx.x);
    auto A_vec_coord = A_thr_layout.get_flat_coord(threadIdx.x);
    
    // set up gB to sB copy
    auto B_thr_layout = make_layout(make_shape(Int<4>{}, Int<32>{}), LayoutRight{});
    auto B_val_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));
    auto B_tiled_copy = make_tiled_copy(CopyAtom{}, B_thr_layout, B_val_layout);
    auto B_thr_copy = B_tiled_copy.get_thread_slice(threadIdx.x);

    // part of sA, sB we touch according to warp tile
    const uint warp_idx = threadIdx.x / WARPSIZE;
    const uint warp_row = warp_idx / (BN / WN);
    const uint warp_col = warp_idx % (BN / WN);
    const uint thread_id_warp = threadIdx.x % WARPSIZE;
    const uint thread_row_subtile = thread_id_warp / (WSUBN / TN);
    const uint thread_col_subtile = thread_id_warp % (WSUBN / TN);
    Tensor sA_warp = local_tile(sA, make_shape(BK, WM), make_coord(Int<0>{}, warp_row));
    Tensor sB_warp = local_tile(sB, make_shape(BK, WN), make_coord(Int<0>{}, warp_col));


    // part of sA, sB each thread loads for computation
    auto A_col_shape = make_shape(Int<1>{}, TM);
    auto B_row_shape = make_shape(Int<1>{}, TN);
    Tensor sA_threadtile_cols = zipped_divide(sA_warp, A_col_shape); // ((1, 4), (16, 16))
    Tensor sB_threadtile_rows = zipped_divide(sB_warp, B_row_shape);
   
    // part of gC each thread should write
    Tensor gC_warptile = local_tile(gC, make_shape(WM, WN), make_coord(warp_row, warp_col));
    Tensor gC_threadtiles = zipped_divide(gC_warptile, shape(C_thread_layout));

    // registers
    Tensor regM = make_tensor_like<float>(make_layout(make_shape(TM, WMITER)));
    Tensor regN = make_tensor_like<float>(make_layout(make_shape(TN, WNITER)));
    Tensor thread_results = make_tensor_like<float>(
        make_layout(make_shape(make_shape(TM, TN), make_shape(WMITER, WNITER)))
    );
    clear(thread_results);
    clear(regM);
    clear(regN);

    auto max_tile_idx = shape<2>(gA);
    for (int tile_idx = 0; tile_idx < max_tile_idx; tile_idx++) {
        // load tiles

        Tensor gA_tile = gA(_, _, tile_idx);
        auto A_thr_src = A_thr_copy.partition_S(gA_tile); // ((4, 1), 4, 1) each thread does 4 vectorized loads
        auto A_frag = make_fragment_like(A_thr_src);
        copy(A_tiled_copy, A_thr_src, A_frag);
        CUTE_UNROLL
        for (int rest_m = 0; rest_m < size<1>(A_frag); rest_m++) { // size<1>(A_frag) = 4
            CUTE_UNROLL
            for (int rest_k = 0; rest_k < size<2>(A_frag); rest_k++) { // size<2>(A_frag) = 1
                Tensor sA_to_w = local_tile(
                    sA,
                    make_shape(Int<4>{}, Int<1>{}),
                    make_coord(get<1>(A_vec_coord), get<0>(A_vec_coord) + rest_m * size<0>(A_thr_layout))
                );
                CUTE_UNROLL
                for (int v = 0; v < size<0,0>(A_frag); v++) {
                    sA_to_w(v) = A_frag(make_coord(v, Int<0>{}), rest_m, rest_k);
                }
            }
        }
        
        Tensor gB_tile = gB(_, _, tile_idx);
        auto B_thr_src = B_thr_copy.partition_S(gB_tile);
        auto B_thr_dst = B_thr_copy.partition_D(sB);
        copy(B_tiled_copy, B_thr_src, B_thr_dst);

        __syncthreads();

        // compute partial results for this tile
        CUTE_UNROLL
        for (int dot_idx = 0; dot_idx < BK; dot_idx++) {
            // smem to rmem
            CUTE_UNROLL
            for (int subtile_row = 0; subtile_row < WMITER; subtile_row++) {
                Tensor sA_to_r = sA_threadtile_cols(_, make_coord(
                    dot_idx,
                    subtile_row * (WSUBM / TM) + thread_row_subtile));
                copy(sA_to_r, regM(_, subtile_row));
            }

            CUTE_UNROLL
            for (int subtile_col = 0; subtile_col < WNITER; subtile_col++) {
                Tensor sB_to_r = sB_threadtile_rows(_, make_coord(
                    dot_idx,
                    subtile_col * (WSUBN / TN) + thread_col_subtile));
                copy(sB_to_r, regN(_, subtile_col));
            }

            // matmul
            CUTE_UNROLL
            for (int subtile_row = 0; subtile_row < WMITER; subtile_row++) {
                CUTE_UNROLL
                for (int subtile_col = 0; subtile_col < WNITER; subtile_col++) {
                    CUTE_UNROLL
                    for (int i = 0; i < TM; i++) {
                        CUTE_UNROLL
                        for (int j = 0; j < TN; j++) {
                            thread_results(
                                make_coord(make_coord(i, j),
                                make_coord(subtile_row, subtile_col))
                            ) += regM(i, subtile_row) * regN(j, subtile_col);
                        }
                    }
                }
            }


        }
        __syncthreads();
    }

    // write the results
    CUTE_UNROLL
    for (uint subtile_row = 0; subtile_row < WMITER; subtile_row++) {
        CUTE_UNROLL
        for (uint subtile_col = 0; subtile_col < WNITER; subtile_col++) {
            auto threadtile = gC_threadtiles(make_coord(_, _), make_coord(
                subtile_row * (WSUBM / TM) + thread_row_subtile,
                subtile_col * (WSUBN / TN) + thread_col_subtile));
            CUTE_UNROLL
            for (uint i = 0; i < TM; i++) {
                CUTE_UNROLL
                for (uint j = 0; j < TN; j += 4) {
                    Tensor tmp = make_tensor<float>(make_shape(Int<1>{}, Int<4>{}));
                    Tensor dst = local_tile(threadtile, make_shape(Int<1>{}, Int<4>{}), make_coord(i, j / 4));
                    CUTE_UNROLL
                    for (int k = 0; k < size(tmp); k++) {
                        tmp(k) = alpha * thread_results(
                            make_coord(make_coord(i, j + k), make_coord(subtile_row, subtile_col))
                        ) + beta * dst(k);
                    }
                    copy_aligned(tmp, dst);
                }
            }
        }
    }
} 


void launch_sgemm_warp_tiling_cute(
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
    auto A_tiler = make_shape(BM, BK);
    auto B_tiler = make_shape(BK, BN);
    auto C_tiler = make_shape(BM, BN);

    // define smem layouts
    auto A_shared_layout = make_layout(make_shape(BK, BM), LayoutRight{});
    auto B_shared_layout = make_layout(make_shape(BK, BN), LayoutRight{});

    // define thread layouts
    auto A_thread_layout = make_layout(make_shape(BM, BK), LayoutRight{});
    auto B_thread_layout = make_layout(make_shape(BK, BN), LayoutRight{});
    auto C_thread_layout = make_layout(make_shape(TM, TN));


    dim3 grid(ceil_div(N, BN), ceil_div(M, BM));
    dim3 block((BM * BN) / (WM * WN) * WARPSIZE);

    sgemm_warp_tiling_cute<<<grid, block>>>(
        M, N, K, alpha, d_A, d_B, beta, d_C,
        A_strides, B_strides, C_strides,
        A_tiler, B_tiler, C_tiler,
        A_shared_layout, B_shared_layout,
        A_thread_layout, B_thread_layout, C_thread_layout
    );
}


} // namespace k10
