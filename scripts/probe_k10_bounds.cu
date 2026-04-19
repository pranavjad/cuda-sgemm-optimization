#include <cstdio>

#include <cute/tensor.hpp>

int main() {
  using namespace cute;

  using Element = float;
  using CopyOp = UniversalCopy<uint_byte_t<16>>;
  using CopyAtom = Copy_Atom<CopyOp, Element>;

  auto BM = Int<128>{};
  auto BN = Int<128>{};
  auto BK = Int<16>{};
  auto TM = Int<8>{};
  auto TN = Int<4>{};
  auto WM = Int<64>{};
  auto WN = Int<64>{};
  auto WNITER = Int<4>{};
  auto WMITER = (WM * WN) / (WNITER * TM * TN * Int<32>{});
  auto WSUBM = WM / WMITER;
  auto WSUBN = WN / WNITER;

  auto A_shared_layout = make_layout(make_shape(BK, BM), LayoutRight{});
  auto B_shared_layout = make_layout(make_shape(BK, BN), LayoutRight{});
  auto C_layout = make_layout(make_shape(BM, BN), LayoutRight{});

  auto sA = make_tensor(counting_iterator<int>{}, A_shared_layout);
  auto sB = make_tensor(counting_iterator<int>{}, B_shared_layout);
  auto gC = make_tensor(counting_iterator<int>{}, C_layout);

  auto A_thr_layout = make_layout(make_shape(Int<64>{}, Int<2>{}));
  auto A_val_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));
  auto A_tiled_copy = make_tiled_copy(CopyAtom{}, A_thr_layout, A_val_layout);

  auto B_thr_layout = make_layout(make_shape(Int<4>{}, Int<32>{}));
  auto B_val_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));
  auto B_tiled_copy = make_tiled_copy(CopyAtom{}, B_thr_layout, B_val_layout);

  auto gA_tile = make_tensor(counting_iterator<int>{}, make_layout(make_shape(BM, BK), LayoutRight{}));
  auto gB_tile = make_tensor(counting_iterator<int>{}, make_layout(make_shape(BK, BN), LayoutRight{}));

  int max_a_store = -1;
  int max_b_dst = -1;
  int max_c_dst = -1;
  bool bad = false;

  for (int tid = 0; tid < 128; ++tid) {
    auto A_thr_copy = A_tiled_copy.get_thread_slice(tid);
    auto A_thr_src = A_thr_copy.partition_S(gA_tile);
    auto A_frag = make_fragment_like(A_thr_src);
    auto A_thr_coord = A_thr_layout.get_flat_coord(tid);

    for (int rest_m = 0; rest_m < size<1>(A_frag); ++rest_m) {
      for (int rest_k = 0; rest_k < size<2>(A_frag); ++rest_k) {
        auto sA_to_w = local_tile(
                    sA,
                    make_shape(Int<4>{}, Int<1>{}),
                    make_coord(rest_k * size<1>(A_thr_layout) + get<1>(A_thr_coord),
                               get<0>(A_thr_coord) + rest_m * size<0>(A_thr_layout)));
        for (int v = 0; v < size<0,0>(A_frag); ++v) {
          int off = sA_to_w(v);
          max_a_store = off > max_a_store ? off : max_a_store;
          if (off < 0 || off >= cosize(A_shared_layout)) {
            std::printf("bad A tid=%d rest_m=%d rest_k=%d v=%d off=%d\n", tid, rest_m, rest_k, v, off);
            bad = true;
          }
        }
      }
    }

    auto B_thr_copy = B_tiled_copy.get_thread_slice(tid);
    auto B_dst = B_thr_copy.partition_D(sB);
    for (int i = 0; i < size(B_dst); ++i) {
      int off = B_dst(i);
      max_b_dst = off > max_b_dst ? off : max_b_dst;
      if (off < 0 || off >= cosize(B_shared_layout)) {
        std::printf("bad B tid=%d i=%d off=%d\n", tid, i, off);
        bad = true;
      }
    }

    int warp_idx = tid / 32;
    int warp_row = warp_idx / (size(BN) / size(WN));
    int warp_col = warp_idx % (size(BN) / size(WN));
    int thread_id_warp = tid % 32;
    int thread_row_subtile = thread_id_warp / (size(WSUBN) / size(TN));
    int thread_col_subtile = thread_id_warp % (size(WSUBN) / size(TN));

    auto gC_warptile = local_tile(gC, make_shape(WM, WN), make_coord(warp_row, warp_col));
    auto gC_threadtiles = zipped_divide(gC_warptile, make_shape(TM, TN));
    for (int subtile_row = 0; subtile_row < size(WMITER); ++subtile_row) {
      for (int subtile_col = 0; subtile_col < size(WNITER); ++subtile_col) {
        auto threadtile = gC_threadtiles(
            make_coord(_, _),
            make_coord(subtile_row * (size(WSUBM) / size(TM)) + thread_row_subtile,
                       subtile_col * (size(WSUBN) / size(TN)) + thread_col_subtile));
        for (int i = 0; i < size(threadtile); ++i) {
          int off = threadtile(i);
          max_c_dst = off > max_c_dst ? off : max_c_dst;
          if (off < 0 || off >= cosize(C_layout)) {
            std::printf("bad C tid=%d i=%d off=%d\n", tid, i, off);
            bad = true;
          }
        }
      }
    }
  }

  std::printf("cosize A=%d max_a_store=%d\n", int(cosize(A_shared_layout)), max_a_store);
  std::printf("cosize B=%d max_b_dst=%d\n", int(cosize(B_shared_layout)), max_b_dst);
  std::printf("cosize C=%d max_c_dst=%d\n", int(cosize(C_layout)), max_c_dst);
  std::printf("%s\n", bad ? "BAD" : "OK");
}
