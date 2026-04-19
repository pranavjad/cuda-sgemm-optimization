#include <cstdio>

#include <cute/tensor.hpp>

int main() {
  using namespace cute;

  using Element = float;
  using CopyOp = UniversalCopy<uint_byte_t<16>>;
  using CopyAtom = Copy_Atom<CopyOp, Element>;

  auto BM = Int<128>{};
  auto BK = Int<8>{};
  auto K = Int<4096>{};

  auto gA = make_tensor(counting_iterator<int>{}, make_layout(make_shape(BM, BK), make_stride(K, Int<1>{})));
  auto sA_physical = make_tensor(counting_iterator<int>{}, make_layout(make_shape(BK, BM), LayoutRight{}));

  auto sA_as_gA = make_tensor(
      sA_physical.data(),
      make_layout(make_shape(BM, BK), make_stride(Int<1>{}, BM)));

  auto thr_layout = make_layout(make_shape(Int<64>{}, Int<2>{}));
  auto val_layout = make_layout(make_shape(Int<1>{}, Int<4>{}));
  auto tiled_copy = make_tiled_copy(CopyAtom{}, thr_layout, val_layout);
  auto thr_copy = tiled_copy.get_thread_slice(0);

  auto src = thr_copy.partition_S(gA);
  auto dst_wrong = thr_copy.partition_D(sA_physical);
  auto dst_transposed = thr_copy.partition_D(sA_as_gA);

  std::printf("tiled_copy S layout: ");
  print(src);
  std::printf("\n");

  std::printf("partition_D physical sA (BK,BM): ");
  print(dst_wrong);
  std::printf("\n");
  for (int i = 0; i < size(dst_wrong) && i < 8; ++i) {
    std::printf("  wrong %d -> ", i);
    print(dst_wrong(i));
    std::printf("\n");
  }

  std::printf("partition_D transposed view of sA as (BM,BK): ");
  print(dst_transposed);
  std::printf("\n");
  for (int i = 0; i < size(dst_transposed) && i < 8; ++i) {
    std::printf("  transposed %d -> ", i);
    print(dst_transposed(i));
    std::printf("\n");
  }
}
