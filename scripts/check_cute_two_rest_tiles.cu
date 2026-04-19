#include <cstdio>

#include <cute/tensor.hpp>

int main() {
  using namespace cute;

  auto tensor = make_identity_tensor(
      make_shape(make_shape(Int<1>{}, Int<4>{}),
                 make_shape(Int<128>{}, Int<2>{})));

  std::printf("tensor: ");
  print(tensor);
  std::printf("\n");

  auto rest_to_orig = logical_product(
      make_layout(make_shape(Int<64>{}, Int<2>{}),
                  make_stride(Int<1>{}, Int<64>{})),
      Int<2>{});
  std::printf("logical_product rest layout: ");
  print(rest_to_orig);
  std::printf("\n");
  auto split_rest_tensor = tensor.compose(_, rest_to_orig);

  std::printf("split rest tensor: ");
  print(split_rest_tensor);
  std::printf("\n");

  auto thr_layout = make_layout(make_shape(Int<64>{}, Int<2>{}));

  int tid = 0;
  auto thr_coord = thr_layout.get_flat_coord(tid);
  auto thr_tensor = split_rest_tensor(_, make_coord(make_coord(get<0>(thr_coord), _), get<1>(thr_coord)));

  std::printf("thread %d tensor: ", tid);
  print(thr_tensor);
  std::printf("\n");
  std::printf("shape: ");
  print(shape(thr_tensor));
  std::printf("\n");

  for (int i = 0; i < size(thr_tensor); ++i) {
    std::printf("  flat %d -> ", i);
    print(thr_tensor(i));
    std::printf("\n");
  }

  tid = 7;
  auto thr7_coord = thr_layout.get_flat_coord(tid);
  auto thr7_tensor = split_rest_tensor(_, make_coord(make_coord(get<0>(thr7_coord), _), get<1>(thr7_coord)));
  std::printf("thread %d tensor: ", tid);
  print(thr7_tensor);
  std::printf("\n");
  for (int i = 0; i < size(thr7_tensor); ++i) {
    std::printf("  flat %d -> ", i);
    print(thr7_tensor(i));
    std::printf("\n");
  }

  tid = 64;
  auto thr64_coord = thr_layout.get_flat_coord(tid);
  auto thr64_tensor = split_rest_tensor(_, make_coord(make_coord(get<0>(thr64_coord), _), get<1>(thr64_coord)));
  std::printf("thread %d tensor: ", tid);
  print(thr64_tensor);
  std::printf("\n");
  for (int i = 0; i < size(thr64_tensor); ++i) {
    std::printf("  flat %d -> ", i);
    print(thr64_tensor(i));
    std::printf("\n");
  }
}
