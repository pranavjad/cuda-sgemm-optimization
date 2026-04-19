#include <cstdio>

#include <cute/tensor.hpp>

int main() {
  using namespace cute;

  auto BM = Int<128>{};
  auto BN = Int<128>{};
  auto TM = Int<8>{};
  auto TN = Int<4>{};
  auto WM = Int<64>{};
  auto WN = Int<64>{};

  auto C_layout = make_layout(make_shape(BM, BN), LayoutRight{});
  auto gC = make_tensor(counting_iterator<int>{}, C_layout);
  auto gC_warptile = local_tile(gC, make_shape(WM, WN), make_coord(0, 0));

  auto c_layout_tiler = make_layout(make_shape(TM, TN));
  auto c_shape_tiler  = make_shape(TM, TN);

  auto by_layout = zipped_divide(gC_warptile, c_layout_tiler);
  auto by_shape  = zipped_divide(gC_warptile, c_shape_tiler);

  std::printf("gC_warptile: ");
  print(gC_warptile);
  std::printf("\nby_layout: ");
  print(by_layout);
  std::printf("\nby_shape : ");
  print(by_shape);
  std::printf("\n");

  for (int rest_m = 0; rest_m < 8; ++rest_m) {
    auto tile_layout = by_layout(make_coord(_, _), make_coord(rest_m, 0));
    auto tile_shape = by_shape(make_coord(_, _), make_coord(rest_m, 0));
    std::printf("rest_m=%d layout first=%d shape first=%d\n", rest_m, int(tile_layout(0)), int(tile_shape(0)));
  }
}
