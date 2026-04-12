#include <cstdio>

#include <cute/tensor.hpp>

template <class Layout>
void print_mapping(char const* name, Layout const& layout) {
  using namespace cute;

  std::printf("%s\n", name);
  std::printf("  layout = ");
  print(layout);
  std::printf("\n");

  for (int i = 0; i < 16; ++i) {
    auto coord = layout.get_flat_coord(i);
    std::printf("  %2d -> ", i);
    print(coord);
    std::printf(", layout(coord) = ");
    print(layout(coord));
    std::printf("\n");
  }
}

int main() {
  using namespace cute;

  auto left = make_layout(make_shape(Int<8>{}, Int<8>{}));
  auto right = make_layout(make_shape(Int<8>{}, Int<8>{}), LayoutRight{});

  print_mapping("default make_layout(shape)", left);
  print_mapping("make_layout(shape, LayoutRight{})", right);
}
