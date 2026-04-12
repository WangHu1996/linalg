#include "dense/dense.h"

int main() {
  print_simd_type();

  // dense dynamic
  std::println("dense dynamic 2x2:");
  linalg::dense_dynamic<double> dense;
  dense.set_shape({2, 2});
  dense.fill(2);
  dense.print();

  // dense static
  std::println("dense static 2x2:");
  linalg::dense_static<double, 2, 2> dense_s;
  dense_s.fill(3);
  dense_s.print();

  return 0;
}