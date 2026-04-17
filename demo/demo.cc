#include "include_md_all.h"

#include <print>

int main() {
  md::vector<double, 2> vec(3, 3);
  vec.set_arange();
  vec.print();

  return 0;
}