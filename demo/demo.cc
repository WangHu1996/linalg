#include "dense/dense.h"

#include <print>

int main() {
  linalg::dense_static<double, 3, 3, std::layout_right> m_s;
  m_s.set_arange(1);
  m_s.print();

  linalg::dense_dynamic<double, std::layout_left> m_d(3, 3);
  m_d.set_arange(1);
  m_d.print();

  return 0;
}