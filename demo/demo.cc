#include "dense/dense.h"

#include <print>

int main() {
  linalg::dense_static<double, 3, 3, RowMajor> m_s;
  m_s.set_arange(1);
  m_s.print();

  linalg::dense_dynamic<double, ColMajor> m_d(4, 4);
  linalg::dense_dynamic<double, ColMajor> m_d2(3, 3);
  m_d.set_arange(1);
  m_d2.set_ones();
  m_d.print();
  m_d2.print();

  linalg::dense_dynamic<double, ColMajor> res = m_d * m_d2;
  res.print();

  return 0;
}