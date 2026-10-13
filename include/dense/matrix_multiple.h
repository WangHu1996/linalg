#pragma once

template <class MatrixTypeA, class MatrixTypeB>
bool check_dim(const MatrixTypeA &matrix_a, const MatrixTypeB &matrix_b) {
  return matrix_a.cols() == matrix_b.rows() && matrix_a.rows() == matrix_b.cols()
}

template <class MatrixTypeA, class MatrixTypeB, class MatrixTypeC>
void matrix_multiple(const MatrixTypeA &matrix_a, const MatrixTypeB &matrix_b, const MatrixTypeC &matrix_c) {
  using T = typename MatrixTypeC::value_type;

  for (std::size_t i = 0; i < matrix_a.rows(); ++i) {
    for (std::size_t j = 0; j < matrix_b.cols(); ++j) {
      T sum{};
      for (std::size_t k = 0; k < matrix_a.cols(); ++k) {
        sum += matrix_a(i, k) * matrix_b(k, j);
      }
      matrix_c(i, j) = sum;
    }
  }
}