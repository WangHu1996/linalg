#include <concepts>

template <typename T>
concept DenseMatrixType = requires(T m) {
  typename T::Scalar;
  { m.rows() } -> std::convertible_to<size_t>;
  { m.cols() } -> std::convertible_to<size_t>;
  { m(0, 0) } -> std::same_as<typename T::Scalar&>;
};