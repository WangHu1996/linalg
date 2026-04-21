#ifndef __LINALG_DENSE_DYNAMIC__
#define __LINALG_DENSE_DYNAMIC__

#include "dense_base.h"
#include "include_md_all.h"

namespace linalg {

template <typename T = double, typename Layout = std::layout_left>
class dense_dynamic : public md::heap_storage<T>,
                      public md::multi_dim_dynamic<dense_dynamic<T, Layout>, T, 2, Layout>,
                      public md::iterator_contiguous<dense_dynamic<T, Layout>, T>,
                      public md::fill_op<dense_dynamic<T, Layout>, T> {
 public:
  using simd_policy = md::aligned_policy;
  using value_type = T;
  using layout_type = Layout;

  using Storage = md::heap_storage<T>;
  using MultiDim = md::multi_dim_dynamic<dense_dynamic<T, Layout>, T, 2, Layout>;
  using Iterator = md::iterator_contiguous<dense_dynamic<T, Layout>, T>;
  using FillOps = md::fill_op<dense_dynamic<T, Layout>, T>;

  // ============ 构造函数 ============

  dense_dynamic() = default;

  explicit dense_dynamic(const std::array<size_t, 2>& shape)
      : Storage(calculate_size(shape)), MultiDim(shape, std::make_index_sequence<2>{}) {}

  template <typename... Sizes>
    requires(sizeof...(Sizes) == 2 && (std::convertible_to<Sizes, size_t> && ...))
  explicit dense_dynamic(Sizes... sizes) : dense_dynamic(std::array<size_t, 2>{static_cast<size_t>(sizes)...}) {}

  // 拷贝/移动
  dense_dynamic(const dense_dynamic& other) : Storage(other.size()), MultiDim() {
    MultiDim::shape_ = other.shape_;
    MultiDim::init_mdspan(std::make_index_sequence<2>{});
    std::copy(other.begin(), other.end(), this->begin());
  }

  dense_dynamic(dense_dynamic&& other) noexcept = default;

  dense_dynamic& operator=(const dense_dynamic& other) {
    if (this != &other) {
      Storage::resize(other.size());
      MultiDim::shape_ = other.extents();
      MultiDim::init_mdspan(std::make_index_sequence<2>{});
      std::copy(other.begin(), other.end(), this->begin());
    }
    return *this;
  }

  dense_dynamic& operator=(dense_dynamic&& other) noexcept = default;

  // ============ 形状修改 ============

  void set_shape(const std::array<size_t, 2>& shape) {
    if (shape == MultiDim::shape_ && !MultiDim::mdspan_.empty()) {
      return;
    }
    Storage::resize(calculate_size(shape));
    MultiDim::shape_ = shape;
    MultiDim::init_mdspan(std::make_index_sequence<2>{});
  }

  template <typename... Sizes>
    requires(sizeof...(Sizes) == 2 && (std::convertible_to<Sizes, size_t> && ...))
  void set_shape(Sizes... sizes) {
    set_shape(std::array<size_t, 2>{static_cast<size_t>(sizes)...});
  }

  // ============ 转发接口 ============

  using Storage::data;
  using Storage::size;
  using Storage::used_size;
  using Storage::capacity;

  using MultiDim::extents;
  using MultiDim::extent;
  using MultiDim::rank;
  using MultiDim::operator();
  using MultiDim::operator[];
  using MultiDim::at;
  using MultiDim::get_1d_index;
  using MultiDim::get_md_index;
  using MultiDim::get_dim_index;
  using MultiDim::print;
  using MultiDim::mdspan;

  using Iterator::begin;
  using Iterator::end;
  using Iterator::cbegin;
  using Iterator::cend;
  using Iterator::rbegin;
  using Iterator::rend;
  using Iterator::crbegin;
  using Iterator::crend;

  using FillOps::fill;
  using FillOps::set_zeros;
  using FillOps::set_ones;
  using FillOps::set_arange;
  using FillOps::set_random_uniform;
  using FillOps::set_random_normal;

  // ============ 行列大小 ============

  const size_t rows() const noexcept { return extent(0); }
  const size_t cols() const noexcept { return extent(1); }
};

}  // namespace linalg

#endif  // __LINALG_DENSE_DYNAMIC__