#ifndef __LINALG_DENSE_STATIC__
#define __LINALG_DENSE_STATIC__

#include "dense_base.h"
#include "include_md_all.h"

namespace linalg {

template <typename T, size_t Rows, size_t Cols, typename Layout = std::layout_left>
class dense_static : public md::stack_storage<T, Rows, Cols>,
                     public md::multi_dim_static<dense_static<T, Rows, Cols, Layout>, T, Layout, Rows, Cols>,
                     public md::iterator_contiguous<dense_static<T, Rows, Cols, Layout>, T>,
                     public md::fill_op<dense_static<T, Rows, Cols, Layout>, T> {
 public:
  using simd_policy = md::aligned_policy;
  using value_type = T;
  using layout_type = Layout;

  using Storage = md::stack_storage<T, Rows, Cols>;
  using MultiDim = md::multi_dim_static<dense_static<T, Rows, Cols, Layout>, T, Layout, Rows, Cols>;
  using Iterator = md::iterator_contiguous<dense_static<T, Rows, Cols, Layout>, T>;
  using FillOps = md::fill_op<dense_static<T, Rows, Cols, Layout>, T>;

  // ============ 构造函数 ============

  dense_static() = default;

  ~dense_static() = default;

  dense_static(const dense_static& other) = default;

  dense_static(dense_static&& other) = default;

  dense_static& operator=(const dense_static& other) = default;

  dense_static& operator=(dense_static&& other) = default;

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

  const size_t rows() const noexcept { return Rows; }
  const size_t cols() const noexcept { return Cols; }
};

}  // namespace linalg

#endif  // __LINALG_DENSE_STATIC__