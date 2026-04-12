#ifndef __LINALG_DENSE_DYNAMIC__
#define __LINALG_DENSE_DYNAMIC__

#include "dense_base.h"
#include "include_md_all.h"

namespace linalg {

template <typename T = double, typename Layout = std::layout_left>
class dense_dynamic {
 public:
  md::vector<T, 2, Layout> data_;

 public:
  void set_shape(const std::array<std::size_t, 2>& shape) { data_.set_shape(shape); }

  void print() { data_.print(); }

  void fill(T v) { data_.fill(v); }
};

}  // namespace linalg

#endif  // __LINALG_DENSE_DYNAMIC__