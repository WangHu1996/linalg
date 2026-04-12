#ifndef __LINALG_DENSE_STATIC__
#define __LINALG_DENSE_STATIC__

#include "dense_base.h"
#include "include_md_all.h"

namespace linalg {

template <typename T, size_t Rows, size_t Cols, typename Layout = std::layout_left>
class dense_static {
 public:
  md::array<T, Layout, Rows, Cols> data_;

 public:
  void print() { data_.print(); }

  void fill(T v) { data_.fill(v); }
};

}  // namespace linalg

#endif  // __LINALG_DENSE_STATIC__