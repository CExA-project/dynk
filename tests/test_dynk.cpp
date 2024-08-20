#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <gtest/gtest.h>

#include "dynk/dynamic_kokkos.hpp"

namespace {

void test_parallel_for_range(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView data("data", 10);

  dynk::parallel_for(
      isExecutedOnDevice, "label", Kokkos::RangePolicy(0, 10),
      KOKKOS_LAMBDA(int const i) {
        if (isExecutedOnDevice) {
          data.d_view(i) = i;
        } else {
          data.h_view(i) = i;
        }
      });
  if (isExecutedOnDevice) {
    data.template modify<typename DualView::execution_space>();
  } else {
    data.template modify<typename DualView::host_mirror_space>();
  }

  data.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(data.h_view(5), 5);
}

} // namespace

TEST(test_parallel_for, test_range) {
  test_parallel_for_range(true);
  test_parallel_for_range(false);
}

namespace {

void test_parallel_for_range_simple(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView data("data", 10);

  dynk::parallel_for(
      isExecutedOnDevice, "label", 10, KOKKOS_LAMBDA(int const i) {
        if (isExecutedOnDevice) {
          data.d_view(i) = i;
        } else {
          data.h_view(i) = i;
        }
      });
  if (isExecutedOnDevice) {
    data.template modify<typename DualView::execution_space>();
  } else {
    data.template modify<typename DualView::host_mirror_space>();
  }

  data.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(data.h_view(5), 5);
}

} // namespace

TEST(test_parallel_for, test_range_simple) {
  test_parallel_for_range_simple(true);
  test_parallel_for_range_simple(false);
}

namespace {

void test_parallel_for_mdrange(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int **>;
  DualView data("data", 10, 10);

  dynk::parallel_for(
      isExecutedOnDevice, "label",
      Kokkos::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {10, 10}),
      KOKKOS_LAMBDA(int const i, int const j) {
        if (isExecutedOnDevice) {
          data.d_view(i, j) = i * 100 + j;
        } else {
          data.h_view(i, j) = i * 100 + j;
        }
      });
  if (isExecutedOnDevice) {
    data.template modify<typename DualView::execution_space>();
  } else {
    data.template modify<typename DualView::host_mirror_space>();
  }

  data.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(data.h_view(4, 6), 406);
}

} // namespace

TEST(test_parallel_for, test_mdrange) {
  test_parallel_for_mdrange(true);
  test_parallel_for_mdrange(false);
}

namespace {

void test_parallel_reduce_range(bool const isExecutedOnDevice) {
  int value = 0;
  dynk::parallel_reduce(
      isExecutedOnDevice, "label", Kokkos::RangePolicy(0, 10),
      KOKKOS_LAMBDA(int const i, int &valueLocal) { valueLocal += 1; }, value);

  EXPECT_EQ(value, 10);
}

} // namespace

TEST(test_parallel_reduce, test_range) {
  test_parallel_reduce_range(true);
  test_parallel_reduce_range(false);
}
