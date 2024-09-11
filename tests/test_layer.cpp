#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <gtest/gtest.h>

#include "dynk/layer.hpp"

void test_parallel_for_range(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  auto dataV = dynk::getViewAnonymous(dataDV, isExecutedOnDevice);
  dynk::parallel_for(
      isExecutedOnDevice, "label", dynk::RangePolicyCreator(0, 10),
      KOKKOS_LAMBDA(int const i) { dataV(i) = i; });
  dynk::setModified(dataDV, isExecutedOnDevice);

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(5), 5);
}

TEST(test_parallel_for, test_range) {
  test_parallel_for_range(true);
  test_parallel_for_range(false);
}

void test_parallel_for_range_simple(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  auto dataV = dynk::getViewAnonymous(dataDV, isExecutedOnDevice);
  dynk::parallel_for(
      isExecutedOnDevice, "label", 10,
      KOKKOS_LAMBDA(int const i) { dataV(i) = i; });
  dynk::setModified(dataDV, isExecutedOnDevice);

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(5), 5);
}

TEST(test_parallel_for, test_range_simple) {
  test_parallel_for_range_simple(true);
  test_parallel_for_range_simple(false);
}

void test_parallel_for_mdrange(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int **>;
  DualView dataDV("data", 10, 10);

  auto dataV = dynk::getViewAnonymous(dataDV, isExecutedOnDevice);
  dynk::parallel_for(
      isExecutedOnDevice, "label",
      dynk::MDRangePolicyCreator<2>({0, 0}, {10, 10}),
      KOKKOS_LAMBDA(int const i, int const j) { dataV(i, j) = i * 100 + j; });
  dynk::setModified(dataDV, isExecutedOnDevice);

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(4, 6), 406);
}

TEST(test_parallel_for, test_mdrange) {
  test_parallel_for_mdrange(true);
  test_parallel_for_mdrange(false);
}

void test_parallel_reduce_range(bool const isExecutedOnDevice) {
  int value = 0;
  dynk::parallel_reduce(
      isExecutedOnDevice, "label", dynk::RangePolicyCreator(0, 10),
      KOKKOS_LAMBDA(int const i, int &valueLocal) { valueLocal += 1; }, value);

  EXPECT_EQ(value, 10);
}

TEST(test_parallel_reduce, test_range) {
  test_parallel_reduce_range(true);
  test_parallel_reduce_range(false);
}
