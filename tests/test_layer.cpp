#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <gtest/gtest.h>

#include "dynk/layer.hpp"

TEST(test_get_view, test_default) {
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  auto dataD = dynk::getView(dataDV, true);
  EXPECT_EQ(dataD.data(), dataDV.template view<DeviceSpace>().data());

  auto dataH = dynk::getView(dataDV, false);
  EXPECT_EQ(dataH.data(), dataDV.template view<Kokkos::HostSpace>().data());
}

TEST(test_get_view, test_non_default) {
  using DeviceSpace = Kokkos::DefaultExecutionSpace::memory_space;
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  auto dataH1 = dynk::getView<Kokkos::HostSpace>(dataDV, true);
  EXPECT_EQ(dataH1.data(), dataDV.template view<Kokkos::HostSpace>().data());

  auto dataH2 =
      dynk::getView<Kokkos::HostSpace, Kokkos::HostSpace>(dataDV, false);
  EXPECT_EQ(dataH2.data(), dataDV.template view<Kokkos::HostSpace>().data());

  auto dataD1 = dynk::getView<DeviceSpace>(dataDV, true);
  EXPECT_EQ(dataD1.data(), dataDV.template view<DeviceSpace>().data());

  auto dataD2 = dynk::getView<DeviceSpace, DeviceSpace>(dataDV, false);
  EXPECT_EQ(dataD2.data(), dataDV.template view<DeviceSpace>().data());
}

void test_parallel_for_range(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  auto dataV = dynk::getView(dataDV, isExecutedOnDevice);
  dynk::parallel_for(
      isExecutedOnDevice, "label", dynk::RangePolicy(0, 10),
      KOKKOS_LAMBDA(int const i) { dataV(i) = i; });
  dynk::setModified(dataDV, isExecutedOnDevice);

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(5), 5);
}

TEST(test_parallel_for, test_range) {
  test_parallel_for_range(true);
  test_parallel_for_range(false);
}

void test_parallel_for_range_unsync(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  // pre-alter data
  auto dataAlteration = dataDV.template view<Kokkos::HostSpace>();
  Kokkos::deep_copy(dataAlteration, 10);
  dataDV.template modify<Kokkos::HostSpace>();

  auto dataV = dynk::getSyncedView(dataDV, isExecutedOnDevice);
  dynk::parallel_for(
      isExecutedOnDevice, "label", dynk::RangePolicy(0, 10),
      KOKKOS_LAMBDA(int const i) { dataV(i) += i; });
  dynk::setModified(dataDV, isExecutedOnDevice);

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(5), 15);
}

TEST(test_parallel_for, test_range_unsync) {
  test_parallel_for_range_unsync(true);
  test_parallel_for_range_unsync(false);
}

void test_parallel_for_range_simple(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  auto dataV = dynk::getView(dataDV, isExecutedOnDevice);
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

  auto dataV = dynk::getView(dataDV, isExecutedOnDevice);
  dynk::parallel_for(
      isExecutedOnDevice, "label",
      dynk::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {10, 10}),
      KOKKOS_LAMBDA(int const i, int const j) { dataV(i, j) = i * 100 + j; });
  dynk::setModified(dataDV, isExecutedOnDevice);

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(4, 6), 406);
}

TEST(test_parallel_for, test_mdrange) {
  test_parallel_for_mdrange(true);
  test_parallel_for_mdrange(false);
}

void test_parallel_for_mdrange_tile(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int **>;
  DualView dataDV("data", 10, 10);

  auto dataV = dynk::getView(dataDV, isExecutedOnDevice);
  dynk::parallel_for(
      isExecutedOnDevice, "label",
      dynk::MDRangePolicy<Kokkos::Rank<2>>({0, 0}, {10, 10}, {2, 2}),
      KOKKOS_LAMBDA(int const i, int const j) { dataV(i, j) = i * 100 + j; });
  dynk::setModified(dataDV, isExecutedOnDevice);

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(4, 6), 406);
}

TEST(test_parallel_for, test_mdrange_tile) {
  test_parallel_for_mdrange_tile(true);
  test_parallel_for_mdrange_tile(false);
}

void test_parallel_reduce_range(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  auto dataV = dynk::getView(dataDV, isExecutedOnDevice);
  int value = 0;
  dynk::parallel_reduce(
      isExecutedOnDevice, "label", dynk::RangePolicy(0, 10),
      KOKKOS_LAMBDA(int const i, int &valueLocal) {
        dataV(i) = i;
        valueLocal += 1;
      },
      value);
  dynk::setModified(dataDV, isExecutedOnDevice);

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(5), 5);
  EXPECT_EQ(value, 10);
}

TEST(test_parallel_reduce, test_range) {
  test_parallel_reduce_range(true);
  test_parallel_reduce_range(false);
}
