#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <gtest/gtest.h>

#include "dynk/dynamic.hpp"

template <typename View>
struct ParallelForRangeFunctor {
    View mDataV;

    explicit ParallelForRangeFunctor(View const dataV) : mDataV(dataV) {}

    KOKKOS_FUNCTION void operator()(int const i) const {
        mDataV(i) = i;
    }
};

void test_parallel_for_range(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  dynk::dynamicLaunch(
      isExecutedOnDevice, [&]<typename ExecutionSpace, typename MemorySpace>() {
        auto dataV = dynk::getView<MemorySpace>(dataDV);
        Kokkos::parallel_for(
            "label", Kokkos::RangePolicy<ExecutionSpace>(0, 10),
            ParallelForRangeFunctor(dataV)
            );
        dynk::setModified<MemorySpace>(dataDV);
      });

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(5), 5);
}

TEST(test_parallel_for, test_range) {
  test_parallel_for_range(true);
  test_parallel_for_range(false);
}

struct ParallelReduceRangeFunctor {
    KOKKOS_FUNCTION void operator()(int const i, int &valueLocal) const {
        valueLocal++;
    }
};

void test_parallel_reduce_range(bool const isExecutedOnDevice) {
  int value = 0;
  dynk::dynamicLaunch(
      isExecutedOnDevice, [&]<typename ExecutionSpace, typename MemorySpace>() {
        Kokkos::parallel_reduce(
            "label", Kokkos::RangePolicy<ExecutionSpace>(0, 10),
            ParallelReduceRangeFunctor(),
            value);
      });

  EXPECT_EQ(value, 10);
}

TEST(test_parallel_reduce, test_range) {
  test_parallel_reduce_range(true);
  test_parallel_reduce_range(false);
}
