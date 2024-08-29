#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <gtest/gtest.h>

#include "dynk/wrapper.hpp"

template <typename View>
struct ParallelForRangeFunctor {
    View mDataV;

    explicit ParallelForRangeFunctor(View const dataV) : mDataV(dataV) {}

    KOKKOS_FUNCTION void operator()(int const i) const {
        mDataV(i) = i;
    }
};

#ifdef ENABLE_CXX20_FEATURES

void test_parallel_for_range_functor(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  dynk::wrap(
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

TEST(test_parallel_for, test_range_functor) {
  test_parallel_for_range_functor(true);
  test_parallel_for_range_functor(false);
}

#endif // ifdef ENABLE_CXX20_FEATURES

template <typename ExecutionSpace, typename MemorySpace, typename DualView>
void doParallelForFreeFunction(DualView &dataDV) {
    auto dataV = dynk::getView<MemorySpace>(dataDV);
    Kokkos::parallel_for(
        "label", Kokkos::RangePolicy<ExecutionSpace>(0, 10),
        KOKKOS_LAMBDA (int const i) {
        dataV(i) = i;
        }
        );
    dynk::setModified<MemorySpace>(dataDV);
}

#ifdef ENABLE_CXX20_FEATURES

void test_parallel_for_range_free_function(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  dynk::wrap(
      isExecutedOnDevice, [&]<typename ExecutionSpace, typename MemorySpace>() {
      doParallelForFreeFunction<ExecutionSpace, MemorySpace>(dataDV);
      });

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(5), 5);
}

TEST(test_parallel_for, test_range_free_function) {
  test_parallel_for_range_free_function(true);
  test_parallel_for_range_free_function(false);
}

#endif // ifdef ENABLE_CXX20_FEATURES

void test_parallel_for_range_free_function_twice(bool const isExecutedOnDevice) {
  using DualView = Kokkos::DualView<int *>;
  DualView dataDV("data", 10);

  dynk::wrap(
      isExecutedOnDevice,
      [&] () {
          doParallelForFreeFunction<Kokkos::DefaultExecutionSpace, typename Kokkos::DefaultExecutionSpace::memory_space>(dataDV);
      },
      [&] () {
          doParallelForFreeFunction<Kokkos::DefaultHostExecutionSpace, typename Kokkos::DefaultHostExecutionSpace::memory_space>(dataDV);
      }
      );

  dataDV.template sync<typename DualView::host_mirror_space>();
  EXPECT_EQ(dataDV.h_view(5), 5);
}

TEST(test_parallel_for, test_range_free_function_twice) {
  test_parallel_for_range_free_function_twice(true);
  test_parallel_for_range_free_function_twice(false);
}

struct ParallelReduceRangeFunctor {
    KOKKOS_FUNCTION void operator()(int const, int &valueLocal) const {
        valueLocal++;
    }
};

#ifdef ENABLE_CXX20_FEATURES

void test_parallel_reduce_range_functor(bool const isExecutedOnDevice) {
  int value = 0;
  dynk::wrap(
      isExecutedOnDevice, [&]<typename ExecutionSpace, typename MemorySpace>() {
        Kokkos::parallel_reduce(
            "label", Kokkos::RangePolicy<ExecutionSpace>(0, 10),
            ParallelReduceRangeFunctor(),
            value);
      });

  EXPECT_EQ(value, 10);
}

TEST(test_parallel_reduce, test_range_functor) {
  test_parallel_reduce_range_functor(true);
  test_parallel_reduce_range_functor(false);
}

#endif // ifdef ENABLE_CXX20_FEATURES

template <typename ExecutionSpace, typename MemorySpace>
void doParallelReduceFreeFunction(int &value) {
    Kokkos::parallel_reduce(
        "label", Kokkos::RangePolicy<ExecutionSpace>(0, 10),
        KOKKOS_LAMBDA (int const, int &valueLocal) {
            valueLocal++;
        },
        value);
}

#ifdef ENABLE_CXX20_FEATURES

void test_parallel_reduce_range_free_function(bool const isExecutedOnDevice) {
  int value = 0;
  dynk::wrap(
      isExecutedOnDevice, [&]<typename ExecutionSpace, typename MemorySpace>() {
      doParallelReduceFreeFunction<ExecutionSpace, MemorySpace>(value);
      });

  EXPECT_EQ(value, 10);
}

TEST(test_parallel_reduce, test_range_free_function) {
  test_parallel_reduce_range_free_function(true);
  test_parallel_reduce_range_free_function(false);
}

#endif // ifdef ENABLE_CXX20_FEATURES

void test_parallel_reduce_range_free_function_twice(bool const isExecutedOnDevice) {
  int value = 0;
  dynk::wrap(
      isExecutedOnDevice,
      [&] () {
          doParallelReduceFreeFunction<Kokkos::DefaultExecutionSpace, typename Kokkos::DefaultExecutionSpace::memory_space>(value);
      },
      [&] () {
          doParallelReduceFreeFunction<Kokkos::DefaultHostExecutionSpace, typename Kokkos::DefaultHostExecutionSpace::memory_space>(value);
      }
      );

  EXPECT_EQ(value, 10);
}

TEST(test_parallel_reduce, test_range_free_function_twice) {
  test_parallel_reduce_range_free_function_twice(true);
  test_parallel_reduce_range_free_function_twice(false);
}
