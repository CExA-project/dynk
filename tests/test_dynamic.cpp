#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include <gtest/gtest.h>

#include "dynk/dynamic.hpp"

namespace {

void test() {
    auto func1 = [=] <typename T1> () {
        Kokkos::parallel_for(
                "kernel",
                Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace, dynk::use_annotated_operator>(0, 10),
                dynk::KokkosLambdaAdapter([=] (int const i) {
                    T1 a = i;
                    Kokkos::printf("from %i, %i\n", i, a);
                    })
                );
    };
    func1.template operator()<int>();
};

// void test_parallel_for_range(bool const isExecutedOnDevice) {
//   using DualView = Kokkos::DualView<int *>;
//   DualView dataDV("data", 10);
//
//   dynk::dynamicLaunch(
//       isExecutedOnDevice, [&]<typename ExecutionSpace, typename MemorySpace>() {
//         auto dataV = dynk::getView<MemorySpace>(dataDV);
//         auto f = [=] (int const i) { dataV(i) = i; };
//         Kokkos::parallel_for(
//             "label", Kokkos::RangePolicy<ExecutionSpace>(0, 10),
//             dynk::KokkosLambdaAdapter(f)
//             );
//         dynk::setModified<MemorySpace>(dataDV);
//       });
//
//   dataDV.template sync<typename DualView::host_mirror_space>();
//   EXPECT_EQ(dataDV.h_view(5), 5);
// }

} // namespace

// TEST(test_parallel_for, test_range) {
//   test_parallel_for_range(true);
//   test_parallel_for_range(false);
// }
//
// namespace {
//
// void test_parallel_reduce_range(bool const isExecutedOnDevice) {
//   int value = 0;
//   dynk::dynamicLaunch(
//       isExecutedOnDevice, [&]<typename ExecutionSpace, typename MemorySpace>() {
//         Kokkos::parallel_reduce(
//             "label", Kokkos::RangePolicy<ExecutionSpace>(0, 10),
//             dynk::KokkosLambdaAdapter([=] (int const i, int &valueLocal) { valueLocal++; }),
//             value);
//       });
//
//   EXPECT_EQ(value, 10);
// }
//
// } // namespace
//
// TEST(test_parallel_reduce, test_range) {
//   test_parallel_reduce_range(true);
//   test_parallel_reduce_range(false);
// }
