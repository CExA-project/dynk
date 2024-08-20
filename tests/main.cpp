#include <Kokkos_Core.hpp>
#include <gtest/gtest.h>

int main(int argc, char **argv) {
  Kokkos::ScopeGuard kokkos(argc, argv);

  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
