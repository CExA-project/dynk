#include <string>

#include <Kokkos_Core.hpp>

#include "class.hpp"

int main(int argc, char *argv[]) {
  int size = 10;
  if (argc > 1) {
    size = std::stoi(argv[1]);
  }

  Kokkos::ScopeGuard kokkos(argc, argv);

  TestArray<float> array1(size);
  TestArray<float> array2(size);

  array1.fill();
  array2.fill();

  array1 += array2;

  array1.synchronize();
  array1.show();
}
