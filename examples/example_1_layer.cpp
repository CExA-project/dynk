#include <Kokkos_Core.hpp>

#include "dynk/layer.hpp"

#include "class.hpp"

template <typename T>
TestArray<T> &TestArray<T>::operator+=(TestArray<T> const &other) {
  bool isExecutedOnDevice = true;

  auto dataV = dynk::getView(mData, isExecutedOnDevice);
  auto otherV = dynk::getView(other.mData, isExecutedOnDevice);

  dynk::parallel_for(
      isExecutedOnDevice, "perform += for test array", size(),
      KOKKOS_CLASS_LAMBDA(int const i) { dataV(i) += otherV(i); });

  dynk::setModified(mData, isExecutedOnDevice);

  return *this;
}

#include "main.hpp"
