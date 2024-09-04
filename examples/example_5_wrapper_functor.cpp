#include <Kokkos_Core.hpp>

#include "dynk/wrapper.hpp"

#include "class.hpp"

template <typename View> struct PlusEqualFunctor {
  View mDataV;
  View mOtherV;

  explicit PlusEqualFunctor(View const view1, View const view2)
      : mDataV(view1), mOtherV(view2) {}

  KOKKOS_FUNCTION void operator()(int const i) const {
    mDataV(i) += mOtherV(i);
  }
};

template <typename T>
TestArray<T> &TestArray<T>::operator+=(TestArray<T> const &other) {
  bool isExecutedOnDevice = true;

  dynk::wrap(
      isExecutedOnDevice, [&]<typename ES, typename MS>() {
        auto dataV = dynk::getView<MS>(mData);
        auto otherV = dynk::getView<MS>(other.mData);
        Kokkos::parallel_for("perform += for test array",
                             Kokkos::RangePolicy<ES>(0, size()),
                             PlusEqualFunctor(dataV, otherV));
        dynk::setModified<MS>(mData);
      });

  return *this;
}

#include "main.hpp"
