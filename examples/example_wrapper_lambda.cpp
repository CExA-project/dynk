#include <Kokkos_Core.hpp>

#include "dynk/wrapper.hpp"

#include "class.hpp"

template <typename T>
TestArray<T>& TestArray<T>::operator+=(TestArray<T> const& other) {
    bool isExecutedOnDevice = true;

    dynk::wrap(
            isExecutedOnDevice,
            [&] <typename ES, typename MS> () {
            auto dataV = dynk::getView<MS>(mData);
            auto otherV = dynk::getView<MS>(other.mData);
            Kokkos::parallel_for(
                    "perform += for test array",
                    Kokkos::RangePolicy<ES>(0, size()),
                    KOKKOS_CLASS_LAMBDA (int const i) { dataV(i) += otherV(i); }
                    );
            dynk::setModified<MS>(mData);
            }
            );

    return *this;
}

#include "main.hpp"
