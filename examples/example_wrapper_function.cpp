#include <Kokkos_Core.hpp>

#include "dynk/wrapper.hpp"

#include "class.hpp"

template <typename ES, typename MS, typename TestArray>
void plusEqual(TestArray &data, TestArray const& other) {
    auto dataV = dynk::getView<MS>(data.data());
    auto otherV = dynk::getView<MS>(other.data());
    Kokkos::parallel_for(
            "perform += for test array",
            Kokkos::RangePolicy<ES>(0, data.size()),
            KOKKOS_LAMBDA (int const i) { dataV(i) += otherV(i); }
            );
    dynk::setModified<MS>(data.data());
}

template <typename T>
TestArray<T>& TestArray<T>::operator+=(TestArray<T> const& other) {
    bool isExecutedOnDevice = true;

    dynk::wrap(
            isExecutedOnDevice,
            [&] <typename ES, typename MS> () {
            plusEqual<ES, MS>(*this, other);
            }
            );

    return *this;
}

#include "main.hpp"
