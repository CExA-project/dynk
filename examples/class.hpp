#ifndef __TEST_ARRAY_HPP__
#define __TEST_ARRAY_HPP__

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

template <typename T>
class TestArray {
    Kokkos::DualView<T*> mData;

    public:

    TestArray() = default;

    explicit TestArray(Kokkos::DualView<T*> const data) : mData(data) {}

    explicit TestArray(int const n) : mData("test array", n) {}

    void fill(T const value = 1) {
        Kokkos::deep_copy(mData.h_view, value);
        mData.template modify<typename Kokkos::DefaultHostExecutionSpace>();
        synchronize();
    }

    TestArray<T>& operator+=(TestArray<T> const& other);

    Kokkos::DualView<T*>& data() {
        return mData;
    }

    Kokkos::DualView<T*> const& data() const {
        return mData;
    }

    int size() const {
        return mData.d_view.size();
    }

    void synchronize() {
        mData.template sync<typename Kokkos::DefaultHostExecutionSpace>();
        mData.template sync<typename Kokkos::DefaultExecutionSpace>();
    }

    void show() const {
        Kokkos::printf("Showing '%s'\n", mData.h_view.label().c_str());
        Kokkos::parallel_for(
                "show test array",
                Kokkos::RangePolicy<Kokkos::Serial>(0, size()),
                KOKKOS_CLASS_LAMBDA (int const i) {
                Kokkos::printf("%f\n", mData.h_view(i));
                }
                );
    }
};

#endif // #ifndef __TEST_ARRAY_HPP__
