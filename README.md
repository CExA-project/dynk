# Dynamic Kokkos

Dynamic Kokkos (or "dynk" for shorter) is a header-only library that proposes helper functions to run parallel block regions on either the device or on the host, the choice being done at runtime.

## Install

### With CMake

The best way is to use CMake.

#### As a subdirectory

Get the library in your project:

```sh
git clone https://github.com/cexa-project/dynk.git path/to/dynk
```

In your main CMake file:

```cmake
add_subdirectory(path/to/dynk)

target_link_libraries(
    my-lib
    PRIVATE
        Dynk::dynk
)
```

#### As a locally available dependency

Get, then install the project:

```sh
git clone https://github.com/cexa-project/dynk.git
cd dynk
cmake -B build -DCMAKE_INSTALL_PREFIX=path/to/install -DCMAKE_BUILD_TYPE=Release # other Kokkos options here if needed
cmake --install build
```

In your main CMake file:

```cmake
find_package(Dynk REQUIRED)

target_link_libraries(
    my-lib
    PRIVATE
        Dynk::dynk
)
```

### Copy files

Alternatively, you can also copy `include/dynk` in your project and start using it.

## Use

The library offers the *wrapper* approach and the *parallel functions* approach.

### Wrapper approach

With this approach, you would wrap your data access and your parallel block construct in the `dynk::wrap` function:

```cpp
#include "dynk/wrapper.hpp"

template <typename View>
struct Functor {
    View mDataV;

    explicit Functor(View const dataV) : mDataV(dataV) {}

    KOKKOS_FUNCTION void operator()(int const i) const {
        mDataV(i) = i;
    }
};

void soSomething() {
    Kokkos::DualView<int *> dataDV("data", 10);
    bool isExecutedOnDevice = true;  // can be changed at will

    dynk::wrap(
        isExecutedOnDevice,
        [&]<typename ExecutionSpace, typename MemorySpace>() {
            // notice the two template arguments above

            // acquire data
            auto dataV = dynk::getView<MemorySpace>(dataDV);

            // usual parallel for
            Kokkos::parallel_for(
                "label",
                Kokkos::RangePolicy<ExecutionSpace>(0, 10),
                Functor(dataV)
                );

            // set data as modified (if necessary)
            dynk::setModified<MemorySpace>(dataDV);
            }
        );
}
```

Note the use of a templated lambda, which is only available with C++20.
Build of the wrapper can be disabled in the CMake options.

Note that the Kokkos kernel is supplied as a functor.
Due to Cuda limitations, as of Cuda 12.5, it is not possible to define an extended lambda (i.e. a lambda with attributes `__host__ __device__`) within a generic lambda (i.e. a templated lambda).

`dynk::wrap` would create two versions of the passed templated lambda: one for the device (with default execution space and default execution space's default memory space), and one for the host (with default host executions space and default host execution space's default memory space).
Depending on the passed Boolean `isExecutedOnDevice`, the former or the later would be executed.
Note that the resulting binary would be about twice the size, but this is the price to pay for dynamic execution.
For more details, please check the documented source code.

The advantage of this approach is its small impact at build time (lightweight library), and the fact that it lets the user do Kokkos code using regular Kokkos functions.
When the dynamic approach is not desired anymore in the user's code, it would be pretty easy to get rid of the library and obtain a plain Kokkos code.
On the other hand, the disadvantage is the C++20 requirement and the obligation, at least for now, to pass the kernel as a functor (with all the boilerplate code it requires).

### Parallel functions approach

The signature of the familiar `parallel_for` and `parallel_reduce` functions is left similar, the list of arguments is prepended with a Boolean value, indicating if the code should run on the device or not.
Note that only the most common uses that appear in the documentation are reproduced.

#### What is supported so far

- Parallel constructs
  - `parallel_for`
  - `parallel_reduce`
- Execution policies
  - `RangePolicy`
  - Implicit `RangePolicy` with only the number of elements
  - `MDRangePolicy`

In your C++ files, you would replace your existing `parallel_for` or `parallel_reduce` by their equivalent from dynk:

```cpp
#include "dynk/dynamic_kokkos.hpp"

...

bool isExecutableOnDevice = true;
dynk::parallel_for(
  isExecutedOnDevice, "my label", Kokkos::RangePolicy(0, 10),
  KOKKOS_LAMBDA(int const i) {
    Kokkos::printf("hello from %i\n", i);
  });
```

All you have to do is replace the `Kokkos` namespace of the parallel construct by `dynk`, and add the boolean as its first argument.
