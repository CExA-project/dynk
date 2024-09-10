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

### With FetchContent

In your main CMake file:

```cmake
include(FetchContent)
FetchContent_Declare(
    dynk
    URL https://github.com/CExA-project/dynk/archive/refs/tags/0.1.0.tar.gz
)
FetchContent_MakeAvailable(dynk)

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

## Tests

You can build tests with the CMake option `DYNK_ENABLE_TESTS`, and run them with `ctest`.

If you don't have a GPU available when compiling, you have to disable the CMake option `DYNK_ENABLE_GTEST_DISCOVER_TESTS`.

## Examples

You can build examples with the CMake option `DYNK_ENABLE_EXAMPLES`.
They should be run individually.

## Use

The library offers the *layer* approach and the *wrapper* approach.
Here is the breakdown of their support (also for their sub-approaches):

| Backend | Compiler | Minimum version | Layer              | Wrapper `if`       | Wrapper 2 functions | Wrapper function | Wrapper functor | Wrapper lambda |
|---------|----------|-----------------|--------------------|--------------------|---------------------|------------------|-----------------|----------------|
| CUDA    | NVCC     |                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :warning:        | :warning:       | :x:            |
| CUDA    | Clang    |                 | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :warning:        | :warning:       | :warning:      |
| HIP     | ROCm     | 5.5.1           | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :warning:        | :warning:       | :warning:      |
| SYCL    | Intel    | 2024.0.2        | :heavy_check_mark: | :heavy_check_mark: | :heavy_check_mark:  | :warning:        | :warning:       | :warning:      |

Where :warning: means C++20 required, and :x: cannot be built.

Note that that in any approach, the resulting binary would always contain the two compiled kernels, and hence may become quite heavy, but this is the price to pay for dynamic execution.
For more details, please check the documented source code.

### Wrapper approach

#### Wrapper lambda approach

With this approach, you would wrap your data access and your parallel block construct in the `dynk::wrap` function:

```cpp
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include "dynk/wrapper.hpp"

void doSomething() {
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
                KOKKOS_LAMBDA (int const i) {
                    dataV(i) = i;
                }
                );

            // set data as modified (if necessary)
            dynk::setModified<MemorySpace>(dataDV);
            }
        );
}
```

With this approach, we have two lambdas: one, templated for the execution space and memory space, that contains the parallel block and the data management, and the other which is a classic Kokkos lambda.
Note that templated lambdas are only available with C++20.
By default, the CMake configuration sets the language standard accordingly, this can be disabled with the CMake option `DYNK_ENABLE_CXX20_FEATURES`.

`dynk::wrap` would create two versions of the passed templated lambda: one for the device (with default execution space and default execution space's default memory space), and one for the host (with default host executions space and default host execution space's default memory space).
Depending on the passed Boolean `isExecutedOnDevice`, the former or the later would be executed.

The advantage of this approach is its small impact at build time (lightweight library), and the fact that it lets the user do Kokkos code using regular Kokkos functions.
When the dynamic approach is not desired anymore in the user's code, it would be pretty easy to get rid of the library and obtain a plain Kokkos code.
On the other hand, the disadvantage is the C++20 requirement and its lack of support with NVCC.

This approach works with Clang, ROCm and the Intel LLVM compiler.
However, this does not work with NVCC, as of Cuda 12.5: is not possible to define an extended lambda (i.e. a lambda with attributes `__host__ __device__`) within a generic lambda (i.e. a templated lambda) with this compiler.
By default, the CMake configuration allows to build test cases using this feature, this can be disabled with the CMake option `DYNK_ENABLE_EXTENDED_LAMBDA_IN_GENERIC_LAMBDA`.

For that, we propose a derived approach.

#### Wrapper functor approach

```cpp
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include "dynk/wrapper.hpp"

template <typename View>
struct Functor {
    View mDataV;

    explicit Functor(View const dataV) : mDataV(dataV) {}

    KOKKOS_FUNCTION void operator()(int const i) const {
        mDataV(i) = i;
    }
};

void doSomething() {
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

Note that the Kokkos kernel is supplied as a functor now.
This works with NVCC.

As using a functor is cumbersome another approach is possible.

#### Wrapper function approach

It is possible to define the parallel block and the data management in a templated free function, where the Kokkos kernel is supplied as a regular lambda:

```cpp
#include "dynk/wrapper.hpp"

template <typename ExecutionSpace, typename MemorySpace, typename DualView>
void doSomethingFreeFunction(DualView &dataDV) {
    // notice the two space template arguments above

    // acquire data
    auto dataV = dynk::getView<MemorySpace>(dataDV);

    // usual parallel for
    Kokkos::parallel_for(
        "label",
        Kokkos::RangePolicy<ExecutionSpace>(0, 10),
        KOKKOS_LAMBDA(int const i) {
            mDataV(i) = i;
        }
        );

    // set data as modified (if necessary)
    dynk::setModified<MemorySpace>(dataDV);
}

void doSomething() {
    Kokkos::DualView<int *> dataDV("data", 10);
    bool isExecutedOnDevice = true;  // can be changed at will

    dynk::wrap(
        isExecutedOnDevice,
        [&]<typename ExecutionSpace, typename MemorySpace>() {
        // notice the two template arguments above
        doSomethingFreeFunction(dataDV);
        }
        );
}
```

This still requires to scatter the code, which makes it less readable.

#### Wrapper 2 functions approach

If enabling C++20 is not possible, `dynk::wrap` is overloaded so that it accepts two functions: one for the device execution, one for the host execution.
There is no need to resort to a templated lambda anymore.

```cpp
void doSomething() {
    Kokkos::DualView<int *> dataDV("data", 10);
    bool isExecutedOnDevice = true;  // can be changed at will

    dynk::wrap(
        isExecutedOnDevice,
        [&] () {
        doSomethingFreeFunction<Kokkos::DefaultExecutionSpace, typename Kokkos::DefaultExecutionSpace::memory_space>(dataDV);
        },
        [&] () {
        doSomethingFreeFunction<Kokkos::DefaultHostExecutionSpace, typename Kokkos::DefaultHostExecutionSpace::memory_space>(dataDV);
        }
        );
}
```

At this rate however, the helper is just a fancy `if`.

#### Wrapper `if` approach

Just ditching the `dynk::wrap` and the extra lambdas:

```cpp
void doSomething() {
    Kokkos::DualView<int *> dataDV("data", 10);
    bool isExecutedOnDevice = true;  // can be changed at will

    if (isExecutedOnDevice) {
        doSomethingFreeFunction<Kokkos::DefaultExecutionSpace, typename Kokkos::DefaultExecutionSpace::memory_space>(dataDV);
    else {
        doSomethingFreeFunction<Kokkos::DefaultHostExecutionSpace, typename Kokkos::DefaultHostExecutionSpace::memory_space>(dataDV);
    }
}
```

### Layer approach

The layer approach aims to propose alternate versions of the `parallel_for` and `parallel_reduce` functions, with a similar signature.
The list of arguments is prepended with a Boolean value, indicating if the code should run on the device or not.
Note that this approach requires to reimplement Kokkos feature and is *difficult to maintain* for the long run.
Only the most common uses that appear in the documentation are reproduced.

In your C++ files, you would replace your existing `parallel_for` or `parallel_reduce` by their equivalent from dynk:

```cpp
#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>
#include "dynk/dynamic_kokkos.hpp"

void doSomething() {
    Kokkos::DualView<int *> dataDV("data", 10);
    bool isExecutedOnDevice = true;  // can be changed at will

    // acquire data
    auto dataV = dynk::getViewAnonymous(dataDV, isExecutedOnDevice);

    // modified parallel for
    dynk::parallel_for(
        isExecutedOnDevice, "label", Kokkos::RangePolicy(0, 10),
        KOKKOS_LAMBDA (int const i) {
        dataV(i) = i;
        }
        );

    // set data as modified (if necessary)
    dynk::setModified(dataDV, isExecutedOnDevice);
}
```

This approach requires to replace the `Kokkos` namespace of the parallel construct by `dynk`, and to add the Boolean value as its first argument.
Accessing the right View from a DualView is made with the `dynk::getViewAnonymous` function, that returns a Kokkos View in an `Kokkos::AnonymousSpace`.
This type of View has no known memory space at compile time, it is defined by affectation dynamically during runtime.
Please note that this feature, though available in the public Kokkos API, is *not documented*.

#### What is supported so far

- Parallel constructs
  - `parallel_for`
  - `parallel_reduce`
- Execution policies
  - `RangePolicy`
  - Implicit `RangePolicy` with only the number of elements
  - `MDRangePolicy`
