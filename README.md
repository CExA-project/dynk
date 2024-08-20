# Dynamic Kokkos

Dynamic Kokkos (or "dynk" for shorter) is a header-only library that proposes helper functions to run parallel block regions on either the device or on the host, the choice being done at runtime.
The signature of the familiar `parallel_for` and `parallel_reduce` functions is left similar, the list of arguments is prepended with a Boolean value, indicating if the code should run on the device or not.
Note that only the most common uses that appear in the documentation are reproduced.

## What is supported so far

- Parallel constructs
  - `parallel_for`
  - `parallel_reduce`
- Execution policies
  - `RangePolicy`
  - Implicit `RangePolicy` with only the number of elements
  - `MDRangePolicy`

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

Alternatively, you can also copy `include/dynk/dynamic_kokkos.hpp` in your project and start using it.

## Use

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
