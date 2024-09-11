#ifndef __DYNK_WRAPPER_HPP__
#define __DYNK_WRAPPER_HPP__

/**
 * Wrapper approach.
 *
 * This approach proposes to encapsulate the parallel block and memory
 * management in templated functions or templated lambdas, and to dynamically
 * execute whichever is requested at runtime.
 */

#include <Kokkos_Core.hpp>

#include "dynk/dual_view.hpp"

namespace dynk {

/**
 * Allow to dynamically launch a parallel block region on device or on host by
 * giving a templated launcher.
 *
 * @tparam ParallelLauncher Type of the functor.
 * @tparam DeviceExecutionSpace Kokkos execution space for device execution,
 * defaults to Kokkos default execution space.
 * @tparam DeviceMemorySpace Kokkos memory space for device memory, defaults to
 * Kokkos default execution space's default memory space.
 * @tparam HostExecutionSpace Kokkos execution space for host execution,
 * defaults to Kokkos default host execution space.
 * @tparam HostMemorySpace Kokkos memory space for host memory, defaults to
 * Kokkos default host execution space's default memory space.
 * @param isExecutedOnDevice If `true`, the parallel block region is launched
 * for execution on the device, otherwise on the host.
 * @param parallelLauncher Functor to launch that contains a parallel block
 * region.
 */
template <
    typename ParallelLauncher,
    typename DeviceExecutionSpace = Kokkos::DefaultExecutionSpace,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostExecutionSpace = Kokkos::DefaultHostExecutionSpace,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
void wrap(bool const isExecutedOnDevice,
          ParallelLauncher const &parallelLauncher) {
  // NOTE: Beware the ugly syntax below! We're calling a templated functor, for
  // which the parenthesis operator is actually templated, hence the need to
  // exhibit the call to `operator()`. As the operator is a method of the
  // object, the `template` keyword is needed to understand the `<>` syntax.
  if (isExecutedOnDevice) {
    // launch for device execution
    parallelLauncher
        .template operator()<DeviceExecutionSpace, DeviceMemorySpace>();
  } else {
    // launch for host execution
    parallelLauncher.template operator()<HostExecutionSpace, HostMemorySpace>();
  }
}

/**
 * Allow to dynamically launch a parallel block region on device or on host by
 * giving two specialized launchers.
 *
 * @tparam ParallelLauncherDevice Type of the device launcher functor.
 * @tparam ParallelLauncherHost Type of the host launcher functor.
 * @param isExecutedOnDevice If `true`, the parallel block region is launched
 * for execution on the device, otherwise on the host.
 * @param parallelLauncherDevice Device launcher functor that contains a
 * parallel block region.
 * @param parallelLauncherHost Host launcher functor that contains a parallel
 * block region.
 */
template <typename ParallelLauncherDevice, typename ParallelLauncherHost>
void wrap(bool const isExecutedOnDevice,
          ParallelLauncherDevice const &parallelLauncherDevice,
          ParallelLauncherHost const &parallelLauncherHost) {
  if (isExecutedOnDevice) {
    // launch for device execution
    parallelLauncherDevice();
  } else {
    // launch for host execution
    parallelLauncherHost();
  }
}

} // namespace dynk

#endif // ifndef __DYNK_WRAPPER_HPP__
