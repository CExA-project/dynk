#ifndef __DYNAMIC_HPP__
#define __DYNAMIC_HPP__

#include <Kokkos_Core.hpp>

namespace dynk {

/**
 * Allow to dynamically launch a parallel block region on device or on host.
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
 * @param parallelLauncher Functor to launch that contains a parallel block region.
 */
template <typename ParallelLauncher,
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
    parallelLauncher.template operator()<
        DeviceExecutionSpace, DeviceMemorySpace>();
  } else {
    // launch for host execution
    parallelLauncher.template
    operator()<HostExecutionSpace, HostMemorySpace>();
  }
}

/**
 * Get a View of a DualView for the requested memory space.
 * The DualView memory spaces should match with the ones expected. This
 * function is just a pass-through to manipulate the DualView with a simpler
 * syntax.
 *
 * @tparam MemorySpace Memory space requested.
 * @tparam DualView Type of the DualView.
 * @param dualView DualView to take a view from.
 * @return View in the requested memory space.
 */
template <typename MemorySpace, typename DualView>
auto getView(DualView &dualView) {
  return dualView.template view<MemorySpace>();
}

/**
 * Set that a DualView was modified in the requested memory space.
 * The DualView memory spaces should match with the ones expected. This
 * function is just a pass-through to manipulate the DualView with a simpler
 * syntax.
 *
 * @tparam MemorySpace Memory space requested.
 * @tparam DualView Type of the DualView.
 * @param dualView DualView to set.
 */
template <typename MemorySpace, typename DualView>
void setModified(DualView &dualView) {
  dualView.template modify<MemorySpace>();
}

} // namespace dynk

#endif // ifndef __DYNAMIC_HPP__
