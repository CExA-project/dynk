#ifndef __DYNAMIC_KOKKOS_HPP__
#define __DYNAMIC_KOKKOS_HPP__

/**
 * This code proposes helper functions to run parallel block regions on either
 * the device or on the host, the choice being done at runtime. The signature
 * of the familiar `parallel_for` and `parallel_reduce` functions is left
 * similar, the list of paramuments is prepended with a Boolean value,
 * indicating if the code should run on the device (i.e. on
 * `Kokkos::DefaultExecutionSpace` by default) or not (i.e. on
 * `Kokkos::DefaultHostExecutionSpace` by default). Note that only the most
 * common uses that appear in the documentation are reproduced.
 *
 * Though this could be pretty useful, I personnaly think this is not a wise
 * idea. I guess there is a reason why Kokkos didn't introduce such a dynamic
 * execution space selection in the first place.
 */

#include <string>

#include <Kokkos_Core.hpp>

#include "dynk/dual_view.hpp"

namespace dynk {

namespace impl {

/**
 * Recreate a range execution policy for the requested execution space.
 *
 * @tparam ExecutionSpace Requested execution space.
 * @param policy Execution policy to recreate.
 * @return Recreated execution policy.
 */
template <typename ExecutionSpace>
auto recreateExecutionPolicy(Kokkos::RangePolicy<> const &policy) {
  return Kokkos::RangePolicy<ExecutionSpace>(policy.begin(), policy.end());
}

/**
 * Recreate a simple range execution policy for the requested execution
 * space.
 *
 * @tparam ExecutionSpace Requested execution space.
 * @param end Number of elements to work on.
 * @return Recreated execution policy.
 */
template <typename ExecutionSpace> auto recreateExecutionPolicy(int const end) {
  return Kokkos::RangePolicy<ExecutionSpace>(0, end);
}

/**
 * Recreate a multi-dimensional range execution policy for the requested
 * execution space.
 *
 * @tparam ExecutionSpace Requested execution space.
 * @param policy Execution policy to recreate.
 * @return Recreated execution policy.
 */
template <typename ExecutionSpace, typename Rank>
auto recreateExecutionPolicy(Kokkos::MDRangePolicy<Rank> const &policy) {
  return Kokkos::MDRangePolicy<ExecutionSpace, Rank>(
      policy.m_lower, policy.m_upper, policy.m_tile);
}

} // namespace impl

/**
 * Parallel for that can be executed dynamically on device or on host
 * depending on a Boolean parameter.
 *
 * @tparam ExecutionPolicy Type of the execution policy.
 * @tparam Kernel Type of the kernel.
 * @tparam DeviceExecutionSpace Kokkos execution space for device execution,
 * defaults to Kokkos default execution space.
 * @tparam HostExecutionSpace Kokkos execution space for host execution,
 * defaults to Kokkos default host execution space.
 * @param isExecutedOnDevice If `true`, the parallel for is executed on the
 * device, otherwise on the host.
 * @param label Label of the kernel.
 * @param dummyExecutionPolicy Execution policy that will be adapted for device
 * and host execution.
 * @param kernel Kernel to execute withing a Kokkos parallel for region.
 */
template <
    typename ExecutionPolicy, typename Kernel,
    typename DeviceExecutionSpace = Kokkos::DefaultExecutionSpace,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostExecutionSpace = Kokkos::DefaultHostExecutionSpace,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
void parallel_for(bool const isExecutedOnDevice, std::string const &label,
                  ExecutionPolicy const &dummyExecutionPolicy,
                  Kernel const &kernel) {
  if (isExecutedOnDevice) {
    // device execution
    Kokkos::parallel_for(label,
                         impl::recreateExecutionPolicy<DeviceExecutionSpace>(
                             dummyExecutionPolicy),
                         kernel);
  } else {
    // host execution
    Kokkos::parallel_for(
        label,
        impl::recreateExecutionPolicy<HostExecutionSpace>(dummyExecutionPolicy),
        kernel);
  }
}

/**
 * Parallel reduce that can be executed dynamically on device or on host
 * depending on a Boolean parameter.
 *
 * @tparam ExecutionPolicy Type of the execution policy.
 * @tparam Kernel Type of the kernel.
 * @tparam Reducer Type of the reducers.
 * @tparam DeviceExecutionSpace Kokkos execution space for device execution,
 * defaults to Kokkos default execution space.
 * @tparam HostExecutionSpace Kokkos execution space for host execution,
 * defaults to Kokkos default host execution space.
 * @param isExecutedOnDevice If `true`, the parallel for is executed on the
 * device, otherwise on the host.
 * @param label Label of the kernel.
 * @param dummyExecutionPolicy Execution policy that will be adapted for device
 * and host execution.
 * @param kernel Kernel to execute withing a Kokkos parallel for region.
 */
template <
    typename ExecutionPolicy, typename Kernel, typename... Reducer,
    typename DeviceExecutionSpace = Kokkos::DefaultExecutionSpace,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostExecutionSpace = Kokkos::DefaultHostExecutionSpace,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
void parallel_reduce(bool const isExecutedOnDevice, std::string const &label,
                     ExecutionPolicy const &dummyExecutionPolicy,
                     Kernel const &kernel, Reducer &... reducers) {
  if (isExecutedOnDevice) {
    // device execution
    Kokkos::parallel_reduce(label,
                            impl::recreateExecutionPolicy<DeviceExecutionSpace>(
                                dummyExecutionPolicy),
                            kernel, reducers...);
  } else {
    // host execution
    Kokkos::parallel_reduce(
        label,
        impl::recreateExecutionPolicy<HostExecutionSpace>(dummyExecutionPolicy),
        kernel, reducers...);
  }
}

} // namespace dynk

#endif // __DYNAMIC_KOKKOS_HPP__
