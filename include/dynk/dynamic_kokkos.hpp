#ifndef __DYNAMIC_KOKKOS_HPP__
#define __DYNAMIC_KOKKOS_HPP__

/**
 * This code proposes helper functions to run parallel block regions on either
 * the device or on the host, the choice being done at runtime. The signature
 * of the familiar `parallel_for` and `parallel_reduce` functions is left
 * similar, the list of arguments is prepended with a Boolean value, indicating
 * if the code should run on the device (i.e. on
 * `Kokkos::DefaultExecutionSpace` by default) or not (i.e. on
 * `Kokkos::DefaultHostExecutionSpace` by default). Note that only the most
 * common uses that appear in the documentation are reproduced.
 *
 * Though this could be pretty useful, I personnaly think this is not a wise
 * idea. I guess there is a reason why Kokkos didn't introduce such a dynamic
 * execution space selection in the first place.
 */

#include <string>

#include "Kokkos_Core.hpp"

namespace dynk {

namespace impl {

/**
 * @brief Recreate a range execution policy for the requested execution space.
 * @tparam ExecutionSpace Requested execution space.
 * @arg policy Execution policy to recreate.
 * @return Recreated execution policy.
 */
template <typename ExecutionSpace>
auto recreateExecutionPolicy(Kokkos::RangePolicy<> const &policy) {
  return Kokkos::RangePolicy<ExecutionSpace>(policy.begin(), policy.end());
}

/**
 * @brief Recreate a simple range execution policy for the requested execution
 * space.
 * @tparam ExecutionSpace Requested execution space.
 * @arg end Number of elements to work on.
 * @return Recreated execution policy.
 */
template <typename ExecutionSpace> auto recreateExecutionPolicy(int const end) {
  return Kokkos::RangePolicy<ExecutionSpace>(0, end);
}

/**
 * @brief Recreate a multi-dimensional range execution policy for the requested
 * execution space.
 * @tparam ExecutionSpace Requested execution space.
 * @arg policy Execution policy to recreate.
 * @return Recreated execution policy.
 */
template <typename ExecutionSpace, typename Rank>
auto recreateExecutionPolicy(Kokkos::MDRangePolicy<Rank> const &policy) {
  return Kokkos::MDRangePolicy<ExecutionSpace, Rank>(
      policy.m_lower, policy.m_upper, policy.m_tile);
}

} // namespace impl

/**
 * @brief Parallel for that can be executed dynamically on device or on host
 * depending on a Boolean parameter.
 * @tparam ExecutionPolicy Type of the execution policy.
 * @tparam Kernel Type of the kernel.
 * @tparam DeviceExecutionSpace Kokkos execution space for device execution,
 * defaults to Kokkos default execution space.
 * @tparam HostExecutionSpace Kokkos execution space for host execution,
 * defaults to Kokkos default host execution space.
 * @arg isExecutedOnDevice If `true`, the parallel for is executed on the
 * device, otherwise on the host.
 * @arg label Label of the kernel.
 * @arg dummyExecutionPolicy Execution policy that will be adapted for device
 * and host execution.
 * @arg kernel Kernel to execute withing a Kokkos parallel for region.
 */
template <typename ExecutionPolicy, typename Kernel,
          typename DeviceExecutionSpace = Kokkos::DefaultExecutionSpace,
          typename HostExecutionSpace = Kokkos::DefaultHostExecutionSpace>
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
 * @brief Parallel reduce that can be executed dynamically on device or on host
 * depending on a Boolean parameter.
 * @tparam ExecutionPolicy Type of the execution policy.
 * @tparam Kernel Type of the kernel.
 * @tparam Reducer Type of the reducers.
 * @tparam DeviceExecutionSpace Kokkos execution space for device execution,
 * defaults to Kokkos default execution space.
 * @tparam HostExecutionSpace Kokkos execution space for host execution,
 * defaults to Kokkos default host execution space.
 * @arg isExecutedOnDevice If `true`, the parallel for is executed on the
 * device, otherwise on the host.
 * @arg label Label of the kernel.
 * @arg dummyExecutionPolicy Execution policy that will be adapted for device
 * and host execution.
 * @arg kernel Kernel to execute withing a Kokkos parallel for region.
 */
template <typename ExecutionPolicy, typename Kernel, typename... Reducer,
          typename DeviceExecutionSpace = Kokkos::DefaultExecutionSpace,
          typename HostExecutionSpace = Kokkos::DefaultHostExecutionSpace>
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
