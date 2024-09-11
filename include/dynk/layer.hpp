#ifndef __DYNAMIC_KOKKOS_HPP__
#define __DYNAMIC_KOKKOS_HPP__

/**
 * This code proposes helper functions to run parallel block regions on either
 * the device or on the host, the choice being done at runtime. The signature
 * of the familiar `parallel_for` and `parallel_reduce` functions is left
 * similar, the list of paramuments is prepended with a Boolean value,
 * indicating if the code should run on the device (i.e. on
 * `Kokkos::DefaultExecutionSpace` by default) or not (i.e. on
 * `Kokkos::DefaultHostExecutionSpace` by default), and the execution policy is
 * replaced by a custom object to create one for any execution space. Note
 * that only the most common uses that appear in the documentation are
 * reproduced.
 *
 * Though this could be pretty useful, I personnaly think this is not a wise
 * idea. I guess there is a reason why Kokkos didn't introduce such a dynamic
 * execution space selection in the first place.
 */

#include <array>
#include <initializer_list>
#include <memory>
#include <string>

#include <Kokkos_Core.hpp>

#include "dynk/dual_view.hpp"

namespace dynk {

/**
 * Store parameters to create a `Kokkos::RangePolicy`.
 */
class RangePolicyCreator {
  std::size_t mBegin;
  std::size_t mEnd;

public:
  RangePolicyCreator(std::size_t const begin, std::size_t const end)
      : mBegin(begin), mEnd(end) {}

  /**
   * Create a `Kokkor::RangePolicy`.
   *
   * @tparam ExecutionSpace Execution space of the execution policy.
   * @return Execution policy.
   */
  template <typename ExecutionSpace> auto getPolicy() const {
    return Kokkos::RangePolicy<ExecutionSpace>(mBegin, mEnd);
  }
};

/**
 * Store parameters to create a `Kokkos::MDRangePolicy`.
 *
 * @tparam rank Rank of the multidimensional range.
 */
template <std::size_t rank> class MDRangePolicyCreator {
  Kokkos::Array<std::size_t, rank> mBegin;
  Kokkos::Array<std::size_t, rank> mEnd;
  Kokkos::Array<std::size_t, rank> mTile;

public:
  MDRangePolicyCreator(Kokkos::Array<std::size_t, rank> begin,
                       Kokkos::Array<std::size_t, rank> end,
                       Kokkos::Array<std::size_t, rank> tile = {})
      : mBegin(begin), mEnd(end), mTile(tile) {}

  /**
   * Create a `Kokkor::MDRangePolicy`.
   *
   * @tparam ExecutionSpace Execution space of the execution policy.
   * @return Execution policy.
   */
  template <typename ExecutionSpace> auto getPolicy() const {
    return Kokkos::MDRangePolicy<ExecutionSpace, Kokkos::Rank<rank>>(
        mBegin, mEnd, mTile);
  }
};

namespace impl {

/**
 * Get an execution policy from an integer.
 *
 * @tparam ExecutionSpace Execution space of the execution policy.
 * @tparam SizeType Type of the size.
 * @param end Last iteration to perform.
 * @return Single-dimension execution policy.
 */
template <typename ExecutionSpace, typename SizeType,
          typename Enable = std::enable_if_t<std::is_integral_v<SizeType>>>
auto getPolicy(SizeType const end) {
  return Kokkos::RangePolicy<ExecutionSpace>(0, end);
}

/**
 * Get an execution policy from an execution policy creator.
 *
 * @tparam ExecutionSpace Execution space of the execution policy.
 * @tparam ExecutionPolicyCreator Type of the execution policy creator.
 * @param executionPolicyCreator Execution policy creator.
 * @return Execution policy.
 */
template <typename ExecutionSpace, typename ExecutionPolicyCreator,
          typename Enable =
              std::enable_if_t<!std::is_integral_v<ExecutionPolicyCreator>>>
auto getPolicy(ExecutionPolicyCreator const &executionPolicyCreator) {
  return executionPolicyCreator.template getPolicy<ExecutionSpace>();
}

} // namespace impl

/**
 * Parallel for that can be executed dynamically on device or on host
 * depending on a Boolean parameter.
 *
 * @tparam ExecutionPolicyCreator Type of the execution policy creator.
 * @tparam Kernel Type of the kernel.
 * @tparam DeviceExecutionSpace Kokkos execution space for device execution,
 * defaults to Kokkos default execution space.
 * @tparam HostExecutionSpace Kokkos execution space for host execution,
 * defaults to Kokkos default host execution space.
 * @param isExecutedOnDevice If `true`, the parallel for is executed on the
 * device, otherwise on the host.
 * @param label Label of the kernel.
 * @param executionPolicyCreator Object containing the parameters to create an
 * execution policy.
 * @param kernel Kernel to execute withing a Kokkos parallel for region.
 */
template <
    typename ExecutionPolicyCreator, typename Kernel,
    typename DeviceExecutionSpace = Kokkos::DefaultExecutionSpace,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostExecutionSpace = Kokkos::DefaultHostExecutionSpace,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
void parallel_for(bool const isExecutedOnDevice, std::string const &label,
                  ExecutionPolicyCreator const &executionPolicyCreator,
                  Kernel const &kernel) {
  if (isExecutedOnDevice) {
    // device execution
    Kokkos::parallel_for(
        label, impl::getPolicy<DeviceExecutionSpace>(executionPolicyCreator),
        kernel);
  } else {
    // host execution
    Kokkos::parallel_for(
        label, impl::getPolicy<HostExecutionSpace>(executionPolicyCreator),
        kernel);
  }
}

/**
 * Parallel reduce that can be executed dynamically on device or on host
 * depending on a Boolean parameter.
 *
 * @tparam ExecutionPolicyCreator Type of the execution policy creator.
 * @tparam Kernel Type of the kernel.
 * @tparam Reducer Type of the reducers.
 * @tparam DeviceExecutionSpace Kokkos execution space for device execution,
 * defaults to Kokkos default execution space.
 * @tparam HostExecutionSpace Kokkos execution space for host execution,
 * defaults to Kokkos default host execution space.
 * @param isExecutedOnDevice If `true`, the parallel for is executed on the
 * device, otherwise on the host.
 * @param label Label of the kernel.
 * @param executionPolicyCreator Object containing the parameters to create an
 * execution policy.
 * @param kernel Kernel to execute withing a Kokkos parallel for region.
 */
template <
    typename ExecutionPolicyCreator, typename Kernel, typename... Reducer,
    typename DeviceExecutionSpace = Kokkos::DefaultExecutionSpace,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostExecutionSpace = Kokkos::DefaultHostExecutionSpace,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
void parallel_reduce(bool const isExecutedOnDevice, std::string const &label,
                     ExecutionPolicyCreator const &executionPolicyCreator,
                     Kernel const &kernel, Reducer &...reducers) {
  if (isExecutedOnDevice) {
    // device execution
    Kokkos::parallel_reduce(
        label,
        executionPolicyCreator.template getPolicy<DeviceExecutionSpace>(),
        kernel, reducers...);
  } else {
    // host execution
    Kokkos::parallel_reduce(
        label, executionPolicyCreator.template getPolicy<HostExecutionSpace>(),
        kernel, reducers...);
  }
}

} // namespace dynk

#endif // __DYNAMIC_KOKKOS_HPP__
