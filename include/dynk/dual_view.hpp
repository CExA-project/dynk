#ifndef __DUAL_VIEW_HPP__
#define __DUAL_VIEW_HPP__

#include <Kokkos_Core.hpp>
#include <Kokkos_DualView.hpp>

namespace dynk {

/**
 * Get a View of a DualView for the requested memory space.
 *
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

template <
    typename T, typename... P,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
Kokkos::View<T, Kokkos::AnonymousSpace, P...>
getViewAnonymous(Kokkos::DualView<T, P...> &dualView,
                 bool const isExecutedOnDevice) {
  if (isExecutedOnDevice) {
    return getView<DeviceMemorySpace>(dualView);
  } else {
    return getView<HostMemorySpace>(dualView);
  }
}

template <
    typename T, typename... P,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
Kokkos::View<T, Kokkos::AnonymousSpace, P...>
getViewAnonymous(Kokkos::DualView<T, P...> const& dualView,
                 bool const isExecutedOnDevice) {
  if (isExecutedOnDevice) {
    return getView<DeviceMemorySpace>(dualView);
  } else {
    return getView<HostMemorySpace>(dualView);
  }
}

/**
 * Get a View of a DualView for the requested memory space which is
 * synchronized if needed.
 *
 * The DualView memory spaces should match with the ones expected. This
 * function is just a pass-through to manipulate the DualView with a simpler
 * syntax.
 *
 * @tparam MemorySpace Memory space requested.
 * @tparam DualView Type of the DualView.
 * @param dualView DualView to take a view from.
 * @return View in the requested memory space, synchronized.
 */
template <typename MemorySpace, typename DualView>
auto getSyncedView(DualView &dualView) {
  dualView.template sync<MemorySpace>();
  return getView<MemorySpace>(dualView);
}

/**
 * Mark a DualView as modified in the requested memory space.
 *
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

/**
 * Mark a DualView as modified in the requested memory space dynamically.
 *
 * @tparam DualView Type of the DualView.
 * @tparam DeviceMemorySpace Device memory space. Defaults to Kokkos default
 * execution space's default memory.
 * @tparam HostMemorySpace Host memory space. Defaults to Kokkos default host
 * execution space's default memory.
 * @param dualView DualView to set.
 * @param isExecutedOnDevice If `true`, mark device view as modified, otherwise
 * mark host view as modified.
 */
template <
    typename DualView,
    typename DeviceMemorySpace = Kokkos::DefaultExecutionSpace::memory_space,
    typename HostMemorySpace = Kokkos::DefaultHostExecutionSpace::memory_space>
void setModified(DualView &dualView, bool const isExecutedOnDevice) {
  if (isExecutedOnDevice) {
    setModified<DeviceMemorySpace>(dualView);
  } else {
    setModified<HostMemorySpace>(dualView);
  }
}

} // namespace dynk

#endif // ifndef __DUAL_VIEW_HPP__
