#ifndef __DYNAMIC_HPP__
#define __DYNAMIC_HPP__

#include <Kokkos_Core.hpp>

namespace dynk {

/**
 * @brief Allow to dynamically launch a parallel block region on device or on host.
 * @tparam ParallelLauncher Type of the lamdba.
 * @tparam DeviceExecutionSpace Kokkos execution space for device execution,
 * defaults to Kokkos default execution space.
 * @tparam HostExecutionSpace Kokkos execution space for host execution,
 * defaults to Kokkos default host execution space.
 * @param isExecutedOnDevice If `true`, the parallel block region is launched
 * for execution on the device, otherwise on the host.
 * @param parallelLauncher Lambda to launch that contains a parallel block region.
 */
template <typename ParallelLauncher,
          typename DeviceExecutionSpace = Kokkos::DefaultExecutionSpace,
          typename HostExecutionSpace = Kokkos::DefaultHostExecutionSpace>
void dynamicLaunch(bool const isExecutedOnDevice,
                   ParallelLauncher const &parallelLauncher) {
  // NOTE: Beware the ugly syntax below! We're calling a templated lambda, for
  // which the parenthesis operator is actually templated, hence the need to
  // exhibit the call to `operator()`. As the operator is a method of the
  // object, the `template` keyword is needed to understand the `<>` syntax.
  if (isExecutedOnDevice) {
    // launch for device execution
    parallelLauncher.template operator()<
        DeviceExecutionSpace, typename DeviceExecutionSpace::memory_space>();
  } else {
    // launch for host execution
    parallelLauncher.template
    operator()<HostExecutionSpace, typename HostExecutionSpace::memory_space>();
  }
}

template <typename MemorySpace, typename DualView>
auto getView(DualView &dualView) {
  return dualView.template view<MemorySpace>();
}

template <typename MemorySpace, typename DualView>
void setModified(DualView &dualView) {
  dualView.template modify<MemorySpace>();
}

struct use_annotated_operator {};

template<typename Lambda>
class KokkosLambdaAdapter {
    Lambda mLambda;

    public:

    explicit KokkosLambdaAdapter(Lambda const& lambda) : mLambda(lambda) {}

    template<typename... Args>
    void operator()(Args&... args) {
        mLambda(args...);
    }

    template<typename... Args>
    KOKKOS_FUNCTION void operator()(use_annotated_operator, Args&... args) {
        mLambda(args...);
    }
};

} // namespace dynk

#endif // ifndef __DYNAMIC_HPP__
