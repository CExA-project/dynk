# Changelog

## Unversioned changes

## Version 0.4.0

- Removed `dynk::getAnonymousView` in favor to a new signature of `dynk::getView`.
- Added `dynk::getSyncedView` with the Boolean value signature.
- Fixed a bug in `dual_view.hpp` template arguments order.

## Version 0.3.0

- Use fences for the layer approach, after and before the parallel block.
- Add documentation.

## Version 0.2.0

- Use custom execution policies (`dynk::RangePolicy` and `dynk::MDRangePolicy`) that create Kokkos execution policies instead of recreating Kokkos execution policies from themselves.

## Version 0.1.0

First version.
