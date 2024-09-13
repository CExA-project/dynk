# Changelog

## Version 0.1.0

First version.

## Version 0.2.0

- Use custom execution policies (`dynk::RangePolicy` and `dynk::MDRangePolicy`) that create Kokkos execution policies instead of recreating Kokkos execution policies from themselves.

## Unversioned changes

- Use fences for the layer approach, after and before the parallel block.
