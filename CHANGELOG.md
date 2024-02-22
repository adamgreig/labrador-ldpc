# Changelog

## [v1.2.1] - 2024-02-22

* Fix accidental deletion of working area zero-initialisation in min-sum
  decoder which could lead to different decoding results depending on
  working area re-use (#14).

## [v1.2.0] - 2024-02-14

* Fix some possible panics (#10)
* Add new `hard_bit()` method to `DecodeFrom` trait to simplify converting LLRs to binary bits (#11)

## [v1.1.1] - 2023-10-07

* Change `encode_parity` to only require a shared reference for `data`

## [v1.1.0] - 2023-09-21

* Add `encode_parity()` to `EncodeInto` to permit encoding parity separately
  from a codeword (#5)
* Make many accessor methods `const` (#5)
* Mark some methods `inline`
* Performance improvement on min-sum decoding with f32/f64 LLRs

## [v1.0.1] - 2020-11-26

* Add `#[repr(C)]` to `LDPCCode` to prevent a warning when used in FFI.
* C API added, see [labrador-ldpc-capi].

[labrador-ldpc-capi]: https://github.com/adamgreig/labrador-ldpc/tree/master/capi

## [v1.0.0] - 2017-06-03

Initial release.

[v1.2.1]: https://github.com/adamgreig/labrador-ldpc/releases/tag/v1.2.1
[v1.2.0]: https://github.com/adamgreig/labrador-ldpc/releases/tag/v1.2.0
[v1.1.1]: https://github.com/adamgreig/labrador-ldpc/releases/tag/v1.1.1
[v1.1.0]: https://github.com/adamgreig/labrador-ldpc/releases/tag/v1.1.0
[v1.0.1]: https://github.com/adamgreig/labrador-ldpc/releases/tag/v1.0.1
[v1.0.0]: https://github.com/adamgreig/labrador-ldpc/releases/tag/v1.0.0
