# Labrador-LDPC

A crate for encoding and decoding a selection of low-density parity check
(LDPC) error correcting codes. Currently, the CCSDS 231.1-O-1 TC codes at rate
r=1/2 with dimensions k=128, k=256, and k=512, and the CCSDS 131.0-B-2 TM codes
at rates r=1/2, r=2/3, and r=4/5 with dimensions k=1024 and k=4096 are
supported.

No dependencies, `no_std`. Designed for both high-performance decoding and
resource-constrained embedded scenarios.

[Documentation](https://docs.rs/labrador-ldpc)

[Repository](https://github.com/adamgreig/labrador-ldpc)

[C API](https://github.com/adamgreig/labrador-ldpc/tree/master/capi)
