#![no_std]
#![deny(missing_docs)]

//! Labrador-LDPC implements a selection of LDPC error correcting codes,
//! including encoders and decoders.
//!
//! It is designed for use with other Labrador components but does not have any dependencies
//! on anything (including `std`) and thus may be used totally standalone. It is reasonably
//! efficient on both serious computers and on small embedded systems. Considerations have
//! been made to accommodate both use cases.
//!
//! No memory allocations are made inside this crate so most methods require you to pass in
//! an allocated block of memory for them to initialise, and then later require you to pass it
//! back when it must be read. Check individual method documentation for further details.
//!
//! Please note this library is still in version 0 and so the API is likely to change.
//! In particular the current interface for passing initialised values (`g`, `cs`, etc)
//! into encoders and decoders is not ergonomic and is likely to change. On the other
//! hand the codes themselves will not change (although new ones may be added) and so newer
//! versions of the library will still be able to communicate with older versions indefinitely.
//!
//! ## Codes
//!
//! Several codes are available in a range of lengths and rates. Current codes come from two
//! sets of CCSDS recommendations, their TC (telecommand) short-length low-rate codes, and their
//! TM (telemetry) higher-length various-rates codes.
//!
//! The TC codes are available in rate r=1/2 and dimensions k=128, k=256, and k=512.
//! They are the same codes defined in CCSDS document 231.1-O-1 and subsequent revisions (although
//! the n=256 code is eventually removed, it lives on here as it's quite useful).
//!
//! The TM codes are available in r=1/2, r=2/3, and r=4/5, for dimensions k=1024 and k=4096.
//! They are the same codes defined in CCSDS document 131.0-B-2 and subsequent revisions.
//!
//! For more information on the codes themselves please see the CCSDS publications:
//! https://public.ccsds.org/default.aspx
//!
//! **Which code should I pick?**: for short and highly-reliable messages, the TC codes make sense,
//! especially if they need to be decoded on a constrained system such as an embedded platform.
//! For most other data transfer, the TM codes are more flexible and generally better suited.
//!
//! ## Encoders
//!
//! There are two encoders available:
//!
//! * The low-memory encoder, `encode_small`, which is as much as a hundred times slower but
//!   does not need to have the generator matrix in RAM at all, and so has a low memory overhead.
//!   The only memory required is that needed for the input and output codewords in a compact
//!   binary representation (i.e. one u8 per eight bits of data).
//!
//! * The fast encoder, `encode_fast`, is much quicker, but requires the generator matrix `g`
//!   to have been initialised at runtime by the `initialise_generator` method.
//!   The precise overhead depends on the code in use, and is available as a constant
//!   `CodeParams.generator_len` (length of a u32 array), in addition to the memory required
//!   for the input and output data.
//!
//! ## Decoders
//!
//! There are two decoders available:
//!
//! * The low-memory decoder, `decode_bf`, uses a bit flipping algorithm with hard information.
//!   This is quite far from optimal for decoding, but requires much less RAM and does still
//!   correct some errors. It's only really useful on something without a hardware floating
//!   point unit or with very little memory available.
//! * The high-performance decoder, `decode_mp`, uses a modified min-sum decoding algorithm to
//!   perform near-optimal decoding albeit with much higher memory overhead.
//!
//! Please see the individual decoders for more details on their requirements.
//!
//! ## Example
//!
//! ```
//! extern crate labrador_ldpc;
//! use labrador_ldpc::LDPCCode;
//!
//! fn main() {
//!     // Pick the TC128 code, n=128 k=64 (that's 8 bytes of user data encoded into 16 bytes)
//!     let code = LDPCCode::TC128;
//!
//!     // Generate some data to encode
//!     let txdata: Vec<u8> = (0..8).collect();
//!
//!     // Allocate memory for the encoded data
//!     let mut txcode = vec![0u8; code.n()/8];
//!
//!     // Encode
//!     code.encode_small(&txdata, &mut txcode);
//!
//!     // Allocate memory to store the log-likelihood ratios of the "received" bits
//!     let mut llrs = vec![0f32; code.n()];
//!
//!     // Convert the txcode bits into log likelihood ratios
//!     code.hard_to_llrs(&txcode, &mut llrs, -3.0);
//!
//!     // Erase and corrupt some bits
//!     llrs[1] = 0.0;
//!     llrs[5] = -llrs[5];
//!
//!     // Allocate memory for decoded data
//!     let mut rxcode = vec![0u8; code.output_len()];
//!
//!     // Initialise the data needed to run a decoder
//!     let mut ci = vec![0; code.sparse_paritycheck_ci_len()];
//!     let mut cs = vec![0; code.sparse_paritycheck_cs_len()];
//!     let mut vi = vec![0; code.sparse_paritycheck_vi_len()];
//!     let mut vs = vec![0; code.sparse_paritycheck_vs_len()];
//!     code.init_sparse_paritycheck(&mut ci, &mut cs, &mut vi, &mut vs);
//!
//!     // Allocate some memory for the decoder's working area
//!     let mut working = vec![0f32; code.decode_mp_working_len()];
//!
//!     // Decode
//!     code.decode_mp(&ci, &cs, &vi, &vs, &llrs, &mut rxcode, &mut working);
//!
//!     // We decoded a whole codeword, so extract just the data into a new Vec
//!     let rxdata = rxcode[0..8].to_vec();
//!
//!     // Check
//!     assert_eq!(rxdata, txdata);
//! }
//! ```
//!

#[cfg(test)]
#[macro_use]
extern crate std;

pub mod codes;
pub mod encoder;
pub mod decoder;
pub use codes::{LDPCCode};
