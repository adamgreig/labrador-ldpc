// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

//! This module provides decoding functions for turning codewords into data.
//!
//! Please refer to the `decode_ms` and `decode_bf` methods on
//! [`LDPCCode`](../codes/enum.LDPCCode.html) for more details.

use core::i8;
use core::i16;
use core::i32;
use core::f32;
use core::f64;

use core::ops::{Add,AddAssign,Neg,Sub};

use ::codes::LDPCCode;

// Ugh gross yuck.
//
// No `f32::abs()` available with `no_std`, and it's not worth bringing in some
// dependency just to get it. This is however used right in the hottest decoder
// loop and it's so much faster than the obvious `if f < 0 { -f } else { f }`.
fn fabs(f: f64) -> f64 {
    unsafe {
        let x: u64 = *((&f as *const f64) as *const u64) & 0x7FFFFFFFFFFFFFFF;
        *((&x as *const u64) as *const f64)
    }
}
fn fabsf(f: f32) -> f32 {
    unsafe {
        let x: u32 = *((&f as *const f32) as *const u32) & 0x7FFFFFFF;
        *((&x as *const u32) as *const f32)
    }
}

/// Trait for types that the min-sum decoder can operate with.
///
/// Implemented for `i8`, `i16`, `i32`, `f32`, and `f64`.
pub trait DecodeFrom:
    Sized + Clone + Copy + PartialEq + Add + AddAssign +
    Neg<Output=Self> + Sub<Output=Self> + PartialOrd
{
    /// 1 in T
    fn one()            -> Self;
    /// 0 in T
    fn zero()           -> Self;
    /// Maximum value T can represent
    fn maxval()         -> Self;
    /// Absolute value of self
    fn abs(&self)       -> Self;
}

impl DecodeFrom for i8 {
    fn one()            -> i8 { 1 }
    fn zero()           -> i8 { 0 }
    fn maxval()         -> i8 { i8::MAX }
    fn abs(&self)       -> i8 { i8::abs(*self) }
}
impl DecodeFrom for i16 {
    fn one()            -> i16 { 1 }
    fn zero()           -> i16 { 0 }
    fn maxval()         -> i16 { i16::MAX }
    fn abs(&self)       -> i16 { i16::abs(*self) }
}
impl DecodeFrom for i32 {
    fn one()            -> i32 { 1 }
    fn zero()           -> i32 { 0 }
    fn maxval()         -> i32 { i32::MAX }
    fn abs(&self)       -> i32 { i32::abs(*self) }
}
impl DecodeFrom for f32 {
    fn one()            -> f32 { 1.0 }
    fn zero()           -> f32 { 0.0 }
    fn maxval()         -> f32 { f32::MAX }
    fn abs(&self)       -> f32 { fabsf(*self) }
}
impl DecodeFrom for f64 {
    fn one()            -> f64 { 1.0 }
    fn zero()           -> f64 { 0.0 }
    fn maxval()         -> f64 { f64::MAX }
    fn abs(&self)       -> f64 { fabs(*self) }
}

impl LDPCCode {

    /// Get the length of [u8] required for the working area of `decode_bf`.
    ///
    /// Equal to n + punctured_bits.
    pub fn decode_bf_working_len(&self) -> usize {
        self.n() + self.punctured_bits()
    }

    /// Get the length of [T] required for the working area of `decode_ms`.
    ///
    /// Equal to 2 * paritycheck_sum. XXX
    pub fn decode_ms_working_len(&self) -> usize {
        2 * self.paritycheck_sum() as usize
    }

    /// Get the length of [u8] required for the output of any decoder.
    ///
    /// Equal to (n+punctured_bits)/8.
    pub fn output_len(&self) -> usize {
        (self.n() + self.punctured_bits()) / 8
    }

    /// Hard erasure decoding algorithm.
    ///
    /// Used to preprocess punctured codes before attempting bit-flipping decoding,
    /// as the bit-flipping algorithm cannot handle erasures.
    ///
    /// The basic idea is:
    ///
    /// * For each erased bit `a`:
    ///     * For each check `i` that `a` is associated with:
    ///         * If `a` is the only erasure that `i` is associated with,
    ///           then compute the parity of `i`, and cast a vote for the
    ///           value of `a` that would give even parity
    ///         * Otherwise ignore `i`
    ///     * If there is a majority vote, set `a` to the winning value, and mark
    ///       it not longer erased. Otherwise, leave it erased.
    ///
    /// This is based on the paper:
    /// Novel multi-Gbps bit-flipping decoders for punctured LDPC codes,
    /// by Archonta, Kanistras, and Paliouras, MOCAST 2016.
    ///
    /// * `ci`, `cs`, `vi`, and `vs` must all have been initialised appropriately.
    /// * `codeword` must be (n+p)/8 long (`self.output_len()`), with the first n/8 bytes already
    ///              set to the received hard information, and the punctured bits at the end will
    ///              be updated.
    /// * `working` must be (n+p) bytes long (`self.decode_bf_working_len()`).
    ///
    /// Returns `(success, number of iterations run)`. Note that `success`=false only indicates
    /// that not every punctured bit was correctly recovered; many may have been successful.
    fn decode_erasures(&self, codeword: &mut [u8], working: &mut [u8], maxiters: usize) -> (bool, usize)
    {
        //assert_eq!(ci.len(), self.sparse_paritycheck_ci_len());
        //assert_eq!(cs.len(), self.sparse_paritycheck_cs_len());
        //assert_eq!(vi.len(), self.sparse_paritycheck_vi_len());
        //assert_eq!(vs.len(), self.sparse_paritycheck_vs_len());
        /*
        assert_eq!(codeword.len(), self.output_len());
        assert_eq!(working.len(), self.decode_bf_working_len());

        let n = self.n();
        let p = self.punctured_bits();

        // Rename working area
        let erasures = working;

        // Initialise erasures
        for e in &mut erasures[..n] { *e = 0; }
        for e in &mut erasures[n..] { *e = 1; }

        // Initialise punctured part of output
        for c in &mut codeword[n/8..] { *c = 0; }

        // Track how many bits we've fixed
        let mut bits_fixed = 0;

        for iter in 0..maxiters {

            // For each punctured bit
            for a in n..n+p {

                // Skip bits we have since fixed
                if erasures[a] == 0 {
                    continue;
                }

                // Track votes for 0 (negative) or 1 (positive)
                let mut votes = 0;

                // For each check this bit is associated with
                for i in &vi[vs[a] as usize .. vs[a+1] as usize] {
                    let i = *i as usize;
                    let mut parity = 0;

                    // See what the check parity is, and quit without voting if this check has
                    // any other erasures
                    let mut only_one_erasure = true;
                    for b in &ci[cs[i] as usize .. cs[i+1] as usize] {
                        let b = *b as usize;

                        // Skip if this is the bit we're currently considering
                        if a == b {
                            continue;
                        }

                        // If we see another erasure, stop
                        if erasures[b] == 1 {
                            only_one_erasure = false;
                            break;
                        }

                        // Otherwise add up parity for this check
                        parity += (codeword[b/8] >> (7-(b%8))) & 1;
                    }

                    // Cast a vote if we didn't see any other erasures
                    if only_one_erasure {
                        votes += if parity & 1 == 1 { 1 } else { -1 };
                    }
                }

                // If we have a majority vote one way or the other, great!
                // Set ourselves to the majority vote value and clear our erasure status.
                if votes != 0 {
                    erasures[a] = 0;
                    bits_fixed += 1;

                    if votes > 0 {
                        codeword[a/8] |=   1<<(7-(a%8));
                    } else {
                        codeword[a/8] &= !(1<<(7-(a%8)));
                    }
                }

                if bits_fixed == p {
                    return (true, iter);
                }

            }
        }

    */
        // If we got this far we have not succeeded
        (false, maxiters)
    }

    /// TODO update these docs
    /// Bit flipping decoder.
    ///
    /// This algorithm is quick but only operates on hard information and consequently leaves a
    /// lot of error-correcting capability behind. It is around 1dB worse than the message passing
    /// decoder. However, it requires much less memory and uses no floating point operations.
    ///
    /// Requires:
    ///
    /// * `ci` and `cs` must have been initialised from `init_sparse_paritycheck_checks()`
    /// * For all codes where `punctured_bits` is not zero (all TM codes),
    ///   `vi` and `vs` must additionally have been initialised, using `init_sparse_paritycheck()`,
    ///   otherwise you may pass in None for these.
    /// * `input` must be `n/8` long, where each bit is the received hard information
    /// * `output` must be `(n+p)/8` (=`self.output_len()`) bytes long and is written with the
    ///   decoded codeword, so the user data is present in the first `k/8` bytes.
    /// * `working` must be `n+p` (=`self.decode_bf_working_len()`) bytes long.
    ///
    /// Returns `(decoding success, iters)`. For punctured codes, `iters` includes iterations
    /// of the erasure decoding algorithm which is run first.
    ///
    /// ## Panics
    /// * `ci.len()` must be exactly `self.sparse_paritycheck_ci_len()`.
    /// * `cs.len()` must be exactly `self.sparse_paritycheck_cs_len()`.
    /// * `vi.unwrap().len()`, if required, must be exactly `self.sparse_paritycheck_vi_len()`.
    /// * `vs.unwrap().len()`, if required, must be exactly `self.sparse_paritycheck_vs_len()`.
    /// * `input.len()` must be exactly `self.n()/8`
    /// * `output.len()` must be exactly `self.output_len()`.
    /// * `working.len()` must be exactly `self.decode_bf_working_len()`.
    pub fn decode_bf(&self, input: &[u8], output: &mut [u8], working: &mut [u8], maxiters: usize) -> (bool, usize)
    {
        assert_eq!(input.len(), self.n()/8);
        assert_eq!(output.len(), self.output_len());
        assert_eq!(working.len(), self.decode_bf_working_len());
        output[..self.n()/8].copy_from_slice(input);

        // top bit of each entry in working is parity check
        // remaining 7 bits is counter of violations

        for iter in 0..maxiters {
            for v in &mut working[..] { *v = 0 }

            // Calculate the parity of each parity check
            for (check, var) in self.iter_paritychecks() {
                if output[var/8] >> (7-(var%8)) & 1 == 1 {
                    working[check] ^= 0x80;
                }
            }

            // Count how many parity violations each variable is associated with
            let mut max_violations = 0;
            for (check, var) in self.iter_paritychecks() {
                if working[check] & 0x80 == 0x80 {
                    working[var] += 1;
                    if working[var] & 0x7F > max_violations {
                        max_violations = working[var] & 0x7F;
                    }
                }
            }

            if max_violations == 0 {
                // With no violations, we've decoded successfully
                return (true, iter);
            } else {
                // Otherwise, flip all the bits that have the maximum number of violations
                for (var, violations) in working.iter().enumerate() {
                    if *violations & 0x7F == max_violations {
                        output[var/8] ^= 1<<(7-(var%8));
                    }
                }
            }
        }

        (false, maxiters)
    }

    /// TODO update these docs
    /// Message passing based min-sum decoder.
    ///
    /// This algorithm is slower, requires more memory, and ideally operates on soft information,
    /// but it provides very close to optimal decoding. If you don't have soft information but do
    /// have the channel BER, you can use `decode_hard_to_llrs_with_ber` to go from hard
    /// information (bytes from a receiver) to soft information (LLRs). If you don't have that, you
    /// can use `decode_hard_to_llrs` to generate arbitrary LLRs from the hard information.
    ///
    /// Requires:
    ///
    /// * `ci`, `cs`, `vi`, `vs` must all be initialised from from `init_sparse_paritycheck()`,
    /// * `llrs` must be `n` long, with positive numbers more likely to be 0.
    /// * `output` must be allocated to (n+p)/8 bytes, of which the first k/8 bytes will be set
    ///   to the decoded message (and the rest to the parity bits of the complete codeword)
    /// * `working` is the working area which must be provided and must have
    ///   `decode_ms_working_len` elements, equal to 2*paritycheck_sum.
    ///
    /// Returns decoding success and the number of iterations run for.
    ///
    /// ## Panics
    /// * `ci.len()` must be exactly `self.sparse_paritycheck_ci_len()`.
    /// * `cs.len()` must be exactly `self.sparse_paritycheck_cs_len()`.
    /// * `vi.len()` must be exactly `self.sparse_paritycheck_vi_len()`.
    /// * `vs.len()` must be exactly `self.sparse_paritycheck_vs_len()`.
    /// * `llrs.len()` must be exactly `self.n()`
    /// * `output.len()` must be exactly `self.output_len()`.
    /// * `working.len()` must be exactly `self.decode_ms_working_len()`.
    pub fn decode_ms<T: DecodeFrom>(&self, llrs: &[T], output: &mut [u8], working: &mut [T], maxiters: usize) -> (bool, usize)
    {
        assert_eq!(llrs.len(), self.n());
        assert_eq!(output.len(), self.output_len());
        assert_eq!(working.len(), self.decode_ms_working_len());

        for w in &mut working[..] { *w = T::zero() }
        let (u, v) = working.split_at_mut(self.decode_ms_working_len() / 2);

        let n = self.n();
        let k = self.k();
        let p = self.punctured_bits();
        let mut va = vec![T::zero(); n + p];
        let mut ui_min1 = vec![T::zero(); n + p - k];
        let mut ui_min2 = vec![T::zero(); n + p - k];
        let mut ui_sgns = vec![0u8; (n + p - k) / 8];
        let mut parities = vec![0u8; (n + p - k) / 8];

        for iter in 0..maxiters {
            va[..llrs.len()].copy_from_slice(llrs);
            for x in &mut va[llrs.len()..] { *x = T::zero() }

            for (idx, (check, var)) in self.iter_paritychecks().enumerate() {
                if v[idx].abs() == ui_min1[check] {
                    u[idx] = ui_min2[check];
                } else {
                    u[idx] = ui_min1[check];
                }
                if ui_sgns[check/8] >> (check%8) & 1 == 1 {
                    u[idx] = -u[idx];
                }
                if v[idx] < T::zero() {
                    u[idx] = -u[idx];
                }

                // accumulate incoming messages to each variable
                va[var] += u[idx];
            }

            for o in &mut output[..] { *o = 0 }
            for var in 0..(self.n() + self.punctured_bits()) {
                if va[var] <= T::zero() {
                    output[var/8] |= 1 << (7 - (var%8));
                }
            }

            for x in &mut ui_min1[..] { *x = T::maxval() }
            for x in &mut ui_min2[..] { *x = T::maxval() }
            for x in &mut ui_sgns[..] { *x = 0 }
            for x in &mut parities[..] { *x = 0 }
            for (idx, (check, var)) in self.iter_paritychecks().enumerate() {
                // output messages to each parity check
                let new_v_ai = va[var] - u[idx];
                if v[idx] != T::zero() && (new_v_ai >= T::zero()) != (v[idx] >= T::zero()) {
                    v[idx] = T::zero();
                } else {
                    v[idx] = new_v_ai;
                }

                // accumulate two minimums
                if v[idx].abs() < ui_min1[check] {
                    ui_min2[check] = ui_min1[check];
                    ui_min1[check] = v[idx].abs();
                } else if v[idx].abs() < ui_min2[check] {
                    ui_min2[check] = v[idx].abs();
                }

                // accumulate signs
                if v[idx] < T::zero() {
                    ui_sgns[check/8] ^= 1<<(check%8);
                }

                // accumulate parity
                if output[var/8] >> (7-(var%8)) & 1 == 1 {
                    parities[check/8] ^= 1<<(check%8);
                }
            }

            // check parities
            if *parities.iter().max().unwrap() == 0 {
                return (true, iter);
            }
        }

        (false, maxiters)
    }

    /// Convert hard information into LLRs.
    ///
    /// The min-sum decoding used in `decode_ms` is invariant to linear scaling
    /// in LLR, so it doesn't matter which value is picked so long as the sign
    /// is correct. This function just assigns -/+ 1.0 for 1/0 bits.
    ///
    /// `input` must be n/8 long, `llrs` must be n long.
    ///
    /// ## Panics
    /// * `input.len()` must be exactly `self.n()/8`
    /// * `llrs.len()` must be exactly `self.n()`
    pub fn hard_to_llrs(&self, input: &[u8], llrs: &mut [f32]) {
        assert_eq!(input.len(), self.n()/8, "input.len() != n/8");
        assert_eq!(llrs.len(), self.n(), "llrs.len() != n");
        let llr = -1.0f32;
        for (idx, byte) in input.iter().enumerate() {
            for i in 0..8 {
                llrs[idx*8 + i] = if (byte >> (7-i)) & 1 == 1 { llr } else { -llr };
            }
        }
    }

    /// Convert LLRs into hard information.
    ///
    /// `llrs` must be n long, `output` must be n/8 long.
    ///
    /// ## Panics
    /// * `input.len()` must be exactly `self.n()/8`
    /// * `llrs.len()` must be exactly `self.n()`
    pub fn llrs_to_hard(&self, llrs: &[f32], output: &mut [u8]) {
        assert_eq!(llrs.len(), self.n(), "llrs.len() != n");
        assert_eq!(output.len(), self.n()/8, "output.len() != n/8");

        for o in &mut output[..] { *o = 0 }

        for (i, llr) in llrs.iter().enumerate() {
            if *llr < 0.0 {
                output[i/8] |= 1 << (7 - (i%8));
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::v1::*;

    use ::codes::{LDPCCode, CodeParams,
                  TC128_PARAMS,  TC256_PARAMS,  TC512_PARAMS,
                  TM1280_PARAMS, TM1536_PARAMS, TM2048_PARAMS,
                  TM5120_PARAMS, TM6144_PARAMS, TM8192_PARAMS};

    const CODES: [LDPCCode;  9] = [LDPCCode::TC128,   LDPCCode::TC256,   LDPCCode::TC512,
                                   LDPCCode::TM1280,  LDPCCode::TM1536,  LDPCCode::TM2048,
                                   LDPCCode::TM5120,  LDPCCode::TM6144,  LDPCCode::TM8192,
    ];

    const PARAMS: [CodeParams; 9] = [TC128_PARAMS,  TC256_PARAMS,  TC512_PARAMS,
                                     TM1280_PARAMS, TM1536_PARAMS, TM2048_PARAMS,
                                     TM5120_PARAMS, TM6144_PARAMS, TM8192_PARAMS,
    ];

    #[test]
    fn test_decode_ms_working_len() {
        for (code, param) in CODES.iter().zip(PARAMS.iter()) {
            assert_eq!(code.decode_ms_working_len(), param.decode_ms_working_len);
        }
    }

    #[test]
    fn test_decode_bf_working_len() {
        for (code, param) in CODES.iter().zip(PARAMS.iter()) {
            assert_eq!(code.decode_bf_working_len(), param.decode_bf_working_len);
        }
    }

    #[test]
    fn test_output_len() {
        for (code, param) in CODES.iter().zip(PARAMS.iter()) {
            assert_eq!(code.output_len(), param.output_len);
        }
    }

    #[test]
    fn test_hard_to_llrs() {
        let code = LDPCCode::TC128;
        let hard = vec![255, 254, 253, 252, 251, 250, 249, 248,
                        203, 102, 103, 120, 107,  30, 157, 169];
        let mut llrs = vec![0f32; code.n()];
        let llr = -1.0;
        code.hard_to_llrs(&hard, &mut llrs);
        assert_eq!(llrs, vec![
             llr,  llr,  llr,  llr,  llr,  llr,  llr,  llr,
             llr,  llr,  llr,  llr,  llr,  llr,  llr, -llr,
             llr,  llr,  llr,  llr,  llr,  llr, -llr,  llr,
             llr,  llr,  llr,  llr,  llr,  llr, -llr, -llr,
             llr,  llr,  llr,  llr,  llr, -llr,  llr,  llr,
             llr,  llr,  llr,  llr,  llr, -llr,  llr, -llr,
             llr,  llr,  llr,  llr,  llr, -llr, -llr,  llr,
             llr,  llr,  llr,  llr,  llr, -llr, -llr, -llr,
             llr,  llr, -llr, -llr,  llr, -llr,  llr,  llr,
            -llr,  llr,  llr, -llr, -llr,  llr,  llr, -llr,
            -llr,  llr,  llr, -llr, -llr,  llr,  llr,  llr,
            -llr,  llr,  llr,  llr,  llr, -llr, -llr, -llr,
            -llr,  llr,  llr, -llr,  llr, -llr,  llr,  llr,
            -llr, -llr, -llr,  llr,  llr,  llr,  llr, -llr,
             llr, -llr, -llr,  llr,  llr,  llr, -llr,  llr,
             llr, -llr,  llr, -llr,  llr, -llr, -llr,  llr]);
    }

    #[test]
    fn test_llrs_to_hard() {
        let code = LDPCCode::TC128;
        let llr = -1.0;
        let llrs = vec![
             llr,  llr,  llr,  llr,  llr,  llr,  llr,  llr,
             llr,  llr,  llr,  llr,  llr,  llr,  llr, -llr,
             llr,  llr,  llr,  llr,  llr,  llr, -llr,  llr,
             llr,  llr,  llr,  llr,  llr,  llr, -llr, -llr,
             llr,  llr,  llr,  llr,  llr, -llr,  llr,  llr,
             llr,  llr,  llr,  llr,  llr, -llr,  llr, -llr,
             llr,  llr,  llr,  llr,  llr, -llr, -llr,  llr,
             llr,  llr,  llr,  llr,  llr, -llr, -llr, -llr,
             llr,  llr, -llr, -llr,  llr, -llr,  llr,  llr,
            -llr,  llr,  llr, -llr, -llr,  llr,  llr, -llr,
            -llr,  llr,  llr, -llr, -llr,  llr,  llr,  llr,
            -llr,  llr,  llr,  llr,  llr, -llr, -llr, -llr,
            -llr,  llr,  llr, -llr,  llr, -llr,  llr,  llr,
            -llr, -llr, -llr,  llr,  llr,  llr,  llr, -llr,
             llr, -llr, -llr,  llr,  llr,  llr, -llr,  llr,
             llr, -llr,  llr, -llr,  llr, -llr, -llr,  llr];
        let mut hard = vec![0u8; code.n()/8];
        code.llrs_to_hard(&llrs, &mut hard);
        assert_eq!(hard, vec![255, 254, 253, 252, 251, 250, 249, 248,
                              203, 102, 103, 120, 107,  30, 157, 169]);
    }

    #[test]
    fn test_decode_bf_tc() {
        let code = LDPCCode::TC256;

        // Make up some TX data
        let txdata: Vec<u8> = (0..16).collect();
        let mut txcode = vec![0u8; code.n()/8];
        code.copy_encode(&txdata, &mut txcode);

        // Copy to rx
        let mut rxcode = txcode.clone();

        // Corrupt some bits
        rxcode[0] = 0xFF;

        // Allocate working area and output area
        let mut working = vec![0u8; code.decode_bf_working_len()];
        let mut output = vec![0u8; code.output_len()];

        // Run decoder
        let (success, _) = code.decode_bf(&rxcode, &mut output, &mut working, 50);

        assert!(success);
        assert_eq!(&txcode[..], &output[..txcode.len()]);

    }
    #[test]
    fn test_decode_ms() {
        let code = LDPCCode::TM1280;

        // Make up a TX codeword
        let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
        let mut txcode = vec![0u8; code.n()/8];
        code.copy_encode(&txdata, &mut txcode);

        // Copy it and corrupt the first bit
        let mut rxcode = txcode.clone();
        rxcode[0] ^= 1<<7;

        // Convert the hard data to LLRs
        let mut llrs = vec![0i8; code.n()];
        for (idx, byte) in rxcode.iter().enumerate() {
            for i in 0..8 {
                llrs[idx*8 + i] = if (byte >> (7-i)) & 1 == 1 { -1i8 } else { 1i8 };
            }
        }
        //code.hard_to_llrs(&rxcode, &mut llrs);

        // Allocate working area and output area
        let mut working = vec![0i8; code.decode_ms_working_len()];
        let mut decoded = vec![0u8; code.output_len()];

        // Run decoder
        let (success, iters) = code.decode_ms(&llrs, &mut decoded, &mut working, 50);

        assert_eq!(&decoded[..8], &txcode[..8]);
        assert!(success);
    }
}
