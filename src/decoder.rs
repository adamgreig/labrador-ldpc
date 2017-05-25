// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

//! This module provides decoding functions for turning codewords into data.
//!
//! Please refer to the `decode_mp` and `decode_bf` methods on
//! [`LDPCCode`](../codes/enum.LDPCCode.html) for more details.

use core::f32;

use ::codes::LDPCCode;

/// Ugh gross yuck.
///
/// No `f32::abs()` available with `no_std`, and it's not worth bringing in some
/// dependency just to get it. This is however used right in the hottest decoder
/// loop and it's so much faster than the obvious if f < 0 { -f } else { f }.
fn fabsf(f: f32) -> f32 {
    unsafe {
        let x: u32 = *((&f as *const f32) as *const u32) & 0x7FFFFFFF;
        *((&x as *const u32) as *const f32)
    }
}

const MP_MAX_ITERS: usize = 20;

impl LDPCCode {

    /// Get the length of [f32] required for the working area of `decode_bf`.
    ///
    /// Equal to n + punctured_bits.
    pub fn decode_bf_working_len(&self) -> usize {
        self.n() + self.punctured_bits()
    }

    /// Get the length of [f32] required for the working area of `decode_mp`.
    ///
    /// Equal to 2 * paritycheck_sum.
    pub fn decode_mp_working_len(&self) -> usize {
        2 * self.paritycheck_sum() as usize
    }

    /// Get the length of [u8] required for the output of any decoder.
    ///
    /// Equal to (n+punctured_bits)/8.
    pub fn output_len(&self) -> usize {
        (self.n() + self.punctured_bits()) / 8
    }


    /// Message passing based decoder.
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
    ///   `decode_mp_working_len` elements, equal to 2*paritycheck_sum.
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
    /// * `working.len()` must be exactly `self.decode_mp_working_len()`.
    pub fn decode_mp(&self, ci: &[u16], cs: &[u16], vi: &[u16], vs: &[u16],
                     llrs: &[f32], output: &mut [u8], working: &mut [f32]) -> (bool, usize)
    {
        assert_eq!(ci.len(), self.sparse_paritycheck_ci_len());
        assert_eq!(cs.len(), self.sparse_paritycheck_cs_len());
        assert_eq!(vi.len(), self.sparse_paritycheck_vi_len());
        assert_eq!(vs.len(), self.sparse_paritycheck_vs_len());
        assert_eq!(llrs.len(), self.n());
        assert_eq!(output.len(), self.output_len());
        assert_eq!(working.len(), self.decode_mp_working_len());

        let n = self.n();
        let p = self.punctured_bits();

        // Initialise working area to all zeros
        for w in &mut working[..] {
            *w = 0.0;
        }

        // Split working area:
        // * u(i->a) will hold messages from checks (i) to variables (a)
        // * v(a->i) will hold messages from variables (a) to checks (i)
        let (u, v) = working.split_at_mut(self.decode_mp_working_len() / 2);

        for iter in 0..MP_MAX_ITERS {
            // Track whether we've had any parity violations this iteration.
            // If we haven't at the end of an iteration, we can stop immediately.
            // Set to false as soon as a single violation is encountered.
            let mut parity_ok = true;

            // Clear the output, we'll OR bits into it as we run through variable nodes
            for o in &mut output[..] {
                *o = 0;
            }

            // Update variable nodes' messages to check nodes.
            // For each variable node, for each check node connected to it,
            // initialise this message v(a->i) to the LLR (or 0 for punctured bits),
            // and then add on all of the incoming messages not from the current check node.
            // Additionally we sum all u(i->a) for this a to marginalise this variable
            // node and see if the hard decoding gives a valid codeword, which is out
            // signal to stop iteration.
            for a in 0..(n+p) {
                // Sum up the marginal LLR for a, starting with llrs[a] or 0 for punctured bits.
                let mut llr_a = if a < n { llrs[a] } else { 0.0 };

                // For each check node i connected to this variable node a
                for a_i in vs[a]..vs[a+1] {
                    let a_i = a_i as usize;
                    let i = vi[a_i] as usize;
                    let prev_v_ai = v[a_i];
                    v[a_i] = if a < n { llrs[a] } else { 0.0 };

                    // For each check node j connected to variable node a
                    for j in &vi[vs[a] as usize..vs[a+1] as usize] {
                        let j = *j as usize;

                        // We need to find where the incoming messages u(j->a) are stored in u.
                        // This means going through every variable node connected to check node j,
                        // and seeing if it's equal to a, and if so, using that message.
                        // This loop could be replaced by another index table the same size as ci,
                        // which might save time if this section proves to be slow.
                        for (j_b, b) in ci[cs[j] as usize .. cs[j+1] as usize].iter().enumerate() {
                            let j_b = cs[j] as usize + j_b;
                            if a == *b as usize {
                                // Sum up just the incoming messages not from i for v(a->i)
                                if j != i {
                                    v[a_i] += u[j_b];
                                }

                                // Sum up all incoming messages for llr_a
                                llr_a += u[j_b];

                                // Once we find a, we can stop looking through variables on j.
                                break;
                            }
                        }
                    }

                    // Our min sum correction trick is to zero any messages that have changed
                    // sign since last time, as per Savin 2009 http://arxiv.org/abs/0803.1090v2
                    if prev_v_ai != 0.0 && (v[a_i] >= 0.0) != (prev_v_ai >= 0.0) {
                        v[a_i] = 0.0;
                    }
                }

                // Hard decode the marginal LLR for a to determine this output bit.
                // (We previously zeroed output, so just set 1 bits).
                if llr_a <= 0.0 {
                    output[a/8] |= 1 << (7 - (a%8));
                }
            }

            // Update check nodes' messages to variable nodes (i=0..(n-k+p)).
            // For each check node, for each variable node connected to it,
            // initialise the message u(i->a) to f32::MAX and then find the minimum of all
            // incoming messages as well as the product of all their signs.
            // Additionally we use this loop to keep track of the parity sum for this check
            // node under hard decoding, and use that to see if the overall message has been
            // decoded yet.
            for (i, cs_ss) in cs.windows(2).enumerate() {
                let mut parity = 0;
                let (cs_start, cs_end) = (cs_ss[0] as usize, cs_ss[1] as usize);

                // For each variable node a connected to check node i
                for i_a in cs_start..cs_end {
                    let a = ci[i_a] as usize;
                    let mut sgnprod = 1.0f32;
                    let mut minacc = f32::MAX;

                    // For each variable node b connected to check node i
                    for b in &ci[cs_start..cs_end] {
                        let b = *b as usize;
                        // Don't process the message from the current variable node
                        if b == a {
                            continue;
                        }

                        // We need to find where the incoming messages v(b->i) are stored in v.
                        // As with the u(j->a) messages, we need to go through each check node j
                        // associated with variable node b, and if j==i we use the message.
                        // This loop could also be replaced by another index table the same size
                        // as vi, which might be useful here.
                        for (b_j, j) in vi[vs[b] as usize .. vs[b+1] as usize].iter().enumerate() {
                            let b_j = vs[b] as usize + b_j;
                            if i == *j as usize {
                                if v[b_j] < 0.0 {
                                    sgnprod = -sgnprod;
                                }
                                let abs_v_bj = fabsf(v[b_j]);
                                if abs_v_bj < minacc {
                                    minacc = abs_v_bj;
                                }

                                // As soon as we find ourselves, we can stop looking.
                                break;
                            }
                        }
                    }

                    // Update u(i->a) with our accumulated minimum and appropriate sign.
                    u[i_a] = sgnprod * minacc;

                    // Work out this node's parity
                    parity += (output[a/8] >> (7 - (a%8))) & 1;
                }

                // Odd parity is bad parity
                if (parity & 1) == 1 {
                    parity_ok = false;
                }
            }

            // If every parity check was satisfied, we're done.
            if parity_ok {
                return (true, iter);
            }
        }

        // If we finished the loop, we ran out of iterations :(
        (false, MP_MAX_ITERS)
    }

    /// Convert hard information and a channel BER into appropriate LLRs.
    ///
    /// Can be used to feed the message passing algorithm soft-ish information.
    ///
    /// `llr` is the log likelihood ratio to use for 1-bits, and must be negative.
    /// If you know the probability of a bit being wrong, P(X=1|Y=0)=ε, then
    /// `llr` is ln(ε/(1-ε)). It doesn't vastly matter if you don't know, try `llr`=-3.
    ///
    /// `input` must be n/8 long, `llrs` must be n long.
    ///
    /// ## Panics
    /// * `input.len()` must be exactly `self.n()/8`
    /// * `llrs.len()` must be exactly `self.n()`
    pub fn hard_to_llrs(&self, input: &[u8], llrs: &mut [f32], llr: f32) {
        assert_eq!(input.len(), self.n()/8);
        assert_eq!(llrs.len(), self.n());
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
        for o in &mut output[..] {
            *o = 0;
        }

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
    fn test_decode_mp_working_len() {
        for (code, param) in CODES.iter().zip(PARAMS.iter()) {
            assert_eq!(code.decode_mp_working_len(), param.decode_mp_working_len);
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
        let llr = -3.0;
        code.hard_to_llrs(&hard, &mut llrs, llr);
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
        let llr = -3.0;
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
    fn test_decode_mp() {
        let code = LDPCCode::TC128;

        // Initialise the parity check matrix
        let mut ci = vec![0; code.sparse_paritycheck_ci_len()];
        let mut cs = vec![0; code.sparse_paritycheck_cs_len()];
        let mut vi = vec![0; code.sparse_paritycheck_vi_len()];
        let mut vs = vec![0; code.sparse_paritycheck_vs_len()];
        code.init_sparse_paritycheck(&mut ci, &mut cs, &mut vi, &mut vs);

        // Make up a TX codeword (the same one we used to test the encoder)
        let txcode = vec![255, 254, 253, 252, 251, 250, 249, 248,
                          203, 102, 103, 120, 107,  30, 157, 169];

        // Copy it and corrupt a few bits
        let mut rxcode = txcode.clone();
        rxcode[5] ^= 1<<4 | 1<<2;
        rxcode[6] ^= 1<<5 | 1<<3;

        // Convert the hard data to LLRs
        let mut llrs = vec![0f32; code.n()];
        code.hard_to_llrs(&rxcode, &mut llrs, -3.0);

        // Allocate working area and output area
        let mut working = vec![0f32; code.decode_mp_working_len()];
        let mut decoded = vec![0u8; code.output_len()];

        // Run decoder
        let (success, _) = code.decode_mp(&ci, &cs, &vi, &vs, &llrs, &mut decoded, &mut working);

        assert!(success);
        assert_eq!(&decoded[..8], &txcode[..8]);
    }
}
