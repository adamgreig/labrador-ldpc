// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

//! This module contains the available LDPC codes, and the associated constants and methods to
//! load their generator and parity check matrices.

// We have a bunch of expressions with +0 for clarity of where the 0 comes from
#![cfg_attr(feature = "cargo-clippy", allow(identity_op))]

/// This module contains the constants representing the generator matrices.
///
/// They are in a compact form: for each systematic generator matrix, we take just the
/// parity bits (the n-k columns on the right), and then we take just the first row for
/// each circulant (i.e. instead of `k` rows, we take `k/circulant_size` rows), and then
/// pack the bits for that row into `u64`s.
///
/// This is relatively easy to unpack at runtime into a full size generator matrix,
/// by just loading each row, then making a copy rotated right by one bit for each
/// of `circulant_size` rows.
mod compact_generators;

/// This module contains the constants representing the parity check matrices.
///
/// They are in different forms for the TC and TM codes. The representation is explained
/// inside the module's source code. Expanding from this representation to an in-memory
/// parity check matrix or sparse representation thereof is a little involved.
mod compact_parity_checks;

/// Available LDPC codes, and methods to encode and decode them.
///
/// * The TC codes are the Telecommand LDPC codes from CCSDS document 231.1-O-1.
/// * The TM codes are the Telemetry LDPC codes from CCSDS document 131.0-B-2.
/// * For full details please see: https://public.ccsds.org/default.aspx
///
/// For code parameters see the [`CodeParams`](struct.CodeParams.html) structs also in this module:
/// [`TC128_PARAMS`](constant.TC128_PARAMS.html) etc.
#[derive(Copy,Clone,Debug,Eq,PartialEq,Hash)]
pub enum LDPCCode {
    /// n=128 k=64 r=1/2
    TC128,

    /// n=256 k=128 r=1/2
    TC256,

    /// n=512 k=256 r=1/2
    TC512,

    /// n=1280 k=1024 r=4/5
    TM1280,

    /// n=1536 k=1024 r=2/3
    TM1536,

    /// n=2048 k=1024 r=1/2
    TM2048,

    /// n=5120 k=4096 r=4/5
    TM5120,

    /// n=6144 k=4096 r=2/3
    TM6144,

    /// n=8192 k=4096 r=1/2
    TM8192,
}

/// Parameters for a given LDPC code.
pub struct CodeParams {
    /// Block length (number of bits transmitted/received, aka code length).
    pub n: usize,

    /// Data length (number of bits of user information, aka code dimension).
    pub k: usize,

    /// Number of parity bits not transmitted.
    pub punctured_bits: usize,

    /// Sub-matrix size (used in parity check matrix construction).
    pub submatrix_size: usize,

    /// Circulant block size (used in generator matrix construction).
    pub circulant_size: usize,

    /// Sum of the parity check matrix (number of parity check edges).
    pub paritycheck_sum: u32,

    // Almost everything below here can probably vanish once const fn is available,
    // as they can all be represented as simple equations of the parameters above.

    /// Length of the working area required for the bit-flipping decoder.
    /// Equal to n+punctured_bits.
    pub decode_bf_working_len: usize,

    /// Length of the working area required for the message-passing decoder.
    /// Equal to 2*paritycheck_sum.
    pub decode_mp_working_len: usize,

    /// Length of output required from any decoder.
    /// Equal to (n+punctured_bits)/8.
    pub output_len: usize,
}

/// Code parameters for the TC128 code
pub const TC128_PARAMS: CodeParams = CodeParams {
    n: 128,
    k: 64,
    punctured_bits: 0,
    submatrix_size: 128/8,
    circulant_size: 128/8,
    paritycheck_sum: 512,

    decode_bf_working_len: 128 + 0,
    decode_mp_working_len: 2 * 512,
    output_len: 128/8,
};

/// Code parameters for the TC256 code
pub const TC256_PARAMS: CodeParams = CodeParams {
    n: 256,
    k: 128,
    punctured_bits: 0,
    submatrix_size: 256/8,
    circulant_size: 256/8,
    paritycheck_sum: 1024,

    decode_bf_working_len: 256 + 0,
    decode_mp_working_len: 2 * 1024,
    output_len: 256/8,
};

/// Code parameters for the TC512 code
pub const TC512_PARAMS: CodeParams = CodeParams {
    n: 512,
    k: 256,
    punctured_bits: 0,
    submatrix_size: 512/8,
    circulant_size: 512/8,
    paritycheck_sum: 2048,

    decode_bf_working_len: 512 + 0,
    decode_mp_working_len: 2 * 2048,
    output_len: 512/8,
};

/// Code parameters for the TM1280 code
pub const TM1280_PARAMS: CodeParams = CodeParams {
    n: 1280,
    k: 1024,
    punctured_bits: 128,
    submatrix_size: 128,
    circulant_size: 128/4,
    paritycheck_sum: 4992,

    decode_bf_working_len: 1280 + 128,
    decode_mp_working_len: 2 * 4992,
    output_len: (1280 + 128)/8,
};

/// Code parameters for the TM1536 code
pub const TM1536_PARAMS: CodeParams = CodeParams {
    n: 1536,
    k: 1024,
    punctured_bits: 256,
    submatrix_size: 256,
    circulant_size: 256/4,
    paritycheck_sum: 5888,

    decode_bf_working_len: 1536 + 256,
    decode_mp_working_len: 2 * 5888,
    output_len: (1536 + 256)/8,
};

/// Code parameters for the TM2048 code
pub const TM2048_PARAMS: CodeParams = CodeParams {
    n: 2048,
    k: 1024,
    punctured_bits: 512,
    submatrix_size: 512,
    circulant_size: 512/4,
    paritycheck_sum: 7680,

    decode_bf_working_len: 2048 + 512,
    decode_mp_working_len: 2 * 7680,
    output_len: (2048 + 512)/8,
};

/// Code parameters for the TM5120 code
pub const TM5120_PARAMS: CodeParams = CodeParams {
    n: 5120,
    k: 4096,
    punctured_bits: 512,
    submatrix_size: 512,
    circulant_size: 512/4,
    paritycheck_sum: 19968,

    decode_bf_working_len: 5120 + 512,
    decode_mp_working_len: 2 * 19968,
    output_len: (5120 + 512)/8,
};

/// Code parameters for the TM6144 code
pub const TM6144_PARAMS: CodeParams = CodeParams {
    n: 6144,
    k: 4096,
    punctured_bits: 1024,
    submatrix_size: 1024,
    circulant_size: 1024/4,
    paritycheck_sum: 23552,

    decode_bf_working_len: 6144 + 1024,
    decode_mp_working_len: 2 * 23552,
    output_len: (6144 + 1024)/8,
};

/// Code parameters for the TM8192 code
pub const TM8192_PARAMS: CodeParams = CodeParams {
    n: 8192,
    k: 4096,
    punctured_bits: 2048,
    submatrix_size: 2048,
    circulant_size: 2048/4,
    paritycheck_sum: 30720,

    decode_bf_working_len: 8192 + 2048,
    decode_mp_working_len: 2 * 30720,
    output_len: (8192 + 2048)/8,
};

pub struct ParityIter {
    phi: &'static [[u16; 26]; 4],
    prototype: &'static [[[u8; 11]; 4]],
    prototype_rows_sub1: usize,
    prototype_cols_sub1: usize,
    prototype_subs_sub1: usize,
    m: usize,
    divm: usize,
    modm: usize,
    modmd4: usize,
    rowidx: usize,
    colidx: usize,
    sub_mat_idx: usize,
    sub_mat: u8,
    check: usize,
}

impl Iterator for ParityIter {
    type Item = (usize, usize);

    #[inline]
    fn next(&mut self) -> Option<(usize, usize)> {
        use self::compact_parity_checks::{HI, HP, THETA_K};
        // Loop over rows of the prototype
        loop {
            // Loop over columns of the prototype
            loop {
                // Loop over the three sub-prototypes we have to sum for each cell of the prototype
                loop {
                    // Identity matrix with a right-shift
                    if self.sub_mat & HI == HI && self.check < self.m {
                        let rot = (self.sub_mat & 0x3F) as usize;
                        let chk = self.rowidx * self.m + self.check;
                        let var = if rot == 0 {
                            self.colidx * self.m + self.check
                        } else {
                            self.colidx * self.m + ((self.check + rot) & self.modm)
                        };
                        self.check += 1;
                        return Some((chk, var));
                    }

                    // Permutation matrix using theta and phi lookup tables
                    if self.sub_mat & HP == HP && self.check < self.m {
                        let k = (self.sub_mat & 0x3F) as usize;
                        let chk = self.rowidx * self.m + self.check;
                        let pi = self.m/4
                                  * ((THETA_K[k-1] as usize + ((4*self.check)>>self.divm)) % 4)
                                 + ((self.phi[(4*self.check)>>self.divm][k-1] as usize
                                     + self.check) & self.modmd4);
                        let var = self.colidx * self.m + pi;
                        self.check += 1;
                        return Some((chk, var));
                    }

                    self.check = 0;

                    // Advance which of the three sub-matrices we're summing.
                    // If sub_mat is 0, there won't be any new ones to sum, so stop there too.
                    if self.sub_mat != 0 && self.sub_mat_idx < self.prototype_subs_sub1 {
                        self.sub_mat_idx += 1;
                        self.sub_mat = self.prototype[self.sub_mat_idx][self.rowidx][self.colidx];
                    } else {
                        self.sub_mat_idx = 0;
                        break;
                    }
                }

                // Advance colidx. The number of active columns depends on the prototype.
                if self.colidx < self.prototype_cols_sub1 {
                    self.colidx += 1;
                    self.sub_mat = self.prototype[self.sub_mat_idx][self.rowidx][self.colidx];
                } else {
                    self.colidx = 0;
                    break;
                }
            }

            // Advance rowidx. The number of rows depends on the prototype.
            if self.rowidx < self.prototype_rows_sub1 {
                self.rowidx += 1;
                self.sub_mat = self.prototype[self.sub_mat_idx][self.rowidx][self.colidx];
            } else {
                return None;
            }
        }
    }
}

impl LDPCCode {
    /// Get the code parameters for a specific LDPC code
    pub fn params(&self) -> CodeParams {
        match *self {
            LDPCCode::TC128  => TC128_PARAMS,
            LDPCCode::TC256  => TC256_PARAMS,
            LDPCCode::TC512  => TC512_PARAMS,
            LDPCCode::TM1280 => TM1280_PARAMS,
            LDPCCode::TM1536 => TM1536_PARAMS,
            LDPCCode::TM2048 => TM2048_PARAMS,
            LDPCCode::TM5120 => TM5120_PARAMS,
            LDPCCode::TM6144 => TM6144_PARAMS,
            LDPCCode::TM8192 => TM8192_PARAMS,
        }
    }

    /// Get the code length (number of codeword bits)
    pub fn n(&self) -> usize {
        self.params().n
    }

    /// Get the code dimension (number of information bits)
    pub fn k(&self) -> usize {
        self.params().k
    }

    /// Get the number of punctured bits (parity bits not transmitted)
    pub fn punctured_bits(&self) -> usize {
        self.params().punctured_bits
    }

    /// Get the size of the sub-matrices used to define the parity check matrix
    pub fn submatrix_size(&self) -> usize {
        self.params().submatrix_size
    }

    /// Get the size of the sub-matrices used to define the generator matrix
    pub fn circulant_size(&self) -> usize {
        self.params().circulant_size
    }

    /// Get the sum of the parity check matrix (total number of parity check edges)
    pub fn paritycheck_sum(&self) -> u32 {
        self.params().paritycheck_sum
    }

    /// Get the reference to the compact generator matrix for this code
    pub fn compact_generator(&self) -> &'static [u64] {
        match *self {
            LDPCCode::TC128  => &compact_generators::TC128_G,
            LDPCCode::TC256  => &compact_generators::TC256_G,
            LDPCCode::TC512  => &compact_generators::TC512_G,
            LDPCCode::TM1280 => &compact_generators::TM1280_G,
            LDPCCode::TM1536 => &compact_generators::TM1536_G,
            LDPCCode::TM2048 => &compact_generators::TM2048_G,
            LDPCCode::TM5120 => &compact_generators::TM5120_G,
            LDPCCode::TM6144 => &compact_generators::TM6144_G,
            LDPCCode::TM8192 => &compact_generators::TM8192_G,
        }
    }

    pub fn iter_paritychecks(&self) -> ParityIter {
        match *self {
            LDPCCode::TC128  | LDPCCode::TC256  | LDPCCode::TC512 => self.iter_paritychecks_tc(),
            LDPCCode::TM1280 | LDPCCode::TM1536 | LDPCCode::TM2048 |
            LDPCCode::TM5120 | LDPCCode::TM6144 | LDPCCode::TM8192 => self.iter_paritychecks_tm(),
        }
    }

    fn iter_paritychecks_tc(&self) -> ParityIter {
        let prototype = match *self {
            LDPCCode::TC128 => &compact_parity_checks::TC128_H,
            LDPCCode::TC256 => &compact_parity_checks::TC256_H,
            LDPCCode::TC512 => &compact_parity_checks::TC512_H,
            // This function is only called with TC codes.
            _               => unreachable!(),
        };

        let m = self.submatrix_size();

        ParityIter {
            phi: &self::compact_parity_checks::PHI_J_K_M128, prototype,
            prototype_cols_sub1: 8-1, prototype_rows_sub1: 4-1, prototype_subs_sub1: 2-1,
            m, divm: m.trailing_zeros() as usize, modm: m-1, modmd4: (m/4)-1,
            rowidx: 0, colidx: 0, sub_mat_idx: 0, sub_mat: prototype[0][0][0], check: 0,
        }
    }

    fn iter_paritychecks_tm(&self) -> ParityIter {
        let m = self.submatrix_size();
        let phi = match m {
            128  => &self::compact_parity_checks::PHI_J_K_M128,
            256  => &self::compact_parity_checks::PHI_J_K_M256,
            512  => &self::compact_parity_checks::PHI_J_K_M512,
            1024 => &self::compact_parity_checks::PHI_J_K_M1024,
            2048 => &self::compact_parity_checks::PHI_J_K_M2048,
            4096 => &self::compact_parity_checks::PHI_J_K_M4096,
            8192 => &self::compact_parity_checks::PHI_J_K_M8192,
            _    => unreachable!(),
        };

        let prototype_cols = (self.n() + self.punctured_bits()) / m;
        let prototype = match prototype_cols {
            5  => &self::compact_parity_checks::TM_R12_H,
            7  => &self::compact_parity_checks::TM_R23_H,
            11 => &self::compact_parity_checks::TM_R45_H,
            _  => unreachable!(),
        };

        ParityIter {
            phi, prototype, prototype_cols_sub1: prototype_cols - 1,
            prototype_rows_sub1: 3-1, prototype_subs_sub1: 3-1,
            m, divm: m.trailing_zeros() as usize, modm: m-1, modmd4: (m/4)-1,
            rowidx: 0, colidx: 0, sub_mat_idx: 0, check: 0, sub_mat: prototype[0][0][0],
        }
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::v1::*;

    use super::{LDPCCode, CodeParams,
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

    fn crc32_u16(crc: u32, data: u32) -> u32 {
        let mut crc = crc ^ data;
        for _ in 0..16 {
            let mask = if crc & 1 == 0 { 0 } else { 0xFFFFFFFFu32 };
            crc = (crc >> 1) ^ (0xEDB88320 & mask);
        }
        crc
    }

    #[test]
    fn test_iter_parity() {
        // These CRC results have been manually verified and should only change if
        // the ordering of checks returned from the iterator changes.
        let crc_results = [0x13A9D28D, 0xC3CC7625, 0x66EA9A48,
                           0xB643C99E, 0x8169E0CF, 0x599A0807,
                           0xD0E794B1, 0xBD0AB764, 0x9003014C];
        for (idx, code) in CODES.iter().enumerate() {
            let mut count = 0;
            let mut crc = 0xFFFFFFFFu32;
            for (check, var) in code.iter_paritychecks() {
                count += 1;
                crc = crc32_u16(crc, check as u32);
                crc = crc32_u16(crc, var as u32);
            }
            assert_eq!(count, code.paritycheck_sum() as usize);
            assert_eq!(crc, crc_results[idx]);
        }
    }
}
