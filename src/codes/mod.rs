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

    // Not yet included due to complexity of computing the compact generator matrix.
    // To be included in the future.
    // /// n=20480 k=16384 r=4/5
    //TM20480,

    // /// n=24576 k=16384 r=2/3
    //TM24576,

    // /// n=32768 k=16384 r=1/2
    //TM32768,
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

    /// Length of the sparse parity check ci array in u16. Equal to paritycheck_sum.
    pub sparse_paritycheck_ci_len: usize,

    /// Length of the sparse parity check cs array in u16. Equal to n-k+punctured_bits+1.
    pub sparse_paritycheck_cs_len: usize,

    /// Length of the sparse parity check vi array in u16. Equal to paritycheck_sum.
    pub sparse_paritycheck_vi_len: usize,

    /// Length of the sparse parity check vs array in u16. Equal to n+punctured_bits+1.
    pub sparse_paritycheck_vs_len: usize,

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

    sparse_paritycheck_ci_len: 512,
    sparse_paritycheck_cs_len: 128 - 64 + 0 + 1,
    sparse_paritycheck_vi_len: 512,
    sparse_paritycheck_vs_len: 128 + 0 + 1,
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

    sparse_paritycheck_ci_len: 1024,
    sparse_paritycheck_cs_len: 256 - 128 + 0 + 1,
    sparse_paritycheck_vi_len: 1024,
    sparse_paritycheck_vs_len: 256 + 0 + 1,
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

    sparse_paritycheck_ci_len: 2048,
    sparse_paritycheck_cs_len: 512 - 256 + 0 + 1,
    sparse_paritycheck_vi_len: 2048,
    sparse_paritycheck_vs_len: 512 + 0 + 1,
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

    sparse_paritycheck_ci_len: 4992,
    sparse_paritycheck_cs_len: 1280 - 1024 + 128 + 1,
    sparse_paritycheck_vi_len: 4992,
    sparse_paritycheck_vs_len: 1280 + 128 + 1,
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

    sparse_paritycheck_ci_len: 5888,
    sparse_paritycheck_cs_len: 1536 - 1024 + 256 + 1,
    sparse_paritycheck_vi_len: 5888,
    sparse_paritycheck_vs_len: 1536 + 256 + 1,
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

    sparse_paritycheck_ci_len: 7680,
    sparse_paritycheck_cs_len: 2048 - 1024 + 512 + 1,
    sparse_paritycheck_vi_len: 7680,
    sparse_paritycheck_vs_len: 2048 + 512 + 1,
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

    sparse_paritycheck_ci_len: 19968,
    sparse_paritycheck_cs_len: 5120 - 4096 + 512 + 1,
    sparse_paritycheck_vi_len: 19968,
    sparse_paritycheck_vs_len: 5120 + 512 + 1,
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

    sparse_paritycheck_ci_len: 23552,
    sparse_paritycheck_cs_len: 6144 - 4096 + 1024 + 1,
    sparse_paritycheck_vi_len: 23552,
    sparse_paritycheck_vs_len: 6144 + 1024 + 1,
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

    sparse_paritycheck_ci_len: 30720,
    sparse_paritycheck_cs_len: 8192 - 4096 + 2048 + 1,
    sparse_paritycheck_vi_len: 30720,
    sparse_paritycheck_vs_len: 8192 + 2048 + 1,
    decode_bf_working_len: 8192 + 2048,
    decode_mp_working_len: 2 * 30720,
    output_len: (8192 + 2048)/8,
};

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

    /// Get the length of [u16] required for the sparse parity check ci array.
    ///
    /// Equal to paritycheck_sum.
    pub fn sparse_paritycheck_ci_len(&self) -> usize {
        self.paritycheck_sum() as usize
    }

    /// Get the length of [u16] required for the sparse parity check cs array.
    ///
    /// Equal to n - k + punctured_bits + 1.
    pub fn sparse_paritycheck_cs_len(&self) -> usize {
        self.n() - self.k() + self.punctured_bits() + 1
    }

    /// Get the length of [u16] required for the sparse parity check vi array.
    ///
    /// Equal to paritycheck_sum.
    pub fn sparse_paritycheck_vi_len(&self) -> usize {
        self.paritycheck_sum() as usize
    }

    /// Get the length of [u16] required for the sparse parity check vs array.
    ///
    /// Equal to n + punctured_bits + 1.
    pub fn sparse_paritycheck_vs_len(&self) -> usize {
        self.n() + self.punctured_bits() + 1
    }

    /// Initialises the sparse representation of the parity check matrix.
    ///
    /// The sparse representation consists of four arrays:
    ///
    /// * `ci` and `vi` contain the indices of the non-zero entries along each row (check nodes)
    ///   and column (variable nodes) of the full parity check matrix, allowing iteration through
    ///   the parity matrix connections from check to variable node or from variable to check node.
    /// * `cs` and `vs` contain the offset into `ci` and `vi` for a given check or column index.
    ///   This means the list of indices of all variable nodes involved in check i starts at
    ///   `ci[cs[i]]` and it ends at `ci[cs[i+1]]`. `cs` has (n-k+p+1) entries, while `vs` has
    ///   (n+p+1) entries, in both cases with the final entry set to the length of `ci` and `vi`
    ///   respectively.
    ///
    /// The references to `ci`, `cs`, `vi`, and `vs` must all be preallocated to the correct size,
    /// available as a `const` in `CodeParams.sparse_paritycheck_ci_len` etc, and at runtime
    /// from `LDPCCode.sparse_paritycheck_ci_len()` etc.
    ///
    /// ## Panics
    /// * `ci.len()` must be exactly `self.sparse_paritycheck_ci_len()`.
    /// * `cs.len()` must be exactly `self.sparse_paritycheck_cs_len()`.
    /// * `vi.len()` must be exactly `self.sparse_paritycheck_vi_len()`.
    /// * `vs.len()` must be exactly `self.sparse_paritycheck_vs_len()`.
    pub fn init_sparse_paritycheck(&self, ci: &mut [u16], cs: &mut [u16],
                                   vi: &mut [u16], vs: &mut [u16])
    {
        assert_eq!(ci.len(), self.sparse_paritycheck_ci_len());
        assert_eq!(cs.len(), self.sparse_paritycheck_cs_len());
        assert_eq!(vi.len(), self.sparse_paritycheck_vi_len());
        assert_eq!(vs.len(), self.sparse_paritycheck_vs_len());

        self.init_sparse_paritycheck_checks(ci, cs);
        self.init_sparse_paritycheck_variables(ci, cs, vi, vs);
    }

    /// Initialises just the checks (`ci` and `cs`) in the sparse representation of the parity
    /// check matrix, useful for the bit flipping decoder which does not need `vi` or `vs`.
    ///
    /// See `init_sparse_paritycheck` for further details.
    ///
    /// ## Panics
    /// * `ci.len()` must be exactly `self.sparse_paritycheck_ci_len()`.
    /// * `cs.len()` must be exactly `self.sparse_paritycheck_cs_len()`.
    pub fn init_sparse_paritycheck_checks(&self, ci: &mut [u16], cs: &mut[u16]) {
        assert_eq!(ci.len(), self.sparse_paritycheck_ci_len());
        assert_eq!(cs.len(), self.sparse_paritycheck_cs_len());

        match *self {
            LDPCCode::TC128  | LDPCCode::TC256  | LDPCCode::TC512  =>
                self.init_sparse_paritycheck_checks_tc(ci, cs),
            LDPCCode::TM1280 | LDPCCode::TM1536 | LDPCCode::TM2048 |
            LDPCCode::TM5120 | LDPCCode::TM6144 | LDPCCode::TM8192 =>
                self.init_sparse_paritycheck_checks_tm(ci, cs),
        }
    }

    /// Initialises just the variables (`vi` and `vs`) in the sparse representation of the parity
    /// check matrix. Requires that the checks `ci` and `cs` have already been initialised.
    ///
    /// See `init_sparse_paritycheck` for further details.
    ///
    /// ## Panics
    /// * `ci.len()` must be exactly `self.sparse_paritycheck_ci_len()`.
    /// * `cs.len()` must be exactly `self.sparse_paritycheck_cs_len()`.
    /// * `vi.len()` must be exactly `self.sparse_paritycheck_vi_len()`.
    /// * `vs.len()` must be exactly `self.sparse_paritycheck_vs_len()`.
    pub fn init_sparse_paritycheck_variables(&self, ci: &[u16], cs: &[u16],
                                             vi: &mut[u16], vs: &mut[u16])
    {
        assert_eq!(ci.len(), self.sparse_paritycheck_ci_len());
        assert_eq!(cs.len(), self.sparse_paritycheck_cs_len());
        assert_eq!(vi.len(), self.sparse_paritycheck_vi_len());
        assert_eq!(vs.len(), self.sparse_paritycheck_vs_len());

        let n = self.n();
        let p = self.punctured_bits();

        let mut vi_idx = 0usize;

        // For each variable of the full parity check matrix (0..n+p)
        for (variable, vs_variable) in vs.iter_mut().take(n+p).enumerate() {
            // Record the starting index for this check
            *vs_variable = vi_idx as u16;

            // For each (start, stop) pair in cs,
            // aka each check (or row) of the parity check matrix, 0 through n-k+p
            for (check, cs_ss) in cs.windows(2).enumerate() {
                // Go through each variable this check is connected to
                for ci_variable in ci[cs_ss[0] as usize .. cs_ss[1] as usize].iter() {
                    // If we see ourselves in this row's connections, then
                    // this check should be listed against our variable
                    if *ci_variable as usize == variable {
                        vi[vi_idx] = check as u16;
                        vi_idx += 1;
                    }
                }
            }
        }

        vs[n+p] = vi_idx as u16;
    }

    /// Initialise sparse check nodes (`ci` and `cs`) for TC codes.
    fn init_sparse_paritycheck_checks_tc(&self, ci: &mut [u16], cs: &mut [u16]) {
        use self::compact_parity_checks::{HI, HP};

        assert_eq!(ci.len(), self.sparse_paritycheck_ci_len());
        assert_eq!(cs.len(), self.sparse_paritycheck_cs_len());

        let n = self.n();
        let k = self.k();
        let m = self.submatrix_size();

        assert!(m.is_power_of_two());

        // Compiler doesn't know m is a power of two, so we'll work out the mask for %
        // to save it having to do expensive division operations
        let modm = m - 1;
        let divm = m.trailing_zeros();

        let prototype = match *self {
            LDPCCode::TC128 => compact_parity_checks::TC128_H,
            LDPCCode::TC256 => compact_parity_checks::TC256_H,
            LDPCCode::TC512 => compact_parity_checks::TC512_H,
            // This function is only called with TC codes.
            _               => unreachable!(),
        };

        let mut ci_idx = 0;

        // For each check in the full parity check matrix (each row, 0..(n-k))
        for (check, cs_check) in cs.iter_mut().take(n-k).enumerate() {
            // Index of the sub-matrix for this check
            let check_block = check >> divm;
            // Check number inside this block
            let block_check = check & modm;

            // Record the start index of this check
            *cs_check = ci_idx;

            // For each variable of the full parity check matrix (each column)
            for variable in 0..n {
                // Index of the sub-matrix for this variable
                let variable_block = variable >> divm;
                // variableumn number inside this block
                let block_variable = variable & modm;

                // Take the relevant prototype entry and extract its rotation
                let subm = prototype[check_block][variable_block];
                let rot = (subm & 0x3F) as usize;

                // For the identity matrix just check if j==i.
                if subm & HI == HI && block_variable == block_check {
                    ci[ci_idx as usize] = variable as u16;
                    ci_idx += 1;
                }

                // Rotated identity matrix. Check if j==(i+r)%m.
                if subm & HP == HP && block_variable == (block_check + rot) & modm {
                    ci[ci_idx as usize] = variable as u16;
                    ci_idx += 1;
                }
            }

        }

        // Record the final entry.
        cs[n - k] = ci_idx;
    }

    /// Initialise sparse check nodes (`ci` and `cs`) for TM codes.
    fn init_sparse_paritycheck_checks_tm(&self, ci: &mut [u16], cs: &mut [u16]) {
        use self::compact_parity_checks::{HI, HP, TM_R12_H, TM_R23_H, TM_R45_H};

        assert_eq!(ci.len(), self.sparse_paritycheck_ci_len());
        assert_eq!(cs.len(), self.sparse_paritycheck_cs_len());

        let mut ci_idx = 0;

        let n = self.n();
        let k = self.k();
        let m = self.submatrix_size();
        let p = self.punctured_bits();

        assert!(m.is_power_of_two());

        // Compiler doesn't know m is a power of two, so we'll work out the mask for %
        // to save it having to do expensive division operations
        let modm = m - 1;
        let divm = m.trailing_zeros();
        let modmd4 = (m/4) - 1;

        // Fetch whichever phi lookup table is appropriate for our M
        let phi_j_k = match m {
            128  => &self::compact_parity_checks::PHI_J_K_M128,
            256  => &self::compact_parity_checks::PHI_J_K_M256,
            512  => &self::compact_parity_checks::PHI_J_K_M512,
            1024 => &self::compact_parity_checks::PHI_J_K_M1024,
            2048 => &self::compact_parity_checks::PHI_J_K_M2048,
            4096 => &self::compact_parity_checks::PHI_J_K_M4096,
            8192 => &self::compact_parity_checks::PHI_J_K_M8192,
            _    => unreachable!(),
        };
        let theta_k = &self::compact_parity_checks::THETA_K;

        // For each check in the full parity check matrix (each row)
        for (check, cs_check) in cs.iter_mut().take(n-k+p).enumerate() {
            // Check number inside this block
            let block_check = check & modm;

            // Record the start index of this check
            *cs_check = ci_idx;

            // For each block (submatrix) in the prototype matrix row
            for variable_block in 0..((n+p)/m) {

                // Determine which prototype to sum for this check and variable
                // (the variable_block_offset is used to shift the prototype to the left)
                let (prototype, variable_block_offset) = match (n + p) / m {
                    5  =>                              (&TM_R12_H, 0),
                    7  => if variable_block < 2 {      (&TM_R23_H, 0) }
                          else {                       (&TM_R12_H, 2) },
                    11 => if variable_block < 4 {      (&TM_R45_H, 0) }
                          else if variable_block < 6 { (&TM_R23_H, 4) }
                          else {                       (&TM_R12_H, 6) },
                    _  => unreachable!(),
                };

                // For each variable node in this prototype block row
                for block_variable in 0..m {

                    // For each of those prototype entries, work out whether their parity bit is
                    // set for this check and variable, and sum over the three sub matrices.
                    let mut pbit = 0;
                    for proto in prototype {
                        let subm = proto[check >> divm][variable_block - variable_block_offset];
                        if subm == 0 {
                            // After a 0 we won't find anything further, so can stop here
                            break;
                        } else if subm & HI == HI {
                            // Identity matrix is simple, just check block_variable==block_check
                            if block_variable == block_check {
                                pbit ^= 1;
                            }
                        } else if subm & HP == HP {
                            // Permutation submatrix:
                            // Extract k from lower bits
                            let k = (subm & 0x3F) as usize;
                            // Compute pi(i)
                            let pi_i = m/4
                                       * ((theta_k[k-1] as usize + ((4*block_check)>>divm)) % 4)
                                       + ((phi_j_k[(4*block_check)>>divm][k-1] as usize
                                           + block_check)
                                          & modmd4);
                            if block_variable == pi_i {
                                pbit ^= 1;
                            }
                        }
                    }

                    // If the parity bit ends up set, record this variable against this check.
                    if pbit == 1 {
                        ci[ci_idx as usize] = ((variable_block * m) + block_variable) as u16;
                        ci_idx += 1;
                    }
                }
            }
        }

        // Record the final entry.
        cs[n - k + p] = ci_idx;
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

    fn crc32_u16(data: &[u16]) -> u32 {
        let mut crc = 0xFFFFFFFFu32;
        for x in data {
            crc ^= *x as u32;
            for _ in 0..16 {
                let mask = if crc & 1 == 0 { 0 } else { 0xFFFFFFFFu32 };
                crc = (crc >> 1) ^ (0xEDB88320 & mask);
            }
        }
        !crc
    }

    #[test]
    fn test_sparse_paritycheck_len() {
        for (code, param) in CODES.iter().zip(PARAMS.iter()) {
            assert_eq!(code.sparse_paritycheck_ci_len(), param.sparse_paritycheck_ci_len);
            assert_eq!(code.sparse_paritycheck_cs_len(), param.sparse_paritycheck_cs_len);
            assert_eq!(code.sparse_paritycheck_vi_len(), param.sparse_paritycheck_vi_len);
            assert_eq!(code.sparse_paritycheck_vs_len(), param.sparse_paritycheck_vs_len);
        }
    }

    #[test]
    fn test_sparse_paritycheck() {
        let mut crc_results: Vec<(u32, u32, u32, u32)> = Vec::new();
        for code in CODES[..6].iter() {
            let mut ci = vec![0; code.sparse_paritycheck_ci_len()];
            let mut cs = vec![0; code.sparse_paritycheck_cs_len()];
            let mut vi = vec![0; code.sparse_paritycheck_vi_len()];
            let mut vs = vec![0; code.sparse_paritycheck_vs_len()];
            code.init_sparse_paritycheck(&mut ci, &mut cs, &mut vi, &mut vs);
            crc_results.push((crc32_u16(&ci), crc32_u16(&cs), crc32_u16(&vi), crc32_u16(&vs)));
        }

        // These sets of CRC32s are known good results from the original C implementation
        assert_eq!(crc_results, vec![
            (0xB7E800BD, 0x6C4C3709, 0xEACD656A, 0x41998815),
            (0x90C64BFC, 0x9D4CF128, 0x8B4E54F1, 0x3A21F54D),
            (0xE7135070, 0xA87336D5, 0x071B76FF, 0x80992086),
            (0x07699182, 0xF5386F36, 0x3951ACFF, 0x2C89D420),
            (0x6DFECCF6, 0xE3AC8063, 0xDC800AEB, 0xD737D4FD),
            (0x6805D4C6, 0x5F00D915, 0x4139AA3E, 0xE7FDABD1),
        ]);
    }

    #[test]
    #[ignore]
    fn test_sparse_paritycheck_slow() {
        let mut crc_results: Vec<(u32, u32, u32, u32)> = Vec::new();
        for code in CODES[6..].iter() {
            let mut ci = vec![0; code.sparse_paritycheck_ci_len()];
            let mut cs = vec![0; code.sparse_paritycheck_cs_len()];
            let mut vi = vec![0; code.sparse_paritycheck_vi_len()];
            let mut vs = vec![0; code.sparse_paritycheck_vs_len()];
            code.init_sparse_paritycheck(&mut ci, &mut cs, &mut vi, &mut vs);
            crc_results.push((crc32_u16(&ci), crc32_u16(&cs), crc32_u16(&vi), crc32_u16(&vs)));
        }

        // These CRC32s were originally generated by this program so only check consistency.
        assert_eq!(crc_results, vec![
            (0xE80235D1, 0x32250FDF, 0xDB9A2980, 0xB750D9CA),
            (0xF4539510, 0x6A88E342, 0xDC592FC2, 0x73046340),
            (0x4EF927D2, 0x8EBFC56C, 0x49BD9D35, 0x2C840D3B),
        ]);
    }
}
