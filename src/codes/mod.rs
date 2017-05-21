//! This module contains the available LDPC codes, and the associated constants and methods to
//! load their generator and parity check matrices.

/// This module contains the constants representing the generator matrices.
///
/// They are in a compact form: for each systematic generator matrix, we take just the
/// parity bits (the n-k columns on the right), and then we take just the first row for
/// each circulant (i.e. instead of k rows, we take k/circulant_size rows), and then
/// pack the bits for that row into `u32`s.
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

/// Available LDPC codes.
///
/// The TC codes are the Telecommand codes from CCSDS document 231.1-O-1.
/// The TM codes are the Telemetry codes from CCSDS document 131.0-B-2.
/// https://public.ccsds.org/default.aspx
///
/// For code parameters see the const CodeParams structs later in this module: `TC128_PARAMS` etc.
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

pub static CODES: [LDPCCode;  9] = [LDPCCode::TC128,   LDPCCode::TC256,   LDPCCode::TC512,
                                    LDPCCode::TM1280,  LDPCCode::TM1536,  LDPCCode::TM2048,
                                    LDPCCode::TM5120,  LDPCCode::TM6144,  LDPCCode::TM8192,
                                    //LDPCCode::TM20480, LDPCCode::TM24576, LDPCCode::TM32768,
];

/// Parameters for a given LDPC code.
///
/// * `n` is the block length (number of bits transmitted/received)
/// * `k` is the data length (number of bits of user information)
/// * `punctured_bits` is the number of parity bits not transmitted
/// * `submatrix_size` is the sub-matrix size (used in code construction)
/// * `circulant_size` is the circulant block size (used in code construction)
/// * `parity_sum` is the sum of the H matrix (number of parity check edges)
pub struct CodeParams {
    pub code: LDPCCode,
    pub n: usize,
    pub k: usize,
    pub punctured_bits: usize,
    pub submatrix_size: usize,
    pub circulant_size: usize,
    pub parity_sum: u32,

    compact_generator: &'static [u32],
}

/// Code parameters for the TC128 code
pub static TC128_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TC128,
    n: 128,
    k: 64,
    punctured_bits: 0,
    submatrix_size: 128/8,
    circulant_size: 128/8,
    parity_sum: 512,

    compact_generator: &compact_generators::TC128_G,
};

/// Code parameters for the TC256 code
pub static TC256_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TC256,
    n: 256,
    k: 128,
    punctured_bits: 0,
    submatrix_size: 256/8,
    circulant_size: 256/8,
    parity_sum: 1024,

    compact_generator: &compact_generators::TC256_G,
};

/// Code parameters for the TC512 code
pub static TC512_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TC512,
    n: 512,
    k: 256,
    punctured_bits: 0,
    submatrix_size: 512/8,
    circulant_size: 512/8,
    parity_sum: 2048,

    compact_generator: &compact_generators::TC512_G,
};

/// Code parameters for the TM1280 code
pub static TM1280_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TM1280,
    n: 1280,
    k: 1024,
    punctured_bits: 128,
    submatrix_size: 128,
    circulant_size: 128/4,
    parity_sum: 4992,

    compact_generator: &compact_generators::TM1280_G,
};

/// Code parameters for the TM1536 code
pub static TM1536_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TM1536,
    n: 1536,
    k: 1024,
    punctured_bits: 256,
    submatrix_size: 256,
    circulant_size: 256/4,
    parity_sum: 5888,

    compact_generator: &compact_generators::TM1536_G,
};

/// Code parameters for the TM2048 code
pub static TM2048_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TM2048,
    n: 2048,
    k: 1024,
    punctured_bits: 512,
    submatrix_size: 512,
    circulant_size: 512/4,
    parity_sum: 7680,

    compact_generator: &compact_generators::TM2048_G,
};

/// Code parameters for the TM5120 code
pub static TM5120_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TM5120,
    n: 5120,
    k: 4096,
    punctured_bits: 512,
    submatrix_size: 512,
    circulant_size: 512/4,
    parity_sum: 0,

    compact_generator: &compact_generators::TM5120_G,
};

/// Code parameters for the TM6144 code
pub static TM6144_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TM6144,
    n: 6144,
    k: 4096,
    punctured_bits: 1024,
    submatrix_size: 1024,
    circulant_size: 1024/4,
    parity_sum: 0,

    compact_generator: &compact_generators::TM6144_G,
};

/// Code parameters for the TM8192 code
pub static TM8192_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TM8192,
    n: 8192,
    k: 4096,
    punctured_bits: 2048,
    submatrix_size: 2048,
    circulant_size: 2048/4,
    parity_sum: 0,

    compact_generator: &compact_generators::TM8192_G,
};

/*
 * Not yet included. See comment in the LDPCCode definition above.

/// Code parameters for the TM20480 code
pub static TM20480_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TM20480,
    n: 20480,
    k: 16384,
    punctured_bits: 2048,
    submatrix_size: 2048
    circulant_size: 2048/4,
    parity_sum: 0,

    compact_generator: &TM20480_G,
};

/// Code parameters for the TM24576 code
pub static TM24576_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TM24576,
    n: 24576,
    k: 16384,
    punctured_bits: 4096,
    submatrix_size: 4096
    circulant_size: 4096/4,
    parity_sum: 0,

    compact_generator: &TM24576_G,
};

/// Code parameters for the TM32768 code
pub static TM32768_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TM32768,
    n: 32768,
    k: 16384,
    punctured_bits: 8192,
    submatrix_size: 8192
    circulant_size: 8192/4,
    parity_sum: 0,

    compact_generator: &TM32768_G,
};

*/

impl LDPCCode {
    /// Get the code parameters for a specific LDPC code
    pub fn params(&self) -> &'static CodeParams {
        match *self {
            LDPCCode::TC128 => &TC128_PARAMS,
            LDPCCode::TC256 => &TC256_PARAMS,
            LDPCCode::TC512 => &TC512_PARAMS,
            LDPCCode::TM1280 => &TM1280_PARAMS,
            LDPCCode::TM1536 => &TM1536_PARAMS,
            LDPCCode::TM2048 => &TM2048_PARAMS,
            LDPCCode::TM5120 => &TM5120_PARAMS,
            LDPCCode::TM6144 => &TM6144_PARAMS,
            LDPCCode::TM8192 => &TM8192_PARAMS,
        }
    }

    /// Get the length in u8 required for the full generator matrix. Equal to k*(n-k) / 8.
    pub fn generator_len_u8(&self) -> usize {
        let params = self.params();
        (params.k * (params.n - params.k)) / 8
    }

    /// Get the length in u32 required for the full generator matrix. Equal to k*(n-k) / 32.
    pub fn generator_len_u32(&self) -> usize {
        self.generator_len_u8() / 4
    }

    /// Initialise a full generator matrix, expanded from the compact circulant form.
    ///
    /// This allows quicker encoding at the cost of higher memory usage.
    /// Note that this will only initialise the parity part of G, and not the
    /// identity matrix, since all supported codes are systematic. This matches
    /// what's expected by the non-compact encoder function.
    pub fn init_generator(&self, g: &mut [u32]) {
        assert_eq!(g.len(), self.generator_len_u32());

        let params = self.params();
        let gc = params.compact_generator;
        let b = params.circulant_size;
        let r = params.n - params.k;

        // For each block of the output matrix
        for (blockidx, block) in g.chunks_mut(b * r/32).enumerate() {
            // Copy the first row from the compact matrix
            block[..r/32].copy_from_slice(&gc[(blockidx  )*(r/32) .. (blockidx+1)*(r/32)]);

            // For each subsequent row, copy from the row above and then
            // rotate right by one.
            for rowidx in 1..b {
                let (prev_row, row) = block[(rowidx-1)*r/32 .. (rowidx+1)*r/32].split_at_mut(r/32);
                row.copy_from_slice(prev_row);

                // For each block in the row
                for rowblockidx in 0..r/b {
                    if b >= 32 {
                        // In the simpler case, blocks are at least one word.
                        // Just take the final bit as the initial carry, then
                        // move through rotating each word.
                        let rowblock = &mut row[(rowblockidx  )*(b/32) .. (rowblockidx+1)*(b/32)];
                        let mut carry = rowblock.last().unwrap() & 1;
                        for word in rowblock.iter_mut() {
                            let newcarry = *word & 1;
                            *word = (carry << 31) | (*word >> 1);
                            carry = newcarry;
                        }
                    } else if b == 16 {
                        // In the more annoying case, blocks are less than one
                        // word, so we'll have to rotate inside the words.
                        // So far this can only happen for 16-bit blocks, so
                        // we'll special case that.
                        let byteidx = rowblockidx * 2;
                        let shift = if byteidx % 4 == 0 { 16 } else { 0 };
                        let mut block = (row[byteidx/4] >> shift) & 0xFFFF;
                        block = ((block & 1) << 15) | (block >> 1);
                        row[byteidx/4] &= !(0xFFFF << shift);
                        row[byteidx/4] |= block << shift;
                    } else {
                        panic!();
                    }
                }
            }
        }

        // We'll be using this generator matrix by XORing with data interpreted
        // as u32, but that data will have been a u8 array, so the bytes will
        // be the wrong way around on little endian platforms.
        // Instead of fixing this at encode time, we can fix it once here.
        for x in g.iter_mut() {
            *x = x.to_be();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CODES;

    fn crc32(data: &[u32]) -> u32 {
        let mut crc = 0xFFFFFFFFu32;
        for x in data {
            crc ^= *x;
            for _ in 0..32 {
                let mask = if crc & 1 == 0 { 0 } else { 0xFFFFFFFFu32 };
                crc = (crc >> 1) ^ (0xEDB88320 & mask);
            }
        }
        !crc
    }

    #[test]
    fn test_generator_matrix() {
        let mut crc_results = Vec::new();
        for code in CODES.iter() {
            let mut g = vec![0; code.generator_len_u32()];
            code.init_generator(&mut g);
            crc_results.push(crc32(&g));
        }

        // The first six CRC32s are known good CRC32 results from the original C implementation,
        // the remainder were originally generated by this program so only check consistency.
        assert_eq!(crc_results, vec![0xDC64D486, 0xD78B5564, 0x6AF9EC6A,
                                     0x452FE118, 0xBCCBA8D0, 0x1597B6F6,
                                     0xab79c637, 0x450a2213, 0xdd3f049b]);

    }
}
