#![allow(dead_code)]

/// Available LDPC codes.
///
/// The TC codes are the Telecommand codes from CCSDS document 231.1-O-1.
/// The TM codes are the Telemetry codes from CCSDS document 131.0-B-2.
/// https://public.ccsds.org/default.aspx
pub enum LDPCCode {
    TC128,
    TC256,
    TC512,
}

/// Parameters for a given LDPC code.
///
/// `n` is the block length (number of bits transmitted/received).
/// `k` is the data length (number of bits of user information).
/// `punctured_bits` is the number of parity bits not transmitted
/// `submatrix_size` is the sub-matrix size (used in code construction)
/// `circulant_size` is the circulant block size (used in code construction)
/// `parity_sum` is the sum of the H matrix (number of parity check edges)
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

/// Compact generator matrix for the TC128 code
static TC128_G: [u32; 4 * 2] = [
    0x0E69166B, 0xEF4C0BC2, 0x7766137E, 0xBB248418,
    0xC480FEB9, 0xCD53A713, 0x4EAA22FA, 0x465EEA11,
];

/// Compact generator matrix for the TC256 code
static TC256_G: [u32; 4 * 4] = [
    0x73F5E839, 0x0220CE51, 0x36ED68E9, 0xF39EB162,
    0xBAC812C0, 0xBCD24379, 0x4786D928, 0x5A09095C,
    0x7DF83F76, 0xA5FF4C38, 0x8E6C0D4E, 0x025EB712,
    0xBAA37B32, 0x60CB31C5, 0xD0F66A31, 0xFAF511BC,
];

/// Compact generator matrix for the TC512 code
static TC512_G: [u32; 4 * 8] = [
    0x1D21794A, 0x22761FAE, 0x59945014, 0x257E130D,
    0x74D60540, 0x03794014, 0x2DADEB9C, 0xA25EF12E,
    0x60E0B662, 0x3C5CE512, 0x4D2C81EC, 0xC7F469AB,
    0x20678DBF, 0xB7523ECE, 0x2B54B906, 0xA9DBE98C,
    0xF6739BCF, 0x54273E77, 0x167BDA12, 0x0C6C4774,
    0x4C071EFF, 0x5E32A759, 0x3138670C, 0x095C39B5,
    0x28706BD0, 0x45300258, 0x2DAB85F0, 0x5B9201D0,
    0x8DFDEE2D, 0x9D84CA88, 0xB371FAE6, 0x3A4EB07E,
];

/// Code parameters for the TC128 code
pub static TC128_PARAMS: CodeParams = CodeParams {
    code: LDPCCode::TC128,
    n: 128,
    k: 64,
    punctured_bits: 0,
    submatrix_size: 128/8,
    circulant_size: 128/8,
    parity_sum: 512,

    compact_generator: &TC128_G,
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

    compact_generator: &TC256_G,
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

    compact_generator: &TC512_G,
};

/// Constants used to define the parity check matrices for the TC codes.
///
/// This representation mirrors that in CCSDS 231.1-O-1, and is expanded
/// at runtime to create sparse-encoded parity check matrices.
/// Each constant represents a single MxM sub-matrix, where M=n/8.
/// HZ: All-zero matrix
/// HI: Identity matrix
/// HP: Phi: nth right circular shift of I, with lower 5 bits for n
/// HS: HI+HP
const HZ: u8 = (0 << 6);
const HI: u8 = (1 << 6);
const HP: u8 = (2 << 6);
const HS: u8 = (HI | HP);

/// Compact parity matrix for the TC128 code
static TC128_H: [[u8; 8]; 4] = [
    [HS| 7, HP| 2, HP|14, HP| 6, HZ   , HP| 0, HP|13, HI   ],
    [HP| 6, HS|15, HP| 0, HP| 1, HI   , HZ   , HP| 0, HP| 7],
    [HP| 4, HP| 1, HS|15, HP|14, HP|11, HI   , HZ   , HP| 3],
    [HP| 0, HP| 1, HP| 9, HS|13, HP|14, HP| 1, HI   , HZ   ],
];

/// Compact parity matrix for the TC256 code
static TC256_H: [[u8; 8]; 4] = [
    [HS|31, HP|15, HP|25, HP| 0, HZ   , HP|20, HP|12, HI   ],
    [HP|28, HS|30, HP|29, HP|24, HI   , HZ   , HP| 1, HP|20],
    [HP| 8, HP| 0, HS|28, HP| 1, HP|29, HI   , HZ   , HP|21],
    [HP|18, HP|30, HP| 0, HS|30, HP|25, HP|26, HI   , HZ   ],
];

/// Compact parity matrix for the TC512 code
static TC512_H: [[u8; 8]; 4] = [
    [HS|63, HP|30, HP|50, HP|25, HZ   , HP|43, HP|62, HI   ],
    [HP|56, HS|61, HP|50, HP|23, HI   , HZ   , HP|37, HP|26],
    [HP|16, HP| 0, HS|55, HP|27, HP|56, HI   , HZ   , HP|43],
    [HP|35, HP|56, HP|62, HS|11, HP|58, HP| 3, HI   , HZ   ],
];

impl LDPCCode {
    /// Get the code parameters for a specific LDPC code
    pub fn params(&self) -> &'static CodeParams {
        match *self {
            LDPCCode::TC128 => &TC128_PARAMS,
            LDPCCode::TC256 => &TC256_PARAMS,
            LDPCCode::TC512 => &TC512_PARAMS,
        }
    }

    /// Get the size (in u8s) required for the full generator matrix.
    /// Equal to k*(n-k) / 8.
    pub fn generator_u8s(&self) -> usize {
        let params = self.params();
        (params.k * (params.n - params.k)) / 8
    }

    /// Get the length (in u32s) required for the full generator matrix.
    /// Equal to generator_size()/4.
    pub fn generator_u32s(&self) -> usize {
        self.generator_u8s() / 4
    }

    /// Initialise a full generator matrix, expanded from the compact circulant
    /// form.
    ///
    /// This allows quicker encoding at the cost of higher memory usage.
    /// Note that this will only initialise the parity part of G, and not the
    /// identity matrix, since all supported codes are systematic. This matches
    /// what's expected by the non-compact encoder function.
    pub fn init_generator(&self, g: &mut [u32]) {
        assert_eq!(g.len(), self.generator_u32s());

        let params = self.params();
        let gc = params.compact_generator;
        let b = params.circulant_size;
        let r = params.n - params.k;

        // For each block of the output matrix
        for (blockidx, block) in g.chunks_mut(b * r/32).enumerate() {
            // Copy the first row from the compact matrix
            block[..r/32].copy_from_slice(&gc[(blockidx  )*(r/32) ..
                                              (blockidx+1)*(r/32)]);

            // For each subsequent row, copy from the row above and then
            // rotate right by one.
            for rowidx in 1..b {
                let (prev_row, row) = block[(rowidx-1)*r/32 .. (rowidx+1)*r/32]
                                      .split_at_mut(r/32);
                row.copy_from_slice(prev_row);

                // For each block in the row
                for rowblockidx in 0..r/b {
                    if b >= 32 {
                        // In the simpler case, blocks are at least one word.
                        // Just take the final bit as the initial carry, then
                        // move through rotating each word.
                        let rowblock = &mut row[(rowblockidx  )*(b/32) ..
                                                (rowblockidx+1)*(b/32)];
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
    use super::{LDPCCode};

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
        let mut g128 = vec![0; LDPCCode::TC128.generator_u32s()];
        LDPCCode::TC128.init_generator(&mut g128);
        let crc_g128 = crc32(&g128);
        assert_eq!(crc_g128, 0xDC64D486);

        let mut g256 = vec![0; LDPCCode::TC256.generator_u32s()];
        LDPCCode::TC256.init_generator(&mut g256);
        let crc_g256 = crc32(&g256);
        assert_eq!(crc_g256, 0xD78B5564);

        let mut g512 = vec![0; LDPCCode::TC512.generator_u32s()];
        LDPCCode::TC512.init_generator(&mut g512);
        let crc_g512 = crc32(&g512);
        assert_eq!(crc_g512, 0x6AF9EC6A);
    }
}
