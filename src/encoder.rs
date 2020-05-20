// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

//! This module provides the encoding function for turning data into codewords.
//!
//! Please refer to the `encode` and `copy_encode` methods on
//! [`LDPCCode`](../codes/enum.LDPCCode.html) for more details.

// We have a couple of expressions with +0 for clarity of where the 0 comes from
#![allow(clippy::identity_op)]

use core::slice;

use crate::codes::LDPCCode;

/// Trait for the types of codeword we can encode into.
///
/// We implement this for u8 (the standard but slow option), and u32 and u64 which give speedups.
pub trait EncodeInto {
    /// Given `codeword` which has the first k bits set to the data to transmit,
    /// sets the remaining n-k parity bits.
    ///
    /// Returns a `&mut [u8]` view on `codeword`.
    fn encode<'a>(code: &LDPCCode, codeword: &'a mut[Self]) -> &'a mut [u8]
        where Self: Sized;

    /// First copies `data` into the first k bits of `codeword`, then calls `encode`.
    fn copy_encode<'a>(code: &LDPCCode, data: &[u8], codeword: &'a mut[Self]) -> &'a mut [u8]
        where Self: Sized;

    /// Returns the bit length for this type
    fn bitlength() -> usize;
}

impl EncodeInto for u8 {
    fn encode<'a>(code: &LDPCCode, codeword: &'a mut[Self]) -> &'a mut [u8] {
        let k = code.k();
        let r = code.n() - code.k();
        let b = code.circulant_size();
        let gc = code.compact_generator();
        let row_len = r/64;

        // Scope the split of codeword into (data, parity)
        {
            // Split codeword into data and parity sections and then zero the parity bits
            let (data, parity) = codeword.split_at_mut(k / 8);
            for x in parity.iter_mut() { *x = 0; }

            // For each rotation of the generator circulants
            for offset in 0..b {
                // For each row of circulants
                for crow in 0..k/b {
                    // Data bit (row of full generator matrix)
                    let bit = crow*b + offset;
                    if data[bit/8] >> (7-(bit%8)) & 1 == 1 {
                        // If bit is set, XOR the generator constant in
                        for (idx, circ) in gc[crow*row_len..(crow+1)*row_len].iter().enumerate() {
                            parity[idx*8 + 7] ^= (*circ >>  0) as u8;
                            parity[idx*8 + 6] ^= (*circ >>  8) as u8;
                            parity[idx*8 + 5] ^= (*circ >> 16) as u8;
                            parity[idx*8 + 4] ^= (*circ >> 24) as u8;
                            parity[idx*8 + 3] ^= (*circ >> 32) as u8;
                            parity[idx*8 + 2] ^= (*circ >> 40) as u8;
                            parity[idx*8 + 1] ^= (*circ >> 48) as u8;
                            parity[idx*8 + 0] ^= (*circ >> 56) as u8;
                        }
                    }
                }
                // Now simulate the right-rotation of the generator by left-rotating the parity
                for block in 0..r/b {
                    let parityblock = &mut parity[block*b/8 .. (block+1)*b/8];
                    let mut carry = parityblock[0] >> 7;
                    for x in parityblock.iter_mut().rev() {
                        let c = *x >> 7;
                        *x = (*x<<1) | carry;
                        carry = c;
                    }
                }
            }
        }

        // Return a &mut [u8] view on the codeword
        codeword
    }

    fn copy_encode<'a>(code: &LDPCCode, data: &[u8], codeword: &'a mut[Self]) -> &'a mut [u8] {
        codeword[..data.len()].copy_from_slice(data);
        Self::encode(code, codeword)
    }

    fn bitlength() -> usize { 8 }
}

impl EncodeInto for u32 {
    fn encode<'a>(code: &LDPCCode, codeword: &'a mut[Self]) -> &'a mut [u8] {
        let k = code.k();
        let r = code.n() - code.k();
        let b = code.circulant_size();
        let gc = code.compact_generator();
        let row_len = r/64;

        // Scope the split of codeword into (data, parity)
        {
            // Split codeword into data and parity sections and then zero the parity bits
            let (data, parity) = codeword.split_at_mut(k / 32);
            for x in parity.iter_mut() { *x = 0; }

            // We treat data as a &[u8] so we bit-index it correctly despite endianness
            let data = unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, data.len()*4) };

            // For each rotation of the generator circulants
            for offset in 0..b {
                // For each row of circulants
                for crow in 0..k/b {
                    // Data bit (row of full generator matrix)
                    let bit = crow*b + offset;
                    if data[bit/8] >> (7-(bit%8)) & 1 == 1 {
                        // If bit is set, XOR the generator constant in
                        for (idx, circ) in gc[crow*row_len..(crow+1)*row_len].iter().enumerate() {
                            parity[idx*2 + 1] ^= (*circ >>  0) as u32;
                            parity[idx*2 + 0] ^= (*circ >> 32) as u32;
                        }
                    }
                }
                // Now simulate the right-rotation of the generator by left-rotating the parity
                if b >= 32 {
                    for block in 0..r/b {
                        let parityblock = &mut parity[block*b/32 .. (block+1)*b/32];
                        let mut carry = parityblock[0] >> 31;
                        for x in parityblock.iter_mut().rev() {
                            let c = *x >> 31;
                            *x = (*x<<1) | carry;
                            carry = c;
                        }
                    }
                } else if b == 16 {
                    // For small blocks we must rotate inside each parity word instead
                    for x in parity.iter_mut() {
                        let block1 = *x & 0xFFFF_0000;
                        let block2 = *x & 0x0000_FFFF;
                        *x =   (((block1<<1)|(block1>>15)) & 0xFFFF_0000)
                             | (((block2<<1)|(block2>>15)) & 0x0000_FFFF);
                    }
                }
            }

            // Need to compensate for endianness
            for x in parity.iter_mut() {
                *x = x.to_be();
            }
        }

        // Return a &mut [u8] view on the codeword
        unsafe {
            slice::from_raw_parts_mut::<'a>(codeword.as_mut_ptr() as *mut u8, codeword.len() * 4)
        }
    }

    fn copy_encode<'a>(code: &LDPCCode, data: &[u8], codeword: &'a mut[Self]) -> &'a mut [u8] {
        let codeword_u8 = unsafe {
            slice::from_raw_parts_mut::<'a>(codeword.as_mut_ptr() as *mut u8, codeword.len() * 4)
        };
        codeword_u8[..data.len()].copy_from_slice(data);
        Self::encode(code, codeword)
    }

    fn bitlength() -> usize { 32 }
}

impl EncodeInto for u64 {
    fn encode<'a>(code: &LDPCCode, codeword: &'a mut[Self]) -> &'a mut [u8] {
        let k = code.k();
        let r = code.n() - code.k();
        let b = code.circulant_size();
        let gc = code.compact_generator();
        let row_len = r/64;

        // Scope the split of codeword into (data, parity)
        {
            // Split codeword into data and parity sections and then zero the parity bits
            let (data, parity) = codeword.split_at_mut(k / 64);
            for x in parity.iter_mut() { *x = 0; }

            // We treat data as a &[u8] so we bit-index it correctly despite endianness
            let data = unsafe { slice::from_raw_parts(data.as_ptr() as *const u8, data.len()*8) };

            // For each rotation of the generator circulants
            for offset in 0..b {
                // For each row of circulants
                for crow in 0..k/b {
                    // Data bit (row of full generator matrix)
                    let bit = crow*b + offset;
                    if data[bit/8] >> (7-(bit%8)) & 1 == 1 {
                        // If bit is set, XOR the generator constant in
                        for (idx, circ) in gc[crow*row_len..(crow+1)*row_len].iter().enumerate() {
                            parity[idx] ^= *circ;
                        }
                    }
                }
                // Now simulate the right-rotation of the generator by left-rotating the parity
                if b >= 64 {
                    for block in 0..r/b {
                        let parityblock = &mut parity[block*b/64 .. (block+1)*b/64];
                        let mut carry = parityblock[0] >> 63;
                        for x in parityblock.iter_mut().rev() {
                            let c = *x >> 63;
                            *x = (*x<<1) | carry;
                            carry = c;
                        }
                    }
                } else if b == 32 {
                    // For small blocks we must rotate inside each parity word instead
                    for x in parity.iter_mut() {
                        let block1 = *x & 0xFFFFFFFF_00000000;
                        let block2 = *x & 0x00000000_FFFFFFFF;
                        *x =   (((block1<<1)|(block1>>31)) & 0xFFFFFFFF_00000000)
                             | (((block2<<1)|(block2>>31)) & 0x00000000_FFFFFFFF);
                    }
                } else if b == 16 {
                    for x in parity.iter_mut() {
                        let block1 = *x & 0xFFFF_0000_0000_0000;
                        let block2 = *x & 0x0000_FFFF_0000_0000;
                        let block3 = *x & 0x0000_0000_FFFF_0000;
                        let block4 = *x & 0x0000_0000_0000_FFFF;
                        *x =   (((block1<<1)|(block1>>15)) & 0xFFFF_0000_0000_0000)
                             | (((block2<<1)|(block2>>15)) & 0x0000_FFFF_0000_0000)
                             | (((block3<<1)|(block3>>15)) & 0x0000_0000_FFFF_0000)
                             | (((block4<<1)|(block4>>15)) & 0x0000_0000_0000_FFFF);
                    }
                }
            }

            // Need to compensate for endianness
            for x in parity.iter_mut() {
                *x = x.to_be();
            }
        }

        // Return a &mut [u8] view on the codeword
        unsafe {
            slice::from_raw_parts_mut::<'a>(codeword.as_mut_ptr() as *mut u8, codeword.len() * 8)
        }
    }

    fn copy_encode<'a>(code: &LDPCCode, data: &[u8], codeword: &'a mut[Self]) -> &'a mut [u8] {
        let codeword_u8 = unsafe {
            slice::from_raw_parts_mut::<'a>(codeword.as_mut_ptr() as *mut u8, codeword.len() * 8)
        };
        codeword_u8[..data.len()].copy_from_slice(data);
        Self::encode(code, codeword)
    }

    fn bitlength() -> usize { 64 }
}

impl LDPCCode {

    /// Encode a codeword. This function assumes the first k bits of `codeword` have already
    /// been set to your data, and will set the remaining n-k bits appropriately.
    ///
    /// `codeword` must be exactly n bits long.
    ///
    /// You can give `codeword` in `u8`, `u32`, or `u64`.
    /// The larger types are faster and are interpreted as packed bytes in little endian.
    ///
    /// Returns a view of `codeword` in &mut [u8] which may be convenient if you
    /// passed in a larger type but want to use the output as bytes. You can just
    /// not use the return value if you wish to keep your original view on `codeword`.
    pub fn encode<'a, T>(&self, codeword: &'a mut [T]) -> &'a mut [u8]
        where T: EncodeInto
    {
        assert_eq!(codeword.len() * T::bitlength(), self.n(), "codeword must be n bits long");
        EncodeInto::encode(self, codeword)
    }

    /// Encode a codeword, first copying in the data.
    ///
    /// This is the same as `encode` except you can pass the data which must be k bits long in as
    /// `&[u8]` and it will be copied into the first part of `codeword`, which must be n bits long.
    ///
    /// Returns a view of `codeword` in &mut [u8] which may be convenient if you
    /// passed in a larger type but want to use the output as bytes. You can just
    /// not use the return value if you wish to keep your original view on `codeword`.
    pub fn copy_encode<'a, T>(&self, data: &[u8], codeword: &'a mut [T]) -> &'a mut [u8]
        where T: EncodeInto
    {
        assert_eq!(data.len() * 8, self.k(), "data must be k bits long");
        assert_eq!(codeword.len() * T::bitlength(), self.n(), "codeword must be n bits long");
        EncodeInto::copy_encode(self, data, codeword)
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::v1::*;

    use crate::codes::LDPCCode;

    macro_rules! test_encode {
        ($code:path, $parity:expr) => {
            let code = $code;
            let parity = $parity;
            let txdata: Vec<u8> = (0..code.k()/8).map(|i| i as u8).collect();

            // First check we can encode OK in the totally normal way
            let mut txcode = vec![0u8; code.n()/8];
            txcode[..code.k()/8].copy_from_slice(&txdata);
            let rxcode = code.encode(&mut txcode);
            let (rxdata, rxparity) = rxcode.split_at(code.k()/8);
            assert_eq!(rxdata, &txdata[..]);
            assert_eq!(rxparity, &parity[..]);

            // Now check copy_encode works
            let mut txcode = vec![0u8; code.n()/8];
            let rxcode = code.copy_encode(&txdata, &mut txcode);
            let (rxdata, rxparity) = rxcode.split_at(code.k()/8);
            assert_eq!(rxdata, &txdata[..]);
            assert_eq!(rxparity, &parity[..]);

            // Now check for u32 version
            let mut txcode = vec![0u32; code.n()/32];
            let rxcode = code.copy_encode(&txdata, &mut txcode);
            let (rxdata, rxparity) = rxcode.split_at(code.k()/8);
            assert_eq!(rxdata, &txdata[..]);
            assert_eq!(rxparity, &parity[..]);

            // Now check for u64 version
            let mut txcode = vec![0u64; code.n()/64];
            let rxcode = code.copy_encode(&txdata, &mut txcode);
            let (rxdata, rxparity) = rxcode.split_at(code.k()/8);
            assert_eq!(rxdata, &txdata[..]);
            assert_eq!(rxparity, &parity[..]);
        };
    }

    #[test]
    fn test_encode() {
        test_encode!(LDPCCode::TC128,
                     [0x34, 0x99, 0x98, 0x87, 0x94, 0xE1, 0x62, 0x56]);

        test_encode!(LDPCCode::TC256,
                     [0x8C, 0x99, 0x21, 0x34, 0xAD, 0xB0, 0xCF, 0xD2,
                      0x2D, 0xA5, 0xF7, 0x7F, 0xBB, 0x42, 0x34, 0xCD]);

        test_encode!(LDPCCode::TC512,
                     [0xBC, 0x92, 0x1C, 0x98, 0xCC, 0xE2, 0x6C, 0xE8,
                      0x12, 0x3A, 0x97, 0xFF, 0x73, 0x5B, 0xF6, 0x9E,
                      0x08, 0xCB, 0x48, 0xC4, 0xC3, 0x00, 0x83, 0x0F,
                      0x30, 0xE0, 0x98, 0x59, 0xD6, 0x06, 0x7E, 0xBF]);

        test_encode!(LDPCCode::TM1280,
                     [0xF1, 0x68, 0xE0, 0x79, 0x45, 0xE3, 0x08, 0xAE,
                      0xEF, 0xD1, 0x68, 0x56, 0x60, 0x0A, 0x90, 0xFA,
                      0xF6, 0x55, 0xA2, 0x01, 0x60, 0x77, 0xF7, 0xE0,
                      0xFA, 0xB5, 0x49, 0x06, 0xDD, 0x6D, 0xCD, 0x7D]);

        test_encode!(LDPCCode::TM1536,
                     [0x99, 0x4D, 0x02, 0x17, 0x53, 0x87, 0xC8, 0xDD,
                      0x42, 0x2E, 0x46, 0x29, 0x06, 0x6A, 0x02, 0x6D,
                      0xE1, 0xAB, 0xB9, 0xA2, 0xAA, 0xE0, 0xF2, 0xE9,
                      0xF6, 0xAA, 0xE6, 0xF0, 0x42, 0x1E, 0x52, 0x44,
                      0x5F, 0x62, 0xD1, 0xA8, 0x8F, 0xB2, 0x01, 0x78,
                      0xB1, 0xD6, 0x2D, 0x0B, 0xD6, 0xB1, 0x4A, 0x6C,
                      0x93, 0x26, 0x69, 0xAA, 0xE0, 0x55, 0x1A, 0xD9,
                      0x9B, 0x94, 0x35, 0x27, 0x3F, 0x30, 0x91, 0x83]);

        test_encode!(LDPCCode::TM2048,
                     [0xEE, 0xA9, 0xAA, 0xAF, 0x98, 0xD9, 0x16, 0xCE,
                      0x6C, 0x2B, 0x28, 0x2D, 0x1A, 0x5B, 0x94, 0x4C,
                      0xA4, 0xF1, 0xB3, 0xD3, 0x1A, 0xEC, 0x58, 0x5A,
                      0xB3, 0xE6, 0xA4, 0xC4, 0x0D, 0xFB, 0x4F, 0x4D,
                      0xD8, 0x07, 0xA1, 0xAD, 0x0A, 0xE9, 0x62, 0xC4,
                      0xD4, 0x0B, 0xAD, 0xA1, 0x06, 0xE5, 0x6E, 0xC8,
                      0xA6, 0x68, 0x6A, 0xD5, 0xE6, 0xAC, 0x09, 0xBE,
                      0x3F, 0xF1, 0xF3, 0x4C, 0x7F, 0x35, 0x90, 0x27,
                      0xF2, 0x64, 0x69, 0x03, 0x83, 0x37, 0x42, 0x91,
                      0x21, 0xB7, 0xBA, 0xD0, 0x50, 0xE4, 0x91, 0x42,
                      0xE4, 0x0D, 0x64, 0x19, 0x70, 0x84, 0xA5, 0xB7,
                      0x86, 0x6F, 0x06, 0x7B, 0x12, 0xE6, 0xC7, 0xD5,
                      0xAB, 0x10, 0xDB, 0x03, 0x4F, 0xF6, 0x8A, 0xFE,
                      0x17, 0xAC, 0x67, 0xBF, 0xF3, 0x4A, 0x36, 0x42,
                      0x04, 0xAE, 0x85, 0xB3, 0xB6, 0x47, 0xCE, 0xC4,
                      0x0F, 0xA5, 0x8E, 0xB8, 0xBD, 0x4C, 0xC5, 0xCF]);

        test_encode!(LDPCCode::TM5120,
                     [0x4A, 0xA9, 0xB6, 0x89, 0x47, 0xB9, 0xAA, 0x41,
                      0x0E, 0xED, 0xF2, 0xCD, 0x03, 0xFD, 0xEE, 0x05,
                      0x40, 0xD1, 0x74, 0x9E, 0xD5, 0x99, 0x69, 0x47,
                      0x2C, 0xBD, 0x18, 0xF2, 0xB9, 0xF5, 0x05, 0x2B,
                      0x35, 0x4B, 0xB6, 0x02, 0x30, 0xBE, 0xE2, 0x24,
                      0x3A, 0x44, 0xB9, 0x0D, 0x3F, 0xB1, 0xED, 0x2B,
                      0x93, 0xF7, 0xE3, 0x6C, 0x0A, 0x66, 0xF8, 0x2D,
                      0xB4, 0xD0, 0xC4, 0x4B, 0x2D, 0x41, 0xDF, 0x0A,
                      0xAC, 0xCB, 0xF4, 0xD7, 0xC8, 0x0E, 0x3A, 0x9A,
                      0xBF, 0xD8, 0xE7, 0xC4, 0xDB, 0x1D, 0x29, 0x89,
                      0x4A, 0x65, 0xEB, 0x61, 0xE2, 0x2F, 0xC0, 0x33,
                      0x22, 0x0D, 0x83, 0x09, 0x8A, 0x47, 0xA8, 0x5B,
                      0xE7, 0x0F, 0xFA, 0xC7, 0x12, 0x35, 0x24, 0xEF,
                      0x4F, 0xA7, 0x52, 0x6F, 0xBA, 0x9D, 0x8C, 0x47,
                      0x31, 0xB1, 0x14, 0x3F, 0xB1, 0x14, 0x7F, 0x5D,
                      0x95, 0x15, 0xB0, 0x9B, 0x15, 0xB0, 0xDB, 0xF9]);

        test_encode!(LDPCCode::TM6144,
                     [0xA7, 0x42, 0xB8, 0x6D, 0x95, 0x7B, 0x9C, 0x90,
                      0xA8, 0x18, 0x30, 0xC2, 0x95, 0x08, 0x7F, 0xD7,
                      0xAB, 0x4E, 0xB4, 0x61, 0x99, 0x77, 0x90, 0x9C,
                      0xA4, 0x14, 0x3C, 0xCE, 0x99, 0x04, 0x73, 0xDB,
                      0x04, 0x59, 0xF4, 0xC0, 0xAD, 0x9F, 0xBA, 0x49,
                      0x1E, 0x53, 0x0A, 0x05, 0x5A, 0x9D, 0x1F, 0xDF,
                      0x67, 0x3A, 0x97, 0xA3, 0xCE, 0xFC, 0xD9, 0x2A,
                      0x7D, 0x30, 0x69, 0x66, 0x39, 0xFE, 0x7C, 0xBC,
                      0x70, 0x17, 0x15, 0x08, 0x56, 0xBD, 0x0B, 0xAC,
                      0x89, 0x12, 0x85, 0xB0, 0xF0, 0x88, 0x7F, 0x07,
                      0x1F, 0x78, 0x7A, 0x67, 0x39, 0xD2, 0x64, 0xC3,
                      0xE6, 0x7D, 0xEA, 0xDF, 0x9F, 0xE7, 0x10, 0x68,
                      0x17, 0x89, 0x95, 0x5C, 0x41, 0x92, 0x42, 0x05,
                      0xBD, 0x62, 0x80, 0x2B, 0x67, 0x59, 0xB2, 0xEB,
                      0x17, 0x89, 0x95, 0x5C, 0x41, 0x92, 0x42, 0x05,
                      0xBD, 0x62, 0x80, 0x2B, 0x67, 0x59, 0xB2, 0xEB,
                      0x3B, 0x15, 0xFB, 0xBE, 0xF2, 0x9B, 0xCE, 0x22,
                      0x77, 0xDC, 0xEB, 0x28, 0x03, 0xA7, 0x83, 0xAD,
                      0x7D, 0x53, 0xBD, 0xF8, 0xB4, 0xDD, 0x88, 0x64,
                      0x31, 0x9A, 0xAD, 0x6E, 0x45, 0xE1, 0xC5, 0xEB,
                      0x92, 0x3E, 0xBA, 0x63, 0x0F, 0x0F, 0x74, 0x3B,
                      0x96, 0x7F, 0x2F, 0xA5, 0x09, 0x43, 0xD0, 0xA6,
                      0x9C, 0x30, 0xB4, 0x6D, 0x01, 0x01, 0x7A, 0x35,
                      0x98, 0x71, 0x21, 0xAB, 0x07, 0x4D, 0xDE, 0xA8,
                      0xD0, 0xC9, 0x86, 0x0D, 0x68, 0xA3, 0xDA, 0x41,
                      0x50, 0xCF, 0x10, 0x6A, 0xA9, 0x24, 0xD0, 0x06,
                      0x12, 0x0B, 0x44, 0xCF, 0xAA, 0x61, 0x18, 0x83,
                      0x92, 0x0D, 0xD2, 0xA8, 0x6B, 0xE6, 0x12, 0xC4,
                      0x1B, 0x75, 0x76, 0xF7, 0xC3, 0xAF, 0x84, 0xE2,
                      0x16, 0xA6, 0xE4, 0x44, 0x06, 0x4F, 0x54, 0x98,
                      0xDC, 0xB2, 0xB1, 0x30, 0x04, 0x68, 0x43, 0x25,
                      0xD1, 0x61, 0x23, 0x83, 0xC1, 0x88, 0x93, 0x5F]);

        test_encode!(LDPCCode::TM8192,
                     [0xF6, 0x00, 0x56, 0xCD, 0x23, 0x63, 0x1A, 0xED,
                      0x7D, 0x7C, 0xF0, 0x17, 0x7C, 0xF1, 0x96, 0x73,
                      0x8C, 0xB7, 0xE0, 0xF4, 0x34, 0xF6, 0xB7, 0x3C,
                      0x89, 0x44, 0x85, 0x74, 0xA3, 0x27, 0x44, 0xF7,
                      0x0A, 0xFC, 0xAA, 0x31, 0xDF, 0x9F, 0xE6, 0x11,
                      0x81, 0x80, 0x0C, 0xEB, 0x80, 0x0D, 0x6A, 0x8F,
                      0x70, 0x4B, 0x1C, 0x08, 0xC8, 0x0A, 0x4B, 0xC0,
                      0x75, 0xB8, 0x79, 0x88, 0x5F, 0xDB, 0xB8, 0x0B,
                      0x5E, 0x53, 0x18, 0x0C, 0xB4, 0x32, 0x45, 0x92,
                      0x71, 0x93, 0xFE, 0xAD, 0xDF, 0x98, 0x55, 0xE8,
                      0x62, 0xB5, 0xFB, 0xBB, 0x4D, 0x94, 0x01, 0x2D,
                      0x22, 0xAD, 0x21, 0x55, 0x44, 0xED, 0x44, 0xC2,
                      0x61, 0x6C, 0x27, 0x33, 0x8B, 0x0D, 0x7A, 0xAD,
                      0x4E, 0xAC, 0xC1, 0x92, 0xE0, 0xA7, 0x6A, 0xD7,
                      0x5D, 0x8A, 0xC4, 0x84, 0x72, 0xAB, 0x3E, 0x12,
                      0x1D, 0x92, 0x1E, 0x6A, 0x7B, 0xD2, 0x7B, 0xFD,
                      0x59, 0x80, 0xA5, 0x02, 0xF2, 0xDD, 0x10, 0xCD,
                      0x8B, 0x8F, 0x52, 0xC3, 0x00, 0x65, 0xAD, 0xF7,
                      0xFB, 0xDF, 0x35, 0xB9, 0xCB, 0xB1, 0x90, 0x75,
                      0x68, 0xFC, 0x36, 0x33, 0x9D, 0x79, 0x18, 0xD0,
                      0x3A, 0xE3, 0xC6, 0x61, 0x91, 0xBE, 0x73, 0xAE,
                      0xE8, 0xEC, 0x31, 0xA0, 0x63, 0x06, 0xCE, 0x94,
                      0x98, 0xBC, 0x56, 0xDA, 0xA8, 0xD2, 0xF3, 0x16,
                      0x0B, 0x9F, 0x55, 0x50, 0xFE, 0x1A, 0x7B, 0xB3,
                      0x2D, 0xF0, 0xEA, 0x30, 0x5C, 0x71, 0xE6, 0xD8,
                      0x21, 0xE7, 0xC4, 0x1F, 0x68, 0xA5, 0x95, 0xAC,
                      0x3B, 0x2F, 0x62, 0xBC, 0x72, 0xF3, 0x2F, 0x9C,
                      0xB5, 0x0F, 0x9A, 0x27, 0x42, 0x5B, 0x0B, 0x49,
                      0x50, 0x8D, 0x97, 0x4D, 0x21, 0x0C, 0x9B, 0xA5,
                      0x5C, 0x9A, 0xB9, 0x62, 0x15, 0xD8, 0xE8, 0xD1,
                      0x46, 0x52, 0x1F, 0xC1, 0x0F, 0x8E, 0x52, 0xE1,
                      0xC8, 0x72, 0xE7, 0x5A, 0x3F, 0x26, 0x76, 0x34,
                      0x2C, 0xFB, 0x43, 0xA9, 0xBD, 0x5C, 0x53, 0x87,
                      0xD3, 0xF9, 0x40, 0x32, 0xED, 0x43, 0x7F, 0x96,
                      0x85, 0xAD, 0xC9, 0x0F, 0xE4, 0x5E, 0x11, 0xFA,
                      0xAF, 0x32, 0xBC, 0xD2, 0xC4, 0x2B, 0xC6, 0xE6,
                      0x65, 0xB2, 0x0A, 0xE0, 0xF4, 0x15, 0x1A, 0xCE,
                      0x9A, 0xB0, 0x09, 0x7B, 0xA4, 0x0A, 0x36, 0xDF,
                      0xCC, 0xE4, 0x80, 0x46, 0xAD, 0x17, 0x58, 0xB3,
                      0xE6, 0x7B, 0xF5, 0x9B, 0x8D, 0x62, 0x8F, 0xAF,
                      0x35, 0x97, 0x39, 0x80, 0xDC, 0x57, 0x52, 0xD1,
                      0xDB, 0x00, 0x55, 0x0B, 0x5E, 0x8E, 0x5C, 0xB6,
                      0x15, 0xB4, 0xF8, 0x9F, 0xAB, 0xF6, 0xCD, 0x60,
                      0xD0, 0xE1, 0x13, 0x02, 0x1B, 0x61, 0x87, 0x73,
                      0x9E, 0x3C, 0x92, 0x2B, 0x77, 0xFC, 0xF9, 0x7A,
                      0x70, 0xAB, 0xFE, 0xA0, 0xF5, 0x25, 0xF7, 0x1D,
                      0xBE, 0x1F, 0x53, 0x34, 0x00, 0x5D, 0x66, 0xCB,
                      0x7B, 0x4A, 0xB8, 0xA9, 0xB0, 0xCA, 0x2C, 0xD8,
                      0x31, 0xAD, 0x0C, 0x43, 0xE1, 0xE1, 0x81, 0x9F,
                      0xA3, 0x69, 0x90, 0x4C, 0xE6, 0x1F, 0x41, 0x61,
                      0xFB, 0x82, 0xD7, 0x97, 0x29, 0xC3, 0x30, 0x15,
                      0xA2, 0x72, 0xE7, 0x88, 0x85, 0xAD, 0x0F, 0x98,
                      0x9A, 0x06, 0xA7, 0xE8, 0x4A, 0x4A, 0x2A, 0x34,
                      0x08, 0xC2, 0x3B, 0xE7, 0x4D, 0xB4, 0xEA, 0xCA,
                      0x50, 0x29, 0x7C, 0x3C, 0x82, 0x68, 0x9B, 0xBE,
                      0x09, 0xD9, 0x4C, 0x23, 0x2E, 0x06, 0xA4, 0x33,
                      0xED, 0xA5, 0x15, 0xF4, 0x69, 0x70, 0x0B, 0xFF,
                      0x35, 0x9C, 0x94, 0x0B, 0x2E, 0xB3, 0x47, 0xCD,
                      0xEB, 0xC4, 0x3F, 0xF5, 0x82, 0x98, 0xCD, 0x72,
                      0x78, 0xF9, 0xEA, 0x4F, 0xF7, 0x49, 0x4B, 0xC3,
                      0x5E, 0x16, 0xA6, 0x47, 0xDA, 0xC3, 0xB8, 0x4C,
                      0x86, 0x2F, 0x27, 0xB8, 0x9D, 0x00, 0xF4, 0x7E,
                      0x58, 0x77, 0x8C, 0x46, 0x31, 0x2B, 0x7E, 0xC1,
                      0xCB, 0x4A, 0x59, 0xFC, 0x44, 0xFA, 0xF8, 0x70]);
    }
}
