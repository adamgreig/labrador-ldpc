// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

//! This module provides the encoding function for turning data into codewords.
//!
//! Please refer to the `encode` method on [`LDPCCode`](../codes/enum.LDPCCode.html)
//! for more details.

// We have a couple of expressions with +0 for clarity of where the 0 comes from
#![cfg_attr(feature="cargo-clippy", allow(identity_op))]

use core::slice;

use ::codes::LDPCCode;

impl LDPCCode {

    /// Encode `data` into `codeword`.
    ///
    /// `data` must be k/8 long, and `codeword` must be n/8 long.
    ///
    /// The fastest routine is used when `codeword` is 64-bit aligned,
    /// reasonably fast when 32-bit aligned, and least fast otherwise.
    ///
    /// The compact generators must also be 64-bit aligned for fastest
    /// performance; but you'll have to guarantee this at link time somehow.
    ///
    /// All of `data` will be copied into the start of `codeword`,
    /// and then the additional parity bits computed.
    pub fn encode(&self, data: &[u8], codeword: &mut [u8]) {
        assert_eq!(data.len(), self.k()/8);
        assert_eq!(codeword.len(), self.n()/8);

        // Copy data into systematic part of codeword
        codeword[..data.len()].copy_from_slice(data);

        // Zero the remaining parity part of the codeword
        let mut parity = &mut codeword[data.len()..];
        for x in parity.iter_mut() { *x = 0; }

        let parity_addr = (parity.as_ptr() as *const usize) as usize;
        let gc_addr = (self.compact_generator().as_ptr() as *const usize) as usize;

        // Encode in u8 or u32 chunks depending on the alignment of `parity`
        if parity_addr % 8 == 0 && gc_addr % 8 == 0 {
            self.encode_aligned_u64(data, parity);
        } else if parity_addr % 4 == 0 {
            self.encode_aligned_u32(data, parity);
        } else {
            self.encode_unaligned(data, parity);
        }
    }

    /// Encode when `parity` is 32-bit aligned.
    ///
    /// We cast `parity` to a `&mut [u32]` and perform the parity additions as a single
    /// 32-bit XOR, which is respectably faster than four 8-bit XORs and shuffling.
    fn encode_aligned_u32<'a>(&self, data: &[u8], parity: &'a mut [u8]) {
        assert_eq!(parity.len() % 4, 0);
        assert_eq!(((parity.as_ptr() as *const usize) as usize) % 4, 0);

        let k = self.k();
        let n = self.n();
        let b = self.circulant_size();
        let gc = self.compact_generator();
        let r = n - k;
        let words_per_row = r/32;

        // We've checked parity has a length that's a multiple of 4 and an address
        // which is 4-byte aligned, so this really ought to be OK. For luck we also
        // ensure the new slice has the same lifetime as the old slice.
        let parity = unsafe {
            slice::from_raw_parts_mut::<'a>(parity.as_mut_ptr() as *mut u32, parity.len()/4)
        };

        // Because we'll be rotating the parity bits to simulate the rotation of the
        // generator matrix circulants, we operate in strides of the block size at
        // this outer level. So we'll XOR each row of the generator, then rotate
        // parity, and do it again at a bit offset of +1, repeat.
        for offset in 0..b {

            // For each row of the compact generator matrix, aka each row of circulants
            for block in 0..k/b {
                // This is the data bit (i.e. row of the full generator matrix)
                let bit = block*b + offset;

                // If the bit is set we will XOR the circulant row into our parity bits
                if data[bit/8] >> (7-(bit%8)) & 1 == 1 {
                    let g_row = &gc[block*words_per_row .. (block+1)*words_per_row];
                    for (i, gword) in g_row.iter().enumerate() {
                        parity[i] ^= *gword;
                    }
                }
            }

            // Now we need to left shift the parity bit blocks by 1.
            // For most codes b>=32 and we can operate word-wise, but for the small
            // TC128 code b=16 and we need to rotate within each u32.
            if b >= 32 {
                // For each circulant
                for block in 0..r/b {
                    // Cut out the parity bits for this circulant and perform a multi-word ROL
                    let parityblock = &mut parity[block*b/32 .. (block+1)*b/32];
                    let mut prevc = parityblock[0] >> 31;
                    for x in parityblock.iter_mut().rev() {
                        let c = *x >> 31;
                        *x = (*x<<1) | prevc;
                        prevc = c;
                    }
                }
            } else {
                // For each u32, ROL the two inner u16
                for x in parity.iter_mut() {
                    let block1 = *x & 0xFFFF_0000;
                    let block2 = *x & 0x0000_FFFF;
                    *x =   (((block1<<1)|(block1>>15)) & 0xFFFF_0000)
                         | (((block2<<1)|(block2>>15)) & 0x0000_FFFF);
                }
            }
        }

        // Compensate for interpreting [u8 u8 u8 u8] as [u32]
        for x in parity.iter_mut() {
            *x = x.to_be();
        }

    }
    /// Encode when `parity` is 64-bit aligned.
    ///
    /// We cast `parity` to a `&mut [u64]` and perform the parity additions as a single
    /// 64-bit XOR, which is much faster than eight 8-bit XORs and shuffling.
    fn encode_aligned_u64<'a>(&self, data: &[u8], parity: &'a mut [u8]) {
        assert_eq!(parity.len() % 8, 0);
        assert_eq!(((parity.as_ptr() as *const usize) as usize) % 8, 0);

        let k = self.k();
        let n = self.n();
        let b = self.circulant_size();
        let gc = self.compact_generator();
        let r = n - k;
        let words_per_row = r/64;

        // Check the generator matrix is also 64-bit aligned...
        assert_eq!(((gc.as_ptr() as *const usize) as usize) % 8, 0);

        // We've checked parity has a length that's a multiple of 8 and an address
        // which is 8-byte aligned, so this really ought to be OK. For luck we also
        // ensure the new slice has the same lifetime as the old slice.
        let parity = unsafe {
            slice::from_raw_parts_mut::<'a>(parity.as_mut_ptr() as *mut u64, parity.len()/8)
        };
        // Also get a &[u64] reference to gc.
        let gc = unsafe { slice::from_raw_parts(gc.as_ptr() as *const u64, gc.len()/2) };

        // Because we'll be rotating the parity bits to simulate the rotation of the
        // generator matrix circulants, we operate in strides of the block size at
        // this outer level. So we'll XOR each row of the generator, then rotate
        // parity, and do it again at a bit offset of +1, repeat.
        for offset in 0..b {

            // For each row of the compact generator matrix, aka each row of circulants
            for block in 0..k/b {
                // This is the data bit (i.e. row of the full generator matrix)
                let bit = block*b + offset;

                // If the bit is set we will XOR the circulant row into our parity bits
                if data[bit/8] >> (7-(bit%8)) & 1 == 1 {
                    let g_row = &gc[block*words_per_row .. (block+1)*words_per_row];
                    for (i, gword) in g_row.iter().enumerate() {
                        parity[i] ^= *gword;
                    }
                }
            }

            // Now we need to left shift the parity bit blocks by 1.
            // For some codes b>=64 and we can operate word-wise, but for the cases
            // where b=32 and b=16 we need to rotate inside the words instead.
            if b >= 64 {
                // For each circulant
                for block in 0..r/b {
                    // Cut out the parity bits for this circulant and perform a multi-word ROL
                    let parityblock = &mut parity[block*b/64 .. (block+1)*b/64];
                    let mut prevc = parityblock[0] >> 63;
                    for x in parityblock.iter_mut().rev() {
                        let c = *x >> 63;
                        *x = (*x<<1) | prevc;
                        prevc = c;
                    }
                }
            } else if b == 32 {
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
            } else {
                panic!();
            }
        }

        // We need to compensate for our weird interpretation of the generator and the data.
        // Since the generator was u32 and we read it as u64 we've shuffled the two u32s around
        // already, so we need to change the endianness of each u32 inside each u64.
        // So we'll reinterpret it as a &[u32] and go from there.
        let parity = unsafe {
            slice::from_raw_parts_mut::<'a>(parity.as_mut_ptr() as *mut u32, parity.len()*2)
        };
        for x in parity.iter_mut() {
            *x = x.to_be();
        }

    }

    /// Encode for any alignment of `parity`.
    ///
    /// This is about half the speed of `encode_aligned`.
    fn encode_unaligned(&self, data: &[u8], parity: &mut [u8]) {
        let k = self.k();
        let n = self.n();
        let b = self.circulant_size();
        let gc = self.compact_generator();
        let r = n - k;
        let words_per_row = r/32;

        // Because we'll be rotating the parity bits to simulate the rotation of the
        // generator matrix circulants, we operate in strides of the block size at
        // this outer level. So we'll XOR each row of the generator, then rotate
        // parity, and do it again at a bit offset of +1, repeat.
        for offset in 0..b {
            // For each row of the compact generator matrix, aka each row of circulants
            for block in 0..k/b {
                // This is the data bit (i.e. row of the full generator matrix)
                let bit = block*b + offset;
                // If the bit is set we will XOR the circulant row into our parity bits
                if data[bit/8] >> (7-(bit%8)) & 1 == 1 {
                    let row = block * words_per_row;
                    for (gword_idx, gword) in gc[row .. row+words_per_row].iter().enumerate() {
                        parity[gword_idx*4 + 3] ^= (*gword >>  0) as u8;
                        parity[gword_idx*4 + 2] ^= (*gword >>  8) as u8;
                        parity[gword_idx*4 + 1] ^= (*gword >> 16) as u8;
                        parity[gword_idx*4 + 0] ^= (*gword >> 24) as u8;
                    }
                }
            }
            // Now we left-rotate all the parity bits. Unlike in the aligned case,
            // since we're rotating through u8 we won't have issues with b=16.
            for block in 0..r/b {
                let parityblock = &mut parity[block*b/8 .. (block+1)*b/8];
                let mut prevc = parityblock[0] >> 7;
                for x in parityblock.iter_mut().rev() {
                    let c = *x >> 7;
                    *x = (*x<<1) | prevc;
                    prevc = c;
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::prelude::v1::*;

    use ::codes::LDPCCode;

    #[test]
    fn test_encode() {
        let code = LDPCCode::TC128;
        let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
        let mut txcode = vec![0u8; code.n()/8];
        code.encode(&txdata, &mut txcode);
        assert_eq!(txcode, vec![255, 254, 253, 252, 251, 250, 249, 248,
                                203, 102, 103, 120, 107,  30, 157, 169]);
    }
}
