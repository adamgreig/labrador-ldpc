//! This module provides encoding functions for turning data into codewords.

use ::codes::LDPCCode;

impl LDPCCode {
    /// Encode `data` into `codeword` without requiring the full generator matrix in-memory.
    ///
    /// This is slower than `encode_fast` but doesn't require `g`.
    ///
    /// `data` must be k/8 long, `codeword` must be n/8 long.
    pub fn encode_small(&self, data: &[u8], codeword: &mut [u8]) {
        assert_eq!(data.len(), self.k()/8);
        assert_eq!(codeword.len(), self.n()/8);

        let k = self.k();
        let n = self.n();
        let b = self.circulant_size();
        let gc = self.compact_generator();
        let r = n - k;

        // Compiler doesn't know b is a power of two, so we'll work out the
        // mask for % and the shift for / ourselves.
        let modb = b - 1;
        let divb = b.trailing_zeros();

        // Copy data into systematic part of codeword
        codeword[..data.len()].copy_from_slice(data);

        // Zero the parity part of the codeword so we can XOR into it
        for x in &mut codeword[data.len()..] {
            *x = 0;
        }

        // For each parity check equation (column of the P part of G)
        for check in 0..r {
            let mut parity = 0;
            // For each input data byte
            for (dbyte_idx, dbyte) in data.iter().enumerate() {
                // For each bit in this byte, MSbit first
                for i in 0..8 {
                    // If this bit is set
                    if (dbyte>>(7-i) & 1) == 1 {
                        // Work out what bit position (row) we're at
                        let j = dbyte_idx*8 + i;

                        // For each row below the one stored in gc, everything is rotated right,
                        // so work out the bit position offset to cancel this rotation.
                        // Might need to wrap around.
                        let mut check_offset: isize = (j & modb) as isize;
                        if check_offset > (check & modb) as isize {
                            check_offset -= b as isize;
                        }
                        let offset_check = (check as isize - check_offset) as usize;

                        // Pick the relevant compact generator constant for this data bit (row)
                        // and check (column). We are skipping (j/b) rows of (r/32) words above,
                        // and then (i-io)/32 words left, to get to the word we want.
                        let gcword = gc[(j>>divb)*(r/32) + offset_check/32];

                        // Add to our running parity check if the relevant bit is set
                        if (gcword >> (31 - (offset_check % 32))) & 1 == 1 {
                            parity ^= 1;
                        }
                    }
                }
            }

            // If the parity bit ended up set, update the codeword
            if parity == 1 {
                codeword[(k/8) + (check/8)] |= parity << (7 - (check%8));
            }
        }
    }

    /// Encode `data` into `codeword` using the full generator matrix `g`.
    ///
    /// `g` must have been initialised using `init_generator_matrix()`,
    /// `data` must be k/8 long, and `codeword` must be n/8 long.
    ///
    /// Additionally both `data` and `codeword` must be 32-bit aligned.
    ///
    /// The lifetimes are just for internal use.
    pub fn encode_fast<'data, 'codeword>(&self, g: &[u32], data: &'data [u8],
                                          codeword: &'codeword mut [u8])
    {
        #[cfg(no_std)]
        use core::slice;
        #[cfg(not(no_std))]
        use std::slice;

        assert_eq!(g.len(), self.generator_len());
        assert_eq!(data.len(), self.k()/8);
        assert_eq!(data.len() % 4, 0);
        assert_eq!(((data.as_ptr() as *const usize) as usize) % 4, 0);
        assert_eq!(codeword.len(), self.n()/8);
        assert_eq!(codeword.len() % 4, 0);
        assert_eq!(((codeword.as_ptr() as *const usize) as usize) % 4, 0);

        let k = self.k();
        let r = self.n() - self.k();
        let words_per_row = r/32;

        // Treat the data and codeword as a &[u32] instead of an &[u8].
        // This should always be safe since we won't exceed the bounds of the original array,
        // and all groups of four u8 are valid u32. We explicitly tie the lifetimes to the
        // input slices.
        // This nets us a 2-3x speedup for encoding so seems quite worthwhile.
        let data_u32 = unsafe {
            slice::from_raw_parts::<'data>(data.as_ptr() as *mut u32, data.len()/4)
        };
        let codeword_u32 = unsafe {
            slice::from_raw_parts_mut::<'codeword>(codeword.as_mut_ptr() as *mut u32,
                                                   codeword.len()/4)
        };

        // Copy data into systematic part of codeword
        codeword_u32[..data_u32.len()].copy_from_slice(data_u32);

        // Zero the parity part of codeword so we can XOR into it later
        for x in &mut codeword_u32[data_u32.len()..] {
            *x = 0;
        }

        // For each u32 of data
        for (dword_idx, dword) in data_u32.iter().enumerate() {
            // Need to swap the data around to big endian to get the bit ordering right.
            // This is still faster than processing the data byte by byte.
            let dword = dword.to_be();
            // For each bit in dword, MSbit first
            for i in 0..32 {
                // If the bit is set
                if (dword>>(31-i) & 1) == 1 {
                    // For each word of the generator matrix row
                    let row = (dword_idx*32 + i) * words_per_row;
                    for (gword_idx, gword) in g[row .. row+words_per_row].iter().enumerate() {
                        codeword_u32[k/32 + gword_idx] ^= *gword;
                    }
                }
            }
        }
    }

    /// Encode `data` into `codeword` using the full generator matrix `g`.
    ///
    /// `g` must have been initialised using `init_generator_matrix()`,
    /// `data` must be k/8 long, and `codeword` must be n/8 long.
    pub fn encode_fast_safe(&self, g: &[u32], data: &[u8], codeword: &mut [u8]) {
        assert_eq!(g.len(), self.generator_len());
        assert_eq!(data.len(), self.k()/8);
        assert_eq!(codeword.len(), self.n()/8);

        let k = self.k();
        let r = self.n() - self.k();
        let words_per_row = r/32;

        // Copy data into systematic part of codeword
        codeword[..data.len()].copy_from_slice(data);

        // Zero the parity part of codeword so we can XOR into it later
        for x in &mut codeword[k/8..] {
            *x = 0;
        }

        // For each byte of data
        for (byte_idx, byte) in data.iter().enumerate() {
            // For each bit in the byte, MSbit first
            for i in 0..8 {
                // If the bit is set
                if ((*byte)>>(7-i) & 1) == 1 {
                    // For each word of the generator matrix row
                    let row = (byte_idx*8 + i) * words_per_row;
                    for (word_idx, word) in g[row .. row + words_per_row].iter().enumerate() {
                        // Remember that the contents of g have already been made big endian,
                        // so the 8 MSbits correspond to the left-most 8 bits of the generator.
                        codeword[k/8 + word_idx*4 + 0] ^= (*word >>  0) as u8;
                        codeword[k/8 + word_idx*4 + 1] ^= (*word >>  8) as u8;
                        codeword[k/8 + word_idx*4 + 2] ^= (*word >> 16) as u8;
                        codeword[k/8 + word_idx*4 + 3] ^= (*word >> 24) as u8;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use ::codes::LDPCCode;

    #[test]
    fn test_encode_small() {
        let code = LDPCCode::TC128;
        let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
        let mut txcode = vec![0u8; code.n()/8];
        code.encode_small(&txdata, &mut txcode);
        assert_eq!(txcode, vec![255, 254, 253, 252, 251, 250, 249, 248,
                                203, 102, 103, 120, 107,  30, 157, 169]);
    }

    #[test]
    fn test_encode_fast() {
        let code = LDPCCode::TC128;
        let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
        let mut txcode = vec![0u8; code.n()/8];
        let mut g = vec![0u32; code.generator_len()];
        code.init_generator(&mut g);
        code.encode_fast(&g, &txdata, &mut txcode);
        assert_eq!(txcode, vec![255, 254, 253, 252, 251, 250, 249, 248,
                                203, 102, 103, 120, 107,  30, 157, 169]);
    }
}
