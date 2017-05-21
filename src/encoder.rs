//! This module provides encoding functions for turning data into codewords.

use ::codes::LDPCCode;

impl LDPCCode {
    /// Encode `data` into `codeword` without requiring the full generator matrix in-memory.
    ///
    /// This is slower than `encode_fast` but doesn't require `g`.
    ///
    /// `data` must be k/8 long, `codeword` must be n/8 long.
    pub fn encode_small(&self, _data: &[u8], _codeword: &mut [u8]) {
    }

    /// Encode `data` into `codeword` using the full generator matrix `g`.
    ///
    /// `g` must have been initialised using `init_generator_matrix()`,
    /// `data` must be k/8 long, and `codeword` must be n/8 long.
    pub fn encode_fast(&self, g: &[u32], data: &[u8], codeword: &mut [u8]) {
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

    /// Encode `data` into `codeword` using the full generator matrix `g`.
    ///
    /// This version uses an unsafe cast to access data and codeword as &[u32],
    /// which should be fine always. It's 2-3x quicker than `encode_fast`, too.
    ///
    /// `g` must have been initialised using `init_generator_matrix()`,
    /// `data` must be k/8 long, and `codeword` must be n/8 long.
    pub fn encode_xfast(&self, g: &[u32], data: &[u8], codeword: &mut [u8]) {
        #[cfg(test)]
        use std::slice;
        #[cfg(not(test))]
        use core::slice;

        assert_eq!(g.len(), self.generator_len());
        assert_eq!(data.len(), self.k()/8);
        assert_eq!(data.len() % 4, 0);
        assert_eq!(codeword.len(), self.n()/8);
        assert_eq!(codeword.len() % 4, 0);

        let k = self.k();
        let r = self.n() - self.k();
        let words_per_row = r/32;

        // Treat the data and codeword as a &[u32] instead of an &[u8].
        // This should always be safe since we won't exceed the bounds of the original array,
        // and all groups of four u8 are valid u32. It gets us a 2-3x speedup.
        let data_u32 = unsafe { slice::from_raw_parts(data.as_ptr() as *mut u32, data.len()/4) };
        let codeword_u32 = unsafe { slice::from_raw_parts_mut(codeword.as_mut_ptr() as *mut u32,
                                                              codeword.len()/4) };

        // Copy data into systematic part of codeword
        codeword_u32[..data_u32.len()].copy_from_slice(data_u32);

        // Zero the parity part of codeword so we can XOR into it later
        for x in &mut codeword_u32[k/32..] {
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
                if ((dword)>>(31-i) & 1) == 1 {
                    // For each word of the generator matrix row
                    let row = (dword_idx*32 + i) * words_per_row;
                    for (gword_idx, gword) in g[row .. row+words_per_row].iter().enumerate() {
                        codeword_u32[k/32 + gword_idx] ^= *gword;
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
    fn test_encode_fast() {
        let code = LDPCCode::TC128;
        let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
        let mut txcode = vec![0u8; code.n()/8];
        let mut g = vec![0u32; code.generator_len()];
        code.init_generator(&mut g);
        code.encode_fast(&g, &txdata, &mut txcode);
        println!("{:?}", txcode);
        assert_eq!(txcode, vec![255, 254, 253, 252, 251, 250, 249, 248,
                                203, 102, 103, 120, 107,  30, 157, 169]);
    }

    #[test]
    fn test_encode_xfast() {
        let code = LDPCCode::TC128;
        let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
        let mut txcode = vec![0u8; code.n()/8];
        let mut g = vec![0u32; code.generator_len()];
        code.init_generator(&mut g);
        code.encode_xfast(&g, &txdata, &mut txcode);
        println!("{:?}", txcode);
        assert_eq!(txcode, vec![255, 254, 253, 252, 251, 250, 249, 248,
                                203, 102, 103, 120, 107,  30, 157, 169]);
    }
}
