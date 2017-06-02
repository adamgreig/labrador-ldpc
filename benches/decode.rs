// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

#![feature(test)]
extern crate test;
use test::Bencher;

extern crate labrador_ldpc;
use labrador_ldpc::LDPCCode;

macro_rules! bench_decode_bf {
    ($fn: ident, $code: path) => {
        #[bench]
        fn $fn(b: &mut Bencher) {
            let code = $code;

            // Generate some data and encode it
            let txdata: Vec<u8> = (0..code.k()/8).map(|i| i as u8).collect();
            let mut txcode = vec![0u8; code.n()/8];
            code.copy_encode(&txdata, &mut txcode);

            // Copy it and flip some bits
            let mut rxcode = txcode.clone();
            rxcode[0] ^= (1<<7) | (1<<5) | (1<<3);

            // Allocate working area and output area
            let mut working = vec![0u8; code.decode_bf_working_len()];
            let mut output = vec![0u8; code.output_len()];

            // Run decoder
            b.iter(|| {
                let (success, _) = code.decode_bf(&rxcode, &mut output, &mut working, 50);
                assert!(success);
            });
        }
    }
}

macro_rules! bench_decode_ms {
    ($fn: ident, $code: path, $ty: ty) => {
        #[bench]
        fn $fn(b: &mut Bencher) {
            let code = $code;

            // Generate some data and encode it
            let txdata: Vec<u8> = (0..code.k()/8).map(|i| i as u8).collect();
            let mut txcode = vec![0u8; code.n()/8];
            code.copy_encode(&txdata, &mut txcode);

            // Copy it and flip some bits
            let mut rxcode = txcode.clone();
            rxcode[0] ^= (1<<7) | (1<<5) | (1<<3);

            // Convert the hard data to LLRs
            let mut llrs = vec![0 as $ty; code.n()];
            code.hard_to_llrs(&rxcode, &mut llrs);

            // Allocate working area and output area
            let mut working = vec![0 as $ty; code.decode_ms_working_len()];
            let mut working_u8 = vec![0u8; code.decode_ms_working_u8_len()];
            let mut output = vec![0u8; code.output_len()];

            // Run decoder
            b.iter(|| {
                let (success, _) = code.decode_ms(&llrs, &mut output, &mut working,
                                                  &mut working_u8, 50);
                assert!(success);
            });
        }
    }
}

bench_decode_bf!(bench_decode_bf_tc128, LDPCCode::TC128);
bench_decode_bf!(bench_decode_bf_tc256, LDPCCode::TC256);
bench_decode_bf!(bench_decode_bf_tc512, LDPCCode::TC512);
bench_decode_bf!(bench_decode_bf_tm1280, LDPCCode::TM1280);
bench_decode_bf!(bench_decode_bf_tm1536, LDPCCode::TM1536);
bench_decode_bf!(bench_decode_bf_tm2048, LDPCCode::TM2048);
bench_decode_bf!(bench_decode_bf_tm5120, LDPCCode::TM5120);
bench_decode_bf!(bench_decode_bf_tm6144, LDPCCode::TM6144);
bench_decode_bf!(bench_decode_bf_tm8192, LDPCCode::TM8192);

bench_decode_ms!(bench_decode_ms_tc128_i8, LDPCCode::TC128, i8);
bench_decode_ms!(bench_decode_ms_tc256_i8, LDPCCode::TC256, i8);
bench_decode_ms!(bench_decode_ms_tc512_i8, LDPCCode::TC512, i8);
bench_decode_ms!(bench_decode_ms_tm1280_i8, LDPCCode::TM1280, i8);
bench_decode_ms!(bench_decode_ms_tm1536_i8, LDPCCode::TM1536, i8);
bench_decode_ms!(bench_decode_ms_tm2048_i8, LDPCCode::TM2048, i8);
bench_decode_ms!(bench_decode_ms_tm5120_i8, LDPCCode::TM5120, i8);
bench_decode_ms!(bench_decode_ms_tm6144_i8, LDPCCode::TM6144, i8);
bench_decode_ms!(bench_decode_ms_tm8192_i8, LDPCCode::TM8192, i8);

bench_decode_ms!(bench_decode_ms_tc128_f32, LDPCCode::TC128, f32);
bench_decode_ms!(bench_decode_ms_tc256_f32, LDPCCode::TC256, f32);
bench_decode_ms!(bench_decode_ms_tc512_f32, LDPCCode::TC512, f32);
bench_decode_ms!(bench_decode_ms_tm1280_f32, LDPCCode::TM1280, f32);
bench_decode_ms!(bench_decode_ms_tm1536_f32, LDPCCode::TM1536, f32);
bench_decode_ms!(bench_decode_ms_tm2048_f32, LDPCCode::TM2048, f32);
bench_decode_ms!(bench_decode_ms_tm5120_f32, LDPCCode::TM5120, f32);
bench_decode_ms!(bench_decode_ms_tm6144_f32, LDPCCode::TM6144, f32);
bench_decode_ms!(bench_decode_ms_tm8192_f32, LDPCCode::TM8192, f32);
