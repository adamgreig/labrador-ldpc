// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

#![feature(test)]
extern crate test;
use test::Bencher;

extern crate labrador_ldpc;
use labrador_ldpc::LDPCCode;

macro_rules! bench_encode {
    ($fn: ident, $code:path, $ty:ty, $tylen:expr) => {
        #[bench]
        fn $fn(b: &mut Bencher) {
            let code = $code;
            let txdata: Vec<u8> = (0..code.k()/8).map(|i| i as u8).collect();
            let mut txcode: Vec<$ty> = vec![0; code.n()/$tylen];
            b.iter(|| { code.copy_encode(&txdata, &mut txcode); } );
        }
    }
}

bench_encode!(bench_encode_tc128_u08, LDPCCode::TC128, u8,   8);
bench_encode!(bench_encode_tc128_u32, LDPCCode::TC128, u32, 32);
bench_encode!(bench_encode_tc128_u64, LDPCCode::TC128, u64, 64);

bench_encode!(bench_encode_tc256_u08, LDPCCode::TC256, u8,   8);
bench_encode!(bench_encode_tc256_u32, LDPCCode::TC256, u32, 32);
bench_encode!(bench_encode_tc256_u64, LDPCCode::TC256, u64, 64);

bench_encode!(bench_encode_tc512_u08, LDPCCode::TC512, u8,   8);
bench_encode!(bench_encode_tc512_u32, LDPCCode::TC512, u32, 32);
bench_encode!(bench_encode_tc512_u64, LDPCCode::TC512, u64, 64);

bench_encode!(bench_encode_tm1280_u08, LDPCCode::TM1280, u8,   8);
bench_encode!(bench_encode_tm1280_u32, LDPCCode::TM1280, u32, 32);
bench_encode!(bench_encode_tm1280_u64, LDPCCode::TM1280, u64, 64);

bench_encode!(bench_encode_tm1536_u08, LDPCCode::TM1536, u8,   8);
bench_encode!(bench_encode_tm1536_u32, LDPCCode::TM1536, u32, 32);
bench_encode!(bench_encode_tm1536_u64, LDPCCode::TM1536, u64, 64);

bench_encode!(bench_encode_tm2048_u08, LDPCCode::TM2048, u8,   8);
bench_encode!(bench_encode_tm2048_u32, LDPCCode::TM2048, u32, 32);
bench_encode!(bench_encode_tm2048_u64, LDPCCode::TM2048, u64, 64);

bench_encode!(bench_encode_tm5120_u08, LDPCCode::TM5120, u8,   8);
bench_encode!(bench_encode_tm5120_u32, LDPCCode::TM5120, u32, 32);
bench_encode!(bench_encode_tm5120_u64, LDPCCode::TM5120, u64, 64);

bench_encode!(bench_encode_tm6144_u08, LDPCCode::TM6144, u8,   8);
bench_encode!(bench_encode_tm6144_u32, LDPCCode::TM6144, u32, 32);
bench_encode!(bench_encode_tm6144_u64, LDPCCode::TM6144, u64, 64);

bench_encode!(bench_encode_tm8192_u08, LDPCCode::TM8192, u8,   8);
bench_encode!(bench_encode_tm8192_u32, LDPCCode::TM8192, u32, 32);
bench_encode!(bench_encode_tm8192_u64, LDPCCode::TM8192, u64, 64);
