// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

#![feature(test)]
extern crate test;
use test::Bencher;

extern crate labrador_ldpc;
use labrador_ldpc::LDPCCode;

#[bench]
fn bench_decode_mp(b: &mut Bencher) {
    let code = LDPCCode::TM5120;

    let mut ci = vec![0; code.sparse_paritycheck_ci_len()];
    let mut cs = vec![0; code.sparse_paritycheck_cs_len()];
    let mut vi = vec![0; code.sparse_paritycheck_vi_len()];
    let mut vs = vec![0; code.sparse_paritycheck_vs_len()];
    code.init_sparse_paritycheck(&mut ci, &mut cs, &mut vi, &mut vs);

    let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
    let mut txcode = vec![0u8; code.n()/8];
    code.copy_encode(&txdata, &mut txcode);

    // Copy it and corrupt the first bit
    let mut rxcode = txcode.clone();
    rxcode[0] ^= 1<<7;
    rxcode[2] ^= 1<<5;
    rxcode[4] ^= 1<<3;

    // Convert the hard data to LLRs
    let mut llrs = vec![0f32; code.n()];
    code.hard_to_llrs(&rxcode, &mut llrs);

    // Allocate working area and output area
    let mut working = vec![0f32; code.decode_mp_working_len()];
    let mut rxdata = vec![0u8; code.output_len()];

    // Run decoder
    b.iter(|| code.decode_mp(&ci, &cs, &vi, &vs, &llrs, &mut rxdata, &mut working));

    b.bytes = (code.k() as u64) / 8;
}

#[bench]
fn bench_decode_mp_new(b: &mut Bencher) {
    let code = LDPCCode::TM5120;

    let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
    let mut txcode = vec![0u8; code.n()/8];
    code.copy_encode(&txdata, &mut txcode);

    // Copy it and corrupt the first bit
    let mut rxcode = txcode.clone();
    rxcode[0] ^= 1<<7;
    rxcode[2] ^= 1<<5;
    rxcode[4] ^= 1<<3;

    // Convert the hard data to LLRs
    let mut llrs = vec![0i8; code.n()];
    //code.hard_to_llrs(&rxcode, &mut llrs);
    for (idx, byte) in rxcode.iter().enumerate() {
        for i in 0..8 {
            llrs[idx*8 + i] = if (byte >> (7-i)) & 1 == 1 { -1i8 } else { 1i8 };
        }
    }

    // Allocate working area and output area
    let mut working = vec![0i8; code.decode_mp_working_len()];
    let mut rxdata = vec![0u8; code.output_len()];

    // Run decoder
    b.iter(|| code.decode_mp_new(&llrs, &mut rxdata, &mut working));

    b.bytes = (code.k() as u64) / 8;
}

#[bench]
fn bench_decode_bf(b: &mut Bencher) {
    let code = LDPCCode::TC256;

    let mut ci = vec![0; code.sparse_paritycheck_ci_len()];
    let mut cs = vec![0; code.sparse_paritycheck_cs_len()];
    let mut vi = vec![0; code.sparse_paritycheck_vi_len()];
    let mut vs = vec![0; code.sparse_paritycheck_vs_len()];
    code.init_sparse_paritycheck(&mut ci, &mut cs, &mut vi, &mut vs);

    let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
    let mut txcode = vec![0u8; code.n()/8];
    code.copy_encode(&txdata, &mut txcode);

    // Copy to rx
    let mut rxcode = txcode.clone();
    //rxcode[0] ^= 1<<7;
    rxcode[0] ^= 0xFF;
    rxcode[2] ^= 0xFF;

    // Allocate working area and output area
    let mut working = vec![0u8; code.decode_bf_working_len()];
    let mut output = vec![0u8; code.output_len()];

    // Run decoder
    b.iter(|| {
        let (_, iters) = code.decode_bf(&ci, &cs, Some(&vi), Some(&vs), &rxcode, &mut output, &mut working);
        assert_eq!(iters, 50);
    });

    b.bytes = (code.k() as u64) / 8;
}

#[bench]
fn bench_decode_bf_new(b: &mut Bencher) {
    let code = LDPCCode::TC256;

    let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
    let mut txcode = vec![0u8; code.n()/8];
    code.copy_encode(&txdata, &mut txcode);

    // Copy to rx
    let mut rxcode = txcode.clone();
    //rxcode[0] ^= 1<<7;
    rxcode[0] ^= 0xFF;
    rxcode[2] ^= 0xFF;

    // Allocate working area and output area
    let mut working = vec![0u8; code.decode_bf_working_len()];
    let mut output = vec![0u8; code.output_len()];

    // Run decoder
    b.iter(|| {
        let (_, iters) = code.decode_bf_new(&rxcode, &mut output, &mut working);
        assert_eq!(iters, 50);
    });

    b.bytes = (code.k() as u64) / 8;
}

#[bench]
fn bench_iter_tc(b: &mut Bencher) {
    let code = LDPCCode::TC512;
    let sum = code.paritycheck_sum() as usize;
    b.iter(|| assert_eq!(sum, code.iter_paritychecks_tc().count()));
}

#[bench]
fn bench_iter_tm(b: &mut Bencher) {
    let code = LDPCCode::TM5120;
    let sum = code.paritycheck_sum() as usize;
    b.iter(|| assert_eq!(sum, code.iter_paritychecks_tm().count()));
}

#[bench]
fn bench_iter_tc_cs(b: &mut Bencher) {
    let code = LDPCCode::TC512;
    let mut ci = vec![0; code.sparse_paritycheck_ci_len()];
    let mut cs = vec![0; code.sparse_paritycheck_cs_len()];
    code.init_sparse_paritycheck_checks(&mut ci, &mut cs);

    b.iter(|| {
        let mut count = 0;
        for (i, cs_ss) in cs.windows(2).enumerate() {
            test::black_box(i);
            let (cs_start, cs_end) = (cs_ss[0] as usize, cs_ss[1] as usize);
            for a in &ci[cs_start..cs_end] {
                test::black_box(a);
                count += 1;
            }
        }
        assert_eq!(count, code.paritycheck_sum() as usize);
    });
}

#[bench]
fn bench_iter_tm_cs(b: &mut Bencher) {
    let code = LDPCCode::TM5120;
    let mut ci = vec![0; code.sparse_paritycheck_ci_len()];
    let mut cs = vec![0; code.sparse_paritycheck_cs_len()];
    code.init_sparse_paritycheck_checks(&mut ci, &mut cs);

    b.iter(|| {
        let mut count = 0;
        for (i, cs_ss) in cs.windows(2).enumerate() {
            test::black_box(i);
            let (cs_start, cs_end) = (cs_ss[0] as usize, cs_ss[1] as usize);
            for a in &ci[cs_start..cs_end] {
                test::black_box(a);
                count += 1;
            }
        }
        assert_eq!(count, code.paritycheck_sum() as usize);
    });
}
