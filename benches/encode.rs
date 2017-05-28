// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

#![feature(test)]
extern crate test;
use test::Bencher;

extern crate labrador_ldpc;
use labrador_ldpc::LDPCCode;

#[bench]
fn bench_encode_aligned_u64(b: &mut Bencher) {
    let code = LDPCCode::TM1280;
    let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
    let mut txcode = vec![0u8; code.n()/8];

    b.iter(|| code.encode(&txdata, &mut txcode) );
}

#[bench]
fn bench_encode_aligned_u32(b: &mut Bencher) {
    let code = LDPCCode::TM1280;
    let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
    let mut txcode = vec![0u8; code.n()/8+4];

    b.iter(|| code.encode(&txdata, &mut txcode[4..]) );
}

#[bench]
fn bench_encode_unaligned(b: &mut Bencher) {
    let code = LDPCCode::TM1280;
    let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
    let mut txcode = vec![0u8; code.n()/8+1];

    b.iter(|| code.encode(&txdata, &mut txcode[1..]) );
}
