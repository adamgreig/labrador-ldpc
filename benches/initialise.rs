// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

#![feature(test)]
extern crate test;
use test::Bencher;

extern crate labrador_ldpc;
use labrador_ldpc::LDPCCode;

#[bench]
fn bench_init_generator(b: &mut Bencher) {
    let code = LDPCCode::TM1280;
    let mut g = vec![0u32; code.generator_len()];

    b.iter(|| code.init_generator(&mut g) );
}

#[bench]
fn bench_init_tm_paritycheck(b: &mut Bencher) {
    let code = LDPCCode::TM1280;
    let mut h = vec![0u32; code.paritycheck_len()];

    b.iter(|| code.init_paritycheck(&mut h) );
}

#[bench]
fn bench_init_tc_paritycheck(b: &mut Bencher) {
    let code = LDPCCode::TC512;
    let mut h = vec![0u32; code.paritycheck_len()];

    b.iter(|| code.init_paritycheck(&mut h) );
}

#[bench]
fn bench_init_tm_sparse_paritycheck_checks(b: &mut Bencher) {
    let code = LDPCCode::TM1280;
    let mut ci = vec![0u16; code.sparse_paritycheck_ci_len()];
    let mut cs = vec![0u16; code.sparse_paritycheck_cs_len()];

    b.iter(|| code.init_sparse_paritycheck_checks(&mut ci, &mut cs) );
}

#[bench]
fn bench_init_tc_sparse_paritycheck_checks(b: &mut Bencher) {
    let code = LDPCCode::TC512;
    let mut ci = vec![0u16; code.sparse_paritycheck_ci_len()];
    let mut cs = vec![0u16; code.sparse_paritycheck_cs_len()];

    b.iter(|| code.init_sparse_paritycheck_checks(&mut ci, &mut cs) );
}

#[bench]
fn bench_init_tm_sparse_paritycheck_variables(b: &mut Bencher) {
    let code = LDPCCode::TM1280;
    let mut ci = vec![0u16; code.sparse_paritycheck_ci_len()];
    let mut cs = vec![0u16; code.sparse_paritycheck_cs_len()];
    let mut vi = vec![0u16; code.sparse_paritycheck_vi_len()];
    let mut vs = vec![0u16; code.sparse_paritycheck_vs_len()];

    code.init_sparse_paritycheck_checks(&mut ci, &mut cs);

    b.iter(|| code.init_sparse_paritycheck_variables(&ci, &cs, &mut vi, &mut vs) );
}

#[bench]
fn bench_init_tc_sparse_paritycheck_variables(b: &mut Bencher) {
    let code = LDPCCode::TC512;
    let mut ci = vec![0u16; code.sparse_paritycheck_ci_len()];
    let mut cs = vec![0u16; code.sparse_paritycheck_cs_len()];
    let mut vi = vec![0u16; code.sparse_paritycheck_vi_len()];
    let mut vs = vec![0u16; code.sparse_paritycheck_vs_len()];

    code.init_sparse_paritycheck_checks(&mut ci, &mut cs);

    b.iter(|| code.init_sparse_paritycheck_variables(&ci, &cs, &mut vi, &mut vs) );
}
