#![feature(test)]
extern crate test;
use test::Bencher;

extern crate labrador_ldpc;
use labrador_ldpc::LDPCCode;

#[bench]
fn bench_decode_mp(b: &mut Bencher) {
    let code = LDPCCode::TM2048;

    let mut g = vec![0u32; code.generator_len()];
    code.init_generator(&mut g);

    let mut ci = vec![0; code.sparse_paritycheck_ci_len()];
    let mut cs = vec![0; code.sparse_paritycheck_cs_len()];
    let mut vi = vec![0; code.sparse_paritycheck_vi_len()];
    let mut vs = vec![0; code.sparse_paritycheck_vs_len()];
    code.init_sparse_paritycheck(&mut ci, &mut cs, &mut vi, &mut vs);

    let txdata: Vec<u8> = (0..code.k()/8).map(|i| !(i as u8)).collect();
    let mut txcode = vec![0u8; code.n()/8];
    code.encode_fast(&g, &txdata, &mut txcode);

    // Copy it and corrupt the first bit
    let mut rxcode = txcode.clone();
    rxcode[0] ^= 1<<7;

    // Convert the hard data to LLRs
    let mut llrs = vec![0f32; code.n()];
    code.hard_to_llrs(&rxcode, &mut llrs);

    // Allocate working area and output area
    let mut working = vec![0f32; code.decode_mp_working_len()];
    let mut rxdata = vec![0u8; code.output_len()];

    // Run decoder
    b.iter(|| code.decode_mp(&ci, &cs, &vi, &vs, &llrs, &mut rxdata, &mut working));
}
