// Copyright 2017 Adam Greig
// Licensed under the MIT license, see LICENSE for details.

#![feature(test)]
extern crate test;
use test::Bencher;

extern crate labrador_ldpc;
use labrador_ldpc::LDPCCode;

macro_rules! bench_iter_paritychecks {
    ($fn: ident, $code: path) => {
        #[bench]
        fn $fn(b: &mut Bencher) {
            b.iter(|| {
                for (count, var) in $code.iter_paritychecks() {
                    test::black_box(count);
                    test::black_box(var);
                }
            });
        }
    }
}

bench_iter_paritychecks!(bench_iter_paritychecks_tc128, LDPCCode::TC128);
bench_iter_paritychecks!(bench_iter_paritychecks_tc256, LDPCCode::TC256);
bench_iter_paritychecks!(bench_iter_paritychecks_tc512, LDPCCode::TC512);
bench_iter_paritychecks!(bench_iter_paritychecks_tm1280, LDPCCode::TM1280);
bench_iter_paritychecks!(bench_iter_paritychecks_tm1536, LDPCCode::TM1536);
bench_iter_paritychecks!(bench_iter_paritychecks_tm2048, LDPCCode::TM2048);
bench_iter_paritychecks!(bench_iter_paritychecks_tm5120, LDPCCode::TM5120);
bench_iter_paritychecks!(bench_iter_paritychecks_tm6144, LDPCCode::TM6144);
bench_iter_paritychecks!(bench_iter_paritychecks_tm8192, LDPCCode::TM8192);
