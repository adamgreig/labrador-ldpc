use std::sync::atomic::{AtomicU64, AtomicBool, Ordering};
use rand::Rng;
use rand_distr::{Normal, Distribution};
use labrador_ldpc::codes::LDPCCode;

/// Runs the min-sum decoder for a random codeword of the given code at the specified SNR.
///
/// Returns number of bits in error.
fn ms_trial(code: LDPCCode, snr_db: f32) -> u32 {
    let mut txcode = vec![0u8; code.n()/8];
    rand::thread_rng().fill(&mut txcode[..]);
    code.encode(&mut txcode);
    let mut llrs = vec![0f32; code.n()];
    code.hard_to_llrs(&txcode[..], &mut llrs[..]);
    let noise = Normal::new(0.0, 1.0 / f32::powf(10.0, snr_db / 10.0)).unwrap();
    for llr in llrs.iter_mut() {
        *llr += noise.sample(&mut rand::thread_rng());
    }
    let mut working = vec![0f32; code.decode_ms_working_len()];
    let mut working_u8 = vec![0u8; code.output_len() - code.k()/8];
    let mut output = vec![0u8; code.output_len()];
    code.decode_ms(&llrs, &mut output, &mut working, &mut working_u8, 100);
    let mut errors = 0;
    let txdata = &txcode[..code.k()/8];
    for (tx, rx) in txdata.iter().zip(output.iter()) {
        errors += (tx ^ rx).count_ones();
    }
    errors
}

/// Run many iterations of the min-sum decoder for each of the specified SNRs.
///
/// Returns a vec of BERs corresponding to each SNR.
fn ms_decoder_many_trials(code: LDPCCode, snrs_db: &[f32]) -> Vec<f32> {
    let mut bers = vec![0f32; snrs_db.len()];
    for (idx, snr_db) in snrs_db.iter().enumerate() {
        let count = AtomicU64::new(0);
        let term = AtomicBool::new(false);
        rayon::scope(|s| {
            s.spawn_broadcast(|_, _| {
                while !term.load(Ordering::Relaxed) {
                    let errs = ms_trial(code, *snr_db) as u64;
                    count.fetch_add((1 << 32) | errs, Ordering::Relaxed);
                }
            });
            loop {
                let v = count.load(Ordering::Relaxed);
                let n_trials = v >> 32;
                let n_errors = v & 0xffff_ffff;
                if (n_trials * code.k() as u64) > 50_000_000 || n_errors > 5_000 {
                    term.store(true, Ordering::Relaxed);
                    break;
                }
                std::thread::sleep(std::time::Duration::from_millis(10));
            }
        });
        let v = count.load(Ordering::Relaxed);
        let n_trials = v >> 32;
        let n_bits = n_trials * code.k() as u64;
        let n_errors = u64::max(1, v & 0xffff_ffff);
        bers[idx] = (n_errors as f32) / (code.k() as f32 * n_trials as f32);
        println!("{code:?},{snr_db:.2},{n_trials},{n_bits},{n_errors},{:.5e}", bers[idx]);
    }
    bers
}

fn main() {
    let snrs_db = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0, 2.1, 2.2];
    ms_decoder_many_trials(LDPCCode::TC512, &snrs_db);
}
