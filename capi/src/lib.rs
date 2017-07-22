#![no_std]
#![feature(lang_items)]

extern crate labrador_ldpc;

use labrador_ldpc::LDPCCode;
use core::slice;

#[lang="eh_personality"] extern fn eh_personality() {}
#[lang="panic_fmt"] #[no_mangle] pub extern fn panic_fmt() { loop {}}

#[no_mangle]
pub extern fn labrador_ldpc_code_n(code: LDPCCode) -> usize {
    code.n()
}

#[no_mangle]
pub extern fn labrador_ldpc_code_k(code: LDPCCode) -> usize {
    code.k()
}

#[no_mangle]
pub extern fn labrador_ldpc_encode(code: LDPCCode, codeword: *mut u8) {
    // TODO use alignment of codeword to optimally choose type to cast to
    let codeword: &mut[u8] = unsafe { slice::from_raw_parts_mut(codeword, code.n()/8) };
    code.encode(codeword);
}

#[no_mangle]
pub extern fn labrador_ldpc_copy_encode(code: LDPCCode, data: *const u8, codeword: *mut u8) {
    // TODO use alignment of codeword to optimally choose type to cast to
    let data: &[u8] = unsafe { slice::from_raw_parts(data, code.k()/8) };
    let codeword: &mut[u8] = unsafe { slice::from_raw_parts_mut(codeword, code.n()/8) };
    code.copy_encode(data, codeword);
}

#[no_mangle]
pub extern fn labrador_ldpc_bf_working_len(code: LDPCCode) -> usize {
    code.decode_bf_working_len()
}

#[no_mangle]
pub extern fn labrador_ldpc_ms_working_len(code: LDPCCode) -> usize {
    code.decode_ms_working_len()
}

#[no_mangle]
pub extern fn labrador_ldpc_ms_working_u8_len(code: LDPCCode) -> usize {
    code.decode_ms_working_u8_len()
}

#[no_mangle]
pub extern fn labrador_ldpc_output_len(code: LDPCCode) -> usize {
    code.output_len()
}

#[no_mangle]
pub extern fn labrador_ldpc_decode_bf(code: LDPCCode, input: *const u8, output: *mut u8,
                                      working: *mut u8, max_iters: usize,
                                      iters_run: *mut usize) -> bool
{
    let input: &[u8] = unsafe { slice::from_raw_parts(input, code.n()/8) };
    let output: &mut[u8] = unsafe { slice::from_raw_parts_mut(output, code.output_len()) };
    let working: &mut[u8] = unsafe { slice::from_raw_parts_mut(working, code.decode_bf_working_len()) };
    let (result, iters) = code.decode_bf(input, output, working, max_iters);
    if !iters_run.is_null() {
        unsafe { *iters_run = iters };
    }
    result
}

fn decode_ms<T>(code: LDPCCode, llrs: *const T, output: *mut u8, working: *mut T,
                working_u8: *mut u8, max_iters: usize, iters_run: *mut usize) -> bool
    where T: labrador_ldpc::decoder::DecodeFrom
{
    let llrs: &[T] = unsafe { slice::from_raw_parts(llrs, code.n()) };
    let output: &mut[u8] = unsafe { slice::from_raw_parts_mut(output, code.output_len()) };
    let working: &mut[T] = unsafe { slice::from_raw_parts_mut(working, code.decode_ms_working_len()) };
    let working_u8: &mut[u8] = unsafe { slice::from_raw_parts_mut(working_u8, code.decode_ms_working_u8_len()) };
    let (result, iters) = code.decode_ms(llrs, output, working, working_u8, max_iters);
    if !iters_run.is_null() {
        unsafe { *iters_run = iters };
    }
    result
}

#[no_mangle]
pub extern fn labrador_ldpc_decode_ms_i8(code: LDPCCode, llrs: *const i8, output: *mut u8,
                                         working: *mut i8, working_u8: *mut u8, max_iters: usize,
                                         iters_run: *mut usize) -> bool
{
    decode_ms::<i8>(code, llrs, output, working, working_u8, max_iters, iters_run)
}

#[no_mangle]
pub extern fn labrador_ldpc_decode_ms_i16(code: LDPCCode, llrs: *const i16, output: *mut u8,
                                         working: *mut i16, working_u8: *mut u8, max_iters: usize,
                                         iters_run: *mut usize) -> bool
{
    decode_ms::<i16>(code, llrs, output, working, working_u8, max_iters, iters_run)
}

#[no_mangle]
pub extern fn labrador_ldpc_decode_ms_f32(code: LDPCCode, llrs: *const f32, output: *mut u8,
                                         working: *mut f32, working_u8: *mut u8, max_iters: usize,
                                         iters_run: *mut usize) -> bool
{
    decode_ms::<f32>(code, llrs, output, working, working_u8, max_iters, iters_run)
}

#[no_mangle]
pub extern fn labrador_ldpc_decode_ms_f64(code: LDPCCode, llrs: *const f64, output: *mut u8,
                                         working: *mut f64, working_u8: *mut u8, max_iters: usize,
                                         iters_run: *mut usize) -> bool
{
    decode_ms::<f64>(code, llrs, output, working, working_u8, max_iters, iters_run)
}

fn hard_to_llrs<T>(code: LDPCCode, input: *const u8, llrs: *mut T)
    where T: labrador_ldpc::decoder::DecodeFrom
{
    let input: &[u8] = unsafe { slice::from_raw_parts(input, code.n()/8) };
    let llrs: &mut[T] = unsafe { slice::from_raw_parts_mut(llrs, code.n()) };
    code.hard_to_llrs(input, llrs);
}

#[no_mangle]
pub extern fn labrador_ldpc_hard_to_llrs_i8(code: LDPCCode, input: *const u8, llrs: *mut i8) {
    hard_to_llrs::<i8>(code, input, llrs);
}

#[no_mangle]
pub extern fn labrador_ldpc_hard_to_llrs_i16(code: LDPCCode, input: *const u8, llrs: *mut i16) {
    hard_to_llrs::<i16>(code, input, llrs);
}

#[no_mangle]
pub extern fn labrador_ldpc_hard_to_llrs_f32(code: LDPCCode, input: *const u8, llrs: *mut f32) {
    hard_to_llrs::<f32>(code, input, llrs);
}

#[no_mangle]
pub extern fn labrador_ldpc_hard_to_llrs_f64(code: LDPCCode, input: *const u8, llrs: *mut f64) {
    hard_to_llrs::<f64>(code, input, llrs);
}
