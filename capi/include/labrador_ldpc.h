/* Labrador-LDPC C API
 * Copyright 2017 Adam Greig
 * Licensed under the MIT license, see LICENSE for details.
 * See README or https://docs.rs/labrador-ldpc-capi for documentation.
 */

#ifndef LABRADOR_LDPC_CAPI
#define LABRADOR_LDPC_CAPI

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

/* Available LDPC codes.
 * For further details refer to:
 * https://docs.rs/labrador-ldpc/1.0.0/labrador_ldpc/codes/enum.LDPCCode.html
 */
enum labrador_ldpc_code {
    LABRADOR_LDPC_CODE_TC128,
    LABRADOR_LDPC_CODE_TC256,
    LABRADOR_LDPC_CODE_TC512,
    LABRADOR_LDPC_CODE_TM1280,
    LABRADOR_LDPC_CODE_TM1536,
    LABRADOR_LDPC_CODE_TM2048,
    LABRADOR_LDPC_CODE_TM5120,
    LABRADOR_LDPC_CODE_TM6144,
    LABRADOR_LDPC_CODE_TM8192,
};

/* Returns the code length n (number of codeword bits) for a given code. */
size_t labrador_ldpc_code_n(enum labrador_ldpc_code code);

/* Returns the code dimension k (number of data bits) for a given code. */
size_t labrador_ldpc_code_k(enum labrador_ldpc_code code);

/* Encode the first k/8 bytes of `codeword` into the rest of `codeword`,
 * using the `code` LDPC code.
 */
void labrador_ldpc_encode(enum labrador_ldpc_code code, uint8_t *codeword);

/* Encode all of `data` (k/8 bytes long) into `codeword` (n/8 bytes long),
 * first copying `data` into `codeword`, using the `code` LDPC code.
 */
void labrador_ldpc_copy_encode(enum labrador_ldpc_code code,
                               const uint8_t* data, uint8_t* codeword);

/* Returns the required length of the working area for the BF decoder. */
size_t labrador_ldpc_bf_working_len(enum labrador_ldpc_code code);

/* Returns the required length of the uint8_t working area for the MS decoder.
 */
size_t labrador_ldpc_ms_working_u8_len(enum labrador_ldpc_code code);

/* Returns the required length of the LLR-type working area for the MS decoder.
 */
size_t labrador_ldpc_ms_working_len(enum labrador_ldpc_code code);

/* Returns the required length for the output of any decoder. */
size_t labrador_ldpc_output_len(enum labrador_ldpc_code code);

/* Run the BF decoder:
 *
 * `code` is the LDPC code in use.
 * `input` is a received codeword, with each bit representing one received bit.
 * `output` is filled with the decoded codeword, and must be the length given
 *          by `labrador_ldpc_output_len`.
 * `working` is the working area and must be the length given by
 *           `labrador_ldpc_bf_working_len`.
 * `max_iters` is the maximum number of iterations to run for, e.g. 50.
 * `iters_run`, if not NULL, is set to the number of iterations actually run.
 *
 * Returns true on decoding success or false on failure.
 */
bool labrador_ldpc_decode_bf(enum labrador_ldpc_code code,
                             const uint8_t *input, uint8_t *output,
                             uint8_t *working, size_t max_iters,
                             size_t *iters_run);

/* Run the MS decoder:
 *
 * `code` is the LDPC code in use.
 * `llrs` contains n entries, one per received bit, with positive numbers
 *        more likely to be a 0 bit.
 * `output` is filled with the decoded codeword, and must be the length given
 *          by `labrador_ldpc_output_len`.
 * `working` is the main working area and must be the length given by
 *           `labrador_ldpc_ms_working_len`.
 * `working_u8` is the secondary working area and must be the length given by
 *              `labrador_ldpc_ms_u8_working_len`.
 * `max_iters` is the maximum number of iterations to run for, e.g. 200.
 * `iters_run`, if not NULL, is set to the number of iterations actually run.
 *
 * Returns true on decoding success or false on failure.
 *
 * Four variants are provided which use different types for the LLRs.
 * For details on this choice please refer to:
 * https://docs.rs/labrador-ldpc/1.0.0/labrador_ldpc/codes/enum.LDPCCode.html,
 * section "Log Likelihood Ratios and choice of T".
 */
bool labrador_ldpc_decode_ms_i8(enum labrador_ldpc_code code,
                                const int8_t* llrs, uint8_t *output,
                                int8_t* working, uint8_t *working_u8,
                                size_t max_iters, size_t* iters_run);
bool labrador_ldpc_decode_ms_i16(enum labrador_ldpc_code code,
                                const int16_t* llrs, uint8_t *output,
                                int16_t* working, uint8_t *working_u8,
                                size_t max_iters, size_t* iters_run);
bool labrador_ldpc_decode_ms_f32(enum labrador_ldpc_code code,
                                const float* llrs, uint8_t *output,
                                float* working, uint8_t *working_u8,
                                size_t max_iters, size_t* iters_run);
bool labrador_ldpc_decode_ms_f64(enum labrador_ldpc_code code,
                                const double* llrs, uint8_t *output,
                                double* working, uint8_t *working_u8,
                                size_t max_iters, size_t* iters_run);

/* Convert hard information into LLRs.
 *
 * Assigns -/+ 1 for 1/0 bits.
 *
 * `input` must be n/8 bytes long.
 * `llrs` must be n entries long.
 *
 * Available in four variants for different LLR types.
 */
void labrador_ldpc_hard_to_llrs_i8(enum labrador_ldpc_code code,
                                   const uint8_t* input, int8_t* llrs);
void labrador_ldpc_hard_to_llrs_i16(enum labrador_ldpc_code code,
                                    const uint8_t* input, int16_t* llrs);
void labrador_ldpc_hard_to_llrs_f32(enum labrador_ldpc_code code,
                                    const uint8_t* input, float* llrs);
void labrador_ldpc_hard_to_llrs_f64(enum labrador_ldpc_code code,
                                    const uint8_t* input, double* llrs);

#endif /* LABRADOR_LDPC_CAPI */
