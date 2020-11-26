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
 *
 * For further details refer to:
 * https://docs.rs/labrador-ldpc/1.0.1/labrador_ldpc/codes/enum.LDPCCode.html
 */
enum labrador_ldpc_code {
    LABRADOR_LDPC_CODE_TC128    = 0,
    LABRADOR_LDPC_CODE_TC256    = 1,
    LABRADOR_LDPC_CODE_TC512    = 2,
    LABRADOR_LDPC_CODE_TM1280   = 3,
    LABRADOR_LDPC_CODE_TM1536   = 4,
    LABRADOR_LDPC_CODE_TM2048   = 5,
    LABRADOR_LDPC_CODE_TM5120   = 6,
    LABRADOR_LDPC_CODE_TM6144   = 7,
    LABRADOR_LDPC_CODE_TM8192   = 8,
};

/* Useful constants for each code, for statically allocating required memory.
 *
 * Each can be accessed directly as `LABRADOR_LDPC_N_TC512`, or with the
 * code as a constant parameter, as `LABRADOR_LDPC_N(TC512)`, or with the code
 * as another define, for example:
 * #define CODE TC512
 * int n = LABRADOR_LDPC_N(CODE);
 *
 * For further details refer to:
 * https://docs.rs/labrador-ldpc/1.0.1/labrador_ldpc/codes/struct.CodeParams.html
 */
#define LABRADOR_LDPC_CODE_(CODE) LABRADOR_LDPC_CODE_##CODE
#define LABRADOR_LDPC_CODE(CODE)  LABRADOR_LDPC_CODE_(CODE)

#define LABRADOR_LDPC_N_TC128  (128)
#define LABRADOR_LDPC_N_TC256  (256)
#define LABRADOR_LDPC_N_TC512  (512)
#define LABRADOR_LDPC_N_TM1280 (1280)
#define LABRADOR_LDPC_N_TM1536 (1536)
#define LABRADOR_LDPC_N_TM2048 (2048)
#define LABRADOR_LDPC_N_TM5120 (5120)
#define LABRADOR_LDPC_N_TM6144 (6140)
#define LABRADOR_LDPC_N_TM8192 (8192)
#define LABRADOR_LDPC_N_(CODE) LABRADOR_LDPC_N_##CODE
#define LABRADOR_LDPC_N(CODE)  LABRADOR_LDPC_N_(CODE)

#define LABRADOR_LDPC_K_TC128  (64)
#define LABRADOR_LDPC_K_TC256  (128)
#define LABRADOR_LDPC_K_TC512  (256)
#define LABRADOR_LDPC_K_TM1280 (1024)
#define LABRADOR_LDPC_K_TM1536 (1024)
#define LABRADOR_LDPC_K_TM2048 (1024)
#define LABRADOR_LDPC_K_TM5120 (4096)
#define LABRADOR_LDPC_K_TM6144 (4096)
#define LABRADOR_LDPC_K_TM8192 (4096)
#define LABRADOR_LDPC_K_(CODE) LABRADOR_LDPC_K_##CODE
#define LABRADOR_LDPC_K(CODE)  LABRADOR_LDPC_K_(CODE)

#define LABRADOR_LDPC_BF_WORKING_LEN_TC128  (128)
#define LABRADOR_LDPC_BF_WORKING_LEN_TC256  (256)
#define LABRADOR_LDPC_BF_WORKING_LEN_TC512  (512)
#define LABRADOR_LDPC_BF_WORKING_LEN_TM1280 (1408)
#define LABRADOR_LDPC_BF_WORKING_LEN_TM1536 (1792)
#define LABRADOR_LDPC_BF_WORKING_LEN_TM2048 (2560)
#define LABRADOR_LDPC_BF_WORKING_LEN_TM5120 (5632)
#define LABRADOR_LDPC_BF_WORKING_LEN_TM6140 (7168)
#define LABRADOR_LDPC_BF_WORKING_LEN_TM8192 (10240)
#define LABRADOR_LDPC_BF_WORKING_LEN_(CODE) LABRADOR_LDPC_BF_WORKING_LEN_##CODE
#define LABRADOR_LDPC_BF_WORKING_LEN(CODE)  LABRADOR_LDPC_BF_WORKING_LEN_(CODE)

#define LABRADOR_LDPC_MS_WORKING_LEN_TC128  (1280)
#define LABRADOR_LDPC_MS_WORKING_LEN_TC256  (2560)
#define LABRADOR_LDPC_MS_WORKING_LEN_TC512  (5120)
#define LABRADOR_LDPC_MS_WORKING_LEN_TM1280 (12160)
#define LABRADOR_LDPC_MS_WORKING_LEN_TM1536 (15104)
#define LABRADOR_LDPC_MS_WORKING_LEN_TM2048 (20992)
#define LABRADOR_LDPC_MS_WORKING_LEN_TM5120 (48640)
#define LABRADOR_LDPC_MS_WORKING_LEN_TM6140 (60416)
#define LABRADOR_LDPC_MS_WORKING_LEN_TM8192 (83968)
#define LABRADOR_LDPC_MS_WORKING_LEN_(CODE) LABRADOR_LDPC_MS_WORKING_LEN_##CODE
#define LABRADOR_LDPC_MS_WORKING_LEN(CODE)  LABRADOR_LDPC_MS_WORKING_LEN_(CODE)

#define LABRADOR_LDPC_MS_WORKING_U8_LEN_TC128  (8)
#define LABRADOR_LDPC_MS_WORKING_U8_LEN_TC256  (16)
#define LABRADOR_LDPC_MS_WORKING_U8_LEN_TC512  (32)
#define LABRADOR_LDPC_MS_WORKING_U8_LEN_TM1280 (48)
#define LABRADOR_LDPC_MS_WORKING_U8_LEN_TM1536 (96)
#define LABRADOR_LDPC_MS_WORKING_U8_LEN_TM2048 (192)
#define LABRADOR_LDPC_MS_WORKING_U8_LEN_TM5120 (192)
#define LABRADOR_LDPC_MS_WORKING_U8_LEN_TM6140 (384)
#define LABRADOR_LDPC_MS_WORKING_U8_LEN_TM8192 (768)
#define LABRADOR_LDPC_MS_WORKING_U8_LEN_(CODE) LABRADOR_LDPC_MS_WORKING_U8_LEN_##CODE
#define LABRADOR_LDPC_MS_WORKING_U8_LEN(CODE)  LABRADOR_LDPC_MS_WORKING_U8_LEN_(CODE)

#define LABRADOR_LDPC_OUTPUT_LEN_TC128  (16)
#define LABRADOR_LDPC_OUTPUT_LEN_TC256  (32)
#define LABRADOR_LDPC_OUTPUT_LEN_TC512  (64)
#define LABRADOR_LDPC_OUTPUT_LEN_TM1280 (176)
#define LABRADOR_LDPC_OUTPUT_LEN_TM1536 (224)
#define LABRADOR_LDPC_OUTPUT_LEN_TM2048 (320)
#define LABRADOR_LDPC_OUTPUT_LEN_TM5120 (704)
#define LABRADOR_LDPC_OUTPUT_LEN_TM6140 (896)
#define LABRADOR_LDPC_OUTPUT_LEN_TM8192 (1280)
#define LABRADOR_LDPC_OUTPUT_LEN_(CODE) LABRADOR_LDPC_OUTPUT_LEN_##CODE
#define LABRADOR_LDPC_OUTPUT_LEN(CODE)  LABRADOR_LDPC_OUTPUT_LEN_(CODE)

/* Returns the code length n (number of codeword bits) for a given code. */
size_t labrador_ldpc_code_n(enum labrador_ldpc_code code);

/* Returns the code dimension k (number of data bits) for a given code. */
size_t labrador_ldpc_code_k(enum labrador_ldpc_code code);

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

/* Encode the first k/8 bytes of `codeword` into the rest of `codeword`,
 * using the `code` LDPC code.
 *
 * If `codeword` is 4-byte aligned, encoding is performed 32 bits at a time,
 * which is usually faster than byte at a time.
 */
void labrador_ldpc_encode(enum labrador_ldpc_code code, uint8_t *codeword);

/* Encode all of `data` (k/8 bytes long) into `codeword` (n/8 bytes long),
 * first copying `data` into `codeword`, using the `code` LDPC code.
 *
 * If `codeword` is 4-byte aligned, encoding is performed 32 bits at a time,
 * which is usually faster than byte at a time.
 */
void labrador_ldpc_copy_encode(enum labrador_ldpc_code code,
                               const uint8_t* data, uint8_t* codeword);

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
 * https://docs.rs/labrador-ldpc/1.0.1/labrador_ldpc/codes/enum.LDPCCode.html,
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

/* Convert LLRs into hard information.
 *
 * Assumes positive numbers are more likely to be 0 bits.
 *
 * `llrs` must be n entries long.
 * `output` must be n/8 long.
 *
 * Available in four variants for different LLR types.
 */
void labrador_ldpc_llrs_to_hard_i8(enum labrador_ldpc_code code,
                                   const int8_t* llrs, uint8_t* output);
void labrador_ldpc_llrs_to_hard_i16(enum labrador_ldpc_code code,
                                    const int16_t* llrs, uint8_t* output);
void labrador_ldpc_llrs_to_hard_f32(enum labrador_ldpc_code code,
                                    const float* llrs, uint8_t* output);
void labrador_ldpc_llrs_to_hard_f64(enum labrador_ldpc_code code,
                                    const double* llrs, uint8_t* output);

#endif /* LABRADOR_LDPC_CAPI */
