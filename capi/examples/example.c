/* Example use of Labrador-LDPC C API.
 * Copyright 2017 Adam Greig.
 * Licensed under the MIT license, see LICENSE for details.
 */

/* Build me like this:
 * gcc example.c -I ../include -o example -L../target/release -llabrador_ldpc
 */

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "labrador_ldpc.h"

/* We can statically allocate all required memory using these macros.
 * You could also allocate dynamically using labrador_ldpc_code_n() etc
 * to find the required sizes.
 * You can just change `CODE` here to any other supported code and
 * everything else will adapt appropriately.
 */
#define CODE TC128
uint8_t message[LABRADOR_LDPC_K(CODE)/8];
uint8_t codeword[LABRADOR_LDPC_N(CODE)/8];
float llrs[LABRADOR_LDPC_N(CODE)];
float working[LABRADOR_LDPC_MS_WORKING_LEN(CODE)];
uint8_t working_u8[LABRADOR_LDPC_MS_WORKING_U8_LEN(CODE)];
uint8_t output[LABRADOR_LDPC_OUTPUT_LEN(CODE)];

int main() {

    /* We can either pick a code directly:
     * enum labrador_ldpc_code code = LABRADOR_LDPC_CODE_TC128
     * or we can use this macro to look up a constant defined earlier:
     */
    enum labrador_ldpc_code code = LABRADOR_LDPC_CODE(CODE);
    size_t code_n = labrador_ldpc_code_n(code);
    size_t code_k = labrador_ldpc_code_k(code);
    printf("Code %d: n=%lu k=%lu\n", code, code_n, code_k);

    /* Make up a message to encode, in this case 0, 1, 2, ..., k/8 */
    printf("Encoding\n");
    for(int i=0; i<code_k/8; i++) {
        message[i] = i;
    }

    /* Copy the message into the codeword and encode it.
     * We could instead have just filled `codeword` with the data,
     * and then called `labrador_ldpc_encode` directly.
     */
    labrador_ldpc_copy_encode(code, message, codeword);

    printf("Codeword:\n");
    for(int i=0; i<code_n/8; i++) {
        printf("%02X ", codeword[i]);
    }
    printf("\n\n");

    /* Now we'll simulate a corruption by erasing the entire last byte
     * of user information.
     */
    printf("Erasing last byte\n");
    codeword[code_k/8 - 1] = 0;
    printf("Corrupted codeword:\n");
    for(int i=0; i<code_n/8; i++) {
        printf("%02X ", codeword[i]);
    }
    printf("\n\n");

    /* The MS (min-sum) decoder requires soft inputs, but since we haven't
     * simulated a real channel, we only have hard information. We can use
     * this function to convert the hard information to appropriate LLRs.
     */
    printf("Converting to LLRs\n\n");
    labrador_ldpc_hard_to_llrs_f32(code, codeword, llrs);

    /* Now we'll run the MS decoder for at most 200 iters. */
    printf("Decoding\n");
    size_t iters_run;
    bool result = labrador_ldpc_decode_ms_f32(code, llrs, output, working,
                                              working_u8, 200, &iters_run);

    if(result) {
        printf("Decoding successful\n");
    } else {
        printf("Decoding failed\n");
    }

    printf("Decoded Codeword:\n");
    for(int i=0; i<code_n/8; i++) {
        printf("%02X ", output[i]);
    }
    printf("\n");
}
