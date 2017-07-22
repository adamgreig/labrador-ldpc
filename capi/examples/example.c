#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdbool.h>

#include "labrador_ldpc.h"

int main() {

    enum labrador_ldpc_code code = LABRADOR_LDPC_CODE_TC128;

    printf("Code %d: n=%lu k=%lu\n", code,
           labrador_ldpc_code_n(code), labrador_ldpc_code_k(code));

    printf("Encoding\n");

    uint8_t data[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    uint8_t codeword[16];

    labrador_ldpc_copy_encode(code, data, codeword);

    printf("Codeword: ");
    for(int i=0; i<16; i++) {
        printf("%02X ", codeword[i]);
    }
    printf("\n\n");

    printf("Erasing last byte\n");
    codeword[15] = 0;
    printf("Codeword: ");
    for(int i=0; i<16; i++) {
        printf("%02X ", codeword[i]);
    }
    printf("\n\n");

    printf("Converting to LLRs\n");
    float llrs[128];
    labrador_ldpc_hard_to_llrs_f32(code, codeword, llrs);

    printf("Decoding\n");

    uint8_t *working_u8 = malloc(labrador_ldpc_ms_working_u8_len(code));
    float *working_f32 = malloc(labrador_ldpc_ms_working_len(code)
                                * sizeof(float));
    uint8_t *output = malloc(labrador_ldpc_output_len(code));
    size_t iters_run;

    bool result = labrador_ldpc_decode_ms_f32(code, llrs, output, working_f32,
                                              working_u8, 200, &iters_run);

    if(result) {
        printf("Decoding successful\n");
    } else {
        printf("Decoding failed\n");
    }

    printf("Codeword: ");
    for(int i=0; i<16; i++) {
        printf("%02X ", output[i]);
    }
    printf("\n");
}
