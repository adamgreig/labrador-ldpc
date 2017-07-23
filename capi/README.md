# Labrador-LDPC C API

This crate contains a C API for the Labrador LDPC library, which is written in 
Rust. The C API compiles to a small static library you can link in to existing
C code as though it were a C library, suitable for use on embedded and regular
systems.

Labrador-LDPC is a library for encoding and decoding the telemetry and 
telecommand LDPC codes specified by the CCSDS.

For the main documentation of the underlying LDPC library, please refer to
the [main documentation](https://docs.rs/labrador-ldpc).


## Quick Example
```c
    /* Choose a code, here the "telecommand (128, 64)" code.
     * See the main documentation for the list of all available codes.
     */
    enum labrador_ldpc_code code = LABRADOR_LDPC_CODE_TC128;

    /* Make up some message to encode. 8 is k/8 for this code.
     * You can also get 8 from `LABRADOR_LDPC_CODE_K(TC128)/8`,
     * or from `labrador_ldpc_code_k(code)/8`
     */
    uint8_t message[8] = {0, 1, 2, 3, 4, 5, 6, 7};

    /* Allocate space for the codeword, n/8 bytes long. */
    uint8_t codeword[16];

    /* Encode the message to an LDPC codeword. */
    labrador_ldpc_copy_encode(code, message, codeword);

    /* We'll zero out a whole byte to give the decoder something to do. */
    codeword[5] = 0;

    /* Allocate space for the decoder. We'll use the BF (bit-flipping)
     * decoder, which is lower performance but faster and lower overhead.
     * See `examples/example.c` for an example using the MS (min-sum) decoder.
     * The 128 length here is `LABRADOR_LDPC_BF_WORKING_LEN(TC128)`, or
     * `labrador_ldpc_bf_working_len(code)`.
     */
    uint8_t working[128];

    /* Allocate space for the output decoded codeword. */
    uint8_t decoded[16];

    /* Run the decoder. Returns true on success. */
    labrador_ldpc_decode_bf(code, codeword, decoded, working, 50, NULL);
```

## Building the C API

You will need Rust installed. Try [rustup.rs](https://rustup.rs/). Building 
should then be as simple as:
```
cargo build --release
```

The generated library is placed in `target/release/liblabrador_ldpc.a`.

Xargo is recommended for building cross platform, for example:
```
xargo build --release --target thumbv7em-none-eabihf
```

## Using the C API in your project

Include the header file, `include/labrador_ldpc.h`, in your project.

You should be able to link like any other static library, for example to build
the example program (in `examples/example.c`):
```
gcc examples/example.c -I include -L target/release -llabrador_ldpc -o example
```

### Memory Allocation

Labrador-LDPC does not itself perform any dynamic allocation, so all memory
required must be provided already allocated by the user, either statically or
dynamically.

The amount of memory required depends on the LDPC code in use. There are tables
in the main library documentation detailing the exact numbers for each code,
and methods are provided to obtain these numbers both
statically and dynamically for a given code:

* Dynamically, use the functions `labrador_ldpc_code_n(code)`,
  `labrador_ldpc_code_k(code)`, etc, to find the required sizes.

* Statically, use the macros `LABRADOR_LDPC_N(TC128)`, etc, to find
  the required sizes. You can also use `LABRADOR_LDPC_CODE(TC128)` to
  obtain the `enum labrador_ldpc_code` for a given code name.
  See `examples/example.c` for a completely statically allocated example.

Each function in `labrador_ldpc.h` has a comment describing the required
lengths of each parameter. Encoding only requires enough memory to store the
encoded codeword, while the decoders all have some working area requirement.

Additionally there is a fixed read-only data requirement for code-related 
constants which weighs in at around 15kB. In the future it might be possible
to enable the linker to strip unused constants, but otherwise you are 
recommended to use the Rust library directly.

### Encoding

Two encoding functions are provided:

* `labrador_ldpc_encode(code, uint8_t *codeword)` takes `codeword` with the 
  first k bits already set to your message, and fills in the remaining
  bits (n in total) with the LDPC codeword.
* `labrador_ldpc_copy_encode(code, uint8_t *data, uint8_t *codeword)` first
  copies `data` (k bits long) into `codeword`, then proceeds to encode.

The `copy_encode` function is just a convenience wrapper if you happen to have
the message already in a smaller buffer or you wish to reuse the message buffer 
etc.

### Decoding

Two decoders are provided, `bf` (bit-flipping) and `ms` (min-sum). Please refer
to the main documentation for a detailed comparison. The C API for them is:

* `bool labrador_ldpc_decode_bf(code, uint8_t* input, uint8_t* output,
  uint8_t* working, size_t max_iters, size_t *iters_run)`, which decodes
  received noisy data from `input` into `output`, using a provided working area
  and running for the given maximum iteration count. The number of iterations
  actually run is written to `iters_run` if it is not NULL. Returns true if
  decoding succeeded.
* `bool labrador_ldpc_decode_ms_T(code, T* llrs, uint8_t* output, T* working,
  uint8_t* working_u8, size_t max_iters, size_t* iters_run)`, which decodes
  received log-likelihood ratios (LLRs) from `llrs` into `output`. The type
  of `llrs` can be `int8_t`, `int16_t`, `float`, or `double`. There are two
  required working areas, one of the same type as `llrs` and one `uint8_t`.
  The decoder runs for the given maximum iteration count. The number of
  iterations actually run is written to `iters_run` if it is not NULL. Returns
  true if decoding succeeded.

Two helper functions for decoding are provided, to convert between
hard data (packed `uint8_t` where each bit is the best-estimate of
that received bit) and log likelihood ratios (a probabilistic measure
of each bit being 0 or 1):

* `labrador_ldpc_hard_to_llrs_T(code, uint8_t input, T* llrs)` converts
  hard information from `input` into LLRs in `llrs`. `T` may be `int8_t`,
  `int16_t`, `float`, or `double`.
* `labrador_ldpc_llrs_to_hard_T(code, T* llrs, uint8_t output)` converts
  LLRs from `llrs` into hard information in `output`. `T` may be `int8_t`,
  `int16_t`, `float`, or `double`.

See the short example above for `decode_bf` usage, and `examples/example.c`
for `decode_ms` usage.
