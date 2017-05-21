#![allow(dead_code)]

/*
 * Constants used to define the parity check matrices for the TC codes.
 *
 * This representation mirrors that in CCSDS 231.1-O-1, and is expanded
 * at runtime initialisation to create sparse-encoded parity check matrices.
 * Each constant represents a single MxM sub-matrix, where M=n/8.
 * * HZ: All-zero matrix
 * * HI: Identity matrix
 * * HP: Phi: nth right circular shift of I, with lower 5 bits for n
 * * HS: HI+HP
 */

pub const HZ: u8 = (0 << 6);
pub const HI: u8 = (1 << 6);
pub const HP: u8 = (2 << 6);
pub const HS: u8 = (HI | HP);

/// Compact parity matrix for the TC128 code
pub static TC128_H: [[u8; 8]; 4] = [
    [HS| 7, HP| 2, HP|14, HP| 6, HZ   , HP| 0, HP|13, HI   ],
    [HP| 6, HS|15, HP| 0, HP| 1, HI   , HZ   , HP| 0, HP| 7],
    [HP| 4, HP| 1, HS|15, HP|14, HP|11, HI   , HZ   , HP| 3],
    [HP| 0, HP| 1, HP| 9, HS|13, HP|14, HP| 1, HI   , HZ   ],
];

/// Compact parity matrix for the TC256 code
pub static TC256_H: [[u8; 8]; 4] = [
    [HS|31, HP|15, HP|25, HP| 0, HZ   , HP|20, HP|12, HI   ],
    [HP|28, HS|30, HP|29, HP|24, HI   , HZ   , HP| 1, HP|20],
    [HP| 8, HP| 0, HS|28, HP| 1, HP|29, HI   , HZ   , HP|21],
    [HP|18, HP|30, HP| 0, HS|30, HP|25, HP|26, HI   , HZ   ],
];

/// Compact parity matrix for the TC512 code
pub static TC512_H: [[u8; 8]; 4] = [
    [HS|63, HP|30, HP|50, HP|25, HZ   , HP|43, HP|62, HI   ],
    [HP|56, HS|61, HP|50, HP|23, HI   , HZ   , HP|37, HP|26],
    [HP|16, HP| 0, HS|55, HP|27, HP|56, HI   , HZ   , HP|43],
    [HP|35, HP|56, HP|62, HS|11, HP|58, HP| 3, HI   , HZ   ],
];

/* Parity check matrices corresponding to the TM codes.
 *
 * This representation mirrors the definition in CCSDS 131.0-B-1,
 * and can be expanded at runtime to create the actual matrix in memory.
 * Each const represents a single MxM sub-matrix, where M is a function
 * of the information block length and the rate:
 *
 * -----------------------------------------
 * |k     | Rate 1/2 | Rate 2/3 | Rate 4/5 |
 * -----------------------------------------
 * |1024  |      512 |      256 |      128 |
 * |4096  |     2048 |     1024 |      512 |
 * |16384 |     8192 |     4096 |     2048 |
 * -----------------------------------------
 *
 * The HP constant is now used for PI_K which goes via the lookup table below.
 * The HZ constant is an MxM zero block.
 * The HI macro an MxM identity, as previously.
 *
 * Each matrix is defined in three parts which are to be added together mod 2.
 *
 * Additionally the matrices for the higher rate codes are assumed to have the previous (lower)
 * rate's parity check matrix right-appended (forming a fatter matrix). For example,
 * H_2/3 = [... | H_1/2], as per section CCSDS 131.0-B-2 7.4.2.
 *
 * The PI_K function is:
 * pi_k(i) = M/4 (( theta_k + floor(4i / M)) mod 4) + (phi_k( floor(4i / M), M ) + i) mod M/4
 *
 * The sizes _should_ be 5x3x3 and 2x3x3 and 4x3x3 for rates 1/2, 2/3, and 4/5, respectively,
 * but this makes it an unholy pain to pass around to the parity expanding functions.
 * So for the sake of type safety we expand these to all be the same shape,
 * sacrificing 36 bytes.
 */

/// Compact parity matrix for the rate-1/2 TM codes
pub static TM_R12_H: [[[u8; 5]; 3]; 3] = [
    [
        [HZ   , HZ   , HI   , HZ   , HI   ],
        [HI   , HI   , HZ   , HI   , HP| 2],
        [HI   , HP| 5, HZ   , HP| 7, HI   ],
    ], [
        [0    , 0    , 0    , 0    , HP| 1],
        [0    , 0    , 0    , 0    , HP| 3],
        [0    , HP| 6, 0    , HP| 8, 0    ],
    ], [
        [0    , 0    , 0    , 0    , 0    ],
        [0    , 0    , 0    , 0    , HP| 4],
        [0    , 0    , 0    , 0    , 0    ],
    ]
];

/// Compact parity left submatrix for the rate-2/3 TM codes
pub static TM_R23_H: [[[u8; 5]; 3]; 3] = [
    [
        [HZ   , HZ   , 0, 0, 0],
        [HP| 9, HI   , 0, 0, 0],
        [HI   , HP|12, 0, 0, 0],
    ], [
        [0    , 0    , 0, 0, 0],
        [HP|10, 0    , 0, 0, 0],
        [0    , HP|13, 0, 0, 0],
    ], [
        [0    , 0    , 0, 0, 0],
        [HP|11, 0    , 0, 0, 0],
        [0    , HP|14, 0, 0, 0],
    ]
];

/// Compact parity left submatrix for the rate-4/5 TM codes
pub static TM_R45_H: [[[u8; 5]; 3]; 3] = [
    [
        [HZ   , HZ   , HZ   , HZ   , 0],
        [HP|21, HI   , HP|15, HI   , 0],
        [HI   , HP|24, HI   , HP|18, 0],
    ], [
        [0    , 0    , 0    , 0    , 0],
        [HP|22, 0    , HP|16, 0    , 0],
        [0    , HP|25, 0    , HP|19, 0],
    ], [
        [0    , 0    , 0    , 0    , 0],
        [HP|23, 0    , HP|17, 0    , 0],
        [0    , HP|26, 0    , HP|20, 0],
    ]
];


/// Theta constants. Looked up against (k-1) from k=1 to k=26.
pub static THETA_K: [u8; 26] = [3, 0, 1, 2, 2, 3, 0, 1, 0, 1, 2, 0, 2,
                            3, 0, 1, 2, 0, 1, 2, 0, 1, 2, 1, 2, 3];


/// Phi constants for M=128.
///
/// First index is j (from 0 to 3), second index is k-1 (from k=1 to k=26).
/// These are split up by M because we only need one set for each TM code, so the linker can throw
/// away the unused constants.
///
/// Ideally the first four would be u8 instead of u16 to save space, but this
/// complicates actually using them -- the monomorphised generic function
/// would likely take more text space than just making them all u16 for 104 extra bytes per code.
pub static PHI_J_K_M128: [[u16; 26]; 4] = [
    [1, 22, 0, 26, 0, 10, 5, 18, 3, 22, 3, 8, 25, 25, 2, 27, 7, 7, 15, 10, 4,
     19, 7, 9, 26, 17],
    [0, 27, 30, 28, 7, 1, 8, 20, 26, 24, 4, 12, 23, 15, 15, 22, 31, 3, 29, 21,
     2, 5, 11, 26, 9, 17],
    [0, 12, 30, 18, 10, 16, 13, 9, 7, 15, 16, 18, 4, 23, 5, 3, 29, 11, 4, 8, 2,
     11, 11, 3, 15, 13],
    [0, 13, 19, 14, 15, 20, 17, 4, 4, 11, 17, 20, 8, 22, 19, 15, 5, 21, 17, 9,
     20, 18, 31, 13, 2, 18],
];

/// Phi constants for M=256. See docs for PHI_J_K_M128.
pub static PHI_J_K_M256: [[u16; 26]; 4] = [
    [59, 18, 52, 23, 11, 7, 22, 25, 27, 30, 43, 14, 46, 62, 44, 12, 38, 47, 1,
     52, 61, 10, 55, 7, 12, 2],
    [0, 32, 21, 36, 30, 29, 44, 29, 39, 14, 22, 15, 48, 55, 39, 11, 1, 50, 40,
     62, 27, 38, 40, 15, 11, 18],
    [0, 46, 45, 27, 48, 37, 41, 13, 9, 49, 36, 10, 11, 18, 54, 40, 27, 35, 25,
     46, 24, 33, 18, 37, 35, 21],
    [0, 44, 51, 12, 15, 12, 4, 7, 2, 30, 53, 23, 29, 37, 42, 48, 4, 10, 18, 56,
     9, 11, 23, 8, 7, 24],
];

/// Phi constants for M=512. See docs for PHI_J_K_M128.
pub static PHI_J_K_M512: [[u16; 26]; 4] = [
    [16, 103, 105, 0, 50, 29, 115, 30, 92, 78, 70, 66, 39, 84, 79, 70, 29, 32,
     45, 113, 86, 1, 42, 118, 33, 126],
    [0, 53, 74, 45, 47, 0, 59, 102, 25, 3, 88, 65, 62, 68, 91, 70, 115, 31,
     121, 45, 56, 54, 108, 14, 30, 116],
    [0, 8, 119, 89, 31, 122, 1, 69, 92, 47, 11, 31, 19, 66, 49, 81, 96, 38, 83,
     42, 58, 24, 25, 92, 38, 120],
    [0, 35, 97, 112, 64, 93, 99, 94, 103, 91, 3, 6, 39, 113, 92, 119, 74, 73,
     116, 31, 127, 98, 23, 38, 18, 62],
];

/// Phi constants for M=1024. See docs for PHI_J_K_M128.
pub static PHI_J_K_M1024: [[u16; 26]; 4] = [
    [160, 241, 185, 251, 209, 103, 90, 184, 248, 12, 111, 66, 173, 42, 157,
     174, 104, 144, 43, 181, 250, 202, 68, 177, 170, 89],
    [0, 182, 249, 65, 70, 141, 237, 77, 55, 12, 227, 42, 52, 243, 179, 250,
     247, 164, 17, 31, 149, 105, 183, 153, 177, 19],
    [0, 35, 167, 214, 84, 206, 122, 67, 147, 54, 23, 93, 20, 197, 46, 162, 101,
     76, 78, 253, 124, 143, 63, 41, 214, 70],
    [0, 162, 7, 31, 164, 11, 237, 125, 133, 99, 105, 17, 97, 91, 211, 128, 82,
     115, 248, 62, 26, 140, 121, 12, 41, 249],
];

/// Phi constants for M=2048. See docs for PHI_J_K_M128.
pub static PHI_J_K_M2048: [[u16; 26]; 4] = [
    [108, 126, 238, 481, 96, 28, 59, 225, 323, 28, 386, 305, 34, 510, 147, 199,
     347, 391, 165, 414, 97, 158, 86, 168, 506, 489],
    [0, 375, 436, 350, 260, 84, 318, 382, 169, 213, 67, 313, 242, 188, 1, 306,
     397, 80, 33, 7, 447, 336, 424, 134, 152, 492],
    [0, 219, 16, 263, 415, 403, 184, 279, 198, 307, 432, 240, 454, 294, 479,
     289, 373, 104, 141, 270, 439, 333, 399, 14, 277, 412],
    [0, 312, 503, 388, 48, 7, 185, 328, 254, 202, 285, 11, 168, 127, 8, 437,
     475, 85, 419, 459, 468, 209, 311, 211, 510, 320],
];

/// Phi constants for M=4096. See docs for PHI_J_K_M128.
pub static PHI_J_K_M4096: [[u16; 26]; 4] = [
    [226, 618, 404, 32, 912, 950, 534, 63, 971, 304, 409, 708, 719, 176, 743,
     759, 674, 958, 984, 11, 413, 925, 687, 752, 867, 323],
    [0, 767, 227, 247, 284, 370, 482, 273, 886, 634, 762, 184, 696, 413, 854,
     544, 864, 82, 1009, 437, 36, 562, 816, 452, 290, 778],
    [0, 254, 790, 642, 248, 899, 328, 518, 477, 404, 698, 160, 497, 100, 518,
     92, 464, 592, 198, 856, 235, 134, 542, 545, 777, 483],
    [0, 285, 554, 809, 185, 49, 101, 82, 898, 627, 154, 65, 81, 823, 50, 413,
     462, 175, 715, 537, 722, 37, 488, 179, 430, 264],
];

/// Phi constants for M=8192. See docs for PHI_J_K_M128.
pub static PHI_J_K_M8192: [[u16; 26]; 4] = [
    [1148, 2032, 249, 1807, 485, 1044, 717, 873, 364, 1926, 1241, 1769, 532,
     768, 1138, 965, 141, 1527, 505, 1312, 1840, 709, 1427, 989, 1925, 270],
    [0, 1822, 203, 882, 1989, 957, 1705, 1083, 1072, 354, 1942, 446, 1456,
     1940, 1660, 1661, 587, 708, 1466, 433, 1345, 867, 1551, 2041, 1383, 1790],
    [0, 318, 494, 1467, 757, 1085, 1630, 64, 689, 1300, 148, 777, 1431, 659,
     352, 1177, 836, 1572, 348, 1040, 779, 476, 191, 1393, 1752, 1627],
    [0, 1189, 458, 460, 1039, 1000, 1265, 1223, 874, 1292, 1491, 631, 464, 461,
     844, 392, 922, 256, 1986, 19, 266, 471, 1166, 1300, 1033, 1606]
];
