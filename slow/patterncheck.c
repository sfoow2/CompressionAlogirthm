#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <immintrin.h>
#include <time.h>
#include <pthread.h>

#define INPUT_PATH "C:\\Users\\lukac\\Documents\\compressor\\compresseddata.bin"
#define N 64

#define NUM_CHUNKS 25
#define HEADER_SKIP 64

#define SEED_MIN 1u
#define SEED_MAX 0xFFFFFFu   /* 3-byte seed range: 1..16777215 */

/* ones (0..8) -> max(ones, 8-ones): the majority-vote score for one phase.
 * A real lookup table instead of a branch/ternary each time. */
static const int MAXTAB[9] = {8,7,6,5,4,5,6,7,8};

/* binary entropy in bits: H(0.5)=1 (no compressibility), H(1.0)=0 bits
 * (perfectly predictable, costs nothing). Used for entropy-corrected
 * profit = csize*(1-H(p)) - overhead, which is 0 at p=0.5 unlike the
 * naive accuracy*(csize-overhead) formula. */
static double binary_entropy(double p) {
    if (p <= 0.0 || p >= 1.0) return 0.0;
    return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
}

/* hash_byte(i, seed): same murmur3-style finalizer used as HASH_XOR in
 * hash_cpu_ref.c. ks[i] depends only on (i, seed) directly -- no state
 * is carried from ks[i-1] to ks[i], unlike a PRNG chain. That means
 * every one of the 64 iterations is independent (only the phase_acc
 * accumulators are loop-carried, and those are cheap associative
 * adds), so the CPU can pipeline consecutive iterations for free --
 * this is what makes it faster than xorshift here, not just a
 * different keystream. */
static inline uint32_t hash_mix(uint32_t x) {
    x ^= x >> 16; x *= 0x85EBCA6Bu;
    x ^= x >> 13; x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return x;
}
static inline uint8_t hash_byte(uint32_t i, uint32_t seed) {
    return (uint8_t)hash_mix(i ^ (seed * 0x9E3779B1u));
}

/* Search all 3-byte seeds for the one whose MSB stream best matches
 * some repeating period-8 pattern. Since the pattern repeats every 8
 * bytes and N=64 is a multiple of 8, byte i's contribution depends
 * only on i%8 ("phase"), so the best pattern for a given seed is just
 * the per-phase majority bit -- no need to test all 256 patterns
 * explicitly.
 *
 * Scalar reference implementation -- kept only to verify the AVX-512
 * version below produces byte-identical results. Not used for the
 * real (large) chunk scan. */
static void search_chunk_scalar(const unsigned char *buf,
                                 uint32_t *out_seed, uint8_t *out_pattern, int *out_matches) {
    int best_matches = -1;
    uint32_t best_seed = 0;
    int best_phase_ones[8] = {0};

    for (uint32_t seed = SEED_MIN; seed <= SEED_MAX; seed++) {
        int phase_ones[8] = {0};
        for (int i = 0; i < N; i++) {
            uint8_t ks = hash_byte((uint32_t)i, seed);
            uint8_t nv = (uint8_t)(buf[i] + ks);
            phase_ones[i & 7] += (nv >> 7) & 1;
        }

        int matches = 0;
        for (int j = 0; j < 8; j++) matches += MAXTAB[phase_ones[j]];

        if (matches > best_matches) {
            best_matches = matches;
            best_seed = seed;
            for (int j = 0; j < 8; j++) best_phase_ones[j] = phase_ones[j];
        }
    }

    uint8_t pattern = 0;
    for (int j = 0; j < 8; j++) {
        if (best_phase_ones[j] >= 4) pattern |= (1u << (7 - j));
    }

    *out_seed = best_seed;
    *out_pattern = pattern;
    *out_matches = best_matches;
}

/* AVX-512 version: identical algorithm to search_chunk_scalar, but
 * tests 16 seeds at once by packing them into the 16 lanes of a
 * 512-bit vector (this CPU has AVX512F/BW/VL). All ops (shift, xor,
 * multiply, add, and) exist as direct per-lane integer intrinsics, so
 * each vector instruction does the work of 16 scalar instructions --
 * same single thread, wider ALU op, no OS threads involved.
 *
 * Caching on top of the raw width increase:
 *  - buf[i] and the loop index i are each broadcast into a vector once
 *    up front (128 total) instead of re-broadcast inside the
 *    ~1M-iteration group loop.
 *  - seed*golden is the same for all 64 bytes of a group, so it's
 *    computed once per group of 16 seeds, not once per byte.
 *  - the majority-vote score (ones -> max(ones,8-ones)) is computed
 *    arithmetically in-vector (sub + max) and reduced with the native
 *    AVX-512 horizontal-max instruction, so the expensive per-lane
 *    MAXTAB lookup + array store only happens on the rare group that
 *    actually beats the current best, not on every one of the ~1M
 *    groups. */
static void search_chunk_avx512(const unsigned char *buf,
                                 uint32_t *out_seed, uint8_t *out_pattern, int *out_matches) {
    __m512i buf_bcast[N];
    __m512i i_bcast[N];
    for (int i = 0; i < N; i++) {
        buf_bcast[i] = _mm512_set1_epi32(buf[i]);
        i_bcast[i]   = _mm512_set1_epi32(i);
    }
    const __m512i mask_ff  = _mm512_set1_epi32(0xFF);
    const __m512i eight    = _mm512_set1_epi32(8);
    const __m512i c_golden = _mm512_set1_epi32((int)0x9E3779B1u);
    const __m512i c_m1     = _mm512_set1_epi32((int)0x85EBCA6Bu);
    const __m512i c_m2     = _mm512_set1_epi32((int)0xC2B2AE35u);

    int best_matches = -1;
    uint32_t best_seed = 0;
    int best_phase_ones[8] = {0};

    uint32_t group_count = (SEED_MAX - SEED_MIN + 1) / 16; /* full groups of 16 seeds */

    for (uint32_t g = 0; g < group_count; g++) {
        uint32_t base = SEED_MIN + g * 16;
        __m512i seed_v = _mm512_setr_epi32((int)(base),   (int)(base+1),  (int)(base+2),  (int)(base+3),
                                            (int)(base+4), (int)(base+5),  (int)(base+6),  (int)(base+7),
                                            (int)(base+8), (int)(base+9),  (int)(base+10), (int)(base+11),
                                            (int)(base+12),(int)(base+13), (int)(base+14), (int)(base+15));
        __m512i seed_mul = _mm512_mullo_epi32(seed_v, c_golden);

        __m512i phase_acc[8];
        for (int j = 0; j < 8; j++) phase_acc[j] = _mm512_setzero_si512();

        for (int i = 0; i < N; i++) {
            __m512i x = _mm512_xor_si512(i_bcast[i], seed_mul);
            x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
            x = _mm512_mullo_epi32(x, c_m1);
            x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 13));
            x = _mm512_mullo_epi32(x, c_m2);
            x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));

            __m512i ks  = _mm512_and_si512(x, mask_ff);
            __m512i sum = _mm512_add_epi32(ks, buf_bcast[i]);
            __m512i nv  = _mm512_and_si512(sum, mask_ff);
            __m512i bit = _mm512_srli_epi32(nv, 7); /* nv < 256, so this is 0 or 1 */

            phase_acc[i & 7] = _mm512_add_epi32(phase_acc[i & 7], bit);
        }

        __m512i matches_vec = _mm512_setzero_si512();
        for (int j = 0; j < 8; j++) {
            __m512i ones  = phase_acc[j];
            __m512i zeros = _mm512_sub_epi32(eight, ones);
            __m512i m     = _mm512_max_epi32(ones, zeros);
            matches_vec   = _mm512_add_epi32(matches_vec, m);
        }
        int group_best = _mm512_reduce_max_epi32(matches_vec);

        if (group_best > best_matches) {
            int lanes[8][16]; /* [phase][lane] */
            for (int j = 0; j < 8; j++) _mm512_storeu_si512((void *)lanes[j], phase_acc[j]);
            int match_lane[16];
            _mm512_storeu_si512((void *)match_lane, matches_vec);

            for (int lane = 0; lane < 16; lane++) {
                if (match_lane[lane] == group_best) {
                    best_matches = group_best;
                    best_seed = base + (uint32_t)lane;
                    for (int j = 0; j < 8; j++) best_phase_ones[j] = lanes[j][lane];
                    break; /* lowest-seed lane wins ties, same as scalar version */
                }
            }
        }
    }

    /* tail seeds not covered by a full group of 16, handled scalar */
    for (uint32_t seed = SEED_MIN + group_count * 16; seed <= SEED_MAX; seed++) {
        int phase_ones[8] = {0};
        for (int i = 0; i < N; i++) {
            uint8_t ks = hash_byte((uint32_t)i, seed);
            uint8_t nv = (uint8_t)(buf[i] + ks);
            phase_ones[i & 7] += (nv >> 7) & 1;
        }
        int matches = 0;
        for (int j = 0; j < 8; j++) matches += MAXTAB[phase_ones[j]];
        if (matches > best_matches) {
            best_matches = matches;
            best_seed = seed;
            for (int j = 0; j < 8; j++) best_phase_ones[j] = phase_ones[j];
        }
    }

    uint8_t pattern = 0;
    for (int j = 0; j < 8; j++) {
        if (best_phase_ones[j] >= 4) pattern |= (1u << (7 - j));
    }

    *out_seed = best_seed;
    *out_pattern = pattern;
    *out_matches = best_matches;
}

/* Same search as search_chunk_avx512, but run once across all
 * num_chunks chunks simultaneously. hash_byte(i, seed) doesn't depend
 * on the chunk's data -- only on (i, seed) -- so calling
 * search_chunk_avx512 once per chunk recomputes the identical
 * keystream num_chunks times over. Here the loop is inverted: for
 * each group of 16 seeds, the 64-byte keystream is computed exactly
 * once (cached in ks_vec, 64 * 64 bytes = 4KB, well inside L1) and
 * then reused for all num_chunks chunks before moving to the next
 * seed group. The expensive part (2 multiplies/byte in the hash) runs
 * 1x per seed instead of num_chunks times; only the cheap per-chunk
 * evaluation (add, mask, extract, accumulate) still runs num_chunks
 * times. buf[i] is also broadcast once per chunk up front instead of
 * inside the ~1M-iteration seed loop. */
static void search_all_chunks_avx512(const unsigned char (*chunks)[N], int num_chunks,
                                      uint32_t *out_seed, uint8_t *out_pattern, int *out_matches) {
    __m512i (*buf_bcast)[N] = malloc(sizeof(__m512i[N]) * (size_t)num_chunks);
    __m512i i_bcast[N];
    for (int i = 0; i < N; i++) i_bcast[i] = _mm512_set1_epi32(i);
    for (int ck = 0; ck < num_chunks; ck++)
        for (int i = 0; i < N; i++)
            buf_bcast[ck][i] = _mm512_set1_epi32(chunks[ck][i]);

    const __m512i mask_ff  = _mm512_set1_epi32(0xFF);
    const __m512i eight    = _mm512_set1_epi32(8);
    const __m512i c_golden = _mm512_set1_epi32((int)0x9E3779B1u);
    const __m512i c_m1     = _mm512_set1_epi32((int)0x85EBCA6Bu);
    const __m512i c_m2     = _mm512_set1_epi32((int)0xC2B2AE35u);

    for (int ck = 0; ck < num_chunks; ck++) { out_matches[ck] = -1; out_seed[ck] = 0; }
    int (*best_phase_ones)[8] = calloc((size_t)num_chunks, sizeof(int[8]));

    uint32_t group_count = (SEED_MAX - SEED_MIN + 1) / 16;

    for (uint32_t g = 0; g < group_count; g++) {
        uint32_t base = SEED_MIN + g * 16;
        __m512i seed_v = _mm512_setr_epi32((int)(base),   (int)(base+1),  (int)(base+2),  (int)(base+3),
                                            (int)(base+4), (int)(base+5),  (int)(base+6),  (int)(base+7),
                                            (int)(base+8), (int)(base+9),  (int)(base+10), (int)(base+11),
                                            (int)(base+12),(int)(base+13), (int)(base+14), (int)(base+15));
        __m512i seed_mul = _mm512_mullo_epi32(seed_v, c_golden);

        /* keystream computed once for this seed group, reused for every chunk below */
        __m512i ks_vec[N];
        for (int i = 0; i < N; i++) {
            __m512i x = _mm512_xor_si512(i_bcast[i], seed_mul);
            x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
            x = _mm512_mullo_epi32(x, c_m1);
            x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 13));
            x = _mm512_mullo_epi32(x, c_m2);
            x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
            ks_vec[i] = _mm512_and_si512(x, mask_ff);
        }

        for (int ck = 0; ck < num_chunks; ck++) {
            __m512i phase_acc[8];
            for (int j = 0; j < 8; j++) phase_acc[j] = _mm512_setzero_si512();

            for (int i = 0; i < N; i++) {
                __m512i sum = _mm512_add_epi32(ks_vec[i], buf_bcast[ck][i]);
                __m512i nv  = _mm512_and_si512(sum, mask_ff);
                __m512i bit = _mm512_srli_epi32(nv, 7);
                phase_acc[i & 7] = _mm512_add_epi32(phase_acc[i & 7], bit);
            }

            __m512i matches_vec = _mm512_setzero_si512();
            for (int j = 0; j < 8; j++) {
                __m512i ones  = phase_acc[j];
                __m512i zeros = _mm512_sub_epi32(eight, ones);
                __m512i m     = _mm512_max_epi32(ones, zeros);
                matches_vec   = _mm512_add_epi32(matches_vec, m);
            }
            int group_best = _mm512_reduce_max_epi32(matches_vec);

            if (group_best > out_matches[ck]) {
                int lanes[8][16];
                for (int j = 0; j < 8; j++) _mm512_storeu_si512((void *)lanes[j], phase_acc[j]);
                int match_lane[16];
                _mm512_storeu_si512((void *)match_lane, matches_vec);

                for (int lane = 0; lane < 16; lane++) {
                    if (match_lane[lane] == group_best) {
                        out_matches[ck] = group_best;
                        out_seed[ck] = base + (uint32_t)lane;
                        for (int j = 0; j < 8; j++) best_phase_ones[ck][j] = lanes[j][lane];
                        break;
                    }
                }
            }
        }
    }

    /* tail seeds not covered by a full group of 16, handled scalar, all chunks */
    for (uint32_t seed = SEED_MIN + group_count * 16; seed <= SEED_MAX; seed++) {
        uint8_t ks[N];
        for (int i = 0; i < N; i++) ks[i] = hash_byte((uint32_t)i, seed);

        for (int ck = 0; ck < num_chunks; ck++) {
            int phase_ones[8] = {0};
            for (int i = 0; i < N; i++) {
                uint8_t nv = (uint8_t)(chunks[ck][i] + ks[i]);
                phase_ones[i & 7] += (nv >> 7) & 1;
            }
            int matches = 0;
            for (int j = 0; j < 8; j++) matches += MAXTAB[phase_ones[j]];
            if (matches > out_matches[ck]) {
                out_matches[ck] = matches;
                out_seed[ck] = seed;
                for (int j = 0; j < 8; j++) best_phase_ones[ck][j] = phase_ones[j];
            }
        }
    }

    for (int ck = 0; ck < num_chunks; ck++) {
        uint8_t pattern = 0;
        for (int j = 0; j < 8; j++) {
            if (best_phase_ones[ck][j] >= 4) pattern |= (1u << (7 - j));
        }
        out_pattern[ck] = pattern;
    }

    free(buf_bcast);
    free(best_phase_ones);
}

/* --- variable chunk-size search, for the profit-vs-chunk-size sweep ---
 * Same algorithm as search_all_chunks_avx512, generalized so the chunk
 * size (csize, must be a multiple of 8) is a runtime parameter instead
 * of the fixed N=64. The pattern period is still 8 bits, but the
 * number of repeats per phase is now csize/8 instead of a hardcoded 8,
 * so the majority-vote score is max(ones, repeats-ones) rather than
 * the fixed MAXTAB[9]. Chunks are passed flattened (chunks_flat[ck*csize+i])
 * since a variable-length row size can't be expressed as a plain
 * pointer-to-array type in C. */
static inline int phase_score(int ones, int repeats) {
    int zeros = repeats - ones;
    return ones >= zeros ? ones : zeros;
}

static void search_chunk_scalar_n(const unsigned char *buf, int csize,
                                   uint32_t *out_seed, uint8_t *out_pattern, int *out_matches) {
    int repeats = csize / 8;
    int best_matches = -1;
    uint32_t best_seed = 0;
    int best_phase_ones[8] = {0};

    for (uint32_t seed = SEED_MIN; seed <= SEED_MAX; seed++) {
        int phase_ones[8] = {0};
        for (int i = 0; i < csize; i++) {
            uint8_t ks = hash_byte((uint32_t)i, seed);
            uint8_t nv = (uint8_t)(buf[i] + ks);
            phase_ones[i & 7] += (nv >> 7) & 1;
        }
        int matches = 0;
        for (int j = 0; j < 8; j++) matches += phase_score(phase_ones[j], repeats);
        if (matches > best_matches) {
            best_matches = matches;
            best_seed = seed;
            for (int j = 0; j < 8; j++) best_phase_ones[j] = phase_ones[j];
        }
    }
    uint8_t pattern = 0;
    for (int j = 0; j < 8; j++) if (best_phase_ones[j] * 2 >= repeats) pattern |= (1u << (7 - j));
    *out_seed = best_seed;
    *out_pattern = pattern;
    *out_matches = best_matches;
}

/* seed_min/seed_max let a caller widen the seed range (e.g. to 4-byte
 * seeds) without touching the 3-byte-seed behavior everywhere else --
 * search_all_chunks_avx512_n below is a thin wrapper over this using
 * the default 3-byte SEED_MIN/SEED_MAX. */
static void search_all_chunks_avx512_n_range(const unsigned char *chunks_flat, int csize, int num_chunks,
                                              uint32_t seed_min, uint32_t seed_max,
                                              uint32_t *out_seed, uint8_t *out_pattern, int *out_matches) {
    int repeats = csize / 8;

    __m512i *buf_bcast = malloc(sizeof(__m512i) * (size_t)csize * (size_t)num_chunks);
    __m512i *i_bcast   = malloc(sizeof(__m512i) * (size_t)csize);
    __m512i *ks_vec    = malloc(sizeof(__m512i) * (size_t)csize);
    for (int i = 0; i < csize; i++) i_bcast[i] = _mm512_set1_epi32(i);
    for (int ck = 0; ck < num_chunks; ck++)
        for (int i = 0; i < csize; i++)
            buf_bcast[(size_t)ck * csize + i] = _mm512_set1_epi32(chunks_flat[(size_t)ck * csize + i]);

    const __m512i mask_ff   = _mm512_set1_epi32(0xFF);
    const __m512i repeats_v = _mm512_set1_epi32(repeats);
    const __m512i c_golden  = _mm512_set1_epi32((int)0x9E3779B1u);
    const __m512i c_m1      = _mm512_set1_epi32((int)0x85EBCA6Bu);
    const __m512i c_m2      = _mm512_set1_epi32((int)0xC2B2AE35u);

    for (int ck = 0; ck < num_chunks; ck++) { out_matches[ck] = -1; out_seed[ck] = 0; }
    int (*best_phase_ones)[8] = calloc((size_t)num_chunks, sizeof(int[8]));

    uint32_t group_count = (seed_max - seed_min + 1) / 16;

    for (uint32_t g = 0; g < group_count; g++) {
        uint32_t base = seed_min + g * 16;
        __m512i seed_v = _mm512_setr_epi32((int)(base),   (int)(base+1),  (int)(base+2),  (int)(base+3),
                                            (int)(base+4), (int)(base+5),  (int)(base+6),  (int)(base+7),
                                            (int)(base+8), (int)(base+9),  (int)(base+10), (int)(base+11),
                                            (int)(base+12),(int)(base+13), (int)(base+14), (int)(base+15));
        __m512i seed_mul = _mm512_mullo_epi32(seed_v, c_golden);

        for (int i = 0; i < csize; i++) {
            __m512i x = _mm512_xor_si512(i_bcast[i], seed_mul);
            x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
            x = _mm512_mullo_epi32(x, c_m1);
            x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 13));
            x = _mm512_mullo_epi32(x, c_m2);
            x = _mm512_xor_si512(x, _mm512_srli_epi32(x, 16));
            ks_vec[i] = _mm512_and_si512(x, mask_ff);
        }

        for (int ck = 0; ck < num_chunks; ck++) {
            __m512i phase_acc[8];
            for (int j = 0; j < 8; j++) phase_acc[j] = _mm512_setzero_si512();

            for (int i = 0; i < csize; i++) {
                __m512i sum = _mm512_add_epi32(ks_vec[i], buf_bcast[(size_t)ck * csize + i]);
                __m512i nv  = _mm512_and_si512(sum, mask_ff);
                __m512i bit = _mm512_srli_epi32(nv, 7);
                phase_acc[i & 7] = _mm512_add_epi32(phase_acc[i & 7], bit);
            }

            __m512i matches_vec = _mm512_setzero_si512();
            for (int j = 0; j < 8; j++) {
                __m512i ones  = phase_acc[j];
                __m512i zeros = _mm512_sub_epi32(repeats_v, ones);
                __m512i m     = _mm512_max_epi32(ones, zeros);
                matches_vec   = _mm512_add_epi32(matches_vec, m);
            }
            int group_best = _mm512_reduce_max_epi32(matches_vec);

            if (group_best > out_matches[ck]) {
                int lanes[8][16];
                for (int j = 0; j < 8; j++) _mm512_storeu_si512((void *)lanes[j], phase_acc[j]);
                int match_lane[16];
                _mm512_storeu_si512((void *)match_lane, matches_vec);

                for (int lane = 0; lane < 16; lane++) {
                    if (match_lane[lane] == group_best) {
                        out_matches[ck] = group_best;
                        out_seed[ck] = base + (uint32_t)lane;
                        for (int j = 0; j < 8; j++) best_phase_ones[ck][j] = lanes[j][lane];
                        break;
                    }
                }
            }
        }
    }

    /* tail seeds not covered by a full group of 16, handled scalar, all chunks */
    uint8_t *ks = malloc((size_t)csize);
    for (uint32_t seed = seed_min + group_count * 16; seed <= seed_max; seed++) {
        for (int i = 0; i < csize; i++) ks[i] = hash_byte((uint32_t)i, seed);

        for (int ck = 0; ck < num_chunks; ck++) {
            int phase_ones[8] = {0};
            for (int i = 0; i < csize; i++) {
                uint8_t nv = (uint8_t)(chunks_flat[(size_t)ck * csize + i] + ks[i]);
                phase_ones[i & 7] += (nv >> 7) & 1;
            }
            int matches = 0;
            for (int j = 0; j < 8; j++) matches += phase_score(phase_ones[j], repeats);
            if (matches > out_matches[ck]) {
                out_matches[ck] = matches;
                out_seed[ck] = seed;
                for (int j = 0; j < 8; j++) best_phase_ones[ck][j] = phase_ones[j];
            }
        }
    }
    free(ks);

    for (int ck = 0; ck < num_chunks; ck++) {
        uint8_t pattern = 0;
        for (int j = 0; j < 8; j++) {
            if (best_phase_ones[ck][j] * 2 >= repeats) pattern |= (1u << (7 - j));
        }
        out_pattern[ck] = pattern;
    }

    free(buf_bcast);
    free(i_bcast);
    free(ks_vec);
    free(best_phase_ones);
}

static void search_all_chunks_avx512_n(const unsigned char *chunks_flat, int csize, int num_chunks,
                                        uint32_t *out_seed, uint8_t *out_pattern, int *out_matches) {
    search_all_chunks_avx512_n_range(chunks_flat, csize, num_chunks, SEED_MIN, SEED_MAX,
                                      out_seed, out_pattern, out_matches);
}

/* --- multithreaded whole-file scan ---
 * Splits the file's non-overlapping 64-byte chunks across THREADS OS
 * threads (real cores, not the single-thread SIMD tricks above -- for
 * a job this size the search space itself is the bottleneck, not
 * instruction-level parallelism). Each thread reuses the already
 * -validated search_all_chunks_avx512 unchanged, processing its
 * assigned chunk range in batches (so the keystream is regenerated
 * only once per BATCH chunks, not once per chunk, same amortization
 * idea as the single-threaded batched version -- just applied per
 * thread). Since the whole file is one contiguous in-memory buffer,
 * chunk k's 64 bytes are just filebuf + k*64: no copying needed, each
 * thread's batch is a direct cast of a slice of that buffer. */
#define SCAN_BATCH 4096

typedef struct {
    const unsigned char *filebuf;
    long chunk_start, chunk_end; /* [start, end) */
    uint32_t *out_seed;
    uint8_t *out_pattern;
    int *out_matches;
} scan_thread_arg;

static void *scan_thread_fn(void *arg_) {
    scan_thread_arg *arg = (scan_thread_arg *)arg_;
    for (long b = arg->chunk_start; b < arg->chunk_end; b += SCAN_BATCH) {
        long count = arg->chunk_end - b;
        if (count > SCAN_BATCH) count = SCAN_BATCH;
        const unsigned char (*chunks)[N] = (const unsigned char (*)[N])(arg->filebuf + (size_t)b * N);
        search_all_chunks_avx512(chunks, (int)count,
                                  arg->out_seed + b, arg->out_pattern + b, arg->out_matches + b);
    }
    return NULL;
}

/* Runs the scan over chunks [0, chunk_count) using `threads` OS threads,
 * splitting the range into contiguous per-thread slices. */
static void scan_whole_file_mt(const unsigned char *filebuf, long chunk_count, int threads,
                                uint32_t *out_seed, uint8_t *out_pattern, int *out_matches) {
    pthread_t *tid = malloc(sizeof(pthread_t) * (size_t)threads);
    scan_thread_arg *args = malloc(sizeof(scan_thread_arg) * (size_t)threads);

    long per_thread = chunk_count / threads;
    long start = 0;
    for (int t = 0; t < threads; t++) {
        long end = (t == threads - 1) ? chunk_count : start + per_thread;
        args[t] = (scan_thread_arg){ filebuf, start, end, out_seed, out_pattern, out_matches };
        pthread_create(&tid[t], NULL, scan_thread_fn, &args[t]);
        start = end;
    }
    for (int t = 0; t < threads; t++) pthread_join(tid[t], NULL);

    free(tid);
    free(args);
}

static double wall_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1e9;
}

/* --- multithreaded chunk-size sweep ---
 * Different chunk sizes are independent searches, so instead of one
 * thread per size (uneven: csize=65536 takes ~1000x longer than
 * csize=64), this is a work queue: SWEEP_THREADS workers each pull the
 * next not-yet-claimed size index from a shared counter (guarded by a
 * mutex) and run it, so a thread that finishes a cheap small size
 * immediately picks up the next unclaimed size instead of sitting
 * idle. Each worker opens its own FILE* since fseek/fread on a shared
 * FILE* from multiple threads would race. */
typedef struct {
    int csize;
    int samples_requested;
    int samples_loaded;
    double accuracy_pct;
    double profit;
    double seconds;
    int ok;
} sweep_result;

typedef struct {
    const int *sizes;
    const int *samples;
    int n;
    int *next_idx;
    pthread_mutex_t *lock;
    long filesize;
    int header_skip;
    int overhead_bits;
    sweep_result *results;
} sweep_pool_arg;

static void *sweep_worker_fn(void *arg_) {
    sweep_pool_arg *arg = (sweep_pool_arg *)arg_;
    for (;;) {
        int idx;
        pthread_mutex_lock(arg->lock);
        idx = *arg->next_idx;
        if (idx < arg->n) (*arg->next_idx)++;
        pthread_mutex_unlock(arg->lock);
        if (idx >= arg->n) break;

        int csize = arg->sizes[idx];
        int samples = arg->samples[idx];
        sweep_result *res = &arg->results[idx];
        res->csize = csize;
        res->samples_requested = samples;
        res->ok = 0;

        long avail = arg->filesize - arg->header_skip - csize;
        if (avail < 0) continue;
        long stride = (samples > 1) ? avail / (samples - 1) : 0;

        FILE *fp = fopen(INPUT_PATH, "rb");
        if (!fp) continue;

        unsigned char *sbuf = malloc((size_t)csize * samples);
        int sloaded = 0;
        for (int k = 0; k < samples; k++) {
            long off = arg->header_skip + k * stride;
            fseek(fp, off, SEEK_SET);
            if (fread(sbuf + (size_t)sloaded * csize, 1, (size_t)csize, fp) != (size_t)csize) break;
            sloaded++;
        }
        fclose(fp);
        if (sloaded == 0) { free(sbuf); continue; }

        uint32_t *sw_seed = malloc(sizeof(uint32_t) * sloaded);
        uint8_t  *sw_pat  = malloc(sizeof(uint8_t) * sloaded);
        int      *sw_m    = malloc(sizeof(int) * sloaded);

        double t0 = wall_seconds();
        search_all_chunks_avx512_n(sbuf, csize, sloaded, sw_seed, sw_pat, sw_m);
        double dt = wall_seconds() - t0;

        double acc_sum = 0.0;
        for (int k = 0; k < sloaded; k++) acc_sum += (double)sw_m[k] / csize;
        double acc_frac = acc_sum / sloaded;

        res->samples_loaded = sloaded;
        res->accuracy_pct = acc_frac * 100.0;
        res->profit = acc_frac * (csize - arg->overhead_bits);
        res->seconds = dt;
        res->ok = 1;

        free(sbuf); free(sw_seed); free(sw_pat); free(sw_m);
    }
    return NULL;
}

/* Runs the whole sweep across SWEEP_THREADS worker threads pulling
 * from a shared work queue; returns results in the same order as
 * sizes[]/samples[] regardless of completion order. */
static void run_sweep_mt(const int *sizes, const int *samples, int n,
                          long filesize, int header_skip, int overhead_bits, int num_threads,
                          sweep_result *results) {
    int next_idx = 0;
    pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
    sweep_pool_arg arg = { sizes, samples, n, &next_idx, &lock, filesize, header_skip, overhead_bits, results };

    if (num_threads > n) num_threads = n;
    pthread_t *tid = malloc(sizeof(pthread_t) * (size_t)num_threads);
    for (int t = 0; t < num_threads; t++) pthread_create(&tid[t], NULL, sweep_worker_fn, &arg);
    for (int t = 0; t < num_threads; t++) pthread_join(tid[t], NULL);
    free(tid);
    pthread_mutex_destroy(&lock);
}

int main(void) {
    FILE *f = fopen(INPUT_PATH, "rb");
    if (!f) { perror(INPUT_PATH); return 1; }

    unsigned char buf[N];
    size_t n = fread(buf, 1, N, f);
    fclose(f);

    int counts[256] = {0};
    for (size_t i = 0; i < n; i++) counts[buf[i]]++;

    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / n;
        entropy -= p * log2(p);
    }

    printf("Read %zu bytes from %s\n", n, INPUT_PATH);
    printf("Entropy: %f bits/byte\n", entropy);

    printf("Bytes (decimal):\n");
    for (size_t i = 0; i < n; i++) {
        printf("%d ", buf[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    if (n % 16 != 0) printf("\n");

    /* slow reference: byte-at-a-time fgetc, ~199M function calls for
     * this file. Kept only to verify the fast version below counts the
     * same bytes before trusting its speedup. */
    clock_t t_slow0 = clock();
    FILE *fall = fopen(INPUT_PATH, "rb");
    if (!fall) { perror(INPUT_PATH); return 1; }
    long total_slow = 0;
    long counts_slow[256] = {0};
    int c;
    while ((c = fgetc(fall)) != EOF) {
        counts_slow[c]++;
        total_slow++;
    }
    fclose(fall);
    double t_slow = (double)(clock() - t_slow0) / CLOCKS_PER_SEC;

    /* fast version: read in large chunks with fread, histogram each
     * chunk in a tight loop -- no per-byte function-call overhead */
    clock_t t_fast0 = clock();
    FILE *ffast = fopen(INPUT_PATH, "rb");
    if (!ffast) { perror(INPUT_PATH); return 1; }
    long total_fast = 0;
    long counts_fast[256] = {0};
    size_t iobuf_size = 4 * 1024 * 1024;
    unsigned char *iobuf = malloc(iobuf_size);
    size_t got;
    while ((got = fread(iobuf, 1, iobuf_size, ffast)) > 0) {
        for (size_t j = 0; j < got; j++) counts_fast[iobuf[j]]++;
        total_fast += (long)got;
    }
    free(iobuf);
    fclose(ffast);
    double t_fast = (double)(clock() - t_fast0) / CLOCKS_PER_SEC;

    int mismatch_entropy = (total_slow != total_fast);
    for (int i = 0; i < 256 && !mismatch_entropy; i++)
        if (counts_slow[i] != counts_fast[i]) mismatch_entropy = 1;
    if (mismatch_entropy) {
        fprintf(stderr, "fast byte-histogram MISMATCH vs fgetc reference -- not trustworthy\n");
        return 1;
    }

    double full_entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (counts_fast[i] == 0) continue;
        double p = (double)counts_fast[i] / total_fast;
        full_entropy -= p * log2(p);
    }
    printf("Whole file entropy (%ld bytes): %f bits/byte\n", total_fast, full_entropy);
    printf("[timing] whole-file histogram -- fgetc: %.3fs   fread+buffer: %.3fs  (%.1fx)\n",
           t_slow, t_fast, t_slow / t_fast);

    /* --- brute-force 3-byte hash-keystream seed search for best
     * repeating MSB pattern, run independently over NUM_CHUNKS chunks
     * spread across the file (skipping the first HEADER_SKIP bytes,
     * which look like a header rather than coded payload) --- */
    FILE *fc = fopen(INPUT_PATH, "rb");
    if (!fc) { perror(INPUT_PATH); return 1; }
    fseek(fc, 0, SEEK_END);
    long filesize = ftell(fc);

    long available = filesize - HEADER_SKIP - N;
    if (available < 0) available = 0;
    long stride = (NUM_CHUNKS > 1) ? available / (NUM_CHUNKS - 1) : 0;

    /* self-check: AVX-512 search must match the scalar reference
     * exactly on real data before we trust it for the full run */
    {
        fseek(fc, HEADER_SKIP, SEEK_SET);
        unsigned char check_chunk[N];
        if (fread(check_chunk, 1, N, fc) == N) {
            uint32_t s_seed, v_seed; uint8_t s_pat, v_pat; int s_m, v_m;
            search_chunk_scalar(check_chunk, &s_seed, &s_pat, &s_m);
            search_chunk_avx512(check_chunk, &v_seed, &v_pat, &v_m);
            if (s_seed != v_seed || s_pat != v_pat || s_m != v_m) {
                fprintf(stderr, "AVX-512 self-check FAILED: scalar seed=%u pat=%u m=%d vs avx512 seed=%u pat=%u m=%d\n",
                        s_seed, s_pat, s_m, v_seed, v_pat, v_m);
                fclose(fc);
                return 1;
            }
            printf("AVX-512 self-check passed (matches scalar reference exactly)\n");
        }
    }

    /* load all chunks up front so the batched search can share one
     * keystream pass across all of them */
    unsigned char (*chunks)[N] = malloc(sizeof(unsigned char[N]) * NUM_CHUNKS);
    long offsets[NUM_CHUNKS];
    int loaded = 0;
    for (int k = 0; k < NUM_CHUNKS; k++) {
        long offset = HEADER_SKIP + k * stride;
        fseek(fc, offset, SEEK_SET);
        if (fread(chunks[k], 1, N, fc) != N) break;
        offsets[k] = offset;
        loaded++;
    }
    fclose(fc);

    printf("\nMSB pattern search across %d chunks of %d bytes (file size %ld bytes):\n",
           loaded, N, filesize);

    /* baseline: one independent full seed-sweep per chunk (recomputes
     * the keystream from scratch each time) */
    clock_t t_baseline0 = clock();
    uint32_t *base_seed = malloc(sizeof(uint32_t) * loaded);
    uint8_t  *base_pat  = malloc(sizeof(uint8_t) * loaded);
    int      *base_m    = malloc(sizeof(int) * loaded);
    for (int k = 0; k < loaded; k++)
        search_chunk_avx512(chunks[k], &base_seed[k], &base_pat[k], &base_m[k]);
    double baseline_time = (double)(clock() - t_baseline0) / CLOCKS_PER_SEC;

    /* batched: single seed-sweep shared across all chunks */
    clock_t t_batch0 = clock();
    uint32_t *batch_seed = malloc(sizeof(uint32_t) * loaded);
    uint8_t  *batch_pat  = malloc(sizeof(uint8_t) * loaded);
    int      *batch_m    = malloc(sizeof(int) * loaded);
    search_all_chunks_avx512((const unsigned char (*)[N])chunks, loaded, batch_seed, batch_pat, batch_m);
    double batch_time = (double)(clock() - t_batch0) / CLOCKS_PER_SEC;

    /* cross-check: batched result must exactly match the already-validated
     * per-chunk search before the speedup is trusted */
    int mismatch = 0;
    double accuracy_sum = 0.0;
    for (int k = 0; k < loaded; k++) {
        if (base_seed[k] != batch_seed[k] || base_pat[k] != batch_pat[k] || base_m[k] != batch_m[k]) {
            fprintf(stderr, "MISMATCH chunk %d: baseline seed=%u pat=%u m=%d vs batched seed=%u pat=%u m=%d\n",
                    k, base_seed[k], base_pat[k], base_m[k], batch_seed[k], batch_pat[k], batch_m[k]);
            mismatch = 1;
            continue;
        }
        double acc = 100.0 * batch_m[k] / 64.0;
        accuracy_sum += acc;
        printf("chunk %2d @ offset %10ld: seed=%7u (0x%06X) pattern=", k, offsets[k], batch_seed[k], batch_seed[k]);
        for (int b = 7; b >= 0; b--) putchar((batch_pat[k] >> b) & 1 ? '1' : '0');
        printf(" accuracy=%2d/64 (%.1f%%)\n", batch_m[k], acc);
    }

    if (mismatch) {
        fprintf(stderr, "batched search does not match baseline -- not trustworthy, fix before using\n");
        return 1;
    }

    printf("Average accuracy across %d chunks: %.1f%%\n", loaded, accuracy_sum / loaded);
    printf("[timing] baseline (per-chunk keystream): %.3fs\n", baseline_time);
    printf("[timing] batched  (shared keystream):     %.3fs  (%.2fx)\n",
           batch_time, baseline_time / batch_time);

    free(chunks);
    free(base_seed); free(base_pat); free(base_m);
    free(batch_seed); free(batch_pat); free(batch_m);

    /* --- multithreaded whole-file scan: validate on a small subset and
     * measure real throughput before committing to the full file, which
     * even multithreaded is expected to take hours.
     * Already validated in a prior run -- skipped here (RUN_MT_SCAN=0)
     * to avoid paying its ~4 minutes again while iterating on the
     * chunk-size sweep below. Flip to 1 to re-run it. */
#define RUN_MT_SCAN 0
    if (RUN_MT_SCAN) {
        FILE *fw = fopen(INPUT_PATH, "rb");
        if (!fw) { perror(INPUT_PATH); return 1; }
        fseek(fw, 0, SEEK_END);
        long wsize = ftell(fw);
        fseek(fw, 0, SEEK_SET);
        unsigned char *filebuf = malloc((size_t)wsize);
        clock_t t_read0 = clock();
        size_t rd = fread(filebuf, 1, (size_t)wsize, fw);
        fclose(fw);
        double read_time = (double)(clock() - t_read0) / CLOCKS_PER_SEC;
        if (rd != (size_t)wsize) { fprintf(stderr, "short read\n"); free(filebuf); return 1; }

        long chunk_count = wsize / N;
        printf("\nWhole-file scan sizing: %ld bytes -> %ld non-overlapping %d-byte chunks (%.3fs to load)\n",
               wsize, chunk_count, N, read_time);

        int TRUST_CHUNKS = 200;
        int MT_TEST_CHUNKS = 8000;
        int threads = 16;

        uint32_t *trust_seed = malloc(sizeof(uint32_t) * TRUST_CHUNKS);
        uint8_t  *trust_pat  = malloc(sizeof(uint8_t) * TRUST_CHUNKS);
        int      *trust_m    = malloc(sizeof(int) * TRUST_CHUNKS);
        clock_t t_trust0 = clock();
        search_all_chunks_avx512((const unsigned char (*)[N])filebuf, TRUST_CHUNKS, trust_seed, trust_pat, trust_m);
        double trust_time = (double)(clock() - t_trust0) / CLOCKS_PER_SEC;

        uint32_t *mt_seed = malloc(sizeof(uint32_t) * MT_TEST_CHUNKS);
        uint8_t  *mt_pat  = malloc(sizeof(uint8_t) * MT_TEST_CHUNKS);
        int      *mt_m    = malloc(sizeof(int) * MT_TEST_CHUNKS);
        clock_t t_mt0 = clock();
        scan_whole_file_mt(filebuf, MT_TEST_CHUNKS, threads, mt_seed, mt_pat, mt_m);
        double mt_time = (double)(clock() - t_mt0) / CLOCKS_PER_SEC;

        int mt_mismatch = 0;
        for (int k = 0; k < TRUST_CHUNKS; k++) {
            if (trust_seed[k] != mt_seed[k] || trust_pat[k] != mt_pat[k] || trust_m[k] != mt_m[k]) {
                fprintf(stderr, "MT MISMATCH chunk %d: trusted seed=%u pat=%u m=%d vs mt seed=%u pat=%u m=%d\n",
                        k, trust_seed[k], trust_pat[k], trust_m[k], mt_seed[k], mt_pat[k], mt_m[k]);
                mt_mismatch = 1;
            }
        }

        if (mt_mismatch) {
            fprintf(stderr, "multithreaded scan does not match trusted single-thread reference -- not trustworthy\n");
        } else {
            printf("multithreaded scan validated: first %d chunks match trusted single-thread reference exactly\n", TRUST_CHUNKS);
            printf("[timing] trusted single-thread, %d chunks:  %.3fs (%.5fs/chunk)\n",
                   TRUST_CHUNKS, trust_time, trust_time / TRUST_CHUNKS);
            printf("[timing] %d threads, %d chunks:            %.3fs (%.5fs/chunk, %.2fx per-chunk speedup)\n",
                   threads, MT_TEST_CHUNKS, mt_time, mt_time / MT_TEST_CHUNKS,
                   (trust_time / TRUST_CHUNKS) / (mt_time / MT_TEST_CHUNKS));

            double mt_accuracy_sum = 0.0;
            int mt_min = 65, mt_max = -1;
            for (int k = 0; k < MT_TEST_CHUNKS; k++) {
                double acc = 100.0 * mt_m[k] / 64.0;
                mt_accuracy_sum += acc;
                if (mt_m[k] < mt_min) mt_min = mt_m[k];
                if (mt_m[k] > mt_max) mt_max = mt_m[k];
            }
            printf("Accuracy over %d chunks (%d threads): avg=%.2f%%  min=%d/64 (%.1f%%)  max=%d/64 (%.1f%%)\n",
                   MT_TEST_CHUNKS, threads, mt_accuracy_sum / MT_TEST_CHUNKS,
                   mt_min, 100.0 * mt_min / 64.0, mt_max, 100.0 * mt_max / 64.0);

            double per_chunk_mt = mt_time / MT_TEST_CHUNKS;
            double eta_seconds = per_chunk_mt * (double)chunk_count;
            printf("Projected full-file scan (%ld chunks, %d threads): %.1f minutes (%.2f hours)\n",
                   chunk_count, threads, eta_seconds / 60.0, eta_seconds / 3600.0);
        }

        free(filebuf);
        free(trust_seed); free(trust_pat); free(trust_m);
        free(mt_seed); free(mt_pat); free(mt_m);
    }

    /* --- chunk-size sweep for profit ---
     * profit(csize) = accuracy_fraction * (csize - overhead_bits), where
     * overhead_bits = 3-byte seed + 1-byte pattern = 32 bits, the fixed
     * cost of describing which seed/pattern reconstructs this chunk's
     * predicted MSB stream, regardless of chunk size. csize itself is
     * the number of MSB bits (one per byte) a perfect predictor could
     * save. Sweep several chunk sizes (multiples of 8, so the period-8
     * pattern divides evenly) and see which maximizes expected profit.
     * Already run once (took 30 minutes -- the largest size dominated
     * wall time since parallelism here was across sizes, not within
     * the big one). Skipped by default so it doesn't re-fire; flip to
     * 1 to rerun. */
#define RUN_CHUNK_SWEEP 0
    if (RUN_CHUNK_SWEEP) {
        const int OVERHEAD_BITS = 3 * 8 + 1 * 8; /* 32 */
        /* cost per size is ~csize * samples, so sample count is tapered
         * down as csize grows to keep total sweep time bounded (a fixed
         * 25 samples at csize=65536 would alone take ~10 minutes) */
        const int SWEEP_SIZES[]   = {  64,  96, 128, 192, 256, 384, 512, 768,1024,1536,2048,3072,4096,6144,8192,12288,16384,24576,32768,49152,65536};
        const int SWEEP_SAMPLES[] = {  25,  25,  20,  15,  12,   8,   6,   5,   5,   4,   4,   3,   3,   3,   3,    3,    3,    3,    3,    3,    3};
        const int NUM_SWEEP = (int)(sizeof(SWEEP_SIZES) / sizeof(SWEEP_SIZES[0]));

        FILE *fs = fopen(INPUT_PATH, "rb");
        if (!fs) { perror(INPUT_PATH); return 1; }
        fseek(fs, 0, SEEK_END);
        long sfilesize = ftell(fs);

        /* correctness spot-check: generalized (variable-csize) AVX-512
         * search must match its own scalar reference at a size other
         * than the already-proven N=64, before the sweep's numbers are
         * trusted */
        {
            int check_csize = 32;
            long avail = sfilesize - HEADER_SKIP - check_csize;
            unsigned char *cbuf = malloc((size_t)check_csize);
            fseek(fs, HEADER_SKIP + (avail > 0 ? avail / 3 : 0), SEEK_SET);
            fread(cbuf, 1, (size_t)check_csize, fs);

            uint32_t sc_seed, sv_seed; uint8_t sc_pat, sv_pat; int sc_m, sv_m;
            search_chunk_scalar_n(cbuf, check_csize, &sc_seed, &sc_pat, &sc_m);
            search_all_chunks_avx512_n(cbuf, check_csize, 1, &sv_seed, &sv_pat, &sv_m);
            if (sc_seed != sv_seed || sc_pat != sv_pat || sc_m != sv_m) {
                fprintf(stderr, "chunk-size-sweep self-check FAILED at csize=%d: scalar seed=%u pat=%u m=%d vs avx512 seed=%u pat=%u m=%d\n",
                        check_csize, sc_seed, sc_pat, sc_m, sv_seed, sv_pat, sv_m);
                free(cbuf);
                fclose(fs);
                return 1;
            }
            printf("\nchunk-size-sweep self-check passed at csize=%d (matches scalar reference exactly)\n", check_csize);
            free(cbuf);
        }

        fclose(fs); /* worker threads open their own handles */

        int SWEEP_THREADS = 16;
        printf("\nProfit vs chunk size (overhead = %d bits, tapered sample count per size, %d worker threads):\n",
               OVERHEAD_BITS, SWEEP_THREADS);

        sweep_result *results = calloc((size_t)NUM_SWEEP, sizeof(sweep_result));
        double t_sweep0 = wall_seconds();
        run_sweep_mt(SWEEP_SIZES, SWEEP_SAMPLES, NUM_SWEEP, sfilesize, HEADER_SKIP, OVERHEAD_BITS, SWEEP_THREADS, results);
        double sweep_time = wall_seconds() - t_sweep0;

        double best_profit = -1e18;
        int best_csize = 0;
        for (int s = 0; s < NUM_SWEEP; s++) {
            sweep_result *r = &results[s];
            if (!r->ok) {
                printf("  csize=%6d  (skipped -- not enough file left for even 1 sample)\n", SWEEP_SIZES[s]);
                continue;
            }
            printf("  csize=%6d  samples=%2d  accuracy=%6.2f%%  profit=%9.4f bits  (%.3fs)\n",
                   r->csize, r->samples_loaded, r->accuracy_pct, r->profit, r->seconds);
            if (r->profit > best_profit) { best_profit = r->profit; best_csize = r->csize; }
        }
        free(results);

        printf("Best chunk size: %d bytes, profit=%.4f bits\n", best_csize, best_profit);
        printf("[timing] whole sweep wall time (%d threads): %.2fs\n", SWEEP_THREADS, sweep_time);
    }

    /* --- stitched-chunks entropy test: 3-byte seed, 64-byte chunks ---
     * Take 25 ADJACENT 64-byte chunks (one contiguous 1600-byte block,
     * not the spread-out stride-sampled chunks used earlier), find each
     * chunk's own best (seed, pattern) independently, apply each
     * chunk's raw MSB-flip transform to its own 64 bytes, then glue the
     * 25 transformed chunks back into one 1600-byte block in their
     * original order. Measuring entropy over the whole stitched block
     * (1600 samples) instead of per-64-byte-chunk (only 64 samples)
     * gives a much less noisy entropy estimate. The total reduction is
     * then divided by 25 to express it as an average net bits saved
     * per 64-byte block. */
    {
        const int csize = 64;
        const int nblocks = 25;
        const int total = csize * nblocks; /* 1600 */

        FILE *fo = fopen(INPUT_PATH, "rb");
        if (!fo) { perror(INPUT_PATH); return 1; }
        fseek(fo, HEADER_SKIP, SEEK_SET);
        unsigned char *orig = malloc((size_t)total);
        size_t got = fread(orig, 1, (size_t)total, fo);
        fclose(fo);
        if (got != (size_t)total) { fprintf(stderr, "short read\n"); free(orig); return 1; }

        uint32_t seeds[25]; uint8_t patterns[25]; int matches[25];
        search_all_chunks_avx512_n(orig, csize, nblocks, seeds, patterns, matches);

        unsigned char *stitched = malloc((size_t)total);
        printf("\n25 adjacent csize=%d blocks (contiguous %d-byte region):\n", csize, total);
        for (int b = 0; b < nblocks; b++) {
            const unsigned char *src = orig + (size_t)b * csize;
            unsigned char *dst = stitched + (size_t)b * csize;
            for (int i = 0; i < csize; i++) {
                uint8_t ks = hash_byte((uint32_t)i, seeds[b]);
                uint8_t hashed = (uint8_t)(src[i] + ks); /* same intermediate the search fit the pattern to */
                int bit = (patterns[b] >> (7 - (i & 7))) & 1;
                dst[i] = bit ? (unsigned char)(hashed ^ 0x80) : hashed;
            }
            printf("  block %2d: seed=%7u pattern=", b, seeds[b]);
            for (int bit = 7; bit >= 0; bit--) putchar((patterns[b] >> bit) & 1 ? '1' : '0');
            printf(" accuracy=%2d/%d (%.1f%%)\n", matches[b], csize, 100.0 * matches[b] / csize);
        }

        double before_entropy = 0.0, after_entropy = 0.0;
        long counts_before[256] = {0}, counts_after[256] = {0};
        for (int i = 0; i < total; i++) { counts_before[orig[i]]++; counts_after[stitched[i]]++; }
        for (int v = 0; v < 256; v++) {
            if (counts_before[v]) { double p = (double)counts_before[v] / total; before_entropy -= p * log2(p); }
            if (counts_after[v])  { double p = (double)counts_after[v]  / total; after_entropy  -= p * log2(p); }
        }

        double total_bits_before = before_entropy * total;
        double total_bits_after  = after_entropy * total;
        double total_reduction   = total_bits_before - total_bits_after;
        double net_per_block     = total_reduction / nblocks;

        printf("\nStitched %d-byte block (%d chunks of %d bytes each):\n", total, nblocks, csize);
        printf("  entropy before: %.4f bits/byte  (%.2f bits total)\n", before_entropy, total_bits_before);
        printf("  entropy after:  %.4f bits/byte  (%.2f bits total)\n", after_entropy, total_bits_after);
        printf("  total reduction: %.2f bits\n", total_reduction);
        printf("  net per block (reduction / %d): %.4f bits\n", nblocks, net_per_block);
        printf("  %s\n", total_reduction > 0 ? "LOWER (helped)" : "NOT LOWER (did not help)");

        printf("\nByte value frequency, before vs after (only values that appear, or that changed):\n");
        printf("  %-6s %-8s %-8s %-8s\n", "value", "before", "after", "delta");
        int distinct_before = 0, distinct_after = 0;
        for (int v = 0; v < 256; v++) {
            if (counts_before[v]) distinct_before++;
            if (counts_after[v]) distinct_after++;
            if (counts_before[v] || counts_after[v]) {
                printf("  0x%02X   %-8ld %-8ld %+ld\n", v, counts_before[v], counts_after[v],
                       counts_after[v] - counts_before[v]);
            }
        }
        printf("  distinct values before: %d, after: %d\n", distinct_before, distinct_after);

        /* Now that the transform actually applies hash(i,seed) before
         * the conditional flip, check MSB=1 rate before (raw, expect
         * ~50%) vs after (stitched output, expect close to 1-accuracy
         * if the flip is doing its job of canceling out predicted 1s). */
        {
            long raw_msb1 = 0, out_msb1 = 0;
            for (int idx = 0; idx < total; idx++) {
                raw_msb1 += (orig[idx] >> 7) & 1;
                out_msb1 += (stitched[idx] >> 7) & 1;
            }
            printf("\nMSB=1 rate: raw data %.2f%% (expect ~50%%) vs transformed output %.2f%% (expect low, near 1-accuracy)\n",
                   100.0 * raw_msb1 / total, 100.0 * out_msb1 / total);
        }

        free(orig); free(stitched);
    }

    return 0;
}
