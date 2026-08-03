#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

#define FILE_PATH "C:\\Users\\lukac\\Documents\\compressor\\actuallstuff\\CurrentDataFile.bin"
#define CHUNK_BYTES 16    // bytes read per chunk -- each byte expands to 4 crumbs (2 bits each)
#define NUM_CHUNKS (32)      // number of chunks to process before stopping
/* CHUNK_BYTES, STAGE1_BITS and STAGE2_BITS are NOT independent -- the best
   chunk size depends on the seed budget and vice versa, so changing one without
   re-tuning the others loses real net every time. They have been mismatched
   three times now. Do not touch one alone.

   Current setting: 16 B / 64 crumbs, stage1 18 bits, stage2 6 bits (prngbit).
   Picked from a 677-config exact grid search (sweeplab.c) maximising NET PER
   CHUNK, 256-4096 chunks per point:

       64 crumbs (16 B) @ 18 + s2=6   +4.093   <- current
       48 crumbs (12 B) @ 18 + s2=0   +4.065
       48 crumbs (12 B) @ 16 + s2=0   +4.064
       64 crumbs (16 B) @ 20 + s2=4   +4.028
       56 crumbs (14 B) @ 21, no s2   +3.981
      128 crumbs (32 B) @ 21, no s2   +3.203   the old setting

   The old header's table (48 crumbs @ 20 = +4.057, 56 @ 23 = +4.181) is
   reproduced by this grid to within noise -- what had drifted was the FILE:
   CHUNK_BYTES sat at 32, a point that table never covered and which measures
   +3.20. Most of the +0.89 gain here is simply returning to the documented
   operating point.

   Stage 2 earns +0.719 of the +4.093 at this setting. That is real but it is
   strongly conditional on chunk size: at 12-16 B stage 2 adds +0.28..+0.72,
   while at 5-6 B it goes NEGATIVE (-0.01..-0.03). Retune it if CHUNK_BYTES
   moves. The families are interchangeable -- a 6-candidate prngbit and the
   6-candidate period-4 pattern set measure identical within noise (paired
   difference -0.007 +/- 0.158), and merging them into one 12-candidate set
   loses (-0.023 +/- 0.019), because doubling the candidates buys about one bit
   of entropy deficit and costs exactly one bit to name.

   NET PER BYTE peaks somewhere else entirely: 5 B / 20 crumbs @ 8 bits with no
   stage 2, +0.209 bits/byte against this config's +0.126. If total file size
   ever becomes the objective instead of per-chunk net, that is the config, and
   stage 2 should be dropped when you go there.

   One caveat on the number itself. Roughly +2.1 of it is the plug-in entropy
   estimator's small-sample bias, (A-1)/(2*ln2) = 2.164 bits, which is already
   there at zero search and zero bits paid. Measured against each chunk's own
   raw entropy instead of the 2.0 ceiling this config scores +1.97. The bias is
   per CHUNK regardless of chunk size, which is why shrinking the chunk inflates
   the per-byte figure so dramatically -- at 3 B it reads +0.89 bits/byte on the
   2.0 baseline but only +0.088 against raw. */
#define SEED_BITS 21   /* legacy: only the self-test's seed masking uses this */
/* 1ULL, not 1L: long is 32-bit on Windows, so 1L<<31 overflows to negative
   (loop never runs) and 1L<<32 upward is undefined. uint64 is good to 63. */
#define SEED_COUNT (1ULL << SEED_BITS)  // seeds swept per chunk

#define CRUMBS_PER_CHUNK (CHUNK_BYTES * 4)
#define MAX_CRUMB_ENTROPY 2.0   // ceiling for a 4-symbol (2-bit) alphabet

static double crumb_entropy(const long freq[4], long total) {
    double entropy = 0.0;
    for (int i = 0; i < 4; i++) {
        if (freq[i] == 0) continue;
        double p = (double)freq[i] / (double)total;
        entropy -= p * log2(p);
    }
    return entropy;
}

/* splitmix64 finaliser: every input bit flips ~half the output bits, so nearby
   (seed, pos) pairs give unrelated crumb streams. Widened from the old 32-bit
   lowbias32 mixer -- see the aliasing note in prng_crumb below. */
static uint64_t avalanche_hash(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/* pos * PHI64 depends only on the position, never on the seed, so the whole
   table is built once for the run instead of recomputed on every inner step. */
static uint64_t pos_key[CRUMBS_PER_CHUNK];
static void init_pos_keys(void) {
    for (uint32_t i = 0; i < CRUMBS_PER_CHUNK; i++)
        pos_key[i] = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
}

/* The crumb stream is  avalanche_hash( avalanche_hash(seed) ^ pos_key[pos] ).
   The inner avalanche_hash(seed) is invariant across positions, so it is split
   out as seed_key and hoisted to the top of the seed loop -- it used to be
   recomputed once per crumb, which was half of all hash work in the program.

   Hashing the seed to a full 64-bit key before folding in pos is what stops
   seeds aliasing. That has bitten twice: the original (seed << 16) ^ pos packing
   made every seed at or above 2^16 a duplicate of one already swept, and the
   uint32 mixer that replaced it did the same at 2^32 -- invisible while
   SEED_BITS was 16, fatal the moment it goes past 32. avalanche_hash is a
   bijection on uint64, so distinct seeds give distinct keys up to SEED_BITS 63. */
static inline uint64_t prng_seed_key(uint64_t seed) {
    return avalanche_hash(seed);
}
static inline unsigned char prng_crumb_keyed(uint64_t seed_key, uint32_t pos) {
    return (unsigned char)(avalanche_hash(seed_key ^ pos_key[pos]) & 0x3u);
}

/* ---------------------------------------------------------------- pattern add
   Stage 2. No instruction tag is spent because there is only ever this one
   instruction; the whole overhead is its argument.

   The wider 12-bit-mask version is parked in
   later/compressthisdamthing_patternadd12.c; it lost, and the period sweep
   there showed why -- each extra period bit buys one more bit of search and
   costs one more bit to name, so widening cancels almost exactly. */
/* ------------------------------------------------------- prngbit amp add
   Stage 2 is now a PRNG bitstream: a stage-2 seed generates one bit per crumb,
   and where that bit is 1 the crumb gets amp added mod 4. Aperiodic and full
   length, so unlike the period-4 patterns it is not confined to residue
   classes.

   Cost = STAGE2_BITS (the bitstream seed) + log2(3) for the amp. amp=0 is
   excluded because it is the identity for every bitstream; amps 1,2,3 are all
   distinct here, because the relabeling partner of (B,a) is (~B,-a) and the
   complement of a PRNG bitstream is essentially never another seed's output.
   That is a real difference from the fixed patterns, where complements WERE in
   the family and halved the space.

   The whole search is EXACT -- every (stage1 seed, stage2 seed, amp) triple is
   scored. That is affordable only because of the bitset trick below, and it
   matters because a top-K beam was what crippled every earlier stage 2. To
   keep it exact the two seed widths are runtime parameters, so the total bit
   budget can be split between the stages and compared against spending all of
   it on stage 1 alone. */
static int      STAGE1_BITS = 18;
static int      STAGE2_BITS = 6;
static uint64_t STAGE1_COUNT, STAGE2_COUNT;
static int      NWORDS;                   /* 64-bit words per crumb bitset */
static uint64_t *S2BITS;                  /* [STAGE2_COUNT][NWORDS] */
static double   S2_COST;                  /* STAGE2_BITS + log2(3) */

/* Salted and shifted differently from the stage-1 crumb generator so a stage-2
   seed can never accidentally reproduce a stage-1 keystream's low bits. */
static inline uint64_t s2_seed_key(uint64_t t) {
    return avalanche_hash(t ^ 0xA5A5A5A5A5A5A5A5ULL);
}
static inline int s2_bit(uint64_t key, uint32_t pos) {
    return (int)((avalanche_hash(key ^ pos_key[pos]) >> 33) & 1u);
}

/* The bitstream depends only on (seed, position) -- never on the data -- so
   every one of them is built once for the whole run, not per chunk and
   certainly not per stage-1 seed. */
static void init_s2bits(void) {
    NWORDS = (CRUMBS_PER_CHUNK + 63) / 64;
    S2BITS = calloc((size_t)STAGE2_COUNT * (size_t)NWORDS, sizeof(uint64_t));
    if (!S2BITS) { fprintf(stderr, "out of memory for stage-2 bitstreams\n"); exit(1); }
    for (uint64_t t = 0; t < STAGE2_COUNT; t++) {
        uint64_t key = s2_seed_key(t);
        uint64_t *row = S2BITS + t * (uint64_t)NWORDS;
        for (int i = 0; i < CRUMBS_PER_CHUNK; i++)
            if (s2_bit(key, (uint32_t)i)) row[i >> 6] |= 1ULL << (i & 63);
    }
}

/* LT[f] = f*log2(f), so a histogram scores in 4 lookups instead of 4 log2 calls */
static double LT[CRUMBS_PER_CHUNK + 1];
static void init_lt(void) {
    LT[0] = 0.0;
    for (int f = 1; f <= CRUMBS_PER_CHUNK; f++) LT[f] = (double)f * log2((double)f);
}

static void s2_apply(unsigned char *c, int n, uint64_t t, int amp) {
    uint64_t key = s2_seed_key(t);
    for (int i = 0; i < n; i++)
        if (s2_bit(key, (uint32_t)i)) c[i] = (unsigned char)((c[i] + amp) & 0x3);
}
static void s2_invert(unsigned char *c, int n, uint64_t t, int amp) {
    uint64_t key = s2_seed_key(t);
    for (int i = 0; i < n; i++)
        if (s2_bit(key, (uint32_t)i)) c[i] = (unsigned char)((c[i] - amp) & 0x3);
}

/* Round-trip check: raw -> stage1 -> stage2 -> undo stage2 -> undo stage1.
   Both stages are add-mod-4 so this cannot really fail, which is exactly why
   it is cheap enough to just run every time rather than reason about. */
static int verify_roundtrip(void) {
    unsigned char orig[CRUMBS_PER_CHUNK], work[CRUMBS_PER_CHUNK];
    srand(4242);
    for (int trial = 0; trial < 200; trial++) {
        int amp = rand() % 3 + 1;
        uint64_t t2 = (uint64_t)rand() % STAGE2_COUNT;
        uint64_t seed = (uint64_t)rand() % STAGE1_COUNT;
        uint64_t k = prng_seed_key(seed);
        for (int i = 0; i < CRUMBS_PER_CHUNK; i++) orig[i] = (unsigned char)(rand() & 3);
        for (int i = 0; i < CRUMBS_PER_CHUNK; i++)
            work[i] = (unsigned char)((orig[i] + prng_crumb_keyed(k, (uint32_t)i)) & 3);
        s2_apply(work, CRUMBS_PER_CHUNK, t2, amp);
        s2_invert(work, CRUMBS_PER_CHUNK, t2, amp);
        for (int i = 0; i < CRUMBS_PER_CHUNK; i++)
            work[i] = (unsigned char)((work[i] - prng_crumb_keyed(k, (uint32_t)i)) & 3);
        if (memcmp(work, orig, CRUMBS_PER_CHUNK) != 0) return 1;
    }
    return 0;
}

int main(int argc, char **argv) {
    int n_chunks = argc > 1 ? atoi(argv[1]) : NUM_CHUNKS;
    if (argc > 2) STAGE1_BITS = atoi(argv[2]);
    if (argc > 3) STAGE2_BITS = atoi(argv[3]);
    if (STAGE1_BITS < 1 || STAGE1_BITS > 40 || STAGE2_BITS < 0 || STAGE2_BITS > 22) {
        fprintf(stderr, "stage1 bits 1..40, stage2 bits 0..22\n");
        return 1;
    }
    STAGE1_COUNT = 1ULL << STAGE1_BITS;
    STAGE2_COUNT = 1ULL << STAGE2_BITS;

    FILE *f = fopen(FILE_PATH, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", FILE_PATH);
        return 1;
    }

    init_pos_keys();
    init_lt();
    if (verify_roundtrip()) { fprintf(stderr, "round-trip self-test FAILED\n"); return 1; }

    S2_COST = (double)STAGE2_BITS + log2(3.0);
    init_s2bits();
    printf("stage1 %d bits (%llu seeds)  +  stage2 prngbit %d bits (%llu streams) x amp{1,2,3}\n"
           "  stage2 cost = %d + log2(3) = %.3f bits;   total budget = %.3f bits\n"
           "  EXACT joint search over all %llu triples per chunk (no beam)\n\n",
           STAGE1_BITS, (unsigned long long)STAGE1_COUNT,
           STAGE2_BITS, (unsigned long long)STAGE2_COUNT,
           STAGE2_BITS, S2_COST, (double)STAGE1_BITS + S2_COST,
           (unsigned long long)(STAGE1_COUNT * STAGE2_COUNT * 3));

    unsigned char bytebuf[CHUNK_BYTES];
    unsigned char crumbs[CRUMBS_PER_CHUNK];

    // chunk 0 with its winning seed actually applied, kept for the dump at the end
    unsigned char chunk0_out[CRUMBS_PER_CHUNK];
    long long     chunk0_seed = -1;
    int           chunk0_len  = 0;

    // running net stats across every chunk actually processed
    double net_sum = 0.0, net_min = 0.0, net_max = 0.0;
    long   net_n = 0;
    int    net_min_chunk = -1, net_max_chunk = -1;

    /* Second accounting, run alongside the first. The net above scores the
       transformed chunk against the 2.0 ceiling: it credits the coder with
       knowing the transformed histogram, but charges the do-nothing
       alternative as if the SAME coder had to store raw crumbs at a flat 2
       bits each. That is the one place the two sides are not held to the same
       standard -- a coder that gets a frequency model for the transformed
       stream gets one for the untransformed stream too, and the raw chunk's
       own entropy is 1.98, not 2.00.

       Charging both sides the same way, with an adaptive coder:
           do nothing : n*H_raw   + C
           transform  : n*H_best  + C + SEED_BITS
       The model-learning cost C -- roughly (A-1)/2*log2(n) ~ 10.5 bits at
       n=128 -- appears on BOTH sides and cancels, which is exactly why it
       does not need to be estimated. What survives is
           net_vs_raw = (H_raw - H_best)*n - SEED_BITS
       and the gap between the two numbers is (2.0 - H_raw)*n, whose expected
       value for a uniform 4-symbol source is (A-1)/(2*ln2) = 2.164 bits. That
       is the same 2.164 already named in the header comment as estimator
       bias; it enters here, through the baseline, and nowhere else. */
    double netraw_sum = 0.0, raw_ent_sum = 0.0;

    // stage-1+2 totals, scored the same two ways as stage 1 alone
    double p_net_sum = 0.0, p_netraw_sum = 0.0;
    double p_net_min = 0.0, p_net_max = 0.0;
    int    p_min_chunk = -1, p_max_chunk = -1;

    for (int chunk = 0; chunk < n_chunks; chunk++) {
        size_t got = fread(bytebuf, 1, CHUNK_BYTES, f);
        if (got == 0) {
            printf("=== EOF reached before chunk %d ===\n", chunk);
            break;
        }

        long freq[4] = {0, 0, 0, 0};
        long total = 0;
        for (size_t b = 0; b < got; b++) {
            unsigned char byte = bytebuf[b];
            for (int shift = 6; shift >= 0; shift -= 2) {
                unsigned char crumb = (byte >> shift) & 0x3;
                crumbs[total] = crumb;
                freq[crumb]++;
                total++;
            }
        }

        double entropy = crumb_entropy(freq, total);

        printf("chunk %-4d  entropy=%.4f/%.1f  freq: 00=%-5ld 01=%-5ld 10=%-5ld 11=%-5ld\n",
               chunk, entropy, MAX_CRUMB_ENTROPY, freq[0], freq[1], freq[2], freq[3]);

        // sweep every seed in the SEED_BITS-wide space, add it (mod 4) onto
        // this chunk's crumb stream, and keep whichever seed drives entropy lowest
        long long best_seed = -1;
        double best_entropy = MAX_CRUMB_ENTROPY + 1.0;
        long best_freq[4] = {0, 0, 0, 0};

        /* One fused pass. Splitting the stage-1 histogram by index parity costs
           nothing extra -- the crumbs are being bucketed anyway -- and once
           c0[] and c1[] exist, both parity candidates are 4 adds each rather
           than another scan of the chunk. So the EXACT joint optimum over all
           2^SEED_BITS * 2 pairs costs essentially the same as stage 1 alone. */
        const double base_D = 2.0 * (double)total - (double)total * log2((double)total);
        double   p_best_D    = -HUGE_VAL;
        uint64_t p_best_seed = 0;
        uint64_t p_best_t2   = 0;
        int      p_best_amp  = 1;
        long     p_best_freq[4] = {0, 0, 0, 0};

        int nw = (total + 63) / 64;
        for (uint64_t seed = 0; seed < STAGE1_COUNT; seed++) {
            uint64_t seed_key = prng_seed_key(seed);   // hoisted out of the position loop

            /* Stage-1 stream as four bitsets, one per symbol value: SM[v] has
               bit i set iff the stage-1 crumb at position i equals v. Built
               once per stage-1 seed, then reused by every stage-2 seed. */
            uint64_t SM[4][(CRUMBS_PER_CHUNK + 63) / 64];
            memset(SM, 0, sizeof(SM));
            long freq2[4] = {0, 0, 0, 0};
            for (long i = 0; i < total; i++) {
                unsigned char x = (unsigned char)((crumbs[i] + prng_crumb_keyed(seed_key, (uint32_t)i)) & 0x3);
                SM[x][i >> 6] |= 1ULL << (i & 63);
                freq2[x]++;
            }

            double e = crumb_entropy(freq2, total);
            if (e < best_entropy) {
                best_entropy = e;
                best_seed = (long long)seed;
                best_freq[0] = freq2[0]; best_freq[1] = freq2[1];
                best_freq[2] = freq2[2]; best_freq[3] = freq2[3];
            }

            /* g1[v] = how many crumbs of value v sit where the bitstream is 1.
               That is an AND plus a popcount per value -- 8 word ops for a
               128-crumb chunk instead of rescanning 128 crumbs, which is what
               makes the exact triple search affordable. Crumbs under a 1 bit
               move from bucket v to v+amp; the rest stay put. */
            for (uint64_t t2 = 0; t2 < STAGE2_COUNT; t2++) {
                const uint64_t *B = S2BITS + t2 * (uint64_t)NWORDS;
                long g1[4];
                for (int v = 0; v < 4; v++) {
                    long c = 0;
                    for (int w = 0; w < nw; w++) c += __builtin_popcountll(SM[v][w] & B[w]);
                    g1[v] = c;
                }
                for (int amp = 1; amp <= 3; amp++) {
                    long f[4];
                    for (int v = 0; v < 4; v++)
                        f[v] = freq2[v] - g1[v] + g1[(v - amp) & 3];
                    double D = base_D + LT[f[0]] + LT[f[1]] + LT[f[2]] + LT[f[3]];
                    if (D > p_best_D) {
                        p_best_D = D; p_best_seed = seed;
                        p_best_t2 = t2; p_best_amp = amp;
                        p_best_freq[0] = f[0]; p_best_freq[1] = f[1];
                        p_best_freq[2] = f[2]; p_best_freq[3] = f[3];
                    }
                }
            }
        }
        double p_best_ent = MAX_CRUMB_ENTROPY - p_best_D / (double)total;


        // the sweep only ever scored histograms; materialise the winner for chunk 0
        if (chunk == 0 && best_seed >= 0) {
            uint64_t k = prng_seed_key(p_best_seed);
            for (long i = 0; i < total; i++)
                chunk0_out[i] = (unsigned char)((crumbs[i] + prng_crumb_keyed(k, (uint32_t)i)) & 0x3);
            s2_apply(chunk0_out, (int)total, p_best_t2, p_best_amp);
            chunk0_seed = (long long)p_best_seed;
            chunk0_len  = (int)total;
        }

        double net = (MAX_CRUMB_ENTROPY - best_entropy) * (double)total - STAGE1_BITS;
        if (net_n == 0 || net < net_min) { net_min = net; net_min_chunk = chunk; }
        if (net_n == 0 || net > net_max) { net_max = net; net_max_chunk = chunk; }
        net_sum += net;
        net_n++;

        netraw_sum  += (entropy - best_entropy) * (double)total - STAGE1_BITS;
        raw_ent_sum += entropy;

        double p_net    = (MAX_CRUMB_ENTROPY - p_best_ent) * (double)total - STAGE1_BITS - S2_COST;
        double p_netraw = (entropy            - p_best_ent) * (double)total - STAGE1_BITS - S2_COST;
        if (net_n == 1 || p_net < p_net_min) { p_net_min = p_net; p_min_chunk = chunk; }
        if (net_n == 1 || p_net > p_net_max) { p_net_max = p_net; p_max_chunk = chunk; }
        p_net_sum    += p_net;
        p_netraw_sum += p_netraw;

        printf("  best add seed: seed=%-11lld (of %llu)  entropy=%.4f/%.1f  net=%.2f bits  freq: 00=%-5ld 01=%-5ld 10=%-5ld 11=%-5ld\n",
               best_seed, (unsigned long long)STAGE1_COUNT, best_entropy, MAX_CRUMB_ENTROPY, net,
               best_freq[0], best_freq[1], best_freq[2], best_freq[3]);
        printf("  + prngbit add: s1=%-11llu s2=%-8llu amp=%d  entropy=%.4f/%.1f  net=%.2f bits  freq: 00=%-5ld 01=%-5ld 10=%-5ld 11=%-5ld  (delta %+.2f)\n",
               (unsigned long long)p_best_seed, (unsigned long long)p_best_t2, p_best_amp,
               p_best_ent, MAX_CRUMB_ENTROPY, p_net,
               p_best_freq[0], p_best_freq[1], p_best_freq[2], p_best_freq[3], p_net - net);

        if (got < (size_t)CHUNK_BYTES) {
            printf("=== EOF reached mid-chunk ===\n");
            break;
        }
    }

    if (net_n > 0) {
        printf("\n=== net over %ld chunk%s (cb=%d, seed_bits=%d) ===\n",
               net_n, net_n == 1 ? "" : "s", CHUNK_BYTES, STAGE1_BITS);
        printf("  avg net = %7.2f bits/chunk\n", net_sum / (double)net_n);
        printf("  min net = %7.2f bits  (chunk %d)\n", net_min, net_min_chunk);
        printf("  max net = %7.2f bits  (chunk %d)\n", net_max, net_max_chunk);
        printf("  total   = %7.2f bits over %ld bytes\n",
               net_sum, net_n * (long)CHUNK_BYTES);

        double raw_mean = raw_ent_sum / (double)net_n;
        double gap      = net_sum / (double)net_n - netraw_sum / (double)net_n;
        printf("\n  --- same run, baseline held to the same standard ---\n");
        printf("  mean raw entropy      = %.4f/%.1f  (not 2.0 -- the raw chunk has a\n"
               "                                     histogram too, and any coder that\n"
               "                                     models the transformed one models it)\n",
               raw_mean, MAX_CRUMB_ENTROPY);
        printf("  net vs 2.0 ceiling    = %+7.3f bits/chunk   (the number above)\n",
               net_sum / (double)net_n);
        printf("  net vs raw entropy    = %+7.3f bits/chunk   (adaptive-coder model cost\n"
               "                                              cancels between the two sides)\n",
               netraw_sum / (double)net_n);
        printf("  baseline gap          = %+7.3f bits/chunk   (expected (A-1)/(2*ln2) = %.3f\n"
               "                                              for a uniform 4-symbol source)\n",
               gap, 3.0 / (2.0 * log(2.0)));

        printf("\n  --- stage1 (%d b) + prngbit add (%d b seed + 1.585 b amp = %.3f b) ---\n",
               STAGE1_BITS, STAGE2_BITS, S2_COST);
        printf("  vs 2.0 ceiling : stage1 %+7.3f  ->  +pattern %+7.3f   (delta %+7.3f)\n",
               net_sum / (double)net_n, p_net_sum / (double)net_n,
               (p_net_sum - net_sum) / (double)net_n);
        printf("  vs raw entropy : stage1 %+7.3f  ->  +pattern %+7.3f   (delta %+7.3f)\n",
               netraw_sum / (double)net_n, p_netraw_sum / (double)net_n,
               (p_netraw_sum - netraw_sum) / (double)net_n);
        printf("  min %+.2f (chunk %d)   max %+.2f (chunk %d)\n",
               p_net_min, p_min_chunk, p_net_max, p_max_chunk);
    }

    if (chunk0_len > 0) {
        printf("\n=== chunk 0 after prng add (seed=%lld) ===\n", chunk0_seed);
        printf("  %d crumbs, decimal 0-3:\n   ", chunk0_len);
        for (int i = 0; i < chunk0_len; i++) {
            printf(" %d", chunk0_out[i]);
            if ((i & 31) == 31 && i + 1 < chunk0_len) printf("\n   ");
        }
        printf("\n  %d bytes, decimal 0-255:\n   ", chunk0_len / 4);
        for (int i = 0; i + 3 < chunk0_len; i += 4) {
            int b = (chunk0_out[i]     << 6) | (chunk0_out[i + 1] << 4)
                  | (chunk0_out[i + 2] << 2) |  chunk0_out[i + 3];
            printf(" %d", b);
            if (((i / 4) & 15) == 15 && i + 4 < chunk0_len) printf("\n   ");
        }
        printf("\n");
    }

    fclose(f);
    return 0;
}
