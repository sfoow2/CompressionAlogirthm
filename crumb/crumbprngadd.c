#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <immintrin.h>   // AVX-512 (F + VPOPCNTDQ) -- confirmed present on this machine (11900H)

#define FILE_PATH "C:\\Users\\lukac\\Documents\\compressor\\actuallstuff\\CurrentDataFile.bin"

static int STAGE1_BITS  = 26;
static int CHUNK_BYTES  = 9;   // bytes read per chunk -- each byte expands to 4 crumbs (2 bits each)
/* 26/9 chosen 2026-08-21 via CrumbReduceCuda.cu: large-sample, held-out,
   random-chunk-sampled sweep under the forced-1peak selection (see
   FORCE_SHAPE below) found net/chunk peaking on a broad, flat plateau at
   36-44 crumbs (cb 9-11), s1 24-26, avg net ~2.92-2.96 bits/chunk -- a real
   jump from the old 24/15 default's ~2.5. cb=9 and cb=11 are a statistical
   tie (each won one of two independent validation runs); 9 was picked
   arbitrarily between them. Non-byte-aligned crumb counts (37, 38) were
   also swept and lost to their byte-aligned neighbors, so this is not an
   arbitrary rounding -- byte alignment measurably matters on this data. */
#define NUM_CHUNKS (32)         // number of chunks to process before stopping
    
#define MAX_CRUMB_ENTROPY 2.0   // ceiling for a 4-symbol (2-bit) alphabet
#define DUMP_CHUNK 3           // which chunk's crumbs/bytes get dumped at the end

static int crumbs_per_chunk;    // = CHUNK_BYTES * 4, set once CHUNK_BYTES is known
static int nw;                  // = ceil(crumbs_per_chunk / 64), words per value bitset

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
   table is built once for the run instead of recomputed on every inner step.
   Sized at runtime now (was a fixed CRUMBS_PER_CHUNK array) so CHUNK_BYTES
   can change without a recompile. */
static uint64_t *pos_key;
static void init_pos_keys(void) {
    pos_key = malloc((size_t)crumbs_per_chunk * sizeof(uint64_t));
    if (!pos_key) { fprintf(stderr, "out of memory for pos_key\n"); exit(1); }
    for (int i = 0; i < crumbs_per_chunk; i++)
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
   STAGE1_BITS was 16, fatal the moment it goes past 32. avalanche_hash is a
   bijection on uint64, so distinct seeds give distinct keys up to STAGE1_BITS 63. */
static inline uint64_t prng_seed_key(uint64_t seed) {
    return avalanche_hash(seed);
}
static inline unsigned char prng_crumb_keyed(uint64_t seed_key, uint32_t pos) {
    return (unsigned char)(avalanche_hash(seed_key ^ pos_key[pos]) & 0x3u);
}


static uint64_t STAGE1_COUNT;

/* ---------------------------------------------------------- keystream cache
   The layer-1 keystream is a function of (seed, position) only -- it never
   depends on the data -- so hashing it per chunk repeats identical work once
   per chunk. It is instead hashed ONCE here and kept as bitsets:
       KSM[seed][v] = the positions where seed's keystream crumb equals v
   The stage-1 output bitsets for a chunk then come out as
       SM[v] = OR over u of ( DM[u] AND KSM[seed][(v-u) & 3] )
   where DM[u] is the chunk's own value-u bitset. That is 16 ANDs and 12 ORs of
   nw words in place of `total` avalanche_hash calls per (chunk, seed) -- about
   32 word ops instead of 640 at 64 crumbs -- and the saving compounds with the
   chunk count, since the hashing is now amortised over the whole run.

   Costs 2^STAGE1_BITS * 4 * nw * 8 bytes. nw only grows once CHUNK_BYTES
   exceeds 16 (64 crumbs > one 64-bit word) -- below that, changing crumb
   count does NOT change the cache size at all, only STAGE1_BITS does. The
   real number for whatever settings are in use is printed at startup below;
   trust that printout over any comment here. */
static uint64_t *KSM;

static void build_ksm(void) {
    size_t words = (size_t)STAGE1_COUNT * 4 * (size_t)nw;
    KSM = calloc(words, sizeof(uint64_t));
    if (!KSM) {
        fprintf(stderr, "out of memory for keystream cache (%.1f MB); "
                        "lower STAGE1_BITS\n", words * 8.0 / 1e6);
        exit(1);
    }
    for (uint64_t s = 0; s < STAGE1_COUNT; s++) {
        uint64_t key = prng_seed_key(s);
        uint64_t *row = KSM + (size_t)s * 4 * (size_t)nw;
        for (int i = 0; i < crumbs_per_chunk; i++)
            row[(size_t)prng_crumb_keyed(key, (uint32_t)i) * (size_t)nw + (i >> 6)]
                |= 1ULL << (i & 63);
    }

}

/* LT[f] = f*log2(f), so a histogram scores in 4 lookups instead of 4 log2 calls */
static double *LT;
static void init_lt(void) {
    LT = malloc((size_t)(crumbs_per_chunk + 1) * sizeof(double));
    if (!LT) { fprintf(stderr, "out of memory for LT\n"); exit(1); }
    LT[0] = 0.0;
    for (int f = 1; f <= crumbs_per_chunk; f++) LT[f] = (double)f * log2((double)f);
}

/* Round-trip: raw -> layer 1 -> undo layer 1. An add mod 4 cannot really fail,
   which is exactly why it is cheap enough to just run every time. */
static int verify_roundtrip(void) {
    unsigned char *orig = malloc((size_t)crumbs_per_chunk);
    unsigned char *work = malloc((size_t)crumbs_per_chunk);
    int failed = 0;
    srand(4242);
    for (int trial = 0; trial < 200 && !failed; trial++) {
        uint64_t seed = (uint64_t)rand() % STAGE1_COUNT;
        uint64_t k = prng_seed_key(seed);
        for (int i = 0; i < crumbs_per_chunk; i++) orig[i] = (unsigned char)(rand() & 3);
        for (int i = 0; i < crumbs_per_chunk; i++)
            work[i] = (unsigned char)((orig[i] + prng_crumb_keyed(k, (uint32_t)i)) & 3);
        for (int i = 0; i < crumbs_per_chunk; i++)
            work[i] = (unsigned char)((work[i] - prng_crumb_keyed(k, (uint32_t)i)) & 3);
        if (memcmp(work, orig, (size_t)crumbs_per_chunk) != 0) failed = 1;
    }
    free(orig); free(work);
    return failed;
}

/* per-depth bests, and the scoring constants the recursion needs */
/* ns per seed PER WORD of chunk, measured on this machine: 11.8 ns at nw=1
   (cb=14), from the marginal cost between a 256- and a 1024-chunk run so that
   cache build and startup are differenced out. Cost per seed is FLAT in the
   seed count -- that is exactly what "time = c * 2^seed_bits" means -- and
   scales with nw because the inner body runs once per 64-crumb word. */
#define NS_PER_COMBO 11.8

static const char *fmt_time(double s) {
    static char buf[64];
    if      (s <      90.0) snprintf(buf, sizeof buf, "%.1f s",  s);
    else if (s <    5400.0) snprintf(buf, sizeof buf, "%.1f min", s / 60.0);
    else if (s <  172800.0) snprintf(buf, sizeof buf, "%.1f h",  s / 3600.0);
    else if (s < 3.15e9)    snprintf(buf, sizeof buf, "%.1f days", s / 86400.0);
    else                    snprintf(buf, sizeof buf, "%.3g years", s / 3.156e7);
    return buf;
}

/* best deficit per chunk, filled by the sweep (seed-outermost, so per-chunk
   results only exist once it finishes) */
static double *bestD;

/* ---------------------------------------------------- histogram shape types
   The post-transform histograms fall into a handful of recognisable shapes.
   Classified on the bins SORTED DESCENDING and measured as ratios to the
   largest bin, so the label depends on shape alone and not on chunk size.

       a >= b >= c >= d,  rb = b/a, rc = c/a, rd = d/a

   Note what this is and is not. Net is a deterministic function of the
   histogram -- net = (2 - H(h))*n - bits -- so these labels are a coarse
   binning of net itself, not independent information about a chunk. They are
   here to make the shape distribution visible, not to feed a decision. */
static const char *TYPE_NAME[5] = {
    "3way-bal", "2big-1med", "1peak", "2peak", "flat" };

/* The winning seed per chunk is no longer the unconstrained entropy-deficit
   best -- it is the best deficit among seeds whose resulting histogram
   classifies as FORCE_SHAPE (must match a TYPE_NAME index above). A chunk
   with zero qualifying seeds in the whole STAGE1_COUNT sweep is possible at
   small seed_bits, so every consumer of best_s/bestD below must treat
   best_s[c] == -1 as "no shape-matching seed found" rather than assuming the
   search always succeeds the way the old unconstrained version did. */
#define FORCE_SHAPE 2   /* 1peak */

static int hist_type(const long f[4]) {
    long s[4] = { f[0], f[1], f[2], f[3] };
    for (int i = 0; i < 3; i++)          /* sort descending, n=4 so bubble is fine */
        for (int j = 0; j < 3 - i; j++)
            if (s[j] < s[j+1]) { long t = s[j]; s[j] = s[j+1]; s[j+1] = t; }
    if (s[0] == 0) return 4;
    double rb = (double)s[1] / s[0], rc = (double)s[2] / s[0], rd = (double)s[3] / s[0];
    if (rd > 0.60) return 4;             /* all four bins comparable        */
    if (rc > 0.65) return 0;             /* three similar, fourth spent     */
    if (rb > 0.80) return 3;             /* two tied peaks over a small rest */
    if (rb < 0.55) return 2;             /* one bin dominates everything    */
    return 1;                            /* steady descent                  */
}

int main(int argc, char **argv) {
    int n_chunks = argc > 1 ? atoi(argv[1]) : NUM_CHUNKS;
    if (argc > 2) STAGE1_BITS = atoi(argv[2]);
    if (argc > 3) CHUNK_BYTES = atoi(argv[3]);
    if (STAGE1_BITS < 1 || STAGE1_BITS > 40) {
        fprintf(stderr,
            "usage: %s [chunks] [seed_bits] [chunk_bytes]\n"
            "  seed_bits 1..40, chunk_bytes >=1 (default 16).\n"
            "  CTDT_SHOW=n sets how many per-chunk rows to print.\n"
            "  CTDT_FORCE_TYPE=0..4 makes CTDT_HIST dump each chunk's best\n"
            "  net>0 seed WITHIN that shape type instead of the global best.\n",
            argv[0]);
        return 1;
    }
    if (CHUNK_BYTES < 1) {
        fprintf(stderr, "chunk_bytes must be >= 1\n");
        return 1;
    }
    crumbs_per_chunk = CHUNK_BYTES * 4;
    nw = (crumbs_per_chunk + 63) / 64;
    STAGE1_COUNT = 1ULL << STAGE1_BITS;

    const char *file_path = argc > 4 ? argv[4] : FILE_PATH;
    FILE *f = fopen(file_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", file_path);
        return 1;
    }
    /* CTDT_SKIP=n starts n chunks into the file. This exists so a config picked
       as the best of a grid can be re-scored on chunks it was NOT picked on --
       without it, every "confirmation" at a larger chunk count still contains
       the sample that did the selecting, and best-of-N selection inflates by
       about 1 SE on shared data. */
    long skip_chunks = 0;
    { const char *e = getenv("CTDT_SKIP"); if (e) skip_chunks = atol(e); }
    if (skip_chunks > 0 &&
        fseek(f, skip_chunks * (long)CHUNK_BYTES, SEEK_SET) != 0) {
        fprintf(stderr, "cannot skip %ld chunks\n", skip_chunks);
        return 1;
    }

    init_pos_keys();
    init_lt();
    build_ksm();
    if (verify_roundtrip()) { fprintf(stderr, "round-trip self-test FAILED\n"); return 1; }

    {
        double combos  = (double)STAGE1_COUNT;
        double cachemb = combos * 4 * nw * 8 / 1e6;
        /* The estimate is SEARCH ONLY and deliberately so: the cache is already
           built by the time this prints (build_ksm ran above), and it is not a
           meaningful cost anyway -- measured 0.26 s to build the whole 2^21
           table, against 26 s of search at 1024 chunks. Startup is a rounding
           error; the search is 2^seed_bits per chunk and that is what matters.
           NS_PER_COMBO is calibrated per WORD of chunk, since the inner body is
           4 ANDs + 3 ORs + 1 popcount per value per word. */
        double secs = combos * NS_PER_COMBO * nw * 1e-9 * (double)n_chunks;
        double full = combos * NS_PER_COMBO * nw * 1e-9
                    * (198766075.0 / (double)CHUNK_BYTES);
        printf("seed %d bits (%llu seeds), chunk %d B = %d crumbs (nw=%d)\n",
               STAGE1_BITS, (unsigned long long)STAGE1_COUNT,
               CHUNK_BYTES, crumbs_per_chunk, nw);
        printf("  overhead %d bits/chunk;  EXACT search over %.3g seeds; cache %.1f MB\n",
               STAGE1_BITS, combos, cachemb);
        printf("  est. search %s for %d chunks", fmt_time(secs), n_chunks);
        printf("  |  whole 198 MB file: %s\n\n", fmt_time(full));
        fflush(stdout);
    }

    unsigned char *bytebuf   = malloc((size_t)CHUNK_BYTES);
    unsigned char *crumbs    = malloc((size_t)crumbs_per_chunk);

    // chunk DUMP_CHUNK with its winning seed actually applied, kept for the dump at the end
    unsigned char *chunk0_out = malloc((size_t)crumbs_per_chunk);
    long long     chunk0_seed = -1;
    int           chunk0_len  = 0;

    // running net stats across every chunk actually processed
    double net_sum = 0.0, net_min = 0.0, net_max = 0.0;
    long   net_n = 0;
    double netraw_sum = 0.0, raw_ent_sum = 0.0;
    int    net_min_chunk = -1, net_max_chunk = -1;

    long   *all_total  = malloc((size_t)n_chunks * sizeof(long));
    double *all_rawent = malloc((size_t)n_chunks * sizeof(double));
    long   *all_rawfrq = malloc((size_t)n_chunks * 4 * sizeof(long));
    /* Transposed [u][w][c] layout (chunk axis innermost/contiguous), not the
       natural [c][u][w] -- so the AVX-512 search loop below can load 8
       consecutive chunks' DM words with one vector load instead of a gather.
       Same axis-to-vectorize-must-be-contiguous lesson as coalesced GPU
       memory access. calloc zeroes it once; no per-chunk memset needed since
       each chunk's slot is only ever written by that chunk. */
    uint64_t *all_DM_T = calloc((size_t)4 * (size_t)nw * (size_t)n_chunks, sizeof(uint64_t));
    double *best_D     = malloc((size_t)n_chunks * sizeof(double));
    long long *best_s  = malloc((size_t)n_chunks * sizeof(long long));
    long long *best_t  = malloc((size_t)n_chunks * sizeof(long long));
    long long *best_u  = malloc((size_t)n_chunks * sizeof(long long));
    double *bestL2_D   = malloc((size_t)n_chunks * sizeof(double));
    double *bestL1_D   = malloc((size_t)n_chunks * sizeof(double));
    /* calloc, not malloc: a chunk with no FORCE_SHAPE-matching seed leaves
       best_f untouched by the search loop, and hist_type(0,0,0,0) (via the
       s[0]==0 check) safely falls out as "flat" instead of reading garbage. */
    long   *best_f     = calloc((size_t)n_chunks * 4, sizeof(long));
    unsigned char *chunk0_raw = malloc((size_t)crumbs_per_chunk);
    /* per (chunk, shape type) best deficit among seeds that are BOTH net>0 AND
       classify to that type -- lets us ask "if every chunk were forced onto
       shape X, how many could still clear net>0, and what would the net be"
       instead of always taking the single best seed regardless of shape. */
    double    *bestD_ty = malloc((size_t)n_chunks * 5 * sizeof(double));
    long long *best_s_ty = malloc((size_t)n_chunks * 5 * sizeof(long long));
    long      *best_f_ty = malloc((size_t)n_chunks * 5 * 4 * sizeof(long));
    if (!bytebuf || !crumbs || !chunk0_out || !all_total || !all_rawent || !all_rawfrq
        || !all_DM_T || !best_D || !best_s || !best_f || !chunk0_raw
        || !bestD_ty || !best_s_ty || !best_f_ty) {
        fprintf(stderr, "out of memory for per-chunk state\n"); return 1;
    }

    int nc = 0;
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        size_t got = fread(bytebuf, 1, (size_t)CHUNK_BYTES, f);
        if (got == 0) break;
        long freq[4] = {0, 0, 0, 0};
        long total = 0;
        for (size_t b = 0; b < got; b++)
            for (int shift = 6; shift >= 0; shift -= 2) {
                unsigned char crumb = (unsigned char)((bytebuf[b] >> shift) & 0x3);
                crumbs[total] = crumb;
                all_DM_T[((size_t)crumb * nw + (total >> 6)) * (size_t)n_chunks + nc]
                    |= 1ULL << (total & 63);
                freq[crumb]++;
                total++;
            }
        if (nc == DUMP_CHUNK) memcpy(chunk0_raw, crumbs, (size_t)total);
        all_total[nc]  = total;
        all_rawent[nc] = crumb_entropy(freq, total);
        for (int v = 0; v < 4; v++) all_rawfrq[nc*4+v] = freq[v];
        best_D[nc] = -HUGE_VAL; best_s[nc] = -1; best_t[nc] = 0;
        bestL1_D[nc] = -HUGE_VAL; bestL2_D[nc] = -HUGE_VAL; best_u[nc] = 0;
        for (int t = 0; t < 5; t++) { bestD_ty[nc*5+t] = -HUGE_VAL; best_s_ty[nc*5+t] = -1; }
        nc++;
        if (got < (size_t)CHUNK_BYTES) break;
    }
    if (nc == 0) { fprintf(stderr, "no chunks read\n"); return 1; }

    bestD = malloc((size_t)n_chunks * sizeof(double));
    if (!bestD) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int c = 0; c < n_chunks; c++) bestD[c] = -HUGE_VAL;

    /* per-chunk count of how many of the STAGE1_COUNT seeds give a net
       (deficit - STAGE1_BITS) that is positive vs. non-positive -- tallied
       across the WHOLE sweep, not just at the best seed. */
    long *cnt_pos = calloc((size_t)n_chunks, sizeof(long));
    long *cnt_neg = calloc((size_t)n_chunks, sizeof(long));
    if (!cnt_pos || !cnt_neg) { fprintf(stderr, "out of memory for net counters\n"); return 1; }
    const double net_cost = (double)STAGE1_BITS;

    /* Score by entropy DEFICIT, not entropy, so the inner loop never calls
       log2(): (2-H)*n = 2n - n*log2(n) + SUM_v f_v*log2(f_v), and the sum is
       four LT[] lookups. Minimising H is the same as maximising this. */
    double base_D_full = 2.0 * (double)crumbs_per_chunk
                       - (double)crumbs_per_chunk * log2((double)crumbs_per_chunk);

    /* AVX-512 fast path: at nw==1 (chunk <=64 crumbs, true for every setting
       this project has found competitive), the keystream is 4 scalar words
       per seed -- broadcast them ONCE per seed, then AND/OR/popcount 8
       chunks at a time via VPOPCNTDQ instead of one chunk at a time. Falls
       back to the original scalar path (now reading the transposed layout)
       for nw>1 and for the <8-chunk remainder. Verified bit-for-bit against
       the pre-SIMD scalar version before being trusted -- see
       compressthisdamthing_scalar_ref.c. */
    for (uint64_t seed = 0; seed < STAGE1_COUNT; seed++) {
        const uint64_t *K = KSM + (size_t)seed * 4 * (size_t)nw;
        int c = 0;

        if (nw == 1) {
            __m512i kb[4][4];
            for (int v = 0; v < 4; v++)
                for (int u = 0; u < 4; u++)
                    kb[v][u] = _mm512_set1_epi64((long long)K[(size_t)((v - u) & 3)]);

            for (; c + 8 <= nc; c += 8) {
                __m512i dm[4];
                for (int u = 0; u < 4; u++)
                    dm[u] = _mm512_loadu_si512((const void *)&all_DM_T[(size_t)u * n_chunks + c]);

                uint64_t fr[4][8];
                for (int v = 0; v < 4; v++) {
                    __m512i acc = _mm512_setzero_si512();
                    for (int u = 0; u < 4; u++)
                        acc = _mm512_or_si512(acc, _mm512_and_si512(dm[u], kb[v][u]));
                    _mm512_storeu_si512((void *)fr[v], _mm512_popcnt_epi64(acc));
                }

                for (int lane = 0; lane < 8; lane++) {
                    int cc = c + lane;
                    long f0 = (long)fr[0][lane], f1 = (long)fr[1][lane],
                         f2 = (long)fr[2][lane], f3 = (long)fr[3][lane];
                    double D = base_D_full + LT[f0] + LT[f1] + LT[f2] + LT[f3];
                    if (D - net_cost > 0.0) cnt_pos[cc]++; else cnt_neg[cc]++;
                    long fr4[4] = { f0, f1, f2, f3 };
                    int ty = hist_type(fr4);
                    if (ty == FORCE_SHAPE && D > bestD[cc]) {
                        bestD[cc] = D;
                        best_s[cc] = (long long)seed;
                        best_f[cc*4+0] = f0; best_f[cc*4+1] = f1;
                        best_f[cc*4+2] = f2; best_f[cc*4+3] = f3;
                    }
                    if (D - net_cost > 0.0) {
                        if (D > bestD_ty[cc*5+ty]) {
                            bestD_ty[cc*5+ty] = D;
                            best_s_ty[cc*5+ty] = (long long)seed;
                            long *bf = &best_f_ty[(cc*5+ty)*4];
                            bf[0] = f0; bf[1] = f1; bf[2] = f2; bf[3] = f3;
                        }
                    }
                }
            }
        }

        for (; c < nc; c++) {
            long fr[4];
            for (int v = 0; v < 4; v++) {
                long q = 0;
                for (int w = 0; w < nw; w++) {
                    uint64_t acc = 0;
                    for (int u = 0; u < 4; u++)
                        acc |= all_DM_T[((size_t)u * nw + w) * (size_t)n_chunks + c]
                             & K[(size_t)((v - u) & 3) * (size_t)nw + w];
                    q += __builtin_popcountll(acc);
                }
                fr[v] = q;
            }
            double D = base_D_full + LT[fr[0]] + LT[fr[1]] + LT[fr[2]] + LT[fr[3]];
            if (D - net_cost > 0.0) cnt_pos[c]++; else cnt_neg[c]++;
            int ty = hist_type(fr);
            if (ty == FORCE_SHAPE && D > bestD[c]) {
                bestD[c] = D;
                best_s[c] = (long long)seed;
                for (int v = 0; v < 4; v++) best_f[c*4+v] = fr[v];
            }
            if (D - net_cost > 0.0) {
                if (D > bestD_ty[c*5+ty]) {
                    bestD_ty[c*5+ty] = D;
                    best_s_ty[c*5+ty] = (long long)seed;
                    long *bf = &best_f_ty[(c*5+ty)*4];
                    for (int v = 0; v < 4; v++) bf[v] = fr[v];
                }
            }
        }
    }

    const double cost = STAGE1_BITS;

    FILE *hf = NULL;
    { const char *e = getenv("CTDT_HIST");
      if (e) {
          hf = fopen(e, "wb");
          if (!hf) { fprintf(stderr, "cannot write %s\n", e); return 1; }
      } }

    /* CTDT_FORCE_TYPE=0..4 redirects the CTDT_HIST dump from each chunk's
       unconstrained best seed to its best net>0 seed WITHIN that shape type --
       so every line in the file is the same TYPE_NAME shape (or the sentinel
       -1/-1/-1/-1 on chunks where that shape never clears net>0 at all). */
    int force_type = -1;
    { const char *e = getenv("CTDT_FORCE_TYPE");
      if (e) {
          force_type = atoi(e);
          if (force_type < 0 || force_type > 4) {
              fprintf(stderr, "CTDT_FORCE_TYPE must be 0..4\n"); return 1;
          }
      } }

    long   tcount[5] = {0,0,0,0,0};   /* chunks per histogram shape */
    double tnet[5]   = {0,0,0,0,0};

    int show = 24;
    { const char *e = getenv("CTDT_SHOW"); if (e) show = atoi(e); }
    if (show > nc) show = nc;
    if (show > 0) {
        printf("=== per chunk (first %d of %d) ===\n", show, nc);
        printf("  %5s %8s %8s %10s  %-18s %-18s %-10s %8s  %12s %12s\n",
               "chunk", "H_raw", "H_best", "seed",
               "histogram raw", "histogram best", "type", "net",
               "seeds net>0", "seeds net<=0");
    }

    long no_shape = 0;   /* chunks where NO seed in the whole sweep gave FORCE_SHAPE */

    for (int chunk = 0; chunk < nc; chunk++) {
        long   total     = all_total[chunk];
        double entropy   = all_rawent[chunk];
        int    have_best = best_s[chunk] >= 0;
        double best_entropy = have_best
            ? MAX_CRUMB_ENTROPY - bestD[chunk] / (double)total : MAX_CRUMB_ENTROPY;

        if (hf) {
            if (!have_best)
                fprintf(hf, "-1/-1/-1/-1\n");
            else if (force_type < 0)
                fprintf(hf, "%ld/%ld/%ld/%ld\n",
                        best_f[chunk*4+0], best_f[chunk*4+1],
                        best_f[chunk*4+2], best_f[chunk*4+3]);
            else if (bestD_ty[chunk*5+force_type] > -HUGE_VAL) {
                long *bf = &best_f_ty[(chunk*5+force_type)*4];
                fprintf(hf, "%ld/%ld/%ld/%ld\n", bf[0], bf[1], bf[2], bf[3]);
            } else
                fprintf(hf, "-1/-1/-1/-1\n");
        }

        if (chunk < show) {
            char hr[32], hb[32];
            snprintf(hr, sizeof hr, "%ld/%ld/%ld/%ld",
                     all_rawfrq[chunk*4+0], all_rawfrq[chunk*4+1],
                     all_rawfrq[chunk*4+2], all_rawfrq[chunk*4+3]);
            if (!have_best) {
                char none[32];
                snprintf(none, sizeof none, "(no %s)", TYPE_NAME[FORCE_SHAPE]);
                printf("  %5d %8.4f %8s %10s  %-18s %-18s %-10s %8s  %12ld %12ld\n",
                       chunk, entropy, "--", "--", hr, none,
                       "--", "--", cnt_pos[chunk], cnt_neg[chunk]);
            } else {
                snprintf(hb, sizeof hb, "%ld/%ld/%ld/%ld",
                         best_f[chunk*4+0], best_f[chunk*4+1],
                         best_f[chunk*4+2], best_f[chunk*4+3]);
                printf("  %5d %8.4f %8.4f %10lld  %-18s %-18s %-10s %+8.2f  %12ld %12ld\n",
                       chunk, entropy, best_entropy, best_s[chunk], hr, hb,
                       TYPE_NAME[hist_type(&best_f[chunk*4])],
                       bestD[chunk] - cost, cnt_pos[chunk], cnt_neg[chunk]);
            }
        }

        if (!have_best) { no_shape++; continue; }

        { int ty = hist_type(&best_f[chunk*4]);
          tcount[ty]++; tnet[ty] += bestD[chunk] - cost; }

        double net = bestD[chunk] - cost;
        if (net_n == 0 || net < net_min) { net_min = net; net_min_chunk = chunk; }
        if (net_n == 0 || net > net_max) { net_max = net; net_max_chunk = chunk; }
        net_sum += net;
        net_n++;
        netraw_sum  += (entropy - best_entropy) * (double)total - cost;
        raw_ent_sum += entropy;

        if (chunk == DUMP_CHUNK && best_s[DUMP_CHUNK] >= 0) {
            uint64_t k = prng_seed_key((uint64_t)best_s[DUMP_CHUNK]);
            for (long i = 0; i < total; i++)
                chunk0_out[i] = (unsigned char)((chunk0_raw[i] + prng_crumb_keyed(k, (uint32_t)i)) & 0x3);
            chunk0_seed = best_s[DUMP_CHUNK];
            chunk0_len  = (int)total;
        }
    }

    if (no_shape)
        printf("  %ld/%d chunks had NO seed producing %s in the whole %.3g-seed "
               "sweep (excluded from the net stats below)\n",
               no_shape, nc, TYPE_NAME[FORCE_SHAPE], (double)STAGE1_COUNT);

    if (net_n > 0) {
        printf("=== net over %ld chunk%s (cb=%d, %.0f-bit seed) ===\n",
               net_n, net_n == 1 ? "" : "s", CHUNK_BYTES, cost);
        printf("  avg net = %7.3f bits/chunk   min %+.2f (chunk %d)   max %+.2f (chunk %d)\n",
               net_sum / (double)net_n, net_min, net_min_chunk, net_max, net_max_chunk);
        printf("  vs raw entropy instead of the 2.0 ceiling: %+.3f bits/chunk\n",
               netraw_sum / (double)net_n);

        printf("\n  --- histogram shape mix ---\n");
        for (int k = 0; k < 5; k++) {
            if (tcount[k])
                printf("  %-10s %6ld  %5.1f%%   mean net %+7.3f\n",
                       TYPE_NAME[k], tcount[k],
                       100.0 * tcount[k] / (double)net_n, tnet[k] / tcount[k]);
            else
                printf("  %-10s %6d   0.0%%\n", TYPE_NAME[k], 0);
        }

        /* Different question from the mix above: not "what shape did the best
           seed land on", but "if I FORCE this chunk onto shape X (giving up
           whatever seed the unconstrained search actually preferred), can I
           still find a net>0 seed of that shape, and what net does it cost
           me". A type with high coverage here is a candidate for a shared/
           implicit type across chunks -- you would not need to transmit the
           type per chunk if (almost) every chunk can hit it anyway. */
        printf("\n  --- forced-type coverage (best net>0 seed WITHIN each shape) ---\n");
        printf("  %-10s %8s %7s   %10s %10s\n",
               "type", "coverage", "of nc", "mean net", "vs unconstrained");
        for (int t = 0; t < 5; t++) {
            long   cover = 0;
            double sumnet = 0.0, sumgap = 0.0;
            for (int c = 0; c < nc; c++) {
                if (bestD_ty[c*5+t] > -HUGE_VAL) {
                    cover++;
                    double net_t = bestD_ty[c*5+t] - cost;
                    sumnet += net_t;
                    sumgap += (bestD[c] - cost) - net_t;   /* best - forced, always >= 0 */
                }
            }
            if (cover)
                printf("  %-10s %6ld/%-4d %6.1f%%   %+10.3f %+10.3f\n",
                       TYPE_NAME[t], cover, nc, 100.0 * cover / (double)nc,
                       sumnet / cover, sumgap / cover);
            else
                printf("  %-10s %6ld/%-4d %6.1f%%\n", TYPE_NAME[t], 0L, nc, 0.0);
        }
    }

    if (hf) {
        fclose(hf);
        printf("\n  wrote %d per-chunk histograms to %s\n",
               nc, getenv("CTDT_HIST"));
    }

    if (chunk0_len > 0) {
        printf("\n=== chunk %d after layer-1 add (seed=%lld) ===\n", DUMP_CHUNK, chunk0_seed);
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

    /* CTDT_RESIDUAL=path writes the actual layer-1 OUTPUT (winning seed applied,
       packed 4 crumbs/byte) for every processed chunk, not just DUMP_CHUNK --
       so a downstream tool (e.g. chunkgrid_cuda) can resweep chunk-size x S1
       fresh on the real post-layer-1 residual instead of the raw file, to test
       whether any grain still has exploitable skew. Crumb values are recovered
       from all_DM_T (one-hot per position across the 4 value planes) since the
       plain crumb array itself is not retained per-chunk after the search. */
    { const char *e = getenv("CTDT_RESIDUAL");
      if (e) {
          FILE *rf = fopen(e, "wb");
          if (!rf) { fprintf(stderr, "cannot write %s\n", e); return 1; }
          for (int c = 0; c < nc; c++) {
              /* no FORCE_SHAPE seed found for this chunk -- key 0 leaves the
                 keystream well-defined (still a valid, just unsearched, seed)
                 rather than reading the -1 sentinel as a huge bogus seed. */
              uint64_t k = prng_seed_key(best_s[c] >= 0 ? (uint64_t)best_s[c] : 0);
              int total = (int)all_total[c];
              for (int i = 0; i < total; i += 4) {
                  int b = 0;
                  for (int j = 0; j < 4; j++) {
                      int pos = i + j;
                      int w = pos >> 6, bit = pos & 63;
                      unsigned char crumb = 0;
                      for (int v = 0; v < 4; v++) {
                          uint64_t word = all_DM_T[((size_t)v * nw + w) * (size_t)n_chunks + c];
                          if ((word >> bit) & 1ULL) { crumb = (unsigned char)v; break; }
                      }
                      unsigned char out = (unsigned char)((crumb + prng_crumb_keyed(k, (uint32_t)pos)) & 3u);
                      b = (b << 2) | out;
                  }
                  unsigned char byte = (unsigned char)b;
                  fwrite(&byte, 1, 1, rf);
              }
          }
          fclose(rf);
          fprintf(stderr, "wrote residual (%d chunks, %d bytes/chunk) to %s\n", nc, CHUNK_BYTES, e);
      }
    }

    fclose(f);
    return 0;
}
