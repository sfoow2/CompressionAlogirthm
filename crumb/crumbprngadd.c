#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include <immintrin.h>
#include <omp.h>

#define FILE_PATH "C:\\Users\\lukac\\Documents\\compressor\\actuallstuff\\CurrentDataFile.bin"

static int STAGE1_BITS  = 16;
static int CHUNK_BYTES  = 7;   // bytes read per chunk -- each byte expands to 4 crumbs (2 bits each)

#define NUM_CHUNKS (8)         // number of chunks to process before stopping

#define MAX_CRUMB_ENTROPY 2.0   // ceiling for a 4-symbol (2-bit) alphabet
#define DUMP_CHUNK 4           // which chunk's crumbs/bytes get dumped at the end

static int crumbs_per_chunk;    // = CHUNK_BYTES * 4, set once CHUNK_BYTES is known

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
   (seed, pos) pairs give unrelated crumb streams. */
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
static uint64_t *pos_key;
static void init_pos_keys(void) {
    pos_key = malloc((size_t)crumbs_per_chunk * sizeof(uint64_t));
    if (!pos_key) { fprintf(stderr, "out of memory for pos_key\n"); exit(1); }
    for (int i = 0; i < crumbs_per_chunk; i++)
        pos_key[i] = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
}

/* The crumb stream is  avalanche_hash( avalanche_hash(seed) ^ pos_key[pos] ).
   The inner avalanche_hash(seed) is invariant across positions, so it is split
   out as seed_key and hoisted to the top of any per-seed loop.

   Hashing the seed to a full 64-bit key before folding in pos is what stops
   seeds aliasing -- avalanche_hash is a bijection on uint64, so distinct
   seeds give distinct keys up to STAGE1_BITS 63. */
static inline uint64_t prng_seed_key(uint64_t seed) {
    return avalanche_hash(seed);
}
static inline unsigned char prng_crumb_keyed(uint64_t seed_key, uint32_t pos) {
    return (unsigned char)(avalanche_hash(seed_key ^ pos_key[pos]) & 0x3u);
}
static void seed_keystream(uint64_t seed, unsigned char *out) {
    uint64_t key = prng_seed_key(seed);
    for (int i = 0; i < crumbs_per_chunk; i++) out[i] = prng_crumb_keyed(key, (uint32_t)i);
}

static uint64_t STAGE1_COUNT;

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

static const char *fmt_time(double s) {
    static char buf[64];
    if      (s <      90.0) snprintf(buf, sizeof buf, "%.1f s",  s);
    else if (s <    5400.0) snprintf(buf, sizeof buf, "%.1f min", s / 60.0);
    else if (s <  172800.0) snprintf(buf, sizeof buf, "%.1f h",  s / 3600.0);
    else if (s < 3.15e9)    snprintf(buf, sizeof buf, "%.1f days", s / 86400.0);
    else                    snprintf(buf, sizeof buf, "%.3g years", s / 3.156e7);
    return buf;
}

/* best deficit per chunk, filled by the exhaustive seed-search loop */
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

/* The winning seed per chunk is simply the best deficit among ALL
   STAGE1_COUNT seeds, found by exhaustive scan -- shape-UNRESTRICTED (see
   exhaustive_query_chunk below). FORCE_SHAPE/hist_type are kept only to
   LABEL the winning histogram's shape for the shape-mix display below, not
   to filter candidates -- classifying the winner never changes which seed
   wins. */
#define FORCE_SHAPE 2   /* 1peak -- label only, see above */

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

/* ---- AVX-512 avalanche_hash, 8 lanes at once (needs AVX512F/DQ/BW,
   confirmed present on this machine) ---- */
static inline __m512i avalanche_hash_x8(__m512i x) {
    x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 30));
    x = _mm512_mullo_epi64(x, _mm512_set1_epi64((long long)0xbf58476d1ce4e5b9ULL));
    x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 27));
    x = _mm512_mullo_epi64(x, _mm512_set1_epi64((long long)0x94d049bb133111ebULL));
    x = _mm512_xor_si512(x, _mm512_srli_epi64(x, 31));
    return x;
}

/* Computes the (f0,f1,f2,f3) histogram for 16 seeds against `data` at once --
   two interleaved 8-lane groups, so the CPU has a second, unrelated
   multiply-latency chain to execute while the first is still in flight
   (validated ~1.5x over scalar with 0 mismatches in MIHSimdScore.c,
   2026-08-22). Histogram-only, not D/shape -- exhaustive_query_chunk below
   applies the exact same LT/hist_type/bestD bookkeeping the scalar fallback
   uses, so tie-breaking (first max wins, ascending seed order) is untouched. */
static void score_x16_hist(const uint64_t seeds[16], const unsigned char *data,
                            long f0[16], long f1[16], long f2[16], long f3[16]) {
    __m512i sraw0 = _mm512_loadu_si512((const void*)(seeds + 0));
    __m512i sraw1 = _mm512_loadu_si512((const void*)(seeds + 8));
    __m512i skey0 = avalanche_hash_x8(sraw0);
    __m512i skey1 = avalanche_hash_x8(sraw1);

    __m512i c0a = _mm512_setzero_si512(), c1a = _mm512_setzero_si512();
    __m512i c2a = _mm512_setzero_si512(), c3a = _mm512_setzero_si512();
    __m512i c0b = _mm512_setzero_si512(), c1b = _mm512_setzero_si512();
    __m512i c2b = _mm512_setzero_si512(), c3b = _mm512_setzero_si512();
    const __m512i one = _mm512_set1_epi64(1);
    const __m512i three = _mm512_set1_epi64(3);

    for (int p = 0; p < crumbs_per_chunk; p++) {
        __m512i pk = _mm512_set1_epi64((long long)pos_key[p]);
        __m512i dv = _mm512_set1_epi64(data[p]);

        __m512i ha = avalanche_hash_x8(_mm512_xor_si512(skey0, pk));
        __m512i hb = avalanche_hash_x8(_mm512_xor_si512(skey1, pk));
        __m512i outa = _mm512_and_si512(_mm512_add_epi64(dv, _mm512_and_si512(ha, three)), three);
        __m512i outb = _mm512_and_si512(_mm512_add_epi64(dv, _mm512_and_si512(hb, three)), three);

        __mmask8 a0 = _mm512_cmpeq_epi64_mask(outa, _mm512_setzero_si512()), b0 = _mm512_cmpeq_epi64_mask(outb, _mm512_setzero_si512());
        __mmask8 a1 = _mm512_cmpeq_epi64_mask(outa, one),   b1 = _mm512_cmpeq_epi64_mask(outb, one);
        __mmask8 a2 = _mm512_cmpeq_epi64_mask(outa, _mm512_set1_epi64(2)), b2 = _mm512_cmpeq_epi64_mask(outb, _mm512_set1_epi64(2));
        __mmask8 a3 = _mm512_cmpeq_epi64_mask(outa, three), b3 = _mm512_cmpeq_epi64_mask(outb, three);
        c0a = _mm512_mask_add_epi64(c0a, a0, c0a, one); c0b = _mm512_mask_add_epi64(c0b, b0, c0b, one);
        c1a = _mm512_mask_add_epi64(c1a, a1, c1a, one); c1b = _mm512_mask_add_epi64(c1b, b1, c1b, one);
        c2a = _mm512_mask_add_epi64(c2a, a2, c2a, one); c2b = _mm512_mask_add_epi64(c2b, b2, c2b, one);
        c3a = _mm512_mask_add_epi64(c3a, a3, c3a, one); c3b = _mm512_mask_add_epi64(c3b, b3, c3b, one);
    }

    uint64_t t0[16], t1[16], t2[16], t3[16];
    _mm512_storeu_si512((void*)(t0 + 0), c0a); _mm512_storeu_si512((void*)(t0 + 8), c0b);
    _mm512_storeu_si512((void*)(t1 + 0), c1a); _mm512_storeu_si512((void*)(t1 + 8), c1b);
    _mm512_storeu_si512((void*)(t2 + 0), c2a); _mm512_storeu_si512((void*)(t2 + 8), c2b);
    _mm512_storeu_si512((void*)(t3 + 0), c3a); _mm512_storeu_si512((void*)(t3 + 8), c3b);
    for (int lane = 0; lane < 16; lane++) {
        f0[lane] = (long)t0[lane]; f1[lane] = (long)t1[lane];
        f2[lane] = (long)t2[lane]; f3[lane] = (long)t3[lane];
    }
}

/* All per-query mutable state used to live in file-scope globals. That was
   fine single-threaded, but every chunk's seed search is fully independent
   of every other chunk's -- nothing about it needs to be shared -- so
   bundling this state into a per-thread QueryCtx (one instance per OpenMP
   thread, not one per chunk) is what makes the per-chunk loop in main()
   safe to run with #pragma omp parallel for. */
typedef struct {
    unsigned char *scratch_ks;
} QueryCtx;

static void ctx_init(QueryCtx *ctx) {
    ctx->scratch_ks = malloc((size_t)crumbs_per_chunk);
    if (!ctx->scratch_ks) {
        fprintf(stderr, "out of memory for per-thread query context\n"); exit(1);
    }
}
static void ctx_free(QueryCtx *ctx) {
    free(ctx->scratch_ks);
}

/* Runs one chunk's query by trying every seed in [0, STAGE1_COUNT) and
   keeping the single best-D one in bestD[c]/best_s[c]/best_f[c*4..] --
   shape-UNRESTRICTED: D is computed for every seed regardless of its
   histogram's shape, so taking the best regardless of shape is free.
   16 seeds at a time go through the AVX-512 histogram kernel above; the
   <16 remainder falls back to scalar. Returns the number of seeds examined
   (always STAGE1_COUNT) and how many scored net > 0, for the diagnostic
   columns in the per-chunk table. */
static double g_t_score = 0.0;   /* profiling accumulator, see the breakdown printout */
static void exhaustive_query_chunk(QueryCtx *ctx, int c, const unsigned char *data, double base_D_full,
                                    long long *best_s, long *best_f,
                                    long *out_cand_n, long *out_cand_pos) {
    double t0 = omp_get_wtime();
    long cpos = 0;
    uint64_t seeds16[16];
    uint64_t s = 0;
    for (; s + 16 <= STAGE1_COUNT; s += 16) {
        for (int lane = 0; lane < 16; lane++) seeds16[lane] = s + (uint64_t)lane;
        long f0[16], f1[16], f2[16], f3[16];
        score_x16_hist(seeds16, data, f0, f1, f2, f3);
        for (int lane = 0; lane < 16; lane++) {
            double D = base_D_full + LT[f0[lane]] + LT[f1[lane]] + LT[f2[lane]] + LT[f3[lane]];
            if (D - STAGE1_BITS > 0.0) cpos++;
            if (D > bestD[c]) {
                bestD[c] = D;
                best_s[c] = (long long)seeds16[lane];
                best_f[c*4+0] = f0[lane]; best_f[c*4+1] = f1[lane];
                best_f[c*4+2] = f2[lane]; best_f[c*4+3] = f3[lane];
            }
        }
    }
    for (; s < STAGE1_COUNT; s++) {
        seed_keystream(s, ctx->scratch_ks);
        long f0 = 0, f1 = 0, f2 = 0, f3 = 0;
        for (int p = 0; p < crumbs_per_chunk; p++) {
            unsigned char out = (unsigned char)((data[p] + ctx->scratch_ks[p]) & 3);
            if      (out == 0) f0++;
            else if (out == 1) f1++;
            else if (out == 2) f2++;
            else               f3++;
        }
        double D = base_D_full + LT[f0] + LT[f1] + LT[f2] + LT[f3];
        if (D - STAGE1_BITS > 0.0) cpos++;
        if (D > bestD[c]) {
            bestD[c] = D;
            best_s[c] = (long long)s;
            best_f[c*4+0] = f0; best_f[c*4+1] = f1; best_f[c*4+2] = f2; best_f[c*4+3] = f3;
        }
    }
    double t1 = omp_get_wtime();
    #pragma omp atomic
    g_t_score += (t1 - t0);

    *out_cand_n = (long)STAGE1_COUNT;
    *out_cand_pos = cpos;
}

int main(int argc, char **argv) {
    /* Single-threaded by default -- the AVX-512 scoring speedup (score_x16_hist)
       applies regardless of thread count, but multithreading is opt-in via
       CTDT_THREADS=n (n>1) rather than the default, since single-thread wall
       time is what "how fast is the search itself" actually means here. */
    { int nthreads = 1;
      const char *e = getenv("CTDT_THREADS");
      if (e) nthreads = atoi(e);
      omp_set_num_threads(nthreads); }

    int n_chunks = argc > 1 ? atoi(argv[1]) : NUM_CHUNKS;
    if (argc > 2) STAGE1_BITS = atoi(argv[2]);
    if (argc > 3) CHUNK_BYTES = atoi(argv[3]);
    if (STAGE1_BITS < 1 || STAGE1_BITS > 40) {
        fprintf(stderr,
            "usage: %s [chunks] [seed_bits] [chunk_bytes]\n"
            "  seed_bits 1..40, chunk_bytes >=1 (default 16).\n"
            "  CTDT_SHOW=n sets how many per-chunk rows to print.\n"
            "  CTDT_THREADS=n sets thread count (default 1).\n",
            argv[0]);
        return 1;
    }
    if (CHUNK_BYTES < 1) {
        fprintf(stderr, "chunk_bytes must be >= 1\n");
        return 1;
    }
    crumbs_per_chunk = CHUNK_BYTES * 4;
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
    if (verify_roundtrip()) { fprintf(stderr, "round-trip self-test FAILED\n"); return 1; }

    printf("seed %d bits (%llu seeds), chunk %d B = %d crumbs -- exhaustive search\n",
           STAGE1_BITS, (unsigned long long)STAGE1_COUNT, CHUNK_BYTES, crumbs_per_chunk);
    printf("  threads: %d\n\n", omp_get_max_threads());
    fflush(stdout);

    unsigned char *bytebuf   = malloc((size_t)CHUNK_BYTES);

    // chunk DUMP_CHUNK with its winning seed actually applied, kept for the dump at the end
    unsigned char *chunk0_out = malloc((size_t)crumbs_per_chunk);
    long long     chunk0_seed = -1;
    int           chunk0_len  = 0;

    // running net stats across every chunk actually processed
    double net_sum = 0.0, net_min = 0.0, net_max = 0.0;
    long   net_n = 0;
    double netraw_sum = 0.0;
    int    net_min_chunk = -1, net_max_chunk = -1;

    long   *all_total  = malloc((size_t)n_chunks * sizeof(long));
    double *all_rawent = malloc((size_t)n_chunks * sizeof(double));
    long   *all_rawfrq = malloc((size_t)n_chunks * 4 * sizeof(long));
    unsigned char *all_crumbs = malloc((size_t)n_chunks * (size_t)crumbs_per_chunk);
    long long *best_s  = malloc((size_t)n_chunks * sizeof(long long));
    /* calloc, not malloc: a chunk that somehow examines zero seeds leaves
       best_f untouched by the query, and hist_type(0,0,0,0) (via the
       s[0]==0 check) safely falls out as "flat" instead of reading garbage. */
    long   *best_f     = calloc((size_t)n_chunks * 4, sizeof(long));
    unsigned char *chunk0_raw = malloc((size_t)crumbs_per_chunk);
    long *cand_count     = malloc((size_t)n_chunks * sizeof(long));
    long *cand_pos_count = malloc((size_t)n_chunks * sizeof(long));
    if (!bytebuf || !chunk0_out || !all_total || !all_rawent
        || !all_rawfrq || !all_crumbs || !best_s || !best_f || !chunk0_raw
        || !cand_count || !cand_pos_count) {
        fprintf(stderr, "out of memory for per-chunk state\n"); return 1;
    }

    int nc = 0;
    for (int chunk = 0; chunk < n_chunks; chunk++) {
        size_t got = fread(bytebuf, 1, (size_t)CHUNK_BYTES, f);
        if (got == 0) break;
        long freq[4] = {0, 0, 0, 0};
        long total = 0;
        unsigned char *dst = all_crumbs + (size_t)nc * crumbs_per_chunk;
        for (size_t b = 0; b < got; b++)
            for (int shift = 6; shift >= 0; shift -= 2) {
                unsigned char crumb = (unsigned char)((bytebuf[b] >> shift) & 0x3);
                dst[total] = crumb;
                freq[crumb]++;
                total++;
            }
        if (nc == DUMP_CHUNK) memcpy(chunk0_raw, dst, (size_t)total);
        all_total[nc]  = total;
        all_rawent[nc] = crumb_entropy(freq, total);
        for (int v = 0; v < 4; v++) all_rawfrq[nc*4+v] = freq[v];
        nc++;
        if (got < (size_t)CHUNK_BYTES) break;
    }
    if (nc == 0) { fprintf(stderr, "no chunks read\n"); return 1; }

    bestD = malloc((size_t)n_chunks * sizeof(double));
    if (!bestD) { fprintf(stderr, "out of memory\n"); return 1; }
    for (int c = 0; c < n_chunks; c++) { bestD[c] = -HUGE_VAL; best_s[c] = -1; }

    double base_D_full = 2.0 * (double)crumbs_per_chunk
                       - (double)crumbs_per_chunk * log2((double)crumbs_per_chunk);
    const double cost = STAGE1_BITS;

    /* Calibrate on a small prefix BEFORE committing to the full n_chunks run --
       the whole point is to know if n_chunks is too many chunks up front, not
       after already paying for the whole thing. Per-chunk cost is fixed (it's
       always exactly STAGE1_COUNT seeds), so a small sample calibrates well.
       Run multithreaded here too (not single-thread) so ms/chunk reflects the
       SAME throughput the real run below will actually get. */
    int calib_n = nc < 50 ? nc : 50;
    double t_query0 = omp_get_wtime();
    #pragma omp parallel
    {
        QueryCtx ctx; ctx_init(&ctx);
        #pragma omp for schedule(dynamic, 4)
        for (int c = 0; c < calib_n; c++)
            exhaustive_query_chunk(&ctx, c, all_crumbs + (size_t)c * crumbs_per_chunk, base_D_full,
                                    best_s, best_f, &cand_count[c], &cand_pos_count[c]);
        ctx_free(&ctx);
    }
    double t_calib = omp_get_wtime();
    double calib_secs    = t_calib - t_query0;
    double ms_per_chunk  = 1000.0 * calib_secs / calib_n;
    printf("  calibration: %.3f ms/chunk wall (from first %d chunk%s, %d threads) -> estimated %s for all %d chunks\n",
           ms_per_chunk, calib_n, calib_n == 1 ? "" : "s", omp_get_max_threads(),
           fmt_time(ms_per_chunk / 1000.0 * nc), nc);
    printf("  |  whole 198 MB file (%.0f chunks) at this rate: %s\n\n",
           198766075.0 / CHUNK_BYTES, fmt_time(ms_per_chunk / 1000.0 * (198766075.0 / CHUNK_BYTES)));
    fflush(stdout);

    #pragma omp parallel
    {
        QueryCtx ctx; ctx_init(&ctx);
        #pragma omp for schedule(dynamic, 4)
        for (int c = calib_n; c < nc; c++)
            exhaustive_query_chunk(&ctx, c, all_crumbs + (size_t)c * crumbs_per_chunk, base_D_full,
                                    best_s, best_f, &cand_count[c], &cand_pos_count[c]);
        ctx_free(&ctx);
    }
    double t_query1 = omp_get_wtime();

    double query_secs = t_query1 - t_query0;
    printf("  actual: %s for %d chunks (%.3f ms/chunk wall, %llu seeds/chunk)\n",
           fmt_time(query_secs), nc, 1000.0 * query_secs / nc, (unsigned long long)STAGE1_COUNT);
    printf("  |  score time: %.2f s [summed across threads]\n\n", g_t_score);

    FILE *hf = NULL;
    { const char *e = getenv("CTDT_HIST");
      if (e) {
          hf = fopen(e, "wb");
          if (!hf) { fprintf(stderr, "cannot write %s\n", e); return 1; }
      } }

    long   tcount[5] = {0,0,0,0,0};   /* chunks per histogram shape */
    double tnet[5]   = {0,0,0,0,0};

    int show = 24;
    { const char *e = getenv("CTDT_SHOW"); if (e) show = atoi(e); }
    if (show > nc) show = nc;
    if (show > 0) {
        printf("=== per chunk (first %d of %d) ===\n", show, nc);
        printf("  %5s %8s %8s %10s  %-18s %-18s %-10s %8s  %10s %10s\n",
               "chunk", "H_raw", "H_best", "seed",
               "histogram raw", "histogram best", "type", "net",
               "seeds", "seeds net>0");
    }

    long no_shape = 0;   /* chunks with zero seeds (only possible if STAGE1_COUNT is 0) */

    for (int chunk = 0; chunk < nc; chunk++) {
        long   total     = all_total[chunk];
        double entropy   = all_rawent[chunk];
        int    have_best = best_s[chunk] >= 0;
        double best_entropy = have_best
            ? MAX_CRUMB_ENTROPY - bestD[chunk] / (double)total : MAX_CRUMB_ENTROPY;

        if (hf) {
            if (!have_best)
                fprintf(hf, "-1/-1/-1/-1\n");
            else
                fprintf(hf, "%ld/%ld/%ld/%ld\n",
                        best_f[chunk*4+0], best_f[chunk*4+1],
                        best_f[chunk*4+2], best_f[chunk*4+3]);
        }

        if (chunk < show) {
            char hr[32], hb[32];
            snprintf(hr, sizeof hr, "%ld/%ld/%ld/%ld",
                     all_rawfrq[chunk*4+0], all_rawfrq[chunk*4+1],
                     all_rawfrq[chunk*4+2], all_rawfrq[chunk*4+3]);
            if (!have_best) {
                printf("  %5d %8.4f %8s %10s  %-18s %-18s %-10s %8s  %10ld %10ld\n",
                       chunk, entropy, "--", "--", hr, "(no candidate)",
                       "--", "--", cand_count[chunk], cand_pos_count[chunk]);
            } else {
                snprintf(hb, sizeof hb, "%ld/%ld/%ld/%ld",
                         best_f[chunk*4+0], best_f[chunk*4+1],
                         best_f[chunk*4+2], best_f[chunk*4+3]);
                printf("  %5d %8.4f %8.4f %10lld  %-18s %-18s %-10s %+8.2f  %10ld %10ld\n",
                       chunk, entropy, best_entropy, best_s[chunk], hr, hb,
                       TYPE_NAME[hist_type(&best_f[chunk*4])],
                       bestD[chunk] - cost, cand_count[chunk], cand_pos_count[chunk]);
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

        if (chunk == DUMP_CHUNK && best_s[DUMP_CHUNK] >= 0) {
            uint64_t k = prng_seed_key((uint64_t)best_s[DUMP_CHUNK]);
            for (long i = 0; i < total; i++)
                chunk0_out[i] = (unsigned char)((chunk0_raw[i] + prng_crumb_keyed(k, (uint32_t)i)) & 0x3);
            chunk0_seed = best_s[DUMP_CHUNK];
            chunk0_len  = (int)total;
        }
    }

    if (no_shape)
        printf("  %ld/%d chunks: no seeds examined (STAGE1_COUNT was 0 --\n"
               "  excluded from the net stats below)\n",
               no_shape, nc);

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
       so a downstream tool can resweep chunk-size x S1 fresh on the real
       post-layer-1 residual instead of the raw file. */
    { const char *e = getenv("CTDT_RESIDUAL");
      if (e) {
          FILE *rf = fopen(e, "wb");
          if (!rf) { fprintf(stderr, "cannot write %s\n", e); return 1; }
          for (int c = 0; c < nc; c++) {
              /* no candidate found for this chunk -- key 0 leaves the
                 keystream well-defined (still a valid, just unsearched, seed)
                 rather than reading the -1 sentinel as a huge bogus seed. */
              uint64_t k = prng_seed_key(best_s[c] >= 0 ? (uint64_t)best_s[c] : 0);
              int total = (int)all_total[c];
              const unsigned char *src = all_crumbs + (size_t)c * crumbs_per_chunk;
              for (int i = 0; i < total; i += 4) {
                  int b = 0;
                  for (int j = 0; j < 4; j++) {
                      int pos = i + j;
                      unsigned char out = (unsigned char)((src[pos] + prng_crumb_keyed(k, (uint32_t)pos)) & 3u);
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
