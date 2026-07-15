#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>

typedef unsigned char u8;
typedef unsigned int u32;

static const char *PATH = "C:\\Users\\lukac\\Documents\\compressor\\compresseddata.bin";
static const char *LAYER2_PATH = "C:\\Users\\lukac\\Documents\\compressor\\layer2.bin";

#define CHUNK_BYTES 8
#define NUM_CHUNKS  32*16
#define SEED_BITS   16

static double entropy(const u8 *d, size_t n) {
    u32 f[256]={0}; for(size_t i=0;i<n;i++) f[d[i]]++;
    double H=0.0;
    for(int v=0;v<256;v++) if(f[v]){double p=(double)f[v]/n; H-=p*log2(p);}
    return H;
}

/* Sum of |consecutive byte differences| over a chunk -- the "smoothness"
 * score used to pick the best seed. Lower means neighboring transformed
 * bytes vary less, which favors delta/RLE-style downstream encoding.
 * Plain O(n) integer pass, cheap enough to call once per seed trial
 * (2^24 trials/chunk) without needing a lookup-table trick. */
static long delta_sum(const u8 *d, int n) {
    long s = 0;
    for (int i = 1; i < n; i++) {
        int diff = (int)d[i] - (int)d[i - 1];
        s += diff < 0 ? -diff : diff;
    }
    return s;
}

/* Avalanche seed hash: Murmur3-style finalizer, stateless per-position
 * keystream. Full avalanche means a 1-bit seed change flips ~half the
 * output bits, so nearby seeds don't produce correlated byte streams.
 * seed_term = seed*0x9E3779B1u is precomputed by the caller once per seed
 * instead of once per position -- it doesn't depend on pos, and this hash
 * runs CHUNK_BYTES times per seed trial. */
static inline u8 avalanche_hash_term(u32 pos, u32 seed_term) {
    u32 x = pos ^ seed_term;
    x ^= x >> 16; x *= 0x85EBCA6Bu;
    x ^= x >> 13; x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return (u8)x;
}

/* Rebuild the transformed chunk for a known-winning seed. Cheap (runs once
 * per chunk, not per seed trial), so no need for the delta_sum hot-loop. */
static void apply_seed(const u8 *chunk, u8 *out, u32 seed) {
    u32 seed_term = seed * 0x9E3779B1u;
    for (int j = 0; j < CHUNK_BYTES; j++)
        out[j] = (u8)(chunk[j] + avalanche_hash_term((u32)j, seed_term));
}

/* Exact inverse of apply_seed: avalanche_hash_term(j, seed) is a pure
 * function of position and seed (no data-dependence), so subtracting the
 * same keystream back off (mod 256) recovers the original chunk bytes
 * exactly, given the seed that was used to transform them. */
static void unapply_seed(const u8 *out, u8 *chunk, u32 seed) {
    u32 seed_term = seed * 0x9E3779B1u;
    for (int j = 0; j < CHUNK_BYTES; j++)
        chunk[j] = (u8)(out[j] - avalanche_hash_term((u32)j, seed_term));
}

/* qsort comparator for joint_entropy_L: sorts window *offsets* by memcmp of
 * the L bytes starting there. g_sort_d/g_sort_L are set by the caller just
 * before qsort -- fine since this program is single-threaded (same pattern
 * as the per-chunk seed-search state elsewhere in this file). */
static const u8 *g_sort_d;
static int g_sort_L;
static int cmp_window(const void *a, const void *b) {
    return memcmp(g_sort_d + *(const int*)a, g_sort_d + *(const int*)b, (size_t)g_sort_L);
}

/* Shannon entropy of the distribution of L-byte windows, i.e. joint entropy
 * H(X_1..X_L) in bits over the whole L-gram, not per-byte. Windows are
 * compared by raw memcmp (via offset sort) instead of packing into a fixed
 * integer, so L isn't capped at 4 -- needed up to L=11 for order-10. Runs
 * once per chunk/order (not in the seed brute-force loop), so a
 * straightforward sort-and-count is fine. */
static double joint_entropy_L(const u8 *d, int n, int L) {
    int nw = n - L + 1;
    if (nw <= 0) return 0.0;
    int *idx = malloc((size_t)nw * sizeof(int));
    for (int i = 0; i < nw; i++) idx[i] = i;
    g_sort_d = d; g_sort_L = L;
    qsort(idx, nw, sizeof(int), cmp_window);
    double H = 0.0;
    int i = 0;
    while (i < nw) {
        int j = i;
        while (j < nw && memcmp(d + idx[i], d + idx[j], (size_t)L) == 0) j++;
        double p = (double)(j - i) / nw;
        H -= p * log2(p);
        i = j;
    }
    free(idx);
    return H;
}

/* Order-k conditional entropy H(X_i | previous k bytes), in bits/byte.
 * order-0 is the plain byte-value entropy; order-k (k>=1) is the standard
 * H_joint(k+1) - H_joint(k) decomposition (same identity reduce2.c's
 * order1test uses for order-1, generalized here up through order-10). */
static double order_k_entropy(const u8 *d, int n, int k) {
    if (k == 0) return joint_entropy_L(d, n, 1);
    return joint_entropy_L(d, n, k + 1) - joint_entropy_L(d, n, k);
}

/* Pearson correlation between the byte stream and itself shifted by `lag`
 * positions -- order-0 (lag=0) is a sequence correlated with itself, always
 * 1.0; order-1/2/3 measure linear dependency at 1/2/3-byte separation. */
static double autocorr_lag(const u8 *d, int n, int lag) {
    if (lag == 0) return 1.0;
    int m = n - lag;
    if (m <= 1) return 0.0;
    double sx = 0, sy = 0, sxx = 0, syy = 0, sxy = 0;
    for (int i = 0; i < m; i++) {
        double x = d[i], y = d[i + lag];
        sx += x; sy += y; sxx += x * x; syy += y * y; sxy += x * y;
    }
    double num = m * sxy - sx * sy;
    double den = sqrt((m * sxx - sx * sx) * (m * syy - sy * sy));
    return (den == 0.0) ? 0.0 : num / den;
}

/* Globally optimal seed assignment across all chunks (Viterbi/DP), instead
 * of the greedy chunk-by-chunk pick this replaces: a greedy search reacts
 * to whatever the previous chunk already committed to, so it can miss a
 * cheaper path that requires a slightly worse earlier chunk. Here the DP
 * state is the chunk's last output byte (0..255) -- for every chunk, for
 * every possible previous-state, for every candidate seed, extend the
 * cheapest path so far by (this chunk's intra delta_sum) + (boundary jump
 * from prev state into this chunk's first byte). The result is the exact
 * seed sequence minimizing total delta_sum over the ENTIRE concatenated
 * stream, matching what delta-encoding the whole output would show.
 * Cost is O(nchunks * nseeds * 256); fine for nseeds up to a few thousand
 * (current SEED_BITS=8 -> 256 seeds). A much larger seed space would need
 * a 1D distance-transform trick to avoid the inner 256-wide scan. */
static long solve_seeds_dp(const u8 *buf, size_t nchunks, u32 *out_seeds) {
    u32 nseeds = 1u << SEED_BITS;
    long dp_prev[256], dp_cur[256];
    u32 *choice_seed = malloc(nchunks * 256 * sizeof(u32));
    int *choice_prev = malloc(nchunks * 256 * sizeof(int));
    if (!choice_seed || !choice_prev) { fprintf(stderr, "out of memory in solve_seeds_dp\n"); exit(1); }

    u8 tmp[CHUNK_BYTES];
    for (size_t c = 0; c < nchunks; c++) {
        const u8 *chunk = buf + c * CHUNK_BYTES;
        for (int i = 0; i < 256; i++) dp_cur[i] = LONG_MAX;

        for (u32 seed = 0; seed < nseeds; seed++) {
            u32 seed_term = seed * 0x9E3779B1u;
            for (int j = 0; j < CHUNK_BYTES; j++)
                tmp[j] = (u8)(chunk[j] + avalanche_hash_term((u32)j, seed_term));
            long intra = delta_sum(tmp, CHUNK_BYTES);
            u8 first = tmp[0], last = tmp[CHUNK_BYTES - 1];

            if (c == 0) {
                if (intra < dp_cur[last]) {
                    dp_cur[last] = intra;
                    choice_seed[last] = seed;
                    choice_prev[last] = -1;
                }
            } else {
                for (int prev = 0; prev < 256; prev++) {
                    if (dp_prev[prev] == LONG_MAX) continue;
                    long cost = dp_prev[prev] + intra + abs((int)first - prev);
                    if (cost < dp_cur[last]) {
                        dp_cur[last] = cost;
                        choice_seed[c * 256 + last] = seed;
                        choice_prev[c * 256 + last] = prev;
                    }
                }
            }
        }
        memcpy(dp_prev, dp_cur, sizeof(dp_prev));
    }

    long best_total = LONG_MAX;
    int best_state = 0;
    for (int i = 0; i < 256; i++)
        if (dp_prev[i] < best_total) { best_total = dp_prev[i]; best_state = i; }

    int state = best_state;
    for (size_t c = nchunks; c-- > 0; ) {
        u32 seed = choice_seed[c * 256 + state];
        out_seeds[c] = seed;
        state = choice_prev[c * 256 + state];
    }

    free(choice_seed);
    free(choice_prev);
    return best_total;
}

int main(void) {
    FILE *fp = fopen(PATH, "rb");
    if (!fp) { fprintf(stderr, "failed to open %s\n", PATH); return 1; }

    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (size < 0) { fprintf(stderr, "failed to stat %s\n", PATH); fclose(fp); return 1; }

    u8 *buf = malloc((size_t)size);
    if (!buf) { fprintf(stderr, "out of memory\n"); fclose(fp); return 1; }

    size_t nread = fread(buf, 1, (size_t)size, fp);
    fclose(fp);
    if (nread != (size_t)size) { fprintf(stderr, "short read on %s\n", PATH); free(buf); return 1; }

    double H = entropy(buf, nread);
    printf("file:    %s\n", PATH);
    printf("size:    %zu bytes\n", nread);
    printf("entropy: %.6f bits/byte\n\n", H);

    size_t nchunks = nread / CHUNK_BYTES;
    if (nchunks > NUM_CHUNKS) nchunks = NUM_CHUNKS;
    if (nchunks == 0) { fprintf(stderr, "input smaller than one %d-byte chunk\n", CHUNK_BYTES); free(buf); return 1; }

    printf("DP seed search (%d-bit seed space per chunk), minimizing whole-stream delta_sum:\n", SEED_BITS);
    u8 *global_transformed = malloc(nchunks * CHUNK_BYTES);
    u32 *seeds = malloc(nchunks * sizeof(u32));
    if (!global_transformed || !seeds) { fprintf(stderr, "out of memory\n"); free(buf); return 1; }
    long dp_total = solve_seeds_dp(buf, nchunks, seeds);
    for (size_t c = 0; c < nchunks; c++) {
        const u8 *chunk = buf + c * CHUNK_BYTES;
        apply_seed(chunk, global_transformed + c * CHUNK_BYTES, seeds[c]);
        printf("chunk %2zu: seed=0x%06X  intra_D_start=%ld  intra_D_end=%ld\n",
               c, seeds[c], delta_sum(chunk, CHUNK_BYTES),
               delta_sum(global_transformed + c * CHUNK_BYTES, CHUNK_BYTES));
    }

    size_t gn = nchunks * CHUNK_BYTES;
    long whole_before = delta_sum(buf, (int)gn);
    long whole_after  = delta_sum(global_transformed, (int)gn);
    printf("\nwhole-stream delta_sum over all %zu bytes as one continuous run: before=%ld  after=%ld\n",
           gn, whole_before, whole_after);
    printf("DP-reported optimum: %ld (should equal 'after' above)\n", dp_total);

    /* layer2: signed mod-256 delta of layer1 (global_transformed), plus the
     * anchor byte (layer1[0]) needed to decode it back. This is the only
     * form of the delta stream that's actually invertible -- see
     * unapply_seed/the round-trip check below -- unlike the |delta| used
     * for display/scoring elsewhere in this file. layer1 itself is no
     * longer written out; layer2 (+ seeds.bin) carries the same information. */
    u8 *layer2 = malloc(gn - 1);
    if (!layer2) { fprintf(stderr, "out of memory\n"); free(buf); return 1; }
    for (size_t i = 1; i < gn; i++)
        layer2[i - 1] = (u8)(global_transformed[i] - global_transformed[i - 1]);
    u8 anchor = global_transformed[0];
    printf("\nglobal stats over all %zu bytes (before -> after):\n", gn);
    {
        double ent_before = order_k_entropy(buf, (int)gn, 0);
        double ent_after  = order_k_entropy(global_transformed, (int)gn, 0);
        double corr_before = autocorr_lag(buf, (int)gn, 0);
        double corr_after  = autocorr_lag(global_transformed, (int)gn, 0);
        double net_0 = (ent_before - ent_after) * gn;
        printf("    order-0 entropy: %8.4f -> %8.4f bps   order-0 corr: %+.4f -> %+.4f   net=%.2f bits\n",
               ent_before, ent_after, corr_before, corr_after, net_0);
    }

    if (gn >= 2) {
        u8 *delta_before = malloc(gn - 1);
        u8 *delta_after  = malloc(gn - 1);
        if (!delta_before || !delta_after) { fprintf(stderr, "out of memory\n"); free(buf); return 1; }
        for (size_t i = 1; i < gn; i++) {
            int db = (int)buf[i] - (int)buf[i - 1];
            int da = (int)global_transformed[i] - (int)global_transformed[i - 1];
            delta_before[i - 1] = (u8)(db < 0 ? -db : db);
            delta_after[i - 1]  = (u8)(da < 0 ? -da : da);
        }
        double dent_before = entropy(delta_before, gn - 1);
        double dent_after  = entropy(delta_after, gn - 1);
        double dnet = (dent_before - dent_after) * (double)(gn - 1);
        printf("    delta order-0 entropy: %8.4f -> %8.4f bps   over %zu consecutive |delta| values   net=%.2f bits\n",
               dent_before, dent_after, gn - 1, dnet);
        free(delta_before);
        free(delta_after);
    }

    /* Dump raw chunks, layer2 (anchor + signed delta stream), and the seed
     * list so an external harness can throw real compressors at both
     * bytestreams and compare against raw_compressed + seed_overhead
     * honestly. layer1 (transformed_chunks.bin) isn't written anymore --
     * layer2 + seeds.bin already carries everything needed to reconstruct
     * it (see the round-trip check below), so it was pure redundancy. */
    FILE *fraw = fopen("raw_chunks.bin", "wb");
    if (fraw) { fwrite(buf, 1, gn, fraw); fclose(fraw); }
    FILE *fl2 = fopen(LAYER2_PATH, "wb");
    if (fl2) {
        fwrite(&anchor, 1, 1, fl2);
        fwrite(layer2, 1, gn - 1, fl2);
        fclose(fl2);
    }
    int seed_bytes = (SEED_BITS + 7) / 8;
    FILE *fseeds = fopen("seeds.bin", "wb");
    if (fseeds) {
        for (size_t c = 0; c < nchunks; c++)
            fwrite(&seeds[c], 1, (size_t)seed_bytes, fseeds); /* little-endian low bytes */
        fclose(fseeds);
    }
    printf("\nwrote raw_chunks.bin, %s (anchor+deltas), seeds.bin (%d bytes/seed x %zu chunks)\n",
           LAYER2_PATH, seed_bytes, nchunks);

    size_t nprint = gn < 64 ? gn : 64;
    printf("\nfirst %zu bytes of transformed_chunks.bin (decimal):\n", nprint);
    for (size_t i = 0; i < nprint; i++)
        printf("%d%s", global_transformed[i], (i + 1 < nprint) ? " " : "\n");

    printf("\nconsecutive |delta| of those same %zu bytes (%zu values):\n", nprint, nprint - 1);
    for (size_t i = 1; i < nprint; i++) {
        int diff = (int)global_transformed[i] - (int)global_transformed[i - 1];
        printf("%d%s", diff < 0 ? -diff : diff, (i + 1 < nprint) ? " " : "\n");
    }

    long delta_before_64 = delta_sum(buf, (int)nprint);
    long delta_after_64  = delta_sum(global_transformed, (int)nprint);
    printf("\nnet delta_sum over these same %zu bytes: raw=%ld  transformed_chunks.bin=%ld  net=%ld\n",
           nprint, delta_before_64, delta_after_64, delta_before_64 - delta_after_64);

    /* Round-trip check: layer2 (signed mod-256 deltas + anchor byte) decodes
     * back to layer1, then unapply_seed per chunk decodes layer1 back to
     * layer0 -- verified byte-for-byte against the real original data. This
     * uses signed mod-256 deltas, not the |delta| used for display/scoring
     * above, since only the signed version is actually invertible. */
    printf("\nround-trip check (layer2 -> layer1 -> layer0):\n");
    u8 *layer1_decoded = malloc(gn);
    u8 *layer0_decoded = malloc(gn);
    if (!layer1_decoded || !layer0_decoded) { fprintf(stderr, "out of memory\n"); free(buf); return 1; }

    layer1_decoded[0] = anchor;
    for (size_t i = 1; i < gn; i++)
        layer1_decoded[i] = (u8)(layer1_decoded[i - 1] + layer2[i - 1]);

    size_t layer1_mismatches = 0;
    for (size_t i = 0; i < gn; i++)
        if (layer1_decoded[i] != global_transformed[i]) layer1_mismatches++;
    printf("  layer2 -> layer1: %s (%zu / %zu bytes mismatched)\n",
           layer1_mismatches == 0 ? "PASS" : "FAIL", layer1_mismatches, gn);

    for (size_t c = 0; c < nchunks; c++)
        unapply_seed(layer1_decoded + c * CHUNK_BYTES, layer0_decoded + c * CHUNK_BYTES, seeds[c]);

    size_t layer0_mismatches = 0;
    for (size_t i = 0; i < gn; i++)
        if (layer0_decoded[i] != buf[i]) layer0_mismatches++;
    printf("  layer1 -> layer0: %s (%zu / %zu bytes mismatched)\n",
           layer0_mismatches == 0 ? "PASS" : "FAIL", layer0_mismatches, gn);
    printf("  overall: %s\n",
           (layer1_mismatches == 0 && layer0_mismatches == 0) ? "PASS -- fully reversible" : "FAIL");

    free(layer2);
    free(layer1_decoded);
    free(layer0_decoded);

    free(seeds);
    free(global_transformed);
    free(buf);
    return 0;
}
