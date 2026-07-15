#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef unsigned char u8;
typedef unsigned int u32;

static const char *PATH = "C:\\Users\\lukac\\Documents\\compressor\\compresseddata.bin";

#define CHUNK_BYTES 8
#define NUM_CHUNKS  32
#define SEED_BITS   8

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

/* Brute-force every 24-bit seed for one CHUNK_BYTES-byte chunk, adding the
 * hash keystream to each byte (mod 256), and keep the seed whose result
 * has the lowest delta_sum (smoothest consecutive-byte variation). */
static u32 best_seed_for_chunk(const u8 *chunk, long *out_start_D, long *out_end_D) {
    long D0 = delta_sum(chunk, CHUNK_BYTES);
    long bestD = D0;
    u32 bestSeed = 0;
    u8 tmp[CHUNK_BYTES];

    u32 nseeds = 1u << SEED_BITS;
    for (u32 seed = 0; seed < nseeds; seed++) {
        u32 seed_term = seed * 0x9E3779B1u;
        for (int j = 0; j < CHUNK_BYTES; j++)
            tmp[j] = (u8)(chunk[j] + avalanche_hash_term((u32)j, seed_term));
        long D = delta_sum(tmp, CHUNK_BYTES);
        if (D < bestD) { bestD = D; bestSeed = seed; }
    }

    *out_start_D = D0;
    *out_end_D = bestD;
    return bestSeed;
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

    printf("per-chunk %d-byte avalanche-seed search (%d-bit seed space), minimizing delta_sum:\n", CHUNK_BYTES, SEED_BITS);
    long total_delta_saved = 0;
    long min_delta_saved = 0, max_delta_saved = 0;
    u8 *global_transformed = malloc(nchunks * CHUNK_BYTES);
    u32 *seeds = malloc(nchunks * sizeof(u32));
    if (!global_transformed || !seeds) { fprintf(stderr, "out of memory\n"); free(buf); return 1; }
    for (size_t c = 0; c < nchunks; c++) {
        const u8 *chunk = buf + c * CHUNK_BYTES;
        long D0, D1;
        u32 seed = best_seed_for_chunk(chunk, &D0, &D1);
        long delta_saved = D0 - D1;
        total_delta_saved += delta_saved;
        if (c == 0 || delta_saved < min_delta_saved) min_delta_saved = delta_saved;
        if (c == 0 || delta_saved > max_delta_saved) max_delta_saved = delta_saved;
        apply_seed(chunk, global_transformed + c * CHUNK_BYTES, seed);
        seeds[c] = seed;

        printf("chunk %2zu: seed=0x%06X  D_start=%ld  D_end=%ld  saved=%ld\n",
               c, seed, D0, D1, delta_saved);
    }
    printf("\ntotal delta reduction over %zu chunks: %ld\n",
           nchunks, total_delta_saved);
    printf("avg delta reduction per chunk: %.2f   min: %ld   max: %ld\n",
           (double)total_delta_saved / (double)nchunks, min_delta_saved, max_delta_saved);

    size_t gn = nchunks * CHUNK_BYTES;
    printf("\nglobal stats over all %zu bytes (before -> after):\n", gn);
    for (int k = 0; k <= 5; k++) {
        double ent_before = order_k_entropy(buf, (int)gn, k);
        double ent_after  = order_k_entropy(global_transformed, (int)gn, k);
        double corr_before = autocorr_lag(buf, (int)gn, k);
        double corr_after  = autocorr_lag(global_transformed, (int)gn, k);
        double net_k = (ent_before - ent_after) * gn;
        printf("    order-%d entropy: %8.4f -> %8.4f bps   order-%d corr: %+.4f -> %+.4f   net=%.2f bits\n",
               k, ent_before, ent_after, k, corr_before, corr_after, net_k);
    }

    /* Dump raw chunks, seed-transformed chunks, and the seed list so an
     * external harness can throw real compressors at both bytestreams and
     * compare against raw_compressed + seed_overhead honestly. */
    FILE *fraw = fopen("raw_chunks.bin", "wb");
    if (fraw) { fwrite(buf, 1, gn, fraw); fclose(fraw); }
    FILE *ftr = fopen("transformed_chunks.bin", "wb");
    if (ftr) { fwrite(global_transformed, 1, gn, ftr); fclose(ftr); }
    int seed_bytes = (SEED_BITS + 7) / 8;
    FILE *fseeds = fopen("seeds.bin", "wb");
    if (fseeds) {
        for (size_t c = 0; c < nchunks; c++)
            fwrite(&seeds[c], 1, (size_t)seed_bytes, fseeds); /* little-endian low bytes */
        fclose(fseeds);
    }
    printf("\nwrote raw_chunks.bin, transformed_chunks.bin, seeds.bin (%d bytes/seed x %zu chunks)\n",
           seed_bytes, nchunks);

    size_t nprint = gn < 64 ? gn : 64;
    printf("\nfirst %zu bytes of transformed_chunks.bin (decimal):\n", nprint);
    for (size_t i = 0; i < nprint; i++)
        printf("%d%s", global_transformed[i], (i + 1 < nprint) ? " " : "\n");

    free(seeds);
    free(global_transformed);
    free(buf);
    return 0;
}
