#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <time.h>
#include "cvs.c"

typedef uint8_t u8;

// xlgx[x] = x * log2(x), with xlgx[0] = 0
// entropy = log2(n) - sum(xlgx[counts[i]]) / n
static double xlgx[4097];

static void init_entropy_table(void) {
    xlgx[0] = 0.0;
    for (int x = 1; x <= 4096; x++)
        xlgx[x] = (double)x * log2((double)x);
}

static uint64_t xorshift_state = 0x123456789ABCDEF0ULL;

void xorshift_seed(uint64_t seed) {
    xorshift_state = seed ? seed : 0x123456789ABCDEF0ULL;
}

u8 xorshift(void) {
    xorshift_state ^= xorshift_state << 13;
    xorshift_state ^= xorshift_state >> 7;
    xorshift_state ^= xorshift_state << 17;
    return (u8)xorshift_state;
}

u8 randNum_add(void) { u8 a = xorshift(); u8 b = xorshift(); return a | b; }
u8 randNum_and(void) { u8 a = xorshift(); u8 b = xorshift(); return a & b; }

static uint64_t shuffle_rng(uint64_t *s) {
    *s ^= *s << 13; *s ^= *s >> 7; *s ^= *s << 17;
    return *s;
}

void do_shuffle(u8 *data, int size, uint64_t seed) {
    int *perm = malloc(size * sizeof(int));
    u8  *tmp  = malloc(size);
    for (int i = 0; i < size; i++) perm[i] = i;
    uint64_t s = seed ? seed : 1;
    for (int i = size - 1; i > 0; i--) {
        int j = (int)(shuffle_rng(&s) % (uint64_t)(i + 1));
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (int i = 0; i < size; i++) tmp[perm[i]] = data[i];
    memcpy(data, tmp, size);
    free(perm); free(tmp);
}

void do_unshuffle(u8 *data, int size, uint64_t seed) {
    int *perm = malloc(size * sizeof(int));
    u8  *tmp  = malloc(size);
    for (int i = 0; i < size; i++) perm[i] = i;
    uint64_t s = seed ? seed : 1;
    for (int i = size - 1; i > 0; i--) {
        int j = (int)(shuffle_rng(&s) % (uint64_t)(i + 1));
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (int i = 0; i < size; i++) tmp[i] = data[perm[i]];
    memcpy(data, tmp, size);
    free(perm); free(tmp);
}

double entropy(u8 *data, int size) {
    if (size <= 0) return 0.0;
    int counts[256] = {0};
    for (int i = 0; i < size; i++) counts[data[i]]++;
    double sum = 0.0;
    for (int i = 0; i < 256; i++)
        if (counts[i]) sum += xlgx[counts[i]];
    return log2((double)size) - sum / size;
}

// GF(256) with AES primitive polynomial x^8+x^4+x^3+x+1 (0x11b)
// gf256_mul(a,b): Galois field multiplication (NOT the same as a*b mod 256)
// gf256_inv_table[v]: multiplicative inverse of v in GF(256)
static u8 gf256_log[256];
static u8 gf256_exp[512];   // doubled table: gf256_exp[i+255]=gf256_exp[i]
static u8 gf256_inv_table[256];

static void init_gf256_tables(void) {
    // Generator 0x03 is a primitive root of GF(256, 0x11b); 0x02 only has order 51.
    int x = 1;
    for (int i = 0; i < 255; i++) {
        gf256_exp[i] = (u8)x;
        gf256_log[x] = (u8)i;
        // multiply x by {03} = xtime(x) XOR x
        int xt = ((x << 1) & 0xFF) ^ ((x & 0x80) ? 0x1b : 0);
        x = xt ^ x;
    }
    gf256_log[0] = 0;   // log(0) undefined; set to 0 safely
    for (int i = 255; i < 512; i++) gf256_exp[i] = gf256_exp[i - 255];
    gf256_inv_table[0] = 0;
    for (int i = 1; i < 256; i++)
        gf256_inv_table[i] = gf256_exp[255 - gf256_log[i]];
}

static inline u8 gf256_mul(u8 a, u8 b) {
    if (!a || !b) return 0;
    return gf256_exp[gf256_log[a] + gf256_log[b]];
}

// Rotate the bits of a byte left by k positions (k in 0..7)
static inline u8 rotl8(u8 v, int k) {
    k &= 7;
    if (!k) return v;
    return (u8)(((unsigned)v << k) | ((unsigned)v >> (8 - k)));
}

double FindBest(u8 *data, int size) {
    u8 *buf = malloc(size);
    double best_entropy = 9.0;
    int best_seed = 0;

    for (int seed = 0; seed <= 65536; seed++) {
        xorshift_seed(seed);
        for (int i = 0; i < size; i++)
            buf[i] = data[i] + randNum_add();
        double e = entropy(buf, size);
        if (e < best_entropy) { best_entropy = e; best_seed = seed; }
    }

    free(buf);
    printf("[add]      seed=%5d  entropy=%lf\n", best_seed, best_entropy);
    return best_entropy;
}

double FindBestResidual(u8 *data, int size) {
    u8 *buf = malloc(size);
    double best_entropy = 9.0;
    int best_seed = 0;

    for (int seed = 0; seed <= 65536; seed++) {
        xorshift_seed(seed);
        for (int i = 0; i < size; i++)
            buf[i] = data[i] - randNum_add();
        double e = entropy(buf, size);
        if (e < best_entropy) { best_entropy = e; best_seed = seed; }
    }

    xorshift_seed(best_seed);
    int near_zero = 0;
    for (int i = 0; i < size; i++) {
        u8 r = data[i] - randNum_add();
        if (r <= 5 || r >= 251) near_zero++;
        buf[i] = r;
    }

    free(buf);
    printf("[residual] seed=%5d  entropy=%lf  near-zero=%d/%d\n",
           best_seed, best_entropy, near_zero, size);
    return best_entropy;
}

typedef struct { int seed; int score; } SeedScore;

int cmp_score_desc(const void *a, const void *b) {
    return ((SeedScore*)b)->score - ((SeedScore*)a)->score;
}

static inline int pattern_applies(int i, int pat) {
    switch (pat) {
        case 0: return  (i & 1);
        case 1: return !(i & 1);
        case 2: return (i%4==1 || i%4==2);
        case 3: return (i%4==0 || i%4==3);
        case 4: return (i%4==0 || i%4==1);
        case 5: return (i%4==2 || i%4==3);
        case 6: return (i%4 != 3);
        default: return (i%4 != 0);
    }
}

static int score_seed_pattern(u8 *data, int len, int seed, int pat) {
    int counts[256] = {0};
    int score = 0;
    xorshift_seed(seed);
    for (int i = 0; i < len; i++) {
        u8 v = pattern_applies(i, pat) ? (u8)(data[i] + randNum_add()) : data[i];
        score += counts[v];
        counts[v]++;
    }
    return score;
}

static int score_seed_pattern_sub(u8 *data, int len, int seed, int pat) {
    int counts[256] = {0};
    int score = 0;
    xorshift_seed(seed);
    for (int i = 0; i < len; i++) {
        u8 v = pattern_applies(i, pat) ? (u8)(data[i] - randNum_add()) : data[i];
        score += counts[v];
        counts[v]++;
    }
    return score;
}

static int score_seed_and(u8 *data, int len, int seed) {
    int counts[256] = {0};
    int score = 0;
    xorshift_seed(seed);
    for (int i = 0; i < len; i++) {
        u8 v = data[i] + randNum_and();
        score += counts[v];
        counts[v]++;
    }
    return score;
}

static int score_seed(u8 *data, int len, int seed) {
    int counts[256] = {0};
    int score = 0;
    xorshift_seed(seed);
    for (int i = 0; i < len; i++) {
        u8 v = data[i] + randNum_add();
        score += counts[v];
        counts[v]++;
    }
    return score;
}

static int score_seed_sub(u8 *data, int len, int seed) {
    int counts[256] = {0};
    int score = 0;
    xorshift_seed(seed);
    for (int i = 0; i < len; i++) {
        u8 v = data[i] - randNum_add();
        score += counts[v];
        counts[v]++;
    }
    return score;
}

double FindBestThreePhase(u8 *data, int size) {
    int num_seeds = 65537;
    SeedScore *ss = malloc(num_seeds * sizeof(SeedScore));

    for (int s = 0; s < num_seeds; s++) {
        ss[s].seed  = s;
        ss[s].score = score_seed(data, 256, s);
    }
    qsort(ss, num_seeds, sizeof(SeedScore), cmp_score_desc);
    int k1 = 5000;

    for (int k = 0; k < k1; k++)
        ss[k].score = score_seed(data, 2048, ss[k].seed);
    qsort(ss, k1, sizeof(SeedScore), cmp_score_desc);
    int k2 = 500;

    long best_score = 0;
    int  best_seed  = 0;
    for (int k = 0; k < k2; k++) {
        long sc = score_seed(data, size, ss[k].seed);
        if (sc > best_score) { best_score = sc; best_seed = ss[k].seed; }
    }
    free(ss);

    u8 *buf = malloc(size);
    xorshift_seed(best_seed);
    for (int i = 0; i < size; i++)
        buf[i] = data[i] + randNum_add();
    double e = entropy(buf, size);
    printf("[3-phase]   seed=%5d  entropy=%lf\n", best_seed, e);
    free(buf);
    return e;
}

double FindBest24Bit(u8 *data, int size) {
    int num_seeds = 1 << 24;
    SeedScore *ss = malloc((size_t)num_seeds * sizeof(SeedScore));
    if (!ss) { printf("out of memory\n"); return 9.0; }

    for (int s = 0; s < num_seeds; s++) {
        ss[s].seed  = s;
        ss[s].score = score_seed(data, 32, s);
    }
    qsort(ss, num_seeds, sizeof(SeedScore), cmp_score_desc);
    int k1 = 30000;

    for (int k = 0; k < k1; k++)
        ss[k].score = score_seed(data, 512, ss[k].seed);
    qsort(ss, k1, sizeof(SeedScore), cmp_score_desc);
    int k2 = 3000;

    for (int k = 0; k < k2; k++)
        ss[k].score = score_seed(data, 2048, ss[k].seed);
    qsort(ss, k2, sizeof(SeedScore), cmp_score_desc);
    int k3 = 300;

    int best_score = 0, best_seed = 0;
    for (int k = 0; k < k3; k++) {
        int sc = score_seed(data, size, ss[k].seed);
        if (sc > best_score) { best_score = sc; best_seed = ss[k].seed; }
    }
    free(ss);

    u8 *buf = malloc(size);
    xorshift_seed(best_seed);
    for (int i = 0; i < size; i++)
        buf[i] = data[i] + randNum_add();
    double e = entropy(buf, size);
    printf("[24-bit]    seed=%8d  entropy=%lf\n", best_seed, e);
    free(buf);
    return e;
}

static void do_shuffle_fast(u8 *data, int *perm, u8 *tmp_buf, int n, uint64_t seed) {
    for (int i = 0; i < n; i++) perm[i] = i;
    uint64_t s = seed ? seed : 1;
    for (int i = n - 1; i > 0; i--) {
        int j = (int)(shuffle_rng(&s) % (uint64_t)(i + 1));
        int t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    for (int i = 0; i < n; i++) tmp_buf[perm[i]] = data[i];
    memcpy(data, tmp_buf, n);
}

static void print_stats(const char *label, u8 *data, int size) {
    int counts[256] = {0};
    long sum = 0;
    for (int i = 0; i < size; i++) { counts[data[i]]++; sum += data[i]; }
    int mode = 0;
    for (int i = 1; i < 256; i++) if (counts[i] > counts[mode]) mode = i;
    printf("%s: avg=%.2f  mode=%d (x%d)\n",
           label, (double)sum / size, mode, counts[mode]);
}


double RunModules(u8 *data, int size) {
    const int CHUNK = 256;
    int num_chunks = (size + CHUNK - 1) / CHUNK;

    u8  *cur   = malloc(size);
    u8  *tmp   = malloc(size);
    u8  *orig  = malloc(size);
    int *prngs = malloc(num_chunks * sizeof(int));
    int *pats  = malloc(num_chunks * sizeof(int));
    memcpy(cur, data, size);
    memcpy(orig, data, size);

    // --- parameter logs for reversal ---
    int ran1 = 0;
    int *seeds1      = malloc(num_chunks * sizeof(int));
    int ran2 = 0;
    int *seeds2      = malloc(num_chunks * sizeof(int));
    int *pats2       = malloc(num_chunks * sizeof(int));
    int ran3 = 0;
    int *shifts3_log = malloc(num_chunks * sizeof(int));
    int ran4a = 0;
    int *shifts4a_log= malloc(num_chunks * sizeof(int));
    int ran4c = 0;
    int *sH4_log     = malloc(num_chunks * sizeof(int));
    int *sC4_log     = malloc(num_chunks * sizeof(int));
    int *pats4c      = malloc(num_chunks * sizeof(int));

    // multi-pass logs: [pass_index * num_chunks + chunk]
    int *seeds5_log  = malloc(64  * num_chunks * sizeof(int));
    u8  *dirs5_log   = malloc(64  * num_chunks);
    int n5 = 0;
    int *shifts6_log = malloc(128 * num_chunks * sizeof(int));
    int n6 = 0;
    u8  *xors7_log   = malloc(128 * num_chunks);
    int n7 = 0;
    int *seeds8_log  = malloc(64  * num_chunks * sizeof(int));
    int n8 = 0;
    u8  *as9_log     = malloc(64  * num_chunks);
    u8  *bs9_log     = malloc(64  * num_chunks);
    int n9 = 0;
    u8  *ag11_log    = malloc(64  * num_chunks);
    u8  *bg11_log    = malloc(64  * num_chunks);
    int n11 = 0;
    int *seeds12_log = malloc(32  * num_chunks * sizeof(int));
    u8  *as12_log    = malloc(32  * num_chunks);
    u8  *dirs12_log  = malloc(32  * num_chunks);
    int n12 = 0;
    u8  *rots13_log  = malloc(64  * num_chunks);
    int n13 = 0;

    // per-outer start indices (index [num_outers] = sentinel)
    int outer_n5[9],  outer_n6[9],  outer_n7[9],  outer_n8[9];
    int outer_n9[9],  outer_n11[9], outer_n12[9], outer_n13[9];
    int num_outers = 0;

    double e0 = entropy(cur, size);
    printf("start:    entropy=%lf  (%d chunks x %d bytes)\n", e0, num_chunks, CHUNK);

    // --- Module 1: OR-PRNG applied to every byte ---
    double e1, e2;
    {
        clock_t t0 = clock();
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK;
            int len = (off + CHUNK <= size) ? CHUNK : (size - off);
            int best_sc = 0, best_seed = 0;
            for (int seed = 0; seed < 65536; seed++) {
                int sc = score_seed(cur + off, len, seed);
                if (sc > best_sc) { best_sc = sc; best_seed = seed; }
            }
            prngs[c] = best_seed;
        }
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK;
            int len = (off + CHUNK <= size) ? CHUNK : (size - off);
            xorshift_seed(prngs[c]);
            for (int i = 0; i < len; i++)
                tmp[off + i] = cur[off + i] + randNum_add();
        }
        e1 = entropy(tmp, size);
        if (e1 < e0) {
            ran1 = 1;
            memcpy(seeds1, prngs, num_chunks * sizeof(int));
            memcpy(cur, tmp, size);
            printf("module 1: entropy=%lf  profit=%+.2f  time=%.1fs\n",
                   e1, (e0 - e1) * size, (double)(clock() - t0) / CLOCKS_PER_SEC);
        } else {
            e1 = e0;
            printf("module 1: skipped (no improvement)  time=%.1fs\n",
                   (double)(clock() - t0) / CLOCKS_PER_SEC);
        }
    }

    // --- Module 2: OR-PRNG applied only to pattern-selected bytes ---
    {
        clock_t t0 = clock();
        int nm2[4] = {0};
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK;
            int len = (off + CHUNK <= size) ? CHUNK : (size - off);
            int best_sc = 0, best_seed = 0, best_pat = 0;
            for (int pat = 0; pat < 4; pat++) {
                for (int seed = 0; seed < 65536; seed++) {
                    int sc = score_seed_pattern(cur + off, len, seed, pat);
                    if (sc > best_sc) { best_sc = sc; best_seed = seed; best_pat = pat; }
                }
            }
            prngs[c] = best_seed;
            pats[c]  = best_pat;
        }
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK;
            int len = (off + CHUNK <= size) ? CHUNK : (size - off);
            int pat = pats[c];
            nm2[pat]++;
            xorshift_seed(prngs[c]);
            for (int i = 0; i < len; i++)
                tmp[off + i] = pattern_applies(i, pat)
                    ? (u8)(cur[off + i] + randNum_add())
                    : cur[off + i];
        }
        e2 = entropy(tmp, size);
        if (e2 < e1) {
            ran2 = 1;
            memcpy(seeds2, prngs, num_chunks * sizeof(int));
            memcpy(pats2,  pats,  num_chunks * sizeof(int));
            memcpy(cur, tmp, size);
            printf("module 2: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                   "  [odd:%d even:%d 0110:%d 1001:%d]\n",
                   e2, (e1 - e2) * size, (e0 - e2) * size,
                   (double)(clock() - t0) / CLOCKS_PER_SEC,
                   nm2[0], nm2[1], nm2[2], nm2[3]);
        } else {
            e2 = e1;
            printf("module 2: skipped (no improvement)  time=%.1fs\n",
                   (double)(clock() - t0) / CLOCKS_PER_SEC);
        }
    }

    // --- Module 3: cross-chunk constant-shift alignment ---
    double e3;
    {
        clock_t t0 = clock();
        int *shifts3 = malloc(num_chunks * sizeof(int));
        int global[256] = {0};
        for (int i = 0; i < size; i++) global[cur[i]]++;
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK;
            int len = (off + CHUNK <= size) ? CHUNK : (size - off);
            int local[256] = {0};
            for (int i = 0; i < len; i++) local[cur[off+i]]++;
            int rest[256];
            for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
            int best_s = 0, best_xcorr = -1;
            for (int s = 0; s < 256; s++) {
                int xcorr = 0;
                for (int v = 0; v < 256; v++)
                    xcorr += local[v] * rest[(v + s) & 0xFF];
                if (xcorr > best_xcorr) { best_xcorr = xcorr; best_s = s; }
            }
            shifts3[c] = best_s;
        }
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK;
            int len = (off + CHUNK <= size) ? CHUNK : (size - off);
            for (int i = 0; i < len; i++)
                tmp[off+i] = (u8)(cur[off+i] + (u8)shifts3[c]);
        }
        e3 = entropy(tmp, size);
        if (e3 < e2) {
            ran3 = 1;
            memcpy(shifts3_log, shifts3, num_chunks * sizeof(int));
            int nshifted = 0;
            for (int c = 0; c < num_chunks; c++) if (shifts3[c]) nshifted++;
            memcpy(cur, tmp, size);
            printf("module 3: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                   "  [chunk-align  shifted:%d/%d]\n",
                   e3, (e2 - e3) * size, (e0 - e3) * size,
                   (double)(clock() - t0) / CLOCKS_PER_SEC,
                   nshifted, num_chunks);
        } else {
            e3 = e2;
            printf("module 3: skipped (no improvement)  [chunk-align]\n");
        }
        free(shifts3);
    }

    // --- Module 4a: second cross-chunk alignment pass ---
    double e4a;
    {
        clock_t t0 = clock();
        int *shifts4a = malloc(num_chunks * sizeof(int));
        int global[256] = {0};
        for (int i = 0; i < size; i++) global[cur[i]]++;
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            int local[256] = {0};
            for (int i = 0; i < len; i++) local[cur[off+i]]++;
            int rest[256];
            for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
            int best_s = 0, best_xcorr = -1;
            for (int s = 0; s < 256; s++) {
                int xcorr = 0;
                for (int v = 0; v < 256; v++) xcorr += local[v] * rest[(v+s)&0xFF];
                if (xcorr > best_xcorr) { best_xcorr = xcorr; best_s = s; }
            }
            shifts4a[c] = best_s;
        }
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            for (int i = 0; i < len; i++) tmp[off+i] = (u8)(cur[off+i] + (u8)shifts4a[c]);
        }
        e4a = entropy(tmp, size);
        if (e4a < e3) {
            ran4a = 1;
            memcpy(shifts4a_log, shifts4a, num_chunks * sizeof(int));
            int nshifted = 0;
            for (int c = 0; c < num_chunks; c++) if (shifts4a[c]) nshifted++;
            memcpy(cur, tmp, size);
            printf("module 4a: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                   "  [chunk-align-2  shifted:%d/%d]\n",
                   e4a, (e3-e4a)*size, (e0-e4a)*size,
                   (double)(clock()-t0)/CLOCKS_PER_SEC, nshifted, num_chunks);
        } else {
            e4a = e3;
            printf("module 4a: skipped  [chunk-align-2]\n");
        }
        free(shifts4a);
    }

    // --- Module 4c: H/C two-constant cross-chunk alignment ---
    double e4c;
    {
        clock_t t0 = clock();
        int *sH4 = malloc(num_chunks * sizeof(int));
        int *sC4 = malloc(num_chunks * sizeof(int));
        int gH[256] = {0}, gC[256] = {0};
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            int pat = pats[c];
            for (int i = 0; i < len; i++) {
                if (pattern_applies(i, pat)) gH[cur[off+i]]++;
                else                         gC[cur[off+i]]++;
            }
        }
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            int pat = pats[c];
            int lH[256] = {0}, lC[256] = {0};
            for (int i = 0; i < len; i++) {
                if (pattern_applies(i, pat)) lH[cur[off+i]]++;
                else                         lC[cur[off+i]]++;
            }
            int rH[256], rC[256];
            for (int v = 0; v < 256; v++) { rH[v] = gH[v]-lH[v]; rC[v] = gC[v]-lC[v]; }
            int best_sH = 0, best_xcH = -1;
            for (int s = 0; s < 256; s++) {
                int xc = 0;
                for (int v = 0; v < 256; v++) xc += lH[v] * rH[(v+s)&0xFF];
                if (xc > best_xcH) { best_xcH = xc; best_sH = s; }
            }
            int best_sC = 0, best_xcC = -1;
            for (int s = 0; s < 256; s++) {
                int xc = 0;
                for (int v = 0; v < 256; v++) xc += lC[v] * rC[(v+s)&0xFF];
                if (xc > best_xcC) { best_xcC = xc; best_sC = s; }
            }
            sH4[c] = best_sH;
            sC4[c] = best_sC;
        }
        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            int pat = pats[c];
            for (int i = 0; i < len; i++) {
                int s = pattern_applies(i, pat) ? sH4[c] : sC4[c];
                tmp[off+i] = (u8)(cur[off+i] + (u8)s);
            }
        }
        e4c = entropy(tmp, size);
        if (e4c < e4a) {
            ran4c = 1;
            memcpy(sH4_log,  sH4,  num_chunks * sizeof(int));
            memcpy(sC4_log,  sC4,  num_chunks * sizeof(int));
            memcpy(pats4c,   pats, num_chunks * sizeof(int));
            memcpy(cur, tmp, size);
            printf("module 4c: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                   "  [hc-align]\n",
                   e4c, (e4a-e4c)*size, (e0-e4c)*size,
                   (double)(clock()-t0)/CLOCKS_PER_SEC);
        } else {
            e4c = e4a;
            printf("module 4c: skipped  [hc-align]\n");
        }
        free(sH4); free(sC4);
    }

    // --- Module 16: Gray code encoding per byte ---
    // Converts bytes between binary and Gray code (g = b ^ (b>>1)).
    // Triggered ~37% of seeds; helps when byte values cluster in a narrow range.
    double e16;
    int ran16 = 0; // 0=skip, 1=binary-to-Gray (b^(b>>1)), 2=Gray-to-binary (inverse)
    {
        clock_t t0 = clock();
        for (int i = 0; i < size; i++) tmp[i] = cur[i] ^ (cur[i] >> 1);
        double e_b2g = entropy(tmp, size);
        u8 *tmp2 = malloc(size);
        for (int i = 0; i < size; i++) {
            u8 g = cur[i], b = g;
            g >>= 1; while (g) { b ^= g; g >>= 1; }
            tmp2[i] = b;
        }
        double e_g2b = entropy(tmp2, size);
        if (e_b2g < e4c && e_b2g <= e_g2b) {
            ran16 = 1; memcpy(cur, tmp, size); e16 = e_b2g;
            printf("module 16: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs  [gray-b2g]\n",
                   e16, (e4c-e16)*size, (e0-e16)*size, (double)(clock()-t0)/CLOCKS_PER_SEC);
        } else if (e_g2b < e4c) {
            ran16 = 2; memcpy(cur, tmp2, size); e16 = e_g2b;
            printf("module 16: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs  [gray-g2b]\n",
                   e16, (e4c-e16)*size, (e0-e16)*size, (double)(clock()-t0)/CLOCKS_PER_SEC);
        } else {
            e16 = e4c;
            printf("module 16: skipped  [gray]\n");
        }
        free(tmp2);
    }

    // --- Module 20: even/odd byte deinterleave ---
    // Splits the stream into even-indexed bytes (first half) and odd-indexed bytes (second half).
    // Triggers when per-position PRNG modules (M5, M12) create an even/odd distribution asymmetry.
    double e20;
    int ran20 = 0;
    {
        clock_t t0 = clock();
        for (int i = 0; i < size/2; i++) {
            tmp[i]          = cur[i*2];
            tmp[i + size/2] = cur[i*2 + 1];
        }
        e20 = entropy(tmp, size);
        if (e20 < e16) {
            ran20 = 1; memcpy(cur, tmp, size);
            printf("module 20: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs  [even-odd-split]\n",
                   e20, (e16-e20)*size, (e0-e20)*size, (double)(clock()-t0)/CLOCKS_PER_SEC);
        } else {
            e20 = e16;
        }
    }

    // --- Module 21: cross-chunk XOR butterfly ---
    // Treats block as 16x256 matrix. At each byte position, applies 4-stage XOR butterfly
    // across the 16 chunk values, concentrating inter-chunk correlation into low-index slots.
    // Inverse: same butterfly stages applied in reverse order.
    double e21;
    int ran21 = 0;
    {
        clock_t t0 = clock();
        memcpy(tmp, cur, size);
        int fwd_strides[4] = {8, 4, 2, 1};
        for (int si = 0; si < 4; si++) {
            int st = fwd_strides[si];
            for (int j = 0; j < CHUNK; j++)
                for (int c = 0; c < num_chunks; c++)
                    if ((c % (2*st)) < st)
                        tmp[(c+st)*CHUNK + j] ^= tmp[c*CHUNK + j];
        }
        e21 = entropy(tmp, size);
        if (e21 < e20) {
            ran21 = 1; memcpy(cur, tmp, size);
            printf("module 21: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs  [chunk-xor-butterfly]\n",
                   e21, (e20-e21)*size, (e0-e21)*size, (double)(clock()-t0)/CLOCKS_PER_SEC);
        } else {
            e21 = e20;
        }
    }

    // --- Module 22: nibble plane separation ---
    // For each pair (cur[2i], cur[2i+1]): pack high nibbles together, low nibbles together.
    // High nibble plane is concentrated near 6 when mode=0x60; separating it lowers combined H.
    double e22;
    int ran22 = 0;
    {
        clock_t t0 = clock();
        for (int i = 0; i < size/2; i++) {
            tmp[i]          = (cur[2*i] >> 4) | (cur[2*i+1] & 0xF0);
            tmp[i + size/2] = (cur[2*i] & 0xF) | ((cur[2*i+1] & 0xF) << 4);
        }
        e22 = entropy(tmp, size);
        if (e22 < e21) {
            ran22 = 1; memcpy(cur, tmp, size);
            printf("module 22: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs  [nibble-split]\n",
                   e22, (e21-e22)*size, (e0-e22)*size, (double)(clock()-t0)/CLOCKS_PER_SEC);
        } else {
            e22 = e21;
        }
    }

    // --- Outer convergence loop: modules 5-12 ---
    // Cycles all transforms until a full pass makes no improvement.
    double e_cur = e22;

    // precompute Z/256Z multiplicative inverses for odd values (used by M9, M12)
    u8 inv256[128];
    for (int k = 0; k < 128; k++) {
        int a = 2*k + 1;
        for (int inv = 1; inv < 256; inv += 2)
            if (((a * inv) & 0xFF) == 1) { inv256[k] = (u8)inv; break; }
    }

    int outer_passes = 0;
    for (int outer = 0; outer < 8; outer++) {
        double e_outer_start = e_cur;
        outer_n5[outer]  = n5;  outer_n6[outer]  = n6;
        outer_n7[outer]  = n7;  outer_n8[outer]  = n8;
        outer_n9[outer]  = n9;  outer_n11[outer] = n11; outer_n12[outer] = n12;
        outer_n13[outer] = n13;
        if (outer > 0)
            printf("--- outer pass %d (entropy=%.6f) ---\n", outer + 1, e_cur);

        // --- Module 5: iterated per-chunk AND-PRNG cross-chunk xcorr ---
        {
            double e_before = e_cur;
            clock_t t0 = clock();
            SeedScore *ss = malloc(65537 * sizeof(SeedScore));
            int *seeds5 = malloc(num_chunks * sizeof(int));
            u8  *dirs5  = malloc(num_chunks);
            int *lc = malloc(256 * sizeof(int));
            int *ls = malloc(256 * sizeof(int));
            int passes = 0;

            for (int iter = 0; iter < 8; iter++) {
                int global[256] = {0};
                for (int i = 0; i < size; i++) global[cur[i]]++;
                int any_found = 0;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    int local[256] = {0};
                    for (int i = 0; i < len; i++) local[cur[off+i]]++;
                    int rest[256];
                    for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
                    int baseline = 0;
                    for (int v = 0; v < 256; v++) baseline += local[v] * rest[v];
                    int pfx = len < 64 ? len : 64;
                    for (int s = 0; s <= 65536; s++) {
                        memset(lc, 0, 256*sizeof(int)); memset(ls, 0, 256*sizeof(int));
                        xorshift_seed(s);
                        for (int i = 0; i < pfx; i++) {
                            u8 r = randNum_and();
                            lc[(u8)(cur[off+i] + r)]++;
                            ls[(u8)(cur[off+i] - r)]++;
                        }
                        int xc = 0, xs = 0;
                        for (int v = 0; v < 256; v++) { xc += lc[v]*rest[v]; xs += ls[v]*rest[v]; }
                        ss[s].seed = s; ss[s].score = xc > xs ? xc : xs;
                    }
                    qsort(ss, 65537, sizeof(SeedScore), cmp_score_desc);
                    int best_s = -1, best_xcorr = baseline;
                    u8 best_dir = 0;
                    for (int k = 0; k < 2000; k++) {
                        int s = ss[k].seed;
                        memset(lc, 0, 256*sizeof(int)); memset(ls, 0, 256*sizeof(int));
                        xorshift_seed(s);
                        for (int i = 0; i < len; i++) {
                            u8 r = randNum_and();
                            lc[(u8)(cur[off+i] + r)]++;
                            ls[(u8)(cur[off+i] - r)]++;
                        }
                        int xc = 0, xs = 0;
                        for (int v = 0; v < 256; v++) { xc += lc[v]*rest[v]; xs += ls[v]*rest[v]; }
                        if (xc > best_xcorr) { best_xcorr = xc; best_s = s; best_dir = 0; }
                        if (xs > best_xcorr) { best_xcorr = xs; best_s = s; best_dir = 1; }
                    }
                    seeds5[c] = best_s; dirs5[c] = best_dir;
                    if (best_s >= 0) any_found = 1;
                }
                if (!any_found) break;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    if (seeds5[c] < 0) { memcpy(tmp+off, cur+off, len); continue; }
                    xorshift_seed(seeds5[c]);
                    for (int i = 0; i < len; i++) {
                        u8 r = randNum_and();
                        tmp[off+i] = dirs5[c] ? (u8)(cur[off+i]-r) : (u8)(cur[off+i]+r);
                    }
                }
                double e_try = entropy(tmp, size);
                if (e_try >= e_cur) break;
                memcpy(seeds5_log + n5*num_chunks, seeds5, num_chunks*sizeof(int));
                memcpy(dirs5_log  + n5*num_chunks, dirs5,  num_chunks);
                n5++;
                memcpy(cur, tmp, size);
                e_cur = e_try;
                passes++;
            }
            free(ss); free(seeds5); free(dirs5); free(lc); free(ls);
            if (passes > 0)
                printf("module 5:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                       "  [and-xcorr  passes:%d]\n",
                       e_cur, (e_before-e_cur)*size, (e0-e_cur)*size,
                       (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        }

        // --- Module 6: iterated cross-chunk re-alignment ---
        {
            double e_before = e_cur;
            clock_t t0 = clock();
            int *shifts6 = malloc(num_chunks * sizeof(int));
            int passes = 0;
            for (int iter = 0; iter < 16; iter++) {
                int global[256] = {0};
                for (int i = 0; i < size; i++) global[cur[i]]++;
                int any_nonzero = 0;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    int local[256] = {0};
                    for (int i = 0; i < len; i++) local[cur[off+i]]++;
                    int rest[256];
                    for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
                    int best_s = 0, best_xcorr = -1;
                    for (int s = 0; s < 256; s++) {
                        int xcorr = 0;
                        for (int v = 0; v < 256; v++) xcorr += local[v] * rest[(v+s)&0xFF];
                        if (xcorr > best_xcorr) { best_xcorr = xcorr; best_s = s; }
                    }
                    shifts6[c] = best_s;
                    if (best_s) any_nonzero = 1;
                }
                if (!any_nonzero) break;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    for (int i = 0; i < len; i++) tmp[off+i] = (u8)(cur[off+i] + (u8)shifts6[c]);
                }
                double e_try = entropy(tmp, size);
                if (e_try >= e_cur) break;
                memcpy(shifts6_log + n6*num_chunks, shifts6, num_chunks*sizeof(int));
                n6++;
                memcpy(cur, tmp, size);
                e_cur = e_try;
                passes++;
            }
            free(shifts6);
            if (passes > 0)
                printf("module 6:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                       "  [chunk-align-3  passes:%d]\n",
                       e_cur, (e_before-e_cur)*size, (e0-e_cur)*size,
                       (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        }

        // --- Module 7: iterated per-chunk XOR-constant alignment ---
        {
            double e_before = e_cur;
            clock_t t0 = clock();
            u8 *xors7 = malloc(num_chunks);
            int passes = 0;
            for (int iter = 0; iter < 16; iter++) {
                int global[256] = {0};
                for (int i = 0; i < size; i++) global[cur[i]]++;
                int any_nonzero = 0;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    int local[256] = {0};
                    for (int i = 0; i < len; i++) local[cur[off+i]]++;
                    int rest[256];
                    for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
                    int best_x = 0, best_xcorr = -1;
                    for (int x = 0; x < 256; x++) {
                        int xcorr = 0;
                        for (int v = 0; v < 256; v++) xcorr += local[v] * rest[v ^ x];
                        if (xcorr > best_xcorr) { best_xcorr = xcorr; best_x = x; }
                    }
                    xors7[c] = (u8)best_x;
                    if (best_x) any_nonzero = 1;
                }
                if (!any_nonzero) break;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    for (int i = 0; i < len; i++) tmp[off+i] = cur[off+i] ^ xors7[c];
                }
                double e_try = entropy(tmp, size);
                if (e_try >= e_cur) break;
                memcpy(xors7_log + n7*num_chunks, xors7, num_chunks);
                n7++;
                memcpy(cur, tmp, size);
                e_cur = e_try;
                passes++;
            }
            free(xors7);
            if (passes > 0)
                printf("module 7:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                       "  [xor-align  passes:%d]\n",
                       e_cur, (e_before-e_cur)*size, (e0-e_cur)*size,
                       (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        }

        // --- Module 13: per-chunk cyclic bit rotation ---
        // rotl8(v, k) permutes the histogram of a chunk identically to how an
        // additive shift (M6) does — but via a different group action on {0..255}.
        // The 8-rotation group is not a subgroup of Z/256Z addition, so this finds
        // alignments that M6 cannot reach. Overhead: 3 bits per chunk.
        {
            double e_before = e_cur;
            clock_t t0 = clock();
            u8 *rots13 = malloc(num_chunks);
            int passes = 0;
            for (int iter = 0; iter < 8; iter++) {
                int global[256] = {0};
                for (int i = 0; i < size; i++) global[cur[i]]++;
                int any_nonzero = 0;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    int local[256] = {0};
                    for (int i = 0; i < len; i++) local[cur[off+i]]++;
                    int rest[256];
                    for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
                    int best_k = 0, best_xcorr = -1;
                    for (int k = 0; k < 8; k++) {
                        int xcorr = 0;
                        for (int v = 0; v < 256; v++)
                            xcorr += local[v] * rest[rotl8((u8)v, k)];
                        if (xcorr > best_xcorr) { best_xcorr = xcorr; best_k = k; }
                    }
                    rots13[c] = (u8)best_k;
                    if (best_k) any_nonzero = 1;
                }
                if (!any_nonzero) break;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    for (int i = 0; i < len; i++)
                        tmp[off+i] = rotl8(cur[off+i], rots13[c]);
                }
                double e_try = entropy(tmp, size);
                if (e_try >= e_cur) break;
                memcpy(rots13_log + n13*num_chunks, rots13, num_chunks);
                n13++;
                memcpy(cur, tmp, size);
                e_cur = e_try;
                passes++;
            }
            free(rots13);
            if (passes > 0)
                printf("module 13: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                       "  [bit-rotate  passes:%d]\n",
                       e_cur, (e_before-e_cur)*size, (e0-e_cur)*size,
                       (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        }

        // --- Module 8: iterated per-chunk XOR AND-PRNG ---
        {
            double e_before = e_cur;
            clock_t t0 = clock();
            SeedScore *ss = malloc(65537 * sizeof(SeedScore));
            int *seeds8 = malloc(num_chunks * sizeof(int));
            int *lx = malloc(256 * sizeof(int));
            int passes = 0;
            for (int iter = 0; iter < 8; iter++) {
                int global[256] = {0};
                for (int i = 0; i < size; i++) global[cur[i]]++;
                int any_found = 0;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    int local[256] = {0};
                    for (int i = 0; i < len; i++) local[cur[off+i]]++;
                    int rest[256];
                    for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
                    int baseline = 0;
                    for (int v = 0; v < 256; v++) baseline += local[v] * rest[v];
                    int pfx = len < 64 ? len : 64;
                    for (int s = 0; s <= 65536; s++) {
                        memset(lx, 0, 256*sizeof(int));
                        xorshift_seed(s);
                        for (int i = 0; i < pfx; i++) lx[cur[off+i] ^ randNum_and()]++;
                        int xcorr = 0;
                        for (int v = 0; v < 256; v++) xcorr += lx[v] * rest[v];
                        ss[s].seed = s; ss[s].score = xcorr;
                    }
                    qsort(ss, 65537, sizeof(SeedScore), cmp_score_desc);
                    int best_s = -1, best_xcorr = baseline;
                    for (int k = 0; k < 2000; k++) {
                        int s = ss[k].seed;
                        memset(lx, 0, 256*sizeof(int));
                        xorshift_seed(s);
                        for (int i = 0; i < len; i++) lx[cur[off+i] ^ randNum_and()]++;
                        int xcorr = 0;
                        for (int v = 0; v < 256; v++) xcorr += lx[v] * rest[v];
                        if (xcorr > best_xcorr) { best_xcorr = xcorr; best_s = s; }
                    }
                    seeds8[c] = best_s;
                    if (best_s >= 0) any_found = 1;
                }
                if (!any_found) break;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    if (seeds8[c] < 0) { memcpy(tmp+off, cur+off, len); continue; }
                    xorshift_seed(seeds8[c]);
                    for (int i = 0; i < len; i++) tmp[off+i] = cur[off+i] ^ randNum_and();
                }
                double e_try = entropy(tmp, size);
                if (e_try >= e_cur) break;
                memcpy(seeds8_log + n8*num_chunks, seeds8, num_chunks*sizeof(int));
                n8++;
                memcpy(cur, tmp, size);
                e_cur = e_try;
                passes++;
            }
            free(ss); free(seeds8); free(lx);
            if (passes > 0)
                printf("module 8:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                       "  [xor-and-xcorr  passes:%d]\n",
                       e_cur, (e_before-e_cur)*size, (e0-e_cur)*size,
                       (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        }

        // --- Module 9: iterated per-chunk Z/256Z affine alignment ---
        // v -> (a*v + b) mod 256, a odd. Uses inv256 from outer scope.
        {
            double e_before = e_cur;
            clock_t t0 = clock();
            u8 *as9 = malloc(num_chunks);
            u8 *bs9 = malloc(num_chunks);
            int *ha9 = malloc(256 * sizeof(int));
            int passes = 0;
            for (int iter = 0; iter < 8; iter++) {
                int global[256] = {0};
                for (int i = 0; i < size; i++) global[cur[i]]++;
                int any_nonidentity = 0;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    int local[256] = {0};
                    for (int i = 0; i < len; i++) local[cur[off+i]]++;
                    int rest[256];
                    for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
                    int baseline = 0;
                    for (int v = 0; v < 256; v++) baseline += local[v] * rest[v];
                    int best_xcorr = baseline, best_ka = 0, best_b = 0;
                    for (int ka = 1; ka < 128; ka++) {
                        int a_inv = inv256[ka];
                        for (int w = 0; w < 256; w++)
                            ha9[w] = local[(a_inv * w) & 0xFF];
                        for (int b = 0; b < 256; b++) {
                            int xcorr = 0;
                            for (int w = 0; w < 256; w++)
                                xcorr += ha9[w] * rest[(w + b) & 0xFF];
                            if (xcorr > best_xcorr) {
                                best_xcorr = xcorr; best_ka = ka; best_b = b;
                            }
                        }
                    }
                    as9[c] = (u8)(2*best_ka + 1);
                    bs9[c] = (u8)best_b;
                    if (best_ka != 0 || best_b != 0) any_nonidentity = 1;
                }
                if (!any_nonidentity) break;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    u8 a = as9[c], b = bs9[c];
                    for (int i = 0; i < len; i++)
                        tmp[off+i] = (u8)((a * (unsigned)cur[off+i] + b) & 0xFF);
                }
                double e_try = entropy(tmp, size);
                if (e_try >= e_cur) break;
                memcpy(as9_log + n9*num_chunks, as9, num_chunks);
                memcpy(bs9_log + n9*num_chunks, bs9, num_chunks);
                n9++;
                memcpy(cur, tmp, size);
                e_cur = e_try;
                passes++;
            }
            free(ha9); free(as9); free(bs9);
            if (passes > 0)
                printf("module 9:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                       "  [affine-align  passes:%d]\n",
                       e_cur, (e_before-e_cur)*size, (e0-e_cur)*size,
                       (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        }


        // --- Module 11: iterated per-chunk GF(256) affine (option C) ---
        // v -> gf256_mul(a, v) XOR b; a in GF(256)*, b in GF(256).
        // Different from M9 (Z/256Z arithmetic) and M7 (XOR-only, i.e. a_gf=1 subgroup).
        // Same xcorr trick as M9 but with GF256 inverse and XOR-lag correlation:
        //   xcorr(a,b) = sum_u Ha_gf[u] * rest[u^b]
        //   where Ha_gf[u] = local[gf256_mul(gf256_inv[a], u)]
        // Skip a=1 (identity mult) — that's M7's domain, already handled.
        {
            double e_before = e_cur;
            clock_t t0 = clock();
            u8  *ag11 = malloc(num_chunks);
            u8  *bg11 = malloc(num_chunks);
            int *ha11 = malloc(256 * sizeof(int));
            int passes = 0;
            for (int iter = 0; iter < 8; iter++) {
                int global[256] = {0};
                for (int i = 0; i < size; i++) global[cur[i]]++;
                int any_nonidentity = 0;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    int local[256] = {0};
                    for (int i = 0; i < len; i++) local[cur[off+i]]++;
                    int rest[256];
                    for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
                    int baseline = 0;
                    for (int v = 0; v < 256; v++) baseline += local[v] * rest[v];
                    int best_xcorr = baseline, best_ag = 1, best_bg = 0;
                    for (int ag = 2; ag < 256; ag++) {    // skip ag=1 (M7 territory)
                        u8 ag_inv = gf256_inv_table[ag];
                        for (int w = 0; w < 256; w++)
                            ha11[w] = local[gf256_mul(ag_inv, (u8)w)];
                        for (int b = 0; b < 256; b++) {
                            int xcorr = 0;
                            for (int w = 0; w < 256; w++)
                                xcorr += ha11[w] * rest[w ^ b];
                            if (xcorr > best_xcorr) {
                                best_xcorr = xcorr; best_ag = ag; best_bg = b;
                            }
                        }
                    }
                    ag11[c] = (u8)best_ag;
                    bg11[c] = (u8)best_bg;
                    if (best_ag != 1 || best_bg != 0) any_nonidentity = 1;
                }
                if (!any_nonidentity) break;
                for (int c = 0; c < num_chunks; c++) {
                    int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    u8 ag = ag11[c], bg = bg11[c];
                    for (int i = 0; i < len; i++)
                        tmp[off+i] = gf256_mul(ag, cur[off+i]) ^ bg;
                }
                double e_try = entropy(tmp, size);
                if (e_try >= e_cur) break;
                memcpy(ag11_log + n11*num_chunks, ag11, num_chunks);
                memcpy(bg11_log + n11*num_chunks, bg11, num_chunks);
                n11++;
                memcpy(cur, tmp, size);
                e_cur = e_try;
                passes++;
            }
            free(ha11); free(ag11); free(bg11);
            if (passes > 0)
                printf("module 11: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                       "  [gf256-affine  passes:%d]\n",
                       e_cur, (e_before-e_cur)*size, (e0-e_cur)*size,
                       (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        }

        // --- Module 12: iterated per-chunk Z/256Z affine + AND-PRNG (option A) ---
        // v -> (a*v + r_i(seed)) mod 256, odd a, AND-PRNG stream.
        // Joint search: M9-style xcorr ranks top-3 odd multipliers; for each,
        // M5-style two-phase seed search (64-byte proxy -> top-1000 full eval).
        // Outer loop lets M9 set the multiplier and M12 refine the PRNG stream,
        // or M12 can find a different multiplier altogether if that works better.
        {
            double e_before = e_cur;
            clock_t t0 = clock();
            SeedScore *ss12 = malloc(65537 * sizeof(SeedScore));
            int *seeds12 = malloc(num_chunks * sizeof(int));
            u8  *as12    = malloc(num_chunks);
            u8  *dirs12  = malloc(num_chunks);
            int *lc12 = malloc(256 * sizeof(int));
            int *ls12 = malloc(256 * sizeof(int));
            int *ha12 = malloc(256 * sizeof(int));
            u8  *utmp12 = malloc(CHUNK);
            int passes = 0;

            for (int iter = 0; iter < 4; iter++) {
                int global[256] = {0};
                for (int i = 0; i < size; i++) global[cur[i]]++;
                int any_found = 0;

                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    int local[256] = {0};
                    for (int i = 0; i < len; i++) local[cur[off+i]]++;
                    int rest[256];
                    for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];
                    int baseline = 0;
                    for (int v = 0; v < 256; v++) baseline += local[v] * rest[v];

                    // rank top-3 odd multipliers (ka=1..127) by best-b xcorr
                    int top_ka[3] = {-1, -1, -1};
                    int top_xc[3] = {-1, -1, -1};
                    for (int ka = 1; ka < 128; ka++) {
                        int a_inv = inv256[ka];
                        for (int w = 0; w < 256; w++)
                            ha12[w] = local[(a_inv * w) & 0xFF];
                        int bx = 0;
                        for (int b = 0; b < 256; b++) {
                            int xcorr = 0;
                            for (int w = 0; w < 256; w++)
                                xcorr += ha12[w] * rest[(w + b) & 0xFF];
                            if (xcorr > bx) bx = xcorr;
                        }
                        // insert into sorted top-3 (descending)
                        if (bx > top_xc[2]) {
                            top_xc[2] = bx; top_ka[2] = ka;
                            if (top_xc[2] > top_xc[1]) {
                                int t;
                                t = top_xc[1]; top_xc[1] = top_xc[2]; top_xc[2] = t;
                                t = top_ka[1]; top_ka[1] = top_ka[2]; top_ka[2] = t;
                            }
                            if (top_xc[1] > top_xc[0]) {
                                int t;
                                t = top_xc[0]; top_xc[0] = top_xc[1]; top_xc[1] = t;
                                t = top_ka[0]; top_ka[0] = top_ka[1]; top_ka[1] = t;
                            }
                        }
                    }

                    // for each top multiplier: M5-style proxy then full search
                    int best_s = -1, best_xcorr = baseline;
                    u8 best_a = 1, best_dir = 0;
                    int pfx = len < 64 ? len : 64;

                    for (int r = 0; r < 3; r++) {
                        if (top_ka[r] < 0) continue;
                        u8 a = (u8)(2 * top_ka[r] + 1);
                        for (int i = 0; i < len; i++)
                            utmp12[i] = (u8)((a * (unsigned)cur[off+i]) & 0xFF);

                        // phase 1: proxy (64 bytes), all seeds
                        for (int s = 0; s <= 65536; s++) {
                            memset(lc12, 0, 256*sizeof(int));
                            memset(ls12, 0, 256*sizeof(int));
                            xorshift_seed(s);
                            for (int i = 0; i < pfx; i++) {
                                u8 rr = randNum_and();
                                lc12[(u8)(utmp12[i] + rr)]++;
                                ls12[(u8)(utmp12[i] - rr)]++;
                            }
                            int xc = 0, xs = 0;
                            for (int v = 0; v < 256; v++) { xc += lc12[v]*rest[v]; xs += ls12[v]*rest[v]; }
                            ss12[s].seed = s; ss12[s].score = xc > xs ? xc : xs;
                        }
                        qsort(ss12, 65537, sizeof(SeedScore), cmp_score_desc);

                        // phase 2: full eval of top 1000
                        for (int k = 0; k < 1000; k++) {
                            int s = ss12[k].seed;
                            memset(lc12, 0, 256*sizeof(int));
                            memset(ls12, 0, 256*sizeof(int));
                            xorshift_seed(s);
                            for (int i = 0; i < len; i++) {
                                u8 rr = randNum_and();
                                lc12[(u8)(utmp12[i] + rr)]++;
                                ls12[(u8)(utmp12[i] - rr)]++;
                            }
                            int xc = 0, xs = 0;
                            for (int v = 0; v < 256; v++) { xc += lc12[v]*rest[v]; xs += ls12[v]*rest[v]; }
                            if (xc > best_xcorr) { best_xcorr = xc; best_s = s; best_a = a; best_dir = 0; }
                            if (xs > best_xcorr) { best_xcorr = xs; best_s = s; best_a = a; best_dir = 1; }
                        }
                    }

                    seeds12[c] = best_s;
                    as12[c]    = best_a;
                    dirs12[c]  = best_dir;
                    if (best_s >= 0) any_found = 1;
                }

                if (!any_found) break;

                for (int c = 0; c < num_chunks; c++) {
                    int off = c * CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                    if (seeds12[c] < 0) { memcpy(tmp+off, cur+off, len); continue; }
                    u8 a = as12[c];
                    xorshift_seed(seeds12[c]);
                    for (int i = 0; i < len; i++) {
                        u8 u = (u8)((a * (unsigned)cur[off+i]) & 0xFF);
                        u8 rr = randNum_and();
                        tmp[off+i] = dirs12[c] ? (u8)(u - rr) : (u8)(u + rr);
                    }
                }
                double e_try = entropy(tmp, size);
                if (e_try >= e_cur) break;
                memcpy(seeds12_log + n12*num_chunks, seeds12, num_chunks*sizeof(int));
                memcpy(as12_log    + n12*num_chunks, as12,    num_chunks);
                memcpy(dirs12_log  + n12*num_chunks, dirs12,  num_chunks);
                n12++;
                memcpy(cur, tmp, size);
                e_cur = e_try;
                passes++;
            }

            free(ss12); free(seeds12); free(as12); free(dirs12);
            free(lc12); free(ls12); free(ha12); free(utmp12);

            if (passes > 0)
                printf("module 12: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                       "  [affine-prng  passes:%d]\n",
                       e_cur, (e_before-e_cur)*size, (e0-e_cur)*size,
                       (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        }

        outer_passes++;
        num_outers++;
        if (e_cur >= e_outer_start) break;
    }
    // sentinel: end-of-log indices for the last outer pass
    outer_n5[num_outers]  = n5;  outer_n6[num_outers]  = n6;
    outer_n7[num_outers]  = n7;  outer_n8[num_outers]  = n8;
    outer_n9[num_outers]  = n9;  outer_n11[num_outers] = n11; outer_n12[num_outers] = n12;
    outer_n13[num_outers] = n13;

    if (outer_passes > 1)
        printf("outer loop: converged in %d passes\n", outer_passes);

    {
        int counts[256] = {0}; long sum = 0;
        for (int i = 0; i < size; i++) { counts[cur[i]]++; sum += cur[i]; }
        int mode = 0;
        for (int i = 1; i < 256; i++) if (counts[i] > counts[mode]) mode = i;
        printf("after : entropy=%lf  total=%+.2f  avg=%.2f  mode=%d (x%d)\n",
               e_cur, (e0-e_cur)*size, (double)sum/size, mode, counts[mode]);

        int delta[256] = {0};
        for (int i = 0; i < size-1; i++) delta[(u8)(cur[i+1] - cur[i])]++;
        double h_delta;
        {
            int n = size-1; double s = 0.0;
            for (int v = 0; v < 256; v++) if (delta[v]) s += xlgx[delta[v]];
            h_delta = log2((double)n) - s / n;
        }
        int dmode = 0;
        for (int v = 1; v < 256; v++) if (delta[v] > delta[dmode]) dmode = v;
        printf("struct: delta_entropy=%.4f  delta_mode=%d (x%d)\n",
               h_delta, dmode, delta[dmode]);

        int bit1[8] = {0};
        for (int i = 0; i < size; i++)
            for (int b = 0; b < 8; b++)
                if (cur[i] & (1<<b)) bit1[b]++;
        printf("struct: bit_frac");
        for (int b = 7; b >= 0; b--) printf(" b%d=%.2f", b, (double)bit1[b]/size);
        printf("\n");

        int xdelta[256] = {0};
        for (int j = 0; j < CHUNK && j+CHUNK < size; j++)
            xdelta[(u8)(cur[j] ^ cur[j+CHUNK])]++;
        int xmode = 0;
        for (int v = 1; v < 256; v++) if (xdelta[v] > xdelta[xmode]) xmode = v;
        printf("struct: cross_chunk_xor_mode=%d (x%d of %d pairs)\n",
               xmode, xdelta[xmode], CHUNK);
    }
    // --- REVERSAL + VERIFICATION ---
    u8 *rev = malloc(size);
    memcpy(rev, cur, size);

    // undo outer loop modules in reverse outer-pass order, reverse module order
    for (int outer = num_outers - 1; outer >= 0; outer--) {
        // M12
        for (int p = outer_n12[outer+1]-1; p >= outer_n12[outer]; p--) {
            for (int c = 0; c < num_chunks; c++) {
                int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                int seed = seeds12_log[p*num_chunks+c];
                if (seed < 0) continue;
                u8 a = as12_log[p*num_chunks+c];
                u8 dir = dirs12_log[p*num_chunks+c];
                u8 a_inv = inv256[(a-1)/2];
                xorshift_seed(seed);
                for (int i = 0; i < len; i++) {
                    u8 rr = randNum_and();
                    u8 v = rev[off+i];
                    u8 u = dir ? (u8)(v+rr) : (u8)(v-rr);
                    rev[off+i] = (u8)((a_inv * (unsigned)u) & 0xFF);
                }
            }
        }
        // M11
        for (int p = outer_n11[outer+1]-1; p >= outer_n11[outer]; p--) {
            for (int c = 0; c < num_chunks; c++) {
                int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                u8 ag = ag11_log[p*num_chunks+c];
                u8 bg = bg11_log[p*num_chunks+c];
                for (int i = 0; i < len; i++)
                    rev[off+i] = gf256_mul(gf256_inv_table[ag], rev[off+i] ^ bg);
            }
        }
        // M9
        for (int p = outer_n9[outer+1]-1; p >= outer_n9[outer]; p--) {
            for (int c = 0; c < num_chunks; c++) {
                int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                u8 a = as9_log[p*num_chunks+c];
                u8 b = bs9_log[p*num_chunks+c];
                u8 a_inv = inv256[(a-1)/2];
                for (int i = 0; i < len; i++)
                    rev[off+i] = (u8)((a_inv * (unsigned)((rev[off+i]-b) & 0xFF)) & 0xFF);
            }
        }
        // M8 (XOR is self-inverse)
        for (int p = outer_n8[outer+1]-1; p >= outer_n8[outer]; p--) {
            for (int c = 0; c < num_chunks; c++) {
                int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                int seed = seeds8_log[p*num_chunks+c];
                if (seed < 0) continue;
                xorshift_seed(seed);
                for (int i = 0; i < len; i++)
                    rev[off+i] ^= randNum_and();
            }
        }
        // M13
        for (int p = outer_n13[outer+1]-1; p >= outer_n13[outer]; p--) {
            for (int c = 0; c < num_chunks; c++) {
                int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                u8 k = rots13_log[p*num_chunks+c];
                u8 inv_k = (8 - k) & 7;
                for (int i = 0; i < len; i++)
                    rev[off+i] = rotl8(rev[off+i], inv_k);
            }
        }
        // M7 (XOR is self-inverse)
        for (int p = outer_n7[outer+1]-1; p >= outer_n7[outer]; p--) {
            for (int c = 0; c < num_chunks; c++) {
                int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                u8 x = xors7_log[p*num_chunks+c];
                for (int i = 0; i < len; i++)
                    rev[off+i] ^= x;
            }
        }
        // M6
        for (int p = outer_n6[outer+1]-1; p >= outer_n6[outer]; p--) {
            for (int c = 0; c < num_chunks; c++) {
                int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                u8 s = (u8)shifts6_log[p*num_chunks+c];
                for (int i = 0; i < len; i++)
                    rev[off+i] = (u8)(rev[off+i] - s);
            }
        }
        // M5
        for (int p = outer_n5[outer+1]-1; p >= outer_n5[outer]; p--) {
            for (int c = 0; c < num_chunks; c++) {
                int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
                int seed = seeds5_log[p*num_chunks+c];
                if (seed < 0) continue;
                u8 dir = dirs5_log[p*num_chunks+c];
                xorshift_seed(seed);
                for (int i = 0; i < len; i++) {
                    u8 rr = randNum_and();
                    rev[off+i] = dir ? (u8)(rev[off+i]+rr) : (u8)(rev[off+i]-rr);
                }
            }
        }
    }

    // undo M22 (nibble split: reconstruct original bytes from packed nibble planes)
    if (ran22) {
        u8 *rb = malloc(size);
        for (int i = 0; i < size/2; i++) {
            rb[2*i]   = ((rev[i] & 0xF) << 4) | (rev[i + size/2] & 0xF);
            rb[2*i+1] = (rev[i] & 0xF0) | ((rev[i + size/2] >> 4) & 0xF);
        }
        memcpy(rev, rb, size);
        free(rb);
    }

    // undo M21 (XOR butterfly: apply same butterfly stages in reverse order)
    if (ran21) {
        int inv_strides[4] = {1, 2, 4, 8};
        for (int si = 0; si < 4; si++) {
            int st = inv_strides[si];
            for (int j = 0; j < CHUNK; j++)
                for (int c = 0; c < num_chunks; c++)
                    if ((c % (2*st)) < st)
                        rev[(c+st)*CHUNK + j] ^= rev[c*CHUNK + j];
        }
    }

    // undo M20 (even/odd split: reinterleave the two halves)
    if (ran20) {
        u8 *rb = malloc(size);
        for (int i = 0; i < size/2; i++) {
            rb[2*i]   = rev[i];
            rb[2*i+1] = rev[i + size/2];
        }
        memcpy(rev, rb, size);
        free(rb);
    }

    // undo M16 (Gray code: b2g inverse is g2b, and vice versa)
    if (ran16 == 1) { // was b2g → undo with g2b
        for (int i = 0; i < size; i++) {
            u8 g = rev[i], b = g;
            g >>= 1; while (g) { b ^= g; g >>= 1; }
            rev[i] = b;
        }
    } else if (ran16 == 2) { // was g2b → undo with b2g
        for (int i = 0; i < size; i++) rev[i] = rev[i] ^ (rev[i] >> 1);
    }

    // undo single-pass modules in reverse: M4c, M4a, M3, M2, M1
    if (ran4c) {
        for (int c = 0; c < num_chunks; c++) {
            int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            int pat = pats4c[c];
            for (int i = 0; i < len; i++) {
                u8 s = (u8)(pattern_applies(i, pat) ? sH4_log[c] : sC4_log[c]);
                rev[off+i] = (u8)(rev[off+i] - s);
            }
        }
    }
    if (ran4a) {
        for (int c = 0; c < num_chunks; c++) {
            int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            for (int i = 0; i < len; i++)
                rev[off+i] = (u8)(rev[off+i] - (u8)shifts4a_log[c]);
        }
    }
    if (ran3) {
        for (int c = 0; c < num_chunks; c++) {
            int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            for (int i = 0; i < len; i++)
                rev[off+i] = (u8)(rev[off+i] - (u8)shifts3_log[c]);
        }
    }
    if (ran2) {
        for (int c = 0; c < num_chunks; c++) {
            int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            int pat = pats2[c];
            xorshift_seed(seeds2[c]);
            for (int i = 0; i < len; i++)
                if (pattern_applies(i, pat))
                    rev[off+i] = (u8)(rev[off+i] - randNum_add());
        }
    }
    if (ran1) {
        for (int c = 0; c < num_chunks; c++) {
            int off = c*CHUNK, len = (off+CHUNK<=size)?CHUNK:(size-off);
            xorshift_seed(seeds1[c]);
            for (int i = 0; i < len; i++)
                rev[off+i] = (u8)(rev[off+i] - randNum_add());
        }
    }

    int mismatches = 0;
    for (int i = 0; i < size; i++) if (rev[i] != orig[i]) mismatches++;
    if (mismatches == 0)
        printf("verify: PASS — all %d bytes recovered exactly\n", size);
    else
        printf("verify: FAIL — %d/%d bytes mismatch\n", mismatches, size);
    free(rev);

    free(cur); free(tmp); free(orig); free(prngs); free(pats);
    free(seeds1); free(seeds2); free(pats2);
    free(shifts3_log); free(shifts4a_log);
    free(sH4_log); free(sC4_log); free(pats4c);
    free(seeds5_log); free(dirs5_log);
    free(shifts6_log); free(xors7_log);
    free(seeds8_log);
    free(as9_log); free(bs9_log);
    free(ag11_log); free(bg11_log);
    free(seeds12_log); free(as12_log); free(dirs12_log);
    free(rots13_log);
    return (e0 - e_cur) * size;
}

int main() {
    init_entropy_table();
    init_gf256_tables();
    int size = 4096;
    u8 *data = malloc(size);
    CSV_XY *csv = csv_xy_open("output.csv", "seeds", "profit","null");
    for (int seeds = 0; seeds < 1; seeds++){
        xorshift_seed((uint64_t)(seeds + 1) * 0x9e3779b97f4a7c15ULL);
        for (int i = 0; i < size; i++)
            data[i] = xorshift();

        print_stats("before", data, size);
        csv_xy_add(csv, seeds, RunModules(data, size),0);
    }

    csv_xy_close(csv);
    free(data);

    return 0;
}
