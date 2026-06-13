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

// prediction coding: find seed where PRNG(seed) best approximates data
// residual = data - PRNG, clustered near 0 if PRNG ~= data
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

    // show residual histogram around 0
    xorshift_seed(best_seed);
    int near_zero = 0;
    for (int i = 0; i < size; i++) {
        u8 r = data[i] - randNum_add();
        if (r <= 5 || r >= 251) near_zero++;  // within +-5 of 0 mod 256
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


// pattern_applies: which byte positions get the PRNG treatment in module 2
// 0=odd  1=even  2=0110(pos 1,2 of every 4)  3=1001(pos 0,3 of every 4)
static inline int pattern_applies(int i, int pat) {
    switch (pat) {
        case 0: return  (i & 1);                   // odd
        case 1: return !(i & 1);                   // even
        case 2: return (i%4==1 || i%4==2);         // 0110
        case 3: return (i%4==0 || i%4==3);         // 1001
        case 4: return (i%4==0 || i%4==1);         // 1100
        case 5: return (i%4==2 || i%4==3);         // 0011
        case 6: return (i%4 != 3);                 // 1110  (skip last of every 4)
        default: return (i%4 != 0);                // 0111  (skip first of every 4)
    }
}

// module 2 scoring: PRNG only applied to positions selected by pattern
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

// same but subtracts PRNG — pushes bytes in the opposite direction from add
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

    // phase 1: 256 bytes, all seeds -> keep top 5000
    for (int s = 0; s < num_seeds; s++) {
        ss[s].seed  = s;
        ss[s].score = score_seed(data, 256, s);
    }
    qsort(ss, num_seeds, sizeof(SeedScore), cmp_score_desc);
    int k1 = 5000;

    // phase 2: 2048 bytes, top 5000 -> keep top 500
    for (int k = 0; k < k1; k++) {
        ss[k].score = score_seed(data, 2048, ss[k].seed);
    }
    qsort(ss, k1, sizeof(SeedScore), cmp_score_desc);
    int k2 = 500;

    // phase 3: full size, top 200 -> pick winner
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
    int num_seeds = 1 << 24;  // 16,777,216
    SeedScore *ss = malloc((size_t)num_seeds * sizeof(SeedScore));
    if (!ss) { printf("out of memory\n"); return 9.0; }

    // phase 1: 32 bytes, all 16M seeds -> keep top 30000
    for (int s = 0; s < num_seeds; s++) {
        ss[s].seed  = s;
        ss[s].score = score_seed(data, 32, s);
    }
    qsort(ss, num_seeds, sizeof(SeedScore), cmp_score_desc);
    int k1 = 30000;

    // phase 2: 512 bytes, top 30000 -> keep top 3000
    for (int k = 0; k < k1; k++)
        ss[k].score = score_seed(data, 512, ss[k].seed);
    qsort(ss, k1, sizeof(SeedScore), cmp_score_desc);
    int k2 = 3000;

    // phase 3: 2048 bytes, top 3000 -> keep top 300
    for (int k = 0; k < k2; k++)
        ss[k].score = score_seed(data, 2048, ss[k].seed);
    qsort(ss, k2, sizeof(SeedScore), cmp_score_desc);
    int k3 = 300;

    // phase 4: full size, top 300 -> pick winner
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

// malloc-free shuffle for hot loops — caller owns perm (size*sizeof(int)) and tmp_buf (size bytes)
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
    int *prngs = malloc(num_chunks * sizeof(int));
    int *pats  = malloc(num_chunks * sizeof(int));
    memcpy(cur, data, size);

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
    // patterns: 0=odd  1=even  2=0110  3=1001
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
    // Modules 1 and 2 optimised each chunk independently, leaving each chunk with
    // its own local mode (the value its seed/pattern happened to concentrate bytes at).
    // Different seeds → different local modes.  Cross-chunk collisions were never
    // explicitly optimised.  For each chunk we search the 256 possible constant shifts
    // and pick the one that maximises Σ_v local[v] * rest[(v+s)%256] — i.e. the
    // cross-correlation of the chunk's distribution against the rest of the buffer.
    // Within-chunk collisions are invariant to constant shifts (bijection), so this
    // search can only create new collisions, never destroy existing ones.
    // Overhead: 1 byte per chunk (shift amount) — fully reversible.
    double e3;
    {
        clock_t t0 = clock();
        int *shifts3 = malloc(num_chunks * sizeof(int));

        // global histogram over current data
        int global[256] = {0};
        for (int i = 0; i < size; i++) global[cur[i]]++;

        for (int c = 0; c < num_chunks; c++) {
            int off = c * CHUNK;
            int len = (off + CHUNK <= size) ? CHUNK : (size - off);

            int local[256] = {0};
            for (int i = 0; i < len; i++) local[cur[off+i]]++;

            // rest-of-buffer histogram (global minus this chunk)
            int rest[256];
            for (int v = 0; v < 256; v++) rest[v] = global[v] - local[v];

            // find shift s* = argmax_s Σ_v local[v] * rest[(v+s) % 256]
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
            int s   = shifts3[c];
            for (int i = 0; i < len; i++)
                tmp[off+i] = (u8)(cur[off+i] + (u8)s);
        }

        e3 = entropy(tmp, size);
        if (e3 < e2) {
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
    // Module 3's shifts changed the global histogram, so chunk-vs-rest cross-correlations
    // are in a new state. A second pass often finds residual misalignment to fix.
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
            int s = shifts4a[c];
            for (int i = 0; i < len; i++) tmp[off+i] = (u8)(cur[off+i] + (u8)s);
        }
        e4a = entropy(tmp, size);
        if (e4a < e3) {
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
    // Module 3 shifted the whole chunk (H and C move together). Here we give H and C
    // their own independent constant shifts, each chosen by cross-correlating that
    // group's distribution against the rest of the buffer. Cross-chunk gains for the
    // two groups factorise, so s_H* and s_C* can be found with two 256-element searches
    // rather than a 256x256 grid.  The within-chunk H-C relative shift (s_H - s_C) is
    // a free parameter; we measure global entropy and only apply if it improves.
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

    // --- Module 5: iterated per-chunk AND-PRNG cross-chunk xcorr ---
    // AND-PRNG (mean~64, small shifts) scored by cross-chunk histogram overlap.
    // Iterated: each pass may reshape histograms, enabling the next pass to find
    // more. Stops when no chunk improves or entropy doesn't decrease.
    // Two-phase search per pass: 64-byte proxy for phase 1, top 2000 full evaluation.
    double e5 = e4c;
    {
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
                seeds5[c] = best_s;
                dirs5[c]  = best_dir;
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
            if (e_try >= e5) break;
            memcpy(cur, tmp, size);
            e5 = e_try;
            passes++;
        }

        free(ss); free(seeds5); free(dirs5); free(lc); free(ls);

        if (passes > 0)
            printf("module 5:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                   "  [and-xcorr  passes:%d]\n",
                   e5, (e4c-e5)*size, (e0-e5)*size,
                   (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        else
            printf("module 5:  skipped  [and-xcorr]\n");
    }

    // --- Module 6: iterated cross-chunk re-alignment ---
    // Module 5's AND-PRNG reshaped histograms, making the earlier alignment stale.
    // Repeat the xcorr constant-shift pass until convergence: each pass may shift
    // chunks whose optimal shift changed because neighbouring chunks moved.
    double e6 = e5;
    {
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
            if (e_try >= e6) break;

            memcpy(cur, tmp, size);
            e6 = e_try;
            passes++;
        }

        free(shifts6);

        if (passes > 0)
            printf("module 6:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                   "  [chunk-align-3  passes:%d]\n",
                   e6, (e5-e6)*size, (e0-e6)*size,
                   (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        else
            printf("module 6:  skipped  [chunk-align-3]\n");
    }

    // --- Module 7: iterated per-chunk XOR-constant alignment ---
    // XOR with a constant permutes values via bit-flips, orthogonal to the additive
    // shifts in modules 3/4a/4c/6. Cross-chunk xcorr after XOR-x is
    //   sum_v local[v] * rest[v^x]  — O(256^2) per chunk, trivially reversible.
    double e7 = e6;
    {
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
            if (e_try >= e7) break;
            memcpy(cur, tmp, size);
            e7 = e_try;
            passes++;
        }

        free(xors7);

        if (passes > 0)
            printf("module 7:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                   "  [xor-align  passes:%d]\n",
                   e7, (e6-e7)*size, (e0-e7)*size,
                   (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        else
            printf("module 7:  skipped  [xor-align]\n");
    }

    // --- Module 8: iterated per-chunk XOR AND-PRNG ---
    // Fills the 2x2 design matrix:
    //   additive+fixed=M6  additive+PRNG=M5  XOR+fixed=M7  XOR+PRNG=M8
    // AND-PRNG (mean~64) flips ~2 bits per byte on average, leaving high-order bits
    // intact. This reshapes the histogram spread (not just the mode like M7).
    // Same two-phase seed search as M5; XOR has no +/- direction so we only need
    // one direction per seed.
    double e8 = e7;
    {
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
                    for (int i = 0; i < pfx; i++) {
                        u8 r = randNum_and();
                        lx[cur[off+i] ^ r]++;
                    }
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
            if (e_try >= e8) break;
            memcpy(cur, tmp, size);
            e8 = e_try;
            passes++;
        }

        free(ss); free(seeds8); free(lx);

        if (passes > 0)
            printf("module 8:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                   "  [xor-and-xcorr  passes:%d]\n",
                   e8, (e7-e8)*size, (e0-e8)*size,
                   (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        else
            printf("module 8:  skipped  [xor-and-xcorr]\n");
    }

    // --- Module 9: iterated per-chunk affine alignment ---
    // Covers the full affine group Aff(Z/256Z): v -> (a*v + b) mod 256, a odd, any b.
    // Module 6 is the a=1 subgroup; module 7 is XOR (over GF(2)^8, a different algebra).
    // Multipliers a=3,5,...,255 stretch/compress value spacing in modular arithmetic.
    //
    // Xcorr formula: sum_v local[v] * rest[(a*v+b)%256]
    //   = sum_u Ha[u] * rest[(u+b)%256]  where Ha[u] = local[a_inv*u % 256]
    // For each a: precompute Ha (O(256)), then scan 256 lags (O(256^2)).
    // Total: 128 * 256^2 = 8.4M per chunk per iteration.
    // Inverse: v -> a_inv*(v - b) mod 256, where a_inv*a = 1 mod 256.
    double e9 = e8;
    {
        clock_t t0 = clock();
        u8 *as9 = malloc(num_chunks);
        u8 *bs9 = malloc(num_chunks);
        int *ha  = malloc(256 * sizeof(int));
        int passes = 0;

        // precompute multiplicative inverses mod 256 for all 128 odd values
        u8 inv256[128];
        for (int k = 0; k < 128; k++) {
            int a = 2*k + 1;
            for (int inv = 1; inv < 256; inv += 2)
                if (((a * inv) & 0xFF) == 1) { inv256[k] = (u8)inv; break; }
        }

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

                // baseline xcorr: identity (a=1, b=0)
                int baseline = 0;
                for (int v = 0; v < 256; v++) baseline += local[v] * rest[v];
                int best_xcorr = baseline, best_ka = 0, best_b = 0;

                for (int ka = 1; ka < 128; ka++) {   // skip ka=0 (a=1, handled by M6)
                    int a_inv = inv256[ka];
                    for (int w = 0; w < 256; w++)
                        ha[w] = local[(a_inv * w) & 0xFF];
                    for (int b = 0; b < 256; b++) {
                        int xcorr = 0;
                        for (int w = 0; w < 256; w++)
                            xcorr += ha[w] * rest[(w + b) & 0xFF];
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
            if (e_try >= e9) break;
            memcpy(cur, tmp, size);
            e9 = e_try;
            passes++;
        }

        free(ha); free(as9); free(bs9);

        if (passes > 0)
            printf("module 9:  entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs"
                   "  [affine-align  passes:%d]\n",
                   e9, (e8-e9)*size, (e0-e9)*size,
                   (double)(clock()-t0)/CLOCKS_PER_SEC, passes);
        else
            printf("module 9:  skipped  [affine-align]\n");
    }

    {
        int counts[256] = {0}; long sum = 0;
        for (int i = 0; i < size; i++) { counts[cur[i]]++; sum += cur[i]; }
        int mode = 0;
        for (int i = 1; i < 256; i++) if (counts[i] > counts[mode]) mode = i;
        printf("after : entropy=%lf  total=%+.2f  avg=%.2f  mode=%d (x%d)\n",
               e9, (e0-e9)*size, (double)sum/size, mode, counts[mode]);

        // --- Structural diagnostics ---
        // delta distribution: how concentrated are adjacent-byte differences?
        int delta[256] = {0};
        for (int i = 0; i < size-1; i++) delta[(u8)(cur[i+1] - cur[i])]++;
        double h_delta = entropy((u8*)NULL, 0); // placeholder
        { // compute entropy of delta array manually
            int n = size-1;
            double s = 0.0;
            for (int v = 0; v < 256; v++) if (delta[v]) s += xlgx[delta[v]];
            h_delta = log2((double)n) - s / n;
        }
        int dmode = 0;
        for (int v = 1; v < 256; v++) if (delta[v] > delta[dmode]) dmode = v;
        printf("struct: delta_entropy=%.4f  delta_mode=%d (x%d)\n",
               h_delta, dmode, delta[dmode]);

        // per-bit statistics: fraction of 1s in each bit plane
        int bit1[8] = {0};
        for (int i = 0; i < size; i++)
            for (int b = 0; b < 8; b++)
                if (cur[i] & (1<<b)) bit1[b]++;
        printf("struct: bit_frac");
        for (int b = 7; b >= 0; b--) printf(" b%d=%.2f", b, (double)bit1[b]/size);
        printf("\n");

        // cross-chunk same-position XOR: how different are corresponding positions?
        int xdelta[256] = {0};
        for (int j = 0; j < CHUNK && j+CHUNK < size; j++)
            xdelta[(u8)(cur[j] ^ cur[j+CHUNK])]++;
        int xmode = 0;
        for (int v = 1; v < 256; v++) if (xdelta[v] > xdelta[xmode]) xmode = v;
        printf("struct: cross_chunk_xor_mode=%d (x%d of %d pairs)\n",
               xmode, xdelta[xmode], CHUNK);
    }
    free(cur); free(tmp); free(prngs); free(pats);
    return (e0 - e9) * size;
}

int main() {
    init_entropy_table();
    int size = 4096;
    u8 *data = malloc(size);
    CSV_XY *csv = csv_xy_open("output.csv", "seeds", "profit","null");
    for (int seeds = 0; seeds < 20; seeds++){
        srand(seeds);
        for (int i = 0; i < size; i++)
            data[i] = rand() % 256;

        print_stats("before", data, size);
        csv_xy_add(csv, seeds, RunModules(data, size),0);
    }

    csv_xy_close(csv);
    free(data);

    return 0;
}
