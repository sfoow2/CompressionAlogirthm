#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <math.h>
#include <string.h>
#include <time.h>

typedef uint8_t u8;

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
    for (int i = 0; i < size; i++)
        counts[data[i]]++;
    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / size;
        entropy -= p * log2(p);
    }
    return entropy;
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

void IterativeSearch(u8 *data, int size, int max_iters) {
    const int CHUNK = 256;
    int num_chunks = (size + CHUNK - 1) / CHUNK;

    u8  *cur   = malloc(size);
    u8  *tmp   = malloc(size);
    int *perm  = malloc(size * sizeof(int));
    int *prngs = malloc(num_chunks * sizeof(int));
    memcpy(cur, data, size);

    double start_e = entropy(cur, size);
    double e = start_e;
    printf("iter  0: entropy=%lf  (%d chunks x %d bytes)\n", e, num_chunks, CHUNK);

    for (int iter = 1; iter <= max_iters; iter++) {
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

        double best_e = entropy(tmp, size);
        if (best_e >= e) break;

        uint64_t shuf_seed = (uint64_t)iter * 6364136223846793005ULL;
        memcpy(cur, tmp, size);
        do_shuffle_fast(cur, perm, tmp, size, shuf_seed);

        double profit = (e       - best_e) * size;
        double total  = (start_e - best_e) * size;
        printf("iter %2d: entropy=%lf  profit=%+.2f  total=%+.2f  time=%.1fs\n",
               iter, best_e, profit, total,
               (double)(clock() - t0) / CLOCKS_PER_SEC);
        e = best_e;
    }

    free(cur); free(tmp); free(perm); free(prngs);
}

void main() {
    srand(14);
    int size = 4096;
    u8 *data = malloc(size);
    for (int i = 0; i < size; i++)
        data[i] = rand() % 256;

    double start = entropy(data, size);
    printf("Starting = %lf\n\n", start);

    IterativeSearch(data, size, 10);

    free(data);
}
