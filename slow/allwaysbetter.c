/* prng_inspect.c — BCrypt block, greedy-stack PRNG (K=1) with variable seed width.
 *
 * Each instruction encodes:
 *   [ bit_length(B) bits: B ]  [ B bits: seed ]  [ 2 bits: op ]
 *
 * op: 0=ADD (src+r), 1=XOR (src^r), 2=SUB (src-r), 3=RSUB (r-src)
 * All four are reversible. RSUB is self-inverse.
 *
 * overhead = B + bit_length(B) + 2
 * net      = (S_before - S_after) * N - overhead
 * value    = net / overhead
 *
 * Build:  gcc -O2 -o prng_inspect prng_inspect.c -lm -lbcrypt
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <bcrypt.h>

typedef unsigned char      u8;
typedef unsigned int       u32;
typedef unsigned long long u64;

#define N          4096
#define MAX_B        32
#define SCAN_SEEDS  256
#define FULL_SEEDS 65535
#define TOP_B        8

static inline u8 xs32_next(u32 *s) {
    u32 x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return (u8)(x ^ (x >> 8) ^ (x >> 16) ^ (x >> 24));
}

static double entropy_of(const u8 *buf, int n) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[buf[i]]++;
    double S = 0.0;
    for (int v = 0; v < 256; v++)
        if (f[v]) { double p = (double)f[v] / n; S -= p * log2(p); }
    return S;
}

/* overhead = B (seed) + bit_length(B) (width field) + 2 (op bits) */
static double overhead_for_B(int B) {
    int bits = 0, b = B;
    while (b > 1) { bits++; b >>= 1; }
    return (double)(B + bits + 1 + 2);   /* +2 for 2-bit op field */
}

/* Evaluate entropy for a given seed, trying all 4 ops in one pass.
 * op: 0=ADD, 1=XOR, 2=SUB, 3=RSUB
 * Returns the lowest entropy and writes the winning op to *out_op. */
static double eval_best_op(const u8 *buf, u32 seed, int *out_op) {
    u32 s = seed;
    int f[4][256];
    memset(f, 0, sizeof(f));
    for (int i = 0; i < N; i++) {
        u8 b = buf[i], r = xs32_next(&s);
        f[0][(u8)(b + r)]++;   /* ADD  */
        f[1][b ^ r]++;         /* XOR  */
        f[2][(u8)(b - r)]++;   /* SUB  */
        f[3][(u8)(r - b)]++;   /* RSUB */
    }
    double best_S = 1e18; *out_op = 0;
    for (int op = 0; op < 4; op++) {
        double S = 0.0;
        for (int v = 0; v < 256; v++)
            if (f[op][v]) { double p = (double)f[op][v] / N; S -= p * log2(p); }
        if (S < best_S) { best_S = S; *out_op = op; }
    }
    return best_S;
}

static const char *op_name(int op) {
    static const char *names[] = {"ADD", "XOR", "SUB", "RSUB"};
    return names[op & 3];
}

static void apply_prng_op(const u8 *src, u8 *dst, u32 seed, int op) {
    u32 s = seed;
    for (int i = 0; i < N; i++) {
        u8 r = xs32_next(&s);
        switch (op) {
            case 0: dst[i] = (u8)(src[i] + r); break;   /* ADD  */
            case 1: dst[i] =  src[i] ^ r;       break;   /* XOR  */
            case 2: dst[i] = (u8)(src[i] - r); break;   /* SUB  */
            case 3: dst[i] = (u8)(r - src[i]); break;   /* RSUB */
        }
    }
}

static int max_seeds_for_B(int B) {
    if (B >= 32) return FULL_SEEDS;
    u64 all = ((u64)1 << B) - 1;
    return (int)(all < FULL_SEEDS ? all : FULL_SEEDS);
}

static u32 scan_seed(int B, int i) {
    if (B <= 8) return (u32)(i + 1);
    u64 mask = (B < 32) ? ((u64)1 << B) - 1 : 0xFFFFFFFFULL;
    u64 v = (u64)(i + 1) * 6364136223846793005ULL + 1442695040888963407ULL;
    return (u32)((v & mask) | 1);
}

static u32 full_seed(int B, int i) {
    if (B <= 16) return (u32)(i + 1);
    u64 mask = (B < 32) ? ((u64)1 << B) - 1 : 0xFFFFFFFFULL;
    u64 v = (u64)(i + 1) * 6364136223846793005ULL + 1442695040888963407ULL;
    return (u32)((v & mask) | 1);
}

static double best_instruction(const u8 *buf, double S_cur,
                               int *out_B, u32 *out_seed, int *out_op, double *out_S) {
    double scan_val[MAX_B + 1];
    int    order[MAX_B];

    for (int B = 1; B <= MAX_B; B++) {
        int n = (B <= 8) ? max_seeds_for_B(B) : SCAN_SEEDS;
        double bv = -1e18, bS = 1e18; u32 bs = 1; int bop = 0;
        for (int i = 0; i < n; i++) {
            u32 seed = scan_seed(B, i);
            int op; double S = eval_best_op(buf, seed, &op);
            double oh = overhead_for_B(B);
            double v  = ((S_cur - S) * N - oh) / oh;
            if (v > bv) { bv = v; bS = S; bs = seed; bop = op; }
        }
        scan_val[B] = bv;
        order[B - 1] = B;
    }

    for (int i = 0; i < TOP_B; i++)
        for (int j = i + 1; j < MAX_B; j++)
            if (scan_val[order[j]] > scan_val[order[i]]) {
                int t = order[i]; order[i] = order[j]; order[j] = t;
            }

    double best_val = -1e18;
    *out_B = 1; *out_seed = 1; *out_op = 0; *out_S = S_cur;

    for (int ci = 0; ci < TOP_B; ci++) {
        int B = order[ci];
        int n = max_seeds_for_B(B);
        double bv = -1e18, bS = 1e18; u32 bs = 1; int bop = 0;
        for (int i = 0; i < n; i++) {
            u32 seed = full_seed(B, i);
            int op; double S = eval_best_op(buf, seed, &op);
            double oh = overhead_for_B(B);
            double v  = ((S_cur - S) * N - oh) / oh;
            if (v > bv) { bv = v; bS = S; bs = seed; bop = op; }
        }
        if (bv > best_val) {
            best_val = bv; *out_B = B; *out_seed = bs; *out_op = bop; *out_S = bS;
        }
    }

    return best_val;
}

int main(void) {
    static u8 work[N], tmp[N];
    BCryptGenRandom(NULL, work, N, BCRYPT_USE_SYSTEM_PREFERRED_RNG);

    double S_initial = entropy_of(work, N);
    printf("initial entropy: %.6f bps  (%d bytes)\n\n", S_initial, N);
    printf("  rnd   B  oh   op   seed          H_before    H_after     net    value\n");
    printf("  ---  --  --   --   ----------    --------    -------     ---    -----\n");

    int    round     = 0;
    double total_oh  = 0.0;

    for (;;) {
        double S_cur = entropy_of(work, N);
        int best_B, best_op; u32 best_seed; double best_S;
        double best_val = best_instruction(work, S_cur, &best_B, &best_seed, &best_op, &best_S);

        double oh  = overhead_for_B(best_B);
        double net = (S_cur - best_S) * N - oh;

        if (net <= 0.0) {
            printf("  (best net=%.1f bits -- stopping)\n", net);
            break;
        }

        round++;
        printf("  %3d  %2d  %2.0f   %-4s  %10u    %.6f  %.6f   %+.1f   %+.4f\n",
               round, best_B, oh, op_name(best_op), best_seed,
               S_cur, best_S, net, best_val);

        total_oh += oh;
        apply_prng_op(work, tmp, best_seed, best_op);
        memcpy(work, tmp, N);
    }

    double S_final   = entropy_of(work, N);
    double reduction = (S_initial - S_final) * N;
    printf("\n%d instructions | total overhead: %.0f bits | entropy reduction: %.1f bits | net: %.1f bits\n",
           round, total_oh, reduction, reduction - total_oh);
    printf("H: %.6f -> %.6f bps\n", S_initial, S_final);

    return 0;
}
