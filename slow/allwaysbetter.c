/* prng_inspect.c — BCrypt block, greedy-stack PRNG with:
 *   1. Variable seed width B (1..32), self-delimiting encoding
 *   2. K interleaved streams (K=1,2,4,8)
 *   3. 8 reversible ops (3-bit op field): ADD XOR SUB RSUB ROTL ROTR XNOR NADD
 *   4. Hill-climbing for large-B seeds (B>16)
 *
 * Instruction encoding (self-delimiting):
 *   [ bit_length(K) bits ]  [ bit_length(B) bits ]  [ B bits: seed ]  [ 3 bits: op ]
 *
 * overhead = bit_length(K) + bit_length(B) + B + 3
 * value    = net / overhead,   net = (S_before - S_after)*N - overhead
 *
 * Build: gcc -O2 -o prng_inspect prng_inspect.c -lm -lbcrypt
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <bcrypt.h>

typedef unsigned char      u8;
typedef unsigned int       u32;
typedef unsigned long long u64;

#define N            4096
#define MAX_B          32
#define N_OPS           8   /* 3-bit op field */
#define SCAN_SEEDS    256
#define FULL_SEEDS  65535
#define TOP_CANDS       8   /* top (B,K) pairs for full search */

static const int  K_VALS[] = {1, 2, 4, 8};
#define N_K 4

static const char *OP_NAME[N_OPS] = {
    "ADD", "XOR", "SUB", "RSUB", "ROTL", "ROTR", "XNOR", "NADD"
};

/* op semantics:
 *  ADD  = (src + r) mod 256
 *  XOR  = src ^ r
 *  SUB  = (src - r) mod 256
 *  RSUB = (r - src) mod 256   (self-inverse)
 *  ROTL = rotate src left  by (r & 7) bits
 *  ROTR = rotate src right by (r & 7) bits
 *  XNOR = ~(src ^ r)           (self-inverse)
 *  NADD = (~src + r) mod 256   (self-inverse)
 */

static inline u8 xs32_step(u32 *s) {
    u32 x = *s; x ^= x<<13; x ^= x>>17; x ^= x<<5; *s = x;
    return (u8)(x ^ (x>>8) ^ (x>>16) ^ (x>>24));
}

static double entropy_of(const u8 *buf, int n) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[buf[i]]++;
    double S = 0.0;
    for (int v = 0; v < 256; v++)
        if (f[v]) { double p = (double)f[v]/n; S -= p*log2(p); }
    return S;
}

static int bit_length(int n) { int b=0; while(n>0){b++;n>>=1;} return b; }

static double overhead_bits(int B, int K) {
    return (double)(bit_length(K) + bit_length(B) + B + 3);
}

/* Evaluate all 8 ops for given (K, seed) in one pass. Returns best entropy. */
static double eval_best(const u8 *buf, int K, u32 seed, int *out_op) {
    u32 st[8]; st[0] = seed;
    for (int k = 1; k < K; k++) { st[k] = st[k-1]; xs32_step(&st[k]); }

    int f[N_OPS][256]; memset(f, 0, sizeof f);
    u32 cs[8]; memcpy(cs, st, (size_t)K * sizeof(u32));

    for (int i = 0; i < N; i++) {
        u8 b = buf[i], r = xs32_step(&cs[i % K]);
        int rot = r & 7, rrot = (8 - rot) & 7;
        f[0][(u8)(b + r)]++;
        f[1][b ^ r]++;
        f[2][(u8)(b - r)]++;
        f[3][(u8)(r - b)]++;
        f[4][rot ? (u8)((b<<rot)|(b>>rrot)) : b]++;   /* ROTL */
        f[5][rot ? (u8)((b>>rot)|(b<<rrot)) : b]++;   /* ROTR */
        f[6][(u8)(~(b ^ r))]++;                        /* XNOR */
        f[7][(u8)(~b + r)]++;                          /* NADD */
    }

    double best_S = 1e18; *out_op = 0;
    for (int op = 0; op < N_OPS; op++) {
        double S = 0.0;
        for (int v = 0; v < 256; v++)
            if (f[op][v]) { double p = (double)f[op][v]/N; S -= p*log2(p); }
        if (S < best_S) { best_S = S; *out_op = op; }
    }
    return best_S;
}

/* Steepest-ascent hill-climb in seed space for given (B, K).
 * Each iteration tries all B bit-flips, keeps the best, repeats. */
static double hill_climb(const u8 *buf, double S_cur, int B, int K,
                         u32 *io_seed, int *io_op) {
    double oh = overhead_bits(B, K);
    u32 seed = *io_seed;
    int  op;  double S = eval_best(buf, K, seed, &op);
    double val = ((S_cur - S)*N - oh) / oh;

    int improved = 1;
    while (improved) {
        improved = 0;
        int best_bit = -1; double best_val = val, best_S = S; int best_op = op;
        for (int bit = 0; bit < B; bit++) {
            u32 ts = seed ^ (1u << bit); if (!ts) continue;
            int to; double tS = eval_best(buf, K, ts, &to);
            double tv = ((S_cur - tS)*N - oh) / oh;
            if (tv > best_val) { best_val=tv; best_S=tS; best_bit=bit; best_op=to; }
        }
        if (best_bit >= 0) {
            seed ^= (1u << best_bit); val=best_val; S=best_S; op=best_op; improved=1;
        }
    }
    *io_seed = seed; *io_op = op;
    return S;
}

static u32 scan_seed(int B, int i) {
    if (B <= 8) return (u32)(i + 1);
    u64 mask = (B < 32) ? ((u64)1<<B)-1 : 0xFFFFFFFFULL;
    u64 v = (u64)(i+1)*6364136223846793005ULL + 1442695040888963407ULL;
    return (u32)((v & mask) | 1);
}

static u32 full_seed(int B, int i) {
    if (B <= 16) return (u32)(i + 1);
    u64 mask = (B < 32) ? ((u64)1<<B)-1 : 0xFFFFFFFFULL;
    u64 v = (u64)(i+1)*6364136223846793005ULL + 1442695040888963407ULL;
    return (u32)((v & mask) | 1);
}

static int max_seeds(int B) {
    if (B >= 32) return FULL_SEEDS;
    u64 all = ((u64)1<<B) - 1;
    return (int)(all < FULL_SEEDS ? all : FULL_SEEDS);
}

typedef struct { int B, K; double val, S; u32 seed; int op; } Cand;

static double best_instruction(const u8 *buf, double S_cur,
                               int *oB, int *oK, u32 *oseed, int *oop, double *oS) {
    /* Phase 1: quick scan — K=1 only for speed, all B=1..MAX_B */
    Cand phase1[MAX_B];
    for (int B = 1; B <= MAX_B; B++) {
        int n = (B <= 8) ? max_seeds(B) : SCAN_SEEDS;
        double oh = overhead_bits(B, 1);
        double bv=-1e18, bS=1e18; u32 bs=1; int bop=0;
        for (int i = 0; i < n; i++) {
            u32 seed = scan_seed(B, i);
            int op; double S = eval_best(buf, 1, seed, &op);
            double v = ((S_cur-S)*N - oh)/oh;
            if (v > bv) { bv=v; bS=S; bs=seed; bop=op; }
        }
        phase1[B-1] = (Cand){B, 1, bv, bS, bs, bop};
    }

    /* Sort phase1 to pick top TOP_CANDS B values */
    for (int i = 0; i < TOP_CANDS; i++)
        for (int j = i+1; j < MAX_B; j++)
            if (phase1[j].val > phase1[i].val) {
                Cand t=phase1[i]; phase1[i]=phase1[j]; phase1[j]=t;
            }

    /* Phase 2: full search + all K values + hill-climb for top B candidates */
    double best_val = -1e18;
    *oB=1; *oK=1; *oseed=1; *oop=0; *oS=S_cur;

    for (int ci = 0; ci < TOP_CANDS; ci++) {
        int B = phase1[ci].B;
        int nf = max_seeds(B);

        for (int ki = 0; ki < N_K; ki++) {
            int K = K_VALS[ki];
            double oh = overhead_bits(B, K);
            double bv=-1e18, bS=1e18; u32 bs=1; int bop=0;

            for (int i = 0; i < nf; i++) {
                u32 seed = full_seed(B, i);
                int op; double S = eval_best(buf, K, seed, &op);
                double v = ((S_cur-S)*N - oh)/oh;
                if (v > bv) { bv=v; bS=S; bs=seed; bop=op; }
            }

            /* Hill-climb for large B where search is incomplete */
            if (B > 16) bS = hill_climb(buf, S_cur, B, K, &bs, &bop);
            bv = ((S_cur-bS)*N - oh)/oh;

            if (bv > best_val) {
                best_val=bv; *oB=B; *oK=K; *oseed=bs; *oop=bop; *oS=bS;
            }
        }
    }
    return best_val;
}

static void apply_op(const u8 *src, u8 *dst, int K, u32 seed, int op) {
    u32 st[8]; st[0] = seed;
    for (int k = 1; k < K; k++) { st[k] = st[k-1]; xs32_step(&st[k]); }
    u32 cs[8]; memcpy(cs, st, (size_t)K * sizeof(u32));
    for (int i = 0; i < N; i++) {
        u8 b = src[i], r = xs32_step(&cs[i % K]);
        int rot = r & 7, rrot = (8 - rot) & 7;
        switch (op) {
            case 0: dst[i] = (u8)(b+r);  break;
            case 1: dst[i] = b^r;         break;
            case 2: dst[i] = (u8)(b-r);  break;
            case 3: dst[i] = (u8)(r-b);  break;
            case 4: dst[i] = rot ? (u8)((b<<rot)|(b>>rrot)) : b; break;
            case 5: dst[i] = rot ? (u8)((b>>rot)|(b<<rrot)) : b; break;
            case 6: dst[i] = (u8)(~(b^r)); break;
            case 7: dst[i] = (u8)(~b+r);  break;
        }
    }
}

int main(void) {
    static u8 orig[N], work[N], tmp[N];
    BCryptGenRandom(NULL, orig, N, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
    memcpy(work, orig, N);

    double S_initial = entropy_of(work, N);
    printf("initial entropy: %.6f bps  (%d bytes)\n\n", S_initial, N);
    printf("  rnd  K   B   oh   op     seed         H_before    H_after     net    value\n");
    printf("  ---  -  --   --   ----   ----------   --------    -------     ---    -----\n");

    int    round    = 0;
    double total_oh = 0.0;

    for (;;) {
        double S_cur = entropy_of(work, N);
        int bB, bK, bop; u32 bseed; double bS;
        double bval = best_instruction(work, S_cur, &bB, &bK, &bseed, &bop, &bS);

        double oh  = overhead_bits(bB, bK);
        double net = (S_cur - bS)*N - oh;
        if (net <= 0.0) {
            printf("  (best net=%.1f bits -- stopping)\n", net);
            break;
        }

        round++;
        printf("  %3d  %d  %2d   %2.0f   %-4s   %10u   %.6f   %.6f    %+.1f   %+.4f\n",
               round, bK, bB, oh, OP_NAME[bop], bseed,
               S_cur, bS, net, bval);

        total_oh += oh;
        apply_op(work, tmp, bK, bseed, bop);
        memcpy(work, tmp, N);
    }

    double S_final   = entropy_of(work, N);
    double reduction = (S_initial - S_final)*N;
    printf("\n%d instructions | OH: %.0f bits | reduction: %.1f bits | net: %.1f bits\n",
           round, total_oh, reduction, reduction - total_oh);
    printf("H: %.6f -> %.6f bps\n", S_initial, S_final);

    /* Write both blocks for compressor3 comparison */
    FILE *f;
    f = fopen("original.bin", "wb");  fwrite(orig, 1, N, f); fclose(f);
    f = fopen("transformed.bin","wb"); fwrite(work, 1, N, f); fclose(f);
    printf("wrote original.bin and transformed.bin\n");
    return 0;
}
