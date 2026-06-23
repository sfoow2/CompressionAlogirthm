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
#define PATIENCE       64   /* quit seed scan after this many non-improving seeds */

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

/* log2 lookup table: g_xl2x[c] = c * log2(c), g_xl2x[0] = 0.
 * Entropy: S = log2(N) - (1/N) * sum_v(g_xl2x[freq[v]])  */
static double g_xl2x[N + 1];
static double g_log2N;

static void init_lut(void) {
    g_log2N = log2((double)N);
    g_xl2x[0] = 0.0;
    for (int c = 1; c <= N; c++) g_xl2x[c] = c * log2((double)c);
}

static double entropy_of(const u8 *buf, int n) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[buf[i]]++;
    double xsum = 0.0;
    for (int v = 0; v < 256; v++) xsum += g_xl2x[f[v]];
    return g_log2N - xsum / n;
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
        int no_imp1 = 0;
        for (int i = 0; i < n; i++) {
            u32 seed = scan_seed(B, i);
            int op; double S = eval_best(buf, 1, seed, &op);
            double v = ((S_cur-S)*N - oh)/oh;
            if (v > bv) { bv=v; bS=S; bs=seed; bop=op; no_imp1=0; }
            else if (++no_imp1 >= PATIENCE) break;
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
            int no_imp2 = 0;
            for (int i = 0; i < nf; i++) {
                u32 seed = full_seed(B, i);
                int op; double S = eval_best(buf, K, seed, &op);
                double v = ((S_cur-S)*N - oh)/oh;
                if (v > bv) { bv=v; bS=S; bs=seed; bop=op; no_imp2=0; }
                else if (++no_imp2 >= PATIENCE) break;
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

/* =====================================================================
 * Split layer: 8 fixed 4-bit cycling patterns, 8-bit amp
 * Byte i is in group 1 if (pattern >> (i%4)) & 1, else group 0.
 * amp = (mode_group0 - mode_group1) mod 256
 * Apply: group-1 bytes += amp    Invert: group-1 bytes -= amp
 * Overhead: 3 bits (pattern index) + 8 bits (amp) = 11 bits constant.
 * ===================================================================== */

#define SPLIT_MIN_B   2
#define SPLIT_MAX_B  32
#define SPLIT_SAMPLE 256*2   /* patterns tried when 2^B > 2^16 */

static int s_bitlen(int n) { int b = 0; while (n > 0) { b++; n >>= 1; } return b; }
/* overhead = 5 (B field, covers 2..32) + B (pattern bits) + 2 (op) + 8 (amp) */
static double split_oh(int B) { return (double)(5 + B + 2 + 8); }

/* split layer uses ops 0-3 (ADD/XOR/NADD/ROTL) — 2 op bits */

typedef struct { int B; u32 pat; u8 amp; int op; } SplitInstr;

static inline u8 rotl8(u8 b, int r) { r &= 7; return r ? (u8)((b<<r)|(b>>(8-r))) : b; }
static inline u8 rotr8(u8 b, int r) { r &= 7; return r ? (u8)((b>>r)|(b<<(8-r))) : b; }

/* 8-op set (ops 0-3 used by split layer, 0-7 used by strided layer):
 * 0=ADD 1=XOR 2=NADD 3=ROTL 4=SUB 5=XNOR 6=RSUB 7=ROTR */
static const char *OP8_NAME[8] = {"ADD","XOR","NADD","ROTL","SUB","XNOR","RSUB","ROTR"};

static inline u8 byte_op(u8 b, u8 amp, int op) {
    switch (op) {
        case 0: return (u8)(b + amp);           /* ADD  */
        case 1: return (u8)(b ^ amp);           /* XOR  */
        case 2: return (u8)(~b + amp);          /* NADD (self-inverse) */
        case 3: return rotl8(b, amp);           /* ROTL */
        case 4: return (u8)(b - amp);           /* SUB  */
        case 5: return (u8)(~(b ^ amp));        /* XNOR (self-inverse) */
        case 6: return (u8)(amp - b);           /* RSUB (self-inverse) */
        default:return rotr8(b, amp);           /* ROTR */
    }
}

static inline u8 byte_op_inv(u8 b, u8 amp, int op) {
    switch (op) {
        case 0: return (u8)(b - amp);           /* ADD^-1 = SUB */
        case 1: return (u8)(b ^ amp);           /* XOR self-inverse */
        case 2: return (u8)(~b + amp);          /* NADD self-inverse */
        case 3: return rotr8(b, amp);           /* ROTL^-1 = ROTR */
        case 4: return (u8)(b + amp);           /* SUB^-1 = ADD */
        case 5: return (u8)(~(b ^ amp));        /* XNOR self-inverse */
        case 6: return (u8)(amp - b);           /* RSUB self-inverse */
        default:return rotl8(b, amp);           /* ROTR^-1 = ROTL */
    }
}

static double split_best(const u8 *buf, double S_cur, SplitInstr *out) {
    double best_val = -1e18;
    out->B = SPLIT_MIN_B; out->pat = 0; out->amp = 0; out->op = 0;

    for (int B = SPLIT_MIN_B; B <= SPLIT_MAX_B; B++) {
        double oh = split_oh(B);
        u32 mask = (B < 32) ? ((1u << B) - 1u) : 0xFFFFFFFFu;

        /* precompute B meta-group histograms: byte i belongs to group i%B */
        int fmeta[SPLIT_MAX_B][256];
        memset(fmeta, 0, (size_t)B * 256 * sizeof(int));
        for (int i = 0; i < N; i++) fmeta[i % B][buf[i]]++;

        /* B<=8: exhaustive over all 2^B patterns; B>8: 256 random samples */
        long n_pats = (B <= 8) ? (1L << B) : SPLIT_SAMPLE;

        for (long pi = 0; pi < n_pats; pi++) {
            u32 pat;
            if (B <= 8) {
                pat = (u32)pi;
            } else {
                u64 v = (u64)(pi + 1) * 6364136223846793005ULL + 1442695040888963407ULL;
                pat = (u32)(v & mask);
            }

            /* combine meta-group histograms into f0/f1 based on pattern bits */
            int f0[256] = {0}, f1[256] = {0};
            for (int b = 0; b < B; b++) {
                int *dst = ((pat >> b) & 1) ? f1 : f0;
                for (int v = 0; v < 256; v++) dst[v] += fmeta[b][v];
            }

            /* exhaustive search over all 256 amps × 4 ops
             * rf[w] = f0[w] + f1[op_inv(w, amp)]  (avoids per-byte scatter) */
            for (int amp = 0; amp < 256; amp++) {
                for (int op = 0; op < 4; op++) {
                    double xsum = 0.0;
                    for (int w = 0; w < 256; w++)
                        xsum += g_xl2x[f0[w] + f1[byte_op_inv((u8)w, (u8)amp, op)]];
                    double S_new = g_log2N - xsum * (1.0 / N);
                    double net = (S_cur - S_new) * N - oh;
                    double val = net / oh;
                    if (val > best_val) {
                        best_val = val; out->B = B; out->pat = pat;
                        out->amp = (u8)amp; out->op = op;
                    }
                }
            }
        }
    }
    return best_val;
}

static void split_apply(u8 *buf, SplitInstr t) {
    for (int i = 0; i < N; i++)
        if ((t.pat >> (i % t.B)) & 1)
            buf[i] = byte_op(buf[i], t.amp, t.op);
}

static void split_invert(u8 *buf, SplitInstr t) {
    for (int i = 0; i < N; i++)
        if ((t.pat >> (i % t.B)) & 1)
            buf[i] = byte_op_inv(buf[i], t.amp, t.op);
}

typedef struct { int rounds; double S_out, total_oh; } SplitResult;

static SplitResult run_split_layer(u8 *work, double S_in, int verbose) {
    double total_oh = 0.0, S_cur = S_in;
    int round = 0;
    if (verbose) {
        printf("\n--- split layer ---\n");
        printf("  rnd   B  pat         op   amp  H_before    H_after     net     OH    value\n");
        printf("  ---  --  ----------  ---  ---  --------    -------     ---     --    -----\n");
    }
    for (;;) {
        SplitInstr best;
        double val = split_best(work, S_cur, &best);
        if (val <= 0.0) {
            if (verbose) printf("  (best value=%.4f -- stopping)\n", val);
            break;
        }
        double oh = split_oh(best.B);
        split_apply(work, best);
        double S_new = entropy_of(work, N);
        double net = (S_cur - S_new) * N - oh;
        if (verbose)
            printf("  %3d  %2d  0x%08X  %-4s  %3u  %.6f   %.6f    %+.4f   %.0f   %+.4f\n",
                   round + 1, best.B, best.pat, OP8_NAME[best.op], best.amp,
                   S_cur, S_new, net, oh, val);
        S_cur = S_new;
        total_oh += split_oh(best.B);
        round++;
    }
    return (SplitResult){ round, S_cur, total_oh };
}

/* =====================================================================
 * Strided-add layer: every stride-th byte (phase=0), op(byte, amp)
 * stride width W: 1..6 bits, covering strides 1..2^W
 * overhead = 3 (W field, W in 1..6) + W (stride value) + 3 (op) + 8 (amp)
 *          = W + 14  (ranges 15..20)
 * ===================================================================== */

#define STRIDED_MAX 64   /* max stride = 2^6 */

/* minimum bits needed to store stride s */
static int stride_bits(int s) { int w = 1; while ((1 << w) < s) w++; return w; }
static double strided_oh(int s) { return (double)(stride_bits(s) + 3 + 8); }

typedef struct { int stride; u8 amp; int op; } StridedInstr;

static double strided_best(const u8 *buf, double S_cur, StridedInstr *out) {
    double best_val = -1e18;
    out->stride = 1; out->amp = 0; out->op = 0;

    /* precompute full histogram once */
    int ftotal[256] = {0};
    for (int i = 0; i < N; i++) ftotal[buf[i]]++;

    for (int s = 1; s <= STRIDED_MAX; s++) {
        /* histogram of selected bytes (phase=0: positions 0,s,2s,...) */
        int fsel[256] = {0};
        for (int i = 0; i < N; i += s) fsel[buf[i]]++;

        /* unselected = total - selected */
        int funsel[256];
        for (int v = 0; v < 256; v++) funsel[v] = ftotal[v] - fsel[v];

        double oh = strided_oh(s);
        /* merged[v] = funsel[v] + fsel[byte_op_inv(v, amp, op)] */
        for (int amp = 0; amp < 256; amp++) {
            for (int op = 0; op < 8; op++) {
                double xsum = 0.0;
                for (int v = 0; v < 256; v++)
                    xsum += g_xl2x[funsel[v] + fsel[byte_op_inv((u8)v, (u8)amp, op)]];
                double S_new = g_log2N - xsum * (1.0 / N);
                double net = (S_cur - S_new) * N - oh;
                double val = net / oh;
                if (val > best_val) {
                    best_val = val; out->stride = s; out->amp = (u8)amp; out->op = op;
                }
            }
        }
    }
    return best_val;
}

static void strided_apply(u8 *buf, StridedInstr t) {
    for (int i = 0; i < N; i += t.stride)
        buf[i] = byte_op(buf[i], t.amp, t.op);
}

static void strided_invert(u8 *buf, StridedInstr t) {
    for (int i = 0; i < N; i += t.stride)
        buf[i] = byte_op_inv(buf[i], t.amp, t.op);
}

typedef struct { int rounds; double S_out, total_oh; } StridedResult;

static StridedResult run_strided_layer(u8 *work, double S_in, int verbose) {
    double total_oh = 0.0, S_cur = S_in;
    int round = 0;
    if (verbose) {
        printf("\n--- strided layer (W+14 OH/instr) ---\n");
        printf("  rnd  stride  op    amp  H_before    H_after     net     OH    value\n");
        printf("  ---  ------  ----  ---  --------    -------     ---     --    -----\n");
    }
    for (;;) {
        StridedInstr best;
        double val = strided_best(work, S_cur, &best);
        if (val <= 0.0) {
            if (verbose) printf("  (best value=%.4f -- stopping)\n", val);
            break;
        }
        strided_apply(work, best);
        double S_new = entropy_of(work, N);
        double oh  = strided_oh(best.stride);
        double net = (S_cur - S_new) * N - oh;
        if (verbose)
            printf("  %3d  %6d  %-4s  %3u  %.6f   %.6f    %+.4f   %.0f   %+.4f\n",
                   round + 1, best.stride, OP8_NAME[best.op], best.amp,
                   S_cur, S_new, net, oh, val);
        S_cur = S_new;
        total_oh += oh;
        round++;
    }
    return (StridedResult){ round, S_cur, total_oh };
}

typedef struct {
    int prng_rounds, split_rounds, strided_rounds;
    double S_init, S_after_prng, S_after_split, S_final;
    double prng_oh, split_oh, strided_oh;
} BlockResult;

static BlockResult run_block(const u8 *input, u8 *transformed, int verbose) {
    static u8 work[N], tmp[N];
    memcpy(work, input, N);

    double S_initial = entropy_of(work, N);
    if (verbose) {
        printf("initial entropy: %.6f bps  (%d bytes)\n\n", S_initial, N);
        printf("--- prng layer ---\n");
        printf("  rnd  K   B   oh   op     seed         H_before    H_after     net    value\n");
        printf("  ---  -  --   --   ----   ----------   --------    -------     ---    -----\n");
    }

    int    prng_rounds = 0;
    double prng_oh     = 0.0;

    for (;;) {
        double S_cur = entropy_of(work, N);
        int bB, bK, bop; u32 bseed; double bS;
        double bval = best_instruction(work, S_cur, &bB, &bK, &bseed, &bop, &bS);

        double oh  = overhead_bits(bB, bK);
        double net = (S_cur - bS)*N - oh;
        if (net <= 0.0) {
            if (verbose) printf("  (best net=%.1f bits -- stopping)\n", net);
            break;
        }

        prng_rounds++;
        if (verbose)
            printf("  %3d  %d  %2d   %2.0f   %-4s   %10u   %.6f   %.6f    %+.4f   %+.4f\n",
                   prng_rounds, bK, bB, oh, OP_NAME[bop], bseed,
                   S_cur, bS, net, bval);

        prng_oh += oh;
        apply_op(work, tmp, bK, bseed, bop);
        memcpy(work, tmp, N);
    }

    double S_after_prng = entropy_of(work, N);
    SplitResult    sr = run_split_layer(work, S_after_prng, verbose);
    StridedResult  tr = run_strided_layer(work, sr.S_out, verbose);

    if (transformed) memcpy(transformed, work, N);
    return (BlockResult){ prng_rounds, sr.rounds, tr.rounds,
                          S_initial, S_after_prng, sr.S_out, tr.S_out,
                          prng_oh, sr.total_oh, tr.total_oh };
}

int main(int argc, char **argv) {
    init_lut();
    /* gen mode: prng_inspect gen <N> [file]  — write N random 4096-byte blocks */
    if (argc >= 2 && strcmp(argv[1], "gen") == 0) {
        int NB = (argc >= 3) ? atoi(argv[2]) : 20;
        const char *out = (argc >= 4) ? argv[3] : "test_blocks.bin";
        FILE *f = fopen(out, "wb");
        if (!f) { perror(out); return 1; }
        for (int b = 0; b < NB; b++) {
            u8 blk[N];
            BCryptGenRandom(NULL, blk, N, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
            fwrite(blk, 1, N, f);
        }
        fclose(f);
        printf("generated %d blocks (%d bytes) -> %s\n", NB, NB * N, out);
        return 0;
    }

    /* load data: file (multi-block ok) or fresh BCrypt random */
    int NB = 1;
    u8 *all = NULL;

    if (argc >= 2) {
        FILE *f = fopen(argv[1], "rb");
        if (!f) { perror(argv[1]); return 1; }
        fseek(f, 0, SEEK_END);
        long sz = ftell(f);
        rewind(f);
        NB = (int)(sz / N);
        if (NB < 1) { fprintf(stderr, "file too small (need %d bytes)\n", N); fclose(f); return 1; }
        all = malloc((size_t)NB * N);
        if (!all) { fprintf(stderr, "oom\n"); return 1; }
        fread(all, 1, (size_t)NB * N, f);
        fclose(f);
        printf("input: %s  (%d block%s)\n", argv[1], NB, NB > 1 ? "s" : "");
    } else {
        all = malloc(N);
        if (!all) { fprintf(stderr, "oom\n"); return 1; }
        BCryptGenRandom(NULL, all, N, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
        printf("input: BCrypt random\n");
    }

    int verbose = (NB == 1);
    double total_net = 0.0;
    u8 transformed[N];

    for (int b = 0; b < NB; b++) {
        BlockResult r = run_block(all + (size_t)b * N, verbose ? transformed : NULL, verbose);
        double prng_red    = (r.S_init        - r.S_after_prng)  * N;
        double split_red   = (r.S_after_prng  - r.S_after_split) * N;
        double strided_red = (r.S_after_split - r.S_final)       * N;
        double total_red   = prng_red + split_red + strided_red;
        double net         = total_red - r.prng_oh - r.split_oh - r.strided_oh;
        total_net += net;

        if (verbose) {
            printf("\nPRNG:    %2d instrs | OH: %5.0f bits | reduction: %6.1f bits | net: %+.1f\n",
                   r.prng_rounds,    r.prng_oh,    prng_red,    prng_red    - r.prng_oh);
            printf("SPLIT:   %2d instrs | OH: %5.0f bits | reduction: %6.1f bits | net: %+.1f\n",
                   r.split_rounds,   r.split_oh,   split_red,   split_red   - r.split_oh);
            printf("STRIDED: %2d instrs | OH: %5.0f bits | reduction: %6.1f bits | net: %+.1f\n",
                   r.strided_rounds, r.strided_oh, strided_red, strided_red - r.strided_oh);
            printf("TOTAL:              | OH: %5.0f bits | reduction: %6.1f bits | net: %+.1f\n",
                   r.prng_oh + r.split_oh + r.strided_oh, total_red, net);
            printf("H: %.6f -> %.6f -> %.6f -> %.6f bps\n",
                   r.S_init, r.S_after_prng, r.S_after_split, r.S_final);
            FILE *f;
            f = fopen("original.bin",    "wb"); fwrite(all, 1, N, f); fclose(f);
            f = fopen("transformed.bin", "wb"); fwrite(transformed, 1, N, f); fclose(f);
            printf("wrote original.bin and transformed.bin\n");
        } else {
            printf("  block %3d: %.6f->%.6f->%.6f->%.6f  p=%d s=%d t=%d  net=%+7.1f\n",
                   b, r.S_init, r.S_after_prng, r.S_after_split, r.S_final,
                   r.prng_rounds, r.split_rounds, r.strided_rounds, net);
            fflush(stdout);
        }
    }

    if (NB > 1)
        printf("\ntotal net: %.1f bits  (avg %.1f / block)\n", total_net, total_net / NB);

    free(all);
    return 0;
}
