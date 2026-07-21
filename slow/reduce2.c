/*
 * reduce2.c — clean, table-driven, stacking entropy reducer (v1 core)
 *
 * Applies a sequence of reversible byte transforms to a 4096-byte block to lower
 * the cost of coding it with the downstream per-block Dirichlet(alpha) coder
 * (compressor3-style). Each transform ("layer") is chosen greedily by the net it
 * buys: net = (coder_bits_saved) - (instruction overhead bits).
 * Pass a=0 on the command line for the legacy objective (empirical order-0
 * entropy); a=<val> sets the coder's alpha (default 24, matching compressor3.c's
 * widened grid — see the g_alpha definition below for calibration history).
 *
 * Architecture: a REGISTRY of instruction descriptors, each with its own
 * search / apply / invert. Every round searches the FULL registry (in
 * parallel) and applies whichever candidate saves the most real bits.
 *
 * Registry pruned to the 9 instruction types that actually clear their own
 * overhead under the real-cost objective on BCrypt-random data — every other
 * family tried (stride/phase value ops, split/periodic constant ops, delta/
 * predictive ops, REFLECT) measured 0 fires across 100+ blocks, even when
 * auctioned directly against the post-PRNG residual. See reduce2-project /
 * reduce2-methodology memory for the full history and the sigma^2 theory
 * this converged on: an instruction can only pay for its own description
 * cost if its candidate pool re-rolls enough of the block to have real
 * outcome spread — value-relabelling ops don't, full/partial keystream
 * churn does.
 *
 * All 9 surviving types are "live": there is no more PRNG-tier/normal-tier
 * split, no fallback auction, and the per-block instruction count is capped
 * at PRNG_WIN (7), matching a flat 3-bit count field in the bitstream.
 *
 * Build:  gcc -O2 -fopenmp -o reduce2 reduce2.c -lm -lbcrypt
 * Run:    ./reduce2 [NB] [file]   a=<alpha> s=<seeds> o=<depth> b=<width> d=<0|1>
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <time.h>
#include <windows.h>
#include <bcrypt.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;

#ifndef BLOCK
#define BLOCK       4096        /* override with -DBLOCK=8192/16384/32768 */
#endif
#define MAXINSTR   16            /* real max/block is PRNG_WIN=7; small margin */
/* Live-tier greedy window: forward-search paths (o= flag) fit inside it, and
 * the count field is a flat 3 bits (0-7). Keep xs16 streams shorter than the
 * 65535 period: single-stream types walk BLOCK steps, so BLOCK<=32768. */
#define PRNG_WIN 7

/* globals visible to all search functions */
static int g_diag = 0;
/* PRNG seed-search space (power of 2). The seed field is charged and stored
 * at exactly log2(g_seed_lim) bits, so shrinking the space is an honest
 * speed-vs-gain trade: 16x fewer seeds costs the max-over-seeds ~0.6 sigma
 * but saves 4 bits of overhead on every PRNG instruction. s=<val> cmdline. */
static u32 g_seed_lim  = 65536;
static int g_seed_bits = 16;
/* PRNG_BSXOR mask densities: fraction of ones = threshold/256 (~50/40/20/10%) */
static const u8 BSXOR_TH[4] = { 128, 102, 51, 26 };

/* ---- instruction type ids: the 9 that fire under the real-cost objective ---- */
enum {
    T_PRNGADD4 = 0, /* 4-stream PRNG add: pos%4 selects stream, all derived from 1 master seed */
    T_PRNGADD8,     /* 8-stream PRNG add: pos%8 selects stream                                  */
    T_PLANEPRNG,    /* PRNG XOR bit plane k with xs16 LSB; amp=seed|(k<<16); self-inverse        */
    T_PRNGBIT,      /* PRNG selects bit pos (0-7) per byte; XOR that bit with (amp>>bit_pos)&1  */
    T_PRNGROT8,     /* 8-stream PRNG rotate-left by ks&7 (ks8); amp=seed                         */
    T_PRNGMUL8,     /* 8-stream PRNG multiply by odd ks|1 (ks8); amp=seed                        */
    T_PRNGNIB8,     /* 8-stream PRNG nibble-add lo+=ks&F, hi+=ks>>4 (ks8); amp=seed              */
    T_PRNGBSXOR,    /* prng_bitstream_xor_amp: seeded single-stream bitmask at density
                        ~50/40/20/10% ones (stream byte < threshold), masked bytes XOR one
                        8-bit amp; self-inverse; amp = seed | (density<<16) | (xoramp<<18).
                        densities >50% are redundant: mask-complement == free relabel + mask.
                        Weak at alpha=32 (3/300 fires) but nonzero; kept. Its siblings PAT_XOR/
                        PRNG_BSROT/PRNG_BSNIB (period-mask XOR / masked-rotate / masked-nibadd)
                        measured 0/300 fires at alpha=32 -- the flatter curvature that comes with
                        the widened compressor3 grid (see reduce2-project memory) starves partial-
                        churn instructions much harder than full-churn ones -- and were removed. */
    T_PRNGGFMUL8,   /* 8-stream PRNG multiply in GF(2^8) (AES reduction poly 0x11B) by a nonzero
                        keystream byte; genuinely nonlinear (no carries, unlike mod-256 MUL8);
                        amp=seed */
    NTYPES          /* = 9 */
};

typedef struct { u8 type; int stride, phase; u32 amp; } Instr;

/* ============================================================ *
 *  entropy machinery                                            *
 * ============================================================ */

#define LN2 0.69314718055994530942

/* Objective = real code length of the downstream per-block Dirichlet(alpha)
 * coder (compressor3-style: counts reset each block, P(v)=(c_v+a)/(t+256a)):
 *
 *   coded_bits = C(n,a) - sum_v ctab[c_v]      with
 *   ctab[k]    = log2 G(k+a) - log2 G(a)       (ctab[0]=0)
 *   C(n,a)     = log2 G(n+256a) - log2 G(256a)
 *
 * C(n,a) is constant for fixed n, so maximising S = sum ctab[c_v] minimises
 * the bits the coder actually emits, and net = dS - OH is real bits saved.
 * g_alpha = 0 selects the legacy objective ctab[k] = k*log2(k) (empirical
 * order-0 entropy). */
static double g_alpha = 32.0;    /* a=<val> on cmdline; 0 = legacy empirical.
                                     Calibrated to compressor3.c's new grid ceiling
                                     (widened from 16.0 -- see compressor3.c). Tried
                                     24 and 32 as the search objective; both converge
                                     to the same real end-to-end pipeline cost once
                                     coded at their own best fixed alpha, so 32 was
                                     kept for the simpler search=code=ceiling story. */
static double ctab[BLOCK + 1];

static void init_ctab(void) {
    ctab[0] = 0.0;
    if (g_alpha > 0.0) {
        double la = lgamma(g_alpha);
        for (int k = 1; k <= BLOCK; k++)
            ctab[k] = (lgamma((double)k + g_alpha) - la) / LN2;
    } else {
        for (int k = 1; k <= BLOCK; k++) ctab[k] = (double)k * log2((double)k);
    }
}

/* S = sum ctab[freq[v]]. Larger S = cheaper for the downstream coder. */
static double S_from_freq(const int *f) {
    double s = 0.0;
    for (int v = 0; v < 256; v++) s += ctab[f[v]];
    return s;
}
static void freq_of(const u8 *d, int n, int *f) {
    memset(f, 0, 256 * sizeof(int));
    for (int i = 0; i < n; i++) f[d[i]]++;
}
static double S_of(const u8 *d, int n) {
    int f[256];
    freq_of(d, n, f);
    return S_from_freq(f);
}
/* bits the downstream Dirichlet coder actually spends on the block
 * (legacy mode g_alpha==0: empirical order-0 entropy) */
static double cost_bits(const u8 *d, int n) {
    double S = S_of(d, n);
    if (g_alpha > 0.0)
        return (lgamma((double)n + 256.0 * g_alpha) - lgamma(256.0 * g_alpha)) / LN2 - S;
    return (double)n * log2((double)n) - S;
}

/* empirical order-0 entropy of an arbitrary-length byte stream (diagnostics
 * only; counts may exceed BLOCK, so this must not index ctab) */
static double stream_entropy_bits(const u8 *d, long n) {
    long f[256] = {0};
    double H = 0.0;
    for (long i = 0; i < n; i++) f[d[i]]++;
    for (int v = 0; v < 256; v++)
        if (f[v]) H += (double)f[v] * log2((double)n / (double)f[v]);
    return H;
}

/* ============================================================ *
 *  overhead model (bits)                                        *
 * ============================================================ */

/* Per-op overhead = (amortized type cost) + param fields. Search functions
 * bake in a flat TAGB placeholder; best_instr() swaps it for the real
 * taglen() cost of whichever type actually won, so the two always agree. */
#define TAGB   6.0
#define OH_PRNGADD4    (TAGB + (double)g_seed_bits)
#define OH_PRNGADD8    (TAGB + (double)g_seed_bits)
#define OH_PLANEPRNG   (TAGB + (double)g_seed_bits + 3.0)   /* seed + 3-bit plane index k */
#define OH_PRNGBIT     (TAGB + (double)g_seed_bits + 8.0)   /* seed + 8-bit flip-mask */
#define OH_PRNGSEED    (TAGB + (double)g_seed_bits)         /* ROT8/MUL8/NIB8: seed only */
#define OH_PRNGBSXOR   (TAGB + (double)g_seed_bits + 10.0)  /* seed + 2-bit density + 8-bit amp */

/* ============================================================ *
 *  apply / invert primitives (in place)                         *
 * ============================================================ */

static inline u8 addnib(u8 v, int lo, int hi) {
    return (u8)((((v & 0xF) + lo) & 0xF) | ((((v >> 4) + hi) & 0xF) << 4));
}

/* xorshift16 stream byte at step i (state advanced per call) */
static inline u8 xs16_next(u16 *s) {
    u16 x = *s;
    x ^= x << 7; x ^= x >> 9; x ^= x << 8;
    *s = x;
    return (u8)(x ^ (x >> 8));
}
static inline int xs16_b3(u16 *s) {
    u16 x = *s;
    x ^= x << 7; x ^= x >> 9; x ^= x << 8;
    *s = x;
    return (int)((x ^ (x >> 8)) & 7);
}
static void ap_prngbit(u8 *d, int n, u32 amp) {
    u16 seed = (u16)(amp & 0xFFFF);
    u8  fmask = (u8)((amp >> 16) & 0xFF);
    for (int i = 0; i < n; i++) {
        int b = xs16_b3(&seed);
        d[i] ^= (u8)(((fmask >> b) & 1) << b);
    }
}

/* BYTE_ROT primitive: circular bit rotation left by k. Inverse: rotate right
 * by k. Still needed by PRNG_ROT8's per-byte op. */
static inline u8 byterot_fwd(u8 v, int k) { return (u8)((v << k) | (v >> (8 - k))); }
static inline u8 byterot_inv(u8 v, int k) { return (u8)((v >> k) | (v << (8 - k))); }

/* Newton/Hensel-lift multiplicative inverse mod 256 (needed by PRNG_MUL8's
 * invert, which must undo a multiply-by-odd-constant bijection). */
static u8 mul_inv256(u8 a) {
    u8 x = 1;
    x = (u8)(x * (2 - a * x));  /* mod 4  */
    x = (u8)(x * (2 - a * x));  /* mod 16 */
    x = (u8)(x * (2 - a * x));  /* mod 256 */
    return x;
}

/* GF(2^8) multiplication (AES reduction poly 0x11B) for PRNG_GFMUL8: genuinely
 * different algebra from mod-256 multiply (no carries). g_gfmul is precomputed
 * once; g_gfinv[a] gives the b with gf_mul(a,b)=1. */
static u8 g_gfmul[256][256];
static u8 g_gfinv[256];

static u8 gf_mul_slow(u8 a, u8 b, u32 poly) {
    u8 p = 0;
    for (int c = 0; c < 8; c++) {
        if (b & 1) p ^= a;
        u8 carry = a & 0x80;
        a = (u8)(a << 1);
        if (carry) a ^= (u8)poly;
        b = (u8)(b >> 1);
    }
    return p;
}
static void init_gf(void) {
    u32 poly = 0x11B;
    for (int a = 0; a < 256; a++)
        for (int b = 0; b < 256; b++)
            g_gfmul[a][b] = gf_mul_slow((u8)a, (u8)b, poly);
    g_gfinv[0] = 0;
    for (int a = 1; a < 256; a++)
        for (int b = 1; b < 256; b++)
            if (g_gfmul[a][b] == 1) { g_gfinv[a] = (u8)b; break; }
}

/* Derive N sub-seeds from master by successive xs16 steps, then interleave by pos%N. */
static void ap_prngadd4(u8 *d, int n, u32 amp) {
    u16 ms = (u16)(amp & 0xFFFF);
    u16 s[4]; s[0]=ms; s[1]=xs16_next(&ms); s[2]=xs16_next(&ms); s[3]=xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + xs16_next(&s[i & 3]));
}
static void inv_prngadd4(u8 *d, int n, u32 amp) {
    u16 ms = (u16)(amp & 0xFFFF);
    u16 s[4]; s[0]=ms; s[1]=xs16_next(&ms); s[2]=xs16_next(&ms); s[3]=xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - xs16_next(&s[i & 3]));
}
static void ap_prngadd8(u8 *d, int n, u32 amp) {
    u16 ms = (u16)(amp & 0xFFFF);
    u16 s[8];
    s[0]=ms;
    for (int k=1;k<8;k++) s[k]=xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + xs16_next(&s[i & 7]));
}
static void inv_prngadd8(u8 *d, int n, u32 amp) {
    u16 ms = (u16)(amp & 0xFFFF);
    u16 s[8];
    s[0]=ms;
    for (int k=1;k<8;k++) s[k]=xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - xs16_next(&s[i & 7]));
}

/* PLANE_PRNG: for bit plane k, XOR each byte's bit k with xs16 LSB. Self-inverse. */
static void ap_planeprng(u8 *d, int n, u32 amp) {
    u16 s = (u16)(amp & 0xFFFF);
    u8 kmask = (u8)(1u << ((amp >> 16) & 7));
    for (int i = 0; i < n; i++) {
        u16 r = xs16_next(&s);
        if (r & 1) d[i] ^= kmask;
    }
}

/* Extra PRNG-tier families: same 8-stream derivation, different per-byte op.
 * Apply/invert regenerate the stream so decode needs no tables. */
static void gen_s8(u32 amp, u16 *s) {
    u16 ms = (u16)(amp & 0xFFFF); s[0] = ms;
    for (int k = 1; k < 8; k++) s[k] = xs16_next(&ms);
}
static void ap_prngrot8(u8 *d, int n, u32 amp) {
    u16 s[8]; gen_s8(amp, s);
    for (int i = 0; i < n; i++) {
        int r = xs16_next(&s[i & 7]) & 7;
        if (r) d[i] = byterot_fwd(d[i], r);
    }
}
static void inv_prngrot8(u8 *d, int n, u32 amp) {
    u16 s[8]; gen_s8(amp, s);
    for (int i = 0; i < n; i++) {
        int r = xs16_next(&s[i & 7]) & 7;
        if (r) d[i] = byterot_inv(d[i], r);
    }
}
static void ap_prngmul8(u8 *d, int n, u32 amp) {
    u16 s[8]; gen_s8(amp, s);
    for (int i = 0; i < n; i++)
        d[i] = (u8)(d[i] * (u8)(xs16_next(&s[i & 7]) | 1));
}
static void inv_prngmul8(u8 *d, int n, u32 amp) {
    u16 s[8]; gen_s8(amp, s);
    for (int i = 0; i < n; i++)
        d[i] = (u8)(d[i] * mul_inv256((u8)(xs16_next(&s[i & 7]) | 1)));
}
static void ap_prnggfmul8(u8 *d, int n, u32 amp) {
    u16 s[8]; gen_s8(amp, s);
    for (int i = 0; i < n; i++) {
        u8 m = xs16_next(&s[i & 7]); if (!m) m = 1;
        d[i] = g_gfmul[d[i]][m];
    }
}
static void inv_prnggfmul8(u8 *d, int n, u32 amp) {
    u16 s[8]; gen_s8(amp, s);
    for (int i = 0; i < n; i++) {
        u8 m = xs16_next(&s[i & 7]); if (!m) m = 1;
        d[i] = g_gfmul[d[i]][g_gfinv[m]];
    }
}
static void ap_prngnib8(u8 *d, int n, u32 amp) {
    u16 s[8]; gen_s8(amp, s);
    for (int i = 0; i < n; i++) {
        u8 k = xs16_next(&s[i & 7]);
        d[i] = addnib(d[i], k & 0xF, (k >> 4) & 0xF);
    }
}
static void inv_prngnib8(u8 *d, int n, u32 amp) {
    u16 s[8]; gen_s8(amp, s);
    for (int i = 0; i < n; i++) {
        u8 k = xs16_next(&s[i & 7]);
        d[i] = addnib(d[i], (-(int)(k & 0xF)) & 0xF, (-(int)((k >> 4) & 0xF)) & 0xF);
    }
}

/* PRNG_BSXOR: regenerate the single xs16 byte stream from the seed; positions
 * whose stream byte is under the density threshold get XORed by the 8-bit amp.
 * Self-inverse. amp field = seed | (density<<16) | (xoramp<<18). */
static void ap_prngbsxor(u8 *d, int n, u32 af) {
    u16 s  = (u16)(af & 0xFFFF);
    u8  th = BSXOR_TH[(af >> 16) & 3];
    u8  a  = (u8)((af >> 18) & 0xFF);
    for (int i = 0; i < n; i++)
        if (xs16_next(&s) < th) d[i] ^= a;
}

/* dispatch: apply / invert any instruction in place */
static void apply_instr(u8 *d, int n, Instr t) {
    switch (t.type) {
        case T_PRNGADD4:  ap_prngadd4(d, n, t.amp); break;
        case T_PRNGADD8:  ap_prngadd8(d, n, t.amp); break;
        case T_PLANEPRNG: ap_planeprng(d, n, t.amp); break;
        case T_PRNGBIT:   ap_prngbit(d, n, t.amp); break;
        case T_PRNGROT8:  ap_prngrot8(d, n, t.amp); break;
        case T_PRNGMUL8:  ap_prngmul8(d, n, t.amp); break;
        case T_PRNGNIB8:  ap_prngnib8(d, n, t.amp); break;
        case T_PRNGBSXOR: ap_prngbsxor(d, n, t.amp); break;
        case T_PRNGGFMUL8: ap_prnggfmul8(d, n, t.amp); break;
    }
}
static void invert_instr(u8 *d, int n, Instr t) {
    switch (t.type) {
        case T_PRNGADD4:  inv_prngadd4(d, n, t.amp); break;
        case T_PRNGADD8:  inv_prngadd8(d, n, t.amp); break;
        case T_PLANEPRNG: ap_planeprng(d, n, t.amp); break;   /* self-inv */
        case T_PRNGBIT:   ap_prngbit(d, n, t.amp); break;     /* self-inv */
        case T_PRNGROT8:  inv_prngrot8(d, n, t.amp); break;
        case T_PRNGMUL8:  inv_prngmul8(d, n, t.amp); break;
        case T_PRNGNIB8:  inv_prngnib8(d, n, t.amp); break;
        case T_PRNGBSXOR: ap_prngbsxor(d, n, t.amp); break;           /* self-inv */
        case T_PRNGGFMUL8: inv_prnggfmul8(d, n, t.amp); break;
    }
}

/* ============================================================ *
 *  searches: each fills *out and returns best net (may be <0)   *
 * ============================================================ */

/* Precomputed PRNG keystreams. The stream bytes are data-independent, so
 * per-search regeneration would be pure waste — and the loop-carried
 * dependency in xs16 is exactly what stalls the pipeline. Generated once:
 *   g_ks4  [seed*BLOCK+i]      ADD4 stream byte
 *   g_ks8  [seed*BLOCK+i]      ADD8/ROT8/MUL8/NIB8 shared stream byte
 *   g_kb3  [seed*BLOCK+i]      PRNG_BIT bit-position 0-7
 *   g_ks1  [seed*BLOCK+i]      PRNG_BSXOR stream byte
 *   g_kbit [seed*BLOCK/8+i>>3] PLANE_PRNG flip bit, packed
 * Apply/invert still regenerate on the fly (once per applied instr, so the
 * decoder needs no tables). */
static u8 *g_ks4, *g_ks8, *g_kb3, *g_kbit, *g_ks1;
static u8 g_rtab[8][256];   /* rotate-left lookup; filled here so threaded searches never init it */

static void init_keystreams(void) {
    for (int r = 0; r < 8; r++)
        for (int v = 0; v < 256; v++)
            g_rtab[r][v] = r ? byterot_fwd((u8)v, r) : (u8)v;
    size_t nb = (size_t)g_seed_lim * BLOCK;
    g_ks4  = malloc(nb);
    g_ks8  = malloc(nb);
    g_kb3  = malloc(nb);
    g_ks1  = malloc(nb);
    g_kbit = malloc(nb / 8);
    if (!g_ks4 || !g_ks8 || !g_kb3 || !g_ks1 || !g_kbit) {
        fprintf(stderr, "keystream tables: oom (%zu MB needed)\n", (nb * 4 + nb / 8) >> 20);
        exit(1);
    }
    memset(g_kbit, 0, nb / 8);
    for (u32 seed = 1; seed < g_seed_lim; seed++) {
        u16 ms, s4[4], s8[8], sp, sb, s1;
        u8 *k4 = g_ks4 + (size_t)seed * BLOCK;
        u8 *k8 = g_ks8 + (size_t)seed * BLOCK;
        u8 *k3 = g_kb3 + (size_t)seed * BLOCK;
        u8 *k1 = g_ks1 + (size_t)seed * BLOCK;
        u8 *kb = g_kbit + (size_t)seed * (BLOCK / 8);
        ms = (u16)seed; s4[0] = ms;
        for (int k = 1; k < 4; k++) s4[k] = xs16_next(&ms);
        ms = (u16)seed; s8[0] = ms;
        for (int k = 1; k < 8; k++) s8[k] = xs16_next(&ms);
        sp = (u16)seed; sb = (u16)seed; s1 = (u16)seed;
        for (int i = 0; i < BLOCK; i++) {
            k4[i] = xs16_next(&s4[i & 3]);
            k8[i] = xs16_next(&s8[i & 7]);
            k3[i] = (u8)xs16_b3(&sb);
            k1[i] = xs16_next(&s1);
            if (xs16_next(&sp) & 1) kb[i >> 3] |= (u8)(1 << (i & 7));
        }
    }
}

/* N-stream PRNG_ADD variants: single master seed, N derived streams by pos%N. */
static double search_prngadd4(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < g_seed_lim; seed++) {
        const u8 *ks = g_ks4 + (size_t)seed * BLOCK;
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + ks[i])]++;
        double net = (S_from_freq(f) - Sb) - OH_PRNGADD4;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGADD4; out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}
static double search_prngadd8(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < g_seed_lim; seed++) {
        const u8 *ks = g_ks8 + (size_t)seed * BLOCK;
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + ks[i])]++;
        double net = (S_from_freq(f) - Sb) - OH_PRNGADD8;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGADD8; out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* PLANE_PRNG: for each xs16 seed, compute flip/noflip histograms once, then test all 8 planes. */
static double search_planeprng(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; int bk = 0; u32 bseed = 1;
    for (u32 seed = 1; seed < g_seed_lim; seed++) {
        const u8 *kb = g_kbit + (size_t)seed * (BLOCK / 8);
        int flip[256] = {0}, noflip[256] = {0};
        for (int i = 0; i < n; i++) {
            if ((kb[i >> 3] >> (i & 7)) & 1) flip[d[i]]++;
            else                             noflip[d[i]]++;
        }
        for (int k = 0; k < 8; k++) {
            int f[256] = {0};
            int kmask = 1 << k;
            for (int v = 0; v < 256; v++) {
                f[v]         += noflip[v];
                f[v ^ kmask] += flip[v];
            }
            double net = (S_from_freq(f) - Sb) - OH_PLANEPRNG;
            if (net > best) { best = net; bk = k; bseed = seed; }
        }
    }
    out->type = T_PLANEPRNG; out->stride = 0; out->phase = 0;
    out->amp = bseed | ((u32)bk << 16);
    return best;
}

/* PRNG_BIT: for each seed, build 8 disjoint groups per bit pos; greedy flip mask. */
static double search_prngbit(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18;
    u16 best_seed = 1; u8 best_fmask = 0;
    for (u32 s = 1; s < g_seed_lim; s++) {
        const u8 *k3 = g_kb3 + (size_t)s * BLOCK;
        int grp[8][256]; memset(grp, 0, sizeof grp);
        for (int i = 0; i < n; i++) grp[k3[i]][d[i]]++;
        u8 fmask = 0;
        for (int j = 0; j < 8; j++) {
            int bv = 1 << j;
            double dS = 0.0;
            for (int v = 0; v < 256; v++) {
                int gv = grp[j][v]; if (!gv) continue;
                int gvf = grp[j][v ^ bv];
                dS += ctab[total[v]-gv+gvf] + ctab[total[v^bv]+gv-gvf]
                    - ctab[total[v]] - ctab[total[v^bv]];
            }
            if (dS > 0.0) fmask |= (u8)(1 << j);
        }
        int cur[256]; memcpy(cur, total, sizeof cur);
        for (int j = 0; j < 8; j++) {
            if (!((fmask >> j) & 1)) continue;
            int bv = 1 << j;
            for (int v = 0; v < 256; v++) {
                if (!grp[j][v]) continue;
                cur[v] -= grp[j][v]; cur[v^bv] += grp[j][v];
            }
        }
        double net = (S_from_freq(cur) - Sb) - OH_PRNGBIT;
        if (net > best) { best = net; best_seed = (u16)s; best_fmask = fmask; }
    }
    out->type = T_PRNGBIT; out->stride = 0; out->phase = 0;
    out->amp = (u32)best_seed | ((u32)best_fmask << 16);
    return best;
}

/* extra PRNG families: identical loop shape, different per-byte op, shared tables */
static double search_prngrot8(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < g_seed_lim; seed++) {
        const u8 *ks = g_ks8 + (size_t)seed * BLOCK;
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[g_rtab[ks[i] & 7][d[i]]]++;
        double net = (S_from_freq(f) - Sb) - OH_PRNGSEED;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGROT8; out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}
static double search_prngmul8(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < g_seed_lim; seed++) {
        const u8 *ks = g_ks8 + (size_t)seed * BLOCK;
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] * (u8)(ks[i] | 1))]++;
        double net = (S_from_freq(f) - Sb) - OH_PRNGSEED;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGMUL8; out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}
static double search_prnggfmul8(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < g_seed_lim; seed++) {
        const u8 *ks = g_ks8 + (size_t)seed * BLOCK;
        int f[256] = {0};
        for (int i = 0; i < n; i++) {
            u8 m = ks[i]; if (!m) m = 1;
            f[g_gfmul[d[i]][m]]++;
        }
        double net = (S_from_freq(f) - Sb) - OH_PRNGSEED;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGGFMUL8; out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

static double search_prngnib8(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < g_seed_lim; seed++) {
        const u8 *ks = g_ks8 + (size_t)seed * BLOCK;
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[addnib(d[i], ks[i] & 0xF, (ks[i] >> 4) & 0xF)]++;
        double net = (S_from_freq(f) - Sb) - OH_PRNGSEED;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGNIB8; out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- Walsh-Hadamard Transform helper (used by PAT_XOR and PRNG_BSXOR) ---- */
static void wht256(int *a) {
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += len<<1)
            for (int j = 0; j < len; j++) {
                int u = a[i+j], v = a[i+j+len];
                a[i+j] = u+v; a[i+j+len] = u-v;
            }
}

/* Correlation helper shared with PRNG_BSXOR's search below: given the WHT of
 * the full histogram and of a candidate subset histogram, finds the XOR
 * constant that best concentrates the subset when merged back, via the WHT
 * correlation prod[k]=(WT[k]-Whit[k])*Whit[k], then an exact ctab check of
 * that one candidate. (Formerly also used by PAT_XOR, removed 2026-07-20:
 * 0/300 fires at alpha=32 — see reduce2-project memory.) */
static double patxor_eval(const int *total, const int *WT,
                          const int *hit, const int *Whit, u32 *c_out) {
    long long prod[256];
    for (int k = 0; k < 256; k++)
        prod[k] = (long long)(WT[k] - Whit[k]) * (long long)Whit[k];
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += len << 1)
            for (int j = 0; j < len; j++) {
                long long u = prod[i+j], v = prod[i+j+len];
                prod[i+j] = u + v; prod[i+j+len] = u - v;
            }
    int bc = 1; long long bcv = prod[1];
    for (int c = 2; c < 256; c++)
        if (prod[c] > bcv) { bcv = prod[c]; bc = c; }
    double S = 0.0;
    for (int v = 0; v < 256; v++)
        S += ctab[total[v] - hit[v] + hit[v ^ bc]];
    *c_out = (u32)bc;
    return S;
}

/* PRNG_BSXOR search: per (density, seed) the mask is fixed by the ks1 table,
 * so finding the best XOR amp is a subset-histogram problem — reuse
 * patxor_eval (WHT correlation proxy + exact ctab check). All three fields
 * (seed, density, amp) are stored, so the charge is honest. */
static double search_prngbsxor_range(const u8 *d, int n, double Sb, Instr *out,
                                     int dlo, int dhi) {
    int total[256]; freq_of(d, n, total);
    int WT[256]; memcpy(WT, total, sizeof WT); wht256(WT);
    double oh = OH_PRNGBSXOR;
    double best = -1e18; u32 bamp = 1;
    for (int dens = dlo; dens <= dhi; dens++) {
        u8 th = BSXOR_TH[dens];
        for (u32 seed = 1; seed < g_seed_lim; seed++) {
            const u8 *ks = g_ks1 + (size_t)seed * BLOCK;
            int hit[256] = {0};
            for (int i = 0; i < n; i++)
                if (ks[i] < th) hit[d[i]]++;
            int Whit[256]; memcpy(Whit, hit, sizeof Whit); wht256(Whit);
            u32 c; double S = patxor_eval(total, WT, hit, Whit, &c);
            double net = (S - Sb) - oh;
            if (net > best) { best = net; bamp = seed | ((u32)dens << 16) | (c << 18); }
        }
    }
    out->type = T_PRNGBSXOR; out->stride = 0; out->phase = 0; out->amp = bamp;
    return best;
}
static double search_prngbsxor_lo(const u8 *d, int n, double Sb, Instr *out) { return search_prngbsxor_range(d, n, Sb, out, 0, 1); }
static double search_prngbsxor_hi(const u8 *d, int n, double Sb, Instr *out) { return search_prngbsxor_range(d, n, Sb, out, 2, 3); }

/* registry of selectable instructions. All rows are always searched every
 * round now — the old prng-tier/normal-tier split doesn't exist any more
 * since every surviving type IS a live-tier type. */
typedef double (*SearchFn)(const u8 *, int, double, Instr *);
typedef struct { const char *name; SearchFn search; } InstrDesc;

static const InstrDesc REGISTRY[] = {
    { "PRNG_ADD4",     search_prngadd4     },
    { "PRNG_ADD8",     search_prngadd8     },
    { "PLANE_PRNG",    search_planeprng    },
    { "PRNG_BIT",      search_prngbit      },
    { "PRNG_ROT8",     search_prngrot8     },
    { "PRNG_MUL8",     search_prngmul8     },
    { "PRNG_NIB8",     search_prngnib8     },
    /* prng_bitstream_xor_amp: two density-range rows for thread balance */
    { "PRNG_BSXOR_LO", search_prngbsxor_lo },
    { "PRNG_BSXOR_HI", search_prngbsxor_hi },
    { "PRNG_GFMUL8",   search_prnggfmul8  },
};
#define NREG ((int)(sizeof(REGISTRY)/sizeof(REGISTRY[0])))
static const char *TYPE_NAME[NTYPES] = {
    "PRNG_ADD4", "PRNG_ADD8", "PLANE_PRNG", "PRNG_BIT",   "PRNG_ROT8",
    "PRNG_MUL8", "PRNG_NIB8", "PRNG_BSXOR", "PRNG_GFMUL8"
};

/* Static prefix code for the type tag, rebuilt as an exact canonical
 * Huffman tree over a clean 300-block fire distribution at the CURRENT
 * default alpha=32 (ADD4=80, ADD8=83, PLANE=11, BIT=3, ROT8=29, MUL8=37,
 * NIB8=18, BSXOR=3, GFMUL8=25; n=289). This replaces an earlier version of
 * this tree that was mistakenly calibrated on alpha=16 fire data and then
 * applied at alpha=32 — the two regimes have qualitatively different
 * winners (alpha=32's flatter curvature favours full-churn types much more
 * strongly; see the PAT_XOR/BSROT/BSNIB removal note on T_PRNGBSXOR above).
 *   00 ADD4  01 ADD8 | 100 GFMUL8  101 ROT8  110 MUL8
 *   1110 NIB8 | 11110 PLANE | 111110 BIT  111111 BSXOR
 * Kraft sum = 2*2^-2 + 3*2^-3 + 2^-4 + 2^-5 + 2*2^-6 = 1.0 exactly.
 * taglen() is the exact serialized cost, used both to score candidates and
 * to bill them in the accept test. Re-derive if the fire distribution
 * drifts meaningfully. */
static double taglen(int type) {
    switch (type) {
        case T_PRNGADD4: case T_PRNGADD8:                      return 2.0;
        case T_PRNGGFMUL8: case T_PRNGROT8: case T_PRNGMUL8:   return 3.0;
        case T_PRNGNIB8:                                       return 4.0;
        case T_PLANEPRNG:                                      return 5.0;
        default:                                                /* BIT/BSXOR */
        case T_PRNGBIT: case T_PRNGBSXOR:                      return 6.0;
    }
}

/* per-family candidates from the most recent best_instr call (lookahead input) */
static Instr  g_cands[NREG];
static double g_cand_nets[NREG];
static int    g_ncand;
static int    g_beam = 0;      /* b=<K> cmdline: lookahead width per node (default 3) */
static int    g_look = 0;      /* o=<D> cmdline: forward-search path depth, 0/1 = greedy */

/* best selectable instruction for the current data. The registry loop is
 * OpenMP-parallel: each row writes only its own slot, and every search is
 * pure (reads the block + tables initialised before threading starts). */
static double best_instr(const u8 *d, int n, Instr *out) {
    double Sb = S_of(d, n);
    static Instr  all_c[NREG];
    static double all_net[NREG];
    #pragma omp parallel for schedule(dynamic)
    for (int r = 0; r < NREG; r++) {
        Instr cand;
        double net = REGISTRY[r].search(d, n, Sb, &cand);
        /* swap the flat TAGB baked into every OH_ formula for the real
         * prefix-code length of this type */
        net = net + TAGB - taglen(cand.type);
        all_c[r] = cand; all_net[r] = net;
    }
    double best = -1e18; Instr bi = {0};
    g_ncand = NREG;
    for (int r = 0; r < NREG; r++) {
        if (g_diag) printf("    [diag] %-12s net=%+.2f\n", REGISTRY[r].name, all_net[r]);
        g_cands[r] = all_c[r]; g_cand_nets[r] = all_net[r];
        if (all_net[r] > best) { best = all_net[r]; bi = all_c[r]; }
    }
    *out = bi;
    return best;
}

/* Forward path search (the "order 2-5" test): value of the best sequence of
 * up to `depth` instructions from this state, expanding the top `width`
 * family candidates per node; 0 means nothing profitable. A path may open
 * with a solo-negative instruction if its suffix repays it. Sets
 * *first/*first_net to the opening move of the winning path; later greedy
 * rounds re-plan and deterministically re-find the suffix. The tree walk is
 * serial; the heavy work (each node's registry sweep) is parallel inside
 * best_instr. Cost ~ width^depth registry sweeps per committed instruction. */
static double lookahead_score(const u8 *d, int n, int depth, int width,
                              Instr *first, double *first_net) {
    Instr  cbe[NREG]; double cne[NREG]; int order[NREG];
    Instr  dummy;
    best_instr(d, n, &dummy);
    int m = g_ncand;
    if (m == 0) return 0.0;
    memcpy(cbe, g_cands, sizeof(Instr) * (size_t)m);
    memcpy(cne, g_cand_nets, sizeof(double) * (size_t)m);
    for (int i = 0; i < m; i++) order[i] = i;
    for (int i = 0; i < m - 1; i++)
        for (int j = i + 1; j < m; j++)
            if (cne[order[j]] > cne[order[i]]) { int t = order[i]; order[i] = order[j]; order[j] = t; }
    int K = (width < m) ? width : m;
    double bestv = 0.0;
    *first = cbe[order[0]]; *first_net = cne[order[0]];
    for (int k = 0; k < K; k++) {
        int c = order[k];
        if (cne[c] <= -30.0) break;      /* too deep a hole for any suffix */
        double v = cne[c];
        if (depth > 1) {
            u8 scr[BLOCK];
            memcpy(scr, d, n);
            apply_instr(scr, n, cbe[c]);
            Instr f2; double n2;
            v += lookahead_score(scr, n, depth - 1, width, &f2, &n2);
        }
        if (v > bestv) { bestv = v; *first = cbe[c]; *first_net = cne[c]; }
    }
    return bestv;
}

/* ============================================================ *
 *  compress (greedy)                                            *
 * ============================================================ */

static double binary_entropy(double p) {
    if (p <= 0.0 || p >= 1.0) return 0.0;
    return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
}

/* data-dependent diagnostic scratch + the "XOR-with-previous-byte" probe */
static u8 g_scr[BLOCK];
static void ap_xprev(u8 *d, int n, int s) {
    for (int i = n - 1; i >= s; i--) d[i] ^= d[i - s];
}

/* Per-layer structure dump: order-0 entropy alone hides *where* the redundancy is.
 * This prints the byte-value spread (mean/stdev/distinct), the entropy of each bit
 * plane (catches structure a byte-histogram view misses, e.g. a stuck high bit), and
 * the best short XOR-with-previous-byte stride (a cheap periodicity/correlation probe)
 * so a human can see what each layer exposed or used up. */
static void print_layer_struct(const u8 *d, int n) {
    int f[256]; freq_of(d, n, f);
    int distinct = 0;
    for (int v = 0; v < 256; v++) if (f[v]) distinct++;

    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += d[i];
    double mean = sum / n;
    double var = 0.0;
    for (int i = 0; i < n; i++) { double dv = d[i] - mean; var += dv * dv; }
    var /= n;
    double stdev = sqrt(var);

    double bitH[8], bitHavg = 0.0;
    for (int b = 0; b < 8; b++) {
        long ones = 0;
        for (int i = 0; i < n; i++) ones += (d[i] >> b) & 1;
        bitH[b] = binary_entropy((double)ones / n);
        bitHavg += bitH[b];
    }
    bitHavg /= 8.0;

    int top_val = 0, top_cnt = 0;
    for (int v = 0; v < 256; v++) if (f[v] > top_cnt) { top_cnt = f[v]; top_val = v; }

    int best_s = 0; double best_bps = 1e18;
    for (int s = 1; s <= 8 && s < n; s++) {
        memcpy(g_scr, d, n);
        ap_xprev(g_scr, n, s);
        double e = cost_bits(g_scr, n) / n;
        if (e < best_bps) { best_bps = e; best_s = s; }
    }

    printf("      [struct] distinct=%3d/256  mean=%6.1f  top=0x%02X(%d)  sd=%5.1f  bitH(avg)=%.3f"
           "  bitH/plane(lo->hi)=[%.2f %.2f %.2f %.2f %.2f %.2f %.2f %.2f]"
           "  best-xorprev: s%d -> %.4f bps\n",
           distinct, mean, top_val, top_cnt, stdev, bitHavg,
           bitH[0], bitH[1], bitH[2], bitH[3], bitH[4], bitH[5], bitH[6], bitH[7],
           best_s, best_bps);
}

/* run greedy to convergence; nets[i] receives the net bits saved by ilist[i] */
static double greedy_run(u8 *d, int n, Instr *ilist, double *nets, int *ni, int verbose) {
    double gained = 0.0;
    for (;;) {
        if (*ni >= PRNG_WIN) break;        /* count-field ceiling (flat 3-bit header) */
        Instr t; double net = 0.0; int accepted = 0;
        if (g_look >= 2 && g_alpha > 0.0) {
            /* receding-horizon forward search: commit the opening move of the
             * best path of up to o= instructions; the accept decision is the
             * whole path's summed real net, so a solo-negative opener is fine
             * when its suffix repays it */
            int deff = g_look;
            if (deff > PRNG_WIN - *ni) deff = PRNG_WIN - *ni;
            double solo;
            double pv = lookahead_score(d, n, deff, (g_beam > 0) ? g_beam : 3, &t, &solo);
            if (pv > 0.0) { net = solo; accepted = 1; }
        } else {
            net = best_instr(d, n, &t);
            if (net > 0.0) accepted = 1;
        }
        if (!accepted) break;
        double e0 = cost_bits(d, n) / n;
        apply_instr(d, n, t);
        nets[*ni] = net;
        ilist[(*ni)++] = t;
        gained += net;
        if (verbose) {
            /* compute mean and mode of data after applying instruction */
            int f[256]; freq_of(d, n, f);
            double sum = 0.0;
            for (int i = 0; i < n; i++) sum += d[i];
            int top_v = 0, top_c = 0;
            for (int v = 0; v < 256; v++) if (f[v] > top_c) { top_c = f[v]; top_v = v; }
            printf("  %-12s s%-3d p%-3d a%-6u  %.4f -> %.4f bps  net=%+.1f  mean=%5.1f  top=0x%02X(%d)\n",
                   TYPE_NAME[t.type], t.stride, t.phase, t.amp,
                   e0, cost_bits(d, n) / n, net, sum / n, top_v, top_c);
            print_layer_struct(d, n);
        }
    }
    return gained;
}

static double compress(u8 *d, int n, Instr *ilist, double *nets, int *ni, int verbose) {
    *ni = 0;
    if (verbose) {
        printf("  INPUT        (before any layer)         %.4f bps\n", cost_bits(d, n) / n);
        print_layer_struct(d, n);
    }
    return greedy_run(d, n, ilist, nets, ni, verbose);
}

/* ============================================================ *
 *  decode (apply inverses in reverse order)                     *
 * ============================================================ */

static void decompress(u8 *d, int n, const Instr *ilist, int ni) {
    for (int i = ni - 1; i >= 0; i--) invert_instr(d, n, ilist[i]);
}

/* ============================================================ *
 *  Compact instruction bit-stream                               *
 *                                                                *
 *  Layout per instruction:                                      *
 *    type : static prefix code, 2-4 bits (see taglen())         *
 *    amp  : type-specific width (seed bits, pattern bits, etc.) *
 *  Header: flat 3-bit instruction count (0..PRNG_WIN=7).        *
 * ============================================================ */

typedef struct { u8 *buf; int pos; } BitBuf;

static void bb_put(BitBuf *b, u32 val, int nbits) {
    for (int i = nbits - 1; i >= 0; i--) {
        int p = b->pos++;
        b->buf[p >> 3] |= (u8)(((val >> i) & 1) << (7 - (p & 7)));
    }
}
static u32 bb_get(BitBuf *b, int nbits) {
    u32 v = 0;
    for (int i = 0; i < nbits; i++) {
        int p = b->pos++;
        v = (v << 1) | ((b->buf[p >> 3] >> (7 - (p & 7))) & 1);
    }
    return v;
}

/* type tag: static prefix code mirroring taglen() exactly */
static void bb_put_tag(BitBuf *b, int type) {
    switch (type) {
        case T_PRNGADD4:    bb_put(b, 0x0,  2); break;  /* 00     */
        case T_PRNGADD8:    bb_put(b, 0x1,  2); break;  /* 01     */
        case T_PRNGGFMUL8:  bb_put(b, 0x4,  3); break;  /* 100    */
        case T_PRNGROT8:    bb_put(b, 0x5,  3); break;  /* 101    */
        case T_PRNGMUL8:    bb_put(b, 0x6,  3); break;  /* 110    */
        case T_PRNGNIB8:    bb_put(b, 0xE,  4); break;  /* 1110   */
        case T_PLANEPRNG:   bb_put(b, 0x1E, 5); break;  /* 11110  */
        case T_PRNGBIT:     bb_put(b, 0x3E, 6); break;  /* 111110 */
        case T_PRNGBSXOR:   bb_put(b, 0x3F, 6); break;  /* 111111 */
    }
}
static int bb_get_tag(BitBuf *b) {
    if (bb_get(b, 1) == 0) return bb_get(b, 1) ? T_PRNGADD8 : T_PRNGADD4; /* 01/00 */
    if (bb_get(b, 1) == 0) {                                   /* 10.   */
        if (bb_get(b, 1) == 0) return T_PRNGGFMUL8;             /* 100   */
        return T_PRNGROT8;                                      /* 101   */
    }
    if (bb_get(b, 1) == 0) return T_PRNGMUL8;                   /* 110   */
    if (bb_get(b, 1) == 0) return T_PRNGNIB8;                   /* 1110  */
    if (bb_get(b, 1) == 0) return T_PLANEPRNG;                  /* 11110 */
    return bb_get(b, 1) ? T_PRNGBSXOR : T_PRNGBIT;              /* 111111/111110 */
}

static void bb_put_instr(BitBuf *b, Instr t) {
    bb_put_tag(b, t.type);
    switch (t.type) {
        case T_PRNGADD4: case T_PRNGADD8:
        case T_PRNGROT8: case T_PRNGMUL8: case T_PRNGNIB8: case T_PRNGGFMUL8:
            bb_put(b, t.amp & (g_seed_lim - 1), g_seed_bits); break;
        case T_PLANEPRNG:
            bb_put(b, t.amp & (g_seed_lim - 1), g_seed_bits);
            bb_put(b, (t.amp >> 16) & 7, 3); break;
        case T_PRNGBIT:
            bb_put(b, t.amp & (g_seed_lim - 1), g_seed_bits);
            bb_put(b, (t.amp >> 16) & 0xFF, 8); break;
        case T_PRNGBSXOR:
            bb_put(b, t.amp & (g_seed_lim - 1), g_seed_bits);
            bb_put(b, (t.amp >> 16) & 3, 2);             /* density */
            bb_put(b, (t.amp >> 18) & 0xFF, 8);          /* XOR amp */
            break;
    }
}

static Instr bb_get_instr(BitBuf *b) {
    Instr t = {0, 0, 0, 0};
    t.type = (u8)bb_get_tag(b);
    switch (t.type) {
        case T_PRNGADD4: case T_PRNGADD8:
        case T_PRNGROT8: case T_PRNGMUL8: case T_PRNGNIB8: case T_PRNGGFMUL8:
            t.amp = bb_get(b, g_seed_bits); break;
        case T_PLANEPRNG: {
            u32 seed = bb_get(b, g_seed_bits), k = bb_get(b, 3);
            t.amp = seed | (k << 16); break;
        }
        case T_PRNGBIT: {
            u32 seed = bb_get(b, g_seed_bits), fm = bb_get(b, 8);
            t.amp = seed | (fm << 16); break;
        }
        case T_PRNGBSXOR: {
            u32 seed = bb_get(b, g_seed_bits);
            u32 dens = bb_get(b, 2);
            u32 a    = bb_get(b, 8);
            t.amp = seed | (dens << 16) | (a << 18);
            break;
        }
    }
    return t;
}

/* Max bits per instruction: PRNG_BIT/BSXOR at 6-bit tag + 16-bit seed + up to
 * 10 bits of payload = 32 bits */
#define COMPACT_BUF_BYTES ((MAXINSTR * 32 + 3 + 7) / 8)

/* Pack instruction list into compact bit-stream. Returns bit count.
 * Header: flat 3-bit count (0..PRNG_WIN=7, exact fit, no escape needed). */
static int pack_ilist(const Instr *ilist, int ni, u8 *buf) {
    memset(buf, 0, COMPACT_BUF_BYTES);
    BitBuf b = { buf, 0 };
    bb_put(&b, (u32)ni, 3);
    for (int i = 0; i < ni; i++) bb_put_instr(&b, ilist[i]);
    return b.pos;
}

/* Unpack. Returns instruction count. */
static int unpack_ilist(const u8 *buf, Instr *ilist_out) {
    BitBuf b = { (u8 *)buf, 0 };
    int ni = (int)bb_get(&b, 3);
    for (int i = 0; i < ni; i++) ilist_out[i] = bb_get_instr(&b);
    return ni;
}

/* ============================================================ *
 *  main                                                         *
 * ============================================================ */

/* validate apply/invert for every instruction type independently */
static int selftest(void) {
    u8 base[BLOCK], a[BLOCK];
    u32 st = 99u;
    for (int i = 0; i < BLOCK; i++) { st = st * 1103515245u + 12345u; base[i] = (u8)(st >> 16); }

    Instr tv[] = {
        { T_PRNGADD4,  0, 0, 1234u },
        { T_PRNGADD8,  0, 0, 5678u },
        { T_PLANEPRNG, 0, 0, (1234u | (3u << 16)) },
        { T_PRNGBIT,   0, 0, (0xABCDu | (0xA5u << 16)) },
        { T_PRNGROT8,  0, 0, 2468u },
        { T_PRNGMUL8,  0, 0, 1357u },
        { T_PRNGNIB8,  0, 0, 3579u },
        { T_PRNGBSXOR,         0, 0, (777u  | (0u << 16) | (0x3Cu << 18)) },
        { T_PRNGBSXOR,         0, 0, (1234u | (3u << 16) | (0xA5u << 18)) },
        { T_PRNGGFMUL8,        0, 0, 8642u },
    };
    int nt = (int)(sizeof(tv) / sizeof(tv[0])), fails = 0;
    for (int i = 0; i < nt; i++) {
        memcpy(a, base, BLOCK);
        apply_instr(a, BLOCK, tv[i]);
        invert_instr(a, BLOCK, tv[i]);
        if (memcmp(a, base, BLOCK) != 0) {
            printf("  SELFTEST FAIL: %s\n", TYPE_NAME[tv[i].type]);
            fails++;
        }
    }
    printf("selftest: %d/%d apply-invert pairs OK\n", nt - fails, nt);
    return fails;
}

/* last-block instruction list, accessible from main for serialisation */
static Instr  g_ilist[MAXINSTR];
static double g_nets[MAXINSTR];
static int    g_last_ni = 0;

/* overhead in bits for a single applied instruction (mirrors search OH formulas) */
static double instr_oh(Instr t) {
    switch (t.type) {
        case T_PRNGADD4:  return OH_PRNGADD4;
        case T_PRNGADD8:  return OH_PRNGADD8;
        case T_PLANEPRNG: return OH_PLANEPRNG;
        case T_PRNGBIT:   return OH_PRNGBIT;
        case T_PRNGROT8: case T_PRNGMUL8: case T_PRNGNIB8: case T_PRNGGFMUL8:
                          return OH_PRNGSEED;
        case T_PRNGBSXOR: return OH_PRNGBSXOR;
        default:          return 0.0;
    }
}

/* reduce one block, verify round-trip, accumulate per-type counts + net stats */
static double do_block(u8 *data, int n, int *counts,
                       double *type_net_sum, double *type_net_max, double *type_oh_sum,
                       int verbose, int *ok_out) {
    u8 orig[BLOCK];
    memcpy(orig, data, n);
    g_last_ni = 0;
    double net = compress(data, n, g_ilist, g_nets, &g_last_ni, verbose);
    for (int i = 0; i < g_last_ni; i++) {
        int t = g_ilist[i].type;
        counts[t]++;
        type_net_sum[t] += g_nets[i];
        if (g_nets[i] > type_net_max[t]) type_net_max[t] = g_nets[i];
        type_oh_sum[t]  += instr_oh(g_ilist[i]);
    }

    u8 dec[BLOCK];
    memcpy(dec, data, n);
    decompress(dec, n, g_ilist, g_last_ni);
    *ok_out = (memcmp(dec, orig, n) == 0);
    return net;
}

int main(int argc, char **argv) {
    /* a=<alpha> sets the objective's Dirichlet concentration (a=0 => legacy
     * empirical-entropy objective); s=<seeds> sets the PRNG seed-search space
     * (power of 2, 256..65536; charged/stored at log2(s) bits); o=<depth>/
     * b=<width> control the forward path search; d=1 prints every family's
     * net each round. Parsed before the tables are built. Valid in every mode. */
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "a=", 2) == 0) g_alpha = atof(argv[i] + 2);
        if (strncmp(argv[i], "b=", 2) == 0) g_beam = atoi(argv[i] + 2);
        if (strncmp(argv[i], "d=", 2) == 0) g_diag = atoi(argv[i] + 2);
        if (strncmp(argv[i], "o=", 2) == 0) {
            g_look = atoi(argv[i] + 2);
            if (g_look > PRNG_WIN) g_look = PRNG_WIN;
        }
        if (strncmp(argv[i], "s=", 2) == 0) {
            u32 v = (u32)atoi(argv[i] + 2);
            if (v < 256 || v > 65536 || (v & (v - 1))) {
                fprintf(stderr, "s= must be a power of 2 in 256..65536\n");
                return 1;
            }
            g_seed_lim = v;
            g_seed_bits = 0;
            while ((1u << g_seed_bits) < v) g_seed_bits++;
        }
    }
    init_ctab();
    init_gf();
    {
        clock_t tk = clock();
        init_keystreams();
        if (g_alpha > 0.0)
            printf("objective: Dirichlet(alpha=%g) code length | seeds=%u (%d-bit field, tables %.0f ms)%s\n",
                   g_alpha, g_seed_lim, g_seed_bits,
                   (double)(clock() - tk) / CLOCKS_PER_SEC * 1000.0,
                   g_look >= 2 ? " | forward search on" : "");
        else
            printf("objective: empirical order-0 entropy (legacy) | seeds=%u\n", g_seed_lim);
        if (g_look >= 2)
            printf("forward search: depth o=%d, width b=%d\n", g_look, (g_beam > 0) ? g_beam : 3);
    }

    if (argc > 1 && strcmp(argv[1], "selftest") == 0) return selftest() ? 2 : 0;

    /* ---- firstlayer mode: one best_instr call per block, type win histogram ---- */
    if (argc > 1 && strcmp(argv[1], "firstlayer") == 0) {
        int NB = (argc > 2) ? atoi(argv[2]) : 20;
        if (NB < 1) NB = 1;
        u8 *all = malloc((size_t)NB * BLOCK);
        if (!all) { fprintf(stderr, "oom\n"); return 1; }
        size_t need = (size_t)NB * BLOCK, got = 0;
        FILE *fc = fopen("bcrypt.bin", "rb");
        if (fc) { got = fread(all, 1, need, fc); fclose(fc); }
        if (got < need) {
            BCryptGenRandom(NULL, all + got, (ULONG)(need - got), BCRYPT_USE_SYSTEM_PREFERRED_RNG);
            fc = fopen("bcrypt.bin", "wb");
            if (fc) { fwrite(all, 1, need, fc); fclose(fc); }
        }
        printf("firstlayer: %d blocks\n", NB);
        int wins[NTYPES] = {0};
        double net_sum[NTYPES] = {0};
        clock_t t0 = clock();
        for (int b = 0; b < NB; b++) {
            u8 *data = all + (size_t)b * BLOCK;
            Instr best; double net = best_instr(data, BLOCK, &best);
            if (net > 0.0) { wins[best.type]++; net_sum[best.type] += net; }
            double ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
            printf("  block %3d: winner=%-12s  net=%+7.1f  (%.0f ms elapsed)\n",
                   b, net > 0.0 ? TYPE_NAME[best.type] : "(none)", net, ms);
            fflush(stdout);
        }
        double ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;
        /* sort types by win count descending */
        int order[NTYPES]; for (int i = 0; i < NTYPES; i++) order[i] = i;
        for (int i = 0; i < NTYPES-1; i++)
            for (int j = i+1; j < NTYPES; j++)
                if (wins[order[j]] > wins[order[i]]) { int tmp=order[i]; order[i]=order[j]; order[j]=tmp; }
        printf("\n=== first-layer wins across %d blocks (%.0f ms) ===\n", NB, ms);
        printf("  %-14s  %5s  %8s\n", "type", "wins", "avg net");
        printf("  %-14s  %5s  %8s\n", "----", "----", "-------");
        for (int i = 0; i < NTYPES; i++) {
            int t = order[i];
            if (wins[t] == 0) continue;
            printf("  %-14s  %5d  %+8.1f\n", TYPE_NAME[t], wins[t],
                   net_sum[t] / wins[t]);
        }
        free(all);
        return 0;
    }

    /* ---- default: many blocks, aggregate ---- */
    /* Usage: reduce2 [NB] [file]  — file defaults to BCrypt random */
    int NB = 1;
    const char *src_file = NULL;
    for (int i = 1; i < argc; i++) {
        if (strncmp(argv[i], "a=", 2) == 0 || strncmp(argv[i], "s=", 2) == 0 ||
            strncmp(argv[i], "b=", 2) == 0 || strncmp(argv[i], "o=", 2) == 0 ||
            strncmp(argv[i], "d=", 2) == 0) continue;
        int v = atoi(argv[i]);
        if (v > 0) NB = v;
        else       src_file = argv[i];
    }
    u8 *all = malloc((size_t)NB * BLOCK);
    if (!all) { fprintf(stderr, "oom\n"); return 1; }

    size_t need = (size_t)NB * BLOCK, got = 0;
    if (src_file) {
        FILE *f = fopen(src_file, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", src_file); return 1; }
        got = fread(all, 1, need, f); fclose(f);
        if (got < (size_t)BLOCK) { fprintf(stderr, "file too small\n"); return 1; }
        NB = (int)(got / BLOCK);  /* use however many full blocks fit */
        printf("input: %s (%d blocks x %d bytes)\n", src_file, NB, BLOCK);
    } else {
        const char *default_file = "C:\\Users\\lukac\\Documents\\compressor\\compresseddata.bin";
        FILE *f = fopen(default_file, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", default_file); return 1; }
        got = fread(all, 1, need, f); fclose(f);
        if (got < (size_t)BLOCK) { fprintf(stderr, "file too small\n"); return 1; }
        NB = (int)(got / BLOCK);
        printf("input: %s (%d blocks x %d bytes)\n", default_file, NB, BLOCK);
    }

    FILE *fcomp = fopen("compressed.bin", "wb");
    if (!fcomp) { fprintf(stderr, "cannot open compressed.bin\n"); return 1; }

    int counts[NTYPES] = {0};
    double type_net_sum[NTYPES] = {0};
    double type_net_max[NTYPES];
    double type_oh_sum[NTYPES]  = {0};
    for (int t = 0; t < NTYPES; t++) type_net_max[t] = 0.0;
    double total_net = 0.0, total_ein = 0.0, total_eout = 0.0;
    double total_overhead = 0.0;
    int total_ni = 0, fails = 0, compact_fails = 0;
    long total_compact_bits = 0;
    /* flat 8-byte-per-instruction buffer for entropy comparison */
    u8 *ibuf = malloc((size_t)NB * MAXINSTR * 8);
    int ibuf_n = 0;
    u8 *compact_buf = malloc(COMPACT_BUF_BYTES);
    Instr *unpacked_tmp = malloc(MAXINSTR * sizeof(Instr));
    clock_t t0 = clock();

    printf("\n=== compressing %d blocks ===\n", NB);
    for (int b = 0; b < NB; b++) {
        u8 *data = all + (size_t)b * BLOCK;
        double e_in = cost_bits(data, BLOCK);
        int ok = 0;
        double net = do_block(data, BLOCK, counts, type_net_sum, type_net_max, type_oh_sum, NB == 1, &ok);
        double e_out = cost_bits(data, BLOCK);
        double raw  = e_in - e_out;
        double oh   = raw - net;
        /* compact pack + round-trip verify */
        int cbits = compact_buf ? pack_ilist(g_ilist, g_last_ni, compact_buf) : 0;
        int cok = 0;
        if (compact_buf && unpacked_tmp) {
            int ni2 = unpack_ilist(compact_buf, unpacked_tmp);
            cok = (ni2 == g_last_ni);
            for (int i = 0; cok && i < ni2; i++)
                cok = (unpacked_tmp[i].type   == g_ilist[i].type   &&
                       unpacked_tmp[i].stride == g_ilist[i].stride &&
                       unpacked_tmp[i].phase  == g_ilist[i].phase  &&
                       unpacked_tmp[i].amp    == g_ilist[i].amp);
        }
        total_compact_bits += cbits;
        if (!cok) compact_fails++;

        total_net += net; total_ein += e_in; total_eout += e_out;
        total_overhead += oh; total_ni += g_last_ni;
        if (!ok) fails++;
        printf("  block %2d: %.4f -> %.4f bps  net=%+.1f  %s  [%d instrs  raw=%+.1f  OH=%.1f bits  compact=%d bits %s]\n",
               b, e_in / BLOCK, e_out / BLOCK, net, ok ? "ok" : "FAIL",
               g_last_ni, raw, oh, cbits, cok ? "ok" : "FAIL");
        if (NB == 1) {
            int fq[256]; freq_of(data, BLOCK, fq);
            printf("\n--- frequency dump (compressed block) ---\n");
            printf("  val  freq  |  val  freq  |  val  freq  |  val  freq\n");
            for (int row = 0; row < 64; row++) {
                for (int col = 0; col < 4; col++) {
                    int v = row + col * 64;
                    printf("  %3d %5d  ", v, fq[v]);
                    if (col < 3) printf("|");
                }
                printf("\n");
            }
            printf("-----------------------------------------\n\n");
        }
        fflush(stdout);

        /* serialise instructions for entropy measurement */
        for (int i = 0; ibuf && i < g_last_ni; i++) {
            ibuf[ibuf_n++] = (u8)g_ilist[i].type;
            ibuf[ibuf_n++] = (u8)g_ilist[i].stride;
            ibuf[ibuf_n++] = (u8)(g_ilist[i].phase & 0xFF);
            ibuf[ibuf_n++] = (u8)(g_ilist[i].phase >> 8);
            ibuf[ibuf_n++] = (u8)( g_ilist[i].amp        & 0xFF);
            ibuf[ibuf_n++] = (u8)((g_ilist[i].amp >>  8) & 0xFF);
            ibuf[ibuf_n++] = (u8)((g_ilist[i].amp >> 16) & 0xFF);
            ibuf[ibuf_n++] = (u8)((g_ilist[i].amp >> 24) & 0xFF);
        }

        fwrite(data, 1, BLOCK, fcomp);
    }
    fclose(fcomp);
    double ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;

    int fired = 0;
    for (int t = 0; t < NTYPES; t++) if (counts[t]) fired++;

    printf("\n=== aggregate over %d blocks (%.0f ms) ===\n", NB, ms);
    printf("avg input:  %.4f bps     avg output: %.4f bps\n",
           total_ein / (NB * BLOCK), total_eout / (NB * BLOCK));
    printf("total net:  %.1f bits   (avg %.1f / block)\n", total_net, total_net / NB);
    printf("total instrs: %d (avg %.1f/block)   total OH: %.1f bits (avg %.1f/block)\n",
           total_ni, (double)total_ni / NB, total_overhead, total_overhead / NB);
    if (ibuf && ibuf_n > 0) {
        double ibits = stream_entropy_bits(ibuf, ibuf_n);
        printf("instr stream (flat 8B/instr): %.4f bps  (%d bytes)  entropy=%.0f bits/block\n",
               ibits / ibuf_n, ibuf_n, ibits / NB);
    }
    printf("instr stream (compact):       %ld bits/block total  (%.1f bits/block avg)  %s\n",
           total_compact_bits / NB,
           (double)total_compact_bits / NB,
           compact_fails ? "*** COMPACT FAIL ***" : "round-trip ok");
    free(ibuf);
    free(compact_buf);
    free(unpacked_tmp);
    printf("round-trip: %s (%d/%d blocks)\n", fails ? "*** FAIL ***" : "OK", NB - fails, NB);
    printf("types fired (across all blocks): %d / %d\n\n", fired, NTYPES);
    printf("  %-14s %6s  %8s  %8s  %12s\n", "type", "fires", "avg net", "top net", "effectivness");
    printf("  %-14s %6s  %8s  %8s  %12s\n", "----", "-----", "-------", "-------", "------------");
    for (int t = 0; t < NTYPES; t++) {
        if (counts[t] == 0) {
            printf("  %-14s %6d\n", TYPE_NAME[t], 0);
        } else {
            double eff = (type_oh_sum[t] > 0.0) ? type_net_sum[t] / type_oh_sum[t] : 0.0;
            printf("  %-14s %6d  %+8.1f  %+8.1f  %11.2fx\n",
                   TYPE_NAME[t], counts[t],
                   type_net_sum[t] / counts[t],
                   type_net_max[t],
                   eff);
        }
    }

    free(all);
    return fails ? 2 : 0;
}
