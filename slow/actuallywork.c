/*
 * reduce2.c — clean, table-driven, stacking entropy reducer (v1 core)
 *
 * Applies a sequence of reversible byte transforms to a 4096-byte block to lower
 * its order-0 Shannon entropy. Each transform ("layer") is chosen greedily by the
 * net it buys: net = (entropy_bits_saved) - (instruction overhead bits).
 *
 * Architecture: a REGISTRY of instruction descriptors, each with its own
 * search / apply / invert / overhead. The greedy loop asks every instruction for
 * its best move, applies the single best, and repeats until nothing nets positive.
 *
 * Pruned down to the 7 instruction types that demonstrably fire on real BCrypt
 * data, with a full inverse decoder and round-trip verification.
 *
 * Build:  gcc -O2 -o reduce2 reduce2.c -lm -lbcrypt
 * Run:    ./reduce2 [input.bin]   (no arg => generated test block)
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

#define BLOCK     4096
#define MAXINSTR  8192

/* ---- instruction type ids ---- */
enum {
    T_XORP = 0,  /* XOR amp at (stride,phase) subset       */
    T_NIBSW,     /* nibble swap at (stride,phase) subset   */
    T_ANIBS,     /* nibble-wise add (lo+=al, hi+=ah)       */
    T_QUADADD,   /* 4-way pos split, 4 ADD amps            */
    T_PRNGD,     /* PRNG dual: even/odd pos XOR two xs16 seeds  */
    T_OCTNIBX,   /* 8-way pos split, 8 nibble (lo-nibble only) XOR amps */
    T_STRIDEADD, /* ADD amp (nonzero) at (stride,phase) subset, stride in 1..64 */
    T_XORDELTA,  /* XOR with predecessor in group: d[i] ^= d[i-stride] (last->first) */
    T_PRNGBIT,   /* PRNG selects bit pos (0-7) per byte; XOR that bit with (amp>>bit_pos)&1 */
    T_BYTEROT,   /* circular bit rotation: v = (v<<k)|(v>>(8-k)), k=1..7 */
    T_HALFXOR,   /* value-conditional XOR: bytes<128 XOR amp_lo, bytes>=128 XOR amp_hi */
    NTYPES
};

typedef struct { u8 type; int stride, phase; u32 amp; } Instr;

/* ============================================================ *
 *  entropy machinery                                            *
 * ============================================================ */

static double hlog[BLOCK + 1];   /* hlog[k] = k * log2(k), hlog[0]=0 */

static void init_hlog(void) {
    hlog[0] = 0.0;
    for (int k = 1; k <= BLOCK; k++) hlog[k] = (double)k * log2((double)k);
}

/* S = sum hlog[freq[v]]. Larger S = more concentrated = lower entropy. */
static double S_from_freq(const int *f) {
    double s = 0.0;
    for (int v = 0; v < 256; v++) s += hlog[f[v]];
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
/* total order-0 entropy of the block, in bits */
static double entropy_bits(const u8 *d, int n) {
    return (double)n * log2((double)n) - S_of(d, n);
}

/* ============================================================ *
 *  overhead model (bits): 6-bit tag + param fields              *
 * ============================================================ */

/* Per-op overhead = (amortized type cost) + param fields. With the instruction stream
 * entropy-coded, a repeated op type costs ~2-3 bits, not a flat 6-bit tag — this is what
 * lets small +1-bit ops clear the bar and stack (high coverage). */
#define TAGB   4.0    /* honest fixed-width tag: ceil(log2(16)) = 4 bits, 16-slot type space */
#define SB     6.0    /* stride field (1..64 = 6 bits) */
#define PB     6.0    /* phase field  (0..63 = 6 bits) */
#define OH_XORP   (TAGB + SB + PB + 8.0)
#define OH_NIBSW  (TAGB + SB + PB)
#define OH_NIBS   (TAGB + SB + PB + 8.0)
#define OH_QUAD   (TAGB + SB + PB + 32.0)
#define OH_PRNGD  (TAGB + 32.0)          /* 2 independent 16-bit seeds, honest */
#define OH_OCTNIBX (TAGB + SB + PB + 32.0) /* 8 bands * 4-bit lo-nibble XOR amp each */
#define OH_STRIDEADD (TAGB + SB + PB + 8.0)    /* stride 1..64, phase 0..63, amp 8 bits (nonzero) */
#define OH_XORDELTA (TAGB + SB + PB)        /* no amp: result is fully determined by stride/phase */
#define OH_PRNGBIT      (TAGB + 16.0 + 8.0)    /* 16-bit seed + 8-bit flip-mask, no stride/phase */
#define OH_BYTEROT      (TAGB + SB + PB + 3.0) /* rotation k in 1..7 */
#define OH_HALFXOR      (TAGB + SB + PB + 7.0 + 7.0) /* amp_lo (7 bits) + amp_hi (7 bits) */

/* ============================================================ *
 *  apply / invert primitives (in place)                         *
 * ============================================================ */

static inline u8 nibswap(u8 v) { return (u8)((v << 4) | (v >> 4)); }

static inline u8 addnib(u8 v, int lo, int hi) {
    return (u8)((((v & 0xF) + lo) & 0xF) | ((((v >> 4) + hi) & 0xF) << 4));
}

static void ap_xorp(u8 *d, int n, int s, int p, u32 a) {
    for (int i = p; i < n; i += s) d[i] ^= (u8)a;
}
static void ap_nibsw(u8 *d, int n, int s, int p) {
    for (int i = p; i < n; i += s) d[i] = nibswap(d[i]);
}
static void ap_strideadd(u8 *d, int n, int s, int p, u32 a) {
    for (int i = p; i < n; i += s) d[i] = (u8)(d[i] + a);
}
static void inv_strideadd(u8 *d, int n, int s, int p, u32 a) {
    for (int i = p; i < n; i += s) d[i] = (u8)(d[i] - a);
}
/* xorshift16 stream byte at step i (state advanced per call) */
static inline u8 xs16_next(u16 *s) {
    u16 x = *s;
    x ^= x << 7; x ^= x >> 9; x ^= x << 8;
    *s = x;
    return (u8)(x ^ (x >> 8));
}
/* PRNG_BIT helper: same xorshift16 update as xs16_next but returns 3-bit value (0-7).
 * XOR-folds both bytes of the state so all 16 seed bits influence the output. */
static inline int xs16_b3(u16 *s) {
    u16 x = *s;
    x ^= x << 7; x ^= x >> 9; x ^= x << 8;
    *s = x;
    return (int)((x ^ (x >> 8)) & 7);
}
/* PRNG_BIT: for each byte, PRNG selects bit_pos in [0,7]; XOR that bit with
 * (flip_mask >> bit_pos) & 1. Self-inverse (same op undoes it). amp[15:0]=seed, amp[23:16]=flip_mask. */
static void ap_prngbit(u8 *d, int n, u32 amp) {
    u16 seed  = (u16)(amp & 0xFFFF);
    u8  fmask = (u8)((amp >> 16) & 0xFF);
    for (int i = 0; i < n; i++) {
        int b = xs16_b3(&seed);
        d[i] ^= (u8)(((fmask >> b) & 1) << b);
    }
}
/* BYTE_ROT: circular bit rotation left by k. Inverse: rotate right by k (= left by 8-k). */
static inline u8 byterot_fwd(u8 v, int k) { return (u8)((v << k) | (v >> (8 - k))); }
static inline u8 byterot_inv(u8 v, int k) { return (u8)((v >> k) | (v << (8 - k))); }
static void ap_byterot(u8 *d, int n, int s, int p, int k) {
    for (int i = p; i < n; i += s) d[i] = byterot_fwd(d[i], k);
}
static void inv_byterot(u8 *d, int n, int s, int p, int k) {
    for (int i = p; i < n; i += s) d[i] = byterot_inv(d[i], k);
}

/* HALF_XOR: value-conditional XOR. Bytes 0-127 XOR with amp_lo (7 bits, bit7 untouched),
 * bytes 128-255 XOR with amp_hi (7 bits, bit7 untouched). Self-inverse because bit7 is
 * never modified, so the decoder always knows which branch was taken. amp[6:0]=lo, amp[14:8]=hi. */
static void ap_halfxor(u8 *d, int n, int s, int p, u32 amp) {
    u8 lo = (u8)(amp & 0x7F), hi = (u8)((amp >> 8) & 0x7F);
    for (int i = p; i < n; i += s) {
        if (d[i] & 0x80) d[i] ^= hi;
        else              d[i] ^= lo;
    }
}

/* out[i] ^= in[i-s]; high->low. Not a selectable instruction any more, but kept
 * for the per-layer "best short XOR-prev stride" diagnostic probe. */
static void ap_xprev(u8 *d, int n, int s) {
    for (int i = n - 1; i >= s; i--) d[i] ^= d[i - s];
}
static void ap_anibs(u8 *d, int n, int s, int p, u32 amp) {
    int lo = amp & 0xF, hi = (amp >> 4) & 0xF;
    for (int i = p; i < n; i += s) d[i] = addnib(d[i], lo, hi);
}
static void inv_anibs(u8 *d, int n, int s, int p, u32 amp) {
    int lo = (-(int)(amp & 0xF)) & 0xF, hi = (-(int)((amp >> 4) & 0xF)) & 0xF;
    for (int i = p; i < n; i += s) d[i] = addnib(d[i], lo, hi);
}
/* 4-way position split ADD (complement to QUAD_XOR) */
static void ap_quadadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 a[4] = { (u8)(amp), (u8)(amp>>8), (u8)(amp>>16), (u8)(amp>>24) };
    int k = 0;
    for (int i = p; i < n; i += s, k++) d[i] = (u8)(d[i] + a[k & 3]);
}
static void inv_quadadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 a[4] = { (u8)(amp), (u8)(amp>>8), (u8)(amp>>16), (u8)(amp>>24) };
    int k = 0;
    for (int i = p; i < n; i += s, k++) d[i] = (u8)(d[i] - a[k & 3]);
}
/* PRNG dual: even positions XOR xs16(seed1), odd positions XOR xs16(seed2). Self-inverse. */
static void ap_prngd(u8 *d, int n, u32 amp) {
    u16 s1 = (u16)(amp & 0xFFFF), s2 = (u16)((amp >> 16) & 0xFFFF);
    for (int i = 0; i < n; i++) {
        u8 b = (i & 1) ? xs16_next(&s2) : xs16_next(&s1);
        d[i] ^= b;
    }
}
/* 8-way position split (k % 8), lo-nibble-only XOR per band. Self-inverse. */
static void ap_octnibx(u8 *d, int n, int s, int p, u32 amp) {
    u8 x[8];
    for (int g = 0; g < 8; g++) x[g] = (u8)((amp >> (g * 4)) & 0xF);
    int k = 0;
    for (int i = p; i < n; i += s, k++) {
        int g = k % 8;
        d[i] = (u8)((d[i] & 0xF0) | ((d[i] ^ x[g]) & 0x0F));
    }
}

/* XOR_DELTA: d[i] ^= d[i-stride] processed last->first so each element uses its
 * original predecessor. Inverse is the same XOR but first->last (predecessor already
 * restored at that point). Neither direction has an amp — output is data-determined. */
static void ap_xordelta(u8 *d, int n, int s, int p) {
    int last = p; while (last + s < n) last += s;
    for (int i = last; i >= p + s; i -= s) d[i] ^= d[i - s];
}
static void inv_xordelta(u8 *d, int n, int s, int p) {
    for (int i = p + s; i < n; i += s) d[i] ^= d[i - s];
}

/* dispatch: apply / invert any instruction in place */
static void apply_instr(u8 *d, int n, Instr t) {
    switch (t.type) {
        case T_XORP:    ap_xorp(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBSW:   ap_nibsw(d, n, t.stride, t.phase); break;
        case T_ANIBS:   ap_anibs(d, n, t.stride, t.phase, t.amp); break;
        case T_QUADADD: ap_quadadd(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGD:   ap_prngd(d, n, t.amp); break;
        case T_OCTNIBX: ap_octnibx(d, n, t.stride, t.phase, t.amp); break;
        case T_STRIDEADD: ap_strideadd(d, n, t.stride, t.phase, t.amp); break;
        case T_XORDELTA:  ap_xordelta(d, n, t.stride, t.phase); break;
        case T_PRNGBIT:   ap_prngbit(d, n, t.amp); break;
        case T_BYTEROT:   ap_byterot(d, n, t.stride, t.phase, (int)t.amp); break;
        case T_HALFXOR:   ap_halfxor(d, n, t.stride, t.phase, t.amp); break;
    }
}
static void invert_instr(u8 *d, int n, Instr t) {
    switch (t.type) {
        case T_XORP:    ap_xorp(d, n, t.stride, t.phase, t.amp); break;       /* self-inv */
        case T_NIBSW:   ap_nibsw(d, n, t.stride, t.phase); break;             /* self-inv */
        case T_ANIBS:   inv_anibs(d, n, t.stride, t.phase, t.amp); break;
        case T_QUADADD: inv_quadadd(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGD:   ap_prngd(d, n, t.amp); break;                         /* self-inv */
        case T_OCTNIBX: ap_octnibx(d, n, t.stride, t.phase, t.amp); break;    /* self-inv */
        case T_STRIDEADD: inv_strideadd(d, n, t.stride, t.phase, t.amp); break;
        case T_XORDELTA:  inv_xordelta(d, n, t.stride, t.phase); break;
        case T_PRNGBIT:   ap_prngbit(d, n, t.amp); break;                     /* self-inv */
        case T_BYTEROT:   inv_byterot(d, n, t.stride, t.phase, (int)t.amp); break;
        case T_HALFXOR:   ap_halfxor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
    }
}

/* ============================================================ *
 *  searches: each fills *out and returns best net (may be <0)   *
 * ============================================================ */

#define MAX_STRIDE 64           /* search strides 1..64 */
#define PRNG_SEEDS 65536        /* try seeds 1..65535 */

/* XOR / ADD at (stride,phase) via the frequency-table trick: O(256) per amp */
static double search_xorp(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 0; a < 256; a++) {
                int rf[256];
                memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[u ^ a] += hit[u];
                double S = S_from_freq(rf);
                double net = (S - Sb) - OH_XORP;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->type = T_XORP; out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}
/* fixed-map subset bijection (NIB_SWAP): freq trick, no amp loop */
static double search_nibsw(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int rf[256];
            for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
            for (int u = 0; u < 256; u++) rf[nibswap((u8)u)] += hit[u];
            double net = (S_from_freq(rf) - Sb) - OH_NIBSW;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->type = T_NIBSW; out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}
/* data-dependent ops: apply to a scratch copy and measure S directly */
static u8 g_scr[BLOCK];
/* STRIDE_ADD: same frequency-table trick as XOR_PHASE, but ADD instead of XOR, an
 * expanded stride range of 1..33, and amp restricted to nonzero (adding 0 is a no-op). */
static double search_strideadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 1;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256];
                memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[(u + a) & 255] += hit[u];
                double net = (S_from_freq(rf) - Sb) - OH_STRIDEADD;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->type = T_STRIDEADD; out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}
/* ADD_NIBS: brute 256 amps via freq trick */
static double search_anibs(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int amp = 1; amp < 256; amp++) {
                int lo = amp & 0xF, hi = (amp >> 4) & 0xF;
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[addnib((u8)u, lo, hi)] += hit[u];
                double net = (S_from_freq(rf) - Sb) - OH_NIBS;
                if (net > best) { best = net; bs = s; bp = p; bamp = (u32)amp; }
            }
        }
    }
    out->type = T_ANIBS; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}
/* k-way position split ADD: coordinate descent on K add amps */
static double search_ksplit_add(const u8 *d, int n, double Sb, Instr *out, int K, int type, double oh) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        for (int p = 0; p < s; p++) {
            int ph[4][256]; memset(ph, 0, sizeof ph);
            int k = 0;
            for (int i = p; i < n; i += s, k++) ph[k % K][d[i]]++;
            int amps[4] = {0,0,0,0};
            int cur[256]; memcpy(cur, total, sizeof cur);
            for (int pass = 0; pass < 5; pass++) {
                int changed = 0;
                for (int g = 0; g < K; g++) {
                    for (int w = 0; w < 256; w++) cur[w] -= ph[g][(w - amps[g]) & 255];
                    int ba = 0; double bS = -1e18;
                    for (int a = 0; a < 256; a++) {
                        double S = 0.0;
                        for (int w = 0; w < 256; w++) S += hlog[cur[w] + ph[g][(w - a) & 255]];
                        if (S > bS) { bS = S; ba = a; }
                    }
                    if (ba != amps[g]) { amps[g] = ba; changed = 1; }
                    for (int w = 0; w < 256; w++) cur[w] += ph[g][(w - amps[g]) & 255];
                }
                if (!changed) break;
            }
            double S = S_from_freq(cur);
            double net = (S - Sb) - oh;
            if (net > best) {
                best = net; bs = s; bp = p;
                bamp = (u32)amps[0] | ((u32)amps[1]<<8) | ((u32)amps[2]<<16) | ((u32)amps[3]<<24);
            }
        }
    }
    out->type = type; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}
static double search_quadadd(const u8 *d, int n, double Sb, Instr *out) {
    return search_ksplit_add(d, n, Sb, out, 4, T_QUADADD, OH_QUAD);
}
/* PRNG dual: two xs16 seeds for even/odd positions; coord descent */
static double search_prngd(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18;
    u16 best_s1 = 1, best_s2 = 1;
    /* pass 1: fix s2=1, scan s1 */
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        u16 s1 = (u16)seed, s2 = 1;
        for (int i = 0; i < n; i++) {
            u8 b = (i & 1) ? xs16_next(&s2) : xs16_next(&s1);
            f[d[i] ^ b]++;
        }
        double net = (S_from_freq(f) - Sb) - OH_PRNGD;
        if (net > best) { best = net; best_s1 = (u16)seed; }
    }
    /* pass 2: fix best s1, scan s2 */
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        u16 s1 = best_s1, s2 = (u16)seed;
        for (int i = 0; i < n; i++) {
            u8 b = (i & 1) ? xs16_next(&s2) : xs16_next(&s1);
            f[d[i] ^ b]++;
        }
        double net = (S_from_freq(f) - Sb) - OH_PRNGD;
        if (net > best) { best = net; best_s2 = (u16)seed; }
    }
    /* pass 3: second coordinate-descent round -- fix best s2, rescan s1 */
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        u16 s1 = (u16)seed, s2 = best_s2;
        for (int i = 0; i < n; i++) {
            u8 b = (i & 1) ? xs16_next(&s2) : xs16_next(&s1);
            f[d[i] ^ b]++;
        }
        double net = (S_from_freq(f) - Sb) - OH_PRNGD;
        if (net > best) { best = net; best_s1 = (u16)seed; }
    }
    out->type = T_PRNGD; out->stride = 0; out->phase = 0;
    out->amp = (u32)best_s1 | ((u32)best_s2 << 16);
    return best;
}
/* 8-way position split (k%8), lo-nibble-only XOR amp per band -- finer position
 * granularity than QUAD_XOR at the same overhead, since amps are nibble- not byte-wide. */
static double search_octnibx(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        for (int p = 0; p < s; p++) {
            int ph[8][256]; memset(ph, 0, sizeof ph);
            int k = 0;
            for (int i = p; i < n; i += s, k++) ph[k % 8][d[i]]++;
            int amps[8] = {0,0,0,0,0,0,0,0};
            int cur[256]; memcpy(cur, total, sizeof cur);
            for (int pass = 0; pass < 4; pass++) {
                int changed = 0;
                for (int g = 0; g < 8; g++) {
                    for (int w = 0; w < 256; w++) {
                        int src = (w & 0xF0) | ((w & 0x0F) ^ amps[g]);
                        cur[w] -= ph[g][src];
                    }
                    int ba = 0; double bS = -1e18;
                    for (int a = 0; a < 16; a++) {
                        double S = 0.0;
                        for (int w = 0; w < 256; w++) {
                            int src = (w & 0xF0) | ((w & 0x0F) ^ a);
                            S += hlog[cur[w] + ph[g][src]];
                        }
                        if (S > bS) { bS = S; ba = a; }
                    }
                    if (ba != amps[g]) changed = 1;
                    amps[g] = ba;
                    for (int w = 0; w < 256; w++) {
                        int src = (w & 0xF0) | ((w & 0x0F) ^ amps[g]);
                        cur[w] += ph[g][src];
                    }
                }
                if (!changed) break;
            }
            double S = S_from_freq(cur);
            double net = (S - Sb) - OH_OCTNIBX;
            if (net > best) {
                best = net; bs = s; bp = p;
                bamp = 0;
                for (int g = 0; g < 8; g++) bamp |= ((u32)amps[g]) << (g * 4);
            }
        }
    }
    out->type = T_OCTNIBX; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* XOR_DELTA: data-dependent (apply to scratch and measure S) */
static double search_xordelta(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        for (int p = 0; p < s; p++) {
            memcpy(g_scr, d, n);
            ap_xordelta(g_scr, n, s, p);
            double net = (S_of(g_scr, n) - Sb) - OH_XORDELTA;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->type = T_XORDELTA; out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}
/* PRNG_BIT search: for each seed, build 8 disjoint groups (one per bit position
 * assigned by the PRNG). Since each byte belongs to exactly one group, the 8 amp-bit
 * decisions are fully independent: flip bit j iff doing so alone improves S. The
 * optimal flip_mask is therefore readable off in one O(8*256) pass per seed. */
static double search_prngbit(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18;
    u16 best_seed = 1; u8 best_fmask = 0;

    for (u32 s = 1; s < 65536; s++) {
        u16 st = (u16)s;
        int grp[8][256]; memset(grp, 0, sizeof grp);
        for (int i = 0; i < n; i++) grp[xs16_b3(&st)][d[i]]++;

        /* for each bit j independently: does flipping bit j of group j improve S? */
        u8 fmask = 0;
        for (int j = 0; j < 8; j++) {
            int bv = 1 << j;
            double dS = 0.0;
            for (int v = 0; v < 256; v++) {
                int gv = grp[j][v];
                if (!gv) continue;
                int gvf = grp[j][v ^ bv];
                /* new_total[v] = total[v] - gv + gvf
                   new_total[v^bv] = total[v^bv] + gv - gvf   */
                dS += hlog[total[v]     - gv  + gvf]
                    + hlog[total[v^bv]  + gv  - gvf]
                    - hlog[total[v]]
                    - hlog[total[v^bv]];
            }
            if (dS > 0.0) fmask |= (u8)(1 << j);
        }

        /* compute exact S after applying all chosen flips */
        int cur[256]; memcpy(cur, total, sizeof cur);
        for (int j = 0; j < 8; j++) {
            if (!((fmask >> j) & 1)) continue;
            int bv = 1 << j;
            for (int v = 0; v < 256; v++) {
                if (!grp[j][v]) continue;
                cur[v]      -= grp[j][v];
                cur[v ^ bv] += grp[j][v];
            }
        }
        double net = (S_from_freq(cur) - Sb) - OH_PRNGBIT;
        if (net > best) { best = net; best_seed = (u16)s; best_fmask = fmask; }
    }

    out->type = T_PRNGBIT; out->stride = 0; out->phase = 0;
    out->amp = (u32)best_seed | ((u32)best_fmask << 16);
    return best;
}

/* BYTE_ROT: frequency-table trick, k=1..7 left-rotations. */
static double search_byterot(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bk = 1;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k = 1; k <= 7; k++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[byterot_fwd((u8)u, k)] += hit[u];
                double net = (S_from_freq(rf) - Sb) - OH_BYTEROT;
                if (net > best) { best = net; bs = s; bp = p; bk = (u32)k; }
            }
        }
    }
    out->type = T_BYTEROT; out->stride = bs; out->phase = bp; out->amp = bk;
    return best;
}

/* HALF_XOR: build separate 128-entry freq tables for lo-half and hi-half bytes.
 * The two XOR amps are independent (disjoint value ranges, bit7 untouched), so
 * we find each optimally with the standard freq trick — one O(128*128) pass each. */
static double search_halfxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        for (int p = 0; p < s; p++) {
            int flo[128] = {0}, fhi[128] = {0};
            for (int i = p; i < n; i += s) {
                u8 v = d[i];
                if (v & 0x80) fhi[v & 0x7F]++; else flo[v]++;
            }
            int blo_arr[128], bhi_arr[128];
            for (int v = 0; v < 128; v++) {
                blo_arr[v] = total[v]       - flo[v];
                bhi_arr[v] = total[v + 128] - fhi[v];
            }
            int alo = 0; double bSlo = -1e18;
            for (int a = 0; a < 128; a++) {
                double S = 0.0;
                for (int v = 0; v < 128; v++) S += hlog[blo_arr[v] + flo[v ^ a]];
                if (S > bSlo) { bSlo = S; alo = a; }
            }
            int ahi = 0; double bShi = -1e18;
            for (int a = 0; a < 128; a++) {
                double S = 0.0;
                for (int v = 0; v < 128; v++) S += hlog[bhi_arr[v] + fhi[v ^ a]];
                if (S > bShi) { bShi = S; ahi = a; }
            }
            double net = (bSlo + bShi - Sb) - OH_HALFXOR;
            if (net > best) {
                best = net; bs = s; bp = p;
                bamp = (u32)alo | ((u32)ahi << 8);
            }
        }
    }
    out->type = T_HALFXOR; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* registry of selectable instructions */
typedef double (*SearchFn)(const u8 *, int, double, Instr *);
typedef struct { const char *name; SearchFn search; } InstrDesc;

static const InstrDesc REGISTRY[] = {
    { "XOR_PHASE",  search_xorp },
    { "NIB_SWAP",   search_nibsw },
    { "ADD_NIBS",   search_anibs },
    { "QUAD_ADD",   search_quadadd },
    { "PRNG_DUAL",  search_prngd },
    { "OCT_NIBX",   search_octnibx },
    { "STRIDE_ADD", search_strideadd },
    { "XOR_DELTA",  search_xordelta },
    { "PRNG_BIT",   search_prngbit },
    { "BYTE_ROT",   search_byterot },
    { "HALF_XOR",   search_halfxor },
};
#define NREG ((int)(sizeof(REGISTRY)/sizeof(REGISTRY[0])))
static const char *TYPE_NAME[NTYPES] = {
    "XOR_PHASE","NIB_SWAP","ADD_NIBS","QUAD_ADD","PRNG_DUAL","OCT_NIBX",
    "STRIDE_ADD","XOR_DELTA","PRNG_BIT","BYTE_ROT","HALF_XOR"
};

static int g_diag = 0;
/* best selectable instruction for the current data */
static double best_instr(const u8 *d, int n, Instr *out) {
    double Sb = S_of(d, n);
    double best = -1e18; Instr bi = {0};
    for (int r = 0; r < NREG; r++) {
        Instr cand;
        double net = REGISTRY[r].search(d, n, Sb, &cand);
        if (g_diag) printf("    [diag] %-12s net=%+.2f\n", REGISTRY[r].name, net);
        if (net > best) { best = net; bi = cand; }
    }
    *out = bi;
    return best;
}

/* ============================================================ *
 *  compress (greedy)                                            *
 * ============================================================ */

static double binary_entropy(double p) {
    if (p <= 0.0 || p >= 1.0) return 0.0;
    return -(p * log2(p) + (1.0 - p) * log2(1.0 - p));
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
        double e = entropy_bits(g_scr, n) / n;
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
        if (*ni >= MAXINSTR) break;
        Instr t;
        double net = best_instr(d, n, &t);
        if (net <= 0.0) break;
        double e0 = entropy_bits(d, n) / n;
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
                   e0, entropy_bits(d, n) / n, net, sum / n, top_v, top_c);
            print_layer_struct(d, n);
        }
    }
    return gained;
}

static double compress(u8 *d, int n, Instr *ilist, double *nets, int *ni, int verbose) {
    *ni = 0;
    if (verbose) {
        printf("  INPUT        (before any layer)         %.4f bps\n", entropy_bits(d, n) / n);
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
 *  main                                                         *
 * ============================================================ */

/* validate apply/invert for every instruction type independently */
static int selftest(void) {
    u8 base[BLOCK], a[BLOCK];
    u32 st = 99u;
    for (int i = 0; i < BLOCK; i++) { st = st * 1103515245u + 12345u; base[i] = (u8)(st >> 16); }

    Instr tv[] = {
        { T_XORP,     3, 1, 0x5A },
        { T_NIBSW,    3, 1, 0 },
        { T_ANIBS,    3, 1, 0x35 },
        { T_QUADADD,  3, 1, 0x11223344u },
        { T_PRNGD,    0, 0, (1234u|(5678u<<16)) },
        { T_OCTNIBX,  3, 1, 0x12345678u },
        { T_STRIDEADD,3, 1, 0x37 },
        { T_XORDELTA, 3, 1, 0 },
        { T_PRNGBIT,  0, 0, (0xABCDu | (0xA5u << 16)) },
        { T_BYTEROT,  3, 1, 3 },
        { T_HALFXOR,  3, 1, (0x2Au | (0x55u << 8)) },
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

/* reduce one block, verify round-trip, accumulate per-type counts + net stats */
static double do_block(u8 *data, int n, int *counts,
                       double *type_net_sum, double *type_net_max,
                       int verbose, int *ok_out) {
    u8 orig[BLOCK];
    memcpy(orig, data, n);
    static Instr ilist[MAXINSTR];
    static double nets[MAXINSTR];
    int ni = 0;
    double net = compress(data, n, ilist, nets, &ni, verbose);
    for (int i = 0; i < ni; i++) {
        int t = ilist[i].type;
        counts[t]++;
        type_net_sum[t] += nets[i];
        if (nets[i] > type_net_max[t]) type_net_max[t] = nets[i];
    }

    u8 dec[BLOCK];
    memcpy(dec, data, n);
    decompress(dec, n, ilist, ni);
    *ok_out = (memcmp(dec, orig, n) == 0);
    return net;
}

int main(int argc, char **argv) {
    init_hlog();

    if (argc > 1 && strcmp(argv[1], "selftest") == 0) return selftest() ? 2 : 0;

    /* ---- default: many BCrypt blocks, aggregate ---- */
    int NB = (argc > 1) ? atoi(argv[1]) : 1;
    if (NB < 1) NB = 1;
    u8 *all = malloc((size_t)NB * BLOCK);
    if (!all) { fprintf(stderr, "oom\n"); return 1; }

    size_t need = (size_t)NB * BLOCK, got = 0;
    FILE *f = fopen("bcrypt.bin", "rb");
    if (f) { got = fread(all, 1, need, f); fclose(f); }
    if (got == need) {
        printf("input: bcrypt.bin (cached BCrypt, %d blocks x %d bytes)\n", NB, BLOCK);
    } else {
        BCryptGenRandom(NULL, all, (ULONG)need, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
        f = fopen("bcrypt.bin", "wb");
        if (f) { fwrite(all, 1, need, f); fclose(f); }
        printf("input: fresh BCryptGenRandom (%d blocks x %d bytes, saved bcrypt.bin)\n", NB, BLOCK);
    }

    FILE *fcomp = fopen("compressed.bin", "wb");
    if (!fcomp) { fprintf(stderr, "cannot open compressed.bin\n"); return 1; }

    int counts[NTYPES] = {0};
    double type_net_sum[NTYPES] = {0};
    double type_net_max[NTYPES];
    for (int t = 0; t < NTYPES; t++) type_net_max[t] = 0.0;
    double total_net = 0.0, total_ein = 0.0, total_eout = 0.0;
    int fails = 0;
    clock_t t0 = clock();

    printf("\n=== compressing %d blocks ===\n", NB);
    for (int b = 0; b < NB; b++) {
        u8 *data = all + (size_t)b * BLOCK;
        double e_in = entropy_bits(data, BLOCK);
        int ok = 0;
        double net = do_block(data, BLOCK, counts, type_net_sum, type_net_max, NB == 1, &ok);
        double e_out = entropy_bits(data, BLOCK);
        total_net += net; total_ein += e_in; total_eout += e_out;
        if (!ok) fails++;
        printf("  block %2d: %.4f -> %.4f bps  net=%+.1f  %s\n",
               b, e_in / BLOCK, e_out / BLOCK, net, ok ? "ok" : "FAIL");
        fflush(stdout);

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
    printf("round-trip: %s (%d/%d blocks)\n", fails ? "*** FAIL ***" : "OK", NB - fails, NB);
    printf("types fired (across all blocks): %d / %d\n\n", fired, NTYPES);
    printf("  %-14s %6s  %8s  %8s\n", "type", "fires", "avg net", "top net");
    printf("  %-14s %6s  %8s  %8s\n", "----", "-----", "-------", "-------");
    for (int t = 0; t < NTYPES; t++) {
        if (counts[t] == 0) {
            printf("  %-14s %6d\n", TYPE_NAME[t], 0);
        } else {
            printf("  %-14s %6d  %+8.1f  %+8.1f\n",
                   TYPE_NAME[t], counts[t],
                   type_net_sum[t] / counts[t],
                   type_net_max[t]);
        }
    }

    free(all);
    return fails ? 2 : 0;
}
