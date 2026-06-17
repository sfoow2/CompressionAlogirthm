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

#define BLOCK      4096
#define MAXINSTR   8192
#define MAX_STRIDE 64           /* search strides 1..MAX_STRIDE */

/* globals visible to all search functions */
static int g_diag        = 0;
static int g_fast_search = 0;     /* when 1, skip slow searches in scramble lookahead */
static int g_stride_lim  = MAX_STRIDE; /* capped to 8 during lookahead (64× speedup)  */

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

    T_HALFADD,   /* value-conditional ADD mod 128: lo-half += amp_lo, hi-half += amp_hi     */
    T_BITCONDXOR,/* condition on bit k (0-6); XOR other 7 bits with mask_0 or mask_1        */
    T_BYTEMUL,   /* multiply each byte by odd constant a mod 256; a in 3..255 odd            */
    T_PRNGMSB,   /* PRNG 1-bit stream XOR'd into bit 7 of every byte; 16-bit seed            */
    T_LAGXOR,    /* data[i] ^= data[i+lag] left-to-right; amp=lag; no stride/phase           */
    T_ADDPHASE,  /* data[i] += amp mod 256 at (stride,phase) subset                          */
    T_TRIPLEXOR, /* 3-way pos split XOR: a0|(a1<<8)|(a2<<16)                                 */
    T_QUADXOR,   /* 4-way pos split XOR: a0|(a1<<8)|(a2<<16)|(a3<<24)                       */
    T_VALUEXOR,  /* bit-k conditional XOR: amp=k|(alo<<3)|(ahi<<11); preserves bit k         */
    T_SCRAMBLE,  /* block permutation; amp=type index 0..13                                  */
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
#define TAGB   5.0    /* 5 bits: ceil(log2(32)) = 5 bits, 32-slot type space                  */
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

#define OH_HALFADD      (TAGB + SB + PB + 7.0 + 7.0) /* amp_lo 7 bits + amp_hi 7 bits      */
#define OH_BITCONDXOR   (TAGB + SB + PB + 3.0 + 7.0 + 7.0) /* k (3 bits) + 2x 7-bit masks */
#define OH_BYTEMUL      (TAGB + SB + PB + 7.0)               /* odd multiplier a (7 bits)   */
#define OH_PRNGMSB      (TAGB + 16.0)                         /* 16-bit seed only, no stride */
#define OH_LAGXOR    (TAGB + 8.0)                             /* 8-bit lag, no stride/phase  */
#define OH_ADDPHASE  (TAGB + SB + PB + 8.0)                  /* same fields as XOR_PHASE    */
#define OH_TRIPLEXOR (TAGB + SB + PB + 24.0)                 /* 3x 8-bit XOR amps           */
#define OH_QUADXOR   (TAGB + SB + PB + 32.0)                 /* 4x 8-bit XOR amps           */
#define OH_VALUEXOR  (TAGB + SB + PB + 17.0)                 /* k(3)+alo(7)+ahi(7) bits     */
#define OH_SCRAMBLE  (TAGB + 4.0)                             /* 4-bit scramble type, no s/p */

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

/* PRNG_MSB: XOR bit 7 of every byte with a 1-bit PRNG stream. The xorshift16 state is
 * folded (x ^ (x>>8)) & 1 so all 16 seed bits influence every output bit. Self-inverse:
 * running the same seed again re-flips the same set of bit-7s, restoring the original. */
static void ap_prngmsb(u8 *d, int n, u32 amp) {
    u16 st = (u16)(amp & 0xFFFF);
    for (int i = 0; i < n; i++) {
        u16 x = st;
        x ^= x << 7; x ^= x >> 9; x ^= x << 8;
        st = x;
        d[i] ^= (u8)(((x ^ (x >> 8)) & 1) << 7);
    }
}

/* BYTE_MUL: multiply each byte by an odd constant a mod 256. All odd a are coprime to 256
 * so the map is a bijection. Inverse: multiply by a^{-1} mod 256, computed via 3 Newton
 * iterations (Hensel lift: x_{n+1} = x_n*(2-a*x_n) doubles precision each step).
 * Produces non-linear value permutations that XOR/ADD cannot replicate. amp = a (odd). */
static u8 mul_inv256(u8 a) {
    u8 x = 1;
    x = (u8)(x * (2 - a * x));  /* mod 4  */
    x = (u8)(x * (2 - a * x));  /* mod 16 */
    x = (u8)(x * (2 - a * x));  /* mod 256 */
    return x;
}
static void ap_bytemul(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)(amp & 0xFF);
    for (int i = p; i < n; i += s) d[i] = (u8)(d[i] * a);
}
static void inv_bytemul(u8 *d, int n, int s, int p, u32 amp) {
    u8 inv = mul_inv256((u8)(amp & 0xFF));
    for (int i = p; i < n; i += s) d[i] = (u8)(d[i] * inv);
}

/* BIT_COND_XOR: condition on bit k (k=0..6; k=7 is covered by HALF_XOR with lower overhead).
 * If bit k of the byte is 0: XOR the other 7 bits with mask_0.
 * If bit k of the byte is 1: XOR the other 7 bits with mask_1.
 * Bit k is never modified, so the decoder always knows which mask was applied → self-inverse.
 * Encoding: amp[2:0]=k, amp[9:3]=mask_0 (7-bit), amp[16:10]=mask_1 (7-bit).
 * Each mask is a 7-bit value; the full 8-bit XOR applied has bit k = 0, produced by
 * expand7(m, k): insert a 0 at position k into the 7-bit compact representation. */
static inline u8 bcx_expand(u8 m7, int k) {
    return (u8)((m7 & ((1 << k) - 1)) | ((m7 >> k) << (k + 1)));
}
static inline u8 bcx_compact(u8 v, int k) {
    return (u8)((v & ((1 << k) - 1)) | ((v >> (k + 1)) << k));
}
static void ap_bitcondxor(u8 *d, int n, int s, int p, u32 amp) {
    int k  = (int)(amp & 7);
    u8 m0  = bcx_expand((u8)((amp >>  3) & 0x7F), k);
    u8 m1  = bcx_expand((u8)((amp >> 10) & 0x7F), k);
    for (int i = p; i < n; i += s) {
        if ((d[i] >> k) & 1) d[i] ^= m1;
        else                  d[i] ^= m0;
    }
}

/* HALF_ADD: value-conditional ADD mod 128. Bytes 0-127 (bit7=0) add amp_lo mod 128;
 * bytes 128-255 (bit7=1) add amp_hi mod 128. Bit7 is never touched, so the decoder
 * always knows which branch to reverse. Unlike HALF_XOR (butterfly rearrangement),
 * HALF_ADD circularly rotates each half of the value histogram — different folding.
 * amp[6:0]=amp_lo, amp[13:7]=amp_hi. Inverse: subtract instead of add. */
static void ap_halfadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 lo = (u8)(amp & 0x7F), hi = (u8)((amp >> 7) & 0x7F);
    for (int i = p; i < n; i += s) {
        if (d[i] & 0x80) d[i] = (u8)(0x80 | ((d[i] + hi) & 0x7F));
        else              d[i] = (u8)((d[i] + lo) & 0x7F);
    }
}
static void inv_halfadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 lo = (u8)(amp & 0x7F), hi = (u8)((amp >> 7) & 0x7F);
    for (int i = p; i < n; i += s) {
        if (d[i] & 0x80) d[i] = (u8)(0x80 | ((d[i] - hi) & 0x7F));
        else              d[i] = (u8)((d[i] - lo) & 0x7F);
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

/* LAG_XOR: d[i] ^= d[i+lag] left-to-right. Inverse is right-to-left (each d[i+lag]
 * was never modified by a later step, so the same XOR undoes itself in reverse). */
static void ap_lagxor(u8 *d, int n, u32 amp) {
    int lag = (int)(amp & 0xFF);
    for (int i = 0; i + lag < n; i++) d[i] ^= d[i + lag];
}
static void inv_lagxor(u8 *d, int n, u32 amp) {
    int lag = (int)(amp & 0xFF);
    for (int i = n - lag - 1; i >= 0; i--) d[i] ^= d[i + lag];
}

/* ADD_PHASE: cyclic byte addition mod 256 at (stride,phase) subset. */
static void ap_addphase(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] += (u8)(amp & 0xFF);
}
static void inv_addphase(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] -= (u8)(amp & 0xFF);
}

/* TRIPLE_XOR: 3-way position split, independent XOR per group. Self-inverse. */
static void ap_triplexor(u8 *d, int n, int s, int p, u32 amp) {
    u8 a[3] = { (u8)(amp&0xFF), (u8)((amp>>8)&0xFF), (u8)((amp>>16)&0xFF) };
    int k = 0;
    for (int i = p; i < n; i += s, k++) d[i] ^= a[k % 3];
}

/* QUAD_XOR: 4-way position split, independent XOR per group. Self-inverse. */
static void ap_quadxor(u8 *d, int n, int s, int p, u32 amp) {
    u8 a[4] = { (u8)(amp&0xFF), (u8)((amp>>8)&0xFF),
                (u8)((amp>>16)&0xFF), (u8)(amp>>24) };
    int k = 0;
    for (int i = p; i < n; i += s, k++) d[i] ^= a[k & 3];
}

/* VALUE_XOR: XOR each byte with alo (if bit k=0) or ahi (if bit k=1).
 * alo and ahi both have bit k=0, so bit k of each byte is preserved → self-inverse. */
static void ap_valuexor(u8 *d, int n, int s, int p, u32 amp) {
    int k    = (int)(amp & 7);
    u8 mask  = (u8)(1 << k);
    u8 alo   = (u8)((amp >>  3) & 0xFF);
    u8 ahi   = (u8)((amp >> 11) & 0xFF);
    for (int i = p; i < n; i += s)
        d[i] ^= (u8)((d[i] & mask) ? ahi : alo);
}

/* ---- SCRAMBLE: block-level permutations ---------------------------------- */
static void sc_interleave(const u8 *src, u8 *dst, int n, int s) {
    int w = n / s;
    for (int i = 0; i < n; i++) dst[(i % s) * w + (i / s)] = src[i];
}
static void sc_bitplane(const u8 *src, u8 *dst, int n) {
    int ps = n / 8;
    memset(dst, 0, n);
    for (int i = 0; i < n; i++)
        for (int b = 0; b < 8; b++)
            if ((src[i] >> b) & 1)
                dst[b * ps + i/8] |= (u8)(1 << (i % 8));
}
static void sc_nibplane(const u8 *src, u8 *dst, int n) {
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        dst[i]      = (u8)((src[2*i] & 0x0F) | ((src[2*i+1] & 0x0F) << 4));
        dst[i+half] = (u8)((src[2*i] >> 4)   | ((src[2*i+1] >> 4)   << 4));
    }
}
static void sc_blkrev(const u8 *src, u8 *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[n-1-i];
}
/* Inverse of sc_bitplane: reconstruct original bytes from bit-plane layout.
 * src[b*ps + i/8] bit (i%8) holds src_orig[i] bit b → reverse: for each i, collect
 * each bit b from its bit-plane byte back into the original byte at position i. */
static void inv_sc_bitplane(const u8 *src, u8 *dst, int n) {
    int ps = n / 8;
    memset(dst, 0, n);
    for (int i = 0; i < n; i++)
        for (int b = 0; b < 8; b++)
            if ((src[b * ps + i/8] >> (i % 8)) & 1)
                dst[i] |= (u8)(1 << b);
}
static void sc_xorfold(const u8 *src, u8 *dst, int n) {
    int h = n / 2;
    for (int i = 0; i < h; i++) dst[i] = src[i] ^ src[i+h];
    memcpy(dst+h, src+h, h);
}

/* Apply scramble type si in-place via a stack buffer. Returns 1 on success, 0 if n
 * doesn't meet the divisibility requirement for that scramble. */
static int ap_scramble_si(int si, u8 *d, int n) {
    u8 tmp[BLOCK];
    switch (si) {
        case  0: sc_interleave(d, tmp, n, 2);    break;
        case  1: sc_interleave(d, tmp, n, 4);    break;
        case  2: sc_interleave(d, tmp, n, 8);    break;
        case  3: sc_interleave(d, tmp, n, 16);   break;
        case  4: sc_interleave(d, tmp, n, 32);   break;
        case  5: sc_interleave(d, tmp, n, 64);   break;
        case  6: sc_bitplane(d, tmp, n);          break;
        case  7: sc_nibplane(d, tmp, n);          break;
        case  8: sc_blkrev(d, tmp, n);            break;
        case  9: sc_xorfold(d, tmp, n);           break;
        case 10: sc_interleave(d, tmp, n, 1024);  break;
        case 11: sc_interleave(d, tmp, n, 512);   break;
        case 12: sc_interleave(d, tmp, n, 256);   break;
        case 13: sc_interleave(d, tmp, n, 128);   break;
        default: return 0;
    }
    memcpy(d, tmp, n);
    return 1;
}

/* Inverse: IL-s inverse = IL-(n/s); bit-plane, block-rev, xor-fold are self-inverse;
 * nibble-plane has its own inverse. For n=4096 this covers all 14 scramble types. */
static void inv_scramble_si(int si, u8 *d, int n) {
    u8 tmp[BLOCK];
    switch (si) {
        case 0: sc_interleave(d, tmp, n, n/2); memcpy(d, tmp, n); break; /* IL-2048 */
        case 1:  ap_scramble_si(10, d, n); break;  /* IL-1024 */
        case 2:  ap_scramble_si(11, d, n); break;  /* IL-512  */
        case 3:  ap_scramble_si(12, d, n); break;  /* IL-256  */
        case 4:  ap_scramble_si(13, d, n); break;  /* IL-128  */
        case 5:  ap_scramble_si( 5, d, n); break;  /* IL-64 self-inv */
        case 6: { u8 tmp2[BLOCK]; inv_sc_bitplane(d, tmp2, n); memcpy(d, tmp2, n); break; }
        case 7: {                                   /* nibble-plane special inverse */
            int half = n/2;
            for (int i = 0; i < half; i++) {
                tmp[2*i]   = (u8)((d[i] & 0xF) | ((d[i+half] & 0xF) << 4));
                tmp[2*i+1] = (u8)(((d[i]>>4) & 0xF) | (((d[i+half]>>4) & 0xF) << 4));
            }
            memcpy(d, tmp, n);
            break;
        }
        case 8:  ap_scramble_si( 8, d, n); break;  /* block-rev self-inv */
        case 9:  ap_scramble_si( 9, d, n); break;  /* xor-fold self-inv */
        case 10: ap_scramble_si( 1, d, n); break;  /* IL-4 */
        case 11: ap_scramble_si( 2, d, n); break;  /* IL-8 */
        case 12: ap_scramble_si( 3, d, n); break;  /* IL-16 */
        case 13: ap_scramble_si( 4, d, n); break;  /* IL-32 */
    }
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
        case T_HALFADD:   ap_halfadd(d, n, t.stride, t.phase, t.amp); break;
        case T_BITCONDXOR:ap_bitcondxor(d, n, t.stride, t.phase, t.amp); break;
        case T_BYTEMUL:   ap_bytemul(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGMSB:   ap_prngmsb(d, n, t.amp); break;
        case T_LAGXOR:    ap_lagxor(d, n, t.amp); break;
        case T_ADDPHASE:  ap_addphase(d, n, t.stride, t.phase, t.amp); break;
        case T_TRIPLEXOR: ap_triplexor(d, n, t.stride, t.phase, t.amp); break;
        case T_QUADXOR:   ap_quadxor(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEXOR:  ap_valuexor(d, n, t.stride, t.phase, t.amp); break;
        case T_SCRAMBLE:  ap_scramble_si((int)t.amp, d, n); break;
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
        case T_HALFADD:   inv_halfadd(d, n, t.stride, t.phase, t.amp); break;
        case T_BITCONDXOR:ap_bitcondxor(d, n, t.stride, t.phase, t.amp); break; /* self-inv */
        case T_BYTEMUL:   inv_bytemul(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGMSB:   ap_prngmsb(d, n, t.amp); break;                      /* self-inv */
        case T_LAGXOR:    inv_lagxor(d, n, t.amp); break;
        case T_ADDPHASE:  inv_addphase(d, n, t.stride, t.phase, t.amp); break;
        case T_TRIPLEXOR: ap_triplexor(d, n, t.stride, t.phase, t.amp); break; /* self-inv */
        case T_QUADXOR:   ap_quadxor(d, n, t.stride, t.phase, t.amp); break;   /* self-inv */
        case T_VALUEXOR:  ap_valuexor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_SCRAMBLE:  inv_scramble_si((int)t.amp, d, n); break;
    }
}

/* ============================================================ *
 *  searches: each fills *out and returns best net (may be <0)   *
 * ============================================================ */

#define PRNG_SEEDS 65536        /* try seeds 1..65535 */

/* XOR / ADD at (stride,phase) via the frequency-table trick: O(256) per amp */
static double search_xorp(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
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
    for (int s = 1; s <= g_stride_lim; s++) {
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
    for (int s = 1; s <= g_stride_lim; s++) {
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
    for (int s = 1; s <= g_stride_lim; s++) {
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
    for (int s = 1; s <= g_stride_lim; s++) {
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
    for (int s = 1; s <= g_stride_lim; s++) {
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
    for (int s = 1; s <= g_stride_lim; s++) {
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
    for (int s = 1; s <= g_stride_lim; s++) {
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
    for (int s = 1; s <= g_stride_lim; s++) {
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



/* HALF_ADD: freq-table trick for ADD-mod-128 within each half. For a shift of `a`,
 * the subset element that ends up at value v came from (v-a) mod 128 in the same half. */
static double search_halfadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        for (int p = 0; p < s; p++) {
            int flo[128] = {0}, fhi[128] = {0};
            for (int i = p; i < n; i += s) {
                u8 v = d[i];
                if (v & 0x80) fhi[v & 0x7F]++; else flo[v]++;
            }
            int blo[128], bhi[128];
            for (int v = 0; v < 128; v++) {
                blo[v] = total[v]       - flo[v];
                bhi[v] = total[v + 128] - fhi[v];
            }
            int alo = 0; double bSlo = -1e18;
            for (int a = 0; a < 128; a++) {
                double S = 0.0;
                for (int v = 0; v < 128; v++) S += hlog[blo[v] + flo[(v - a + 128) & 0x7F]];
                if (S > bSlo) { bSlo = S; alo = a; }
            }
            int ahi = 0; double bShi = -1e18;
            for (int a = 0; a < 128; a++) {
                double S = 0.0;
                for (int v = 0; v < 128; v++) S += hlog[bhi[v] + fhi[(v - a + 128) & 0x7F]];
                if (S > bShi) { bShi = S; ahi = a; }
            }
            double net = (bSlo + bShi - Sb) - OH_HALFADD;
            if (net > best) { best = net; bs = s; bp = p; bamp = (u32)alo | ((u32)ahi << 7); }
        }
    }
    out->type = T_HALFADD; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* BYTE_MUL: frequency-table trick — for each odd multiplier a, the subset element that
 * ends up at value w came from w * a^{-1} mod 256. We enumerate all 127 non-trivial odd
 * multipliers (skip a=1 = identity) and find the one with best net. */
static double search_bytemul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 3;
    for (int s = 1; s <= g_stride_lim; s++) {
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 3; a < 256; a += 2) {  /* skip a=1 (identity) */
                int rf[256];
                memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[(u * a) & 0xFF] += hit[u];
                double net = (S_from_freq(rf) - Sb) - OH_BYTEMUL;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->type = T_BYTEMUL; out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* BIT_COND_XOR: for each (s,p,k), build compact hit tables for the two bit-k halves,
 * then find the best 7-bit XOR mask for each half independently (disjoint value ranges). */
static double search_bitcondxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            for (int k = 0; k < 7; k++) {   /* k=7 already covered by HALF_XOR */
                /* compact hit into two 128-entry tables by bit k */
                int h0[128] = {0}, h1[128] = {0};
                for (int v = 0; v < 256; v++) {
                    if (!hit[v]) continue;
                    if ((v >> k) & 1) h1[bcx_compact((u8)v, k)] += hit[v];
                    else              h0[bcx_compact((u8)v, k)] += hit[v];
                }
                /* base counts for non-subset bytes in each half */
                int b0[128], b1[128];
                for (int v = 0; v < 128; v++) {
                    u8 fv0 = bcx_expand((u8)v, k);
                    u8 fv1 = fv0 | (u8)(1 << k);
                    b0[v] = total[fv0] - h0[v];
                    b1[v] = total[fv1] - h1[v];
                }
                /* find best 7-bit XOR mask per half */
                int bm0 = 0; double bS0 = -1e18;
                for (int m = 0; m < 128; m++) {
                    double S = 0.0;
                    for (int v = 0; v < 128; v++) S += hlog[b0[v] + h0[v ^ m]];
                    if (S > bS0) { bS0 = S; bm0 = m; }
                }
                int bm1 = 0; double bS1 = -1e18;
                for (int m = 0; m < 128; m++) {
                    double S = 0.0;
                    for (int v = 0; v < 128; v++) S += hlog[b1[v] + h1[v ^ m]];
                    if (S > bS1) { bS1 = S; bm1 = m; }
                }
                double net = (bS0 + bS1 - Sb) - OH_BITCONDXOR;
                if (net > best) {
                    best = net; bs = s; bp = p;
                    bamp = (u32)k | ((u32)bm0 << 3) | ((u32)bm1 << 10);
                }
            }
        }
    }
    out->type = T_BITCONDXOR; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* PRNG_MSB: scan all 65535 seeds. For each seed, one pass builds g1[v] = count of bytes
 * at value v that get their bit7 flipped (PRNG output = 1). Then S is computed by
 * swapping each pair (v, v^0x80) proportionally. Only 16 bits of overhead since there
 * is no flip-mask (bit7 is the only target) and no stride/phase. */
static double search_prngmsb(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18;
    u16 best_seed = 1;
    for (u32 s = 1; s < 65536; s++) {
        u16 st = (u16)s;
        int g1[256] = {0};
        for (int i = 0; i < n; i++) {
            u16 x = st;
            x ^= x << 7; x ^= x >> 9; x ^= x << 8;
            st = x;
            if ((x ^ (x >> 8)) & 1) g1[d[i]]++;
        }
        /* new_freq[v] = (total[v] - g1[v]) + g1[v^0x80] */
        int f[256];
        for (int v = 0; v < 256; v++) f[v] = total[v];
        for (int v = 0; v < 128; v++) {
            int glo = g1[v], ghi = g1[v + 128];
            f[v]       += ghi - glo;
            f[v + 128] += glo - ghi;
        }
        double net = (S_from_freq(f) - Sb) - OH_PRNGMSB;
        if (net > best) { best = net; best_seed = (u16)s; }
    }
    out->type = T_PRNGMSB; out->stride = 0; out->phase = 0; out->amp = best_seed;
    return best;
}

/* ---- Walsh-Hadamard Transform helpers (for fast XOR-amp search) ---- */
static void wht256(int *a) {
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += len<<1)
            for (int j = 0; j < len; j++) {
                int u = a[i+j], v = a[i+j+len];
                a[i+j] = u+v; a[i+j+len] = u-v;
            }
}
/* XOR-correlation proxy: find best amp in 1..255 for Σ hlog[A[v]+B[v^amp]].
 * WHT gives proxy scores; top-3 are exact-verified with hlog. */
static int xor_best_wht(const int *A, const int *B, double *Sout) {
    int ha[256], hb[256];
    memcpy(ha, A, 256*sizeof(int));
    memcpy(hb, B, 256*sizeof(int));
    wht256(ha); wht256(hb);
    long long prod[256];
    for (int k = 0; k < 256; k++) prod[k] = (long long)ha[k]*hb[k];
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += len<<1)
            for (int j = 0; j < len; j++) {
                long long u = prod[i+j], v = prod[i+j+len];
                prod[i+j] = u+v; prod[i+j+len] = u-v;
            }
    long long c0 = -(1LL<<32), c1 = c0, c2 = c0;
    int a0 = 1, a1 = 2, a2 = 3;
    for (int amp = 1; amp < 256; amp++) {
        long long c = prod[amp];
        if      (c > c0) { c2=c1;a2=a1; c1=c0;a1=a0; c0=c;a0=amp; }
        else if (c > c1) { c2=c1;a2=a1; c1=c;a1=amp; }
        else if (c > c2) { c2=c;a2=amp; }
    }
    int best = a0; double bestS = -1e30;
    int cands[3] = {a0, a1, a2};
    for (int t = 0; t < 3; t++) {
        double S = 0.0;
        for (int v = 0; v < 256; v++) S += hlog[A[v]+B[v^cands[t]]];
        if (S > bestS) { bestS = S; best = cands[t]; }
    }
    *Sout = bestS;
    return best;
}

/* LAG_XOR: one O(n) pass per lag, up to 255 lags. O(255n) total — very fast. */
static double search_lagxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int blag = 1;
    for (int lag = 1; lag <= 255 && lag < n; lag++) {
        int f[256];
        memcpy(f, total, sizeof f);
        for (int i = 0; i + lag < n; i++) { f[d[i]]--; f[d[i] ^ d[i+lag]]++; }
        double net = (S_from_freq(f) - Sb) - OH_LAGXOR;
        if (net > best) { best = net; blag = lag; }
    }
    out->type = T_LAGXOR; out->stride = 0; out->phase = 0; out->amp = (u32)blag;
    return best;
}

/* ADD_PHASE: brute-force 255 cyclic-add amps at each (stride,phase). */
static double search_addphase(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 1;
    for (int s = 1; s <= g_stride_lim; s++) {
        for (int p = 0; p < s; p++) {
            int phF[256]={0};
            for (int i = p; i < n; i += s) phF[d[i]]++;
            int dv[256];
            for (int v = 0; v < 256; v++) dv[v] = total[v] - phF[v];
            int rf[256];
            for (int a = 1; a < 256; a++) {
                for (int v = 0; v < 256; v++) rf[v] = dv[v] + phF[(v-a)&0xFF];
                double net = (S_from_freq(rf) - Sb) - OH_ADDPHASE;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->type = T_ADDPHASE; out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* TRIPLE_XOR: 3-pass coordinate descent, each pass uses WHT to find best XOR amp. */
static double search_triplexor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        for (int p = 0; p < s; p++) {
            int phF0[256]={0}, phF1[256]={0}, phF2[256]={0};
            int k = 0;
            for (int i = p; i < n; i += s, k++) {
                if      (k%3==0) phF0[d[i]]++;
                else if (k%3==1) phF1[d[i]]++;
                else             phF2[d[i]]++;
            }
            int td0[256], tx1[256], dx1[256], tx2[256], dx2[256];
            for (int v=0;v<256;v++) td0[v]=total[v]-phF0[v];
            double S1; int a0 = xor_best_wht(td0, phF0, &S1);
            for (int v=0;v<256;v++) tx1[v]=td0[v]+phF0[v^a0];
            for (int v=0;v<256;v++) dx1[v]=tx1[v]-phF1[v];
            double S2; int a1 = xor_best_wht(dx1, phF1, &S2);
            for (int v=0;v<256;v++) tx2[v]=dx1[v]+phF1[v^a1];
            for (int v=0;v<256;v++) dx2[v]=tx2[v]-phF2[v];
            double S3; int a2 = xor_best_wht(dx2, phF2, &S3);
            double net = (S3 - Sb) - OH_TRIPLEXOR;
            if (net > best) { best=net; bs=s; bp=p; bamp=(u32)(a0|(a1<<8)|(a2<<16)); }
        }
    }
    out->type=T_TRIPLEXOR; out->stride=bs; out->phase=bp; out->amp=bamp;
    return best;
}

/* QUAD_XOR: 4-pass coordinate descent via WHT. */
static double search_quadxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        for (int p = 0; p < s; p++) {
            int phF0[256]={0}, phF1[256]={0}, phF2[256]={0}, phF3[256]={0};
            int k = 0;
            for (int i = p; i < n; i += s, k++) {
                switch (k&3) {
                    case 0: phF0[d[i]]++; break; case 1: phF1[d[i]]++; break;
                    case 2: phF2[d[i]]++; break; default: phF3[d[i]]++; break;
                }
            }
            int td0[256], tx1[256], dx1[256], tx2[256], dx2[256], tx3[256], dx3[256];
            for (int v=0;v<256;v++) td0[v]=total[v]-phF0[v];
            double S1; int a0=xor_best_wht(td0,phF0,&S1);
            for (int v=0;v<256;v++) tx1[v]=td0[v]+phF0[v^a0];
            for (int v=0;v<256;v++) dx1[v]=tx1[v]-phF1[v];
            double S2; int a1=xor_best_wht(dx1,phF1,&S2);
            for (int v=0;v<256;v++) tx2[v]=dx1[v]+phF1[v^a1];
            for (int v=0;v<256;v++) dx2[v]=tx2[v]-phF2[v];
            double S3; int a2=xor_best_wht(dx2,phF2,&S3);
            for (int v=0;v<256;v++) tx3[v]=dx2[v]+phF2[v^a2];
            for (int v=0;v<256;v++) dx3[v]=tx3[v]-phF3[v];
            double S4; int a3=xor_best_wht(dx3,phF3,&S4);
            double net = (S4 - Sb) - OH_QUADXOR;
            if (net > best) {
                best=net; bs=s; bp=p;
                bamp=(u32)a0|((u32)a1<<8)|((u32)a2<<16)|((u32)a3<<24);
            }
        }
    }
    out->type=T_QUADXOR; out->stride=bs; out->phase=bp; out->amp=bamp;
    return best;
}

/* VALUE_XOR: for each bit position k, find best alo (for bit-k=0 group) and ahi (for
 * bit-k=1 group) independently — the two groups are XOR-closed under amps with bit k=0,
 * so they don't interact. Brute-force 127 amps × 128 values per group. */
static double search_valuexor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs=1, bp=0; u32 bamp=0;
    for (int s = 1; s <= g_stride_lim; s++) {
        for (int p = 0; p < s; p++) {
            int phF[256]={0};
            for (int i = p; i < n; i += s) phF[d[i]]++;
            int dv[256];
            for (int v=0;v<256;v++) dv[v]=total[v]-phF[v];
            for (int k = 0; k < 8; k++) {
                int mask = 1 << k;
                /* best alo: amp with bit k=0 for the bit-k=0 value group */
                int balo=0; double bS0=-1e30;
                for (int a=0; a<256; a++) {
                    if ((a>>k)&1) continue;
                    if (a==0) continue;
                    double S=0.0;
                    for (int v=0;v<256;v++) if (!((v>>k)&1)) S+=hlog[dv[v]+phF[v^a]];
                    if (S>bS0){bS0=S;balo=a;}
                }
                /* best ahi: amp with bit k=0 for the bit-k=1 value group */
                int bahi=0; double bS1=-1e30;
                for (int a=0; a<256; a++) {
                    if ((a>>k)&1) continue;
                    if (a==0) continue;
                    double S=0.0;
                    for (int v=0;v<256;v++) if ((v>>k)&1) S+=hlog[dv[v]+phF[v^a]];
                    if (S>bS1){bS1=S;bahi=a;}
                }
                /* combined S for (balo, bahi) */
                double S=0.0;
                for (int v=0;v<256;v++) {
                    int a = ((v>>k)&1) ? bahi : balo;
                    S += hlog[dv[v]+phF[v^a]];
                }
                double net = (S - Sb) - OH_VALUEXOR;
                if (net > best) {
                    best=net; bs=s; bp=p;
                    bamp=(u32)(k|(balo<<3)|(bahi<<11));
                }
            }
        }
    }
    out->type=T_VALUEXOR; out->stride=bs; out->phase=bp; out->amp=bamp;
    return best;
}

/* registry of selectable instructions */
typedef double (*SearchFn)(const u8 *, int, double, Instr *);
typedef struct { const char *name; SearchFn search; int slow; } InstrDesc;

static const InstrDesc REGISTRY[] = {
    { "XOR_PHASE",  search_xorp,      0 },
    { "NIB_SWAP",   search_nibsw,     0 },
    { "ADD_NIBS",   search_anibs,     0 },
    { "QUAD_ADD",   search_quadadd,   0 },
    { "PRNG_DUAL",  search_prngd,     1 },   /* slow: 65535-seed scan */
    { "OCT_NIBX",   search_octnibx,   0 },
    { "STRIDE_ADD", search_strideadd, 0 },
    { "XOR_DELTA",  search_xordelta,  0 },
    { "PRNG_BIT",   search_prngbit,   1 },   /* slow: 65535-seed scan */
    { "BYTE_ROT",   search_byterot,   0 },
    { "HALF_XOR",   search_halfxor,   0 },
    { "HALF_ADD",     search_halfadd,    0 },
    { "BITCONDXOR",   search_bitcondxor, 0 },
    { "BYTE_MUL",     search_bytemul,    0 },
    { "PRNG_MSB",     search_prngmsb,    1 }, /* slow: 65535-seed scan */
    { "LAG_XOR",      search_lagxor,     0 },
    { "ADD_PHASE",    search_addphase,   0 },
    { "TRIPLE_XOR",   search_triplexor,  0 },
    { "QUAD_XOR",     search_quadxor,    0 },
    { "VALUE_XOR",    search_valuexor,   1 }, /* slow: O(stride²×8×127×128) brute force */
    /* T_SCRAMBLE has no search entry — handled by try_scramble() */
};
#define NREG ((int)(sizeof(REGISTRY)/sizeof(REGISTRY[0])))
static const char *TYPE_NAME[NTYPES] = {
    "XOR_PHASE","NIB_SWAP","ADD_NIBS","QUAD_ADD","PRNG_DUAL","OCT_NIBX",
    "STRIDE_ADD","XOR_DELTA","PRNG_BIT","BYTE_ROT","HALF_XOR","HALF_ADD","BITCONDXOR","BYTE_MUL",
    "PRNG_MSB","LAG_XOR","ADD_PHASE","TRIPLE_XOR","QUAD_XOR","VALUE_XOR","SCRAMBLE"
};


/* best selectable instruction for the current data */
static double best_instr(const u8 *d, int n, Instr *out) {
    double Sb = S_of(d, n);
    double best = -1e18; Instr bi = {0};
    for (int r = 0; r < NREG; r++) {
        if (g_fast_search && REGISTRY[r].slow) continue;
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

/* Last-resort scramble: when greedy is stuck, try each of the 14 block permutations.
 * For each, run a 5-iteration fast-mode lookahead to estimate unlocked gain. Apply the
 * best if (entropy_delta + unlocked_net - overhead) > 0. Returns 1 if a scramble was
 * applied, 0 if none helped. */
static int try_scramble(u8 *d, int n, Instr *ilist, double *nets, int *ni, int verbose) {
    int fbase[256]; freq_of(d, n, fbase);
    double Sb = S_from_freq(fbase);
    int best_si = -1;
    double best_gain = 0.0, best_edelta = 0.0;

    g_fast_search = 1;
    g_stride_lim  = 8;   /* cap stride in lookahead to keep each search call ~1ms */
    for (int si = 0; si < 14; si++) {
        u8 scbuf[BLOCK];
        memcpy(scbuf, d, n);
        if (!ap_scramble_si(si, scbuf, n)) continue;

        int fsc[256]; freq_of(scbuf, n, fsc);
        double edelta = S_from_freq(fsc) - Sb;

        /* 5-iter lookahead (fast mode: no slow PRNG searches) */
        u8 tmp[BLOCK]; memcpy(tmp, scbuf, n);
        double unlocked = 0.0;
        for (int iter = 0; iter < 5; iter++) {
            Instr t; double net = best_instr(tmp, n, &t);
            if (net <= 0.0) break;
            apply_instr(tmp, n, t);
            unlocked += net;
        }

        double gain = edelta + unlocked - OH_SCRAMBLE;
        if (gain > best_gain) { best_gain = gain; best_si = si; best_edelta = edelta; }
    }
    g_fast_search = 0;
    g_stride_lim  = MAX_STRIDE;

    if (best_si < 0) return 0;

    ap_scramble_si(best_si, d, n);
    Instr sc = { T_SCRAMBLE, 0, 0, (u32)best_si };
    if (*ni < MAXINSTR) { nets[*ni] = best_edelta - OH_SCRAMBLE; ilist[(*ni)++] = sc; }
    if (verbose)
        printf("  %-12s si=%-3d              %.4f bps  net=%+.1f\n",
               "SCRAMBLE", best_si, entropy_bits(d, n) / n, best_edelta - OH_SCRAMBLE);
    return 1;
}

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
    double total = greedy_run(d, n, ilist, nets, ni, verbose);
    /* after greedy stalls, try block permutations to unlock more gains */
    for (;;) {
        if (!try_scramble(d, n, ilist, nets, ni, verbose)) break;
        total += greedy_run(d, n, ilist, nets, ni, verbose);
    }
    return total;
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

        { T_HALFADD,    3, 1, (0x13u | (0x41u << 7)) },
        { T_BITCONDXOR, 3, 1, (3u | (0x55u << 3) | (0x2Au << 10)) },
        { T_BYTEMUL,    3, 1, 3u },
        { T_PRNGMSB,    0, 0, 0x1A2Bu },
        { T_LAGXOR,     0, 0, 46u },
        { T_ADDPHASE,   3, 1, 0x37u },
        { T_TRIPLEXOR,  3, 1, 0x112233u },
        { T_QUADXOR,    3, 1, 0x11223344u },
        { T_VALUEXOR,   3, 1, (3u|(0x24u<<3)|(0x12u<<11)) }, /* k=3, alo=0x24 bit3=0, ahi=0x12 bit3=0 */
        { T_SCRAMBLE,   0, 0, 0u },   /* IL-2 */
        { T_SCRAMBLE,   0, 0, 6u },   /* bit-plane */
        { T_SCRAMBLE,   0, 0, 8u },   /* block-reverse */
        { T_SCRAMBLE,   0, 0, 9u },   /* xor-fold */
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
