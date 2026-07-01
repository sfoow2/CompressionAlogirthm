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

#define BLOCK       4096
#define MAXINSTR   8192
#define MAX_STRIDE 256          /* search strides 1..MAX_STRIDE */

/* globals visible to all search functions */
static int g_diag        = 0;
static int g_stride_lim  = MAX_STRIDE;

/* ---- instruction type ids ---- */
/* Tried and removed (0 fires on BCrypt random data in GPU greedy search):
 *   BP_XOR     — 0 fires; bit-plane cross-XOR needs bit-plane correlation (absent in random)
 *   PRNG_ADD   — 0 fires; dual-seed ADD superseded by PRNG_ADD4/8 with lower overhead
 *   NIBBLE_LUT — bijection of whole block → S is invariant; can never improve entropy
 *   PLANE_DELTA — 3 fires out of 1000 blocks; random data has no bit-plane run structure
 *   RANGE_XOR  — 0 fires; 43-bit overhead too large to clear on 4096-byte random blocks
 */
enum {
    T_XORP = 0,  /* XOR amp at (stride,phase) subset                                        */
    T_ANIBS,     /* nibble-wise add mod 16 (lo+=al, hi+=ah); amp=al|(ah<<4)                 */
    T_STRIDEADD, /* ADD amp (nonzero) at (stride,phase) subset                              */
    T_BYTEROT,   /* circular bit rotation left by k=1..7; amp=k                             */
    T_BYTEMUL,   /* multiply each byte by odd constant a mod 256; amp=a (odd, 3..255)       */
    T_VALUEXOR,  /* bit-k conditional XOR: amp=k|(alo<<3)|(ahi<<11); preserves bit k        */
    T_BITREV,    /* reverse bit order within each byte; self-inverse; stride/phase           */
    T_PRNGADD4,  /* 4-stream PRNG add: pos%4 selects stream, all derived from 1 master seed */
    T_PRNGADD8,  /* 8-stream PRNG add: pos%8 selects stream, aligns with 8-byte Blowfish    */
    T_NIBCXOR,   /* cross-nibble XOR: amp=0→lo^=hi, amp=1→hi^=lo; self-inverse; stride/phase */
    T_CRMBCXOR,  /* cross-crumb XOR: XOR 2-bit crumb k with crumb j; amp=j|(k<<2); self-inv  */
    T_GRAYCODE,  /* Gray code: v→v^(v>>1); stride/phase                                     */
    T_BITASWAP,  /* swap adjacent bits (v&0xAA)>>1|(v&0x55)<<1; self-inverse; stride/phase  */
    T_PLANEPRNG, /* PRNG XOR bit plane k with xs16 LSB; amp=seed|(k<<16); self-inverse       */
    T_REFLECT,   /* Elias remap around mode M: r=(v-M) signed, out=r>=0?2r:(-2r-1); amp=M   */
    T_SPLITADD,  /* N-way split ADD; kidx in phase[8..9]: 0=dual,1=quad,2=octo,3=half        */
    T_SPLITXOR,  /* N-way split XOR; kidx in phase[8..9]: 0=dual,1=quad,2=octo-nibble,3=half */
    T_PRNGBIT,   /* PRNG selects bit pos (0-7) per byte; XOR that bit with (amp>>bit_pos)&1  */
    T_PRNGXOR8,  /* 8-stream PRNG XOR; pos%8 selects stream; self-inverse; amp=seed           */
    T_VALUEMAP4, /* 4-quartile value-conditional lo6 XOR; top-2 bits preserved -> self-inverse */
    T_SPLITBYTEMUL, /* N-way stride-split multiply by odd constants; kidx in phase[8]         */
    T_ROTXOR,    /* compound rotate-left-k then XOR-c; amp=(k-1)|(c<<3); inv=xor-then-rotright */
    T_NIBSWAP,        /* swap hi/lo nibbles v->(v<<4)|(v>>4); self-inverse; stride/phase              */
    T_XORPNP,    /* XOR amp at stride, always phase=0; no phase field stored — saves log2(s) bits  */
    T_DELTA,     /* delta: d[i] -= d[i-stride] (backwards pass); inv: cumsum forward               */
    T_DELTA2,    /* 2nd-order delta (Laplacian): d[i] -= 2*d[i-s] - d[i-2s]; inv: cumsum×2       */
    T_MTF,       /* Move-to-Front remap; inv: MTF decode; no params — overhead = TAGB only        */
    T_PRNGBIT3,  /* 3 PRNG_BIT passes batched; fmask=0xFF; amp=seed1|(seed2<<16), stride=seed3  */
    NTYPES       /* = 28 */
};

typedef struct { u8 type; int stride, phase; u32 amp; u8 consts[256]; } Instr;

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
#define TAGB   6.0    /* 6 bits: ceil(log2(33)) = 6 bits, 33-slot type space */
#define SB     8.0    /* stride field (1..256 = 8 bits) */
/* Phase bits are stride-adaptive: log2(s) instead of fixed 6. Use OH_SP(base, s). */
static inline double pb_bits(int s) { return (s > 1) ? log2((double)s) : 0.0; }
#define OH_SP(base, s)  ((base) + pb_bits(s))
#define OH_XORP_BASE    (TAGB + SB + 8.0)
#define OH_NIBS_BASE    (TAGB + SB + 8.0)
#define OH_STRIDEADD_BASE (TAGB + SB + 8.0)
#define OH_BYTEROT_BASE (TAGB + SB + 3.0)
#define OH_BYTEMUL_BASE (TAGB + SB + 7.0)
#define OH_VALUEXOR_BASE  (TAGB + SB + 17.0)
#define OH_BITREV_BASE    (TAGB + SB)
#define OH_PRNGADD4       (TAGB + 16.0)
#define OH_PRNGADD8       (TAGB + 16.0)
#define OH_NIBCXOR_BASE   (TAGB + SB + 1.0)
#define OH_CRMBCXOR_BASE  (TAGB + SB + 4.0)
#define OH_GRAYCODE_BASE  (TAGB + SB)
#define OH_BITASWAP_BASE  (TAGB + SB)
#define OH_PLANEPRNG      (TAGB + 19.0)  /* 16-bit seed + 3-bit plane index k */
/* SPLIT_ADD: TAGB + SB + 2(kidx) + amp_bits(kidx) + pb_bits(s)
 * kidx=0 dual:2×8=16  kidx=1 quad:4×8=32  kidx=2 octo:8×4=32  kidx=3 half:2×7=14 */
static const double SPLITADD_AMP_BITS[4] = {16.0, 32.0, 32.0, 14.0};
static inline double oh_splitadd(int kidx, int s) {
    return TAGB + SB + 2.0 + SPLITADD_AMP_BITS[kidx] + pb_bits(s);
}
static const double SPLITXOR_AMP_BITS[4] = {16.0, 32.0, 32.0, 14.0};
static inline double oh_splitxor(int kidx, int s) {
    return TAGB + SB + 2.0 + SPLITXOR_AMP_BITS[kidx] + pb_bits(s);
}
#define OH_PRNGBIT      (TAGB + 16.0 + 8.0)  /* 16-bit seed + 8-bit flip-mask */
#define OH_PRNGXOR8    (TAGB + 16.0)
#define OH_NIBSWAP_BASE (TAGB + SB)
static double oh_splitbytemul(int kidx, int s) {
    return TAGB + SB + 1.0 + pb_bits(s) + (kidx == 0 ? 14.0 : 28.0);
}
static double oh_rotxor(int s)      { return TAGB + SB + pb_bits(s) + 11.0; } /* 3-bit k + 8-bit c */
#define OH_DELTA_BASE  (TAGB + SB)  /* type tag + stride only, no phase or amp */
#define OH_DELTA2_BASE (TAGB + SB)  /* same encoding as T_DELTA */
#define OH_MTF_BASE    (TAGB)        /* no parameters at all */
#define OH_PRNGBIT3    (TAGB + 3*(16.0+8.0))  /* 1 type tag + 3×(seed16 + fmask8) */
static double oh_valuemap4(int s)   { return TAGB + SB + pb_bits(s) + 24.0; } /* 4×6-bit constants */

/* ============================================================ *
 *  apply / invert primitives (in place)                         *
 * ============================================================ */


static inline u8 addnib(u8 v, int lo, int hi) {
    return (u8)((((v & 0xF) + lo) & 0xF) | ((((v >> 4) + hi) & 0xF) << 4));
}

static void ap_xorp(u8 *d, int n, int s, int p, u32 a) {
    for (int i = p; i < n; i += s) d[i] ^= (u8)a;
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
/* SPLIT_XOR: self-inverse. phase_full = actual_phase|(kidx<<8).
 * kidx=0 dual:N=2,8-bit  kidx=1 quad:N=4,8-bit  kidx=2 octo-nib:N=8,4-bit  kidx=3 half:value-split 7-bit */
static void ap_splitxor(u8 *d, int n, int s, int phase_full, u32 amp) {
    int p = phase_full & 0xFF, kidx = (phase_full >> 8) & 3, k = 0;
    if (kidx == 3) {
        u8 lo = amp & 0x7F, hi = (amp >> 8) & 0x7F;
        for (int i = p; i < n; i += s) { if (d[i] & 0x80) d[i] ^= hi; else d[i] ^= lo; }
    } else if (kidx == 2) {
        u8 a[8]; for (int g = 0; g < 8; g++) a[g] = (amp >> (g*4)) & 0xF;
        for (int i = p; i < n; i += s, k++) d[i] = (u8)((d[i] & 0xF0) | ((d[i] ^ a[k & 7]) & 0x0F));
    } else {
        int K = (kidx == 0) ? 2 : 4;
        u8 a[4] = { (u8)amp, (u8)(amp>>8), (u8)(amp>>16), (u8)(amp>>24) };
        for (int i = p; i < n; i += s, k++) d[i] ^= a[k % K];
    }
}
/* SPLIT_ADD. phase_full = actual_phase|(kidx<<8). */
static void ap_splitadd(u8 *d, int n, int s, int phase_full, u32 amp) {
    int p = phase_full & 0xFF, kidx = (phase_full >> 8) & 3, k = 0;
    if (kidx == 3) {
        u8 lo = amp & 0x7F, hi = (amp >> 7) & 0x7F;
        for (int i = p; i < n; i += s) {
            if (d[i] & 0x80) d[i] = (u8)(0x80 | ((d[i] + hi) & 0x7F));
            else              d[i] = (u8)((d[i] + lo) & 0x7F);
        }
    } else if (kidx == 2) {
        u8 a[8]; for (int g = 0; g < 8; g++) a[g] = (amp >> (g*4)) & 0xF;
        for (int i = p; i < n; i += s, k++) d[i] = (u8)(d[i] + a[k & 7]);
    } else {
        int K = (kidx == 0) ? 2 : 4;
        u8 a[4] = { (u8)amp, (u8)(amp>>8), (u8)(amp>>16), (u8)(amp>>24) };
        for (int i = p; i < n; i += s, k++) d[i] = (u8)(d[i] + a[k % K]);
    }
}
static void inv_splitadd(u8 *d, int n, int s, int phase_full, u32 amp) {
    int p = phase_full & 0xFF, kidx = (phase_full >> 8) & 3, k = 0;
    if (kidx == 3) {
        u8 lo = amp & 0x7F, hi = (amp >> 7) & 0x7F;
        for (int i = p; i < n; i += s) {
            if (d[i] & 0x80) d[i] = (u8)(0x80 | ((d[i] - hi) & 0x7F));
            else              d[i] = (u8)((d[i] - lo) & 0x7F);
        }
    } else if (kidx == 2) {
        u8 a[8]; for (int g = 0; g < 8; g++) a[g] = (amp >> (g*4)) & 0xF;
        for (int i = p; i < n; i += s, k++) d[i] = (u8)(d[i] - a[k & 7]);
    } else {
        int K = (kidx == 0) ? 2 : 4;
        u8 a[4] = { (u8)amp, (u8)(amp>>8), (u8)(amp>>16), (u8)(amp>>24) };
        for (int i = p; i < n; i += s, k++) d[i] = (u8)(d[i] - a[k % K]);
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

/* NIB_CROSS_XOR: XOR one nibble with the other within each byte.
 * amp=0: lo ^= hi  →  d[i] = (d[i] & 0xF0) | ((d[i] ^ (d[i]>>4)) & 0x0F)
 * amp=1: hi ^= lo  →  d[i] = (d[i] & 0x0F) | (((d[i] ^ d[i]>>4) & 0xF) << 4)
 * Both are self-inverse (hi/lo unchanged in the XOR source, so second apply undoes first). */
static void ap_nibcxor(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        if (amp == 0) d[i] = (u8)((v & 0xF0) | ((v ^ (v >> 4)) & 0x0F));
        else          d[i] = (u8)((v & 0x0F) | (((v ^ (v >> 4)) & 0x0F) << 4));
    }
}

/* CRUMB_CROSS_XOR: XOR 2-bit crumb k with 2-bit crumb j within each byte.
 * Crumb i occupies bits [2i+1:2i]. amp = j|(k<<2), j!=k. Self-inverse. */
static void ap_crmbcxor(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s)
        d[i] ^= (u8)(((d[i] >> (2*j)) & 3) << (2*k));
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

/* BIT_REVERSE: reverse bit order within each byte. Self-inverse. */
static inline u8 bitrev8(u8 v) {
    v = (u8)((v & 0xF0) >> 4 | (v & 0x0F) << 4);
    v = (u8)((v & 0xCC) >> 2 | (v & 0x33) << 2);
    v = (u8)((v & 0xAA) >> 1 | (v & 0x55) << 1);
    return v;
}
static void ap_bitrev(u8 *d, int n, int s, int p) {
    for (int i = p; i < n; i += s) d[i] = bitrev8(d[i]);
}

/* GRAY_CODE: encode each byte as its 8-bit Gray code: v → v ^ (v>>1).
 * Consecutive integers differ by exactly 1 bit, so clusters of nearby values
 * map to codes sharing many bits.
 * Inverse: decode via xor-prefix-sum: g ^= g>>4; g ^= g>>2; g ^= g>>1. */
static inline u8 gray_enc(u8 v) { return (u8)(v ^ (v >> 1)); }
static inline u8 gray_dec(u8 g) { g ^= (g >> 4); g ^= (g >> 2); g ^= (g >> 1); return g; }
static void ap_graycode(u8 *d, int n, int s, int p) {
    for (int i = p; i < n; i += s) d[i] = gray_enc(d[i]);
}
static void inv_graycode(u8 *d, int n, int s, int p) {
    for (int i = p; i < n; i += s) d[i] = gray_dec(d[i]);
}

/* BIT_ADJ_SWAP: swap adjacent bits within each byte at stride/phase positions.
 * Bit 0↔1, 2↔3, 4↔5, 6↔7: v = ((v & 0xAA)>>1) | ((v & 0x55)<<1). Self-inverse. */
static void ap_bitaswap(u8 *d, int n, int s, int p) {
    for (int i = p; i < n; i += s)
        d[i] = (u8)(((d[i] & 0xAA) >> 1) | ((d[i] & 0x55) << 1));
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

/* REFLECT: Elias-style bijection centered on mode M.
 * r = (int8_t)(v - M); out = (r >= 0) ? 2*r : -2*r - 1
 * Maps: M→0, M±1→2/1, M±2→4/3, ..., M+127→254, M-128→255. Bijective on 0..255. */
static void ap_reflect(u8 *d, int n, u8 M) {
    for (int i = 0; i < n; i++) {
        int r = (int8_t)(d[i] - M);
        d[i] = (u8)(r >= 0 ? 2 * r : -2 * r - 1);
    }
}
static void inv_reflect(u8 *d, int n, u8 M) {
    for (int i = 0; i < n; i++) {
        int u = d[i];
        int r = (u & 1) ? -((u + 1) / 2) : (u / 2);
        d[i] = (u8)((M + r) & 0xFF);
    }
}


/* PRNGXOR8: 8-stream PRNG XOR. Self-inverse (same PRNG sequence undoes it). */
static void ap_prngxor8(u8 *d, int n, u32 amp) {
    u16 ms = (u16)(amp & 0xFFFF);
    u16 s[8]; s[0] = ms;
    for (int k = 1; k < 8; k++) s[k] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] ^= xs16_next(&s[i & 7]);
}


/* VALUEMAP4: divide value space into 4 quartiles by top-2 bits (v>>6).
 * XOR the lo-6 bits with quartile constant c[q]; top-2 bits unchanged.
 * Self-inverse: v' top-2 bits == v top-2 bits, so c[v'>>6] == c[v>>6].
 * amp bits: c0[5:0] | c1[11:6] | c2[17:12] | c3[23:18] */
static void ap_valuemap4(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    c[0] = (u8)( amp        & 0x3F);
    c[1] = (u8)((amp >>  6) & 0x3F);
    c[2] = (u8)((amp >> 12) & 0x3F);
    c[3] = (u8)((amp >> 18) & 0x3F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        d[i] = (u8)((v & 0xC0) | ((v ^ c[v >> 6]) & 0x3F));
    }
}

/* SPLITBYTEMUL: N-way stride-split multiply by cycling odd constants.
 * kidx in phase[8]: 0=dual(K=2), 1=quad(K=4).
 * amp encodes K 7-bit codes: a[g] = 2*((amp >> (g*7)) & 0x7F) + 1 (always odd). */
static void ap_splitbytemul(u8 *d, int n, int s, int phase_full, u32 amp) {
    int p = phase_full & 0xFF, kidx = (phase_full >> 8) & 1;
    int K = (kidx == 0) ? 2 : 4;
    u8 a[4];
    for (int g = 0; g < K; g++)
        a[g] = (u8)(2 * ((amp >> (g * 7)) & 0x7F) + 1);
    int k = 0;
    for (int i = p; i < n; i += s, k++)
        d[i] = (u8)(d[i] * a[k % K]);
}
static void inv_splitbytemul(u8 *d, int n, int s, int phase_full, u32 amp) {
    int p = phase_full & 0xFF, kidx = (phase_full >> 8) & 1;
    int K = (kidx == 0) ? 2 : 4;
    u8 ai[4];
    for (int g = 0; g < K; g++) {
        u8 a = (u8)(2 * ((amp >> (g * 7)) & 0x7F) + 1);
        ai[g] = mul_inv256(a);
    }
    int k = 0;
    for (int i = p; i < n; i += s, k++)
        d[i] = (u8)(d[i] * ai[k % K]);
}

/* ROTXOR: rotate byte left by k bits, then XOR with c; at stride/phase.
 * amp = (k-1) | (c<<3); k=1..7 stored as 0..6 in bits[2:0].
 * Inverse: XOR with c first, then rotate right by k. */
static void ap_rotxor(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)(amp & 7) + 1;
    u8  c = (u8)((amp >> 3) & 0xFF);
    for (int i = p; i < n; i += s)
        d[i] = byterot_fwd(d[i], k) ^ c;
}
static void inv_rotxor(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)(amp & 7) + 1;
    u8  c = (u8)((amp >> 3) & 0xFF);
    for (int i = p; i < n; i += s)
        d[i] = byterot_inv(d[i] ^ c, k);
}

/* NIBSWAP: swap high and low nibbles of each byte: v -> (v<<4)|(v>>4). Self-inverse. */
static void ap_nibswap(u8 *d, int n, int s, int p) {
    for (int i = p; i < n; i += s)
        d[i] = (u8)((d[i] << 4) | (d[i] >> 4));
}

/* DELTA: d[i] -= d[i-stride] mod 256, scanning backwards so each position sees the
 * original predecessor. Inverse: cumulative sum forward. */
static void ap_delta(u8 *d, int n, int s) {
    for (int i = n - 1; i >= s; i--) d[i] = (u8)((d[i] - d[i - s]) & 0xFF);
}
static void inv_delta(u8 *d, int n, int s) {
    for (int i = s; i < n; i++) d[i] = (u8)((d[i] + d[i - s]) & 0xFF);
}

/* DELTA2: 2nd-order Laplacian predictor: d[i] -= 2*d[i-s] - d[i-2s].
 * For i in [s, 2s): falls back to 1st-order delta (only one predecessor).
 * Backwards scan ensures predecessors are still original when referenced.
 * Inverse: 2-step cumsum forward. */
static void ap_delta2(u8 *d, int n, int s) {
    for (int i = n - 1; i >= 2 * s; i--)
        d[i] = (u8)((d[i] - 2 * d[i - s] + d[i - 2 * s]) & 0xFF);
    for (int i = (2 * s > n ? n : 2 * s) - 1; i >= s; i--)
        d[i] = (u8)((d[i] - d[i - s]) & 0xFF);
}
static void inv_delta2(u8 *d, int n, int s) {
    for (int i = s; i < n && i < 2 * s; i++)
        d[i] = (u8)((d[i] + d[i - s]) & 0xFF);
    for (int i = 2 * s; i < n; i++)
        d[i] = (u8)((d[i] + 2 * d[i - s] - d[i - 2 * s]) & 0xFF);
}

/* MTF (Move-to-Front): replace each byte with its rank in a recency list (0=most recent).
 * Self-inverse structure: encode and decode use the same list-update logic. */
static void ap_mtf(u8 *d, int n) {
    u8 list[256]; for (int i = 0; i < 256; i++) list[i] = (u8)i;
    for (int i = 0; i < n; i++) {
        int r = 0; while (list[r] != d[i]) r++;
        d[i] = (u8)r;
        /* move to front */
        u8 v = list[r];
        memmove(list + 1, list, r);
        list[0] = v;
    }
}
static void inv_mtf(u8 *d, int n) {
    u8 list[256]; for (int i = 0; i < 256; i++) list[i] = (u8)i;
    for (int i = 0; i < n; i++) {
        int r = (int)d[i];
        u8 v = list[r];
        d[i] = v;
        memmove(list + 1, list, r);
        list[0] = v;
    }
}

/* PRNGBIT3: compact bitstream encoding of 3 T_PRNGBIT passes.
 * amp=pass1_amp, stride=pass2_amp, phase=pass3_amp (each = seed|(fmask<<16)).
 * Never emitted by the greedy loop — pack_ilist collapses 3 T_PRNGBIT into this. */
static void ap_prngbit3(u8 *d, int n, u32 amp, int stride, int phase) {
    ap_prngbit(d, n, amp);
    ap_prngbit(d, n, (u32)stride);
    ap_prngbit(d, n, (u32)phase);
}
static void inv_prngbit3(u8 *d, int n, u32 amp, int stride, int phase) {
    ap_prngbit(d, n, (u32)phase);
    ap_prngbit(d, n, (u32)stride);
    ap_prngbit(d, n, amp);
}


/* dispatch: apply / invert any instruction in place */
static void apply_instr(u8 *d, int n, Instr t) {
    switch (t.type) {
        case T_XORP:     ap_xorp(d, n, t.stride, t.phase, t.amp); break;
        case T_XORPNP:   ap_xorp(d, n, t.stride, 0, t.amp); break;
        case T_ANIBS:    ap_anibs(d, n, t.stride, t.phase, t.amp); break;
        case T_STRIDEADD:ap_strideadd(d, n, t.stride, t.phase, t.amp); break;
        case T_BYTEROT:  ap_byterot(d, n, t.stride, t.phase, (int)t.amp); break;
        case T_BYTEMUL:  ap_bytemul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEXOR: ap_valuexor(d, n, t.stride, t.phase, t.amp); break;
        case T_BITREV:   ap_bitrev(d, n, t.stride, t.phase); break;
        case T_PRNGADD4: ap_prngadd4(d, n, t.amp); break;
        case T_PRNGADD8: ap_prngadd8(d, n, t.amp); break;
        case T_NIBCXOR:  ap_nibcxor(d, n, t.stride, t.phase, t.amp); break;
        case T_CRMBCXOR: ap_crmbcxor(d, n, t.stride, t.phase, t.amp); break;
        case T_GRAYCODE: ap_graycode(d, n, t.stride, t.phase); break;
        case T_BITASWAP: ap_bitaswap(d, n, t.stride, t.phase); break;
        case T_PLANEPRNG:ap_planeprng(d, n, t.amp); break;
        case T_REFLECT:  ap_reflect(d, n, (u8)t.amp); break;
        case T_SPLITADD: ap_splitadd(d, n, t.stride, t.phase, t.amp); break;
        case T_SPLITXOR: ap_splitxor(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGBIT:      ap_prngbit(d, n, t.amp); break;
        case T_PRNGXOR8:     ap_prngxor8(d, n, t.amp); break;
        case T_VALUEMAP4:    ap_valuemap4(d, n, t.stride, t.phase, t.amp); break;
        case T_SPLITBYTEMUL: ap_splitbytemul(d, n, t.stride, t.phase, t.amp); break;
        case T_ROTXOR:       ap_rotxor(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBSWAP:        ap_nibswap(d, n, t.stride, t.phase); break;
        case T_DELTA:          ap_delta(d, n, t.stride); break;
        case T_DELTA2:         ap_delta2(d, n, t.stride); break;
        case T_MTF:            ap_mtf(d, n); break;
        case T_PRNGBIT3:       ap_prngbit3(d, n, t.amp, t.stride, t.phase); break;
    }
}
static void invert_instr(u8 *d, int n, Instr t) {
    switch (t.type) {
        case T_XORP:     ap_xorp(d, n, t.stride, t.phase, t.amp); break;      /* self-inv */
        case T_XORPNP:   ap_xorp(d, n, t.stride, 0, t.amp); break;           /* self-inv */
        case T_ANIBS:    inv_anibs(d, n, t.stride, t.phase, t.amp); break;
        case T_STRIDEADD:inv_strideadd(d, n, t.stride, t.phase, t.amp); break;
        case T_BYTEROT:  inv_byterot(d, n, t.stride, t.phase, (int)t.amp); break;
        case T_BYTEMUL:  inv_bytemul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEXOR: ap_valuexor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_BITREV:   ap_bitrev(d, n, t.stride, t.phase); break;            /* self-inv */
        case T_PRNGADD4: inv_prngadd4(d, n, t.amp); break;
        case T_PRNGADD8: inv_prngadd8(d, n, t.amp); break;
        case T_NIBCXOR:  ap_nibcxor(d, n, t.stride, t.phase, t.amp); break;   /* self-inv */
        case T_CRMBCXOR: ap_crmbcxor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_GRAYCODE: inv_graycode(d, n, t.stride, t.phase); break;
        case T_BITASWAP: ap_bitaswap(d, n, t.stride, t.phase); break;          /* self-inv */
        case T_PLANEPRNG:ap_planeprng(d, n, t.amp); break;                     /* self-inv */
        case T_REFLECT:  inv_reflect(d, n, (u8)t.amp); break;
        case T_SPLITADD: inv_splitadd(d, n, t.stride, t.phase, t.amp); break;
        case T_SPLITXOR: ap_splitxor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_PRNGBIT:      ap_prngbit(d, n, t.amp); break;                      /* self-inv */
        case T_PRNGXOR8:     ap_prngxor8(d, n, t.amp); break;                    /* self-inv */
        case T_VALUEMAP4:    ap_valuemap4(d, n, t.stride, t.phase, t.amp); break; /* self-inv */
        case T_SPLITBYTEMUL: inv_splitbytemul(d, n, t.stride, t.phase, t.amp); break;
        case T_ROTXOR:       inv_rotxor(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBSWAP:        ap_nibswap(d, n, t.stride, t.phase); break;          /* self-inv */
        case T_DELTA:          inv_delta(d, n, t.stride); break;
        case T_DELTA2:         inv_delta2(d, n, t.stride); break;
        case T_MTF:            inv_mtf(d, n); break;
        case T_PRNGBIT3:       inv_prngbit3(d, n, t.amp, t.stride, t.phase); break;
    }
}

/* ============================================================ *
 *  searches: each fills *out and returns best net (may be <0)   *
 * ============================================================ */

#define PRNG_SEEDS 65536        /* try seeds 1..65535 */

/* XOR no-phase: same as XOR_PHASE but always phase=0; overhead = TAGB+SB+8 (no pb_bits) */
static double search_xorpnp(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1; u32 ba = 0;
    const double oh = OH_XORP_BASE; /* no pb_bits(s) — same cost at every stride */
    for (int s = 1; s <= g_stride_lim; s++) {
        int hit[256] = {0};
        for (int i = 0; i < n; i += s) hit[d[i]]++;
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        for (int a = 0; a < 256; a++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ a] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; ba = (u32)a; }
        }
    }
    out->type = T_XORPNP; out->stride = bs; out->phase = 0; out->amp = ba;
    return best;
}

/* XOR / ADD at (stride,phase) via the frequency-table trick: O(256) per amp */
static double search_xorp(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_XORP_BASE, s);
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
                double net = (S - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->type = T_XORP; out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}
/* data-dependent ops: apply to a scratch copy and measure S directly */
static __thread u8 g_scr[BLOCK];

/* DELTA: d[i] -= d[i-stride]. Can't use freq-table trick — must apply to scratch. */
static double search_delta(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; int bs = 1;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_DELTA_BASE;
        memcpy(g_scr, d, n);
        ap_delta(g_scr, n, s);
        int f[256]; freq_of(g_scr, n, f);
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bs = s; }
    }
    out->type = T_DELTA; out->stride = bs; out->phase = 0; out->amp = 0;
    return best;
}

/* DELTA2: 2nd-order Laplacian delta. Same scratch-copy approach as T_DELTA. */
static double search_delta2(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; int bs = 1;
    for (int s = 1; s <= g_stride_lim && 2 * s < n; s++) {
        memcpy(g_scr, d, n);
        ap_delta2(g_scr, n, s);
        int f[256]; freq_of(g_scr, n, f);
        double net = (S_from_freq(f) - Sb) - OH_DELTA2_BASE;
        if (net > best) { best = net; bs = s; }
    }
    out->type = T_DELTA2; out->stride = bs; out->phase = 0; out->amp = 0;
    return best;
}

/* MTF: single global pass — no parameters. */
static double search_mtf(const u8 *d, int n, double Sb, Instr *out) {
    memcpy(g_scr, d, n);
    ap_mtf(g_scr, n);
    int f[256]; freq_of(g_scr, n, f);
    double net = (S_from_freq(f) - Sb) - OH_MTF_BASE;
    out->type = T_MTF; out->stride = 1; out->phase = 0; out->amp = 0;
    return net;
}

/* STRIDE_ADD: same frequency-table trick as XOR_PHASE, but ADD instead of XOR, an
 * expanded stride range of 1..33, and amp restricted to nonzero (adding 0 is a no-op). */
static double search_strideadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 1;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_STRIDEADD_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256];
                memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[(u + a) & 255] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
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
        double oh = OH_SP(OH_NIBS_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int amp = 1; amp < 256; amp++) {
                int lo = amp & 0xF, hi = (amp >> 4) & 0xF;
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[addnib((u8)u, lo, hi)] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bamp = (u32)amp; }
            }
        }
    }
    out->type = T_ANIBS; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}
/* BYTE_ROT: frequency-table trick, k=1..7 left-rotations. */
static double search_byterot(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bk = 1;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_BYTEROT_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k = 1; k <= 7; k++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[byterot_fwd((u8)u, k)] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bk = (u32)k; }
            }
        }
    }
    out->type = T_BYTEROT; out->stride = bs; out->phase = bp; out->amp = bk;
    return best;
}
/* BYTE_MUL: frequency-table trick — for each odd multiplier a, the subset element that
 * ends up at value w came from w * a^{-1} mod 256. We enumerate all 127 non-trivial odd
 * multipliers (skip a=1 = identity) and find the one with best net. */
static double search_bytemul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 3;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_BYTEMUL_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 3; a < 256; a += 2) {  /* skip a=1 (identity) */
                int rf[256];
                memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[(u * a) & 0xFF] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->type = T_BYTEMUL; out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}


/* VALUE_XOR: for each bit position k, find best alo (for bit-k=0 group) and ahi (for
 * bit-k=1 group) independently — the two groups are XOR-closed under amps with bit k=0,
 * so they don't interact. Brute-force 127 amps × 128 values per group. */
static double search_valuexor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs=1, bp=0; u32 bamp=0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_VALUEXOR_BASE, s);
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
                double net = (S - Sb) - oh;
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

/* NIB_CROSS_XOR: try both directions (lo^=hi and hi^=lo) at every stride/phase. */
static double search_nibcxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_NIBCXOR_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            /* try both directions */
            for (int dir = 0; dir < 2; dir++) {
                int rf[256];
                for (int v=0;v<256;v++) rf[v] = total[v] - hit[v];
                for (int v=0;v<256;v++) {
                    int w = (dir == 0)
                        ? ((v & 0xF0) | ((v ^ (v>>4)) & 0x0F))
                        : ((v & 0x0F) | (((v ^ (v>>4)) & 0x0F) << 4));
                    rf[w] += hit[v];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bamp = (u32)dir; }
            }
        }
    }
    out->type = T_NIBCXOR; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}


/* GRAY_CODE: try all stride/phase; use freq-table permutation trick. */
static double search_graycode(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    /* precompute gray_enc permutation table */
    int gmap[256];
    for (int v = 0; v < 256; v++) gmap[v] = gray_enc((u8)v);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_GRAYCODE_BASE, s);
        for (int p = 0; p < s; p++) {
            int rf[256];
            for (int v = 0; v < 256; v++) rf[v] = total[v];
            for (int i = p; i < n; i += s) { rf[d[i]]--; rf[gmap[d[i]]]++; }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->type = T_GRAYCODE; out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}

/* BIT_ADJ_SWAP: fixed bijection, freq-table trick. */
static double search_bitaswap(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_BITASWAP_BASE, s);
        for (int p = 0; p < s; p++) {
            int rf[256];
            for (int v = 0; v < 256; v++) rf[v] = total[v];
            for (int i = p; i < n; i += s) {
                u8 w = (u8)(((d[i] & 0xAA) >> 1) | ((d[i] & 0x55) << 1));
                rf[d[i]]--; rf[w]++;
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->type = T_BITASWAP; out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}

/* CRUMB_CROSS_XOR: try all 12 ordered (j,k) pairs at every stride/phase. */
static double search_crmbcxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_CRMBCXOR_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    int rf[256];
                    for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
                    for (int v = 0; v < 256; v++) {
                        int w = v ^ (((v >> (2*j)) & 3) << (2*k));
                        rf[w] += hit[v];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; bamp = (u32)(j | (k<<2)); }
                }
            }
        }
    }
    out->type = T_CRMBCXOR; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* N-stream PRNG_ADD variants: single master seed, N derived streams by pos%N.
 * Lower OH than PRNG_ADD (16 bits vs 32) and single-pass search. */
static double search_prngadd4(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[4]; s[0]=ms; s[1]=xs16_next(&ms); s[2]=xs16_next(&ms); s[3]=xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + xs16_next(&s[i & 3]))]++;
        double net = (S_from_freq(f) - Sb) - OH_PRNGADD4;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGADD4; out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}
static double search_prngadd8(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[8]; s[0]=ms;
        for (int k=1;k<8;k++) s[k]=xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + xs16_next(&s[i & 7]))]++;
        double net = (S_from_freq(f) - Sb) - OH_PRNGADD8;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGADD8; out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* BIT_REVERSE: fixed permutation — use freq-table trick (O(256) per stride/phase). */
static double search_bitrev(const u8 *d, int n, double Sb, Instr *out) {
    /* precompute bitrev table once */
    static u8 brtab[256]; static int init = 0;
    if (!init) { for (int v=0;v<256;v++) brtab[v]=bitrev8((u8)v); init=1; }
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_BITREV_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            /* new freq: untouched bytes stay, hit bytes permuted by bitrev */
            int rf[256];
            for (int v=0;v<256;v++) rf[v] = total[v] - hit[v];
            for (int v=0;v<256;v++) rf[brtab[v]] += hit[v];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->type = T_BITREV; out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}

/* PLANE_PRNG: for each xs16 seed, compute flip/noflip histograms once, then test all 8 planes. */
static double search_planeprng(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; int bk = 0; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 s = (u16)seed;
        int flip[256] = {0}, noflip[256] = {0};
        for (int i = 0; i < n; i++) {
            u16 r = xs16_next(&s);
            if (r & 1) flip[d[i]]++;
            else        noflip[d[i]]++;
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


/* ---- Walsh-Hadamard Transform helpers (used by search_splitxor) ---- */
static void wht256(int *a) {
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += len<<1)
            for (int j = 0; j < len; j++) {
                int u = a[i+j], v = a[i+j+len];
                a[i+j] = u+v; a[i+j+len] = u-v;
            }
}
static int xor_best_wht(const int *A, const int *B, double *Sout) {
    int ha[256], hb[256];
    memcpy(ha, A, 256*sizeof(int)); memcpy(hb, B, 256*sizeof(int));
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

/* SPLIT_ADD unified search. */
static double search_splitadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bph_full = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        for (int p = 0; p < s; p++) {
            int ph8[8][256]; memset(ph8, 0, sizeof ph8);
            int k = 0;
            for (int i = p; i < n; i += s, k++) ph8[k & 7][d[i]]++;
            int ph4[4][256], ph2[2][256];
            for (int g = 0; g < 4; g++)
                for (int v = 0; v < 256; v++) ph4[g][v] = ph8[g][v] + ph8[g+4][v];
            for (int g = 0; g < 2; g++)
                for (int v = 0; v < 256; v++) ph2[g][v] = ph4[g][v] + ph4[g+2][v];
            #define KSPLIT_COORD(PHarr, K, amax, kidx_val, pack_amp) do { \
                double oh = oh_splitadd(kidx_val, s); \
                int amps[8] = {0}; \
                int cur[256]; memcpy(cur, total, sizeof cur); \
                for (int pass = 0; pass < 5; pass++) { \
                    int changed = 0; \
                    for (int g = 0; g < (K); g++) { \
                        for (int w = 0; w < 256; w++) cur[w] -= PHarr[g][(w - amps[g]) & 255]; \
                        int ba = 0; double bS2 = -1e18; \
                        for (int a = 0; a < (amax); a++) { \
                            double S2 = 0.0; \
                            for (int w = 0; w < 256; w++) S2 += hlog[cur[w] + PHarr[g][(w-a)&255]]; \
                            if (S2 > bS2) { bS2 = S2; ba = a; } \
                        } \
                        if (ba != amps[g]) { amps[g] = ba; changed = 1; } \
                        for (int w = 0; w < 256; w++) cur[w] += PHarr[g][(w - amps[g]) & 255]; \
                    } \
                    if (!changed) break; \
                } \
                double net2 = (S_from_freq(cur) - Sb) - oh; \
                if (net2 > best) { best = net2; bs = s; bph_full = p | ((kidx_val)<<8); pack_amp; } \
            } while(0)
            KSPLIT_COORD(ph2, 2, 256, 0, bamp = (u32)amps[0] | ((u32)amps[1]<<8));
            KSPLIT_COORD(ph4, 4, 256, 1, bamp = (u32)amps[0]|((u32)amps[1]<<8)|((u32)amps[2]<<16)|((u32)amps[3]<<24));
            KSPLIT_COORD(ph8, 8,  16, 2, { bamp=0; for(int g=0;g<8;g++) bamp|=(u32)(amps[g]&0xF)<<(g*4); });
            #undef KSPLIT_COORD
            /* kidx=3: value-split half */
            {
                double oh = oh_splitadd(3, s);
                int flo[128] = {0}, fhi[128] = {0};
                for (int i = p; i < n; i += s) { u8 v=d[i]; if(v&0x80) fhi[v&0x7F]++; else flo[v]++; }
                int blo_b[128], bhi_b[128];
                for (int v=0;v<128;v++) { blo_b[v]=total[v]-flo[v]; bhi_b[v]=total[v+128]-fhi[v]; }
                int alo=0; double bSlo=-1e18;
                for (int a=0;a<128;a++) { double S=0.0; for(int v=0;v<128;v++) S+=hlog[blo_b[v]+flo[(v-a+128)&0x7F]]; if(S>bSlo){bSlo=S;alo=a;} }
                int ahi=0; double bShi=-1e18;
                for (int a=0;a<128;a++) { double S=0.0; for(int v=0;v<128;v++) S+=hlog[bhi_b[v]+fhi[(v-a+128)&0x7F]]; if(S>bShi){bShi=S;ahi=a;} }
                double net2 = (bSlo+bShi-Sb) - oh;
                if (net2 > best) { best=net2; bs=s; bph_full=p|(3<<8); bamp=(u32)alo|((u32)ahi<<7); }
            }
        }
    }
    out->type = T_SPLITADD; out->stride = bs; out->phase = bph_full; out->amp = bamp;
    return best;
}

/* SPLIT_XOR unified search. */
static double search_splitxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bph_full = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        for (int p = 0; p < s; p++) {
            int ph8[8][256]; memset(ph8, 0, sizeof ph8);
            int k = 0;
            for (int i = p; i < n; i += s, k++) ph8[k & 7][d[i]]++;
            int ph4[4][256], ph2[2][256];
            for (int g = 0; g < 4; g++)
                for (int v = 0; v < 256; v++) ph4[g][v] = ph8[g][v] + ph8[g+4][v];
            for (int g = 0; g < 2; g++)
                for (int v = 0; v < 256; v++) ph2[g][v] = ph4[g][v] + ph4[g+2][v];
            for (int kidx = 0; kidx <= 1; kidx++) {
                int K = (kidx == 0) ? 2 : 4;
                int (*ph)[256] = (kidx == 0) ? ph2 : ph4;
                double oh = oh_splitxor(kidx, s);
                int cur[256], amps[4] = {0,0,0,0};
                memcpy(cur, total, sizeof cur);
                for (int g = 0; g < K; g++) {
                    for (int v = 0; v < 256; v++) cur[v] -= ph[g][v ^ amps[g]];
                    double Sg; int ba = xor_best_wht(cur, ph[g], &Sg);
                    amps[g] = ba;
                    for (int v = 0; v < 256; v++) cur[v] += ph[g][v ^ amps[g]];
                }
                double net = (S_from_freq(cur) - Sb) - oh;
                if (net > best) {
                    best = net; bs = s; bph_full = p | (kidx << 8);
                    bamp = 0; for (int g = 0; g < K; g++) bamp |= (u32)amps[g] << (g*8);
                }
            }
            /* kidx=2 octo nibble XOR */
            {
                double oh = oh_splitxor(2, s);
                int amps[8] = {0,0,0,0,0,0,0,0};
                int cur[256]; memcpy(cur, total, sizeof cur);
                for (int pass = 0; pass < 4; pass++) {
                    int changed = 0;
                    for (int g = 0; g < 8; g++) {
                        for (int w = 0; w < 256; w++) cur[w] -= ph8[g][(w&0xF0)|((w^amps[g])&0xF)];
                        int ba = 0; double bS = -1e18;
                        for (int a = 0; a < 16; a++) {
                            double S = 0.0;
                            for (int w = 0; w < 256; w++) S += hlog[cur[w] + ph8[g][(w&0xF0)|((w^a)&0xF)]];
                            if (S > bS) { bS = S; ba = a; }
                        }
                        if (ba != amps[g]) { amps[g] = ba; changed = 1; }
                        for (int w = 0; w < 256; w++) cur[w] += ph8[g][(w&0xF0)|((w^amps[g])&0xF)];
                    }
                    if (!changed) break;
                }
                double net = (S_from_freq(cur) - Sb) - oh;
                if (net > best) {
                    best = net; bs = s; bph_full = p | (2 << 8);
                    bamp = 0; for (int g = 0; g < 8; g++) bamp |= (u32)(amps[g]&0xF) << (g*4);
                }
            }
            /* kidx=3 half XOR */
            {
                double oh = oh_splitxor(3, s);
                int flo[128] = {0}, fhi[128] = {0};
                for (int i = p; i < n; i += s) { u8 v=d[i]; if(v&0x80) fhi[v&0x7F]++; else flo[v]++; }
                int blo_a[128], bhi_a[128];
                for (int v=0;v<128;v++) { blo_a[v]=total[v]-flo[v]; bhi_a[v]=total[v+128]-fhi[v]; }
                int alo=0; double bSlo=-1e18;
                for (int a=0;a<128;a++) { double S=0.0; for(int v=0;v<128;v++) S+=hlog[blo_a[v]+flo[v^a]]; if(S>bSlo){bSlo=S;alo=a;} }
                int ahi=0; double bShi=-1e18;
                for (int a=0;a<128;a++) { double S=0.0; for(int v=0;v<128;v++) S+=hlog[bhi_a[v]+fhi[v^a]]; if(S>bShi){bShi=S;ahi=a;} }
                double net = (bSlo + bShi - Sb) - oh;
                if (net > best) { best=net; bs=s; bph_full=p|(3<<8); bamp=(u32)alo|((u32)ahi<<8); }
            }
        }
    }
    out->type = T_SPLITXOR; out->stride = bs; out->phase = bph_full; out->amp = bamp;
    return best;
}

/* PRNG_BIT: for each seed, build 8 disjoint groups per bit pos; greedy flip mask. */
static double search_prngbit(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18;
    u16 best_seed = 1; u8 best_fmask = 0;
    for (u32 s = 1; s < 65536; s++) {
        u16 st = (u16)s;
        int grp[8][256]; memset(grp, 0, sizeof grp);
        for (int i = 0; i < n; i++) grp[xs16_b3(&st)][d[i]]++;
        u8 fmask = 0;
        for (int j = 0; j < 8; j++) {
            int bv = 1 << j;
            double dS = 0.0;
            for (int v = 0; v < 256; v++) {
                int gv = grp[j][v]; if (!gv) continue;
                int gvf = grp[j][v ^ bv];
                dS += hlog[total[v]-gv+gvf] + hlog[total[v^bv]+gv-gvf]
                    - hlog[total[v]] - hlog[total[v^bv]];
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

/* PRNGBIT3: 3 greedy sub-passes; fmask=0xFF (all planes); uses freq-table trick per pass.
 * Returns net = total_raw - OH_PRNGBIT3. Seeds stored in amp(lo,hi) and stride. */
static double search_prngbit3(const u8 *d, int n, double Sb, Instr *out) {
    u8 tmp[BLOCK];
    memcpy(tmp, d, n);
    u16 seeds[3] = {1, 1, 1};
    double Scur = Sb;
    double total_raw = 0.0;
    for (int pass = 0; pass < 3; pass++) {
        int tot[256]; freq_of(tmp, n, tot);
        double best_S = -1e18; u16 bs = 1;
        for (u32 s = 1; s < 65536; s++) {
            u16 st = (u16)s;
            int grp[8][256]; memset(grp, 0, sizeof grp);
            for (int i = 0; i < n; i++) grp[xs16_b3(&st)][tmp[i]]++;
            int cur[256]; memcpy(cur, tot, sizeof cur);
            for (int j = 0; j < 8; j++) {
                int bv = 1 << j;
                for (int v = 0; v < 256; v++) {
                    if (!grp[j][v]) continue;
                    cur[v]      -= grp[j][v];
                    cur[v ^ bv] += grp[j][v];
                }
            }
            double S = S_from_freq(cur);
            if (S > best_S) { best_S = S; bs = (u16)s; }
        }
        seeds[pass] = bs;
        total_raw += best_S - Scur;
        Scur = best_S;
        ap_prngbit(tmp, n, (u32)bs | (0xFFu << 16));
    }
    double net = total_raw - OH_PRNGBIT3;
    out->type = T_PRNGBIT3;
    out->amp = (u32)seeds[0] | ((u32)seeds[1] << 16);
    out->stride = (int)seeds[2];
    out->phase = 0;
    return net;
}


static double search_prngxor8(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < 4096; seed++) {
        u16 ms = (u16)seed;
        u16 s[8]; s[0]=ms;
        for (int k=1;k<8;k++) s[k]=xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ xs16_next(&s[i & 7]))]++;
        double net = (S_from_freq(f) - Sb) - OH_PRNGXOR8;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGXOR8; out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}


/* VALUEMAP4: for each of 4 quartiles (top-2 bits), find optimal lo-6-bit XOR constant.
 * Quartiles are independent (XOR preserves top-2 bits), so per-quartile greedy is exact. */
static double search_valuemap4(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs=1, bp=0; u32 bamp=0;
    for (int s=1; s<=g_stride_lim; s++) {
        double oh = oh_valuemap4(s);
        for (int p=0; p<s; p++) {
            int hit[256]={0};
            for (int i=p; i<n; i+=s) hit[d[i]]++;
            int base[256];
            for (int v=0;v<256;v++) base[v]=total[v]-hit[v];
            /* per-quartile independent optimisation */
            u32 amp=0;
            for (int q=0; q<4; q++) {
                int bc=0; double bS=-1e30;
                int qlo=q*64;
                for (int c=0; c<64; c++) {
                    double S=0.0;
                    for (int j=0; j<64; j++)
                        S += hlog[base[qlo+j] + hit[qlo+(j^c)]];
                    if (S>bS) { bS=S; bc=c; }
                }
                amp |= (u32)bc << (q*6);
            }
            /* compute combined result freq */
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int v=0;v<256;v++) if (hit[v]) {
                int q=v>>6, c=(int)((amp>>(q*6))&0x3F);
                rf[(v&0xC0)|((v^c)&0x3F)] += hit[v];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net>best) { best=net; bs=s; bp=p; bamp=amp; }
        }
    }
    out->type=T_VALUEMAP4; out->stride=bs; out->phase=bp; out->amp=bamp;
    return best;
}

/* SPLITBYTEMUL: coordinate descent over K odd multipliers (K=2 or 4) cycling by stride-phase.
 * Stride limited to 8 (marked slow). */
static double search_splitbytemul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs=1, bph_full=0; u32 bamp=0;
    int slim = (g_stride_lim < 8) ? g_stride_lim : 8;
    for (int s=1; s<=slim; s++) {
        for (int p=0; p<s; p++) {
            int ph8[8][256]; memset(ph8,0,sizeof ph8);
            int k=0;
            for (int i=p; i<n; i+=s, k++) ph8[k&7][d[i]]++;
            int ph2[2][256], ph4[4][256];
            for (int g=0;g<4;g++) for (int v=0;v<256;v++)
                ph4[g][v]=ph8[g][v]+ph8[g+4][v];
            for (int g=0;g<2;g++) for (int v=0;v<256;v++)
                ph2[g][v]=ph4[g][v]+ph4[g+2][v];
            for (int kidx=0; kidx<=1; kidx++) {
                int K = (kidx==0)?2:4;
                int (*ph)[256] = (kidx==0) ? ph2 : ph4;
                double oh = oh_splitbytemul(kidx, s);
                u8 ac[4]; for (int g=0;g<K;g++) ac[g]=1; /* start with identity */
                int cur[256]; memcpy(cur,total,sizeof cur);
                for (int pass=0; pass<5; pass++) {
                    int changed=0;
                    for (int g=0; g<K; g++) {
                        u8 acg_inv = mul_inv256(ac[g]);
                        for (int w=0;w<256;w++) cur[w] -= ph[g][(u8)((u8)acg_inv*(u8)w)];
                        u8 ba=1; double bS=-1e18;
                        for (int aa=0; aa<128; aa++) {
                            u8 a=(u8)(2*aa+1);
                            u8 ai=mul_inv256(a);
                            double S=0.0;
                            for (int w=0;w<256;w++) S+=hlog[cur[w]+ph[g][(u8)((u8)ai*(u8)w)]];
                            if (S>bS){bS=S;ba=a;}
                        }
                        if (ba!=ac[g]){ac[g]=ba;changed=1;}
                        u8 acg_inv2=mul_inv256(ac[g]);
                        for (int w=0;w<256;w++) cur[w]+=ph[g][(u8)((u8)acg_inv2*(u8)w)];
                    }
                    if (!changed) break;
                }
                double net=(S_from_freq(cur)-Sb)-oh;
                if (net>best) {
                    best=net; bs=s; bph_full=p|(kidx<<8);
                    bamp=0; for(int g=0;g<K;g++) bamp|=(u32)((ac[g]-1)/2)<<(g*7);
                }
            }
        }
    }
    out->type=T_SPLITBYTEMUL; out->stride=bs; out->phase=bph_full; out->amp=bamp;
    return best;
}

/* ROTXOR: for each k=1..7, precompute rotated hit-histogram, then find best XOR constant c
 * via O(256) exhaustive scan (freq-table trick). Stride limited to 8 to keep search fast. */
static double search_rotxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs=1, bp=0; u32 bamp=0;
    int slim = (g_stride_lim < 8) ? g_stride_lim : 8;
    for (int s=1; s<=slim; s++) {
        for (int p=0; p<s; p++) {
            double oh = oh_rotxor(s);
            int hit[256]={0};
            for (int i=p; i<n; i+=s) hit[d[i]]++;
            int base[256];
            for (int v=0;v<256;v++) base[v]=total[v]-hit[v];
            for (int k=1; k<=7; k++) {
                int rhit[256]={0};
                for (int v=0;v<256;v++) rhit[byterot_fwd((u8)v,k)]+=hit[v];
                for (int c=0; c<256; c++) {
                    double S=0.0;
                    for (int w=0;w<256;w++) S+=hlog[base[w]+rhit[w^c]];
                    double net=(S-Sb)-oh;
                    if (net>best) { best=net; bs=s; bp=p; bamp=(u32)(k-1)|((u32)(u8)c<<3); }
                }
            }
        }
    }
    out->type=T_ROTXOR; out->stride=bs; out->phase=bp; out->amp=bamp;
    return best;
}

/* NIBSWAP: fixed permutation (swap hi/lo nibbles) — freq-table trick. */
static double search_nibswap(const u8 *d, int n, double Sb, Instr *out) {
    static u8 nstab[256]; static int nsinit=0;
    if (!nsinit) { for(int v=0;v<256;v++) nstab[v]=(u8)((v<<4)|(v>>4)); nsinit=1; }
    int total[256]; freq_of(d, n, total);
    double best=-1e18; int bs=1, bp=0;
    for (int s=1; s<=g_stride_lim; s++) {
        double oh=OH_SP(OH_NIBSWAP_BASE, s);
        for (int p=0; p<s; p++) {
            int hit[256]={0};
            for (int i=p;i<n;i+=s) hit[d[i]]++;
            int rf[256];
            for (int v=0;v<256;v++) rf[v]=total[v]-hit[v];
            for (int v=0;v<256;v++) rf[nstab[v]]+=hit[v];
            double net=(S_from_freq(rf)-Sb)-oh;
            if (net>best){best=net;bs=s;bp=p;}
        }
    }
    out->type=T_NIBSWAP; out->stride=bs; out->phase=bp; out->amp=0;
    return best;
}

/* REFLECT: search for the best mode M to center the Elias remap on.
 * Reflect is a pure value bijection, so S is invariant — net is always negative.
 * Adding it to the registry lets best_instr() evaluate it; try_reflect() still
 * applies it unconditionally at the end so it always fires at least once. */
static double search_reflect(const u8 *d, int n, double Sb, Instr *out) {
    int f[256]; freq_of(d, n, f);
    int M = 0;
    for (int v = 1; v < 256; v++) if (f[v] > f[M]) M = v;
    out->type = T_REFLECT; out->stride = 0; out->phase = 0; out->amp = (u32)M;
    return -(TAGB + 8.0);   /* bijection: ΔS=0, so net = -overhead */
}

/* registry of selectable instructions */
typedef double (*SearchFn)(const u8 *, int, double, Instr *);
/* prng_first=1: always run on layer 0; other types skipped on layer 0 */
typedef struct { const char *name; SearchFn search; int slow; int prng_first; } InstrDesc;



static const InstrDesc REGISTRY[] = {
    { "XOR_PHASE",   search_xorp,         0, 0 },
    { "ADD_NIBS",    search_anibs,         0, 0 },
    { "STRIDE_ADD",  search_strideadd,     0, 0 },
    { "BYTE_ROT",    search_byterot,       0, 0 },
    { "BYTE_MUL",    search_bytemul,       0, 1 },
    { "VALUE_XOR",   search_valuexor,      1, 0 },
    { "BIT_REV",     search_bitrev,        0, 0 },
    { "PRNG_ADD4",   search_prngadd4,      1, 1 },
    { "PRNG_ADD8",   search_prngadd8,      1, 1 },
    { "NIB_CXOR",    search_nibcxor,       0, 0 },
    { "CRMB_CXOR",   search_crmbcxor,      0, 0 },
    { "GRAY_CODE",   search_graycode,      0, 0 },
    { "BIT_ASWAP",   search_bitaswap,      0, 0 },
    { "PLANE_PRNG",  search_planeprng,     1, 1 },
    { "REFLECT",     search_reflect,       0, 0 },
    { "SPLIT_ADD",   search_splitadd,      1, 0 },
    { "SPLIT_XOR",   search_splitxor,      1, 0 },
    { "PRNG_BIT",    search_prngbit,       1, 1 },
    { "PRNG_XOR8",   search_prngxor8,      1, 1 },
    { "VALUEMAP4",   search_valuemap4,     0, 0 },
    { "SPLT_BYTEMUL",search_splitbytemul,  1, 0 },
    { "ROT_XOR",     search_rotxor,        1, 0 },
    { "NIB_SWAP",    search_nibswap,        0, 0 },
    { "XOR_NP",      search_xorpnp,         0, 0 },
    { "DELTA",       search_delta,          0, 0 },
    { "DELTA2",      search_delta2,         0, 0 },
    { "MTF",         search_mtf,            0, 0 },
    /* T_PRNGBIT3 is NOT in REGISTRY — it's a bitstream-only compact encoding of 3 T_PRNGBIT */
};
#define NREG ((int)(sizeof(REGISTRY)/sizeof(REGISTRY[0])))
static const char *TYPE_NAME[NTYPES] = {
    "XOR_PHASE",  "ADD_NIBS",   "STRIDE_ADD",  "BYTE_ROT",    "BYTE_MUL",
    "VALUE_XOR",  "BIT_REV",    "PRNG_ADD4",   "PRNG_ADD8",   "NIB_CXOR",
    "CRMB_CXOR", "GRAY_CODE", "BIT_ASWAP",  "PLANE_PRNG", "REFLECT",
    "SPLIT_ADD", "SPLIT_XOR", "PRNG_BIT",   "PRNG_XOR8",
    "VALUEMAP4", "SPLT_BMUL", "ROT_XOR",    "NIB_SWAP",   "XOR_NP",
    "DELTA",     "DELTA2",    "MTF",        "PRNGBIT3"
};


/* 0=all types, 1=prng_first only (layers 0-2), 2=non-prng only (layer 3+) */
static __thread int g_search_mode;

/* Adaptive type-tag cost: replaces fixed TAGB with -log2(laplace_smoothed_freq).
 * Repeated types get cheaper; rare types get more expensive. Reset per block. */
static __thread int g_type_freq[NTYPES];
static __thread int g_total_instrs_used;

static double adaptive_tag_cost(int type) {
    /* Denominator frozen at 0.5*18 (the original type count before extension types were added).
     * Keeps the prior stable so adding non-firing types doesn't inflate costs for firing ones. */
    double p = (g_type_freq[type] + 0.5) / (g_total_instrs_used + 9.0);
    return -log2(p);
}

/* best selectable instruction for the current data */
static double best_instr(const u8 *d, int n, Instr *out) {
    double Sb = S_of(d, n);
    double best = -1e18; Instr bi = {0};
    for (int r = 0; r < NREG; r++) {
        if (g_search_mode == 1 && !REGISTRY[r].prng_first) continue;
        if (g_search_mode == 2 &&  REGISTRY[r].prng_first) continue;
        Instr cand;
        double net = REGISTRY[r].search(d, n, Sb, &cand);
        /* swap fixed TAGB for the actual adaptive type-coding cost */
        net = net + TAGB - adaptive_tag_cost(cand.type);
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
    memset(g_type_freq, 0, sizeof g_type_freq);
    g_total_instrs_used = 0;
    double gained = 0.0;
    for (;;) {
        if (*ni >= MAXINSTR) break;
        Instr t;
        g_search_mode = (*ni < 3) ? 1 : 2;  /* layers 0-2: PRNG only; layer 3+: non-PRNG only */
        double net = best_instr(d, n, &t);
        g_search_mode = 0;
        /* net was computed with adaptive_tag_cost; compact encoder uses a fixed 5-bit
         * type tag.  Only accept the instruction if raw entropy reduction > actual
         * compact cost, i.e.  net > TAGB - adaptive_tag_cost(t.type). */
        if (net <= TAGB - adaptive_tag_cost(t.type)) break;
        double e0 = entropy_bits(d, n) / n;
        apply_instr(d, n, t);
        g_type_freq[t.type]++;
        g_total_instrs_used++;
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

/* Always apply REFLECT at the very end: find mode M, Elias-remap around it.
 * Doesn't change order-0 entropy (pure value permutation) but concentrates the
 * histogram around 0x00, which benefits a downstream entropy coder. net=0. */
static void try_reflect(u8 *d, int n, Instr *ilist, double *nets, int *ni, int verbose) {
    int f[256]; freq_of(d, n, f);
    int M = 0;
    for (int v = 1; v < 256; v++) if (f[v] > f[M]) M = v;
    ap_reflect(d, n, (u8)M);
    Instr r = { T_REFLECT, 0, 0, (u32)M };
    if (*ni < MAXINSTR) { nets[*ni] = 0.0; ilist[(*ni)++] = r; }
    if (verbose)
        printf("  %-12s M=0x%02X               %.4f bps  (value remap)\n",
               "REFLECT", M, entropy_bits(d, n) / n);
}

static double instr_oh(Instr t);  /* forward decl; defined after do_block */

static double compress(u8 *d, int n, Instr *ilist, double *nets, int *ni, int verbose) {
    *ni = 0;
    if (verbose) {
        printf("  INPUT        (before any layer)         %.4f bps\n", entropy_bits(d, n) / n);
        print_layer_struct(d, n);
    }
    double total = greedy_run(d, n, ilist, nets, ni, verbose);
    try_reflect(d, n, ilist, nets, ni, verbose);
    return total;
}

/* ============================================================ *
 *  decode (apply inverses in reverse order)                     *
 * ============================================================ */

static void decompress(u8 *d, int n, const Instr *ilist, int ni) {
    for (int i = ni - 1; i >= 0; i--) invert_instr(d, n, ilist[i]);
}

/* ============================================================ *
 *  Compact instruction bit-stream                               *
 *                                                               *
 *  Each field is packed to its natural width (same widths the   *
 *  OH_ model uses), eliminating the wasted bits in the flat     *
 *  8-byte-per-instruction serialization.                        *
 *                                                               *
 *  Layout per instruction:                                       *
 *    type        : 6 bits  (NTYPES=33 <= 64)                   *
 *    stride-1    : 6 bits  (1..64, skip for PRNG/REFLECT)      *
 *    kidx        : 2 bits  (SPLIT* only, before phase)         *
 *    phase       : ceil(log2(stride)) bits                      *
 *    amp         : type-specific width                          *
 *  Header: 16-bit instruction count.                            *
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
/* ceil(log2(s)): number of bits to represent 0..s-1 */
static int phase_bits(int s) {
    int b = 0; while ((1 << b) < s) b++; return b;
}

static void bb_put_instr(BitBuf *b, Instr t) {
    bb_put(b, t.type, 6);
    int s = t.stride, pb = phase_bits(s);
    switch (t.type) {
        case T_PRNGADD4: case T_PRNGADD8:
            bb_put(b, t.amp & 0xFFFF, 16); break;
        case T_PLANEPRNG:
            bb_put(b, t.amp & 0xFFFF, 16);
            bb_put(b, (t.amp >> 16) & 7, 3); break;
        case T_PRNGBIT:
            bb_put(b, t.amp & 0xFFFF, 16);
            bb_put(b, (t.amp >> 16) & 0xFF, 8); break;
        case T_REFLECT:
            bb_put(b, t.amp & 0xFF, 8); break;
        case T_BITREV: case T_GRAYCODE: case T_BITASWAP:
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb); break;
        case T_XORPNP:
            bb_put(b, s - 1, 8); bb_put(b, t.amp & 0xFF, 8); break;   /* no phase */
        case T_DELTA:
            bb_put(b, s - 1, 8); break;                                /* stride only */
        case T_DELTA2:
            bb_put(b, s - 1, 8); break;                                /* stride only */
        case T_MTF:
            break;                                                      /* no params */
        case T_PRNGBIT3: {
            /* amp=pass1, stride=pass2, phase=pass3; each = seed|(fmask<<16) */
            u32 sf[3] = { t.amp, (u32)t.stride, (u32)t.phase };
            for (int k = 0; k < 3; k++) {
                bb_put(b, sf[k] & 0xFFFF, 16);
                bb_put(b, (sf[k] >> 16) & 0xFF, 8);
            }
        } break;
        case T_XORP: case T_ANIBS: case T_STRIDEADD:
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xFF, 8); break;
        case T_BYTEROT:
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb);
            bb_put(b, (t.amp - 1) & 7, 3); break;           /* k=1..7 → 0..6 */
        case T_BYTEMUL:
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb);
            bb_put(b, (t.amp - 3) / 2, 7); break;           /* odd a=3..255 → 0..126 */
        case T_VALUEXOR: {
            /* amp = k(3) | alo(8, bit-k forced 0) | ahi(8, bit-k forced 0)
             * Pack as 17 bits: k(3) + alo with bit-k removed (7) + ahi with bit-k removed (7) */
            int vk   = t.amp & 7;
            int valo = (t.amp >> 3) & 0xFF;
            int vahi = (t.amp >> 11) & 0xFF;
            int mask = (1 << vk) - 1;
            int alo7 = (valo & mask) | ((valo >> (vk+1)) << vk);
            int ahi7 = (vahi & mask) | ((vahi >> (vk+1)) << vk);
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb);
            bb_put(b, (u32)vk, 3); bb_put(b, (u32)alo7, 7); bb_put(b, (u32)ahi7, 7);
            break;
        }
        case T_NIBCXOR:
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 1, 1); break;
        case T_CRMBCXOR:
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xF, 4); break;
        case T_SPLITADD: case T_SPLITXOR: {
            int kidx = (t.phase >> 8) & 3, act_p = t.phase & 0xFF;
            bb_put(b, s - 1, 8); bb_put(b, kidx, 2); bb_put(b, act_p, pb);
            if (kidx == 3) {
                /* kidx=3 has a gap in the bit layout; re-pack as 14 clean bits */
                int lo = t.amp & 0x7F;
                int hi = (t.amp >> (t.type == T_SPLITADD ? 7 : 8)) & 0x7F;
                bb_put(b, lo, 7); bb_put(b, hi, 7);
            } else {
                int ab = (t.type == T_SPLITADD)
                    ? (int)SPLITADD_AMP_BITS[kidx] : (int)SPLITXOR_AMP_BITS[kidx];
                bb_put(b, t.amp, ab);
            }
            break;
        }
        case T_PRNGXOR8:
            bb_put(b, t.amp & 0xFFFF, 16); break;
        case T_VALUEMAP4:
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x00FFFFFFu, 24); break;
        case T_SPLITBYTEMUL: {
            int kidx = (t.phase >> 8) & 1, act_p = t.phase & 0xFF;
            int K = (kidx == 0) ? 2 : 4;
            bb_put(b, s - 1, 8); bb_put(b, (u32)kidx, 1); bb_put(b, (u32)act_p, pb);
            bb_put(b, t.amp, K * 7); break;
        }
        case T_ROTXOR:
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 7, 3);
            bb_put(b, (t.amp >> 3) & 0xFF, 8); break;
        case T_NIBSWAP:
            bb_put(b, s - 1, 8); bb_put(b, t.phase, pb); break;
    }
}

/* Decode instruction body given the already-read type value. */
static Instr bb_get_instr_body(BitBuf *b, u8 type_val) {
    Instr t = {0, 0, 0, 0};
    t.type = type_val;
    int s, pb;
    switch (t.type) {
        case T_PRNGADD4: case T_PRNGADD8:
            t.amp = bb_get(b, 16); break;
        case T_PLANEPRNG: {
            u32 seed = bb_get(b, 16), k = bb_get(b, 3);
            t.amp = seed | (k << 16); break;
        }
        case T_PRNGBIT: {
            u32 seed = bb_get(b, 16), fm = bb_get(b, 8);
            t.amp = seed | (fm << 16); break;
        }
        case T_REFLECT:
            t.amp = bb_get(b, 8); break;
        case T_BITREV: case T_GRAYCODE: case T_BITASWAP:
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb); break;
        case T_XORPNP:
            s = (int)bb_get(b, 8) + 1; t.stride = s; t.phase = 0;
            t.amp = bb_get(b, 8); break;                               /* no phase */
        case T_DELTA:
            s = (int)bb_get(b, 8) + 1; t.stride = s; t.phase = 0; t.amp = 0; break;
        case T_DELTA2:
            s = (int)bb_get(b, 8) + 1; t.stride = s; t.phase = 0; t.amp = 0; break;
        case T_MTF:
            t.stride = 1; t.phase = 0; t.amp = 0; break;              /* no params */
        case T_PRNGBIT3: {
            /* amp=pass1, stride=pass2, phase=pass3; each = seed(16)+fmask(8) */
            u32 s0=bb_get(b,16), f0=bb_get(b,8);
            u32 s1=bb_get(b,16), f1=bb_get(b,8);
            u32 s2=bb_get(b,16), f2=bb_get(b,8);
            t.amp    = s0 | (f0 << 16);
            t.stride = (int)(s1 | (f1 << 16));
            t.phase  = (int)(s2 | (f2 << 16));
        } break;
        case T_XORP: case T_ANIBS: case T_STRIDEADD:
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 8); break;
        case T_BYTEROT:
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 3) + 1; break;
        case T_BYTEMUL:
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 7) * 2 + 3; break;
        case T_VALUEXOR: {
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            int vk   = (int)bb_get(b, 3);
            int alo7 = (int)bb_get(b, 7);
            int ahi7 = (int)bb_get(b, 7);
            /* restore the forced-zero bit k in alo and ahi */
            int mask = (1 << vk) - 1;
            int valo = (alo7 & mask) | ((alo7 >> vk) << (vk+1));
            int vahi = (ahi7 & mask) | ((ahi7 >> vk) << (vk+1));
            t.amp = (u32)(vk | (valo << 3) | (vahi << 11));
            break;
        }
        case T_NIBCXOR:
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 1); break;
        case T_CRMBCXOR:
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 4); break;
        case T_SPLITADD: case T_SPLITXOR: {
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            int kidx = (int)bb_get(b, 2), act_p = (int)bb_get(b, pb);
            t.stride = s; t.phase = act_p | (kidx << 8);
            if (kidx == 3) {
                int lo = (int)bb_get(b, 7); int hi = (int)bb_get(b, 7);
                t.amp = (t.type == T_SPLITADD)
                    ? (u32)(lo | (hi << 7)) : (u32)(lo | (hi << 8));
            } else {
                int ab = (t.type == T_SPLITADD)
                    ? (int)SPLITADD_AMP_BITS[kidx] : (int)SPLITXOR_AMP_BITS[kidx];
                t.amp = bb_get(b, ab);
            }
            break;
        }
        case T_PRNGXOR8:
            t.amp = bb_get(b, 16); break;
        case T_VALUEMAP4:
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 24); break;
        case T_SPLITBYTEMUL: {
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            int kidx = (int)bb_get(b, 1), act_p = (int)bb_get(b, pb);
            t.stride = s; t.phase = act_p | (kidx << 8);
            int K = (kidx == 0) ? 2 : 4;
            t.amp = bb_get(b, K * 7); break;
        }
        case T_ROTXOR: {
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            u32 km = bb_get(b, 3), cm = bb_get(b, 8);
            t.amp = km | (cm << 3); break;
        }
        case T_NIBSWAP:
            s = (int)bb_get(b, 8) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb); break;
    }
    return t;
}
static Instr bb_get_instr(BitBuf *b) {
    return bb_get_instr_body(b, (u8)bb_get(b, 6));
}

/* Max bits per instruction: VALUEXOR stride=256 → 6+8+8+17 = 39 bits (largest non-PHASEOFFSET) */
#define COMPACT_BUF_BYTES ((MAXINSTR * 64 + 16 + 7) / 8)

/* Pack instruction list into compact bit-stream.
 * 3 consecutive T_PRNGBIT instructions are collapsed into one T_PRNGBIT3 token
 * (1 tag + 3×(seed16+fmask8) = 78 bits vs 3×30 = 90 bits, saves 12 bits).
 * The 16-bit header stores the token count (may be less than ni if runs exist). */
static int pack_ilist(const Instr *ilist, int ni, u8 *buf) {
    memset(buf, 0, COMPACT_BUF_BYTES);
    /* count tokens */
    int ni_tok = 0;
    for (int i = 0; i < ni; ) {
        if (i+2 < ni && ilist[i].type==T_PRNGBIT
                     && ilist[i+1].type==T_PRNGBIT
                     && ilist[i+2].type==T_PRNGBIT)
            { ni_tok++; i += 3; }
        else { ni_tok++; i++; }
    }
    BitBuf b = { buf, 0 };
    bb_put(&b, (u32)ni_tok, 16);
    for (int i = 0; i < ni; ) {
        if (i+2 < ni && ilist[i].type==T_PRNGBIT
                     && ilist[i+1].type==T_PRNGBIT
                     && ilist[i+2].type==T_PRNGBIT) {
            bb_put(&b, (u32)T_PRNGBIT3, 6);
            for (int k = 0; k < 3; k++) {
                bb_put(&b, ilist[i+k].amp & 0xFFFF, 16);
                bb_put(&b, (ilist[i+k].amp >> 16) & 0xFF, 8);
            }
            i += 3;
        } else {
            bb_put_instr(&b, ilist[i]);
            i++;
        }
    }
    return b.pos;
}

/* Unpack. Returns LOGICAL instruction count (T_PRNGBIT3 expands to 3 T_PRNGBIT). */
static int unpack_ilist(const u8 *buf, Instr *ilist_out) {
    BitBuf b = { (u8 *)buf, 0 };
    int ni_tok = (int)bb_get(&b, 16);
    int out = 0;
    for (int i = 0; i < ni_tok; i++) {
        u8 type_val = (u8)bb_get(&b, 6);
        if (type_val == T_PRNGBIT3) {
            /* expand to 3 individual T_PRNGBIT instructions */
            for (int k = 0; k < 3; k++) {
                u32 seed = bb_get(&b, 16), fmask = bb_get(&b, 8);
                Instr t = {T_PRNGBIT, 0, 0, seed | (fmask << 16)};
                ilist_out[out++] = t;
            }
        } else {
            ilist_out[out++] = bb_get_instr_body(&b, type_val);
        }
    }
    return out;
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
        { T_XORP,      3, 1, 0x5A },
        { T_ANIBS,     3, 1, 0x35 },
        { T_STRIDEADD, 3, 1, 0x37 },
        { T_BYTEROT,   3, 1, 3 },
        { T_BYTEMUL,   3, 1, 3u },
        { T_VALUEXOR,  3, 1, (3u|(0x24u<<3)|(0x12u<<11)) },
        { T_BITREV,    3, 1, 0 },
        { T_PRNGADD4,  0, 0, 1234u },
        { T_PRNGADD8,  0, 0, 5678u },
        { T_NIBCXOR,   3, 1, 0 },
        { T_NIBCXOR,   3, 1, 1 },
        { T_CRMBCXOR,  3, 1, (0|(2<<2)) },
        { T_GRAYCODE,  3, 1, 0 },
        { T_BITASWAP,  3, 1, 0 },
        { T_PLANEPRNG, 0, 0, (1234u | (3u << 16)) },
        { T_REFLECT,   0, 0, 0x40u },
        { T_SPLITADD,  3, 1|(0<<8), 0x1234u },
        { T_SPLITADD,  3, 1|(1<<8), 0x11223344u },
        { T_SPLITADD,  3, 1|(2<<8), 0x12345678u },
        { T_SPLITADD,  3, 1|(3<<8), (0x13u|(0x41u<<7)) },
        { T_SPLITXOR,  3, 1|(0<<8), 0x1122u },
        { T_SPLITXOR,  3, 1|(1<<8), 0x11223344u },
        { T_SPLITXOR,  3, 1|(2<<8), 0x12345678u },
        { T_SPLITXOR,  3, 1|(3<<8), (0x2Au | (0x55u << 8)) },
        { T_PRNGBIT,      0, 0, (0xABCDu | (0xA5u << 16)) },
        { T_PRNGXOR8,     0, 0, 9999u },
        { T_VALUEMAP4,    3, 1, (0x1Au|(0x2Bu<<6)|(0x0Cu<<12)|(0x3Fu<<18)) },
        { T_SPLITBYTEMUL, 3, 1|(0<<8), (u32)(1u|(2u<<7)) },
        { T_SPLITBYTEMUL, 3, 1|(1<<8), (u32)(1u|(2u<<7)|(3u<<14)|(4u<<21)) },
        { T_ROTXOR,       3, 1, (u32)((3u-1)|((u32)0x42u<<3)) },
        { T_NIBSWAP,      3, 1, 0 },
        { T_DELTA,        4, 0, 0 },
        { T_DELTA2,       3, 0, 0 },
        { T_MTF,          1, 0, 0 },
        { T_PRNGBIT3, (int)(5u|(0xFFu<<16)), (int)(7u|(0xFFu<<16)), (u32)(3u|(0xFFu<<16)) },
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
/* overhead in bits for a single applied instruction (mirrors search OH formulas) */
static double instr_oh(Instr t) {
    switch (t.type) {
        case T_XORP:     return OH_SP(OH_XORP_BASE,      t.stride);
        case T_XORPNP:   return OH_XORP_BASE; /* no phase field */
        case T_DELTA:    return OH_DELTA_BASE;
        case T_DELTA2:   return OH_DELTA2_BASE;
        case T_MTF:      return OH_MTF_BASE;
        case T_PRNGBIT3: return OH_PRNGBIT3;
        case T_ANIBS:    return OH_SP(OH_NIBS_BASE,      t.stride);
        case T_STRIDEADD:return OH_SP(OH_STRIDEADD_BASE, t.stride);
        case T_BYTEROT:  return OH_SP(OH_BYTEROT_BASE,   t.stride);
        case T_BYTEMUL:  return OH_SP(OH_BYTEMUL_BASE,   t.stride);
        case T_VALUEXOR: return OH_SP(OH_VALUEXOR_BASE,  t.stride);
        case T_BITREV:   return OH_SP(OH_BITREV_BASE,    t.stride);
        case T_PRNGADD4: return OH_PRNGADD4;
        case T_PRNGADD8: return OH_PRNGADD8;
        case T_NIBCXOR:  return OH_SP(OH_NIBCXOR_BASE,  t.stride);
        case T_CRMBCXOR: return OH_SP(OH_CRMBCXOR_BASE, t.stride);
        case T_GRAYCODE: return OH_SP(OH_GRAYCODE_BASE, t.stride);
        case T_BITASWAP: return OH_SP(OH_BITASWAP_BASE, t.stride);
        case T_PLANEPRNG:return OH_PLANEPRNG;
        case T_REFLECT:  return 0.0;
        case T_SPLITADD:    return oh_splitadd((t.phase >> 8) & 3, t.stride);
        case T_SPLITXOR:    return oh_splitxor((t.phase >> 8) & 3, t.stride);
        case T_PRNGBIT:      return OH_PRNGBIT;
        case T_PRNGXOR8:     return OH_PRNGXOR8;
        case T_VALUEMAP4:    return oh_valuemap4(t.stride);
        case T_SPLITBYTEMUL: return oh_splitbytemul((t.phase >> 8) & 1, t.stride);
        case T_ROTXOR:       return oh_rotxor(t.stride);
        case T_NIBSWAP:        return OH_SP(OH_NIBSWAP_BASE, t.stride);
        default:             return 0.0;
    }
}

/* per-block result: all stats + ibuf bytes needed for aggregate reporting */
typedef struct {
    double e_in, e_out, net;
    int    ok, ni, cbits, cok;
    int    type_counts[NTYPES];
    double type_net_sum[NTYPES];
    double type_net_max[NTYPES];
    double type_oh_sum[NTYPES];
    int    ibuf_n;
    u8     ibuf[MAXINSTR * 8];
} BlockResult;

/* reduce one block into *r; ilist/nets allocated on heap so stack stays small */
static void do_block(u8 *data, int n, BlockResult *r, int verbose) {
    u8 orig[BLOCK];
    memcpy(orig, data, n);

    Instr  *ilist       = malloc(MAXINSTR * sizeof(Instr));
    double *nets        = malloc(MAXINSTR * sizeof(double));
    u8     *compact_buf = malloc(COMPACT_BUF_BYTES);
    Instr  *unpacked    = malloc(MAXINSTR * sizeof(Instr));
    int ni = 0;

    r->e_in = entropy_bits(data, n);
    r->net  = compress(data, n, ilist, nets, &ni, verbose);
    r->e_out = entropy_bits(data, n);
    r->ni   = ni;

    memset(r->type_counts,  0, sizeof r->type_counts);
    memset(r->type_net_sum, 0, sizeof r->type_net_sum);
    memset(r->type_net_max, 0, sizeof r->type_net_max);
    memset(r->type_oh_sum,  0, sizeof r->type_oh_sum);
    for (int i = 0; i < ni; i++) {
        int t = ilist[i].type;
        r->type_counts[t]++;
        r->type_net_sum[t] += nets[i];
        if (nets[i] > r->type_net_max[t]) r->type_net_max[t] = nets[i];
        r->type_oh_sum[t]  += instr_oh(ilist[i]);
    }

    r->cbits = compact_buf ? pack_ilist(ilist, ni, compact_buf) : 0;
    r->cok   = 0;
    if (compact_buf && unpacked) {
        int ni2 = unpack_ilist(compact_buf, unpacked);
        r->cok  = (ni2 == ni);
        for (int i = 0; r->cok && i < ni2; i++)
            r->cok = (unpacked[i].type   == ilist[i].type   &&
                      unpacked[i].stride == ilist[i].stride &&
                      unpacked[i].phase  == ilist[i].phase  &&
                      unpacked[i].amp    == ilist[i].amp);
    }

    u8 dec[BLOCK];
    memcpy(dec, data, n);
    decompress(dec, n, ilist, ni);
    r->ok = (memcmp(dec, orig, n) == 0);

    r->ibuf_n = 0;
    for (int i = 0; i < ni; i++) {
        r->ibuf[r->ibuf_n++] = (u8)ilist[i].type;
        r->ibuf[r->ibuf_n++] = (u8)ilist[i].stride;
        r->ibuf[r->ibuf_n++] = (u8)(ilist[i].phase & 0xFF);
        r->ibuf[r->ibuf_n++] = (u8)(ilist[i].phase >> 8);
        r->ibuf[r->ibuf_n++] = (u8)( ilist[i].amp        & 0xFF);
        r->ibuf[r->ibuf_n++] = (u8)((ilist[i].amp >>  8) & 0xFF);
        r->ibuf[r->ibuf_n++] = (u8)((ilist[i].amp >> 16) & 0xFF);
        r->ibuf[r->ibuf_n++] = (u8)((ilist[i].amp >> 24) & 0xFF);
    }

    free(ilist); free(nets); free(compact_buf); free(unpacked);
}

typedef struct { u8 *data; BlockResult *res; int verbose; } BlockArg;

static DWORD WINAPI block_thread(LPVOID arg) {
    BlockArg *a = (BlockArg *)arg;
    do_block(a->data, BLOCK, a->res, a->verbose);
    return 0;
}

int main(int argc, char **argv) {
    init_hlog();

    if (argc > 1 && strcmp(argv[1], "selftest") == 0) return selftest() ? 2 : 0;

    /* ---- stridemap: for every stride 1..MAX_STRIDE, print best raw XOR delta-S and overhead ---- */
    if (argc > 1 && strcmp(argv[1], "stridemap") == 0) {
        const char *sf = (argc > 2) ? argv[2] : "compressed.bin";
        u8 data[BLOCK];
        FILE *f = fopen(sf, "rb");
        if (!f) { fprintf(stderr, "cannot open %s\n", sf); return 1; }
        if (fread(data, 1, BLOCK, f) != BLOCK) { fprintf(stderr, "need %d bytes\n", BLOCK); fclose(f); return 1; }
        fclose(f);
        double Sb = S_of(data, BLOCK);
        int total[256]; freq_of(data, BLOCK, total);
        printf("stridemap of %s  (Sb=%.2f, bps=%.4f)\n", sf, Sb, entropy_bits(data, BLOCK) / BLOCK);
        printf("  %5s  %5s  %8s  %8s  %8s  %5s\n", "stride","phase","raw_dS","overhead","net","amp");
        printf("  %5s  %5s  %8s  %8s  %8s  %5s\n", "------","-----","------","--------","---","---");
        for (int s = 1; s <= MAX_STRIDE; s++) {
            double oh = OH_SP(OH_XORP_BASE, s);
            double best_raw = -1e18; int bp = 0; u32 ba = 0;
            for (int p = 0; p < s; p++) {
                int hit[256] = {0};
                for (int i = p; i < BLOCK; i += s) hit[data[i]]++;
                int base[256];
                for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
                for (int a = 0; a < 256; a++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) rf[u ^ a] += hit[u];
                    double raw = S_from_freq(rf) - Sb;
                    if (raw > best_raw) { best_raw = raw; bp = p; ba = (u32)a; }
                }
            }
            if (best_raw > -1.0)  /* only print strides that do something */
                printf("  %5d  %5d  %+8.2f  %8.1f  %+8.2f  %5u\n",
                       s, bp, best_raw, oh, best_raw - oh, ba);
        }
        return 0;
    }

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
            double Sb = S_of(data, BLOCK);
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
    }

    FILE *fcomp = fopen("compressed.bin", "wb");
    if (!fcomp) { fprintf(stderr, "cannot open compressed.bin\n"); return 1; }

    BlockResult *results = calloc(NB, sizeof(BlockResult));
    BlockArg    *args    = malloc(NB * sizeof(BlockArg));
    SYSTEM_INFO si; GetSystemInfo(&si);
    int ncpus = (int)si.dwNumberOfProcessors;
    if (ncpus < 1) ncpus = 1;
    HANDLE *threads = malloc((size_t)ncpus * sizeof(HANDLE));
    if (!results || !args || !threads) { fprintf(stderr, "oom\n"); return 1; }

    clock_t t0 = clock();
    printf("\n=== compressing %d blocks (%d threads) ===\n", NB, ncpus);
    fflush(stdout);

    /* process in batches of ncpus so we never exceed the logical CPU count */
    for (int start = 0; start < NB; ) {
        int batch = NB - start;
        if (batch > ncpus) batch = ncpus;
        for (int i = 0; i < batch; i++) {
            int b = start + i;
            args[b].data    = all + (size_t)b * BLOCK;
            args[b].res     = &results[b];
            args[b].verbose = (NB == 1);
            threads[i] = CreateThread(NULL, 0, block_thread, &args[b], 0, NULL);
            if (!threads[i]) { fprintf(stderr, "CreateThread failed for block %d\n", b); return 1; }
        }
        for (int i = 0; i < batch; i++) {
            WaitForSingleObject(threads[i], INFINITE);
            CloseHandle(threads[i]);
        }
        start += batch;
    }
    double ms = (double)(clock() - t0) / CLOCKS_PER_SEC * 1000.0;

    /* collect results in order, print and accumulate */
    int counts[NTYPES] = {0};
    double type_net_sum[NTYPES] = {0};
    double type_net_max[NTYPES]; for (int t = 0; t < NTYPES; t++) type_net_max[t] = 0.0;
    double type_oh_sum[NTYPES]  = {0};
    double total_net = 0.0, total_ein = 0.0, total_eout = 0.0, total_overhead = 0.0;
    int total_ni = 0, fails = 0, compact_fails = 0;
    long total_compact_bits = 0;
    u8 *ibuf  = malloc((size_t)NB * MAXINSTR * 8);
    int ibuf_n = 0;

    for (int b = 0; b < NB; b++) {
        BlockResult *r = &results[b];
        double raw = r->e_in - r->e_out;
        double oh  = raw - r->net;

        total_net      += r->net;   total_ein  += r->e_in;
        total_eout     += r->e_out; total_overhead += oh;
        total_ni       += r->ni;    total_compact_bits += r->cbits;
        if (!r->ok)  fails++;
        if (!r->cok) compact_fails++;
        for (int t = 0; t < NTYPES; t++) {
            counts[t]       += r->type_counts[t];
            type_net_sum[t] += r->type_net_sum[t];
            if (r->type_net_max[t] > type_net_max[t]) type_net_max[t] = r->type_net_max[t];
            type_oh_sum[t]  += r->type_oh_sum[t];
        }

        printf("  block %2d: %.4f -> %.4f bps  net=%+.1f  %s  [%d instrs  raw=%+.1f  OH=%.1f bits  compact=%d bits %s]\n",
               b, r->e_in / BLOCK, r->e_out / BLOCK, r->net,
               r->ok ? "ok" : "FAIL", r->ni, raw, oh, r->cbits,
               r->cok ? "ok" : "FAIL");
        if (NB == 1) {
            int fq[256]; freq_of(all, BLOCK, fq);
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

        if (ibuf) { memcpy(ibuf + ibuf_n, r->ibuf, r->ibuf_n); ibuf_n += r->ibuf_n; }
        fwrite(all + (size_t)b * BLOCK, 1, BLOCK, fcomp);
    }
    fclose(fcomp);

    int fired = 0;
    for (int t = 0; t < NTYPES; t++) if (counts[t]) fired++;

    printf("\n=== aggregate over %d blocks (%.0f ms, %d threads) ===\n", NB, ms, ncpus);
    printf("avg input:  %.4f bps     avg output: %.4f bps\n",
           total_ein / (NB * BLOCK), total_eout / (NB * BLOCK));
    printf("total net:  %.1f bits   (avg %.1f / block)\n", total_net, total_net / NB);
    printf("total instrs: %d (avg %.1f/block)   total OH: %.1f bits (avg %.1f/block)\n",
           total_ni, (double)total_ni / NB, total_overhead, total_overhead / NB);
    if (ibuf && ibuf_n > 0) {
        double ibps = entropy_bits(ibuf, ibuf_n) / ibuf_n;
        printf("instr stream (flat 8B/instr): %.4f bps  (%d bytes)  entropy=%.0f bits/block\n",
               ibps, ibuf_n, entropy_bits(ibuf, ibuf_n) / NB);
    }
    printf("instr stream (compact):       %ld bits/block total  (%.1f bits/block avg)  %s\n",
           total_compact_bits / NB,
           (double)total_compact_bits / NB,
           compact_fails ? "*** COMPACT FAIL ***" : "round-trip ok");
    free(ibuf); free(results); free(args); free(threads);
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
