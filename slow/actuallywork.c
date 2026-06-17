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
static int g_stride_lim  = MAX_STRIDE;

/* ---- instruction type ids ---- */
enum {
    T_XORP = 0,  /* XOR amp at (stride,phase) subset       */
    T_ANIBS,     /* nibble-wise add (lo+=al, hi+=ah)       */
    T_QUADADD,   /* 4-way pos split, 4 ADD amps            */
    T_PRNGD,     /* PRNG dual: even/odd pos XOR two xs16 seeds  */
    T_OCTNIBX,   /* 8-way pos split, 8 nibble (lo-nibble only) XOR amps */
    T_STRIDEADD, /* ADD amp (nonzero) at (stride,phase) subset, stride in 1..64 */
    T_PRNGBIT,   /* PRNG selects bit pos (0-7) per byte; XOR that bit with (amp>>bit_pos)&1 */
    T_BYTEROT,   /* circular bit rotation: v = (v<<k)|(v>>(8-k)), k=1..7 */
    T_HALFXOR,   /* value-conditional XOR: bytes<128 XOR amp_lo, bytes>=128 XOR amp_hi */
    T_HALFADD,   /* value-conditional ADD mod 128: lo-half += amp_lo, hi-half += amp_hi     */
    T_BYTEMUL,   /* multiply each byte by odd constant a mod 256; a in 3..255 odd            */
    T_TRIPLEXOR, /* 3-way pos split XOR: a0|(a1<<8)|(a2<<16)                                 */
    T_VALUEXOR,  /* bit-k conditional XOR: amp=k|(alo<<3)|(ahi<<11); preserves bit k         */
    T_BITREV,    /* reverse bit order within each byte at (stride,phase) subset; self-inverse  */
    T_BPXOR,     /* per-byte: XOR bit k with bit j (j!=k); amp=j|(k<<3); self-inverse         */
    T_PRNGADD,   /* PRNG add: even/odd positions get xs16-streamed ADD constants; amp=s1|s2<<16 */
    T_NIBLUT,    /* low-nibble permutation LUT: lut[0..7] in amp (4 bits each), lut[8..15] in stride */
    T_NIBCXOR,   /* cross-nibble XOR: amp=0 → lo^=hi, amp=1 → hi^=lo; self-inverse; stride/phase */
    T_CRMBCXOR,  /* cross-crumb XOR: XOR 2-bit crumb k with crumb j; amp=j|(k<<2); self-inverse */
    T_REFLECT,   /* Elias remap around mode M: r=(v-M) signed, out=r>=0?2r:(-2r-1); amp=M    */
    NTYPES       /* = 19 */
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
/* Phase bits are stride-adaptive: log2(s) instead of fixed 6. Use OH_SP(base, s). */
#define OH_XORP_BASE    (TAGB + SB + 8.0)
#define OH_NIBS_BASE    (TAGB + SB + 8.0)
#define OH_QUAD_BASE    (TAGB + SB + 32.0)
#define OH_PRNGD        (TAGB + 32.0)       /* 2 independent 16-bit seeds, no stride/phase */
#define OH_OCTNIBX_BASE (TAGB + SB + 32.0) /* 8 bands * 4-bit lo-nibble XOR amp each */
#define OH_STRIDEADD_BASE (TAGB + SB + 8.0)
#define OH_PRNGBIT      (TAGB + 16.0 + 8.0) /* 16-bit seed + 8-bit flip-mask, no stride/phase */
#define OH_BYTEROT_BASE (TAGB + SB + 3.0)
#define OH_HALFXOR_BASE (TAGB + SB + 14.0)
#define OH_HALFADD_BASE (TAGB + SB + 14.0)
#define OH_BYTEMUL_BASE (TAGB + SB + 7.0)
#define OH_TRIPLEXOR_BASE (TAGB + SB + 24.0)
#define OH_VALUEXOR_BASE  (TAGB + SB + 17.0)
#define OH_BITREV_BASE    (TAGB + SB)           /* no amp — fixed bit reversal */
#define OH_BPXOR          (TAGB + 6.0)          /* j(3 bits) + k(3 bits), no stride/phase */
#define OH_PRNGADD        (TAGB + 32.0)         /* two 16-bit seeds, same as PRNG_DUAL */
#define OH_NIBLUT         (TAGB + 64.0)         /* 16 × 4-bit LUT entries */
#define OH_NIBCXOR_BASE   (TAGB + SB + 1.0)     /* 1 direction bit + stride/phase */
#define OH_CRMBCXOR_BASE  (TAGB + SB + 4.0)     /* 2+2 bits for crumb indices j,k */
/* Stride-adaptive phase cost: log2(s) bits for phase in [0,s). Use in search loops. */
static inline double pb_bits(int s) { return (s > 1) ? log2((double)s) : 0.0; }
#define OH_SP(base, s)  ((base) + pb_bits(s))

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

/* TRIPLE_XOR: 3-way position split, independent XOR per group. Self-inverse. */
static void ap_triplexor(u8 *d, int n, int s, int p, u32 amp) {
    u8 a[3] = { (u8)(amp&0xFF), (u8)((amp>>8)&0xFF), (u8)((amp>>16)&0xFF) };
    int k = 0;
    for (int i = p; i < n; i += s, k++) d[i] ^= a[k % 3];
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

/* NIBBLE_LUT: apply a permutation on 0..15 to the low nibble of every byte.
 * High nibble is untouched. LUT is stored as 16 nibbles:
 *   amp   = lut[0]|(lut[1]<<4)|...|(lut[7]<<28)
 *   stride = lut[8]|(lut[9]<<4)|...|(lut[15]<<28)
 * Inverse uses the inverse permutation, computed from the same LUT. */
static void niblut_unpack(u32 amp, int stride_lut, u8 lut[16]) {
    for (int i = 0; i < 8;  i++) lut[i]   = (u8)((amp         >> (i * 4)) & 0xF);
    for (int i = 0; i < 8;  i++) lut[i+8] = (u8)((stride_lut  >> (i * 4)) & 0xF);
}
static void ap_niblut(u8 *d, int n, u32 amp, int stride_lut) {
    u8 lut[16]; niblut_unpack(amp, stride_lut, lut);
    for (int i = 0; i < n; i++)
        d[i] = (u8)((d[i] & 0xF0) | lut[d[i] & 0x0F]);
}
static void inv_niblut(u8 *d, int n, u32 amp, int stride_lut) {
    u8 lut[16], inv[16]; niblut_unpack(amp, stride_lut, lut);
    for (int i = 0; i < 16; i++) inv[lut[i]] = (u8)i;
    for (int i = 0; i < n; i++)
        d[i] = (u8)((d[i] & 0xF0) | inv[d[i] & 0x0F]);
}

/* PRNG_ADD: like PRNG_DUAL but uses addition instead of XOR. Even positions use
 * stream s1, odd positions use stream s2. Inverse subtracts same sequences. */
static void ap_prngadd(u8 *d, int n, u32 amp) {
    u16 s1 = (u16)(amp & 0xFFFF), s2 = (u16)((amp >> 16) & 0xFFFF);
    for (int i = 0; i < n; i++) {
        u8 b = (i & 1) ? xs16_next(&s2) : xs16_next(&s1);
        d[i] = (u8)(d[i] + b);
    }
}
static void inv_prngadd(u8 *d, int n, u32 amp) {
    u16 s1 = (u16)(amp & 0xFFFF), s2 = (u16)((amp >> 16) & 0xFFFF);
    for (int i = 0; i < n; i++) {
        u8 b = (i & 1) ? xs16_next(&s2) : xs16_next(&s1);
        d[i] = (u8)(d[i] - b);
    }
}

/* BIT_PLANE_XOR: for each byte, XOR bit k with bit j (j!=k). Self-inverse because
 * only bit k changes; bit j is read but not modified, so applying twice restores. */
static void ap_bpxor(u8 *d, int n, u32 amp) {
    int j = (int)(amp & 7), k = (int)((amp >> 3) & 7);
    u8 jmask = (u8)(1 << j), kmask = (u8)(1 << k);
    for (int i = 0; i < n; i++)
        if (d[i] & jmask) d[i] ^= kmask;
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

/* dispatch: apply / invert any instruction in place */
static void apply_instr(u8 *d, int n, Instr t) {
    switch (t.type) {
        case T_XORP:      ap_xorp(d, n, t.stride, t.phase, t.amp); break;
        case T_ANIBS:     ap_anibs(d, n, t.stride, t.phase, t.amp); break;
        case T_QUADADD:   ap_quadadd(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGD:     ap_prngd(d, n, t.amp); break;
        case T_OCTNIBX:   ap_octnibx(d, n, t.stride, t.phase, t.amp); break;
        case T_STRIDEADD: ap_strideadd(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGBIT:   ap_prngbit(d, n, t.amp); break;
        case T_BYTEROT:   ap_byterot(d, n, t.stride, t.phase, (int)t.amp); break;
        case T_HALFXOR:   ap_halfxor(d, n, t.stride, t.phase, t.amp); break;
        case T_HALFADD:   ap_halfadd(d, n, t.stride, t.phase, t.amp); break;
        case T_BYTEMUL:   ap_bytemul(d, n, t.stride, t.phase, t.amp); break;
        case T_TRIPLEXOR: ap_triplexor(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEXOR:  ap_valuexor(d, n, t.stride, t.phase, t.amp); break;
        case T_BITREV:    ap_bitrev(d, n, t.stride, t.phase); break;
        case T_BPXOR:     ap_bpxor(d, n, t.amp); break;
        case T_PRNGADD:   ap_prngadd(d, n, t.amp); break;
        case T_NIBLUT:    ap_niblut(d, n, t.amp, t.stride); break;
        case T_NIBCXOR:   ap_nibcxor(d, n, t.stride, t.phase, t.amp); break;
        case T_CRMBCXOR:  ap_crmbcxor(d, n, t.stride, t.phase, t.amp); break;
        case T_REFLECT:   ap_reflect(d, n, (u8)t.amp); break;
    }
}
static void invert_instr(u8 *d, int n, Instr t) {
    switch (t.type) {
        case T_XORP:      ap_xorp(d, n, t.stride, t.phase, t.amp); break;      /* self-inv */
        case T_ANIBS:     inv_anibs(d, n, t.stride, t.phase, t.amp); break;
        case T_QUADADD:   inv_quadadd(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGD:     ap_prngd(d, n, t.amp); break;                        /* self-inv */
        case T_OCTNIBX:   ap_octnibx(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_STRIDEADD: inv_strideadd(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGBIT:   ap_prngbit(d, n, t.amp); break;                      /* self-inv */
        case T_BYTEROT:   inv_byterot(d, n, t.stride, t.phase, (int)t.amp); break;
        case T_HALFXOR:   ap_halfxor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_HALFADD:   inv_halfadd(d, n, t.stride, t.phase, t.amp); break;
        case T_BYTEMUL:   inv_bytemul(d, n, t.stride, t.phase, t.amp); break;
        case T_TRIPLEXOR: ap_triplexor(d, n, t.stride, t.phase, t.amp); break; /* self-inv */
        case T_VALUEXOR:  ap_valuexor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_BITREV:    ap_bitrev(d, n, t.stride, t.phase); break;            /* self-inv */
        case T_BPXOR:     ap_bpxor(d, n, t.amp); break;                        /* self-inv */
        case T_PRNGADD:   inv_prngadd(d, n, t.amp); break;
        case T_NIBLUT:    inv_niblut(d, n, t.amp, t.stride); break;
        case T_NIBCXOR:   ap_nibcxor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_CRMBCXOR:  ap_crmbcxor(d, n, t.stride, t.phase, t.amp); break; /* self-inv */
        case T_REFLECT:   inv_reflect(d, n, (u8)t.amp); break;
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
static u8 g_scr[BLOCK];
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
/* k-way position split ADD: coordinate descent on K add amps */
static double search_ksplit_add(const u8 *d, int n, double Sb, Instr *out, int K, int type, double oh_base) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = oh_base + pb_bits(s);
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
    return search_ksplit_add(d, n, Sb, out, 4, T_QUADADD, OH_QUAD_BASE);
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
        double oh = OH_SP(OH_OCTNIBX_BASE, s);
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
            double net = (S - Sb) - oh;
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

/* HALF_XOR: build separate 128-entry freq tables for lo-half and hi-half bytes.
 * The two XOR amps are independent (disjoint value ranges, bit7 untouched), so
 * we find each optimally with the standard freq trick — one O(128*128) pass each. */
static double search_halfxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_HALFXOR_BASE, s);
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
            double net = (bSlo + bShi - Sb) - oh;
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
        double oh = OH_SP(OH_HALFADD_BASE, s);
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
            double net = (bSlo + bShi - Sb) - oh;
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

/* TRIPLE_XOR: 3-pass coordinate descent, each pass uses WHT to find best XOR amp. */
static double search_triplexor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_TRIPLEXOR_BASE, s);
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
            double net = (S3 - Sb) - oh;
            if (net > best) { best=net; bs=s; bp=p; bamp=(u32)(a0|(a1<<8)|(a2<<16)); }
        }
    }
    out->type=T_TRIPLEXOR; out->stride=bs; out->phase=bp; out->amp=bamp;
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

/* NIBBLE_LUT: find the best low-nibble permutation by sorting marginal nibble counts.
 * Maps most-common nibble → 0, next → 1, etc. Packs result into amp + stride. */
static double search_niblut(const u8 *d, int n, double Sb, Instr *out) {
    int freq[256]; freq_of(d, n, freq);
    /* marginal count for each low nibble */
    int nib_cnt[16] = {0};
    for (int v=0;v<256;v++) nib_cnt[v & 0xF] += freq[v];
    /* build permutation: sort nibbles by descending frequency → map to 0,1,2,... */
    u8 order[16]; for (int i=0;i<16;i++) order[i]=(u8)i;
    for (int i=0;i<16;i++) /* insertion sort */
        for (int j=i+1;j<16;j++)
            if (nib_cnt[order[j]] > nib_cnt[order[i]]) { u8 t=order[i]; order[i]=order[j]; order[j]=t; }
    u8 lut[16]; /* lut[old_nibble] = new_nibble */
    for (int rank=0;rank<16;rank++) lut[order[rank]] = (u8)rank;
    /* compute actual S after transform */
    int rf[256];
    for (int v=0;v<256;v++) rf[(v & 0xF0) | lut[v & 0xF]] = freq[v];
    double net = (S_from_freq(rf) - Sb) - OH_NIBLUT;
    /* pack lut into amp (lut[0..7]) and stride (lut[8..15]) */
    u32 amp_lut = 0; int str_lut = 0;
    for (int i=0;i<8;i++) amp_lut |= (u32)(lut[i] & 0xF) << (i*4);
    for (int i=0;i<8;i++) str_lut |= (int)((lut[i+8] & 0xF) << (i*4));
    out->type = T_NIBLUT; out->stride = str_lut; out->phase = 0; out->amp = amp_lut;
    return net;
}

/* PRNG_ADD: same 3-pass coordinate descent as PRNG_DUAL but with addition. */
static double search_prngadd(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18;
    u16 best_s1 = 1, best_s2 = 1;
    /* pass 1: fix s2=1, scan s1 */
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        u16 s1 = (u16)seed, s2 = 1;
        for (int i = 0; i < n; i++) {
            u8 b = (i & 1) ? xs16_next(&s2) : xs16_next(&s1);
            f[(u8)(d[i] + b)]++;
        }
        double net = (S_from_freq(f) - Sb) - OH_PRNGADD;
        if (net > best) { best = net; best_s1 = (u16)seed; }
    }
    /* pass 2: fix best s1, scan s2 */
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        u16 s1 = best_s1, s2 = (u16)seed;
        for (int i = 0; i < n; i++) {
            u8 b = (i & 1) ? xs16_next(&s2) : xs16_next(&s1);
            f[(u8)(d[i] + b)]++;
        }
        double net = (S_from_freq(f) - Sb) - OH_PRNGADD;
        if (net > best) { best = net; best_s2 = (u16)seed; }
    }
    /* pass 3: fix best s2, rescan s1 */
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        u16 s1 = (u16)seed, s2 = best_s2;
        for (int i = 0; i < n; i++) {
            u8 b = (i & 1) ? xs16_next(&s2) : xs16_next(&s1);
            f[(u8)(d[i] + b)]++;
        }
        double net = (S_from_freq(f) - Sb) - OH_PRNGADD;
        if (net > best) { best = net; best_s1 = (u16)seed; }
    }
    out->type = T_PRNGADD; out->stride = 0; out->phase = 0;
    out->amp = (u32)best_s1 | ((u32)best_s2 << 16);
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

/* BIT_PLANE_XOR: try all 56 (j,k) pairs with j!=k via freq-table permutation.
 * The transform is self-inverse: perm(v) = (v & jmask) ? v^kmask : v, so
 * new_freq[perm(v)] = old_freq[v]  →  new_freq[v] = old_freq[perm(v)]. */
static double search_bpxor(const u8 *d, int n, double Sb, Instr *out) {
    int freq[256]; freq_of(d, n, freq);
    double best = -1e18; u32 bamp = 0;
    for (int j = 0; j < 8; j++) {
        u8 jmask = (u8)(1 << j);
        for (int k = 0; k < 8; k++) {
            if (k == j) continue;
            u8 kmask = (u8)(1 << k);
            int rf[256];
            for (int v=0;v<256;v++) {
                int src = (v & jmask) ? v ^ kmask : v;
                rf[v] = freq[src];
            }
            double net = (S_from_freq(rf) - Sb) - OH_BPXOR;
            if (net > best) { best = net; bamp = (u32)(j | (k << 3)); }
        }
    }
    out->type = T_BPXOR; out->stride = 0; out->phase = 0; out->amp = bamp;
    return best;
}

/* registry of selectable instructions */
typedef double (*SearchFn)(const u8 *, int, double, Instr *);
typedef struct { const char *name; SearchFn search; int slow; } InstrDesc;

static const InstrDesc REGISTRY[] = {
    { "XOR_PHASE",  search_xorp,      0 },
    { "ADD_NIBS",   search_anibs,     0 },
    { "QUAD_ADD",   search_quadadd,   0 },
    { "PRNG_DUAL",  search_prngd,     1 },   /* slow: 65535-seed scan */
    { "OCT_NIBX",   search_octnibx,   0 },
    { "STRIDE_ADD", search_strideadd, 0 },
    { "PRNG_BIT",   search_prngbit,   1 },   /* slow: 65535-seed scan */
    { "BYTE_ROT",   search_byterot,   0 },
    { "HALF_XOR",   search_halfxor,   0 },
    { "HALF_ADD",   search_halfadd,   0 },
    { "BYTE_MUL",   search_bytemul,   0 },
    { "TRIPLE_XOR", search_triplexor, 0 },
    { "VALUE_XOR",  search_valuexor,  1 }, /* slow: O(stride²×8×127×128) brute force */
    { "BIT_REV",    search_bitrev,    0 },
    { "BP_XOR",     search_bpxor,     0 },
    { "PRNG_ADD",   search_prngadd,   1 },   /* slow: 65535-seed scan × 3 passes */
    { "NIBBLE_LUT", search_niblut,    0 },
    { "NIB_CXOR",   search_nibcxor,   0 },
    { "CRMB_CXOR",  search_crmbcxor,  0 },
};
#define NREG ((int)(sizeof(REGISTRY)/sizeof(REGISTRY[0])))
static const char *TYPE_NAME[NTYPES] = {
    "XOR_PHASE","ADD_NIBS","QUAD_ADD","PRNG_DUAL","OCT_NIBX",
    "STRIDE_ADD","PRNG_BIT","BYTE_ROT","HALF_XOR","HALF_ADD","BYTE_MUL",
    "TRIPLE_XOR","VALUE_XOR","BIT_REV","BP_XOR","PRNG_ADD","NIBBLE_LUT","NIB_CXOR","CRMB_CXOR","REFLECT"
};


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
 *  main                                                         *
 * ============================================================ */

/* validate apply/invert for every instruction type independently */
static int selftest(void) {
    u8 base[BLOCK], a[BLOCK];
    u32 st = 99u;
    for (int i = 0; i < BLOCK; i++) { st = st * 1103515245u + 12345u; base[i] = (u8)(st >> 16); }

    Instr tv[] = {
        { T_XORP,     3, 1, 0x5A },
        { T_ANIBS,    3, 1, 0x35 },
        { T_QUADADD,  3, 1, 0x11223344u },
        { T_PRNGD,    0, 0, (1234u|(5678u<<16)) },
        { T_OCTNIBX,  3, 1, 0x12345678u },
        { T_STRIDEADD,3, 1, 0x37 },
        { T_PRNGBIT,  0, 0, (0xABCDu | (0xA5u << 16)) },
        { T_BYTEROT,  3, 1, 3 },
        { T_HALFXOR,  3, 1, (0x2Au | (0x55u << 8)) },
        { T_HALFADD,  3, 1, (0x13u | (0x41u << 7)) },
        { T_BYTEMUL,  3, 1, 3u },
        { T_TRIPLEXOR,3, 1, 0x112233u },
        { T_VALUEXOR, 3, 1, (3u|(0x24u<<3)|(0x12u<<11)) },
        { T_BITREV,   3, 1, 0 },
        { T_BPXOR,    0, 0, (2u | (5u << 3)) },  /* j=2, k=5 */
        { T_PRNGADD,  0, 0, (1234u | (5678u << 16)) },
        /* NIBLUT: lut = {1,0,3,2,5,4,7,6,9,8,11,10,13,12,15,14} (swap nibble pairs) */
        { T_NIBLUT,  (int)(0x89ABCDEFu), 0, 0x01234567u },
        { T_NIBCXOR, 3, 1, 0 },  /* dir=0: lo ^= hi */
        { T_NIBCXOR,  3, 1, 1 },          /* dir=1: hi ^= lo */
        { T_CRMBCXOR, 3, 1, (0|(2<<2)) }, /* j=0 (bits[1:0]), k=2 (bits[5:4]) */
        { T_REFLECT,  0, 0, 0x40u },
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

/* reduce one block, verify round-trip, accumulate per-type counts + net stats */
static double do_block(u8 *data, int n, int *counts,
                       double *type_net_sum, double *type_net_max,
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
    }

    u8 dec[BLOCK];
    memcpy(dec, data, n);
    decompress(dec, n, g_ilist, g_last_ni);
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
    double total_overhead = 0.0;
    int total_ni = 0, fails = 0;
    /* 8 bytes per instruction: type(1)+stride(1)+phase(2)+amp(4) */
    u8 *ibuf = malloc((size_t)NB * MAXINSTR * 8);
    int ibuf_n = 0;
    clock_t t0 = clock();

    printf("\n=== compressing %d blocks ===\n", NB);
    for (int b = 0; b < NB; b++) {
        u8 *data = all + (size_t)b * BLOCK;
        double e_in = entropy_bits(data, BLOCK);
        int ok = 0;
        double net = do_block(data, BLOCK, counts, type_net_sum, type_net_max, NB == 1, &ok);
        double e_out = entropy_bits(data, BLOCK);
        double raw  = e_in - e_out;
        double oh   = raw - net;
        total_net += net; total_ein += e_in; total_eout += e_out;
        total_overhead += oh; total_ni += g_last_ni;
        if (!ok) fails++;
        printf("  block %2d: %.4f -> %.4f bps  net=%+.1f  %s  [%d instrs  raw=%+.1f  OH=%.1f bits]\n",
               b, e_in / BLOCK, e_out / BLOCK, net, ok ? "ok" : "FAIL",
               g_last_ni, raw, oh);
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
        double ibps = entropy_bits(ibuf, ibuf_n) / ibuf_n;
        printf("instr entropy: %.4f bps  (%d bytes serialised as type+stride+phase+amp)\n",
               ibps, ibuf_n);
    }
    free(ibuf);
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
