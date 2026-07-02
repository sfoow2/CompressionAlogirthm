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
#define MAX_STRIDE 64           /* search strides 1..MAX_STRIDE (phase < stride, so this also caps phase) */
#define MAX_NORMAL_INSTR 64     /* cap on non-PRNG instructions per block; sizes the compact header's count field */

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
    T_PRNGPERM,  /* PRNG-built full 256-perm applied at PRNG-coinflip positions; amp=seed    */
    T_REFLECT,   /* Elias remap around mode M: r=(v-M) signed, out=r>=0?2r:(-2r-1); amp=M   */
    T_SPLITADD,  /* N-way split ADD; kidx in phase[8..9]: 0=dual,1=quad,2=octo,3=half        */
    T_SPLITXOR,  /* N-way split XOR; kidx in phase[8..9]: 0=dual,1=quad,2=octo-nibble,3=half */
    T_PRNGBIT,   /* PRNG selects bit pos (0-7) per byte; XOR that bit with (amp>>bit_pos)&1  */
    T_VALUEMAP4, /* 4-quartile value-conditional lo6 XOR; top-2 bits preserved -> self-inverse */
    T_NIBSWAP,        /* swap hi/lo nibbles v->(v<<4)|(v>>4); self-inverse; stride/phase              */
    T_XORPNP,    /* XOR amp at stride, always phase=0; no phase field stored — saves log2(s) bits  */
    T_GFMUL,     /* multiply each byte by nonzero a in GF(256) (Rijndael field); amp=a; stride/phase */
    T_NIBMUL16,  /* independent odd mul mod 16 per nibble; amp=((a-1)/2)|(((b-1)/2)<<3)              */
    T_NIBCADD,   /* cross-nibble ADD mod16: dir=0 lo+=hi, dir=1 hi+=lo; amp=dir                       */
    T_CRMBCADD,  /* cross-crumb ADD mod4: crumb k += crumb j; amp=j|(k<<2), j!=k                     */
    T_BITSWAP2,  /* swap bits at distance 2: (0,2)(1,3)(4,6)(5,7); self-inverse; stride/phase         */
    T_VALUEADD,  /* sign bit preserved; ADD mod128 to lo7 independently per sign group; amp=alo|(ahi<<7) */
    T_VALUEMUL,  /* sign bit preserved; odd MUL mod128 to lo7 independently per group; amp=aidx|(bidx<<6) */
    T_VALUEMAP4ADD, /* 4-quartile value-conditional lo6 ADD mod64; top-2 bits preserved -> ADD-sib of VALUEMAP4 */
    T_NIBGFMUL,  /* independent GF(16) mul per nibble; amp=(a-1)|((b-1)<<4), a,b in 1..15 (nibble-sib of GF_MUL) */
    T_CRMBMUL,   /* independent GF(4) mul per 2-bit crumb; amp=c0|(c1<<2)|(c2<<4)|(c3<<6), each ci in 0..2 (=a-1) */
    T_VALUEMAP4MUL, /* 4-quartile value-conditional lo6 odd MUL mod64; top-2 bits preserved -> MUL-sib of VALUEMAP4 */
    T_CRMBCROSSMUL, /* crumb k *= (crumb j, or 1 if crumb j==0); amp=j|(k<<2), j!=k                              */
    T_VALUEGFMUL, /* bit-k preserved (searched); GF(128) mul on the other 7 bits, independent per group; amp=k|(aidx<<3)|(bidx<<10) */
    T_NIBCROSSGFMUL, /* nibble k *= (nibble j, or 1 if nibble j==0), GF(16); amp=dir (0: lo*=hi_or_1, 1: hi*=lo_or_1) */
    T_VALUEMAP4GFMUL, /* 4-quartile value-conditional lo6 GF(64) mul; top-2 bits preserved -> GFMUL-sib of VALUEMAP4 */
    T_BITCXOR,   /* bit-level cross-XOR: bit k ^= bit j; amp=j|(k<<3), j!=k; self-inverse; stride/phase */
    T_CRMBIADD,  /* independent ADD mod4 per crumb; amp=c0|(c1<<2)|(c2<<4)|(c3<<6), each ci in 0..3 (ADD-sib of CRMB_MUL) */
    T_GFPOW,     /* nonlinear GF(256) power map v->v^e, e coprime to 255; amp=idx into 128-entry e-table (7 bits, no waste) */
    T_NIBPOW,    /* independent nonlinear GF(16) power map per nibble; amp=idxlo|(idxhi<<3), idx into 8-entry e-table each (nibble-sib of GF_POW) */
    T_VALUEGFPOW, /* bit-k preserved (searched); nonlinear GF(128) power map on the other 7 bits, independent per group; amp=k|(eaidx<<3)|(ebidx<<10) (pow-sib of VALUE_GFMUL) */
    T_VALUEMAP4GFPOW, /* 4-quartile value-conditional GF(64) power map; top-2 bits preserved -> GFPOW-sib of VALUEMAP4GFMUL */
    NTYPES       /* = 42 */
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
#define SB     6.0    /* stride field (1..64 = 6 bits) */
/* PRNG-section tag: the 4 PRNG_TAG_TYPES live in a separate stream-prefix section
 * (see pack_ilist) addressed with a flat 2-bit index instead of the 6-bit TAGB. */
#define PRTAGB 2.0
/* Phase bits are stride-adaptive: log2(s) instead of fixed 6. Use OH_SP(base, s). */
static inline double pb_bits(int s) { return (s > 1) ? log2((double)s) : 0.0; }
#define OH_SP(base, s)  ((base) + pb_bits(s))
#define OH_XORP_BASE    (TAGB + SB + 8.0)
#define OH_NIBS_BASE    (TAGB + SB + 8.0)
#define OH_STRIDEADD_BASE (TAGB + SB + 8.0)
#define OH_BYTEROT_BASE (TAGB + SB + 3.0)
#define OH_BYTEMUL_BASE (TAGB + SB + 7.0)
#define OH_GFMUL_BASE   (TAGB + SB + 8.0)  /* full byte a (1..255), not just odd */
#define OH_NIBMUL16_BASE (TAGB + SB + 6.0) /* 3 bits per nibble multiplier (8 odd values each) */
#define OH_NIBGFMUL_BASE (TAGB + SB + 8.0) /* 4 bits per nibble multiplier (15 nonzero GF16 values each) */
#define OH_CRMBMUL_BASE  (TAGB + SB + 8.0) /* 2 bits per crumb multiplier (3 nonzero GF4 values each), 4 crumbs */
#define OH_CRMBCROSSMUL_BASE (TAGB + SB + 4.0) /* j|(k<<2), same shape as CRMB_CXOR/CRMB_CADD */
#define OH_VALUEGFMUL_BASE (TAGB + SB + 17.0) /* k(3) + aidx(7) + bidx(7): 127 nonzero GF128 values each */
#define OH_NIBCROSSGFMUL_BASE (TAGB + SB + 1.0) /* 1-bit direction, same shape as NIB_CXOR/NIB_CADD */
#define OH_NIBCADD_BASE  (TAGB + SB + 1.0) /* 1-bit direction, same shape as NIB_CXOR */
#define OH_CRMBCADD_BASE (TAGB + SB + 4.0) /* j|(k<<2), same shape as CRMB_CXOR */
#define OH_BITSWAP2_BASE (TAGB + SB)       /* fixed permutation, no amp — same shape as BIT_ASWAP */
#define OH_VALUEADD_BASE (TAGB + SB + 14.0) /* alo(7)+ahi(7), no k field — always top bit */
#define OH_VALUEMUL_BASE (TAGB + SB + 12.0) /* aidx(6)+bidx(6): 64 odd values mod 128 each */
#define OH_VALUEXOR_BASE  (TAGB + SB + 17.0)
#define OH_BITREV_BASE    (TAGB + SB)
#define OH_PRNGADD4       (PRTAGB + 16.0)
#define OH_PRNGADD8       (PRTAGB + 16.0)
#define OH_NIBCXOR_BASE   (TAGB + SB + 1.0)
#define OH_CRMBCXOR_BASE  (TAGB + SB + 4.0)
#define OH_BITCXOR_BASE   (TAGB + SB + 6.0) /* j|(k<<3), j,k in 0..7: 3 bits each */
#define OH_CRMBIADD_BASE  (TAGB + SB + 8.0) /* 2 bits per crumb additive constant (4 values each), 4 crumbs */
#define OH_GFPOW_BASE     (TAGB + SB + 7.0) /* idx into 128-entry coprime-exponent table, no waste */
#define OH_NIBPOW_BASE    (TAGB + SB + 6.0) /* idxlo(3)+idxhi(3): idx into 8-entry coprime-exponent table each */
#define OH_VALUEGFPOW_BASE (TAGB + SB + 17.0) /* k(3) + eaidx(7) + ebidx(7): 126 exponent values each (127 is prime) */
#define OH_GRAYCODE_BASE  (TAGB + SB)
#define OH_BITASWAP_BASE  (TAGB + SB)
#define OH_PRNGPERM       (PRTAGB + 16.0)  /* 16-bit seed only — no plane index needed */
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
#define OH_PRNGBIT      (PRTAGB + 16.0 + 8.0)  /* 16-bit seed + 8-bit flip-mask */
#define OH_NIBSWAP_BASE (TAGB + SB)
static double oh_valuemap4(int s)   { return TAGB + SB + pb_bits(s) + 24.0; } /* 4×6-bit constants */
static double oh_valuemap4mul(int s){ return TAGB + SB + pb_bits(s) + 20.0; } /* 4×5-bit odd multipliers */

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

/* GF_MUL: multiply each byte by a nonzero constant a in GF(256) under the AES/Rijndael
 * reduction polynomial x^8+x^4+x^3+x+1 (0x11B). A genuinely different bijection family
 * from BYTE_MUL's mod-256 integer multiply — XOR-based carry-free arithmetic combines
 * bits in a structurally different way, so it can catch value correlations mod-256
 * multiply misses. Every nonzero element is invertible (GF(256) is a field), so all 255
 * values of a are usable (vs BYTE_MUL's 128 odd-only), still cheaply exhaustive to search.
 * Table precomputed once at startup (see init_gf_mul_tab) so apply/search are O(1) lookups. */
static u8 gf_mul_tab[256][256];
static void init_gf_mul_tab(void) {
    for (int av = 0; av < 256; av++) {
        for (int bv = 0; bv < 256; bv++) {
            u8 a = (u8)av, b = (u8)bv, p = 0;
            for (int i = 0; i < 8; i++) {
                if (b & 1) p ^= a;
                u8 hi = (u8)(a & 0x80);
                a = (u8)(a << 1);
                if (hi) a ^= 0x1B;
                b = (u8)(b >> 1);
            }
            gf_mul_tab[av][bv] = p;
        }
    }
}
static inline u8 gf_mul(u8 a, u8 b) { return gf_mul_tab[a][b]; }
static u8 gf_inv(u8 a) {
    for (int c = 1; c < 256; c++) if (gf_mul(a, (u8)c) == 1) return (u8)c;
    return 1; /* unreachable for a != 0 */
}
static void ap_gfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)(amp & 0xFF);
    for (int i = p; i < n; i += s) d[i] = gf_mul(d[i], a);
}
static void inv_gfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 ainv = gf_inv((u8)(amp & 0xFF));
    for (int i = p; i < n; i += s) d[i] = gf_mul(d[i], ainv);
}

/* GF_POW: nonlinear sibling of GF_MUL/BYTE_MUL — instead of multiplying by a fixed
 * field element (a LINEAR map over GF(256)), raise every byte to a fixed power e in the
 * field's multiplicative group (order 255 = 3*5*17, cyclic, generated by 0x03 under this
 * same x^8+x^4+x^3+x+1 reduction). v^e is invertible over the whole group iff
 * gcd(e,255)=1, with inverse exponent e^-1 mod 255; 0 always maps to 0. There are exactly
 * phi(255)=128 such e values -- a 7-bit index with zero wasted code space. Because
 * exponentiation composes multiplications nonlinearly (unlike GF_MUL's single multiply),
 * this reaches byte-value correlations that no linear GF/mod-256 multiply can. Built via
 * log/antilog tables (discrete log base 0x03) so both directions are O(1) table lookups,
 * same cost profile as GF_MUL. */
static u8 gf_log[256], gf_antilog[256];
static u8 gfpow_elist[128], gfpow_einv[128];
static int gfpow_ne;
static u8 gfpow_tab[128][256], gfpow_itab[128][256];
static void init_gf_log(void) {
    u8 x = 1;
    for (int i = 0; i < 255; i++) {
        gf_antilog[i] = x;
        gf_log[x] = (u8)i;
        x = gf_mul(x, 0x03);
    }
}
static inline u8 gfpow_raw(u8 v, int e) {
    if (v == 0) return 0;
    return gf_antilog[(gf_log[v] * e) % 255];
}
static void init_gfpow(void) {
    gfpow_ne = 0;
    for (int e = 1; e < 255; e++) {
        int a = e, b = 255, x0 = 1, x1 = 0;
        while (b != 0) { int q = a / b, t = b; b = a % b; a = t; t = x1; x1 = x0 - q * x1; x0 = t; }
        if (a != 1) continue; /* gcd(e,255) != 1: not invertible */
        int einv = ((x0 % 255) + 255) % 255;
        int idx = gfpow_ne++;
        gfpow_elist[idx] = (u8)e; gfpow_einv[idx] = (u8)einv;
        for (int v = 0; v < 256; v++) {
            gfpow_tab[idx][v]  = gfpow_raw((u8)v, e);
            gfpow_itab[idx][v] = gfpow_raw((u8)v, einv);
        }
    }
}
static void ap_gfpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *col = gfpow_tab[amp & 0x7F];
    for (int i = p; i < n; i += s) d[i] = col[d[i]];
}
static void inv_gfpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *col = gfpow_itab[amp & 0x7F];
    for (int i = p; i < n; i += s) d[i] = col[d[i]];
}

/* NIB_MUL16: multiply lo/hi nibbles independently by odd constants mod 16 (coprime to 16,
 * so each is a bijection on 0..15). Nibble-granularity sibling of BYTE_MUL, the way
 * ADD_NIBS is to STRIDE_ADD — no carry crosses the nibble boundary. amp packs (a-1)/2 and
 * (b-1)/2 in 3 bits each (8 odd values: 1,3,...,15). */
static u8 inv_mod16(u8 a) {
    for (int c = 1; c < 16; c += 2) if ((u8)((a * c) & 0xF) == 1) return (u8)c;
    return 1; /* unreachable for odd a */
}
static void ap_nibmul16(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)(2 * (amp & 7) + 1);
    u8 b = (u8)(2 * ((amp >> 3) & 7) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = (u8)(((v & 0xF) * a) & 0xF);
        u8 hi = (u8)((((v >> 4) & 0xF) * b) & 0xF);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibmul16(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)(2 * (amp & 7) + 1);
    u8 b = (u8)(2 * ((amp >> 3) & 7) + 1);
    u8 ainv = inv_mod16(a), binv = inv_mod16(b);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = (u8)(((v & 0xF) * ainv) & 0xF);
        u8 hi = (u8)((((v >> 4) & 0xF) * binv) & 0xF);
        d[i] = (u8)(lo | (hi << 4));
    }
}

/* NIB_GFMUL: nibble-granularity sibling of GF_MUL, the way NIB_MUL16 is to BYTE_MUL —
 * multiply lo/hi nibbles independently by nonzero constants in GF(16) under x^4+x+1
 * (0x13). GF(16) is a field, so all 15 nonzero elements are invertible (vs NIB_MUL16's
 * 8 odd-only values) — a different bit-mixing structure at the nibble level, same
 * reasoning as GF_MUL vs BYTE_MUL at the byte level. Table precomputed once at startup. */
static u8 gf16_mul_tab[16][16];
static void init_gf16_mul_tab(void) {
    for (int av = 0; av < 16; av++) {
        for (int bv = 0; bv < 16; bv++) {
            u8 a = (u8)av, b = (u8)bv, p = 0;
            for (int i = 0; i < 4; i++) {
                if (b & 1) p ^= a;
                u8 hi = (u8)(a & 0x8);
                a = (u8)((a << 1) & 0xF);
                if (hi) a ^= 0x3;
                b = (u8)(b >> 1);
            }
            gf16_mul_tab[av][bv] = p;
        }
    }
}
static inline u8 gf16_mul(u8 a, u8 b) { return gf16_mul_tab[a & 0xF][b & 0xF]; }
static u8 gf16_inv(u8 a) {
    for (int c = 1; c < 16; c++) if (gf16_mul(a, (u8)c) == 1) return (u8)c;
    return 1; /* unreachable for a != 0 */
}
static void ap_nibgfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0xF) + 1), b = (u8)(((amp >> 4) & 0xF) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = gf16_mul((u8)(v & 0xF), a);
        u8 hi = gf16_mul((u8)((v >> 4) & 0xF), b);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibgfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0xF) + 1), b = (u8)(((amp >> 4) & 0xF) + 1);
    u8 ainv = gf16_inv(a), binv = gf16_inv(b);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = gf16_mul((u8)(v & 0xF), ainv);
        u8 hi = gf16_mul((u8)((v >> 4) & 0xF), binv);
        d[i] = (u8)(lo | (hi << 4));
    }
}

/* NIB_POW: nibble-granularity sibling of GF_POW — nonlinear power map v->v^e within
 * GF(16)* (order 15 = 3*5, cyclic, generated by 0x02 under x^4+x+1), applied
 * independently per nibble like NIB_GFMUL. phi(15)=8 valid exponents -> 3-bit index per
 * nibble, zero waste. Not redundant with NIB_GFMUL: v^e is nonlinear (not expressible as
 * any single "multiply by a"), so it reaches different value correlations. */
static u8 gf16_log[16], gf16_antilog[16];
static u8 nibpow_elist[8], nibpow_einv[8];
static int nibpow_ne;
static u8 nibpow_tab[8][16], nibpow_itab[8][16];
static void init_gf16_log(void) {
    u8 x = 1;
    for (int i = 0; i < 15; i++) {
        gf16_antilog[i] = x;
        gf16_log[x] = (u8)i;
        x = gf16_mul(x, 0x02);
    }
}
static inline u8 nibpow_raw(u8 v, int e) {
    if (v == 0) return 0;
    return gf16_antilog[(gf16_log[v] * e) % 15];
}
static void init_nibpow(void) {
    nibpow_ne = 0;
    for (int e = 1; e < 15; e++) {
        int a = e, b = 15, x0 = 1, x1 = 0;
        while (b != 0) { int q = a / b, t = b; b = a % b; a = t; t = x1; x1 = x0 - q * x1; x0 = t; }
        if (a != 1) continue;
        int einv = ((x0 % 15) + 15) % 15;
        int idx = nibpow_ne++;
        nibpow_elist[idx] = (u8)e; nibpow_einv[idx] = (u8)einv;
        for (int v = 0; v < 16; v++) {
            nibpow_tab[idx][v]  = nibpow_raw((u8)v, e);
            nibpow_itab[idx][v] = nibpow_raw((u8)v, einv);
        }
    }
}
static void ap_nibpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *tlo = nibpow_tab[amp & 7], *thi = nibpow_tab[(amp >> 3) & 7];
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = tlo[v & 0xF], hi = thi[(v >> 4) & 0xF];
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *tlo = nibpow_itab[amp & 7], *thi = nibpow_itab[(amp >> 3) & 7];
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = tlo[v & 0xF], hi = thi[(v >> 4) & 0xF];
        d[i] = (u8)(lo | (hi << 4));
    }
}

/* CRMB_MUL: crumb-granularity sibling of NIB_GFMUL — multiply each of the 4 independent
 * 2-bit crumbs by a nonzero constant in GF(4) under x^2+x+1. GF(4)* is cyclic of order 3
 * ({1,2,3}), small enough for a direct lookup table instead of a bit-loop. amp packs 4
 * crumb-index fields (2 bits each, value = constant-1, so 0..2 for constants 1..3). */
static inline u8 gf4_mul(u8 a, u8 b) {
    static const u8 tab[4][4] = {
        {0,0,0,0},
        {0,1,2,3},
        {0,2,3,1},
        {0,3,1,2},
    };
    return tab[a & 3][b & 3];
}
static inline u8 gf4_inv(u8 a) {
    static const u8 inv[4] = {0, 1, 3, 2}; /* inv[0] unused */
    return inv[a & 3];
}
static void ap_crmbmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    for (int g = 0; g < 4; g++) c[g] = (u8)(((amp >> (g * 2)) & 3) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) {
            u8 crumb = (u8)((v >> (2 * g)) & 3);
            w = (u8)(w | (gf4_mul(crumb, c[g]) << (2 * g)));
        }
        d[i] = w;
    }
}
static void inv_crmbmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 ci[4];
    for (int g = 0; g < 4; g++) ci[g] = gf4_inv((u8)(((amp >> (g * 2)) & 3) + 1));
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) {
            u8 crumb = (u8)((v >> (2 * g)) & 3);
            w = (u8)(w | (gf4_mul(crumb, ci[g]) << (2 * g)));
        }
        d[i] = w;
    }
}

/* CRMB_IADD: independent-per-crumb ADD mod 4 — direct ADD-sibling of CRMB_MUL (same
 * shape, 4 independent 2-bit fields), just ADD mod 4 instead of GF(4) mul. Unlike
 * CRMB_MUL's multiplier codes (which skip 0, the annihilator), all 4 values 0..3 are
 * valid additive constants — no waste in the 2-bit field. */
static void ap_crmbiadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    for (int g = 0; g < 4; g++) c[g] = (u8)((amp >> (g * 2)) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) {
            u8 crumb = (u8)((v >> (2 * g)) & 3);
            w = (u8)(w | (((crumb + c[g]) & 3) << (2 * g)));
        }
        d[i] = w;
    }
}
static void inv_crmbiadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    for (int g = 0; g < 4; g++) c[g] = (u8)((amp >> (g * 2)) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) {
            u8 crumb = (u8)((v >> (2 * g)) & 3);
            w = (u8)(w | (((crumb - c[g]) & 3) << (2 * g)));
        }
        d[i] = w;
    }
}

/* CRMB_CROSS_MUL: cross-crumb sibling of CRMB_CXOR/CRMB_CADD — crumb k *= crumb j (in
 * GF(4)), instead of XOR/ADD. A raw data-dependent multiplier is NOT invertible whenever
 * crumb j happens to be 0 (multiplying by 0 destroys crumb k's value) — substitute 1 in
 * that case. This is safe and exactly reversible because crumb j is never written by this
 * transform, so the same substitution is recomputed identically at decode time from the
 * untouched crumb j. */
static void ap_crmbcrossmul(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2*j)) & 3);
        u8 mult = cj ? cj : 1;
        u8 ck = (u8)((v >> (2*k)) & 3);
        u8 nck = gf4_mul(ck, mult);
        d[i] = (u8)((v & ~(u8)(3 << (2*k))) | (nck << (2*k)));
    }
}
static void inv_crmbcrossmul(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2*j)) & 3);
        u8 mult = cj ? cj : 1;
        u8 multinv = gf4_inv(mult);
        u8 ck = (u8)((v >> (2*k)) & 3);
        u8 nck = gf4_mul(ck, multinv);
        d[i] = (u8)((v & ~(u8)(3 << (2*k))) | (nck << (2*k)));
    }
}

/* NIB_CROSS_GFMUL: nibble-granularity sibling of CRMB_CROSSMUL — nibble k *= nibble j
 * (in GF(16)), substituting 1 whenever nibble j reads 0 (same zero-annihilator fix, still
 * exactly reversible since nibble j is never written by this transform). Only 2 positions
 * (lo,hi) instead of 4 crumbs, so amp is just a 1-bit direction like NIB_CXOR/NIB_CADD. */
static void ap_nibcrossgfmul(u8 *d, int n, int s, int p, u32 amp) {
    int dir = (int)(amp & 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
        if (dir == 0) { u8 m = hi ? hi : 1; lo = gf16_mul(lo, m); }
        else          { u8 m = lo ? lo : 1; hi = gf16_mul(hi, m); }
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibcrossgfmul(u8 *d, int n, int s, int p, u32 amp) {
    int dir = (int)(amp & 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
        if (dir == 0) { u8 m = hi ? hi : 1; lo = gf16_mul(lo, gf16_inv(m)); }
        else          { u8 m = lo ? lo : 1; hi = gf16_mul(hi, gf16_inv(m)); }
        d[i] = (u8)(lo | (hi << 4));
    }
}

/* NIB_CADD: cross-nibble ADD mod 16 — direct ADD-sibling of NIB_CXOR (same shape, XOR
 * swapped for ADD). Not self-inverse like NIB_CXOR is: dir=0 adds hi into lo (inverse
 * subtracts hi back out); dir=1 adds lo into hi (inverse subtracts lo back out). Either
 * way the untouched nibble is still available at decode time to invert with. */
static void ap_nibcadd(u8 *d, int n, int s, int p, u32 amp) {
    int dir = (int)(amp & 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
        if (dir == 0) lo = (u8)((lo + hi) & 0xF);
        else          hi = (u8)((hi + lo) & 0xF);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibcadd(u8 *d, int n, int s, int p, u32 amp) {
    int dir = (int)(amp & 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
        if (dir == 0) lo = (u8)((lo - hi) & 0xF);
        else          hi = (u8)((hi - lo) & 0xF);
        d[i] = (u8)(lo | (hi << 4));
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

/* VALUE_ADD: ADD-sibling of VALUE_XOR. XOR has no carries, so VALUE_XOR can preserve
 * any bit k by forcing that bit of the amp to 0 — but ADD's carries can still flip a
 * higher bit even when the added constant's own bit k is 0. Restricting to the sign/top
 * bit sidesteps that: the remaining 7 bits are already contiguous at the bottom, so
 * adding mod 128 there never carries into (or out of) the top bit. amp=alo|(ahi<<7). */
static void ap_valueadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0x7F), ahi = (u8)((amp >> 7) & 0x7F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], top = (u8)(v & 0x80), lo7 = (u8)(v & 0x7F);
        u8 nlo7 = (u8)((lo7 + (top ? ahi : alo)) & 0x7F);
        d[i] = (u8)(top | nlo7);
    }
}
static void inv_valueadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0x7F), ahi = (u8)((amp >> 7) & 0x7F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], top = (u8)(v & 0x80), lo7 = (u8)(v & 0x7F);
        u8 nlo7 = (u8)((lo7 - (top ? ahi : alo)) & 0x7F);
        d[i] = (u8)(top | nlo7);
    }
}

/* VALUE_MUL: MUL-sibling of VALUE_ADD, completing the sign-bit-preserved XOR/ADD/MUL
 * trio the same way byte-level ops already have all three. Multiplying the 7 low bits
 * by an odd constant mod 128 is a bijection on 0..127 (128=2^7, odd is coprime); the mod-
 * 256 inverse of an odd number, masked to 7 bits, is also its correct mod-128 inverse
 * (128 divides 256), so mul_inv256 is reused rather than a separate mod-128 routine.
 * amp = aidx|(bidx<<6), a = 2*aidx+1 (64 odd values 1..127). */
static void ap_valuemul(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(2 * (amp & 0x3F) + 1), ahi = (u8)(2 * ((amp >> 6) & 0x3F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], top = (u8)(v & 0x80), lo7 = (u8)(v & 0x7F);
        u8 nlo7 = (u8)((lo7 * (top ? ahi : alo)) & 0x7F);
        d[i] = (u8)(top | nlo7);
    }
}
static void inv_valuemul(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(2 * (amp & 0x3F) + 1), ahi = (u8)(2 * ((amp >> 6) & 0x3F) + 1);
    u8 aloinv = (u8)(mul_inv256(alo) & 0x7F), ahiinv = (u8)(mul_inv256(ahi) & 0x7F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], top = (u8)(v & 0x80), lo7 = (u8)(v & 0x7F);
        u8 nlo7 = (u8)((lo7 * (top ? ahiinv : aloinv)) & 0x7F);
        d[i] = (u8)(top | nlo7);
    }
}

/* VALUE_GFMUL: generalizes VALUE_MUL from "sign bit only" to an arbitrary preserved bit
 * k (searched), using GF(128) field multiply (x^7+x+1) on the compacted other 7 bits
 * instead of mod-128 integer multiply — same GF-vs-int-mult relationship as GF_MUL is to
 * BYTE_MUL. Needs the bit-extraction/reinsertion trick VALUE_PERM used (bit k isn't
 * already contiguous with the rest unless k=7) — that trick was never VALUE_PERM's real
 * problem; GF(128) has only 127 nonzero elements (fully searchable), so this doesn't
 * inherit VALUE_PERM's actual flaw (sparse coverage of an astronomically large
 * permutation space). amp = k|(aidx<<3)|(bidx<<10), a=aidx+1, b=bidx+1, aidx/bidx in 0..126. */
static u8 gf128_mul_tab[128][128];
static void init_gf128_mul_tab(void) {
    for (int av = 0; av < 128; av++) {
        for (int bv = 0; bv < 128; bv++) {
            u8 a = (u8)av, b = (u8)bv, p = 0;
            for (int i = 0; i < 7; i++) {
                if (b & 1) p ^= a;
                u8 hi = (u8)(a & 0x40);
                a = (u8)((a << 1) & 0x7F);
                if (hi) a ^= 0x03;
                b = (u8)(b >> 1);
            }
            gf128_mul_tab[av][bv] = p;
        }
    }
}
static inline u8 gf128_mul(u8 a, u8 b) { return gf128_mul_tab[a & 0x7F][b & 0x7F]; }
static u8 gf128_inv(u8 a) {
    for (int c = 1; c < 128; c++) if (gf128_mul(a, (u8)c) == 1) return (u8)c;
    return 1; /* unreachable for a != 0 */
}
static inline u8 vk_idx(u8 v, int k) { int mask = (1 << k) - 1; return (u8)((v & mask) | ((v >> (k + 1)) << k)); }
static inline u8 vk_unidx(u8 idx, int gbit, int k) { int mask = (1 << k) - 1; return (u8)((idx & mask) | (gbit << k) | ((idx >> k) << (k + 1))); }
static void ap_valuegfmul(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)(amp & 7);
    u8 a = (u8)(((amp >> 3) & 0x7F) + 1), b = (u8)(((amp >> 10) & 0x7F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> k) & 1;
        u8 idx = vk_idx(v, k);
        u8 nidx = gf128_mul(idx, gbit ? b : a);
        d[i] = vk_unidx(nidx, gbit, k);
    }
}
static void inv_valuegfmul(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)(amp & 7);
    u8 a = (u8)(((amp >> 3) & 0x7F) + 1), b = (u8)(((amp >> 10) & 0x7F) + 1);
    u8 ainv = gf128_inv(a), binv = gf128_inv(b);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> k) & 1;
        u8 idx = vk_idx(v, k);
        u8 nidx = gf128_mul(idx, gbit ? binv : ainv);
        d[i] = vk_unidx(nidx, gbit, k);
    }
}

/* VALUE_GFPOW: nonlinear sibling of VALUE_GFMUL — bit-k preserved (same vk_idx/vk_unidx
 * split), but the two groups (gbit=0/1) each get an independent GF(128) power-map
 * exponent instead of a multiplier. GF(128)* has PRIME order 127, so every exponent
 * 1..126 is invertible (no gcd filtering needed) -- 126 values, same 7-bit width as
 * VALUE_GFMUL's multiplier index, so no overhead cost, just a different (nonlinear)
 * transform space, same relationship as GF_POW to GF_MUL. */
static u8 gf128_log[128], gf128_antilog[128];
static void init_gf128_log(void) {
    u8 x = 1;
    for (int i = 0; i < 127; i++) {
        gf128_antilog[i] = x;
        gf128_log[x] = (u8)i;
        x = gf128_mul(x, 0x02);
    }
}
static u8 vgfpow_elist[126], vgfpow_einv[126];
static int vgfpow_ne;
static u8 vgfpow_tab[126][128], vgfpow_itab[126][128];
static inline u8 vgfpow_raw(u8 v, int e) {
    if (v == 0) return 0;
    return gf128_antilog[(gf128_log[v] * e) % 127];
}
static void init_vgfpow(void) {
    vgfpow_ne = 0;
    for (int e = 1; e < 127; e++) {
        int a = e, b = 127, x0 = 1, x1 = 0;
        while (b != 0) { int q = a / b, t = b; b = a % b; a = t; t = x1; x1 = x0 - q * x1; x0 = t; }
        int einv = ((x0 % 127) + 127) % 127;
        int idx = vgfpow_ne++;
        vgfpow_elist[idx] = (u8)e; vgfpow_einv[idx] = (u8)einv;
        for (int v = 0; v < 128; v++) {
            vgfpow_tab[idx][v]  = vgfpow_raw((u8)v, e);
            vgfpow_itab[idx][v] = vgfpow_raw((u8)v, einv);
        }
    }
}
static void ap_valuegfpow(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)(amp & 7);
    const u8 *ta = vgfpow_tab[(amp >> 3) & 0x7F], *tb = vgfpow_tab[(amp >> 10) & 0x7F];
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> k) & 1;
        u8 idx = vk_idx(v, k);
        u8 nidx = gbit ? tb[idx] : ta[idx];
        d[i] = vk_unidx(nidx, gbit, k);
    }
}
static void inv_valuegfpow(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)(amp & 7);
    const u8 *ta = vgfpow_itab[(amp >> 3) & 0x7F], *tb = vgfpow_itab[(amp >> 10) & 0x7F];
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> k) & 1;
        u8 idx = vk_idx(v, k);
        u8 nidx = gbit ? tb[idx] : ta[idx];
        d[i] = vk_unidx(nidx, gbit, k);
    }
}

/* VALUEMAP4_GFMUL: GF(64) sibling of VALUEMAP4_MUL, completing the quartile-level
 * XOR/ADD/MUL-int/MUL-GF family the same way VALUE_GFMUL completes it at the sign-bit
 * level. GF(64) under x^6+x+1 gives all 63 nonzero elements (vs the mod-64 int version's
 * 32 odd-only). amp packs 4 quartile codes, 6 bits each (code -> a = code+1, 1..63). */
static u8 gf64_mul_tab[64][64];
static void init_gf64_mul_tab(void) {
    for (int av = 0; av < 64; av++) {
        for (int bv = 0; bv < 64; bv++) {
            u8 a = (u8)av, b = (u8)bv, p = 0;
            for (int i = 0; i < 6; i++) {
                if (b & 1) p ^= a;
                u8 hi = (u8)(a & 0x20);
                a = (u8)((a << 1) & 0x3F);
                if (hi) a ^= 0x03;
                b = (u8)(b >> 1);
            }
            gf64_mul_tab[av][bv] = p;
        }
    }
}
static inline u8 gf64_mul(u8 a, u8 b) { return gf64_mul_tab[a & 0x3F][b & 0x3F]; }
static u8 gf64_inv(u8 a) {
    for (int c = 1; c < 64; c++) if (gf64_mul(a, (u8)c) == 1) return (u8)c;
    return 1; /* unreachable for a != 0 */
}
static void ap_valuemap4gfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    for (int q = 0; q < 4; q++) c[q] = (u8)(((amp >> (q * 6)) & 0x3F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        d[i] = (u8)((v & 0xC0) | gf64_mul((u8)(v & 0x3F), c[v >> 6]));
    }
}
static void inv_valuemap4gfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4], ci[4];
    for (int q = 0; q < 4; q++) { c[q] = (u8)(((amp >> (q * 6)) & 0x3F) + 1); ci[q] = gf64_inv(c[q]); }
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        d[i] = (u8)((v & 0xC0) | gf64_mul((u8)(v & 0x3F), ci[v >> 6]));
    }
}

/* VALUEMAP4_GFPOW: nonlinear sibling of VALUEMAP4_GFMUL — same quartile split (top 2
 * bits select one of 4 independent groups), but each quartile gets a GF(64) power-map
 * exponent instead of a multiplier. GF(64)* has order 63=7*3^2, phi(63)=36 valid
 * exponents -- fits the same 6-bit-per-quartile field VM4GFMUL already uses (6 bits is
 * the minimum needed for 36 values regardless; some code space unused, same as any
 * non-power-of-2 domain, not a new inefficiency). Same relationship to VM4GFMUL as
 * GF_POW to GF_MUL. */
static u8 gf64_log[64], gf64_antilog[64];
static void init_gf64_log(void) {
    u8 x = 1;
    for (int i = 0; i < 63; i++) {
        gf64_antilog[i] = x;
        gf64_log[x] = (u8)i;
        x = gf64_mul(x, 0x02);
    }
}
static u8 vm4pow_elist[36], vm4pow_einv[36];
static int vm4pow_ne;
static u8 vm4pow_tab[36][64], vm4pow_itab[36][64];
static inline u8 vm4pow_raw(u8 v, int e) {
    if (v == 0) return 0;
    return gf64_antilog[(gf64_log[v] * e) % 63];
}
static void init_vm4pow(void) {
    vm4pow_ne = 0;
    for (int e = 1; e < 63; e++) {
        int a = e, b = 63, x0 = 1, x1 = 0;
        while (b != 0) { int q = a / b, t = b; b = a % b; a = t; t = x1; x1 = x0 - q * x1; x0 = t; }
        if (a != 1) continue; /* gcd(e,63) != 1: not invertible */
        int einv = ((x0 % 63) + 63) % 63;
        int idx = vm4pow_ne++;
        vm4pow_elist[idx] = (u8)e; vm4pow_einv[idx] = (u8)einv;
        for (int v = 0; v < 64; v++) {
            vm4pow_tab[idx][v]  = vm4pow_raw((u8)v, e);
            vm4pow_itab[idx][v] = vm4pow_raw((u8)v, einv);
        }
    }
}
static void ap_valuemap4gfpow(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    for (int q = 0; q < 4; q++) c[q] = (u8)((amp >> (q * 6)) & 0x3F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        d[i] = (u8)((v & 0xC0) | vm4pow_tab[c[v >> 6]][v & 0x3F]);
    }
}
static void inv_valuemap4gfpow(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    for (int q = 0; q < 4; q++) c[q] = (u8)((amp >> (q * 6)) & 0x3F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        d[i] = (u8)((v & 0xC0) | vm4pow_itab[c[v >> 6]][v & 0x3F]);
    }
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

/* BIT_CROSS_XOR: finest-granularity sibling of CRMB_CXOR/NIB_CXOR — XOR single bit k
 * with single bit j within each byte. amp = j|(k<<3), j,k in 0..7, j!=k. Self-inverse
 * (bit j is only read, never written by this transform). Bit-level cross-ADD would be
 * identical to cross-XOR (mod-2 addition = XOR) so it's redundant and not implemented;
 * bit-level cross-MUL is trivial (GF(2)'s only nonzero multiplier is 1, i.e. identity)
 * so it's meaningless and also not implemented — XOR is the only nontrivial op at this
 * granularity. */
static void ap_bitcxor(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 7), k = (int)((amp >> 3) & 7);
    for (int i = p; i < n; i += s)
        d[i] ^= (u8)(((d[i] >> j) & 1) << k);
}

/* CRUMB_CROSS_ADD: ADD-sibling of CRMB_CXOR — same shape (2-bit crumb k gets crumb j
 * added into it, mod 4), just ADD instead of XOR. Not self-inverse: crumb j is left
 * untouched so it's still available at decode time to subtract back out. j must != k
 * (j==k would add a crumb into itself, i.e. double it mod 4 — not invertible since 2 has
 * no inverse mod 4). */
static void ap_crmbcadd(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 src = (u8)((v >> (2*j)) & 3);
        u8 dst = (u8)(((v >> (2*k)) & 3) + src) & 3;
        d[i] = (u8)((v & ~(u8)(3 << (2*k))) | (dst << (2*k)));
    }
}
static void inv_crmbcadd(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 src = (u8)((v >> (2*j)) & 3);
        u8 dst = (u8)(((v >> (2*k)) & 3) - src) & 3;
        d[i] = (u8)((v & ~(u8)(3 << (2*k))) | (dst << (2*k)));
    }
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

/* BIT_SWAP2: swap bits at distance 2 within each byte: 0<->2, 1<->3, 4<->6, 5<->7.
 * A distinct fixed permutation from BIT_ASWAP (distance 1) and NIB_SWAP (distance 4).
 * Self-inverse; no amp needed beyond stride/phase. */
static inline u8 bitswap2(u8 v) {
    u8 a = (u8)(v & 0x33); /* bits 0,1,4,5 */
    u8 b = (u8)(v & 0xCC); /* bits 2,3,6,7 */
    return (u8)((a << 2) | (b >> 2));
}
static void ap_bitswap2(u8 *d, int n, int s, int p) {
    for (int i = p; i < n; i += s) d[i] = bitswap2(d[i]);
}


/* PRNG_PERM: build a full 256-entry permutation from the seed via Fisher-Yates,
 * then continue the same stream as a per-position coin flip deciding whether to
 * apply it. Generalizes PLANE_PRNG's coin-flip partition to an unconstrained
 * bijection on the "heads" side instead of a single-bit XOR. */
static void build_prngperm256(u16 *s, u8 perm[256]) {
    for (int i = 0; i < 256; i++) perm[i] = (u8)i;
    for (int i = 255; i > 0; i--) {
        int j = (int)(xs16_next(s) % (unsigned)(i + 1));
        u8 t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
}
static void ap_prngperm(u8 *d, int n, u32 amp) {
    u16 s = (u16)(amp & 0xFFFF);
    u8 perm[256];
    build_prngperm256(&s, perm);
    for (int i = 0; i < n; i++) {
        u16 r = xs16_next(&s);
        if (r & 1) d[i] = perm[d[i]];
    }
}
static void inv_prngperm(u8 *d, int n, u32 amp) {
    u16 s = (u16)(amp & 0xFFFF);
    u8 perm[256], invp[256];
    build_prngperm256(&s, perm);
    for (int i = 0; i < 256; i++) invp[perm[i]] = (u8)i;
    for (int i = 0; i < n; i++) {
        u16 r = xs16_next(&s);
        if (r & 1) d[i] = invp[d[i]];
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

/* VALUEMAP4_ADD: ADD-sibling of VALUEMAP4 — same top-2-bits-preserved quartile split,
 * ADD mod 64 to the lower 6 bits instead of XOR. Safe for the same reason VALUE_ADD is:
 * the lower 6 bits are already contiguous, so adding mod 64 never carries into the
 * preserved top 2 bits. Not self-inverse (ADD needs an explicit subtract to invert). */
static void ap_valuemap4add(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    c[0] = (u8)( amp        & 0x3F);
    c[1] = (u8)((amp >>  6) & 0x3F);
    c[2] = (u8)((amp >> 12) & 0x3F);
    c[3] = (u8)((amp >> 18) & 0x3F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        d[i] = (u8)((v & 0xC0) | (u8)(((v & 0x3F) + c[v >> 6]) & 0x3F));
    }
}
static void inv_valuemap4add(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    c[0] = (u8)( amp        & 0x3F);
    c[1] = (u8)((amp >>  6) & 0x3F);
    c[2] = (u8)((amp >> 12) & 0x3F);
    c[3] = (u8)((amp >> 18) & 0x3F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        d[i] = (u8)((v & 0xC0) | (u8)(((v & 0x3F) - c[v >> 6]) & 0x3F));
    }
}

/* VALUEMAP4_MUL: completes the XOR/ADD/MUL trio at the quartile level, the way
 * VALUE_XOR/VALUE_ADD/VALUE_MUL already do at the sign-bit level. Multiply the lower 6
 * bits by an odd constant mod 64 per quartile (odd is coprime to 64, so a bijection on
 * 0..63); the mod-256 inverse of an odd number, masked to 6 bits, is also its correct
 * mod-64 inverse (64 divides 256), so mul_inv256 is reused, same trick as VALUE_MUL.
 * amp packs 4 quartile codes (5 bits each: code -> a = 2*code+1, 32 odd values 1..63). */
static void ap_valuemap4mul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    for (int q = 0; q < 4; q++) c[q] = (u8)(2 * ((amp >> (q * 5)) & 0x1F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        d[i] = (u8)((v & 0xC0) | (u8)(((v & 0x3F) * c[v >> 6]) & 0x3F));
    }
}
static void inv_valuemap4mul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4], ci[4];
    for (int q = 0; q < 4; q++) {
        c[q] = (u8)(2 * ((amp >> (q * 5)) & 0x1F) + 1);
        ci[q] = (u8)(mul_inv256(c[q]) & 0x3F);
    }
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        d[i] = (u8)((v & 0xC0) | (u8)(((v & 0x3F) * ci[v >> 6]) & 0x3F));
    }
}

/* NIBSWAP: swap high and low nibbles of each byte: v -> (v<<4)|(v>>4). Self-inverse. */
static void ap_nibswap(u8 *d, int n, int s, int p) {
    for (int i = p; i < n; i += s)
        d[i] = (u8)((d[i] << 4) | (d[i] >> 4));
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
        case T_PRNGPERM: ap_prngperm(d, n, t.amp); break;
        case T_REFLECT:  ap_reflect(d, n, (u8)t.amp); break;
        case T_SPLITADD: ap_splitadd(d, n, t.stride, t.phase, t.amp); break;
        case T_SPLITXOR: ap_splitxor(d, n, t.stride, t.phase, t.amp); break;
        case T_PRNGBIT:      ap_prngbit(d, n, t.amp); break;
        case T_VALUEMAP4:    ap_valuemap4(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBSWAP:        ap_nibswap(d, n, t.stride, t.phase); break;
        case T_GFMUL:          ap_gfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBMUL16:       ap_nibmul16(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBCADD:        ap_nibcadd(d, n, t.stride, t.phase, t.amp); break;
        case T_CRMBCADD:       ap_crmbcadd(d, n, t.stride, t.phase, t.amp); break;
        case T_BITSWAP2:       ap_bitswap2(d, n, t.stride, t.phase); break;
        case T_VALUEADD:       ap_valueadd(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMUL:       ap_valuemul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMAP4ADD:   ap_valuemap4add(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBGFMUL:       ap_nibgfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_CRMBMUL:        ap_crmbmul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMAP4MUL:   ap_valuemap4mul(d, n, t.stride, t.phase, t.amp); break;
        case T_CRMBCROSSMUL:   ap_crmbcrossmul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEGFMUL:     ap_valuegfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBCROSSGFMUL:  ap_nibcrossgfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMAP4GFMUL: ap_valuemap4gfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_BITCXOR:        ap_bitcxor(d, n, t.stride, t.phase, t.amp); break;
        case T_CRMBIADD:       ap_crmbiadd(d, n, t.stride, t.phase, t.amp); break;
        case T_GFPOW:          ap_gfpow(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBPOW:         ap_nibpow(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEGFPOW:     ap_valuegfpow(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMAP4GFPOW: ap_valuemap4gfpow(d, n, t.stride, t.phase, t.amp); break;
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
        case T_PRNGPERM: inv_prngperm(d, n, t.amp); break;
        case T_REFLECT:  inv_reflect(d, n, (u8)t.amp); break;
        case T_SPLITADD: inv_splitadd(d, n, t.stride, t.phase, t.amp); break;
        case T_SPLITXOR: ap_splitxor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_PRNGBIT:      ap_prngbit(d, n, t.amp); break;                      /* self-inv */
        case T_VALUEMAP4:    ap_valuemap4(d, n, t.stride, t.phase, t.amp); break; /* self-inv */
        case T_NIBSWAP:        ap_nibswap(d, n, t.stride, t.phase); break;          /* self-inv */
        case T_GFMUL:          inv_gfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBMUL16:       inv_nibmul16(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBCADD:        inv_nibcadd(d, n, t.stride, t.phase, t.amp); break;
        case T_CRMBCADD:       inv_crmbcadd(d, n, t.stride, t.phase, t.amp); break;
        case T_BITSWAP2:       ap_bitswap2(d, n, t.stride, t.phase); break;          /* self-inv */
        case T_VALUEADD:       inv_valueadd(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMUL:       inv_valuemul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMAP4ADD:   inv_valuemap4add(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBGFMUL:       inv_nibgfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_CRMBMUL:        inv_crmbmul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMAP4MUL:   inv_valuemap4mul(d, n, t.stride, t.phase, t.amp); break;
        case T_CRMBCROSSMUL:   inv_crmbcrossmul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEGFMUL:     inv_valuegfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBCROSSGFMUL:  inv_nibcrossgfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMAP4GFMUL: inv_valuemap4gfmul(d, n, t.stride, t.phase, t.amp); break;
        case T_BITCXOR:        ap_bitcxor(d, n, t.stride, t.phase, t.amp); break;  /* self-inv */
        case T_CRMBIADD:       inv_crmbiadd(d, n, t.stride, t.phase, t.amp); break;
        case T_GFPOW:          inv_gfpow(d, n, t.stride, t.phase, t.amp); break;
        case T_NIBPOW:         inv_nibpow(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEGFPOW:     inv_valuegfpow(d, n, t.stride, t.phase, t.amp); break;
        case T_VALUEMAP4GFPOW: inv_valuemap4gfpow(d, n, t.stride, t.phase, t.amp); break;
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
/* GF_MUL: same frequency-table trick as BYTE_MUL, but the forward map is a GF(256) table
 * lookup instead of a mod-256 multiply. All 255 nonzero a are usable (field, not just
 * odd), skip a=1 (identity). */
static double search_gfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 2;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_GFMUL_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 2; a < 256; a++) {  /* skip a=0 (annihilates) and a=1 (identity) */
                int rf[256];
                memcpy(rf, base, sizeof rf);
                const u8 *col = gf_mul_tab[a];
                for (int u = 0; u < 256; u++) rf[col[u]] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->type = T_GFMUL; out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* GF_POW: same shape as search_gfmul, but iterating the 128-entry coprime-exponent
 * table (gfpow_tab) instead of the 254 nonzero multipliers. */
static double search_gfpow(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_GFPOW_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int idx = 0; idx < gfpow_ne; idx++) {
                int rf[256];
                memcpy(rf, base, sizeof rf);
                const u8 *col = gfpow_tab[idx];
                for (int u = 0; u < 256; u++) rf[col[u]] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)idx; }
            }
        }
    }
    out->type = T_GFPOW; out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* NIB_MUL16: lo/hi nibbles never interact in the output, so a joint 8x8 brute force
 * (rather than a coordinate-descent approximation) is both exact and cheap. */
static double search_nibmul16(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_NIBMUL16_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int ai = 0; ai < 8; ai++) {
                u8 a = (u8)(2 * ai + 1);
                for (int bi = 0; bi < 8; bi++) {
                    u8 b = (u8)(2 * bi + 1);
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 lo = (u8)(((u & 0xF) * a) & 0xF);
                        u8 hi = (u8)((((u >> 4) & 0xF) * b) & 0xF);
                        rf[(u8)(lo | (hi << 4))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; bamp = (u32)ai | ((u32)bi << 3); }
                }
            }
        }
    }
    out->type = T_NIBMUL16; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* NIB_GFMUL: same joint-brute-force shape as NIB_MUL16 — 15x15 combos instead of 8x8,
 * still cheap since nibbles stay independent. */
static double search_nibgfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_NIBGFMUL_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int ai = 0; ai < 15; ai++) {
                u8 a = (u8)(ai + 1);
                for (int bi = 0; bi < 15; bi++) {
                    u8 b = (u8)(bi + 1);
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 lo = gf16_mul((u8)(u & 0xF), a);
                        u8 hi = gf16_mul((u8)((u >> 4) & 0xF), b);
                        rf[(u8)(lo | (hi << 4))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; bamp = (u32)ai | ((u32)bi << 4); }
                }
            }
        }
    }
    out->type = T_NIBGFMUL; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* NIB_POW: same joint-brute-force shape as NIB_GFMUL, but only 8x8=64 combos (fewer
 * than NIB_GFMUL's 15x15, since phi(15)=8 valid exponents vs 15 nonzero multipliers). */
static double search_nibpow(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_NIBPOW_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int ai = 0; ai < nibpow_ne; ai++) {
                const u8 *tlo = nibpow_tab[ai];
                for (int bi = 0; bi < nibpow_ne; bi++) {
                    const u8 *thi = nibpow_tab[bi];
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 lo = tlo[u & 0xF], hi = thi[(u >> 4) & 0xF];
                        rf[(u8)(lo | (hi << 4))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; bamp = (u32)ai | ((u32)bi << 3); }
                }
            }
        }
    }
    out->type = T_NIBPOW; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* CRMB_MUL: 4 independent crumbs, each with only 3 nonzero GF(4) candidates — a full
 * joint 3^4=81-combo brute force per (stride,phase) is still cheap. */
static double search_crmbmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_CRMBMUL_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int c0 = 0; c0 < 3; c0++) for (int c1 = 0; c1 < 3; c1++)
            for (int c2 = 0; c2 < 3; c2++) for (int c3 = 0; c3 < 3; c3++) {
                u8 c[4] = { (u8)(c0+1), (u8)(c1+1), (u8)(c2+1), (u8)(c3+1) };
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    u8 w = 0;
                    for (int g = 0; g < 4; g++) {
                        u8 crumb = (u8)((u >> (2*g)) & 3);
                        w = (u8)(w | (gf4_mul(crumb, c[g]) << (2*g)));
                    }
                    rf[w] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) {
                    best = net; bs = s; bp = p;
                    bamp = (u32)c0 | ((u32)c1 << 2) | ((u32)c2 << 4) | ((u32)c3 << 6);
                }
            }
        }
    }
    out->type = T_CRMBMUL; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* CRMB_IADD: same shape as search_crmbmul, ADD mod4 instead of GF(4) mul (all 4 codes
 * 0..3 are valid additive constants, no annihilator to skip). */
static double search_crmbiadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_CRMBIADD_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int c0 = 0; c0 < 4; c0++) for (int c1 = 0; c1 < 4; c1++)
            for (int c2 = 0; c2 < 4; c2++) for (int c3 = 0; c3 < 4; c3++) {
                u8 c[4] = { (u8)c0, (u8)c1, (u8)c2, (u8)c3 };
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    u8 w = 0;
                    for (int g = 0; g < 4; g++) {
                        u8 crumb = (u8)((u >> (2*g)) & 3);
                        w = (u8)(w | (((crumb + c[g]) & 3) << (2*g)));
                    }
                    rf[w] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) {
                    best = net; bs = s; bp = p;
                    bamp = (u32)c0 | ((u32)c1 << 2) | ((u32)c2 << 4) | ((u32)c3 << 6);
                }
            }
        }
    }
    out->type = T_CRMBIADD; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* NIB_CADD: same shape as search_nibcxor, just the two candidate transforms are ADD
 * instead of XOR. */
static double search_nibcadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_NIBCADD_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            for (int dir = 0; dir < 2; dir++) {
                int rf[256];
                for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
                for (int v = 0; v < 256; v++) {
                    u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
                    u8 w = (dir == 0)
                        ? (u8)(((lo + hi) & 0xF) | (hi << 4))
                        : (u8)(lo | (((hi + lo) & 0xF) << 4));
                    rf[w] += hit[v];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bamp = (u32)dir; }
            }
        }
    }
    out->type = T_NIBCADD; out->stride = bs; out->phase = bp; out->amp = bamp;
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

/* VALUE_ADD: same independent-group-optimization shape as VALUE_XOR, but the top bit
 * splits the value space into two contiguous 128-value halves, so each half's best
 * additive shift (mod 128) can be found directly — no cross-half interaction, so the
 * combined score is just the sum of the two independent optima. */
static double search_valueadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_VALUEADD_BASE, s);
        for (int p = 0; p < s; p++) {
            int phF[256] = {0};
            for (int i = p; i < n; i += s) phF[d[i]]++;
            int dv[256];
            for (int v = 0; v < 256; v++) dv[v] = total[v] - phF[v];
            int balo = 0; double bS0 = -1e30;
            for (int a = 0; a < 128; a++) {
                double S = 0.0;
                for (int lo = 0; lo < 128; lo++) {
                    int vin = (lo - a) & 0x7F;
                    S += hlog[dv[lo] + phF[vin]];
                }
                if (S > bS0) { bS0 = S; balo = a; }
            }
            int bahi = 0; double bS1 = -1e30;
            for (int a = 0; a < 128; a++) {
                double S = 0.0;
                for (int lo = 0; lo < 128; lo++) {
                    int w = 0x80 | lo;
                    int vin = 0x80 | ((lo - a) & 0x7F);
                    S += hlog[dv[w] + phF[vin]];
                }
                if (S > bS1) { bS1 = S; bahi = a; }
            }
            double net = (bS0 + bS1 - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; bamp = (u32)balo | ((u32)bahi << 7); }
        }
    }
    out->type = T_VALUEADD; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* VALUE_MUL: same shape as search_valueadd, but the forward map per candidate is
 * multiply-by-odd-a mod 128 instead of add-a mod 128, so recovering the input for a
 * given output needs the candidate's own mod-128 inverse (via mul_inv256, masked). */
static double search_valuemul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_VALUEMUL_BASE, s);
        for (int p = 0; p < s; p++) {
            int phF[256] = {0};
            for (int i = p; i < n; i += s) phF[d[i]]++;
            int dv[256];
            for (int v = 0; v < 256; v++) dv[v] = total[v] - phF[v];
            int balo = 0; double bS0 = -1e30;
            for (int ai = 0; ai < 64; ai++) {
                u8 ainv = (u8)(mul_inv256((u8)(2 * ai + 1)) & 0x7F);
                double S = 0.0;
                for (int w = 0; w < 128; w++) {
                    int vin = (int)((u8)(w * ainv) & 0x7F);
                    S += hlog[dv[w] + phF[vin]];
                }
                if (S > bS0) { bS0 = S; balo = ai; }
            }
            int bahi = 0; double bS1 = -1e30;
            for (int ai = 0; ai < 64; ai++) {
                u8 ainv = (u8)(mul_inv256((u8)(2 * ai + 1)) & 0x7F);
                double S = 0.0;
                for (int lo = 0; lo < 128; lo++) {
                    int w = 0x80 | lo;
                    int vin = 0x80 | (int)((u8)(lo * ainv) & 0x7F);
                    S += hlog[dv[w] + phF[vin]];
                }
                if (S > bS1) { bS1 = S; bahi = ai; }
            }
            double net = (bS0 + bS1 - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; bamp = (u32)balo | ((u32)bahi << 6); }
        }
    }
    out->type = T_VALUEMUL; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* VALUE_GFMUL: same independent-group-optimization shape as VALUE_MUL, but loops over
 * every preserved bit k (like VALUE_XOR does) instead of being fixed to the sign bit, and
 * uses the GF(128) forward map through vk_idx/vk_unidx instead of a plain mod-128
 * multiply. */
static double search_valuegfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_VALUEGFMUL_BASE, s);
        for (int p = 0; p < s; p++) {
            int phF[256] = {0};
            for (int i = p; i < n; i += s) phF[d[i]]++;
            int dv[256];
            for (int v = 0; v < 256; v++) dv[v] = total[v] - phF[v];
            for (int k = 0; k < 8; k++) {
                int balo = 1; double bS0 = -1e30;
                for (int ai = 1; ai < 128; ai++) {
                    u8 ainv = gf128_inv((u8)ai);
                    double S = 0.0;
                    for (int w = 0; w < 128; w++) {
                        u8 outv = vk_unidx((u8)w, 0, k);
                        u8 inv_v = vk_unidx(gf128_mul((u8)w, ainv), 0, k);
                        S += hlog[dv[outv] + phF[inv_v]];
                    }
                    if (S > bS0) { bS0 = S; balo = ai; }
                }
                int bahi = 1; double bS1 = -1e30;
                for (int ai = 1; ai < 128; ai++) {
                    u8 ainv = gf128_inv((u8)ai);
                    double S = 0.0;
                    for (int w = 0; w < 128; w++) {
                        u8 outv = vk_unidx((u8)w, 1, k);
                        u8 inv_v = vk_unidx(gf128_mul((u8)w, ainv), 1, k);
                        S += hlog[dv[outv] + phF[inv_v]];
                    }
                    if (S > bS1) { bS1 = S; bahi = ai; }
                }
                double net = (bS0 + bS1 - Sb) - oh;
                if (net > best) {
                    best = net; bs = s; bp = p;
                    bamp = (u32)k | ((u32)(balo - 1) << 3) | ((u32)(bahi - 1) << 10);
                }
            }
        }
    }
    out->type = T_VALUEGFMUL; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* VALUE_GFPOW: same shape as search_valuegfmul, iterating the 126-entry power-exponent
 * table (vgfpow_itab, for the inverse lookup) instead of the 127 nonzero multipliers. */
static double search_valuegfpow(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_VALUEGFPOW_BASE, s);
        for (int p = 0; p < s; p++) {
            int phF[256] = {0};
            for (int i = p; i < n; i += s) phF[d[i]]++;
            int dv[256];
            for (int v = 0; v < 256; v++) dv[v] = total[v] - phF[v];
            for (int k = 0; k < 8; k++) {
                int balo = 0; double bS0 = -1e30;
                for (int ei = 0; ei < vgfpow_ne; ei++) {
                    const u8 *itab = vgfpow_itab[ei];
                    double S = 0.0;
                    for (int w = 0; w < 128; w++) {
                        u8 outv = vk_unidx((u8)w, 0, k);
                        u8 inv_v = vk_unidx(itab[w], 0, k);
                        S += hlog[dv[outv] + phF[inv_v]];
                    }
                    if (S > bS0) { bS0 = S; balo = ei; }
                }
                int bahi = 0; double bS1 = -1e30;
                for (int ei = 0; ei < vgfpow_ne; ei++) {
                    const u8 *itab = vgfpow_itab[ei];
                    double S = 0.0;
                    for (int w = 0; w < 128; w++) {
                        u8 outv = vk_unidx((u8)w, 1, k);
                        u8 inv_v = vk_unidx(itab[w], 1, k);
                        S += hlog[dv[outv] + phF[inv_v]];
                    }
                    if (S > bS1) { bS1 = S; bahi = ei; }
                }
                double net = (bS0 + bS1 - Sb) - oh;
                if (net > best) {
                    best = net; bs = s; bp = p;
                    bamp = (u32)k | ((u32)balo << 3) | ((u32)bahi << 10);
                }
            }
        }
    }
    out->type = T_VALUEGFPOW; out->stride = bs; out->phase = bp; out->amp = bamp;
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

/* NIB_CROSS_GFMUL: same shape as search_nibcxor, GF(16) cross-multiply (with the
 * nibble-is-zero -> substitute-1 fix) instead of XOR. */
static double search_nibcrossgfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_NIBCROSSGFMUL_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            for (int dir = 0; dir < 2; dir++) {
                int rf[256];
                for (int v=0;v<256;v++) rf[v] = total[v] - hit[v];
                for (int v=0;v<256;v++) {
                    u8 lo=(u8)(v&0xF), hi=(u8)((v>>4)&0xF);
                    int w;
                    if (dir==0) { u8 m=hi?hi:1; w = (int)(gf16_mul(lo,m)|(hi<<4)); }
                    else        { u8 m=lo?lo:1; w = (int)(lo|(gf16_mul(hi,m)<<4)); }
                    rf[w] += hit[v];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bamp = (u32)dir; }
            }
        }
    }
    out->type = T_NIBCROSSGFMUL; out->stride = bs; out->phase = bp; out->amp = bamp;
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

/* BIT_SWAP2: fixed bijection, same freq-table trick as BIT_ASWAP. */
static double search_bitswap2(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_BITSWAP2_BASE, s);
        for (int p = 0; p < s; p++) {
            int rf[256];
            for (int v = 0; v < 256; v++) rf[v] = total[v];
            for (int i = p; i < n; i += s) {
                u8 w = bitswap2(d[i]);
                rf[d[i]]--; rf[w]++;
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->type = T_BITSWAP2; out->stride = bs; out->phase = bp; out->amp = 0;
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

/* BIT_CROSS_XOR: same shape as search_crmbcxor but bit granularity — try all 56 ordered
 * (j,k) pairs, j,k in 0..7. */
static double search_bitcxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_BITCXOR_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                    if (j == k) continue;
                    int rf[256];
                    for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
                    for (int v = 0; v < 256; v++) {
                        int w = v ^ (((v >> j) & 1) << k);
                        rf[w] += hit[v];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; bamp = (u32)(j | (k<<3)); }
                }
            }
        }
    }
    out->type = T_BITCXOR; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* CRUMB_CROSS_ADD: same shape as search_crmbcxor, ADD instead of XOR. */
static double search_crmbcadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_CRMBCADD_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    int rf[256];
                    for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
                    for (int v = 0; v < 256; v++) {
                        u8 src = (u8)((v >> (2*j)) & 3);
                        u8 dst = (u8)(((v >> (2*k)) & 3) + src) & 3;
                        int w = (v & ~(3 << (2*k))) | (dst << (2*k));
                        rf[w] += hit[v];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; bamp = (u32)(j | (k<<2)); }
                }
            }
        }
    }
    out->type = T_CRMBCADD; out->stride = bs; out->phase = bp; out->amp = bamp;
    return best;
}

/* CRUMB_CROSS_MUL: same shape as search_crmbcadd, GF(4) mul (with the crumb-j==0 ->
 * substitute-1 fix) instead of ADD. */
static double search_crmbcrossmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bamp = 0;
    for (int s = 1; s <= g_stride_lim; s++) {
        double oh = OH_SP(OH_CRMBCROSSMUL_BASE, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    int rf[256];
                    for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
                    for (int v = 0; v < 256; v++) {
                        u8 cj = (u8)((v >> (2*j)) & 3);
                        u8 mult = cj ? cj : 1;
                        u8 ck = (u8)((v >> (2*k)) & 3);
                        u8 nck = gf4_mul(ck, mult);
                        int w = (v & ~(3 << (2*k))) | (nck << (2*k));
                        rf[w] += hit[v];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; bamp = (u32)(j | (k<<2)); }
                }
            }
        }
    }
    out->type = T_CRMBCROSSMUL; out->stride = bs; out->phase = bp; out->amp = bamp;
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

/* PRNG_PERM: for each seed, build the 256-perm (advancing the stream), then continue
 * the same stream for the coin-flip histograms — must mirror ap_prngperm's draw order
 * exactly so the searched seed decodes to the same transform. */
static double search_prngperm(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 s = (u16)seed;
        u8 perm[256];
        build_prngperm256(&s, perm);
        int flip[256] = {0}, noflip[256] = {0};
        for (int i = 0; i < n; i++) {
            u16 r = xs16_next(&s);
            if (r & 1) flip[d[i]]++;
            else        noflip[d[i]]++;
        }
        int f[256] = {0};
        for (int v = 0; v < 256; v++) {
            f[v]        += noflip[v];
            f[perm[v]]  += flip[v];
        }
        double net = (S_from_freq(f) - Sb) - OH_PRNGPERM;
        if (net > best) { best = net; bseed = seed; }
    }
    out->type = T_PRNGPERM; out->stride = 0; out->phase = 0; out->amp = bseed;
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

/* VALUEMAP4_ADD: same per-quartile independent-optimisation shape as VALUEMAP4, ADD
 * mod 64 instead of XOR. Quartiles stay independent since ADD mod 64 can't carry out
 * into the preserved top 2 bits. */
static double search_valuemap4add(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs=1, bp=0; u32 bamp=0;
    for (int s=1; s<=g_stride_lim; s++) {
        double oh = oh_valuemap4(s);
        for (int p=0; p<s; p++) {
            int hit[256]={0};
            for (int i=p; i<n; i+=s) hit[d[i]]++;
            int base[256];
            for (int v=0;v<256;v++) base[v]=total[v]-hit[v];
            u32 amp=0;
            for (int q=0; q<4; q++) {
                int bc=0; double bS=-1e30;
                int qlo=q*64;
                for (int c=0; c<64; c++) {
                    double S=0.0;
                    for (int j=0; j<64; j++)
                        S += hlog[base[qlo+j] + hit[qlo + ((j - c) & 0x3F)]];
                    if (S>bS) { bS=S; bc=c; }
                }
                amp |= (u32)bc << (q*6);
            }
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int v=0;v<256;v++) if (hit[v]) {
                int q=v>>6, c=(int)((amp>>(q*6))&0x3F);
                int w = (v&0xC0) | (u8)(((v&0x3F)+c)&0x3F);
                rf[w] += hit[v];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net>best) { best=net; bs=s; bp=p; bamp=amp; }
        }
    }
    out->type=T_VALUEMAP4ADD; out->stride=bs; out->phase=bp; out->amp=bamp;
    return best;
}

/* VALUEMAP4_MUL: same per-quartile independent-optimisation shape, MUL mod 64 (odd
 * multiplier) instead of ADD/XOR. For candidate multiplier a, the input that maps to
 * output j is j*a^{-1} mod 64 (mul_inv256(a) masked to 6 bits, same trick VALUE_MUL uses
 * for mod 128 — 64 divides 256, so the mod-256 inverse restricted to 6 bits is correct). */
static double search_valuemap4mul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs=1, bp=0; u32 bamp=0;
    for (int s=1; s<=g_stride_lim; s++) {
        double oh = oh_valuemap4mul(s);
        for (int p=0; p<s; p++) {
            int hit[256]={0};
            for (int i=p; i<n; i+=s) hit[d[i]]++;
            int base[256];
            for (int v=0;v<256;v++) base[v]=total[v]-hit[v];
            u32 amp=0;
            for (int q=0; q<4; q++) {
                int bc=0; double bS=-1e30;
                int qlo=q*64;
                for (int c=0; c<32; c++) {
                    u8 a=(u8)(2*c+1);
                    u8 ainv=(u8)(mul_inv256(a)&0x3F);
                    double S=0.0;
                    for (int j=0; j<64; j++) {
                        int vin = (int)((u8)(j*ainv)&0x3F);
                        S += hlog[base[qlo+j] + hit[qlo+vin]];
                    }
                    if (S>bS) { bS=S; bc=c; }
                }
                amp |= (u32)bc << (q*5);
            }
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int v=0;v<256;v++) if (hit[v]) {
                int q=v>>6; u8 a=(u8)(2*((amp>>(q*5))&0x1F)+1);
                int w = (v&0xC0) | (u8)(((v&0x3F)*a)&0x3F);
                rf[w] += hit[v];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net>best) { best=net; bs=s; bp=p; bamp=amp; }
        }
    }
    out->type=T_VALUEMAP4MUL; out->stride=bs; out->phase=bp; out->amp=bamp;
    return best;
}

/* VALUEMAP4_GFMUL: same per-quartile independent-optimisation shape, GF(64) mul instead
 * of mod-64 int mul — all 63 nonzero elements usable (vs 32 odd-only). */
static double search_valuemap4gfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs=1, bp=0; u32 bamp=0;
    for (int s=1; s<=g_stride_lim; s++) {
        double oh = oh_valuemap4(s);
        for (int p=0; p<s; p++) {
            int hit[256]={0};
            for (int i=p; i<n; i+=s) hit[d[i]]++;
            int base[256];
            for (int v=0;v<256;v++) base[v]=total[v]-hit[v];
            u32 amp=0;
            for (int q=0; q<4; q++) {
                int bc=0; double bS=-1e30;
                int qlo=q*64;
                for (int c=1; c<64; c++) {
                    u8 cinv = gf64_inv((u8)c);
                    double S=0.0;
                    for (int w=0; w<64; w++) {
                        int vin = gf64_mul((u8)w, cinv);
                        S += hlog[base[qlo+w] + hit[qlo+vin]];
                    }
                    if (S>bS) { bS=S; bc=c; }
                }
                amp |= (u32)(bc-1) << (q*6);
            }
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int v=0;v<256;v++) if (hit[v]) {
                int q=v>>6; u8 a=(u8)(((amp>>(q*6))&0x3F)+1);
                int w = (v&0xC0) | gf64_mul((u8)(v&0x3F), a);
                rf[w] += hit[v];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net>best) { best=net; bs=s; bp=p; bamp=amp; }
        }
    }
    out->type=T_VALUEMAP4GFMUL; out->stride=bs; out->phase=bp; out->amp=bamp;
    return best;
}

/* VALUEMAP4_GFPOW: same per-quartile-independent shape as search_valuemap4gfmul (quartiles
 * occupy disjoint value ranges so their entropy contributions are separable), iterating
 * the 36-entry power-exponent table instead of the 63 nonzero multipliers. */
static double search_valuemap4gfpow(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs=1, bp=0; u32 bamp=0;
    for (int s=1; s<=g_stride_lim; s++) {
        double oh = oh_valuemap4(s);
        for (int p=0; p<s; p++) {
            int hit[256]={0};
            for (int i=p; i<n; i+=s) hit[d[i]]++;
            int base[256];
            for (int v=0;v<256;v++) base[v]=total[v]-hit[v];
            u32 amp=0;
            for (int q=0; q<4; q++) {
                int bc=0; double bS=-1e30;
                int qlo=q*64;
                for (int ei=0; ei<vm4pow_ne; ei++) {
                    const u8 *itab = vm4pow_itab[ei];
                    double S=0.0;
                    for (int w=0; w<64; w++) {
                        int vin = itab[w];
                        S += hlog[base[qlo+w] + hit[qlo+vin]];
                    }
                    if (S>bS) { bS=S; bc=ei; }
                }
                amp |= (u32)bc << (q*6);
            }
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int v=0;v<256;v++) if (hit[v]) {
                int q=v>>6; u8 ei=(u8)((amp>>(q*6))&0x3F);
                int w = (v&0xC0) | vm4pow_tab[ei][v&0x3F];
                rf[w] += hit[v];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net>best) { best=net; bs=s; bp=p; bamp=amp; }
        }
    }
    out->type=T_VALUEMAP4GFPOW; out->stride=bs; out->phase=bp; out->amp=bamp;
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
    return -8.0;   /* bijection: no entropy change; no type tag (structurally last, see pack_ilist), just the 8-bit M */
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
    { "BYTE_MUL",    search_bytemul,       0, 0 },
    { "VALUE_XOR",   search_valuexor,      1, 0 },
    { "BIT_REV",     search_bitrev,        0, 0 },
    { "PRNG_ADD4",   search_prngadd4,      1, 1 },
    { "PRNG_ADD8",   search_prngadd8,      1, 1 },
    { "NIB_CXOR",    search_nibcxor,       0, 0 },
    { "CRMB_CXOR",   search_crmbcxor,      0, 0 },
    { "GRAY_CODE",   search_graycode,      0, 0 },
    { "BIT_ASWAP",   search_bitaswap,      0, 0 },
    { "PRNG_PERM",   search_prngperm,      1, 1 },
    { "REFLECT",     search_reflect,       0, 0 },
    { "SPLIT_ADD",   search_splitadd,      1, 0 },
    { "SPLIT_XOR",   search_splitxor,      1, 0 },
    { "PRNG_BIT",    search_prngbit,       1, 1 },
    { "VALUEMAP4",   search_valuemap4,     0, 0 },
    { "NIB_SWAP",    search_nibswap,        0, 0 },
    { "XOR_NP",      search_xorpnp,         0, 0 },
    { "GF_MUL",      search_gfmul,          0, 0 },
    { "NIB_MUL16",   search_nibmul16,       0, 0 },
    { "NIB_CADD",    search_nibcadd,        0, 0 },
    { "CRMB_CADD",   search_crmbcadd,       0, 0 },
    { "BIT_SWAP2",   search_bitswap2,       0, 0 },
    { "VALUE_ADD",   search_valueadd,       0, 0 },
    { "VALUE_MUL",   search_valuemul,       0, 0 },
    { "VALMAP4ADD",  search_valuemap4add,   0, 0 },
    { "NIB_GFMUL",   search_nibgfmul,       0, 0 },
    { "CRMB_MUL",    search_crmbmul,        0, 0 },
    { "VALMAP4MUL",  search_valuemap4mul,   0, 0 },
    { "CRMBXMUL",    search_crmbcrossmul,   0, 0 },
    { "VALUE_GFMUL", search_valuegfmul,     1, 0 },
    { "NIBXGFMUL",   search_nibcrossgfmul,  0, 0 },
    { "VM4GFMUL",    search_valuemap4gfmul, 0, 0 },
    { "BIT_CXOR",    search_bitcxor,        0, 0 },
    { "CRMB_IADD",   search_crmbiadd,       0, 0 },
    { "GF_POW",      search_gfpow,          0, 0 },
    { "NIB_POW",     search_nibpow,         0, 0 },
    { "VALUE_GFPOW", search_valuegfpow,     1, 0 },
    { "VM4GFPOW",    search_valuemap4gfpow, 1, 0 },
};
#define NREG ((int)(sizeof(REGISTRY)/sizeof(REGISTRY[0])))
static const char *TYPE_NAME[NTYPES] = {
    "XOR_PHASE",   /*  0 */
    "ADD_NIBS",    /*  1 */
    "STRIDE_ADD",  /*  2 */
    "BYTE_ROT",    /*  3 */
    "BYTE_MUL",    /*  4 */
    "VALUE_XOR",   /*  5 */
    "BIT_REV",     /*  6 */
    "PRNG_ADD4",   /*  7 */
    "PRNG_ADD8",   /*  8 */
    "NIB_CXOR",    /*  9 */
    "CRMB_CXOR",   /* 10 */
    "GRAY_CODE",   /* 11 */
    "BIT_ASWAP",   /* 12 */
    "PRNG_PERM",   /* 13 */
    "REFLECT",     /* 14 */
    "SPLIT_ADD",   /* 15 */
    "SPLIT_XOR",   /* 16 */
    "PRNG_BIT",    /* 17 */
    "VALUEMAP4",   /* 18 */
    "NIB_SWAP",    /* 19 */
    "XOR_NP",      /* 20 */
    "GF_MUL",      /* 21 */
    "NIB_MUL16",   /* 22 */
    "NIB_CADD",    /* 23 */
    "CRMB_CADD",   /* 24 */
    "BIT_SWAP2",   /* 25 */
    "VALUE_ADD",   /* 26 */
    "VALUE_MUL",   /* 27 */
    "VALMAP4ADD",  /* 28 */
    "NIB_GFMUL",   /* 29 */
    "CRMB_MUL",    /* 30 */
    "VALMAP4MUL",  /* 31 */
    "CRMBXMUL",    /* 32 */
    "VALUE_GFMUL", /* 33 */
    "NIBXGFMUL",   /* 34 */
    "VM4GFMUL",    /* 35 */
    "BIT_CXOR",    /* 36 */
    "CRMB_IADD",   /* 37 */
    "GF_POW",      /* 38 */
    "NIB_POW",     /* 39 */
    "VALUE_GFPOW", /* 40 */
    "VM4GFPOW"     /* 41 */
};

/* True per-type tag bit-length for the canonical Huffman code that pack_ilist actually
 * writes (see TAGCODE/bb_get_tag near the bitstream section). Declared here (before
 * adaptive_tag_cost/best_instr/greedy_run) so the search's cost accounting can use the
 * real cost instead of the old flat TAGB or a hypothetical per-block-adaptive estimate
 * that doesn't match what's actually encoded. 0 for PRNG-tagged types and REFLECT,
 * which never go through this path (PRNG types use their own 2-bit PRTAGB; REFLECT is
 * always structurally last, needing no tag at all). Weighted avg 4.97 bits vs flat 6.0
 * (aggregated fire-frequency data across many real blocks this session), max 8 bits. */
static const u8 TAGLEN[NTYPES] = {
    6, 4, 5, 6, 5, 4, 7, 0, 0, 6, 5, 6, 6, 0, 0, 6, 4, 0, 7, 6,
    6, 4, 5, 6, 6, 6, 5, 4, 6, 5, 6, 6, 8, 4, 7, 6, 6, 4, 5, 5,
    5, 8
};

/* PRNG-tagged types: these are the only ones the greedy loop can pick during the
 * PRNG-first layers (see g_search_mode==1 below), so they always land in a
 * contiguous prefix of ilist. pack_ilist exploits that to store them in a
 * dedicated section addressed with a flat 2-bit index instead of the 6-bit TAGB. */
static const u8 PRNG_TAG_TYPES[4] = { T_PRNGADD4, T_PRNGADD8, T_PRNGPERM, T_PRNGBIT };
static int prng_tag_index(u8 type) {
    for (int i = 0; i < 4; i++) if (PRNG_TAG_TYPES[i] == type) return i;
    return -1;
}
static int is_prng_tagged(u8 type) { return prng_tag_index(type) >= 0; }

/* 0=all types, 1=prng_first only (layers 0-2), 2=non-prng only (layer 3+) */
static __thread int g_search_mode;

/* best selectable instruction for the current data */
static double best_instr(const u8 *d, int n, Instr *out) {
    double Sb = S_of(d, n);
    double best = -1e18; Instr bi = {0};
    for (int r = 0; r < NREG; r++) {
        if (g_search_mode == 1 && !REGISTRY[r].prng_first) continue;
        if (g_search_mode == 2 &&  REGISTRY[r].prng_first) continue;
        Instr cand;
        double net = REGISTRY[r].search(d, n, Sb, &cand);
        if (!is_prng_tagged(cand.type)) {
            /* swap fixed TAGB (baked into oh by every OH_*_BASE macro) for the real
             * per-type Huffman tag cost that pack_ilist actually writes */
            net = net + TAGB - (double)TAGLEN[cand.type];
        }
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

/* One-step lookahead tolerance: greedy stops the instant the best candidate doesn't
 * clear its own overhead — but a marginally-negative move can still reshape the residual
 * into something the NEXT layer exploits for more than the first move cost. Only accept
 * such a move if a verified 2-step lookahead shows the COMBINED net is positive — never
 * blindly accept a negative move on faith. */
#define PATIENCE_BITS 5.0

/* run greedy to convergence; nets[i] receives the net bits saved by ilist[i] */
static double greedy_run(u8 *d, int n, Instr *ilist, double *nets, int *ni, int verbose) {
    double gained = 0.0;
    int n_normal_used = 0;  /* count of accepted non-PRNG instructions so far */
    u8 *scratch = malloc(n);
    for (;;) {
        if (*ni >= MAXINSTR) break;
        Instr t;
        g_search_mode = (*ni < 3) ? 1 : 2;  /* layers 0-2: PRNG only; layer 3+: non-PRNG only */
        double net = best_instr(d, n, &t);
        g_search_mode = 0;
        /* net was computed with the real per-type Huffman tag cost (TAGLEN) for normal
         * types; PRNG-tagged types already bake in their true flat 2-bit tag cost (see
         * OH_PRNGADD4 etc), so their net needs no further threshold adjustment. */
        double accept_threshold = is_prng_tagged(t.type) ? 0.0 : (TAGB - (double)TAGLEN[t.type]);
        int accept = (net > accept_threshold);
        if (!accept && scratch) {
            double margin = net - accept_threshold;
            if (margin > -PATIENCE_BITS) {
                memcpy(scratch, d, n);
                apply_instr(scratch, n, t);
                Instr t2;
                g_search_mode = (*ni + 1 < 3) ? 1 : 2;
                double net2 = best_instr(scratch, n, &t2);
                g_search_mode = 0;
                double accept_threshold2 = is_prng_tagged(t2.type) ? 0.0 : (TAGB - (double)TAGLEN[t2.type]);
                double combined = net + ((net2 > accept_threshold2) ? net2 : 0.0);
                if (combined > 0.0) accept = 1;
            }
        }
        if (!accept) break;
        /* Reserve 1 of the MAX_NORMAL_INSTR slots for the trailing REFLECT try_reflect()
         * always appends, so the compact header's normal-count field never overflows. */
        if (!is_prng_tagged(t.type) && n_normal_used >= MAX_NORMAL_INSTR - 1) break;
        double e0 = entropy_bits(d, n) / n;
        apply_instr(d, n, t);
        if (!is_prng_tagged(t.type)) n_normal_used++;
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
            fflush(stdout);
        }
    }
    free(scratch);
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
 *  Header: 3-bit PRNG-prefix count + 6-bit (normal count - 1),   *
 *          i.e. normal count is 1..MAX_NORMAL_INSTR (64).        *
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

/* Canonical Huffman code for the type tag (TAGLEN defined earlier, near TYPE_NAME,
 * so the search logic can also reference true per-type tag cost). */
static const u16 TAGCODE[NTYPES] = {
    46, 0, 14, 47, 15, 1, 124, 0, 0, 48, 16, 49, 50, 0, 0, 51, 2, 0, 125, 52,
    53, 3, 17, 54, 55, 56, 18, 4, 57, 19, 58, 59, 254, 5, 126, 60, 61, 6, 20, 21,
    22, 255
};
/* Decode a type tag written by TAGCODE/TAGLEN: read bits MSB-first, one at a time,
 * until the accumulated (code,length) matches a known entry. Prefix-free by
 * construction (canonical Huffman), so exactly one match exists at the right length. */
static u8 bb_get_tag(BitBuf *b) {
    u32 code = 0;
    for (int len = 1; len <= 8; len++) {
        code = (code << 1) | bb_get(b, 1);
        for (int t = 0; t < NTYPES; t++)
            if (TAGLEN[t] == len && TAGCODE[t] == code) return (u8)t;
    }
    return 0; /* unreachable: table covers all 37 taggable types up to length 8 */
}

/* Encodes a "normal" (Huffman-tagged) instruction. PRNG_TAG_TYPES never reach here —
 * they're written by bb_put_prng_body into the dedicated PRNG section instead. */
static void bb_put_instr(BitBuf *b, Instr t) {
    bb_put(b, TAGCODE[t.type], TAGLEN[t.type]);
    int s = t.stride, pb = phase_bits(s);
    switch (t.type) {
        case T_REFLECT:
            bb_put(b, t.amp & 0xFF, 8); break;
        case T_BITREV: case T_GRAYCODE: case T_BITASWAP: case T_BITSWAP2:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb); break;
        case T_XORPNP:
            bb_put(b, s - 1, 6); bb_put(b, t.amp & 0xFF, 8); break;   /* no phase */
        case T_XORP: case T_ANIBS: case T_STRIDEADD:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xFF, 8); break;
        case T_BYTEROT:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, (t.amp - 1) & 7, 3); break;           /* k=1..7 → 0..6 */
        case T_BYTEMUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
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
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, (u32)vk, 3); bb_put(b, (u32)alo7, 7); bb_put(b, (u32)ahi7, 7);
            break;
        }
        case T_NIBCXOR:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 1, 1); break;
        case T_CRMBCXOR:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xF, 4); break;
        case T_SPLITADD: case T_SPLITXOR: {
            int kidx = (t.phase >> 8) & 3, act_p = t.phase & 0xFF;
            bb_put(b, s - 1, 6); bb_put(b, kidx, 2); bb_put(b, act_p, pb);
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
        case T_VALUEMAP4:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x00FFFFFFu, 24); break;
        case T_NIBSWAP:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb); break;
        case T_GFMUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xFF, 8); break;
        case T_NIBMUL16:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x3F, 6); break;
        case T_NIBCADD:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 1, 1); break;
        case T_CRMBCADD:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xF, 4); break;
        case T_VALUEADD:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x3FFF, 14); break;   /* alo(7) | ahi(7), no gaps */
        case T_VALUEMUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xFFF, 12); break;    /* aidx(6) | bidx(6) */
        case T_VALUEMAP4ADD:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x00FFFFFFu, 24); break;
        case T_NIBGFMUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xFF, 8); break;      /* aidx(4) | bidx(4) */
        case T_CRMBMUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xFF, 8); break;      /* 4 crumb codes, 2 bits each */
        case T_VALUEMAP4MUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xFFFFFu, 20); break; /* 4 quartile codes, 5 bits each */
        case T_CRMBCROSSMUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xF, 4); break;
        case T_VALUEGFMUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x1FFFFu, 17); break; /* k(3) | aidx(7) | bidx(7) */
        case T_NIBCROSSGFMUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 1, 1); break;
        case T_VALUEMAP4GFMUL:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x00FFFFFFu, 24); break;
        case T_BITCXOR:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x3F, 6); break;      /* j(3) | k(3) */
        case T_CRMBIADD:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0xFF, 8); break;      /* 4 crumb constants, 2 bits each */
        case T_GFPOW:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x7F, 7); break;      /* idx into 128-entry exponent table */
        case T_NIBPOW:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x3F, 6); break;      /* idxlo(3) | idxhi(3) */
        case T_VALUEGFPOW:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x1FFFFu, 17); break; /* k(3) | eaidx(7) | ebidx(7) */
        case T_VALUEMAP4GFPOW:
            bb_put(b, s - 1, 6); bb_put(b, t.phase, pb);
            bb_put(b, t.amp & 0x00FFFFFFu, 24); break; /* 4 quartile exponent-idx codes, 6 bits each */
    }
}

/* Decode instruction body given the already-read type value. */
static Instr bb_get_instr_body(BitBuf *b, u8 type_val) {
    Instr t = {0, 0, 0, 0};
    t.type = type_val;
    int s, pb;
    switch (t.type) {
        case T_REFLECT:
            t.amp = bb_get(b, 8); break;
        case T_BITREV: case T_GRAYCODE: case T_BITASWAP: case T_BITSWAP2:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb); break;
        case T_XORPNP:
            s = (int)bb_get(b, 6) + 1; t.stride = s; t.phase = 0;
            t.amp = bb_get(b, 8); break;                               /* no phase */
        case T_XORP: case T_ANIBS: case T_STRIDEADD:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 8); break;
        case T_BYTEROT:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 3) + 1; break;
        case T_BYTEMUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 7) * 2 + 3; break;
        case T_VALUEXOR: {
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
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
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 1); break;
        case T_CRMBCXOR:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 4); break;
        case T_SPLITADD: case T_SPLITXOR: {
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
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
        case T_VALUEMAP4:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 24); break;
        case T_NIBSWAP:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb); break;
        case T_GFMUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 8); break;
        case T_NIBMUL16:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 6); break;
        case T_NIBCADD:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 1); break;
        case T_CRMBCADD:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 4); break;
        case T_VALUEADD:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 14); break;
        case T_VALUEMUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 12); break;
        case T_VALUEMAP4ADD:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 24); break;
        case T_NIBGFMUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 8); break;
        case T_CRMBMUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 8); break;
        case T_VALUEMAP4MUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 20); break;
        case T_CRMBCROSSMUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 4); break;
        case T_VALUEGFMUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 17); break;
        case T_NIBCROSSGFMUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 1); break;
        case T_VALUEMAP4GFMUL:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 24); break;
        case T_BITCXOR:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 6); break;
        case T_CRMBIADD:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 8); break;
        case T_GFPOW:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 7); break;
        case T_NIBPOW:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 6); break;
        case T_VALUEGFPOW:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 17); break;
        case T_VALUEMAP4GFPOW:
            s = (int)bb_get(b, 6) + 1; pb = phase_bits(s);
            t.stride = s; t.phase = (int)bb_get(b, pb);
            t.amp = bb_get(b, 24); break;
    }
    return t;
}
static Instr bb_get_instr(BitBuf *b) {
    return bb_get_instr_body(b, bb_get_tag(b));
}

/* PRNG-tagged instructions are all global (stride=phase=0), so only amp needs encoding.
 * Tag itself is a flat 2-bit index into PRNG_TAG_TYPES, not the 6-bit TAGB. */
static void bb_put_prng_body(BitBuf *b, Instr t) {
    bb_put(b, (u32)prng_tag_index(t.type), 2);
    switch (t.type) {
        case T_PRNGADD4: case T_PRNGADD8:
            bb_put(b, t.amp & 0xFFFF, 16); break;
        case T_PRNGPERM:
            bb_put(b, t.amp & 0xFFFF, 16); break;
        case T_PRNGBIT:
            bb_put(b, t.amp & 0xFFFF, 16);
            bb_put(b, (t.amp >> 16) & 0xFF, 8); break;
    }
}
static Instr bb_get_prng_body(BitBuf *b) {
    Instr t = {0, 0, 0, 0};
    t.type = PRNG_TAG_TYPES[bb_get(b, 2)];
    switch (t.type) {
        case T_PRNGADD4: case T_PRNGADD8:
            t.amp = bb_get(b, 16); break;
        case T_PRNGPERM:
            t.amp = bb_get(b, 16); break;
        case T_PRNGBIT: {
            u32 seed = bb_get(b, 16), fm = bb_get(b, 8);
            t.amp = seed | (fm << 16); break;
        }
    }
    return t;
}

/* Max bits per instruction: VALUEXOR stride=64 → 6+6+6+17 = 35 bits (largest non-PHASEOFFSET) */
#define COMPACT_BUF_BYTES ((MAXINSTR * 64 + 16 + 7) / 8)

/* Pack instruction list into compact bit-stream.
 * Layout: [3-bit PRNG-prefix count n_prng] [6-bit (n_normal - 1)]
 *          [n_prng PRNG-tagged instructions, 2-bit tag each]
 *          [n_normal normal instructions, 6-bit tag each]
 * The PRNG-first greedy layers (g_search_mode==1, capped at 3) guarantee any
 * PRNG_TAG_TYPES instructions form a contiguous prefix of ilist, and greedy_run
 * caps n_normal at MAX_NORMAL_INSTR (64) so it always fits (n_normal - 1) in 6 bits.
 * Total instruction count is derived as n_prng + n_normal, never stored directly. */
/* try_reflect() (see compress()) unconditionally appends exactly one T_REFLECT as the
 * very last ilist entry in every block -- its position alone determines its type, so its
 * 6-bit TAGB type tag is pure redundancy. Skip writing/reading it: a guaranteed, 100%-of-
 * blocks 6-bit/block saving with zero probabilistic risk (unlike a frequency-based code
 * for the other types, this needs no assumption about the data at all). */
static int pack_ilist(const Instr *ilist, int ni, u8 *buf) {
    memset(buf, 0, COMPACT_BUF_BYTES);
    int n_prng = 0;
    while (n_prng < ni && is_prng_tagged(ilist[n_prng].type)) n_prng++;
    int n_normal = ni - n_prng;
    BitBuf b = { buf, 0 };
    bb_put(&b, (u32)n_prng, 2);   /* n_prng in {0,1,2,3}: greedy_run's g_search_mode gates PRNG-tagged types to *ni<3, so it can never exceed 3 */
    bb_put(&b, (u32)(n_normal - 1), 6);
    for (int i = 0; i < n_prng; i++) bb_put_prng_body(&b, ilist[i]);
    for (int i = n_prng; i < ni - 1; i++) bb_put_instr(&b, ilist[i]);
    bb_put(&b, ilist[ni - 1].amp & 0xFF, 8);  /* trailing REFLECT: amp only, no type tag */
    return b.pos;
}

/* Unpack. Returns instruction count. */
static int unpack_ilist(const u8 *buf, Instr *ilist_out) {
    BitBuf b = { (u8 *)buf, 0 };
    int n_prng   = (int)bb_get(&b, 2);
    int n_normal = (int)bb_get(&b, 6) + 1;
    int ni = n_prng + n_normal;
    int i = 0;
    for (; i < n_prng; i++) ilist_out[i] = bb_get_prng_body(&b);
    for (; i < ni - 1; i++) ilist_out[i] = bb_get_instr(&b);
    ilist_out[ni - 1] = (Instr){ T_REFLECT, 0, 0, bb_get(&b, 8) };
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
        { T_PRNGPERM,  0, 0, 1234u },
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
        { T_VALUEMAP4,    3, 1, (0x1Au|(0x2Bu<<6)|(0x0Cu<<12)|(0x3Fu<<18)) },
        { T_NIBSWAP,      3, 1, 0 },
        { T_GFMUL,        3, 1, 0x57u },
        { T_NIBMUL16,     3, 1, (u32)(2u | (3u << 3)) },
        { T_NIBCADD,      3, 1, 1u },
        { T_CRMBCADD,     3, 1, (u32)(0u | (2u << 2)) },
        { T_BITSWAP2,     3, 1, 0 },
        { T_VALUEADD,     3, 1, (u32)(37u | (89u << 7)) },
        { T_VALUEMUL,     3, 1, (u32)(5u | (17u << 6)) },
        { T_VALUEMAP4ADD, 3, 1, (u32)(0x1Au|(0x2Bu<<6)|(0x0Cu<<12)|(0x3Fu<<18)) },
        { T_NIBGFMUL,     3, 1, (u32)(4u | (9u << 4)) },
        { T_CRMBMUL,      3, 1, (u32)(1u | (2u << 2) | (0u << 4) | (1u << 6)) },
        { T_VALUEMAP4MUL, 3, 1, (u32)(3u | (10u << 5) | (20u << 10) | (31u << 15)) },
        { T_CRMBCROSSMUL, 3, 1, (u32)(0u | (2u << 2)) },
        { T_VALUEGFMUL,   3, 1, (u32)(3u | (10u << 3) | (50u << 10)) },
        { T_NIBCROSSGFMUL, 3, 1, 1u },
        { T_VALUEMAP4GFMUL, 3, 1, (u32)(5u | (15u << 6) | (40u << 12) | (62u << 18)) },
        { T_BITCXOR,      3, 1, (u32)(2u | (5u << 3)) },
        { T_CRMBIADD,     3, 1, (u32)(1u | (2u << 2) | (3u << 4) | (0u << 6)) },
        { T_GFPOW,        3, 1, 17u },
        { T_NIBPOW,       3, 1, (u32)(2u | (5u << 3)) },
        { T_VALUEGFPOW,   3, 1, (u32)(4u | (10u << 3) | (60u << 10)) },
        { T_VALUEMAP4GFPOW, 3, 1, (u32)(5u | (15u << 6) | (25u << 12) | (35u << 18)) },
        { T_XORPNP,       3, 0, 0x99u },
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

    /* Dedicated bitstream round-trip test: every taggable (non-PRNG, non-REFLECT) type
     * gets its own entry here, verified through the REAL pack_ilist/unpack_ilist path
     * (Huffman type tag + stride/phase/amp fields), not just apply/invert semantics.
     * This specifically targets rare types (e.g. CRMBXMUL, VM4GFPOW) whose Huffman
     * codes are 8 bits long and may fire only a handful of times in real block runs --
     * relying on real random data to exercise every code path leaves a gap; this closes
     * it deterministically. n_prng=3 and REFLECT-last respect the same invariants
     * pack_ilist assumes (PRNG-tagged prefix capped at 3, REFLECT always last). */
    Instr bv[] = {
        /* PRNG-tagged prefix (n_prng=3, the real maximum) */
        { T_PRNGADD4, 0, 0, 1234u },
        { T_PRNGADD8, 0, 0, 5678u },
        { T_PRNGPERM, 0, 0, 1234u },
        /* every taggable type, one representative each */
        { T_XORP,      3, 1, 0x5A },
        { T_ANIBS,     3, 1, 0x35 },
        { T_STRIDEADD, 3, 1, 0x37 },
        { T_BYTEROT,   3, 1, 3 },
        { T_BYTEMUL,   3, 1, 3u },
        { T_VALUEXOR,  3, 1, (3u|(0x24u<<3)|(0x12u<<11)) },
        { T_BITREV,    3, 1, 0 },
        { T_NIBCXOR,   3, 1, 0 },
        { T_CRMBCXOR,  3, 1, (0|(2<<2)) },
        { T_GRAYCODE,  3, 1, 0 },
        { T_BITASWAP,  3, 1, 0 },
        { T_SPLITADD,  3, 1|(0<<8), 0x1234u },
        { T_SPLITXOR,  3, 1|(3<<8), (0x2Au | (0x55u << 8)) },
        { T_VALUEMAP4,    3, 1, (0x1Au|(0x2Bu<<6)|(0x0Cu<<12)|(0x3Fu<<18)) },
        { T_NIBSWAP,      3, 1, 0 },
        { T_XORPNP,       3, 0, 0x99u },
        { T_GFMUL,        3, 1, 0x57u },
        { T_NIBMUL16,     3, 1, (u32)(2u | (3u << 3)) },
        { T_NIBCADD,      3, 1, 1u },
        { T_CRMBCADD,     3, 1, (u32)(0u | (2u << 2)) },
        { T_BITSWAP2,     3, 1, 0 },
        { T_VALUEADD,     3, 1, (u32)(37u | (89u << 7)) },
        { T_VALUEMUL,     3, 1, (u32)(5u | (17u << 6)) },
        { T_VALUEMAP4ADD, 3, 1, (u32)(0x1Au|(0x2Bu<<6)|(0x0Cu<<12)|(0x3Fu<<18)) },
        { T_NIBGFMUL,     3, 1, (u32)(4u | (9u << 4)) },
        { T_CRMBMUL,      3, 1, (u32)(1u | (2u << 2) | (0u << 4) | (1u << 6)) },
        { T_VALUEMAP4MUL, 3, 1, (u32)(3u | (10u << 5) | (20u << 10) | (31u << 15)) },
        { T_CRMBCROSSMUL, 3, 1, (u32)(0u | (2u << 2)) },
        { T_VALUEGFMUL,   3, 1, (u32)(3u | (10u << 3) | (50u << 10)) },
        { T_NIBCROSSGFMUL, 3, 1, 1u },
        { T_VALUEMAP4GFMUL, 3, 1, (u32)(5u | (15u << 6) | (40u << 12) | (62u << 18)) },
        { T_BITCXOR,      3, 1, (u32)(2u | (5u << 3)) },
        { T_CRMBIADD,     3, 1, (u32)(1u | (2u << 2) | (3u << 4) | (0u << 6)) },
        { T_GFPOW,        3, 1, 17u },
        { T_NIBPOW,       3, 1, (u32)(2u | (5u << 3)) },
        { T_VALUEGFPOW,   3, 1, (u32)(4u | (10u << 3) | (60u << 10)) },
        { T_VALUEMAP4GFPOW, 3, 1, (u32)(5u | (15u << 6) | (25u << 12) | (35u << 18)) },
        /* REFLECT: always last, no type tag */
        { T_REFLECT, 0, 0, 0x40u },
    };
    int nb = (int)(sizeof(bv) / sizeof(bv[0]));
    u8 *packbuf = malloc(COMPACT_BUF_BYTES);
    Instr *unpacked_bv = malloc(MAXINSTR * sizeof(Instr));
    int bfails = 0;
    if (packbuf && unpacked_bv) {
        pack_ilist(bv, nb, packbuf);
        int nb2 = unpack_ilist(packbuf, unpacked_bv);
        if (nb2 != nb) {
            printf("  BITSTREAM SELFTEST FAIL: count mismatch %d != %d\n", nb2, nb);
            bfails++;
        }
        for (int i = 0; i < nb && i < nb2; i++) {
            if (unpacked_bv[i].type   != bv[i].type   ||
                unpacked_bv[i].stride != bv[i].stride ||
                unpacked_bv[i].phase  != bv[i].phase  ||
                unpacked_bv[i].amp    != bv[i].amp) {
                printf("  BITSTREAM SELFTEST FAIL: entry %d (%s): got type=%d s=%d p=%d a=%u, want type=%d s=%d p=%d a=%u\n",
                       i, TYPE_NAME[bv[i].type],
                       unpacked_bv[i].type, unpacked_bv[i].stride, unpacked_bv[i].phase, unpacked_bv[i].amp,
                       bv[i].type, bv[i].stride, bv[i].phase, bv[i].amp);
                bfails++;
            }
        }
    } else {
        printf("  BITSTREAM SELFTEST FAIL: allocation failed\n");
        bfails++;
    }
    printf("selftest: bitstream round-trip of all 37 taggable types + 3 PRNG + REFLECT (%d instrs): %s\n",
           nb, bfails == 0 ? "OK" : "FAIL");
    free(packbuf); free(unpacked_bv);

    return fails + bfails;
}

/* last-block instruction list, accessible from main for serialisation */
/* overhead in bits for a single applied instruction (mirrors search OH formulas) */
static double instr_oh(Instr t) {
    switch (t.type) {
        case T_XORP:     return OH_SP(OH_XORP_BASE,      t.stride);
        case T_XORPNP:   return OH_XORP_BASE; /* no phase field */
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
        case T_PRNGPERM: return OH_PRNGPERM;
        case T_REFLECT:  return 0.0;
        case T_SPLITADD:    return oh_splitadd((t.phase >> 8) & 3, t.stride);
        case T_SPLITXOR:    return oh_splitxor((t.phase >> 8) & 3, t.stride);
        case T_PRNGBIT:      return OH_PRNGBIT;
        case T_VALUEMAP4:    return oh_valuemap4(t.stride);
        case T_NIBSWAP:        return OH_SP(OH_NIBSWAP_BASE, t.stride);
        case T_GFMUL:          return OH_SP(OH_GFMUL_BASE, t.stride);
        case T_NIBMUL16:       return OH_SP(OH_NIBMUL16_BASE, t.stride);
        case T_NIBCADD:        return OH_SP(OH_NIBCADD_BASE, t.stride);
        case T_CRMBCADD:       return OH_SP(OH_CRMBCADD_BASE, t.stride);
        case T_BITSWAP2:       return OH_SP(OH_BITSWAP2_BASE, t.stride);
        case T_VALUEADD:       return OH_SP(OH_VALUEADD_BASE, t.stride);
        case T_VALUEMUL:       return OH_SP(OH_VALUEMUL_BASE, t.stride);
        case T_VALUEMAP4ADD:   return oh_valuemap4(t.stride);
        case T_NIBGFMUL:       return OH_SP(OH_NIBGFMUL_BASE, t.stride);
        case T_CRMBMUL:        return OH_SP(OH_CRMBMUL_BASE, t.stride);
        case T_VALUEMAP4MUL:   return oh_valuemap4mul(t.stride);
        case T_CRMBCROSSMUL:   return OH_SP(OH_CRMBCROSSMUL_BASE, t.stride);
        case T_VALUEGFMUL:     return OH_SP(OH_VALUEGFMUL_BASE, t.stride);
        case T_NIBCROSSGFMUL:  return OH_SP(OH_NIBCROSSGFMUL_BASE, t.stride);
        case T_VALUEMAP4GFMUL: return oh_valuemap4(t.stride);
        case T_BITCXOR:      return OH_SP(OH_BITCXOR_BASE, t.stride);
        case T_CRMBIADD:     return OH_SP(OH_CRMBIADD_BASE, t.stride);
        case T_GFPOW:        return OH_SP(OH_GFPOW_BASE, t.stride);
        case T_NIBPOW:       return OH_SP(OH_NIBPOW_BASE, t.stride);
        case T_VALUEGFPOW:   return OH_SP(OH_VALUEGFPOW_BASE, t.stride);
        case T_VALUEMAP4GFPOW: return oh_valuemap4(t.stride);
        default:             return 0.0;
    }
}

/* Real per-block instruction counts never approach MAXINSTR (capped at 3 PRNG +
 * MAX_NORMAL_INSTR normal = 67 max); this bounds the compact-bitstream dump buffer. */
#define CBUF_MAX_BYTES (((3 + MAX_NORMAL_INSTR) * 40 + 7) / 8)

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
    int    cbuf_n;
    u8     cbuf[CBUF_MAX_BYTES];
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
    /* -1e18 floor, not 0 — the patience mechanism can now genuinely accept a
     * negative-net instruction as part of a lookahead-verified positive combo, and a
     * 0 floor would silently hide that (max stuck at 0 instead of the true, negative,
     * value) whenever every fire of a type happened to be negative. */
    for (int t = 0; t < NTYPES; t++) r->type_net_max[t] = -1e18;
    memset(r->type_oh_sum,  0, sizeof r->type_oh_sum);
    for (int i = 0; i < ni; i++) {
        int t = ilist[i].type;
        r->type_counts[t]++;
        r->type_net_sum[t] += nets[i];
        if (nets[i] > r->type_net_max[t]) r->type_net_max[t] = nets[i];
        r->type_oh_sum[t]  += instr_oh(ilist[i]);
    }

    r->cbits = compact_buf ? pack_ilist(ilist, ni, compact_buf) : 0;
    r->cbuf_n = 0;
    if (compact_buf) {
        int cbytes = (r->cbits + 7) / 8;
        if (cbytes > CBUF_MAX_BYTES) cbytes = CBUF_MAX_BYTES;
        memcpy(r->cbuf, compact_buf, cbytes);
        r->cbuf_n = cbytes;
    }
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
    init_gf_mul_tab();
    init_gf_log();
    init_gfpow();
    init_gf16_mul_tab();
    init_gf16_log();
    init_nibpow();
    init_gf128_mul_tab();
    init_gf128_log();
    init_vgfpow();
    init_gf64_mul_tab();
    init_gf64_log();
    init_vm4pow();

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
    int ncpus = (int)si.dwNumberOfProcessors - 1;  /* leave one core free for background use */
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
    double type_net_max[NTYPES]; for (int t = 0; t < NTYPES; t++) type_net_max[t] = -1e18;
    double type_oh_sum[NTYPES]  = {0};
    double total_net = 0.0, total_ein = 0.0, total_eout = 0.0;
    int total_ni = 0, fails = 0, compact_fails = 0;
    long total_compact_bits = 0;
    u8 *ibuf  = malloc((size_t)NB * MAXINSTR * 8);
    int ibuf_n = 0;
    u8 *cbuf_all = malloc((size_t)NB * CBUF_MAX_BYTES);
    int cbuf_all_n = 0;

    for (int b = 0; b < NB; b++) {
        BlockResult *r = &results[b];
        double raw = r->e_in - r->e_out;
        /* Real net: raw entropy saved minus the actual packed compact-bitstream cost
         * (r->cbits), not the modeled per-instruction overhead the greedy search used
         * to decide accept/reject (that estimate still drives the search itself, since
         * the real bitstream size is only known once the whole instruction list for a
         * block is fixed). */
        double net = raw - (double)r->cbits;

        total_net      += net;       total_ein  += r->e_in;
        total_eout     += r->e_out;
        total_ni       += r->ni;    total_compact_bits += r->cbits;
        if (!r->ok)  fails++;
        if (!r->cok) compact_fails++;
        for (int t = 0; t < NTYPES; t++) {
            counts[t]       += r->type_counts[t];
            type_net_sum[t] += r->type_net_sum[t];
            if (r->type_net_max[t] > type_net_max[t]) type_net_max[t] = r->type_net_max[t];
            type_oh_sum[t]  += r->type_oh_sum[t];
        }

        printf("  block %2d: %.4f -> %.4f bps  net=%+.1f  %s  [%d instrs  raw=%+.1f  compact=%d bits %s]\n",
               b, r->e_in / BLOCK, r->e_out / BLOCK, net,
               r->ok ? "ok" : "FAIL", r->ni, raw, r->cbits,
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
        if (cbuf_all) { memcpy(cbuf_all + cbuf_all_n, r->cbuf, r->cbuf_n); cbuf_all_n += r->cbuf_n; }
        fwrite(all + (size_t)b * BLOCK, 1, BLOCK, fcomp);
    }
    fclose(fcomp);

    int fired = 0, fired_nonprng = 0, n_nonprng = 0;
    for (int t = 0; t < NTYPES; t++) {
        if (counts[t]) fired++;
        if (!is_prng_tagged((u8)t)) {
            n_nonprng++;
            if (counts[t]) fired_nonprng++;
        }
    }

    printf("\n=== aggregate over %d blocks (%.0f ms, %d threads) ===\n", NB, ms, ncpus);
    printf("avg input:  %.4f bps     avg output: %.4f bps\n",
           total_ein / (NB * BLOCK), total_eout / (NB * BLOCK));
    printf("total net:  %.1f bits   (avg %.1f / block)\n", total_net, total_net / NB);
    printf("total instrs: %d (avg %.1f/block)   total compact: %.1f bits (avg %.1f/block)\n",
           total_ni, (double)total_ni / NB, (double)total_compact_bits, (double)total_compact_bits / NB);
    if (ibuf && ibuf_n > 0) {
        double ibps = entropy_bits(ibuf, ibuf_n) / ibuf_n;
        printf("instr stream (flat 8B/instr): %.4f bps  (%d bytes)  entropy=%.0f bits/block\n",
               ibps, ibuf_n, entropy_bits(ibuf, ibuf_n) / NB);
    }
    printf("instr stream (compact):       %ld bits/block total  (%.1f bits/block avg)  %s\n",
           total_compact_bits / NB,
           (double)total_compact_bits / NB,
           compact_fails ? "*** COMPACT FAIL ***" : "round-trip ok");

    /* Dump both instruction-stream representations to disk for external inspection. */
    if (ibuf && ibuf_n > 0) {
        FILE *fi = fopen("instr_flat.bin", "wb");
        if (fi) { fwrite(ibuf, 1, ibuf_n, fi); fclose(fi); }
    }
    if (cbuf_all && cbuf_all_n > 0) {
        FILE *fi = fopen("instr_compact.bin", "wb");
        if (fi) { fwrite(cbuf_all, 1, cbuf_all_n, fi); fclose(fi); }
    }
    printf("wrote instr_flat.bin (%d bytes), instr_compact.bin (%d bytes)\n", ibuf_n, cbuf_all_n);

    free(ibuf); free(cbuf_all); free(results); free(args); free(threads);
    printf("round-trip: %s (%d/%d blocks)\n", fails ? "*** FAIL ***" : "OK", NB - fails, NB);
    printf("types fired (across all blocks): %d / %d non-PRNG  (%d / %d incl. PRNG)\n\n",
           fired_nonprng, n_nonprng, fired, NTYPES);
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
