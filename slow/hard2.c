/*
 * simple.c — clean greedy entropy compressor
 *
 * Goal: reduce order-0 entropy of a block by finding reversible byte transforms
 *       whose entropy reduction exceeds their storage cost.
 *
 * To add a new instruction:
 *   1. Add a value to InstrType enum
 *   2. Add a case in applyInstr()
 *   3. Add a search block in findBest()
 *   That's it.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include <bcrypt.h>

typedef uint8_t u8;

#define BLOCK_SIZE 4096
#define NUM_BLOCKS 50

/* ── log table: hlog[x] = x*log2(x), hlog[0]=0 ─────────────────────────── */
static double hlog[BLOCK_SIZE + 1];
static void init_hlog(void) {
    hlog[0] = 0.0;
    for (int i = 1; i <= BLOCK_SIZE; i++) hlog[i] = i * log2(i);
}

/* ── entropy ─────────────────────────────────────────────────────────────── */
static double entropy(const u8 *data, int n) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[data[i]]++;
    double s = 0.0;
    for (int i = 0; i < 256; i++) s += hlog[f[i]];
    return log2(n) - s / n;
}

/* ── data generation ─────────────────────────────────────────────────────── */
static void fill_random(u8 *buf, int n) {
    BCryptGenRandom(NULL, buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}

/* ── byte helpers ────────────────────────────────────────────────────────── */
static u8 gray_encode(u8 x) { return x ^ (x >> 1); }
static u8 gray_decode(u8 x) { x ^= x>>4; x ^= x>>2; x ^= x>>1; return x; }
static u8 rotl8(u8 v, int n) { n &= 7; return (u8)((v << n) | (v >> (8-n))); }
static u8 rotr8(u8 v, int n) { n &= 7; return (u8)((v >> n) | (v << (8-n))); }
static u8 bit_reverse8(u8 x) {
    x = (u8)(((x & 0xF0) >> 4) | ((x & 0x0F) << 4));
    x = (u8)(((x & 0xCC) >> 2) | ((x & 0x33) << 2));
    x = (u8)(((x & 0xAA) >> 1) | ((x & 0x55) << 1));
    return x;
}

/* ── multiply-inverse table ──────────────────────────────────────────────── */
static u8 mul_inv[256];
static void init_mul_inv(void) {
    for (int k = 1; k < 256; k += 2)
        for (int inv = 1; inv < 256; inv += 2)
            if (((k * inv) & 0xFF) == 1) { mul_inv[k] = (u8)inv; break; }
}

/* ── scramble permutations (used in compress restart loop) ──────────────── */
static void interleave_stride(const u8 *src, u8 *dst, int n, int s) {
    int w = n / s;
    for (int i = 0; i < n; i++) dst[(i % s) * w + (i / s)] = src[i];
}
static void bit_plane_sep(const u8 *src, u8 *dst, int n) {
    int ps = n / 8;
    memset(dst, 0, n);
    for (int i = 0; i < n; i++)
        for (int b = 0; b < 8; b++)
            if ((src[i] >> b) & 1)
                dst[b * ps + i/8] |= (u8)(1 << (i % 8));
}
static void nib_plane_sep(const u8 *src, u8 *dst, int n) {
    int half = n / 2;
    for (int i = 0; i < half; i++) {
        int j = i * 2;
        dst[i]        = (u8)((src[j] & 0x0F) | ((src[j+1] & 0x0F) << 4));
        dst[i + half] = (u8)((src[j] >> 4)   | ((src[j+1] >> 4)   << 4));
    }
}
static void block_reverse(const u8 *src, u8 *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[n - 1 - i];
}
static void xor_fold_scramble(const u8 *src, u8 *dst, int n) {
    int h = n / 2;
    for (int i = 0; i < h; i++) dst[i] = src[i] ^ src[i + h];
    memcpy(dst + h, src + h, h);
}

/* ── instructions ────────────────────────────────────────────────────────── */

typedef enum {
    /*
     * ── phase+stride+amp: 16-bit overhead ─────────────────────────────────
     *
     * These use the frequency-table trick: for a given (stride, phase),
     * build phF[256] once, then evaluate all amp values in O(256) each.
     * Any bijection on Z/256Z parameterized by amp can plug in here.
     */
    XOR_PHASE = 0,  /* data[i] ^= amp                         bijection: v → v^amp    */
    ADD_PHASE,      /* data[i] += amp                         bijection: v → (v+amp)%256 */
    ADD_NIBS,       /* nibble-wise add: lo += amp&0xF, hi += amp>>4  (no carry crossing) */
    XOR_LO_NIB,     /* data[i].lo ^= amp (amp 1..15)                                   */
    XOR_HI_NIB,     /* data[i].hi ^= amp (amp 1..15)                                   */
    ADD_LO_NIB,     /* data[i].lo += amp (amp 1..15)                                   */
    ADD_HI_NIB,     /* data[i].hi += amp (amp 1..15)                                   */
    MUL_ODD,        /* data[i] *= amp (amp odd)               bijection: v → (v*amp)%256 */
    ROTL,           /* rotate-left by amp bits                bijection: v → rotl(v,amp) */
    ROTR,           /* rotate-right by amp bits                                         */
    MODULAR_CHAIN,  /* data[i] += (data[i-stride]*amp)&0xFF  right-to-left             */
    COND_LO_XOR,    /* if (data[i]>>4)==(amp>>4): data[i]^=(amp&0xF)                   */

    /*
     * ── AFFINE_PHASE: amp encodes (m | c<<8) where m is odd, 3..31 ────────
     *   data[i] = (data[i]*m + c) & 0xFF   — the full affine group on Z/256Z
     *   subsumes ADD_PHASE (m=1) and MUL_ODD (c=0) in one instruction
     *   freq-table trick: new_freq[v] = total[v] - phF[v] + phF[(v-c)*modinv[m]&0xFF]
     *   cost: 24 bits (needs m + c + stride + phase)
     */
    AFFINE_PHASE,

    /* ── phase+stride, no amp: 8-bit overhead ───────────────────────────── */
    NIB_SWAP,
    GRAY_ENC,
    GRAY_DEC,
    BIT_REV,

    /* ── stride only: 8-bit overhead ────────────────────────────────────── */
    XOR3,           /* data[i] ^= data[i-s]^data[i-2s]  right-to-left, self-inverse */
    XOR_FOLD,       /* data[i] ^= data[i+n/2]  for i<n/2, self-inverse */
    HI_NIB_FEED,    /* data[i] += data[i-stride]>>4   right-to-left */
    LO_NIB_FEED,    /* data[i] += data[i-stride]&0x0F right-to-left */
    HI2_NIB_FEED,   /* data[i] += data[i-stride]&0xF0 right-to-left */
    CARRY_TOP,      /* data[i] += data[i-stride]>>7   right-to-left */
    TOFFOLI,        /* data[i+2] ^= data[i]&data[i+1]  self-inverse */

    /* ── add new instruction types above this line ── */
    NUM_INSTR_TYPES
} InstrType;

static const char *INSTR_NAMES[NUM_INSTR_TYPES] = {
    "XOR_PHASE","ADD_PHASE","ADD_NIBS","XOR_LO_NIB","XOR_HI_NIB","ADD_LO_NIB","ADD_HI_NIB",
    "MUL_ODD","ROTL","ROTR","MOD_CHAIN","COND_LO_XOR","AFFINE_PHASE",
    "NIB_SWAP","GRAY_ENC","GRAY_DEC","BIT_REV",
    "XOR3","XOR_FOLD",
    "HI_NIB_FEED","LO_NIB_FEED","HI2_NIB_FEED","CARRY_TOP","TOFFOLI"
};

typedef struct { InstrType type; int stride, phase, amp; } Instr;

static double instr_cost(Instr t) {
    switch (t.type) {
        case AFFINE_PHASE: return 24.0;
        case XOR3: case XOR_FOLD:
        case HI_NIB_FEED: case LO_NIB_FEED: case HI2_NIB_FEED: case CARRY_TOP:
        case TOFFOLI:
        case NIB_SWAP: case GRAY_ENC: case GRAY_DEC: case BIT_REV:
            return 8.0;
        default: return 16.0;
    }
}

static void applyInstr(u8 *data, int n, Instr t) {
    int i;
    switch (t.type) {
        case XOR_PHASE:
            for (i = t.phase; i < n; i += t.stride) data[i] ^= (u8)t.amp;
            break;
        case ADD_PHASE:
            for (i = t.phase; i < n; i += t.stride) data[i] += (u8)t.amp;
            break;
        case ADD_NIBS: {
            u8 lo = (u8)(t.amp & 0x0F), hi = (u8)((t.amp >> 4) & 0x0F);
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)(((data[i] + lo) & 0x0F) | (((data[i] >> 4) + hi) & 0x0F) << 4);
            break;
        }
        case XOR_LO_NIB:
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (data[i] & 0xF0) | ((data[i] ^ t.amp) & 0x0F);
            break;
        case XOR_HI_NIB:
            for (i = t.phase; i < n; i += t.stride)
                data[i] ^= (u8)((t.amp & 0x0F) << 4);
            break;
        case ADD_LO_NIB:
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (data[i] & 0xF0) | ((data[i] + t.amp) & 0x0F);
            break;
        case ADD_HI_NIB:
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (data[i] & 0x0F) | ((((data[i] >> 4) + t.amp) & 0x0F) << 4);
            break;
        case MUL_ODD:
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)((data[i] * (u8)t.amp) & 0xFF);
            break;
        case ROTL:
            for (i = t.phase; i < n; i += t.stride) data[i] = rotl8(data[i], t.amp);
            break;
        case ROTR:
            for (i = t.phase; i < n; i += t.stride) data[i] = rotr8(data[i], t.amp);
            break;
        case MODULAR_CHAIN:
            for (i = n-1; i >= t.stride; i--)
                data[i] += (u8)((data[i - t.stride] * (u8)t.amp) & 0xFF);
            break;
        case COND_LO_XOR: {
            int nib_cond = (t.amp >> 4) & 0xF;
            int xv       = t.amp & 0x0F;
            for (i = t.phase; i < n; i += t.stride)
                if ((data[i] >> 4) == nib_cond)
                    data[i] ^= (u8)xv;
            break;
        }
        case AFFINE_PHASE: {
            int m = t.amp & 0xFF;           /* odd multiplier, 3..31 */
            int c = (t.amp >> 8) & 0xFF;    /* additive constant, 0..255 */
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)((data[i] * m + c) & 0xFF);
            break;
        }
        case NIB_SWAP:
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)((data[i] << 4) | (data[i] >> 4));
            break;
        case GRAY_ENC:
            for (i = t.phase; i < n; i += t.stride) data[i] = gray_encode(data[i]);
            break;
        case GRAY_DEC:
            for (i = t.phase; i < n; i += t.stride) data[i] = gray_decode(data[i]);
            break;
        case BIT_REV:
            for (i = t.phase; i < n; i += t.stride) data[i] = bit_reverse8(data[i]);
            break;
        case XOR3:
            for (i = n-1; i >= 2*t.stride; i--)
                data[i] ^= data[i - t.stride] ^ data[i - 2*t.stride];
            break;
        case XOR_FOLD:
            for (i = 0; i < n/2; i++) data[i] ^= data[i + n/2];
            break;
        case HI_NIB_FEED:
            for (i = n-1; i >= t.stride; i--) data[i] += data[i - t.stride] >> 4;
            break;
        case LO_NIB_FEED:
            for (i = n-1; i >= t.stride; i--) data[i] += data[i - t.stride] & 0x0F;
            break;
        case HI2_NIB_FEED:
            for (i = n-1; i >= t.stride; i--) data[i] += data[i - t.stride] & 0xF0;
            break;
        case CARRY_TOP:
            for (i = n-1; i >= t.stride; i--) data[i] += data[i - t.stride] >> 7;
            break;
        case TOFFOLI:
            for (i = 0; i + 2 < n; i += t.stride)
                data[i+2] ^= data[i] & data[i+1];
            break;
        /* ── add new instruction types above this line ── */
        default: break;
    }
}

/* ── search: find the single best instruction for this block ─────────────── */
static Instr findBest(const u8 *data, int n, double *netOut) {
    int total[256] = {0};
    for (int i = 0; i < n; i++) total[data[i]]++;
    double Sbase = 0.0;
    for (int v = 0; v < 256; v++) Sbase += hlog[total[v]];
    double baseE = log2(n) - Sbase / n;

    double bestNet = 0.0;
    Instr best = {XOR_PHASE, 2, 0, 0};

    /*
     * Freq-table trick block — all ops of the form:
     *   "apply bijection f(v, amp) to bytes at positions i%stride==phase"
     *
     * new_freq[v] = total[v] - phF[v] + phF[f_inv(v, amp)]
     * evaluated in O(256) per (stride, phase, amp).
     *
     * Strides 1..32: ~528 (stride,phase) pairs, fast enough.
     */
    {
        int phF[256];
        for (int stride = 1; stride <= 32; stride++) {
            for (int phase = 0; phase < stride; phase++) {
                memset(phF, 0, sizeof phF);
                for (int i = phase; i < n; i += stride) phF[data[i]]++;

                /* XOR_PHASE / ADD_PHASE: amp 1..255 */
                for (int amp = 1; amp < 256; amp++) {
                    double Snf_x=0, Snf_a=0;
                    for (int v = 0; v < 256; v++) {
                        Snf_x += hlog[total[v]-phF[v]+phF[v^amp]];
                        Snf_a += hlog[total[v]-phF[v]+phF[(v-amp)&0xFF]];
                    }
                    if ((Snf_x-Sbase)-16>bestNet){bestNet=(Snf_x-Sbase)-16;best=(Instr){XOR_PHASE,stride,phase,amp};}
                    if ((Snf_a-Sbase)-16>bestNet){bestNet=(Snf_a-Sbase)-16;best=(Instr){ADD_PHASE, stride,phase,amp};}
                }

                /* ADD_NIBS: independent nibble addition, amp 1..255
                 * amp = lo_amp | (hi_amp << 4), both 0-15 (skip 0,0 = identity)
                 * inverse: lo nibble was (v_lo - lo_amp)&0xF, hi nibble was (v_hi - hi_amp)&0xF */
                for (int amp = 1; amp < 256; amp++) {
                    int lo_amp = amp & 0x0F, hi_amp = (amp >> 4) & 0x0F;
                    double Snf = 0.0;
                    for (int v = 0; v < 256; v++) {
                        int old_v = ((v - lo_amp) & 0x0F) | (((v>>4) - hi_amp) & 0x0F) << 4;
                        Snf += hlog[total[v]-phF[v]+phF[old_v]];
                    }
                    if ((Snf-Sbase)-16>bestNet){bestNet=(Snf-Sbase)-16;best=(Instr){ADD_NIBS,stride,phase,amp};}
                }

                /* XOR/ADD nibble ops: amp 1..15 */
                for (int amp = 1; amp < 16; amp++) {
                    double Snf_xlo=0, Snf_xhi=0, Snf_alo=0, Snf_ahi=0;
                    for (int v = 0; v < 256; v++) {
                        Snf_xlo += hlog[total[v]-phF[v]+phF[(v&0xF0)|((v^amp)&0x0F)]];
                        Snf_xhi += hlog[total[v]-phF[v]+phF[v^((amp&0xF)<<4)]];
                        Snf_alo += hlog[total[v]-phF[v]+phF[(v&0xF0)|((v-amp)&0x0F)]];
                        Snf_ahi += hlog[total[v]-phF[v]+phF[(v&0x0F)|((((v>>4)-amp)&0xF)<<4)]];
                    }
                    if ((Snf_xlo-Sbase)-16>bestNet){bestNet=(Snf_xlo-Sbase)-16;best=(Instr){XOR_LO_NIB,stride,phase,amp};}
                    if ((Snf_xhi-Sbase)-16>bestNet){bestNet=(Snf_xhi-Sbase)-16;best=(Instr){XOR_HI_NIB,stride,phase,amp};}
                    if ((Snf_alo-Sbase)-16>bestNet){bestNet=(Snf_alo-Sbase)-16;best=(Instr){ADD_LO_NIB,stride,phase,amp};}
                    if ((Snf_ahi-Sbase)-16>bestNet){bestNet=(Snf_ahi-Sbase)-16;best=(Instr){ADD_HI_NIB,stride,phase,amp};}
                }

                /* MUL_ODD: odd amp 3..255 */
                for (int amp = 3; amp < 256; amp += 2) {
                    double Snf = 0.0;
                    for (int v = 0; v < 256; v++)
                        Snf += hlog[total[v]-phF[v]+phF[(v*mul_inv[amp])&0xFF]];
                    if ((Snf-Sbase)-16>bestNet){bestNet=(Snf-Sbase)-16;best=(Instr){MUL_ODD,stride,phase,amp};}
                }

                /* ROTL / ROTR: amp 1..7 */
                for (int amp = 1; amp <= 7; amp++) {
                    double Snf_l=0, Snf_r=0;
                    for (int v = 0; v < 256; v++) {
                        Snf_l += hlog[total[v]-phF[v]+phF[rotr8((u8)v,amp)]];
                        Snf_r += hlog[total[v]-phF[v]+phF[rotl8((u8)v,amp)]];
                    }
                    if ((Snf_l-Sbase)-16>bestNet){bestNet=(Snf_l-Sbase)-16;best=(Instr){ROTL,stride,phase,amp};}
                    if ((Snf_r-Sbase)-16>bestNet){bestNet=(Snf_r-Sbase)-16;best=(Instr){ROTR,stride,phase,amp};}
                }

                /* NIB_SWAP / GRAY_ENC / GRAY_DEC / BIT_REV (no amp) */
                {
                    double Snf_sw=0, Snf_ge=0, Snf_gd=0, Snf_br=0;
                    for (int v = 0; v < 256; v++) {
                        Snf_sw += hlog[total[v]-phF[v]+phF[(u8)((v<<4)|(v>>4))]];
                        Snf_ge += hlog[total[v]-phF[v]+phF[gray_decode((u8)v)]];
                        Snf_gd += hlog[total[v]-phF[v]+phF[gray_encode((u8)v)]];
                        Snf_br += hlog[total[v]-phF[v]+phF[bit_reverse8((u8)v)]];
                    }
                    if ((Snf_sw-Sbase)-8>bestNet){bestNet=(Snf_sw-Sbase)-8;best=(Instr){NIB_SWAP,stride,phase,0};}
                    if ((Snf_ge-Sbase)-8>bestNet){bestNet=(Snf_ge-Sbase)-8;best=(Instr){GRAY_ENC,stride,phase,0};}
                    if ((Snf_gd-Sbase)-8>bestNet){bestNet=(Snf_gd-Sbase)-8;best=(Instr){GRAY_DEC,stride,phase,0};}
                    if ((Snf_br-Sbase)-8>bestNet){bestNet=(Snf_br-Sbase)-8;best=(Instr){BIT_REV, stride,phase,0};}
                }

                /* COND_LO_XOR: nib_cond 0..15, xv 1..15 — only the nib_cond family of 16 values changes */
                for (int nib_cond = 0; nib_cond < 16; nib_cond++) {
                    for (int xv = 1; xv < 16; xv++) {
                        double delta_S = 0.0;
                        for (int lo = 0; lo < 16; lo++) {
                            int v = (nib_cond<<4)|lo, v_xv = (nib_cond<<4)|(lo^xv);
                            delta_S += hlog[total[v]-phF[v]+phF[v_xv]] - hlog[total[v]];
                        }
                        if (delta_S-16>bestNet){bestNet=delta_S-16;best=(Instr){COND_LO_XOR,stride,phase,(nib_cond<<4)|xv};}
                    }
                }

                /*
                 * AFFINE_PHASE: bijection v → (v*m + c) & 0xFF
                 * m ∈ odd {3,5,7,9,11,13,15,17,19,21,23,25,27,29,31} (15 values)
                 * c ∈ 0..255 (256 values)
                 * inv: (v-c)*modinv[m] & 0xFF
                 * cost: 24 bits (needs extra byte for m vs ADD_PHASE)
                 * Only searched at stride 1..8 to control total work.
                 */
                if (stride <= 8) {
                    for (int m = 3; m <= 31; m += 2) {
                        int minv = mul_inv[m];
                        for (int c = 0; c < 256; c++) {
                            double Snf = 0.0;
                            for (int v = 0; v < 256; v++) {
                                int old_v = ((v - c) * minv) & 0xFF;
                                Snf += hlog[total[v]-phF[v]+phF[old_v]];
                            }
                            double net = (Snf-Sbase)-24.0;
                            if (net>bestNet){bestNet=net;best=(Instr){AFFINE_PHASE,stride,phase,m|(c<<8)};}
                        }
                    }
                }
            }
        }
    }

    /* ── direct-measure block ─────────────────────────────────────────────── */
    {
        u8 *buf = malloc(n);

        /* XOR3: stride 1..12 */
        for (int stride = 1; stride <= 12; stride++) {
            memcpy(buf,data,n); applyInstr(buf,n,(Instr){XOR3,stride,0,0});
            double net=(baseE-entropy(buf,n))*n-8;
            if (net>bestNet){bestNet=net;best=(Instr){XOR3,stride,0,0};}
        }

        /* XOR_FOLD */
        {
            memcpy(buf,data,n); applyInstr(buf,n,(Instr){XOR_FOLD,0,0,0});
            double net=(baseE-entropy(buf,n))*n-8;
            if (net>bestNet){bestNet=net;best=(Instr){XOR_FOLD,0,0,0};}
        }

        /* HI/LO/HI2_NIB_FEED / CARRY_TOP: stride 1..24 */
        for (int stride = 1; stride <= 24; stride++) {
            memcpy(buf,data,n); applyInstr(buf,n,(Instr){HI_NIB_FEED,stride,0,0});
            double net=(baseE-entropy(buf,n))*n-8;
            if (net>bestNet){bestNet=net;best=(Instr){HI_NIB_FEED,stride,0,0};}

            memcpy(buf,data,n); applyInstr(buf,n,(Instr){LO_NIB_FEED,stride,0,0});
            net=(baseE-entropy(buf,n))*n-8;
            if (net>bestNet){bestNet=net;best=(Instr){LO_NIB_FEED,stride,0,0};}

            memcpy(buf,data,n); applyInstr(buf,n,(Instr){HI2_NIB_FEED,stride,0,0});
            net=(baseE-entropy(buf,n))*n-8;
            if (net>bestNet){bestNet=net;best=(Instr){HI2_NIB_FEED,stride,0,0};}

            memcpy(buf,data,n); applyInstr(buf,n,(Instr){CARRY_TOP,stride,0,0});
            net=(baseE-entropy(buf,n))*n-8;
            if (net>bestNet){bestNet=net;best=(Instr){CARRY_TOP,stride,0,0};}
        }

        /* TOFFOLI: stride 1..24 */
        for (int stride = 1; stride <= 24; stride++) {
            memcpy(buf,data,n); applyInstr(buf,n,(Instr){TOFFOLI,stride,0,0});
            double net=(baseE-entropy(buf,n))*n-8;
            if (net>bestNet){bestNet=net;best=(Instr){TOFFOLI,stride,0,0};}
        }

        /* MODULAR_CHAIN: stride 1..16, amp 1..255 */
        for (int stride = 1; stride <= 16; stride++) {
            for (int amp = 1; amp < 256; amp++) {
                memcpy(buf,data,n); applyInstr(buf,n,(Instr){MODULAR_CHAIN,stride,0,amp});
                double net=(baseE-entropy(buf,n))*n-16;
                if (net>bestNet){bestNet=net;best=(Instr){MODULAR_CHAIN,stride,0,amp};}
            }
        }

        free(buf);
    }

    /* ── add new instruction type searches above this line ───────────────── */

    if (netOut) *netOut = bestNet;
    return best;
}

/* ── greedy compress with scramble-restart meta-loop ────────────────────── */
static double compress(u8 *data, int n, int verbose, int *counts) {
    double total = 0.0;

    for (;;) {
        /* primary greedy loop */
        for (;;) {
            double net;
            Instr t = findBest(data, n, &net);
            if (net <= 0.0) break;
            double e0 = entropy(data, n);
            applyInstr(data, n, t);
            total += net;
            counts[t.type]++;
            if (verbose)
                printf("  %-14s s%-2d p%-2d a%-5d  %.6f -> %.6f  net=%.1f\n",
                       INSTR_NAMES[t.type], t.stride, t.phase, t.amp,
                       e0, entropy(data, n), net);
        }

        /*
         * Scramble restart: try layout permutations when greedy is stuck.
         * TRUE net = (E_before_scramble - E_after_both_ops)*n - 8.
         * Correctly handles scrambles that change byte values (bit-plane, etc.)
         */
        int found_scramble = 0;
        double E_before_scramble = entropy(data, n);
        u8 *temp  = malloc(n);
        u8 *after = malloc(n);

#define TRY_SCRAMBLE(label, cond, scramble_call) \
        if (!found_scramble && (cond)) { \
            scramble_call; \
            double net2; Instr t2 = findBest(temp, n, &net2); \
            memcpy(after, temp, n); applyInstr(after, n, t2); \
            double true_net = (E_before_scramble - entropy(after, n)) * n - 8.0; \
            if (true_net > 0) { \
                memcpy(data, temp, n); \
                double e0 = entropy(data, n); \
                applyInstr(data, n, t2); \
                total += true_net; counts[t2.type]++; found_scramble = 1; \
                if (verbose) \
                    printf("  [%-14s] %-14s s%-2d p%-2d a%-5d  %.6f -> %.6f  net=%.1f\n", \
                           label, INSTR_NAMES[t2.type], t2.stride, t2.phase, t2.amp, \
                           e0, entropy(data, n), true_net); \
            } \
        }

        TRY_SCRAMBLE("INTERLEAVE-2",  n%2==0,   interleave_stride(data,temp,n,2))
        TRY_SCRAMBLE("INTERLEAVE-4",  n%4==0,   interleave_stride(data,temp,n,4))
        TRY_SCRAMBLE("INTERLEAVE-8",  n%8==0,   interleave_stride(data,temp,n,8))
        TRY_SCRAMBLE("INTERLEAVE-16", n%16==0,  interleave_stride(data,temp,n,16))
        TRY_SCRAMBLE("INTERLEAVE-32", n%32==0,  interleave_stride(data,temp,n,32))
        TRY_SCRAMBLE("INTERLEAVE-64", n%64==0,  interleave_stride(data,temp,n,64))
        TRY_SCRAMBLE("BIT-PLANE-SEP", n%8==0,   bit_plane_sep(data,temp,n))
        TRY_SCRAMBLE("NIB-PLANE-SEP", n%2==0,   nib_plane_sep(data,temp,n))
        TRY_SCRAMBLE("BLOCK-REVERSE", 1,        block_reverse(data,temp,n))
        TRY_SCRAMBLE("XOR-FOLD-SCR",  n%2==0,   xor_fold_scramble(data,temp,n))

#undef TRY_SCRAMBLE

        free(after);
        free(temp);
        if (!found_scramble) break;
    }
    return total;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void) {
    init_hlog();
    init_mul_inv();
    int counts[NUM_INSTR_TYPES] = {0};
    double sum = 0.0;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        u8 *data = malloc(BLOCK_SIZE);
        fill_random(data, BLOCK_SIZE);
        double e0 = entropy(data, BLOCK_SIZE);
        double net = compress(data, BLOCK_SIZE, /*verbose=*/ b == 0, counts);
        double e1 = entropy(data, BLOCK_SIZE);
        printf("block %2d: %.6f -> %.6f  net=%.1f bits\n", b, e0, e1, net);
        sum += net;
        free(data);
    }
    printf("\navg net: %.1f bits/block over %d blocks\n", sum / NUM_BLOCKS, NUM_BLOCKS);

    printf("\ninstruction usage counts:\n");
    for (int i = 0; i < NUM_INSTR_TYPES; i++)
        printf("  %-16s %d\n", INSTR_NAMES[i], counts[i]);

    return 0;
}
