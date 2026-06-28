/*
 * simple.c â€” greedy entropy compressor
 *
 * All instructions use the frequency-table trick:
 *   new_freq[v] = (total[v]-phF[v]) + phF[inverse(v, amp)]
 * evaluating ALL amp values in O(256) per (stride, phase) pair.
 *
 * DUAL_XOR / DUAL_ADD split a (stride, phase) into even/odd occurrences
 * and apply independent amps to each â€” same pattern as two XOR_PHASE calls
 * at stride*2, but packed into one instruction (24 bits vs 32 bits overhead).
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
#define NUM_BLOCKS 1

/* Overhead model: instructions with stride+phase cost INSTR_BASE + amp bits.
 * Instructions without stride/phase (LAG_XOR, SCRAMBLE) cost INSTR_BASE_NOPHASE + amp bits.
 * INSTR_BASE covers type+stride+phase; INSTR_BASE_NOPHASE covers type only (~5 bits).
 * Amp bits by type: xor/add_phase(8), cond_*(8), value_xor(17), dual(16), dual_mul(14),
 *   nib_swap/pack_xor(0), bit_rotate(3), lag_xor(8), scramble(4), triple(24/21), quad(32/28). */
#define INSTR_BASE          12
#define INSTR_BASE_NOPHASE   5   /* type only: log2(24)~4.6 bits, no stride/phase */
#define INSTR_OHD(ab)         ((double)(INSTR_BASE         + (ab)))
#define INSTR_OHD_NOPHASE(ab) ((double)(INSTR_BASE_NOPHASE + (ab)))

/* â”€â”€ log table: hlog[x] = x*log2(x), hlog[0]=0 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
static double hlog[BLOCK_SIZE + 1];
static int   g_skip_quad = 0;
static void init_hlog(void) {
    hlog[0] = 0.0;
    for (int i = 1; i <= BLOCK_SIZE; i++) hlog[i] = i * log2(i);
}

/* 256-point FFT twiddle factors: g_tw_r[k]=cos(2πk/256), g_tw_i[k]=sin(2πk/256) */
static double g_tw_r[128], g_tw_i[128];
static void init_fft256(void) {
    for (int k = 0; k < 128; k++) {
        g_tw_r[k] = cos(2.0 * M_PI * k / 256.0);
        g_tw_i[k] = sin(2.0 * M_PI * k / 256.0);
    }
}

/* In-place 256-point DFT.  sign=+1: forward (exp(-2πi) twiddle).
 * sign=-1: unnormalized inverse (exp(+2πi) twiddle; result scaled by 256). */
static void fft256_core(double *re, double *im, int sign) {
    /* bit-reversal permutation for n=256 */
    for (int i = 1, j = 0; i < 256; i++) {
        int bit = 128;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) {
            double t; t=re[i]; re[i]=re[j]; re[j]=t;
                      t=im[i]; im[i]=im[j]; im[j]=t;
        }
    }
    for (int len = 2; len <= 256; len <<= 1) {
        int half = len >> 1, tw_step = 256 / len;
        for (int i = 0; i < 256; i += len) {
            for (int j = 0; j < half; j++) {
                int k = j * tw_step;
                double wr = g_tw_r[k], wi = g_tw_i[k];
                double ur = re[i+j], ui = im[i+j];
                /* forward: twiddle = wr - i*wi; inverse: twiddle = wr + i*wi
                 * vr = re_v*wr + sign*im_v*wi
                 * vi = -sign*re_v*wi + im_v*wr */
                double rv = re[i+j+half], iv = im[i+j+half];
                double vr = rv*wr + sign*iv*wi;
                double vi = -sign*rv*wi + iv*wr;
                re[i+j]      = ur+vr; im[i+j]      = ui+vi;
                re[i+j+half] = ur-vr; im[i+j+half] = ui-vi;
            }
        }
    }
}

/* Cyclic ADD cross-correlation: corr[m] = Σ_v dv[v]*phF[(v-m)%256].
 * Uses FFT: C = IFFT(FFT(dv) * conj(FFT(phF))) / 256. */
static void add_cyclic_corr(const int *dv, const int *phF, int *corr) {
    double Ar[256], Ai[256], Br[256], Bi[256];
    for (int v = 0; v < 256; v++) {
        Ar[v]=(double)dv[v]; Ai[v]=0.0;
        Br[v]=(double)phF[v]; Bi[v]=0.0;
    }
    fft256_core(Ar, Ai, +1);
    fft256_core(Br, Bi, +1);
    /* product[k] = FFT(dv)[k] * conj(FFT(phF)[k])
     * (Ar+i*Ai)*(Br-i*Bi) = Ar*Br+Ai*Bi + i*(Ai*Br-Ar*Bi) */
    for (int k = 0; k < 256; k++) {
        double pr = Ar[k]*Br[k] + Ai[k]*Bi[k];
        double pi = Ai[k]*Br[k] - Ar[k]*Bi[k];
        Ar[k]=pr; Ai[k]=pi;
    }
    fft256_core(Ar, Ai, -1);  /* unnormalized inverse → result = 256 * C[m] */
    for (int k = 0; k < 256; k++) corr[k] = (int)round(Ar[k] / 256.0);
}

/* â”€â”€ entropy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
static double entropy(const u8 *data, int n) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[data[i]]++;
    double s = 0.0;
    for (int i = 0; i < 256; i++) s += hlog[f[i]];
    return log2(n) - s / n;
}

/* â”€â”€ data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
static void fill_random(u8 *buf, int n) {
    BCryptGenRandom(NULL, buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}

/* â”€â”€ multiply-inverse table mod 256 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
static u8 mul_inv[256];
static void init_mul_inv(void) {
    for (int k = 1; k < 256; k += 2)
        for (int inv = 1; inv < 256; inv += 2)
            if (((k * inv) & 0xFF) == 1) { mul_inv[k] = (u8)inv; break; }
}


/* â”€â”€ instructions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

typedef enum {
    /* 20-bit overhead (12 base + 8-bit amp) */
    XOR_PHASE = 0,  /* data[i] ^= amp                                          */
    ADD_PHASE,      /* data[i] += amp  (mod 256)                               */
    PACK_XOR,       /* data[phase+2k*s] ^= data[phase+(2k+1)*s]; self-inverse   */
    COND_LO_XOR,    /* if hi-nib==(amp>>4): lo-nib ^= (amp&0xF)               */
    ADD_NIBS,       /* lo-nib += (amp&0xF), hi-nib += (amp>>4), no carry       */
    COND_HI_XOR,    /* if lo-nib==(amp&0xF): hi-nib ^= ((amp>>4)<<4)          */
    COND_LO_ADD,    /* if hi-nib==(amp>>4): lo-nib += (amp&0xF) mod 16        */
    COND_HI_ADD,    /* if lo-nib==(amp&0xF): hi-nib += (amp>>4) mod 16        */
    /* 12-bit overhead (12 base, no amp) */
    NIB_SWAP,       /* data[i] = (data[i]<<4)|(data[i]>>4), self-inverse       */
    /* 15-bit overhead (12 base + 3-bit k) */
    BIT_ROTATE,     /* data[i] = rotl(data[i], k); inverse is rotl(_, 8-k)    */
    /* 29-bit overhead (12 base + 17-bit amp); k=amp&7, alo=amp>>3 (7 eff), ahi=amp>>11 (7 eff) */
    VALUE_XOR,      /* (data[i]>>k)&1 ? ^ahi : ^alo; amps have bit k=0, self-inv */
    /* 28-bit overhead (12 base + 16-bit amp: lo|hi<<8) */
    DUAL_XOR,       /* even ^= (amp&0xFF), odd ^= ((amp>>8)&0xFF)              */
    DUAL_ADD,       /* even += (amp&0xFF), odd += ((amp>>8)&0xFF)              */
    /* 26-bit overhead (12 base + 14-bit amp: idx_lo|idx_hi<<7) */
    DUAL_MUL,       /* even *= m_lo, odd *= m_hi; amp = idx_lo|(idx_hi<<7)    */
    /* 13-bit overhead (5-bit type-only base + 8-bit lag; no stride/phase) */
    LAG_XOR,        /* data[i] ^= data[i+amp], left-to-right; inv right-to-left */
    /* 28-bit overhead (12 base + 16-bit amp: alo|ahi<<8; HAS stride+phase) */
    COND_PREV_XOR,  /* data[i] ^= (prev_in_group>=128)?ahi:alo; chain, amp=lo|(hi<<8) */
    /* 18-bit overhead (12 base + 6 extra for second stride, no amp) */
    POLY_DELTA_XOR, /* data[i] ^= data[i-s1]^data[i-s2] for i>=s2             */
    /* 9-bit overhead (5-bit type-only base + 4-bit index; no stride/phase) */
    SCRAMBLE,       /* position rearrangement; amp encodes scramble type        */
    /* 33-bit overhead (9 base + 3Ã—8 amp) */
    TRIPLE_XOR,     /* group0 ^= a0, group1 ^= a1, group2 ^= a2 (pos mod 3)   */
    TRIPLE_ADD,     /* group0 += a0, group1 += a1, group2 += a2 (pos mod 3)   */
    /* 30-bit overhead (9 base + 3Ã—7 amp) */
    TRIPLE_MUL,     /* group0 *= m0, group1 *= m1, group2 *= m2; idx per 7b   */
    /* 44-bit overhead (12 base + 4x8 amp packed as uint32) */
    QUAD_XOR,       /* group0^a0, group1^a1, group2^a2, group3^a3 (pos mod 4) */
    QUAD_ADD,       /* group0+=a0, group1+=a1, group2+=a2, group3+=a3         */
    /* 40-bit overhead (12 base + 4x7 amp) */
    QUAD_MUL,       /* group0*=m0, ..., group3*=m3; idx per 7b (pos mod 4)    */
    NUM_INSTR_TYPES /* 24 total */
} InstrType;

static const char *INSTR_NAMES[NUM_INSTR_TYPES] = {
    "XOR_PHASE", "ADD_PHASE", "PACK_XOR", "COND_LO_XOR", "ADD_NIBS",
    "COND_HI_XOR", "COND_LO_ADD", "COND_HI_ADD", "NIB_SWAP",
    "BIT_ROTATE", "VALUE_XOR",
    "DUAL_XOR", "DUAL_ADD", "DUAL_MUL", "LAG_XOR",
    "COND_PREV_XOR", "POLY_DELTA_XOR", "SCRAMBLE",
    "TRIPLE_XOR", "TRIPLE_ADD", "TRIPLE_MUL",
    "QUAD_XOR", "QUAD_ADD", "QUAD_MUL"
};

typedef struct { InstrType type; int stride, phase; unsigned int amp; } Instr;

static void interleave_stride(const u8 *src, u8 *dst, int n, int s);
static void bit_plane_sep(const u8 *src, u8 *dst, int n);
static void nib_plane_sep(const u8 *src, u8 *dst, int n);
static void block_reverse(const u8 *src, u8 *dst, int n);
static void xor_fold_scramble(const u8 *src, u8 *dst, int n);

static void applyInstr(u8 *data, int n, Instr t) {
    int i;
    switch (t.type) {
        case XOR_PHASE:
            for (i = t.phase; i < n; i += t.stride)
                data[i] ^= (u8)t.amp;
            break;
        case ADD_PHASE:
            for (i = t.phase; i < n; i += t.stride)
                data[i] += (u8)t.amp;
            break;
        case PACK_XOR: {
            int k = 0;
            for (i = t.phase; i < n; i += t.stride, k++) {
                if (k & 1) continue;
                int j = i + t.stride;
                if (j >= n) continue;
                data[i] ^= data[j];
            }
            break;
        }
        case COND_LO_XOR: {
            int nib_cond = (t.amp >> 4) & 0xF;
            int xv       =  t.amp       & 0x0F;
            for (i = t.phase; i < n; i += t.stride)
                if ((data[i] >> 4) == nib_cond)
                    data[i] ^= (u8)xv;
            break;
        }
        case ADD_NIBS: {
            u8 lo = (u8)(t.amp & 0x0F), hi = (u8)((t.amp >> 4) & 0x0F);
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)(((data[i] + lo) & 0x0F) |
                               ((((data[i] >> 4) + hi) & 0x0F) << 4));
            break;
        }
        case NIB_SWAP:
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)((data[i] << 4) | (data[i] >> 4));
            break;
        case BIT_ROTATE: {
            int k = t.amp & 7;
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)((data[i] << k) | (data[i] >> (8 - k)));
            break;
        }
        case VALUE_XOR: {
            int k = t.amp & 7, mask = 1 << k;
            int alo = (t.amp >> 3) & 0xFF, ahi = (t.amp >> 11) & 0xFF;
            for (i = t.phase; i < n; i += t.stride)
                data[i] ^= (u8)((data[i] & mask) ? ahi : alo);
            break;
        }
        case DUAL_XOR: {
            int lo = t.amp & 0xFF, hi = (t.amp >> 8) & 0xFF;
            int k = 0;
            for (i = t.phase; i < n; i += t.stride, k++)
                data[i] ^= (u8)(k & 1 ? hi : lo);
            break;
        }
        case DUAL_ADD: {
            int lo = t.amp & 0xFF, hi = (t.amp >> 8) & 0xFF;
            int k = 0;
            for (i = t.phase; i < n; i += t.stride, k++)
                data[i] += (u8)(k & 1 ? hi : lo);
            break;
        }
        case COND_HI_XOR: {
            int nc = t.amp & 0xF, xv = (t.amp >> 4) & 0xF;
            for (i = t.phase; i < n; i += t.stride)
                if ((data[i] & 0xF) == nc) data[i] ^= (u8)(xv << 4);
            break;
        }
        case COND_LO_ADD: {
            int nc = (t.amp >> 4) & 0xF, av = t.amp & 0xF;
            for (i = t.phase; i < n; i += t.stride)
                if ((data[i] >> 4) == nc)
                    data[i] = (u8)((data[i] & 0xF0) | ((data[i] + av) & 0xF));
            break;
        }
        case COND_HI_ADD: {
            int nc = t.amp & 0xF, av = (t.amp >> 4) & 0xF;
            for (i = t.phase; i < n; i += t.stride)
                if ((data[i] & 0xF) == nc)
                    data[i] = (u8)((data[i] & 0x0F) | (((data[i] >> 4) + av) & 0xF) << 4);
            break;
        }
        case DUAL_MUL: {
            int m_lo = (t.amp & 0x7F) * 2 + 3, m_hi = ((t.amp >> 7) & 0x7F) * 2 + 3;
            int k = 0;
            for (i = t.phase; i < n; i += t.stride, k++)
                data[i] = (u8)((data[i] * (k & 1 ? m_hi : m_lo)) & 0xFF);
            break;
        }
        case LAG_XOR: {
            int lag = (int)t.amp;
            for (i = 0; i + lag < n; i++)
                data[i] ^= data[i + lag];
            break;
        }
        case COND_PREV_XOR: {
            int alo = t.amp & 0xFF, ahi = (t.amp >> 8) & 0xFF;
            int prev_val = 0, first = 1;
            for (i = t.phase; i < n; i += t.stride) {
                if (first) { first = 0; prev_val = data[i]; continue; }
                data[i] ^= (u8)((prev_val >= 128) ? ahi : alo);
                prev_val = data[i];
            }
            break;
        }
        case POLY_DELTA_XOR: {
            int s1 = t.stride, s2 = t.phase; /* reuse fields: s1<s2 */
            for (i = n - 1; i >= s2; i--)
                data[i] ^= data[i - s1] ^ data[i - s2];
            break;
        }
        case SCRAMBLE: {
            u8 tmp[BLOCK_SIZE];
            int sc = t.amp & 0xF;
            if      (sc == 0) interleave_stride(data, tmp, n, 2);
            else if (sc == 1) interleave_stride(data, tmp, n, 4);
            else if (sc == 2) interleave_stride(data, tmp, n, 8);
            else if (sc == 3) interleave_stride(data, tmp, n, 16);
            else if (sc == 4)  interleave_stride(data, tmp, n, 32);
            else if (sc == 5)  interleave_stride(data, tmp, n, 64);
            else if (sc == 6)  bit_plane_sep(data, tmp, n);
            else if (sc == 7)  nib_plane_sep(data, tmp, n);
            else if (sc == 8)  block_reverse(data, tmp, n);
            else if (sc == 9)  xor_fold_scramble(data, tmp, n);
            else if (sc == 10) interleave_stride(data, tmp, n, 1024);
            else if (sc == 11) interleave_stride(data, tmp, n, 512);
            else if (sc == 12) interleave_stride(data, tmp, n, 256);
            else               interleave_stride(data, tmp, n, 128);
            memcpy(data, tmp, n);
            break;
        }
        case TRIPLE_XOR: {
            int a0=t.amp&0xFF, a1=(t.amp>>8)&0xFF, a2=(t.amp>>16)&0xFF;
            int k=0;
            for (i=t.phase; i<n; i+=t.stride, k++) {
                if      (k%3==0) data[i]^=(u8)a0;
                else if (k%3==1) data[i]^=(u8)a1;
                else             data[i]^=(u8)a2;
            }
            break;
        }
        case TRIPLE_ADD: {
            int a0=t.amp&0xFF, a1=(t.amp>>8)&0xFF, a2=(t.amp>>16)&0xFF;
            int k=0;
            for (i=t.phase; i<n; i+=t.stride, k++) {
                if      (k%3==0) data[i]+=(u8)a0;
                else if (k%3==1) data[i]+=(u8)a1;
                else             data[i]+=(u8)a2;
            }
            break;
        }
        case TRIPLE_MUL: {
            int m0=(t.amp&0x7F)*2+3, m1=((t.amp>>7)&0x7F)*2+3, m2=((t.amp>>14)&0x7F)*2+3;
            int k=0;
            for (i=t.phase; i<n; i+=t.stride, k++) {
                if      (k%3==0) data[i]=(u8)((data[i]*m0)&0xFF);
                else if (k%3==1) data[i]=(u8)((data[i]*m1)&0xFF);
                else             data[i]=(u8)((data[i]*m2)&0xFF);
            }
            break;
        }
        case QUAD_XOR: {
            unsigned a0=t.amp&0xFF, a1=(t.amp>>8)&0xFF, a2=(t.amp>>16)&0xFF, a3=t.amp>>24;
            unsigned aa[4]={a0,a1,a2,a3}; int k=0;
            for (i=t.phase; i<n; i+=t.stride, k++) data[i]^=(u8)aa[k&3];
            break;
        }
        case QUAD_ADD: {
            unsigned a0=t.amp&0xFF, a1=(t.amp>>8)&0xFF, a2=(t.amp>>16)&0xFF, a3=t.amp>>24;
            unsigned aa[4]={a0,a1,a2,a3}; int k=0;
            for (i=t.phase; i<n; i+=t.stride, k++) data[i]+=(u8)aa[k&3];
            break;
        }
        case QUAD_MUL: {
            int m0=(t.amp&0x7F)*2+3, m1=((t.amp>>7)&0x7F)*2+3;
            int m2=((t.amp>>14)&0x7F)*2+3, m3=((t.amp>>21)&0x7F)*2+3;
            int mm[4]={m0,m1,m2,m3}; int k=0;
            for (i=t.phase; i<n; i+=t.stride, k++) data[i]=(u8)((data[i]*mm[k&3])&0xFF);
            break;
        }
        default: break;
    }
}

/* â”€â”€ Walsh-Hadamard helpers for fast XOR-amp search â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

/* In-place Walsh-Hadamard Transform, n=256 */
static void wht256(int *a) {
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += len << 1)
            for (int j = 0; j < len; j++) {
                int u = a[i+j], v = a[i+j+len];
                a[i+j] = u+v; a[i+j+len] = u-v;
            }
}

/* WHT for 128 elements */
static void wht128(int *a) {
    for (int len = 1; len < 128; len <<= 1)
        for (int i = 0; i < 128; i += len << 1)
            for (int j = 0; j < len; j++) {
                int u = a[i+j], v = a[i+j+len];
                a[i+j] = u+v; a[i+j+len] = u-v;
            }
}

/* XOR best amp (idx 1..127) for 128-element remapped arrays. Same structure as
 * xor_best_amp. Used by VALUE_XOR search for each bit-k split subgroup. */
static int xor_best_amp_128(const int *A, const int *B, double *Sout) {
    int ha[128], hb[128];
    for (int i = 0; i < 128; i++) { ha[i] = A[i]; hb[i] = B[i]; }
    wht128(ha); wht128(hb);
    long long prod[128];
    for (int k = 0; k < 128; k++) prod[k] = (long long)ha[k] * hb[k];
    for (int len = 1; len < 128; len <<= 1)
        for (int i = 0; i < 128; i += len << 1)
            for (int j = 0; j < len; j++) {
                long long u = prod[i+j], v = prod[i+j+len];
                prod[i+j] = u+v; prod[i+j+len] = u-v;
            }
    long long c0 = INT64_MIN, c1 = INT64_MIN, c2 = INT64_MIN;
    int a0 = 1, a1 = 2, a2 = 3;
    for (int amp = 1; amp < 128; amp++) {
        long long c = prod[amp];
        if      (c > c0) { c2=c1;a2=a1; c1=c0;a1=a0; c0=c;a0=amp; }
        else if (c > c1) { c2=c1;a2=a1; c1=c;a1=amp; }
        else if (c > c2) { c2=c;a2=amp; }
    }
    int best_amp = a0; double best_S = -1e30;
    int cands[3] = {a0, a1, a2};
    for (int t = 0; t < 3; t++) {
        double S = 0.0;
        for (int i = 0; i < 128; i++) S += hlog[A[i] + B[i ^ cands[t]]];
        if (S > best_S) { best_S = S; best_amp = cands[t]; }
    }
    *Sout = best_S;
    return best_amp;
}

/* Insert 0 at bit-k position into 7-bit idx to produce a value in G0 (bit k=0) */
static int vxor_ins(int idx, int k) {
    return ((idx >> k) << (k + 1)) | (idx & ((1 << k) - 1));
}

/* Find best XOR amp (1..255) for sum_v hlog[A[v] + B[v^amp]].
 * Uses WHT XOR-correlation to find top-3 candidates, then exact verify.
 * Returns best amp; stores exact S in *Sout. */
static int xor_best_amp(const int *A, const int *B, double *Sout) {
    int ha[256], hb[256];
    for (int i = 0; i < 256; i++) { ha[i] = A[i]; hb[i] = B[i]; }
    wht256(ha); wht256(hb);
    /* pointwise product â†’ IWHT gives XOR-correlation Ã— 256 */
    long long prod[256];
    for (int k = 0; k < 256; k++) prod[k] = (long long)ha[k] * hb[k];
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += len << 1)
            for (int j = 0; j < len; j++) {
                long long u = prod[i+j], v = prod[i+j+len];
                prod[i+j] = u+v; prod[i+j+len] = u-v;
            }
    /* find top-3 amps by proxy (skip amp=0) */
    long long c0 = INT64_MIN, c1 = INT64_MIN, c2 = INT64_MIN;
    int       a0 = 1,         a1 = 2,          a2 = 3;
    for (int amp = 1; amp < 256; amp++) {
        long long c = prod[amp];
        if      (c > c0) { c2=c1; a2=a1; c1=c0; a1=a0; c0=c; a0=amp; }
        else if (c > c1) { c2=c1; a2=a1; c1=c; a1=amp; }
        else if (c > c2) { c2=c; a2=amp; }
    }
    /* exact verify top-3 */
    int best_amp = a0; double best_S = -1e30;
    int cands[3] = {a0, a1, a2};
    for (int t = 0; t < 3; t++) {
        double S = 0.0;
        for (int v = 0; v < 256; v++) S += hlog[A[v] + B[v ^ cands[t]]];
        if (S > best_S) { best_S = S; best_amp = cands[t]; }
    }
    *Sout = best_S;
    return best_amp;
}

/* â”€â”€ fast amp-search helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

/* Best cyclic-ADD amp via double-buffer sequential cross-correlation.
 * Stores phF twice so each rotation is a stride-1 slice â†’ auto-vectorises.
 * Top-8 peaks of C[k] = Î£ dv[v]*phF[(v-k)] verified exactly with hlog.
 * Reliable for near-uniform BCrypt data where spike-valley proxy fails. */
static int add_best_amp(const int *dv, const int *phF, int _K, double *Sout) {
    (void)_K;
    int corr[256];
    add_cyclic_corr(dv, phF, corr);
    /* find top-8 peaks in corr[1..255] */
    int top8[8] = {1,2,3,4,5,6,7,8};
    int tv[8];
    for (int i = 0; i < 8; i++) tv[i] = corr[i + 1];
    for (int i = 0; i < 8; i++) for (int j = i+1; j < 8; j++)
        if (tv[j]>tv[i]){ int t=tv[i];tv[i]=tv[j];tv[j]=t; int ti=top8[i];top8[i]=top8[j];top8[j]=ti; }
    for (int k = 9; k < 256; k++) {
        int c = corr[k]; if (c <= tv[7]) continue;
        int pos = 7;
        while (pos > 0 && c > tv[pos-1]) pos--;
        for (int j = 7; j > pos; j--) { top8[j]=top8[j-1]; tv[j]=tv[j-1]; }
        top8[pos] = k; tv[pos] = c;
    }
    /* exact verify top-8 with hlog */
    int best_amp = top8[0]; double best_S = -1e30;
    for (int ci = 0; ci < 8; ci++) {
        int amp = top8[ci];
        double S = 0.0;
        for (int v = 0; v < 256; v++) S += hlog[dv[v]+phF[(v-amp)&0xFF]];
        if (S > best_S) { best_S = S; best_amp = amp; }
    }
    *Sout = best_S; return best_amp;
}

/* Best ADD_NIBS amp: factored nibble proxy via 1-D cyclic correlations. */
static int add_nibs_best_amp(const int *dv, const int *phF, double *Sout) {
    int mlo[16]={0},mhi[16]={0},dlo[16]={0},dhi[16]={0};
    for(int v=0;v<256;v++){mlo[v&0xF]+=phF[v];mhi[v>>4]+=phF[v];dlo[v&0xF]+=dv[v];dhi[v>>4]+=dv[v];}
    int clo[16],chi_c[16];
    for(int la=0;la<16;la++){int c=0;for(int l=0;l<16;l++)c+=dlo[l]*mlo[(l-la)&0xF];clo[la]=c;}
    for(int ha=0;ha<16;ha++){int c=0;for(int h=0;h<16;h++)c+=dhi[h]*mhi[(h-ha)&0xF];chi_c[ha]=c;}
    int lo4[4],hi4[4];
    {int seen[16]={0};for(int k=0;k<4;k++){int bi=0,bv=-1;for(int i=0;i<16;i++)if(!seen[i]&&clo[i]>bv){bv=clo[i];bi=i;}lo4[k]=bi;seen[bi]=1;}}
    {int seen[16]={0};for(int k=0;k<4;k++){int bi=0,bv=-1;for(int i=0;i<16;i++)if(!seen[i]&&chi_c[i]>bv){bv=chi_c[i];bi=i;}hi4[k]=bi;seen[bi]=1;}}
    int best_amp=0x11; double best_S=-1e30;
    for(int i=0;i<4;i++) for(int j=0;j<4;j++){
        int la=lo4[i],ha=hi4[j],amp=la|(ha<<4); if(!amp) continue;
        double S=0.0; for(int v=0;v<256;v++){int ov=((v-la)&0xF)|(((v>>4)-ha)&0xF)<<4;S+=hlog[dv[v]+phF[ov]];}
        if(S>best_S){best_S=S;best_amp=amp;}
    }
    *Sout=best_S; return best_amp;
}

/* Best odd multiplier: 2-level coarse+fine search.
 * Coarse: 8 amps at every 32nd odd (3,35,67,...,227). Fine: Â±16 odd steps around top-3.
 * Reliable for near-uniform data; ~85% coverage of 127 odd amps per pass. */
static int mul_best_amp(const int *dv, const int *phF, double *Sout) {
    /* Level 1: 8 coarse odd amps */
    int   top3a[3] = {3, 35, 67};
    double top3s[3] = {-1e30, -1e30, -1e30};
    for (int m = 3; m < 256; m += 32) {
        double S = 0.0;
        for (int v = 0; v < 256; v++) S += hlog[dv[v]+phF[(v*(int)mul_inv[m])&0xFF]];
        if (S > top3s[0]) { top3s[2]=top3s[1];top3a[2]=top3a[1]; top3s[1]=top3s[0];top3a[1]=top3a[0]; top3s[0]=S;top3a[0]=m; }
        else if (S > top3s[1]) { top3s[2]=top3s[1];top3a[2]=top3a[1]; top3s[1]=S;top3a[1]=m; }
        else if (S > top3s[2]) { top3s[2]=S;top3a[2]=m; }
    }
    /* Level 2: brute force Â±16 odd steps (Â±32 in value) around each top-3 */
    int best_amp = top3a[0]; double best_S = top3s[0];
    for (int ci = 0; ci < 3; ci++) {
        int base = top3a[ci];
        for (int d = -32; d <= 32; d += 2) {
            int m = base + d;
            if (m < 3 || m > 255 || !(m & 1)) continue;
            double S = 0.0;
            for (int v = 0; v < 256; v++) S += hlog[dv[v]+phF[(v*(int)mul_inv[m])&0xFF]];
            if (S > best_S) { best_S = S; best_amp = m; }
        }
    }
    *Sout = best_S; return best_amp;
}

/* â”€â”€ find best instruction â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
static Instr findBest(const u8 *data, int n, double *netOut, int max_stride) {
    int total[256] = {0};
    for (int i = 0; i < n; i++) total[data[i]]++;
    double hlt[256];
    double Sbase = 0.0;
    for (int v = 0; v < 256; v++) { hlt[v] = hlog[total[v]]; Sbase += hlt[v]; }

    double bestNet = 0.0;
    Instr  best    = {XOR_PHASE, 2, 0, 1};

    int phF[256];

    /* LAG_XOR: data[i] ^= data[i+lag] left-to-right; try all lags */
    for (int lag = 1; lag < n && lag <= 255; lag++) {
        int lxfreq[256];
        for (int v = 0; v < 256; v++) lxfreq[v] = total[v];
        for (int ii = 0; ii + lag < n; ii++) {
            lxfreq[data[ii]]--;
            lxfreq[data[ii] ^ data[ii + lag]]++;
        }
        double Slx = 0.0;
        for (int v = 0; v < 256; v++) Slx += hlog[lxfreq[v]];
        double nlx = (Slx - Sbase) - INSTR_OHD_NOPHASE(8);
        if (nlx > bestNet) { bestNet = nlx; best = (Instr){LAG_XOR, 0, 0, (unsigned)lag}; }
    }

    for (int stride = 1; stride <= max_stride; stride++) {
        /* â”€â”€ POLY_DELTA_XOR: limit to stride<=32 to keep search fast â”€â”€â”€â”€â”€ */
        if (stride <= 32) {
        for (int s1 = 1; s1 < stride; s1++) {
            int pdfreq[256] = {0};
            for (int i = 0; i < stride && i < n; i++) pdfreq[data[i]]++;
            for (int i = stride; i < n; i++) pdfreq[data[i] ^ data[i - s1] ^ data[i - stride]]++;
            double Spd = 0.0;
            for (int v = 0; v < 256; v++) Spd += hlog[pdfreq[v]];
            double npd = (Spd - Sbase) - INSTR_OHD(6); /* two strides encoded: +6 bits vs single */
            if (npd > bestNet) {
                bestNet = npd;
                best = (Instr){POLY_DELTA_XOR, s1, stride, 0};
            }
        }
        }

        for (int phase = 0; phase < stride; phase++) {

            /* build phase-position frequency table */
            memset(phF, 0, sizeof phF);
            for (int i = phase; i < n; i += stride) phF[data[i]]++;
            int dv[256];
            for (int v = 0; v < 256; v++) dv[v] = total[v] - phF[v];

            /* â”€â”€ XOR_PHASE: WHT proxy + top-3 exact verify â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
            {
                double Sx;
                int ax = xor_best_amp(dv, phF, &Sx);
                double nx = (Sx - Sbase) - INSTR_OHD(8);
                if (nx > bestNet) { bestNet = nx; best = (Instr){XOR_PHASE, stride, phase, ax}; }
            }
            /* â”€â”€ ADD_PHASE: spike-valley proxy K=6 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
            {
                double Sa; int aa = add_best_amp(dv, phF, 6, &Sa);
                double na = (Sa - Sbase) - INSTR_OHD(8);
                if (na > bestNet) { bestNet = na; best = (Instr){ADD_PHASE, stride, phase, aa}; }
            }

            /* PACK_XOR: XOR even-indexed with adjacent odd-indexed element in stride group */
            {
                int pxfreq[256];
                for (int v = 0; v < 256; v++) pxfreq[v] = total[v];
                int k = 0;
                for (int ii = phase; ii < n; ii += stride, k++) {
                    if (k & 1) continue;
                    int j = ii + stride;
                    if (j >= n) continue;
                    pxfreq[data[ii]]--;
                    pxfreq[data[ii] ^ data[j]]++;
                }
                double Spx = 0.0;
                for (int v = 0; v < 256; v++) Spx += hlog[pxfreq[v]];
                double npx = (Spx - Sbase) - INSTR_OHD(0);
                if (npx > bestNet) { bestNet = npx; best = (Instr){PACK_XOR, stride, phase, 0}; }
            }
            /* COND_PREV_XOR: XOR conditioned on previous stride-group element high bit */
            {
                int phF0[256]={0}, phF1[256]={0};
                int prev_v = 0, cpx_first = 1;
                for (int ii = phase; ii < n; ii += stride) {
                    if (cpx_first) { cpx_first = 0; prev_v = data[ii]; continue; }
                    if (prev_v < 128) phF0[data[ii]]++;
                    else              phF1[data[ii]]++;
                    prev_v = data[ii];
                }
                int bg0[256], bg1[256];
                for (int v = 0; v < 256; v++) {
                    int bg = total[v] - phF0[v] - phF1[v];
                    bg0[v] = bg + phF1[v];
                    bg1[v] = bg;  /* filled in after alo found */
                }
                double Scpx0; int cpx_alo = xor_best_amp(bg0, phF0, &Scpx0);
                for (int v = 0; v < 256; v++) {
                    int bg = total[v] - phF0[v] - phF1[v];
                    bg1[v] = bg + phF0[v ^ cpx_alo];
                }
                double Scpx1; int cpx_ahi = xor_best_amp(bg1, phF1, &Scpx1);
                double Scpx = 0.0;
                for (int v = 0; v < 256; v++) {
                    int bg = total[v] - phF0[v] - phF1[v];
                    Scpx += hlog[bg + phF0[v^cpx_alo] + phF1[v^cpx_ahi]];
                }
                double ncpx = (Scpx - Sbase) - INSTR_OHD(16);
                if (ncpx > bestNet) {
                    bestNet = ncpx;
                    best = (Instr){COND_PREV_XOR, stride, phase, (unsigned)(cpx_alo | (cpx_ahi << 8))};
                }
            }
            /* â”€â”€ COND_LO_XOR: stride<=32 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
            if (stride <= 32) {
            for (int nib_cond = 0; nib_cond < 16; nib_cond++) {
                for (int xv = 1; xv < 16; xv++) {
                    double delta = 0.0;
                    for (int lo = 0; lo < 16; lo++) {
                        int v    = (nib_cond << 4) | lo;
                        int v_xv = (nib_cond << 4) | (lo ^ xv);
                        delta += hlog[dv[v] + phF[v_xv]] - hlt[v];
                    }
                    double nc = delta - INSTR_OHD(8);
                    if (nc > bestNet) { bestNet = nc; best = (Instr){COND_LO_XOR, stride, phase, (nib_cond << 4) | xv}; }
                }
            }
            }

            /* â”€â”€ ADD_NIBS: factored nibble proxy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
            {
                double Sn; int an = add_nibs_best_amp(dv, phF, &Sn);
                double nn = (Sn - Sbase) - INSTR_OHD(8);
                if (nn > bestNet) { bestNet = nn; best = (Instr){ADD_NIBS, stride, phase, an}; }
            }

            /* â”€â”€ NIB_SWAP: self-inverse, cost 8 bits â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
            {
                double Ssw = 0.0;
                for (int v = 0; v < 256; v++) {
                    int sv = ((v << 4) | (v >> 4)) & 0xFF;
                    Ssw += hlog[dv[v] + phF[sv]];
                }
                double nsw = (Ssw - Sbase) - INSTR_OHD(0);
                if (nsw > bestNet) { bestNet = nsw; best = (Instr){NIB_SWAP, stride, phase, 0}; }
            }

            /* â”€â”€ BIT_ROTATE: rotate bits of each byte by k=1..7 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
            for (int k = 1; k <= 7; k++) {
                double Sbr = 0.0;
                for (int v = 0; v < 256; v++) {
                    int rv = ((v >> k) | (v << (8 - k))) & 0xFF; /* rotr(v,k) = original before rotl(k) */
                    Sbr += hlog[dv[v] + phF[rv]];
                }
                double nbr = (Sbr - Sbase) - INSTR_OHD(3);
                if (nbr > bestNet) { bestNet = nbr; best = (Instr){BIT_ROTATE, stride, phase, k}; }
            }

            /* â”€â”€ VALUE_XOR: split by any bit k (0-7); amps have bit k=0 â”€â”€â”€â”€â”€
             * Groups are disjoint under XOR with amp having bit k=0, separable.
             * Search all 8 bit positions; pick the best-net (k, alo, ahi). */
            for (int k = 0; k < 8; k++) {
                int mask = 1 << k;
                /* Remap 128-element subgroups into contiguous index space so
                 * WHT-128 finds the best amp in O(128 log 128) not O(128^2). */
                int A_lo[128], B_lo[128], A_hi[128], B_hi[128];
                for (int i = 0; i < 128; i++) {
                    int v0 = vxor_ins(i, k);
                    A_lo[i] = dv[v0];      B_lo[i] = phF[v0];
                    A_hi[i] = dv[v0|mask]; B_hi[i] = phF[v0|mask];
                }
                double bSlo; int balo_idx = xor_best_amp_128(A_lo, B_lo, &bSlo);
                int balo = vxor_ins(balo_idx, k);
                double bShi; int bahi_idx = xor_best_amp_128(A_hi, B_hi, &bShi);
                int bahi = vxor_ins(bahi_idx, k);
                double Svx = 0.0;
                for (int v = 0; v < 256; v++) {
                    int vx = v ^ ((v & mask) ? bahi : balo);
                    Svx += hlog[dv[v] + phF[vx]];
                }
                double nvx = (Svx - Sbase) - INSTR_OHD(17);
                if (nvx > bestNet) {
                    bestNet = nvx;
                    best = (Instr){VALUE_XOR, stride, phase, (unsigned)(k | (balo << 3) | (bahi << 11))};
                }
            }

            /* â”€â”€ COND_HI_XOR / COND_LO_ADD / COND_HI_ADD: stride<=32 â”€â”€â”€â”€ */
            if (stride <= 32) {

            for (int nc = 0; nc < 16; nc++) {
                for (int xv = 1; xv < 16; xv++) {
                    double delta = 0.0;
                    for (int hi = 0; hi < 16; hi++) {
                        int v    = (hi << 4) | nc;
                        int v_xv = ((hi ^ xv) << 4) | nc;
                        delta += hlog[dv[v] + phF[v_xv]] - hlt[v];
                    }
                    double nc_net = delta - INSTR_OHD(8);
                    if (nc_net > bestNet) { bestNet = nc_net; best = (Instr){COND_HI_XOR, stride, phase, nc | (xv << 4)}; }
                }
            }

            for (int nc = 0; nc < 16; nc++) {
                for (int av = 1; av < 16; av++) {
                    double delta = 0.0;
                    for (int lo = 0; lo < 16; lo++) {
                        int v     = (nc << 4) | lo;
                        int v_inv = (nc << 4) | ((lo - av) & 0xF);
                        delta += hlog[dv[v] + phF[v_inv]] - hlt[v];
                    }
                    double nc_net = delta - INSTR_OHD(8);
                    if (nc_net > bestNet) { bestNet = nc_net; best = (Instr){COND_LO_ADD, stride, phase, (nc << 4) | av}; }
                }
            }

            for (int nc = 0; nc < 16; nc++) {
                for (int av = 1; av < 16; av++) {
                    double delta = 0.0;
                    for (int hi = 0; hi < 16; hi++) {
                        int v     = (hi << 4) | nc;
                        int v_inv = (((hi - av) & 0xF) << 4) | nc;
                        delta += hlog[dv[v] + phF[v_inv]] - hlt[v];
                    }
                    double nc_net = delta - INSTR_OHD(8);
                    if (nc_net > bestNet) { bestNet = nc_net; best = (Instr){COND_HI_ADD, stride, phase, nc | (av << 4)}; }
                }
            }

            } /* end stride<=32 for COND instructions */

            /*
             * â”€â”€ DUAL_XOR and DUAL_ADD â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
             *
             * Split (stride, phase) into even-indexed (k=0,2,...) and
             * odd-indexed (k=1,3,...) occurrences and apply independent amps
             * to each. This is equivalent to two XOR_PHASE/ADD_PHASE calls at
             * stride*2, but encoded in one instruction at 24 bits vs 32 bits.
             *
             * Search: two-pass coordinate descent.
             *   Pass 1: find best amp_lo for even occurrences (phFe).
             *   Pass 2: given amp_lo, find best amp_hi for odd occurrences.
             * Accurate because the two sub-phases modify disjoint byte sets.
             */
            {
                /* build even-occurrence frequency table */
                int phFe[256]; memset(phFe, 0, sizeof phFe);
                {
                    int k = 0;
                    for (int i = phase; i < n; i += stride, k++)
                        if (!(k & 1)) phFe[data[i]]++;
                }
                /* phFo[v] = phF[v] - phFe[v]  (computed inline below) */
                int dte[256];
                for (int v = 0; v < 256; v++) dte[v] = total[v] - phFe[v];

                /* â”€â”€ DUAL_XOR: WHT for both coordinate-descent passes â”€â”€ */
                double best_S_xor;
                int best_lo_xor = xor_best_amp(dte, phFe, &best_S_xor);
                int t2x[256];
                for (int v = 0; v < 256; v++)
                    t2x[v] = dte[v] + phFe[v ^ best_lo_xor];
                /* pass 2: hlog[t2x[v] - phFo[v] + phFo[v^amp]] = hlog[a2[v] + phFo[v^amp]] */
                int phFo[256], a2[256];
                for (int v = 0; v < 256; v++) { phFo[v] = phF[v]-phFe[v]; a2[v] = t2x[v]-phFo[v]; }
                double best_S2_xor;
                int best_hi_xor = xor_best_amp(a2, phFo, &best_S2_xor);
                double net_dxor = (best_S2_xor - Sbase) - INSTR_OHD(16);
                if (net_dxor > bestNet) {
                    bestNet = net_dxor;
                    best = (Instr){DUAL_XOR, stride, phase, best_lo_xor | (best_hi_xor << 8)};
                }

                /* â”€â”€ DUAL_ADD: spike-valley proxy both passes â”€â”€ */
                {
                    double best_S_add;
                    int best_lo_add = add_best_amp(dte, phFe, 6, &best_S_add);
                    int t2a[256];
                    for (int v = 0; v < 256; v++)
                        t2a[v] = dte[v] + phFe[(v - best_lo_add) & 0xFF];
                    int phFo_p2[256], a2[256];
                    for (int v = 0; v < 256; v++) { phFo_p2[v] = phF[v]-phFe[v]; a2[v] = t2a[v]-phFo_p2[v]; }
                    double best_S2_add;
                    int best_hi_add = add_best_amp(a2, phFo_p2, 6, &best_S2_add);
                    double net_dadd = (best_S2_add - Sbase) - INSTR_OHD(16);
                    if (net_dadd > bestNet) {
                        bestNet = net_dadd;
                        best = (Instr){DUAL_ADD, stride, phase, best_lo_add | (best_hi_add << 8)};
                    }
                }

                /* â”€â”€ DUAL_MUL: brute force all strides â”€â”€ */
                {
                    int    best_lo_mul = 3; double best_S_mul  = -1e30;
                    for (int m = 3; m < 256; m += 2) {
                        double S = 0.0;
                        for (int v = 0; v < 256; v++) S += hlog[dte[v]+phFe[(v*(int)mul_inv[m])&0xFF]];
                        if (S > best_S_mul) { best_S_mul = S; best_lo_mul = m; }
                    }
                    int t2m[256];
                    for (int v = 0; v < 256; v++) t2m[v]=dte[v]+phFe[(v*(int)mul_inv[best_lo_mul])&0xFF];
                    int phFo_m[256], a2m[256];
                    for (int v = 0; v < 256; v++){phFo_m[v]=phF[v]-phFe[v];a2m[v]=t2m[v]-phFo_m[v];}
                    int    best_hi_mul = 3; double best_S2_mul = -1e30;
                    for (int m = 3; m < 256; m += 2) {
                        double S = 0.0;
                        for (int v = 0; v < 256; v++){int phFo_vm=phFo_m[(v*(int)mul_inv[m])&0xFF];S+=hlog[a2m[v]+phFo_vm];}
                        if (S > best_S2_mul) { best_S2_mul = S; best_hi_mul = m; }
                    }
                    double net_dmul = (best_S2_mul - Sbase) - INSTR_OHD(14);
                    if (net_dmul > bestNet) {
                        bestNet = net_dmul;
                        best = (Instr){DUAL_MUL, stride, phase, ((best_lo_mul-3)/2)|(((best_hi_mul-3)/2)<<7)};
                    }
                }
            }

            /* â”€â”€ TRIPLE_XOR / TRIPLE_ADD / TRIPLE_MUL: 3-way position split â”€â”€â”€â”€â”€â”€
             * Splits the (stride, phase) sequence into 3 groups by pos mod 3.
             * Overhead: 9+24=33 bits (XOR/ADD), 9+21=30 bits (MUL).
             * Replaces up to 3 individual instructions at 3Ã—17=51 or 3Ã—17=51 bits. */
            {
                int phF0[256]={0}, phF1[256]={0}, phF2[256]={0};
                { int k=0; for (int i=phase; i<n; i+=stride, k++) {
                    if      (k%3==0) phF0[data[i]]++;
                    else if (k%3==1) phF1[data[i]]++;
                    else             phF2[data[i]]++;
                }}
                int td0[256], tx1[256], dx1[256], tx2[256], dx2[256];
                for (int v=0;v<256;v++) td0[v]=total[v]-phF0[v];

                /* TRIPLE_XOR: 3 WHT passes */
                {
                    double Sx1; int ta0=xor_best_amp(td0,phF0,&Sx1);
                    for (int v=0;v<256;v++) tx1[v]=td0[v]+phF0[v^ta0];
                    for (int v=0;v<256;v++) dx1[v]=tx1[v]-phF1[v];
                    double Sx2; int ta1=xor_best_amp(dx1,phF1,&Sx2);
                    for (int v=0;v<256;v++) tx2[v]=dx1[v]+phF1[v^ta1];
                    for (int v=0;v<256;v++) dx2[v]=tx2[v]-phF2[v];
                    double Sx3; int ta2=xor_best_amp(dx2,phF2,&Sx3);
                    double ntx=(Sx3-Sbase)-INSTR_OHD(24);
                    if (ntx>bestNet) { bestNet=ntx; best=(Instr){TRIPLE_XOR,stride,phase,ta0|(ta1<<8)|(ta2<<16)}; }
                }

                /* TRIPLE_ADD: 3 cross-correlation passes */
                {
                    double Sa1; int ta0=add_best_amp(td0,phF0,6,&Sa1);
                    for (int v=0;v<256;v++) tx1[v]=td0[v]+phF0[(v-ta0)&0xFF];
                    for (int v=0;v<256;v++) dx1[v]=tx1[v]-phF1[v];
                    double Sa2; int ta1=add_best_amp(dx1,phF1,6,&Sa2);
                    for (int v=0;v<256;v++) tx2[v]=dx1[v]+phF1[(v-ta1)&0xFF];
                    for (int v=0;v<256;v++) dx2[v]=tx2[v]-phF2[v];
                    double Sa3; int ta2=add_best_amp(dx2,phF2,6,&Sa3);
                    double nta=(Sa3-Sbase)-INSTR_OHD(24);
                    if (nta>bestNet) { bestNet=nta; best=(Instr){TRIPLE_ADD,stride,phase,ta0|(ta1<<8)|(ta2<<16)}; }
                }

                /* TRIPLE_MUL: brute force 3 passes, limit stride<=16 */
                if (stride<=16) {
                    int bm0=3; double bSm0=-1e30;
                    for (int m=3;m<256;m+=2) {
                        double S=0.0;
                        for (int v=0;v<256;v++) S+=hlog[td0[v]+phF0[(v*(int)mul_inv[m])&0xFF]];
                        if (S>bSm0) { bSm0=S; bm0=m; }
                    }
                    for (int v=0;v<256;v++) tx1[v]=td0[v]+phF0[(v*(int)mul_inv[bm0])&0xFF];
                    for (int v=0;v<256;v++) dx1[v]=tx1[v]-phF1[v];
                    int bm1=3; double bSm1=-1e30;
                    for (int m=3;m<256;m+=2) {
                        double S=0.0;
                        for (int v=0;v<256;v++) S+=hlog[dx1[v]+phF1[(v*(int)mul_inv[m])&0xFF]];
                        if (S>bSm1) { bSm1=S; bm1=m; }
                    }
                    for (int v=0;v<256;v++) tx2[v]=dx1[v]+phF1[(v*(int)mul_inv[bm1])&0xFF];
                    for (int v=0;v<256;v++) dx2[v]=tx2[v]-phF2[v];
                    int bm2=3; double bSm2=-1e30;
                    for (int m=3;m<256;m+=2) {
                        double S=0.0;
                        for (int v=0;v<256;v++) S+=hlog[dx2[v]+phF2[(v*(int)mul_inv[m])&0xFF]];
                        if (S>bSm2) { bSm2=S; bm2=m; }
                    }
                    double ntm=(bSm2-Sbase)-INSTR_OHD(21);
                    if (ntm>bestNet) {
                        bestNet=ntm;
                        best=(Instr){TRIPLE_MUL,stride,phase,((bm0-3)/2)|(((bm1-3)/2)<<7)|(((bm2-3)/2)<<14)};
                    }
                }
            }

            /* ── QUAD_XOR / QUAD_ADD / QUAD_MUL: 4-way position split ──────────
             * Splits (stride, phase) into 4 groups by pos mod 4.
             * Overhead: INSTR_OHD(32)=41 (XOR/ADD), INSTR_OHD(28)=37 (MUL).
             * Replaces up to 4 XOR_PHASE at 4×17=68 bits — saves 27 bits. */
            if (!g_skip_quad) {
                int phF0[256]={0}, phF1[256]={0}, phF2[256]={0}, phF3[256]={0};
                { int k=0; for (int i=phase; i<n; i+=stride, k++) {
                    if      ((k&3)==0) phF0[data[i]]++;
                    else if ((k&3)==1) phF1[data[i]]++;
                    else if ((k&3)==2) phF2[data[i]]++;
                    else               phF3[data[i]]++;
                }}
                int td0[256], tx1[256], dx1[256], tx2[256], dx2[256], tx3[256], dx3[256];
                for (int v=0;v<256;v++) td0[v]=total[v]-phF0[v];

                /* QUAD_XOR: 4 WHT passes */
                {
                    double Sq1; int qa0=xor_best_amp(td0,phF0,&Sq1);
                    for (int v=0;v<256;v++) tx1[v]=td0[v]+phF0[v^qa0];
                    for (int v=0;v<256;v++) dx1[v]=tx1[v]-phF1[v];
                    double Sq2; int qa1=xor_best_amp(dx1,phF1,&Sq2);
                    for (int v=0;v<256;v++) tx2[v]=dx1[v]+phF1[v^qa1];
                    for (int v=0;v<256;v++) dx2[v]=tx2[v]-phF2[v];
                    double Sq3; int qa2=xor_best_amp(dx2,phF2,&Sq3);
                    for (int v=0;v<256;v++) tx3[v]=dx2[v]+phF2[v^qa2];
                    for (int v=0;v<256;v++) dx3[v]=tx3[v]-phF3[v];
                    double Sq4; int qa3=xor_best_amp(dx3,phF3,&Sq4);
                    double nqx=(Sq4-Sbase)-INSTR_OHD(32);
                    if (nqx>bestNet) {
                        bestNet=nqx;
                        best=(Instr){QUAD_XOR,stride,phase,
                            (unsigned)qa0|((unsigned)qa1<<8)|((unsigned)qa2<<16)|((unsigned)qa3<<24)};
                    }
                }

                /* QUAD_ADD: 4 cross-correlation passes */
                {
                    double Sa1; int qa0=add_best_amp(td0,phF0,6,&Sa1);
                    for (int v=0;v<256;v++) tx1[v]=td0[v]+phF0[(v-qa0)&0xFF];
                    for (int v=0;v<256;v++) dx1[v]=tx1[v]-phF1[v];
                    double Sa2; int qa1=add_best_amp(dx1,phF1,6,&Sa2);
                    for (int v=0;v<256;v++) tx2[v]=dx1[v]+phF1[(v-qa1)&0xFF];
                    for (int v=0;v<256;v++) dx2[v]=tx2[v]-phF2[v];
                    double Sa3; int qa2=add_best_amp(dx2,phF2,6,&Sa3);
                    for (int v=0;v<256;v++) tx3[v]=dx2[v]+phF2[(v-qa2)&0xFF];
                    for (int v=0;v<256;v++) dx3[v]=tx3[v]-phF3[v];
                    double Sa4; int qa3=add_best_amp(dx3,phF3,6,&Sa4);
                    double nqa=(Sa4-Sbase)-INSTR_OHD(32);
                    if (nqa>bestNet) {
                        bestNet=nqa;
                        best=(Instr){QUAD_ADD,stride,phase,
                            (unsigned)qa0|((unsigned)qa1<<8)|((unsigned)qa2<<16)|((unsigned)qa3<<24)};
                    }
                }

                /* QUAD_MUL: brute force 4 passes, stride<=8 only */
                if (stride<=8) {
                    int bm0=3; double bSm0=-1e30;
                    for (int m=3;m<256;m+=2) {
                        double S=0.0;
                        for (int v=0;v<256;v++) S+=hlog[td0[v]+phF0[(v*(int)mul_inv[m])&0xFF]];
                        if (S>bSm0) { bSm0=S; bm0=m; }
                    }
                    for (int v=0;v<256;v++) tx1[v]=td0[v]+phF0[(v*(int)mul_inv[bm0])&0xFF];
                    for (int v=0;v<256;v++) dx1[v]=tx1[v]-phF1[v];
                    int bm1=3; double bSm1=-1e30;
                    for (int m=3;m<256;m+=2) {
                        double S=0.0; for (int v=0;v<256;v++) S+=hlog[dx1[v]+phF1[(v*(int)mul_inv[m])&0xFF]];
                        if (S>bSm1) { bSm1=S; bm1=m; }
                    }
                    for (int v=0;v<256;v++) tx2[v]=dx1[v]+phF1[(v*(int)mul_inv[bm1])&0xFF];
                    for (int v=0;v<256;v++) dx2[v]=tx2[v]-phF2[v];
                    int bm2=3; double bSm2=-1e30;
                    for (int m=3;m<256;m+=2) {
                        double S=0.0; for (int v=0;v<256;v++) S+=hlog[dx2[v]+phF2[(v*(int)mul_inv[m])&0xFF]];
                        if (S>bSm2) { bSm2=S; bm2=m; }
                    }
                    for (int v=0;v<256;v++) tx3[v]=dx2[v]+phF2[(v*(int)mul_inv[bm2])&0xFF];
                    for (int v=0;v<256;v++) dx3[v]=tx3[v]-phF3[v];
                    int bm3=3; double bSm3=-1e30;
                    for (int m=3;m<256;m+=2) {
                        double S=0.0; for (int v=0;v<256;v++) S+=hlog[dx3[v]+phF3[(v*(int)mul_inv[m])&0xFF]];
                        if (S>bSm3) { bSm3=S; bm3=m; }
                    }
                    double nqm=(bSm3-Sbase)-INSTR_OHD(28);
                    if (nqm>bestNet) {
                        bestNet=nqm;
                        best=(Instr){QUAD_MUL,stride,phase,
                            (unsigned)((bm0-3)/2)|(unsigned)(((bm1-3)/2)<<7)|(unsigned)(((bm2-3)/2)<<14)|(unsigned)(((bm3-3)/2)<<21)};
                    }
                }
            }
        }
    }


    if (netOut) *netOut = bestNet;
    return best;
}

/* â”€â”€ scramble helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
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
        dst[i]        = (u8)((src[2*i] & 0x0F) | ((src[2*i+1] & 0x0F) << 4));
        dst[i + half] = (u8)((src[2*i] >>   4) | ((src[2*i+1] >>   4) << 4));
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

/* â”€â”€ apply scramble type si to dst from src â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
static int apply_scramble(int si, const u8 *src, u8 *dst, int n) {
    if      (si == 0 && n%2==0)    { interleave_stride(src, dst, n, 2);    return 1; }
    else if (si == 1 && n%4==0)    { interleave_stride(src, dst, n, 4);    return 1; }
    else if (si == 2 && n%8==0)    { interleave_stride(src, dst, n, 8);    return 1; }
    else if (si == 3 && n%16==0)   { interleave_stride(src, dst, n, 16);   return 1; }
    else if (si == 4 && n%32==0)   { interleave_stride(src, dst, n, 32);   return 1; }
    else if (si == 5 && n%64==0)   { interleave_stride(src, dst, n, 64);   return 1; }
    else if (si == 6 && n%8==0)    { bit_plane_sep(src, dst, n);            return 1; }
    else if (si == 7 && n%2==0)    { nib_plane_sep(src, dst, n);            return 1; }
    else if (si == 8)              { block_reverse(src, dst, n);            return 1; }
    else if (si == 9 && n%2==0)    { xor_fold_scramble(src, dst, n);       return 1; }
    /* inverse interleaves: inv of IL-s = interleave at stride n/s */
    else if (si == 10 && n%1024==0){ interleave_stride(src, dst, n, 1024); return 1; }  /* inv IL4  */
    else if (si == 11 && n%512==0) { interleave_stride(src, dst, n, 512);  return 1; }  /* inv IL8  */
    else if (si == 12 && n%256==0) { interleave_stride(src, dst, n, 256);  return 1; }  /* inv IL16 */
    else if (si == 13 && n%128==0) { interleave_stride(src, dst, n, 128);  return 1; }  /* inv IL32 */
    return 0;
}

/* â”€â”€ decompressor helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

/* Apply the inverse of scramble si in-place. */
static void invert_scramble(u8 *data, int n, int si) {
    u8 tmp[BLOCK_SIZE];
    /* Inverse map: IL-s â†’ IL-(n/s); bit_plane_sep, block_reverse, xor_fold self-inverse */
    int inv_si = -1;
    switch (si) {
        case 1:  inv_si=10; break;
        case 2:  inv_si=11; break;
        case 3:  inv_si=12; break;
        case 4:  inv_si=13; break;
        case 5:  inv_si=5;  break;
        case 6:  inv_si=6;  break;  /* bit_plane_sep is an involution */
        case 8:  inv_si=8;  break;  /* block_reverse is self-inverse */
        case 9:  inv_si=9;  break;  /* xor_fold is self-inverse */
        case 10: inv_si=1;  break;
        case 11: inv_si=2;  break;
        case 12: inv_si=3;  break;
        case 13: inv_si=4;  break;
    }
    if (si == 0) {
        /* IL-2 inverse = IL-(n/2) */
        interleave_stride(data, tmp, n, n/2);
        memcpy(data, tmp, n);
    } else if (si == 7) {
        /* nib_plane_sep inverse: recombine lo/hi nibble planes */
        int half = n/2;
        for (int i = 0; i < half; i++) {
            tmp[2*i]   = (u8)((data[i] & 0xF) | ((data[i+half] & 0xF) << 4));
            tmp[2*i+1] = (u8)(((data[i]>>4) & 0xF) | (((data[i+half]>>4) & 0xF) << 4));
        }
        memcpy(data, tmp, n);
    } else if (inv_si >= 0) {
        apply_scramble(inv_si, data, tmp, n);
        memcpy(data, tmp, n);
    }
}

/* Decompress: process instruction list in REVERSE ORDER, recovering each amp
 * by running the same selection algorithm on the current (partially undone) data.
 * This works because undoing later instructions first restores the exact data state
 * the compressor had when it applied instruction i, so canonical_amp() agrees. */
static int decompress(u8 *data, int n, const Instr *instrs, int ni) {
    for (int ii = ni - 1; ii >= 0; ii--) {
        Instr t = instrs[ii];
        switch (t.type) {
        case SCRAMBLE:
            invert_scramble(data, n, t.amp & 0xF); break;
        case COND_PREV_XOR: {
            int alo = t.amp & 0xFF, ahi = (t.amp >> 8) & 0xFF;
            int last_ii = t.phase;
            for (int ii = t.phase; ii < n; ii += t.stride) last_ii = ii;
            for (int ii = last_ii; ii >= t.phase + t.stride; ii -= t.stride) {
                int prev_val = data[ii - t.stride];
                data[ii] ^= (u8)((prev_val >= 128) ? ahi : alo);
            }
            break;
        }
        case POLY_DELTA_XOR: {
            int s1 = t.stride, s2 = t.phase;
            for (int i = s2; i < n; i++) data[i] ^= data[i-s1] ^ data[i-s2]; break;
        }
        case XOR_PHASE:
            applyInstr(data, n, t); break;  /* XOR is self-inverse */
        case ADD_PHASE:
            applyInstr(data, n, (Instr){ADD_PHASE, t.stride, t.phase, (256 - t.amp) & 0xFF}); break;
        case PACK_XOR:
            applyInstr(data, n, t); break;  /* self-inverse */
        case LAG_XOR: {
            int lag = (int)t.amp;
            for (int ii = n - lag - 1; ii >= 0; ii--)
                data[ii] ^= data[ii + lag];
            break;
        }
        case ADD_NIBS: {
            int lo = t.amp & 0xF, hi = (t.amp >> 4) & 0xF;
            applyInstr(data, n, (Instr){ADD_NIBS, t.stride, t.phase, (((-hi)&0xF)<<4)|((-lo)&0xF)}); break;
        }
        case NIB_SWAP:
            applyInstr(data, n, t); break;  /* self-inverse */
        case BIT_ROTATE: {
            int k = t.amp & 7;
            applyInstr(data, n, (Instr){BIT_ROTATE, t.stride, t.phase, (8 - k) & 7}); break;
        }
        case VALUE_XOR:
            applyInstr(data, n, t); break;  /* XOR self-inverse, amps preserved */
        case COND_LO_XOR:
            applyInstr(data, n, t); break;  /* XOR self-inverse, same condition */
        case COND_HI_XOR:
            applyInstr(data, n, t); break;
        case COND_LO_ADD: {
            int nc = (t.amp >> 4) & 0xF, av = t.amp & 0xF;
            applyInstr(data, n, (Instr){COND_LO_ADD, t.stride, t.phase, (nc<<4)|((-av)&0xF)}); break;
        }
        case COND_HI_ADD: {
            int nc = t.amp & 0xF, av = (t.amp >> 4) & 0xF;
            applyInstr(data, n, (Instr){COND_HI_ADD, t.stride, t.phase, nc|((-av&0xF)<<4)}); break;
        }
        case DUAL_XOR:
            applyInstr(data, n, t); break;  /* XOR self-inverse */
        case DUAL_ADD: {
            int lo = t.amp & 0xFF, hi = (t.amp >> 8) & 0xFF;
            applyInstr(data, n, (Instr){DUAL_ADD, t.stride, t.phase,
                ((256-lo)&0xFF) | (((256-hi)&0xFF)<<8)}); break;
        }
        case DUAL_MUL: {
            int idx_lo = t.amp & 0x7F, idx_hi = (t.amp >> 7) & 0x7F;
            int m_lo = idx_lo * 2 + 3, m_hi = idx_hi * 2 + 3;
            int inv_lo = (mul_inv[m_lo] - 3) / 2, inv_hi = (mul_inv[m_hi] - 3) / 2;
            applyInstr(data, n, (Instr){DUAL_MUL, t.stride, t.phase, inv_lo | (inv_hi << 7)}); break;
        }
        case TRIPLE_XOR:
            applyInstr(data, n, t); break;  /* XOR is self-inverse */
        case TRIPLE_ADD: {
            int a0=t.amp&0xFF, a1=(t.amp>>8)&0xFF, a2=(t.amp>>16)&0xFF;
            applyInstr(data, n, (Instr){TRIPLE_ADD, t.stride, t.phase,
                ((256-a0)&0xFF)|(((256-a1)&0xFF)<<8)|(((256-a2)&0xFF)<<16)}); break;
        }
        case TRIPLE_MUL: {
            int idx0=t.amp&0x7F, idx1=(t.amp>>7)&0x7F, idx2=(t.amp>>14)&0x7F;
            int m0=idx0*2+3, m1=idx1*2+3, m2=idx2*2+3;
            int inv0=(mul_inv[m0]-3)/2, inv1=(mul_inv[m1]-3)/2, inv2=(mul_inv[m2]-3)/2;
            applyInstr(data, n, (Instr){TRIPLE_MUL, t.stride, t.phase,
                (unsigned)inv0|((unsigned)inv1<<7)|((unsigned)inv2<<14)}); break;
        }
        case QUAD_XOR:
            applyInstr(data, n, t); break;  /* XOR self-inverse */
        case QUAD_ADD: {
            unsigned a0=t.amp&0xFF, a1=(t.amp>>8)&0xFF, a2=(t.amp>>16)&0xFF, a3=t.amp>>24;
            applyInstr(data, n, (Instr){QUAD_ADD, t.stride, t.phase,
                ((256-a0)&0xFF)|(((256-a1)&0xFF)<<8)|(((256-a2)&0xFF)<<16)|(((256-a3)&0xFF)<<24)}); break;
        }
        case QUAD_MUL: {
            int idx0=t.amp&0x7F, idx1=(t.amp>>7)&0x7F, idx2=(t.amp>>14)&0x7F, idx3=(t.amp>>21)&0x7F;
            int m0=idx0*2+3, m1=idx1*2+3, m2=idx2*2+3, m3=idx3*2+3;
            int inv0=(mul_inv[m0]-3)/2, inv1=(mul_inv[m1]-3)/2;
            int inv2=(mul_inv[m2]-3)/2, inv3=(mul_inv[m3]-3)/2;
            applyInstr(data, n, (Instr){QUAD_MUL, t.stride, t.phase,
                (unsigned)inv0|((unsigned)inv1<<7)|((unsigned)inv2<<14)|((unsigned)inv3<<21)}); break;
        }
        default: break;
        }
    }
    return 1;
}

/* â”€â”€ greedy compress â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */

/* Global instruction list for current block (reset at start of compress). */
static Instr g_ilist[4096];
static int   g_ni;

/* Last-resort scramble: called when greedy is fully stuck.
 * For each of the 14 scramble types, runs the full greedy search to convergence
 * on the scrambled data to measure total unlocked net. Applies the best one if
 * the total gain (scramble entropy delta + unlocked net - scramble overhead) > 0. */
static int try_scramble(u8 *data, int n, double *total_net, int *counts, int verbose) {
    int sc_freq[256] = {0};
    for (int i = 0; i < n; i++) sc_freq[data[i]]++;
    double Sbase = 0.0;
    for (int v = 0; v < 256; v++) Sbase += hlog[sc_freq[v]];

    static u8 scbuf[BLOCK_SIZE], tmpwork[BLOCK_SIZE];
    int    best_si     = -1;
    double best_gain   = 0.0;
    double best_edelta = 0.0;

    g_skip_quad = 1;
    for (int si = 0; si < 14; si++) {
        if (!apply_scramble(si, data, scbuf, n)) continue;

        int sc_tot[256] = {0};
        for (int i = 0; i < n; i++) sc_tot[scbuf[i]]++;
        double sc_Sb = 0.0;
        for (int v = 0; v < 256; v++) sc_Sb += hlog[sc_tot[v]];
        double edelta = sc_Sb - Sbase;

        /* quick probe: 5 iterations captures enough signal to rank scrambles */
        memcpy(tmpwork, scbuf, n);
        double temp_net = 0.0;
        for (int iter = 0; iter < 5; iter++) {
            double net;
            Instr t2 = findBest(tmpwork, n, &net, 64);
            if (net <= 0.0) break;
            applyInstr(tmpwork, n, t2);
            temp_net += net;
        }

        double gain = edelta + temp_net - INSTR_OHD_NOPHASE(4);
        if (gain > best_gain) { best_gain = gain; best_si = si; best_edelta = edelta; }
    }
    g_skip_quad = 0;

    if (best_si < 0) return 0;

    double e0 = entropy(data, n);
    applyInstr(data, n, (Instr){SCRAMBLE, 0, 0, best_si});
    g_ilist[g_ni++] = (Instr){SCRAMBLE, 0, 0, best_si};
    *total_net += best_edelta - INSTR_OHD_NOPHASE(4);
    counts[SCRAMBLE]++;
    if (verbose)
        printf("  %-14s si=%-7d %.6f -> %.6f  net=%.1f\n",
               "SCRAMBLE", best_si, e0, entropy(data, n), best_edelta - INSTR_OHD_NOPHASE(4));
    return 1;
}

static double compress(u8 *data, int n, int verbose, int *counts) {
    double total_net = 0.0;
    g_ni = 0;

    for (;;) {
        /* phase 1: regular greedy, no SCRAMBLE */
        for (;;) {
            double net;
            Instr t = findBest(data, n, &net, 64);
            if (net <= 0.0) break;
            double e0 = entropy(data, n);
            applyInstr(data, n, t);
            g_ilist[g_ni++] = t;
            total_net += net;
            counts[t.type]++;
            if (verbose)
                printf("  %-14s s%-2d p%-2d a%-8u  %.6f -> %.6f  net=%.1f\n",
                       INSTR_NAMES[t.type], t.stride, t.phase, t.amp,
                       e0, entropy(data, n), net);
        }

        /* last resort: try every scramble, pick whichever unlocks the most */
        if (!try_scramble(data, n, &total_net, counts, verbose)) break;
    }
    return total_net;
}

/* â”€â”€ main â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€ */
int main(void) {
    init_hlog();
    init_mul_inv();
    init_fft256();
    int counts[NUM_INSTR_TYPES] = {0};
    double sum = 0.0;

    /* load or generate seed data â€” seed.bin ensures reproducible runs */
    static u8 alldata[NUM_BLOCKS * BLOCK_SIZE];
    {
        FILE *sf = fopen("seed.bin", "rb");
        int loaded = 0;
        if (sf) {
            size_t got = fread(alldata, 1, sizeof alldata, sf);
            fclose(sf);
            if (got == sizeof alldata) {
                printf("loaded seed.bin (%d bytes)\n", (int)sizeof alldata);
                loaded = 1;
            } else {
                printf("seed.bin wrong size (%d/%d bytes), regenerating\n",
                       (int)got, (int)sizeof alldata);
            }
        }
        if (!loaded) {
            for (int b = 0; b < NUM_BLOCKS; b++)
                fill_random(alldata + b * BLOCK_SIZE, BLOCK_SIZE);
            sf = fopen("seed.bin", "wb");
            if (sf) { fwrite(alldata, 1, sizeof alldata, sf); fclose(sf); }
            printf("generated and saved seed.bin (%d bytes)\n", (int)sizeof alldata);
        }
    }

    int decomp_fails = 0;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        u8 *data = malloc(BLOCK_SIZE);
        u8 *orig = malloc(BLOCK_SIZE);
        memcpy(data, alldata + b * BLOCK_SIZE, BLOCK_SIZE);
        memcpy(orig, data, BLOCK_SIZE);
        double e0 = entropy(data, BLOCK_SIZE);
        double net = compress(data, BLOCK_SIZE, /*verbose=*/ b == 0, counts);
        double e1 = entropy(data, BLOCK_SIZE);
        printf("block %2d: %.6f -> %.6f  net=%.1f bits  ni=%d  verifying...",
               b, e0, e1, net, g_ni);
        fflush(stdout);
        sum += net;
        /* verify decompressor recovers original */
        u8 *dcmp = malloc(BLOCK_SIZE);
        memcpy(dcmp, data, BLOCK_SIZE);
        decompress(dcmp, BLOCK_SIZE, g_ilist, g_ni);
        if (memcmp(dcmp, orig, BLOCK_SIZE) != 0) {
            printf(" FAIL\n  *** DECOMP FAIL block %d (ni=%d) ***\n", b, g_ni);
            decomp_fails++;
        } else {
            printf(" ok\n");
        }
        fflush(stdout);
        free(dcmp); free(orig); free(data);
    }
    if (decomp_fails == 0)
        printf("decompressor verified OK on all %d blocks\n", NUM_BLOCKS);

    printf("\navg net: %.1f bits/block over %d blocks\n", sum / NUM_BLOCKS, NUM_BLOCKS);
    printf("\ninstruction usage counts:\n");
    for (int i = 0; i < NUM_INSTR_TYPES; i++)
        printf("  %-14s %d\n", INSTR_NAMES[i], counts[i]);

    return 0;
}
