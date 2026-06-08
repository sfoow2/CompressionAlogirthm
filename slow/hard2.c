/*
 * simple.c — greedy entropy compressor
 *
 * All instructions use the frequency-table trick:
 *   new_freq[v] = (total[v]-phF[v]) + phF[inverse(v, amp)]
 * evaluating ALL amp values in O(256) per (stride, phase) pair.
 *
 * DUAL_XOR / DUAL_ADD split a (stride, phase) into even/odd occurrences
 * and apply independent amps to each — same pattern as two XOR_PHASE calls
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

/* Overhead model: every instruction costs INSTR_BASE (type+stride+phase) + amp bits.
 * Change INSTR_BASE to experiment; all per-type overheads update automatically.
 * Amp bits by type: single(8), dual_xor/add(16), dual_mul(14), nib_swap(0), scramble(4).
 * DELTA/POLY_DELTA have no phase/amp — use separate constants. */
#define INSTR_BASE     9
#define INSTR_OHD(ab) ((double)(INSTR_BASE + (ab)))

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

/* ── data ────────────────────────────────────────────────────────────────── */
static void fill_random(u8 *buf, int n) {
    BCryptGenRandom(NULL, buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}

/* ── multiply-inverse table mod 256 ─────────────────────────────────────── */
static u8 mul_inv[256];
static void init_mul_inv(void) {
    for (int k = 1; k < 256; k += 2)
        for (int inv = 1; inv < 256; inv += 2)
            if (((k * inv) & 0xFF) == 1) { mul_inv[k] = (u8)inv; break; }
}

/* ── GF(2^8) multiply using AES polynomial (x^8+x^4+x^3+x+1 = 0x11B) ─── */
static u8 gf_mul_tab[256][256];
static u8 gf_inv_tab[256];
static void init_gf_tables(void) {
    for (int a = 0; a < 256; a++) {
        u8 p = 0, aa = (u8)a;
        for (int b = 0; b < 256; b++) {
            /* carry-less multiply a * b mod 0x11B */
            u8 pp = 0, tmp = aa;
            for (int i = 0; i < 8; i++) {
                if ((b >> i) & 1) pp ^= tmp;
                int hi = tmp & 0x80;
                tmp = (u8)(tmp << 1);
                if (hi) tmp ^= 0x1B;
            }
            gf_mul_tab[a][b] = pp;
            (void)p;
        }
    }
    gf_inv_tab[0] = 0;
    for (int a = 1; a < 256; a++)
        for (int b = 1; b < 256; b++)
            if (gf_mul_tab[a][b] == 1) { gf_inv_tab[a] = (u8)b; break; }
}

/* ── instructions ────────────────────────────────────────────────────────── */

typedef enum {
    /* 17-bit overhead (9 base + 8-bit amp) */
    XOR_PHASE = 0,  /* data[i] ^= amp                                          */
    ADD_PHASE,      /* data[i] += amp  (mod 256)                               */
    MUL_ODD,        /* data[i] *= amp  (amp odd, invertible mod 256)           */
    COND_LO_XOR,    /* if hi-nib==(amp>>4): lo-nib ^= (amp&0xF)               */
    ADD_NIBS,       /* lo-nib += (amp&0xF), hi-nib += (amp>>4), no carry       */
    COND_HI_XOR,    /* if lo-nib==(amp&0xF): hi-nib ^= ((amp>>4)<<4)          */
    COND_LO_ADD,    /* if hi-nib==(amp>>4): lo-nib += (amp&0xF) mod 16        */
    COND_HI_ADD,    /* if lo-nib==(amp&0xF): hi-nib += (amp>>4) mod 16        */
    /* 9-bit overhead */
    NIB_SWAP,       /* data[i] = (data[i]<<4)|(data[i]>>4), self-inverse       */
    /* 12-bit overhead (9 base + 3-bit k) */
    BIT_ROTATE,     /* data[i] = rotl(data[i], k); inverse is rotl(_, 8-k)    */
    /* 23-bit overhead (9 base + 7+7 bits; amps < 128 so MSB preserved) */
    VALUE_XOR,      /* lo-half ^= (amp&0x7F); hi-half ^= ((amp>>7)&0x7F)      */
    /* 25-bit overhead */
    DUAL_XOR,       /* even ^= (amp&0xFF), odd ^= ((amp>>8)&0xFF)              */
    DUAL_ADD,       /* even += (amp&0xFF), odd += ((amp>>8)&0xFF)              */
    /* 23-bit overhead */
    DUAL_MUL,       /* even *= m_lo, odd *= m_hi; amp = idx_lo|(idx_hi<<7)    */
    /* 17-bit overhead */
    GF_MUL,         /* data[i] = gf_mul(data[i], amp) in GF(2^8) AES field    */
    /* 10-bit overhead (no phase, no amp) */
    DELTA_SUB,      /* data[i] -= data[i-stride] for all i>=stride             */
    /* 16-bit overhead (two strides, no phase, no amp) */
    POLY_DELTA_XOR, /* data[i] ^= data[i-s1]^data[i-s2] for i>=s2             */
    /* 13-bit overhead */
    SCRAMBLE,       /* position rearrangement; amp encodes scramble type        */
    /* 33-bit overhead (9 base + 3×8 amp) */
    TRIPLE_XOR,     /* group0 ^= a0, group1 ^= a1, group2 ^= a2 (pos mod 3)   */
    TRIPLE_ADD,     /* group0 += a0, group1 += a1, group2 += a2 (pos mod 3)   */
    /* 30-bit overhead (9 base + 3×7 amp) */
    TRIPLE_MUL,     /* group0 *= m0, group1 *= m1, group2 *= m2; idx per 7b   */
    NUM_INSTR_TYPES /* 19 total */
} InstrType;

static const char *INSTR_NAMES[NUM_INSTR_TYPES] = {
    "XOR_PHASE", "ADD_PHASE", "MUL_ODD", "COND_LO_XOR", "ADD_NIBS",
    "COND_HI_XOR", "COND_LO_ADD", "COND_HI_ADD", "NIB_SWAP",
    "BIT_ROTATE", "VALUE_XOR",
    "DUAL_XOR", "DUAL_ADD", "DUAL_MUL", "GF_MUL",
    "DELTA_SUB", "POLY_DELTA_XOR", "SCRAMBLE",
    "TRIPLE_XOR", "TRIPLE_ADD", "TRIPLE_MUL"
};

typedef struct { InstrType type; int stride, phase, amp; } Instr;

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
        case MUL_ODD:
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)((data[i] * (u8)t.amp) & 0xFF);
            break;
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
            int alo = t.amp & 0x7F, ahi = (t.amp >> 7) & 0x7F;
            for (i = t.phase; i < n; i += t.stride)
                data[i] ^= (u8)(data[i] & 0x80 ? ahi : alo);
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
        case GF_MUL:
            for (i = t.phase; i < n; i += t.stride)
                data[i] = gf_mul_tab[data[i]][t.amp];
            break;
        case DELTA_SUB:
            for (i = n - 1; i >= t.stride; i--)
                data[i] = (u8)((data[i] - data[i - t.stride]) & 0xFF);
            break;
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
        default: break;
    }
}

/* ── Walsh-Hadamard helpers for fast XOR-amp search ─────────────────────── */

/* In-place Walsh-Hadamard Transform, n=256 */
static void wht256(int *a) {
    for (int len = 1; len < 256; len <<= 1)
        for (int i = 0; i < 256; i += len << 1)
            for (int j = 0; j < len; j++) {
                int u = a[i+j], v = a[i+j+len];
                a[i+j] = u+v; a[i+j+len] = u-v;
            }
}

/* Find best XOR amp (1..255) for sum_v hlog[A[v] + B[v^amp]].
 * Uses WHT XOR-correlation to find top-3 candidates, then exact verify.
 * Returns best amp; stores exact S in *Sout. */
static int xor_best_amp(const int *A, const int *B, double *Sout) {
    int ha[256], hb[256];
    for (int i = 0; i < 256; i++) { ha[i] = A[i]; hb[i] = B[i]; }
    wht256(ha); wht256(hb);
    /* pointwise product → IWHT gives XOR-correlation × 256 */
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

/* ── fast amp-search helpers ─────────────────────────────────────────────── */

/* Best cyclic-ADD amp via double-buffer sequential cross-correlation.
 * Stores phF twice so each rotation is a stride-1 slice → auto-vectorises.
 * Top-8 peaks of C[k] = Σ dv[v]*phF[(v-k)] verified exactly with hlog.
 * Reliable for near-uniform BCrypt data where spike-valley proxy fails. */
static int add_best_amp(const int *dv, const int *phF, int _K, double *Sout) {
    (void)_K;
    /* double-buffer: phF2[v] = phF2[v+256] = phF[v]  (512 ints, 2 KB) */
    int phF2[512];
    for (int v = 0; v < 256; v++) phF2[v] = phF2[v + 256] = phF[v];
    /* C[k] = dot(dv, phF2[256-k .. 511-k])  —  stride-1 access, vectorisable */
    int corr[256];
    for (int k = 1; k < 256; k++) {
        const int *p = phF2 + (256 - k);
        int c = 0;
        for (int v = 0; v < 256; v++) c += dv[v] * p[v];
        corr[k] = c;
    }
    /* find top-8 peaks */
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
 * Coarse: 8 amps at every 32nd odd (3,35,67,...,227). Fine: ±16 odd steps around top-3.
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
    /* Level 2: brute force ±16 odd steps (±32 in value) around each top-3 */
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

/* ── find best instruction ───────────────────────────────────────────────── */
static Instr findBest(const u8 *data, int n, double *netOut, int max_stride) {
    int total[256] = {0};
    for (int i = 0; i < n; i++) total[data[i]]++;
    double hlt[256];
    double Sbase = 0.0;
    for (int v = 0; v < 256; v++) { hlt[v] = hlog[total[v]]; Sbase += hlt[v]; }

    double bestNet = 0.0;
    Instr  best    = {XOR_PHASE, 2, 0, 1};

    int phF[256];

    for (int stride = 1; stride <= max_stride; stride++) {
        /* ── DELTA_SUB: data[i] -= data[i-stride], no phase/amp ─────────── */
        {
            int dsfreq[256] = {0};
            for (int i = 0; i < stride && i < n; i++) dsfreq[data[i]]++;
            for (int i = stride; i < n; i++) dsfreq[(data[i] - data[i - stride]) & 0xFF]++;
            double Sds = 0.0;
            for (int v = 0; v < 256; v++) Sds += hlog[dsfreq[v]];
            double nds = (Sds - Sbase) - 10.0;
            if (nds > bestNet) { bestNet = nds; best = (Instr){DELTA_SUB, stride, 0, 0}; }
        }
        /* ── POLY_DELTA_XOR: limit to stride<=32 to keep search fast ───── */
        if (stride <= 32) {
        for (int s1 = 1; s1 < stride; s1++) {
            int pdfreq[256] = {0};
            for (int i = 0; i < stride && i < n; i++) pdfreq[data[i]]++;
            for (int i = stride; i < n; i++) pdfreq[data[i] ^ data[i - s1] ^ data[i - stride]]++;
            double Spd = 0.0;
            for (int v = 0; v < 256; v++) Spd += hlog[pdfreq[v]];
            double npd = (Spd - Sbase) - 16.0;
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

            /* ── XOR_PHASE: WHT proxy + top-3 exact verify ──────────────── */
            {
                double Sx;
                int ax = xor_best_amp(dv, phF, &Sx);
                double nx = (Sx - Sbase) - INSTR_OHD(8);
                if (nx > bestNet) { bestNet = nx; best = (Instr){XOR_PHASE, stride, phase, ax}; }
            }
            /* ── ADD_PHASE: spike-valley proxy K=6 ──────────────────────── */
            {
                double Sa; int aa = add_best_amp(dv, phF, 6, &Sa);
                double na = (Sa - Sbase) - INSTR_OHD(8);
                if (na > bestNet) { bestNet = na; best = (Instr){ADD_PHASE, stride, phase, aa}; }
            }

            /* ── MUL_ODD: brute force for stride<=32 ────────────────────── */
            if (stride <= 32) {
                for (int amp = 3; amp < 256; amp += 2) {
                    double Sm = 0.0;
                    for (int v = 0; v < 256; v++) Sm += hlog[dv[v] + phF[(v*(int)mul_inv[amp])&0xFF]];
                    double nm = (Sm - Sbase) - INSTR_OHD(8);
                    if (nm > bestNet) { bestNet = nm; best = (Instr){MUL_ODD, stride, phase, amp}; }
                }
            }
            /* ── GF_MUL: stride<=8 only to keep overhead manageable ────── */
            if (stride <= 8) {
            for (int amp = 1; amp < 256; amp++) {
                u8 inv_amp = gf_inv_tab[amp];
                double Sgf = 0.0;
                for (int v = 0; v < 256; v++)
                    Sgf += hlog[dv[v] + phF[gf_mul_tab[v][inv_amp]]];
                double ngf = (Sgf - Sbase) - INSTR_OHD(8);
                if (ngf > bestNet) { bestNet = ngf; best = (Instr){GF_MUL, stride, phase, amp}; }
            }
            }

            /* ── COND_LO_XOR: stride<=32 ────────────────────────────────── */
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

            /* ── ADD_NIBS: factored nibble proxy ─────────────────────────── */
            {
                double Sn; int an = add_nibs_best_amp(dv, phF, &Sn);
                double nn = (Sn - Sbase) - INSTR_OHD(8);
                if (nn > bestNet) { bestNet = nn; best = (Instr){ADD_NIBS, stride, phase, an}; }
            }

            /* ── NIB_SWAP: self-inverse, cost 8 bits ─────────────────────── */
            {
                double Ssw = 0.0;
                for (int v = 0; v < 256; v++) {
                    int sv = ((v << 4) | (v >> 4)) & 0xFF;
                    Ssw += hlog[dv[v] + phF[sv]];
                }
                double nsw = (Ssw - Sbase) - INSTR_OHD(0);
                if (nsw > bestNet) { bestNet = nsw; best = (Instr){NIB_SWAP, stride, phase, 0}; }
            }

            /* ── BIT_ROTATE: rotate bits of each byte by k=1..7 ─────────── */
            for (int k = 1; k <= 7; k++) {
                double Sbr = 0.0;
                for (int v = 0; v < 256; v++) {
                    int rv = ((v >> k) | (v << (8 - k))) & 0xFF; /* rotr(v,k) = original before rotl(k) */
                    Sbr += hlog[dv[v] + phF[rv]];
                }
                double nbr = (Sbr - Sbase) - INSTR_OHD(3);
                if (nbr > bestNet) { bestNet = nbr; best = (Instr){BIT_ROTATE, stride, phase, k}; }
            }

            /* ── VALUE_XOR: split by MSB, independent XOR amps (both <128) ─
             * Groups are disjoint under XOR with amp<128, so score is separable.
             * Best amp_lo maximises Σ_{v<128} hlog[dv[v]+phF[v^a]]; similarly hi. */
            {
                int balo = 0; double bSlo = -1e30;
                for (int a = 1; a < 128; a++) {
                    double S = 0.0;
                    for (int v = 0; v < 128; v++) S += hlog[dv[v] + phF[v ^ a]];
                    if (S > bSlo) { bSlo = S; balo = a; }
                }
                int bahi = 0; double bShi = -1e30;
                for (int a = 1; a < 128; a++) {
                    double S = 0.0;
                    for (int v = 128; v < 256; v++) S += hlog[dv[v] + phF[v ^ a]];
                    if (S > bShi) { bShi = S; bahi = a; }
                }
                /* full score: sum both halves plus the unchanged complement */
                double Svx = 0.0;
                for (int v = 0; v < 256; v++) {
                    int vx = (v < 128) ? (v ^ balo) : (v ^ bahi);
                    Svx += hlog[dv[v] + phF[vx]];
                }
                double nvx = (Svx - Sbase) - INSTR_OHD(14);
                if (nvx > bestNet) { bestNet = nvx; best = (Instr){VALUE_XOR, stride, phase, balo | (bahi << 7)}; }
            }

            /* ── COND_HI_XOR / COND_LO_ADD / COND_HI_ADD: stride<=32 ──── */
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
             * ── DUAL_XOR and DUAL_ADD ─────────────────────────────────────
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

                /* ── DUAL_XOR: WHT for both coordinate-descent passes ── */
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

                /* ── DUAL_ADD: spike-valley proxy both passes ── */
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

                /* ── DUAL_MUL: brute force all strides ── */
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

            /* ── TRIPLE_XOR / TRIPLE_ADD / TRIPLE_MUL: 3-way position split ──────
             * Splits the (stride, phase) sequence into 3 groups by pos mod 3.
             * Overhead: 9+24=33 bits (XOR/ADD), 9+21=30 bits (MUL).
             * Replaces up to 3 individual instructions at 3×17=51 or 3×17=51 bits. */
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
        }
    }


    if (netOut) *netOut = bestNet;
    return best;
}

/* ── scramble helpers ────────────────────────────────────────────────────── */
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

/* ── apply scramble type si to dst from src ─────────────────────────────── */
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

/* ── decompressor helpers ────────────────────────────────────────────────── */

/* Apply the inverse of scramble si in-place. */
static void invert_scramble(u8 *data, int n, int si) {
    u8 tmp[BLOCK_SIZE];
    /* Inverse map: IL-s → IL-(n/s); bit_plane_sep, block_reverse, xor_fold self-inverse */
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
        case DELTA_SUB:
            for (int i = t.stride; i < n; i++) data[i] = (u8)((data[i] + data[i - t.stride]) & 0xFF); break;
        case POLY_DELTA_XOR: {
            int s1 = t.stride, s2 = t.phase;
            for (int i = s2; i < n; i++) data[i] ^= data[i-s1] ^ data[i-s2]; break;
        }
        case XOR_PHASE:
            applyInstr(data, n, t); break;  /* XOR is self-inverse */
        case ADD_PHASE:
            applyInstr(data, n, (Instr){ADD_PHASE, t.stride, t.phase, (256 - t.amp) & 0xFF}); break;
        case MUL_ODD:
            applyInstr(data, n, (Instr){MUL_ODD, t.stride, t.phase, mul_inv[t.amp]}); break;
        case GF_MUL:
            applyInstr(data, n, (Instr){GF_MUL, t.stride, t.phase, gf_inv_tab[t.amp]}); break;
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
                inv0|(inv1<<7)|(inv2<<14)}); break;
        }
        default: break;
        }
    }
    return 1;
}

/* ── greedy compress ─────────────────────────────────────────────────────── */

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

    for (int si = 0; si < 14; si++) {
        if (!apply_scramble(si, data, scbuf, n)) continue;

        int sc_tot[256] = {0};
        for (int i = 0; i < n; i++) sc_tot[scbuf[i]]++;
        double sc_Sb = 0.0;
        for (int v = 0; v < 256; v++) sc_Sb += hlog[sc_tot[v]];
        double edelta = sc_Sb - Sbase;

        /* run greedy to convergence on the scrambled copy */
        memcpy(tmpwork, scbuf, n);
        double temp_net = 0.0;
        for (;;) {
            double net;
            Instr t2 = findBest(tmpwork, n, &net, 64);
            if (net <= 0.0) break;
            applyInstr(tmpwork, n, t2);
            temp_net += net;
        }

        double gain = edelta + temp_net - INSTR_OHD(4);
        if (gain > best_gain) { best_gain = gain; best_si = si; best_edelta = edelta; }
    }

    if (best_si < 0) return 0;

    double e0 = entropy(data, n);
    applyInstr(data, n, (Instr){SCRAMBLE, 0, 0, best_si});
    g_ilist[g_ni++] = (Instr){SCRAMBLE, 0, 0, best_si};
    *total_net += best_edelta - INSTR_OHD(4);
    counts[SCRAMBLE]++;
    if (verbose)
        printf("  %-14s si=%-7d %.6f -> %.6f  net=%.1f\n",
               "SCRAMBLE", best_si, e0, entropy(data, n), best_edelta - INSTR_OHD(4));
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
                printf("  %-14s s%-2d p%-2d a%-6d  %.6f -> %.6f  net=%.1f\n",
                       INSTR_NAMES[t.type], t.stride, t.phase, t.amp,
                       e0, entropy(data, n), net);
        }

        /* last resort: try every scramble, pick whichever unlocks the most */
        if (!try_scramble(data, n, &total_net, counts, verbose)) break;
    }
    return total_net;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void) {
    init_hlog();
    init_mul_inv();
    init_gf_tables();
    int counts[NUM_INSTR_TYPES] = {0};
    double sum = 0.0;

    /* load or generate seed data — seed.bin ensures reproducible runs */
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
