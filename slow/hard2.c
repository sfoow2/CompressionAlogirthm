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
    XOR_NIBS,       /* lo-nib ^= (amp&0xF), hi-nib ^= (amp>>4), no condition  */
    /* 9-bit overhead */
    NIB_SWAP,       /* data[i] = (data[i]<<4)|(data[i]>>4), self-inverse       */
    /* 24-bit overhead */
    AFFINE_PHASE,   /* data[i] = (data[i]*m + c)&0xFF  amp = idx|(c<<7), m=2*idx+3  */
    /* 25-bit overhead */
    DUAL_XOR,       /* even ^= (amp&0xFF), odd ^= ((amp>>8)&0xFF)              */
    DUAL_ADD,       /* even += (amp&0xFF), odd += ((amp>>8)&0xFF)              */
    /* 23-bit overhead */
    DUAL_MUL,       /* even *= m_lo, odd *= m_hi; amp = idx_lo|(idx_hi<<7)    */
    /* 10-bit overhead (4 type + 6 stride, no phase, no amp) */
    DELTA_XOR,      /* data[i] ^= data[i-stride] for all i>=stride (right-to-left) */
    /* 13-bit overhead */
    SCRAMBLE,       /* position rearrangement; amp = 0:IL2 1:IL4 2:IL8 3:IL16 4:IL32 5:IL64 6:BITPL 7:NIBPL 8:REV 9:XORFOLD */
    NUM_INSTR_TYPES
} InstrType;

static const char *INSTR_NAMES[NUM_INSTR_TYPES] = {
    "XOR_PHASE", "ADD_PHASE", "MUL_ODD", "COND_LO_XOR", "ADD_NIBS",
    "COND_HI_XOR", "COND_LO_ADD", "COND_HI_ADD", "XOR_NIBS",
    "NIB_SWAP", "AFFINE_PHASE", "DUAL_XOR", "DUAL_ADD", "DUAL_MUL",
    "DELTA_XOR", "SCRAMBLE"
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
        case AFFINE_PHASE: {
            int m = (t.amp & 0x7F) * 2 + 3, c = (t.amp >> 7) & 0xFF;
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)((data[i] * m + c) & 0xFF);
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
        case XOR_NIBS: {
            int xl = t.amp & 0xF, xh = (t.amp >> 4) & 0xF;
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)(((data[i] & 0xF) ^ xl) | ((((data[i] >> 4) ^ xh) & 0xF) << 4));
            break;
        }
        case DUAL_MUL: {
            int m_lo = (t.amp & 0x7F) * 2 + 3, m_hi = ((t.amp >> 7) & 0x7F) * 2 + 3;
            int k = 0;
            for (i = t.phase; i < n; i += t.stride, k++)
                data[i] = (u8)((data[i] * (k & 1 ? m_hi : m_lo)) & 0xFF);
            break;
        }
        case DELTA_XOR:
            for (i = n - 1; i >= t.stride; i--)
                data[i] ^= data[i - t.stride];
            break;
        case SCRAMBLE: {
            u8 tmp[BLOCK_SIZE];
            int sc = t.amp & 0xF;
            if      (sc == 0) interleave_stride(data, tmp, n, 2);
            else if (sc == 1) interleave_stride(data, tmp, n, 4);
            else if (sc == 2) interleave_stride(data, tmp, n, 8);
            else if (sc == 3) interleave_stride(data, tmp, n, 16);
            else if (sc == 4) interleave_stride(data, tmp, n, 32);
            else if (sc == 5) interleave_stride(data, tmp, n, 64);
            else if (sc == 6) bit_plane_sep(data, tmp, n);
            else if (sc == 7) nib_plane_sep(data, tmp, n);
            else if (sc == 8) block_reverse(data, tmp, n);
            else              xor_fold_scramble(data, tmp, n);
            memcpy(data, tmp, n);
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
        /* ── DELTA_XOR: data[i] ^= data[i-stride], no phase/amp ─────────── */
        {
            int dxfreq[256] = {0};
            for (int i = 0; i < stride && i < n; i++) dxfreq[data[i]]++;
            for (int i = stride; i < n; i++) dxfreq[data[i] ^ data[i - stride]]++;
            double Sdx = 0.0;
            for (int v = 0; v < 256; v++) Sdx += hlog[dxfreq[v]];
            double ndx = (Sdx - Sbase) - 10.0;
            if (ndx > bestNet) { bestNet = ndx; best = (Instr){DELTA_XOR, stride, 0, 0}; }
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
                double nx = (Sx - Sbase) - 17.0;
                if (nx > bestNet) { bestNet = nx; best = (Instr){XOR_PHASE, stride, phase, ax}; }
            }
            /* ── ADD_PHASE: brute force ──────────────────────────────────── */
            for (int amp = 1; amp < 256; amp++) {
                double Sa = 0.0;
                for (int v = 0; v < 256; v++)
                    Sa += hlog[dv[v] + phF[(v - amp) & 0xFF]];
                double na = (Sa - Sbase) - 17.0;
                if (na > bestNet) { bestNet = na; best = (Instr){ADD_PHASE, stride, phase, amp}; }
            }

            /* ── MUL_ODD ─────────────────────────────────────────────────── */
            for (int amp = 3; amp < 256; amp += 2) {
                double Sm = 0.0;
                for (int v = 0; v < 256; v++)
                    Sm += hlog[dv[v] + phF[(v * mul_inv[amp]) & 0xFF]];
                double nm = (Sm - Sbase) - 17.0;
                if (nm > bestNet) { bestNet = nm; best = (Instr){MUL_ODD, stride, phase, amp}; }
            }

            /* ── COND_LO_XOR ─────────────────────────────────────────────── */
            for (int nib_cond = 0; nib_cond < 16; nib_cond++) {
                for (int xv = 1; xv < 16; xv++) {
                    double delta = 0.0;
                    for (int lo = 0; lo < 16; lo++) {
                        int v    = (nib_cond << 4) | lo;
                        int v_xv = (nib_cond << 4) | (lo ^ xv);
                        delta += hlog[dv[v] + phF[v_xv]] - hlt[v];
                    }
                    double nc = delta - 17.0;
                    if (nc > bestNet) { bestNet = nc; best = (Instr){COND_LO_XOR, stride, phase, (nib_cond << 4) | xv}; }
                }
            }

            /* ── ADD_NIBS ─────────────────────────────────────────────────── */
            for (int amp = 1; amp < 256; amp++) {
                int lo_a = amp & 0x0F, hi_a = (amp >> 4) & 0x0F;
                double Sn = 0.0;
                for (int v = 0; v < 256; v++) {
                    int old_v = ((v - lo_a) & 0x0F) | (((v >> 4) - hi_a) & 0x0F) << 4;
                    Sn += hlog[dv[v] + phF[old_v]];
                }
                double nn = (Sn - Sbase) - 17.0;
                if (nn > bestNet) { bestNet = nn; best = (Instr){ADD_NIBS, stride, phase, amp}; }
            }

            /* ── NIB_SWAP: self-inverse, cost 8 bits ─────────────────────── */
            {
                double Ssw = 0.0;
                for (int v = 0; v < 256; v++) {
                    int sv = ((v << 4) | (v >> 4)) & 0xFF;
                    Ssw += hlog[dv[v] + phF[sv]];
                }
                double nsw = (Ssw - Sbase) - 9.0;
                if (nsw > bestNet) { bestNet = nsw; best = (Instr){NIB_SWAP, stride, phase, 0}; }
            }

            /* ── COND_HI_XOR: if lo-nib==nc: hi-nib ^= xv ─────────────── */
            for (int nc = 0; nc < 16; nc++) {
                for (int xv = 1; xv < 16; xv++) {
                    double delta = 0.0;
                    for (int hi = 0; hi < 16; hi++) {
                        int v    = (hi << 4) | nc;
                        int v_xv = ((hi ^ xv) << 4) | nc;
                        delta += hlog[dv[v] + phF[v_xv]] - hlt[v];
                    }
                    double nc_net = delta - 17.0;
                    if (nc_net > bestNet) { bestNet = nc_net; best = (Instr){COND_HI_XOR, stride, phase, nc | (xv << 4)}; }
                }
            }

            /* ── COND_LO_ADD: if hi-nib==nc: lo-nib += av ───────────── */
            for (int nc = 0; nc < 16; nc++) {
                for (int av = 1; av < 16; av++) {
                    double delta = 0.0;
                    for (int lo = 0; lo < 16; lo++) {
                        int v     = (nc << 4) | lo;
                        int v_inv = (nc << 4) | ((lo - av) & 0xF);
                        delta += hlog[dv[v] + phF[v_inv]] - hlt[v];
                    }
                    double nc_net = delta - 17.0;
                    if (nc_net > bestNet) { bestNet = nc_net; best = (Instr){COND_LO_ADD, stride, phase, (nc << 4) | av}; }
                }
            }

            /* ── COND_HI_ADD: if lo-nib==nc: hi-nib += av ───────────── */
            for (int nc = 0; nc < 16; nc++) {
                for (int av = 1; av < 16; av++) {
                    double delta = 0.0;
                    for (int hi = 0; hi < 16; hi++) {
                        int v   = (hi << 4) | nc;
                        int v_a = (((hi + av) & 0xF) << 4) | nc;
                        delta += hlog[dv[v] + phF[v_a]] - hlt[v];
                    }
                    double nc_net = delta - 17.0;
                    if (nc_net > bestNet) { bestNet = nc_net; best = (Instr){COND_HI_ADD, stride, phase, nc | (av << 4)}; }
                }
            }

/* ── AFFINE_PHASE: disabled (O(127×256×256) per small-stride phase,
             *   never fires on BCrypt; re-enable if needed for other data) ── */
            if (0 && stride <= 8) {
                for (int m = 3; m <= 255; m += 2) {
                    int minv = mul_inv[m];
                    for (int c = 0; c < 256; c++) {
                        double Sf = 0.0;
                        for (int v = 0; v < 256; v++) {
                            int old_v = ((v - c) * minv) & 0xFF;
                            Sf += hlog[dv[v] + phF[old_v]];
                        }
                        double nf = (Sf - Sbase) - 24.0;
                        if (nf > bestNet) { bestNet = nf; best = (Instr){AFFINE_PHASE, stride, phase, ((m-3)/2) | (c << 7)}; }
                    }
                }
            }

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
                double net_dxor = (best_S2_xor - Sbase) - 25.0;
                if (net_dxor > bestNet) {
                    bestNet = net_dxor;
                    best = (Instr){DUAL_XOR, stride, phase, best_lo_xor | (best_hi_xor << 8)};
                }

                /* ── DUAL_ADD ── */
                int    best_lo_add = 1;
                double best_S_add  = -1e30;
                for (int amp = 1; amp < 256; amp++) {
                    double S = 0.0;
                    for (int v = 0; v < 256; v++)
                        S += hlog[dte[v] + phFe[(v - amp) & 0xFF]];
                    if (S > best_S_add) { best_S_add = S; best_lo_add = amp; }
                }
                int t2a[256];
                for (int v = 0; v < 256; v++)
                    t2a[v] = dte[v] + phFe[(v - best_lo_add) & 0xFF];
                int    best_hi_add = 1;
                double best_S2_add = -1e30;
                for (int amp = 1; amp < 256; amp++) {
                    double S = 0.0;
                    for (int v = 0; v < 256; v++) {
                        int phFo_v     = phF[v]                - phFe[v];
                        int phFo_v_amp = phF[(v - amp) & 0xFF] - phFe[(v - amp) & 0xFF];
                        S += hlog[t2a[v] - phFo_v + phFo_v_amp];
                    }
                    if (S > best_S2_add) { best_S2_add = S; best_hi_add = amp; }
                }
                double net_dadd = (best_S2_add - Sbase) - 25.0;
                if (net_dadd > bestNet) {
                    bestNet = net_dadd;
                    best = (Instr){DUAL_ADD, stride, phase, best_lo_add | (best_hi_add << 8)};
                }

                /* ── DUAL_MUL ── */
                {
                    int    best_lo_mul = 3;
                    double best_S_mul  = -1e30;
                    for (int m = 3; m < 256; m += 2) {
                        double S = 0.0;
                        for (int v = 0; v < 256; v++)
                            S += hlog[dte[v] + phFe[(v * mul_inv[m]) & 0xFF]];
                        if (S > best_S_mul) { best_S_mul = S; best_lo_mul = m; }
                    }
                    int t2m[256];
                    for (int v = 0; v < 256; v++)
                        t2m[v] = dte[v] + phFe[(v * mul_inv[best_lo_mul]) & 0xFF];
                    int    best_hi_mul = 3;
                    double best_S2_mul = -1e30;
                    for (int m = 3; m < 256; m += 2) {
                        double S = 0.0;
                        for (int v = 0; v < 256; v++) {
                            int phFo_v   = phF[v]                           - phFe[v];
                            int phFo_v_m = phF[(v * mul_inv[m]) & 0xFF]     - phFe[(v * mul_inv[m]) & 0xFF];
                            S += hlog[t2m[v] - phFo_v + phFo_v_m];
                        }
                        if (S > best_S2_mul) { best_S2_mul = S; best_hi_mul = m; }
                    }
                    double net_dmul = (best_S2_mul - Sbase) - 23.0;
                    if (net_dmul > bestNet) {
                        bestNet = net_dmul;
                        int idx_lo = (best_lo_mul - 3) / 2;
                        int idx_hi = (best_hi_mul - 3) / 2;
                        best = (Instr){DUAL_MUL, stride, phase, idx_lo | (idx_hi << 7)};
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
    if      (si == 0 && n%2==0)  { interleave_stride(src, dst, n, 2);  return 1; }
    else if (si == 1 && n%4==0)  { interleave_stride(src, dst, n, 4);  return 1; }
    else if (si == 2 && n%8==0)  { interleave_stride(src, dst, n, 8);  return 1; }
    else if (si == 3 && n%16==0) { interleave_stride(src, dst, n, 16); return 1; }
    else if (si == 4 && n%32==0) { interleave_stride(src, dst, n, 32); return 1; }
    else if (si == 5 && n%64==0) { interleave_stride(src, dst, n, 64); return 1; }
    else if (si == 6 && n%8==0)  { bit_plane_sep(src, dst, n);         return 1; }
    else if (si == 7 && n%2==0)  { nib_plane_sep(src, dst, n);         return 1; }
    else if (si == 8)            { block_reverse(src, dst, n);          return 1; }
    else if (si == 9 && n%2==0)  { xor_fold_scramble(src, dst, n);     return 1; }
    return 0;
}

/* ── greedy compress ─────────────────────────────────────────────────────── */
/* try every scramble on current data, apply best if gain > 0; returns 1 if fired */
static int try_scramble(u8 *data, int n, double *total_net, int *counts, int verbose,
                        int max_stride, int max_steps) {
    int sc_freq[256] = {0};
    for (int i = 0; i < n; i++) sc_freq[data[i]]++;
    double Sbase = 0.0;
    for (int v = 0; v < 256; v++) Sbase += hlog[sc_freq[v]];

    static u8 scbuf[BLOCK_SIZE], tmpwork[BLOCK_SIZE];
    int    best_si    = -1;
    double best_gain  = 0.0;
    double best_edelta = 0.0;

    for (int si = 0; si < 10; si++) {
        if (!apply_scramble(si, data, scbuf, n)) continue;

        int sc_tot[256] = {0};
        for (int i = 0; i < n; i++) sc_tot[scbuf[i]]++;
        double sc_Sb = 0.0;
        for (int v = 0; v < 256; v++) sc_Sb += hlog[sc_tot[v]];
        double edelta = sc_Sb - Sbase;

        memcpy(tmpwork, scbuf, n);
        double temp_net = 0.0;
        for (int step = 0; step < max_steps; step++) {
            double net;
            Instr t2 = findBest(tmpwork, n, &net, max_stride);
            if (net <= 0.0) break;
            applyInstr(tmpwork, n, t2);
            temp_net += net;
        }

        double gain = edelta + temp_net - 9.0;
        if (gain > best_gain) { best_gain = gain; best_si = si; best_edelta = edelta; }
    }

    if (best_si < 0) return 0;

    double e0 = entropy(data, n);
    applyInstr(data, n, (Instr){SCRAMBLE, 0, 0, best_si});
    *total_net += best_edelta - 9.0;
    counts[SCRAMBLE]++;
    if (verbose)
        printf("  %-14s si=%-7d %.6f -> %.6f  net=%.1f\n",
               "SCRAMBLE", best_si, e0, entropy(data, n), best_edelta - 9.0);
    return 1;
}

static double compress(u8 *data, int n, int verbose, int *counts) {
    double total_net = 0.0;

    /* initial scramble: WHT proxy, stride<=32, 4 steps — picks best orientation fast */
    try_scramble(data, n, &total_net, counts, verbose, 32, 4);

    for (;;) {
        /* phase 1: regular greedy, no SCRAMBLE */
        for (;;) {
            double net;
            Instr t = findBest(data, n, &net, 64);
            if (net <= 0.0) break;
            double e0 = entropy(data, n);
            applyInstr(data, n, t);
            total_net += net;
            counts[t.type]++;
            if (verbose)
                printf("  %-14s s%-2d p%-2d a%-6d  %.6f -> %.6f  net=%.1f\n",
                       INSTR_NAMES[t.type], t.stride, t.phase, t.amp,
                       e0, entropy(data, n), net);
        }

        /* phase 2: last-resort SCRAMBLE when greedy is stuck */
        if (!try_scramble(data, n, &total_net, counts, verbose, 16, 6)) break;
    }
    return total_net;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void) {
    init_hlog();
    init_mul_inv();
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

    for (int b = 0; b < NUM_BLOCKS; b++) {
        u8 *data = malloc(BLOCK_SIZE);
        memcpy(data, alldata + b * BLOCK_SIZE, BLOCK_SIZE);
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
        printf("  %-14s %d\n", INSTR_NAMES[i], counts[i]);

    return 0;
}
