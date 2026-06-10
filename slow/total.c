/*
 * ans_entropy_test.c — prototype: ANS compress chunks then entropy-reduce the output.
 *
 * Pipeline per chunk:
 *   raw[4096] → ANS compress → ans_out[ans_len]
 *             → greedy entropy reduce (simple.c logic) → reduced[ans_len] + key
 *             → verify: invert key → ANS decompress → must equal raw[4096]
 *
 * Change INPUT_FILE and MAX_CHUNKS to suit.
 * Compile:  gcc -O3 -o ans_entropy_test ans_entropy_test.c -lm
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint8_t  u8;
typedef uint32_t u32;

/* ── configuration ───────────────────────────────────────────────────────── */
#define INPUT_FILE  "C:\\Users\\lukac\\Documents\\compressor\\data"
#define MAX_CHUNKS  20         /* how many 4096-byte chunks to process (SINGLE_BLOCK=0) */
#define BLOCK_SIZE  4096
#define SINGLE_BLOCK       0   /* 1 = treat entire ANS output as one block (no chunking) */
#define VERBOSE            1   /* 1 = print per-instruction detail                       */
#define ANALYSE            1   /* 1 = print pattern profile before entropy reduction      */
#define ANALYSE_MAX_STRIDE 64  /* stride scan for findBest/verification (1..N)           */
#define ANALYSE_TRI_MAX    32  /* stride scan for TRIPLE category (<=ANALYSE_MAX_STRIDE) */
#define HLOG_MAX  (MAX_CHUNKS * BLOCK_SIZE * 2 + 2)  /* hlog/hlogf table size */

/* ═══════════════════════════════════════════════════════════════════════════
 *  ANS CODER  (Subbotin carryless range coder + adaptive order-0 model)
 *  Copied from compressorelement.c — self-contained, no transmitted table.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define RC_TOP  (1u << 24)
#define RC_BOT  (1u << 16)

typedef struct { u32 low, range, code; u8 *buf; int pos, cap; } RC;

static void enc_init(RC *rc, u8 *buf, int cap) {
    rc->low = 0; rc->range = 0xFFFFFFFFu; rc->code = 0;
    rc->buf = buf; rc->pos = 0; rc->cap = cap;
}
static void enc_renorm(RC *rc) {
    while ((rc->low ^ (rc->low + rc->range)) < RC_TOP ||
           (rc->range < RC_BOT && ((rc->range = (0u - rc->low) & (RC_BOT-1)), 1))) {
        if (rc->pos < rc->cap) rc->buf[rc->pos] = (u8)(rc->low >> 24);
        rc->pos++; rc->low <<= 8; rc->range <<= 8;
    }
}
static void enc_encode(RC *rc, u32 cum, u32 freq, u32 tot) {
    rc->range /= tot; rc->low += cum * rc->range; rc->range *= freq; enc_renorm(rc);
}
static void enc_flush(RC *rc) {
    for (int i = 0; i < 4; i++) {
        if (rc->pos < rc->cap) rc->buf[rc->pos] = (u8)(rc->low >> 24);
        rc->pos++; rc->low <<= 8;
    }
}
static void dec_init(RC *rc, u8 *buf, int cap) {
    rc->low = 0; rc->range = 0xFFFFFFFFu; rc->code = 0;
    rc->buf = buf; rc->pos = 0; rc->cap = cap;
    for (int i = 0; i < 4; i++) {
        rc->code = (rc->code << 8) | (rc->pos < rc->cap ? rc->buf[rc->pos] : 0);
        rc->pos++;
    }
}
static void dec_renorm(RC *rc) {
    while ((rc->low ^ (rc->low + rc->range)) < RC_TOP ||
           (rc->range < RC_BOT && ((rc->range = (0u - rc->low) & (RC_BOT-1)), 1))) {
        rc->code = (rc->code << 8) | (rc->pos < rc->cap ? rc->buf[rc->pos] : 0);
        rc->pos++; rc->low <<= 8; rc->range <<= 8;
    }
}
static u32 dec_getfreq(RC *rc, u32 tot) { rc->range /= tot; return (rc->code - rc->low) / rc->range; }
static void dec_decode(RC *rc, u32 cum, u32 freq) {
    rc->low += cum * rc->range; rc->range *= freq; dec_renorm(rc);
}

static u32 m0[256];
static void m0_init(void) { for (int i = 0; i < 256; i++) m0[i] = 1; }
static void m0_bump(int s) {
    m0[s]++;
    u32 tot = 0; for (int j = 0; j < 256; j++) tot += m0[j];
    if (tot >= RC_BOT - 1) for (int j = 0; j < 256; j++) m0[j] = (m0[j] >> 1) | 1;
}

static int o0_compress(const u8 *in, int n, u8 *out, int cap) {
    RC rc; enc_init(&rc, out, cap);
    m0_init();
    for (int i = 0; i < n; i++) {
        u8 s = in[i];
        u32 cum = 0; for (int j = 0; j < s; j++) cum += m0[j];
        u32 tot = 0; for (int j = 0; j < 256; j++) tot += m0[j];
        enc_encode(&rc, cum, m0[s], tot);
        m0_bump(s);
    }
    enc_flush(&rc);
    return rc.pos;
}
static int o0_decompress(const u8 *in, int n, u8 *out, int outlen) {
    RC rc; dec_init(&rc, (u8 *)in, n);
    m0_init();
    for (int i = 0; i < outlen; i++) {
        u32 tot = 0; for (int j = 0; j < 256; j++) tot += m0[j];
        u32 dv  = dec_getfreq(&rc, tot);
        u32 cum = 0; int s = 0;
        for (s = 0; s < 256; s++) { if (cum + m0[s] > dv) break; cum += m0[s]; }
        dec_decode(&rc, cum, m0[s]);
        out[i] = (u8)s;
        m0_bump(s);
    }
    return outlen;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  ENTROPY REDUCER  (greedy search, simple.c logic verbatim)
 * ═══════════════════════════════════════════════════════════════════════════ */

/* ── hlog table ──────────────────────────────────────────────────────────── */
static double hlog[HLOG_MAX];
static int    g_skip_quad = 0;
static void init_hlog(void) {
    hlog[0] = 0.0;
    for (int i = 1; i < HLOG_MAX; i++) hlog[i] = i * log2(i);
}

/* ── analytic period-K XOR/ADD transform ────────────────────────────────── */
#define ANA_MAX_K          64
#define ANA_MODE           1   /* 0=off  1=always (runs every round) */
#define ANA_MAX_INTERLEAVE 128 /* max round slots per chunk (greedy+analytic each get own slot) */

typedef struct { u8 pat[ANA_MAX_K]; int K; int op; int g_ni; } RoundInfo;

static float hlogf[HLOG_MAX];

static void init_hlogf(void) {
    hlogf[0] = 0.0f;
    for (int i = 1; i < HLOG_MAX; i++) hlogf[i] = i * log2f((float)i);
}

static inline int ana_slot(int v, int xp, int op) {
    return op == 0 ? (v ^ xp) : ((v + xp) & 255);
}

static void apply_analytic(u8 *data, int n, u8 *pat, int K, int op) {
    if (op == 0) for (int i = 0; i < n; i++) data[i] ^= pat[i % K];
    else         for (int i = 0; i < n; i++) data[i] = (u8)((data[i] + pat[i % K]) & 0xFF);
}
static void undo_analytic(u8 *data, int n, u8 *pat, int K, int op) {
    if (op == 0) for (int i = 0; i < n; i++) data[i] ^= pat[i % K];
    else         for (int i = 0; i < n; i++) data[i] = (u8)((data[i] - pat[i % K] + 256) & 0xFF);
}

/* Returns net gain in bits (0 = no improvement). Fills best_pat[0..(*best_K)-1]. */
static double find_best_analytic(u8 *data, int n, u8 *best_pat, int *best_K, int op) {
    int freq0[256] = {0};
    for (int i = 0; i < n; i++) freq0[data[i]]++;
    double hsum0 = 0.0;
    for (int i = 0; i < 256; i++) if (freq0[i]) hsum0 += hlogf[freq0[i]];
    double H0 = log2((double)n) - hsum0 / n;

    double overall_net = 0.0;
    *best_K = 0;
    const double logn = log2((double)n);

    for (int K = 2; K <= ANA_MAX_K; K++) {
        int sub[ANA_MAX_K][256];
        memset(sub, 0, (size_t)K * 256 * sizeof(int));
        for (int i = 0; i < n; i++) sub[i % K][data[i]]++;

        u8  nz_val[ANA_MAX_K][256]; int nz_cnt[ANA_MAX_K];
        for (int p = 0; p < K; p++) {
            nz_cnt[p] = 0;
            for (int v = 0; v < 256; v++)
                if (sub[p][v]) nz_val[p][nz_cnt[p]++] = (u8)v;
        }

        u8 pat[ANA_MAX_K] = {0};
        int full[256] = {0};
        for (int p = 0; p < K; p++)
            for (int j = 0; j < nz_cnt[p]; j++)
                full[ana_slot(nz_val[p][j], pat[p], op)] += sub[p][nz_val[p][j]];

        double full_hsum = 0.0;
        for (int v = 0; v < 256; v++) if (full[v]) full_hsum += hlogf[full[v]];

        int   base[256];
        int   bperm[256];
        float delta_j[BLOCK_SIZE + 1];
        float delta_perm[256];
        float hsum_arr[256];

        for (int pass = 0; pass < 8; pass++) {
            int changed = 0;
            for (int p = (pass == 0 ? 1 : 0); p < K; p++) {
                int nc    = nz_cnt[p];
                u8  old_x = pat[p];

                memcpy(base, full, 256 * sizeof(int));
                for (int j = 0; j < nc; j++)
                    base[ana_slot(nz_val[p][j], old_x, op)] -= sub[p][nz_val[p][j]];

                double base_hsum = full_hsum;
                for (int j = 0; j < nc; j++) {
                    int v = nz_val[p][j], sp = sub[p][v];
                    int fw = full[ana_slot(v, old_x, op)];
                    base_hsum += hlogf[fw - sp] - hlogf[fw];
                }

                int max_bw = 0;
                for (int w = 0; w < 256; w++) if (base[w] > max_bw) max_bw = base[w];

                float fbase = (float)base_hsum;
                for (int xp = 0; xp < 256; xp++) hsum_arr[xp] = fbase;

                for (int j = 0; j < nc; j++) {
                    int v  = nz_val[p][j];
                    int sp = sub[p][v];
                    for (int b = 0; b <= max_bw; b++)
                        delta_j[b] = hlogf[b + sp] - hlogf[b];
                    for (int xp = 0; xp < 256; xp++) bperm[xp] = base[ana_slot(v, xp, op)];
                    for (int xp = 0; xp < 256; xp++) delta_perm[xp] = delta_j[bperm[xp]];
                    for (int xp = 0; xp < 256; xp++) hsum_arr[xp] += delta_perm[xp];
                }

                double lbest_H = 1e18; u8 lbest_x = old_x;
                for (int xp = 0; xp < 256; xp++) {
                    double H = logn - hsum_arr[xp] / n;
                    if (H < lbest_H) { lbest_H = H; lbest_x = (u8)xp; }
                }

                if (lbest_x != old_x) {
                    for (int j = 0; j < nc; j++) {
                        int v = nz_val[p][j], sp = sub[p][v];
                        int wo = ana_slot(v, old_x, op);
                        full_hsum += hlogf[full[wo] - sp] - hlogf[full[wo]];
                        full[wo] -= sp;
                        int wn = ana_slot(v, lbest_x, op);
                        full_hsum += hlogf[full[wn] + sp] - hlogf[full[wn]];
                        full[wn] += sp;
                    }
                    pat[p] = lbest_x; changed = 1;
                }
            }
            if (!changed) break;
        }

        double best_H  = logn - full_hsum / n;
        double raw_gain = (H0 - best_H) * n;
        double key_cost = 8.0 * K + 4.0;
        double net      = raw_gain - key_cost;

        if (net > overall_net) {
            overall_net = net;
            *best_K     = K;
            memcpy(best_pat, pat, (size_t)K);
        }
    }

    return overall_net;
}

/* ── CTX256_XOR global transform ─────────────────────────────────────────── */
/* Forward: data[i] ^= amp[original data[i-1]], left-to-right, prev=0 for i=0 */
static void apply_ctx256_xor(u8 *data, int n, const u8 *amp) {
    u8 prev = 0;
    for (int i = 0; i < n; i++) {
        u8 orig = data[i];
        data[i] = orig ^ amp[prev];
        prev = orig;
    }
}
/* Undo: forward pass using recovered originals as context (NOT self-inverse) */
static void undo_ctx256_xor(u8 *data, int n, const u8 *amp) {
    u8 prev = 0;
    for (int i = 0; i < n; i++) {
        u8 orig = data[i] ^ amp[prev];
        prev = orig;
        data[i] = orig;
    }
}
/* Coordinate descent: for each of 256 contexts find best XOR amp jointly.
   Key cost = 256*8 = 2048 bits.  Intended for full ANS stream (n >> 4096). */
static double find_best_ctx256_xor(const u8 *data, int n, u8 *amp_out) {
    static int cond[256][256];
    memset(cond, 0, sizeof cond);
    cond[0][data[0]]++;
    for (int i = 1; i < n; i++) cond[(u8)data[i-1]][data[i]]++;

    int total[256] = {0};
    for (int i = 0; i < n; i++) total[data[i]]++;

    double hs0 = 0.0;
    for (int v = 0; v < 256; v++) if (total[v]) hs0 += hlogf[total[v]];
    double H0 = log2((double)n) - hs0 / n;

    u8  amp[256]; memset(amp, 0, 256);
    int full[256]; memcpy(full, total, 256 * sizeof(int));
    double full_hsum = hs0;
    const double logn = log2((double)n);

    static float delta_j[HLOG_MAX];

    for (int pass = 0; pass < 8; pass++) {
        int changed = 0;
        for (int c = 0; c < 256; c++) {
            u8  old_a = amp[c];
            int nz_v[256], nz_cnt = 0;
            for (int v = 0; v < 256; v++) if (cond[c][v]) nz_v[nz_cnt++] = v;
            if (nz_cnt == 0) continue;

            int base[256]; memcpy(base, full, 256 * sizeof(int));
            double base_hsum = full_hsum;
            for (int j = 0; j < nz_cnt; j++) {
                int v = nz_v[j], sp = cond[c][v], sl = v ^ old_a;
                base_hsum += hlogf[base[sl] - sp] - hlogf[base[sl]];
                base[sl] -= sp;
            }

            int max_bw = 0;
            for (int w = 0; w < 256; w++) if (base[w] > max_bw) max_bw = base[w];

            float hsum_arr[256];
            float fbase = (float)base_hsum;
            for (int xp = 0; xp < 256; xp++) hsum_arr[xp] = fbase;

            for (int j = 0; j < nz_cnt; j++) {
                int v = nz_v[j], sp = cond[c][v];
                for (int b = 0; b <= max_bw; b++) delta_j[b] = hlogf[b + sp] - hlogf[b];
                for (int xp = 0; xp < 256; xp++) hsum_arr[xp] += delta_j[base[v ^ xp]];
            }

            double lbest_H = 1e18; u8 lbest_xp = old_a;
            for (int xp = 0; xp < 256; xp++) {
                double H = logn - hsum_arr[xp] / n;
                if (H < lbest_H) { lbest_H = H; lbest_xp = (u8)xp; }
            }

            if (lbest_xp != old_a) {
                for (int j = 0; j < nz_cnt; j++) {
                    int v = nz_v[j], sp = cond[c][v];
                    int wo = v ^ old_a;
                    full_hsum += hlogf[full[wo] - sp] - hlogf[full[wo]];
                    full[wo] -= sp;
                    int wn = v ^ lbest_xp;
                    full_hsum += hlogf[full[wn] + sp] - hlogf[full[wn]];
                    full[wn] += sp;
                }
                amp[c] = lbest_xp; changed = 1;
            }
        }
        if (!changed) break;
    }

    double best_H = logn - full_hsum / n;
    double raw_gain = (H0 - best_H) * n;
    double net = raw_gain - 2048.0;  /* key = 256*8 bits dense */
    if (net > 0.0) memcpy(amp_out, amp, 256);
    return net;  /* caller checks > 0 before applying */
}

#define INSTR_BASE          12
#define INSTR_BASE_NOPHASE   5
#define INSTR_OHD(ab)         ((double)(INSTR_BASE         + (ab)))
#define INSTR_OHD_NOPHASE(ab) ((double)(INSTR_BASE_NOPHASE + (ab)))

/* ── 256-point FFT ───────────────────────────────────────────────────────── */
static double g_tw_r[128], g_tw_i[128];
static void init_fft256(void) {
    for (int k = 0; k < 128; k++) {
        g_tw_r[k] = cos(2.0 * M_PI * k / 256.0);
        g_tw_i[k] = sin(2.0 * M_PI * k / 256.0);
    }
}
static void fft256_core(double *re, double *im, int sign) {
    for (int i = 1, j = 0; i < 256; i++) {
        int bit = 128;
        while (j & bit) { j ^= bit; bit >>= 1; }
        j ^= bit;
        if (i < j) { double t; t=re[i];re[i]=re[j];re[j]=t; t=im[i];im[i]=im[j];im[j]=t; }
    }
    for (int len = 2; len <= 256; len <<= 1) {
        int half = len >> 1, tw_step = 256 / len;
        for (int i = 0; i < 256; i += len) {
            for (int j = 0; j < half; j++) {
                int k = j * tw_step;
                double wr = g_tw_r[k], wi = g_tw_i[k];
                double ur = re[i+j], ui = im[i+j];
                double rv = re[i+j+half], iv = im[i+j+half];
                double vr = rv*wr + sign*iv*wi, vi2 = -sign*rv*wi + iv*wr;
                re[i+j] = ur+vr; im[i+j] = ui+vi2;
                re[i+j+half] = ur-vr; im[i+j+half] = ui-vi2;
            }
        }
    }
}
/* FFT(total) cache: every caller passes dv = total - phF, and the FFT is
   linear, so FFT(dv) = FFT(total) - FFT(phF). The correlation values are
   exact integers recovered via round(); FP error (~1e-7) is far below 0.5,
   so the integer corr[] output is unchanged. Saves 1 of 3 FFTs per call. */
static double g_fft_tot_r[256], g_fft_tot_i[256];
static void add_cyclic_corr(const int *dv, const int *phF, int *corr) {
    (void)dv;
    double Ar[256], Ai[256], Br[256], Bi[256];
    for (int v = 0; v < 256; v++) { Br[v]=(double)phF[v]; Bi[v]=0.0; }
    fft256_core(Br, Bi, +1);
    for (int k = 0; k < 256; k++) { Ar[k]=g_fft_tot_r[k]-Br[k]; Ai[k]=g_fft_tot_i[k]-Bi[k]; }
    for (int k = 0; k < 256; k++) {
        double pr = Ar[k]*Br[k] + Ai[k]*Bi[k], pi = Ai[k]*Br[k] - Ar[k]*Bi[k];
        Ar[k]=pr; Ai[k]=pi;
    }
    fft256_core(Ar, Ai, -1);
    for (int k = 0; k < 256; k++) corr[k] = (int)round(Ar[k] / 256.0);
}

/* ── entropy ─────────────────────────────────────────────────────────────── */
static double entropy(const u8 *data, int n) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[data[i]]++;
    double s = 0.0;
    for (int i = 0; i < 256; i++) {
        if (f[i] > 0)
            s += f[i] < HLOG_MAX ? hlog[f[i]] : (double)f[i] * log2((double)f[i]);
    }
    return log2((double)n) - s / n;
}

/* ── multiply-inverse table mod 256 ─────────────────────────────────────── */
static u8 mul_inv[256];
static void init_mul_inv(void) {
    for (int k = 1; k < 256; k += 2)
        for (int inv = 1; inv < 256; inv += 2)
            if (((k * inv) & 0xFF) == 1) { mul_inv[k] = (u8)inv; break; }
}

/* ── instruction types ───────────────────────────────────────────────────── */
typedef enum {
    XOR_PHASE = 0, ADD_PHASE, PACK_XOR, COND_LO_XOR, ADD_NIBS,
    COND_HI_XOR, COND_LO_ADD, COND_HI_ADD, NIB_SWAP, BIT_ROTATE,
    VALUE_XOR, DUAL_XOR, DUAL_ADD, DUAL_MUL, LAG_XOR, LAG_ADD,
    COND_PREV_XOR, COND_PREV_ADD, POLY_DELTA_XOR, SCRAMBLE,
    TRIPLE_XOR, TRIPLE_ADD, TRIPLE_MUL,
    QUAD_XOR, QUAD_ADD, QUAD_MUL,
    CTX4_XOR, CTX4_ADD,
    NUM_INSTR_TYPES
} InstrType;
static const char *INSTR_NAMES[NUM_INSTR_TYPES] = {
    "XOR_PHASE","ADD_PHASE","PACK_XOR","COND_LO_XOR","ADD_NIBS",
    "COND_HI_XOR","COND_LO_ADD","COND_HI_ADD","NIB_SWAP",
    "BIT_ROTATE","VALUE_XOR","DUAL_XOR","DUAL_ADD","DUAL_MUL","LAG_XOR","LAG_ADD",
    "COND_PREV_XOR","COND_PREV_ADD","POLY_DELTA_XOR","SCRAMBLE",
    "TRIPLE_XOR","TRIPLE_ADD","TRIPLE_MUL","QUAD_XOR","QUAD_ADD","QUAD_MUL",
    "CTX4_XOR","CTX4_ADD"
};
typedef struct { InstrType type; int stride, phase; unsigned int amp; } Instr;

/* ── forward declarations for scramble helpers ───────────────────────────── */
static void interleave_stride(const u8 *src, u8 *dst, int n, int s);
static void bit_plane_sep(const u8 *src, u8 *dst, int n);
static void nib_plane_sep(const u8 *src, u8 *dst, int n);
static void block_reverse(const u8 *src, u8 *dst, int n);
static void xor_fold_scramble(const u8 *src, u8 *dst, int n);

/* ── applyInstr ──────────────────────────────────────────────────────────── */
static void applyInstr(u8 *data, int n, Instr t) {
    int i;
    switch (t.type) {
    case XOR_PHASE: for(i=t.phase;i<n;i+=t.stride) data[i]^=(u8)t.amp; break;
    case ADD_PHASE: for(i=t.phase;i<n;i+=t.stride) data[i]+=(u8)t.amp; break;
    case PACK_XOR: {
        int k=0; for(i=t.phase;i<n;i+=t.stride,k++){
            if(k&1)continue; int j=i+t.stride; if(j>=n)continue; data[i]^=data[j]; } break; }
    case COND_LO_XOR: { int nc=(t.amp>>4)&0xF, xv=t.amp&0xF;
        for(i=t.phase;i<n;i+=t.stride) if((data[i]>>4)==nc) data[i]^=(u8)xv; break; }
    case ADD_NIBS: { u8 lo=(u8)(t.amp&0xF),hi=(u8)((t.amp>>4)&0xF);
        for(i=t.phase;i<n;i+=t.stride)
            data[i]=(u8)(((data[i]+lo)&0xF)|(((data[i]>>4)+hi&0xF)<<4)); break; }
    case NIB_SWAP: for(i=t.phase;i<n;i+=t.stride) data[i]=(u8)((data[i]<<4)|(data[i]>>4)); break;
    case BIT_ROTATE: { int k=t.amp&7;
        for(i=t.phase;i<n;i+=t.stride) data[i]=(u8)((data[i]<<k)|(data[i]>>(8-k))); break; }
    case VALUE_XOR: { int k=t.amp&7,mask=1<<k,alo=(t.amp>>3)&0xFF,ahi=(t.amp>>11)&0xFF;
        for(i=t.phase;i<n;i+=t.stride) data[i]^=(u8)((data[i]&mask)?ahi:alo); break; }
    case DUAL_XOR: { int lo=t.amp&0xFF,hi=(t.amp>>8)&0xFF,k=0;
        for(i=t.phase;i<n;i+=t.stride,k++) data[i]^=(u8)(k&1?hi:lo); break; }
    case DUAL_ADD: { int lo=t.amp&0xFF,hi=(t.amp>>8)&0xFF,k=0;
        for(i=t.phase;i<n;i+=t.stride,k++) data[i]+=(u8)(k&1?hi:lo); break; }
    case COND_HI_XOR: { int nc=t.amp&0xF,xv=(t.amp>>4)&0xF;
        for(i=t.phase;i<n;i+=t.stride) if((data[i]&0xF)==nc) data[i]^=(u8)(xv<<4); break; }
    case COND_LO_ADD: { int nc=(t.amp>>4)&0xF,av=t.amp&0xF;
        for(i=t.phase;i<n;i+=t.stride)
            if((data[i]>>4)==nc) data[i]=(u8)((data[i]&0xF0)|((data[i]+av)&0xF)); break; }
    case COND_HI_ADD: { int nc=t.amp&0xF,av=(t.amp>>4)&0xF;
        for(i=t.phase;i<n;i+=t.stride)
            if((data[i]&0xF)==nc) data[i]=(u8)((data[i]&0x0F)|(((data[i]>>4)+av)&0xF)<<4); break; }
    case DUAL_MUL: { int ml=(t.amp&0x7F)*2+3,mh=((t.amp>>7)&0x7F)*2+3,k=0;
        for(i=t.phase;i<n;i+=t.stride,k++) data[i]=(u8)((data[i]*(k&1?mh:ml))&0xFF); break; }
    case LAG_XOR: { int lag=(int)t.amp;
        for(i=0;i+lag<n;i++) data[i]^=data[i+lag]; break; }
    case LAG_ADD: { int lag=(int)t.amp;
        for(i=0;i+lag<n;i++) data[i]=(u8)((data[i]+data[i+lag])&0xFF); break; }
    case COND_PREV_XOR: { int alo=t.amp&0xFF,ahi=(t.amp>>8)&0xFF,pv=0,first=1;
        for(i=t.phase;i<n;i+=t.stride){
            if(first){first=0;pv=data[i];continue;}
            data[i]^=(u8)((pv>=128)?ahi:alo); pv=data[i]; } break; }
    case COND_PREV_ADD: { int alo=t.amp&0xFF,ahi=(t.amp>>8)&0xFF,pv=0,first=1;
        for(i=t.phase;i<n;i+=t.stride){
            if(first){first=0;pv=data[i];continue;}
            data[i]=(u8)((data[i]+((pv>=128)?ahi:alo))&0xFF); pv=data[i]; } break; }
    case POLY_DELTA_XOR: { int s1=t.stride,s2=t.phase;
        for(i=n-1;i>=s2;i--) data[i]^=data[i-s1]^data[i-s2]; break; }
    case SCRAMBLE: { u8 tmp[BLOCK_SIZE]; int sc=t.amp&0xF;
        if     (sc==0) interleave_stride(data,tmp,n,2);
        else if(sc==1) interleave_stride(data,tmp,n,4);
        else if(sc==2) interleave_stride(data,tmp,n,8);
        else if(sc==3) interleave_stride(data,tmp,n,16);
        else if(sc==4) interleave_stride(data,tmp,n,32);
        else if(sc==5) interleave_stride(data,tmp,n,64);
        else if(sc==6) bit_plane_sep(data,tmp,n);
        else if(sc==7) nib_plane_sep(data,tmp,n);
        else if(sc==8) block_reverse(data,tmp,n);
        else if(sc==9) xor_fold_scramble(data,tmp,n);
        else if(sc==10) interleave_stride(data,tmp,n,1024);
        else if(sc==11) interleave_stride(data,tmp,n,512);
        else if(sc==12) interleave_stride(data,tmp,n,256);
        else            interleave_stride(data,tmp,n,128);
        memcpy(data,tmp,n); break; }
    case TRIPLE_XOR: { int a0=t.amp&0xFF,a1=(t.amp>>8)&0xFF,a2=(t.amp>>16)&0xFF,k=0;
        for(i=t.phase;i<n;i+=t.stride,k++){
            if(k%3==0)data[i]^=(u8)a0; else if(k%3==1)data[i]^=(u8)a1; else data[i]^=(u8)a2; }
        break; }
    case TRIPLE_ADD: { int a0=t.amp&0xFF,a1=(t.amp>>8)&0xFF,a2=(t.amp>>16)&0xFF,k=0;
        for(i=t.phase;i<n;i+=t.stride,k++){
            if(k%3==0)data[i]+=(u8)a0; else if(k%3==1)data[i]+=(u8)a1; else data[i]+=(u8)a2; }
        break; }
    case TRIPLE_MUL: { int m0=(t.amp&0x7F)*2+3,m1=((t.amp>>7)&0x7F)*2+3,m2=((t.amp>>14)&0x7F)*2+3,k=0;
        for(i=t.phase;i<n;i+=t.stride,k++){
            if(k%3==0)data[i]=(u8)((data[i]*m0)&0xFF);
            else if(k%3==1)data[i]=(u8)((data[i]*m1)&0xFF);
            else data[i]=(u8)((data[i]*m2)&0xFF); } break; }
    case QUAD_XOR: { unsigned a0=t.amp&0xFF,a1=(t.amp>>8)&0xFF,a2=(t.amp>>16)&0xFF,a3=t.amp>>24;
        unsigned aa[4]={a0,a1,a2,a3}; int k=0;
        for(i=t.phase;i<n;i+=t.stride,k++) data[i]^=(u8)aa[k&3]; break; }
    case QUAD_ADD: { unsigned a0=t.amp&0xFF,a1=(t.amp>>8)&0xFF,a2=(t.amp>>16)&0xFF,a3=t.amp>>24;
        unsigned aa[4]={a0,a1,a2,a3}; int k=0;
        for(i=t.phase;i<n;i+=t.stride,k++) data[i]+=(u8)aa[k&3]; break; }
    case QUAD_MUL: { int m0=(t.amp&0x7F)*2+3,m1=((t.amp>>7)&0x7F)*2+3;
        int m2=((t.amp>>14)&0x7F)*2+3,m3=((t.amp>>21)&0x7F)*2+3;
        int mm[4]={m0,m1,m2,m3}; int k=0;
        for(i=t.phase;i<n;i+=t.stride,k++) data[i]=(u8)((data[i]*mm[k&3])&0xFF); break; }
    case CTX4_XOR: { u8 a[4]={(u8)(t.amp&0xFF),(u8)((t.amp>>8)&0xFF),(u8)((t.amp>>16)&0xFF),(u8)(t.amp>>24)};
        u8 prev=0;
        for(i=0;i<n;i++){data[i]^=a[(prev>>6)&3]; prev=data[i];} break; }
    case CTX4_ADD: { u8 a[4]={(u8)(t.amp&0xFF),(u8)((t.amp>>8)&0xFF),(u8)((t.amp>>16)&0xFF),(u8)(t.amp>>24)};
        u8 prev=0;
        for(i=0;i<n;i++){data[i]=(u8)((data[i]+a[(prev>>6)&3])&0xFF); prev=data[i];} break; }
    default: break;
    }
}

/* ── WHT + fast amp search ───────────────────────────────────────────────── */
static void wht256(int *a) {
    for (int len=1;len<256;len<<=1)
        for (int i=0;i<256;i+=len<<1)
            for (int j=0;j<len;j++){int u=a[i+j],v=a[i+j+len];a[i+j]=u+v;a[i+j+len]=u-v;}
}
static void wht128(int *a) {
    for (int len=1;len<128;len<<=1)
        for (int i=0;i<128;i+=len<<1)
            for (int j=0;j<len;j++){int u=a[i+j],v=a[i+j+len];a[i+j]=u+v;a[i+j+len]=u-v;}
}
static int xor_best_amp_128(const int *A, const int *B, double *Sout) {
    int ha[128],hb[128];
    for(int i=0;i<128;i++){ha[i]=A[i];hb[i]=B[i];}
    wht128(ha); wht128(hb);
    long long prod[128];
    for(int k=0;k<128;k++) prod[k]=(long long)ha[k]*hb[k];
    for(int len=1;len<128;len<<=1)
        for(int i=0;i<128;i+=len<<1)
            for(int j=0;j<len;j++){long long u=prod[i+j],v=prod[i+j+len];prod[i+j]=u+v;prod[i+j+len]=u-v;}
    long long c0=INT64_MIN,c1=INT64_MIN,c2=INT64_MIN; int a0=1,a1=2,a2=3;
    for(int amp=1;amp<128;amp++){long long c=prod[amp];
        if(c>c0){c2=c1;a2=a1;c1=c0;a1=a0;c0=c;a0=amp;}
        else if(c>c1){c2=c1;a2=a1;c1=c;a1=amp;}
        else if(c>c2){c2=c;a2=amp;}}
    int best_amp=a0; double best_S=-1e30;
    int cands[3]={a0,a1,a2};
    for(int t=0;t<3;t++){double S=0.0;for(int i=0;i<128;i++)S+=hlog[A[i]+B[i^cands[t]]];
        if(S>best_S){best_S=S;best_amp=cands[t];}}
    *Sout=best_S; return best_amp;
}
static int vxor_ins(int idx, int k) { return ((idx>>k)<<(k+1))|(idx&((1<<k)-1)); }
/* remove bit k from v (v must have bit k == 0), giving a 7-bit half-space index */
static int rbk(int v, int k) { return ((v>>1)&~((1<<k)-1)) | (v&((1<<k)-1)); }
/* WHT(total) cache: every caller passes A = total - B, and the WHT is linear
   over the integers, so WHT(A) = WHT(total) - WHT(B) exactly (bit-identical).
   Saves 1 of 2 forward WHTs per call.  set_freq_cache() must be called after
   computing total[] (done at the top of findBest and analyse_chunk).        */
static int g_wht_tot[256];
static void set_freq_cache(const int *total) {
    for (int i = 0; i < 256; i++) g_wht_tot[i] = total[i];
    wht256(g_wht_tot);
    for (int v = 0; v < 256; v++) { g_fft_tot_r[v] = (double)total[v]; g_fft_tot_i[v] = 0.0; }
    fft256_core(g_fft_tot_r, g_fft_tot_i, +1);
}
static int xor_best_amp(const int *A, const int *B, double *Sout) {
    int ha[256],hb[256];
    for(int i=0;i<256;i++) hb[i]=B[i];
    wht256(hb);
    for(int i=0;i<256;i++) ha[i]=g_wht_tot[i]-hb[i];
    long long prod[256];
    for(int k=0;k<256;k++) prod[k]=(long long)ha[k]*hb[k];
    for(int len=1;len<256;len<<=1)
        for(int i=0;i<256;i+=len<<1)
            for(int j=0;j<len;j++){long long u=prod[i+j],v=prod[i+j+len];prod[i+j]=u+v;prod[i+j+len]=u-v;}
    long long c0=INT64_MIN,c1=INT64_MIN,c2=INT64_MIN; int a0=1,a1=2,a2=3;
    for(int amp=1;amp<256;amp++){long long c=prod[amp];
        if(c>c0){c2=c1;a2=a1;c1=c0;a1=a0;c0=c;a0=amp;}
        else if(c>c1){c2=c1;a2=a1;c1=c;a1=amp;}
        else if(c>c2){c2=c;a2=amp;}}
    int best_amp=a0; double best_S=-1e30;
    int cands[3]={a0,a1,a2};
    for(int t=0;t<3;t++){double S=0.0;for(int v=0;v<256;v++)S+=hlog[A[v]+B[v^cands[t]]];
        if(S>best_S){best_S=S;best_amp=cands[t];}}
    *Sout=best_S; return best_amp;
}
static int add_best_amp(const int *dv, const int *phF, int _K, double *Sout) {
    (void)_K;
    int corr[256]; add_cyclic_corr(dv, phF, corr);
    int top8[8]={1,2,3,4,5,6,7,8},tv[8];
    for(int i=0;i<8;i++) tv[i]=corr[i+1];
    for(int i=0;i<8;i++) for(int j=i+1;j<8;j++)
        if(tv[j]>tv[i]){int t=tv[i];tv[i]=tv[j];tv[j]=t;int ti=top8[i];top8[i]=top8[j];top8[j]=ti;}
    for(int k=9;k<256;k++){int c=corr[k];if(c<=tv[7])continue;
        int pos=7; while(pos>0&&c>tv[pos-1])pos--;
        for(int j=7;j>pos;j--){top8[j]=top8[j-1];tv[j]=tv[j-1];}
        top8[pos]=k;tv[pos]=c;}
    int best_amp=top8[0]; double best_S=-1e30;
    for(int ci=0;ci<8;ci++){int amp=top8[ci]; double S=0.0;
        for(int v=0;v<256;v++) S+=hlog[dv[v]+phF[(v-amp)&0xFF]];
        if(S>best_S){best_S=S;best_amp=amp;}}
    *Sout=best_S; return best_amp;
}
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
        int la=lo4[i],ha=hi4[j],amp=la|(ha<<4); if(!amp)continue;
        double S=0.0; for(int v=0;v<256;v++){int ov=((v-la)&0xF)|(((v>>4)-ha)&0xF)<<4;S+=hlog[dv[v]+phF[ov]];}
        if(S>best_S){best_S=S;best_amp=amp;}}
    *Sout=best_S; return best_amp;
}
static double mul_eval_S(const int *dv, const int *phF, int m) {
    double S=0.0;
    for(int v=0;v<256;v++) S+=hlog[dv[v]+phF[(v*(int)mul_inv[m])&0xFF]];
    return S;
}
static int mul_best_amp(const int *dv, const int *phF, double *Sout) {
    /* memoize S(m): the refine windows overlap each other and the coarse
       probes, so each distinct m is evaluated once (values unchanged) */
    double Sc[256]; u8 seen[256]={0};
    int top3a[3]={3,35,67}; double top3s[3]={-1e30,-1e30,-1e30};
    for(int m=3;m<256;m+=32){
        double S=mul_eval_S(dv,phF,m); Sc[m]=S; seen[m]=1;
        if(S>top3s[0]){top3s[2]=top3s[1];top3a[2]=top3a[1];top3s[1]=top3s[0];top3a[1]=top3a[0];top3s[0]=S;top3a[0]=m;}
        else if(S>top3s[1]){top3s[2]=top3s[1];top3a[2]=top3a[1];top3s[1]=S;top3a[1]=m;}
        else if(S>top3s[2]){top3s[2]=S;top3a[2]=m;}}
    int best_amp=top3a[0]; double best_S=top3s[0];
    for(int ci=0;ci<3;ci++){int base=top3a[ci];
        for(int d=-32;d<=32;d+=2){int m=base+d;
            if(m<3||m>255||!(m&1))continue;
            double S;
            if(seen[m]) S=Sc[m];
            else {S=mul_eval_S(dv,phF,m); Sc[m]=S; seen[m]=1;}
            if(S>best_S){best_S=S;best_amp=m;}}}
    *Sout=best_S; return best_amp;
}

/* ── scramble helpers ────────────────────────────────────────────────────── */
static void interleave_stride(const u8 *src, u8 *dst, int n, int s) {
    int w=n/s; for(int i=0;i<n;i++) dst[(i%s)*w+(i/s)]=src[i];
}
static void bit_plane_sep(const u8 *src, u8 *dst, int n) {
    int ps=n/8; memset(dst,0,n);
    for(int i=0;i<n;i++) for(int b=0;b<8;b++) if((src[i]>>b)&1) dst[b*ps+i/8]|=(u8)(1<<(i%8));
}
static void nib_plane_sep(const u8 *src, u8 *dst, int n) {
    int half=n/2;
    for(int i=0;i<half;i++){dst[i]=(u8)((src[2*i]&0xF)|((src[2*i+1]&0xF)<<4));
                              dst[i+half]=(u8)((src[2*i]>>4)|((src[2*i+1]>>4)<<4));}
}
static void block_reverse(const u8 *src, u8 *dst, int n) { for(int i=0;i<n;i++) dst[i]=src[n-1-i]; }
static void xor_fold_scramble(const u8 *src, u8 *dst, int n) {
    int h=n/2; for(int i=0;i<h;i++) dst[i]=src[i]^src[i+h]; memcpy(dst+h,src+h,h);
}
static int apply_scramble(int si, const u8 *src, u8 *dst, int n) {
    if     (si==0&&n%2==0)   {interleave_stride(src,dst,n,2);   return 1;}
    else if(si==1&&n%4==0)   {interleave_stride(src,dst,n,4);   return 1;}
    else if(si==2&&n%8==0)   {interleave_stride(src,dst,n,8);   return 1;}
    else if(si==3&&n%16==0)  {interleave_stride(src,dst,n,16);  return 1;}
    else if(si==4&&n%32==0)  {interleave_stride(src,dst,n,32);  return 1;}
    else if(si==5&&n%64==0)  {interleave_stride(src,dst,n,64);  return 1;}
    else if(si==6&&n%8==0)   {bit_plane_sep(src,dst,n);          return 1;}
    else if(si==7&&n%2==0)   {nib_plane_sep(src,dst,n);          return 1;}
    else if(si==8)            {block_reverse(src,dst,n);          return 1;}
    else if(si==9&&n%2==0)   {xor_fold_scramble(src,dst,n);     return 1;}
    else if(si==10&&n%1024==0){interleave_stride(src,dst,n,1024);return 1;}
    else if(si==11&&n%512==0) {interleave_stride(src,dst,n,512); return 1;}
    else if(si==12&&n%256==0) {interleave_stride(src,dst,n,256); return 1;}
    else if(si==13&&n%128==0) {interleave_stride(src,dst,n,128); return 1;}
    return 0;
}
static void invert_scramble(u8 *data, int n, int si) {
    u8 tmp[BLOCK_SIZE]; int inv_si=-1;
    switch(si){case 1:inv_si=10;break;case 2:inv_si=11;break;case 3:inv_si=12;break;
               case 4:inv_si=13;break;case 5:inv_si=5;break;case 6:inv_si=6;break;
               case 8:inv_si=8;break;case 9:inv_si=9;break;
               case 10:inv_si=1;break;case 11:inv_si=2;break;case 12:inv_si=3;break;case 13:inv_si=4;break;}
    if(si==0){interleave_stride(data,tmp,n,n/2);memcpy(data,tmp,n);}
    else if(si==7){int half=n/2;
        for(int i=0;i<half;i++){
            tmp[2*i]  =(u8)((data[i]&0xF)|((data[i+half]&0xF)<<4));
            tmp[2*i+1]=(u8)(((data[i]>>4)&0xF)|(((data[i+half]>>4)&0xF)<<4));}
        memcpy(data,tmp,n);}
    else if(inv_si>=0){apply_scramble(inv_si,data,tmp,n);memcpy(data,tmp,n);}
}

/* ── findBest (verbatim from simple.c) ──────────────────────────────────── */
static Instr findBest(const u8 *data, int n, double *netOut, int max_stride) {
    int total[256]={0};
    for(int i=0;i<n;i++) total[data[i]]++;
    double hlt[256],Sbase=0.0;
    for(int v=0;v<256;v++){hlt[v]=hlog[total[v]];Sbase+=hlt[v];}
    set_freq_cache(total);

    double bestNet=0.0;
    Instr  best={XOR_PHASE,2,0,1};
    int phF[256];

    /* LAG_XOR */
    for(int lag=1;lag<n&&lag<=255;lag++){
        int lxfreq[256]; for(int v=0;v<256;v++) lxfreq[v]=total[v];
        for(int ii=0;ii+lag<n;ii++){lxfreq[data[ii]]--;lxfreq[data[ii]^data[ii+lag]]++;}
        double Slx=0.0; for(int v=0;v<256;v++) Slx+=hlog[lxfreq[v]];
        double nlx=(Slx-Sbase)-INSTR_OHD_NOPHASE(8);
        if(nlx>bestNet){bestNet=nlx;best=(Instr){LAG_XOR,0,0,(unsigned)lag};}}

    /* LAG_ADD */
    for(int lag=1;lag<n&&lag<=255;lag++){
        int lafreq[256]; for(int v=0;v<256;v++) lafreq[v]=total[v];
        for(int ii=0;ii+lag<n;ii++){lafreq[data[ii]]--;lafreq[(data[ii]+data[ii+lag])&0xFF]++;}
        double Sla=0.0; for(int v=0;v<256;v++) Sla+=hlog[lafreq[v]];
        double nla=(Sla-Sbase)-INSTR_OHD_NOPHASE(8);
        if(nla>bestNet){bestNet=nla;best=(Instr){LAG_ADD,0,0,(unsigned)lag};}}

    /* CTX4_XOR / CTX4_ADD: 4 contexts keyed on top 2 bits of previous byte */
    {
        int phF4[4][256]={{0},{0},{0},{0}};
        u8 prevC=0;
        for(int i=0;i<n;i++){phF4[(prevC>>6)&3][data[i]]++; prevC=data[i];}
        int dv4[4][256];
        for(int c=0;c<4;c++) for(int v=0;v<256;v++) dv4[c][v]=total[v]-phF4[c][v];
        double S4x=0.0; unsigned amp4x=0;
        for(int c=0;c<4;c++){double Sc; int ac=xor_best_amp(dv4[c],phF4[c],&Sc);
            S4x+=Sc; amp4x|=(unsigned)(ac&0xFF)<<(c*8);}
        double n4x=S4x-4.0*Sbase-INSTR_OHD(32);
        if(n4x>bestNet){bestNet=n4x; best=(Instr){CTX4_XOR,1,0,amp4x};}
        double S4a=0.0; unsigned amp4a=0;
        for(int c=0;c<4;c++){double Sc; int ac=add_best_amp(dv4[c],phF4[c],6,&Sc);
            S4a+=Sc; amp4a|=(unsigned)(ac&0xFF)<<(c*8);}
        double n4a=S4a-4.0*Sbase-INSTR_OHD(32);
        if(n4a>bestNet){bestNet=n4a; best=(Instr){CTX4_ADD,1,0,amp4a};}
    }

    for(int stride=1;stride<=max_stride;stride++){
        /* POLY_DELTA_XOR */
        if(stride<=32){
        for(int s1=1;s1<stride;s1++){
            int pdfreq[256]={0};
            for(int i=0;i<stride&&i<n;i++) pdfreq[data[i]]++;
            for(int i=stride;i<n;i++) pdfreq[data[i]^data[i-s1]^data[i-stride]]++;
            double Spd=0.0; for(int v=0;v<256;v++) Spd+=hlog[pdfreq[v]];
            double npd=(Spd-Sbase)-INSTR_OHD(6);
            if(npd>bestNet){bestNet=npd;best=(Instr){POLY_DELTA_XOR,s1,stride,0};}}}

        for(int phase=0;phase<stride;phase++){
            memset(phF,0,sizeof phF);
            for(int i=phase;i<n;i+=stride) phF[data[i]]++;
            int dv[256]; for(int v=0;v<256;v++) dv[v]=total[v]-phF[v];

            /* XOR_PHASE */ { double Sx; int ax=xor_best_amp(dv,phF,&Sx);
                double nx=(Sx-Sbase)-INSTR_OHD(8);
                if(nx>bestNet){bestNet=nx;best=(Instr){XOR_PHASE,stride,phase,ax};}}
            /* ADD_PHASE */ { double Sa; int aa=add_best_amp(dv,phF,6,&Sa);
                double na=(Sa-Sbase)-INSTR_OHD(8);
                if(na>bestNet){bestNet=na;best=(Instr){ADD_PHASE,stride,phase,aa};}}
            /* PACK_XOR */ {
                int pxfreq[256]; for(int v=0;v<256;v++) pxfreq[v]=total[v];
                int k=0; for(int ii=phase;ii<n;ii+=stride,k++){
                    if(k&1)continue; int j=ii+stride; if(j>=n)continue;
                    pxfreq[data[ii]]--; pxfreq[data[ii]^data[j]]++;}
                double Spx=0.0; for(int v=0;v<256;v++) Spx+=hlog[pxfreq[v]];
                double npx=(Spx-Sbase)-INSTR_OHD(0);
                if(npx>bestNet){bestNet=npx;best=(Instr){PACK_XOR,stride,phase,0};}}
            /* COND_PREV_XOR */ {
                int phF0[256]={0},phF1[256]={0},pv=0,cpx_first=1;
                for(int ii=phase;ii<n;ii+=stride){
                    if(cpx_first){cpx_first=0;pv=data[ii];continue;}
                    if(pv<128) phF0[data[ii]]++; else phF1[data[ii]]++;
                    pv=data[ii];}
                int dv0[256],dv1[256];
                for(int v=0;v<256;v++){dv0[v]=total[v]-phF0[v];dv1[v]=total[v]-phF1[v];}
                double Sx0,Sx1; int ax0=xor_best_amp(dv0,phF0,&Sx0),ax1=xor_best_amp(dv1,phF1,&Sx1);
                double ncpx=(Sx0-Sbase)+(Sx1-Sbase)-INSTR_OHD(16);
                if(ncpx>bestNet){bestNet=ncpx;best=(Instr){COND_PREV_XOR,stride,phase,
                    (unsigned)(ax0|(ax1<<8))};}}
            /* COND_LO_XOR */ {
                double best_sub=-1e30; unsigned best_sub_amp=0x11;
                for(int nib=0;nib<16;nib++){
                    int cf[256]={0}; for(int v=0;v<256;v++) if((v>>4)==nib) cf[v]=phF[v];
                    int bg[256]; for(int v=0;v<256;v++) bg[v]=total[v]-cf[v];
                    int remap[16]={0},remapc[16]={0};
                    for(int v=0;v<256;v++) if((v>>4)==nib){remap[v&0xF]+=cf[v];remapc[v&0xF]+=bg[v];}
                    double Sr=-1e30; int ar=1;
                    {int ha2[128],hb2[128];
                    for(int i=0;i<16;i++){ha2[i]=remapc[i];hb2[i]=remap[i];}
                    for(int i=16;i<128;i++){ha2[i]=0;hb2[i]=0;}
                    int cands2[3]={1,2,3}; long long best_pr=INT64_MIN;
                    {int wa[128],wb[128]; memcpy(wa,ha2,128*sizeof(int)); memcpy(wb,hb2,128*sizeof(int));
                    wht128(wa); wht128(wb);
                    for(int kk=1;kk<16;kk++){long long pp=(long long)wa[kk]*wb[kk];
                        if(pp>best_pr){best_pr=pp;cands2[0]=kk;}}}
                    for(int t=0;t<1;t++){double S2=0.0;
                        for(int i=0;i<16;i++) S2+=hlog[remapc[i]+remap[i^cands2[t]]];
                        if(S2>Sr){Sr=S2;ar=cands2[t];}}
                    double gain_sub=0.0;
                    for(int v=0;v<256;v++) if((v>>4)==nib){
                        int nv=(v&0xF0)|((v^ar)&0xF);
                        gain_sub+=hlog[total[nv]-phF[nv]+phF[v]]-hlog[total[nv]];
                        gain_sub+=hlog[total[v]-phF[v]+phF[nv]]-hlog[total[v]];}
                    if(gain_sub>best_sub){best_sub=gain_sub;best_sub_amp=(unsigned)(nib<<4)|ar;}}
                double ncl=best_sub-INSTR_OHD(8);
                if(ncl>bestNet){bestNet=ncl;best=(Instr){COND_LO_XOR,stride,phase,best_sub_amp};}}
            /* ADD_NIBS */ { double San; int an=add_nibs_best_amp(dv,phF,&San);
                double nan=(San-Sbase)-INSTR_OHD(8);
                if(nan>bestNet){bestNet=nan;best=(Instr){ADD_NIBS,stride,phase,(unsigned)an};}}
            /* COND_HI_XOR */ {
                double best_sub=-1e30; unsigned best_sub_amp=0x11;
                for(int nib=0;nib<16;nib++){
                    int cf[256]={0}; for(int v=0;v<256;v++) if((v&0xF)==nib) cf[v]=phF[v];
                    int remap[16]={0},remapc[16]={0};
                    for(int v=0;v<256;v++) if((v&0xF)==nib){remap[v>>4]+=cf[v];remapc[v>>4]+=total[v]-cf[v];}
                    double Sr=-1e30; int ar=1;
                    {int wa[16],wb[16];
                    for(int i=0;i<16;i++){wa[i]=remapc[i];wb[i]=remap[i];}
                    long long best_pr=INT64_MIN;
                    for(int kk=1;kk<16;kk++){long long pp=(long long)wa[kk]*wb[kk];
                        if(pp>best_pr){best_pr=pp;ar=kk;}}}
                    double gain_sub=0.0;
                    for(int v=0;v<256;v++) if((v&0xF)==nib){
                        int nv=((v^(ar<<4))&0xF0)|(v&0xF);
                        gain_sub+=hlog[total[nv]-phF[nv]+phF[v]]-hlog[total[nv]];
                        gain_sub+=hlog[total[v]-phF[v]+phF[nv]]-hlog[total[v]];}
                    if(gain_sub>best_sub){best_sub=gain_sub;best_sub_amp=(unsigned)((ar<<4)|nib);}}
                double nch=best_sub-INSTR_OHD(8);
                if(nch>bestNet){bestNet=nch;best=(Instr){COND_HI_XOR,stride,phase,best_sub_amp};}}
            /* COND_LO_ADD / COND_HI_ADD */ {
                for(int nib=0;nib<16;nib++){
                    int cf[256]={0}; for(int v=0;v<256;v++) if((v>>4)==nib) cf[v]=phF[v];
                    int remap[16]={0},remapc[16]={0};
                    for(int v=0;v<256;v++) if((v>>4)==nib){remap[v&0xF]+=cf[v];remapc[v&0xF]+=total[v]-cf[v];}
                    int corr16[16]={0}; for(int la=0;la<16;la++) for(int l=0;l<16;l++) corr16[la]+=remapc[l]*remap[(l-la)&0xF];
                    int ar=0; for(int la=1;la<16;la++) if(corr16[la]>corr16[ar]) ar=la;
                    if(!ar) continue;
                    double gain_sub=0.0;
                    for(int v=0;v<256;v++) if((v>>4)==nib){
                        int nv=(v&0xF0)|((v+ar)&0xF);
                        gain_sub+=hlog[total[nv]-phF[nv]+phF[v]]-hlog[total[nv]];
                        gain_sub+=hlog[total[v]-phF[v]+phF[nv]]-hlog[total[v]];}
                    double ncla=gain_sub-INSTR_OHD(8);
                    if(ncla>bestNet){bestNet=ncla;best=(Instr){COND_LO_ADD,stride,phase,(unsigned)(nib<<4)|ar};}}}
            {for(int nib=0;nib<16;nib++){
                    int cf[256]={0}; for(int v=0;v<256;v++) if((v&0xF)==nib) cf[v]=phF[v];
                    int remap[16]={0},remapc[16]={0};
                    for(int v=0;v<256;v++) if((v&0xF)==nib){remap[v>>4]+=cf[v];remapc[v>>4]+=total[v]-cf[v];}
                    int corr16[16]={0}; for(int ha=0;ha<16;ha++) for(int h=0;h<16;h++) corr16[ha]+=remapc[h]*remap[(h-ha)&0xF];
                    int ar=0; for(int ha=1;ha<16;ha++) if(corr16[ha]>corr16[ar]) ar=ha;
                    if(!ar) continue;
                    double gain_sub=0.0;
                    for(int v=0;v<256;v++) if((v&0xF)==nib){
                        int nv=((((v>>4)+ar)&0xF)<<4)|(v&0xF);
                        gain_sub+=hlog[total[nv]-phF[nv]+phF[v]]-hlog[total[nv]];
                        gain_sub+=hlog[total[v]-phF[v]+phF[nv]]-hlog[total[v]];}
                    double ncha=gain_sub-INSTR_OHD(8);
                    if(ncha>bestNet){bestNet=ncha;best=(Instr){COND_HI_ADD,stride,phase,(unsigned)((ar<<4)|nib)};}}}
            /* NIB_SWAP */ { int nsfreq[256]={0};
                for(int i=phase;i<n;i+=stride) nsfreq[((data[i]<<4)&0xF0)|(data[i]>>4)]++;
                int ns_tot[256]; for(int v=0;v<256;v++) ns_tot[v]=total[v]-phF[v]+nsfreq[v];
                double Sns=0.0; for(int v=0;v<256;v++) Sns+=hlog[ns_tot[v]];
                double nns=(Sns-Sbase)-INSTR_OHD(0);
                if(nns>bestNet){bestNet=nns;best=(Instr){NIB_SWAP,stride,phase,0};}}
            /* BIT_ROTATE */ { for(int k=1;k<8;k++){
                int brfreq[256]={0};
                for(int i=phase;i<n;i+=stride) brfreq[((data[i]<<k)|(data[i]>>(8-k)))&0xFF]++;
                int br_tot[256]; for(int v=0;v<256;v++) br_tot[v]=total[v]-phF[v]+brfreq[v&0xFF];
                double Sbr=0.0; for(int v=0;v<256;v++) Sbr+=hlog[br_tot[v]];
                double nbr=(Sbr-Sbase)-INSTR_OHD(3);
                if(nbr>bestNet){bestNet=nbr;best=(Instr){BIT_ROTATE,stride,phase,(unsigned)k};}}}
            /* VALUE_XOR */
            /* Group 0: bytes with bit k=0; Group 1: bytes with bit k=1.
               Both amps must have bit k=0 so groups never cross — the transform
               is then self-inverse (XOR twice = identity). */
            { for(int k=0;k<8;k++){
                int G0[128]={0},G1[128]={0},bg0[128]={0},bg1[128]={0};
                int mask=1<<k;
                for(int v=0;v<256;v++){
                    int hi=rbk(v,k);  /* 7-bit index in the half-space */
                    if(v&mask){G1[hi]+=phF[v];bg1[hi]+=total[v]-phF[v];}
                    else       {G0[hi]+=phF[v];bg0[hi]+=total[v]-phF[v];}}
                double S0,S1;
                int ha0=xor_best_amp_128(bg0,G0,&S0);
                int ha1=xor_best_amp_128(bg1,G1,&S1);
                /* restore to 8-bit amps with bit k=0 */
                int alo=vxor_ins(ha0,k), ahi=vxor_ins(ha1,k);
                double Svx=(S0+S1)-Sbase-INSTR_OHD(17);
                if(Svx>bestNet){bestNet=Svx;best=(Instr){VALUE_XOR,stride,phase,
                    (unsigned)(k|(alo<<3)|(ahi<<11))};}} }
            /* DUAL XOR/ADD/MUL (shared histograms) */ { int lo_phF[256]={0},hi_phF[256]={0};
                int kk=0; for(int i=phase;i<n;i+=stride,kk++){if(kk&1)hi_phF[data[i]]++;else lo_phF[data[i]]++;}
                int lo_dv[256],hi_dv[256]; for(int v=0;v<256;v++){lo_dv[v]=total[v]-lo_phF[v];hi_dv[v]=total[v]-hi_phF[v];}
                double Slo,Shi;
                { int alo=xor_best_amp(lo_dv,lo_phF,&Slo),ahi=xor_best_amp(hi_dv,hi_phF,&Shi);
                  double ndx=(Slo-Sbase)+(Shi-Sbase)-INSTR_OHD(16);
                  if(ndx>bestNet){bestNet=ndx;best=(Instr){DUAL_XOR,stride,phase,(unsigned)(alo|(ahi<<8))};} }
                { int alo=add_best_amp(lo_dv,lo_phF,6,&Slo),ahi=add_best_amp(hi_dv,hi_phF,6,&Shi);
                  double nda=(Slo-Sbase)+(Shi-Sbase)-INSTR_OHD(16);
                  if(nda>bestNet){bestNet=nda;best=(Instr){DUAL_ADD,stride,phase,(unsigned)(alo|(ahi<<8))};} }
                { int idx_lo=(mul_best_amp(lo_dv,lo_phF,&Slo)-3)/2,idx_hi=(mul_best_amp(hi_dv,hi_phF,&Shi)-3)/2;
                  double ndm=(Slo-Sbase)+(Shi-Sbase)-INSTR_OHD(14);
                  if(ndm>bestNet){bestNet=ndm;best=(Instr){DUAL_MUL,stride,phase,(unsigned)(idx_lo|(idx_hi<<7))};} } }
            /* TRIPLE XOR/ADD/MUL (shared histograms) */ if(!g_skip_quad){
                int ph3[3][256]={{0},{0},{0}};
                int kk=0; for(int i=phase;i<n;i+=stride,kk++) ph3[kk%3][data[i]]++;
                int dv3[3][256]; for(int r=0;r<3;r++) for(int v=0;v<256;v++) dv3[r][v]=total[v]-ph3[r][v];
                double S3[3]; int a3[3];
                for(int r=0;r<3;r++) a3[r]=xor_best_amp(dv3[r],ph3[r],&S3[r]);
                double nt3=(S3[0]-Sbase)+(S3[1]-Sbase)+(S3[2]-Sbase)-INSTR_OHD_NOPHASE(3*8);
                if(nt3>bestNet){bestNet=nt3;best=(Instr){TRIPLE_XOR,stride,phase,(unsigned)(a3[0]|(a3[1]<<8)|(a3[2]<<16))};}
                for(int r=0;r<3;r++) a3[r]=add_best_amp(dv3[r],ph3[r],6,&S3[r]);
                nt3=(S3[0]-Sbase)+(S3[1]-Sbase)+(S3[2]-Sbase)-INSTR_OHD_NOPHASE(3*8);
                if(nt3>bestNet){bestNet=nt3;best=(Instr){TRIPLE_ADD,stride,phase,(unsigned)(a3[0]|(a3[1]<<8)|(a3[2]<<16))};}
                int idx3[3];
                for(int r=0;r<3;r++){double Sm; idx3[r]=(mul_best_amp(dv3[r],ph3[r],&Sm)-3)/2;S3[r]=Sm;}
                nt3=(S3[0]-Sbase)+(S3[1]-Sbase)+(S3[2]-Sbase)-INSTR_OHD_NOPHASE(3*7);
                if(nt3>bestNet){bestNet=nt3;best=(Instr){TRIPLE_MUL,stride,phase,(unsigned)(idx3[0]|(idx3[1]<<7)|(idx3[2]<<14))};}}
            if(!g_skip_quad){ /* QUAD XOR/ADD/MUL (shared histograms) */
                int ph4[4][256]={{0},{0},{0},{0}};
                int kk=0; for(int i=phase;i<n;i+=stride,kk++) ph4[kk&3][data[i]]++;
                int dv4[4][256]; for(int r=0;r<4;r++) for(int v=0;v<256;v++) dv4[r][v]=total[v]-ph4[r][v];
                double S4[4]; unsigned a4[4];
                for(int r=0;r<4;r++) a4[r]=(unsigned)xor_best_amp(dv4[r],ph4[r],&S4[r]);
                double nq4=(S4[0]-Sbase)+(S4[1]-Sbase)+(S4[2]-Sbase)+(S4[3]-Sbase)-INSTR_OHD(4*8);
                if(nq4>bestNet){bestNet=nq4;best=(Instr){QUAD_XOR,stride,phase,a4[0]|(a4[1]<<8)|(a4[2]<<16)|(a4[3]<<24)};}
                for(int r=0;r<4;r++) a4[r]=(unsigned)add_best_amp(dv4[r],ph4[r],6,&S4[r]);
                nq4=(S4[0]-Sbase)+(S4[1]-Sbase)+(S4[2]-Sbase)+(S4[3]-Sbase)-INSTR_OHD(4*8);
                if(nq4>bestNet){bestNet=nq4;best=(Instr){QUAD_ADD,stride,phase,a4[0]|(a4[1]<<8)|(a4[2]<<16)|(a4[3]<<24)};}
                int idx4[4];
                for(int r=0;r<4;r++){double Sm; idx4[r]=(mul_best_amp(dv4[r],ph4[r],&Sm)-3)/2;S4[r]=Sm;}
                nq4=(S4[0]-Sbase)+(S4[1]-Sbase)+(S4[2]-Sbase)+(S4[3]-Sbase)-INSTR_OHD(4*7);
                if(nq4>bestNet){bestNet=nq4;best=(Instr){QUAD_MUL,stride,phase,(unsigned)(idx4[0]|(idx4[1]<<7)|(idx4[2]<<14)|(idx4[3]<<21))};}}
            }
        }
    }
    *netOut = bestNet;
    return best;
}

/* ── global instruction list ────────────────────────────────────────────── */
static Instr g_ilist[4096];
static int   g_ni;

/* ── decompress (invert instruction list in reverse order) ──────────────── */
static int do_decompress(u8 *data, int n, const Instr *instrs, int ni) {
    for (int ii = ni-1; ii >= 0; ii--) {
        Instr t = instrs[ii];
        switch(t.type) {
        case SCRAMBLE: invert_scramble(data,n,t.amp&0xF); break;
        case COND_PREV_XOR: { int alo=t.amp&0xFF,ahi=(t.amp>>8)&0xFF;
            int last_ii=t.phase; for(int i=t.phase;i<n;i+=t.stride) last_ii=i;
            for(int i=last_ii;i>=t.phase+t.stride;i-=t.stride){
                int pv=data[i-t.stride]; data[i]^=(u8)((pv>=128)?ahi:alo);} break; }
        case COND_PREV_ADD: { int alo=t.amp&0xFF,ahi=(t.amp>>8)&0xFF;
            int last_ii=t.phase; for(int i=t.phase;i<n;i+=t.stride) last_ii=i;
            for(int i=last_ii;i>=t.phase+t.stride;i-=t.stride){
                int pv=data[i-t.stride]; data[i]=(u8)((data[i]-((pv>=128)?ahi:alo))&0xFF);} break; }
        case POLY_DELTA_XOR: { int s1=t.stride,s2=t.phase;
            for(int i=s2;i<n;i++) data[i]^=data[i-s1]^data[i-s2]; break; }
        case XOR_PHASE: applyInstr(data,n,t); break;
        case ADD_PHASE: applyInstr(data,n,(Instr){ADD_PHASE,t.stride,t.phase,(256-t.amp)&0xFF}); break;
        case PACK_XOR:  applyInstr(data,n,t); break;
        case LAG_XOR: { int lag=(int)t.amp;
            for(int i=n-lag-1;i>=0;i--) data[i]^=data[i+lag]; break; }
        case LAG_ADD: { int lag=(int)t.amp;
            for(int i=n-lag-1;i>=0;i--) data[i]=(u8)((data[i]-data[i+lag])&0xFF); break; }
        case ADD_NIBS: { int lo=t.amp&0xF,hi=(t.amp>>4)&0xF;
            applyInstr(data,n,(Instr){ADD_NIBS,t.stride,t.phase,(unsigned)(((-hi)&0xF)<<4)|((-lo)&0xF)}); break; }
        case NIB_SWAP: applyInstr(data,n,t); break;
        case BIT_ROTATE: { int k=t.amp&7;
            applyInstr(data,n,(Instr){BIT_ROTATE,t.stride,t.phase,(8-k)&7}); break; }
        case VALUE_XOR: applyInstr(data,n,t); break;
        case COND_LO_XOR: applyInstr(data,n,t); break;
        case COND_HI_XOR: applyInstr(data,n,t); break;
        case COND_LO_ADD: { int nc=(t.amp>>4)&0xF,av=t.amp&0xF;
            applyInstr(data,n,(Instr){COND_LO_ADD,t.stride,t.phase,(unsigned)((nc<<4)|((-av)&0xF))}); break; }
        case COND_HI_ADD: { int nc=t.amp&0xF,av=(t.amp>>4)&0xF;
            applyInstr(data,n,(Instr){COND_HI_ADD,t.stride,t.phase,(unsigned)(nc|((-av&0xF)<<4))}); break; }
        case DUAL_XOR: applyInstr(data,n,t); break;
        case DUAL_ADD: { int lo=t.amp&0xFF,hi=(t.amp>>8)&0xFF;
            applyInstr(data,n,(Instr){DUAL_ADD,t.stride,t.phase,((256-lo)&0xFF)|(((256-hi)&0xFF)<<8)}); break; }
        case DUAL_MUL: { int il=t.amp&0x7F,ih=(t.amp>>7)&0x7F;
            int ml=il*2+3,mh=ih*2+3;
            int invl=(mul_inv[ml]-3)/2,invh=(mul_inv[mh]-3)/2;
            applyInstr(data,n,(Instr){DUAL_MUL,t.stride,t.phase,(unsigned)(invl|(invh<<7))}); break; }
        case TRIPLE_XOR: applyInstr(data,n,t); break;
        case TRIPLE_ADD: { int a0=t.amp&0xFF,a1=(t.amp>>8)&0xFF,a2=(t.amp>>16)&0xFF;
            applyInstr(data,n,(Instr){TRIPLE_ADD,t.stride,t.phase,
                ((256-a0)&0xFF)|(((256-a1)&0xFF)<<8)|(((256-a2)&0xFF)<<16)}); break; }
        case TRIPLE_MUL: { int i0=t.amp&0x7F,i1=(t.amp>>7)&0x7F,i2=(t.amp>>14)&0x7F;
            int m0=i0*2+3,m1=i1*2+3,m2=i2*2+3;
            int v0=(mul_inv[m0]-3)/2,v1=(mul_inv[m1]-3)/2,v2=(mul_inv[m2]-3)/2;
            applyInstr(data,n,(Instr){TRIPLE_MUL,t.stride,t.phase,(unsigned)(v0|(v1<<7)|(v2<<14))}); break; }
        case QUAD_XOR: applyInstr(data,n,t); break;
        case QUAD_ADD: { unsigned a0=t.amp&0xFF,a1=(t.amp>>8)&0xFF,a2=(t.amp>>16)&0xFF,a3=t.amp>>24;
            applyInstr(data,n,(Instr){QUAD_ADD,t.stride,t.phase,
                ((256-a0)&0xFF)|(((256-a1)&0xFF)<<8)|(((256-a2)&0xFF)<<16)|(((256-a3)&0xFF)<<24)}); break; }
        case QUAD_MUL: { int i0=t.amp&0x7F,i1=(t.amp>>7)&0x7F,i2=(t.amp>>14)&0x7F,i3=(t.amp>>21)&0x7F;
            int m0=i0*2+3,m1=i1*2+3,m2=i2*2+3,m3=i3*2+3;
            int v0=(mul_inv[m0]-3)/2,v1=(mul_inv[m1]-3)/2,v2=(mul_inv[m2]-3)/2,v3=(mul_inv[m3]-3)/2;
            applyInstr(data,n,(Instr){QUAD_MUL,t.stride,t.phase,(unsigned)(v0|(v1<<7)|(v2<<14)|(v3<<21))}); break; }
        case CTX4_XOR: { u8 a[4]={(u8)(t.amp&0xFF),(u8)((t.amp>>8)&0xFF),(u8)((t.amp>>16)&0xFF),(u8)(t.amp>>24)};
            for(int i=n-1;i>=0;i--){u8 prev=(i>0)?data[i-1]:0; data[i]^=a[(prev>>6)&3];} break; }
        case CTX4_ADD: { u8 a[4]={(u8)(t.amp&0xFF),(u8)((t.amp>>8)&0xFF),(u8)((t.amp>>16)&0xFF),(u8)(t.amp>>24)};
            for(int i=n-1;i>=0;i--){u8 prev=(i>0)?data[i-1]:0; data[i]=(u8)((data[i]-a[(prev>>6)&3])&0xFF);} break; }
        default: break;
        }
    }
    return 1;
}

/* ── chunk pattern analysis ──────────────────────────────────────────────── */
/*
 * Profile all known pattern categories.  For each, estimates the net-bit gain
 * of the best instruction in that category WITHOUT applying it.  Also verifies
 * the overall best instruction on a scratch copy (same e1>=e0 guard as
 * compress_greedy) to separate real structure from scoring noise.
 *
 * Categories:
 *   lag/delta  — LAG_XOR/ADD (lags 1..64), PACK_XOR, POLY_DELTA_XOR
 *   periodic   — TRIPLE XOR/ADD per stride 1..ANALYSE_TRI_MAX, top 5 shown
 *   val-cond   — COND_LO/HI XOR+ADD (strides 1..8)
 *   markov     — CTX4_XOR/ADD, COND_PREV_XOR (strides 1..4)
 *   bit/nibble — BIT_ROTATE, NIB_SWAP, ADD_NIBS (strides 1..8)
 */
static void analyse_chunk(const u8 *data, int n, int cidx) {
    double H = entropy(data, n);
    int total[256]={0};
    for(int i=0;i<n;i++) total[data[i]]++;
    double Sbase=0; for(int v=0;v<256;v++) Sbase+=hlog[total[v]];
    set_freq_cache(total);

    /* ── overall best + verification ─────────────────────────────── */
    double estNet=0;
    Instr best = findBest(data, n, &estNet, ANALYSE_MAX_STRIDE);
    double verNet=0;
    if(estNet>0){
        u8 *tmp=malloc(n); memcpy(tmp,data,n);
        applyInstr(tmp,n,best);
        double e1=entropy(tmp,n); free(tmp);
        if(e1<H) verNet=(H-e1)*n;
    }
    const char *type = verNet>50?"STRONG-STRIDE":verNet>10?"STRIDE":
                       verNet>0?"WEAK":estNet>0?"NOISE":"FLAT";

    /* ── category 1: lag/delta ────────────────────────────────────── */
    double lag_xor=0,lag_add=0; int lag_xk=0,lag_ak=0;
    for(int lag=1;lag<=64;lag++){
        int f[256];
        /* LAG_XOR */
        for(int v=0;v<256;v++) f[v]=total[v];
        for(int i=0;i+lag<n;i++){f[data[i]]--;f[data[i]^data[i+lag]]++;}
        double S=0; for(int v=0;v<256;v++) S+=hlog[f[v]];
        double g=(S-Sbase)-INSTR_OHD_NOPHASE(8);
        if(g>lag_xor){lag_xor=g;lag_xk=lag;}
        /* LAG_ADD */
        for(int v=0;v<256;v++) f[v]=total[v];
        for(int i=0;i+lag<n;i++){f[data[i]]--;f[(data[i]+data[i+lag])&0xFF]++;}
        S=0; for(int v=0;v<256;v++) S+=hlog[f[v]];
        g=(S-Sbase)-INSTR_OHD_NOPHASE(8);
        if(g>lag_add){lag_add=g;lag_ak=lag;}
    }
    /* PACK_XOR: XOR adjacent elements at each even position within stride */
    double pack_xor=0; int pack_s=1;
    for(int s=1;s<=8;s++){
        int f2[256]; for(int v=0;v<256;v++) f2[v]=total[v];
        int k2=0;
        for(int i=0;i<n;i+=s,k2++){
            if(k2&1) continue;
            int j=i+s; if(j>=n) continue;
            f2[data[i]]--; f2[data[i]^data[j]]++;
        }
        double S=0; for(int v=0;v<256;v++) S+=hlog[f2[v]];
        double g=(S-Sbase)-INSTR_OHD(0);
        if(g>pack_xor){pack_xor=g;pack_s=s;}
    }
    /* POLY_DELTA_XOR: second-order XOR predictor over (s1, stride) */
    double pdx=0; int pdx_s1=1,pdx_str=2;
    for(int stride=2;stride<=8;stride++){
        for(int s1=1;s1<stride;s1++){
            int f2[256]={0};
            for(int i=0;i<stride&&i<n;i++) f2[data[i]]++;
            for(int i=stride;i<n;i++) f2[data[i]^data[i-s1]^data[i-stride]]++;
            double S=0; for(int v=0;v<256;v++) S+=hlog[f2[v]];
            double g=(S-Sbase)-INSTR_OHD(6);
            if(g>pdx){pdx=g;pdx_s1=s1;pdx_str=stride;}
        }
    }

    /* ── category 2: periodic (SINGLE/DUAL/TRIPLE/QUAD XOR+ADD+MUL, s1..N) ── */
    /* peri[s][k] = best gain for K=k+1 amps cycling at stride s.
       Overhead: K=1 uses INSTR_OHD(8)=20; K>=2 uses INSTR_OHD_NOPHASE(K*8)=5+K*8. */
    double peri[ANALYSE_TRI_MAX+1][4]; int periK[ANALYSE_TRI_MAX+1];
    for(int k=0;k<4;k++) for(int s=0;s<=ANALYSE_TRI_MAX;s++) peri[s][k]=0;
    for(int s=1;s<=ANALYSE_TRI_MAX;s++){
        for(int p=0;p<s;p++){
            for(int K=1;K<=4;K++){
                int phK[4][256]={{0},{0},{0},{0}};
                int kk=0; for(int i=p;i<n;i+=s,kk++) phK[kk%K][data[i]]++;
                int dvK[4][256];
                for(int r=0;r<K;r++) for(int v=0;v<256;v++) dvK[r][v]=total[v]-phK[r][v];
                double SK[4];
                int ohd = (K==1) ? INSTR_OHD(8) : INSTR_OHD_NOPHASE(K*8);
                /* XOR */
                double gx=-(double)ohd;
                for(int r=0;r<K;r++){xor_best_amp(dvK[r],phK[r],&SK[r]);gx+=SK[r]-Sbase;}
                if(gx>peri[s][K-1]) peri[s][K-1]=gx;
                /* ADD */
                double ga=-(double)ohd;
                for(int r=0;r<K;r++){add_best_amp(dvK[r],phK[r],6,&SK[r]);ga+=SK[r]-Sbase;}
                if(ga>peri[s][K-1]) peri[s][K-1]=ga;
                /* MUL */
                double gm=-(double)ohd;
                for(int r=0;r<K;r++){mul_best_amp(dvK[r],phK[r],&SK[r]);gm+=SK[r]-Sbase;}
                if(gm>peri[s][K-1]) peri[s][K-1]=gm;
            }
        }
    }
    /* per-stride: pick best K, record type */
    double perimax[ANALYSE_TRI_MAX+1];
    for(int s=0;s<=ANALYSE_TRI_MAX;s++){
        perimax[s]=0; periK[s]=1;
        for(int k=0;k<4;k++) if(peri[s][k]>perimax[s]){perimax[s]=peri[s][k];periK[s]=k+1;}
    }
    int pord[ANALYSE_TRI_MAX];
    for(int i=0;i<ANALYSE_TRI_MAX;i++) pord[i]=i+1;
    for(int i=0;i<5;i++) for(int j=i+1;j<ANALYSE_TRI_MAX;j++)
        if(perimax[pord[j]]>perimax[pord[i]]){int t=pord[i];pord[i]=pord[j];pord[j]=t;}

    /* ── category 3: value-conditional (COND_LO/HI XOR+ADD, s1..8) ── */
    /* Uses pairwise-swap gain formula matching findBest exactly.
       Each pair (v, nv) is counted twice in the inner loop — consistent with
       findBest's best_sub scale, so estimates are comparable to its output. */
    double clox=0,cloa=0,chix=0,chia=0;
    for(int s=1;s<=8;s++){
        for(int p=0;p<s;p++){
            int phF[256]={0};
            for(int i=p;i<n;i+=s) phF[data[i]]++;
            /* COND_LO_XOR: condition on high nibble, XOR low nibble */
            for(int nib=0;nib<16;nib++){
                double best=-1e30;
                for(int ar=1;ar<16;ar++){
                    double g=0;
                    for(int v=0;v<256;v++) if((v>>4)==nib){
                        int nv=(v&0xF0)|((v^ar)&0xF);
                        g+=hlog[total[nv]-phF[nv]+phF[v]]-hlog[total[nv]]
                          +hlog[total[v]-phF[v]+phF[nv]]-hlog[total[v]];
                    }
                    if(g>best) best=g;
                }
                double net=best-INSTR_OHD(8); if(net>clox) clox=net;
            }
            /* COND_LO_ADD: condition on high nibble, ADD to low nibble */
            for(int nib=0;nib<16;nib++){
                double best=-1e30;
                for(int ar=1;ar<16;ar++){
                    double g=0;
                    for(int v=0;v<256;v++) if((v>>4)==nib){
                        int nv=(v&0xF0)|((v+ar)&0xF);
                        g+=hlog[total[nv]-phF[nv]+phF[v]]-hlog[total[nv]]
                          +hlog[total[v]-phF[v]+phF[nv]]-hlog[total[v]];
                    }
                    if(g>best) best=g;
                }
                double net=best-INSTR_OHD(8); if(net>cloa) cloa=net;
            }
            /* COND_HI_XOR: condition on low nibble, XOR high nibble */
            for(int nib=0;nib<16;nib++){
                double best=-1e30;
                for(int ar=1;ar<16;ar++){
                    double g=0;
                    for(int v=0;v<256;v++) if((v&0xF)==nib){
                        int nv=((v^(ar<<4))&0xF0)|(v&0xF);
                        g+=hlog[total[nv]-phF[nv]+phF[v]]-hlog[total[nv]]
                          +hlog[total[v]-phF[v]+phF[nv]]-hlog[total[v]];
                    }
                    if(g>best) best=g;
                }
                double net=best-INSTR_OHD(8); if(net>chix) chix=net;
            }
            /* COND_HI_ADD: condition on low nibble, ADD to high nibble */
            for(int nib=0;nib<16;nib++){
                double best=-1e30;
                for(int ar=1;ar<16;ar++){
                    double g=0;
                    for(int v=0;v<256;v++) if((v&0xF)==nib){
                        int nv=(((v>>4)+ar)&0xF)<<4|(v&0xF);
                        g+=hlog[total[nv]-phF[nv]+phF[v]]-hlog[total[nv]]
                          +hlog[total[v]-phF[v]+phF[nv]]-hlog[total[v]];
                    }
                    if(g>best) best=g;
                }
                double net=best-INSTR_OHD(8); if(net>chia) chia=net;
            }
        }
    }

    /* ── category 4: Markov (CTX4 + COND_PREV, s1..4) ──────────────── */
    double ctx4x=0,ctx4a=0;
    {
        int phF4[4][256]={{0},{0},{0},{0}};
        u8 prevC=0;
        for(int i=0;i<n;i++){phF4[(prevC>>6)&3][data[i]]++;prevC=data[i];}
        int dv4[4][256];
        for(int c=0;c<4;c++) for(int v=0;v<256;v++) dv4[c][v]=total[v]-phF4[c][v];
        for(int c=0;c<4;c++){double Sc;xor_best_amp(dv4[c],phF4[c],&Sc);ctx4x+=Sc-Sbase;}
        ctx4x-=INSTR_OHD(32);
        for(int c=0;c<4;c++){double Sc;add_best_amp(dv4[c],phF4[c],6,&Sc);ctx4a+=Sc-Sbase;}
        ctx4a-=INSTR_OHD(32);
    }
    double cpx=0,cpa=0; /* COND_PREV_XOR/ADD best over s1..4 */
    for(int s=1;s<=4;s++){
        for(int p=0;p<s;p++){
            int ph0[256]={0},ph1[256]={0};
            int pv2=0,first=1;
            for(int i=p;i<n;i+=s){
                if(first){first=0;pv2=data[i];continue;}
                if(pv2<128) ph0[data[i]]++; else ph1[data[i]]++;
                pv2=data[i];
            }
            int dv0[256],dv1[256];
            for(int v=0;v<256;v++){dv0[v]=total[v]-ph0[v];dv1[v]=total[v]-ph1[v];}
            double Sx0,Sx1;
            xor_best_amp(dv0,ph0,&Sx0); xor_best_amp(dv1,ph1,&Sx1);
            double gx=(Sx0-Sbase)+(Sx1-Sbase)-INSTR_OHD(16);
            if(gx>cpx) cpx=gx;
            add_best_amp(dv0,ph0,6,&Sx0); add_best_amp(dv1,ph1,6,&Sx1);
            double ga=(Sx0-Sbase)+(Sx1-Sbase)-INSTR_OHD(16);
            if(ga>cpa) cpa=ga;
        }
    }

    /* ── category 5: bit/nibble (BIT_ROTATE, NIB_SWAP, ADD_NIBS, s1..8) ── */
    double bitr=0,nibsw=0,addnibs=0;
    for(int s=1;s<=8;s++){
        for(int p=0;p<s;p++){
            int phF[256]={0};
            for(int i=p;i<n;i+=s) phF[data[i]]++;
            int dv[256]; for(int v=0;v<256;v++) dv[v]=total[v]-phF[v];
            /* ADD_NIBS */
            {double Sn; add_nibs_best_amp(dv,phF,&Sn);
             double g=(Sn-Sbase)-INSTR_OHD(8); if(g>addnibs) addnibs=g;}
            /* NIB_SWAP */
            {int nsfreq[256]={0};
             for(int i=p;i<n;i+=s) nsfreq[((data[i]<<4)&0xF0)|(data[i]>>4)]++;
             int ns_tot[256]; for(int v=0;v<256;v++) ns_tot[v]=dv[v]+nsfreq[v];
             double Sns=0; for(int v=0;v<256;v++) Sns+=hlog[ns_tot[v]];
             double g=(Sns-Sbase)-INSTR_OHD(0); if(g>nibsw) nibsw=g;}
            /* BIT_ROTATE: try all 7 rotations */
            for(int k=1;k<8;k++){
                int brfreq[256]={0};
                for(int i=p;i<n;i+=s) brfreq[((data[i]<<k)|(data[i]>>(8-k)))&0xFF]++;
                int br_tot[256]; for(int v=0;v<256;v++) br_tot[v]=dv[v]+brfreq[v];
                double Sbr=0; for(int v=0;v<256;v++) Sbr+=hlog[br_tot[v]];
                double g=(Sbr-Sbase)-INSTR_OHD(3); if(g>bitr) bitr=g;}
        }
    }

    /* ── category 6: positional entropy gradient ─────────────────────── */
    double hwin[4]; int wsz=n/4;
    for(int w=0;w<4;w++) hwin[w]=entropy(data+w*wsz,wsz);

    /* ── category 7: second-order delta residuals ────────────────────── */
    /* XOR residual: data[i] ^ data[i-1] ^ data[i-2]  (order-2 XOR predictor)
       ADD residual: data[i] - 2*data[i-1] + data[i-2] mod 256 (2nd difference)
       H lower than raw = 2nd-order structure present; equal = none beyond 1st-order. */
    double h_d2x, h_d2a;
    {
        int fx[256]={0}, fa[256]={0};
        for(int i=2;i<n;i++){
            fx[data[i]^data[i-1]^data[i-2]]++;
            fa[((int)data[i]-2*(int)data[i-1]+(int)data[i-2]+512)&0xFF]++;
        }
        double m=(double)(n-2), Sx=0, Sa=0;
        for(int v=0;v<256;v++){
            if(fx[v]) Sx-=fx[v]*log2(fx[v]/m);
            if(fa[v]) Sa-=fa[v]*log2(fa[v]/m);
        }
        h_d2x=Sx; h_d2a=Sa; /* total bits; divide by (n-2) for bpb */
    }

    /* ── category 8: bigram (Markov MI estimate) ─────────────────────── */
    /* MI = H(X) + H(Y) - H(X,Y) ≈ 2H - H_pair_bpb.
       Upward bias ≈ 4 bits total (singletons in 256x256 table at n=4096). */
    double h_pair_bpb, markov_mi;
    {
        static int f2[256][256]; memset(f2,0,sizeof(f2));
        for(int i=1;i<n;i++) f2[data[i-1]][data[i]]++;
        double m=(double)(n-1), S2=0;
        for(int a=0;a<256;a++) for(int b=0;b<256;b++)
            if(f2[a][b]) S2-=f2[a][b]*log2((double)f2[a][b]/m);
        h_pair_bpb=S2/m;
        markov_mi=2.0*H-h_pair_bpb;
    }

    /* ── category 9: bit-plane entropy ──────────────────────────────── */
    /* H per bit position (max=1.0=flat). Deviation reveals XOR-exploitable
       structure at specific bit positions. */
    double bit_h[8];
    for(int k=0;k<8;k++){
        int ones=0;
        for(int v=0;v<256;v++) if((v>>k)&1) ones+=total[v];
        double p=(double)ones/n;
        bit_h[k]=(p>0&&p<1)?-(p*log2(p)+(1-p)*log2(1-p)):0.0;
    }

    /* ── category 10: run-length statistics ─────────────────────────── */
    /* avg_run: geometric mean run length. Random uniform ≈ 1.004 (256/255).
       max_run: expected 2-3 for n=4096 random; long runs = structured data. */
    double avg_run; int max_run;
    {
        int nruns=0, cur=1, mx=1;
        for(int i=1;i<n;i++){
            if(data[i]==data[i-1]) cur++;
            else {nruns++;if(cur>mx)mx=cur;cur=1;}
        }
        nruns++;if(cur>mx)mx=cur;
        max_run=mx; avg_run=(double)n/nruns;
    }

    /* ── print ──────────────────────────────────────────────────────── */
    printf("[chunk %d | H=%.4f | %s]\n",cidx,H,type);
    if(estNet>0)
        printf("  best:  %-16s s%-2d  est=%+.0f  verified=%+.0f bits\n",
               INSTR_NAMES[best.type],best.stride,estNet,verNet);
    else
        printf("  best:  (none > 0 bits)\n");
    printf("\n");
    /* lag/delta */
    printf("  lag/delta  : ");
    if(lag_xor>0) printf("XOR L%d=%+.0f  ",lag_xk,lag_xor); else printf("XOR=(none)  ");
    if(lag_add>0) printf("ADD L%d=%+.0f  ",lag_ak,lag_add); else printf("ADD=(none)  ");
    if(pack_xor>0) printf("PACK s%d=%+.0f  ",pack_s,pack_xor); else printf("PACK=(none)  ");
    if(pdx>0) printf("PDX %d/%d=%+.0f",pdx_s1,pdx_str,pdx); else printf("PDX=(none)");
    printf("\n");
    /* periodic: show top-5 strides with type S/D/T/Q */
    {static const char klab[]="SDTQ";
    printf("  periodic   : ");
    int any=0;
    for(int i=0;i<5;i++) if(perimax[pord[i]]>0){
        printf("%c:s%d=%+.0f  ",klab[periK[pord[i]]-1],pord[i],perimax[pord[i]]);any=1;}
    if(!any) printf("(none above 0 bits)");
    printf("  (S/D/T/Q XOR+ADD+MUL s1..%d)\n",ANALYSE_TRI_MAX);}
    /* val-cond */
    printf("  val-cond   : LO_XOR=%+.0f  LO_ADD=%+.0f  HI_XOR=%+.0f  HI_ADD=%+.0f  (s1..8)\n",
           clox,cloa,chix,chia);
    /* markov */
    printf("  markov     : CTX4_XOR=%+.0f  CTX4_ADD=%+.0f  PREV_XOR=%+.0f  PREV_ADD=%+.0f\n",
           ctx4x,ctx4a,cpx,cpa);
    /* bit/nibble */
    printf("  bit/nibble : BIT_ROTATE=%+.0f  NIB_SWAP=%+.0f  ADD_NIBS=%+.0f  (s1..8)\n",
           bitr,nibsw,addnibs);
    /* positional gradient */
    printf("  pos-grad   : H[0..%d]=%.4f  H[%d..%d]=%.4f  H[%d..%d]=%.4f  H[%d..%d]=%.4f\n",
           wsz-1,hwin[0], wsz,2*wsz-1,hwin[1], 2*wsz,3*wsz-1,hwin[2], 3*wsz,n-1,hwin[3]);
    /* second-order delta residuals */
    printf("  delta2     : XOR=%.4f  ADD=%.4f bpb  (H of 2nd-order residual; raw H=%.4f)\n",
           h_d2x/(n-2), h_d2a/(n-2), H);
    /* bigram Markov MI */
    printf("  bigram     : H_pair=%.4f bpb  MI≈%+.4f bpb  (~4 bit noise bias at n=%d)\n",
           h_pair_bpb, markov_mi, n);
    /* bit-plane entropy */
    printf("  bit-planes : b7=%.3f b6=%.3f b5=%.3f b4=%.3f b3=%.3f b2=%.3f b1=%.3f b0=%.3f\n",
           bit_h[7],bit_h[6],bit_h[5],bit_h[4],bit_h[3],bit_h[2],bit_h[1],bit_h[0]);
    /* run-length */
    printf("  run-len    : avg=%.3f  max=%d  (random: avg≈1.004  max≈2-3)\n",
           avg_run, max_run);
    printf("\n");
}

/* ── greedy compress (simplified: no try_scramble for speed) ─────────────── */
static double compress_greedy(u8 *data, int n, int verbose, int *counts) {
    double total_net = 0.0;
    g_ni = 0;
    for (;;) {
        if (g_ni >= 4094) break;
        double net;
        Instr t = findBest(data, n, &net, 64);
        if (net <= 0.0) break;
        double e0 = entropy(data, n);
        applyInstr(data, n, t);
        double e1 = entropy(data, n);
        /* reject if entropy didn't actually decrease (model error protection) */
        if (e1 >= e0) { do_decompress(data, n, &t, 1); break; }
        g_ilist[g_ni++] = t;
        total_net += (e0 - e1) * n;  /* use actual bit reduction */
        if (counts) counts[t.type]++;
        if (verbose)
            printf("    %-14s s%-2d p%-2d a%-8u  %.6f -> %.6f  net=%.1f\n",
                   INSTR_NAMES[t.type], t.stride, t.phase, t.amp,
                   e0, e1, (e0 - e1) * n);
    }
    return total_net;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  MAIN
 * ═══════════════════════════════════════════════════════════════════════════ */
int main(void) {
    setbuf(stdout, NULL);
    init_hlog();
    init_mul_inv();
    init_fft256();
    if (ANA_MODE) init_hlogf();

    /* ── load whole file (or generate random data) ───────────────────── */
    int   raw_size = 0;
    u8   *raw_data = NULL;

    FILE *f = fopen(INPUT_FILE, "rb");
    if (f) {
        /* cap input so we don't try to read a giant file */
        int cap = MAX_CHUNKS * BLOCK_SIZE;
        raw_data = malloc(cap);
        if (!raw_data) { fprintf(stderr, "OOM\n"); return 1; }
        raw_size = (int)fread(raw_data, 1, cap, f);
        if (raw_size <= 0) { fprintf(stderr, "read error or empty file\n"); return 1; }
        fclose(f);
    } else {
        fprintf(stderr, "error: input file '%s' not found\n", INPUT_FILE);
        return 1;
    }

    printf("Input : %d bytes  H_raw=%.6f bpb\n\n",
           raw_size, entropy(raw_data, raw_size));

    /* ── ANS compress the whole file at once ─────────────────────────── */
    int ans_cap    = raw_size * 2 + 4096;
    u8 *ans_stream = malloc(ans_cap);
    if (!ans_stream) { fprintf(stderr, "OOM\n"); return 1; }

    int ans_len = o0_compress(raw_data, raw_size, ans_stream, ans_cap);

    printf("ANS   : %d → %d bytes  H_ans=%.6f bpb\n\n",
           raw_size, ans_len, entropy(ans_stream, ans_len));

    /* pristine copy for verification */
    u8 *ans_pristine = malloc(ans_len);
    if (!ans_pristine) { fprintf(stderr, "OOM\n"); return 1; }
    memcpy(ans_pristine, ans_stream, ans_len);

#if SINGLE_BLOCK
    /* ── Option A: entropy-reduce the full ANS output as one block ───── */
    double H_ans_full = entropy(ans_stream, ans_len);
    printf("H_ans=%.6f bpb  size=%d bytes\n\n", H_ans_full, ans_len);

    if (ANALYSE) analyse_chunk(ans_stream, ans_len, 0);
    if (VERBOSE) printf("[instruction detail]\n");

    int total_counts[NUM_INSTR_TYPES] = {0};
    double net = compress_greedy(ans_stream, ans_len, VERBOSE, total_counts);
    if (VERBOSE) printf("\n");

    double H_red = entropy(ans_stream, ans_len);
    int ni = g_ni;

    /* round-trip: invert all instructions → should recover ans_pristine */
    u8 *ans_restored = malloc(ans_len);
    memcpy(ans_restored, ans_stream, ans_len);
    do_decompress(ans_restored, ans_len, g_ilist, ni);
    int ans_match = (memcmp(ans_restored, ans_pristine, ans_len) == 0);

    /* ANS decompress to check full pipeline */
    u8 *raw_check = malloc(raw_size);
    o0_decompress(ans_restored, ans_len, raw_check, raw_size);
    int raw_match = (memcmp(raw_check, raw_data, raw_size) == 0);

    printf("H_ans    : %.6f bpb\n", H_ans_full);
    printf("H_red    : %.6f bpb\n", H_red);
    printf("delta-H  : %+.6f bpb\n", H_red - H_ans_full);
    printf("net      : %+.1f bits  (%+.4f bpb)\n",
           net, net / ans_len);
    printf("ni       : %d instructions\n", ni);
    printf("ANS restore    : %s\n", ans_match ? "ok" : "FAIL");
    printf("full round-trip: %s\n\n", raw_match ? "ok" : "FAIL");

    printf("instruction usage:\n");
    for (int i = 0; i < NUM_INSTR_TYPES; i++)
        if (total_counts[i] > 0)
            printf("  %-16s %d\n", INSTR_NAMES[i], total_counts[i]);

    free(ans_restored);
    free(raw_check);

#else
    /* ── CTX256_XOR pre-pass on the full ANS stream ───────────────────── */
    u8  ctx256_amp[256];
    int has_ctx256 = 0;
    {
        double H_pre = entropy(ans_stream, ans_len);
        double cn = find_best_ctx256_xor(ans_stream, ans_len, ctx256_amp);
        printf("CTX256_XOR  n=%-7d  H=%.6f  net=%+.1f bits (raw=%+.1f key=2048)",
               ans_len, H_pre, cn, cn + 2048.0);
        if (cn > 0.0) {
            apply_ctx256_xor(ans_stream, ans_len, ctx256_amp);
            has_ctx256 = 1;
            memcpy(ans_pristine, ans_stream, ans_len);
            printf("  [applied]\n\n");
        } else {
            printf("  [skipped]\n\n");
        }
    }

    /* ── chunked mode: split ANS stream into BLOCK_SIZE chunks ──────── */
    int n_chunks = (ans_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    if (n_chunks > MAX_CHUNKS) n_chunks = MAX_CHUNKS;

    Instr (*ilists)[4096] = malloc(n_chunks * sizeof(*ilists));
    int   *ni_list          = calloc(n_chunks, sizeof(int));
    if (!ilists || !ni_list) { fprintf(stderr, "OOM\n"); return 1; }

    int    total_counts[NUM_INSTR_TYPES] = {0};
    double total_net = 0.0, total_H_ans = 0.0, total_H_red = 0.0;
    double total_ana_net = 0.0;
    int    chunks_ok = 0, chunks_fail = 0;

    RoundInfo (*rounds)[ANA_MAX_INTERLEAVE] =
        calloc(n_chunks, ANA_MAX_INTERLEAVE * sizeof(RoundInfo));
    int *nrounds_arr = calloc(n_chunks, sizeof(int));
    if (!rounds || !nrounds_arr) { fprintf(stderr, "OOM\n"); return 1; }

    printf("%-7s  %-9s  %-7s  %-9s  %-9s  %-10s  %-5s  %-10s  %-4s  %-3s  %s\n",
           "chunk", "H_ans", "csz", "H_red", "delta-H", "net(bits)", "ni",
           "ana_net", "rnd", "ok?", "top instr");
    printf("%-7s  %-9s  %-7s  %-9s  %-9s  %-10s  %-5s  %-10s  %-4s  %-3s  %s\n",
           "-------", "---------", "-------", "---------", "---------",
           "----------", "-----", "----------", "----", "---", "---------");

    for (int c = 0; c < n_chunks; c++) {
        int off = c * BLOCK_SIZE;
        int csz = (off + BLOCK_SIZE <= ans_len) ? BLOCK_SIZE : (ans_len - off);
        u8 *chunk = ans_stream + off;

        double H_chunk = entropy(chunk, csz);

        /* dump chunk 2 raw bytes to file for inspection */
        if (c == 2) {
            FILE *df = fopen("C:\\Users\\lukac\\Documents\\compressor\\chunk2.bin", "wb");
            if (df) { fwrite(chunk, 1, csz, df); fclose(df); printf("[chunk 2 dumped to chunk2.bin]\n"); }
        }

        int counts[NUM_INSTR_TYPES] = {0};
        int verbose_this = (c == 0 && VERBOSE);
        if (ANALYSE) analyse_chunk(chunk, csz, c);
        if (verbose_this) printf("[chunk 0 detail]\n");

        /* ── interleaved greedy + analytic sub-loop until convergence ──────── */
        /* Each analytic application gets its own slot (g_ni=0, K>0).
         * The undo loop already handles mixed/pure slots correctly.         */
        double net_c     = 0.0;
        double ana_net_c = 0.0;
        int    total_ni_c = 0;
        int    s = 0;   /* current slot index into rounds[c] */
        nrounds_arr[c]    = 0;

        for (int iter = 0; iter < 16 && s < ANA_MAX_INTERLEAVE; iter++) {
            /* greedy pass → greedy-only slot */
            int cnt_r[NUM_INSTR_TYPES] = {0};
            double gn = compress_greedy(chunk, csz, (iter == 0 ? verbose_this : 0), cnt_r);
            if (iter == 0 && verbose_this) printf("\n");
            int rni = g_ni;
            if (total_ni_c + rni > 4096) rni = 4096 - total_ni_c;
            memcpy(ilists[c] + total_ni_c, g_ilist, (size_t)rni * sizeof(Instr));
            rounds[c][s].g_ni = rni;
            rounds[c][s].K    = 0;
            total_ni_c += rni;
            for (int i2 = 0; i2 < NUM_INSTR_TYPES; i2++) counts[i2] += cnt_r[i2];
            net_c += gn;
            nrounds_arr[c] = ++s;

            /* analytic sub-loop: keep firing on new periods until no gain */
            double an = 0.0;
            while (s < ANA_MAX_INTERLEAVE && ANA_MODE > 0) {
                u8 px[ANA_MAX_K]; int Kx;
                double nx = find_best_analytic(chunk, csz, px, &Kx, 0);
                u8 pa[ANA_MAX_K]; int Ka;
                double na2 = find_best_analytic(chunk, csz, pa, &Ka, 1);
                double best = 0.0;
                if (nx >= na2 && nx > 0.0) {
                    best = nx;
                    rounds[c][s].K = Kx; rounds[c][s].op = 0; rounds[c][s].g_ni = 0;
                    memcpy(rounds[c][s].pat, px, (size_t)Kx);
                    apply_analytic(chunk, csz, px, Kx, 0);
                    nrounds_arr[c] = ++s;
                } else if (na2 > nx && na2 > 0.0) {
                    best = na2;
                    rounds[c][s].K = Ka; rounds[c][s].op = 1; rounds[c][s].g_ni = 0;
                    memcpy(rounds[c][s].pat, pa, (size_t)Ka);
                    apply_analytic(chunk, csz, pa, Ka, 1);
                    nrounds_arr[c] = ++s;
                } else break;
                an += best;
            }
            ana_net_c += an;

            if (gn <= 0.0 && an <= 0.0) break;
        }

        ni_list[c]    = total_ni_c;
        double H_red_c = entropy(chunk, csz);

        /* round-trip verify: undo in reverse round order */
        u8 *tmp = malloc(csz);
        memcpy(tmp, chunk, csz);
        {
            int gni_pos = total_ni_c;
            for (int r = nrounds_arr[c]-1; r >= 0; r--) {
                if (rounds[c][r].K > 0)
                    undo_analytic(tmp, csz, rounds[c][r].pat,
                                  rounds[c][r].K, rounds[c][r].op);
                gni_pos -= rounds[c][r].g_ni;
                if (rounds[c][r].g_ni > 0)
                    do_decompress(tmp, csz, ilists[c] + gni_pos, rounds[c][r].g_ni);
            }
        }
        int ok = (memcmp(tmp, ans_pristine + off, csz) == 0);
        free(tmp);

        if (ok) chunks_ok++; else { chunks_fail++; printf("  *** FAIL chunk %d ***\n", c); }

        int top_type = 0;
        for (int i = 1; i < NUM_INSTR_TYPES; i++)
            if (counts[i] > counts[top_type]) top_type = i;
        for (int i = 0; i < NUM_INSTR_TYPES; i++) total_counts[i] += counts[i];

        printf("%-7d  %-9.6f  %-7d  %-9.6f  %-+9.6f  %-+10.1f  %-5d  %-+10.1f  %-4d  %-3s  %s\n",
               c, H_chunk, csz, H_red_c, H_red_c - H_chunk, net_c, ni_list[c],
               ana_net_c, nrounds_arr[c],
               ok ? "ok" : "FAIL",
               ni_list[c] > 0 ? INSTR_NAMES[top_type] : "-");

        total_net += net_c; total_H_ans += H_chunk; total_H_red += H_red_c;
        total_ana_net += ana_net_c;
    }

    /* ── SECOND LAYER: ANS2 on entropy-reduced stream, then entropy-reduce ─ */
    printf("\n--- second layer test (diagnostic) ---\n");
    {
        /* ans_stream is still in entropy-reduced state here */
        u8 *red_copy = malloc(ans_len);
        memcpy(red_copy, ans_stream, ans_len);

        int ans2_cap = ans_len + 4096;
        u8 *ans2_buf = malloc(ans2_cap);
        int ans2_len = o0_compress(red_copy, ans_len, ans2_buf, ans2_cap);
        free(red_copy);

        printf("ANS2 adaptive : %d → %d bytes  H=%.6f  (%+.0f bits)\n\n",
               ans_len, ans2_len, entropy(ans2_buf, ans2_len),
               (double)(ans_len - ans2_len) * 8.0);

        u8  *ans2_payload = ans2_buf;
        int  ans2_plen    = ans2_len;

        int n2 = (ans2_plen + BLOCK_SIZE - 1) / BLOCK_SIZE;
        double total_net2 = 0.0, total_ana2 = 0.0;

        printf("%-7s  %-9s  %-9s  %-10s  %-10s\n",
               "chunk2", "H_in", "H_out", "greedy", "analytic");

        for (int c2 = 0; c2 < n2; c2++) {
            int off2 = c2 * BLOCK_SIZE;
            int csz2 = (off2 + BLOCK_SIZE <= ans2_plen) ? BLOCK_SIZE : (ans2_plen - off2);
            u8 *ch2  = ans2_payload + off2;

            double H_in2 = entropy(ch2, csz2);
            int cnt2[NUM_INSTR_TYPES] = {0};
            double gn2 = 0.0, an2 = 0.0;

            /* same greedy+analytic sub-loop as layer-1 (diagnostic, no undo needed) */
            for (int it2 = 0; it2 < 16; it2++) {
                int c2t[NUM_INSTR_TYPES] = {0};
                double g = compress_greedy(ch2, csz2, 0, c2t);
                for (int i2 = 0; i2 < NUM_INSTR_TYPES; i2++) cnt2[i2] += c2t[i2];
                gn2 += g;
                double round_an2 = 0.0;
                while (ANA_MODE > 0) {
                    u8 px[ANA_MAX_K]; int Kx;
                    double nx = find_best_analytic(ch2, csz2, px, &Kx, 0);
                    u8 pa[ANA_MAX_K]; int Ka;
                    double na2x = find_best_analytic(ch2, csz2, pa, &Ka, 1);
                    if (nx >= na2x && nx > 0.0) { round_an2 += nx; apply_analytic(ch2, csz2, px, Kx, 0); }
                    else if (na2x > nx && na2x > 0.0) { round_an2 += na2x; apply_analytic(ch2, csz2, pa, Ka, 1); }
                    else break;
                }
                an2 += round_an2;
                if (g <= 0.0 && round_an2 <= 0.0) break;
            }

            double H_out2 = entropy(ch2, csz2);
            printf("  %-5d  %-9.6f  %-9.6f  %-+10.1f  %-+10.1f\n",
                   c2, H_in2, H_out2, gn2, an2);
            total_net2 += gn2;
            total_ana2 += an2;
        }

        printf("\n2nd layer greedy  : %+.1f bits  (%+.1f/chunk)\n", total_net2, total_net2/n2);
        printf("2nd layer analytic: %+.1f bits  (%+.1f/chunk)\n", total_ana2, total_ana2/n2);
        printf("2nd layer combined: %+.1f bits  (%+.1f/chunk)\n",
               total_net2+total_ana2, (total_net2+total_ana2)/n2);
        free(ans2_buf);
    }

    for (int c = 0; c < n_chunks; c++) {
        int off = c * BLOCK_SIZE;
        int csz = (off + BLOCK_SIZE <= ans_len) ? BLOCK_SIZE : (ans_len - off);
        u8 *data = ans_stream + off;
        int gni_pos = ni_list[c];
        for (int r = nrounds_arr[c]-1; r >= 0; r--) {
            if (rounds[c][r].K > 0)
                undo_analytic(data, csz, rounds[c][r].pat,
                              rounds[c][r].K, rounds[c][r].op);
            gni_pos -= rounds[c][r].g_ni;
            if (rounds[c][r].g_ni > 0)
                do_decompress(data, csz, ilists[c] + gni_pos, rounds[c][r].g_ni);
        }
    }
    int ans_match = (memcmp(ans_stream, ans_pristine, ans_len) == 0);
    if (has_ctx256) undo_ctx256_xor(ans_stream, ans_len, ctx256_amp);
    u8 *raw_check = malloc(raw_size);
    o0_decompress(ans_stream, ans_len, raw_check, raw_size);
    int raw_match = (memcmp(raw_check, raw_data, raw_size) == 0);
    free(raw_check);

    printf("\nchunks processed : %d  (%d ok, %d FAIL)\n", n_chunks, chunks_ok, chunks_fail);
    printf("ANS restore      : %s\nfull round-trip  : %s\n\n",
           ans_match ? "ok" : "FAIL", raw_match ? "ok" : "FAIL");
    printf("avg H_ans     : %.6f bpb\navg H_red     : %.6f bpb\n"
           "avg delta     : %+.6f bpb\n"
           "greedy net    : %+.1f bits  (%+.1f bits/chunk)\n"
           "analytic net  : %+.1f bits  (%+.1f bits/chunk)\n"
           "combined net  : %+.1f bits  (%+.1f bits/chunk)\n",
           total_H_ans/n_chunks, total_H_red/n_chunks,
           (total_H_red-total_H_ans)/n_chunks,
           total_net, total_net/n_chunks,
           total_ana_net, total_ana_net/n_chunks,
           total_net + total_ana_net, (total_net + total_ana_net)/n_chunks);

    printf("\ninstruction usage:\n");
    for (int i = 0; i < NUM_INSTR_TYPES; i++)
        if (total_counts[i] > 0)
            printf("  %-16s %d\n", INSTR_NAMES[i], total_counts[i]);

    free(ilists); free(ni_list);
    free(rounds); free(nrounds_arr);
#endif

    free(raw_data); free(ans_stream); free(ans_pristine);
    return 0;
}
