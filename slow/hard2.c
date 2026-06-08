/*
 * simple.c — greedy entropy compressor
 *
 * All instructions use the frequency-table trick:
 *   new_freq[v] = total[v] - phF[v] + phF[inverse(v, amp)]
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
    /* 16-bit overhead */
    XOR_PHASE = 0,  /* data[i] ^= amp                                          */
    ADD_PHASE,      /* data[i] += amp  (mod 256)                               */
    MUL_ODD,        /* data[i] *= amp  (amp odd, invertible mod 256)           */
    COND_LO_XOR,    /* if hi-nibble==(amp>>4): low-nibble ^= (amp&0xF)         */
    ADD_NIBS,       /* lo-nib += (amp&0xF), hi-nib += (amp>>4), no carry       */
    /* 8-bit overhead */
    NIB_SWAP,       /* data[i] = (data[i]<<4)|(data[i]>>4), self-inverse       */
    /* 24-bit overhead */
    AFFINE_PHASE,   /* data[i] = (data[i]*m + c)&0xFF  amp = idx|(c<<7), m=2*idx+3  */
    DUAL_XOR,       /* even occurrences ^= (amp&0xFF), odd ^= ((amp>>8)&0xFF)  */
    DUAL_ADD,       /* even occurrences += (amp&0xFF), odd += ((amp>>8)&0xFF)  */
    NUM_INSTR_TYPES
} InstrType;

static const char *INSTR_NAMES[NUM_INSTR_TYPES] = {
    "XOR_PHASE", "ADD_PHASE", "MUL_ODD", "COND_LO_XOR", "ADD_NIBS",
    "NIB_SWAP", "AFFINE_PHASE", "DUAL_XOR", "DUAL_ADD"
};

typedef struct { InstrType type; int stride, phase, amp; } Instr;

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
        default: break;
    }
}

/* ── find best instruction ───────────────────────────────────────────────── */
static Instr findBest(const u8 *data, int n, double *netOut) {
    int total[256] = {0};
    for (int i = 0; i < n; i++) total[data[i]]++;
    double Sbase = 0.0;
    for (int v = 0; v < 256; v++) Sbase += hlog[total[v]];

    double bestNet = 0.0;
    Instr  best    = {XOR_PHASE, 2, 0, 1};

    int phF[256];

    for (int stride = 1; stride <= 64; stride++) {
        for (int phase = 0; phase < stride; phase++) {

            /* build phase-position frequency table */
            memset(phF, 0, sizeof phF);
            for (int i = phase; i < n; i += stride) phF[data[i]]++;

            /* ── XOR_PHASE and ADD_PHASE ─────────────────────────────────── */
            for (int amp = 1; amp < 256; amp++) {
                double Sx = 0.0, Sa = 0.0;
                for (int v = 0; v < 256; v++) {
                    Sx += hlog[total[v] - phF[v] + phF[v ^ amp]];
                    Sa += hlog[total[v] - phF[v] + phF[(v - amp) & 0xFF]];
                }
                double nx = (Sx - Sbase) - 16.0;
                double na = (Sa - Sbase) - 16.0;
                if (nx > bestNet) { bestNet = nx; best = (Instr){XOR_PHASE, stride, phase, amp}; }
                if (na > bestNet) { bestNet = na; best = (Instr){ADD_PHASE, stride, phase, amp}; }
            }

            /* ── MUL_ODD ─────────────────────────────────────────────────── */
            for (int amp = 3; amp < 256; amp += 2) {
                double Sm = 0.0;
                for (int v = 0; v < 256; v++)
                    Sm += hlog[total[v] - phF[v] + phF[(v * mul_inv[amp]) & 0xFF]];
                double nm = (Sm - Sbase) - 16.0;
                if (nm > bestNet) { bestNet = nm; best = (Instr){MUL_ODD, stride, phase, amp}; }
            }

            /* ── COND_LO_XOR ─────────────────────────────────────────────── */
            for (int nib_cond = 0; nib_cond < 16; nib_cond++) {
                for (int xv = 1; xv < 16; xv++) {
                    double delta = 0.0;
                    for (int lo = 0; lo < 16; lo++) {
                        int v    = (nib_cond << 4) | lo;
                        int v_xv = (nib_cond << 4) | (lo ^ xv);
                        delta += hlog[total[v] - phF[v] + phF[v_xv]] - hlog[total[v]];
                    }
                    double nc = delta - 16.0;
                    if (nc > bestNet) { bestNet = nc; best = (Instr){COND_LO_XOR, stride, phase, (nib_cond << 4) | xv}; }
                }
            }

            /* ── ADD_NIBS ─────────────────────────────────────────────────── */
            for (int amp = 1; amp < 256; amp++) {
                int lo_a = amp & 0x0F, hi_a = (amp >> 4) & 0x0F;
                double Sn = 0.0;
                for (int v = 0; v < 256; v++) {
                    int old_v = ((v - lo_a) & 0x0F) | (((v >> 4) - hi_a) & 0x0F) << 4;
                    Sn += hlog[total[v] - phF[v] + phF[old_v]];
                }
                double nn = (Sn - Sbase) - 16.0;
                if (nn > bestNet) { bestNet = nn; best = (Instr){ADD_NIBS, stride, phase, amp}; }
            }

            /* ── NIB_SWAP: self-inverse, cost 8 bits ─────────────────────── */
            {
                double Ssw = 0.0;
                for (int v = 0; v < 256; v++) {
                    int sv = ((v << 4) | (v >> 4)) & 0xFF;
                    Ssw += hlog[total[v] - phF[v] + phF[sv]];
                }
                double nsw = (Ssw - Sbase) - 8.0;
                if (nsw > bestNet) { bestNet = nsw; best = (Instr){NIB_SWAP, stride, phase, 0}; }
            }

            /* ── AFFINE_PHASE: v → (v*m + c)&0xFF, m odd 3..255 ────────── */
            /* Restricted to stride <= 8; slower than XOR/ADD per pair. */
            if (stride <= 8) {
                for (int m = 3; m <= 255; m += 2) {
                    int minv = mul_inv[m];
                    for (int c = 0; c < 256; c++) {
                        double Sf = 0.0;
                        for (int v = 0; v < 256; v++) {
                            int old_v = ((v - c) * minv) & 0xFF;
                            Sf += hlog[total[v] - phF[v] + phF[old_v]];
                        }
                        double nf = (Sf - Sbase) - 23.0;
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

                /* ── DUAL_XOR ── */
                int    best_lo_xor = 1;
                double best_S_xor  = -1e30;
                for (int amp = 1; amp < 256; amp++) {
                    double S = 0.0;
                    for (int v = 0; v < 256; v++)
                        S += hlog[total[v] - phFe[v] + phFe[v ^ amp]];
                    if (S > best_S_xor) { best_S_xor = S; best_lo_xor = amp; }
                }
                /* total2[v] = total after applying best_lo_xor to even positions */
                int t2x[256];
                for (int v = 0; v < 256; v++)
                    t2x[v] = total[v] - phFe[v] + phFe[v ^ best_lo_xor];
                int    best_hi_xor = 1;
                double best_S2_xor = -1e30;
                for (int amp = 1; amp < 256; amp++) {
                    double S = 0.0;
                    for (int v = 0; v < 256; v++) {
                        int phFo_v     = phF[v]       - phFe[v];
                        int phFo_v_amp = phF[v ^ amp] - phFe[v ^ amp];
                        S += hlog[t2x[v] - phFo_v + phFo_v_amp];
                    }
                    if (S > best_S2_xor) { best_S2_xor = S; best_hi_xor = amp; }
                }
                double net_dxor = (best_S2_xor - Sbase) - 24.0;
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
                        S += hlog[total[v] - phFe[v] + phFe[(v - amp) & 0xFF]];
                    if (S > best_S_add) { best_S_add = S; best_lo_add = amp; }
                }
                int t2a[256];
                for (int v = 0; v < 256; v++)
                    t2a[v] = total[v] - phFe[v] + phFe[(v - best_lo_add) & 0xFF];
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
                double net_dadd = (best_S2_add - Sbase) - 24.0;
                if (net_dadd > bestNet) {
                    bestNet = net_dadd;
                    best = (Instr){DUAL_ADD, stride, phase, best_lo_add | (best_hi_add << 8)};
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

/* ── greedy compress ─────────────────────────────────────────────────────── */
static double compress(u8 *data, int n, int verbose, int *counts) {
    double total_net = 0.0;

    for (;;) {
        for (;;) {
            double net;
            Instr t = findBest(data, n, &net);
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

        int found = 0;
        double E0 = entropy(data, n);
        u8 *temp  = malloc(n);
        u8 *after = malloc(n);

#define TRY_SCRAMBLE(label, cond, scramble_call) \
        if (!found && (cond)) { \
            scramble_call; \
            double net2; Instr t2 = findBest(temp, n, &net2); \
            memcpy(after, temp, n); applyInstr(after, n, t2); \
            double true_net = (E0 - entropy(after, n)) * n - 8.0; \
            if (true_net > 0) { \
                memcpy(data, temp, n); \
                double e0 = entropy(data, n); \
                applyInstr(data, n, t2); \
                total_net += true_net; counts[t2.type]++; found = 1; \
                if (verbose) \
                    printf("  [%-14s] %-14s s%-2d p%-2d a%-6d  %.6f -> %.6f  net=%.1f\n", \
                           label, INSTR_NAMES[t2.type], t2.stride, t2.phase, t2.amp, \
                           e0, entropy(data, n), true_net); \
            } \
        }

        TRY_SCRAMBLE("INTERLEAVE-2",  n%2==0,  interleave_stride(data, temp, n, 2))
        TRY_SCRAMBLE("INTERLEAVE-4",  n%4==0,  interleave_stride(data, temp, n, 4))
        TRY_SCRAMBLE("INTERLEAVE-8",  n%8==0,  interleave_stride(data, temp, n, 8))
        TRY_SCRAMBLE("INTERLEAVE-16", n%16==0, interleave_stride(data, temp, n, 16))
        TRY_SCRAMBLE("INTERLEAVE-32", n%32==0, interleave_stride(data, temp, n, 32))
        TRY_SCRAMBLE("INTERLEAVE-64", n%64==0, interleave_stride(data, temp, n, 64))
        TRY_SCRAMBLE("BIT-PLANE-SEP", n%8==0,  bit_plane_sep(data, temp, n))
        TRY_SCRAMBLE("NIB-PLANE-SEP", n%2==0,  nib_plane_sep(data, temp, n))
        TRY_SCRAMBLE("BLOCK-REVERSE", 1,       block_reverse(data, temp, n))
        TRY_SCRAMBLE("XOR-FOLD-SCR",  n%2==0,  xor_fold_scramble(data, temp, n))

#undef TRY_SCRAMBLE

        free(after);
        free(temp);
        if (!found) break;
    }
    return total_net;
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
        printf("  %-14s %d\n", INSTR_NAMES[i], counts[i]);

    return 0;
}
