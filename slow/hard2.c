/*
 * simple.c — greedy entropy compressor
 *
 * Six bijective byte-transform instructions, all evaluated via the
 * frequency-table trick:
 *
 *   new_freq[v] = total[v] - phF[v] + phF[inverse(v, amp)]
 *
 * This evaluates ALL amp values for a given (stride, phase) in O(256)
 * each, making exhaustive search essentially free.
 *
 * After the greedy converges, a scramble-restart loop tries layout
 * permutations and commits any that yield a true net gain.
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
    /*
     * All six use the frequency-table trick.
     * Overhead: XOR/ADD/MUL/COND_LO/ADD_NIBS = 16 bits
     *           AFFINE = 24 bits (needs extra byte to encode multiplier m)
     *
     * Search strides: 1..32 for all.
     * AFFINE additionally searches m=3..31 at stride 1..8 (wider grid).
     */
    XOR_PHASE = 0,  /* data[i] ^= amp                                          */
    ADD_PHASE,      /* data[i] += amp  (mod 256)                               */
    MUL_ODD,        /* data[i] *= amp  (amp must be odd for invertibility)     */
    COND_LO_XOR,    /* if hi-nibble == (amp>>4): low-nibble ^= (amp&0xF)       */
    AFFINE_PHASE,   /* data[i] = (data[i]*m + c) & 0xFF   amp = m|(c<<8)      */
    ADD_NIBS,       /* low and high nibbles each get independent modular adds;
                       amp = lo_add | (hi_add<<4), no carry across boundary    */
    NIB_SWAP,       /* data[i] = (data[i]<<4)|(data[i]>>4) — swaps nibbles,
                       self-inverse, no amp; cost 8 bits                       */
    NUM_INSTR_TYPES
} InstrType;

static const char *INSTR_NAMES[NUM_INSTR_TYPES] = {
    "XOR_PHASE", "ADD_PHASE", "MUL_ODD",
    "COND_LO_XOR", "AFFINE_PHASE", "ADD_NIBS", "NIB_SWAP"
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
        case AFFINE_PHASE: {
            int m = t.amp & 0xFF;
            int c = (t.amp >> 8) & 0xFF;
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)((data[i] * m + c) & 0xFF);
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

    for (int stride = 1; stride <= 32; stride++) {
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

            /* ── NIB_SWAP: v → (v<<4)|(v>>4), self-inverse, cost 8 bits ──── */
            {
                double Ssw = 0.0;
                for (int v = 0; v < 256; v++) {
                    int sv = ((v << 4) | (v >> 4)) & 0xFF;
                    Ssw += hlog[total[v] - phF[v] + phF[sv]];
                }
                double nsw = (Ssw - Sbase) - 8.0;
                if (nsw > bestNet) { bestNet = nsw; best = (Instr){NIB_SWAP, stride, phase, 0}; }
            }

            /* ── AFFINE_PHASE: v → (v*m + c) & 0xFF ─────────────────────── */
            /* Searching m ∈ odd 3..31, c ∈ 0..255. Restricted to stride<=8  */
            /* because it's ~15x slower than XOR/ADD per (stride,phase) pair. */
            if (stride <= 8) {
                for (int m = 3; m <= 31; m += 2) {
                    int minv = mul_inv[m];
                    for (int c = 0; c < 256; c++) {
                        double Sf = 0.0;
                        for (int v = 0; v < 256; v++) {
                            int old_v = ((v - c) * minv) & 0xFF;
                            Sf += hlog[total[v] - phF[v] + phF[old_v]];
                        }
                        double nf = (Sf - Sbase) - 24.0;
                        if (nf > bestNet) { bestNet = nf; best = (Instr){AFFINE_PHASE, stride, phase, m | (c << 8)}; }
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

/* ── greedy compress ─────────────────────────────────────────────────────── */
static double compress(u8 *data, int n, int verbose, int *counts) {
    double total_net = 0.0;

    for (;;) {
        /* primary greedy loop */
        for (;;) {
            double net;
            Instr t = findBest(data, n, &net);
            if (net <= 0.0) break;
            double e0 = entropy(data, n);
            applyInstr(data, n, t);
            total_net += net;
            counts[t.type]++;
            if (verbose)
                printf("  %-14s s%-2d p%-2d a%-5d  %.6f -> %.6f  net=%.1f\n",
                       INSTR_NAMES[t.type], t.stride, t.phase, t.amp,
                       e0, entropy(data, n), net);
        }

        /*
         * Scramble-restart: try layout permutations that may expose new
         * patterns. true_net measures total gain vs the pre-scramble baseline,
         * correctly handling scrambles that themselves increase entropy.
         */
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
                    printf("  [%-14s] %-14s s%-2d p%-2d a%-5d  %.6f -> %.6f  net=%.1f\n", \
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
