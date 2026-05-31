#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint8_t u8;

#define SIZE 4096
#define NOPS 64   /* 6-bit op selector */

/* ---- log table for fast entropy ---- */
static double logT[SIZE + 1];

void initLog(void) {
    logT[0] = 0.0;
    for (int k = 1; k <= SIZE; k++)
        logT[k] = (double)k * log2((double)k);
}

double entropy(const u8 *data, size_t n) {
    unsigned freq[256] = {0};
    for (size_t i = 0; i < n; i++) freq[data[i]]++;
    double sum = 0.0;
    for (int i = 0; i < 256; i++) sum += logT[freq[i]];
    return log2((double)n) - sum / (double)n;
}

/* ---- GF(256) helpers ---- */
u8 gfmul(u8 a, u8 b) {
    u8 r = 0;
    for (int i = 0; i < 8; i++) {
        if (b & 1) r ^= a;
        b >>= 1;
        if (a & 0x80) a = (a << 1) ^ 0x1B;
        else a <<= 1;
    }
    return r;
}

static u8 gfInvTable[256];
void initGFInv(void) {
    gfInvTable[0] = 0;
    for (int a = 1; a < 256; a++)
        for (int b = 1; b < 256; b++)
            if (gfmul((u8)a, (u8)b) == 1) { gfInvTable[a] = (u8)b; break; }
}

/* mulInvTable[n] = n^-1 mod 256; valid only for odd n — always pass (x|1) */
static u8 mulInvTable[256];
void initMulInv(void) {
    for (int n = 1; n < 256; n += 2)
        for (int inv = 1; inv < 256; inv += 2)
            if (((n * inv) & 0xFF) == 1) { mulInvTable[n] = (u8)inv; break; }
}

static inline u8 byterev(u8 b) {
    b = (u8)(((b & 0xF0) >> 4) | ((b & 0x0F) << 4));
    b = (u8)(((b & 0xCC) >> 2) | ((b & 0x33) << 2));
    b = (u8)(((b & 0xAA) >> 1) | ((b & 0x55) << 1));
    return b;
}

static inline u8 swp(u8 x) { return (u8)((x << 4) | (x >> 4)); }

/* ---- 64 operations (6-bit selector) --------------------------------
 *
 *  Convention:
 *    prev  = original buf[i-1]  (0 at start)
 *    prev2 = original buf[i-2]  (0 at start)
 *    ix    = (u8)i  (position mod 256)
 *
 *  Multipliers always forced odd via |1 so modular inverse always exists.
 */
static const char *opName[NOPS] = {
    /* 0-15  (original) */
    "ADD",     "XOR",     "MUL",      "ADDLO",
    "SWXOR",   "GFMUL",   "ROL",      "ADDHI",
    "GRAY",    "XORNIB",  "SUBLO",    "SUBHI",
    "NEGADD",  "XORLO",   "XORHI",    "ROR",
    /* 16-31 */
    "SUB",     "ADDX",    "SUBX",     "XORX",
    "ROL2",    "ROR2",    "ROL4",     "BYTEREV",
    "NOT",     "MULX",    "ADDP2",    "SUBP2",
    "XORP2",   "ZIGZAG",  "MULHI",    "MULLO",
    /* 32-63 (new) */
    "ROL3",    "ROR3",    "ROL5",     "ROR5",
    "ROL6",    "ROR6",    "DELTA2",   "ADDSUM",
    "SUBSUM",  "XORADD",  "ADDXOR",   "SUBXOR",
    "MULP2",   "ADDP2HI", "ADDP2LO",  "SUBP2HI",
    "SUBP2LO", "XORP2HI", "XORP2LO",  "GFMUL2",
    "GFMUL3",  "ADDSWP",  "SUBSWP",   "XORSWP",
    "MULSWP",  "ADDXP",   "SUBXP",    "XORXP",
    "MULXP",   "GRAYINV", "GFMUL5",   "GFMULP2"
};

void applyOp(u8 *buf, size_t n, int op) {
    u8 prev2 = 0, prev = 0;
    for (size_t i = 0; i < n; i++) {
        u8 c  = buf[i];
        u8 ix = (u8)i;
        switch (op) {
            /* ---- 0-15 ---- */
            case  0: buf[i] = c + prev; break;
            case  1: buf[i] = c ^ prev; break;
            case  2: buf[i] = c * (prev | 1); break;
            case  3: buf[i] = c + (prev & 0x0F); break;
            case  4: buf[i] = swp(c) ^ prev; break;
            case  5: buf[i] = gfmul(c, prev ? prev : 1); break;
            case  6: buf[i] = (u8)((c << 1) | (c >> 7)); break;
            case  7: buf[i] = c + (prev >> 4); break;
            case  8: buf[i] = c ^ (c >> 1); break;
            case  9: buf[i] = c ^ ((prev & 0x0F) | ((prev & 0x0F) << 4)); break;
            case 10: buf[i] = c - (prev & 0x0F); break;
            case 11: buf[i] = c - (prev >> 4); break;
            case 12: buf[i] = prev - c; break;
            case 13: buf[i] = c ^ (prev & 0x0F); break;
            case 14: buf[i] = c ^ (prev >> 4); break;
            case 15: buf[i] = (u8)((c >> 1) | (c << 7)); break;
            /* ---- 16-31 ---- */
            case 16: buf[i] = c - prev; break;
            case 17: buf[i] = c + ix; break;
            case 18: buf[i] = c - ix; break;
            case 19: buf[i] = c ^ ix; break;
            case 20: buf[i] = (u8)((c << 2) | (c >> 6)); break;
            case 21: buf[i] = (u8)((c >> 2) | (c << 6)); break;
            case 22: buf[i] = swp(c); break;
            case 23: buf[i] = byterev(c); break;
            case 24: buf[i] = ~c; break;
            case 25: buf[i] = c * (ix | 1); break;
            case 26: buf[i] = c + prev2; break;
            case 27: buf[i] = c - prev2; break;
            case 28: buf[i] = c ^ prev2; break;
            case 29: buf[i] = (u8)((c << 1) ^ (u8)(-(c >> 7))); break;
            case 30: buf[i] = c * ((prev >> 4) | 1); break;
            case 31: buf[i] = c * ((prev & 0x0F) | 1); break;
            /* ---- 32-63 ---- */
            case 32: buf[i] = (u8)((c << 3) | (c >> 5)); break;              /* ROL3        */
            case 33: buf[i] = (u8)((c >> 3) | (c << 5)); break;              /* ROR3        */
            case 34: buf[i] = (u8)((c << 5) | (c >> 3)); break;              /* ROL5        */
            case 35: buf[i] = (u8)((c >> 5) | (c << 3)); break;              /* ROR5        */
            case 36: buf[i] = (u8)((c << 6) | (c >> 2)); break;              /* ROL6        */
            case 37: buf[i] = (u8)((c >> 6) | (c << 2)); break;              /* ROR6        */
            case 38: buf[i] = c - 2*prev + prev2; break;                     /* DELTA2      */
            case 39: buf[i] = c + prev + prev2; break;                       /* ADDSUM      */
            case 40: buf[i] = c - prev - prev2; break;                       /* SUBSUM      */
            case 41: buf[i] = c ^ (u8)(prev + prev2); break;                 /* XORADD      */
            case 42: buf[i] = c + (u8)(prev ^ prev2); break;                 /* ADDXOR      */
            case 43: buf[i] = c - (u8)(prev ^ prev2); break;                 /* SUBXOR      */
            case 44: buf[i] = c * (prev2 | 1); break;                        /* MULP2       */
            case 45: buf[i] = c + (prev2 >> 4); break;                       /* ADDP2HI     */
            case 46: buf[i] = c + (prev2 & 0x0F); break;                     /* ADDP2LO     */
            case 47: buf[i] = c - (prev2 >> 4); break;                       /* SUBP2HI     */
            case 48: buf[i] = c - (prev2 & 0x0F); break;                     /* SUBP2LO     */
            case 49: buf[i] = c ^ (prev2 >> 4); break;                       /* XORP2HI     */
            case 50: buf[i] = c ^ (prev2 & 0x0F); break;                     /* XORP2LO     */
            case 51: buf[i] = gfmul(c, 2); break;                            /* GFMUL2      */
            case 52: buf[i] = gfmul(c, 3); break;                            /* GFMUL3      */
            case 53: buf[i] = c + swp(prev); break;                          /* ADDSWP      */
            case 54: buf[i] = c - swp(prev); break;                          /* SUBSWP      */
            case 55: buf[i] = c ^ swp(prev); break;                          /* XORSWP      */
            case 56: buf[i] = c * (swp(prev) | 1); break;                    /* MULSWP      */
            case 57: buf[i] = c + (u8)(ix ^ prev); break;                    /* ADDXP       */
            case 58: buf[i] = c - (u8)(ix ^ prev); break;                    /* SUBXP       */
            case 59: buf[i] = c ^ (u8)(ix + prev); break;                    /* XORXP       */
            case 60: buf[i] = c * ((u8)(ix ^ prev) | 1); break;              /* MULXP       */
            case 61: { u8 g=c; g^=g>>4; g^=g>>2; g^=g>>1; buf[i]=g; break;} /* GRAYINV     */
            case 62: buf[i] = gfmul(c, 5); break;                            /* GFMUL5      */
            case 63: buf[i] = gfmul(c, prev2 ? prev2 : 1); break;            /* GFMULP2     */
        }
        prev2 = prev;
        prev  = c;   /* original (pre-transform) value */
    }
}

void invertOp(u8 *buf, size_t n, int op) {
    u8 prev2 = 0, prev = 0;
    for (size_t i = 0; i < n; i++) {
        u8 c  = buf[i];
        u8 ix = (u8)i;
        u8 orig;
        switch (op) {
            /* ---- 0-15 ---- */
            case  0: orig = c - prev; break;
            case  1: orig = c ^ prev; break;
            case  2: orig = c * mulInvTable[prev | 1]; break;
            case  3: orig = c - (prev & 0x0F); break;
            case  4: { u8 t = c ^ prev; orig = swp(t); break; }
            case  5: orig = gfmul(c, gfInvTable[prev ? prev : 1]); break;
            case  6: orig = (u8)((c >> 1) | (c << 7)); break;
            case  7: orig = c - (prev >> 4); break;
            case  8: { u8 g=c; g^=g>>4; g^=g>>2; g^=g>>1; orig=g; break; }
            case  9: orig = c ^ ((prev & 0x0F) | ((prev & 0x0F) << 4)); break;
            case 10: orig = c + (prev & 0x0F); break;
            case 11: orig = c + (prev >> 4); break;
            case 12: orig = prev - c; break;
            case 13: orig = c ^ (prev & 0x0F); break;
            case 14: orig = c ^ (prev >> 4); break;
            case 15: orig = (u8)((c << 1) | (c >> 7)); break;
            /* ---- 16-31 ---- */
            case 16: orig = c + prev; break;
            case 17: orig = c - ix; break;
            case 18: orig = c + ix; break;
            case 19: orig = c ^ ix; break;
            case 20: orig = (u8)((c >> 2) | (c << 6)); break;
            case 21: orig = (u8)((c << 2) | (c >> 6)); break;
            case 22: orig = swp(c); break;
            case 23: orig = byterev(c); break;
            case 24: orig = ~c; break;
            case 25: orig = c * mulInvTable[ix | 1]; break;
            case 26: orig = c - prev2; break;
            case 27: orig = c + prev2; break;
            case 28: orig = c ^ prev2; break;
            case 29: orig = (u8)((c >> 1) ^ (u8)(-(c & 1))); break;
            case 30: orig = c * mulInvTable[(prev >> 4) | 1]; break;
            case 31: orig = c * mulInvTable[(prev & 0x0F) | 1]; break;
            /* ---- 32-63 ---- */
            case 32: orig = (u8)((c >> 3) | (c << 5)); break;               /* ROL3 inv: ROR3   */
            case 33: orig = (u8)((c << 3) | (c >> 5)); break;               /* ROR3 inv: ROL3   */
            case 34: orig = (u8)((c >> 5) | (c << 3)); break;               /* ROL5 inv: ROR5   */
            case 35: orig = (u8)((c << 5) | (c >> 3)); break;               /* ROR5 inv: ROL5   */
            case 36: orig = (u8)((c >> 6) | (c << 2)); break;               /* ROL6 inv: ROR6   */
            case 37: orig = (u8)((c << 6) | (c >> 2)); break;               /* ROR6 inv: ROL6   */
            case 38: orig = c + 2*prev - prev2; break;                      /* DELTA2           */
            case 39: orig = c - prev - prev2; break;                        /* ADDSUM           */
            case 40: orig = c + prev + prev2; break;                        /* SUBSUM           */
            case 41: orig = c ^ (u8)(prev + prev2); break;                  /* XORADD  self-inv */
            case 42: orig = c - (u8)(prev ^ prev2); break;                  /* ADDXOR           */
            case 43: orig = c + (u8)(prev ^ prev2); break;                  /* SUBXOR           */
            case 44: orig = c * mulInvTable[prev2 | 1]; break;              /* MULP2            */
            case 45: orig = c - (prev2 >> 4); break;                        /* ADDP2HI          */
            case 46: orig = c - (prev2 & 0x0F); break;                      /* ADDP2LO          */
            case 47: orig = c + (prev2 >> 4); break;                        /* SUBP2HI          */
            case 48: orig = c + (prev2 & 0x0F); break;                      /* SUBP2LO          */
            case 49: orig = c ^ (prev2 >> 4); break;                        /* XORP2HI self-inv */
            case 50: orig = c ^ (prev2 & 0x0F); break;                      /* XORP2LO self-inv */
            case 51: orig = gfmul(c, gfInvTable[2]); break;                 /* GFMUL2           */
            case 52: orig = gfmul(c, gfInvTable[3]); break;                 /* GFMUL3           */
            case 53: orig = c - swp(prev); break;                           /* ADDSWP           */
            case 54: orig = c + swp(prev); break;                           /* SUBSWP           */
            case 55: orig = c ^ swp(prev); break;                           /* XORSWP  self-inv */
            case 56: orig = c * mulInvTable[swp(prev) | 1]; break;          /* MULSWP           */
            case 57: orig = c - (u8)(ix ^ prev); break;                     /* ADDXP            */
            case 58: orig = c + (u8)(ix ^ prev); break;                     /* SUBXP            */
            case 59: orig = c ^ (u8)(ix + prev); break;                     /* XORXP   self-inv */
            case 60: orig = c * mulInvTable[(u8)(ix ^ prev) | 1]; break;    /* MULXP            */
            case 61: orig = c ^ (c >> 1); break;                            /* GRAYINV inv:GRAY */
            case 62: orig = gfmul(c, gfInvTable[5]); break;                 /* GFMUL5           */
            case 63: orig = gfmul(c, gfInvTable[prev2 ? prev2 : 1]); break; /* GFMULP2          */
            default: orig = c; break;
        }
        buf[i] = orig;
        prev2 = prev;
        prev  = orig;   /* decoded (original) value */
    }
}

/* ---- Fisher-Yates shuffle / unshuffle ---- */
void yatsShuffle(u8 *buf, size_t n, uint16_t seed) {
    uint32_t s = (uint32_t)seed * 1664525u + 1013904223u;
    for (size_t i = n - 1; i > 0; i--) {
        s = s * 1664525u + 1013904223u;
        size_t j = s % (i + 1);
        u8 t = buf[i]; buf[i] = buf[j]; buf[j] = t;
    }
}

void yatsUnshuffle(u8 *buf, size_t n, uint16_t seed) {
    size_t *idx = malloc(n * sizeof(size_t));
    uint32_t s = (uint32_t)seed * 1664525u + 1013904223u;
    for (size_t i = n - 1; i > 0; i--) {
        s = s * 1664525u + 1013904223u;
        idx[i] = s % (i + 1);
    }
    for (size_t i = 1; i < n; i++) {
        u8 t = buf[i]; buf[i] = buf[idx[i]]; buf[idx[i]] = t;
    }
    free(idx);
}

/* ---- self-test ---- */
int selfTest(void) {
    u8 orig[SIZE], work[SIZE];
    srand(42);
    for (int i = 0; i < SIZE; i++) orig[i] = rand() & 0xFF;

    int ok = 1;
    for (int op = 0; op < NOPS; op++) {
        memcpy(work, orig, SIZE);
        applyOp(work, SIZE, op);
        invertOp(work, SIZE, op);
        if (memcmp(orig, work, SIZE) != 0) {
            printf("  FAIL op=%d (%s)\n", op, opName[op]);
            ok = 0;
        }
    }
    memcpy(work, orig, SIZE);
    yatsShuffle(work, SIZE, 0xABCD);
    yatsUnshuffle(work, SIZE, 0xABCD);
    if (memcmp(orig, work, SIZE) != 0) {
        printf("  FAIL yatsShuffle round-trip\n");
        ok = 0;
    }
    if (ok) printf("Self-test passed: all %d ops + shuffle are reversible.\n\n", NOPS);
    return ok;
}

/* ---- main ---- */
int main(void) {
    initLog();
    initGFInv();
    initMulInv();

    if (!selfTest()) return 1;

    srand(0);
    u8 *data = malloc(SIZE);
    for (int x = 0; x < SIZE; x++) data[x] = rand() % 256;

    double startH = entropy(data, SIZE);
    double curH   = startH;
    printf("Initial entropy: %.6f bits/byte\n\n", startH);

    u8 *tmp  = malloc(SIZE);
    u8 *tmp2 = malloc(SIZE);
    u8 *shuf = malloc(SIZE);   /* scratch for best shuffled image in phase 2b */
    int    step          = 0;
    double totalOverhead = 0.0;

    const double OP_BITS       = 6.0;              /* log2(64)          */
    const double PAIR_BITS     = 2.0 * OP_BITS;    /* 12 bits           */
    const double SHUFFLE_BITS  = 16.0 + OP_BITS;   /* 22 bits           */
    const double SHUPAR_BITS   = 16.0 + PAIR_BITS; /* 28 bits           */

    for (;;) {
        /* ============================================================
         * Phase 1 — best single op
         * ============================================================ */
        int    bestOp = -1;
        double bestH  = curH;
        for (int op = 0; op < NOPS; op++) {
            memcpy(tmp, data, SIZE);
            applyOp(tmp, SIZE, op);
            double h = entropy(tmp, SIZE);
            if (h < bestH) { bestH = h; bestOp = op; }
        }
        double opNet = (bestOp >= 0) ? (curH - bestH) * SIZE - OP_BITS : -1e18;

        /* ============================================================
         * Phase 1b — best op pair (64x64=4096 combos, ~50 ms)
         * Finds synergistic sequences that greedy misses.
         * ============================================================ */
        int    bestP1 = -1, bestP2 = -1;
        double bestPH = curH;
        for (int op1 = 0; op1 < NOPS; op1++) {
            memcpy(tmp, data, SIZE);
            applyOp(tmp, SIZE, op1);            /* tmp = op1(data)     */
            for (int op2 = 0; op2 < NOPS; op2++) {
                memcpy(tmp2, tmp, SIZE);
                applyOp(tmp2, SIZE, op2);
                double h = entropy(tmp2, SIZE);
                if (h < bestPH) { bestPH = h; bestP1 = op1; bestP2 = op2; }
            }
        }
        double pairNet = (bestP1 >= 0) ? (curH - bestPH) * SIZE - PAIR_BITS : -1e18;

        /* ============================================================
         * Phase 2 + 2b — shuffle search (only when best so far is weak)
         * Phase 2  : best shuffle + single op   (22 bits overhead)
         * Phase 2b : best op pair after that seed (28 bits overhead)
         * ============================================================ */
        int      bSOp  = -1;
        uint16_t bSeed = 0;
        double   bSeedH  = curH;
        double   shuNet  = -1e18;

        int    bSP1 = -1, bSP2 = -1;
        double bSPH    = curH;
        double shuPNet = -1e18;

        double bestDirect = (opNet > pairNet) ? opNet : pairNet;
        if (bestDirect < SHUFFLE_BITS) {
            if (bestDirect > 0.0)
                printf("Best direct net=%.1f bits — also searching shuffles...\n", bestDirect);
            else
                printf("No profitable direct action — searching shuffles...\n");
            fflush(stdout);

            /* Phase 2: exhaustive seed search with single op */
            for (uint32_t s = 0; s <= 0xFFFF; s++) {
                memcpy(tmp, data, SIZE);
                yatsShuffle(tmp, SIZE, (uint16_t)s);
                for (int op = 0; op < NOPS; op++) {
                    memcpy(tmp2, tmp, SIZE);
                    applyOp(tmp2, SIZE, op);
                    double h = entropy(tmp2, SIZE);
                    if (h < bSeedH) { bSeedH = h; bSOp = op; bSeed = (uint16_t)s; }
                }
                if ((s & 0x1FFF) == 0) {
                    printf("  seed %5u/65535  shuf+op %.6f\r", (unsigned)s, bSeedH);
                    fflush(stdout);
                }
            }
            printf("  shuf+op done.                            \n");
            shuNet = (bSOp >= 0) ? (curH - bSeedH) * SIZE - SHUFFLE_BITS : -1e18;

            /* Phase 2b: pair search on the best shuffled image */
            memcpy(shuf, data, SIZE);
            yatsShuffle(shuf, SIZE, bSeed);   /* shuf = best_seed(data) */

            for (int op1 = 0; op1 < NOPS; op1++) {
                memcpy(tmp, shuf, SIZE);
                applyOp(tmp, SIZE, op1);
                for (int op2 = 0; op2 < NOPS; op2++) {
                    memcpy(tmp2, tmp, SIZE);
                    applyOp(tmp2, SIZE, op2);
                    double h = entropy(tmp2, SIZE);
                    if (h < bSPH) { bSPH = h; bSP1 = op1; bSP2 = op2; }
                }
            }
            shuPNet = (bSP1 >= 0) ? (curH - bSPH) * SIZE - SHUPAR_BITS : -1e18;
            printf("  shuf+pair done. best=%.6f\n", bSPH);
        }

        /* ============================================================
         * Pick the action with the highest net profit
         * ============================================================ */
        double nets[4] = { opNet, pairNet, shuNet, shuPNet };
        int winner = 0;
        for (int i = 1; i < 4; i++) if (nets[i] > nets[winner]) winner = i;

        if (nets[winner] <= 0.0) break;

        switch (winner) {
        case 0: {  /* single op */
            double gross = (curH - bestH) * SIZE;
            applyOp(data, SIZE, bestOp);
            printf("Step %2d  op=%2d %-9s  "
                   "%.6f->%.6f  gross=%7.2f  ovhd=%4.0f  net=%7.2f bits\n",
                   ++step, bestOp, opName[bestOp], curH, bestH,
                   gross, OP_BITS, opNet);
            totalOverhead += OP_BITS;
            curH = bestH;
            break;
        }
        case 1: {  /* op pair */
            double gross = (curH - bestPH) * SIZE;
            applyOp(data, SIZE, bestP1);
            applyOp(data, SIZE, bestP2);
            printf("Step %2d  pair=%2d+%2d %-9s+%-9s  "
                   "%.6f->%.6f  gross=%7.2f  ovhd=%4.0f  net=%7.2f bits\n",
                   ++step, bestP1, bestP2, opName[bestP1], opName[bestP2],
                   curH, bestPH, gross, PAIR_BITS, pairNet);
            totalOverhead += PAIR_BITS;
            curH = bestPH;
            break;
        }
        case 2: {  /* shuffle + single op */
            double gross = (curH - bSeedH) * SIZE;
            yatsShuffle(data, SIZE, bSeed);
            applyOp(data, SIZE, bSOp);
            printf("Step %2d  shuf(%5u)+op=%2d %-9s  "
                   "%.6f->%.6f  gross=%7.2f  ovhd=%4.0f  net=%7.2f bits\n",
                   ++step, (unsigned)bSeed, bSOp, opName[bSOp],
                   curH, bSeedH, gross, SHUFFLE_BITS, shuNet);
            totalOverhead += SHUFFLE_BITS;
            curH = bSeedH;
            break;
        }
        case 3: {  /* shuffle + op pair */
            double gross = (curH - bSPH) * SIZE;
            yatsShuffle(data, SIZE, bSeed);
            applyOp(data, SIZE, bSP1);
            applyOp(data, SIZE, bSP2);
            printf("Step %2d  shuf(%5u)+pair=%2d+%2d %-9s+%-9s  "
                   "%.6f->%.6f  gross=%7.2f  ovhd=%4.0f  net=%7.2f bits\n",
                   ++step, (unsigned)bSeed, bSP1, bSP2,
                   opName[bSP1], opName[bSP2],
                   curH, bSPH, gross, SHUPAR_BITS, shuPNet);
            totalOverhead += SHUPAR_BITS;
            curH = bSPH;
            break;
        }
        }
    }

    /* ---- summary ---- */
    double grossBits = (startH - curH) * SIZE;
    double netBits   = grossBits - totalOverhead;
    printf("\n+------------------------------------------+\n");
    printf("|             PROFIT SUMMARY               |\n");
    printf("+------------------------------------------+\n");
    printf("| Start entropy  %8.6f bits/byte       |\n", startH);
    printf("| End entropy    %8.6f bits/byte       |\n", curH);
    printf("|                                          |\n");
    printf("| (startH * SIZE) = %11.2f bits       |\n", startH * SIZE);
    printf("| (endH   * SIZE) = %11.2f bits       |\n", curH   * SIZE);
    printf("|                                          |\n");
    printf("| Gross saved    %10.2f bits           |\n", grossBits);
    printf("| Overhead       %10.2f bits           |\n", totalOverhead);
    printf("| Net profit     %10.2f bits           |\n", netBits);
    printf("| Net profit     %10.2f bytes          |\n", netBits / 8.0);
    printf("|                                          |\n");
    printf("| Steps          %10d               |\n", step);
    printf("+------------------------------------------+\n");

    free(data); free(tmp); free(tmp2); free(shuf);
    return 0;
}
