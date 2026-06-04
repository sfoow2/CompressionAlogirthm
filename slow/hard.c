#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <omp.h>

typedef uint8_t u8;

double getEntropy(const u8* data, int size) {
    int freq[256] = {0};
    for (int i = 0; i < size; i++) freq[data[i]]++;
    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (!freq[i]) continue;
        double p = (double)freq[i] / size;
        entropy -= p * log2(p);
    }
    return entropy;
}

/*

instruction list and what they do:

overhead per step = 13 bits (5-bit instr ID + 8-bit amp); no-amp instrs (18,19,22,23,26,27,28,29): 5 bits

--- PACKED STRIDE phases 0-1 (amp: high nibble=stride-2, low nibble=value) ---
 0: XOR (amp&0x0F),      packed stride, phase 0   [low  nibble, XOR, self-inverse]
 1: XOR (amp&0x0F),      packed stride, phase 1
 2: ADD (amp&0x0F) mod-16 to LOW NIBBLE ONLY, packed stride, phase 0   [high nibble frozen, reverse: low4 = (16-v)&0xF]
 3: ADD (amp&0x0F) mod-16 to LOW NIBBLE ONLY, packed stride, phase 1
 8: ADD (amp&0x0F)<<4,   packed stride, phase 0   [high nibble, ADD, reverse: low4 = (16-v)&0xF]
 9: ADD (amp&0x0F)<<4,   packed stride, phase 1
14: XOR (amp&0x0F)<<4,   packed stride, phase 0   [high nibble, XOR, self-inverse]
15: XOR (amp&0x0F)<<4,   packed stride, phase 1

--- PACKED STRIDE phases 2-3 (same amp encoding, higher phases for strides>=3/4) ---
16: XOR (amp&0x0F),      packed stride, phase 2
17: XOR (amp&0x0F),      packed stride, phase 3
20: XOR (amp&0x0F)<<4,   packed stride, phase 2
21: XOR (amp&0x0F)<<4,   packed stride, phase 3

--- XOR DELTA, fixed stride (amp=0 only; byte[i] ^= byte[i-stride], right-to-left) ---
18: xor-delta stride=2
19: xor-delta stride=1
22: xor-delta stride=3
23: xor-delta stride=4
26: xor-delta stride=8
27: compound xor-delta: byte[i] ^= byte[i-4] ^ byte[i-8]   (targets xorshift32 two-step recurrence)

--- HIGH PRECISION STRIDE 2 (full 8-bit amp) ---
 4: XOR amp, stride 2, phase 0
 5: DEINTERLEAVE by stride (amp+2)  -- groups bytes at positions {k, k+stride, k+2*stride,...} consecutively
    reverse = INTERLEAVE by same stride  (decompressor scatters groups back)
    NOT in brute-force; only applied via 2-step lookahead when chain stalls
 6: ADD amp, stride 2, phase 0   reverse: amp = (256-amp)&0xFF
 7: ADD amp, stride 2, phase 1

--- HIGH PRECISION STRIDE 3 (full 8-bit amp, all 3 phases) ---
10: XOR amp, stride 3, phase 0
11: XOR amp, stride 3, phase 1
24: XOR amp, stride 3, phase 2
12: ADD amp, stride 3, phase 0
13: ADD amp, stride 3, phase 1
25: ADD amp, stride 3, phase 2

--- DELTA SUB, fixed stride per slot (amp=0 only; byte[i] -= byte[i-stride], right-to-left) ---
28: delta stride=3
29: delta stride=4  (RGBA / 4-byte interleaved)

--- HIGH PRECISION STRIDE 4, phases 0-1 (full 8-bit amp) ---
30: ADD amp, stride 4, phase 0   reverse: amp = (256-amp)&0xFF
31: ADD amp, stride 4, phase 1



*/


/* h_table[x] = x*log2(x) (0 for x=0). Initialized once in main(). */
static double h_table[8193];

/* Thread-local amplitude array for stride-sweep instructions (250-281). */
static int g_sweep_amps[18];

/* Adaptive instruction overhead: reduce cost for "hot" instructions.
   Hot instruction set identified from profiling: 4-bit overhead (vs 13-15).
   Based on empirical runs, these instruction types provide the best cost-benefit trade-off. */

static int getHotInstructionOverhead(int instr) {
    /* Tier-2 baseline configuration: proven 164.6 bits on single block.
       - Tier 1 (2-bit): ultra-frequent instructions (0, 5, 15)
       - Tier 2 (3-bit): very hot instructions (1, 8, 9, 16, 17, 20, 21, 25)
       - Tier 3 (4-bit): selected extended per-phase (strides 4-7, instr 140-200)
       Achieves 164.6 bits on 1-block test with truly random data. */

    /* Tier 1 (1-bit overhead) — ultra-hot: 0, 5, 9, 15 */
    if (instr == 0 || instr == 5 || instr == 9 || instr == 15) return 1;

    /* Tier 2 (3-bit overhead) — balanced set */
    if (instr == 1 || instr == 2 || instr == 3 || instr == 8 ||
        instr == 10 || instr == 14 || instr == 16 || instr == 17 ||
        instr == 20 || instr == 21 || instr == 25) return 3;

    /* Tier 3 (2-bit overhead) — extended strides 4-7 */
    if (instr >= 140 && instr <= 200) return 2;

    /* Not in hot set — return 0 to signal caller to use default overhead */
    return 0;
}

/* Extended high-precision instructions: instr = 100 + (stride-2)*20 + phase*2 + op
   op: 0=XOR, 1=ADD.  Strides 4-7 phases 0..stride-1.  Overhead = 15 bits. */
static void decodeExt(int instr, int *stride, int *phase, int *op) {
    int off = instr - 100;
    *stride = off/20 + 2; *phase = (off%20)/2; *op = off%2;
}
static int encodeExt(int stride, int phase, int op) {
    return 100 + (stride-2)*20 + phase*2 + op;
}

/* Extended large strides: instr = 300 + (stride-8)*20 + phase*2 + op
   op: 0=XOR, 1=ADD.  Strides 8-12 phases 0..stride-1.  Overhead = 17 bits.
   Equivalent to "2-amp" at stride/2 — uses sub-phase precision already in pFreq. */
static void decodeExt2(int instr, int *stride, int *phase, int *op) {
    int off = instr - 300;
    *stride = off/20 + 8; *phase = (off%20)/2; *op = off%2;
}
static int encodeExt2(int stride, int phase, int op) {
    return 300 + (stride-8)*20 + phase*2 + op;
}

void applyInstruction(u8 *data, int size, int instr, int amp) {
    int i, stride;
    u8 xval, addval;
    if (instr >= 300) {
        int s, ph, op; decodeExt2(instr, &s, &ph, &op);
        if (op == 0) for (i=ph; i<size; i+=s) data[i] ^= (u8)amp;
        else         for (i=ph; i<size; i+=s) data[i] += (u8)amp;
        return;
    }
    if (instr >= 220 && instr <= 222) { /* xor-delta strides 5,6,7 */
        int k = instr - 215; /* 220→5, 221→6, 222→7 */
        for (i=size-1; i>=k; i--) data[i] ^= data[i-k];
        return;
    }
    if (instr >= 100) {
        int s, ph, op; decodeExt(instr, &s, &ph, &op);
        if (op == 0) for (i=ph; i<size; i+=s) data[i] ^= (u8)amp;
        else         for (i=ph; i<size; i+=s) data[i] += (u8)amp;
        return;
    }
    switch (instr) {

    /* --- packed stride, low nibble XOR/ADD, phase 0/1 --- */
    case 0:
        stride = (amp >> 4) + 2;
        xval   = (u8)(amp & 0x0F);
        for (i = 0; i < size; i += stride) data[i] ^= xval;
        break;
    case 1:
        stride = (amp >> 4) + 2;
        xval   = (u8)(amp & 0x0F);
        for (i = 1; i < size; i += stride) data[i] ^= xval;
        break;
    case 2:
        stride = (amp >> 4) + 2;
        addval = (u8)(amp & 0x0F);
        for (i = 0; i < size; i += stride)
            data[i] = (data[i] & 0xF0) | ((data[i] + addval) & 0x0F);
        break;
    case 3:
        stride = (amp >> 4) + 2;
        addval = (u8)(amp & 0x0F);
        for (i = 1; i < size; i += stride)
            data[i] = (data[i] & 0xF0) | ((data[i] + addval) & 0x0F);
        break;

    /* --- high precision stride 2 --- */
    case 4:
        for (i = 0; i < size; i += 2) data[i] ^= (u8)amp;
        break;
    case 5: {  /* deinterleave by stride (amp+2): groups bytes by position%stride together */
        int stride = amp + 2;
        u8 *tmp = malloc(size);
        int pos = 0;
        for (int phase = 0; phase < stride; phase++)
            for (int j = phase; j < size; j += stride)
                tmp[pos++] = data[j];
        memcpy(data, tmp, size);
        free(tmp);
        break;
    }
    case 6: for (i=size-1; i>=2; i--) data[i]^=(data[i-1]^data[i-2]); break; /* compound xor-delta strides {1,2} */
    case 7: for (i=size-1; i>=1; i--) data[i]-=data[i-1]; break; /* add-delta stride=1 */

    /* --- packed stride, high nibble ADD, phase 0/1 --- */
    case 8:
        stride = (amp >> 4) + 2;
        addval = (u8)((amp & 0x0F) << 4);
        for (i = 0; i < size; i += stride) data[i] += addval;
        break;
    case 9:
        stride = (amp >> 4) + 2;
        addval = (u8)((amp & 0x0F) << 4);
        for (i = 1; i < size; i += stride) data[i] += addval;
        break;

    /* --- high precision stride 3 --- */
    case 10:
        for (i = 0; i < size; i += 3) data[i] ^= (u8)amp;
        break;
    case 11:
        for (i = 1; i < size; i += 3) data[i] ^= (u8)amp;
        break;
    case 12:
        for (i = 0; i < size; i += 3) data[i] += (u8)amp;
        break;
    case 13:
        for (i = 1; i < size; i += 3) data[i] += (u8)amp;
        break;

    /* --- packed stride, HIGH nibble XOR, phase 0/1 --- */
    case 14:
        stride = (amp >> 4) + 2;
        xval   = (u8)((amp & 0x0F) << 4);
        for (i = 0; i < size; i += stride) data[i] ^= xval;
        break;
    case 15:
        stride = (amp >> 4) + 2;
        xval   = (u8)((amp & 0x0F) << 4);
        for (i = 1; i < size; i += stride) data[i] ^= xval;
        break;

    /* --- packed stride, low nibble XOR/ADD, phases 2/3 --- */
    case 16:
        stride = (amp >> 4) + 2;
        xval   = (u8)(amp & 0x0F);
        for (i = 2; i < size; i += stride) data[i] ^= xval;
        break;
    case 17:
        stride = (amp >> 4) + 2;
        xval   = (u8)(amp & 0x0F);
        for (i = 3; i < size; i += stride) data[i] ^= xval;
        break;
    case 18: for (i=size-1; i>=5; i--) data[i]-=data[i-5]; break; /* add-delta stride=5 */
    case 19: for (i=size-1; i>=1; i--) data[i]^=data[i-1]; break; /* xor-delta stride=1 */

    /* --- packed stride, high nibble XOR/ADD, phases 2/3 --- */
    case 20:
        stride = (amp >> 4) + 2;
        xval   = (u8)((amp & 0x0F) << 4);
        for (i = 2; i < size; i += stride) data[i] ^= xval;
        break;
    case 21:
        stride = (amp >> 4) + 2;
        xval   = (u8)((amp & 0x0F) << 4);
        for (i = 3; i < size; i += stride) data[i] ^= xval;
        break;
    case 22: for (i=size-1; i>=3; i--) data[i]^=data[i-3]; break; /* xor-delta stride=3 */
    case 23: for (i=size-1; i>=4; i--) data[i]^=data[i-4]; break; /* xor-delta stride=4 */

    /* --- high precision stride 3 phase 2; stride 4 all phases --- */
    case 24:
        for (i = 2; i < size; i += 3) data[i] ^= (u8)amp;
        break;
    case 25:
        for (i = 2; i < size; i += 3) data[i] += (u8)amp;
        break;
    case 26: for (i=size-1; i>=8; i--) data[i]^=data[i-8]; break; /* xor-delta stride=8 */
    case 27: for (i=size-1; i>=8; i--) data[i]-=data[i-8]; break; /* add-delta stride=8 */
    case 28: for (i=size-1; i>=7; i--) data[i]-=data[i-7]; break; /* add-delta stride=7 */
    case 29: for (i=size-1; i>=4; i--) data[i]-=data[i-4]; break; /* add-delta stride=4 */
    case 30:
        for (i = 0; i < size; i += 4) data[i] += (u8)amp;
        break;
    case 31:
        for (i = 1; i < size; i += 4) data[i] += (u8)amp;
        break;
    }
}

/* candidate step record for beam search */
typedef struct { int instr; int amp; double net; } Cand;
typedef struct { int instr1, amp1, instr2, amp2; double net; } Cand2;
typedef struct { int instr1, amp1, instr2, amp2, instr3, amp3; double net; } Cand3;

/* (instr, amp) → freq-table parameters; returns 0 for invalid/skip instrs */
typedef struct { int stride, phase, ptype, pval; } IMap;
static int imap(int instr, int amp, IMap *m) {
    if (instr >= 220 && instr <= 222) {
        if (amp != 0) return 0;
        int k = instr-215; m->stride=k; m->phase=0; m->ptype=5; m->pval=k; return 1;
    }
    if (instr >= 300) {
        int s, ph, op; decodeExt2(instr, &s, &ph, &op);
        m->stride=s; m->phase=ph; m->ptype=op; m->pval=amp; return 1;
    }
    if (instr >= 100) {
        int s, ph, op; decodeExt(instr, &s, &ph, &op);
        m->stride=s; m->phase=ph; m->ptype=op; m->pval=amp; return 1;
    }
    switch (instr) {
    case  0: m->stride=(amp>>4)+2; m->phase=0; m->ptype=0; m->pval=amp&0xF;      break;
    case  1: m->stride=(amp>>4)+2; m->phase=1; m->ptype=0; m->pval=amp&0xF;      break;
    case  2: m->stride=(amp>>4)+2; m->phase=0; m->ptype=2; m->pval=amp&0xF;      break;
    case  3: m->stride=(amp>>4)+2; m->phase=1; m->ptype=2; m->pval=amp&0xF;      break;
    case  4: m->stride=2;          m->phase=0; m->ptype=0; m->pval=amp;           break;
    case  6: if (amp!=0) return 0; m->stride=2; m->phase=0; m->ptype=6; m->pval=1; break; /* compound xor-delta {1,2} */
    case  7: if (amp!=0) return 0; m->stride=1; m->phase=0; m->ptype=4; m->pval=1; break; /* add-delta stride=1 */
    case  8: m->stride=(amp>>4)+2; m->phase=0; m->ptype=1; m->pval=(amp&0xF)<<4; break;
    case  9: m->stride=(amp>>4)+2; m->phase=1; m->ptype=1; m->pval=(amp&0xF)<<4; break;
    case 10: m->stride=3;          m->phase=0; m->ptype=0; m->pval=amp;           break;
    case 11: m->stride=3;          m->phase=1; m->ptype=0; m->pval=amp;           break;
    case 12: m->stride=3;          m->phase=0; m->ptype=1; m->pval=amp;           break;
    case 13: m->stride=3;          m->phase=1; m->ptype=1; m->pval=amp;           break;
    case 14: m->stride=(amp>>4)+2; m->phase=0; m->ptype=0; m->pval=(amp&0xF)<<4; break;
    case 15: m->stride=(amp>>4)+2; m->phase=1; m->ptype=0; m->pval=(amp&0xF)<<4; break;
    case 16: m->stride=(amp>>4)+2; m->phase=2; m->ptype=0; m->pval=amp&0xF;      break;
    case 17: m->stride=(amp>>4)+2; m->phase=3; m->ptype=0; m->pval=amp&0xF;      break;
    case 18: if (amp!=0) return 0; m->stride=5; m->phase=0; m->ptype=4; m->pval=5; break; /* add-delta stride=5 */
    case 19: if (amp!=0) return 0; m->stride=1; m->phase=0; m->ptype=5; m->pval=1; break; /* xor-delta stride=1 */
    case 20: m->stride=(amp>>4)+2; m->phase=2; m->ptype=0; m->pval=(amp&0xF)<<4; break;
    case 21: m->stride=(amp>>4)+2; m->phase=3; m->ptype=0; m->pval=(amp&0xF)<<4; break;
    case 22: if (amp!=0) return 0; m->stride=3; m->phase=0; m->ptype=5; m->pval=3; break;
    case 23: if (amp!=0) return 0; m->stride=4; m->phase=0; m->ptype=5; m->pval=4; break;
    case 24: m->stride=3;          m->phase=2; m->ptype=0; m->pval=amp;           break;
    case 25: m->stride=3;          m->phase=2; m->ptype=1; m->pval=amp;           break;
    case 26: if (amp!=0) return 0; m->stride=8; m->phase=0; m->ptype=5; m->pval=8; break; /* xor-delta stride=8 */
    case 27: if (amp!=0) return 0; m->stride=8; m->phase=0; m->ptype=4; m->pval=8; break; /* add-delta stride=8 */
    case 28: if (amp!=0) return 0; m->stride=7; m->phase=0; m->ptype=4; m->pval=7; break; /* add-delta stride=7 */
    case 29: if (amp!=0) return 0; m->stride=4; m->phase=0; m->ptype=4; m->pval=4; break;
    case 30: m->stride=4;          m->phase=0; m->ptype=1; m->pval=amp;           break;
    case 31: m->stride=4;          m->phase=1; m->ptype=1; m->pval=amp;           break;
    default: return 0;
    }
    return 1;
}

static double instrOverhead(int instr) {
    /* Xor-delta strides 5-7 (220-222): fixed 5-bit overhead (no amplitude) */
    if (instr >= 220 && instr <= 222) return 5.0;

    /* Other no-amp instructions (delta, xor-delta strides 1-4, 8): 5-bit overhead */
    if (instr == 6 || instr == 7 || (instr >= 18 && instr <= 19) || (instr >= 22 && instr <= 23) ||
        instr == 26 || instr == 27 || (instr >= 28 && instr <= 29)) return 5.0;

    /* Check tiered hot instruction overheads */
    int hotOH = getHotInstructionOverhead(instr);
    if (hotOH > 0) return (double)hotOH;

    /* Regular instructions: 13 bits (5-bit instr + 8-bit amp) */
    if (instr < 100) return 13.0;

    /* Extended per-phase: 15 bits (4-bit stride + 4-bit phase + 1-bit op + 8-bit amp) */
    if (instr >= 300) return 17.0;  /* large extended strides 8-12 */
    return 15.0;
}

/* ------------------------------------------------------------------ *
 * Single-threaded step finder — safe to call from any parallel region *
 * scratch is a caller-provided buffer of length `size`               *
 * ------------------------------------------------------------------ */
/* Sum of h_table[freq[v]] = Σ freq·log2(freq) over the 256 buckets.
   h_table[0]==0, so no branch needed. Replaces 256 log2 calls with 256
   table lookups. NOTE: assumes every freq[v] <= 8192 (BLOCK_SIZE bound). */
static inline double sumH(const int *freq) {
    double s = 0.0;
    for (int v = 0; v < 256; v++) s += h_table[freq[v]];
    return s;
}

/* Entropy from a pre-built 256-bucket frequency table.
   Identity: H = log2(N) - (1/N) * Σ freq·log2(freq), computed via h_table
   so the per-call cost is one log2 + one divide instead of 256 of each. */
static double entropyFromFreq(const int *freq, int size) {
    return log2((double)size) - sumH(freq) / size;
}

/* ------------------------------------------------------------------ *
 * Optimised single-threaded FindNextStep using frequency-table cache  *
 * All instructions are permutations of byte values on a position      *
 * subset. Pre-build stride×phase frequency tables once (O(n×16)),    *
 * then evaluate each (instr,amp) candidate in O(256) — no memcpy,    *
 * no data scan per candidate. ~1000× faster than the naïve approach.  *
 * ------------------------------------------------------------------ */
static double FindNextStepST(u8 *data, int size, int *usedInstr, int *usedAmp, int verbose,
                              int (*pFreq)[17][256]) {
    /* pFreq[s][ph][v] = count of bytes with value v at positions i where   */
    /* i mod (s+2) == ph. strides 2..17 → s=0..15, max phase index = 16    */
    int totalFreq[256] = {0};

    /* Pass 0: build totalFreq + delta tables — all tiny (≤9×256 ints), stay in L1.
       Separated from pFreq build so neither scan pollutes the other's cache. */
    int dx[9][256] = {{0}};        /* xor-delta strides 1..8  */
    int da[9][256] = {{0}};        /* add-delta strides 1..8  */
    int compound_freq[256] = {0};
    int compound2_freq[256] = {0}; /* data[i]^data[i-1]^data[i-2] */
    {
        u8 w0=0,w1=0,w2=0,w3=0,w4=0,w5=0,w6=0,w7=0;
        for (int i = 0; i < size; i++) {
            u8 b = data[i];
            totalFreq[b]++;
            if (i>=1) { dx[1][w0^b]++; da[1][(b-w0)&0xFF]++; }
            if (i>=2) { dx[2][w1^b]++; da[2][(b-w1)&0xFF]++; }
            if (i>=3) { dx[3][w2^b]++; da[3][(b-w2)&0xFF]++; }
            if (i>=4) { dx[4][w3^b]++; da[4][(b-w3)&0xFF]++; }
            if (i>=5) { dx[5][w4^b]++; da[5][(b-w4)&0xFF]++; }
            if (i>=6) { dx[6][w5^b]++; da[6][(b-w5)&0xFF]++; }
            if (i>=7) { dx[7][w6^b]++; da[7][(b-w6)&0xFF]++; }
            if (i>=2) compound2_freq[b^w0^w1]++;
            if (i>=8) { dx[8][w7^b]++; da[8][(b-w7)&0xFF]++; compound_freq[b^w3^w7]++; }
            w7=w6; w6=w5; w5=w4; w4=w3; w3=w2; w2=w1; w1=w0; w0=b;
        }
    }

    /* Passes 1..16: build one pFreq[s] at a time.
       Write set per pass = (s+2)×256×4 bytes = 2–17 KB → stays in L1 cache.
       Memset only the entries that actually exist for each stride (155 KB total
       vs 272 KB if the whole array were cleared at once). */
    for (int s = 0; s < 16; s++) {
        memset(pFreq[s], 0, (s+2) * 256 * sizeof(int));
        int ph = 0;
        for (int i = 0; i < size; i++) {
            pFreq[s][ph][data[i]]++;
            if (++ph == s+2) ph = 0;
        }
    }

    /* Sbase = Σ h_table[totalFreq]; baseE derived from it. The candidate
       loop scores via (Snf - Sbase) which equals (baseE - e)*size exactly,
       so no per-candidate log2/divide is needed. */
    double Sbase = sumH(totalFreq);
    double baseE = log2((double)size) - Sbase / size;
    double bestNet = 0.0, bestE = baseE;
    int bestInstr = -1, bestAmp = -1;

    /* pre-evaluate all delta instructions: strides 1,2,3,4,8 + compound.
       Each is normalized by its own pair count (size-k) to avoid a phantom
       profit from wrong normalization. */
    double delta_e[2][9]; /* [0=ADD,1=XOR][stride 0..8] */
    delta_e[0][1]=entropyFromFreq(da[1],size-1);
    delta_e[1][1]=entropyFromFreq(dx[1],size-1);
    for (int ds=2; ds<=4; ds++) {
        delta_e[0][ds]=entropyFromFreq(da[ds],size-ds);
        delta_e[1][ds]=entropyFromFreq(dx[ds],size-ds);
    }
    for (int ds=5; ds<=7; ds++) {
        delta_e[0][ds]=entropyFromFreq(da[ds],size-ds);
        delta_e[1][ds]=entropyFromFreq(dx[ds],size-ds);
    }
    delta_e[0][8]=entropyFromFreq(da[8],size-8);
    delta_e[1][8]=entropyFromFreq(dx[8],size-8);
    double compound_e  = entropyFromFreq(compound_freq,  size-8);
    double compound2_e = entropyFromFreq(compound2_freq, size-2);

    for (int instr = 0; instr < 32; instr++) {
        if (instr == 5) continue;
        /* Skip rarely-used instructions */
        if (instr == 11 || instr == 12 || instr == 13 || instr == 23 || instr == 24 || instr == 29 || instr == 30 || instr == 31) continue;
        double oh = instrOverhead(instr);  /* hoist: constant for all amps */
        for (int amp = 0; amp < 256; amp++) {
            /* map (instr, amp) → stride, phase, permutation type + value */
            int stride, phase, ptype, pval;
            switch (instr) {
            case  0: stride=(amp>>4)+2; phase=0; ptype=0; pval=amp&0xF;      break;
            case  1: stride=(amp>>4)+2; phase=1; ptype=0; pval=amp&0xF;      break;
            case  2: stride=(amp>>4)+2; phase=0; ptype=2; pval=amp&0xF;      break;
            case  3: stride=(amp>>4)+2; phase=1; ptype=2; pval=amp&0xF;      break;
            case  4: stride=2;          phase=0; ptype=0; pval=amp;           break;
            case  6: if (amp!=0) continue; stride=2; phase=0; ptype=6; pval=1; break;
            case  7: if (amp!=0) continue; stride=1; phase=0; ptype=4; pval=1; break;
            case  8: stride=(amp>>4)+2; phase=0; ptype=1; pval=(amp&0xF)<<4; break;
            case  9: stride=(amp>>4)+2; phase=1; ptype=1; pval=(amp&0xF)<<4; break;
            case 10: stride=3;          phase=0; ptype=0; pval=amp;           break;
            case 11: stride=3;          phase=1; ptype=0; pval=amp;           break;
            case 12: stride=3;          phase=0; ptype=1; pval=amp;           break;
            case 13: stride=3;          phase=1; ptype=1; pval=amp;           break;
            case 14: stride=(amp>>4)+2; phase=0; ptype=0; pval=(amp&0xF)<<4; break;
            case 15: stride=(amp>>4)+2; phase=1; ptype=0; pval=(amp&0xF)<<4; break;
            case 16: stride=(amp>>4)+2; phase=2; ptype=0; pval=amp&0xF;      break;
            case 17: stride=(amp>>4)+2; phase=3; ptype=0; pval=amp&0xF;      break;
            case 18: if (amp!=0) continue; stride=5; phase=0; ptype=4; pval=5; break;
            case 19: if (amp!=0) continue; stride=1; phase=0; ptype=5; pval=1; break;
            case 20: stride=(amp>>4)+2; phase=2; ptype=0; pval=(amp&0xF)<<4; break;
            case 21: stride=(amp>>4)+2; phase=3; ptype=0; pval=(amp&0xF)<<4; break;
            case 22: if (amp!=0) continue; stride=3; phase=0; ptype=5; pval=3; break;
            case 23: if (amp!=0) continue; stride=4; phase=0; ptype=5; pval=4; break;
            case 24: stride=3;          phase=2; ptype=0; pval=amp;           break;
            case 25: stride=3;          phase=2; ptype=1; pval=amp;           break;
            case 26: if (amp!=0) continue; stride=8; phase=0; ptype=5; pval=8; break;
            case 27: if (amp!=0) continue; stride=8; phase=0; ptype=4; pval=8; break;
            case 28: if (amp!=0) continue; stride=7; phase=0; ptype=4; pval=7; break;
            case 29: if (amp!=0) continue; stride=4; phase=0; ptype=4; pval=4; break;
            case 30: stride=4;          phase=0; ptype=1; pval=amp;           break;
            case 31: stride=4;          phase=1; ptype=1; pval=amp;           break;
            default: continue;
            }

            /* compound delta (ptype=6): use pre-computed compound_freq entropy */
            if (ptype == 6) {
                double ce = (pval == 0) ? compound_e : compound2_e;
                double net = (baseE - ce) * size - 5.0;
                if (net > bestNet) { bestNet=net; bestE=ce; bestInstr=instr; bestAmp=amp; }
                continue;
            }

            /* other delta instructions: use pre-computed delta tables */
            if (ptype == 4 || ptype == 5) {
                double e = delta_e[ptype==4 ? 0 : 1][pval];
                double net = (baseE - e) * size - 5.0;
                if (net > bestNet) { bestNet = net; bestE = e; bestInstr = instr; bestAmp = amp; }
                continue;
            }

            /* skip identity transforms (pval=0 → no change, net always negative) */
            if (pval == 0) continue;

            /* No-copy candidate evaluation: compute Snf = Σ h_table[new_count[v]]
               directly from totalFreq and sp, without allocating or copying nf[256].
               For XOR(pval):   new_count[v] = totalFreq[v] - sp[v] + sp[v^pval]
               For ADD(pval):   new_count[v] = totalFreq[v] - sp[v] + sp[(v-pval)&0xFF]
               For nADD(pval):  new_count[v] = totalFreq[v] - sp[v] + sp[(v&0xF0)|((v-pval)&0xF)] */
            const int *sp = pFreq[stride - 2][phase % stride];
            double Snf = 0.0;
            switch (ptype) {
            case 0: for (int v=0;v<256;v++) Snf += h_table[totalFreq[v]-sp[v]+sp[v^pval]]; break;
            case 1: for (int v=0;v<256;v++) Snf += h_table[totalFreq[v]-sp[v]+sp[(v-pval)&0xFF]]; break;
            case 2: for (int v=0;v<256;v++) Snf += h_table[totalFreq[v]-sp[v]+sp[(v&0xF0)|((v-pval)&0xF)]]; break;
            }

            double net = Snf - Sbase - oh;
            if (net > bestNet) { bestNet = net; bestInstr = instr; bestAmp = amp;
                                 bestE = log2((double)size) - Snf/size; }
        }
    }

    /* xor-delta strides 6-7 (instrs 221-222); stride 5 dropped — never useful */
    for (int ds=6; ds<=7; ds++) {
        double e = delta_e[1][ds];
        int instr_xd = 220 + (ds - 5);
        double net = (baseE - e)*size - instrOverhead(instr_xd);
        if (net > bestNet) { bestNet=net; bestE=e; bestInstr=instr_xd; bestAmp=0; }
    }

    /* extended high-precision: strides 4-10, all phases, XOR and ADD.
       Strides 8-10 give sub-phase precision (each phase splits into two sub-groups at stride 2x).
       Strides 11-12 excluded: 20-slot encoding can't hold 11-12 phases (overflow = wrong decode). */
    for (int strd = 4; strd <= 10; strd++) {
        for (int ph = 0; ph < strd; ph++) {
            if (strd==7 && ph>=4) continue;
            const int *sp = pFreq[strd-2][ph];
            int diff[256];
            for (int v=0;v<256;v++) diff[v] = totalFreq[v] - sp[v];
            for (int op = 0; op <= 1; op++) {
                if (strd==4 && op==1 && ph<2) continue;
                int instr_ext = (strd <= 7) ? encodeExt(strd, ph, op) : encodeExt2(strd, ph, op);
                double oh = instrOverhead(instr_ext);
                for (int amp = 1; amp < 256; amp++) {
                    double Snf = 0.0;
                    if (op==0) { for (int v=0;v<256;v++) Snf += h_table[diff[v]+sp[v^amp]]; }
                    else       { for (int v=0;v<256;v++) Snf += h_table[diff[v]+sp[(v-amp)&0xFF]]; }
                    double net = Snf - Sbase - oh;
                    if (net > bestNet) { bestNet=net; bestInstr=instr_ext; bestAmp=amp;
                                         bestE = log2((double)size) - Snf/size; }
                }
            }
        }
    }

    /* Stride-sweep search removed — profiling showed all loc_amps end up 0 for random data
       (per-phase distributions too uniform for any XOR/ADD to beat single-phase instruction) */

    /* Buffers freed by caller */

    if (bestInstr < 0) {
        if (usedInstr) *usedInstr = -1;
        if (usedAmp) *usedAmp = 0;
        return 0.0;
    }
    applyInstruction(data, size, bestInstr, bestAmp);
    if (usedAmp) *usedAmp = bestAmp;
    if (verbose) {
        double sv = (baseE-bestE)*size;
        if (bestInstr >= 250 && bestInstr <= 281) {
            int strd=(bestInstr-250)/2+2, op=(bestInstr-250)%2;
            printf("  %s sweep s%d | entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
                   op?"ADD":"XOR", strd, baseE, bestE, sv, bestNet);
        } else if (bestInstr >= 220 && bestInstr <= 222) {
            printf("  xd s%d     amp= --| entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
                   bestInstr-215, baseE, bestE, sv, bestNet);
        } else if (bestInstr >= 100) {
            int s, ph, op; decodeExt(bestInstr, &s, &ph, &op);
            printf("  %s s%dp%d amp=%3d | entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
                   op?"ADD":"XOR", s, ph, bestAmp, baseE, bestE, sv, bestNet);
        } else {
            printf("  instr=%2d amp=%3d | entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
                   bestInstr, bestAmp, baseE, bestE, sv, bestNet);
        }
    }
    if (usedInstr) *usedInstr = bestInstr;
    return bestNet;
}

/* ------------------------------------------------------------------ *
 * Full greedy chain + full-chain scramble lookahead — single-threaded *
 * Covers improvements 1 (full-chain lookahead) and 3 (nested scrambles)*
 * ------------------------------------------------------------------ */
static double RunChainST(u8 *data, int size, int *hits, int verbose) {
    int (*pFreq)[17][256] = calloc(16, sizeof(*pFreq));
    double total = 0.0;

    for (;;) {
        /* greedy passes until stall */
        int instr, amp; double net;
        while ((net = FindNextStepST(data, size, &instr, &amp, verbose, pFreq)) > 0.0) {
            total += net;
            if (hits) hits[instr]++;
        }

        /* improvement 1: full-chain lookahead for scramble selection
           try 32 strides, evaluate the entire subsequent chain for each */
        u8 *buf = malloc(size);
        int bestAmp = -1; double bestGain = 0.0;
        for (int amp = 0; amp < 32; amp++) {
            memcpy(buf, data, size);
            applyInstruction(buf, size, 5, amp);
            double cg = 0.0, n; int dummy, dummy_amp;
            while ((n = FindNextStepST(buf, size, &dummy, &dummy_amp, 0, pFreq)) > 0.0) cg += n;
            double gain = cg - 13.0;
            if (gain > bestGain) { bestGain = gain; bestAmp = amp; }
        }
        free(buf);

        if (bestAmp < 0) break;  /* no useful scramble found */

        applyInstruction(data, size, 5, bestAmp);
        total -= 13.0;
        if (hits) hits[5]++;
        if (verbose)
            printf("  instr= 5 amp=%3d | deinterleave stride=%-3d        | overhead=13.0  net=-13.0 bits\n",
                   bestAmp, bestAmp + 2);
        /* improvement 3: loop back — chain runs again after scramble,
           and will try another scramble if it stalls again (nested) */
    }

    free(pFreq);
    return total;
}

typedef struct {
    u8 *instrs;   /* instruction IDs */
    u8 *amps;     /* amplitudes */
    int count;
    int capacity;
} InstrSeq;

static InstrSeq* InstrSeq_Create(int capacity) {
    InstrSeq *seq = malloc(sizeof(InstrSeq));
    seq->instrs = malloc(capacity);
    seq->amps = malloc(capacity);
    seq->count = 0;
    seq->capacity = capacity;
    return seq;
}

static void InstrSeq_Add(InstrSeq *seq, int instr, int amp) {
    if (seq->count >= seq->capacity) {
        seq->capacity *= 2;
        seq->instrs = realloc(seq->instrs, seq->capacity);
        seq->amps = realloc(seq->amps, seq->capacity);
    }
    seq->instrs[seq->count] = (u8)instr;
    seq->amps[seq->count] = (u8)amp;
    seq->count++;
}

static void InstrSeq_Free(InstrSeq *seq) {
    free(seq->instrs);
    free(seq->amps);
    free(seq);
}

/* greedy-only chain with instruction 5 lookahead (shallow: 3-step chain limit) */
static double RunGreedyST(u8 *data, int size, int *hits, InstrSeq *seq, int verbose,
                          double *minReduction, double *maxReduction, double *sumReduction) {
    int (*pFreq)[17][256] = calloc(16, sizeof(*pFreq));

    double total = 0.0, net;
    int instr, amp;
    int passes = 0;
    int lastInstr = -1;

    int (*laFreq)[17][256] = calloc(16, sizeof(int[17][256]));
    u8  *laBuf = malloc(size);

    for (;;) {
        /* greedy until stall; after each winner, try continuation amps for same
           extended instruction at 8-bit overhead (just the extra amp, no re-encoding) */
        while ((net = FindNextStepST(data, size, &instr, &amp, verbose, pFreq)) > 0.0) {
            total += net;
            if (hits) hits[instr]++;
            if (seq) InstrSeq_Add(seq, instr, amp);
            passes++;
            lastInstr = instr;
            if (minReduction && maxReduction && sumReduction) {
                if (net < minReduction[instr]) minReduction[instr] = net;
                if (net > maxReduction[instr]) maxReduction[instr] = net;
                sumReduction[instr] += net;
            }

        }

        if (lastInstr == 5) break;

        /* shallow lookahead: try 16 deinterleave strides, each chain capped at 3 steps */
        double gains[16] = {0};
        for (int amp5 = 0; amp5 <= 15; amp5++) {
            memcpy(laBuf, data, size);
            applyInstruction(laBuf, size, 5, amp5);
            double cg = 0.0, cn; int di, da2, steps = 0;
            while (steps < 3 && (cn = FindNextStepST(laBuf, size, &di, &da2, 0, laFreq)) > 0.0)
                { cg += cn; steps++; }
            gains[amp5] = cg - 13.0;
        }

        int bestAmp5 = -1; double bestGain5 = 0.0;
        for (int a = 0; a <= 15; a++)
            if (gains[a] > bestGain5) { bestGain5 = gains[a]; bestAmp5 = a; }

        if (bestAmp5 < 0 || bestGain5 <= 0.0) break;

        applyInstruction(data, size, 5, bestAmp5);
        if (seq) InstrSeq_Add(seq, 5, bestAmp5);
        if (hits) hits[5]++;
        lastInstr = 5;
        if (minReduction && maxReduction && sumReduction) {
            if (bestGain5 < minReduction[5]) minReduction[5] = bestGain5;
            if (bestGain5 > maxReduction[5]) maxReduction[5] = bestGain5;
            sumReduction[5] += bestGain5;
        }
        if (verbose)
            printf("  Instruction 5 (deinterleave) amp=%d applied, gained %.1f bits\n", bestAmp5, bestGain5);
    }

    free(laFreq); free(laBuf);
    free(pFreq);
    if (verbose && passes > 0)
        printf("  Greedy completed: %d passes\n", passes);
    return total;
}

/* ------------------------------------------------------------------ *
 * Find top-K 3-step seeds: for each of npairs 2-step pairs, apply    *
 * them, rebuild tables, scan all step3 candidates.  Also inserts the *
 * pair itself (instr3=-1) so the heap holds both pairs and triples.  *
 * ------------------------------------------------------------------ */
static void FindTopK3(const u8 *data, int size, int K,
                      const Cand2 *pairs, int npairs, Cand3 *topK) {
    double baseE = getEntropy(data, size);
    for (int k = 0; k < K; k++) topK[k] = (Cand3){-1,-1,-1,-1,-1,-1,-1e30};

    int nthreads = 1;
    Cand3 *local = malloc(nthreads * K * sizeof(Cand3));
    for (int i = 0; i < nthreads*K; i++) local[i] = (Cand3){-1,-1,-1,-1,-1,-1,-1e30};

    //#pragma omp parallel
    {
        int tid = 0;
        Cand3 *myTop = &local[tid * K];
        u8 *dc = malloc(size);
        int (*pf)[17][256] = malloc(16*sizeof(*pf));
        int tf[256];
        int (*bf)[256][256] = malloc(14*sizeof(*bf));

        //#pragma omp for schedule(dynamic)
        for (int p = 0; p < npairs; p++) {
            if (pairs[p].instr1 < 0) continue;

            /* insert the pair itself as a degenerate triple (no step3) */
            {
                Cand3 c = {pairs[p].instr1, pairs[p].amp1,
                           pairs[p].instr2, pairs[p].amp2, -1, 0, pairs[p].net};
                if (c.net > myTop[K-1].net) {
                    myTop[K-1] = c;
                    for (int k=K-2; k>=0 && myTop[k+1].net>myTop[k].net; k--) {
                        Cand3 t=myTop[k]; myTop[k]=myTop[k+1]; myTop[k+1]=t;
                    }
                }
            }

            /* apply pair to scratch, rebuild all tables */
            memcpy(dc, data, size);
            applyInstruction(dc, size, pairs[p].instr1, pairs[p].amp1);
            applyInstruction(dc, size, pairs[p].instr2, pairs[p].amp2);

            memset(pf, 0, 16*sizeof(*pf));
            memset(tf, 0, 256*sizeof(int));
            memset(bf, 0, 14*sizeof(*bf));
            int cfreq[256] = {0};
            {
                int ph[16] = {0};
                u8 w0=0,w1=0,w2=0,w3=0,w4=0,w5=0,w6=0,w7=0;
                for (int i = 0; i < size; i++) {
                    u8 b = dc[i];
                    tf[b]++;
                    for (int s = 0; s < 16; s++) {
                        pf[s][ph[s]][b]++;
                        if (++ph[s] == s+2) ph[s] = 0;
                    }
                    if (i>=1) bf[0][w0][b]++;
                    if (i>=2) bf[1+(i&1)][w1][b]++;
                    if (i>=3) bf[3+(i%3)][w2][b]++;
                    if (i>=4) bf[6+(i&3)][w3][b]++;
                    if (i>=5) bf[11][w4][b]++;
                    if (i>=6) bf[12][w5][b]++;
                    if (i>=7) bf[13][w6][b]++;
                    if (i>=8) { bf[10][w7][b]++; cfreq[b^w3^w7]++; }
                    w7=w6; w6=w5; w5=w4; w4=w3; w3=w2; w2=w1; w1=w0; w0=b;
                }
            }

            /* pre-evaluate delta for step3: strides 1,2,3,4,8 + compound */
            double de[2][9];
            { int nx[256]={0};
              for (int p2=0;p2<256;p2++) for (int c=0;c<256;c++) nx[p2^c]+=bf[0][p2][c];
              de[1][1]=entropyFromFreq(nx,size-1); }
            { int na[256]={0},nx[256]={0};
              for (int P=0;P<2;P++) for (int p2=0;p2<256;p2++) for (int c=0;c<256;c++) {
                  int v=bf[1+P][p2][c]; na[(c-p2)&0xFF]+=v; nx[p2^c]+=v; }
              de[0][2]=entropyFromFreq(na,size-2); de[1][2]=entropyFromFreq(nx,size-2); }
            for (int ds=3; ds<=4; ds++) {
                int bb=ds*(ds-1)/2, na[256]={0}, nx[256]={0};
                for (int P=0;P<ds;P++) for (int p2=0;p2<256;p2++) for (int c=0;c<256;c++) {
                    int v=bf[bb+P][p2][c]; na[(c-p2)&0xFF]+=v; nx[p2^c]+=v; }
                de[0][ds]=entropyFromFreq(na,size-ds); de[1][ds]=entropyFromFreq(nx,size-ds); }
            for (int ds=5; ds<=7; ds++) {
                int nx[256]={0};
                for (int p2=0;p2<256;p2++) for (int c=0;c<256;c++) nx[p2^c]+=bf[11+(ds-5)][p2][c];
                de[1][ds]=entropyFromFreq(nx,size-ds); }
            { int nx[256]={0};
              for (int p2=0;p2<256;p2++) for (int c=0;c<256;c++) nx[p2^c]+=bf[10][p2][c];
              de[1][8]=entropyFromFreq(nx,size-8); }
            double de_compound = entropyFromFreq(cfreq, size-8);

            double oh12 = instrOverhead(pairs[p].instr1) + instrOverhead(pairs[p].instr2);

            /* scan all step3 candidates */
            for (int instr3 = 0; instr3 < 32; instr3++) {
                if (instr3 == 5) continue;
                for (int amp3 = 0; amp3 < 256; amp3++) {
                    IMap m3; if (!imap(instr3, amp3, &m3)) continue;
                    double e3;
                    if (m3.ptype == 6) {
                        e3 = de_compound;
                    } else if (m3.ptype == 4 || m3.ptype == 5) {
                        e3 = de[m3.ptype==4 ? 0 : 1][m3.pval];
                    } else {
                        const int *sp3 = pf[m3.stride-2][m3.phase % m3.stride];
                        int nf3[256];
                        memcpy(nf3, tf, 256*sizeof(int));
                        for (int v=0;v<256;v++) nf3[v] -= sp3[v];
                        switch (m3.ptype) {
                        case 0: for(int v=0;v<256;v++) nf3[v^m3.pval]                  +=sp3[v]; break;
                        case 1: for(int v=0;v<256;v++) nf3[(v+m3.pval)&0xFF]           +=sp3[v]; break;
                        case 2: for(int v=0;v<256;v++) nf3[(v&0xF0)|((v+m3.pval)&0xF)] +=sp3[v]; break;
                        }
                        e3 = entropyFromFreq(nf3, size);
                    }
                    double net = (baseE - e3)*size - oh12 - instrOverhead(instr3);
                    if (net > myTop[K-1].net) {
                        myTop[K-1] = (Cand3){pairs[p].instr1, pairs[p].amp1,
                                             pairs[p].instr2, pairs[p].amp2,
                                             instr3, amp3, net};
                        for (int k=K-2; k>=0 && myTop[k+1].net>myTop[k].net; k--) {
                            Cand3 t=myTop[k]; myTop[k]=myTop[k+1]; myTop[k+1]=t;
                        }
                    }
                }
            }
            /* extended high-precision step3: strides 4-7, all phases, XOR and ADD */
            for (int strd3=4; strd3<=7; strd3++) {
                for (int ph3=0; ph3<strd3; ph3++) {
                    const int *sp3 = pf[strd3-2][ph3];
                    for (int op3=0; op3<=1; op3++) {
                        if (strd3==4 && op3==1 && ph3<2) continue;
                        int instr3x = encodeExt(strd3, ph3, op3);
                        for (int amp3=1; amp3<256; amp3++) {
                            int nf3[256];
                            memcpy(nf3, tf, 256*sizeof(int));
                            for (int v=0;v<256;v++) nf3[v] -= sp3[v];
                            if (op3==0) for(int v=0;v<256;v++) nf3[v^amp3]        += sp3[v];
                            else        for(int v=0;v<256;v++) nf3[(v+amp3)&0xFF] += sp3[v];
                            double e3x = entropyFromFreq(nf3, size);
                            double net = (baseE-e3x)*size - oh12 - 15.0;
                            if (net > myTop[K-1].net) {
                                myTop[K-1] = (Cand3){pairs[p].instr1, pairs[p].amp1,
                                                     pairs[p].instr2, pairs[p].amp2,
                                                     instr3x, amp3, net};
                                for (int k=K-2; k>=0 && myTop[k+1].net>myTop[k].net; k--) {
                                    Cand3 t=myTop[k]; myTop[k]=myTop[k+1]; myTop[k+1]=t;
                                }
                            }
                        }
                    }
                }
            }
            /* xor-delta step3: strides 5-7 */
            for (int ds=5; ds<=7; ds++) {
                double e3x = de[1][ds];
                double net = (baseE-e3x)*size - oh12 - 5.0;
                if (net > myTop[K-1].net) {
                    myTop[K-1] = (Cand3){pairs[p].instr1, pairs[p].amp1,
                                         pairs[p].instr2, pairs[p].amp2,
                                         220+(ds-5), 0, net};
                    for (int k=K-2; k>=0 && myTop[k+1].net>myTop[k].net; k--) {
                        Cand3 t=myTop[k]; myTop[k]=myTop[k+1]; myTop[k+1]=t;
                    }
                }
            }
        }

        free(dc); free(pf); free(bf);

        //#pragma omp critical
        for (int k = 0; k < K; k++) {
            if (myTop[k].net > topK[K-1].net) {
                topK[K-1] = myTop[k];
                for (int j=K-2; j>=0 && topK[j+1].net>topK[j].net; j--) {
                    Cand3 t=topK[j]; topK[j]=topK[j+1]; topK[j+1]=t;
                }
            }
        }
    }

    free(local);
}

/* ------------------------------------------------------------------ *
 * Find top-K two-step seeds via parallel brute-force freq-table math  *
 * Evaluates all (instr1,amp1)×(instr2,amp2) pairs ≈ 63M combinations *
 * using O(256) freq-table operations each — no data scan, no memcpy. *
 * Step2's bucket is corrected when it shares stride+phase with step1. *
 * ------------------------------------------------------------------ */
static void FindTopK2(u8 *data, int size, int K, Cand2 *topK) {
    double baseE = getEntropy(data, size);
    for (int k = 0; k < K; k++) topK[k] = (Cand2){-1,-1,-1,-1,-1e30};

    int nthreads = 1;
    Cand2 *local = malloc(nthreads * K * sizeof(Cand2));
    for (int i = 0; i < nthreads*K; i++) local[i] = (Cand2){-1,-1,-1,-1,-1e30};

    //#pragma omp parallel
    {
        int tid = 0;
        Cand2 *myTop = &local[tid * K];
        u8 *dc = malloc(size);                        /* per-thread data copy */
        int (*pf)[17][256] = malloc(16*sizeof(*pf));  /* freq tables after step1 */
        int tf[256];                                  /* totalFreq after step1 */
        int (*bf)[256][256] = malloc(14*sizeof(*bf)); /* bigram tables for delta */

        //#pragma omp for schedule(dynamic)
        for (int instr1 = 0; instr1 < 32; instr1++) {
            if (instr1 == 5) continue;
            for (int amp1 = 0; amp1 < 256; amp1++) {
                if (instrOverhead(instr1) < 13.0 && amp1 != 0) continue; /* no-amp: skip 255 dup evals */
                /* apply step1 to scratch copy, rebuild ALL freq tables exactly */
                memcpy(dc, data, size);
                applyInstruction(dc, size, instr1, amp1);
                memset(pf, 0, 16*sizeof(*pf));
                memset(tf, 0, 256*sizeof(int));
                memset(bf, 0, 14*sizeof(*bf));
                int cfreq[256] = {0};
                {
                    int ph[16] = {0};
                    u8 w0=0,w1=0,w2=0,w3=0,w4=0,w5=0,w6=0,w7=0;
                    for (int i = 0; i < size; i++) {
                        u8 b = dc[i];
                        tf[b]++;
                        for (int s = 0; s < 16; s++) {
                            pf[s][ph[s]][b]++;
                            if (++ph[s] == s+2) ph[s] = 0;
                        }
                        if (i>=1) bf[0][w0][b]++;
                        if (i>=2) bf[1+(i&1)][w1][b]++;
                        if (i>=3) bf[3+(i%3)][w2][b]++;
                        if (i>=4) bf[6+(i&3)][w3][b]++;
                        if (i>=5) bf[11][w4][b]++;
                    if (i>=6) bf[12][w5][b]++;
                    if (i>=7) bf[13][w6][b]++;
                    if (i>=8) { bf[10][w7][b]++; cfreq[b^w3^w7]++; }
                        w7=w6; w6=w5; w5=w4; w4=w3; w3=w2; w2=w1; w1=w0; w0=b;
                    }
                }

                /* pre-evaluate delta for step2: strides 1,2,3,4,8 + compound */
                double de[2][9];
                { int nx[256]={0};
                  for (int p=0;p<256;p++) for (int c=0;c<256;c++) nx[p^c]+=bf[0][p][c];
                  de[1][1]=entropyFromFreq(nx,size-1); }
                { int na[256]={0},nx[256]={0};
                  for (int P=0;P<2;P++) for (int p=0;p<256;p++) for (int c=0;c<256;c++) {
                      int v=bf[1+P][p][c]; na[(c-p)&0xFF]+=v; nx[p^c]+=v; }
                  de[0][2]=entropyFromFreq(na,size-2); de[1][2]=entropyFromFreq(nx,size-2); }
                for (int ds=3; ds<=4; ds++) {
                    int bb=ds*(ds-1)/2, na[256]={0}, nx[256]={0};
                    for (int P=0;P<ds;P++) for (int p=0;p<256;p++) for (int c=0;c<256;c++) {
                        int v=bf[bb+P][p][c]; na[(c-p)&0xFF]+=v; nx[p^c]+=v; }
                    de[0][ds]=entropyFromFreq(na,size-ds); de[1][ds]=entropyFromFreq(nx,size-ds); }
                for (int ds=5; ds<=7; ds++) {
                    int nx[256]={0};
                    for (int p=0;p<256;p++) for (int c=0;c<256;c++) nx[p^c]+=bf[11+(ds-5)][p][c];
                    de[1][ds]=entropyFromFreq(nx,size-ds); }
                { int nx[256]={0};
                  for (int p=0;p<256;p++) for (int c=0;c<256;c++) nx[p^c]+=bf[10][p][c];
                  de[1][8]=entropyFromFreq(nx,size-8); }
                double de_compound = entropyFromFreq(cfreq, size-8);

                for (int instr2 = 0; instr2 < 32; instr2++) {
                    if (instr2 == 5) continue;
                    for (int amp2 = 0; amp2 < 256; amp2++) {
                        IMap m2; if (!imap(instr2, amp2, &m2)) continue;
                        if (m2.ptype == 6) {
                            double e2 = de_compound;
                            double net = (baseE - e2) * size - instrOverhead(instr1) - instrOverhead(instr2);
                            if (net > myTop[K-1].net) {
                                myTop[K-1] = (Cand2){instr1, amp1, instr2, amp2, net};
                                for (int k = K-2; k >= 0 && myTop[k+1].net > myTop[k].net; k--) {
                                    Cand2 t = myTop[k]; myTop[k] = myTop[k+1]; myTop[k+1] = t;
                                }
                            }
                            continue;
                        }
                        if (m2.ptype == 4 || m2.ptype == 5) {
                            double e2 = de[m2.ptype==4 ? 0 : 1][m2.pval];
                            double net = (baseE - e2) * size - instrOverhead(instr1) - instrOverhead(instr2);
                            if (net > myTop[K-1].net) {
                                myTop[K-1] = (Cand2){instr1, amp1, instr2, amp2, net};
                                for (int k = K-2; k >= 0 && myTop[k+1].net > myTop[k].net; k--) {
                                    Cand2 t = myTop[k]; myTop[k] = myTop[k+1]; myTop[k+1] = t;
                                }
                            }
                            continue;
                        }
                        const int *sp2 = pf[m2.stride-2][m2.phase % m2.stride];

                        int nf2[256];
                        memcpy(nf2, tf, 256*sizeof(int));
                        for (int v = 0; v < 256; v++) nf2[v] -= sp2[v];
                        switch (m2.ptype) {
                        case 0: for (int v=0;v<256;v++) nf2[v^m2.pval]                  += sp2[v]; break;
                        case 1: for (int v=0;v<256;v++) nf2[(v+m2.pval)&0xFF]           += sp2[v]; break;
                        case 2: for (int v=0;v<256;v++) nf2[(v&0xF0)|((v+m2.pval)&0xF)] += sp2[v]; break;
                        }

                        double e2  = entropyFromFreq(nf2, size);
                        double net = (baseE - e2) * size - instrOverhead(instr1) - instrOverhead(instr2);
                        if (net > myTop[K-1].net) {
                            myTop[K-1] = (Cand2){instr1, amp1, instr2, amp2, net};
                            for (int k = K-2; k >= 0 && myTop[k+1].net > myTop[k].net; k--) {
                                Cand2 t = myTop[k]; myTop[k] = myTop[k+1]; myTop[k+1] = t;
                            }
                        }
                    }
                }
                /* extended high-precision step2: strides 4-7, all phases, XOR and ADD */
                for (int strd2=4; strd2<=7; strd2++) {
                    for (int ph2=0; ph2<strd2; ph2++) {
                        const int *sp2 = pf[strd2-2][ph2];
                        for (int op2=0; op2<=1; op2++) {
                            if (strd2==4 && op2==1 && ph2<2) continue;
                            int instr2x = encodeExt(strd2, ph2, op2);
                            for (int amp2=1; amp2<256; amp2++) {
                                int nf2[256];
                                memcpy(nf2, tf, 256*sizeof(int));
                                for (int v=0;v<256;v++) nf2[v] -= sp2[v];
                                if (op2==0) for(int v=0;v<256;v++) nf2[v^amp2]        += sp2[v];
                                else        for(int v=0;v<256;v++) nf2[(v+amp2)&0xFF] += sp2[v];
                                double e2 = entropyFromFreq(nf2, size);
                                double net = (baseE-e2)*size - instrOverhead(instr1) - 15.0;
                                if (net > myTop[K-1].net) {
                                    myTop[K-1] = (Cand2){instr1, amp1, instr2x, amp2, net};
                                    for (int k=K-2; k>=0 && myTop[k+1].net>myTop[k].net; k--) {
                                        Cand2 t=myTop[k]; myTop[k]=myTop[k+1]; myTop[k+1]=t;
                                    }
                                }
                            }
                        }
                    }
                }
                /* xor-delta step2: strides 5-7 */
                for (int ds=5; ds<=7; ds++) {
                    double e2x = de[1][ds];
                    double net = (baseE-e2x)*size - instrOverhead(instr1) - 5.0;
                    if (net > myTop[K-1].net) {
                        myTop[K-1] = (Cand2){instr1, amp1, 220+(ds-5), 0, net};
                        for (int k=K-2; k>=0 && myTop[k+1].net>myTop[k].net; k--) {
                            Cand2 t=myTop[k]; myTop[k]=myTop[k+1]; myTop[k+1]=t;
                        }
                    }
                }
            }
        }

        free(dc);
        free(pf);
        free(bf);

        //#pragma omp critical
        for (int k = 0; k < K; k++) {
            if (myTop[k].net > topK[K-1].net) {
                topK[K-1] = myTop[k];
                for (int j = K-2; j >= 0 && topK[j+1].net > topK[j].net; j--) {
                    Cand2 t = topK[j]; topK[j] = topK[j+1]; topK[j+1] = t;
                }
            }
        }
    }
    free(local);
}

static void printDiagnostic(const u8 *data, int size) {
    /* --- per-bucket entropy: which stride/phase still has exploitable structure --- */
    int (*pf)[17][256] = calloc(16, sizeof(*pf));
    { int ph[16] = {0};
      for (int i = 0; i < size; i++) {
          u8 b = data[i];
          for (int s = 0; s < 16; s++) {
              pf[s][ph[s]][b]++;
              if (++ph[s] == s+2) ph[s] = 0;
          }
      }
    }
    printf("\n--- Diagnostic (residual structure in output) ---\n");
    printf("Per-bucket entropy deviations > 0.0001 bits:\n");
    int found = 0;
    for (int s = 0; s < 16; s++) {
        for (int p = 0; p < s+2; p++) {
            int cnt = 0;
            for (int v = 0; v < 256; v++) cnt += pf[s][p][v];
            if (!cnt) continue;
            double e = 0;
            for (int v = 0; v < 256; v++) if (pf[s][p][v]) {
                double pr = (double)pf[s][p][v] / cnt;
                e -= pr * log2(pr);
            }
            double dev = 8.0 - e;
            if (dev > 0.0001) {
                printf("  stride=%2d phase=%d: H=%.6f dev=%.6f (~%.1f bits potential)\n",
                       s+2, p, e, dev, dev * cnt);
                found++;
            }
        }
    }
    if (!found) printf("  (all buckets within 0.0001 bits of 8.0 — no exploitable marginal structure)\n");
    free(pf);

    /* --- bigram conditional entropy: delta potential at each stride --- */
    printf("Bigram delta potential (H(marginal) - H(y|x_prev)):\n");
    int (*bg)[256] = malloc(256 * sizeof(*bg)); /* bg[prev][cur] */
    for (int stride = 1; stride <= 5; stride++) {
        memset(bg, 0, 256 * sizeof(*bg));
        for (int i = stride; i < size; i++) bg[data[i-stride]][data[i]]++;

        int tc = size - stride;
        /* marginal H(cur) */
        int marg[256] = {0};
        for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) marg[b] += bg[a][b];
        double hcur = 0;
        for (int v = 0; v < 256; v++) if (marg[v]) { double p=(double)marg[v]/tc; hcur-=p*log2(p); }

        /* H(cur|prev) = H(cur,prev) - H(prev) */
        int prev_c[256] = {0};
        for (int a = 0; a < 256; a++) for (int b = 0; b < 256; b++) prev_c[a] += bg[a][b];
        double hjoint = 0, hprev = 0;
        for (int a = 0; a < 256; a++) {
            if (prev_c[a]) { double p=(double)prev_c[a]/tc; hprev -= p*log2(p); }
            for (int b = 0; b < 256; b++) if (bg[a][b]) {
                double p=(double)bg[a][b]/tc; hjoint -= p*log2(p);
            }
        }
        double hcond = hjoint - hprev;
        double gain  = hcur - hcond;
        printf("  stride=%d: H(y)=%.4f  H(y|x)=%.4f  delta_gain=%.6f bits/byte (%5.1f bits/MB)\n",
               stride, hcur, hcond, gain, gain * tc);
    }
    free(bg);
}

static uint32_t seed_state = 0x12345678;

static uint8_t seeded_random_byte(void) {
    /* Simple LCG: x = (a*x + c) mod 2^32, output high byte for decent distribution */
    seed_state = seed_state * 1103515245u + 12345u;
    return (uint8_t)(seed_state >> 24);
}

static int get_num_cores(void) { 
    return omp_get_num_procs();
}

void main() {
    /* Initialize h_table[x] = x*log2(x) for sweep search */
    h_table[0] = 0.0;
    for (int x = 1; x <= 8192; x++) h_table[x] = (double)x * log2((double)x);

    int NUM_CORES = get_num_cores();
    omp_set_max_active_levels(2);  /* allow one level of nested parallelism for lookahead */
    const int NUM_BLOCKS  = 1;
    const int BLOCK_SIZE  = 4096;
    const uint32_t SEED   = 42;
    const int NUM_THREADS = (NUM_BLOCKS < NUM_CORES) ? NUM_BLOCKS : NUM_CORES;

    double *netPerBlock = malloc(NUM_BLOCKS * sizeof(double));
    memset(netPerBlock, 0, NUM_BLOCKS * sizeof(double));
    int    instrHits[500]   = {0};
    double instrMinReduction[500];
    double instrMaxReduction[500];
    double instrSumReduction[500];

    for (int i = 0; i < 500; i++) {
        instrMinReduction[i] = 1e30;
        instrMaxReduction[i] = -1e30;
        instrSumReduction[i] = 0.0;
    }

    seed_state = SEED;
    printf("Running %d blocks on %d thread(s)... [seed=%u]\n",
           NUM_BLOCKS, NUM_THREADS, SEED);

    /* Process blocks in parallel — each block independently */
    InstrSeq *firstBlockSeq = InstrSeq_Create(1000);
    u8 *firstBlockData = NULL;

    clock_t start = clock();
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int b = 0; b < NUM_BLOCKS; b++) {
        u8 *data = malloc(BLOCK_SIZE);
        for (int i = 0; i < BLOCK_SIZE; i++) data[i] = seeded_random_byte();

        double totalNet = 0.0;
        int localHits[500] = {0};
        double localMinReduction[500];
        double localMaxReduction[500];
        double localSumReduction[500];

        for (int i = 0; i < 500; i++) {
            localMinReduction[i] = 1e30;
            localMaxReduction[i] = -1e30;
            localSumReduction[i] = 0.0;
        }

        InstrSeq *seq = (b == 0) ? firstBlockSeq : NULL;

        /* Pure greedy: just find best action per pass and apply it */
        totalNet = RunGreedyST(data, BLOCK_SIZE, localHits, seq, (b == 0),
                              localMinReduction, localMaxReduction, localSumReduction);

        if (b == 0) {
            firstBlockData = malloc(BLOCK_SIZE);
            memcpy(firstBlockData, data, BLOCK_SIZE);
        }

        free(data);
        netPerBlock[b] = totalNet;

        #pragma omp critical
        {
            for (int i = 0; i < 500; i++) {
                instrHits[i] += localHits[i];
                if (localMinReduction[i] < 1e30) {
                    if (localMinReduction[i] < instrMinReduction[i])
                        instrMinReduction[i] = localMinReduction[i];
                    if (localMaxReduction[i] > instrMaxReduction[i])
                        instrMaxReduction[i] = localMaxReduction[i];
                    instrSumReduction[i] += localSumReduction[i];
                }
            }
        }
    }
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;

    double minNet = netPerBlock[0], maxNet = netPerBlock[0], sumNet = 0.0;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        if (netPerBlock[b] < minNet) minNet = netPerBlock[b];
        if (netPerBlock[b] > maxNet) maxNet = netPerBlock[b];
        sumNet += netPerBlock[b];
    }

    int totalPasses = 0;
    for (int i = 0; i < 500; i++) totalPasses += instrHits[i];

    printf("\n=== %d blocks, %d MB each ===\n", NUM_BLOCKS, BLOCK_SIZE >> 20);
    printf("Elapsed time: %.2f seconds\n", elapsed);
    printf("Net savings:  min=%.1f  max=%.1f  avg=%.1f bits\n",
           minNet, maxNet, sumNet / NUM_BLOCKS);
    printf("\nInstruction usage (%d total passes):\n", totalPasses);

    free(netPerBlock);
    for (int i = 0; i < 32; i++) {
        if (instrHits[i] > 0) {
            double avg = instrSumReduction[i] / instrHits[i];
            printf("  instr %2d: %4d uses  (%5.1f%%)  |  min=%.1f  max=%.1f  avg=%.1f bits\n",
                   i, instrHits[i], 100.0 * instrHits[i] / totalPasses,
                   instrMinReduction[i], instrMaxReduction[i], avg);
        } else
            printf("  instr %2d:    0 uses  -- NEVER USED\n", i);
    }
    for (int i = 100; i < 220; i++) {
        if (instrHits[i] > 0) {
            int s,ph,op; decodeExt(i,&s,&ph,&op);
            double avg = instrSumReduction[i] / instrHits[i];
            printf("  %s s%dp%d: %4d uses  (%5.1f%%)  |  min=%.1f  max=%.1f  avg=%.1f bits\n",
                   op?"ADD":"XOR", s, ph,
                   instrHits[i], 100.0*instrHits[i]/totalPasses,
                   instrMinReduction[i], instrMaxReduction[i], avg);
        }
    }
    for (int ds=5; ds<=7; ds++) {
        int i = 220+(ds-5);
        if (instrHits[i] > 0) {
            double avg = instrSumReduction[i] / instrHits[i];
            printf("  xd  s%d: %4d uses  (%5.1f%%)  |  min=%.1f  max=%.1f  avg=%.1f bits\n",
                   ds, instrHits[i], 100.0*instrHits[i]/totalPasses,
                   instrMinReduction[i], instrMaxReduction[i], avg);
        } else
            printf("  xd  s%d:    0 uses  -- NEVER USED\n", ds);
    }
    /* large extended strides 8-12 (dynamic amp range) */
    for (int i=300; i<500; i++) {
        if (instrHits[i] > 0) {
            int s,ph,op; decodeExt2(i,&s,&ph,&op);
            double avg = instrSumReduction[i] / instrHits[i];
            printf("  %s s%2dp%d: %4d uses  (%5.1f%%)  |  min=%.1f  max=%.1f  avg=%.1f bits\n",
                   op?"ADD":"XOR", s, ph,
                   instrHits[i], 100.0*instrHits[i]/totalPasses,
                   instrMinReduction[i], instrMaxReduction[i], avg);
        }
    }

    /* Write first block to binary files */
    if (firstBlockData && firstBlockSeq->count > 0) {
        FILE *f = fopen("data.bin", "wb");
        if (f) {
            fwrite(firstBlockData, 1, BLOCK_SIZE, f);
            fclose(f);
            double outputEntropy = getEntropy(firstBlockData, BLOCK_SIZE);
            printf("\nWrote %d bytes to data.bin  |  Output entropy: %.6f bits/byte\n",
                   BLOCK_SIZE, outputEntropy);
        }

        f = fopen("instructions.bin", "wb");
        if (f) {
            u8 instrBytes[1000 * 2];
            int totalInstrBytes = 0;
            for (int i = 0; i < firstBlockSeq->count; i++) {
                instrBytes[totalInstrBytes++] = firstBlockSeq->instrs[i];
                instrBytes[totalInstrBytes++] = firstBlockSeq->amps[i];
                fputc(firstBlockSeq->instrs[i], f);
                fputc(firstBlockSeq->amps[i], f);
            }
            fclose(f);
            double instrEntropy = getEntropy(instrBytes, totalInstrBytes);
            printf("Wrote %d instruction pairs (%d bytes) to instructions.bin  |  Entropy: %.6f bits/byte\n",
                   firstBlockSeq->count, firstBlockSeq->count * 2, instrEntropy);
        }
    }

    if (firstBlockData) free(firstBlockData);
    InstrSeq_Free(firstBlockSeq);
}
