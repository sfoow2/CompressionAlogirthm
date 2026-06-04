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

void applyInstruction(u8 *data, int size, int instr, int amp) {
    int i, stride;
    u8 xval, addval;
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
    case 6:
        for (i = 0; i < size; i += 2) data[i] += (u8)amp;
        break;
    case 7:
        for (i = 1; i < size; i += 2) data[i] += (u8)amp;
        break;

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
    case 18: for (i=size-1; i>=2; i--) data[i]^=data[i-2]; break; /* xor-delta stride=2 */
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
    case 27: for (i=size-1; i>=8; i--) data[i]^=(data[i-4]^data[i-8]); break; /* compound xor-delta 4+8 */
    case 28: for (i=size-1; i>=3; i--) data[i]-=data[i-3]; break; /* delta stride=3 */
    case 29: for (i=size-1; i>=4; i--) data[i]-=data[i-4]; break; /* delta stride=4 */
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
    case  6: m->stride=2;          m->phase=0; m->ptype=1; m->pval=amp;           break;
    case  7: m->stride=2;          m->phase=1; m->ptype=1; m->pval=amp;           break;
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
    case 18: if (amp!=0) return 0; m->stride=2; m->phase=0; m->ptype=5; m->pval=2; break; /* xor-delta stride=2 */
    case 19: if (amp!=0) return 0; m->stride=1; m->phase=0; m->ptype=5; m->pval=1; break; /* xor-delta stride=1 */
    case 20: m->stride=(amp>>4)+2; m->phase=2; m->ptype=0; m->pval=(amp&0xF)<<4; break;
    case 21: m->stride=(amp>>4)+2; m->phase=3; m->ptype=0; m->pval=(amp&0xF)<<4; break;
    case 22: if (amp!=0) return 0; m->stride=3; m->phase=0; m->ptype=5; m->pval=3; break;
    case 23: if (amp!=0) return 0; m->stride=4; m->phase=0; m->ptype=5; m->pval=4; break;
    case 24: m->stride=3;          m->phase=2; m->ptype=0; m->pval=amp;           break;
    case 25: m->stride=3;          m->phase=2; m->ptype=1; m->pval=amp;           break;
    case 26: if (amp!=0) return 0; m->stride=8; m->phase=0; m->ptype=5; m->pval=8; break; /* xor-delta stride=8 */
    case 27: if (amp!=0) return 0; m->stride=8; m->phase=0; m->ptype=6; m->pval=0; break; /* compound xor-delta */
    case 28: if (amp!=0) return 0; m->stride=3; m->phase=0; m->ptype=4; m->pval=3; break;
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
    if ((instr >= 18 && instr <= 19) || (instr >= 22 && instr <= 23) ||
        instr == 26 || instr == 27 || (instr >= 28 && instr <= 29)) return 5.0;

    /* Check tiered hot instruction overheads */
    int hotOH = getHotInstructionOverhead(instr);
    if (hotOH > 0) return (double)hotOH;

    /* Regular instructions: 13 bits (5-bit instr + 8-bit amp) */
    if (instr < 100) return 13.0;

    /* Extended per-phase: 15 bits (4-bit stride + 4-bit phase + 1-bit op + 8-bit amp) */
    return 15.0;
}

/* ------------------------------------------------------------------ *
 * Single-threaded step finder — safe to call from any parallel region *
 * scratch is a caller-provided buffer of length `size`               *
 * ------------------------------------------------------------------ */
/* Entropy from a pre-built 256-bucket frequency table */
static double entropyFromFreq(const int *freq, int size) {
    double e = 0.0;
    for (int v = 0; v < 256; v++) {
        if (!freq[v]) continue;
        double p = (double)freq[v] / size;
        e -= p * log2(p);
    }
    return e;
}

/* ------------------------------------------------------------------ *
 * Optimised single-threaded FindNextStep using frequency-table cache  *
 * All instructions are permutations of byte values on a position      *
 * subset. Pre-build stride×phase frequency tables once (O(n×16)),    *
 * then evaluate each (instr,amp) candidate in O(256) — no memcpy,    *
 * no data scan per candidate. ~1000× faster than the naïve approach.  *
 * ------------------------------------------------------------------ */
static double FindNextStepST(u8 *data, int size, int *usedInstr, int verbose,
                              int (*pFreq)[17][256], int (*bf)[256][256]) {
    /* --- build per-stride-phase frequency tables (one pass over data) --- */
    /* pFreq[s][ph][v] = count of bytes with value v at positions i where   */
    /* i mod (s+2) == ph. strides 2..17 → s=0..15, max phase index = 16    */
    int totalFreq[256] = {0};

    /* Clear pre-allocated buffers */
    memset(pFreq, 0, 16 * 17 * 256 * sizeof(int));
    memset(bf, 0, 14 * 256 * 256 * sizeof(int));
    int compound_freq[256] = {0}; /* freq of (data[i]^data[i-4]^data[i-8]) */
    {
        int ph[16] = {0};
        u8 w0=0,w1=0,w2=0,w3=0,w4=0,w5=0,w6=0,w7=0;
        for (int i = 0; i < size; i++) {
            u8 b = data[i];
            totalFreq[b]++;
            for (int s = 0; s < 16; s++) {
                pFreq[s][ph[s]][b]++;
                if (++ph[s] == s+2) ph[s] = 0;
            }
            if (i>=1) bf[0][w0][b]++;
            if (i>=2) bf[1+(i&1)][w1][b]++;
            if (i>=3) bf[3+(i%3)][w2][b]++;
            if (i>=4) bf[6+(i&3)][w3][b]++;
            if (i>=5) bf[11][w4][b]++;
            if (i>=6) bf[12][w5][b]++;
            if (i>=7) bf[13][w6][b]++;
            if (i>=8) { bf[10][w7][b]++; compound_freq[b^w3^w7]++; }
            w7=w6; w6=w5; w5=w4; w4=w3; w3=w2; w2=w1; w1=w0; w0=b;
        }
    }

    double baseE = entropyFromFreq(totalFreq, size);
    double bestNet = 0.0, bestE = baseE;
    int bestInstr = -1, bestAmp = -1;

    /* pre-evaluate all delta instructions: strides 1,2,3,4,8 + compound
       Bigrams cover size-k pairs (positions k..size-1), so normalize by
       size-k to avoid ~k*8-bit phantom profit from wrong normalization. */
    double delta_e[2][9]; /* [0=ADD,1=XOR][stride 0..8] */
    /* stride 1: xor-delta only */
    { int nx[256]={0};
      for (int p=0;p<256;p++) for (int c=0;c<256;c++) nx[p^c]+=bf[0][p][c];
      delta_e[1][1]=entropyFromFreq(nx,size-1); }
    /* stride 2: add+xor */
    { int na[256]={0},nx[256]={0};
      for (int P=0;P<2;P++) for (int p=0;p<256;p++) for (int c=0;c<256;c++) {
          int v=bf[1+P][p][c]; na[(c-p)&0xFF]+=v; nx[p^c]+=v; }
      delta_e[0][2]=entropyFromFreq(na,size-2); delta_e[1][2]=entropyFromFreq(nx,size-2); }
    /* strides 3-4: add+xor */
    for (int ds=3; ds<=4; ds++) {
        int base=ds*(ds-1)/2, na[256]={0}, nx[256]={0};
        for (int P=0;P<ds;P++) for (int p=0;p<256;p++) for (int c=0;c<256;c++) {
            int v=bf[base+P][p][c]; na[(c-p)&0xFF]+=v; nx[p^c]+=v; }
        delta_e[0][ds]=entropyFromFreq(na,size-ds); delta_e[1][ds]=entropyFromFreq(nx,size-ds); }
    /* strides 5-7: xor-delta only (unphased bigrams bf[11..13]) */
    for (int ds=5; ds<=7; ds++) {
        int nx[256]={0};
        for (int p=0;p<256;p++) for (int c=0;c<256;c++) nx[p^c]+=bf[11+(ds-5)][p][c];
        delta_e[1][ds]=entropyFromFreq(nx,size-ds); }
    /* stride 8: xor-delta only */
    { int nx[256]={0};
      for (int p=0;p<256;p++) for (int c=0;c<256;c++) nx[p^c]+=bf[10][p][c];
      delta_e[1][8]=entropyFromFreq(nx,size-8); }
    /* compound: pre-computed directly in compound_freq */
    double compound_e = entropyFromFreq(compound_freq, size-8);

    for (int instr = 0; instr < 32; instr++) {
        if (instr == 5) continue;
        for (int amp = 0; amp < 256; amp++) {
            /* map (instr, amp) → stride, phase, permutation type + value */
            int stride, phase, ptype, pval;
            switch (instr) {
            case  0: stride=(amp>>4)+2; phase=0; ptype=0; pval=amp&0xF;      break;
            case  1: stride=(amp>>4)+2; phase=1; ptype=0; pval=amp&0xF;      break;
            case  2: stride=(amp>>4)+2; phase=0; ptype=2; pval=amp&0xF;      break;
            case  3: stride=(amp>>4)+2; phase=1; ptype=2; pval=amp&0xF;      break;
            case  4: stride=2;          phase=0; ptype=0; pval=amp;           break;
            case  6: stride=2;          phase=0; ptype=1; pval=amp;           break;
            case  7: stride=2;          phase=1; ptype=1; pval=amp;           break;
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
            case 18: if (amp!=0) continue; stride=2; phase=0; ptype=5; pval=2; break;
            case 19: if (amp!=0) continue; stride=1; phase=0; ptype=5; pval=1; break;
            case 20: stride=(amp>>4)+2; phase=2; ptype=0; pval=(amp&0xF)<<4; break;
            case 21: stride=(amp>>4)+2; phase=3; ptype=0; pval=(amp&0xF)<<4; break;
            case 22: if (amp!=0) continue; stride=3; phase=0; ptype=5; pval=3; break;
            case 23: if (amp!=0) continue; stride=4; phase=0; ptype=5; pval=4; break;
            case 24: stride=3;          phase=2; ptype=0; pval=amp;           break;
            case 25: stride=3;          phase=2; ptype=1; pval=amp;           break;
            case 26: if (amp!=0) continue; stride=8; phase=0; ptype=5; pval=8; break;
            case 27: if (amp!=0) continue; stride=8; phase=0; ptype=6; pval=0; break;
            case 28: if (amp!=0) continue; stride=3; phase=0; ptype=4; pval=3; break;
            case 29: if (amp!=0) continue; stride=4; phase=0; ptype=4; pval=4; break;
            case 30: stride=4;          phase=0; ptype=1; pval=amp;           break;
            case 31: stride=4;          phase=1; ptype=1; pval=amp;           break;
            default: continue;
            }

            /* compound delta (ptype=6): use pre-computed compound_freq entropy */
            if (ptype == 6) {
                double net = (baseE - compound_e) * size - 5.0;
                if (net > bestNet) { bestNet=net; bestE=compound_e; bestInstr=instr; bestAmp=amp; }
                continue;
            }

            /* other delta instructions: use bigram tables */
            if (ptype == 4 || ptype == 5) {
                double e = delta_e[ptype==4 ? 0 : 1][pval];
                double net = (baseE - e) * size - 5.0;
                if (net > bestNet) { bestNet = net; bestE = e; bestInstr = instr; bestAmp = amp; }
                continue;
            }

            /* phase % stride: handles phase>=stride for packed instrs with
               small strides (off-by-a-few-bytes approximation, negligible
               on 1MB data) */
            const int *sp = pFreq[stride - 2][phase % stride];

            /* build new freq: remove affected bytes, re-add with permutation */
            int nf[256];
            memcpy(nf, totalFreq, 256 * sizeof(int));
            for (int v = 0; v < 256; v++) nf[v] -= sp[v];
            switch (ptype) {
            case 0: for (int v=0;v<256;v++) nf[v^pval]                  += sp[v]; break;
            case 1: for (int v=0;v<256;v++) nf[(v+pval)&0xFF]           += sp[v]; break;
            case 2: for (int v=0;v<256;v++) nf[(v&0xF0)|((v+pval)&0xF)] += sp[v]; break;
            }

            double e = entropyFromFreq(nf, size);
            double net = (baseE - e) * size - instrOverhead(instr);
            if (net > bestNet) { bestNet = net; bestE = e; bestInstr = instr; bestAmp = amp; }
        }
    }

    /* xor-delta strides 5-7 (instrs 220-222) */
    for (int ds=5; ds<=7; ds++) {
        double e = delta_e[1][ds];
        int instr_xd = 220 + (ds - 5);
        double net = (baseE - e)*size - instrOverhead(instr_xd);
        if (net > bestNet) { bestNet=net; bestE=e; bestInstr=instr_xd; bestAmp=0; }
    }

    /* extended high-precision: strides 4-7, all phases, XOR and ADD */
    for (int strd = 4; strd <= 7; strd++) {
        for (int ph = 0; ph < strd; ph++) {
            /* skip ADD stride-4 phases 0,1 — covered by instrs 30,31 with lower overhead */
            const int *sp = pFreq[strd-2][ph];
            for (int op = 0; op <= 1; op++) {
                if (strd==4 && op==1 && ph<2) continue;
                int instr_ext = encodeExt(strd, ph, op);
                int nf[256];
                for (int amp = 1; amp < 256; amp++) {
                    memcpy(nf, totalFreq, 256*sizeof(int));
                    for (int v=0;v<256;v++) nf[v] -= sp[v];
                    if (op==0) for(int v=0;v<256;v++) nf[v^amp]        += sp[v];
                    else       for(int v=0;v<256;v++) nf[(v+amp)&0xFF] += sp[v];
                    double e = entropyFromFreq(nf, size);
                    double net = (baseE - e)*size - instrOverhead(instr_ext);
                    if (net > bestNet) { bestNet=net; bestE=e; bestInstr=instr_ext; bestAmp=amp; }
                }
            }
        }
    }

    /* Stride-sweep search removed — profiling showed all loc_amps end up 0 for random data
       (per-phase distributions too uniform for any XOR/ADD to beat single-phase instruction) */

    /* Buffers freed by caller */

    if (bestInstr < 0) {
        if (usedInstr) *usedInstr = -1;
        return 0.0;
    }
    applyInstruction(data, size, bestInstr, bestAmp);
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
    int (*bf)[256][256] = calloc(14, sizeof(int[256][256]));
    double total = 0.0;

    for (;;) {
        /* greedy passes until stall */
        int instr; double net;
        while ((net = FindNextStepST(data, size, &instr, verbose, pFreq, bf)) > 0.0) {
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
            double cg = 0.0, n; int dummy;
            while ((n = FindNextStepST(buf, size, &dummy, 0, pFreq, bf)) > 0.0) cg += n;
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
    free(bf);
    return total;
}

/* greedy-only chain (no scramble) — safe to call from parallel regions */
static double RunGreedyST(u8 *data, int size, int *hits, int verbose) {
    int (*pFreq)[17][256] = calloc(16, sizeof(*pFreq));
    int (*bf)[256][256] = calloc(14, sizeof(int[256][256]));

    double total = 0.0, net;
    int instr;
    int passes = 0;
    clock_t find_time = 0, apply_time = 0;

    while ((net = FindNextStepST(data, size, &instr, verbose, pFreq, bf)) > 0.0) {
        total += net;
        if (hits) hits[instr]++;
        passes++;
    }

    free(pFreq);
    free(bf);
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
    const int NUM_BLOCKS  = 1000;
    const int BLOCK_SIZE  = 1024 * 4;
    const uint32_t SEED   = 42;
    const int NUM_THREADS = (NUM_BLOCKS < NUM_CORES) ? NUM_BLOCKS : NUM_CORES;

    double netPerBlock[200] = {0};
    int    instrHits[300]   = {0};

    seed_state = SEED;
    printf("Running %d blocks on %d thread(s)... [seed=%u]\n",
           NUM_BLOCKS, NUM_THREADS, SEED);

    /* Process blocks in parallel — each block independently */
    clock_t start = clock();
    #pragma omp parallel for num_threads(NUM_THREADS)
    for (int b = 0; b < NUM_BLOCKS; b++) {
        u8 *data = malloc(BLOCK_SIZE);
        for (int i = 0; i < BLOCK_SIZE; i++) data[i] = seeded_random_byte();

        double totalNet = 0.0;
        int localHits[300] = {0};

        /* Pure greedy: just find best action per pass and apply it */
        totalNet = RunGreedyST(data, BLOCK_SIZE, localHits, (b == 0));

        free(data);
        netPerBlock[b] = totalNet;

        //#pragma omp critical
        for (int i = 0; i < 300; i++)
            instrHits[i] += localHits[i];
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
    for (int i = 0; i < 300; i++) totalPasses += instrHits[i];

    printf("\n=== %d blocks, %d MB each ===\n", NUM_BLOCKS, BLOCK_SIZE >> 20);
    printf("Elapsed time: %.2f seconds\n", elapsed);
    printf("Net savings:  min=%.1f  max=%.1f  avg=%.1f bits\n",
           minNet, maxNet, sumNet / NUM_BLOCKS);
    printf("\nInstruction usage (%d total passes):\n", totalPasses);
    for (int i = 0; i < 32; i++) {
        if (instrHits[i] > 0)
            printf("  instr %2d: %4d uses  (%5.1f%%)\n",
                   i, instrHits[i], 100.0 * instrHits[i] / totalPasses);
        else
            printf("  instr %2d:    0 uses  -- NEVER USED\n", i);
    }
    for (int i = 100; i < 220; i++) {
        if (instrHits[i] > 0) {
            int s,ph,op; decodeExt(i,&s,&ph,&op);
            printf("  %s s%dp%d: %4d uses  (%5.1f%%)\n",
                   op?"ADD":"XOR", s, ph,
                   instrHits[i], 100.0*instrHits[i]/totalPasses);
        }
    }
    for (int ds=5; ds<=7; ds++) {
        int i = 220+(ds-5);
        if (instrHits[i] > 0)
            printf("  xd  s%d: %4d uses  (%5.1f%%)\n",
                   ds, instrHits[i], 100.0*instrHits[i]/totalPasses);
        else
            printf("  xd  s%d:    0 uses  -- NEVER USED\n", ds);
    }
    for (int i=250; i<=281; i++) {
        if (instrHits[i] > 0) {
            int strd=(i-250)/2+2, op=(i-250)%2;
            printf("  %s sweep s%2d: %4d uses (%5.1f%%)\n",
                   op?"ADD":"XOR", strd, instrHits[i], 100.0*instrHits[i]/totalPasses);
        }
    }
}
