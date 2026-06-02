#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdint.h>
#include <string.h>
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

overhead per step = 13 bits (5-bit instr ID + 8-bit amp)

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
18: ADD (amp&0x0F) mod-16 to LOW NIBBLE ONLY, packed stride, phase 2   [high nibble frozen, reverse: low4 = (16-v)&0xF]
19: ADD (amp&0x0F) mod-16 to LOW NIBBLE ONLY, packed stride, phase 3
20: XOR (amp&0x0F)<<4,   packed stride, phase 2
21: XOR (amp&0x0F)<<4,   packed stride, phase 3
22: ADD (amp&0x0F)<<4,   packed stride, phase 2
23: ADD (amp&0x0F)<<4,   packed stride, phase 3

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
26: delta stride=1  (all bytes)          [reverse: left-to-right add same stride]
27: delta stride=2  (even/odd channels independently)
28: delta stride=3
29: delta stride=4  (RGBA / 4-byte interleaved)

--- HIGH PRECISION STRIDE 4, phases 0-1 (full 8-bit amp) ---
30: ADD amp, stride 4, phase 0   reverse: amp = (256-amp)&0xFF
31: ADD amp, stride 4, phase 1



*/


void applyInstruction(u8 *data, int size, int instr, int amp) {
    int i, stride;
    u8 xval, addval;
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
    case 18:
        stride = (amp >> 4) + 2;
        addval = (u8)(amp & 0x0F);
        for (i = 2; i < size; i += stride)
            data[i] = (data[i] & 0xF0) | ((data[i] + addval) & 0x0F);
        break;
    case 19:
        stride = (amp >> 4) + 2;
        addval = (u8)(amp & 0x0F);
        for (i = 3; i < size; i += stride)
            data[i] = (data[i] & 0xF0) | ((data[i] + addval) & 0x0F);
        break;

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
    case 22:
        stride = (amp >> 4) + 2;
        addval = (u8)((amp & 0x0F) << 4);
        for (i = 2; i < size; i += stride) data[i] += addval;
        break;
    case 23:
        stride = (amp >> 4) + 2;
        addval = (u8)((amp & 0x0F) << 4);
        for (i = 3; i < size; i += stride) data[i] += addval;
        break;

    /* --- high precision stride 3 phase 2; stride 4 all phases --- */
    case 24:
        for (i = 2; i < size; i += 3) data[i] ^= (u8)amp;
        break;
    case 25:
        for (i = 2; i < size; i += 3) data[i] += (u8)amp;
        break;
    case 26: for (i=size-1; i>=1; i--) data[i]-=data[i-1]; break; /* delta stride=1 */
    case 27: for (i=size-1; i>=2; i--) data[i]-=data[i-2]; break; /* delta stride=2 */
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

/* (instr, amp) → freq-table parameters; returns 0 for invalid/skip instrs */
typedef struct { int stride, phase, ptype, pval; } IMap;
static int imap(int instr, int amp, IMap *m) {
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
    case 18: m->stride=(amp>>4)+2; m->phase=2; m->ptype=2; m->pval=amp&0xF;      break;
    case 19: m->stride=(amp>>4)+2; m->phase=3; m->ptype=2; m->pval=amp&0xF;      break;
    case 20: m->stride=(amp>>4)+2; m->phase=2; m->ptype=0; m->pval=(amp&0xF)<<4; break;
    case 21: m->stride=(amp>>4)+2; m->phase=3; m->ptype=0; m->pval=(amp&0xF)<<4; break;
    case 22: m->stride=(amp>>4)+2; m->phase=2; m->ptype=1; m->pval=(amp&0xF)<<4; break;
    case 23: m->stride=(amp>>4)+2; m->phase=3; m->ptype=1; m->pval=(amp&0xF)<<4; break;
    case 24: m->stride=3;          m->phase=2; m->ptype=0; m->pval=amp;           break;
    case 25: m->stride=3;          m->phase=2; m->ptype=1; m->pval=amp;           break;
    case 26: if (amp!=0) return 0; m->stride=1; m->phase=0; m->ptype=4; m->pval=1; break;
    case 27: if (amp!=0) return 0; m->stride=2; m->phase=0; m->ptype=4; m->pval=2; break;
    case 28: if (amp!=0) return 0; m->stride=3; m->phase=0; m->ptype=4; m->pval=3; break;
    case 29: if (amp!=0) return 0; m->stride=4; m->phase=0; m->ptype=4; m->pval=4; break;
    case 30: m->stride=4;          m->phase=0; m->ptype=1; m->pval=amp;           break;
    case 31: m->stride=4;          m->phase=1; m->ptype=1; m->pval=amp;           break;
    default: return 0;
    }
    return 1;
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
static double FindNextStepST(u8 *data, int size, int *usedInstr, int verbose) {
    /* --- build per-stride-phase frequency tables (one pass over data) --- */
    /* pFreq[s][ph][v] = count of bytes with value v at positions i where   */
    /* i mod (s+2) == ph. strides 2..17 → s=0..15, max phase index = 16    */
    int (*pFreq)[17][256] = calloc(16, sizeof(*pFreq));
    int totalFreq[256] = {0};

    {
        int ph[16] = {0};
        for (int i = 0; i < size; i++) {
            u8 b = data[i];
            totalFreq[b]++;
            for (int s = 0; s < 16; s++) {
                pFreq[s][ph[s]][b]++;
                if (++ph[s] == s + 2) ph[s] = 0;
            }
        }
    }

    /* bigram tables for delta: bf[stride*(stride-1)/2 + phase][prev][cur]
       strides 1-4 → 1+2+3+4=10 phase-buckets total */
    int (*bf)[256][256] = calloc(10, sizeof(int[256][256]));
    for (int s = 1; s <= 4; s++) {
        int base = s*(s-1)/2;
        for (int i = s; i < size; i++)
            bf[base + i%s][data[i-s]][data[i]]++;
    }

    double baseE = entropyFromFreq(totalFreq, size);
    double bestE = baseE;
    int bestInstr = -1, bestAmp = -1;

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
            case 18: stride=(amp>>4)+2; phase=2; ptype=2; pval=amp&0xF;      break;
            case 19: stride=(amp>>4)+2; phase=3; ptype=2; pval=amp&0xF;      break;
            case 20: stride=(amp>>4)+2; phase=2; ptype=0; pval=(amp&0xF)<<4; break;
            case 21: stride=(amp>>4)+2; phase=3; ptype=0; pval=(amp&0xF)<<4; break;
            case 22: stride=(amp>>4)+2; phase=2; ptype=1; pval=(amp&0xF)<<4; break;
            case 23: stride=(amp>>4)+2; phase=3; ptype=1; pval=(amp&0xF)<<4; break;
            case 24: stride=3;          phase=2; ptype=0; pval=amp;           break;
            case 25: stride=3;          phase=2; ptype=1; pval=amp;           break;
            case 26: if (amp!=0) continue; stride=1; phase=0; ptype=4; pval=1; break;
            case 27: if (amp!=0) continue; stride=2; phase=0; ptype=4; pval=2; break;
            case 28: if (amp!=0) continue; stride=3; phase=0; ptype=4; pval=3; break;
            case 29: if (amp!=0) continue; stride=4; phase=0; ptype=4; pval=4; break;
            case 30: stride=4;          phase=0; ptype=1; pval=amp;           break;
            case 31: stride=4;          phase=1; ptype=1; pval=amp;           break;
            default: continue;
            }

            /* delta instructions: use bigram tables, skip normal freq-table path */
            if (ptype == 4) {
                int ds = pval, base = ds*(ds-1)/2;
                int nf_d[256] = {0};
                for (int P = 0; P < ds; P++)
                    for (int prev=0; prev<256; prev++)
                        for (int cur=0; cur<256; cur++)
                            nf_d[(cur-prev)&0xFF] += bf[base+P][prev][cur];
                double e = entropyFromFreq(nf_d, size);
                if (e < bestE) { bestE = e; bestInstr = instr; bestAmp = amp; }
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
            if (e < bestE) { bestE = e; bestInstr = instr; bestAmp = amp; }
        }
    }

    free(pFreq);
    free(bf);

    double saved = (baseE - bestE) * size, net = saved - 13.0;
    if (bestInstr < 0 || net <= 0.0) {
        if (usedInstr) *usedInstr = -1;
        return 0.0;
    }
    applyInstruction(data, size, bestInstr, bestAmp);
    if (verbose)
        printf("  instr=%2d amp=%3d | entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
               bestInstr, bestAmp, baseE, bestE, saved, net);
    if (usedInstr) *usedInstr = bestInstr;
    return net;
}

/* ------------------------------------------------------------------ *
 * Full greedy chain + full-chain scramble lookahead — single-threaded *
 * Covers improvements 1 (full-chain lookahead) and 3 (nested scrambles)*
 * ------------------------------------------------------------------ */
static double RunChainST(u8 *data, int size, int *hits, int verbose) {
    double total = 0.0;

    for (;;) {
        /* greedy passes until stall */
        int instr; double net;
        while ((net = FindNextStepST(data, size, &instr, verbose)) > 0.0) {
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
            while ((n = FindNextStepST(buf, size, &dummy, 0)) > 0.0) cg += n;
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

    return total;
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

    int nthreads = omp_get_max_threads();
    Cand2 *local = malloc(nthreads * K * sizeof(Cand2));
    for (int i = 0; i < nthreads*K; i++) local[i] = (Cand2){-1,-1,-1,-1,-1e30};

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Cand2 *myTop = &local[tid * K];
        u8 *dc = malloc(size);                        /* per-thread data copy */
        int (*pf)[17][256] = malloc(16*sizeof(*pf));  /* freq tables after step1 */
        int tf[256];                                  /* totalFreq after step1 */
        int (*bf)[256][256] = malloc(10*sizeof(*bf)); /* bigram tables for delta */

        #pragma omp for schedule(dynamic)
        for (int instr1 = 0; instr1 < 32; instr1++) {
            if (instr1 == 5) continue;
            for (int amp1 = 0; amp1 < 256; amp1++) {
                /* apply step1 to scratch copy, rebuild ALL freq tables exactly */
                memcpy(dc, data, size);
                applyInstruction(dc, size, instr1, amp1);
                memset(pf, 0, 16*sizeof(*pf));
                memset(tf, 0, 256*sizeof(int));
                {
                    int ph[16] = {0};
                    for (int i = 0; i < size; i++) {
                        u8 b = dc[i];
                        tf[b]++;
                        for (int s = 0; s < 16; s++) {
                            pf[s][ph[s]][b]++;
                            if (++ph[s] == s+2) ph[s] = 0;
                        }
                    }
                }
                memset(bf, 0, 10*sizeof(*bf));
                for (int s = 1; s <= 4; s++) {
                    int bbase = s*(s-1)/2;
                    for (int i = s; i < size; i++)
                        bf[bbase + i%s][dc[i-s]][dc[i]]++;
                }

                for (int instr2 = 0; instr2 < 32; instr2++) {
                    if (instr2 == 5) continue;
                    for (int amp2 = 0; amp2 < 256; amp2++) {
                        IMap m2; if (!imap(instr2, amp2, &m2)) continue;
                        if (m2.ptype == 4) {
                            int ds = m2.pval, bbase = ds*(ds-1)/2;
                            int nf_d[256] = {0};
                            for (int P = 0; P < ds; P++)
                                for (int prev=0;prev<256;prev++)
                                    for (int cur=0;cur<256;cur++)
                                        nf_d[(cur-prev)&0xFF] += bf[bbase+P][prev][cur];
                            double e2 = entropyFromFreq(nf_d, size);
                            double net = (baseE - e2) * size - 26.0;
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
                        double net = (baseE - e2) * size - 26.0;
                        if (net > myTop[K-1].net) {
                            myTop[K-1] = (Cand2){instr1, amp1, instr2, amp2, net};
                            for (int k = K-2; k >= 0 && myTop[k+1].net > myTop[k].net; k--) {
                                Cand2 t = myTop[k]; myTop[k] = myTop[k+1]; myTop[k+1] = t;
                            }
                        }
                    }
                }
            }
        }

        free(dc);
        free(pf);
        free(bf);

        #pragma omp critical
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

void main() {
    const int NUM_BLOCKS  = 1;
    const int BLOCK_SIZE  = 1024 * 1024;

    double netPerBlock[200] = {0};
    int    instrHits[32]    = {0};

    printf("Running %d blocks on %d thread(s)...\n",
           NUM_BLOCKS, omp_get_max_threads());

    for (int b = 0; b < NUM_BLOCKS; b++) {
        u8 *data = malloc(BLOCK_SIZE);
        uint32_t rng = (uint32_t)(b + 1);
        for (int i = 0; i < BLOCK_SIZE; i++) {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            data[i] = (u8)rng;
        }

        double totalNet = 0.0;
        int localHits[32] = {0};

        /* ---- improvement 4: beam search ----
           find top-K first steps, run full chain from each in parallel,
           keep the path with the most total net savings               */
        int BEAM_K = 64;
        Cand2   *topK     = malloc(BEAM_K * sizeof(Cand2));
        double  *beamNets = calloc(BEAM_K, sizeof(double));
        u8     **beamData = malloc(BEAM_K * sizeof(u8 *));
        int    **beamHits = malloc(BEAM_K * sizeof(int *));
        for (int k = 0; k < BEAM_K; k++) {
            beamData[k] = malloc(BLOCK_SIZE);
            memcpy(beamData[k], data, BLOCK_SIZE);
            beamHits[k] = calloc(32, sizeof(int));
        }

        FindTopK2(data, BLOCK_SIZE, BEAM_K, topK);

        /* parallel evaluation of K beams — each applies 2-step seed then full chain */
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < BEAM_K; k++) {
            if (topK[k].net <= 0.0) continue;
            applyInstruction(beamData[k], BLOCK_SIZE, topK[k].instr1, topK[k].amp1);
            beamHits[k][topK[k].instr1]++;
            applyInstruction(beamData[k], BLOCK_SIZE, topK[k].instr2, topK[k].amp2);
            beamHits[k][topK[k].instr2]++;
            beamNets[k] = topK[k].net + RunChainST(beamData[k], BLOCK_SIZE, beamHits[k], 0);
        }

        int best = 0;
        for (int k = 1; k < BEAM_K; k++)
            if (beamNets[k] > beamNets[best]) best = k;

        if (topK[best].net > 0.0) {
            u8 *replay = malloc(BLOCK_SIZE);
            memcpy(replay, data, BLOCK_SIZE);
            printf("  [beam %d/%d selected | paths tried: %d]\n", best+1, BEAM_K, BEAM_K);
            {
                double e0 = getEntropy(data, BLOCK_SIZE);
                applyInstruction(replay, BLOCK_SIZE, topK[best].instr1, topK[best].amp1);
                double e1 = getEntropy(replay, BLOCK_SIZE);
                printf("  instr=%2d amp=%3d | entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
                       topK[best].instr1, topK[best].amp1, e0, e1,
                       (e0-e1)*BLOCK_SIZE, (e0-e1)*BLOCK_SIZE - 13.0);
                applyInstruction(replay, BLOCK_SIZE, topK[best].instr2, topK[best].amp2);
                double e2 = getEntropy(replay, BLOCK_SIZE);
                printf("  instr=%2d amp=%3d | entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
                       topK[best].instr2, topK[best].amp2, e1, e2,
                       (e1-e2)*BLOCK_SIZE, (e1-e2)*BLOCK_SIZE - 13.0);
            }
            RunChainST(replay, BLOCK_SIZE, NULL, 1);
            if (b == 0) printDiagnostic(replay, BLOCK_SIZE);
            free(replay);

            memcpy(data, beamData[best], BLOCK_SIZE);
            totalNet = beamNets[best];
            for (int i = 0; i < 32; i++) localHits[i] = beamHits[best][i];
        }

        for (int k = 0; k < BEAM_K; k++) { free(beamData[k]); free(beamHits[k]); }
        free(topK); free(beamNets); free(beamData); free(beamHits);

        free(data);
        netPerBlock[b] = totalNet;

        #pragma omp critical
        for (int i = 0; i < 32; i++)
            instrHits[i] += localHits[i];
    }

    double minNet = netPerBlock[0], maxNet = netPerBlock[0], sumNet = 0.0;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        if (netPerBlock[b] < minNet) minNet = netPerBlock[b];
        if (netPerBlock[b] > maxNet) maxNet = netPerBlock[b];
        sumNet += netPerBlock[b];
    }

    int totalPasses = 0;
    for (int i = 0; i < 32; i++) totalPasses += instrHits[i];

    printf("\n=== %d blocks, %d MB each ===\n", NUM_BLOCKS, BLOCK_SIZE >> 20);
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
}
