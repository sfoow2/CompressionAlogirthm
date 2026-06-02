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

--- HIGH PRECISION STRIDE 4 (full 8-bit amp, all 4 phases) ---
26: XOR amp, stride 4, phase 0
27: XOR amp, stride 4, phase 1
28: XOR amp, stride 4, phase 2
29: XOR amp, stride 4, phase 3
30: ADD amp, stride 4, phase 0
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
    case 26:
        for (i = 0; i < size; i += 4) data[i] ^= (u8)amp;
        break;
    case 27:
        for (i = 1; i < size; i += 4) data[i] ^= (u8)amp;
        break;
    case 28:
        for (i = 2; i < size; i += 4) data[i] ^= (u8)amp;
        break;
    case 29:
        for (i = 3; i < size; i += 4) data[i] ^= (u8)amp;
        break;
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
            case 26: stride=4;          phase=0; ptype=0; pval=amp;           break;
            case 27: stride=4;          phase=1; ptype=0; pval=amp;           break;
            case 28: stride=4;          phase=2; ptype=0; pval=amp;           break;
            case 29: stride=4;          phase=3; ptype=0; pval=amp;           break;
            case 30: stride=4;          phase=0; ptype=1; pval=amp;           break;
            case 31: stride=4;          phase=1; ptype=1; pval=amp;           break;
            default: continue;
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
            case 0: for (int v=0;v<256;v++) nf[v^pval]           += sp[v]; break;
            case 1: for (int v=0;v<256;v++) nf[(v+pval)&0xFF]    += sp[v]; break;
            case 2: for (int v=0;v<256;v++) nf[(v&0xF0)|((v+pval)&0xF)] += sp[v]; break;
            }

            double e = entropyFromFreq(nf, size);
            if (e < bestE) { bestE = e; bestInstr = instr; bestAmp = amp; }
        }
    }

    free(pFreq);

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
 * Find top-K first steps via parallel brute-force                     *
 * bufs: pre-allocated per-thread scratch buffers                      *
 * ------------------------------------------------------------------ */
static void FindTopK(u8 *data, int size, int K, Cand *topK) {
    /* build frequency tables once — same layout as FindNextStepST */
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

    double baseE = entropyFromFreq(totalFreq, size);
    for (int k = 0; k < K; k++) topK[k] = (Cand){-1, -1, -1e30};

    int nthreads = omp_get_max_threads();
    Cand *local = malloc(nthreads * K * sizeof(Cand));
    for (int i = 0; i < nthreads * K; i++) local[i] = (Cand){-1, -1, -1e30};

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        Cand *myTop = &local[tid * K];

        #pragma omp for schedule(dynamic)
        for (int instr = 0; instr < 32; instr++) {
            if (instr == 5) continue;
            for (int amp = 0; amp < 256; amp++) {
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
                case 26: stride=4;          phase=0; ptype=0; pval=amp;           break;
                case 27: stride=4;          phase=1; ptype=0; pval=amp;           break;
                case 28: stride=4;          phase=2; ptype=0; pval=amp;           break;
                case 29: stride=4;          phase=3; ptype=0; pval=amp;           break;
                case 30: stride=4;          phase=0; ptype=1; pval=amp;           break;
                case 31: stride=4;          phase=1; ptype=1; pval=amp;           break;
                default: continue;
                }

                const int *sp = pFreq[stride - 2][phase % stride];
                int nf[256];
                memcpy(nf, totalFreq, 256 * sizeof(int));
                for (int v = 0; v < 256; v++) nf[v] -= sp[v];
                switch (ptype) {
                case 0: for (int v=0;v<256;v++) nf[v^pval]                   += sp[v]; break;
                case 1: for (int v=0;v<256;v++) nf[(v+pval)&0xFF]             += sp[v]; break;
                case 2: for (int v=0;v<256;v++) nf[(v&0xF0)|((v+pval)&0xF)]  += sp[v]; break;
                }

                double e   = entropyFromFreq(nf, size);
                double net = (baseE - e) * size - 13.0;
                if (net > myTop[K-1].net) {
                    myTop[K-1] = (Cand){instr, amp, net};
                    for (int k = K-2; k >= 0 && myTop[k+1].net > myTop[k].net; k--) {
                        Cand t = myTop[k]; myTop[k] = myTop[k+1]; myTop[k+1] = t;
                    }
                }
            }
        }
        #pragma omp critical
        for (int k = 0; k < K; k++) {
            if (myTop[k].net > topK[K-1].net) {
                topK[K-1] = myTop[k];
                for (int j = K-2; j >= 0 && topK[j+1].net > topK[j].net; j--) {
                    Cand t = topK[j]; topK[j] = topK[j+1]; topK[j+1] = t;
                }
            }
        }
    }
    free(local);
    free(pFreq);
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
        Cand    *topK     = malloc(BEAM_K * sizeof(Cand));
        double  *beamNets = calloc(BEAM_K, sizeof(double));
        u8     **beamData = malloc(BEAM_K * sizeof(u8 *));
        int    **beamHits = malloc(BEAM_K * sizeof(int *));
        for (int k = 0; k < BEAM_K; k++) {
            beamData[k] = malloc(BLOCK_SIZE);
            memcpy(beamData[k], data, BLOCK_SIZE);
            beamHits[k] = calloc(32, sizeof(int));
        }

        FindTopK(data, BLOCK_SIZE, BEAM_K, topK);

        /* improvement 2: parallel evaluation of K beams across all threads */
        #pragma omp parallel for schedule(dynamic)
        for (int k = 0; k < BEAM_K; k++) {
            if (topK[k].net <= 0.0) continue;
            applyInstruction(beamData[k], BLOCK_SIZE, topK[k].instr, topK[k].amp);
            beamHits[k][topK[k].instr]++;
            /* RunChainST = improvements 1 (full-chain lookahead) + 3 (nested scrambles) */
            beamNets[k] = topK[k].net + RunChainST(beamData[k], BLOCK_SIZE, beamHits[k], 0);
        }

        int best = 0;
        for (int k = 1; k < BEAM_K; k++)
            if (beamNets[k] > beamNets[best]) best = k;

        if (topK[best].net > 0.0) {
            /* verbose replay of the winning beam's chain for display */
            u8 *replay = malloc(BLOCK_SIZE);
            memcpy(replay, data, BLOCK_SIZE);
            printf("  [beam %d/%d selected | paths tried: %d]\n", best+1, BEAM_K, BEAM_K);
            applyInstruction(replay, BLOCK_SIZE, topK[best].instr, topK[best].amp);
            {
                double bE = getEntropy(data, BLOCK_SIZE);
                double aE = getEntropy(replay, BLOCK_SIZE);
                printf("  instr=%2d amp=%3d | entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
                       topK[best].instr, topK[best].amp, bE, aE,
                       (bE-aE)*BLOCK_SIZE, topK[best].net);
            }
            RunChainST(replay, BLOCK_SIZE, NULL, 1);  /* verbose replay */
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
