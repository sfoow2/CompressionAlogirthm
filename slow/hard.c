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

double FindNextStep(u8 *data, int size, int *usedInstr, int verbose) {
    u8 *data2 = malloc(size);
    double baseE    = getEntropy(data, size);
    double bestE    = baseE;
    int bestInstr   = -1;
    int bestAmp     = -1;

    for (int instr = 0; instr < 32; instr++) {
        if (instr == 5) continue;  /* deinterleave never wins on entropy alone; handled via lookahead */
        for (int amp = 0; amp < 256; amp++) {
            memcpy(data2, data, size);
            applyInstruction(data2, size, instr, amp);
            double e = getEntropy(data2, size);
            if (e < bestE) {
                bestE     = e;
                bestInstr = instr;
                bestAmp   = amp;
            }
        }
    }

    free(data2);

    double savedBits = (baseE - bestE) * size;
    double netBits   = savedBits - 13.0;

    if (bestInstr == -1 || netBits <= 0.0) {
        if (usedInstr) *usedInstr = -1;
        return 0.0;
    }

    applyInstruction(data, size, bestInstr, bestAmp);

    if (verbose)
        printf("  instr=%2d amp=%3d | entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
               bestInstr, bestAmp, baseE, bestE, savedBits, netBits);

    if (usedInstr) *usedInstr = bestInstr;
    return netBits;
}

void main() {
    const int NUM_BLOCKS  = 1;
    const int BLOCK_SIZE  = 1024 * 1024;

    double netPerBlock[200] = {0};
    int    instrHits[32]    = {0};

    printf("Running %d blocks on %d thread(s)...\n",
           NUM_BLOCKS, omp_get_max_threads());

    #pragma omp parallel for schedule(dynamic)
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
        int usedInstr;
        double net;

        for (;;) {
            /* run normal chain until it stalls */
            while ((net = FindNextStep(data, BLOCK_SIZE, &usedInstr, 1)) > 0.0) {
                totalNet += net;
                localHits[usedInstr]++;
            }

            /* chain stalled: try every deinterleave stride, pick the one whose
               best follow-up step covers both overheads (deinterleave + next step) */
            u8 *scTmp = malloc(BLOCK_SIZE);
            int bestSAmp  = -1;
            double bestSGain = 0.0;

            for (int amp = 0; amp < 256; amp++) {
                memcpy(scTmp, data, BLOCK_SIZE);
                applyInstruction(scTmp, BLOCK_SIZE, 5, amp);
                int dummy;
                double nextNet = FindNextStep(scTmp, BLOCK_SIZE, &dummy, 0);
                double gain = nextNet - 13.0;
                if (gain > bestSGain) {
                    bestSGain = gain;
                    bestSAmp  = amp;
                }
            }
            free(scTmp);

            if (bestSAmp == -1)
                break;

            applyInstruction(data, BLOCK_SIZE, 5, bestSAmp);
            totalNet -= 13.0;
            localHits[5]++;
            printf("  scramble: stride=%d | unlocks %.1f bits net\n",
                   bestSAmp + 2, bestSGain);
            /* loop back: run chain again on rearranged data, then try scramble again */
        }

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
