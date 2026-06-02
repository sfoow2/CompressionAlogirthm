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

--- PACKED STRIDE (amp packs stride + value, self-inverse for XOR, negate low nibble for ADD) ---
0:  XOR (amp&0x0F), every ((amp>>4)+2)th byte, phase 0   stride 2-17
1:  XOR (amp&0x0F), every ((amp>>4)+2)th byte, phase 1   stride 2-17
2:  ADD (amp&0x0F), every ((amp>>4)+2)th byte, phase 0   reverse: same instr, low nibble = (16-(amp&0x0F))&0x0F
3:  ADD (amp&0x0F), every ((amp>>4)+2)th byte, phase 1   reverse: same instr, low nibble = (16-(amp&0x0F))&0x0F
14: XOR (amp&0x0F)<<4 to HIGH NIBBLE, every ((amp>>4)+2)th byte, phase 0   self-inverse
15: XOR (amp&0x0F)<<4 to HIGH NIBBLE, every ((amp>>4)+2)th byte, phase 1   self-inverse

--- HIGH PRECISION STRIDE 2 (full 8-bit value, fallback when 4-bit isn't enough) ---
4:  XOR amp, stride 2, phase 0
5:  XOR amp, stride 2, phase 1
6:  ADD amp, stride 2, phase 0   reverse: same instr with amp = (256-amp)&0xFF
7:  ADD amp, stride 2, phase 1   reverse: same instr with amp = (256-amp)&0xFF

--- VALUE-CONDITIONAL (condition preserved because target bits are never touched) ---
8:  XOR (amp&0x0F) to LOW  NIBBLE of bytes where (data[i]>>4) == (amp>>4)   self-inverse
9:  XOR (amp&0x7F) to LOW 7 BITS of bytes >= 128   self-inverse
10: XOR (amp&0x7F) to LOW 7 BITS of bytes <  128   self-inverse
11: ADD (amp&0x7F) mod-128 to LOW 7 BITS of bytes >= 128   reverse: same instr, amp = (amp&0x80)|((128-(amp&0x7F))&0x7F)
12: ADD (amp&0x7F) mod-128 to LOW 7 BITS of bytes <  128   reverse: same instr, amp = (amp&0x80)|((128-(amp&0x7F))&0x7F)
13: ADD (amp&0x0F) mod-16  to LOW NIBBLE  of bytes where (data[i]>>4) == (amp>>4)   reverse: same instr, low nibble = (16-(amp&0x0F))&0x0F



*/


void applyInstruction(u8 *data, int size, int instr, int amp) {
    int i, stride;
    u8 xval, addval, band;
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
        for (i = 0; i < size; i += stride) data[i] += addval;
        break;
    case 3:
        stride = (amp >> 4) + 2;
        addval = (u8)(amp & 0x0F);
        for (i = 1; i < size; i += stride) data[i] += addval;
        break;

    /* --- high precision stride 2 --- */
    case 4:
        for (i = 0; i < size; i += 2) data[i] ^= (u8)amp;
        break;
    case 5:
        for (i = 1; i < size; i += 2) data[i] ^= (u8)amp;
        break;
    case 6:
        for (i = 0; i < size; i += 2) data[i] += (u8)amp;
        break;
    case 7:
        for (i = 1; i < size; i += 2) data[i] += (u8)amp;
        break;

    /* --- value-conditional --- */
    case 8:
        band   = (u8)(amp >> 4);
        xval   = (u8)(amp & 0x0F);
        for (i = 0; i < size; i++)
            if ((data[i] >> 4) == band) data[i] ^= xval;
        break;
    case 9:
        xval = (u8)(amp & 0x7F);
        for (i = 0; i < size; i++)
            if (data[i] >= 128) data[i] ^= xval;
        break;
    case 10:
        xval = (u8)(amp & 0x7F);
        for (i = 0; i < size; i++)
            if (data[i] < 128) data[i] ^= xval;
        break;
    case 11:
        addval = (u8)(amp & 0x7F);
        for (i = 0; i < size; i++)
            if (data[i] >= 128)
                data[i] = (u8)(128 + ((data[i] - 128 + addval) & 0x7F));
        break;
    case 12:
        addval = (u8)(amp & 0x7F);
        for (i = 0; i < size; i++)
            if (data[i] < 128)
                data[i] = (u8)((data[i] + addval) & 0x7F);
        break;
    case 13:
        band   = (u8)(amp >> 4);
        addval = (u8)(amp & 0x0F);
        for (i = 0; i < size; i++)
            if ((data[i] >> 4) == band)
                data[i] = (data[i] & 0xF0) | ((data[i] + addval) & 0x0F);
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
    }
}

double FindNextStep(u8 *data, int size) {
    u8 *data2 = malloc(size);
    double baseE    = getEntropy(data, size);
    double bestE    = baseE;
    int bestInstr   = -1;
    int bestAmp     = -1;

    for (int instr = 0; instr < 16; instr++) {
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
    double netBits   = savedBits - 12.0;

    if (bestInstr == -1 || netBits <= 0.0)
        return 0.0;

    applyInstruction(data, size, bestInstr, bestAmp);

    printf("  instr=%2d amp=%3d | entropy %.6f -> %.6f | saved=%.1f  net=%.1f bits\n",
           bestInstr, bestAmp, baseE, bestE, savedBits, netBits);

    return netBits;
}

void main() {
    int size = 1024 * 1024;
    u8* data = malloc(size);
    uint32_t rng = 12345;
    for (int i = 0; i < size; i++) {
        rng ^= rng << 13;
        rng ^= rng >> 17;
        rng ^= rng << 5;
        data[i] = (u8)rng;
    }

    double baseE = getEntropy(data, size);
    printf("Base entropy: %.6f\n", baseE);

    int pass = 0;
    double totalNet = 0.0;
    double net;
    while ((net = FindNextStep(data, size)) > 0.0) {
        totalNet += net;
        pass++;
    }
    printf("Done: %d passes, total net = %.1f bits (%.1f bytes)\n",
           pass, totalNet, totalNet / 8.0);

    free(data);
}
