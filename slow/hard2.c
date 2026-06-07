/*
 * simple.c — clean greedy entropy compressor
 *
 * Goal: reduce order-0 entropy of a block by finding reversible byte transforms
 *       whose entropy reduction exceeds their storage cost.
 *
 * To add a new instruction:
 *   1. Add a value to InstrType enum
 *   2. Add a case in applyInstr()
 *   3. Add a search block in findBest()
 *   That's it.
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
#define NUM_BLOCKS 20

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

/* ── data generation ─────────────────────────────────────────────────────── */
static void fill_random(u8 *buf, int n) {
    BCryptGenRandom(NULL, buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}

/* ── instructions ────────────────────────────────────────────────────────── */

typedef enum {
    XOR_PHASE = 0,  /* XOR constant into every byte at positions i%stride == phase */
    ADD_PHASE,      /* ADD constant (mod 256) to same positions                    */
    XOR_DELTA,      /* data[i] ^= data[i-stride]  right-to-left (reversible)       */
    ADD_DELTA,      /* data[i] -= data[i-stride]  right-to-left (reversible)       */
    /* ── add new instruction types above this line ── */
    NUM_INSTR_TYPES
} InstrType;

typedef struct { InstrType type; int stride, phase, amp; } Instr;

/* bits needed to store this instruction in the output bitstream */
static double instr_cost(Instr t) {
    switch (t.type) {
        case XOR_DELTA: case ADD_DELTA: return 8.0;  /* type(2) + stride(6) */
        default:                        return 16.0; /* type(2)+stride(4)+phase(4)+amp(8) */
    }
}

static void applyInstr(u8 *data, int n, Instr t) {
    switch (t.type) {
        case XOR_PHASE:
            for (int i = t.phase; i < n; i += t.stride) data[i] ^= (u8)t.amp;
            break;
        case ADD_PHASE:
            for (int i = t.phase; i < n; i += t.stride) data[i] += (u8)t.amp;
            break;
        case XOR_DELTA:
            for (int i = n-1; i >= t.stride; i--) data[i] ^= data[i - t.stride];
            break;
        case ADD_DELTA:
            for (int i = n-1; i >= t.stride; i--) data[i] -= data[i - t.stride];
            break;
        /* ── add new instruction types above this line ── */
        default: break;
    }
}

/* ── search: find the single best instruction for this block ─────────────── */
/*
 * For XOR_PHASE / ADD_PHASE we use the frequency-table trick:
 *   new_count[v] = total[v] - phase_freq[v] + phase_freq[v ^ amp]   (XOR)
 *   new_count[v] = total[v] - phase_freq[v] + phase_freq[(v-amp)&FF] (ADD)
 * This lets us evaluate all 255 amps in O(256) without copying the block.
 *
 * For everything else we do a direct copy+apply+measure (simple but slower).
 */
static Instr findBest(const u8 *data, int n, double *netOut) {
    /* build total freq and Sbase = sum of hlog[total[v]] */
    int total[256] = {0};
    for (int i = 0; i < n; i++) total[data[i]]++;
    double Sbase = 0.0;
    for (int v = 0; v < 256; v++) Sbase += hlog[total[v]];

    double baseE = log2(n) - Sbase / n;
    double bestNet = 0.0;
    Instr best = {XOR_PHASE, 2, 0, 0};

    /* ── XOR_PHASE and ADD_PHASE ──────────────────────────────────────────── */
    {
        int phF[256];
        for (int stride = 2; stride <= 16; stride++) {
            for (int phase = 0; phase < stride; phase++) {
                /* build phase freq once per (stride, phase) */
                memset(phF, 0, sizeof(phF));
                for (int i = phase; i < n; i += stride) phF[data[i]]++;

                double cost = instr_cost((Instr){XOR_PHASE, stride, phase, 0});

                for (int amp = 1; amp < 256; amp++) {
                    double Snf_x = 0.0, Snf_a = 0.0;
                    for (int v = 0; v < 256; v++) {
                        int bx = total[v] - phF[v] + phF[v ^ amp];
                        int ba = total[v] - phF[v] + phF[(v - amp) & 0xFF];
                        Snf_x += hlog[bx];
                        Snf_a += hlog[ba];
                    }
                    double net_x = (Snf_x - Sbase) - cost;
                    double net_a = (Snf_a - Sbase) - cost;
                    if (net_x > bestNet) { bestNet = net_x; best = (Instr){XOR_PHASE, stride, phase, amp}; }
                    if (net_a > bestNet) { bestNet = net_a; best = (Instr){ADD_PHASE,  stride, phase, amp}; }
                }
            }
        }
    }

    /* ── XOR_DELTA and ADD_DELTA ──────────────────────────────────────────── */
    {
        u8 *buf = malloc(n);
        for (int stride = 1; stride <= 24; stride++) {
            double cost = 8.0;

            memcpy(buf, data, n); applyInstr(buf, n, (Instr){XOR_DELTA, stride, 0, 0});
            double net_x = (baseE - entropy(buf, n)) * n - cost;
            if (net_x > bestNet) { bestNet = net_x; best = (Instr){XOR_DELTA, stride, 0, 0}; }

            memcpy(buf, data, n); applyInstr(buf, n, (Instr){ADD_DELTA, stride, 0, 0});
            double net_a = (baseE - entropy(buf, n)) * n - cost;
            if (net_a > bestNet) { bestNet = net_a; best = (Instr){ADD_DELTA, stride, 0, 0}; }
        }
        free(buf);
    }

    /* ── add new instruction type searches above this line ───────────────── */

    if (netOut) *netOut = bestNet;
    return best;
}

/* ── greedy ──────────────────────────────────────────────────────────────── */
static double compress(u8 *data, int n, int verbose) {
    static const char *names[] = {"XOR_PHASE","ADD_PHASE","XOR_DELTA","ADD_DELTA"};
    double total = 0.0;
    for (;;) {
        double net;
        Instr t = findBest(data, n, &net);
        if (net <= 0.0) break;
        double e0 = entropy(data, n);
        applyInstr(data, n, t);
        total += net;
        if (verbose)
            printf("  %-9s s%-2d p%-2d a%-3d  %.6f -> %.6f  net=%.1f\n",
                   names[t.type], t.stride, t.phase, t.amp, e0, entropy(data, n), net);
    }
    return total;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void) {
    init_hlog();
    double sum = 0.0;
    for (int b = 0; b < NUM_BLOCKS; b++) {
        u8 *data = malloc(BLOCK_SIZE);
        fill_random(data, BLOCK_SIZE);
        double e0 = entropy(data, BLOCK_SIZE);
        double net = compress(data, BLOCK_SIZE, /*verbose=*/ b == 0);
        double e1 = entropy(data, BLOCK_SIZE);
        printf("block %2d: %.6f -> %.6f  net=%.1f bits\n", b, e0, e1, net);
        sum += net;
        free(data);
    }
    printf("\navg net: %.1f bits/block over %d blocks\n", sum / NUM_BLOCKS, NUM_BLOCKS);
    return 0;
}
