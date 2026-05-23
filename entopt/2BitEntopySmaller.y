//DO NOT USE THIS CODE IF YOU DO IT WILL BREAK YOUR COMPUTER DONT EVEN TRY LEARNING ABOUT IT IT WILL RESUALT IN AIS BREAKING IDK WHY JUST DONT TRY IT

// entropytest.c
// Streaming xorshift32 — unique key at every position, no period-256 repeat.
// Seed = (s, t) packed into the 32-bit xorshift initial state.
// Search: 65,536 (s,t) pairs. Applies addif2bit and addifNibble to compare.
//
// Compile: gcc -O2 -o entropytest entropytest.c -lm
// Usage:   ./entropytest [data_seed [size_bytes]]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

typedef uint8_t  u8;
typedef uint32_t u32;

// 4 x 2-bit fields, add 0 or 1 per field mod 4 (uses bits 0-3 of k)
static u8 addif2bit(u8 b, u8 k) {
    u8 f0 = (u8)(((b >> 0) & 3) + ((k >> 0) & 1)) & 3;
    u8 f1 = (u8)(((b >> 2) & 3) + ((k >> 1) & 1)) & 3;
    u8 f2 = (u8)(((b >> 4) & 3) + ((k >> 2) & 1)) & 3;
    u8 f3 = (u8)(((b >> 6) & 3) + ((k >> 3) & 1)) & 3;
    return (u8)(f0 | (f1 << 2) | (f2 << 4) | (f3 << 6));
}

// 2 x 4-bit nibbles, add 0 or 1 per nibble mod 16 (uses bits 0-1 of k)
static u8 addifNibble(u8 b, u8 k) {
    u8 n0 = (u8)((b & 0x0F) + ((k >> 0) & 1)) & 0x0F;
    u8 n1 = (u8)((b >> 4)   + ((k >> 1) & 1)) & 0x0F;
    return (u8)(n0 | (n1 << 4));
}

static double entropy(u8 *data, int size) {
    long freq[256] = {0};
    for (int i = 0; i < size; i++) freq[data[i]]++;
    double e = 0.0;
    for (int v = 0; v < 256; v++) {
        if (!freq[v]) continue;
        double p = (double)freq[v] / size;
        e -= p * log2(p);
    }
    return e;
}

// Streaming xorshift32: state evolves continuously, one key byte per position.
// No repeat — every position gets a unique key derived from the running state.
static double searchStream(u8 *data, int size, u8 *dst, int nibble,
                           u8 *out_s, u8 *out_t) {
    double best = 1e30;
    u8 bs = 0, bt = 0;

    for (int s = 0; s < 256; s++) {
        for (int t = 0; t < 256; t++) {
            // Pack (s,t) into the xorshift32 seed. 0x12340000 ensures non-zero state.
            u32 state = (u32)s | ((u32)t << 8) | 0x12340000u;
            int freq[256] = {0};
            for (int i = 0; i < size; i++) {
                state ^= state << 13;
                state ^= state >> 17;
                state ^= state << 5;
                u8 k = (u8)(state >> 24);
                freq[nibble ? addifNibble(data[i], k) : addif2bit(data[i], k)]++;
            }
            double e = 0.0;
            for (int v = 0; v < 256; v++) {
                if (!freq[v]) continue;
                double p = (double)freq[v] / size;
                e -= p * log2(p);
            }
            if (e < best) { best = e; bs = (u8)s; bt = (u8)t; }
        }
    }

    *out_s = bs; *out_t = bt;
    u32 state = (u32)bs | ((u32)bt << 8) | 0x12340000u;
    for (int i = 0; i < size; i++) {
        state ^= state << 13; state ^= state >> 17; state ^= state << 5;
        u8 k = (u8)(state >> 24);
        dst[i] = nibble ? addifNibble(data[i], k) : addif2bit(data[i], k);
    }
    return best;
}

int main(int argc, char **argv) {
    int data_seed = argc >= 2 ? atoi(argv[1]) : 452;
    int size      = argc >= 3 ? atoi(argv[2]) : 5120;

    u8 *data     = malloc(size);
    u8 *out      = malloc(size);
    u8 *best_out = malloc(size);

    srand(data_seed);
    for (int i = 0; i < size; i++) data[i] = (u8)(rand() & 0xFF);

    double base = entropy(data, size);
    printf("seed=%d  size=%d bytes  base entropy=%.6f\n\n", data_seed, size, base);

    double best_ent = base;
    const char *winner = "none";
    memcpy(best_out, data, size);

#define TRY(label, nibble) do {                                               \
    u8 _s, _t;                                                                \
    printf("%-20s ", (label)); fflush(stdout);                                \
    double _e = searchStream(data, size, out, (nibble), &_s, &_t);           \
    printf("s=%-3d t=%-3d  entropy=%.6f  net=%+.2f B\n",                    \
           _s, _t, _e, (base-_e)*size/8.0 - 2.0);                           \
    if (_e < best_ent) { best_ent=_e; memcpy(best_out,out,size); winner=(label); } \
} while(0)

    TRY("2-bit ADD-IF",  0);
    TRY("Nibble ADD-IF", 1);

#undef TRY

    printf("\n=== WINNER: %s ===\n", winner);
    printf("Entropy: %.6f -> %.6f  (reduction %.6f bits)\n",
           base, best_ent, base - best_ent);
    printf("Net savings: %.2f bytes  (after 2-byte overhead)\n",
           (base - best_ent) * size / 8.0 - 2.0);

    FILE *fb = fopen("before.bin", "wb");
    FILE *fa = fopen("after.bin",  "wb");
    fwrite(data,     1, size, fb); fclose(fb);
    fwrite(best_out, 1, size, fa); fclose(fa);

    free(data); free(out); free(best_out);
    return 0;
}
