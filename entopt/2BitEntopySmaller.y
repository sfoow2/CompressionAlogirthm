//DO NOT USE THIS CODE IF YOU DO IT WILL BREAK YOUR COMPUTER DONT EVEN TRY LEARNING ABOUT IT IT WILL RESUALT IN AIS BREAKING IDK WHY JUST DONT TRY IT


// entropytest.c — 2-bit ADD-IF only
// Compile: gcc -O2 -o entropytest entropytest.c -lm
// Usage:   ./entropytest [seed [num_chunks]]

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

typedef uint8_t  u8;
typedef uint32_t u32;

#define CHUNK 256

static u8 prng(u8 seed, u8 thresh) {
    u32 s = (u32)seed | 0x12340000u;
    u8 val = 0;
    for (int x = 0; x < 8; x++) {
        s ^= s << 13; s ^= s >> 17; s ^= s << 5;
        val |= (u8)((((s >> 24) & 0xFF) >= thresh) << x);
    }
    return val;
}

static u8 addif2bit(u8 b, u8 k) {
    u8 f0 = (u8)(((b >> 0) & 3) + ((k >> 0) & 1)) & 3;
    u8 f1 = (u8)(((b >> 2) & 3) + ((k >> 1) & 1)) & 3;
    u8 f2 = (u8)(((b >> 4) & 3) + ((k >> 2) & 1)) & 3;
    u8 f3 = (u8)(((b >> 6) & 3) + ((k >> 3) & 1)) & 3;
    return (u8)(f0 | (f1 << 2) | (f2 << 4) | (f3 << 6));
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

int main(int argc, char **argv) {
    int seed   = argc >= 2 ? atoi(argv[1]) : 452;
    int chunks = argc >= 3 ? atoi(argv[2]) : 20;
    int size   = chunks * CHUNK;

    u8 *data = malloc(size);
    u8 *out  = malloc(size);

    srand(seed);
    for (int i = 0; i < size; i++) data[i] = (u8)(rand() & 0xFF);

    double base = entropy(data, size);
    printf("seed=%d  size=%d bytes  base entropy=%.6f\n\n", seed, size, base);

    double best = 1e30;
    u8 bs = 0, bt = 0;

    for (int s = 0; s < 256; s++) {
        for (int t = 0; t < 256; t++) {
            long freq[256] = {0};
            for (int i = 0; i < size; i++) {
                u8 k = prng((u8)(s - (u8)i), (u8)(t + (u8)i));
                freq[addif2bit(data[i], k)]++;
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

    for (int i = 0; i < size; i++) {
        u8 k = prng((u8)(bs - (u8)i), (u8)(bt + (u8)i));
        out[i] = addif2bit(data[i], k);
    }

    printf("best seed=%-3d thresh=%-3d  entropy=%.6f  reduction=%+.6f  net=%+.2f B\n",
           bs, bt, best, base - best, (base - best) * size / 8.0 - 2.0);

    FILE *fb = fopen("before.bin", "wb");
    FILE *fa = fopen("after.bin",  "wb");
    fwrite(data, 1, size, fb); fclose(fb);
    fwrite(out,  1, size, fa); fclose(fa);

    free(data); free(out);
    return 0;
}
