//DO NOT USE THIS CODE IF YOU DO IT WILL BREAK YOUR COMPUTER DONT EVEN TRY LEARNING ABOUT IT IT WILL RESUALT IN AIS BREAKING IDK WHY JUST DONT TRY IT
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint8_t  u8;
typedef uint64_t u64;

// xorshift64: data generation only
static inline u64 xs64(u64 s) { s^=s<<13; s^=s>>7; s^=s<<17; return s; }

// xoshiro256 without output transform = 100% linear over GF(2), 256-bit state
// All ops are XOR, shift, or rotate — all linear over GF(2)
static void xs256_step(u64 s[4]) {
    u64 t  = s[1] << 17;
    s[2] ^= s[0];  s[3] ^= s[1];
    s[1] ^= s[2];  s[0] ^= s[3];
    s[2] ^= t;
    s[3]  = (s[3] << 45) | (s[3] >> 19);  // rotate = linear
}

static float byteEntropy(const u8 *d, int n) {
    int freq[256] = {0};
    for (int i = 0; i < n; i++) freq[d[i]]++;
    double e = 0.0;
    for (int i = 0; i < 256; i++) {
        if (!freq[i]) continue;
        double p = (double)freq[i] / n;
        e -= p * log2(p);
    }
    return (float)e;
}


int main() {
    const int DATA_SEED = 42;
    const int SIZE      = 1048;

    u64 ds = (u64)DATA_SEED;
    u8 *data = malloc(SIZE);
    for (int i = 0; i < SIZE; i++) { ds = xs64(ds); data[i] = (u8)(ds & 0xFF); }

    u8 *data_bits = malloc(SIZE);
    int data_ones = 0;
    for (int i = 0; i < SIZE; i++) {
        data_bits[i] = (data[i] < 128) ? 1 : 0;
        data_ones += data_bits[i];
    }

    float ent_before = byteEntropy(data, SIZE);
    printf("Data seed: %d (xorshift64) | Size: %d bytes\n", DATA_SEED, SIZE);
    printf("Bytes < 128:  %d / %d (%.1f%%)\n", data_ones, SIZE, 100.0f * data_ones / SIZE);
    printf("Entropy before: %.6f bpb\n", ent_before);

    // Entropy reducer: for each byte, if the bit is 1, add 128 (push low bytes up)
    u8 *out = malloc(SIZE);
    memcpy(out, data, SIZE);
    for (int i = 0; i < SIZE; i++)
        if (data_bits[i]) out[i] += 128;

    int low_after = 0;
    for (int i = 0; i < SIZE; i++) low_after += (out[i] < 128);
    float ent_after = byteEntropy(out, SIZE);
    printf("Entropy after:  %.6f bpb  (delta %.6f)\n", ent_after, ent_before - ent_after);
    printf("Bytes < 128 after: %d / %d (%.1f%%)\n", low_after, SIZE, 100.0f * low_after / SIZE);

    free(data); free(data_bits); free(out);
    return 0;
}
