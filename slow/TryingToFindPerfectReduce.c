#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

typedef uint8_t u8;

static uint32_t xorshift(uint32_t *s) {
    *s ^= *s << 13; *s ^= *s >> 17; *s ^= *s << 5; return *s;
}

double getEntropy(const u8 *data, size_t size) {
    unsigned long freq[256] = {0};
    for (size_t i = 0; i < size; i++) freq[data[i]]++;
    double h = 0.0;
    for (int i = 0; i < 256; i++) {
        if (!freq[i]) continue;
        double p = (double)freq[i] / (double)size;
        h -= p * log2(p);
    }
    return h;
}

void applyXorStream(u8 *d, size_t n, uint32_t seed) {
    uint32_t s = seed;
    for (size_t i = 0; i < n; i++) d[i] ^= (u8)xorshift(&s);
}

// For stride K: greedily find the best XOR seed for each of the K groups.
// Group g = bytes at positions g, g+K, g+2K, ...
// Each group is optimized in sequence, building on the previous group's result.
// Returns final entropy; transformed result is left in `work`.
double greedyStrideGroups(const u8 *data, size_t size, u8 *work, u8 *tmp,
                          int k, uint32_t *out_seeds) {
    memcpy(work, data, size);
    for (int g = 0; g < k; g++) {
        double best = getEntropy(work, size);
        uint32_t bseed = 0;
        for (uint32_t seed = 1; seed <= 65535; seed++) {
            memcpy(tmp, work, size);
            uint32_t s = seed;
            for (size_t i = (size_t)g; i < size; i += k) tmp[i] ^= (u8)xorshift(&s);
            double e = getEntropy(tmp, size);
            if (e < best) { best = e; bseed = seed; }
        }
        out_seeds[g] = bseed;
        if (bseed) {
            uint32_t s = bseed;
            for (size_t i = (size_t)g; i < size; i += k) work[i] ^= (u8)xorshift(&s);
        }
    }
    return getEntropy(work, size);
}

int main(void) {
    const int size = 1024 * 4;
    u8 *data = malloc(size);
    u8 *tmp  = malloc(size);
    u8 *work = malloc(size);

    uint32_t rng = 0xDEADBEEF;
    for (int i = 0; i < size; i++) data[i] = (u8)xorshift(&rng);

    double ent = getEntropy(data, size);
    const double initial = ent;
    double total_overhead = 0.0;
    printf("Starting entropy: %.6f bits/byte\n\n", ent);

    for (int pass = 1; ; pass++) {

        // candidate 1: global XOR stream
        double   xor_ent  = ent;
        uint32_t xor_seed = 0;
        for (uint32_t seed = 1; seed <= 65535; seed++) {
            memcpy(tmp, data, size);
            applyXorStream(tmp, size, seed);
            double e = getEntropy(tmp, size);
            if (e < xor_ent) { xor_ent = e; xor_seed = seed; }
        }

        // candidate 2: stride-K with K independent group seeds (K=2..8)
        double   best_stride_ent  = ent;
        int      best_k           = 0;
        uint32_t best_seeds[8]    = {0};

        for (int k = 2; k <= 8; k++) {
            uint32_t seeds[8] = {0};
            double e = greedyStrideGroups(data, (size_t)size, work, tmp, k, seeds);
            if (e < best_stride_ent) {
                best_stride_ent = e;
                best_k = k;
                memcpy(best_seeds, seeds, k * sizeof(uint32_t));
            }
        }

        // overhead in bits:
        //   XOR stream  = 1 seed (2 bytes)                         = 16 bits
        //   Stride-K    = 1 byte K + 1 byte bitmask (which groups
        //                 are active) + 2 bytes per non-zero seed   = 16 + active*16 bits
        int active = 0;
        for (int g = 0; g < best_k; g++) if (best_seeds[g]) active++;

        double xor_net    = (ent - xor_ent)         * (double)size - 16.0;
        double stride_net = (ent - best_stride_ent) * (double)size - (16.0 + active * 16.0);

        double best_net = (xor_net > stride_net) ? xor_net : stride_net;

        if (best_net <= 0.0) {
            printf("Pass %2d: no profit after overhead — done\n", pass);
            break;
        }

        double best_ent = (xor_net >= stride_net) ? xor_ent : best_stride_ent;

        if (xor_net >= stride_net) {
            applyXorStream(data, size, xor_seed);
            total_overhead += 16.0;
            printf("Pass %2d: XOR stream      seed=%5u  entropy %.6f  net gain %.4f bits\n",
                   pass, xor_seed, best_ent, xor_net);
        } else {
            for (int g = 0; g < best_k; g++) {
                if (best_seeds[g]) {
                    uint32_t s = best_seeds[g];
                    for (size_t i = (size_t)g; i < (size_t)size; i += best_k)
                        data[i] ^= (u8)xorshift(&s);
                }
            }
            total_overhead += 16.0 + active * 16.0;
            printf("Pass %2d: Stride-%d groups  seeds=", pass, best_k);
            for (int g = 0; g < best_k; g++) printf("%u%s", best_seeds[g], g<best_k-1?",":"");
            printf("  entropy %.6f  net gain %.4f bits\n", best_ent, stride_net);
        }
        ent = best_ent;
    }

    double raw_saved = (initial - ent) * (double)size;
    printf("\nInitial:       %.6f bits/byte\n", initial);
    printf("Final:         %.6f bits/byte\n", ent);
    printf("Raw saved:     %.4f bits\n", raw_saved);
    printf("Seed overhead: %.4f bits\n", total_overhead);
    printf("Net saved:     %.4f bits (%.6f bits/byte)\n",
           raw_saved - total_overhead, (raw_saved - total_overhead) / (double)size);

    free(work); free(tmp); free(data);
    return 0;
}
