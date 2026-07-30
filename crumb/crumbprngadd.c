#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define FILE_PATH "C:\\Users\\lukac\\Documents\\compressor\\actuallstuff\\CurrentDataFile.bin"
#define CHUNK_BYTES 11    // bytes read per chunk -- each byte expands to 4 crumbs (2 bits each)
                          // Re-gridded after the prng_crumb aliasing fix. Two independent
                          // 512 KiB replicates (offsets 0 and 100 MB, ~50k chunks per cell,
                          // measured SE ~0.008 bits/chunk), sb=16:
                          //     cb= 9 -> 3.631 / 3.620
                          //     cb=10 -> 3.717 / 3.713
                          //     cb=11 -> 3.743 / 3.750   <-- peak
                          //     cb=12 -> 3.589 / 3.590
                          // cb=11 beats cb=10 by 0.0315 +/- 0.0084 (3.7 sigma pooled); the
                          // replicates agree cell-for-cell within 1 sigma. A smaller 64 KiB
                          // sweep put cb=10 on top, but that gap was 0.8 sigma -- noise that
                          // flipped once the sample grew. Do not re-tune this from short runs:
                          // per-chunk sd is ~1.8-2.1 bits, so ~50k chunks per cell is the
                          // minimum for 3-sigma separation between adjacent cb.
                          // cb=16 measures 3.27, so the old "cb=16 optimum, 3.735" was the
                          // right number on the wrong row.
                          //
                          // CAVEAT: net/chunk is not comparable across chunk sizes -- a bigger
                          // chunk covers more input for the same ~3 bits. Per BYTE there is no
                          // interior optimum at all: net/byte rises monotonically as cb shrinks,
                          // reaching 1.50 bits/byte (18.8%) at cb=1/sb=4 on already-compressed
                          // input, which is impossible. That divergence is the tell that this
                          // metric measures the model-cost accounting gap, not compression.
                          // Priced with a decodable code (enumerative: log2(#histograms) +
                          // log2(multinomial) + seed), every cell of the grid is negative:
                          // -2.67 bits/byte at cb=1, -0.037 at cb=256, approaching 0 from below
                          // as cb grows. The honest optimum is "chunk size infinity" -- i.e. do
                          // not apply the transform. See net/chunk here as an instrument
                          // reading, not a compression ratio.
#define NUM_CHUNKS 3      // number of chunks to process before stopping
#define SEED_BITS 16      // width of the per-chunk PRNG seed space. The old "plateau above 16"
                          // was an artifact of the seed-aliasing bug, not a real saturation.
                          // With distinct seeds each added bit buys ~1 bit of extra entropy
                          // reduction and costs exactly 1 bit of overhead, so net stays flat:
                          // widening the search cannot pay for itself.
#define SEED_COUNT (1L << SEED_BITS)   // seeds swept per chunk

#define CRUMBS_PER_CHUNK (CHUNK_BYTES * 4)
#define MAX_CRUMB_ENTROPY 2.0   // ceiling for a 4-symbol (2-bit) alphabet

static double crumb_entropy(const long freq[4], long total) {
    double entropy = 0.0;
    for (int i = 0; i < 4; i++) {
        if (freq[i] == 0) continue;
        double p = (double)freq[i] / (double)total;
        entropy -= p * log2(p);
    }
    return entropy;
}

/* lowbias32-style integer avalanche mixer: every input bit flips ~half the
   output bits, so nearby (seed, pos) pairs give unrelated crumb streams. */
static uint32_t avalanche_hash(uint32_t x) {
    x ^= x >> 16;
    x *= 0x7feb352dU;
    x ^= x >> 15;
    x *= 0x846ca68bU;
    x ^= x >> 16;
    return x;
}

static unsigned char prng_crumb(unsigned int seed, unsigned int pos) {
    /* Hash the seed into a full 32-bit key before folding in pos. The earlier
       (seed << 16) ^ pos packing aliased: seeds s and s + 2^16 shifted to the
       same value, so every seed at or above 2^16 was an exact duplicate of one
       already swept. avalanche_hash is a bijection on uint32, so distinct seeds
       now give distinct keys for the whole SEED_BITS range. */
    uint32_t h = avalanche_hash(avalanche_hash(seed) ^ (pos * 0x9E3779B1u));
    return (unsigned char)(h & 0x3u);   // uniform over {00,01,10,11}
}

int main(void) {
    FILE *f = fopen(FILE_PATH, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", FILE_PATH);
        return 1;
    }

    unsigned char bytebuf[CHUNK_BYTES];
    unsigned char crumbs[CRUMBS_PER_CHUNK];

    // running net stats across every chunk actually processed
    double net_sum = 0.0, net_min = 0.0, net_max = 0.0;
    long   net_n = 0;
    int    net_min_chunk = -1, net_max_chunk = -1;

    for (int chunk = 0; chunk < NUM_CHUNKS; chunk++) {
        size_t got = fread(bytebuf, 1, CHUNK_BYTES, f);
        if (got == 0) {
            printf("=== EOF reached before chunk %d ===\n", chunk);
            break;
        }

        long freq[4] = {0, 0, 0, 0};
        long total = 0;
        for (size_t b = 0; b < got; b++) {
            unsigned char byte = bytebuf[b];
            for (int shift = 6; shift >= 0; shift -= 2) {
                unsigned char crumb = (byte >> shift) & 0x3;
                crumbs[total] = crumb;
                freq[crumb]++;
                total++;
            }
        }

        double entropy = crumb_entropy(freq, total);

        printf("chunk %-4d  entropy=%.4f/%.1f  freq: 00=%-5ld 01=%-5ld 10=%-5ld 11=%-5ld\n",
               chunk, entropy, MAX_CRUMB_ENTROPY, freq[0], freq[1], freq[2], freq[3]);

        // sweep every seed in the SEED_BITS-wide space, add it (mod 4) onto
        // this chunk's crumb stream, and keep whichever seed drives entropy lowest
        long best_seed = -1;
        double best_entropy = MAX_CRUMB_ENTROPY + 1.0;
        long best_freq[4] = {0, 0, 0, 0};

        for (long seed = 0; seed < SEED_COUNT; seed++) {
            long freq2[4] = {0, 0, 0, 0};
            for (long i = 0; i < total; i++) {
                unsigned char x = (unsigned char)((crumbs[i] + prng_crumb((unsigned int)seed, (unsigned int)i)) & 0x3);
                freq2[x]++;
            }
            double e = crumb_entropy(freq2, total);
            if (e < best_entropy) {
                best_entropy = e;
                best_seed = seed;
                best_freq[0] = freq2[0]; best_freq[1] = freq2[1];
                best_freq[2] = freq2[2]; best_freq[3] = freq2[3];
            }
        }

        double net = (MAX_CRUMB_ENTROPY - best_entropy) * (double)total - SEED_BITS;
        if (net_n == 0 || net < net_min) { net_min = net; net_min_chunk = chunk; }
        if (net_n == 0 || net > net_max) { net_max = net; net_max_chunk = chunk; }
        net_sum += net;
        net_n++;

        printf("  best add seed: seed=%-6ld (of %ld)  entropy=%.4f/%.1f  net=%.2f bits  freq: 00=%-5ld 01=%-5ld 10=%-5ld 11=%-5ld\n",
               best_seed, SEED_COUNT, best_entropy, MAX_CRUMB_ENTROPY, net,
               best_freq[0], best_freq[1], best_freq[2], best_freq[3]);

        if (got < (size_t)CHUNK_BYTES) {
            printf("=== EOF reached mid-chunk ===\n");
            break;
        }
    }

    fclose(f);
    return 0;
}
