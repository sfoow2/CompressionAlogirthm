#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define FILE_PATH "C:\\Users\\lukac\\Documents\\compressor\\actuallstuff\\CurrentDataFile.bin"
#define CHUNK_BYTES 20    // bytes read per chunk -- each byte expands to 4 crumbs (2 bits each)
#define NUM_CHUNKS (64)      // number of chunks to process before stopping
/* Width of the per-chunk PRNG seed space. This and CHUNK_BYTES are NOT
   independent -- the best chunk size depends on the seed budget, so changing
   one without re-tuning the other loses real net. Do not treat them separately.

   CHUNK_BYTES is tuned FOR THIS 16-bit seed. Swept 24-72 crumbs at 6000 chunks
   per size, se ~0.024:

       40 crumbs (10.00 B)  +3.742   <- current; byte-aligned, so expressible here
       41 crumbs (10.25 B)  +3.745   in-sample best, but a max over 49 candidates
                                     and only 0.003 ahead -- not a real difference
       44 crumbs (11.00 B)  +3.706   the old setting
       56 crumbs (14.00 B)  +3.345   optimal at 21 bits, badly mismatched at 16

   The optimum is a plateau over roughly 39-45 crumbs; anything in that band is
   within noise. If the seed budget ever changes, re-sweep: at 23 bits the peak
   moves out to 58 crumbs (+4.094).

   Two caveats on the number itself. Roughly +2.2 of it is the plug-in entropy
   estimator's small-sample bias, (A-1)/(2*ln2) = 2.164 bits, which is already
   there at zero search and zero bits paid -- only ~1.5 comes from the seed
   search. And splitting the budget with a shuffle seed, or flagging between
   add/xor, never beats spending every bit on the plain seed. */
#define SEED_BITS 21
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

    if (net_n > 0) {
        printf("\n=== net over %ld chunk%s (cb=%d, seed_bits=%d) ===\n",
               net_n, net_n == 1 ? "" : "s", CHUNK_BYTES, SEED_BITS);
        printf("  avg net = %7.2f bits/chunk\n", net_sum / (double)net_n);
        printf("  min net = %7.2f bits  (chunk %d)\n", net_min, net_min_chunk);
        printf("  max net = %7.2f bits  (chunk %d)\n", net_max, net_max_chunk);
        printf("  total   = %7.2f bits over %ld bytes\n",
               net_sum, net_n * (long)CHUNK_BYTES);
    }

    fclose(f);
    return 0;
}
