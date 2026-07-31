#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

#define FILE_PATH "C:\\Users\\lukac\\Documents\\compressor\\actuallstuff\\CurrentDataFile.bin"
#define CHUNK_BYTES 20    // bytes read per chunk -- each byte expands to 4 crumbs (2 bits each)
#define NUM_CHUNKS (32)      // number of chunks to process before stopping
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
/* 1ULL, not 1L: long is 32-bit on Windows, so 1L<<31 overflows to negative
   (loop never runs) and 1L<<32 upward is undefined. uint64 is good to 63. */
#define SEED_COUNT (1ULL << SEED_BITS)  // seeds swept per chunk

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

/* splitmix64 finaliser: every input bit flips ~half the output bits, so nearby
   (seed, pos) pairs give unrelated crumb streams. Widened from the old 32-bit
   lowbias32 mixer -- see the aliasing note in prng_crumb below. */
static uint64_t avalanche_hash(uint64_t x) {
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ULL;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebULL;
    x ^= x >> 31;
    return x;
}

/* pos * PHI64 depends only on the position, never on the seed, so the whole
   table is built once for the run instead of recomputed on every inner step. */
static uint64_t pos_key[CRUMBS_PER_CHUNK];
static void init_pos_keys(void) {
    for (uint32_t i = 0; i < CRUMBS_PER_CHUNK; i++)
        pos_key[i] = (uint64_t)i * 0x9E3779B97F4A7C15ULL;
}

/* The crumb stream is  avalanche_hash( avalanche_hash(seed) ^ pos_key[pos] ).
   The inner avalanche_hash(seed) is invariant across positions, so it is split
   out as seed_key and hoisted to the top of the seed loop -- it used to be
   recomputed once per crumb, which was half of all hash work in the program.

   Hashing the seed to a full 64-bit key before folding in pos is what stops
   seeds aliasing. That has bitten twice: the original (seed << 16) ^ pos packing
   made every seed at or above 2^16 a duplicate of one already swept, and the
   uint32 mixer that replaced it did the same at 2^32 -- invisible while
   SEED_BITS was 16, fatal the moment it goes past 32. avalanche_hash is a
   bijection on uint64, so distinct seeds give distinct keys up to SEED_BITS 63. */
static inline uint64_t prng_seed_key(uint64_t seed) {
    return avalanche_hash(seed);
}
static inline unsigned char prng_crumb_keyed(uint64_t seed_key, uint32_t pos) {
    return (unsigned char)(avalanche_hash(seed_key ^ pos_key[pos]) & 0x3u);
}

int main(void) {
    FILE *f = fopen(FILE_PATH, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open file: %s\n", FILE_PATH);
        return 1;
    }

    init_pos_keys();

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
        long long best_seed = -1;
        double best_entropy = MAX_CRUMB_ENTROPY + 1.0;
        long best_freq[4] = {0, 0, 0, 0};

        for (uint64_t seed = 0; seed < SEED_COUNT; seed++) {
            uint64_t seed_key = prng_seed_key(seed);   // hoisted out of the position loop
            long freq2[4] = {0, 0, 0, 0};
            for (long i = 0; i < total; i++) {
                unsigned char x = (unsigned char)((crumbs[i] + prng_crumb_keyed(seed_key, (uint32_t)i)) & 0x3);
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

        printf("  best add seed: seed=%-11lld (of %llu)  entropy=%.4f/%.1f  net=%.2f bits  freq: 00=%-5ld 01=%-5ld 10=%-5ld 11=%-5ld\n",
               best_seed, (unsigned long long)SEED_COUNT, best_entropy, MAX_CRUMB_ENTROPY, net,
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
