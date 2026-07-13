#include <stdio.h>
#include <math.h>
#include <stdint.h>

#define INPUT_PATH "C:\\Users\\lukac\\Documents\\compressor\\compresseddata.bin"
#define N 64

#define NUM_CHUNKS 25
#define HEADER_SKIP 64

/* Search all 3-byte xorshift seeds for the one whose MSB stream best
 * matches some repeating period-8 pattern. Since the pattern repeats
 * every 8 bytes and N=64 is a multiple of 8, byte i's contribution
 * depends only on i%8 ("phase"), so the best pattern for a given seed
 * is just the per-phase majority bit -- no need to test all 256
 * patterns explicitly. */
static void search_chunk(const unsigned char *buf,
                          uint32_t *out_seed, uint8_t *out_pattern, int *out_matches) {
    int best_matches = -1;
    uint32_t best_seed = 0;
    int best_phase_ones[8] = {0};

    for (uint32_t seed = 1; seed <= 0xFFFFFFu; seed++) {
        uint32_t state = seed;
        int phase_ones[8] = {0};
        for (int i = 0; i < N; i++) {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            uint8_t ks = (uint8_t)(state & 0xFF);
            uint8_t nv = (uint8_t)(buf[i] + ks);
            phase_ones[i & 7] += (nv >> 7) & 1;
        }

        int matches = 0;
        for (int j = 0; j < 8; j++) {
            int ones = phase_ones[j];
            int zeros = 8 - ones;
            matches += (ones >= zeros) ? ones : zeros;
        }

        if (matches > best_matches) {
            best_matches = matches;
            best_seed = seed;
            for (int j = 0; j < 8; j++) best_phase_ones[j] = phase_ones[j];
        }
    }

    uint8_t pattern = 0;
    for (int j = 0; j < 8; j++) {
        if (best_phase_ones[j] >= 4) pattern |= (1u << (7 - j));
    }

    *out_seed = best_seed;
    *out_pattern = pattern;
    *out_matches = best_matches;
}

int main(void) {
    FILE *f = fopen(INPUT_PATH, "rb");
    if (!f) { perror(INPUT_PATH); return 1; }

    unsigned char buf[N];
    size_t n = fread(buf, 1, N, f);
    fclose(f);

    int counts[256] = {0};
    for (size_t i = 0; i < n; i++) counts[buf[i]]++;

    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (counts[i] == 0) continue;
        double p = (double)counts[i] / n;
        entropy -= p * log2(p);
    }

    printf("Read %zu bytes from %s\n", n, INPUT_PATH);
    printf("Entropy: %f bits/byte\n", entropy);

    printf("Bytes (decimal):\n");
    for (size_t i = 0; i < n; i++) {
        printf("%d ", buf[i]);
        if ((i + 1) % 16 == 0) printf("\n");
    }
    if (n % 16 != 0) printf("\n");

    FILE *fall = fopen(INPUT_PATH, "rb");
    if (!fall) { perror(INPUT_PATH); return 1; }
    long total = 0;
    long full_counts[256] = {0};
    int c;
    while ((c = fgetc(fall)) != EOF) {
        full_counts[c]++;
        total++;
    }
    fclose(fall);

    double full_entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (full_counts[i] == 0) continue;
        double p = (double)full_counts[i] / total;
        full_entropy -= p * log2(p);
    }
    printf("Whole file entropy (%ld bytes): %f bits/byte\n", total, full_entropy);

    /* --- brute-force 3-byte xorshift seed search for best repeating MSB
     * pattern, run independently over NUM_CHUNKS chunks spread across the
     * file (skipping the first HEADER_SKIP bytes, which look like a
     * header rather than coded payload) --- */
    FILE *fc = fopen(INPUT_PATH, "rb");
    if (!fc) { perror(INPUT_PATH); return 1; }
    fseek(fc, 0, SEEK_END);
    long filesize = ftell(fc);

    long available = filesize - HEADER_SKIP - N;
    if (available < 0) available = 0;
    long stride = (NUM_CHUNKS > 1) ? available / (NUM_CHUNKS - 1) : 0;

    printf("\nMSB pattern search across %d chunks of %d bytes (file size %ld bytes):\n",
           NUM_CHUNKS, N, filesize);

    double accuracy_sum = 0.0;
    for (int k = 0; k < NUM_CHUNKS; k++) {
        long offset = HEADER_SKIP + k * stride;
        fseek(fc, offset, SEEK_SET);
        unsigned char chunk[N];
        size_t got = fread(chunk, 1, N, fc);
        if (got != N) break;

        uint32_t seed; uint8_t pattern; int matches;
        search_chunk(chunk, &seed, &pattern, &matches);

        double acc = 100.0 * matches / 64.0;
        accuracy_sum += acc;

        printf("chunk %2d @ offset %10ld: seed=%7u (0x%06X) pattern=", k, offset, seed, seed);
        for (int b = 7; b >= 0; b--) putchar((pattern >> b) & 1 ? '1' : '0');
        printf(" accuracy=%2d/64 (%.1f%%)\n", matches, acc);
    }
    fclose(fc);

    printf("Average accuracy across %d chunks: %.1f%%\n", NUM_CHUNKS, accuracy_sum / NUM_CHUNKS);

    return 0;
}
