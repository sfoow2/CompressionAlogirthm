#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static double shannon_entropy(const unsigned char *buf, int n) {
    int freq[256] = {0};
    for (int i = 0; i < n; i++) freq[buf[i]]++;
    double h = 0.0;
    for (int i = 0; i < 256; i++) {
        if (freq[i] == 0) continue;
        double p = (double)freq[i] / n;
        h -= p * log2(p);
    }
    return h;
}

#define INPUT_PATH  "C:/Users/lukac/Documents/compressor/compressed.bin"
#define OUTPUT_PATH "C:/Users/lukac/Documents/compressor/aftercheat.bin"
#define BITS_PATH   "C:/Users/lukac/Documents/compressor/bitstream.bin"

#define DATA_SIZE 4096

int main(void) {
    unsigned char data[DATA_SIZE];

    /* read input */
    FILE *fin = fopen(INPUT_PATH, "rb");
    if (!fin) { perror("fopen input"); return 1; }
    if (fread(data, 1, DATA_SIZE, fin) != DATA_SIZE) {
        fprintf(stderr, "input must be exactly 4096 bytes\n");
        fclose(fin);
        return 1;
    }
    fclose(fin);

    printf("entropy before: %.6f bits/byte\n", shannon_entropy(data, DATA_SIZE));

    /* build bitstream: 4096 bits = 512 bytes, one bit per element */
    unsigned char bits[DATA_SIZE / 8];
    memset(bits, 0, sizeof(bits));

    for (int i = 0; i < DATA_SIZE; i++) {
        if (data[i] < 128) {
            bits[i / 8] |= (1 << (7 - (i % 8)));  /* set bit to 1 */
            data[i] += 128;
        }
        /* else bit stays 0, data unchanged */
    }

    printf("entropy after:  %.6f bits/byte\n", shannon_entropy(data, DATA_SIZE));

    /* write modified data */
    FILE *fout = fopen(OUTPUT_PATH, "wb");
    if (!fout) { perror("fopen output"); return 1; }
    fwrite(data, 1, DATA_SIZE, fout);
    fclose(fout);

    /* write bitstream */
    FILE *fbits = fopen(BITS_PATH, "wb");
    if (!fbits) { perror("fopen bitstream"); return 1; }
    fwrite(bits, 1, sizeof(bits), fbits);
    fclose(fbits);

    printf("done: output.bin (%d bytes), bitstream.bin (%zu bytes)\n",
           DATA_SIZE, sizeof(bits));
    return 0;
}
