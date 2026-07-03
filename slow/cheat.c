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

#define INPUT_PATH   "C:/Users/lukac/Documents/compressor/compressedsave.bin"
#define OUT_DIR      "C:/Users/lukac/Documents/compressor"
#define NUM_PASSES   7
#define DATA_SIZE    4096
#define BITS_SIZE    (DATA_SIZE / 8)

int main(void) {
    unsigned char data[DATA_SIZE];
    unsigned char bits_all[BITS_SIZE * NUM_PASSES]; /* [layer1][layer2]...[layer7] */
    char path[256];

    FILE *fin = fopen(INPUT_PATH, "rb");
    if (!fin) { perror("fopen input"); return 1; }
    if (fread(data, 1, DATA_SIZE, fin) != DATA_SIZE) {
        fprintf(stderr, "input must be exactly 4096 bytes\n");
        fclose(fin);
        return 1;
    }
    fclose(fin);

    printf("entropy before: %.6f bits/byte\n", shannon_entropy(data, DATA_SIZE));

    /* pass p (1..7): add = 128>>(p-1), threshold = 256 - add
     * ranges: [128,255] -> [192,255] -> [224,255] -> ... -> [254,255]
     * each layer's bitmask is appended into bits_all so the final file
     * is one continuous bitstream: [layer1][layer2][layer3]...[layer7] */
    for (int p = 1; p <= NUM_PASSES; p++) {
        int add       = 128 >> (p - 1);
        int threshold = 256 - add;
        unsigned char *bits = bits_all + (p - 1) * BITS_SIZE;

        memset(bits, 0, BITS_SIZE);
        for (int i = 0; i < DATA_SIZE; i++) {
            if (data[i] < threshold) {
                bits[i / 8] |= (1 << (7 - (i % 8)));
                data[i] += add;
            }
        }

        printf("pass%d entropy after: %.6f bits/byte  (add=%d threshold=%d)\n",
               p, shannon_entropy(data, DATA_SIZE), add, threshold);
    }

    snprintf(path, sizeof(path), "%s/aftercheat7.bin", OUT_DIR);
    FILE *fout = fopen(path, "wb");
    if (!fout) { perror("fopen output"); return 1; }
    fwrite(data, 1, DATA_SIZE, fout);
    fclose(fout);

    snprintf(path, sizeof(path), "%s/bitstream_all.bin", OUT_DIR);
    FILE *fbits = fopen(path, "wb");
    if (!fbits) { perror("fopen bitstream"); return 1; }
    fwrite(bits_all, 1, sizeof(bits_all), fbits);
    fclose(fbits);

    printf("done: aftercheat7.bin + bitstream_all.bin (%zu bytes, %d layers)\n",
           sizeof(bits_all), NUM_PASSES);

    return 0;
}
