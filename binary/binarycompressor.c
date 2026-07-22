#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define RANS_L (1u << 23)

static void enc_put(uint32_t *r, uint8_t **pptr, uint32_t start, uint32_t freq, uint32_t scale_bits) {
    uint32_t x = *r;
    uint32_t x_max = ((RANS_L >> scale_bits) << 8) * freq;
    if (x >= x_max) {
        *pptr -= 1;
        **pptr = (uint8_t)(x & 0xff);
        x >>= 8;
    }
    *r = ((x / freq) << scale_bits) + (x % freq) + start;
}

static uint32_t dec_get(uint32_t *r, uint32_t scale_bits) {
    return *r & ((1u << scale_bits) - 1);
}

static void dec_advance(uint32_t *r, uint8_t **pptr, uint32_t start, uint32_t freq, uint32_t scale_bits) {
    uint32_t mask = (1u << scale_bits) - 1;
    uint32_t x = *r;
    x = freq * (x >> scale_bits) + (x & mask) - start;
    while (x < RANS_L) {
        x = (x << 8) | **pptr;
        *pptr += 1;
    }
    *r = x;
}

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file> [byte_limit]\n", argv[0]); return 1; }
    const char *path = argv[1];
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    fseek(f, 0, SEEK_END);
    long fsize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (argc > 2) {
        long lim = atol(argv[2]);
        if (lim < fsize) fsize = lim;
    }
    uint8_t *filebuf = malloc(fsize);
    fread(filebuf, 1, fsize, f);
    fclose(f);

    long nbits = fsize * 8;
    uint8_t *bits = malloc(nbits);
    for (long i = 0; i < fsize; i++)
        for (int b = 0; b < 8; b++)
            bits[i*8+b] = (filebuf[i] >> (7-b)) & 1;

    long ones = 0;
    for (long i = 0; i < nbits; i++) ones += bits[i];
    long zeros = nbits - ones;

    uint32_t scale_bits = 16;
    uint32_t M = 1u << scale_bits;
    uint32_t freq1 = (uint32_t)((double)ones / (double)nbits * M + 0.5);
    if (freq1 < 1) freq1 = 1;
    if (freq1 > M - 1) freq1 = M - 1;
    uint32_t freq0 = M - freq1;
    uint32_t start0 = 0, start1 = freq0;

    size_t bufcap = (size_t)nbits + 1024;
    uint8_t *outbuf = malloc(bufcap);
    uint8_t *ptr = outbuf + bufcap;
    uint32_t state = RANS_L;
    for (long i = nbits - 1; i >= 0; i--) {
        uint32_t sym = bits[i];
        uint32_t start = sym ? start1 : start0;
        uint32_t freq  = sym ? freq1  : freq0;
        enc_put(&state, &ptr, start, freq, scale_bits);
    }
    ptr -= 4;
    ptr[0] = (uint8_t)(state >> 0);
    ptr[1] = (uint8_t)(state >> 8);
    ptr[2] = (uint8_t)(state >> 16);
    ptr[3] = (uint8_t)(state >> 24);

    size_t compressed_size = (size_t)((outbuf + bufcap) - ptr);

    uint8_t *dptr = ptr;
    uint32_t dstate;
    dstate  = dptr[0];
    dstate |= (uint32_t)dptr[1] << 8;
    dstate |= (uint32_t)dptr[2] << 16;
    dstate |= (uint32_t)dptr[3] << 24;
    dptr += 4;

    int ok = 1;
    long first_mismatch = -1;
    for (long i = 0; i < nbits; i++) {
        uint32_t slot = dec_get(&dstate, scale_bits);
        uint32_t sym = (slot >= start1) ? 1u : 0u;
        uint32_t start = sym ? start1 : start0;
        uint32_t freq  = sym ? freq1  : freq0;
        if (sym != bits[i] && ok) { ok = 0; first_mismatch = i; }
        dec_advance(&dstate, &dptr, start, freq, scale_bits);
    }

    double p1 = (double)ones / (double)nbits, p0 = 1.0 - p1;
    double bit_entropy = 0.0;
    if (p0 > 0) bit_entropy -= p0 * log2(p0);
    if (p1 > 0) bit_entropy -= p1 * log2(p1);

    double table_bits = ceil(log2((double)M));
    double total_with_table = (double)compressed_size + table_bits / 8.0;

    printf("file: %s (using first %ld bytes)\n", path, fsize);
    printf("raw size: %ld bytes (%ld bits)\n", fsize, nbits);
    printf("ones: %ld (%.4f%%)  zeros: %ld (%.4f%%)\n", ones, 100.0 * p1, zeros, 100.0 * p0);
    printf("quantized freq (scale_bits=%u, M=%u): freq1=%u freq0=%u\n", scale_bits, M, freq1, freq0);
    printf("round-trip check: %s%s\n", ok ? "OK, bit-exact" : "FAILED",
           ok ? "" : "");
    if (!ok) printf("  first mismatch at bit %ld\n", first_mismatch);
    printf("rANS compressed payload: %zu bytes\n", compressed_size);
    printf("theoretical entropy-bound size: %.2f bytes (%.6f bits/bit)\n", nbits * bit_entropy / 8.0, bit_entropy);
    printf("table cost: %.0f bits (%.2f bytes)\n", table_bits, table_bits / 8.0);
    printf("total (table + payload): %.2f bytes = %.4f%% of raw %ld bytes\n",
           total_with_table, 100.0 * total_with_table / fsize, fsize);

    return 0;
}
