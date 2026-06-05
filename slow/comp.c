#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <math.h>

/* Hardcoded input - change to test different files. File is expected to be
 * the stride-transformed output (data.bin), ~4096 bytes, entropy ~7.85. */
#define INPUT_FILE "data.bin"

typedef uint32_t u32;
typedef uint64_t u64;
typedef uint8_t  u8;

/* ============================================================
 *  SUBBOTIN CARRYLESS RANGE CODER
 *  Encoder and decoder use IDENTICAL arithmetic (range/=total
 *  first on both sides) so they never drift out of sync.
 * ============================================================ */

#define TOP  (1u << 24)
#define BOT  (1u << 16)

typedef struct {
    u32 low, range, code;
    u8 *buf;
    int pos, cap;
} RC;

/* ---- encoder ---- */
static void enc_init(RC *rc, u8 *buf, int cap) {
    rc->low = 0; rc->range = 0xFFFFFFFFu; rc->code = 0;
    rc->buf = buf; rc->pos = 0; rc->cap = cap;
}

static void enc_renorm(RC *rc) {
    while ((rc->low ^ (rc->low + rc->range)) < TOP ||
           (rc->range < BOT && ((rc->range = (0u - rc->low) & (BOT - 1)), 1))) {
        if (rc->pos < rc->cap) rc->buf[rc->pos] = (u8)(rc->low >> 24);
        rc->pos++;
        rc->low <<= 8;
        rc->range <<= 8;
    }
}

static void enc_encode(RC *rc, u32 cumfreq, u32 freq, u32 totfreq) {
    rc->range /= totfreq;
    rc->low   += cumfreq * rc->range;
    rc->range *= freq;
    enc_renorm(rc);
}

static void enc_flush(RC *rc) {
    for (int i = 0; i < 4; i++) {
        if (rc->pos < rc->cap) rc->buf[rc->pos] = (u8)(rc->low >> 24);
        rc->pos++;
        rc->low <<= 8;
    }
}

/* ---- decoder ---- */
static void dec_init(RC *rc, u8 *buf, int cap) {
    rc->low = 0; rc->range = 0xFFFFFFFFu; rc->code = 0;
    rc->buf = buf; rc->pos = 0; rc->cap = cap;
    for (int i = 0; i < 4; i++) {
        rc->code = (rc->code << 8) | (rc->pos < rc->cap ? rc->buf[rc->pos] : 0);
        rc->pos++;
    }
}

static void dec_renorm(RC *rc) {
    while ((rc->low ^ (rc->low + rc->range)) < TOP ||
           (rc->range < BOT && ((rc->range = (0u - rc->low) & (BOT - 1)), 1))) {
        rc->code = (rc->code << 8) | (rc->pos < rc->cap ? rc->buf[rc->pos] : 0);
        rc->pos++;
        rc->low <<= 8;
        rc->range <<= 8;
    }
}

static u32 dec_getfreq(RC *rc, u32 totfreq) {
    rc->range /= totfreq;
    return (rc->code - rc->low) / rc->range;
}

static void dec_decode(RC *rc, u32 cumfreq, u32 freq) {
    rc->low   += cumfreq * rc->range;
    rc->range *= freq;
    dec_renorm(rc);
}

/* ============================================================
 *  ORDER-0 ADAPTIVE MODEL  (256 cells - learnable on 4 KB)
 * ============================================================ */

static u32 m0[256];
#define M0_INC    1                /* pseudocount increment (1 = best for ~4KB random-ish data) */
#define M0_LIMIT  (BOT - 1)       /* keep total < BOT for coder precision */

static void m0_init(void) {
    for (int i = 0; i < 256; i++) m0[i] = 1;   /* pseudocount 1 */
}

static void m0_bump(int s) {
    m0[s] += M0_INC;
    /* rescale if total would approach BOT, preserving nonzero counts */
    u32 tot = 0; for (int j = 0; j < 256; j++) tot += m0[j];
    if (tot >= M0_LIMIT) {
        for (int j = 0; j < 256; j++) m0[j] = (m0[j] >> 1) | 1;
    }
}

static int o0_compress(const u8 *in, int n, u8 *out, int cap) {
    RC rc; enc_init(&rc, out, cap);
    m0_init();
    for (int i = 0; i < n; i++) {
        u8 s = in[i];
        u32 cum = 0; for (int j = 0; j < s; j++) cum += m0[j];
        u32 tot = 0; for (int j = 0; j < 256; j++) tot += m0[j];
        enc_encode(&rc, cum, m0[s], tot);
        m0_bump(s);
    }
    enc_flush(&rc);
    return rc.pos;
}

static int o0_decompress(const u8 *in, int n, u8 *out, int outlen) {
    RC rc; dec_init(&rc, (u8 *)in, n);
    m0_init();
    for (int i = 0; i < outlen; i++) {
        u32 tot = 0; for (int j = 0; j < 256; j++) tot += m0[j];
        u32 dv  = dec_getfreq(&rc, tot);
        u32 cum = 0; int s = 0;
        for (s = 0; s < 256; s++) {
            if (cum + m0[s] > dv) break;
            cum += m0[s];
        }
        dec_decode(&rc, cum, m0[s]);
        out[i] = (u8)s;
        m0_bump(s);
    }
    return outlen;
}

/* ============================================================ */

static void run_test(const char *name,
                     int (*comp)(const u8 *, int, u8 *, int),
                     int (*decomp)(const u8 *, int, u8 *, int),
                     const u8 *input, int n) {
    u8 *out = malloc(n + 1024);
    u8 *dec = malloc(n);

    clock_t t0 = clock();
    int csize = comp(input, n, out, n + 1024);
    clock_t t1 = clock();
    decomp(out, csize, dec, n);
    clock_t t2 = clock();

    int match = (memcmp(input, dec, n) == 0);
    int gain  = (n - csize) * 8;

    printf("--- %s ---\n", name);
    printf("  compressed : %d -> %d bytes\n", n, csize);
    printf("  gain       : %+d bits  (%.2f bits/byte)\n",
           gain, (double)csize * 8.0 / n);
    printf("  enc/dec    : %.3f s / %.3f s\n",
           (t1 - t0) / (double)CLOCKS_PER_SEC,
           (t2 - t1) / (double)CLOCKS_PER_SEC);
    if (match) {
        printf("  roundtrip  : OK (decoded == original)\n");
    } else {
        printf("  roundtrip  : FAILED\n");
        for (int i = 0; i < n; i++)
            if (input[i] != dec[i]) {
                printf("    first mismatch at %d: %02x vs %02x\n",
                       i, input[i], dec[i]);
                break;
            }
    }
    printf("\n");

    free(out);
    free(dec);
}

int main(void) {
    FILE *f = fopen(INPUT_FILE, "rb");
    if (!f) { printf("Cannot open %s\n", INPUT_FILE); return 1; }
    fseek(f, 0, SEEK_END); int n = ftell(f); fseek(f, 0, SEEK_SET);
    u8 *input = malloc(n);
    if (fread(input, 1, n, f) != (size_t)n) { printf("read error\n"); return 1; }
    fclose(f);

    /* order-0 entropy of the input, for reference */
    int hist[256] = {0};
    for (int i = 0; i < n; i++) hist[input[i]]++;
    double H = 0;
    for (int i = 0; i < 256; i++) if (hist[i]) {
        double p = (double)hist[i] / n;
        H -= p * (log(p) / log(2.0));
    }
    printf("Input: %s  (%d bytes)\n", INPUT_FILE, n);
    printf("Order-0 entropy: %.6f bits/byte  ->  ideal floor %d bytes (%.0f redundant bits)\n\n",
           H, (int)(H * n / 8.0 + 0.999), (8.0 - H) * n);

    run_test("ORDER-0 adaptive", o0_compress, o0_decompress, input, n);

    free(input);
    return 0;
}
