#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

/* ----------------------------------------------------------------
 * Config
 * ---------------------------------------------------------------- */
#define ALPHA       4           /* Dirichlet pseudocount per symbol  */
#define ALPHABET    256         /* byte alphabet                     */
#define INPUT_PATH  "C:/Users/lukac/Documents/compressor/output.bin"
#define OUTPUT_PATH "C:/Users/lukac/Documents/compressor/compressed.bin"

/* ----------------------------------------------------------------
 * 32-bit arithmetic coding interval landmarks
 * ---------------------------------------------------------------- */
#define AC_MAX  0xFFFFFFFFUL
#define AC_HALF 0x80000000UL
#define AC_QTR  0x40000000UL
#define AC_3QTR 0xC0000000UL

/* ================================================================
 * Bit writer
 * ================================================================ */
typedef struct {
    uint8_t *buf;
    size_t   pos;
    uint8_t  acc;
    int      bits;   /* how many bits are filled in acc (0-7) */
} BitWriter;

static void bw_init(BitWriter *w, uint8_t *buf) {
    w->buf = buf; w->pos = 0; w->acc = 0; w->bits = 0;
}

static void bw_put(BitWriter *w, int b) {
    w->acc = (w->acc << 1) | (b & 1);
    if (++w->bits == 8) {
        w->buf[w->pos++] = w->acc;
        w->acc = 0; w->bits = 0;
    }
}

/* Flush any partial byte (zero-padded) and return total bytes written */
static size_t bw_close(BitWriter *w) {
    if (w->bits)
        w->buf[w->pos++] = w->acc << (8 - w->bits);
    return w->pos;
}

/* ================================================================
 * Adaptive frequency model
 *
 * Initialised with ALPHA counts per symbol (= uniform Dirichlet
 * prior).  Both encoder and decoder use identical init so nothing
 * needs to be stored in the file.
 * ================================================================ */
typedef struct {
    uint32_t f[ALPHABET];
    uint32_t total;
} Model;

static void model_init(Model *m) {
    for (int i = 0; i < ALPHABET; i++) m->f[i] = ALPHA;
    m->total = ALPHA * ALPHABET;   /* 4 × 256 = 1024 */
}

/* Cumulative frequency strictly below symbol sym */
static uint32_t cum_below(const Model *m, int sym) {
    uint32_t c = 0;
    for (int i = 0; i < sym; i++) c += m->f[i];
    return c;
}

/* ================================================================
 * Arithmetic encoder
 * ================================================================ */
typedef struct {
    uint32_t  lo, hi;
    long      pend;   /* E3 carry counter */
    BitWriter bw;
} Enc;

static void enc_init(Enc *e, uint8_t *buf) {
    e->lo = 0; e->hi = AC_MAX; e->pend = 0;
    bw_init(&e->bw, buf);
}

/*
 * Emit one resolved bit plus any pending carry bits.
 * E3 scaling defers output until the top bit is certain.
 * When it finally resolves as 0, all the deferred bits are 1s
 * (and vice-versa) — that is the classic E3 carry trick.
 */
static void emit(Enc *e, int b) {
    bw_put(&e->bw, b);
    for (; e->pend > 0; e->pend--)
        bw_put(&e->bw, !b);
}

static void enc_sym(Enc *e, Model *m, int sym) {
    uint32_t cl = cum_below(m, sym);
    uint32_t ch = cl + m->f[sym];

    /* Narrow the coding interval proportionally to symbol probability */
    uint64_t r = (uint64_t)(e->hi - e->lo) + 1;
    e->hi = e->lo + (uint32_t)((r * ch) / m->total) - 1;
    e->lo = e->lo + (uint32_t)((r * cl) / m->total);

    /*
     * Renormalise: output bits whose value is now certain and
     * rescale the interval to maintain precision.
     *
     *  E1 — entire interval in lower half  → emit 0
     *  E2 — entire interval in upper half  → emit 1
     *  E3 — interval straddles the midpoint but fits in middle
     *        half → defer one carry bit, centre-scale
     */
    for (;;) {
        if (e->hi < AC_HALF) {
            emit(e, 0);
            e->lo =  e->lo        << 1;
            e->hi = (e->hi << 1) | 1;
        } else if (e->lo >= AC_HALF) {
            emit(e, 1);
            e->lo = (e->lo - AC_HALF) << 1;
            e->hi = ((e->hi - AC_HALF) << 1) | 1;
        } else if (e->lo >= AC_QTR && e->hi < AC_3QTR) {
            e->pend++;
            e->lo = (e->lo - AC_QTR) << 1;
            e->hi = ((e->hi - AC_QTR) << 1) | 1;
        } else {
            break;
        }
    }

    m->f[sym]++;
    m->total++;
}

/* Flush the final interval value and return compressed byte count */
static size_t enc_finish(Enc *e) {
    e->pend++;
    emit(e, (e->lo >= AC_QTR) ? 1 : 0);
    return bw_close(&e->bw);
}

/* ================================================================
 * Top-level compress:  byte array  →  file
 *
 * File format (little-endian):
 *   [uint16_t size][data]
 *   size = 0  → passthrough: data is the raw 4096-byte block
 *   size = N  → N bytes of compressed data follow
 * ================================================================ */
int compress(const uint8_t *in, size_t n, const char *path) {
    /* Worst-case output is larger than input for truly random data.
     * 2*n is a safe upper bound for any input. */
    uint8_t *buf = malloc(2 * n + 64);
    if (!buf) return -1;

    Enc   enc; enc_init(&enc, buf);
    Model mdl; model_init(&mdl);

    for (size_t i = 0; i < n; i++)
        enc_sym(&enc, &mdl, in[i]);

    size_t clen = enc_finish(&enc);

    FILE *f = fopen(path, "wb");
    if (!f) { free(buf); return -1; }

    if (clen >= n) {
        /* Data didn't compress — store raw with size = 0 flag */
        uint16_t flag = 0;
        fwrite(&flag, 2, 1, f);
        fwrite(in, 1, n, f);
        printf("Passthrough (incompressible): stored %zu raw bytes\n", n);
    } else {
        uint16_t sz = (uint16_t)clen;
        fwrite(&sz,  2, 1, f);
        fwrite(buf, 1, clen, f);
        printf("Input:      %zu bytes\n", n);
        printf("Compressed: %zu bytes (header included: %zu)\n", clen, clen + 2);
        printf("Saved:      %ld bytes  (%.4f bits/byte)\n",
               (long)n - (long)(clen + 2),
               (double)(clen + 2) * 8.0 / (double)n);
    }

    fclose(f);
    free(buf);
    return 0;
}

int main(void) {
    FILE *fin = fopen(INPUT_PATH, "rb");
    if (!fin) {
        fprintf(stderr, "Cannot open input: %s\n", INPUT_PATH);
        return 1;
    }

    fseek(fin, 0, SEEK_END);
    long fsize = ftell(fin);
    rewind(fin);

    if (fsize <= 0) {
        fprintf(stderr, "Empty or unreadable file\n");
        fclose(fin);
        return 1;
    }

    uint8_t *data = malloc((size_t)fsize);
    if (!data) {
        fprintf(stderr, "Out of memory\n");
        fclose(fin);
        return 1;
    }

    size_t nread = fread(data, 1, (size_t)fsize, fin);
    fclose(fin);

    if (nread != (size_t)fsize) {
        fprintf(stderr, "Read error\n");
        free(data);
        return 1;
    }

    int ret = compress(data, nread, OUTPUT_PATH);
    free(data);
    return ret == 0 ? 0 : 1;
}
