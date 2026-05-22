/*
 * LCG Chain Compressor - Core Test v3
 *
 * Pipeline: LCG (transition structure) → nibble Huffman (frequency bias)
 *   stage 0 = raw  |  stage 1 = LCG only  |  stage 2 = LCG → Huffman
 * Compressor tries all three and keeps the smallest.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <limits.h>

#define BLOCK_SIZE 32

static const uint8_t VALID_A[4] = {1, 5, 9, 13};
static const uint8_t VALID_C[8] = {1, 3, 5, 7, 9, 11, 13, 15};

/* OT[ai][ci][from][to]      = steps (1-16) to reach 'to' from 'from'   */
/* ADV[ai][ci][from][steps-1] = state after 'steps' steps from 'from'   */
static uint8_t OT[4][8][16][16];
static uint8_t ADV[4][8][16][16];

static void build_tables(void) {
    for (int ai = 0; ai < 4; ai++)
        for (int ci = 0; ci < 8; ci++) {
            uint8_t a = VALID_A[ai], c = VALID_C[ci];
            for (int from = 0; from < 16; from++) {
                uint8_t s = (uint8_t)from;
                for (int step = 0; step < 16; step++) {
                    s = (a * s + c) & 0xF;
                    OT[ai][ci][from][s]   = (uint8_t)(step + 1);
                    ADV[ai][ci][from][step] = s;
                }
            }
        }
}

typedef enum { OP_NONE=0, OP_XOR=1, OP_ADD=2, OP_SUB=3 } TransformOp;

/* ------------------------------------------------------------------ */
/* Bit writer / reader                                                  */
/* ------------------------------------------------------------------ */

typedef struct { uint8_t *buf; int byte_pos; int bit_pos; } BitWriter;
typedef struct { const uint8_t *buf; int byte_pos; int bit_pos; } BitReader;

static void write_bit(BitWriter *bw, int bit) {
    if (bw->bit_pos == 0) bw->buf[bw->byte_pos] = 0;
    bw->buf[bw->byte_pos] |= (bit & 1) << (7 - bw->bit_pos);
    if (++bw->bit_pos == 8) { bw->bit_pos = 0; bw->byte_pos++; }
}

static void write_bits(BitWriter *bw, uint32_t value, int n) {
    for (int i = n - 1; i >= 0; i--)
        write_bit(bw, (value >> i) & 1);
}

static int read_bit(BitReader *br) {
    int b = (br->buf[br->byte_pos] >> (7 - br->bit_pos)) & 1;
    if (++br->bit_pos == 8) { br->bit_pos = 0; br->byte_pos++; }
    return b;
}

static uint32_t read_bits(BitReader *br, int n) {
    uint32_t v = 0;
    for (int i = 0; i < n; i++) v = (v << 1) | read_bit(br);
    return v;
}

static int bw_bits(const BitWriter *bw) {
    return bw->byte_pos * 8 + bw->bit_pos;
}

/* ------------------------------------------------------------------ */
/* LCG core                                                             */
/* ------------------------------------------------------------------ */

static inline uint8_t lcg_step(uint8_t state, uint8_t a, uint8_t c) {
    return (a * state + c) & 0xF;
}


/* ------------------------------------------------------------------ */
/* VLC  (tweaked vs v1)                                                 */
/*                                                                      */
/*  offset  code          bits                                          */
/*  1       0             1                                             */
/*  2       10            2                                             */
/*  3       110           3                                             */
/*  4       1110          4                                             */
/*  5-8     11110 xx      7    (saves 1 bit over the old 8-bit escape) */
/*  9-16    11111 xxx     8                                             */
/* ------------------------------------------------------------------ */

static int vlc_bit_length(int offset) {
    if (offset <= 4) return offset;
    if (offset <= 8) return 7;
    return 8;
}

static void write_vlc(BitWriter *bw, int offset) {
    if      (offset == 1) { write_bits(bw, 0x0,  1); }
    else if (offset == 2) { write_bits(bw, 0x2,  2); }
    else if (offset == 3) { write_bits(bw, 0x6,  3); }
    else if (offset == 4) { write_bits(bw, 0xE,  4); }
    else if (offset <= 8) { write_bits(bw, 0x1E, 5); write_bits(bw, offset - 5, 2); }
    else                  { write_bits(bw, 0x1F, 5); write_bits(bw, offset - 9, 3); }
}

static int read_vlc(BitReader *br) {
    if (!read_bit(br)) return 1;
    if (!read_bit(br)) return 2;
    if (!read_bit(br)) return 3;
    if (!read_bit(br)) return 4;
    if (!read_bit(br)) return (int)read_bits(br, 2) + 5;   /* 11110xx -> 5..8  */
    return               (int)read_bits(br, 3) + 9;         /* 11111xxx -> 9..16 */
}

/* ------------------------------------------------------------------ */
/* Transforms                                                           */
/* ------------------------------------------------------------------ */

static void apply_transform(const uint8_t *in, int n,
                             TransformOp op, uint8_t K, uint8_t *out) {
    for (int i = 0; i < n; i++) {
        switch (op) {
            case OP_XOR: out[i] =  in[i] ^ K;              break;
            case OP_ADD: out[i] = (in[i] + K)      & 0xF;  break;
            case OP_SUB: out[i] = (in[i] - K + 16) & 0xF;  break;
            default:     out[i] =  in[i];                   break;
        }
    }
}

static void invert_transform(uint8_t *data, int n, TransformOp op, uint8_t K) {
    for (int i = 0; i < n; i++) {
        switch (op) {
            case OP_XOR: data[i] ^= K;                          break;
            case OP_ADD: data[i] = (data[i] - K + 16) & 0xF;   break;
            case OP_SUB: data[i] = (data[i] + K)      & 0xF;   break;
            default: break;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Scoring and tuning                                                   */
/* ------------------------------------------------------------------ */

/* Returns seed(4) + VLC bits for data[1..n-1] with given (ai,ci). */
static int score_block(const uint8_t *data, int n, int ai, int ci) {
    int bits = 4;   /* seed */
    uint8_t state = data[0];
    for (int i = 1; i < n; i++) {
        bits += vlc_bit_length(OT[ai][ci][state][data[i]]);
        state = data[i];
    }
    return bits;
}

static void tune_block(const uint8_t *data, int n, int *out_ai, int *out_ci) {
    int best = INT_MAX;
    *out_ai = 0; *out_ci = 0;
    for (int ai = 0; ai < 4; ai++) {
        for (int ci = 0; ci < 8; ci++) {
            int s = score_block(data, n, ai, ci);
            if (s < best) { best = s; *out_ai = ai; *out_ci = ci; }
        }
    }
}

/* ------------------------------------------------------------------ */
/* Block compress / decompress                                          */
/*                                                                      */
/* Format:                                                              */
/*   Raw:             [0] [n*4 raw nibble bits]                         */
/*   Comp no-xform:   [1][0] [2b a_idx][3b c_idx][4b seed] [VLC...]    */
/*   Comp w/-xform:   [1][1] [2b op][4b K] [2b a_idx][3b c_idx]        */
/*                    [4b seed] [VLC...]                                */
/*                                                                      */
/* Header overhead:                                                     */
/*   no-xform:  1+1+2+3 = 7 bits  (score_block already has the 4b seed)*/
/*   w/-xform:  1+1+2+4+2+3 = 13 bits                                  */
/*   raw:       1 bit                                                   */
/* ------------------------------------------------------------------ */

static int compress_block(BitWriter *bw, const uint8_t *data, int n) {
    uint8_t tmp[BLOCK_SIZE];
    int raw_bits = 1 + n * 4;

    /* No-transform baseline */
    int best_ai, best_ci;
    tune_block(data, n, &best_ai, &best_ci);
    int best_bits = 7 + score_block(data, n, best_ai, best_ci);
    TransformOp best_op = OP_NONE;
    uint8_t best_K = 0;

    /* Try all (op, K) combinations */
    for (int oi = 1; oi <= 3; oi++) {
        for (uint8_t K = 1; K < 16; K++) {
            apply_transform(data, n, (TransformOp)oi, K, tmp);
            int ai, ci;
            tune_block(tmp, n, &ai, &ci);
            int bits = 13 + score_block(tmp, n, ai, ci);
            if (bits < best_bits) {
                best_bits = bits;
                best_op   = (TransformOp)oi;
                best_K    = K;
                best_ai   = ai;
                best_ci   = ci;
            }
        }
    }

    /* Fall back to raw if compression doesn't win */
    if (best_bits >= raw_bits) {
        write_bits(bw, 0, 1);
        for (int i = 0; i < n; i++) write_bits(bw, data[i], 4);
        return raw_bits;
    }

    /* Source for encoding: original or transformed */
    const uint8_t *src = data;
    if (best_op != OP_NONE) {
        apply_transform(data, n, best_op, best_K, tmp);
        src = tmp;
    }

    /* Write header */
    write_bits(bw, 1, 1);   /* comp_flag */
    if (best_op == OP_NONE) {
        write_bits(bw, 0, 1);   /* transform_flag = 0 */
    } else {
        write_bits(bw, 1, 1);   /* transform_flag = 1 */
        write_bits(bw, (uint32_t)best_op, 2);
        write_bits(bw, best_K, 4);
    }
    write_bits(bw, best_ai, 2);
    write_bits(bw, best_ci, 3);
    write_bits(bw, src[0], 4);   /* seed */

    /* VLC offsets */
    uint8_t state = src[0];
    for (int i = 1; i < n; i++) {
        write_vlc(bw, OT[best_ai][best_ci][state][src[i]]);
        state = src[i];
    }

    return best_bits;
}

static void decompress_block(BitReader *br, int n, uint8_t *out) {
    int comp_flag = read_bit(br);
    if (!comp_flag) {
        for (int i = 0; i < n; i++) out[i] = (uint8_t)read_bits(br, 4);
        return;
    }

    TransformOp op = OP_NONE;
    uint8_t K = 0;
    if (read_bit(br)) {   /* transform_flag */
        op = (TransformOp)read_bits(br, 2);
        K  = (uint8_t)read_bits(br, 4);
    }

    int ai = (int)read_bits(br, 2);
    int ci = (int)read_bits(br, 3);
    out[0] = (uint8_t)read_bits(br, 4);

    uint8_t state = out[0];
    for (int i = 1; i < n; i++) {
        int offset = read_vlc(br);
        state = ADV[ai][ci][state][offset - 1];
        out[i] = state;
    }

    if (op != OP_NONE) invert_transform(out, n, op, K);
}

/* ------------------------------------------------------------------ */
/* Full pass: compress / decompress a nibble array                      */
/* ------------------------------------------------------------------ */

static int compress_nibbles(const uint8_t *nibs, int n, uint8_t *out) {
    BitWriter bw = { .buf = out, .byte_pos = 0, .bit_pos = 0 };
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        int bn = (i + BLOCK_SIZE <= n) ? BLOCK_SIZE : (n - i);
        compress_block(&bw, nibs + i, bn);
    }
    return (bw_bits(&bw) + 7) / 8;
}

static void decompress_nibbles(const uint8_t *in, int n_nibs, uint8_t *out) {
    BitReader br = { .buf = in, .byte_pos = 0, .bit_pos = 0 };
    for (int i = 0; i < n_nibs; i += BLOCK_SIZE) {
        int bn = (i + BLOCK_SIZE <= n_nibs) ? BLOCK_SIZE : (n_nibs - i);
        decompress_block(&br, bn, out + i);
    }
}

/* ------------------------------------------------------------------ */
/* Byte <-> nibble helpers                                              */
/* ------------------------------------------------------------------ */

static void bytes_to_nibs(const uint8_t *bytes, int n, uint8_t *nibs) {
    for (int i = 0; i < n; i++) {
        nibs[i*2]   = bytes[i] >> 4;
        nibs[i*2+1] = bytes[i] & 0xF;
    }
}

static void nibs_to_bytes(const uint8_t *nibs, int n_bytes, uint8_t *bytes) {
    for (int i = 0; i < n_bytes; i++)
        bytes[i] = (nibs[i*2] << 4) | nibs[i*2+1];
}

/* ------------------------------------------------------------------ */
/* Nibble Huffman  (stage 2: exploits frequency bias in LCG output)    */
/*                                                                      */
/* Treats every byte as two 4-bit nibbles → 16 symbols.                */
/* Header: 8 bytes = 16 code-lengths packed 4 bits each (0 = absent).  */
/* ------------------------------------------------------------------ */

static void huff_build(const int freq[16], int len[16]) {
    int f[31] = {0}, par[31];
    for (int i = 0; i < 31; i++) par[i] = -1;
    for (int i = 0; i < 16; i++) f[i] = freq[i];
    int n = 16;
    for (;;) {
        int m1 = -1, m2 = -1;
        for (int i = 0; i < n; i++) {
            if (par[i] >= 0 || (i < 16 && !f[i])) continue;
            if (m1 < 0 || f[i] < f[m1]) { m2 = m1; m1 = i; }
            else if (m2 < 0 || f[i] < f[m2]) { m2 = i; }
        }
        if (m2 < 0) break;
        f[n] = f[m1] + f[m2];
        par[m1] = par[m2] = n++;
    }
    for (int i = 0; i < 16; i++) {
        if (!freq[i]) { len[i] = 0; continue; }
        int d = 0, c = i;
        while (par[c] >= 0) { d++; c = par[c]; }
        len[i] = d ? d : 1;
    }
}

static void huff_codes(const int len[16], uint32_t code[16]) {
    int cnt[17] = {0};
    for (int i = 0; i < 16; i++) if (len[i]) cnt[len[i]]++;
    uint32_t nxt[17] = {0}, c = 0;
    for (int b = 1; b <= 16; b++) { nxt[b] = c; c = (c + cnt[b]) << 1; }
    for (int i = 0; i < 16; i++) code[i] = len[i] ? nxt[len[i]]++ : 0;
}

/* Returns compressed byte count (8-byte header + coded nibbles). */
static int huff_compress(const uint8_t *in, int n, uint8_t *out) {
    int freq[16] = {0};
    for (int i = 0; i < n; i++) { freq[in[i]>>4]++; freq[in[i]&0xF]++; }
    int len[16]; uint32_t code[16];
    huff_build(freq, len);
    huff_codes(len, code);
    for (int i = 0; i < 8; i++)
        out[i] = (uint8_t)(((len[i*2]&0xF)<<4)|(len[i*2+1]&0xF));
    BitWriter bw = { .buf=out+8, .byte_pos=0, .bit_pos=0 };
    for (int i = 0; i < n; i++) {
        int hi = in[i]>>4, lo = in[i]&0xF;
        write_bits(&bw, code[hi], len[hi]);
        write_bits(&bw, code[lo], len[lo]);
    }
    return 8 + (bw_bits(&bw)+7)/8;
}

static void huff_decompress(const uint8_t *in, int n_out, uint8_t *out) {
    int len[16]; uint32_t code[16];
    for (int i = 0; i < 8; i++) { len[i*2]=in[i]>>4; len[i*2+1]=in[i]&0xF; }
    huff_codes(len, code);
    BitReader br = { .buf=in+8, .byte_pos=0, .bit_pos=0 };
    for (int i = 0; i < n_out; i++) {
        uint8_t byte = 0;
        for (int nib = 0; nib < 2; nib++) {
            uint32_t bits = 0;
            int done = 0;
            for (int b = 1; b <= 16 && !done; b++) {
                bits = (bits<<1)|read_bit(&br);
                for (int s = 0; s < 16 && !done; s++) {
                    if (len[s]==b && code[s]==bits) {
                        byte = nib ? (byte|(uint8_t)s) : (uint8_t)(s<<4);
                        done = 1;
                    }
                }
            }
        }
        out[i] = byte;
    }
}

/* ------------------------------------------------------------------ */
/* LZSS  (stage 3/4: removes repetition — best for text/source code)   */
/*                                                                      */
/* Window: 4096 bytes (12-bit offset).  Min match: 3.  Max: 258.       */
/* Token: flag(1) + literal(8)  OR  flag(1) + offset(12) + length(8). */
/* ------------------------------------------------------------------ */
#define LZ_WIN 4096
#define LZ_MIN 3
#define LZ_MAX 258

static int lz_compress(const uint8_t *in, int n, uint8_t *out) {
    BitWriter bw = { .buf=out, .byte_pos=0, .bit_pos=0 };
    int i = 0;
    while (i < n) {
        int best_len = 0, best_off = 0;
        int start = (i > LZ_WIN) ? i - LZ_WIN : 0;
        int limit  = (n - i < LZ_MAX) ? n - i : LZ_MAX;
        for (int j = start; j < i; j++) {
            int len = 0;
            while (len < limit && in[j + len] == in[i + len]) len++;
            if (len > best_len) { best_len = len; best_off = i - j; }
        }
        if (best_len >= LZ_MIN) {
            write_bits(&bw, 1, 1);
            write_bits(&bw, (uint32_t)(best_off - 1), 12);
            write_bits(&bw, (uint32_t)(best_len - LZ_MIN), 8);
            i += best_len;
        } else {
            write_bits(&bw, 0, 1);
            write_bits(&bw, in[i], 8);
            i++;
        }
    }
    return (bw_bits(&bw) + 7) / 8;
}

static void lz_decompress(const uint8_t *in, int n_out, uint8_t *out) {
    BitReader br = { .buf=in, .byte_pos=0, .bit_pos=0 };
    int i = 0;
    while (i < n_out) {
        if (!read_bit(&br)) {
            out[i++] = (uint8_t)read_bits(&br, 8);
        } else {
            int off = (int)read_bits(&br, 12) + 1;
            int len = (int)read_bits(&br, 8)  + LZ_MIN;
            int src = i - off;
            for (int k = 0; k < len; k++) out[i + k] = out[src + k];
            i += len;
        }
    }
}

/* ------------------------------------------------------------------ */
/* Full pipeline                                                         */
/*                                                                      */
/* Tries five strategies, picks the smallest:                           */
/*   0 = raw                                                            */
/*   1 = LCG              (sequences, gradients, generated data)        */
/*   2 = LCG → Huffman    (same + exploits remaining frequency bias)    */
/*   3 = LZ77             (repetitive data, source code)                */
/*   4 = LZ77 → Huffman   (text, ASCII — best all-rounder)             */
/*                                                                      */
/* Stages 2 and 4 store intermediate size as uint16 at bytes [1..2].   */
/* ------------------------------------------------------------------ */

static int compress_full(const uint8_t *data, int n, uint8_t *out) {
    int n_nibs = n * 2;
    uint8_t *nibs = malloc(n_nibs);
    uint8_t *lcg  = malloc(n_nibs * 2 + 4);
    uint8_t *lh   = malloc(n_nibs * 2 + 16);
    uint8_t *lz   = malloc(n * 2 + 16);
    uint8_t *lzh  = malloc(n * 2 + 16);

    bytes_to_nibs(data, n, nibs);
    int lcg_sz  = compress_nibbles(nibs, n_nibs, lcg);
    int lh_sz   = huff_compress(lcg, lcg_sz, lh);
    int lz_sz   = lz_compress(data, n, lz);
    int lzh_sz  = huff_compress(lz, lz_sz, lzh);

    int best = n + 1, sel = 0;
    if (lcg_sz  + 1 < best) { best = lcg_sz  + 1; sel = 1; }
    if (lh_sz   + 3 < best) { best = lh_sz   + 3; sel = 2; }
    if (lz_sz   + 1 < best) { best = lz_sz   + 1; sel = 3; }
    if (lzh_sz  + 3 < best) { best = lzh_sz  + 3; sel = 4; }

    out[0] = (uint8_t)sel;
    switch (sel) {
        case 0: memcpy(out+1, data, n); break;
        case 1: memcpy(out+1, lcg, lcg_sz); break;
        case 2:
            out[1] = (uint8_t)(lcg_sz & 0xFF); out[2] = (uint8_t)(lcg_sz >> 8);
            memcpy(out+3, lh, lh_sz); break;
        case 3: memcpy(out+1, lz, lz_sz); break;
        case 4:
            out[1] = (uint8_t)(lz_sz & 0xFF); out[2] = (uint8_t)(lz_sz >> 8);
            memcpy(out+3, lzh, lzh_sz); break;
    }
    free(nibs); free(lcg); free(lh); free(lz); free(lzh);
    return best;
}

static void decompress_full(const uint8_t *in, int n_out, uint8_t *out) {
    int sel = in[0];
    if (sel == 0) {
        memcpy(out, in+1, n_out);
    } else if (sel == 1) {
        uint8_t *nibs = malloc(n_out*2);
        decompress_nibbles(in+1, n_out*2, nibs);
        nibs_to_bytes(nibs, n_out, out);
        free(nibs);
    } else if (sel == 2) {
        int lcg_sz = (int)in[1] | ((int)in[2]<<8);
        uint8_t *lcg  = malloc(lcg_sz);
        uint8_t *nibs = malloc(n_out*2);
        huff_decompress(in+3, lcg_sz, lcg);
        decompress_nibbles(lcg, n_out*2, nibs);
        nibs_to_bytes(nibs, n_out, out);
        free(lcg); free(nibs);
    } else if (sel == 3) {
        lz_decompress(in+1, n_out, out);
    } else {
        int lz_sz = (int)in[1] | ((int)in[2]<<8);
        uint8_t *lz = malloc(lz_sz);
        huff_decompress(in+3, lz_sz, lz);
        lz_decompress(lz, n_out, out);
        free(lz);
    }
}

/* ------------------------------------------------------------------ */
/* Test runner                                                           */
/* ------------------------------------------------------------------ */

static const char *stage_name[] = { "[raw    ]", "[lcg    ]", "[lcg+huf]", "[lz     ]", "[lz+huf ]" };

static void run_test(const char *name, const uint8_t *data, int n_bytes) {
    uint8_t *comp     = malloc(n_bytes * 4 + 16);
    uint8_t *dec_data = malloc(n_bytes);

    int comp_bytes = compress_full(data, n_bytes, comp);
    decompress_full(comp, n_bytes, dec_data);

    int ok = (memcmp(data, dec_data, n_bytes) == 0);
    float ratio = 100.0f * comp_bytes / n_bytes;
    printf("  %-40s  raw=%5d B  comp=%5d B  %5.1f%%  %s  %s\n",
           name, n_bytes, comp_bytes, ratio,
           ok ? "OK" : "FAIL", stage_name[comp[0]]);

    free(comp); free(dec_data);
}

/* ------------------------------------------------------------------ */
/* main                                                                 */
/* ------------------------------------------------------------------ */

int main(void) {
    build_tables();
    printf("=== LCG Chain Compressor v4: LCG + LZ77 + Huffman pipeline ===\n\n");

    /* ---- Baselines ---- */
    printf("Baselines (K-search not expected to activate):\n");

    {
        int n = 128;
        uint8_t *d = malloc(n);
        uint8_t v = 0;
        for (int i = 0; i < n; i++) {
            uint8_t hi = v; v = (v + 3) & 0xF;
            uint8_t lo = v; v = (v + 3) & 0xF;
            d[i] = (hi << 4) | lo;
        }
        run_test("arithmetic +3 mod 16", d, n);
        free(d);
    }

    {
        int n = 128;
        uint8_t *d = malloc(n);
        uint8_t st = 0;
        for (int i = 0; i < n; i++) {
            uint8_t hi = lcg_step(st, 5, 3); st = hi;
            uint8_t lo = lcg_step(st, 5, 3); st = lo;
            d[i] = (hi << 4) | lo;
        }
        run_test("LCG sequence a=5 c=3", d, n);
        free(d);
    }

    {
        int n = 128;
        uint8_t *d = calloc(n, 1);
        run_test("all zeros (raw fallback expected)", d, n);
        free(d);
    }

    {
        int n = 128;
        uint8_t *d = malloc(n);
        srand(42);
        for (int i = 0; i < n; i++) d[i] = rand() & 0xFF;
        run_test("random data (raw fallback expected)", d, n);
        free(d);
    }

    /* ---- K-search tests ---- */
    printf("\nK-search (transform needed to compress):\n");

    {
        /* LCG sequence XOR-masked: K-search must find XOR K=7 */
        int n = 256;
        uint8_t *d = malloc(n);
        uint8_t st = 0;
        for (int i = 0; i < n; i++) {
            uint8_t hi = lcg_step(st, 5, 3); st = hi;
            uint8_t lo = lcg_step(st, 5, 3); st = lo;
            d[i] = ((hi ^ 7) << 4) | (lo ^ 7);
        }
        run_test("LCG a=5,c=3 XOR-masked K=7", d, n);
        free(d);
    }

    {
        /* Arithmetic +3, all nibbles ADD-shifted by 5 */
        int n = 256;
        uint8_t *d = malloc(n);
        uint8_t v = 0;
        for (int i = 0; i < n; i++) {
            uint8_t hi = (v + 5) & 0xF; v = (v + 3) & 0xF;
            uint8_t lo = (v + 5) & 0xF; v = (v + 3) & 0xF;
            d[i] = (hi << 4) | lo;
        }
        run_test("arithmetic +3, ADD-shifted by 5", d, n);
        free(d);
    }

    {
        /* LCG a=9 c=7 SUB-masked by 3 */
        int n = 256;
        uint8_t *d = malloc(n);
        uint8_t st = 2;
        for (int i = 0; i < n; i++) {
            uint8_t hi = lcg_step(st, 9, 7); st = hi;
            uint8_t lo = lcg_step(st, 9, 7); st = lo;
            d[i] = (((hi - 3 + 16) & 0xF) << 4) | ((lo - 3 + 16) & 0xF);
        }
        run_test("LCG a=9,c=7 SUB-masked K=3", d, n);
        free(d);
    }

    /* ---- Real-world data simulations ---- */
    printf("\nSimulated real-world data:\n");

    {
        /* Audio: random walk, each nibble differs from previous by at most 1 */
        int n = 512;
        uint8_t *d = malloc(n);
        srand(1234);
        uint8_t v = 8;
        for (int i = 0; i < n; i++) {
            int d1 = (rand() % 3) - 1;
            uint8_t hi = (v + d1 + 16) & 0xF;
            int d2 = (rand() % 3) - 1;
            uint8_t lo = (hi + d2 + 16) & 0xF;
            v = lo;
            d[i] = (hi << 4) | lo;
        }
        run_test("audio random walk (step <= 1)", d, n);
        free(d);
    }

    {
        /* Audio: larger steps, bias toward +1 */
        int n = 512;
        uint8_t *d = malloc(n);
        srand(5678);
        uint8_t v = 4;
        for (int i = 0; i < n; i++) {
            /* bias: +1 most of the time, ±2 occasionally */
            int r = rand() % 10;
            int delta = (r < 6) ? 1 : (r < 8) ? 0 : (r == 8) ? 2 : -1;
            uint8_t hi = (v + delta + 16) & 0xF;
            r = rand() % 10;
            delta = (r < 6) ? 1 : (r < 8) ? 0 : (r == 8) ? 2 : -1;
            uint8_t lo = (hi + delta + 16) & 0xF;
            v = lo;
            d[i] = (hi << 4) | lo;
        }
        run_test("audio biased +1 drift", d, n);
        free(d);
    }

    {
        /* Image gradient: linear ramp 0->15->0->15 ... */
        int n = 512;
        uint8_t *d = malloc(n);
        for (int i = 0; i < n; i++) {
            uint8_t hi = (i * 2)     & 0xF;
            uint8_t lo = (i * 2 + 1) & 0xF;
            d[i] = (hi << 4) | lo;
        }
        run_test("image gradient (linear ramp)", d, n);
        free(d);
    }

    {
        /*
         * Image gradient XOR-masked (simulates a sensor encoding trick).
         * High nibble XOR 5, low nibble XOR 5 - K-search should recover it.
         */
        int n = 512;
        uint8_t *d = malloc(n);
        for (int i = 0; i < n; i++) {
            uint8_t hi = ((i * 2)     & 0xF) ^ 5;
            uint8_t lo = ((i * 2 + 1) & 0xF) ^ 5;
            d[i] = (hi << 4) | lo;
        }
        run_test("image gradient XOR-masked K=5", d, n);
        free(d);
    }

    {
        /*
         * ASCII text: real printable chars in 0x20-0x7E range.
         * High nibbles cluster at 2,4,5,6,7. Low nibbles are varied.
         */
        int n = 512;
        uint8_t *d = malloc(n);
        const char *src =
            "the quick brown fox jumps over the lazy dog  "
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz  "
            "hello world this is a test of ascii text compression  "
            "with the lcg chain compressor algorithm version two  "
            "one two three four five six seven eight nine ten  ";
        int slen = (int)strlen(src);
        for (int i = 0; i < n; i++) d[i] = (uint8_t)src[i % slen];
        run_test("ASCII text (a-z A-Z spaces)", d, n);
        free(d);
    }

    {
        /* Sensor: slow +1 drift, occasional ±3 spike */
        int n = 512;
        uint8_t *d = malloc(n);
        srand(999);
        uint8_t v = 5;
        for (int i = 0; i < n; i++) {
            int r = rand() % 16;
            int delta = (r < 12) ? 1 : (r < 14) ? 0 : (r == 14) ? 3 : -3;
            uint8_t hi = (v + delta + 16) & 0xF;
            r = rand() % 16;
            delta = (r < 12) ? 1 : (r < 14) ? 0 : (r == 14) ? 3 : -3;
            uint8_t lo = (hi + delta + 16) & 0xF;
            v = lo;
            d[i] = (hi << 4) | lo;
        }
        run_test("sensor drift (+1 bias, rare spikes)", d, n);
        free(d);
    }

    /* ---- Real file ---- */
    printf("\nReal file test:\n");
    {
        /* Try to open the companion source file */
        FILE *f = fopen("LGC.c", "rb");
        if (!f) f = fopen("test_core.c", "rb");
        if (f) {
            fseek(f, 0, SEEK_END);
            long fsz = ftell(f);
            fseek(f, 0, SEEK_SET);
            int n = (int)((fsz < 8192) ? fsz : 8192);
            n &= ~1;   /* round down to even bytes */
            uint8_t *d = malloc(n);
            fread(d, 1, n, f);
            fclose(f);
            run_test("C source file (LGC.c or test_core.c)", d, n);
            free(d);
        } else {
            printf("  (file not found)\n");
        }
    }

    /* ---- Scale ---- */
    printf("\nScale tests:\n");

    {
        int n = 4096;
        uint8_t *d = malloc(n);
        uint8_t st = 0;
        for (int i = 0; i < n; i++) {
            uint8_t hi = lcg_step(st, 5, 3); st = hi;
            uint8_t lo = lcg_step(st, 5, 3); st = lo;
            d[i] = (hi << 4) | lo;
        }
        run_test("4 KB LCG sequence", d, n);
        free(d);
    }

    {
        int n = 4096;
        uint8_t *d = malloc(n);
        srand(77);
        for (int i = 0; i < n; i++) d[i] = rand() & 0xFF;
        run_test("4 KB random", d, n);
        free(d);
    }

    printf("\nDone.\n");
    return 0;
}
