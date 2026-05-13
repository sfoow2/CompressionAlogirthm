*/
DO NOT USE THIS CODE AS IT IS UNSAFE AND WILL BREAK YOUR COMPUTER DONT EVEN TRY TO RUN IT IT WILL END UP BREAKING EVERYTHING

/*


/*
 * prng_compress.c  —  nibble-pair PRNG compression with per-pass scramble search
 *
 * MULTI-PASS + SCRAMBLE
 * =====================
 * Before each compression pass the encoder brute-forces a reversible
 * per-nibble scramble (operation + parameter k) together with the PRNG
 * settings.  The combination that produces the smallest compressed output
 * on the tuning sample is committed, then the scramble is applied to the
 * full current nibble stream before encoding.
 *
 * Scramble operations (all 4-bit, all reversible):
 *   ADD  k   (n+k)  mod 16        inverse: SUB k
 *   SUB  k   (n-k)  mod 16        inverse: ADD k
 *   XOR  k    n ^ k               inverse: XOR k
 *   ROL  k   rotate-left  k bits  inverse: ROR k
 *   ROR  k   rotate-right k bits  inverse: ROL k
 *   FLIP     bit-reverse nibble   inverse: FLIP
 *   NOT      ~n & 0xF             inverse: NOT
 *
 * Header layout:
 *   Byte 0        : MAGIC (0xC0)
 *   Byte 1        : passes (1..MAX_PASSES)
 *   For each pass p = 0 .. passes-1:
 *     Byte 2+3*p  : packed PRNG settings high byte
 *     Byte 3+3*p  : packed PRNG settings low byte
 *     Byte 4+3*p  : scramble byte = (op << 4) | (k & 0xF)
 *   Payload starts at byte 2 + 3*passes.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <limits.h>
#include <omp.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define MAGIC        0xC0u
#define MAX_PASSES   4

#define CR_MIN  1
#define CR_MAX  7
#define SW_MIN  5
#define SW_MAX  80
#define SW_STEP 1
#define HW_MIN  0
#define HW_MAX  40
#define HW_STEP 1

#define NSEEDS       16
#define WINDOW        8
#define BLOCK         32
#define HISTORY_LEN   4

/* ── settings pack/unpack ── */

static u16 pack_settings(int cr, int sw, int hw)
{
    int cr_val = cr - CR_MIN;           /* 0-6, needs 3 bits */
    int sw_val = sw - SW_MIN;           /* 0-75, needs 7 bits */
    int hw_val = hw - HW_MIN;           /* 0-40, needs 6 bits */
    return (u16)((cr_val << 13) | (sw_val << 6) | hw_val);
}

static void unpack_settings(u16 p, int *cr, int *sw, int *hw)
{
    *cr = ((p >> 13) & 0x07) + CR_MIN;
    *sw = ((p >>  6) & 0x7F) + SW_MIN;
    *hw = (p & 0x3F) + HW_MIN;
}

/* ── nibble scramble ── */

typedef enum {
    SCRAM_ADD  = 0,
    SCRAM_SUB  = 1,
    SCRAM_XOR  = 2,
    SCRAM_ROL  = 3,
    SCRAM_ROR  = 4,
    SCRAM_FLIP = 5,
    SCRAM_NOT  = 6
} ScramOp;

static u8 rol4(u8 n, int k) { k &= 3; return (u8)(((n << k) | (n >> (4-k))) & 0xF); }
static u8 ror4(u8 n, int k) { k &= 3; return (u8)(((n >> k) | (n << (4-k))) & 0xF); }
static u8 flip4(u8 n)       { return (u8)(((n&1)<<3)|((n&2)<<1)|((n&4)>>1)|((n&8)>>3)); }

static u8 scramble_nib(u8 n, ScramOp op, int k)
{
    n &= 0xF;
    switch (op) {
        case SCRAM_ADD:  return (u8)((n + k) & 0xF);
        case SCRAM_SUB:  return (u8)((n - k + 16) & 0xF);
        case SCRAM_XOR:  return (u8)(n ^ (k & 0xF));
        case SCRAM_ROL:  return rol4(n, k);
        case SCRAM_ROR:  return ror4(n, k);
        case SCRAM_FLIP: return flip4(n);
        case SCRAM_NOT:  return (u8)((~n) & 0xF);
        default:         return n;
    }
}

static u8 unscramble_nib(u8 n, ScramOp op, int k)
{
    n &= 0xF;
    switch (op) {
        case SCRAM_ADD:  return (u8)((n - k + 16) & 0xF);
        case SCRAM_SUB:  return (u8)((n + k) & 0xF);
        case SCRAM_XOR:  return (u8)(n ^ (k & 0xF));
        case SCRAM_ROL:  return ror4(n, k);
        case SCRAM_ROR:  return rol4(n, k);
        case SCRAM_FLIP: return flip4(n);
        case SCRAM_NOT:  return (u8)((~n) & 0xF);
        default:         return n;
    }
}

static u8 pack_scramble(ScramOp op, int k)
{ return (u8)(((int)op << 4) | (k & 0xF)); }

static void unpack_scramble(u8 b, ScramOp *op, int *k)
{ *op = (ScramOp)((b >> 4) & 0x7); *k = b & 0xF; }

/* ── PRNG (thread-local for parallelization) ── */

static int G_CR = 3;
static int G_SW = 40;
static int G_HW = 15;
#pragma omp threadprivate(G_CR, G_SW, G_HW)

typedef struct {
    u32 lcg;
    u8  last;
    u8  history[HISTORY_LEN];
    int hpos;
    u32 bias[16];
} NibblePRNG;

static u32 lcg_step(u32 *s, u8 fb)
{ *s = *s * 1664525u + 1013904223u + (u32)fb; return *s; }

static void nibble_seed(NibblePRNG *m, u8 seed)
{
    m->lcg  = (u32)seed * 2654435761u ^ 0xDEADBEEFu;
    m->last = seed & 0x0Fu;
    m->hpos = 0;
    memset(m->history, m->last, sizeof m->history);
}

static u8 nibble_next(NibblePRNG *m)
{
    int CR = G_CR, SW = G_SW, HW = G_HW;
    for (int i = 0; i < 16; i++)
        m->bias[i] = (lcg_step(&m->lcg, m->last) >> 16) & 0xFFu;
    for (int d = -CR; d <= CR; d++) {
        int idx  = ((int)m->last + d + 16) & 0x0F;
        int dist = d < 0 ? -d : d;
        m->bias[idx] += (u32)(CR - dist + 1) * 3u;
    }
    m->bias[m->last] += (u32)SW;
    for (int h = 0; h < HISTORY_LEN; h++)
        m->bias[m->history[h]] += (u32)HW;
    u64 total = 0;
    for (int i = 0; i < 16; i++) total += m->bias[i];
    u64 r = ((u64)(lcg_step(&m->lcg, m->last) >> 8) << 16)
          |  (u64)(lcg_step(&m->lcg, m->last) >> 16);
    u8 out = 0; u64 acc = 0;
    for (int i = 0; i < 16; i++) {
        acc += m->bias[i];
        if ((r % total) < acc) { out = (u8)i; break; }
    }
    m->history[m->hpos] = m->last;
    m->hpos = (m->hpos + 1) % HISTORY_LEN;
    m->last = out;
    return out;
}

static int enc_seed[16][16];
static int enc_off [16][16];
static NibblePRNG g_prng;
#pragma omp threadprivate(enc_seed, enc_off, g_prng)

static void build_table(void)
{
    for (int a = 0; a < 16; a++)
        for (int b = 0; b < 16; b++)
            enc_seed[a][b] = enc_off[a][b] = -1;
    for (int s = 0; s < NSEEDS; s++) {
        nibble_seed(&g_prng, (u8)s);
        u8 a = nibble_next(&g_prng);
        for (int w = 0; w < WINDOW; w++) {
            u8 b = nibble_next(&g_prng);
            if (enc_seed[a][b] == -1) {
                enc_seed[a][b] = s;
                enc_off [a][b] = w;
            }
        }
    }
}

#define TUNE_SAMPLE 4096  /* increased from 512 to get representative sample */

/* ── bit stream (unchanged) ── */

typedef struct { u8 *buf; int cap; int bp; } BS;

static void bs_init(BS *b, u8 *buf, int cap)
{ b->buf = buf; b->cap = cap; b->bp = 0; memset(buf, 0, (size_t)cap); }

static void bs_w(BS *b, u32 v, int n)
{
    for (int i = 0; i < n; i++) {
        int by = b->bp / 8, bi = b->bp % 8;
        if (by < b->cap && ((v >> i) & 1u)) b->buf[by] |= (u8)(1u << bi);
        b->bp++;
    }
}

static u32 bs_r(BS *b, int n)
{
    u32 v = 0;
    for (int i = 0; i < n; i++) {
        int by = b->bp / 8, bi = b->bp % 8;
        if (by < b->cap && ((b->buf[by] >> bi) & 1u)) v |= (1u << i);
        b->bp++;
    }
    return v;
}

static int bs_bytes(const BS *b) { return (b->bp + 7) / 8; }

/* ── nibble/byte conversion (unchanged) ── */

static int bytes_to_nibbles(const u8 *in, int len, u8 *out)
{
    for (int i = 0; i < len; i++) {
        out[2*i    ] = (in[i] >> 4) & 0x0Fu;
        out[2*i + 1] =  in[i]       & 0x0Fu;
    }
    return 2 * len;
}

static int nibbles_to_bytes(const u8 *in, int nib_len, u8 *out)
{
    int out_len = nib_len / 2;
    for (int i = 0; i < out_len; i++)
        out[i] = (u8)((in[2*i] << 4) | (in[2*i + 1] & 0x0Fu));
    return out_len;
}

/* ── block encode/decode (unchanged) ── */

static int encode_block(const u8 *in, int count, u8 *out, int *in_used)
{
    u8  comp[BLOCK], seed_s[BLOCK], off_s[BLOCK], raw_s[BLOCK];
    int nslots = 0, ndone = 0;
    while (ndone < count && nslots < BLOCK) {
        int a = in[ndone] & 0xF;
        if (ndone + 1 < count) {
            int b = in[ndone + 1] & 0xF;
            int s = enc_seed[a][b];
            if (s >= 0) {
                comp[nslots]   = 1;
                seed_s[nslots] = (u8)s;
                off_s [nslots] = (u8)enc_off[a][b];
                nslots++; ndone += 2; continue;
            }
        }
        comp[nslots]  = 0;
        raw_s[nslots] = (u8)(in[ndone] & 0xF);
        nslots++; ndone++;
    }
    if (in_used) *in_used = ndone;
    u32 flags = 0;
    for (int i = 0; i < nslots; i++)
        if (comp[i]) flags |= (1u << i);
    u8 pbuf[128]; memset(pbuf, 0, sizeof pbuf);
    BS pbs; bs_init(&pbs, pbuf, (int)sizeof pbuf);
    for (int i = 0; i < nslots; i++) {
        if (comp[i]) { bs_w(&pbs, seed_s[i], 4); bs_w(&pbs, off_s[i], 3); }
        else            bs_w(&pbs, raw_s[i], 4);
    }
    int comp_bytes = 5 + bs_bytes(&pbs);  /* 4 bytes flags + 1 byte nslots */
    int raw_bytes  = 5 + ndone;            /* 4 bytes flags + 1 byte ndone */
    if (comp_bytes >= raw_bytes) {
        out[0] = 0xFF; out[1] = 0xFF; out[2] = 0xFF; out[3] = 0xFF;
        out[4] = (u8)ndone;
        for (int i = 0; i < ndone; i++) out[5 + i] = in[i] & 0x0Fu;
        return raw_bytes;
    }
    out[0] = (u8)(flags >> 24); out[1] = (u8)(flags >> 16);
    out[2] = (u8)(flags >> 8);  out[3] = (u8)flags;
    out[4] = (u8)nslots;
    memcpy(out + 5, pbuf, bs_bytes(&pbs));
    return comp_bytes;
}

static int decode_block(const u8 *in, int in_len, u8 *out, int *consumed)
{
    if (in_len < 5) return 0;
    u32 flags = ((u32)in[0] << 24) | ((u32)in[1] << 16) | ((u32)in[2] << 8) | (u32)in[3];
    if (flags == 0xFFFFFFFFu) {
        int cnt = (int)in[4];
        if (cnt > in_len - 5) cnt = in_len - 5;
        for (int i = 0; i < cnt; i++) out[i] = in[5 + i] & 0x0Fu;
        *consumed = 5 + cnt;
        return cnt;
    }
    int nslots = (int)in[4];
    BS  pbs;
    pbs.buf = (u8 *)(in + 5); pbs.cap = in_len - 5; pbs.bp = 0;
    int pos = 0;
    for (int sl = 0; sl < nslots; sl++) {
        if ((flags >> sl) & 1u) {
            u8  seed = (u8)bs_r(&pbs, 4);
            int off  = (int)bs_r(&pbs, 3);
            nibble_seed(&g_prng, seed);
            u8 a = nibble_next(&g_prng);
            for (int i = 0; i < off; i++) nibble_next(&g_prng);
            u8 b = nibble_next(&g_prng);
            out[pos++] = a; out[pos++] = b;
        } else {
            out[pos++] = (u8)bs_r(&pbs, 4);
        }
    }
    int bits = 0;
    for (int sl = 0; sl < nslots; sl++)
        bits += ((flags >> sl) & 1u) ? 7 : 4;
    *consumed = 5 + (bits + 7) / 8;
    return pos;
}

static int encode_stream(const u8 *in, int ilen, u8 *out, int ocap)
{
    int ip = 0, op = 0;
    while (ip < ilen) {
        int chunk = (ilen - ip < BLOCK) ? (ilen - ip) : BLOCK;
        u8  blk[BLOCK * 2 + 8];
        int in_used = 0;
        int blen = encode_block(in + ip, chunk, blk, &in_used);
        if (op + blen > ocap) break;
        memcpy(out + op, blk, blen);
        op += blen; ip += in_used;
    }
    return op;
}

static void analyze_compression(const u8 *in, int ilen)
{
    int total_matched = 0, total_raw = 0, total_blocks = 0, total_bits = 0;

    int ip = 0;
    while (ip < ilen) {
        int chunk = (ilen - ip < BLOCK) ? (ilen - ip) : BLOCK;
        u8  comp[BLOCK];
        int nslots = 0, ndone = 0;

        while (ndone < chunk && nslots < BLOCK) {
            int a = in[ip + ndone] & 0xF;
            if (ndone + 1 < chunk) {
                int b = in[ip + ndone + 1] & 0xF;
                if (enc_seed[a][b] >= 0) {
                    comp[nslots] = 1;
                    total_matched++;
                    total_bits += 7;  /* 4 bits seed + 3 bits offset */
                    nslots++; ndone += 2; continue;
                }
            }
            comp[nslots] = 0;
            total_raw++;
            total_bits += 4;
            nslots++; ndone++;
        }

        total_blocks++;
        total_bits += 16;  /* flags byte + count byte per block */
        ip += ndone;
    }

    int total_bytes_data = (total_bits + 7) / 8;
    int total_bytes_with_header = 5 + total_bytes_data;

    fprintf(stderr, "  [ANALYSIS]\n");
    fprintf(stderr, "    Input:          %d nibbles\n", ilen);
    fprintf(stderr, "    Matched pairs:  %d (%.1f%%)\n", total_matched, 100.0f*total_matched*2/ilen);
    fprintf(stderr, "    Raw nibbles:    %d (%.1f%%)\n", total_raw, 100.0f*total_raw/ilen);
    fprintf(stderr, "    Blocks:         %d\n", total_blocks);
    fprintf(stderr, "    Data bits:      %d matched*7 + %d raw*4 = %d bits\n",
            total_matched, total_raw, total_bits - total_blocks*16);
    fprintf(stderr, "    Block overhead: %d blocks * 16 bits = %d bits\n", total_blocks, total_blocks*16);
    fprintf(stderr, "    Payload bytes:  %d (%.1f%% of input)\n", total_bytes_data, 100.0f*total_bytes_data/((ilen+1)/2));
    fprintf(stderr, "    Header:         5 bytes\n");
    fprintf(stderr, "    Total:          %d bytes (%.1f%% of input)\n", total_bytes_with_header, 100.0f*total_bytes_with_header/((ilen+1)/2));
}

static int decode_stream(const u8 *in, int ilen, u8 *out, int ocap)
{
    int ip = 0, op = 0;
    while (ip < ilen && op < ocap) {
        u8  syms[BLOCK * 2 + 4];
        int consumed = 0;
        int ns = decode_block(in + ip, ilen - ip, syms, &consumed);
        if (consumed <= 0) break;
        for (int i = 0; i < ns && op < ocap; i++) out[op++] = syms[i];
        ip += consumed;
    }
    return op;
}

/* ── joint scramble + PRNG tuner ── */

typedef struct { int cr, sw, hw, csz; float ratio; } Settings;
typedef struct { Settings prng; ScramOp op; int k; } TuneResult;

static const char *scram_name(ScramOp op)
{
    switch (op) {
        case SCRAM_ADD:  return "ADD";
        case SCRAM_SUB:  return "SUB";
        case SCRAM_XOR:  return "XOR";
        case SCRAM_ROL:  return "ROL";
        case SCRAM_ROR:  return "ROR";
        case SCRAM_FLIP: return "FLP";
        case SCRAM_NOT:  return "NOT";
        default:         return "???";
    }
}

static TuneResult tune_scramble(const u8 *data, int len)
{
    /* ops and their k iteration ranges */
    static const int OPS[]  = { SCRAM_ADD, SCRAM_SUB, SCRAM_XOR,
                                 SCRAM_ROL, SCRAM_ROR, SCRAM_FLIP, SCRAM_NOT };
    static const int KMAX[] = { 16, 16, 16, 4, 4, 1, 1 };
    enum { N_OPS = 7 };

    int slen = (len < TUNE_SAMPLE) ? len : TUNE_SAMPLE;

    TuneResult best;
    best.prng.cr  = CR_MIN; best.prng.sw = SW_MIN; best.prng.hw = HW_MIN;
    best.prng.csz = slen * 3; best.prng.ratio = 999.0f;
    best.op = SCRAM_ADD; best.k = 0;

    u8 *scratch = (u8 *)malloc((size_t)slen);
    u8 *enc     = (u8 *)malloc((size_t)(slen * 3 + 64));
    if (!scratch || !enc) { free(scratch); free(enc); return best; }

    int tries = 0;
    int best_per_op[7];
    for (int j = 0; j < 7; j++) best_per_op[j] = INT_MAX;

    for (int oi = 0; oi < N_OPS; oi++) {
        ScramOp op = (ScramOp)OPS[oi];
        int     km = KMAX[oi];

        int best_for_op = INT_MAX;
        int best_k_for_op = -1;

        for (int k = 0; k < km; k++) {
            /* apply this scramble to the tuning sample */
            for (int i = 0; i < slen; i++)
                scratch[i] = scramble_nib(data[i], op, k);

            /* find best PRNG for this (op,k) combo */
            int best_sz_for_this_k = INT_MAX;
            int local_tries = 0;

            #pragma omp parallel for collapse(3) reduction(min:best_sz_for_this_k) \
                reduction(+:local_tries)
            for (int cr = CR_MIN; cr <= CR_MAX; cr++)
            for (int sw = SW_MIN; sw <= SW_MAX; sw += SW_STEP)
            for (int hw = HW_MIN; hw <= HW_MAX; hw += HW_STEP) {
                G_CR = cr; G_SW = sw; G_HW = hw;
                build_table();
                int sz = encode_stream(scratch, slen, enc, slen * 3 + 64);
                local_tries++;

                if (sz < best_sz_for_this_k) {
                    best_sz_for_this_k = sz;
                }

                if (sz < best.prng.csz) {
                    #pragma omp critical
                    {
                        if (sz < best.prng.csz) {
                            best.prng.cr  = cr; best.prng.sw = sw; best.prng.hw = hw;
                            best.prng.csz = sz;
                            best.prng.ratio = (float)sz / (float)slen;
                            best.op = op; best.k  = k;
                        }
                    }
                }
            }
            tries += local_tries;

            if (best_sz_for_this_k < best_for_op) {
                best_for_op = best_sz_for_this_k;
                best_k_for_op = k;
            }
        }
        best_per_op[oi] = best_for_op;
    }

    fprintf(stderr, "  [TUNE] best: %s:%d with CR=%d SW=%d HW=%d → %d bytes (%.1f%%), %d combos tried\n",
            scram_name(best.op), best.k, best.prng.cr, best.prng.sw, best.prng.hw,
            best.prng.csz, 100.0f*best.prng.csz/slen, tries);

    free(scratch); free(enc);
    return best;
}

/* ── compress ── */

int compress(const u8 *in, int ilen, u8 *out, int ocap, int *used)
{
    *used = 0;

    int enc_cap = ilen * 2 + 256;
    int nib_cap = ilen * 2 + 256;

    u8 *enc_a = (u8 *)malloc((size_t)enc_cap);
    u8 *enc_b = (u8 *)malloc((size_t)enc_cap);
    u8 *nib_a = (u8 *)malloc((size_t)nib_cap);
    u8 *nib_b = (u8 *)malloc((size_t)nib_cap);
    u8 *scr   = (u8 *)malloc((size_t)nib_cap);  /* scratch for pre-encode scramble */
    if (!enc_a || !enc_b || !nib_a || !nib_b || !scr) {
        free(enc_a); free(enc_b); free(nib_a); free(nib_b); free(scr);
        return 0;
    }

    const u8 *cur_nibs    = in;
    int        cur_nib_len = ilen;

    u8 *enc_good     = enc_a;
    u8 *enc_try      = enc_b;
    int enc_good_len = 0;

    int     pass_cr[MAX_PASSES], pass_sw[MAX_PASSES], pass_hw[MAX_PASSES];
    ScramOp pass_op[MAX_PASSES];
    int     pass_k [MAX_PASSES];
    int passes = 0;

    u8 *nib_cur = nib_a;
    u8 *nib_alt = nib_b;

    for (int p = 0; p < MAX_PASSES; p++) {
        TuneResult tr = tune_scramble(cur_nibs, cur_nib_len);

        /* apply the best-found scramble to the full current nibble stream */
        for (int i = 0; i < cur_nib_len; i++)
            scr[i] = scramble_nib(cur_nibs[i], tr.op, tr.k);

        G_CR = tr.prng.cr; G_SW = tr.prng.sw; G_HW = tr.prng.hw;
        build_table();
        analyze_compression(scr, cur_nib_len);
        int enc_len = encode_stream(scr, cur_nib_len, enc_try, enc_cap);

        /*
         * Would accepting this pass shrink the total output?
         * New header cost: 2 (magic+passes) + 3*(passes+1) bytes.
         * Compare against raw input (pass 0) or previous best total (pass >0).
         */
        int new_header = 2 + 3 * (passes + 1);
        int new_total  = new_header + enc_len;
        int prev_total = (passes == 0) ? ilen : (2 + 3 * passes + enc_good_len);
        int accept = new_total < prev_total;

        fprintf(stderr, "[PASS %d] %d nibs → %d B total=%d (prev=%d) %s scram=%s:%d CR=%d\n",
                p, cur_nib_len, enc_len, new_total, prev_total,
                accept ? "✓" : "✗", scram_name(tr.op), tr.k, tr.prng.cr);

        if (!accept) break;

        pass_cr[passes] = tr.prng.cr;
        pass_sw[passes] = tr.prng.sw;
        pass_hw[passes] = tr.prng.hw;
        pass_op[passes] = tr.op;
        pass_k [passes] = tr.k;
        passes++;

        u8 *tmp = enc_good; enc_good = enc_try; enc_try = tmp;
        enc_good_len = enc_len;

        if (p + 1 < MAX_PASSES) {
            int nn = bytes_to_nibbles(enc_good, enc_good_len, nib_cur);
            cur_nibs    = nib_cur;
            cur_nib_len = nn;
            u8 *ntmp = nib_cur; nib_cur = nib_alt; nib_alt = ntmp;
        }
    }

    if (passes == 0) {
        int n = (ilen <= ocap) ? ilen : ocap;
        memcpy(out, in, n);
        free(enc_a); free(enc_b); free(nib_a); free(nib_b); free(scr);
        return ilen;
    }

    *used = 1;

    int op = 0;
    out[op++] = (u8)MAGIC;
    out[op++] = (u8)passes;
    for (int p = 0; p < passes; p++) {
        u16 pk = pack_settings(pass_cr[p], pass_sw[p], pass_hw[p]);
        out[op++] = (u8)(pk >> 8);
        out[op++] = (u8)(pk & 0xFF);
        out[op++] = pack_scramble(pass_op[p], pass_k[p]);
    }

    if (op + enc_good_len <= ocap) memcpy(out + op, enc_good, enc_good_len);
    op += enc_good_len;

    free(enc_a); free(enc_b); free(nib_a); free(nib_b); free(scr);
    return op;
}

/* ── decompress ── */

int decompress(const u8 *in, int ilen, u8 *out, int ocap)
{
    if (ilen < 1) return 0;
    if (in[0] != MAGIC) {
        int n = (ilen < ocap) ? ilen : ocap;
        memcpy(out, in, n);
        return n;
    }
    if (ilen < 2) return 0;

    int passes = (int)in[1];
    if (passes < 1 || passes > MAX_PASSES) return 0;

    int hdr_total = 2 + 3 * passes;
    if (ilen < hdr_total) return 0;

    int     pass_cr[MAX_PASSES], pass_sw[MAX_PASSES], pass_hw[MAX_PASSES];
    ScramOp pass_op[MAX_PASSES];
    int     pass_k [MAX_PASSES];

    for (int p = 0; p < passes; p++) {
        int off = 2 + 3 * p;
        u16 pk = ((u16)in[off] << 8) | in[off + 1];
        unpack_settings(pk, &pass_cr[p], &pass_sw[p], &pass_hw[p]);
        unpack_scramble(in[off + 2], &pass_op[p], &pass_k[p]);
    }

    const u8 *payload = in + hdr_total;
    int        plen   = ilen - hdr_total;

    int nib_cap  = (ocap + plen) * 2 + 256;
    int byte_cap = nib_cap / 2 + 256;

    u8 *nib_a    = (u8 *)malloc((size_t)nib_cap);
    u8 *nib_b    = (u8 *)malloc((size_t)nib_cap);
    u8 *byte_buf = (u8 *)malloc((size_t)byte_cap);
    if (!nib_a || !nib_b || !byte_buf) {
        free(nib_a); free(nib_b); free(byte_buf);
        return 0;
    }

    if (plen > byte_cap) plen = byte_cap;
    memcpy(byte_buf, payload, plen);
    int cur_byte_len = plen;

    u8 *nib_cur = nib_a;
    u8 *nib_alt = nib_b;

    /* undo passes in reverse: decode compressed stream, then un-scramble */
    for (int p = passes - 1; p >= 0; p--) {
        G_CR = pass_cr[p]; G_SW = pass_sw[p]; G_HW = pass_hw[p];
        build_table();

        int nib_len = decode_stream(byte_buf, cur_byte_len, nib_cur, nib_cap);

        /* reverse this pass's scramble */
        for (int i = 0; i < nib_len; i++)
            nib_cur[i] = unscramble_nib(nib_cur[i], pass_op[p], pass_k[p]);

        if (p == 0) {
            int out_len = (nib_len < ocap) ? nib_len : ocap;
            memcpy(out, nib_cur, out_len);
            free(nib_a); free(nib_b); free(byte_buf);
            return out_len;
        }

        /* convert unscrambled nibbles back to bytes for the next (outer) pass */
        int new_byte_len = nibbles_to_bytes(nib_cur, nib_len, byte_buf);
        cur_byte_len = new_byte_len;
        u8 *tmp = nib_cur; nib_cur = nib_alt; nib_alt = tmp;
    }

    free(nib_a); free(nib_b); free(byte_buf);
    return 0;
}

/* ── test harness ── */

static u32 xr = 0x12345678u;
static u8 xrand_nib(void)
{ xr ^= xr << 13; xr ^= xr >> 17; xr ^= xr << 5; return xr & 0xFu; }

static int run_test(const char *name, const u8 *data, int len)
{
    int max_out = len * 4 + 256;
    u8 *comp = (u8 *)malloc((size_t)max_out);
    u8 *dec  = (u8 *)malloc((size_t)(len * 16 + 256));
    if (!comp || !dec) { free(comp); free(dec); return 0; }

    int used = 0;
    int clen = compress(data, len, comp, max_out, &used);
    int dlen = decompress(comp, clen, dec, len * 16 + 256);

    int ok = (dlen == len);
    int first_bad = -1;
    if (ok) {
        for (int i = 0; i < len; i++) {
            if ((data[i] & 0xF) != (dec[i] & 0xF)) {
                ok = 0;
                if (first_bad < 0) first_bad = i;
            }
        }
    }

    if (used) {
        /* read back scramble info from the header for display */
        int passes = (int)comp[1];  /* header: [MAGIC][passes][prng+scr per pass...] */
        char scr_str[64] = "";
        for (int pp = 0; pp < passes && pp < MAX_PASSES; pp++) {
            ScramOp op; int k;
            unpack_scramble(comp[4 + 3*pp], &op, &k);
            char tmp[16];
            snprintf(tmp, sizeof tmp, "%s%s:%d", pp > 0 ? "," : "", scram_name(op), k);
            strncat(scr_str, tmp, sizeof(scr_str) - strlen(scr_str) - 1);
        }
        printf("  %-38s  %4d -> %4d B  (%.1f%%)  p=%d  [%s]  %s\n",
               name, len, clen, 100.0f * clen / len, passes, scr_str,
               ok ? "OK" : "MISMATCH");
    } else {
        printf("  %-38s  %4d -> %4d B  (raw)                    %s\n",
               name, len, clen, ok ? "OK" : "MISMATCH");
    }

    if (first_bad >= 0) {
        printf("    first mismatch nibble %d: got %X expected %X\n",
               first_bad, dec[first_bad] & 0xF, data[first_bad] & 0xF);
        printf("    input  context: ");
        for (int i = first_bad - 2; i < first_bad + 6 && i < len;  i++) {
            if (i < 0) continue;
            printf(i == first_bad ? "[%X]" : " %X ", data[i] & 0xF);
        }
        printf("\n    output context: ");
        for (int i = first_bad - 2; i < first_bad + 6 && i < dlen; i++) {
            if (i < 0) continue;
            printf(i == first_bad ? "[%X]" : " %X ", dec[i] & 0xF);
        }
        printf("\n");
    } else if (dlen != len) {
        printf("    length mismatch: decoded %d, expected %d\n", dlen, len);
    }

    free(comp); free(dec);
    return ok;
}

int main(void)
{
    int pass = 0, fail = 0;

    /* PRNG settings pack/unpack */
    {
        int errs = 0;
        for (int cr=CR_MIN;cr<=CR_MAX;cr++)
        for (int sw=SW_MIN;sw<=SW_MAX;sw+=SW_STEP)
        for (int hw=HW_MIN;hw<=HW_MAX;hw+=HW_STEP) {
            int cr2,sw2,hw2;
            unpack_settings(pack_settings(cr,sw,hw), &cr2,&sw2,&hw2);
            if (cr2!=cr||sw2!=sw||hw2!=hw) errs++;
        }
        int n = (CR_MAX-CR_MIN+1)*((SW_MAX-SW_MIN)/SW_STEP+1)*((HW_MAX-HW_MIN)/HW_STEP+1);
        printf("settings pack/unpack (%d combinations): %s\n", n, errs ? "FAIL" : "OK");
    }

    /* scramble pack/unpack and round-trip */
    {
        int errs_pack = 0, errs_rt = 0;
        for (int oi = 0; oi <= 6; oi++)
        for (int k = 0; k < 16; k++) {
            ScramOp op2; int k2;
            unpack_scramble(pack_scramble((ScramOp)oi, k), &op2, &k2);
            if ((int)op2 != oi || k2 != k) errs_pack++;
            for (int n = 0; n < 16; n++) {
                u8 s = scramble_nib((u8)n, (ScramOp)oi, k);
                u8 u = unscramble_nib(s, (ScramOp)oi, k);
                if (u != (u8)n) errs_rt++;
            }
        }
        printf("scramble pack/unpack  (%d combinations): %s\n", 7*16,
               errs_pack ? "FAIL" : "OK");
        printf("scramble round-trip   (%d nibble*op*k):  %s\n", 16*7*16,
               errs_rt ? "FAIL" : "OK");
    }

    /* single-block encode/decode */
    {
        printf("\nsingle-block encode/decode:\n");
        G_CR=3; G_SW=40; G_HW=15; build_table();
        u8 input[] = {0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,
                      0x9,0xA,0xB,0xC,0xD,0xE,0xF,0x0};
        u8 blk[64], out[64];
        int in_used=0, consumed=0;
        int blen = encode_block(input, 16, blk, &in_used);
        int olen = decode_block(blk, blen, out, &consumed);
        int ok = (olen==in_used && consumed==blen);
        for(int i=0;i<in_used&&ok;i++) if((input[i]&0xF)!=(out[i]&0xF)) ok=0;
        printf("  encode_block/decode_block 16 nibbles: %s  "
               "(encoded %d B, decoded %d nibbles)\n",
               ok?"OK":"FAIL", blen, olen);
        if (ok) pass++; else fail++;
    }

    printf("\nfull compress/decompress:\n");
    printf("  %-38s  %18s  p  scramble(s)       rt\n", "test", "size");
    printf("  %s\n",
           "------------------------------------------------------------------------");

    #define T(label, data, len) \
        do { int r = run_test(label, data, len); if(r) pass++; else fail++; } while(0)


    xr = 0x11223344u;
    { enum{N=4096}; u8 *d=(u8*)malloc(N); for(int i=0;i<N;i++) d[i]=xrand_nib();
      T("random 4096",d,N); free(d); 
    }



    printf("  %s\n",
           "------------------------------------------------------------------------");
    printf("\nresult: %d passed, %d failed\n", pass, fail);
    return fail > 0 ? 1 : 0;
}

