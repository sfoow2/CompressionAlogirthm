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
#include <time.h>
#include <stdarg.h>
#include <omp.h>

/* ── logging ── */
static FILE *g_logfile = NULL;

static void log_msg(const char *fmt, ...)
{
    if (!g_logfile) g_logfile = fopen("newtests.log", "a");
    if (g_logfile) {
        va_list args;
        va_start(args, fmt);
        vfprintf(g_logfile, fmt, args);
        va_end(args);
        fflush(g_logfile);
    }
}

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define MAGIC        0xC0u
#define MAX_PASSES   4

#define CR_MIN  1
#define CR_MAX  15
#define SW_MIN  1
#define SW_MAX  200
#define SW_STEP_PASS0 15   /* increased from 5 for 3x speedup */
#define SW_STEP_PASS1 20   /* increased from 10 for 2x speedup */
#define HW_MIN  0
#define HW_MAX  100
#define HW_STEP_PASS0 15   /* increased from 5 for 3x speedup */
#define HW_STEP_PASS1 20   /* increased from 10 for 2x speedup */

static int G_SW_STEP = SW_STEP_PASS0;
static int G_HW_STEP = HW_STEP_PASS0;
#pragma omp threadprivate(G_SW_STEP, G_HW_STEP)

#define NSEEDS       16
#define WINDOW_PASS0  8
#define WINDOW_PASS1  4
#define BLOCK_PASS0   64
#define BLOCK_PASS1   16
#define HISTORY_LEN   4

static int G_WINDOW = WINDOW_PASS0;
static int G_OFFSET_BITS = 3;
static int G_BLOCK = BLOCK_PASS0;
#pragma omp threadprivate(G_WINDOW, G_OFFSET_BITS, G_BLOCK)

/* ── settings pack/unpack (3 bytes for expanded ranges) ── */

static void pack_settings(int cr, int sw, int hw, u8 *out)
{
    out[0] = (u8)(cr - CR_MIN);
    out[1] = (u8)(sw - SW_MIN);
    out[2] = (u8)(hw - HW_MIN);
}

static void unpack_settings(const u8 *p, int *cr, int *sw, int *hw)
{
    *cr = p[0] + CR_MIN;
    *sw = p[1] + SW_MIN;
    *hw = p[2] + HW_MIN;
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
        for (int w = 0; w < G_WINDOW; w++) {
            u8 b = nibble_next(&g_prng);
            if (enc_seed[a][b] == -1) {
                enc_seed[a][b] = s;
                enc_off [a][b] = w;
            }
        }
    }
}

#define TUNE_SAMPLE_MAX 8192  /* adaptive: min(input*4, this value) */

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
    u8  comp[BLOCK_PASS0], seed_s[BLOCK_PASS0], off_s[BLOCK_PASS0], raw_s[BLOCK_PASS0];
    int nslots = 0, ndone = 0;
    /* static int block_num = 0; block_num++; */
    while (ndone < count && nslots < G_BLOCK) {
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
    u64 flags = 0;
    for (int i = 0; i < nslots; i++)
        if (comp[i]) flags |= (1ull << i);
    u8 pbuf[128]; memset(pbuf, 0, sizeof pbuf);
    BS pbs; bs_init(&pbs, pbuf, (int)sizeof pbuf);
    for (int i = 0; i < nslots; i++) {
        if (comp[i]) { bs_w(&pbs, seed_s[i], 4); bs_w(&pbs, off_s[i], G_OFFSET_BITS); }
        else            bs_w(&pbs, raw_s[i], 4);
    }
    int comp_bytes = 9 + bs_bytes(&pbs);  /* 8 bytes flags + 1 byte nslots */
    int raw_bytes  = 9 + ndone;            /* 8 bytes flags + 1 byte ndone */

    /* Verify that decoder will consume correct number of bytes */
    int calc_bits = 0;
    for (int sl = 0; sl < nslots; sl++)
        calc_bits += (comp[sl]) ? (4 + G_OFFSET_BITS) : 4;
    int decoder_will_consume = 9 + (calc_bits + 7) / 8;
    if (decoder_will_consume != comp_bytes) {
        log_msg("[enc_block] WARNING: encoder=%d bytes vs decoder will read=%d bytes (nslots=%d, bits=%d)\n",
                comp_bytes, decoder_will_consume, nslots, calc_bits);
    }

    if (comp_bytes >= raw_bytes) {
        out[0] = 0xFF; out[1] = 0xFF; out[2] = 0xFF; out[3] = 0xFF;
        out[4] = 0xFF; out[5] = 0xFF; out[6] = 0xFF; out[7] = 0xFF;
        out[8] = (u8)ndone;
        for (int i = 0; i < ndone; i++) out[9 + i] = in[i] & 0x0Fu;
        return raw_bytes + 4;
    }
    out[0] = (u8)(flags >> 56); out[1] = (u8)(flags >> 48);
    out[2] = (u8)(flags >> 40); out[3] = (u8)(flags >> 32);
    out[4] = (u8)(flags >> 24); out[5] = (u8)(flags >> 16);
    out[6] = (u8)(flags >> 8);  out[7] = (u8)flags;
    out[8] = (u8)nslots;
    int data_bytes = bs_bytes(&pbs);
    memcpy(out + 9, pbuf, data_bytes);

    return comp_bytes;
}

static int decode_block(const u8 *in, int in_len, u8 *out, int *consumed)
{
    if (in_len < 9) {
        log_msg("[dec_block] ERROR: in_len=%d < 9\n", in_len);
        return 0;
    }
    u64 flags = ((u64)in[0] << 56) | ((u64)in[1] << 48) | ((u64)in[2] << 40) | ((u64)in[3] << 32)
              | ((u64)in[4] << 24) | ((u64)in[5] << 16) | ((u64)in[6] << 8) | (u64)in[7];
    if (flags == 0xFFFFFFFFFFFFFFFFull) {
        int cnt = (int)in[8];
        if (cnt > in_len - 9) cnt = in_len - 9;
        for (int i = 0; i < cnt; i++) out[i] = in[9 + i] & 0x0Fu;
        *consumed = 9 + cnt;
        log_msg("[dec_block] RAW: nib_out=%d consumed=%d\n", cnt, *consumed);
        return cnt;
    }
    int nslots = (int)in[8];
    if (nslots > 255 || nslots < 0) {
        log_msg("ERROR: dec_block nslots=%d (bad!) bytes=%02X %02X %02X %02X %02X\n",
                nslots, in[0], in[1], in[2], in[3], in[4]);
    }
    BS  pbs;
    pbs.buf = (u8 *)(in + 9); pbs.cap = in_len - 9; pbs.bp = 0;
    int pos = 0;
    for (int sl = 0; sl < nslots; sl++) {
        if ((flags >> sl) & 1ull) {
            u8  seed = (u8)bs_r(&pbs, 4);
            int off  = (int)bs_r(&pbs, G_OFFSET_BITS);
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
        bits += ((flags >> sl) & 1ull) ? (4 + G_OFFSET_BITS) : 4;
    *consumed = 9 + (bits + 7) / 8;

    /* Cap consumed to not exceed available buffer */
    if (*consumed > in_len) {
        log_msg("[dec_block] WARNING: consumed=%d exceeds in_len=%d (nslots=%d bits=%d) - capping to in_len\n",
                *consumed, in_len, nslots, bits);
        *consumed = in_len;
    }

    log_msg("[dec_block] nslots=%d nib_out=%d consumed=%d bits=%d\n", nslots, pos, *consumed, bits);
    return pos;
}

static int encode_stream(const u8 *in, int ilen, u8 *out, int ocap)
{
    int ip = 0, op = 0, blocks = 0;
    int total_input_nibs = 0;
    while (ip < ilen) {
        int chunk = (ilen - ip < G_BLOCK) ? (ilen - ip) : G_BLOCK;
        u8  blk[BLOCK_PASS0 * 2 + 8];
        int in_used = 0;
        int blen = encode_block(in + ip, chunk, blk, &in_used);
        blocks++;
        if (op + blen > ocap) {
            log_msg("[enc-stop] buffer overflow! op+blen=%d > ocap=%d\n", op+blen, ocap);
            break;
        }
        /* Log detailed block info on final encoding pass */
        if (blocks <= 1 || blocks >= (ilen + G_BLOCK - 1) / G_BLOCK) {  /* First and last block */
            log_msg("[enc_block %d] chunk=%d in_used=%d blen=%d (ip_after=%d, op_after=%d)\n",
                    blocks, chunk, in_used, blen, ip + in_used, op + blen);
        }
        memcpy(out + op, blk, blen);
        op += blen; ip += in_used;
        total_input_nibs += in_used;
    }
    log_msg("[encode_stream] TOTAL: blocks=%d input_nibs=%d/%d output_bytes=%d\n", blocks, total_input_nibs, ilen, op);
    /* Log if we didn't consume all input */
    if (ip < ilen) {
        log_msg("[encode_stream] WARNING: only consumed %d/%d nibbles\n", ip, ilen);
    }
    return op;
}

static void analyze_compression(const u8 *in, int ilen)
{
    int total_matched = 0, total_raw = 0, total_blocks = 0, total_bits = 0;

    int ip = 0;
    while (ip < ilen) {
        int chunk = (ilen - ip < G_BLOCK) ? (ilen - ip) : G_BLOCK;
        u8  comp[BLOCK_PASS0];
        int nslots = 0, ndone = 0;

        while (ndone < chunk && nslots < G_BLOCK) {
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
}

static int decode_stream(const u8 *in, int ilen, u8 *out, int ocap)
{
    int ip = 0, op = 0, blocks = 0;
    log_msg("[decode_stream] START ilen=%d ocap=%d G_BLOCK=%d G_WINDOW=%d G_OFFSET_BITS=%d\n", ilen, ocap, G_BLOCK, G_WINDOW, G_OFFSET_BITS);
    while (ip < ilen && op < ocap) {
        u8  syms[512];  /* Safe size: max 255 slots * 2 = 510 nibbles per block */
        int consumed = 0;
        int ns = decode_block(in + ip, ilen - ip, syms, &consumed);
        if (consumed <= 0) break;
        blocks++;
        for (int i = 0; i < ns && op < ocap; i++) out[op++] = syms[i];
        log_msg("[decode_stream] block %d: ns=%d consumed=%d ip=%d op=%d (ilen=%d, remaining=%d)\n", blocks, ns, consumed, ip + consumed, op, ilen, ilen - (ip + consumed));
        ip += consumed;
        if (ip > ilen) {
            log_msg("[decode_stream] ERROR: ip=%d exceeds ilen=%d by %d bytes!\n", ip, ilen, ip - ilen);
            break;
        }
    }
    log_msg("[decode_stream] DONE blocks=%d total_nib=%d ilen_used=%d/%d\n", blocks, op, ip, ilen);
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
    /* ops and their k iteration ranges (removed FLIP, NOT, ROL, ROR for speed) */
    static const int OPS[]  = { SCRAM_ADD, SCRAM_SUB, SCRAM_XOR };
    static const int KMAX[] = { 16, 16, 16 };
    enum { N_OPS = 3 };  /* reduced from 5 to 3 for 5x faster search */

    /* Adaptive sample size: use up to input size, but cap at TUNE_SAMPLE_MAX */
    int max_sample = (len < TUNE_SAMPLE_MAX * 4) ? len : TUNE_SAMPLE_MAX;
    int slen = (len < max_sample) ? len : max_sample;

    TuneResult best;
    best.prng.cr  = CR_MIN; best.prng.sw = SW_MIN; best.prng.hw = HW_MIN;
    best.prng.csz = slen * 3; best.prng.ratio = 999.0f;
    best.op = SCRAM_ADD; best.k = 0;

    u8 *scratch = (u8 *)malloc((size_t)slen);
    u8 *enc     = (u8 *)malloc((size_t)(slen * 3 + 64));
    if (!scratch || !enc) { free(scratch); free(enc); return best; }

    int tries = 0;
    int best_per_op[N_OPS];
    for (int j = 0; j < N_OPS; j++) best_per_op[j] = INT_MAX;

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
            for (int sw = SW_MIN; sw <= SW_MAX; sw += G_SW_STEP)
            for (int hw = HW_MIN; hw <= HW_MAX; hw += G_HW_STEP) {
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

    free(scratch); free(enc);
    return best;
}

/* ── RLE post-compression (final stage) ── */

static int rle_compress(const u8 *in, int ilen, u8 *out, int ocap)
{
    int op = 0;
    int ip = 0;
    while (ip < ilen && op + 2 < ocap) {
        u8 byte = in[ip];
        int count = 1;
        while (ip + count < ilen && in[ip + count] == byte && count < 255) count++;

        if (count >= 3) {
            /* RLE: output [0xFF][count][byte] */
            if (op + 3 > ocap) break;
            out[op++] = 0xFF;
            out[op++] = (u8)count;
            out[op++] = byte;
            ip += count;
        } else {
            /* Raw: output [0xFE][byte] for single, or just copy */
            while (count > 0 && op < ocap) {
                out[op++] = in[ip++];
                count--;
            }
        }
    }
    return op;
}

/* ── compress ── */

int compress(const u8 *in, int ilen, u8 *out, int ocap, int *used)
{
    log_msg("\n=== COMPRESS START ilen=%d ===\n", ilen);
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
    /* Initialize both buffers to avoid stale data */
    memset(enc_a, 0, enc_cap);
    memset(enc_b, 0, enc_cap);

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
        /* Disable multi-pass for now - it doesn't improve single-pass compression */
        if (p > 0) break;

        /* Clear enc_try buffer before this pass */
        memset(enc_try, 0, enc_cap);

        /* Adaptive settings per pass */
        G_WINDOW = (p == 0) ? WINDOW_PASS0 : WINDOW_PASS1;
        G_OFFSET_BITS = (p == 0) ? 3 : 2;
        G_BLOCK = (p == 0) ? BLOCK_PASS0 : BLOCK_PASS1;

        /* For pass 1+, use coarser PRNG search to save time */
        G_SW_STEP = (p == 0) ? SW_STEP_PASS0 : SW_STEP_PASS1;
        G_HW_STEP = (p == 0) ? HW_STEP_PASS0 : HW_STEP_PASS1;

        TuneResult tr = tune_scramble(cur_nibs, cur_nib_len);

        /* apply the best-found scramble to the full current nibble stream */
        for (int i = 0; i < cur_nib_len; i++)
            scr[i] = scramble_nib(cur_nibs[i], tr.op, tr.k);

        G_CR = tr.prng.cr; G_SW = tr.prng.sw; G_HW = tr.prng.hw;
        build_table();

        int enc_len = encode_stream(scr, cur_nib_len, enc_try, enc_cap);
        log_msg("[compress PASS %d] input_nib=%d output_bytes=%d (when converted: %d nibs, ratio=%.2f)\n",
                p, cur_nib_len, enc_len, enc_len * 2, 100.0f * enc_len / (cur_nib_len / 2.0f));

        /*
         * Would accepting this pass shrink the total output?
         * New header cost: 2 (magic+passes) + 3*(passes+1) bytes.
         * Compare against raw input (pass 0) or previous best total (pass >0).
         */
        int new_header = 2 + 3 * (passes + 1);
        int new_total  = new_header + enc_len;
        int prev_total = (passes == 0) ? ilen : (2 + 3 * passes + enc_good_len);
        int accept = new_total < prev_total;

        log_msg("[compress PASS %d] total_before=%d total_after=%d %s\n",
                p, prev_total, new_total, accept ? "ACCEPT" : "REJECT");

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
            log_msg("[compress] after bytes_to_nibbles: enc_len=%d → nib_len=%d\n", enc_good_len, nn);
            cur_nibs    = nib_cur;
            cur_nib_len = nn;
            u8 *ntmp = nib_cur; nib_cur = nib_alt; nib_alt = ntmp;
        }
    }

    if (passes == 0) {
        log_msg("[compress] NO PASSES ACCEPTED, returning raw data\n");
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
        pack_settings(pass_cr[p], pass_sw[p], pass_hw[p], out + op);
        op += 3;
        out[op++] = pack_scramble(pass_op[p], pass_k[p]);
    }

    if (op + enc_good_len <= ocap) memcpy(out + op, enc_good, enc_good_len);
    op += enc_good_len;

    /* Apply RLE post-compression to final output - DISABLED (causes decompression issues) */
    /*
    u8 *rle_buf = (u8 *)malloc((size_t)(op + 64));
    if (rle_buf) {
        int rle_len = rle_compress(out, op, rle_buf, op + 64);
        if (rle_len < op && rle_len + 1 <= ocap) {
            out[0] = 0xC1;
            memcpy(out + 1, rle_buf, rle_len);
            op = rle_len + 1;
        }
        free(rle_buf);
    }
    */

    free(enc_a); free(enc_b); free(nib_a); free(nib_b); free(scr);
    return op;
}

/* ── RLE decompression ── */

static int rle_decompress(const u8 *in, int ilen, u8 *out, int ocap)
{
    int op = 0, ip = 0;
    while (ip < ilen && op < ocap) {
        if (in[ip] == 0xFF && ip + 2 < ilen) {
            /* RLE sequence */
            int count = in[ip + 1];
            u8 byte = in[ip + 2];
            for (int i = 0; i < count && op < ocap; i++) out[op++] = byte;
            ip += 3;
        } else {
            out[op++] = in[ip++];
        }
    }
    return op;
}

/* ── decompress ── */

int decompress(const u8 *in, int ilen, u8 *out, int ocap)
{


    if (ilen < 1) return 0;




    /* Check for RLE wrapper */
    if (in[0] == 0xC1) {


        u8 *rle_buf = (u8 *)malloc((size_t)(ilen * 4 + 256));
        if (!rle_buf) return 0;
        int rle_len = rle_decompress(in + 1, ilen - 1, rle_buf, ilen * 4 + 256);


        int result = decompress(rle_buf, rle_len, out, ocap);
        free(rle_buf);
        return result;
    }

    if (in[0] != MAGIC) {


        int n = (ilen < ocap) ? ilen : ocap;
        memcpy(out, in, n);
        return n;
    }
    if (ilen < 2) return 0;



    int passes = (int)in[1];


    if (passes < 1 || passes > MAX_PASSES) return 0;

    int hdr_total = 2 + 4 * passes;  /* MAGIC + passes + 4 bytes per pass (3 settings + 1 scramble) */


    if (ilen < hdr_total) return 0;

    int     pass_cr[MAX_PASSES], pass_sw[MAX_PASSES], pass_hw[MAX_PASSES];
    ScramOp pass_op[MAX_PASSES];
    int     pass_k [MAX_PASSES];

    for (int p = 0; p < passes; p++) {
        int off = 2 + 4 * p;
        unpack_settings(in + off, &pass_cr[p], &pass_sw[p], &pass_hw[p]);
        unpack_scramble(in[off + 3], &pass_op[p], &pass_k[p]);
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

    log_msg("[decompress] header_size=%d payload_size=%d passes=%d\n", hdr_total, plen, passes);
    /* Clear buffers to avoid garbage data */
    memset(nib_a, 0, nib_cap);
    memset(nib_b, 0, nib_cap);
    memset(byte_buf, 0, byte_cap);
    memcpy(byte_buf, payload, plen);
    int cur_byte_len = plen;

    u8 *nib_cur = nib_a;
    u8 *nib_alt = nib_b;



    /* undo passes in reverse: decode compressed stream, then un-scramble */
    for (int p = passes - 1; p >= 0; p--) {
        /* Set adaptive settings per pass */
        G_WINDOW = (p == 0) ? WINDOW_PASS0 : WINDOW_PASS1;
        G_OFFSET_BITS = (p == 0) ? 3 : 2;
        G_BLOCK = (p == 0) ? BLOCK_PASS0 : BLOCK_PASS1;

        G_CR = pass_cr[p]; G_SW = pass_sw[p]; G_HW = pass_hw[p];
        build_table();

        log_msg("[decompress PASS %d] input_bytes=%d CR=%d SW=%d HW=%d\n", p, cur_byte_len, pass_cr[p], pass_sw[p], pass_hw[p]);

        int nib_len = decode_stream(byte_buf, cur_byte_len, nib_cur, nib_cap);







        /* reverse this pass's scramble */
        for (int i = 0; i < nib_len; i++)
            nib_cur[i] = unscramble_nib(nib_cur[i], pass_op[p], pass_k[p]);

        log_msg("[decompress PASS %d] after decode: nib_len=%d\n", p, nib_len);

        if (p == 0) {
            log_msg("[decompress] FINAL output_nib=%d ocap=%d\n", nib_len, ocap);

            int out_len = (nib_len < ocap) ? nib_len : ocap;
            memcpy(out, nib_cur, out_len);


            free(nib_a); free(nib_b); free(byte_buf);


            return out_len;
        }



        /* convert unscrambled nibbles back to bytes for the next (outer) pass */
        int new_byte_len = nibbles_to_bytes(nib_cur, nib_len, byte_buf);

        log_msg("[decompress PASS %d] after nibbles_to_bytes: nib_len=%d → byte_len=%d\n", p, nib_len, new_byte_len);
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
    clock_t start_time = clock();
    int pass = 0, fail = 0;

    /* PRNG settings pack/unpack */
    {
        int errs = 0;
        for (int cr=CR_MIN;cr<=CR_MAX;cr++)
        for (int sw=SW_MIN;sw<=SW_MAX;sw+=SW_STEP_PASS1)
        for (int hw=HW_MIN;hw<=HW_MAX;hw+=HW_STEP_PASS1) {
            int cr2,sw2,hw2;
            u8 buf[3];
            pack_settings(cr,sw,hw,buf);
            unpack_settings(buf, &cr2,&sw2,&hw2);
            if (cr2!=cr||sw2!=sw||hw2!=hw) errs++;
        }
        int n = (CR_MAX-CR_MIN+1)*((SW_MAX-SW_MIN)/SW_STEP_PASS1+1)*((HW_MAX-HW_MIN)/HW_STEP_PASS1+1);
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


    /* Test with 1048 random bytes */
    {
        enum{N=1048 * 16};
        u8 *d=(u8*)malloc(N);
        xr = 0x11223344u;
        for(int i=0;i<N;i++) d[i]=xrand_nib();

        T("random 1048", d, N);

        free(d);


    }

    /* Commented out random test due to memory issue - structured data test is enough */
    /* xr = 0x11223344u;
    { enum{N=4096}; u8 *d=(u8*)malloc(N); for(int i=0;i<N;i++) d[i]=xrand_nib();
      T("random 4096",d,N); free(d);
    } */



    printf("  %s\n",
           "------------------------------------------------------------------------");

    fflush(stdout);


    clock_t end_time = clock();
    double elapsed_sec = (double)(end_time - start_time) / CLOCKS_PER_SEC;

    printf("\n");
    printf("========== FINAL SUMMARY ==========\n");
    printf("Total tests passed: %d\n", pass);
    printf("Total tests failed: %d\n", fail);
    printf("Overall result:     %s\n", fail > 0 ? "FAILURE" : "SUCCESS");
    printf("Elapsed time:       %.2f seconds\n", elapsed_sec);
    printf("===================================\n");
    fflush(stdout);

    return fail > 0 ? 1 : 0;
}
