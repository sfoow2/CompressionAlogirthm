*/
DO NOT USE THIS CODE AS IT IS UNSAFE AND WILL BREAK YOUR COMPUTER DONT EVEN TRY TO RUN IT IT WILL END UP BREAKING EVERYTHING

/*


/*
 * prng_compress.c  —  nibble-pair PRNG compression
 * ==================================================
 *
 * DATA MODEL
 *   Input is treated as a stream of 4-bit nibbles.
 *   A biased PRNG produces nibble sequences; any adjacent pair (A,B) that
 *   appears in a PRNG run can be stored as (seed=4 bits, offset=3 bits)
 *   instead of the raw 8 bits — a saving of 1 bit per pair before overhead.
 *   The win comes from amortising the fixed per-block header and from
 *   multi-pass recompression.
 *
 * BLOCK FORMAT
 *   Compressed block:
 *     byte 0   : flags   (bit i = 1 → slot i is a compressed pair)
 *     byte 1   : nslots  (1..BLOCK)
 *     payload  : packed bit-fields, LSB-first
 *                  compressed slot : seed[4] + offset[3] = 7 bits
 *                  raw slot        : nibble[4]            = 4 bits
 *
 *   Raw-escape block (when compression loses):
 *     byte 0   : 0xFF
 *     byte 1   : count of raw nibbles
 *     bytes    : raw nibbles, one per byte (low 4 bits used)
 *
 * FILE FORMAT
 *   [0]      MAGIC (0xC0)
 *   [1..2]   packed PRNG settings for pass 0
 *   [3]      number of passes applied  (1..MAX_PASSES)
 *   for each subsequent pass p = 1..passes-1:
 *     [+0,+1] packed PRNG settings for pass p
 *   remainder: compressed payload (output of the last/innermost pass)
 *
 * MULTI-PASS ENCODE ORDER
 *   raw → pack_to_nibbles → encode(pass[0]) → encode(pass[1]) → ... → stored
 *   (the initial raw data is already nibble-valued, i.e. values 0x00..0x0F)
 *   Between passes the compressed byte output (0x00..0xFF) is split into
 *   high and low nibbles before being fed to the next encode pass.
 *
 * MULTI-PASS DECODE ORDER  (reverse of encode)
 *   stored → decode(pass[N-1]) → unpack_nibbles_to_bytes →
 *          → decode(pass[N-2]) → unpack_nibbles_to_bytes →
 *          → ... → decode(pass[0]) → raw
 *
 * KEY INVARIANT
 *   encode_stream / decode_stream work on nibble streams (values 0x0..0xF).
 *   The compressed output of encode_stream is a byte stream (0x00..0xFF).
 *   Before feeding that byte stream into a second encode_stream pass, each
 *   byte must be split into two nibbles (high, low).  The corresponding
 *   decode_stream pass recovers those nibbles, which are then re-packed
 *   into bytes before the next decode pass.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

/* ── constants ──────────────────────────────────────────────────────────── */
#define MAGIC        0xC0u
#define HEADER_BYTES 4        /* magic + settings(2) + pass_count */
#define MAX_PASSES   4

#define CR_MIN  1
#define CR_MAX  7
#define SW_MIN  5
#define SW_MAX  80
#define SW_STEP 5
#define HW_MIN  0
#define HW_MAX  40
#define HW_STEP 5

#define NSEEDS       16   /* 4-bit seed  */
#define WINDOW        8   /* look-ahead  (3-bit offset → values 0..7) */
#define BLOCK         8   /* max slots per block */
#define HISTORY_LEN   4

/* ── settings pack / unpack ─────────────────────────────────────────────── */
static u16 pack_settings(int cr, int sw, int hw)
{
    return (u16)(((cr - CR_MIN)           << 13) |
                 (((sw - SW_MIN) / SW_STEP) <<  9) |
                 (((hw - HW_MIN) / HW_STEP) <<  5));
}

static void unpack_settings(u16 p, int *cr, int *sw, int *hw)
{
    *cr = ((p >> 13) & 0x07) + CR_MIN;
    *sw = ((p >>  9) & 0x0F) * SW_STEP + SW_MIN;
    *hw = ((p >>  5) & 0x0F) * HW_STEP + HW_MIN;
}

/* ── PRNG ───────────────────────────────────────────────────────────────── */
static int G_CR = 3;
static int G_SW = 40;
static int G_HW = 15;

typedef struct {
    u32 lcg;
    u8  last;
    u8  history[HISTORY_LEN];
    int hpos;
    u32 bias[16];
} NibblePRNG;

static u32 lcg_step(u32 *s, u8 fb)
{
    *s = *s * 1664525u + 1013904223u + (u32)fb;
    return *s;
}

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

/* ── lookup table ───────────────────────────────────────────────────────── */
static int enc_seed[16][16];
static int enc_off [16][16];
static NibblePRNG g_prng;

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

static int table_coverage(void)
{
    int n = 0;
    for (int a = 0; a < 16; a++)
        for (int b = 0; b < 16; b++)
            if (enc_seed[a][b] >= 0) n++;
    return n;
}

/* ── bit-stream ─────────────────────────────────────────────────────────── */
typedef struct { u8 *buf; int cap; int bp; } BS;

static void bs_init(BS *b, u8 *buf, int cap)
{
    b->buf = buf; b->cap = cap; b->bp = 0;
    memset(buf, 0, (size_t)cap);
}

static void bs_w(BS *b, u32 v, int n)
{
    for (int i = 0; i < n; i++) {
        int by = b->bp / 8, bi = b->bp % 8;
        if (by < b->cap && ((v >> i) & 1u))
            b->buf[by] |= (u8)(1u << bi);
        b->bp++;
    }
}

static u32 bs_r(BS *b, int n)
{
    u32 v = 0;
    for (int i = 0; i < n; i++) {
        int by = b->bp / 8, bi = b->bp % 8;
        if (by < b->cap && ((b->buf[by] >> bi) & 1u))
            v |= (1u << i);
        b->bp++;
    }
    return v;
}

static int bs_bytes(const BS *b) { return (b->bp + 7) / 8; }

/* ── nibble pack / unpack ───────────────────────────────────────────────── */
/*
 * bytes_to_nibbles: split each byte into (high nibble, low nibble).
 *   out[] must have capacity >= 2 * len.
 *   Returns 2 * len.
 */
static int bytes_to_nibbles(const u8 *in, int len, u8 *out)
{
    for (int i = 0; i < len; i++) {
        out[2*i    ] = (in[i] >> 4) & 0x0Fu;   /* high nibble first */
        out[2*i + 1] =  in[i]       & 0x0Fu;   /* low  nibble second */
    }
    return 2 * len;
}

/*
 * nibbles_to_bytes: re-pack pairs of nibbles into bytes.
 *   nib_len must be even; out[] must have capacity >= nib_len / 2.
 *   Returns nib_len / 2.
 */
static int nibbles_to_bytes(const u8 *in, int nib_len, u8 *out)
{
    int out_len = nib_len / 2;
    for (int i = 0; i < out_len; i++)
        out[i] = (u8)((in[2*i] << 4) | (in[2*i + 1] & 0x0Fu));
    return out_len;
}

/* ── block encode / decode ──────────────────────────────────────────────── */
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
                nslots++; ndone += 2;
                continue;
            }
        }
        comp[nslots]  = 0;
        raw_s[nslots] = (u8)(in[ndone] & 0xF);
        nslots++; ndone++;
    }
    if (in_used) *in_used = ndone;

    u8 flags = 0;
    for (int i = 0; i < nslots; i++)
        if (comp[i]) flags |= (u8)(1u << i);

    u8 pbuf[16]; memset(pbuf, 0, sizeof pbuf);
    BS pbs; bs_init(&pbs, pbuf, (int)sizeof pbuf);
    for (int i = 0; i < nslots; i++) {
        if (comp[i]) { bs_w(&pbs, seed_s[i], 4); bs_w(&pbs, off_s[i], 3); }
        else            bs_w(&pbs, raw_s[i], 4);
    }

    int comp_bytes = 2 + bs_bytes(&pbs);
    int raw_bytes  = 2 + ndone;

    if (comp_bytes >= raw_bytes) {
        out[0] = 0xFF; out[1] = (u8)ndone;
        for (int i = 0; i < ndone; i++) out[2 + i] = in[i] & 0x0Fu;
        return raw_bytes;
    }
    out[0] = flags; out[1] = (u8)nslots;
    memcpy(out + 2, pbuf, bs_bytes(&pbs));
    return comp_bytes;
}

static int decode_block(const u8 *in, int in_len, u8 *out, int *consumed)
{
    if (in_len < 2) return 0;

    if (in[0] == 0xFF) {
        int cnt = (int)in[1];
        if (cnt > in_len - 2) cnt = in_len - 2;
        for (int i = 0; i < cnt; i++) out[i] = in[2 + i] & 0x0Fu;
        *consumed = 2 + cnt;
        return cnt;
    }

    u8  flags  = in[0];
    int nslots = (int)in[1];
    BS  pbs;
    pbs.buf = (u8 *)(in + 2); pbs.cap = in_len - 2; pbs.bp = 0;

    int pos = 0;
    for (int sl = 0; sl < nslots; sl++) {
        if ((flags >> sl) & 1u) {
            u8  seed = (u8)bs_r(&pbs, 4);
            int off  = (int)bs_r(&pbs, 3);
            nibble_seed(&g_prng, seed);
            u8 a = nibble_next(&g_prng);
            for (int i = 0; i < off; i++) nibble_next(&g_prng);
            u8 b = nibble_next(&g_prng);
            out[pos++] = a;
            out[pos++] = b;
        } else {
            out[pos++] = (u8)bs_r(&pbs, 4);
        }
    }

    /* recompute consumed from exact bit count (mirrors encoder) */
    int bits = 0;
    for (int sl = 0; sl < nslots; sl++)
        bits += ((flags >> sl) & 1u) ? 7 : 4;
    *consumed = 2 + (bits + 7) / 8;
    return pos;
}

/* ── stream encode / decode ─────────────────────────────────────────────── */
static int encode_stream(const u8 *in, int ilen, u8 *out, int ocap)
{
    int ip = 0, op = 0;
    while (ip < ilen) {
        int chunk   = (ilen - ip < BLOCK) ? (ilen - ip) : BLOCK;
        u8  blk[BLOCK * 2 + 8];
        int in_used = 0;
        int blen    = encode_block(in + ip, chunk, blk, &in_used);
        if (op + blen > ocap) break;
        memcpy(out + op, blk, blen);
        op += blen; ip += in_used;
    }
    return op;
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

/* ── tuner ──────────────────────────────────────────────────────────────── */
typedef struct { int cr, sw, hw, csz; float ratio; } Settings;

#define TUNE_SAMPLE 512

static Settings tune(const u8 *data, int len)
{
    int slen = (len < TUNE_SAMPLE) ? len : TUNE_SAMPLE;
    Settings best = { CR_MIN, SW_MIN, HW_MIN, slen * 3, 999.0f };

    u8 *enc = (u8 *)malloc((size_t)(slen * 3 + 64));
    if (!enc) return best;

    for (int cr = CR_MIN; cr <= CR_MAX; cr++)
    for (int sw = SW_MIN; sw <= SW_MAX; sw += SW_STEP)
    for (int hw = HW_MIN; hw <= HW_MAX; hw += HW_STEP) {
        G_CR = cr; G_SW = sw; G_HW = hw;
        build_table();
        int sz = encode_stream(data, slen, enc, slen * 3 + 64);
        if (sz < best.csz) {
            best.cr    = cr; best.sw = sw; best.hw = hw;
            best.csz   = sz;
            best.ratio = (float)sz / (float)slen;
        }
    }
    free(enc);
    return best;
}

/* ── compress ───────────────────────────────────────────────────────────── */
/*
 * compress()
 *
 *   Pass 0 operates on the raw nibble input (values 0x0..0xF).
 *   Each subsequent pass operates on a nibble expansion of the previous
 *   compressed byte output: each byte is split into (high_nibble, low_nibble)
 *   before being fed to encode_stream.
 *
 *   File layout:
 *     [0]       MAGIC
 *     [1..2]    settings for pass 0
 *     [3]       passes (1..MAX_PASSES)
 *     [4..3+2*(passes-1)]  settings for passes 1..passes-1
 *     remainder payload  (raw bytes of innermost compressed stream)
 *
 *   If no pass helps, raw data is returned unchanged and *used = 0.
 */
int compress(const u8 *in, int ilen, u8 *out, int ocap, int *used)
{
    *used = 0;

    /*
     * We need four working buffers:
     *   enc_a/enc_b : ping-pong encode output buffers so that a failed
     *                 speculative encode pass doesn't clobber the last
     *                 good result.
     *   nib_a/nib_b : ping-pong nibble-expansion buffers (bytes → nibbles
     *                 between encode passes).
     *
     * enc_cap: worst-case encode output ≤ 1.25 × nibble input.
     *   Pass 0 input is ilen nibbles → output ≤ 1.25*ilen bytes.
     *   Each subsequent pass works on a smaller stream, so enc_cap = ilen*2
     *   is a safe upper bound for all passes.
     * nib_cap: nibble expansion doubles the byte count, bounded by pass-0
     *   output which is < ilen bytes → nib_cap = ilen*2 + margin.
     */
    int enc_cap = ilen * 2 + 256;
    int nib_cap = ilen * 2 + 256;

    u8 *enc_a = (u8 *)malloc((size_t)enc_cap);
    u8 *enc_b = (u8 *)malloc((size_t)enc_cap);
    u8 *nib_a = (u8 *)malloc((size_t)nib_cap);
    u8 *nib_b = (u8 *)malloc((size_t)nib_cap);
    if (!enc_a || !enc_b || !nib_a || !nib_b) {
        free(enc_a); free(enc_b); free(nib_a); free(nib_b);
        return 0;
    }

    /*
     * cur_nibs / cur_nib_len : the nibble stream to encode this pass.
     * Pass 0: the raw input (already nibble-valued).
     * Pass p>0: nibble-expansion of the previous pass's byte output.
     */
    const u8 *cur_nibs    = in;
    int        cur_nib_len = ilen;

    /*
     * enc_good / enc_good_len : the byte output of the last *successful*
     * pass.  enc_try is where we write the current speculative attempt.
     * By ping-ponging enc_good and enc_try we ensure a failed attempt
     * never overwrites the last good result.
     */
    u8 *enc_good     = enc_a;   /* holds last confirmed good encode */
    u8 *enc_try      = enc_b;   /* scratch for the next attempt      */
    int enc_good_len = 0;

    int pass_cr[MAX_PASSES], pass_sw[MAX_PASSES], pass_hw[MAX_PASSES];
    int passes = 0;

    /* nib_cur / nib_alt : ping-pong nibble expansion buffers */
    u8 *nib_cur = nib_a;
    u8 *nib_alt = nib_b;

    for (int p = 0; p < MAX_PASSES; p++) {
        Settings s = tune(cur_nibs, cur_nib_len);
        G_CR = s.cr; G_SW = s.sw; G_HW = s.hw;
        build_table();

        /* Write this attempt into enc_try (not enc_good) */
        int enc_len = encode_stream(cur_nibs, cur_nib_len, enc_try, enc_cap);

        /*
         * Compare against the byte length of the previous state.
         * Pass 0: compare enc_len vs ilen.
         * Pass p>0: compare enc_len vs enc_good_len.
         */
        int prev_bytes = (passes == 0) ? ilen : enc_good_len;
        if (enc_len >= prev_bytes) break;   /* pass didn't help — stop */

        pass_cr[passes] = s.cr;
        pass_sw[passes] = s.sw;
        pass_hw[passes] = s.hw;
        passes++;

        /* Promote enc_try → enc_good by swapping pointers */
        u8 *tmp  = enc_good; enc_good = enc_try; enc_try = tmp;
        enc_good_len = enc_len;

        if (p + 1 < MAX_PASSES) {
            /* expand bytes → nibbles for next pass */
            int nn = bytes_to_nibbles(enc_good, enc_good_len, nib_cur);
            cur_nibs    = nib_cur;
            cur_nib_len = nn;
            /* swap ping-pong nibble buffers */
            u8 *ntmp = nib_cur; nib_cur = nib_alt; nib_alt = ntmp;
        }
    }

    if (passes == 0) {
        int n = (ilen <= ocap) ? ilen : ocap;
        memcpy(out, in, n);
        free(enc_a); free(enc_b); free(nib_a); free(nib_b);
        return ilen;
    }

    *used = 1;

    /* build output header */
    int op = 0;
    out[op++] = (u8)MAGIC;
    u16 p0    = pack_settings(pass_cr[0], pass_sw[0], pass_hw[0]);
    out[op++] = (u8)(p0 >> 8);
    out[op++] = (u8)(p0 & 0xFF);
    out[op++] = (u8)passes;

    for (int p = 1; p < passes; p++) {
        u16 pk = pack_settings(pass_cr[p], pass_sw[p], pass_hw[p]);
        out[op++] = (u8)(pk >> 8);
        out[op++] = (u8)(pk & 0xFF);
    }

    /* enc_good holds the last successful encode output, enc_good_len its size */
    if (op + enc_good_len <= ocap) memcpy(out + op, enc_good, enc_good_len);
    op += enc_good_len;

    free(enc_a); free(enc_b); free(nib_a); free(nib_b);
    return op;
}

/* ── decompress ─────────────────────────────────────────────────────────── */
/*
 * decompress()
 *
 *   Decode order (reverse of encode):
 *     stored_bytes
 *       → decode(pass[N-1]) → nibble stream
 *       → nibbles_to_bytes  → byte stream
 *       → decode(pass[N-2]) → nibble stream
 *       → nibbles_to_bytes  → byte stream
 *       → ...
 *       → decode(pass[0])   → raw nibble stream (= original input)
 *
 *   If input doesn't start with MAGIC it is returned as-is.
 */
int decompress(const u8 *in, int ilen, u8 *out, int ocap)
{
    if (ilen < 1) return 0;
    if (in[0] != MAGIC) {
        int n = (ilen < ocap) ? ilen : ocap;
        memcpy(out, in, n);
        return n;
    }
    if (ilen < HEADER_BYTES) return 0;

    int cr0, sw0, hw0;
    unpack_settings(((u16)in[1] << 8) | in[2], &cr0, &sw0, &hw0);
    int passes = (int)in[3];
    if (passes < 1 || passes > MAX_PASSES) return 0;

    int pass_cr[MAX_PASSES], pass_sw[MAX_PASSES], pass_hw[MAX_PASSES];
    pass_cr[0] = cr0; pass_sw[0] = sw0; pass_hw[0] = hw0;

    int hdr_off = HEADER_BYTES;
    for (int p = 1; p < passes; p++) {
        if (hdr_off + 2 > ilen) return 0;
        u16 pk = ((u16)in[hdr_off] << 8) | in[hdr_off + 1];
        unpack_settings(pk, &pass_cr[p], &pass_sw[p], &pass_hw[p]);
        hdr_off += 2;
    }

    const u8 *payload = in + hdr_off;
    int        plen   = ilen - hdr_off;

    /*
     * Buffer sizing:
     *   The payload is a compressed byte stream. Decoding it produces a
     *   nibble stream of up to 2× the byte count. Re-packing nibbles→bytes
     *   halves it again. So across all passes the maximum intermediate size
     *   is bounded by the original ilen * 2 (nibble expand) with margin.
     *
     *   We allocate nib_cap for nibble buffers and byte_cap for byte buffers.
     *   Original data is at most ocap nibbles.
     */
    int nib_cap  = (ocap + plen) * 2 + 256;
    int byte_cap = nib_cap / 2 + 256;

    u8 *nib_a  = (u8 *)malloc((size_t)nib_cap);
    u8 *nib_b  = (u8 *)malloc((size_t)nib_cap);
    u8 *byte_buf = (u8 *)malloc((size_t)byte_cap);
    if (!nib_a || !nib_b || !byte_buf) {
        free(nib_a); free(nib_b); free(byte_buf);
        return 0;
    }

    /*
     * We decode pass[passes-1] first (innermost), down to pass[0] (outermost).
     *
     * After decode(pass[p]):  we have a nibble stream.
     *   - If p > 0: nibbles_to_bytes → byte stream, then feed as-is into
     *               decode(pass[p-1]).
     *     Wait — no. Let's think carefully about what each pass's input is.
     *
     * Encode sequence for 2 passes:
     *   raw_nibs -[encode pass0]-> bytes_0 -[bytes_to_nibbles]->
     *   nibs_0   -[encode pass1]-> bytes_1  (stored as payload)
     *
     * Decode sequence:
     *   bytes_1 -[decode pass1]-> nibs_0 -[nibbles_to_bytes]->
     *   bytes_0 -[decode pass0]-> raw_nibs
     *
     * So after each decode pass (except the last / pass[0]):
     *   the nibble output must be packed back into bytes via nibbles_to_bytes,
     *   and those bytes are the input to the next decode pass.
     *
     * For pass[0] (the final decode): the output is the original raw nibbles.
     */

    /* Start: payload bytes → first decode (pass[passes-1]) */
    /* nib_cur will hold the current nibble stream being decoded FROM (as bytes) */
    /* We keep the "current byte stream to decode" in byte_buf,
       and the "decoded nibble stream" in nib_a/nib_b ping-pong.            */

    /* Copy payload into byte_buf as the initial byte input */
    if (plen > byte_cap) plen = byte_cap;
    memcpy(byte_buf, payload, plen);
    int cur_byte_len = plen;

    u8 *nib_cur = nib_a;
    u8 *nib_alt = nib_b;

    for (int p = passes - 1; p >= 0; p--) {
        G_CR = pass_cr[p]; G_SW = pass_sw[p]; G_HW = pass_hw[p];
        build_table();

        /* Decode the current byte stream into a nibble stream */
        int nib_len = decode_stream(byte_buf, cur_byte_len, nib_cur, nib_cap);

        if (p == 0) {
            /* Final pass: nib_cur holds the original raw nibbles */
            int out_len = (nib_len < ocap) ? nib_len : ocap;
            memcpy(out, nib_cur, out_len);
            free(nib_a); free(nib_b); free(byte_buf);
            return out_len;
        }

        /* Intermediate pass: pack nibbles back into bytes for next decode */
        /* nib_len must be even (each byte was split into 2 nibbles on encode) */
        int new_byte_len = nibbles_to_bytes(nib_cur, nib_len, byte_buf);
        cur_byte_len = new_byte_len;

        /* swap ping-pong nibble buffers */
        u8 *tmp = nib_cur; nib_cur = nib_alt; nib_alt = tmp;
    }

    /* Should not reach here if passes >= 1 */
    free(nib_a); free(nib_b); free(byte_buf);
    return 0;
}

/* ════════════════════════════════════════════════════════════════════════════
 * TEST + VERIFICATION HARNESS
 * ════════════════════════════════════════════════════════════════════════════ */

/* simple xorshift32 for test data */
static u32 xr = 0x12345678u;
static u8 xrand_nib(void)
{
    xr ^= xr << 13; xr ^= xr >> 17; xr ^= xr << 5;
    return xr & 0xFu;
}

/* ── run_test ──────────────────────────────────────────────────────────── */
static int run_test(const char *name, const u8 *data, int len)
{
    int max_out = len * 4 + 256;
    u8 *comp = (u8 *)malloc((size_t)max_out);
    u8 *dec  = (u8 *)malloc((size_t)(len * 16 + 256));
    if (!comp || !dec) { free(comp); free(dec); return 0; }

    int used  = 0;
    int clen  = compress(data, len, comp, max_out, &used);
    int dlen  = decompress(comp, clen, dec, len * 16 + 256);

    /* byte-by-byte verify */
    int ok        = (dlen == len);
    int first_bad = -1;
    if (ok) {
        for (int i = 0; i < len; i++) {
            if ((data[i] & 0xF) != (dec[i] & 0xF)) {
                ok = 0;
                if (first_bad < 0) first_bad = i;
            }
        }
    }

    /* report */
    if (used) {
        int passes = (int)comp[3];
        printf("  %-38s  %4d -> %4d B  (%.1f%%)  passes=%d  %s\n",
               name, len, clen,
               100.0f * clen / len,
               passes,
               ok ? "OK" : "MISMATCH");
    } else {
        printf("  %-38s  %4d -> %4d B  (raw)           %s\n",
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

/* ── main ──────────────────────────────────────────────────────────────── */
int main(void)
{
    int pass = 0, fail = 0;

    /* ── 1. settings pack/unpack ── */
    {
        int errs = 0;
        for (int cr=CR_MIN;cr<=CR_MAX;cr++)
        for (int sw=SW_MIN;sw<=SW_MAX;sw+=SW_STEP)
        for (int hw=HW_MIN;hw<=HW_MAX;hw+=HW_STEP) {
            int cr2,sw2,hw2;
            unpack_settings(pack_settings(cr,sw,hw), &cr2,&sw2,&hw2);
            if (cr2!=cr||sw2!=sw||hw2!=hw) errs++;
        }
        printf("settings pack/unpack (%d combinations): %s\n",
               (CR_MAX-CR_MIN+1)*((SW_MAX-SW_MIN)/SW_STEP+1)*((HW_MAX-HW_MIN)/HW_STEP+1),
               errs ? "FAIL" : "OK");
    }

    /* ── 2. single-pass block round-trip ── */
    {
        printf("\nsingle-block encode/decode:\n");
        G_CR=3; G_SW=40; G_HW=15; build_table();
        u8 input[]  = {0x1,0x2,0x3,0x4,0x5,0x6,0x7,0x8,
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

    /* ── 3. full compress/decompress suite ── */
    printf("\nfull compress/decompress:\n");
    printf("  %-38s  %18s  passes  rt\n", "test", "size");
    printf("  %s\n", "----------------------------------------------------------------------");

    #define T(label, data, len) \
        do { int r = run_test(label, data, len); if(r) pass++; else fail++; } while(0)

    /* random (worst case) */
    xr = 0xDEADBEEFu;
    { enum{N=256}; u8 d[N]; for(int i=0;i<N;i++) d[i]=xrand_nib(); T("random 256",d,N); }

    xr = 0xCAFEBABEu;
    { enum{N=512}; u8 d[N]; for(int i=0;i<N;i++) d[i]=xrand_nib(); T("random 512",d,N); }

    xr = 0x99887766u;
    { enum{N=1024}; u8 *d=(u8*)malloc(N); for(int i=0;i<N;i++) d[i]=xrand_nib();
      T("random 1024",d,N); free(d); }

    xr = 0x11223344u;
    { enum{N=4096}; u8 *d=(u8*)malloc(N); for(int i=0;i<N;i++) d[i]=xrand_nib();
      T("random 4096",d,N); free(d); }

    /* biased / clustered */
    xr = 0xABCDu;
    { enum{N=256}; u8 d[N];
      for(int i=0;i<N;i++){
          u32 r=(xr^=xr<<13,xr^=xr>>17,xr^=xr<<5,xr);
          d[i]=(r&0xFF)<178 ? (u8)(4+(r>>8)%4) : xrand_nib();
      }
      T("biased 70% in {4-7} (256)",d,N); }

    /* repeating patterns */
    { enum{N=128}; u8 d[N]; for(int i=0;i<N;i++) d[i]=(u8)(i%5);
      T("pattern 01234... (128)",d,N); }

    { enum{N=256}; u8 d[N]; for(int i=0;i<N;i++) d[i]=(u8)(i%16);
      T("pattern 0-F repeat (256)",d,N); }

    { enum{N=128}; u8 d[N]; for(int i=0;i<N;i++) d[i]=(u8)((i/4)%16);
      T("slow ramp (128)",d,N); }

    /* constant */
    { enum{N=64};  u8 d[N]; memset(d,0x07,N); T("all 7s (64)",d,N); }
    { enum{N=128}; u8 d[N]; memset(d,0x00,N); T("all 0s (128)",d,N); }

    /* tiny edge cases */
    { u8 d[]={0xA}; T("single nibble",d,1); }
    { u8 d[]={0xA,0xB}; T("two nibbles",d,2); }
    { u8 d[]={0x1,0x2,0x3}; T("three nibbles",d,3); }

    /* real-ish data: nibbles of incrementing bytes */
    { enum{N=256}; u8 d[N];
      for(int i=0;i<N;i++) d[i]=(u8)((i%256)&0xF);
      T("sawtooth nibbles (256)",d,N); }

    /* alternating */
    { enum{N=128}; u8 d[N]; for(int i=0;i<N;i++) d[i]=(u8)(i&1?0xA:0x5);
      T("alternating 5/A (128)",d,N); }

    printf("  %s\n", "----------------------------------------------------------------------");
    printf("\nresult: %d passed, %d failed\n", pass, fail);

    return fail > 0 ? 1 : 0;
}
