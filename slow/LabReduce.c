/*
 * instrlab.c — instruction-testing lab for entropy-reducing byte transforms
 *
 * Stripped-down sibling of reduce2.c, built for fast iteration on NEW
 * instruction ideas rather than for a production bitstream. Differences
 * from reduce2.c:
 *   - single hardcoded input file, first BLOCK bytes only (one block, no
 *     multi-block aggregation, no threading, no bitstream packing)
 *   - ONE PASS: every registered instruction's search() runs exactly once
 *     against the untouched input block. No greedy stacking, no repeated
 *     re-search after applying a move — this is a screening tool, not a
 *     compressor. It just tells you, for THIS block, what every
 *     instruction's best move and net bit-savings would be.
 *   - overhead is priced per instruction: TAG_BITS (6, same for every type)
 *     + that type's own amp field width, +, ONLY for instructions that are
 *     genuinely stride-based, a stride field (6 bits, 1..64 range) + phase
 *     (log2(stride) bits). Use oh_strided()/oh_flat() when writing a new
 *     search function.
 *
 *   IMPORTANT correctness rule, learned the hard way while adding this
 *   batch: a transform that only looks at a byte's OWN bits — cross-nibble,
 *   cross-bit, self-shift-fold, "condition on my own value" tricks like
 *   VALUE_XOR/POPCOUNT_XOR — is STILL just some fixed function g(v) applied
 *   identically to every byte, no matter how cleverly g is built internally.
 *   If g happens to be a bijection (nearly all of these are, since that's
 *   what makes them reversible) and it's applied to 100% of the block, it's
 *   just alphabet relabeling — provably zero effect on order-0 entropy,
 *   exactly like a plain whole-block GF_MUL. These types MUST use
 *   oh_strided() (stride/phase restriction) to have any chance of mattering.
 *   Only genuinely cross-BYTE transforms (DELTA, CBC_XOR, WORD_ADD16,
 *   BLOCKDIFF2, FEISTEL_HALF — anything where byte i's output depends on
 *   ANOTHER byte's value) or position-varying ones (PRNG streams,
 *   FIXED_KS_XOR, WINDOW_XOR, RUNPARITY_XOR, ADAPT_LMS) can safely use
 *   oh_flat() and touch 100% of the block.
 *
 *   Registry pattern — 3 functions + 1 line, nothing else to touch:
 *     apply(d, n, stride, phase, amp)   — forward transform, in place
 *     invert(d, n, stride, phase, amp)  — inverse transform, in place
 *     search(d, n, Sb, out)             — sweeps this type's parameter
 *                                          space, fills out->stride/phase
 *                                          /amp with the best combo, and
 *                                          returns NET bits saved (raw
 *                                          entropy gain minus this type's
 *                                          own overhead, via oh_strided()/
 *                                          oh_flat() — NOT raw S delta;
 *                                          the framework does no further
 *                                          adjustment). Don't set
 *                                          out->type, the framework does.
 *   Then add { "NAME", search_x, ap_x, inv_x } to REGISTRY[] below.
 *   Self-inverse transforms just repeat the apply fn as both apply/invert.
 *
 * Build:  gcc -O2 -o instrlab instrlab.c -lm
 * Run:    ./instrlab            (one pass over the hardcoded block)
 *         ./instrlab selftest   (apply/invert round-trip check per type)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#ifdef _OPENMP
#include <omp.h>
#else
/* fall back to serial behavior if built without -fopenmp */
static inline int omp_get_max_threads(void) { return 1; }
static inline int omp_get_thread_num(void) { return 0; }
#endif

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;

#define BLOCK      4096
#define MAX_STRIDE 64

static const char *INPUT_FILE = "C:\\Users\\lukac\\Documents\\compressor\\compresseddata.bin";

/* ============================================================ *
 *  entropy machinery (identical math to reduce2.c)              *
 * ============================================================ */

static double hlog[BLOCK + 1];
static void init_hlog(void) {
    hlog[0] = 0.0;
    for (int k = 1; k <= BLOCK; k++) hlog[k] = (double)k * log2((double)k);
}
static double S_from_freq(const int *f) {
    double s = 0.0;
    for (int v = 0; v < 256; v++) s += hlog[f[v]];
    return s;
}
static void freq_of(const u8 *d, int n, int *f) {
    memset(f, 0, 256 * sizeof(int));
    for (int i = 0; i < n; i++) f[d[i]]++;
}
static double S_of(const u8 *d, int n) {
    int f[256]; freq_of(d, n, f);
    return S_from_freq(f);
}
static double entropy_bits(const u8 *d, int n) {
    return (double)n * log2((double)n) - S_of(d, n);
}

/* ============================================================ *
 *  overhead pricing: tag + stride + phase + amp                 *
 * ============================================================ */

#define TAG_BITS    5.0   /* flat per-instruction type tag, every type pays this -- placeholder, fix properly once NREG is large */
#define STRIDE_BITS 6.0   /* stride field, fixed width for the 1..64 range */

/* log2(s) lookup table -- phase_bits() is called from the innermost stride
 * loop of nearly every search() function (every candidate stride, every
 * layer), so it's by far the hottest log2() call site in the program.
 * Widest observed stride bound is AFFINE_FULL_MAX_STRIDE=96; table covers
 * 0..255 with headroom, falling back to log2() itself outside that range. */
static double log2_tab[256];
static void init_log2_tab(void) {
    log2_tab[0] = 0.0;
    log2_tab[1] = 0.0;
    for (int k = 2; k < 256; k++) log2_tab[k] = log2((double)k);
}
static inline double phase_bits(int s) {
    if (s <= 1) return 0.0;
    if (s < 256) return log2_tab[s];
    return log2((double)s);
}
/* strided instruction: tag + stride + phase + amp_bits */
static inline double oh_strided(double amp_bits, int s) {
    return TAG_BITS + STRIDE_BITS + phase_bits(s) + amp_bits;
}
/* whole-block instruction, no stride/phase field (e.g. PRNG streams) */
static inline double oh_flat(double amp_bits) {
    return TAG_BITS + amp_bits;
}

/* ============================================================ *
 *  instruction record                                           *
 * ============================================================ */

typedef struct { int type; int stride, phase; u32 amp; } Instr;

/* xorshift16 stream byte, state advanced per call (used by PRNG_ADD4/XOR4) */
static inline u8 xs16_next(u16 *s) {
    u16 x = *s;
    x ^= x << 7; x ^= x >> 9; x ^= x << 8;
    *s = x;
    return (u8)(x ^ (x >> 8));
}
#define PRNG_SEEDS 65536

/* ============================================================ *
 *  GF(256) multiply, x^8+x^4+x^3+x+1 -- shared by several instructions *
 * ============================================================ */

static u8 gf_mul_tab[256][256];
static void init_gf_mul_tab(void) {
    for (int av = 0; av < 256; av++) {
        for (int bv = 0; bv < 256; bv++) {
            u8 a = (u8)av, b = (u8)bv, p = 0;
            for (int i = 0; i < 8; i++) {
                if (b & 1) p ^= a;
                u8 hi = (u8)(a & 0x80);
                a = (u8)(a << 1);
                if (hi) a ^= 0x1B;
                b = (u8)(b >> 1);
            }
            gf_mul_tab[av][bv] = p;
        }
    }
}
static inline u8 gf_mul(u8 a, u8 b) { return gf_mul_tab[a][b]; }
static u8 gf_inv(u8 a) {
    for (int c = 1; c < 256; c++) if (gf_mul(a, (u8)c) == 1) return (u8)c;
    return 1; /* unreachable for a != 0 */
}

/* ============================================================ *
 *  example instructions -- add new ones the same way             *
 * ============================================================ */

/* ---- XOR_PHASE: d[i] ^= amp at i = phase, phase+stride, ...  self-inverse ---- */
/* amp domain: 1..255 (255 values) -> 8 bits */
static void ap_xorp(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] ^= (u8)amp;
}
static double search_xorp(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 1;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[u ^ a] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- STRIDE_ADD: d[i] += amp (mod 256) at (stride,phase); inverse subtracts ---- */
/* amp domain: 1..255 (255 values) -> 8 bits */
static void ap_strideadd(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] = (u8)(d[i] + amp);
}
static void inv_strideadd(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] = (u8)(d[i] - amp);
}
static double search_strideadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 1;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[(u + a) & 255] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- BYTE_ROT: circular left rotate by k=1..7, whole block (not stride-based) ---- */
/* amp domain: 1..7 (7 values) -> 3 bits */
static inline u8 byterot_fwd(u8 v, int k) { return (u8)((v << k) | (v >> (8 - k))); }
static inline u8 byterot_inv(u8 v, int k) { return (u8)((v >> k) | (v << (8 - k))); }
static void ap_byterot(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] = byterot_fwd(d[i], (int)amp);
}
static void inv_byterot(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] = byterot_inv(d[i], (int)amp);
}
static double search_byterot(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double oh = oh_flat(3.0);
    double best = -1e18; u32 bk = 1;
    for (int k = 1; k <= 7; k++) {
        int rf[256] = {0};
        for (int u = 0; u < 256; u++) rf[byterot_fwd((u8)u, k)] += total[u];
        double net = (S_from_freq(rf) - Sb) - oh;
        if (net > best) { best = net; bk = (u32)k; }
    }
    out->stride = 0; out->phase = 0; out->amp = bk;
    return best;
}

/* ---- BYTE_MUL: multiply by odd constant mod 256, whole block (not stride-based) ---- */
/* amp domain: odd 3..255 (127 values) -> 7 bits */
static u8 mul_inv256(u8 a) {
    u8 x = 1;
    x = (u8)(x * (2 - a * x));  /* mod 4   */
    x = (u8)(x * (2 - a * x));  /* mod 16  */
    x = (u8)(x * (2 - a * x));  /* mod 256 */
    return x;
}
static void ap_bytemul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] * a);
}
static void inv_bytemul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 inv = mul_inv256((u8)amp);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] * inv);
}
static double search_bytemul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double oh = oh_flat(7.0);
    double best = -1e18; u32 ba = 3;
    for (int a = 3; a < 256; a += 2) {  /* odd only, skip a=1 (identity) */
        int rf[256] = {0};
        for (int u = 0; u < 256; u++) rf[(u * a) & 0xFF] += total[u];
        double net = (S_from_freq(rf) - Sb) - oh;
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- GF_MUL: multiply by nonzero constant in GF(256) ---- */
/* amp domain: 2..255 (254 values) -> 8 bits */
static void ap_gfmul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int i = 0; i < n; i++) d[i] = gf_mul(d[i], a);
}
static void inv_gfmul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 ainv = gf_inv((u8)amp);
    for (int i = 0; i < n; i++) d[i] = gf_mul(d[i], ainv);
}
static double search_gfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double oh = oh_flat(8.0);
    double best = -1e18; u32 ba = 2;
    for (int a = 2; a < 256; a++) {  /* skip a=0 (annihilates) and a=1 (identity) */
        int rf[256] = {0};
        const u8 *col = gf_mul_tab[a];
        for (int u = 0; u < 256; u++) rf[col[u]] += total[u];
        double net = (S_from_freq(rf) - Sb) - oh;
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- BIT_REV: reverse bit order within each byte, whole block; self-inverse ---- */
/* no amp -> 0 bits; not stride-based, so this is a single fixed candidate, no sweep */
static inline u8 bitrev8(u8 v) {
    v = (u8)((v & 0xF0) >> 4 | (v & 0x0F) << 4);
    v = (u8)((v & 0xCC) >> 2 | (v & 0x33) << 2);
    v = (u8)((v & 0xAA) >> 1 | (v & 0x55) << 1);
    return v;
}
static void ap_bitrev(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i < n; i++) d[i] = bitrev8(d[i]);
}
static double search_bitrev(const u8 *d, int n, double Sb, Instr *out) {
    static u8 brtab[256]; static int init = 0;
    if (!init) { for (int v = 0; v < 256; v++) brtab[v] = bitrev8((u8)v); init = 1; }
    int total[256]; freq_of(d, n, total);
    int rf[256] = {0};
    for (int v = 0; v < 256; v++) rf[brtab[v]] += total[v];
    double net = (S_from_freq(rf) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- PRNG_ADD4: 4-stream xorshift16 ADD, stream picked by pos%4, seed searched exhaustively ---- */
/* whole-block, no stride/phase; amp domain: seed 1..65535 -> 16 bits */
static void ap_prngadd4(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[4]; st[0] = ms; st[1] = xs16_next(&ms); st[2] = xs16_next(&ms); st[3] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + xs16_next(&st[i & 3]));
}
static void inv_prngadd4(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[4]; st[0] = ms; st[1] = xs16_next(&ms); st[2] = xs16_next(&ms); st[3] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - xs16_next(&st[i & 3]));
}
static double search_prngadd4(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[4]; s[0] = ms; s[1] = xs16_next(&ms); s[2] = xs16_next(&ms); s[3] = xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + xs16_next(&s[i & 3]))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- PRNG_XOR4: same 4-stream derivation as PRNG_ADD4, XOR instead of ADD -- self-inverse ---- */
static void ap_prngxor4(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[4]; st[0] = ms; st[1] = xs16_next(&ms); st[2] = xs16_next(&ms); st[3] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] ^= xs16_next(&st[i & 3]);
}
static double search_prngxor4(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[4]; s[0] = ms; s[1] = xs16_next(&ms); s[2] = xs16_next(&ms); s[3] = xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ xs16_next(&s[i & 3]))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- PRNG_ADD<N> / PRNG_XOR<N> family: same xorshift16 multi-stream
 * mechanism as PRNG_ADD4/8/2/16, filling in stream counts 3,5,6,7,32 --
 * stream index picked by pos%N (modulo, since N isn't always a power of
 * 2) instead of a bitmask. Generated via macro: identical mechanism,
 * only the stream count differs. */
#define DEFINE_PRNG_ADDN(SUF, NSTR) \
static void ap_prngadd##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u16 ms = (u16)(amp & 0xFFFF); \
    u16 st[NSTR]; st[0] = ms; for (int k = 1; k < (NSTR); k++) st[k] = xs16_next(&ms); \
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + xs16_next(&st[i % (NSTR)])); \
} \
static void inv_prngadd##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u16 ms = (u16)(amp & 0xFFFF); \
    u16 st[NSTR]; st[0] = ms; for (int k = 1; k < (NSTR); k++) st[k] = xs16_next(&ms); \
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - xs16_next(&st[i % (NSTR)])); \
} \
static double search_prngadd##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double oh = oh_flat(16.0); \
    double best = -1e18; u32 bseed = 1; \
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) { \
        u16 ms = (u16)seed; \
        u16 s[NSTR]; s[0] = ms; for (int k = 1; k < (NSTR); k++) s[k] = xs16_next(&ms); \
        int f[256] = {0}; \
        for (int i = 0; i < n; i++) f[(u8)(d[i] + xs16_next(&s[i % (NSTR)]))]++; \
        double net = (S_from_freq(f) - Sb) - oh; \
        if (net > best) { best = net; bseed = seed; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = bseed; \
    return best; \
}
#define DEFINE_PRNG_XORN(SUF, NSTR) \
static void ap_prngxor##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u16 ms = (u16)(amp & 0xFFFF); \
    u16 st[NSTR]; st[0] = ms; for (int k = 1; k < (NSTR); k++) st[k] = xs16_next(&ms); \
    for (int i = 0; i < n; i++) d[i] ^= xs16_next(&st[i % (NSTR)]); \
} \
static double search_prngxor##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double oh = oh_flat(16.0); \
    double best = -1e18; u32 bseed = 1; \
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) { \
        u16 ms = (u16)seed; \
        u16 s[NSTR]; s[0] = ms; for (int k = 1; k < (NSTR); k++) s[k] = xs16_next(&ms); \
        int f[256] = {0}; \
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ xs16_next(&s[i % (NSTR)]))]++; \
        double net = (S_from_freq(f) - Sb) - oh; \
        if (net > best) { best = net; bseed = seed; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = bseed; \
    return best; \
}
DEFINE_PRNG_ADDN(3, 3)
DEFINE_PRNG_XORN(3, 3)
DEFINE_PRNG_ADDN(5, 5)
DEFINE_PRNG_XORN(5, 5)
DEFINE_PRNG_ADDN(6, 6)
DEFINE_PRNG_XORN(6, 6)
DEFINE_PRNG_ADDN(7, 7)
DEFINE_PRNG_XORN(7, 7)
DEFINE_PRNG_ADDN(32, 32)
DEFINE_PRNG_XORN(32, 32)

/* ---- VALUE_XOR: bit k of v is preserved; the OTHER 7 bits get XORed by alo
 * (bit k=0) or ahi (bit k=1), both with bit k forced to 0 so it never
 * changes. Self-inverse. Still a pointwise bijection of v alone -> needs
 * stride/phase like every plain operator (see header note). Search uses an
 * independent-group optimization: since alo only ever touches the bit-k=0
 * half of the value space and ahi only the bit-k=1 half, and those halves
 * never share output bins, the best alo and best ahi can be found
 * independently instead of jointly (128+128 candidates instead of 128*128). */
static inline u8 vx_insert0(int idx, int k) {
    int mask = (1 << k) - 1;
    return (u8)((idx & mask) | ((idx >> k) << (k + 1)));
}
static void ap_valuexor(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)(amp & 7);
    u8 mask = (u8)(1 << k);
    u8 alo = (u8)((amp >> 3) & 0xFF);
    u8 ahi = (u8)((amp >> 11) & 0xFF);
    for (int i = p; i < n; i += s)
        d[i] ^= (u8)((d[i] & mask) ? ahi : alo);
}
static double search_valuexor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(17.0, s);  /* k(3) + aidx(7) + bidx(7) */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k = 0; k < 8; k++) {
                u8 mask = (u8)(1 << k);
                double bestlo = -1e18; int baidx = 0;
                for (int aidx = 0; aidx < 128; aidx++) {
                    u8 alo = vx_insert0(aidx, k);
                    double s0 = 0.0;
                    for (int v = 0; v < 256; v++) if (!(v & mask))
                        s0 += hlog[base[v] + hit[v ^ alo]];
                    if (s0 > bestlo) { bestlo = s0; baidx = aidx; }
                }
                double besthi = -1e18; int bbidx = 0;
                for (int bidx = 0; bidx < 128; bidx++) {
                    u8 ahi = vx_insert0(bidx, k);
                    double s1 = 0.0;
                    for (int v = 0; v < 256; v++) if (v & mask)
                        s1 += hlog[base[v] + hit[v ^ ahi]];
                    if (s1 > besthi) { besthi = s1; bbidx = bidx; }
                }
                double net = (bestlo + besthi - Sb) - oh;
                if (net > best) {
                    best = net; bs = s; bp = p;
                    ba = (u32)k | ((u32)vx_insert0(baidx, k) << 3) | ((u32)vx_insert0(bbidx, k) << 11);
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- POPCOUNT_XOR: hi nibble (never touched) supplies a fixed
 * popcount-threshold condition (T=2, "at least half the hi-nibble bits
 * set"); lo nibble gets XORed by alo or ahi accordingly. Self-inverse.
 * Same pointwise-bijection family as VALUE_XOR -> needs stride/phase. T is
 * fixed rather than searched to keep the search tractable (searching T too
 * would 5x the cost for little expected gain). */
#define POPCXOR_T 2
static void ap_popcxor(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        int hipc = __builtin_popcount((unsigned)((v >> 4) & 0xF));
        u8 x = (hipc >= POPCXOR_T) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_popcxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    static int hipc_of[256]; static int hinit = 0;
    if (!hinit) { for (int v = 0; v < 256; v++) hipc_of[v] = __builtin_popcount((unsigned)((v >> 4) & 0xF)); hinit = 1; }
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);  /* alo(4) + ahi(4) */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (hipc_of[u] >= POPCXOR_T) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- NIB_CXOR: cross-nibble XOR. dir=0: lo^=hi (hi untouched); dir=1:
 * hi^=lo (lo untouched). Self-inverse. Pointwise bijection -> stride/phase. */
static void ap_nibcxor(u8 *d, int n, int s, int p, u32 amp) {
    int dir = (int)(amp & 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
        if (dir == 0) lo = (u8)(lo ^ hi); else hi = (u8)(hi ^ lo);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static double search_nibcxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(1.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int dir = 0; dir < 2; dir++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    u8 lo = (u8)(u & 0xF), hi = (u8)((u >> 4) & 0xF);
                    u8 w = (dir == 0) ? (u8)((lo ^ hi) | (hi << 4)) : (u8)(lo | ((hi ^ lo) << 4));
                    rf[w] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)dir; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- NIB_CADD: cross-nibble ADD mod 16, ADD-sibling of NIB_CXOR. Not
 * self-inverse (dir=0 inverts by subtracting hi back out, dir=1 by
 * subtracting lo back out). Pointwise bijection -> stride/phase. */
static void ap_nibcadd(u8 *d, int n, int s, int p, u32 amp) {
    int dir = (int)(amp & 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
        if (dir == 0) lo = (u8)((lo + hi) & 0xF); else hi = (u8)((hi + lo) & 0xF);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibcadd(u8 *d, int n, int s, int p, u32 amp) {
    int dir = (int)(amp & 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
        if (dir == 0) lo = (u8)((lo - hi) & 0xF); else hi = (u8)((hi - lo) & 0xF);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static double search_nibcadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(1.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int dir = 0; dir < 2; dir++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    u8 lo = (u8)(u & 0xF), hi = (u8)((u >> 4) & 0xF);
                    u8 w = (dir == 0) ? (u8)(((lo + hi) & 0xF) | (hi << 4)) : (u8)(lo | (((hi + lo) & 0xF) << 4));
                    rf[w] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)dir; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- SELFSHIFT_XOR: v ^= v>>k, self-shift bit diffusion (Gray code is the
 * k=1 case). Not self-inverse for general k -- real inverse via the
 * standard "unshift-right-xor" trick: v = w ^ (w>>k) ^ (w>>2k) ^ ...
 * Pointwise bijection -> stride/phase. */
static inline u8 selfshift_fwd(u8 v, int k) { return (u8)(v ^ (v >> k)); }
static inline u8 selfshift_inv(u8 w, int k) {
    u8 v = w;
    for (int shift = k; shift < 8; shift += k) v ^= (u8)(w >> shift);
    return v;
}
static void ap_selfshift(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)amp;
    for (int i = p; i < n; i += s) d[i] = selfshift_fwd(d[i], k);
}
static void inv_selfshift(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)amp;
    for (int i = p; i < n; i += s) d[i] = selfshift_inv(d[i], k);
}
static double search_selfshift(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bk = 1;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(3.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k = 1; k <= 7; k++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[selfshift_fwd((u8)u, k)] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bk = (u32)k; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = bk;
    return best;
}

/* ---- BIT_MAJ3: target bit k (>=3) XORed with majority(bit0,bit1,bit2);
 * key bits 0,1,2 are never touched, so decoder recomputes the same
 * majority value from the untouched output. Self-inverse. Pointwise
 * bijection -> stride/phase. NOTE: reduce2.c already tried this exact
 * shape (BIT_MAJ3) and found zero value even with tag cost zeroed out --
 * kept here for completeness / to confirm the same result on this data. */
static inline int maj3(int a, int b, int c) { return (a & b) | (b & c) | (a & c); }
static void ap_bitmaj3(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)amp;
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        int m = maj3(v & 1, (v >> 1) & 1, (v >> 2) & 1);
        d[i] = (u8)(v ^ (u8)(m << k));
    }
}
static double search_bitmaj3(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bk = 3;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(3.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k = 3; k <= 7; k++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    int m = maj3(u & 1, (u >> 1) & 1, (u >> 2) & 1);
                    rf[u ^ (m << k)] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bk = (u32)k; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = bk;
    return best;
}

/* ---- BIT_MAJ3_HI: same majority-of-3 mechanism as BIT_MAJ3, but the key
 * bits are the TOP three (5,6,7) instead of the bottom three (0,1,2), and
 * the target bit k ranges 0..4 (below the key bits) instead of 3..7
 * (above them) -- a mirrored bit-position choice testing whether
 * high-bit correlation structure differs from low-bit. */
static void ap_bitmaj3_hi(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)amp;
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        int m = maj3((v >> 5) & 1, (v >> 6) & 1, (v >> 7) & 1);
        d[i] = (u8)(v ^ (u8)(m << k));
    }
}
/* ---- BIT_MAJ3_K<N> family: same majority-of-3 mechanism as BIT_MAJ3
 * (key={0,1,2})/BIT_MAJ3_HI (key={5,6,7}), sliding the 3-bit contiguous
 * key window to different positions -- key={1,2,3}/{2,3,4}/{4,5,6},
 * target bit k ranges over the remaining contiguous block on one side. */
#define DEFINE_BITMAJ3_CONTIG(SUF, K0, K1, K2, TSTART, TEND) \
static void ap_bitmaj3_k##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    int k = (int)amp; \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; \
        int m = maj3((v >> (K0)) & 1, (v >> (K1)) & 1, (v >> (K2)) & 1); \
        d[i] = (u8)(v ^ (u8)(m << k)); \
    } \
} \
static double search_bitmaj3_k##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 bk = (TSTART); \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(3.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int k = (TSTART); k <= (TEND); k++) { \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int u = 0; u < 256; u++) { \
                    int m = maj3((u >> (K0)) & 1, (u >> (K1)) & 1, (u >> (K2)) & 1); \
                    rf[u ^ (m << k)] += hit[u]; \
                } \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; bk = (u32)k; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = bk; \
    return best; \
}
DEFINE_BITMAJ3_CONTIG(1, 1, 2, 3, 4, 7)
DEFINE_BITMAJ3_CONTIG(2, 2, 3, 4, 5, 7)
DEFINE_BITMAJ3_CONTIG(4, 4, 5, 6, 0, 3)

static double search_bitmaj3_hi(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bk = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(3.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k = 0; k <= 4; k++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    int m = maj3((u >> 5) & 1, (u >> 6) & 1, (u >> 7) & 1);
                    rf[u ^ (m << k)] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bk = (u32)k; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = bk;
    return best;
}

/* ---- DELTA: d[i] -= d[i-lag] (mod 256) for i>=lag; first `lag` bytes
 * untouched. Depends on ANOTHER byte's value (which varies by position), so
 * unlike the pointwise-bijection family above this genuinely escapes the
 * whole-block-bijection trap without needing stride/phase. Apply must run
 * DESCENDING (so d[i-lag] is still original when read); invert must run
 * ASCENDING (so d[i-lag] is already recovered before it's needed). */
static void ap_delta(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag = (int)amp;
    for (int i = n - 1; i >= lag; i--) d[i] = (u8)(d[i] - d[i - lag]);
}
static void inv_delta(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag = (int)amp;
    for (int i = lag; i < n; i++) d[i] = (u8)(d[i] + d[i - lag]);
}
static double search_delta(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 blag = 1;
    static u8 scr[BLOCK];
    for (int lag = 1; lag <= 64; lag++) {
        memcpy(scr, d, (size_t)n);
        for (int i = n - 1; i >= lag; i--) scr[i] = (u8)(scr[i] - scr[i - lag]);
        double net = (S_of(scr, n) - Sb) - oh_flat(6.0);
        if (net > best) { best = net; blag = (u32)lag; }
    }
    out->stride = 0; out->phase = 0; out->amp = blag;
    return best;
}

/* ---- CBC_XOR: cumulative running XOR chain, output[i] = d[i] ^ output[i-1]
 * (output[-1] = IV). Depends on the CHAIN of prior OUTPUTS, a different
 * avalanche shape than plain delta. Self-inverse pattern (own apply/invert
 * pair, not literally the same function, since IV bookkeeping differs). */
static void ap_cbcxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 prev = (u8)amp;
    for (int i = 0; i < n; i++) { d[i] ^= prev; prev = d[i]; }
}
static void inv_cbcxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 prev = (u8)amp;
    for (int i = 0; i < n; i++) { u8 cur = d[i]; d[i] ^= prev; prev = cur; }
}
static double search_cbcxor(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 biv = 0;
    static u8 scr[BLOCK];
    for (int iv = 0; iv < 256; iv++) {
        u8 prev = (u8)iv;
        for (int i = 0; i < n; i++) { scr[i] = (u8)(d[i] ^ prev); prev = scr[i]; }
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; biv = (u32)iv; }
    }
    out->stride = 0; out->phase = 0; out->amp = biv;
    return best;
}

/* ---- WINDOW_XOR: partition the block into W=16-byte windows; each
 * window's XOR-sum is invariant under uniformly XORing all W (even) bytes
 * by the same constant, so the per-window constant C_w = gf_mul(summary_w,
 * a) can be recomputed identically from the TRANSFORMED data at decode
 * time. Self-inverse. Escapes the whole-block trap because C_w varies by
 * window (position-dependent), not a single fixed constant everywhere. */
#define WXOR_W 16
static void ap_winxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int base = 0; base + WXOR_W <= n; base += WXOR_W) {
        u8 sum = 0;
        for (int j = 0; j < WXOR_W; j++) sum ^= d[base + j];
        u8 c = gf_mul(sum, a);
        for (int j = 0; j < WXOR_W; j++) d[base + j] ^= c;
    }
}
static double search_winxor(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        for (int base = 0; base + WXOR_W <= n; base += WXOR_W) {
            u8 sum = 0;
            for (int j = 0; j < WXOR_W; j++) sum ^= d[base + j];
            u8 c = gf_mul(sum, (u8)a);
            for (int j = 0; j < WXOR_W; j++) scr[base + j] = (u8)(d[base + j] ^ c);
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- RUNPARITY_XOR: running XOR-so-far of all STRICTLY PRIOR original
 * bytes selects which of two constants (alo/ahi) gets XORed into d[i],
 * based on bit 0 of that running parity. Self-inverse pattern: decode
 * processes ascending, recomputing the identical running parity from
 * already-recovered prefix bytes before undoing position i. */
static void ap_runparity(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 parity = 0;
    for (int i = 0; i < n; i++) {
        u8 orig = d[i];
        d[i] ^= (parity & 1) ? ahi : alo;
        parity ^= orig;
    }
}
static void inv_runparity(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 parity = 0;
    for (int i = 0; i < n; i++) {
        u8 orig = (u8)(d[i] ^ ((parity & 1) ? ahi : alo));
        d[i] = orig;
        parity ^= orig;
    }
}
static double search_runparity(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int alo = 0; alo < 256; alo++) {
        for (int ahi = 0; ahi < 256; ahi++) {
            u8 parity = 0;
            for (int i = 0; i < n; i++) {
                u8 orig = d[i];
                scr[i] = (u8)(orig ^ ((parity & 1) ? (u8)ahi : (u8)alo));
                parity ^= orig;
            }
            double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
            if (net > best) { best = net; ba = (u32)alo | ((u32)ahi << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- ADAPT_LMS: sign-sign LMS adaptive linear predictor. w starts at
 * w0-8 (amp in 0..15 -> w in -8..7); each step predicts p=(w*prev)>>4 from
 * the ORIGINAL previous byte (tracked in a local var, captured before
 * overwrite -- NOT re-read from d[], which would already hold the prior
 * step's residual), stores residual = d[i]-p, then nudges w by +-1 based on
 * the residual's sign. Decoder replays the identical w trajectory since it
 * only ever uses already-reconstructed data. First byte untouched. */
static void ap_lms(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w = (int)amp - 8;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)((w * (int)prev) >> 4);
        u8 residual = (u8)(orig_i - pred);
        d[i] = residual;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w += sgn; if (w > 7) w = 7; if (w < -8) w = -8;
        prev = orig_i;
    }
}
static void inv_lms(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w = (int)amp - 8;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 residual = d[i];
        u8 pred = (u8)((w * (int)prev) >> 4);
        u8 orig_i = (u8)(residual + pred);
        d[i] = orig_i;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w += sgn; if (w > 7) w = 7; if (w < -8) w = -8;
        prev = orig_i;
    }
}
static double search_lms(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bw0 = 8;
    static u8 scr[BLOCK];
    for (int w0 = 0; w0 < 16; w0++) {
        memcpy(scr, d, (size_t)n);
        int w = w0 - 8;
        u8 prev = scr[0];
        for (int i = 1; i < n; i++) {
            u8 orig_i = scr[i];
            u8 pred = (u8)((w * (int)prev) >> 4);
            u8 residual = (u8)(orig_i - pred);
            scr[i] = residual;
            int r = (int)residual;
            int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
            w += sgn; if (w > 7) w = 7; if (w < -8) w = -8;
            prev = orig_i;
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(4.0);
        if (net > best) { best = net; bw0 = (u32)w0; }
    }
    out->stride = 0; out->phase = 0; out->amp = bw0;
    return best;
}

/* ---- FIXED_KS_XOR: XOR against a fixed, hardcoded pseudo-random table --
 * no seed, no search. Both encoder and decoder regenerate the SAME table
 * from a fixed constant, so this costs only the type tag, zero amp bits.
 * Escapes the whole-block trap because the table varies by POSITION,
 * exactly like a PRNG stream, just without a searchable seed. */
static u8 fixed_keystream[BLOCK];
static void init_fixed_keystream(void) {
    u16 st = 0xC0FF;
    for (int i = 0; i < BLOCK; i++) fixed_keystream[i] = xs16_next(&st);
}
static void ap_fixedks(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i < n; i++) d[i] ^= fixed_keystream[i];
}
static double search_fixedks(const u8 *d, int n, double Sb, Instr *out) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[(u8)(d[i] ^ fixed_keystream[i])]++;
    double net = (S_from_freq(f) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- WORD_ADD16: treat adjacent byte pairs as one big-endian 16-bit word,
 * add a 16-bit constant mod 65536. Genuinely escapes the whole-block trap
 * (unlike a bitwise word-XOR, which would just decompose into two
 * independent byte-level XORs) because ADD's carry can propagate from the
 * low byte into the high byte -- real cross-byte coupling. */
static void ap_wordadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 c = (u16)amp;
    for (int i = 0; i + 1 < n; i += 2) {
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]);
        w = (u16)(w + c);
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF);
    }
}
static void inv_wordadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 c = (u16)amp;
    for (int i = 0; i + 1 < n; i += 2) {
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]);
        w = (u16)(w - c);
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF);
    }
}
static double search_wordadd(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bc = 1;
    static u8 scr[BLOCK];
    for (u32 c = 1; c < 65536; c++) {
        for (int i = 0; i + 1 < n; i += 2) {
            u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]);
            w = (u16)(w + (u16)c);
            scr[i] = (u8)(w >> 8); scr[i + 1] = (u8)(w & 0xFF);
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
        if (net > best) { best = net; bc = c; }
    }
    out->stride = 0; out->phase = 0; out->amp = bc;
    return best;
}

/* ---- BLOCKDIFF2: fixed invertible 2x2 GF(256) MDS-style diffusion matrix
 * applied to adjacent byte pairs -- genuine cross-byte mixing (b0' depends
 * on BOTH b0 and b1), escapes the whole-block trap the same way DELTA does.
 * No search parameter: matrix is fixed at compile time, single candidate. */
#define BD_M00 2
#define BD_M01 1
#define BD_M10 1
#define BD_M11 1
static u8 bd_inv00, bd_inv01, bd_inv10, bd_inv11;
static void init_blockdiff(void) {
    u8 det = (u8)(gf_mul(BD_M00, BD_M11) ^ gf_mul(BD_M01, BD_M10));
    u8 di = gf_inv(det);
    bd_inv00 = gf_mul(di, BD_M11);
    bd_inv01 = gf_mul(di, BD_M01);
    bd_inv10 = gf_mul(di, BD_M10);
    bd_inv11 = gf_mul(di, BD_M00);
}
static void ap_blockdiff(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i + 1 < n; i += 2) {
        u8 b0 = d[i], b1 = d[i + 1];
        d[i]     = (u8)(gf_mul(BD_M00, b0) ^ gf_mul(BD_M01, b1));
        d[i + 1] = (u8)(gf_mul(BD_M10, b0) ^ gf_mul(BD_M11, b1));
    }
}
static void inv_blockdiff(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i + 1 < n; i += 2) {
        u8 b0 = d[i], b1 = d[i + 1];
        d[i]     = (u8)(gf_mul(bd_inv00, b0) ^ gf_mul(bd_inv01, b1));
        d[i + 1] = (u8)(gf_mul(bd_inv10, b0) ^ gf_mul(bd_inv11, b1));
    }
}
static double search_blockdiff(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_blockdiff(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- FEISTEL_HALF: single Feistel round on the two 2048-byte halves L,R:
 * L[i] ^= GF_mul(R[i], a); R stays untouched. R serves as the untouched-
 * sibling "key" (Axis H) -- since XOR-by-anything is always invertible,
 * this works for any a, not just bijective mixing functions. Self-inverse
 * (R never touched, so re-applying cancels the first pass). */
static void ap_feistel(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    int half = n / 2;
    for (int i = 0; i < half; i++) d[i] ^= gf_mul(d[half + i], a);
}
static double search_feistel(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    int half = n / 2;
    for (int a = 1; a < 256; a++) {
        for (int i = 0; i < half; i++) scr[i] = (u8)(d[i] ^ gf_mul(d[half + i], (u8)a));
        memcpy(scr + half, d + half, (size_t)(n - half));
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- PATTERN_XOR: explicit 8-bit repeating pattern (period 8) selects
 * which positions get XORed by a shared constant -- a genuinely different
 * subset shape than modular stride/phase (arbitrary bit pattern instead of
 * "every s-th position"). amp packs pattern(8) + xor constant(8) = 16
 * bits; whole-block (the pattern itself IS the positional selector, no
 * separate stride field needed). */
#define PATXA_L 8
static void ap_patxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 pat = (u8)(amp & 0xFF);
    u8 c = (u8)((amp >> 8) & 0xFF);
    for (int i = 0; i < n; i++)
        if ((pat >> (i % PATXA_L)) & 1) d[i] ^= c;
}
static double search_patxor(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 ba = 0;
    for (int pat = 1; pat < 256; pat++) {
        int hit[256] = {0}, tot[256] = {0};
        for (int i = 0; i < n; i++) {
            tot[d[i]]++;
            if ((pat >> (i % PATXA_L)) & 1) hit[d[i]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v];
        for (int c = 1; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; ba = (u32)pat | ((u32)c << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- DIAG_XOR: reshape 4096 -> 64x64 grid, select a diagonal (constant
 * row-col, offset -63..63), XOR that diagonal's bytes by a shared constant.
 * A subset shape only visible once the block is viewed as 2D -- invisible
 * to both linear stride and row/column-aligned selection. */
#define GRID_N 64
static void ap_diagxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)n;
    int off = (int)(amp & 0x7F) - 63;
    u8 c = (u8)((amp >> 7) & 0xFF);
    for (int row = 0; row < GRID_N; row++) {
        int col = row - off;
        if (col >= 0 && col < GRID_N) d[row * GRID_N + col] ^= c;
    }
}
static double search_diagxor(const u8 *d, int n, double Sb, Instr *out) {
    (void)n;
    double oh = oh_flat(15.0);
    double best = -1e18; u32 ba = 0;
    int total[256]; freq_of(d, BLOCK, total);
    for (int off = -63; off <= 63; off++) {
        int hit[256] = {0};
        for (int row = 0; row < GRID_N; row++) {
            int col = row - off;
            if (col >= 0 && col < GRID_N) hit[d[row * GRID_N + col]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        for (int c = 1; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; ba = (u32)(off + 63) | ((u32)c << 7); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- DIAG_XOR_W32: same diagonal-XOR mechanism as DIAG_XOR, but on a
 * 32x128 grid instead of 64x64 -- a rectangular grid makes "diagonal"
 * mean a very different subset of byte positions than the square grid,
 * since row-col spans -31..127 rather than -63..63. */
#define DXW_W 32
#define DXW_H 128
static void ap_diagxor_w32(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)n;
    int off = (int)(amp & 0xFF) - (DXW_W - 1);
    u8 c = (u8)((amp >> 8) & 0xFF);
    for (int row = 0; row < DXW_H; row++) {
        int col = row - off;
        if (col >= 0 && col < DXW_W) d[row * DXW_W + col] ^= c;
    }
}
static double search_diagxor_w32(const u8 *d, int n, double Sb, Instr *out) {
    (void)n;
    double oh = oh_flat(16.0);
    double best = -1e18; u32 ba = 0;
    int total[256]; freq_of(d, BLOCK, total);
    for (int off = -(DXW_W - 1); off <= (DXW_H - 1); off++) {
        int hit[256] = {0};
        for (int row = 0; row < DXW_H; row++) {
            int col = row - off;
            if (col >= 0 && col < DXW_W) hit[d[row * DXW_W + col]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        for (int c = 1; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; ba = (u32)(off + (DXW_W - 1)) | ((u32)c << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- DIAG_ADD_<grid>_<dir> family: generalizes DIAG_ADD (64x64, main
 * diagonal) to every other valid grid factorization of 4096=2^12
 * (2x2048,4x1024,8x512,16x256,32x128,128x32,256x16,512x8,1024x4,2048x2 --
 * 64x64 is the only one already used) crossed with BOTH diagonal
 * directions (main: row-col; anti: row+col). 11 grids x 2 directions =
 * 22 combos, minus the 1 already existing (64x64 main) = 21 genuinely
 * distinct new instructions -- the natural parameter space here tops
 * out at 21, not 32; padding further would mean contrived duplicates of
 * the same mechanism rather than a real new partition. */
#define DEFINE_DIAG_ADD_GRID(SUF, WIDTH, HEIGHT, ANTI) \
static void ap_diagadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)n; \
    int offmin = (ANTI) ? 0 : -((WIDTH) - 1); \
    int off = (int)(amp & 0xFFF) + offmin; \
    u8 c = (u8)((amp >> 12) & 0xFF); \
    for (int row = 0; row < (HEIGHT); row++) { \
        int col = (ANTI) ? (off - row) : (row - off); \
        if (col >= 0 && col < (WIDTH)) d[row * (WIDTH) + col] = (u8)(d[row * (WIDTH) + col] + c); \
    } \
} \
static void inv_diagadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)n; \
    int offmin = (ANTI) ? 0 : -((WIDTH) - 1); \
    int off = (int)(amp & 0xFFF) + offmin; \
    u8 c = (u8)((amp >> 12) & 0xFF); \
    for (int row = 0; row < (HEIGHT); row++) { \
        int col = (ANTI) ? (off - row) : (row - off); \
        if (col >= 0 && col < (WIDTH)) d[row * (WIDTH) + col] = (u8)(d[row * (WIDTH) + col] - c); \
    } \
} \
static double search_diagadd_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    (void)n; \
    double oh = oh_flat(20.0); \
    double best = -1e18; u32 ba = 0; \
    int total[256]; freq_of(d, BLOCK, total); \
    int offmin = (ANTI) ? 0 : -((WIDTH) - 1); \
    int offmax = (ANTI) ? ((WIDTH) + (HEIGHT) - 2) : ((HEIGHT) - 1); \
    for (int off = offmin; off <= offmax; off++) { \
        int hit[256] = {0}; \
        for (int row = 0; row < (HEIGHT); row++) { \
            int col = (ANTI) ? (off - row) : (row - off); \
            if (col >= 0 && col < (WIDTH)) hit[d[row * (WIDTH) + col]]++; \
        } \
        int base[256]; \
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
        for (int c = 1; c < 256; c++) { \
            int rf[256]; memcpy(rf, base, sizeof rf); \
            for (int u = 0; u < 256; u++) rf[(u + c) & 255] += hit[u]; \
            double net = (S_from_freq(rf) - Sb) - oh; \
            if (net > best) { best = net; ba = (u32)(off - offmin) | ((u32)c << 12); } \
        } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_DIAG_ADD_GRID(g2_main,    2, 2048, 0)
DEFINE_DIAG_ADD_GRID(g2_anti,    2, 2048, 1)
DEFINE_DIAG_ADD_GRID(g4_main,    4, 1024, 0)
DEFINE_DIAG_ADD_GRID(g4_anti,    4, 1024, 1)
DEFINE_DIAG_ADD_GRID(g8_main,    8, 512,  0)
DEFINE_DIAG_ADD_GRID(g8_anti,    8, 512,  1)
DEFINE_DIAG_ADD_GRID(g16_main,   16, 256, 0)
DEFINE_DIAG_ADD_GRID(g16_anti,   16, 256, 1)
DEFINE_DIAG_ADD_GRID(g32_main,   32, 128, 0)
DEFINE_DIAG_ADD_GRID(g32_anti,   32, 128, 1)
DEFINE_DIAG_ADD_GRID(g64_anti,   64, 64,  1)
DEFINE_DIAG_ADD_GRID(g128_main,  128, 32, 0)
DEFINE_DIAG_ADD_GRID(g128_anti,  128, 32, 1)
DEFINE_DIAG_ADD_GRID(g256_main,  256, 16, 0)
DEFINE_DIAG_ADD_GRID(g256_anti,  256, 16, 1)
DEFINE_DIAG_ADD_GRID(g512_main,  512, 8,  0)
DEFINE_DIAG_ADD_GRID(g512_anti,  512, 8,  1)
DEFINE_DIAG_ADD_GRID(g1024_main, 1024, 4, 0)
DEFINE_DIAG_ADD_GRID(g1024_anti, 1024, 4, 1)
DEFINE_DIAG_ADD_GRID(g2048_main, 2048, 2, 0)
DEFINE_DIAG_ADD_GRID(g2048_anti, 2048, 2, 1)

/* ---- VALUEMAP4_XOR: top 2 bits of v (never touched) pick one of 4
 * quartiles; the lower 6 bits get XORed by that quartile's own constant.
 * Self-inverse. Pointwise bijection of v alone -> needs stride/phase. */
static void ap_valuemap4(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    for (int g = 0; g < 4; g++) c[g] = (u8)((amp >> (g * 6)) & 0x3F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int g = (v >> 6) & 3;
        d[i] = (u8)((v & 0xC0) | ((v & 0x3F) ^ c[g]));
    }
}
static double search_valuemap4(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(24.0, s);  /* 4 quartiles x 6-bit constant */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            u32 amp = 0;
            for (int g = 0; g < 4; g++) {
                /* the 4 quartiles never share output bins (top 2 bits
                 * untouched), so each quartile's best 6-bit constant can be
                 * found independently, same trick as VALUE_XOR. */
                double bestg = -1e18; int bc = 0;
                for (int c = 0; c < 64; c++) {
                    double sg = 0.0;
                    for (int lo = 0; lo < 64; lo++) {
                        int v = (g << 6) | lo;
                        sg += hlog[base[v] + hit[(g << 6) | (lo ^ c)]];
                    }
                    if (sg > bestg) { bestg = sg; bc = c; }
                }
                amp |= (u32)bc << (g * 6);
            }
            /* recompute total net now that all 4 quartiles are fixed */
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) {
                int g = (u >> 6) & 3;
                u8 c = (u8)((amp >> (g * 6)) & 0x3F);
                rf[(u & 0xC0) | ((u & 0x3F) ^ c)] += hit[u];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = amp; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- RANGE_COND_XOR: hi nibble (untouched) compared against a fixed
 * threshold T=8 picks alo or ahi to XOR into the lo nibble. Magnitude
 * threshold rather than popcount (POPCNT_XOR) or a raw bit (VALUE_XOR) --
 * a different split shape. Self-inverse. Pointwise bijection -> stride/phase. */
#define RANGE_T 8
static void ap_rangexor(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (hi >= RANGE_T) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_rangexor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hi = (u >> 4) & 0xF;
                        u8 x = (hi >= RANGE_T) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- MAGCLASS_XOR: hi nibble's bit-length (position of its highest set
 * bit, 0..4 -- an exponential/logarithmic bucketing, not a linear
 * threshold like RANGE_XOR or a bit-count like POPCNT_XOR) compared to a
 * fixed class boundary picks alo/ahi for the lo nibble. Self-inverse.
 * Pointwise bijection -> stride/phase. */
#define MAGCLASS_T 3
static inline int bitlen4(int v) {
    int n = 0;
    while (v) { n++; v >>= 1; }
    return n;
}
static void ap_magclass(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (bitlen4(hi) >= MAGCLASS_T) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_magclass(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    static int bl_of[256]; static int binit = 0;
    if (!binit) { for (int v = 0; v < 256; v++) bl_of[v] = bitlen4((v >> 4) & 0xF); binit = 1; }
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (bl_of[u] >= MAGCLASS_T) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- CRMB_CXOR: cross-crumb XOR. Crumb j (2-bit group, untouched) gets
 * XORed into crumb k (j != k). Self-inverse. Pointwise bijection -> stride/
 * phase. Finer granularity than NIB_CXOR (4 groups of 2 bits vs 2 groups
 * of 4 bits). */
static void ap_crmbcxor(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3);
        u8 ck = (u8)((v >> (2 * k)) & 3);
        u8 nck = (u8)(ck ^ cj);
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (nck << (2 * k)));
    }
}
static double search_crmbcxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);  /* j(2) + k(2) */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 cj = (u8)((u >> (2 * j)) & 3);
                        u8 ck = (u8)((u >> (2 * k)) & 3);
                        u8 w = (u8)((u & ~(u8)(3 << (2 * k))) | ((ck ^ cj) << (2 * k)));
                        rf[w] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)j | ((u32)k << 2); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- BIT_CXOR: cross-bit XOR. Bit j (untouched) gets XORed into bit k
 * (j != k). Self-inverse. Pointwise bijection -> stride/phase. Finest
 * granularity of the cross-part family (single bits, vs crumbs/nibbles). */
static void ap_bitcxor(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 7), k = (int)((amp >> 3) & 7);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        int bj = (v >> j) & 1, bk = (v >> k) & 1;
        int nbk = bk ^ bj;
        d[i] = (u8)((v & ~(u8)(1 << k)) | ((u8)nbk << k));
    }
}
static double search_bitcxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(6.0, s);  /* j(3) + k(3) */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int j = 0; j < 8; j++) {
                for (int k = 0; k < 8; k++) {
                    if (j == k) continue;
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int bj = (u >> j) & 1, bk = (u >> k) & 1;
                        int nbk = bk ^ bj;
                        u8 w = (u8)((u & ~(u8)(1 << k)) | ((u8)nbk << k));
                        rf[w] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)j | ((u32)k << 3); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- BIT_CXOR2_<j1j2> / BIT_CXOR3_<t> family: generalizes BIT_CXOR's
 * single-source cross-bit XOR to TWO or THREE fixed source bits XORed
 * together before being XORed into a searched target bit -- same
 * mechanism family as BIT_MAJ3/5 (multi-source conditioning) but with
 * XOR combine instead of majority vote. Since BIT_CXOR already
 * exhaustively searches its only free (j,k) pair in one instruction,
 * fixing the SOURCE set and searching only the target bit (mirroring
 * BIT_MAJ3_K1/K2/K4's precedent) is what gives room for many distinct
 * instructions: 28 = all C(8,2) two-source-bit pairs, plus 4 handpicked
 * three-source-bit combos, for 32 total. */
#define DEFINE_BITCXOR2(SUF, J1, J2) \
static void ap_bitcxor2_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    int k = (int)(amp & 7); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; \
        int bj = ((v >> (J1)) & 1) ^ ((v >> (J2)) & 1); \
        int bk = (v >> k) & 1; \
        int nbk = bk ^ bj; \
        d[i] = (u8)((v & ~(u8)(1 << k)) | ((u8)nbk << k)); \
    } \
} \
static double search_bitcxor2_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(3.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int k = 0; k < 8; k++) { \
                if (k == (J1) || k == (J2)) continue; \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int u = 0; u < 256; u++) { \
                    int bj = ((u >> (J1)) & 1) ^ ((u >> (J2)) & 1); \
                    int bk = (u >> k) & 1; \
                    int nbk = bk ^ bj; \
                    u8 w = (u8)((u & ~(u8)(1 << k)) | ((u8)nbk << k)); \
                    rf[w] += hit[u]; \
                } \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)k; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_BITCXOR2(p01, 0, 1)
DEFINE_BITCXOR2(p02, 0, 2)
DEFINE_BITCXOR2(p03, 0, 3)
DEFINE_BITCXOR2(p04, 0, 4)
DEFINE_BITCXOR2(p05, 0, 5)
DEFINE_BITCXOR2(p06, 0, 6)
DEFINE_BITCXOR2(p07, 0, 7)
DEFINE_BITCXOR2(p12, 1, 2)
DEFINE_BITCXOR2(p13, 1, 3)
DEFINE_BITCXOR2(p14, 1, 4)
DEFINE_BITCXOR2(p15, 1, 5)
DEFINE_BITCXOR2(p16, 1, 6)
DEFINE_BITCXOR2(p17, 1, 7)
DEFINE_BITCXOR2(p23, 2, 3)
DEFINE_BITCXOR2(p24, 2, 4)
DEFINE_BITCXOR2(p25, 2, 5)
DEFINE_BITCXOR2(p26, 2, 6)
DEFINE_BITCXOR2(p27, 2, 7)
DEFINE_BITCXOR2(p34, 3, 4)
DEFINE_BITCXOR2(p35, 3, 5)
DEFINE_BITCXOR2(p36, 3, 6)
DEFINE_BITCXOR2(p37, 3, 7)
DEFINE_BITCXOR2(p45, 4, 5)
DEFINE_BITCXOR2(p46, 4, 6)
DEFINE_BITCXOR2(p47, 4, 7)
DEFINE_BITCXOR2(p56, 5, 6)
DEFINE_BITCXOR2(p57, 5, 7)
DEFINE_BITCXOR2(p67, 6, 7)
#define DEFINE_BITCXOR3(SUF, J1, J2, J3) \
static void ap_bitcxor3_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    int k = (int)(amp & 7); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; \
        int bj = ((v >> (J1)) & 1) ^ ((v >> (J2)) & 1) ^ ((v >> (J3)) & 1); \
        int bk = (v >> k) & 1; \
        int nbk = bk ^ bj; \
        d[i] = (u8)((v & ~(u8)(1 << k)) | ((u8)nbk << k)); \
    } \
} \
static double search_bitcxor3_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(3.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int k = 0; k < 8; k++) { \
                if (k == (J1) || k == (J2) || k == (J3)) continue; \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int u = 0; u < 256; u++) { \
                    int bj = ((u >> (J1)) & 1) ^ ((u >> (J2)) & 1) ^ ((u >> (J3)) & 1); \
                    int bk = (u >> k) & 1; \
                    int nbk = bk ^ bj; \
                    u8 w = (u8)((u & ~(u8)(1 << k)) | ((u8)nbk << k)); \
                    rf[w] += hit[u]; \
                } \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)k; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_BITCXOR3(t012, 0, 1, 2)
DEFINE_BITCXOR3(t345, 3, 4, 5)
DEFINE_BITCXOR3(t135, 1, 3, 5)
DEFINE_BITCXOR3(t246, 2, 4, 6)
DEFINE_BITCXOR3(t013, 0, 1, 3)
DEFINE_BITCXOR3(t014, 0, 1, 4)
DEFINE_BITCXOR3(t015, 0, 1, 5)
DEFINE_BITCXOR3(t016, 0, 1, 6)
DEFINE_BITCXOR3(t017, 0, 1, 7)
DEFINE_BITCXOR3(t023, 0, 2, 3)
DEFINE_BITCXOR3(t024, 0, 2, 4)
DEFINE_BITCXOR3(t025, 0, 2, 5)
DEFINE_BITCXOR3(t026, 0, 2, 6)
DEFINE_BITCXOR3(t027, 0, 2, 7)
DEFINE_BITCXOR3(t034, 0, 3, 4)
DEFINE_BITCXOR3(t035, 0, 3, 5)
DEFINE_BITCXOR3(t036, 0, 3, 6)
DEFINE_BITCXOR3(t037, 0, 3, 7)
DEFINE_BITCXOR3(t045, 0, 4, 5)
DEFINE_BITCXOR3(t046, 0, 4, 6)
DEFINE_BITCXOR3(t047, 0, 4, 7)
DEFINE_BITCXOR3(t056, 0, 5, 6)
DEFINE_BITCXOR3(t057, 0, 5, 7)
DEFINE_BITCXOR3(t067, 0, 6, 7)
DEFINE_BITCXOR3(t123, 1, 2, 3)
DEFINE_BITCXOR3(t124, 1, 2, 4)
DEFINE_BITCXOR3(t125, 1, 2, 5)
DEFINE_BITCXOR3(t126, 1, 2, 6)
DEFINE_BITCXOR3(t127, 1, 2, 7)
DEFINE_BITCXOR3(t134, 1, 3, 4)
DEFINE_BITCXOR3(t136, 1, 3, 6)
DEFINE_BITCXOR3(t137, 1, 3, 7)
DEFINE_BITCXOR3(t145, 1, 4, 5)
DEFINE_BITCXOR3(t146, 1, 4, 6)
DEFINE_BITCXOR3(t147, 1, 4, 7)
DEFINE_BITCXOR3(t156, 1, 5, 6)

/* ---- DELTA2: second-order delta, d[i] -= 2*d[i-1] - d[i-2] (mod 256) for
 * i>=2; first 2 bytes untouched. No searched parameter (fixed formula).
 * Same descending-apply / ascending-invert pattern as DELTA, just using
 * two priors -- must read ORIGINAL d[i-1],d[i-2] before they're
 * overwritten (descending), and recover them before they're needed
 * (ascending). */
static void ap_delta2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = n - 1; i >= 2; i--)
        d[i] = (u8)(d[i] - 2 * d[i - 1] + d[i - 2]);
}
static void inv_delta2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 2; i < n; i++)
        d[i] = (u8)(d[i] + 2 * d[i - 1] - d[i - 2]);
}
static double search_delta2(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_delta2(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- LINPRED2: linear predictor from two priors with SEARCHED integer
 * coefficients: pred = (a*d[i-1] + b*d[i-2]) >> 2, residual = d[i]-pred,
 * for i>=2. a,b in -4..3 (searched). Generalizes DELTA2's fixed (2,-1)
 * coefficients. Same descending-apply / ascending-invert pattern. */
static void ap_linpred2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int a = (int)(amp & 7) - 4, b = (int)((amp >> 3) & 7) - 4;
    for (int i = n - 1; i >= 2; i--) {
        int pred = (a * (int)d[i - 1] + b * (int)d[i - 2]) >> 2;
        d[i] = (u8)((int)d[i] - pred);
    }
}
static void inv_linpred2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int a = (int)(amp & 7) - 4, b = (int)((amp >> 3) & 7) - 4;
    for (int i = 2; i < n; i++) {
        int pred = (a * (int)d[i - 1] + b * (int)d[i - 2]) >> 2;
        d[i] = (u8)((int)d[i] + pred);
    }
}
static double search_linpred2(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int aidx = 0; aidx < 8; aidx++) {
        for (int bidx = 0; bidx < 8; bidx++) {
            memcpy(scr, d, (size_t)n);
            u32 amp = (u32)aidx | ((u32)bidx << 3);
            ap_linpred2(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh_flat(6.0);
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- EQNEIGH_XOR: condition on whether the TWO PRIOR bytes were equal
 * (d[i-1]==d[i-2]) -- NOT whether d[i] equals d[i-1], which would be
 * circular (decoder doesn't know d[i] yet, that's what it's recovering).
 * The condition is checked against ORIGINAL values tracked in local
 * variables (updated forward each step), never re-read from the array,
 * since by the time position i is processed the array may already hold
 * this same instruction's transformed values at i-1/i-2. First two bytes
 * untouched (no pair to compare yet). */
static void ap_eqneigh(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        if (i >= 2) {
            int cond = (p1 == p2);
            d[i] = (u8)(orig_i ^ (cond ? ahi : alo));
        }
        p2 = p1; p1 = orig_i;
    }
}
static void inv_eqneigh(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i;
        if (i >= 2) {
            int cond = (p1 == p2);
            orig_i = (u8)(d[i] ^ (cond ? ahi : alo));
            d[i] = orig_i;
        } else {
            orig_i = d[i];
        }
        p2 = p1; p1 = orig_i;
    }
}
static double search_eqneigh(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int alo = 0; alo < 256; alo++) {
        for (int ahi = 0; ahi < 256; ahi++) {
            u8 p1 = d[0], p2 = 0;
            for (int i = 0; i < n; i++) scr[i] = d[i];
            for (int i = 1; i < n; i++) {
                u8 orig_i = d[i];
                if (i >= 2) {
                    int cond = (p1 == p2);
                    scr[i] = (u8)(orig_i ^ ((cond ? ahi : alo)));
                }
                p2 = p1; p1 = orig_i;
            }
            double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
            if (net > best) { best = net; ba = (u32)alo | ((u32)ahi << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- MIRROR_XOR: d[i] ^= gf_mul(d[n-1-i], a) for i < n/2; the mirror
 * partner n-1-i is untouched (untouched-sibling trick again, Axis H), a
 * different pairing rule than FEISTEL_HLF's half-split (d[0] pairs with
 * d[n-1], not d[n/2]). Self-inverse. */
static void ap_mirrorxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int i = 0; i < n / 2; i++) d[i] ^= gf_mul(d[n - 1 - i], a);
}
static double search_mirrorxor(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        for (int i = 0; i < n / 2; i++) scr[i] = (u8)(d[i] ^ gf_mul(d[n - 1 - i], (u8)a));
        for (int i = n / 2; i < n; i++) scr[i] = d[i];
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- MIRROR_XOR_HALF: same untouched-sibling mirror mechanism as
 * MIRROR_XOR, but reflecting within the first HALF of the block only
 * (d[i] paired with d[n/2-1-i] for i<n/4) instead of across the whole
 * block (d[i] paired with d[n-1-i] for i<n/2) -- a shorter-range
 * reflection distance, leaving the entire second half of the block
 * untouched rather than just the second quarter. */
static void ap_mirrorxor_half(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    int h = n / 2;
    for (int i = 0; i < h / 2; i++) d[i] ^= gf_mul(d[h - 1 - i], a);
}
static double search_mirrorxor_half(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    int h = n / 2;
    for (int a = 1; a < 256; a++) {
        for (int i = 0; i < h / 2; i++) scr[i] = (u8)(d[i] ^ gf_mul(d[h - 1 - i], (u8)a));
        for (int i = h / 2; i < n; i++) scr[i] = d[i];
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- MIRROR_XOR_QUARTER: same untouched-sibling mirror mechanism as
 * MIRROR_XOR/MIRROR_XOR_HALF, one step shorter still -- reflecting
 * within the first QUARTER of the block (d[i] paired with d[n/4-1-i]
 * for i<n/8), leaving the remaining 7/8 of the block untouched. */
static void ap_mirrorxor_quarter(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    int q = n / 4;
    for (int i = 0; i < q / 2; i++) d[i] ^= gf_mul(d[q - 1 - i], a);
}
static double search_mirrorxor_quarter(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    int q = n / 4;
    for (int a = 1; a < 256; a++) {
        for (int i = 0; i < q / 2; i++) scr[i] = (u8)(d[i] ^ gf_mul(d[q - 1 - i], (u8)a));
        for (int i = q / 2; i < n; i++) scr[i] = d[i];
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- MULTINEIGH_XOR: d[i] ^= d[i-1] ^ d[i-2] (both ORIGINAL, tracked via
 * local vars updated forward, never re-read from the array) for i>=2 --
 * combines TWO priors instead of DELTA's one. NOT self-inverse: re-running
 * apply() would capture the just-written TRANSFORMED value into the
 * tracker for i>=2, corrupting the chain from i=3 onward even though each
 * individual XOR step is algebraically self-canceling. Needs a genuine
 * separate inverse that solves for orig_i instead of re-deriving it. */
static void ap_multineigh(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        if (i >= 2) d[i] = (u8)(orig_i ^ p1 ^ p2);
        p2 = p1; p1 = orig_i;
    }
}
static void inv_multineigh(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i;
        if (i >= 2) { orig_i = (u8)(d[i] ^ p1 ^ p2); d[i] = orig_i; }
        else orig_i = d[i];
        p2 = p1; p1 = orig_i;
    }
}
static double search_multineigh(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_multineigh(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- ERRDIFF_XOR: running ADD-accumulated statistic (bit 0 of the mod-256
 * running sum of original bytes so far) selects alo/ahi -- same shape as
 * RUNPARITY_XOR but accumulated via ADD instead of XOR, a genuinely
 * different running statistic (sum vs parity). Self-inverse pattern. */
static void ap_errdiff(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 acc = 0;
    for (int i = 0; i < n; i++) {
        u8 orig = d[i];
        d[i] = (u8)(orig ^ ((acc & 1) ? ahi : alo));
        acc = (u8)(acc + orig);
    }
}
static void inv_errdiff(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 acc = 0;
    for (int i = 0; i < n; i++) {
        u8 orig = (u8)(d[i] ^ ((acc & 1) ? ahi : alo));
        d[i] = orig;
        acc = (u8)(acc + orig);
    }
}
static double search_errdiff(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int alo = 0; alo < 256; alo++) {
        for (int ahi = 0; ahi < 256; ahi++) {
            u8 acc = 0;
            for (int i = 0; i < n; i++) {
                u8 orig = d[i];
                scr[i] = (u8)(orig ^ ((acc & 1) ? (u8)ahi : (u8)alo));
                acc = (u8)(acc + orig);
            }
            double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
            if (net > best) { best = net; ba = (u32)alo | ((u32)ahi << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- DETREND: subtract a linear trend pred(i) = ((a*i)>>10 + b) mod 256
 * across the block. Position used as a continuous covariate rather than a
 * fixed periodic pattern (stride) or a neighbor value (delta) -- catches
 * slow monotonic drift. No sequencing constraint (pred depends only on i,
 * not on data), so apply/invert are simple elementwise formulas. */
static inline int detrend_pred(int i, int a, int b) { return (((a * i) >> 10) + b) & 0xFF; }
static void ap_detrend(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int a = (int)(amp & 0xF) - 8, b = (int)((amp >> 4) & 0xFF);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - detrend_pred(i, a, b));
}
static void inv_detrend(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int a = (int)(amp & 0xF) - 8, b = (int)((amp >> 4) & 0xFF);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + detrend_pred(i, a, b));
}
static double search_detrend(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int aidx = 0; aidx < 16; aidx++) {
        int a = aidx - 8;
        for (int b = 0; b < 256; b++) {
            for (int i = 0; i < n; i++) scr[i] = (u8)(d[i] - detrend_pred(i, a, b));
            double net = (S_of(scr, n) - Sb) - oh_flat(12.0);  /* a(4) + b(8) */
            if (net > best) { best = net; ba = (u32)aidx | ((u32)b << 4); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- SINTREND: subtract a smooth sinusoid A*sin(2*pi*i*freq/256 + phase)
 * across the block -- sits between DETREND's monotonic trend and stride's
 * hard on/off periodic pattern, catching smooth periodic drift neither
 * handles. Integer sine table (not floating point at runtime, though the
 * table itself is float-computed once at init) for exact determinism. No
 * sequencing constraint, elementwise formula like DETREND. Search space
 * kept small (6 freqs x 32 phases x 4 amplitudes) to stay tractable. */
#define INSTRLAB_PI 3.14159265358979323846
static int sin256[256];
static void init_sin256(void) {
    for (int k = 0; k < 256; k++) sin256[k] = (int)lround(63.0 * sin(2.0 * INSTRLAB_PI * k / 256.0));
}
static const int SINFREQ[6] = { 1, 2, 4, 8, 16, 32 };
static const int SINAMP[4]  = { 8, 16, 32, 63 };
static inline int sintrend_pred(int i, int freq, int phase, int amp) {
    int idx = (phase * 8 + i * freq) & 0xFF;
    return (amp * sin256[idx]) >> 6;
}
static void ap_sintrend(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int freq = SINFREQ[amp & 7], phase = (int)((amp >> 3) & 0x1F), a = SINAMP[(amp >> 8) & 3];
    for (int i = 0; i < n; i++) d[i] = (u8)((int)d[i] - sintrend_pred(i, freq, phase, a));
}
static void inv_sintrend(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int freq = SINFREQ[amp & 7], phase = (int)((amp >> 3) & 0x1F), a = SINAMP[(amp >> 8) & 3];
    for (int i = 0; i < n; i++) d[i] = (u8)((int)d[i] + sintrend_pred(i, freq, phase, a));
}
static double search_sintrend(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int fi = 0; fi < 6; fi++) {
        for (int phase = 0; phase < 32; phase++) {
            for (int ai = 0; ai < 4; ai++) {
                for (int i = 0; i < n; i++)
                    scr[i] = (u8)((int)d[i] - sintrend_pred(i, SINFREQ[fi], phase, SINAMP[ai]));
                double net = (S_of(scr, n) - Sb) - oh_flat(10.0);  /* freq(3)+phase(5)+amp(2) */
                if (net > best) { best = net; ba = (u32)fi | ((u32)phase << 3) | ((u32)ai << 8); }
            }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- PRNG_BIT: PRNG selects WHICH single bit (0..7) to flip per byte --
 * qualitatively different from PRNG_ADD4/XOR4 (which apply a value across
 * all 8 bits via a byte stream). Self-inverse (flipping the same bit
 * twice cancels, and replaying from the same seed regenerates the
 * identical bit-position sequence). Simplified from reduce2.c's PRNG_BIT:
 * always flips the selected bit (dropping the separate flip-mask search,
 * which would make seed x fmask ~68 billion candidate-ops -- intractable
 * here; this keeps just the seed search, same cost as PRNG_ADD4). */
static inline int xs16_b3(u16 *s) {
    u16 x = *s;
    x ^= x << 7; x ^= x >> 9; x ^= x << 8;
    *s = x;
    return (int)((x ^ (x >> 8)) & 7);
}
static void ap_prngbit(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 seed = (u16)(amp & 0xFFFF);
    for (int i = 0; i < n; i++) {
        int b = xs16_b3(&seed);
        d[i] ^= (u8)(1 << b);
    }
}
static double search_prngbit(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 s = (u16)seed;
        int f[256] = {0};
        for (int i = 0; i < n; i++) {
            int b = xs16_b3(&s);
            f[(u8)(d[i] ^ (1 << b))]++;
        }
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- PRNG_SUBSET_XOR: PRNG coin-flip picks a subset of positions, and
 * every position in that subset gets XORed by the SAME fixed constant
 * (0xFF, invert all bits) -- a cheaper, less powerful cousin of the full
 * per-position PRNG streams (PRNG_ADD4/XOR4/BIT), searching only a seed
 * rather than seed+per-position-value. Self-inverse. */
static void ap_prngsubset(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 seed = (u16)(amp & 0xFFFF);
    for (int i = 0; i < n; i++) {
        u8 coin = xs16_next(&seed);
        if (coin & 1) d[i] ^= 0xFF;
    }
}
static double search_prngsubset(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 s = (u16)seed;
        int f[256] = {0};
        for (int i = 0; i < n; i++) {
            u8 coin = xs16_next(&s);
            f[(u8)(d[i] ^ ((coin & 1) ? 0xFF : 0))]++;
        }
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- HASH_XOR: XOR each byte by hash(i, seed), a STATELESS per-position
 * hash (standard Murmur3-style integer finalizer) -- unlike PRNG_ADD4/
 * XOR4/BIT/SUBSET, which all carry a running PRNG state forward, this
 * computes each position's key independently, embarrassingly parallel.
 * Self-inverse. */
static inline u8 hash_byte(u32 i, u32 seed) {
    u32 x = i ^ (seed * 0x9E3779B1u);
    x ^= x >> 16; x *= 0x85EBCA6Bu;
    x ^= x >> 13; x *= 0xC2B2AE35u;
    x ^= x >> 16;
    return (u8)x;
}
static void ap_hashxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] ^= hash_byte((u32)i, amp);
}
static double search_hashxor(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ hash_byte((u32)i, seed))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- HASH_XOR2: same stateless per-position hash mechanism as HASH_XOR,
 * but a different finalizer (SplitMix32-style constants/shifts instead
 * of Murmur3-style) -- a genuinely different bit-mixing function, not
 * just a different seed within the same mixer. Self-inverse. */
static inline u8 hash_byte2(u32 i, u32 seed) {
    u32 x = i + seed * 0x9E3779B9u;
    x ^= x >> 16; x *= 0x21F0AAADu;
    x ^= x >> 15; x *= 0x735A2D97u;
    x ^= x >> 15;
    return (u8)x;
}
static void ap_hashxor2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] ^= hash_byte2((u32)i, amp);
}
static double search_hashxor2(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ hash_byte2((u32)i, seed))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- MTF: move-to-front. Maintains a 256-entry recency list; each byte
 * is replaced by its current RANK in that list, then moved to the front.
 * Escapes the whole-block trap via running STATE (rank depends on
 * everything seen so far, not on the byte's own value alone). No searched
 * parameter. */
static void ap_mtf(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 list[256];
    for (int k = 0; k < 256; k++) list[k] = (u8)k;
    for (int i = 0; i < n; i++) {
        u8 v = d[i];
        int r = 0;
        while (list[r] != v) r++;
        d[i] = (u8)r;
        for (int k = r; k > 0; k--) list[k] = list[k - 1];
        list[0] = v;
    }
}
static void inv_mtf(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 list[256];
    for (int k = 0; k < 256; k++) list[k] = (u8)k;
    for (int i = 0; i < n; i++) {
        int r = d[i];
        u8 v = list[r];
        d[i] = v;
        for (int k = r; k > 0; k--) list[k] = list[k - 1];
        list[0] = v;
    }
}
static double search_mtf(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_mtf(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- ADPCM: adaptive-step delta modulation. Tracks a running level L
 * (clamped to 0..255) and a step size that doubles when consecutive
 * residual signs agree, halves when they disagree -- distinct from
 * ADAPT_LMS's adaptive WEIGHT (multiplicative predictor); this adapts the
 * ADDITIVE step itself. Decoder replays the identical (step, L, sign)
 * trajectory since it only ever uses the transmitted residual byte, which
 * matches what encode used to update its own state. */
static void ap_adpcm(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int step = 1 << (int)(amp & 3);
    int L = d[0];
    int last_sign = 0;
    for (int i = 1; i < n; i++) {
        int orig = d[i];
        u8 residual = (u8)(orig - L);
        d[i] = residual;
        int r = (int)residual;
        int sign = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        if (sign != 0 && sign == last_sign) { step *= 2; if (step > 32) step = 32; }
        else if (sign != 0) { step /= 2; if (step < 1) step = 1; }
        L += sign * step;
        if (L < 0) L = 0; if (L > 255) L = 255;
        last_sign = sign;
    }
}
static void inv_adpcm(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int step = 1 << (int)(amp & 3);
    int L = d[0];
    int last_sign = 0;
    for (int i = 1; i < n; i++) {
        u8 residual = d[i];
        int orig = (int)((u8)(residual + (u8)L));
        d[i] = (u8)orig;
        int r = (int)residual;
        int sign = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        if (sign != 0 && sign == last_sign) { step *= 2; if (step > 32) step = 32; }
        else if (sign != 0) { step /= 2; if (step < 1) step = 1; }
        L += sign * step;
        if (L < 0) L = 0; if (L > 255) L = 255;
        last_sign = sign;
    }
}
static double search_adpcm(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int s0 = 0; s0 < 4; s0++) {
        memcpy(scr, d, (size_t)n);
        ap_adpcm(scr, n, 0, 0, (u32)s0);
        double net = (S_of(scr, n) - Sb) - oh_flat(2.0);
        if (net > best) { best = net; ba = (u32)s0; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- EMA_BIAS: subtract a running exponential moving average of original
 * values (fixed-point, 8 fractional bits) -- a smoothed running predictor,
 * distinct from DELTA's raw-immediate-neighbor and ADPCM's step-adapted
 * level. shift controls the smoothing factor (searched, 2..5). */
static void ap_emabias(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int shift = 2 + (int)(amp & 3);
    int ema = d[0] << 8;
    for (int i = 1; i < n; i++) {
        int orig = d[i];
        int pred = ema >> 8;
        d[i] = (u8)(orig - pred);
        ema += ((orig << 8) - ema) >> shift;
    }
}
static void inv_emabias(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int shift = 2 + (int)(amp & 3);
    int ema = d[0] << 8;
    for (int i = 1; i < n; i++) {
        u8 residual = d[i];
        int pred = ema >> 8;
        int orig = (int)((u8)(residual + (u8)pred));
        d[i] = (u8)orig;
        ema += ((orig << 8) - ema) >> shift;
    }
}
static double search_emabias(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int sh = 0; sh < 4; sh++) {
        memcpy(scr, d, (size_t)n);
        ap_emabias(scr, n, 0, 0, (u32)sh);
        double net = (S_of(scr, n) - Sb) - oh_flat(2.0);
        if (net > best) { best = net; ba = (u32)sh; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- EMA_BIAS_WIDE: same running-EMA predictor as EMA_BIAS, but the
 * searched smoothing shift spans 1..8 (3 bits) instead of 2..5 (2 bits) --
 * covers both faster-adapting (shift=1) and much slower/smoother
 * (shift=8) exponential predictors than the original's narrower range. */
static void ap_emabias_wide(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int shift = 1 + (int)(amp & 7);
    int ema = d[0] << 8;
    for (int i = 1; i < n; i++) {
        int orig = d[i];
        int pred = ema >> 8;
        d[i] = (u8)(orig - pred);
        ema += ((orig << 8) - ema) >> shift;
    }
}
static void inv_emabias_wide(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int shift = 1 + (int)(amp & 7);
    int ema = d[0] << 8;
    for (int i = 1; i < n; i++) {
        u8 residual = d[i];
        int pred = ema >> 8;
        int orig = (int)((u8)(residual + (u8)pred));
        d[i] = (u8)orig;
        ema += ((orig << 8) - ema) >> shift;
    }
}
static double search_emabias_wide(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int sh = 0; sh < 8; sh++) {
        memcpy(scr, d, (size_t)n);
        ap_emabias_wide(scr, n, 0, 0, (u32)sh);
        double net = (S_of(scr, n) - Sb) - oh_flat(3.0);
        if (net > best) { best = net; ba = (u32)sh; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- PIECEWISE_ADD: split the block into K=4 contiguous chunks, each
 * with its own independently-fit ADD constant (32 bits = 4 x 8-bit
 * constants, fits exactly in the amp field) -- catches a block with
 * different local regimes and a seam between them, which one global
 * STRIDE_ADD constant can't. Search fits each chunk's constant against
 * the OTHER chunks' ORIGINAL (untransformed) histograms as an
 * approximation (the constants technically interact through the shared
 * output histogram; this greedy-independent fit is not globally optimal
 * but is fast and the actually-applied/reported net is still exact). */
#define PW_K 4
static void ap_piecewise(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int chunk = n / PW_K;
    for (int c = 0; c < PW_K; c++) {
        u8 add = (u8)((amp >> (c * 8)) & 0xFF);
        int start = c * chunk, end = (c == PW_K - 1) ? n : start + chunk;
        for (int i = start; i < end; i++) d[i] = (u8)(d[i] + add);
    }
}
static void inv_piecewise(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int chunk = n / PW_K;
    for (int c = 0; c < PW_K; c++) {
        u8 add = (u8)((amp >> (c * 8)) & 0xFF);
        int start = c * chunk, end = (c == PW_K - 1) ? n : start + chunk;
        for (int i = start; i < end; i++) d[i] = (u8)(d[i] - add);
    }
}
static double search_piecewise(const u8 *d, int n, double Sb, Instr *out) {
    int chunk = n / PW_K;
    int total[256]; freq_of(d, n, total);
    u32 amp = 0;
    for (int c = 0; c < PW_K; c++) {
        int start = c * chunk, end = (c == PW_K - 1) ? n : start + chunk;
        int hit[256] = {0};
        for (int i = start; i < end; i++) hit[d[i]]++;
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        double bestg = -1e18; int ba = 0;
        for (int a = 0; a < 256; a++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[(u + a) & 255] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; ba = a; }
        }
        amp |= (u32)ba << (c * 8);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_piecewise(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(32.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

/* ---- BITPLANE_XOR: pick one bit-plane (0..7); pack that plane's bits
 * across every group of 8 consecutive positions into a conceptual "packed
 * byte" and XOR it by a constant c -- equivalently, position i's bit
 * `plane` flips iff bit (i%8) of c is set. A period-8 pattern restricted
 * to a SINGLE bit-plane, distinct from PATTERN_XOR (whole-byte flip) and
 * from the cross-bit family (pointwise within one byte). Self-inverse. */
static void ap_bitplane(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int plane = (int)(amp & 7);
    u8 c = (u8)((amp >> 3) & 0xFF);
    u8 planemask = (u8)(1 << plane);
    for (int i = 0; i < n; i++) {
        int j = i % 8;
        if ((c >> j) & 1) d[i] ^= planemask;
    }
}
static double search_bitplane(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(11.0);  /* plane(3) + c(8) */
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int plane = 0; plane < 8; plane++) {
        for (int c = 0; c < 256; c++) {
            u32 amp = (u32)plane | ((u32)c << 3);
            memcpy(scr, d, (size_t)n);
            ap_bitplane(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh;
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- GF_POW: nonlinear sibling of GF_MUL -- raise every byte to a fixed
 * power e in GF(256)'s multiplicative group (order 255), for e coprime to
 * 255 (phi(255)=128 valid exponents, a 7-bit index with no waste).
 * Exponentiation composes multiplications nonlinearly, unlike GF_MUL's
 * single multiply. Pointwise bijection -> stride/phase. */
static u8 gf_log[256], gf_antilog[256];
static void init_gf_log(void) {
    u8 x = 1;
    for (int i = 0; i < 255; i++) {
        gf_antilog[i] = x;
        gf_log[x] = (u8)i;
        x = gf_mul(x, 0x03);
    }
}
static u8 gfpow_elist[128], gfpow_einv[128];
static int gfpow_ne;
static u8 gfpow_tab[128][256], gfpow_itab[128][256];
static inline u8 gfpow_raw(u8 v, int e) {
    if (v == 0) return 0;
    return gf_antilog[(gf_log[v] * e) % 255];
}
static void init_gfpow(void) {
    gfpow_ne = 0;
    for (int e = 1; e < 255; e++) {
        int a = e, b = 255, x0 = 1, x1 = 0;
        while (b != 0) { int q = a / b, t = b; b = a % b; a = t; t = x1; x1 = x0 - q * x1; x0 = t; }
        if (a != 1) continue;
        int einv = ((x0 % 255) + 255) % 255;
        int idx = gfpow_ne++;
        gfpow_elist[idx] = (u8)e; gfpow_einv[idx] = (u8)einv;
        for (int v = 0; v < 256; v++) {
            gfpow_tab[idx][v]  = gfpow_raw((u8)v, e);
            gfpow_itab[idx][v] = gfpow_raw((u8)v, einv);
        }
    }
}
static void ap_gfpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *col = gfpow_tab[amp & 0x7F];
    for (int i = p; i < n; i += s) d[i] = col[d[i]];
}
static void inv_gfpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *col = gfpow_itab[amp & 0x7F];
    for (int i = p; i < n; i += s) d[i] = col[d[i]];
}
static double search_gfpow(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(7.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int idx = 0; idx < gfpow_ne; idx++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                const u8 *col = gfpow_tab[idx];
                for (int u = 0; u < 256; u++) rf[col[u]] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)idx; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- AFFINE: v -> a*v+b mod 256, bundling MUL+ADD under one instruction
 * tag (Axis D). Pointwise bijection -> stride/phase. SIMPLIFIED: a full
 * joint (a,b) search across all strides is tens of billions of ops
 * (127 odd multipliers x 256 offsets x 256-merge x ~2080 stride/phase
 * pairs), far too slow here -- fixed at a=3 (smallest interesting
 * multiplier), searching only the additive offset b. */
#define AFFINE_A 3
static void ap_affine(u8 *d, int n, int s, int p, u32 amp) {
    u8 b = (u8)amp;
    for (int i = p; i < n; i += s) d[i] = (u8)(AFFINE_A * d[i] + b);
}
static void inv_affine(u8 *d, int n, int s, int p, u32 amp) {
    u8 b = (u8)amp;
    u8 ainv = mul_inv256(AFFINE_A);
    for (int i = p; i < n; i += s) d[i] = (u8)((d[i] - b) * ainv);
}
static double search_affine(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bb = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            int hitm[256] = {0};
            for (int u = 0; u < 256; u++) hitm[(AFFINE_A * u) & 0xFF] += hit[u];
            for (int b = 0; b < 256; b++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int w = 0; w < 256; w++) rf[(w + b) & 0xFF] += hitm[w];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bb = (u32)b; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = bb;
    return best;
}

/* ---- AFFINE_A<N> family: same v->a*v+b mod 256 mechanism as AFFINE/
 * AFFINE_A5/AFFINE_A7, but a DIFFERENT fixed odd multiplier `a` per
 * variant (odd is required and sufficient for a*v mod 256 to be a
 * bijection, since gcd(a,256)=1). Each is a genuinely different
 * pointwise bijection -- the full joint (a,b) search is intractable
 * (noted at AFFINE's definition), so each candidate `a` gets its own
 * instruction slot with only `b` searched, exactly like the existing
 * A5/A7 siblings. Generated via macro since the mechanism is identical
 * across all of them, only the compile-time constant differs. */
#define DEFINE_AFFINE_VARIANT(SUF, AVAL) \
static void ap_affine_v##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 b = (u8)amp; \
    for (int i = p; i < n; i += s) d[i] = (u8)((AVAL) * d[i] + b); \
} \
static void inv_affine_v##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 b = (u8)amp; \
    u8 ainv = mul_inv256((u8)(AVAL)); \
    for (int i = p; i < n; i += s) d[i] = (u8)((d[i] - b) * ainv); \
} \
static double search_affine_v##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 bb = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            int hitm[256] = {0}; \
            for (int u = 0; u < 256; u++) hitm[((AVAL) * u) & 0xFF] += hit[u]; \
            for (int b = 0; b < 256; b++) { \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int w = 0; w < 256; w++) rf[(w + b) & 0xFF] += hitm[w]; \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; bb = (u32)b; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = bb; \
    return best; \
}
DEFINE_AFFINE_VARIANT(9, 9)
DEFINE_AFFINE_VARIANT(11, 11)
DEFINE_AFFINE_VARIANT(13, 13)
DEFINE_AFFINE_VARIANT(15, 15)
DEFINE_AFFINE_VARIANT(17, 17)
DEFINE_AFFINE_VARIANT(19, 19)
DEFINE_AFFINE_VARIANT(21, 21)
DEFINE_AFFINE_VARIANT(23, 23)
DEFINE_AFFINE_VARIANT(25, 25)
DEFINE_AFFINE_VARIANT(27, 27)
DEFINE_AFFINE_VARIANT(29, 29)
DEFINE_AFFINE_VARIANT(31, 31)
DEFINE_AFFINE_VARIANT(37, 37)
DEFINE_AFFINE_VARIANT(41, 41)
DEFINE_AFFINE_VARIANT(43, 43)
DEFINE_AFFINE_VARIANT(47, 47)
DEFINE_AFFINE_VARIANT(53, 53)
DEFINE_AFFINE_VARIANT(59, 59)
DEFINE_AFFINE_VARIANT(61, 61)
DEFINE_AFFINE_VARIANT(63, 63)
DEFINE_AFFINE_VARIANT(33, 33)
DEFINE_AFFINE_VARIANT(35, 35)
DEFINE_AFFINE_VARIANT(39, 39)
DEFINE_AFFINE_VARIANT(45, 45)
DEFINE_AFFINE_VARIANT(49, 49)
DEFINE_AFFINE_VARIANT(51, 51)
DEFINE_AFFINE_VARIANT(55, 55)
DEFINE_AFFINE_VARIANT(57, 57)
DEFINE_AFFINE_VARIANT(65, 65)
DEFINE_AFFINE_VARIANT(67, 67)
DEFINE_AFFINE_VARIANT(69, 69)
DEFINE_AFFINE_VARIANT(71, 71)
DEFINE_AFFINE_VARIANT(73, 73)
DEFINE_AFFINE_VARIANT(75, 75)
DEFINE_AFFINE_VARIANT(77, 77)
DEFINE_AFFINE_VARIANT(79, 79)
DEFINE_AFFINE_VARIANT(81, 81)
DEFINE_AFFINE_VARIANT(83, 83)
DEFINE_AFFINE_VARIANT(85, 85)
DEFINE_AFFINE_VARIANT(87, 87)
DEFINE_AFFINE_VARIANT(89, 89)
DEFINE_AFFINE_VARIANT(91, 91)
DEFINE_AFFINE_VARIANT(93, 93)
DEFINE_AFFINE_VARIANT(95, 95)
DEFINE_AFFINE_VARIANT(97, 97)
DEFINE_AFFINE_VARIANT(99, 99)
DEFINE_AFFINE_VARIANT(101, 101)
DEFINE_AFFINE_VARIANT(103, 103)
DEFINE_AFFINE_VARIANT(105, 105)
DEFINE_AFFINE_VARIANT(107, 107)
DEFINE_AFFINE_VARIANT(109, 109)
DEFINE_AFFINE_VARIANT(111, 111)
DEFINE_AFFINE_VARIANT(113, 113)
DEFINE_AFFINE_VARIANT(115, 115)
DEFINE_AFFINE_VARIANT(117, 117)
DEFINE_AFFINE_VARIANT(119, 119)
DEFINE_AFFINE_VARIANT(121, 121)
DEFINE_AFFINE_VARIANT(123, 123)
DEFINE_AFFINE_VARIANT(125, 125)
DEFINE_AFFINE_VARIANT(127, 127)
DEFINE_AFFINE_VARIANT(129, 129)
DEFINE_AFFINE_VARIANT(131, 131)
DEFINE_AFFINE_VARIANT(133, 133)
DEFINE_AFFINE_VARIANT(135, 135)
DEFINE_AFFINE_VARIANT(137, 137)
DEFINE_AFFINE_VARIANT(139, 139)
DEFINE_AFFINE_VARIANT(141, 141)
DEFINE_AFFINE_VARIANT(143, 143)
DEFINE_AFFINE_VARIANT(145, 145)
DEFINE_AFFINE_VARIANT(147, 147)
DEFINE_AFFINE_VARIANT(149, 149)
DEFINE_AFFINE_VARIANT(151, 151)
DEFINE_AFFINE_VARIANT(153, 153)
DEFINE_AFFINE_VARIANT(155, 155)
DEFINE_AFFINE_VARIANT(157, 157)
DEFINE_AFFINE_VARIANT(159, 159)
DEFINE_AFFINE_VARIANT(161, 161)
DEFINE_AFFINE_VARIANT(163, 163)
DEFINE_AFFINE_VARIANT(165, 165)
DEFINE_AFFINE_VARIANT(167, 167)
DEFINE_AFFINE_VARIANT(169, 169)
DEFINE_AFFINE_VARIANT(171, 171)
DEFINE_AFFINE_VARIANT(173, 173)
DEFINE_AFFINE_VARIANT(175, 175)
DEFINE_AFFINE_VARIANT(177, 177)
DEFINE_AFFINE_VARIANT(179, 179)
DEFINE_AFFINE_VARIANT(181, 181)
DEFINE_AFFINE_VARIANT(183, 183)
DEFINE_AFFINE_VARIANT(185, 185)
DEFINE_AFFINE_VARIANT(187, 187)
DEFINE_AFFINE_VARIANT(189, 189)
DEFINE_AFFINE_VARIANT(191, 191)
DEFINE_AFFINE_VARIANT(193, 193)
DEFINE_AFFINE_VARIANT(195, 195)
DEFINE_AFFINE_VARIANT(197, 197)
DEFINE_AFFINE_VARIANT(199, 199)
DEFINE_AFFINE_VARIANT(201, 201)
DEFINE_AFFINE_VARIANT(203, 203)
DEFINE_AFFINE_VARIANT(205, 205)
DEFINE_AFFINE_VARIANT(207, 207)
DEFINE_AFFINE_VARIANT(209, 209)
DEFINE_AFFINE_VARIANT(211, 211)
DEFINE_AFFINE_VARIANT(213, 213)
DEFINE_AFFINE_VARIANT(215, 215)
DEFINE_AFFINE_VARIANT(217, 217)
DEFINE_AFFINE_VARIANT(219, 219)
DEFINE_AFFINE_VARIANT(221, 221)
DEFINE_AFFINE_VARIANT(223, 223)
DEFINE_AFFINE_VARIANT(225, 225)
DEFINE_AFFINE_VARIANT(227, 227)
DEFINE_AFFINE_VARIANT(229, 229)
DEFINE_AFFINE_VARIANT(231, 231)
DEFINE_AFFINE_VARIANT(233, 233)
DEFINE_AFFINE_VARIANT(235, 235)
DEFINE_AFFINE_VARIANT(237, 237)
DEFINE_AFFINE_VARIANT(239, 239)
DEFINE_AFFINE_VARIANT(241, 241)
DEFINE_AFFINE_VARIANT(243, 243)
DEFINE_AFFINE_VARIANT(245, 245)
DEFINE_AFFINE_VARIANT(247, 247)
DEFINE_AFFINE_VARIANT(249, 249)
DEFINE_AFFINE_VARIANT(251, 251)
DEFINE_AFFINE_VARIANT(253, 253)
DEFINE_AFFINE_VARIANT(255, 255)

/* ---- AFFINE_FULL: joint (a,b) search for v->a*v+b mod256, replacing the
 * need to hand-pick which `a` gets its own fixed instruction -- covers
 * ALL 128 valid odd multipliers instead of the ~30 we enumerated by
 * hand, while sharing the per-(stride,phase) histogram computation
 * across all of them (each separate AFFINE_A<N> instruction redundantly
 * rebuilt that same histogram from scratch).
 *
 * Two speed tricks make this tractable (an exhaustive 128a x 256b joint
 * sweep at every stride/phase is ~50B+ ops, too slow):
 *  1. PRUNING: `hlog` is nonlinear in the bin-sum, so there's no exact
 *     shortcut (no FFT/convolution trick applies, no smoothness in a to
 *     climb) -- but a cheap proxy (how concentrated is `a`'s remapped
 *     histogram onto its single heaviest bin?) correlates well with
 *     "will some b make this a good fit", so only the AFFINE_PRUNE_K
 *     most-concentrated candidates get the full exact b-sweep. This is
 *     a heuristic -- it can occasionally miss the true best `a` -- not
 *     a proof of optimality.
 *  2. FUSED B-LOOP: since v=(w+b) mod 256 is a bijection of w for any
 *     fixed b, `rf[(w+b)&0xFF] = base[(w+b)&0xFF] + hitm[w]` touches
 *     every output bin exactly once -- no separate memcpy(base) needed
 *     first, cutting the per-b inner work from 3 passes to 2.
 *
 * Stride range extended to 1..96 (vs the base MAX_STRIDE=64) since the
 * above savings create some headroom; pushed further (e.g. to 256)
 * the runtime becomes tens of minutes, not worth it for the marginal
 * extra stride coverage. */
#define AFFINE_FULL_MAX_STRIDE 96
#define AFFINE_PRUNE_K 16
static void ap_affinefull(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)(2 * (amp & 0x7F) + 1);
    u8 b = (u8)((amp >> 7) & 0xFF);
    for (int i = p; i < n; i += s) d[i] = (u8)(a * d[i] + b);
}
static void inv_affinefull(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)(2 * (amp & 0x7F) + 1);
    u8 b = (u8)((amp >> 7) & 0xFF);
    u8 ainv = mul_inv256(a);
    for (int i = p; i < n; i += s) d[i] = (u8)((d[i] - b) * ainv);
}
static double search_affinefull(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= AFFINE_FULL_MAX_STRIDE; s++) {
        double oh = oh_strided(15.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];

            /* stage 1: cheap peakiness proxy (heaviest bin under this a) */
            int cand_aidx[128], cand_score[128];
            for (int aidx = 0; aidx < 128; aidx++) {
                u8 a = (u8)(2 * aidx + 1);
                int hitm[256] = {0};
                int maxbin = 0;
                for (int u = 0; u < 256; u++) {
                    int w = (a * u) & 0xFF;
                    hitm[w] += hit[u];
                    if (hitm[w] > maxbin) maxbin = hitm[w];
                }
                cand_aidx[aidx] = aidx;
                cand_score[aidx] = maxbin;
            }
            int topn = AFFINE_PRUNE_K;
            for (int i = 0; i < topn; i++) {
                int best_j = i;
                for (int j = i + 1; j < 128; j++) if (cand_score[j] > cand_score[best_j]) best_j = j;
                int ts = cand_score[i]; cand_score[i] = cand_score[best_j]; cand_score[best_j] = ts;
                int ta = cand_aidx[i]; cand_aidx[i] = cand_aidx[best_j]; cand_aidx[best_j] = ta;
            }

            /* stage 2: exact entropy scoring, top candidates only */
            for (int t = 0; t < topn; t++) {
                int aidx = cand_aidx[t];
                u8 a = (u8)(2 * aidx + 1);
                int hitm[256] = {0};
                for (int u = 0; u < 256; u++) hitm[(a * u) & 0xFF] += hit[u];
                for (int b = 0; b < 256; b++) {
                    int rf[256];
                    for (int w = 0; w < 256; w++) {
                        int v = (w + b) & 0xFF;
                        rf[v] = base[v] + hitm[w];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)aidx | ((u32)b << 7); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VIGENERE_XOR: short repeating L=4-byte key, cycling by position mod
 * 4 -- bundles what would otherwise be 4 separate XOR_PHASE(stride=4)
 * picks under one instruction tag (Axis D). Search fits each key byte
 * independently via its own O(256) frequency trick (greedy approximation,
 * same caveat as PIECEWISE_ADD -- the 4 residue classes' output histograms
 * technically interact, this ignores that for tractability; the reported
 * net is still an exact measurement of the actually-applied transform).
 * Self-inverse. */
#define VIG_L 4
static void ap_vigenere(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 key[VIG_L];
    for (int k = 0; k < VIG_L; k++) key[k] = (u8)((amp >> (k * 8)) & 0xFF);
    for (int i = 0; i < n; i++) d[i] ^= key[i % VIG_L];
}
static double search_vigenere(const u8 *d, int n, double Sb, Instr *out) {
    u32 amp = 0;
    for (int r = 0; r < VIG_L; r++) {
        int hit[256] = {0}, tot[256] = {0};
        for (int i = 0; i < n; i++) {
            tot[d[i]]++;
            if (i % VIG_L == r) hit[d[i]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v];
        double bestg = -1e18; int bc = 0;
        for (int c = 0; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; bc = c; }
        }
        amp |= (u32)bc << (r * 8);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_vigenere(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(32.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

/* ---- RANK_XOR: hi nibble's frequency RANK among the 16 possible hi
 * values (computed once from the whole block's hi-nibble histogram --
 * hi is never touched by this transform, so decoder recomputes the
 * identical ranking) picks alo/ahi for the lo nibble. Self-inverse.
 * Pointwise bijection -> stride/phase. K (the rank threshold) fixed at 8
 * rather than searched, to keep the search tractable. */
#define RANKXOR_K 8
static void rankxor_compute_rank(const u8 *d, int n, int *rank) {
    int freq[16] = {0};
    for (int i = 0; i < n; i++) freq[(d[i] >> 4) & 0xF]++;
    int order[16]; for (int h = 0; h < 16; h++) order[h] = h;
    for (int a = 0; a < 16; a++)
        for (int b = a + 1; b < 16; b++)
            if (freq[order[b]] > freq[order[a]]) { int t = order[a]; order[a] = order[b]; order[b] = t; }
    for (int r = 0; r < 16; r++) rank[order[r]] = r;
}
static void ap_rankxor(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    int rank[16]; rankxor_compute_rank(d, n, rank);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (rank[hi] < RANKXOR_K) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_rankxor(const u8 *d, int n, double Sb, Instr *out) {
    int rank[16]; rankxor_compute_rank(d, n, rank);
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);  /* alo(4) + ahi(4) */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hi = (u >> 4) & 0xF;
                        u8 x = (rank[hi] < RANKXOR_K) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- RUNLEN_XOR: condition on whether the current position is inside a
 * run of >= K consecutive equal ORIGINAL bytes so far (K fixed at 1 here
 * to keep the search to just alo/ahi) -- richer than EQNEIGH_XOR's plain
 * "same as immediately-previous" flag, since it's a genuine run length,
 * not just a pairwise check. run/prev tracked via local vars updated with
 * ORIGINAL values, same safe pattern as EQNEIGH_XOR/MULTINEIGH_XOR. */
#define RUNLEN_K 1
static void ap_runlen(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 prev = d[0]; int run = 0;
    for (int i = 1; i < n; i++) {
        u8 orig = d[i];
        int cond = (run >= RUNLEN_K);
        d[i] = (u8)(orig ^ (cond ? ahi : alo));
        run = (orig == prev) ? run + 1 : 0;
        prev = orig;
    }
}
static void inv_runlen(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 prev = d[0]; int run = 0;
    for (int i = 1; i < n; i++) {
        int cond = (run >= RUNLEN_K);
        u8 orig = (u8)(d[i] ^ (cond ? ahi : alo));
        d[i] = orig;
        run = (orig == prev) ? run + 1 : 0;
        prev = orig;
    }
}
static double search_runlen(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int alo = 0; alo < 256; alo++) {
        for (int ahi = 0; ahi < 256; ahi++) {
            u8 prev = d[0]; int run = 0;
            scr[0] = d[0];
            for (int i = 1; i < n; i++) {
                u8 orig = d[i];
                int cond = (run >= RUNLEN_K);
                scr[i] = (u8)(orig ^ ((cond ? ahi : alo)));
                run = (orig == prev) ? run + 1 : 0;
                prev = orig;
            }
            double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
            if (net > best) { best = net; ba = (u32)alo | ((u32)ahi << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- PRNG_PERM: build a full 256-entry permutation via seeded
 * Fisher-Yates, then apply it only at positions selected by a coin-flip
 * drawn from the SAME PRNG stream continuing after the shuffle (not
 * applied to 100% of positions -- that would be a whole-block bijection,
 * useless per the Master Principle; the coin-flip subset selection is
 * what makes this work, same as every other PRNG-tier instruction). */
static void prngperm_build(u16 seed, u8 *perm, u16 *st_out) {
    for (int i = 0; i < 256; i++) perm[i] = (u8)i;
    u16 st = seed;
    for (int i = 255; i > 0; i--) {
        int j = xs16_next(&st) % (i + 1);
        u8 t = perm[i]; perm[i] = perm[j]; perm[j] = t;
    }
    *st_out = st;
}
static void ap_prngperm(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 perm[256]; u16 st;
    prngperm_build((u16)amp, perm, &st);
    for (int i = 0; i < n; i++) {
        u8 coin = xs16_next(&st);
        if (coin & 1) d[i] = perm[d[i]];
    }
}
static void inv_prngperm(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 perm[256], invperm[256]; u16 st;
    prngperm_build((u16)amp, perm, &st);
    for (int i = 0; i < 256; i++) invperm[perm[i]] = (u8)i;
    for (int i = 0; i < n; i++) {
        u8 coin = xs16_next(&st);
        if (coin & 1) d[i] = invperm[d[i]];
    }
}
static double search_prngperm(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u8 perm[256]; u16 st;
        prngperm_build((u16)seed, perm, &st);
        int f[256] = {0};
        for (int i = 0; i < n; i++) {
            u8 coin = xs16_next(&st);
            f[(coin & 1) ? perm[d[i]] : d[i]]++;
        }
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- PRNG_PERM_BIT<k> / PRNG_PERM_E<t>/S<t>/Q<t> family: same seeded
 * Fisher-Yates-permutation-applied-to-a-coinflip-subset mechanism as
 * PRNG_PERM (which tests bit0 of the continuing stream, ~50% density),
 * varying WHICH bit is tested (bit1..bit7, still ~50% density but a
 * genuinely different selected subset for the same seed) or using a
 * coarser multi-bit threshold test for finer density control (eighths,
 * sixteenths, quarters) -- different modulus width means a different
 * dependency on the stream's bits even at overlapping nominal density.
 * 7 bit-position + 7 eighths + 15 sixteenths + 3 quarters = 32. */
#define DEFINE_PRNGPERM_BIT(SUF, BITIDX) \
static void ap_prngperm_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 perm[256]; u16 st; \
    prngperm_build((u16)amp, perm, &st); \
    for (int i = 0; i < n; i++) { \
        u8 coin = xs16_next(&st); \
        if ((coin >> (BITIDX)) & 1) d[i] = perm[d[i]]; \
    } \
} \
static void inv_prngperm_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 perm[256], invperm[256]; u16 st; \
    prngperm_build((u16)amp, perm, &st); \
    for (int i = 0; i < 256; i++) invperm[perm[i]] = (u8)i; \
    for (int i = 0; i < n; i++) { \
        u8 coin = xs16_next(&st); \
        if ((coin >> (BITIDX)) & 1) d[i] = invperm[d[i]]; \
    } \
} \
static double search_prngperm_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double oh = oh_flat(16.0); \
    double best = -1e18; u32 bseed = 1; \
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) { \
        u8 perm[256]; u16 st; \
        prngperm_build((u16)seed, perm, &st); \
        int f[256] = {0}; \
        for (int i = 0; i < n; i++) { \
            u8 coin = xs16_next(&st); \
            f[((coin >> (BITIDX)) & 1) ? perm[d[i]] : d[i]]++; \
        } \
        double net = (S_from_freq(f) - Sb) - oh; \
        if (net > best) { best = net; bseed = seed; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = bseed; \
    return best; \
}
DEFINE_PRNGPERM_BIT(bit1, 1)
DEFINE_PRNGPERM_BIT(bit2, 2)
DEFINE_PRNGPERM_BIT(bit3, 3)
DEFINE_PRNGPERM_BIT(bit4, 4)
DEFINE_PRNGPERM_BIT(bit5, 5)
DEFINE_PRNGPERM_BIT(bit6, 6)
DEFINE_PRNGPERM_BIT(bit7, 7)
#define DEFINE_PRNGPERM_THRESH(SUF, MASKBITS, THRESH) \
static void ap_prngperm_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 perm[256]; u16 st; \
    prngperm_build((u16)amp, perm, &st); \
    for (int i = 0; i < n; i++) { \
        u8 coin = xs16_next(&st); \
        if ((coin & ((1 << (MASKBITS)) - 1)) < (THRESH)) d[i] = perm[d[i]]; \
    } \
} \
static void inv_prngperm_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 perm[256], invperm[256]; u16 st; \
    prngperm_build((u16)amp, perm, &st); \
    for (int i = 0; i < 256; i++) invperm[perm[i]] = (u8)i; \
    for (int i = 0; i < n; i++) { \
        u8 coin = xs16_next(&st); \
        if ((coin & ((1 << (MASKBITS)) - 1)) < (THRESH)) d[i] = invperm[d[i]]; \
    } \
} \
static double search_prngperm_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double oh = oh_flat(16.0); \
    double best = -1e18; u32 bseed = 1; \
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) { \
        u8 perm[256]; u16 st; \
        prngperm_build((u16)seed, perm, &st); \
        int f[256] = {0}; \
        for (int i = 0; i < n; i++) { \
            u8 coin = xs16_next(&st); \
            f[((coin & ((1 << (MASKBITS)) - 1)) < (THRESH)) ? perm[d[i]] : d[i]]++; \
        } \
        double net = (S_from_freq(f) - Sb) - oh; \
        if (net > best) { best = net; bseed = seed; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = bseed; \
    return best; \
}
DEFINE_PRNGPERM_THRESH(e1, 3, 1)
DEFINE_PRNGPERM_THRESH(e2, 3, 2)
DEFINE_PRNGPERM_THRESH(e3, 3, 3)
DEFINE_PRNGPERM_THRESH(e4, 3, 4)
DEFINE_PRNGPERM_THRESH(e5, 3, 5)
DEFINE_PRNGPERM_THRESH(e6, 3, 6)
DEFINE_PRNGPERM_THRESH(e7, 3, 7)
DEFINE_PRNGPERM_THRESH(s1, 4, 1)
DEFINE_PRNGPERM_THRESH(s2, 4, 2)
DEFINE_PRNGPERM_THRESH(s3, 4, 3)
DEFINE_PRNGPERM_THRESH(s4, 4, 4)
DEFINE_PRNGPERM_THRESH(s5, 4, 5)
DEFINE_PRNGPERM_THRESH(s6, 4, 6)
DEFINE_PRNGPERM_THRESH(s7, 4, 7)
DEFINE_PRNGPERM_THRESH(s8, 4, 8)
DEFINE_PRNGPERM_THRESH(s9, 4, 9)
DEFINE_PRNGPERM_THRESH(s10, 4, 10)
DEFINE_PRNGPERM_THRESH(s11, 4, 11)
DEFINE_PRNGPERM_THRESH(s12, 4, 12)
DEFINE_PRNGPERM_THRESH(s13, 4, 13)
DEFINE_PRNGPERM_THRESH(s14, 4, 14)
DEFINE_PRNGPERM_THRESH(s15, 4, 15)
DEFINE_PRNGPERM_THRESH(q1, 2, 1)
DEFINE_PRNGPERM_THRESH(q2, 2, 2)
DEFINE_PRNGPERM_THRESH(q3, 2, 3)
/* ---- PRNG_PERM_W<t>/X<t>: sparser density thresholds than the E/S/Q
 * families above -- 32nds (W1..W16, up to 50% density) and 64ths
 * (X1..X16, up to 25% density). Added because PRNG_PERM_S1 (the
 * sparsest sixteenths threshold, ~6% density) was the winning
 * instruction at layer 3 of the layered pass on real data -- pushing
 * toward even sparser thresholds tests whether the trend continues. */
DEFINE_PRNGPERM_THRESH(w1, 5, 1)
DEFINE_PRNGPERM_THRESH(w2, 5, 2)
DEFINE_PRNGPERM_THRESH(w3, 5, 3)
DEFINE_PRNGPERM_THRESH(w4, 5, 4)
DEFINE_PRNGPERM_THRESH(w5, 5, 5)
DEFINE_PRNGPERM_THRESH(w6, 5, 6)
DEFINE_PRNGPERM_THRESH(w7, 5, 7)
DEFINE_PRNGPERM_THRESH(w8, 5, 8)
DEFINE_PRNGPERM_THRESH(w9, 5, 9)
DEFINE_PRNGPERM_THRESH(w10, 5, 10)
DEFINE_PRNGPERM_THRESH(w11, 5, 11)
DEFINE_PRNGPERM_THRESH(w12, 5, 12)
DEFINE_PRNGPERM_THRESH(w13, 5, 13)
DEFINE_PRNGPERM_THRESH(w14, 5, 14)
DEFINE_PRNGPERM_THRESH(w15, 5, 15)
DEFINE_PRNGPERM_THRESH(w16, 5, 16)
DEFINE_PRNGPERM_THRESH(x1, 6, 1)
DEFINE_PRNGPERM_THRESH(x2, 6, 2)
DEFINE_PRNGPERM_THRESH(x3, 6, 3)
DEFINE_PRNGPERM_THRESH(x4, 6, 4)
DEFINE_PRNGPERM_THRESH(x5, 6, 5)
DEFINE_PRNGPERM_THRESH(x6, 6, 6)
DEFINE_PRNGPERM_THRESH(x7, 6, 7)
DEFINE_PRNGPERM_THRESH(x8, 6, 8)
DEFINE_PRNGPERM_THRESH(x9, 6, 9)
DEFINE_PRNGPERM_THRESH(x10, 6, 10)
DEFINE_PRNGPERM_THRESH(x11, 6, 11)
DEFINE_PRNGPERM_THRESH(x12, 6, 12)
DEFINE_PRNGPERM_THRESH(x13, 6, 13)
DEFINE_PRNGPERM_THRESH(x14, 6, 14)
DEFINE_PRNGPERM_THRESH(x15, 6, 15)
DEFINE_PRNGPERM_THRESH(x16, 6, 16)
DEFINE_PRNGPERM_THRESH(y1, 7, 1)
DEFINE_PRNGPERM_THRESH(y2, 7, 2)
DEFINE_PRNGPERM_THRESH(y3, 7, 3)
DEFINE_PRNGPERM_THRESH(y4, 7, 4)
DEFINE_PRNGPERM_THRESH(y5, 7, 5)
DEFINE_PRNGPERM_THRESH(y6, 7, 6)
DEFINE_PRNGPERM_THRESH(y7, 7, 7)
DEFINE_PRNGPERM_THRESH(y8, 7, 8)
DEFINE_PRNGPERM_THRESH(y9, 7, 9)
DEFINE_PRNGPERM_THRESH(y10, 7, 10)
DEFINE_PRNGPERM_THRESH(y11, 7, 11)
DEFINE_PRNGPERM_THRESH(y12, 7, 12)
DEFINE_PRNGPERM_THRESH(y13, 7, 13)
DEFINE_PRNGPERM_THRESH(y14, 7, 14)
DEFINE_PRNGPERM_THRESH(y15, 7, 15)
DEFINE_PRNGPERM_THRESH(y16, 7, 16)
DEFINE_PRNGPERM_THRESH(y17, 7, 17)
DEFINE_PRNGPERM_THRESH(y18, 7, 18)
DEFINE_PRNGPERM_THRESH(y19, 7, 19)
DEFINE_PRNGPERM_THRESH(y20, 7, 20)
DEFINE_PRNGPERM_THRESH(y21, 7, 21)
DEFINE_PRNGPERM_THRESH(y22, 7, 22)
DEFINE_PRNGPERM_THRESH(y23, 7, 23)
DEFINE_PRNGPERM_THRESH(y24, 7, 24)
DEFINE_PRNGPERM_THRESH(y25, 7, 25)
DEFINE_PRNGPERM_THRESH(y26, 7, 26)
DEFINE_PRNGPERM_THRESH(y27, 7, 27)
DEFINE_PRNGPERM_THRESH(y28, 7, 28)
DEFINE_PRNGPERM_THRESH(y29, 7, 29)
DEFINE_PRNGPERM_THRESH(y30, 7, 30)
DEFINE_PRNGPERM_THRESH(y31, 7, 31)
DEFINE_PRNGPERM_THRESH(y32, 7, 32)
DEFINE_PRNGPERM_THRESH(y33, 7, 33)
DEFINE_PRNGPERM_THRESH(y34, 7, 34)
DEFINE_PRNGPERM_THRESH(y35, 7, 35)
DEFINE_PRNGPERM_THRESH(y36, 7, 36)
DEFINE_PRNGPERM_THRESH(y37, 7, 37)
DEFINE_PRNGPERM_THRESH(y38, 7, 38)
DEFINE_PRNGPERM_THRESH(y39, 7, 39)
DEFINE_PRNGPERM_THRESH(y40, 7, 40)
DEFINE_PRNGPERM_THRESH(y41, 7, 41)
DEFINE_PRNGPERM_THRESH(y42, 7, 42)
DEFINE_PRNGPERM_THRESH(y43, 7, 43)
DEFINE_PRNGPERM_THRESH(y44, 7, 44)
DEFINE_PRNGPERM_THRESH(y45, 7, 45)
DEFINE_PRNGPERM_THRESH(y46, 7, 46)
DEFINE_PRNGPERM_THRESH(y47, 7, 47)
DEFINE_PRNGPERM_THRESH(y48, 7, 48)
DEFINE_PRNGPERM_THRESH(y49, 7, 49)
DEFINE_PRNGPERM_THRESH(y50, 7, 50)
DEFINE_PRNGPERM_THRESH(y51, 7, 51)
DEFINE_PRNGPERM_THRESH(y52, 7, 52)
DEFINE_PRNGPERM_THRESH(y53, 7, 53)
DEFINE_PRNGPERM_THRESH(y54, 7, 54)
DEFINE_PRNGPERM_THRESH(y55, 7, 55)
DEFINE_PRNGPERM_THRESH(y56, 7, 56)
DEFINE_PRNGPERM_THRESH(y57, 7, 57)
DEFINE_PRNGPERM_THRESH(y58, 7, 58)
DEFINE_PRNGPERM_THRESH(y59, 7, 59)
DEFINE_PRNGPERM_THRESH(y60, 7, 60)
DEFINE_PRNGPERM_THRESH(y61, 7, 61)
DEFINE_PRNGPERM_THRESH(y62, 7, 62)
DEFINE_PRNGPERM_THRESH(y63, 7, 63)
DEFINE_PRNGPERM_THRESH(y64, 7, 64)

/* ---- SYNDROME_XOR: partition into W=8 windows, compute each window's
 * XOR-syndrome; if it deviates from 0 (a coding-theory-style parity
 * check), XOR the whole window by ahi, else by alo -- a binary
 * anomaly-detection framing, coarser than WINDOW_XOR's data-derived
 * multiplicative key. Same invariant-syndrome trick (W even) makes it
 * self-inverse. */
#define SYN_W 8
static void ap_syndrome(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    for (int base = 0; base + SYN_W <= n; base += SYN_W) {
        u8 syn = 0;
        for (int j = 0; j < SYN_W; j++) syn ^= d[base + j];
        u8 c = (syn != 0) ? ahi : alo;
        for (int j = 0; j < SYN_W; j++) d[base + j] ^= c;
    }
}
static double search_syndrome(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int alo = 0; alo < 256; alo++) {
        for (int ahi = 0; ahi < 256; ahi++) {
            for (int base = 0; base + SYN_W <= n; base += SYN_W) {
                u8 syn = 0;
                for (int j = 0; j < SYN_W; j++) syn ^= d[base + j];
                u8 c = (syn != 0) ? (u8)ahi : (u8)alo;
                for (int j = 0; j < SYN_W; j++) scr[base + j] = (u8)(d[base + j] ^ c);
            }
            double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
            if (net > best) { best = net; ba = (u32)alo | ((u32)ahi << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- PAETH2D: reshape 4096 -> 64x64 grid, PNG-style Paeth predictor
 * using left+up+upleft neighbors (whichever of the three is closest to
 * left+up-upleft is the prediction). No searched parameter. Needs a
 * genuinely different reversibility strategy than the 1D neighbor family:
 * with THREE different neighbor positions, a simple local-variable
 * tracker doesn't work. apply() snapshots the untouched original into a
 * scratch copy and always predicts from that (trivially correct, never
 * mutated); invert() predicts directly from the in-place array, which is
 * safe because raster-order traversal guarantees every neighbor position
 * (same row earlier column, or anywhere in the previous row) has already
 * been recovered by the time it's needed. */
#define P2D_N 64
static inline int paeth_pred(int left, int up, int upleft) {
    int pr = left + up - upleft;
    int pa = abs(pr - left), pb = abs(pr - up), pc = abs(pr - upleft);
    if (pa <= pb && pa <= pc) return left;
    if (pb <= pc) return up;
    return upleft;
}
static void ap_paeth2d(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 orig[BLOCK];
    memcpy(orig, d, (size_t)n);
    for (int r = 0; r < P2D_N; r++) {
        for (int c = 0; c < P2D_N; c++) {
            int i = r * P2D_N + c;
            int left   = (c > 0) ? orig[r * P2D_N + (c - 1)] : 0;
            int up     = (r > 0) ? orig[(r - 1) * P2D_N + c] : 0;
            int upleft = (r > 0 && c > 0) ? orig[(r - 1) * P2D_N + (c - 1)] : 0;
            d[i] = (u8)(orig[i] - paeth_pred(left, up, upleft));
        }
    }
}
static void inv_paeth2d(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int r = 0; r < P2D_N; r++) {
        for (int c = 0; c < P2D_N; c++) {
            int i = r * P2D_N + c;
            int left   = (c > 0) ? d[r * P2D_N + (c - 1)] : 0;
            int up     = (r > 0) ? d[(r - 1) * P2D_N + c] : 0;
            int upleft = (r > 0 && c > 0) ? d[(r - 1) * P2D_N + (c - 1)] : 0;
            d[i] = (u8)(d[i] + paeth_pred(left, up, upleft));
        }
    }
}
static double search_paeth2d(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_paeth2d(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- PAETH2D_W32: same PNG-style Paeth predictor as PAETH2D, but
 * reshaped as a 32x128 grid instead of 64x64 -- much longer rows change
 * which pixels count as "up"/"upleft" neighbors, giving different
 * correlation capture on data with a different natural row period. */
#define P2D_W 32
#define P2D_H 128
static void ap_paeth2d_w32(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 orig[BLOCK];
    memcpy(orig, d, (size_t)n);
    for (int r = 0; r < P2D_H; r++) {
        for (int c = 0; c < P2D_W; c++) {
            int i = r * P2D_W + c;
            int left   = (c > 0) ? orig[r * P2D_W + (c - 1)] : 0;
            int up     = (r > 0) ? orig[(r - 1) * P2D_W + c] : 0;
            int upleft = (r > 0 && c > 0) ? orig[(r - 1) * P2D_W + (c - 1)] : 0;
            d[i] = (u8)(orig[i] - paeth_pred(left, up, upleft));
        }
    }
}
static void inv_paeth2d_w32(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int r = 0; r < P2D_H; r++) {
        for (int c = 0; c < P2D_W; c++) {
            int i = r * P2D_W + c;
            int left   = (c > 0) ? d[r * P2D_W + (c - 1)] : 0;
            int up     = (r > 0) ? d[(r - 1) * P2D_W + c] : 0;
            int upleft = (r > 0 && c > 0) ? d[(r - 1) * P2D_W + (c - 1)] : 0;
            d[i] = (u8)(d[i] + paeth_pred(left, up, upleft));
        }
    }
}
static double search_paeth2d_w32(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_paeth2d_w32(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- JOINT_XOR: 4 cells indexed by BOTH position parity (i&1) AND value
 * bucket (top bit of d[i]), each with its own independently-fit 7-bit XOR
 * constant (top bit forced 0 in every constant, so it's never flipped --
 * otherwise the decoder couldn't recompute the same bucket from the
 * transformed byte). Catches position x value interaction effects neither
 * axis alone can see. Self-inverse. Search fits each cell independently
 * (greedy approximation, same caveat as PIECEWISE_ADD/VIGENERE). */
static void ap_jointxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 c[4];
    for (int k = 0; k < 4; k++) c[k] = (u8)((amp >> (k * 7)) & 0x7F);
    for (int i = 0; i < n; i++) {
        int phase = i & 1;
        int bucket = (d[i] >> 7) & 1;
        d[i] ^= c[phase * 2 + bucket];
    }
}
static double search_jointxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    u32 amp = 0;
    for (int cell = 0; cell < 4; cell++) {
        int phase = cell >> 1, bucket = cell & 1;
        int hit[256] = {0};
        for (int i = 0; i < n; i++)
            if ((i & 1) == phase && ((d[i] >> 7) & 1) == bucket) hit[d[i]]++;
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        double bestg = -1e18; int bc = 0;
        for (int c = 0; c < 128; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; bc = c; }
        }
        amp |= (u32)bc << (cell * 7);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_jointxor(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(28.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

/* ---- ANCHOR_XOR: designate d[0] as an untouched anchor byte; derive one
 * XOR constant from it (c = a*anchor+b, affine, mod 256) and XOR it into
 * every OTHER byte uniformly. Included for empirical testing even though
 * theory predicts it's dominated: excluding just 1 byte out of 4096 is
 * negligibly different from a whole-block-uniform bijection (Master
 * Principle), which is provably useless -- expect this to score at or
 * near 0 net. Self-inverse. */
static void ap_anchorxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)(amp & 0xFF), b = (u8)((amp >> 8) & 0xFF);
    u8 c = (u8)(a * d[0] + b);
    for (int i = 1; i < n; i++) d[i] ^= c;
}
static double search_anchorxor(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a += 2) {  /* odd a for a well-defined affine map, though only used forward here */
        for (int b = 0; b < 256; b++) {
            u8 c = (u8)(a * d[0] + b);
            scr[0] = d[0];
            for (int i = 1; i < n; i++) scr[i] = (u8)(d[i] ^ c);
            double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
            if (net > best) { best = net; ba = (u32)a | ((u32)b << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- WORD_XOR16: XOR adjacent byte pairs by a 16-bit constant. Included
 * for empirical testing even though theory predicts it's dominated: XOR
 * has no carry, so this decomposes exactly into two independent byte-
 * level phase-XORs (stride 2, phase 0 and phase 1) -- whatever this finds
 * should be equal to or worse than running XOR_PHASE(stride=2) twice
 * independently, since here both bytes' constants are locked together
 * under one search rather than each optimized on its own. Self-inverse. */
static void ap_wordxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 c0 = (u8)(amp & 0xFF), c1 = (u8)((amp >> 8) & 0xFF);
    for (int i = 0; i + 1 < n; i += 2) { d[i] ^= c0; d[i + 1] ^= c1; }
}
static double search_wordxor(const u8 *d, int n, double Sb, Instr *out) {
    /* NOTE: an earlier version scored candidates by summing S computed
     * SEPARATELY over the even-position and odd-position sub-histograms
     * (2048 items each) and subtracting the TRUE combined 4096-item Sb --
     * mathematically incoherent (hlog isn't additive like that: both
     * c0-transformed even bytes and c1-transformed odd bytes can land on
     * the SAME output byte value 0..255, unlike VALUE_XOR's bit-k split
     * where the two groups are guaranteed disjoint output ranges). That
     * produced a wildly wrong net (-3945) despite the transform itself
     * being perfectly reversible -- the bug was only in scoring, not in
     * apply/invert. Fixed by picking c0, c1 independently as a heuristic
     * (still doesn't account for their interaction, same caveat as
     * PIECEWISE_ADD/VIGENERE) but then re-measuring the TRUE combined S
     * by actually applying the transform, like those do. */
    int tot0[256] = {0}, tot1[256] = {0};
    for (int i = 0; i + 1 < n; i += 2) { tot0[d[i]]++; tot1[d[i + 1]]++; }
    double oh = oh_flat(16.0);
    double bestc0 = -1e18; int bc0 = 0;
    for (int c = 0; c < 256; c++) {
        double S = 0.0;
        int rf[256] = {0};
        for (int u = 0; u < 256; u++) rf[u ^ c] += tot0[u];
        for (int v = 0; v < 256; v++) S += hlog[rf[v]];
        if (S > bestc0) { bestc0 = S; bc0 = c; }
    }
    double bestc1 = -1e18; int bc1 = 0;
    for (int c = 0; c < 256; c++) {
        double S = 0.0;
        int rf[256] = {0};
        for (int u = 0; u < 256; u++) rf[u ^ c] += tot1[u];
        for (int v = 0; v < 256; v++) S += hlog[rf[v]];
        if (S > bestc1) { bestc1 = S; bc1 = c; }
    }
    u32 amp = (u32)bc0 | ((u32)bc1 << 8);
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_wordxor(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh;
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

/* ---- QR_XOR: hi nibble's quadratic-residue class mod 5 (is (hi mod 5) a
 * perfect square mod 5, i.e. in {0,1,4}?) picks alo/ahi for the lo
 * nibble -- an algebraic/number-theoretic split shape, structurally
 * different from the popcount/magnitude/threshold splits already done.
 * Self-inverse. Pointwise bijection -> stride/phase. */
static int is_qr5[16];
static void init_qr5(void) {
    int sq[5] = { 0 };
    for (int x = 0; x < 5; x++) sq[(x * x) % 5] = 1;
    for (int h = 0; h < 16; h++) is_qr5[h] = sq[h % 5];
}
static void ap_qrxor(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = is_qr5[hi] ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_qrxor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hi = (u >> 4) & 0xF;
                        u8 x = is_qr5[hi] ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- QR_XOR_MOD7: same quadratic-residue split mechanism as QR_XOR, but
 * modulus 7 instead of modulus 5 (QRs mod 7 are {0,1,2,4} instead of
 * {0,1,4}) -- a different residue class gives a different partition of
 * the 16 hi-nibble values into the two groups. Self-inverse. */
static int is_qr7[16];
static void init_qr7(void) {
    int sq[7] = { 0 };
    for (int x = 0; x < 7; x++) sq[(x * x) % 7] = 1;
    for (int h = 0; h < 16; h++) is_qr7[h] = sq[h % 7];
}
static void ap_qrxor_mod7(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = is_qr7[hi] ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_qrxor_mod7(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hi = (u >> 4) & 0xF;
                        u8 x = is_qr7[hi] ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- CTXTABLE_XOR: maintain a 256-entry table indexed by the previous
 * byte's value, holding the last-seen byte that followed that context.
 * Predict d[i] as table[prev], XOR it in, then update table[prev] to the
 * newly-observed original. A context-conditioned delta (predicts from
 * "what usually follows this byte" rather than a fixed neighbor formula),
 * the adaptive-context idea from the predictive family. No searched
 * parameter. */
static void ap_ctxtable(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 table[256]; memset(table, 0, sizeof table);
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig = d[i];
        u8 pred = table[prev];
        d[i] = (u8)(orig ^ pred);
        table[prev] = orig;
        prev = orig;
    }
}
static void inv_ctxtable(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 table[256]; memset(table, 0, sizeof table);
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 pred = table[prev];
        u8 orig = (u8)(d[i] ^ pred);
        d[i] = orig;
        table[prev] = orig;
        prev = orig;
    }
}
static double search_ctxtable(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_ctxtable(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- GF2_LINEAR: an invertible GF(2)-linear byte map (arbitrary 8x8
 * binary matrix applied to the byte as a bit-vector), built by COMPOSING
 * two already-proven-invertible primitives (rotate-left-3, then
 * self-shift-fold k=2) rather than risking an unverified arbitrary
 * matrix -- composition of invertible maps is always invertible, and the
 * inverse is just the reverse-order composition of the two known
 * inverses. Genuinely different bit-mixing shape than either primitive
 * alone. Pointwise bijection -> stride/phase. No searched parameter. */
static void ap_gf2linear(u8 *d, int n, int s, int p, u32 amp) {
    (void)amp;
    for (int i = p; i < n; i += s) d[i] = selfshift_fwd(byterot_fwd(d[i], 3), 2);
}
static void inv_gf2linear(u8 *d, int n, int s, int p, u32 amp) {
    (void)amp;
    for (int i = p; i < n; i += s) d[i] = byterot_inv(selfshift_inv(d[i], 2), 3);
}
static double search_gf2linear(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(0.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int rf[256];
            for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
            for (int u = 0; u < 256; u++) rf[selfshift_fwd(byterot_fwd((u8)u, 3), 2)] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}

/* ---- LATIN_CROSS: cross-crumb operator using a genuine Latin square (not
 * ADD/XOR/GF-mul) -- crumb k -> latin4[crumb_k][crumb_j], crumb j
 * untouched. Any Latin square's columns are each a permutation of
 * 0..3 by definition, which is exactly what guarantees invertibility for
 * a fixed key crumb j. Generalizes CRMB_CXOR/CRMB_CADD beyond algebraic
 * operations to an arbitrary reversible lookup table. Pointwise
 * bijection -> stride/phase. */
static const u8 latin4[4][4] = {
    { 0, 1, 2, 3 },
    { 1, 0, 3, 2 },
    { 2, 3, 1, 0 },
    { 3, 2, 0, 1 },
};
static u8 latin4_inv[4][4];
static void init_latin4(void) {
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 4; k++)
            latin4_inv[latin4[k][j]][j] = (u8)k;
}
static void ap_latincross(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3);
        u8 ck = (u8)((v >> (2 * k)) & 3);
        u8 nck = latin4[ck][cj];
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (nck << (2 * k)));
    }
}
static void inv_latincross(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3);
        u8 nck = (u8)((v >> (2 * k)) & 3);
        u8 ck = latin4_inv[nck][cj];
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (ck << (2 * k)));
    }
}
static double search_latincross(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);  /* j(2) + k(2) */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 cj = (u8)((u >> (2 * j)) & 3);
                        u8 ck = (u8)((u >> (2 * k)) & 3);
                        u8 w = (u8)((u & ~(u8)(3 << (2 * k))) | (latin4[ck][cj] << (2 * k)));
                        rf[w] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)j | ((u32)k << 2); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- WORD_GFMUL: genuine GF(65536) field multiply on adjacent byte
 * pairs (big-endian 16-bit word). Reduction polynomial x^16+x^5+x^3+x+1
 * (0x002B, implicit x^16 bit). NOTE: an earlier attempt used the
 * CRC-16-CCITT polynomial (0x1021) on the WRONG assumption that CRC
 * generators are field-irreducible -- they're chosen for error-detection
 * distance, not irreducibility, and 0x1021 is in fact reducible (most
 * elements had no inverse, caught immediately by selftest). 0x002B was
 * verified via Fermat's little theorem (a^65535 == 1 for 200 random
 * nonzero a, consistent with a genuine order-65536 field) before use.
 * Word-granularity field arithmetic is a richer structure than GF(256):
 * more nonzero elements, can catch correlations only visible jointly
 * across 2 bytes. Inverse found by brute-force search (65536 candidates,
 * done once per invert() call -- cheap, no log/antilog table needed). */
static inline u16 gf65536_mul(u16 a, u16 b) {
    u16 p = 0;
    for (int i = 0; i < 16; i++) {
        if (b & 1) p ^= a;
        int hibit = a & 0x8000;
        a = (u16)(a << 1);
        if (hibit) a ^= 0x002B;
        b = (u16)(b >> 1);
    }
    return p;
}
static u16 gf65536_inv(u16 a) {
    for (u32 c = 1; c < 65536; c++) if (gf65536_mul(a, (u16)c) == 1) return (u16)c;
    return 1;
}
static void ap_wordgfmul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 a = (u16)amp;
    for (int i = 0; i + 1 < n; i += 2) {
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]);
        w = gf65536_mul(w, a);
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF);
    }
}
static void inv_wordgfmul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ainv = gf65536_inv((u16)amp);
    for (int i = 0; i + 1 < n; i += 2) {
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]);
        w = gf65536_mul(w, ainv);
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF);
    }
}
static double search_wordgfmul(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 2;
    static u8 scr[BLOCK];
    for (u32 a = 2; a < 65536; a++) {
        for (int i = 0; i + 1 < n; i += 2) {
            u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]);
            w = gf65536_mul(w, (u16)a);
            scr[i] = (u8)(w >> 8); scr[i + 1] = (u8)(w & 0xFF);
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
        if (net > best) { best = net; ba = a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- WORD_GFMUL_ODD: same GF(65536) word-multiply mechanism as
 * WORD_GFMUL, but pairs bytes at ODD offsets (1,2),(3,4),... instead of
 * (0,1),(2,3),... -- byte 0 and the last unpaired byte are left
 * untouched. Catches cross-byte correlation on the opposite pairing
 * phase from WORD_GFMUL. */
static void ap_wordgfmul_odd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 a = (u16)amp;
    for (int i = 1; i + 1 < n; i += 2) {
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]);
        w = gf65536_mul(w, a);
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF);
    }
}
static void inv_wordgfmul_odd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ainv = gf65536_inv((u16)amp);
    for (int i = 1; i + 1 < n; i += 2) {
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]);
        w = gf65536_mul(w, ainv);
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF);
    }
}
static double search_wordgfmul_odd(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 2;
    static u8 scr[BLOCK];
    for (u32 a = 2; a < 65536; a++) {
        memcpy(scr, d, (size_t)n);
        for (int i = 1; i + 1 < n; i += 2) {
            u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]);
            w = gf65536_mul(w, (u16)a);
            scr[i] = (u8)(w >> 8); scr[i + 1] = (u8)(w & 0xFF);
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
        if (net > best) { best = net; ba = a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- WORD_GFMUL_POLY2 / WORD_GFMUL_POLY2_ODD / WORD_GFMUL_POLY3 /
 * WORD_GFMUL_POLY3_ODD: same GF(65536) word-multiply mechanism as
 * WORD_GFMUL/WORD_GFMUL_ODD, but with a DIFFERENT reduction polynomial --
 * a genuinely different field (different multiplication table entirely,
 * not just a relabeling). Both new polynomials (0x100B, 0x002D) were
 * verified irreducible the same way the original 0x002B was: a^65535==1
 * for 300 random nonzero a, checked via a standalone verifier program
 * before use (0x1021/CRC-CCITT was re-confirmed REDUCIBLE by the same
 * verifier as a sanity check). Crossed with both pairing phases
 * (even/odd) from WORD_GFMUL/WORD_GFMUL_ODD. */
static inline u16 gf65536_mul_poly(u16 a, u16 b, u16 poly) {
    u16 p = 0;
    for (int i = 0; i < 16; i++) {
        if (b & 1) p ^= a;
        int hibit = a & 0x8000;
        a = (u16)(a << 1);
        if (hibit) a ^= poly;
        b = (u16)(b >> 1);
    }
    return p;
}
static u16 gf65536_inv_poly(u16 a, u16 poly) {
    for (u32 c = 1; c < 65536; c++) if (gf65536_mul_poly(a, (u16)c, poly) == 1) return (u16)c;
    return 1;
}
#define DEFINE_WORDGFMUL_POLY(SUF, POLYVAL, ISTART) \
static void ap_wordgfmul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u16 a = (u16)amp; \
    for (int i = (ISTART); i + 1 < n; i += 2) { \
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]); \
        w = gf65536_mul_poly(w, a, (POLYVAL)); \
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF); \
    } \
} \
static void inv_wordgfmul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u16 ainv = gf65536_inv_poly((u16)amp, (POLYVAL)); \
    for (int i = (ISTART); i + 1 < n; i += 2) { \
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]); \
        w = gf65536_mul_poly(w, ainv, (POLYVAL)); \
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF); \
    } \
} \
static double search_wordgfmul_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double best = -1e18; u32 ba = 2; \
    static u8 scr[BLOCK]; \
    for (u32 a = 2; a < 65536; a++) { \
        memcpy(scr, d, (size_t)n); \
        for (int i = (ISTART); i + 1 < n; i += 2) { \
            u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]); \
            w = gf65536_mul_poly(w, (u16)a, (POLYVAL)); \
            scr[i] = (u8)(w >> 8); scr[i + 1] = (u8)(w & 0xFF); \
        } \
        double net = (S_of(scr, n) - Sb) - oh_flat(16.0); \
        if (net > best) { best = net; ba = a; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_WORDGFMUL_POLY(poly2, 0x100B, 0)
DEFINE_WORDGFMUL_POLY(poly2_odd, 0x100B, 1)
DEFINE_WORDGFMUL_POLY(poly3, 0x002D, 0)
DEFINE_WORDGFMUL_POLY(poly3_odd, 0x002D, 1)

/* ---- WORD_GFMUL_LE / WORD_GFMUL_ODD_LE: same GF(65536) mechanism and
 * reduction polynomial as WORD_GFMUL/WORD_GFMUL_ODD, but LITTLE-endian
 * word order (d[i] | d[i+1]<<8) instead of big-endian -- a different
 * 16-bit numeric value for the exact same byte pair, so a genuinely
 * different multiplication result even though the field is identical. */
#define DEFINE_WORDGFMUL_LE(SUF, ISTART) \
static void ap_wordgfmul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u16 a = (u16)amp; \
    for (int i = (ISTART); i + 1 < n; i += 2) { \
        u16 w = (u16)((u16)d[i] | ((u16)d[i + 1] << 8)); \
        w = gf65536_mul(w, a); \
        d[i] = (u8)(w & 0xFF); d[i + 1] = (u8)(w >> 8); \
    } \
} \
static void inv_wordgfmul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u16 ainv = gf65536_inv((u16)amp); \
    for (int i = (ISTART); i + 1 < n; i += 2) { \
        u16 w = (u16)((u16)d[i] | ((u16)d[i + 1] << 8)); \
        w = gf65536_mul(w, ainv); \
        d[i] = (u8)(w & 0xFF); d[i + 1] = (u8)(w >> 8); \
    } \
} \
static double search_wordgfmul_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double best = -1e18; u32 ba = 2; \
    static u8 scr[BLOCK]; \
    for (u32 a = 2; a < 65536; a++) { \
        memcpy(scr, d, (size_t)n); \
        for (int i = (ISTART); i + 1 < n; i += 2) { \
            u16 w = (u16)((u16)d[i] | ((u16)d[i + 1] << 8)); \
            w = gf65536_mul(w, (u16)a); \
            scr[i] = (u8)(w & 0xFF); scr[i + 1] = (u8)(w >> 8); \
        } \
        double net = (S_of(scr, n) - Sb) - oh_flat(16.0); \
        if (net > best) { best = net; ba = a; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_WORDGFMUL_LE(le, 0)
DEFINE_WORDGFMUL_LE(odd_le, 1)

/* ---- WORD_GFMUL_SPARSE_EVEN / WORD_GFMUL_SPARSE_ODD: same GF(65536)
 * word-multiply as WORD_GFMUL, but only every OTHER word is transformed
 * -- word-index 0,2,4,... (byte pairs (0,1),(4,5),(8,9)...) for EVEN, or
 * word-index 1,3,5,... (byte pairs (2,3),(6,7),(10,11)...) for ODD. The
 * skipped words are left completely untouched, a genuinely different
 * "which subset gets multiplied" shape than the dense pairing. */
#define DEFINE_WORDGFMUL_SPARSE(SUF, WORDPHASE) \
static void ap_wordgfmul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u16 a = (u16)amp; \
    for (int i = 0; i + 1 < n; i += 2) { \
        if (((i / 2) & 1) != (WORDPHASE)) continue; \
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]); \
        w = gf65536_mul(w, a); \
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF); \
    } \
} \
static void inv_wordgfmul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u16 ainv = gf65536_inv((u16)amp); \
    for (int i = 0; i + 1 < n; i += 2) { \
        if (((i / 2) & 1) != (WORDPHASE)) continue; \
        u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]); \
        w = gf65536_mul(w, ainv); \
        d[i] = (u8)(w >> 8); d[i + 1] = (u8)(w & 0xFF); \
    } \
} \
static double search_wordgfmul_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double best = -1e18; u32 ba = 2; \
    static u8 scr[BLOCK]; \
    for (u32 a = 2; a < 65536; a++) { \
        memcpy(scr, d, (size_t)n); \
        for (int i = 0; i + 1 < n; i += 2) { \
            if (((i / 2) & 1) != (WORDPHASE)) continue; \
            u16 w = (u16)(((u16)d[i] << 8) | d[i + 1]); \
            w = gf65536_mul(w, (u16)a); \
            scr[i] = (u8)(w >> 8); scr[i + 1] = (u8)(w & 0xFF); \
        } \
        double net = (S_of(scr, n) - Sb) - oh_flat(16.0); \
        if (net > best) { best = net; ba = a; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_WORDGFMUL_SPARSE(sparse_even, 0)
DEFINE_WORDGFMUL_SPARSE(sparse_odd, 1)

/* ---- PRIME257_MUL: multiply mod 257 (prime) -- unlike mod-256 (only odd
 * residues invertible) or GF(256) (characteristic 2), ALL 256 nonzero
 * residues mod 257 are invertible. The catch: bytes only cover 0..255,
 * but Z/257 has 257 elements (0..256), so multiplying two in-range values
 * can land on residue 256, which doesn't fit in a byte. Fix: for a
 * given multiplier a, exactly one byte value v0 maps to 256 (patch it to
 * map to v1 instead, where v1 is whatever WOULD have mapped from 256);
 * this keeps the map an exact bijection on 0..255. For a=1 no patch is
 * needed (v0/v1 both fall outside the byte range, i.e. the sentinel
 * value 256, so the patch condition never triggers). Pointwise bijection
 * -> stride/phase. Per-a forward tables precomputed once (not inside the
 * stride/phase loop) since they don't depend on stride or phase. */
static u32 prime257_inv(u32 a) {
    for (u32 c = 1; c < 257; c++) if ((a * c) % 257 == 1) return c;
    return 1;
}
static u8 prime257_fwd[255][256];
static u8 prime257_inv_tab[255][256];
static void init_prime257(void) {
    for (int aidx = 0; aidx < 255; aidx++) {
        u32 a = (u32)aidx + 2;
        u32 ainv = prime257_inv(a);
        u32 v0 = (256 * ainv) % 257, v1 = (a * 256) % 257;
        for (int v = 0; v < 256; v++) {
            u8 fw = (v0 < 256 && v == (int)v0) ? (u8)v1 : (u8)((a * (u32)v) % 257);
            prime257_fwd[aidx][v] = fw;
        }
        for (int w = 0; w < 256; w++) {
            u8 iv = (v1 < 256 && w == (int)v1) ? (u8)v0 : (u8)((ainv * (u32)w) % 257);
            prime257_inv_tab[aidx][w] = iv;
        }
    }
}
static void ap_prime257(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *col = prime257_fwd[amp % 255];
    for (int i = p; i < n; i += s) d[i] = col[d[i]];
}
static void inv_prime257(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *col = prime257_inv_tab[amp % 255];
    for (int i = p; i < n; i += s) d[i] = col[d[i]];
}
static double search_prime257(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int aidx = 0; aidx < 255; aidx++) {
                const u8 *col = prime257_fwd[aidx];
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[col[u]] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)aidx; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- PRIME<P>_MUL family: generalizes PRIME257_MUL to other primes
 * P>256 -- each such prime gives m=P-256 "overflow" outputs (residues
 * that land in [256,P-1] when multiplying by `a` mod P) that need
 * patching back into the byte range. Provable-correct generalization of
 * the single-patch trick: since v->a*v mod P is injective on the input
 * domain [0,255], its image has exactly 256 elements, of which
 * (256-m) land in-range and m overflow; by counting, exactly m byte
 * values in [0,255] are then "orphaned" (never hit by an in-range
 * output), so pairing the m overflow outputs with the m orphans (both
 * sorted ascending) gives an exact bijection on [0,255] for ANY m, not
 * just m=1 (P=257's special case). Primes below 256 do NOT work this
 * way (pigeonhole guarantees collisions when the field is smaller than
 * the byte domain), so this family only extends upward. */
#define DEFINE_PRIMEP_MUL(SUF, PVAL) \
static u32 primeP_inv_##SUF(u32 a) { \
    for (u32 c = 1; c < (PVAL); c++) if ((a * c) % (PVAL) == 1) return c; \
    return 1; \
} \
static u8 primeP_fwd_##SUF[(PVAL) - 2][256]; \
static u8 primeP_inv_tab_##SUF[(PVAL) - 2][256]; \
static void init_primeP_##SUF(void) { \
    for (int aidx = 0; aidx < (PVAL) - 2; aidx++) { \
        u32 a = (u32)aidx + 2; \
        u32 raw[256]; \
        int used[PVAL]; for (int r = 0; r < (PVAL); r++) used[r] = 0; \
        for (int v = 0; v < 256; v++) { raw[v] = (a * (u32)v) % (PVAL); used[raw[v]] = 1; } \
        int overflow_v[PVAL]; u32 overflow_r[PVAL]; int novf = 0; \
        for (int v = 0; v < 256; v++) if (raw[v] > 255) { overflow_v[novf] = v; overflow_r[novf] = raw[v]; novf++; } \
        for (int a1 = 0; a1 < novf; a1++) for (int b1 = a1 + 1; b1 < novf; b1++) \
            if (overflow_r[b1] < overflow_r[a1]) { \
                u32 tr = overflow_r[a1]; overflow_r[a1] = overflow_r[b1]; overflow_r[b1] = tr; \
                int tv = overflow_v[a1]; overflow_v[a1] = overflow_v[b1]; overflow_v[b1] = tv; \
            } \
        int orphan[PVAL]; int norph = 0; \
        for (int w = 0; w < 256; w++) if (!used[w]) orphan[norph++] = w; \
        for (int k = 0; k < novf; k++) raw[overflow_v[k]] = (u32)orphan[k]; \
        for (int v = 0; v < 256; v++) primeP_fwd_##SUF[aidx][v] = (u8)raw[v]; \
        for (int v = 0; v < 256; v++) primeP_inv_tab_##SUF[aidx][primeP_fwd_##SUF[aidx][v]] = (u8)v; \
    } \
} \
static void ap_primeP_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    const u8 *col = primeP_fwd_##SUF[amp % ((PVAL) - 2)]; \
    for (int i = p; i < n; i += s) d[i] = col[d[i]]; \
} \
static void inv_primeP_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    const u8 *col = primeP_inv_tab_##SUF[amp % ((PVAL) - 2)]; \
    for (int i = p; i < n; i += s) d[i] = col[d[i]]; \
} \
static double search_primeP_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(9.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int aidx = 0; aidx < (PVAL) - 2; aidx++) { \
                const u8 *col = primeP_fwd_##SUF[aidx]; \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int u = 0; u < 256; u++) rf[col[u]] += hit[u]; \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)aidx; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_PRIMEP_MUL(263, 263)
DEFINE_PRIMEP_MUL(269, 269)
DEFINE_PRIMEP_MUL(271, 271)
DEFINE_PRIMEP_MUL(277, 277)
DEFINE_PRIMEP_MUL(281, 281)
DEFINE_PRIMEP_MUL(283, 283)
DEFINE_PRIMEP_MUL(293, 293)
DEFINE_PRIMEP_MUL(307, 307)
DEFINE_PRIMEP_MUL(311, 311)
DEFINE_PRIMEP_MUL(313, 313)
DEFINE_PRIMEP_MUL(317, 317)
DEFINE_PRIMEP_MUL(331, 331)
DEFINE_PRIMEP_MUL(337, 337)
DEFINE_PRIMEP_MUL(347, 347)
DEFINE_PRIMEP_MUL(349, 349)
DEFINE_PRIMEP_MUL(353, 353)
DEFINE_PRIMEP_MUL(359, 359)
DEFINE_PRIMEP_MUL(367, 367)
DEFINE_PRIMEP_MUL(373, 373)
DEFINE_PRIMEP_MUL(379, 379)
DEFINE_PRIMEP_MUL(383, 383)
DEFINE_PRIMEP_MUL(389, 389)
DEFINE_PRIMEP_MUL(397, 397)
DEFINE_PRIMEP_MUL(401, 401)
DEFINE_PRIMEP_MUL(409, 409)
DEFINE_PRIMEP_MUL(419, 419)
DEFINE_PRIMEP_MUL(421, 421)
DEFINE_PRIMEP_MUL(431, 431)
DEFINE_PRIMEP_MUL(433, 433)
DEFINE_PRIMEP_MUL(439, 439)
DEFINE_PRIMEP_MUL(443, 443)
DEFINE_PRIMEP_MUL(449, 449)
DEFINE_PRIMEP_MUL(457, 457)
DEFINE_PRIMEP_MUL(461, 461)
DEFINE_PRIMEP_MUL(463, 463)
DEFINE_PRIMEP_MUL(467, 467)
DEFINE_PRIMEP_MUL(479, 479)
DEFINE_PRIMEP_MUL(487, 487)
DEFINE_PRIMEP_MUL(491, 491)
DEFINE_PRIMEP_MUL(499, 499)
DEFINE_PRIMEP_MUL(503, 503)
DEFINE_PRIMEP_MUL(509, 509)
DEFINE_PRIMEP_MUL(521, 521)
DEFINE_PRIMEP_MUL(523, 523)
DEFINE_PRIMEP_MUL(541, 541)
DEFINE_PRIMEP_MUL(547, 547)
DEFINE_PRIMEP_MUL(557, 557)
DEFINE_PRIMEP_MUL(563, 563)
DEFINE_PRIMEP_MUL(569, 569)
DEFINE_PRIMEP_MUL(571, 571)
DEFINE_PRIMEP_MUL(577, 577)
DEFINE_PRIMEP_MUL(587, 587)
DEFINE_PRIMEP_MUL(593, 593)
DEFINE_PRIMEP_MUL(599, 599)
DEFINE_PRIMEP_MUL(601, 601)
DEFINE_PRIMEP_MUL(607, 607)
DEFINE_PRIMEP_MUL(613, 613)
DEFINE_PRIMEP_MUL(617, 617)
DEFINE_PRIMEP_MUL(619, 619)
DEFINE_PRIMEP_MUL(631, 631)
DEFINE_PRIMEP_MUL(641, 641)
DEFINE_PRIMEP_MUL(643, 643)
DEFINE_PRIMEP_MUL(647, 647)
DEFINE_PRIMEP_MUL(653, 653)
DEFINE_PRIMEP_MUL(659, 659)
DEFINE_PRIMEP_MUL(661, 661)
DEFINE_PRIMEP_MUL(673, 673)
DEFINE_PRIMEP_MUL(677, 677)
DEFINE_PRIMEP_MUL(683, 683)
DEFINE_PRIMEP_MUL(691, 691)
DEFINE_PRIMEP_MUL(701, 701)
DEFINE_PRIMEP_MUL(709, 709)
DEFINE_PRIMEP_MUL(719, 719)
DEFINE_PRIMEP_MUL(727, 727)
DEFINE_PRIMEP_MUL(733, 733)
DEFINE_PRIMEP_MUL(739, 739)
DEFINE_PRIMEP_MUL(743, 743)
DEFINE_PRIMEP_MUL(751, 751)
DEFINE_PRIMEP_MUL(757, 757)
DEFINE_PRIMEP_MUL(761, 761)
DEFINE_PRIMEP_MUL(769, 769)
DEFINE_PRIMEP_MUL(773, 773)
DEFINE_PRIMEP_MUL(787, 787)
DEFINE_PRIMEP_MUL(797, 797)
DEFINE_PRIMEP_MUL(809, 809)
DEFINE_PRIMEP_MUL(811, 811)
DEFINE_PRIMEP_MUL(821, 821)
DEFINE_PRIMEP_MUL(823, 823)
DEFINE_PRIMEP_MUL(827, 827)
DEFINE_PRIMEP_MUL(829, 829)
DEFINE_PRIMEP_MUL(839, 839)
DEFINE_PRIMEP_MUL(853, 853)
DEFINE_PRIMEP_MUL(857, 857)
DEFINE_PRIMEP_MUL(859, 859)
DEFINE_PRIMEP_MUL(863, 863)
DEFINE_PRIMEP_MUL(877, 877)
DEFINE_PRIMEP_MUL(881, 881)
DEFINE_PRIMEP_MUL(883, 883)
DEFINE_PRIMEP_MUL(887, 887)
DEFINE_PRIMEP_MUL(907, 907)
DEFINE_PRIMEP_MUL(911, 911)
DEFINE_PRIMEP_MUL(919, 919)
DEFINE_PRIMEP_MUL(929, 929)
DEFINE_PRIMEP_MUL(937, 937)
DEFINE_PRIMEP_MUL(941, 941)
DEFINE_PRIMEP_MUL(947, 947)
DEFINE_PRIMEP_MUL(953, 953)
DEFINE_PRIMEP_MUL(967, 967)
DEFINE_PRIMEP_MUL(971, 971)
DEFINE_PRIMEP_MUL(977, 977)
DEFINE_PRIMEP_MUL(983, 983)
DEFINE_PRIMEP_MUL(991, 991)
DEFINE_PRIMEP_MUL(997, 997)
DEFINE_PRIMEP_MUL(1009, 1009)
DEFINE_PRIMEP_MUL(1013, 1013)
DEFINE_PRIMEP_MUL(1019, 1019)
DEFINE_PRIMEP_MUL(1021, 1021)
DEFINE_PRIMEP_MUL(1031, 1031)
DEFINE_PRIMEP_MUL(1033, 1033)
DEFINE_PRIMEP_MUL(1039, 1039)
DEFINE_PRIMEP_MUL(1049, 1049)
DEFINE_PRIMEP_MUL(1051, 1051)
DEFINE_PRIMEP_MUL(1061, 1061)
DEFINE_PRIMEP_MUL(1063, 1063)
DEFINE_PRIMEP_MUL(1069, 1069)
DEFINE_PRIMEP_MUL(1087, 1087)
DEFINE_PRIMEP_MUL(1091, 1091)
DEFINE_PRIMEP_MUL(1093, 1093)
DEFINE_PRIMEP_MUL(1097, 1097)
DEFINE_PRIMEP_MUL(1103, 1103)
DEFINE_PRIMEP_MUL(1109, 1109)
DEFINE_PRIMEP_MUL(1117, 1117)
DEFINE_PRIMEP_MUL(1123, 1123)
DEFINE_PRIMEP_MUL(1129, 1129)
DEFINE_PRIMEP_MUL(1151, 1151)
DEFINE_PRIMEP_MUL(1153, 1153)
DEFINE_PRIMEP_MUL(1163, 1163)
DEFINE_PRIMEP_MUL(1171, 1171)
DEFINE_PRIMEP_MUL(1181, 1181)
DEFINE_PRIMEP_MUL(1187, 1187)
DEFINE_PRIMEP_MUL(1193, 1193)
DEFINE_PRIMEP_MUL(1201, 1201)
DEFINE_PRIMEP_MUL(1213, 1213)
DEFINE_PRIMEP_MUL(1217, 1217)
DEFINE_PRIMEP_MUL(1223, 1223)
DEFINE_PRIMEP_MUL(1229, 1229)
DEFINE_PRIMEP_MUL(1231, 1231)
DEFINE_PRIMEP_MUL(1237, 1237)
DEFINE_PRIMEP_MUL(1249, 1249)
DEFINE_PRIMEP_MUL(1259, 1259)
DEFINE_PRIMEP_MUL(1277, 1277)
DEFINE_PRIMEP_MUL(1279, 1279)
DEFINE_PRIMEP_MUL(1283, 1283)
DEFINE_PRIMEP_MUL(1289, 1289)
DEFINE_PRIMEP_MUL(1291, 1291)
DEFINE_PRIMEP_MUL(1297, 1297)
DEFINE_PRIMEP_MUL(1301, 1301)
DEFINE_PRIMEP_MUL(1303, 1303)
DEFINE_PRIMEP_MUL(1307, 1307)
DEFINE_PRIMEP_MUL(1319, 1319)
DEFINE_PRIMEP_MUL(1321, 1321)
DEFINE_PRIMEP_MUL(1327, 1327)
DEFINE_PRIMEP_MUL(1361, 1361)
DEFINE_PRIMEP_MUL(1367, 1367)
DEFINE_PRIMEP_MUL(1373, 1373)
DEFINE_PRIMEP_MUL(1381, 1381)
DEFINE_PRIMEP_MUL(1399, 1399)
DEFINE_PRIMEP_MUL(1409, 1409)
DEFINE_PRIMEP_MUL(1423, 1423)
DEFINE_PRIMEP_MUL(1427, 1427)
DEFINE_PRIMEP_MUL(1429, 1429)
DEFINE_PRIMEP_MUL(1433, 1433)
DEFINE_PRIMEP_MUL(1439, 1439)
DEFINE_PRIMEP_MUL(1447, 1447)
DEFINE_PRIMEP_MUL(1451, 1451)
DEFINE_PRIMEP_MUL(1453, 1453)
DEFINE_PRIMEP_MUL(1459, 1459)
DEFINE_PRIMEP_MUL(1471, 1471)
DEFINE_PRIMEP_MUL(1481, 1481)
DEFINE_PRIMEP_MUL(1483, 1483)
DEFINE_PRIMEP_MUL(1487, 1487)
DEFINE_PRIMEP_MUL(1489, 1489)
DEFINE_PRIMEP_MUL(1493, 1493)
DEFINE_PRIMEP_MUL(1499, 1499)
DEFINE_PRIMEP_MUL(1511, 1511)
DEFINE_PRIMEP_MUL(1523, 1523)
DEFINE_PRIMEP_MUL(1531, 1531)
DEFINE_PRIMEP_MUL(1543, 1543)

/* ---- COMPAND: a non-linear value-warping bijection (μ-law/A-law-style),
 * built via a construction GUARANTEED to be a valid permutation rather
 * than a rounded/truncated curve that risks collisions: compute
 * raw[v]=v^gamma for a fixed gamma, then RANK the 256 raw values (a rank
 * assignment over a fixed-size set is always a bijection by definition,
 * regardless of the underlying curve's rounding behavior). Distinctly
 * shaped from the algebraic operators (XOR/ADD/MUL/GF) -- a smooth
 * non-linear curve rather than a linear-or-field operation. Pointwise
 * bijection -> stride/phase. gamma searched over a small fixed set. */
static u8 compand_tab[4][256], compand_inv_tab[4][256];
static const double COMPAND_GAMMA[4] = { 0.3, 0.5, 2.0, 3.0 };
static void init_compand(void) {
    for (int g = 0; g < 4; g++) {
        double raw[256];
        int idx[256];
        for (int v = 0; v < 256; v++) { raw[v] = pow((double)v, COMPAND_GAMMA[g]); idx[v] = v; }
        for (int a = 1; a < 256; a++) {
            int cur = idx[a]; double curval = raw[cur];
            int b = a - 1;
            while (b >= 0 && raw[idx[b]] > curval) { idx[b + 1] = idx[b]; b--; }
            idx[b + 1] = cur;
        }
        for (int rank = 0; rank < 256; rank++) {
            compand_tab[g][idx[rank]] = (u8)rank;
            compand_inv_tab[g][rank] = (u8)idx[rank];
        }
    }
}
static void ap_compand(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *tab = compand_tab[amp & 3];
    for (int i = p; i < n; i += s) d[i] = tab[d[i]];
}
static void inv_compand(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *tab = compand_inv_tab[amp & 3];
    for (int i = p; i < n; i += s) d[i] = tab[d[i]];
}
static double search_compand(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(2.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int g = 0; g < 4; g++) {
                const u8 *tab = compand_tab[g];
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[tab[u]] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)g; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- COMPAND2: same rank-based warping-bijection mechanism as COMPAND,
 * but a different fixed gamma set -- milder curves closer to 1.0 (plus
 * one steeper outlier) instead of COMPAND's more extreme {0.3,0.5,2,3},
 * giving finer-grained warping shapes for data whose value distribution
 * needs a gentler nonlinear correction. */
static u8 compand2_tab[4][256], compand2_inv_tab[4][256];
static const double COMPAND2_GAMMA[4] = { 0.7, 0.85, 1.15, 1.5 };
static void init_compand2(void) {
    for (int g = 0; g < 4; g++) {
        double raw[256];
        int idx[256];
        for (int v = 0; v < 256; v++) { raw[v] = pow((double)v, COMPAND2_GAMMA[g]); idx[v] = v; }
        for (int a = 1; a < 256; a++) {
            int cur = idx[a]; double curval = raw[cur];
            int b = a - 1;
            while (b >= 0 && raw[idx[b]] > curval) { idx[b + 1] = idx[b]; b--; }
            idx[b + 1] = cur;
        }
        for (int rank = 0; rank < 256; rank++) {
            compand2_tab[g][idx[rank]] = (u8)rank;
            compand2_inv_tab[g][rank] = (u8)idx[rank];
        }
    }
}
static void ap_compand2(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *tab = compand2_tab[amp & 3];
    for (int i = p; i < n; i += s) d[i] = tab[d[i]];
}
static void inv_compand2(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *tab = compand2_inv_tab[amp & 3];
    for (int i = p; i < n; i += s) d[i] = tab[d[i]];
}
static double search_compand2(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(2.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int g = 0; g < 4; g++) {
                const u8 *tab = compand2_tab[g];
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[tab[u]] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)g; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- COMPAND<N> family: same rank-based warping-bijection mechanism as
 * COMPAND/COMPAND2, with further distinct fixed gamma sets -- COMPAND3
 * spans a wider extreme range than COMPAND's {0.3,0.5,2,3}, COMPAND4
 * uses asymmetric fractional/near-1 values. Generated via macro since
 * the rank-based table-building mechanism is identical, only the gamma
 * constants differ. */
#define DEFINE_COMPAND_N(SUF, G0, G1, G2, G3) \
static u8 compand##SUF##_tab[4][256], compand##SUF##_inv_tab[4][256]; \
static const double COMPAND##SUF##_GAMMA[4] = { (G0), (G1), (G2), (G3) }; \
static void init_compand##SUF(void) { \
    for (int g = 0; g < 4; g++) { \
        double raw[256]; \
        int idx[256]; \
        for (int v = 0; v < 256; v++) { raw[v] = pow((double)v, COMPAND##SUF##_GAMMA[g]); idx[v] = v; } \
        for (int a = 1; a < 256; a++) { \
            int cur = idx[a]; double curval = raw[cur]; \
            int b = a - 1; \
            while (b >= 0 && raw[idx[b]] > curval) { idx[b + 1] = idx[b]; b--; } \
            idx[b + 1] = cur; \
        } \
        for (int rank = 0; rank < 256; rank++) { \
            compand##SUF##_tab[g][idx[rank]] = (u8)rank; \
            compand##SUF##_inv_tab[g][rank] = (u8)idx[rank]; \
        } \
    } \
} \
static void ap_compand##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    const u8 *tab = compand##SUF##_tab[amp & 3]; \
    for (int i = p; i < n; i += s) d[i] = tab[d[i]]; \
} \
static void inv_compand##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    const u8 *tab = compand##SUF##_inv_tab[amp & 3]; \
    for (int i = p; i < n; i += s) d[i] = tab[d[i]]; \
} \
static double search_compand##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(2.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int g = 0; g < 4; g++) { \
                const u8 *tab = compand##SUF##_tab[g]; \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int u = 0; u < 256; u++) rf[tab[u]] += hit[u]; \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)g; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_COMPAND_N(3, 0.15, 0.25, 4.0, 6.0)
DEFINE_COMPAND_N(4, 0.6, 0.95, 1.05, 1.4)
DEFINE_COMPAND_N(5, 0.1, 0.4, 0.6, 0.9)
DEFINE_COMPAND_N(6, 1.1, 1.3, 1.6, 2.0)
DEFINE_COMPAND_N(7, 2.5, 3.5, 5.0, 8.0)
DEFINE_COMPAND_N(8, 0.2, 1.8, 3.0, 4.5)

/* ---- SPARSE_XOR: explicit position-list selection -- unlike every rule-
 * based selector above (stride, value, neighbor state: all free to
 * recompute), this enumerates K=4 SPECIFIC positions to XOR by a shared
 * constant, paying real bits to store which positions (12 bits each,
 * enough for 0..4095). Only worth it when the benefit is concentrated in
 * a handful of individually anomalous positions no implicit rule
 * characterizes cheaply. amp: 4 x 12-bit positions (48 bits) doesn't fit
 * u32, so this uses only K=2 positions (24 bits) + 8-bit constant = 32
 * bits, fits exactly. Self-inverse. Search: greedily pick the single most
 * anomalous position, then the second-most (holding the first fixed). */
#define SPARSE_K 2
static void ap_sparsexor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)n;
    u8 c = (u8)(amp & 0xFF);
    int pos0 = (int)((amp >> 8) & 0xFFF), pos1 = (int)((amp >> 20) & 0xFFF);
    d[pos0] ^= c; d[pos1] ^= c;
}
static double search_sparsexor(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; u32 ba = 0;
    for (int c = 1; c < 256; c++) {
        /* first position: best single-position XOR by c */
        double bestg1 = -1e18; int bp0 = 0;
        for (int i0 = 0; i0 < n; i0++) {
            int rf[256]; memcpy(rf, total, sizeof rf);
            rf[d[i0]]--; rf[d[i0] ^ c]++;
            double S = S_from_freq(rf);
            if (S > bestg1) { bestg1 = S; bp0 = i0; }
        }
        /* second position: best additional single-position XOR by c, given the first is already applied */
        int mid[256]; memcpy(mid, total, sizeof mid);
        mid[d[bp0]]--; mid[d[bp0] ^ c]++;
        double bestg2 = -1e18; int bp1 = 0;
        for (int i1 = 0; i1 < n; i1++) {
            if (i1 == bp0) continue;
            int rf[256]; memcpy(rf, mid, sizeof rf);
            rf[d[i1]]--; rf[d[i1] ^ c]++;
            double S = S_from_freq(rf);
            if (S > bestg2) { bestg2 = S; bp1 = i1; }
        }
        double net = (bestg2 - Sb) - oh_flat(32.0);
        if (net > best) { best = net; ba = (u32)c | ((u32)bp0 << 8) | ((u32)bp1 << 20); }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- AUTOPERIOD_XOR: same mechanism as XOR_PHASE (subset selection via
 * modular stride/phase, shared XOR constant), but searches strides
 * 65..256 instead of 1..64 -- autocorrelation-guided period detection
 * for periodicity too long for XOR_PHASE's small-stride assumption,
 * rather than a genuinely different mechanism. Self-inverse. */
#define AUTOPERIOD_MIN 65
#define AUTOPERIOD_MAX 256
static void ap_autoperiod(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] ^= (u8)amp;
}
static double search_autoperiod(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = AUTOPERIOD_MIN, bp = 0; u32 ba = 1;
    for (int s = AUTOPERIOD_MIN; s <= AUTOPERIOD_MAX; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[u ^ a] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- AUTOPERIOD2_XOR: same mechanism as AUTOPERIOD_XOR, but searches
 * strides 257..512 instead of 65..256 -- reaches for even longer
 * periodicity than AUTOPERIOD_XOR's range covers, same self-inverse
 * modular-stride-subset-XOR pattern. */
#define AUTOPERIOD2_MIN 257
#define AUTOPERIOD2_MAX 512
static void ap_autoperiod2(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] ^= (u8)amp;
}
static double search_autoperiod2(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = AUTOPERIOD2_MIN, bp = 0; u32 ba = 1;
    for (int s = AUTOPERIOD2_MIN; s <= AUTOPERIOD2_MAX; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[u ^ a] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- AUTOPERIOD3_XOR: same mechanism as AUTOPERIOD_XOR/AUTOPERIOD2_XOR,
 * but searches strides 513..768 -- the next segment further out, same
 * self-inverse modular-stride-subset-XOR pattern. */
#define AUTOPERIOD3_MIN 513
#define AUTOPERIOD3_MAX 768
static void ap_autoperiod3(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] ^= (u8)amp;
}
static double search_autoperiod3(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = AUTOPERIOD3_MIN, bp = 0; u32 ba = 1;
    for (int s = AUTOPERIOD3_MIN; s <= AUTOPERIOD3_MAX; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[u ^ a] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- BITREV_IDX_XOR: select positions where the 12-bit BIT-REVERSED
 * index (van der Corput reindexing; n=4096=2^12) mod s == p, instead of
 * the raw linear index -- a different locality notion than XOR_PHASE's
 * modular stride: positions far apart linearly can be "close" under
 * bit-reversal, and vice versa. Self-inverse. */
static u16 bitrev12_tab[BLOCK];
static void init_bitrev12(void) {
    for (int i = 0; i < BLOCK; i++) {
        int r = 0;
        for (int b = 0; b < 12; b++) r |= ((i >> b) & 1) << (11 - b);
        bitrev12_tab[i] = (u16)r;
    }
}
static void ap_bitrevidx(u8 *d, int n, int s, int p, u32 amp) {
    u8 c = (u8)amp;
    for (int i = 0; i < n; i++) if (bitrev12_tab[i] % s == (u16)p) d[i] ^= c;
}
static double search_bitrevidx(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 1;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = 0; i < n; i++) if (bitrev12_tab[i] % s == (u16)p) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[u ^ a] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- HILBERT_DELTA: reshape 4096 -> 64x64 grid and walk it in Hilbert
 * space-filling-curve order (standard public-domain d2xy algorithm);
 * apply lag-1 delta ALONG that walk instead of linear/raster order --
 * a fundamentally different adjacency notion (two positions far apart
 * linearly, or in raster 2D, can be Hilbert-neighbors, and vice versa).
 * Same descending-apply/ascending-invert reversibility pattern as DELTA,
 * generalized to the Hilbert sequence instead of the linear one -- since
 * the Hilbert walk visits every position exactly once (a bijection),
 * "haven't reached this step yet" still means "still original". No
 * searched parameter. */
static int hilbert_seq[BLOCK];
static void hilbert_d2xy(int order_n, int d, int *x, int *y) {
    int rx, ry, t = d;
    *x = 0; *y = 0;
    for (int s = 1; s < order_n; s *= 2) {
        rx = 1 & (t / 2);
        ry = 1 & (t ^ rx);
        if (ry == 0) {
            if (rx == 1) { *x = s - 1 - *x; *y = s - 1 - *y; }
            int tmp = *x; *x = *y; *y = tmp;
        }
        *x += s * rx; *y += s * ry;
        t /= 4;
    }
}
static void init_hilbert(void) {
    for (int d = 0; d < BLOCK; d++) {
        int x, y; hilbert_d2xy(64, d, &x, &y);
        hilbert_seq[d] = y * 64 + x;
    }
}
static void ap_hilbertdelta(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp; (void)n;
    for (int k = BLOCK - 1; k >= 1; k--) {
        int cur = hilbert_seq[k], prev = hilbert_seq[k - 1];
        d[cur] = (u8)(d[cur] - d[prev]);
    }
}
static void inv_hilbertdelta(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp; (void)n;
    for (int k = 1; k < BLOCK; k++) {
        int cur = hilbert_seq[k], prev = hilbert_seq[k - 1];
        d[cur] = (u8)(d[cur] + d[prev]);
    }
}
static double search_hilbertdelta(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_hilbertdelta(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- BWT_MTF: Burrows-Wheeler transform followed by move-to-front. BWT
 * ALONE is a pure position permutation (sort all cyclic rotations,
 * output the last column) -- entropy-invariant by itself per the Master
 * Principle, exactly like transpose/shuffle. Its whole point is to
 * cluster similar contexts together so a FOLLOWING adaptive pass (MTF)
 * can exploit them; testing BWT alone would just show net = -overhead,
 * so this tests the combined hypothesis, which is the only way BWT can
 * show a nonzero result. Standard LF-mapping inverse (textbook
 * algorithm, but easy to get subtly wrong -- ties in the rotation sort
 * are broken by start index for a well-defined total order). The
 * primary index (which sorted row holds the un-rotated original) is
 * data-derived, not searched -- computed once and stored in amp (12
 * bits) since the decoder cannot reconstruct it from the BWT output
 * alone. */
static const u8 *g_bwt_ptr; static int g_bwt_n;
static int bwt_cmp(const void *pa, const void *pb) {
    int a = *(const int *)pa, b = *(const int *)pb;
    for (int k = 0; k < g_bwt_n; k++) {
        u8 ca = g_bwt_ptr[(a + k) % g_bwt_n];
        u8 cb = g_bwt_ptr[(b + k) % g_bwt_n];
        if (ca != cb) return (int)ca - (int)cb;
    }
    return a - b;
}
static void bwt_forward(const u8 *d, int n, u8 *L, int *primary) {
    static int idx[BLOCK];
    for (int i = 0; i < n; i++) idx[i] = i;
    g_bwt_ptr = d; g_bwt_n = n;
    qsort(idx, (size_t)n, sizeof(int), bwt_cmp);
    for (int r = 0; r < n; r++) {
        int start = idx[r];
        L[r] = d[(start - 1 + n) % n];
        if (start == 0) *primary = r;
    }
}
static void bwt_inverse(const u8 *L, int n, int primary, u8 *out) {
    int cnt[256] = {0};
    for (int i = 0; i < n; i++) cnt[L[i]]++;
    int cum[256]; int running = 0;
    for (int v = 0; v < 256; v++) { cum[v] = running; running += cnt[v]; }
    static int rankarr[BLOCK];
    int seen[256] = {0};
    for (int i = 0; i < n; i++) { rankarr[i] = seen[L[i]]; seen[L[i]]++; }
    static int LF[BLOCK];
    for (int i = 0; i < n; i++) LF[i] = cum[L[i]] + rankarr[i];
    int row = primary;
    for (int i = n - 1; i >= 0; i--) {
        out[i] = L[row];
        row = LF[row];
    }
}
static void ap_bwtmtf(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 L[BLOCK];
    int primary;
    bwt_forward(d, n, L, &primary);
    memcpy(d, L, (size_t)n);
    ap_mtf(d, n, 0, 0, 0);
}
static void inv_bwtmtf(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int primary = (int)(amp & 0xFFF);
    inv_mtf(d, n, 0, 0, 0);
    static u8 out[BLOCK];
    bwt_inverse(d, n, primary, out);
    memcpy(d, out, (size_t)n);
}
static double search_bwtmtf(const u8 *d, int n, double Sb, Instr *out) {
    static u8 L[BLOCK];
    int primary;
    bwt_forward(d, n, L, &primary);
    ap_mtf(L, n, 0, 0, 0);
    double net = (S_of(L, n) - Sb) - oh_flat(12.0);
    out->stride = 0; out->phase = 0; out->amp = (u32)primary;
    return net;
}

/* ---- HADAMARD4: a small transform-domain mix -- 2-stage butterfly
 * network across groups of 4 bytes, reusing BLOCKDIFF2's already-
 * verified fixed 2x2 GF(256) matrix at two different pairings (adjacent,
 * then far). A GENUINE integer/real WHT butterfly is (a,b)->(a+b,a-b),
 * which needs DIVISION BY 2 to invert -- impossible mod 256 since 2 has
 * no inverse there. Doing the butterfly in GF(256) field arithmetic
 * instead sidesteps that (fields have no non-invertible nonzero
 * elements), at the cost of testing a small 4-point mix rather than a
 * full 12-stage 4096-point WHT (deferred: full recursive WHT is
 * significantly more code/risk for uncertain payoff here). No searched
 * parameter; reuses bd_inv00..11 computed by init_blockdiff(). */
static void hadamard4_stage(u8 *a, u8 *b) {
    u8 na = (u8)(gf_mul(BD_M00, *a) ^ gf_mul(BD_M01, *b));
    u8 nb = (u8)(gf_mul(BD_M10, *a) ^ gf_mul(BD_M11, *b));
    *a = na; *b = nb;
}
static void hadamard4_stage_inv(u8 *a, u8 *b) {
    u8 na = (u8)(gf_mul(bd_inv00, *a) ^ gf_mul(bd_inv01, *b));
    u8 nb = (u8)(gf_mul(bd_inv10, *a) ^ gf_mul(bd_inv11, *b));
    *a = na; *b = nb;
}
static void ap_hadamard4(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int base = 0; base + 4 <= n; base += 4) {
        hadamard4_stage(&d[base + 0], &d[base + 1]);
        hadamard4_stage(&d[base + 2], &d[base + 3]);
        hadamard4_stage(&d[base + 0], &d[base + 2]);
        hadamard4_stage(&d[base + 1], &d[base + 3]);
    }
}
static void inv_hadamard4(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int base = 0; base + 4 <= n; base += 4) {
        hadamard4_stage_inv(&d[base + 0], &d[base + 2]);
        hadamard4_stage_inv(&d[base + 1], &d[base + 3]);
        hadamard4_stage_inv(&d[base + 0], &d[base + 1]);
        hadamard4_stage_inv(&d[base + 2], &d[base + 3]);
    }
}
static double search_hadamard4(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_hadamard4(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- BITREV_STRIDE: bit-reversal restricted to a (stride,phase) subset,
 * unlike the existing whole-block BIT_REV (which measured exactly
 * raw dS=0, confirmed empirically -- any pointwise bijection applied to
 * 100% of the block is entropy-invariant). Same fix that made VALUE_XOR/
 * POPCNT_XOR/etc. work once given stride/phase, applied to bit-reversal.
 * Self-inverse (involution). */
static void ap_bitrevstride(u8 *d, int n, int s, int p, u32 amp) {
    (void)amp;
    for (int i = p; i < n; i += s) d[i] = bitrev8(d[i]);
}
static double search_bitrevstride(const u8 *d, int n, double Sb, Instr *out) {
    static u8 brtab[256]; static int init = 0;
    if (!init) { for (int v = 0; v < 256; v++) brtab[v] = bitrev8((u8)v); init = 1; }
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(0.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int rf[256];
            for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
            for (int v = 0; v < 256; v++) rf[brtab[v]] += hit[v];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}

/* ---- REFLECT: Elias zigzag remap around the block's mode M: r=(v-M)
 * signed, out = r>=0 ? 2r : (-2r-1). A pure value permutation (bijection
 * applied to 100% of the block, no conditioning) -- predicted, per the
 * Master Principle, to show exactly raw dS=0 regardless of M, same as
 * whole-block GF_MUL/BYTE_ROT/BIT_REV. Included specifically as an
 * empirical confirmation: REFLECT is real (reduce2.c uses it to
 * concentrate the histogram near 0 for a downstream coder's benefit),
 * it just doesn't move THIS metric, which is the whole point of the
 * distinction this project draws between "lowers S" and "helps a
 * downstream coder without changing S". Self-inverse pattern (own
 * apply/invert since the zigzag encode/decode differ). */
static void ap_reflect(u8 *d, int n, int s, int p, u32 amp) {
    /* NOTE: an earlier version used the plain integer difference
     * (int)d[i]-(int)M, which ranges -255..255 -- too wide for the
     * zigzag output to fit in a byte (e.g. M=0, v=255 gives out=510,
     * silently truncated by the u8 cast, colliding with other inputs).
     * Fixed: take the mod-256 difference and reinterpret it as a signed
     * byte in -128..127, which zigzags exactly onto 0..255. */
    (void)s; (void)p;
    u8 M = (u8)amp;
    for (int i = 0; i < n; i++) {
        u8 diff = (u8)(d[i] - M);
        int r = (diff < 128) ? (int)diff : (int)diff - 256;
        int out = (r >= 0) ? (2 * r) : (-2 * r - 1);
        d[i] = (u8)out;
    }
}
static void inv_reflect(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 M = (u8)amp;
    for (int i = 0; i < n; i++) {
        int out = d[i];
        int r = (out % 2 == 0) ? (out / 2) : (-(out + 1) / 2);
        u8 diff = (u8)r;
        d[i] = (u8)(diff + M);
    }
}
static double search_reflect(const u8 *d, int n, double Sb, Instr *out) {
    int f[256]; freq_of(d, n, f);
    int M = 0;
    for (int v = 1; v < 256; v++) if (f[v] > f[M]) M = v;
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_reflect(scr, n, 0, 0, (u32)M);
    double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
    out->stride = 0; out->phase = 0; out->amp = (u32)M;
    return net;
}

/* ---- LINPRED3: linear predictor from THREE priors with searched integer
 * coefficients: pred = (a*d[i-1] + b*d[i-2] + c*d[i-3]) >> 2, for i>=3.
 * Genuinely higher-order than LINPRED2/DELTA2. Same descending-apply /
 * ascending-invert pattern. Coefficients kept small (-2..1, 2 bits each)
 * to keep the joint search tractable. */
static void ap_linpred3(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int a = (int)(amp & 3) - 2, b = (int)((amp >> 2) & 3) - 2, c = (int)((amp >> 4) & 3) - 2;
    for (int i = n - 1; i >= 3; i--) {
        int pred = (a * (int)d[i - 1] + b * (int)d[i - 2] + c * (int)d[i - 3]) >> 2;
        d[i] = (u8)((int)d[i] - pred);
    }
}
static void inv_linpred3(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int a = (int)(amp & 3) - 2, b = (int)((amp >> 2) & 3) - 2, c = (int)((amp >> 4) & 3) - 2;
    for (int i = 3; i < n; i++) {
        int pred = (a * (int)d[i - 1] + b * (int)d[i - 2] + c * (int)d[i - 3]) >> 2;
        d[i] = (u8)((int)d[i] + pred);
    }
}
static double search_linpred3(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int aidx = 0; aidx < 4; aidx++) {
        for (int bidx = 0; bidx < 4; bidx++) {
            for (int cidx = 0; cidx < 4; cidx++) {
                memcpy(scr, d, (size_t)n);
                u32 amp = (u32)aidx | ((u32)bidx << 2) | ((u32)cidx << 4);
                ap_linpred3(scr, n, 0, 0, amp);
                double net = (S_of(scr, n) - Sb) - oh_flat(6.0);
                if (net > best) { best = net; ba = amp; }
            }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- REGIONAL_ANCHOR: split into 4 regions; each region's OWN first
 * byte (untouched) derives that region's XOR constant via a shared
 * GF(256) multiplier a: c_r = gf_mul(anchor_r, a). Unlike the earlier
 * global ANCHOR_XOR (proven dominated by STRIDE_ADD -- excluding 1 byte
 * out of 4096 is negligibly different from a whole-block-uniform
 * constant), this constant genuinely VARIES by region (4 distinct
 * anchors), escaping the same trap ANCHOR_XOR fell into. Self-inverse. */
#define REGION_K 4
static void ap_regionalanchor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    int rlen = n / REGION_K;
    for (int r = 0; r < REGION_K; r++) {
        int start = r * rlen, end = (r == REGION_K - 1) ? n : start + rlen;
        u8 c = gf_mul(d[start], a);
        for (int i = start + 1; i < end; i++) d[i] ^= c;
    }
}
static double search_regionalanchor(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        memcpy(scr, d, (size_t)n);
        ap_regionalanchor(scr, n, 0, 0, (u32)a);
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- NIB_SWAP_STRIDE: v -> (v<<4)|(v>>4), a fixed involution (swap hi/lo
 * nibbles), restricted to a (stride,phase) subset -- same fix as
 * BITREV_STRIDE, since this is also just a pointwise bijection that would
 * be entropy-invariant applied to 100% of the block. Self-inverse. */
static inline u8 nibswap8(u8 v) { return (u8)((v << 4) | (v >> 4)); }
static void ap_nibswapstride(u8 *d, int n, int s, int p, u32 amp) {
    (void)amp;
    for (int i = p; i < n; i += s) d[i] = nibswap8(d[i]);
}
static double search_nibswapstride(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(0.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int rf[256];
            for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
            for (int v = 0; v < 256; v++) rf[nibswap8((u8)v)] += hit[v];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}

/* ---- BIT_ASWAP: swap adjacent bit pairs (v&0xAA)>>1 | (v&0x55)<<1, a
 * fixed involution restricted to a (stride,phase) subset. Same
 * pointwise-bijection-needs-subset logic. Self-inverse. */
static inline u8 bitaswap8(u8 v) { return (u8)(((v & 0xAA) >> 1) | ((v & 0x55) << 1)); }
static void ap_bitaswap(u8 *d, int n, int s, int p, u32 amp) {
    (void)amp;
    for (int i = p; i < n; i += s) d[i] = bitaswap8(d[i]);
}
static double search_bitaswap(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(0.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int rf[256];
            for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
            for (int v = 0; v < 256; v++) rf[bitaswap8((u8)v)] += hit[v];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}

/* ---- CRMB_GFMUL: independent GF(4) multiply per 2-bit crumb (4 crumbs,
 * each by its own nonzero GF(4) constant 1..3, field x^2+x+1) -- each
 * crumb's output stays within its own 2-bit slice, so the 4 sub-problems
 * are completely independent (unlike cross-crumb ops, no untouched-
 * sibling trick needed; each crumb is simply its own small bijection).
 * Pointwise bijection -> stride/phase. */
static inline u8 gf4_mul(u8 a, u8 b) {
    static const u8 tab[4][4] = { {0,0,0,0}, {0,1,2,3}, {0,2,3,1}, {0,3,1,2} };
    return tab[a & 3][b & 3];
}
static inline u8 gf4_inv(u8 a) { static const u8 inv[4] = {0,1,3,2}; return inv[a & 3]; }
static void ap_crmbgfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4];
    for (int g = 0; g < 4; g++) c[g] = (u8)(((amp >> (g * 2)) & 3) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) {
            u8 crumb = (u8)((v >> (2 * g)) & 3);
            w = (u8)(w | (gf4_mul(crumb, c[g]) << (2 * g)));
        }
        d[i] = w;
    }
}
static void inv_crmbgfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 ci[4];
    for (int g = 0; g < 4; g++) ci[g] = gf4_inv((u8)(((amp >> (g * 2)) & 3) + 1));
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) {
            u8 crumb = (u8)((v >> (2 * g)) & 3);
            w = (u8)(w | (gf4_mul(crumb, ci[g]) << (2 * g)));
        }
        d[i] = w;
    }
}
static double search_crmbgfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int amp = 0; amp < 81; amp++) {
                u8 c[4]; int t = amp;
                for (int g = 0; g < 4; g++) { c[g] = (u8)((t % 3) + 1); t /= 3; }
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    u8 w = 0;
                    for (int g = 0; g < 4; g++) {
                        u8 crumb = (u8)((u >> (2 * g)) & 3);
                        w = (u8)(w | (gf4_mul(crumb, c[g]) << (2 * g)));
                    }
                    rf[w] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) {
                    best = net; bs = s; bp = p;
                    ba = (u32)((c[0]-1) | ((c[1]-1)<<2) | ((c[2]-1)<<4) | ((c[3]-1)<<6));
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- NIB_GFMUL: independent GF(16) multiply per nibble (lo/hi, each by
 * its own nonzero GF(16) constant 1..15, field x^4+x+1). Nibble-
 * granularity sibling of CRMB_GFMUL. Pointwise bijection -> stride/phase. */
static u8 gf16_mul_tab[16][16];
static void init_gf16_mul_tab(void) {
    for (int av = 0; av < 16; av++) {
        for (int bv = 0; bv < 16; bv++) {
            u8 a = (u8)av, b = (u8)bv, p = 0;
            for (int i = 0; i < 4; i++) {
                if (b & 1) p ^= a;
                u8 hi = (u8)(a & 0x8);
                a = (u8)((a << 1) & 0xF);
                if (hi) a ^= 0x3;
                b = (u8)(b >> 1);
            }
            gf16_mul_tab[av][bv] = p;
        }
    }
}
static inline u8 gf16_mul(u8 a, u8 b) { return gf16_mul_tab[a & 0xF][b & 0xF]; }
static u8 gf16_inv(u8 a) {
    for (int c = 1; c < 16; c++) if (gf16_mul(a, (u8)c) == 1) return (u8)c;
    return 1;
}
static void ap_nibgfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0xF) + 1), b = (u8)(((amp >> 4) & 0xF) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = gf16_mul((u8)(v & 0xF), a);
        u8 hi = gf16_mul((u8)((v >> 4) & 0xF), b);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibgfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0xF) + 1), b = (u8)(((amp >> 4) & 0xF) + 1);
    u8 ainv = gf16_inv(a), binv = gf16_inv(b);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = gf16_mul((u8)(v & 0xF), ainv);
        u8 hi = gf16_mul((u8)((v >> 4) & 0xF), binv);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static double search_nibgfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int aidx = 0; aidx < 15; aidx++) {
                for (int bidx = 0; bidx < 15; bidx++) {
                    u8 a = (u8)(aidx + 1), b = (u8)(bidx + 1);
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 lo = gf16_mul((u8)(u & 0xF), a);
                        u8 hi = gf16_mul((u8)((u >> 4) & 0xF), b);
                        rf[(u8)(lo | (hi << 4))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)aidx | ((u32)bidx << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUE_GFMUL: bit k=0 (fixed, never touched) preserved; the other 7
 * bits get GF(128)-multiplied (field x^7+x+1) by a (if bit0=0) or b (if
 * bit0=1) -- a richer, nonlinear-mixing sibling of VALUE_XOR. Self-
 * inverse pattern needs a real inverse (GF multiply, not XOR). Pointwise
 * bijection -> stride/phase. Same independent-group decomposition as
 * VALUE_XOR (the two groups never share output bins since bit0 is
 * preserved), k fixed at 0 to keep the search tractable. */
static u8 gf128_mul_tab[128][128];
static void init_gf128_mul_tab(void) {
    for (int av = 0; av < 128; av++) {
        for (int bv = 0; bv < 128; bv++) {
            u8 a = (u8)av, b = (u8)bv, p = 0;
            for (int i = 0; i < 7; i++) {
                if (b & 1) p ^= a;
                u8 hi = (u8)(a & 0x40);
                a = (u8)((a << 1) & 0x7F);
                if (hi) a ^= 0x03;
                b = (u8)(b >> 1);
            }
            gf128_mul_tab[av][bv] = p;
        }
    }
}
static inline u8 gf128_mul(u8 a, u8 b) { return gf128_mul_tab[a & 0x7F][b & 0x7F]; }
static u8 gf128_inv(u8 a) {
    for (int c = 1; c < 128; c++) if (gf128_mul(a, (u8)c) == 1) return (u8)c;
    return 1;
}
static inline u8 v0_idx(u8 v) { return (u8)(v >> 1); }
static inline u8 v0_unidx(u8 idx, int gbit) { return (u8)((idx << 1) | gbit); }
static void ap_valuegfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = v & 1;
        u8 idx = v0_idx(v);
        u8 nidx = gf128_mul(idx, gbit ? b : a);
        d[i] = v0_unidx(nidx, gbit);
    }
}
static void inv_valuegfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1);
    u8 ainv = gf128_inv(a), binv = gf128_inv(b);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = v & 1;
        u8 idx = v0_idx(v);
        u8 nidx = gf128_mul(idx, gbit ? binv : ainv);
        d[i] = v0_unidx(nidx, gbit);
    }
}
static double search_valuegfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(14.0, s);  /* aidx(7) + bidx(7) */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            double bestlo = -1e18; int baidx = 0;
            for (int aidx = 0; aidx < 127; aidx++) {
                u8 a = (u8)(aidx + 1);
                double s0 = 0.0;
                for (int idx = 0; idx < 128; idx++) {
                    u8 vout = v0_unidx(idx, 0);
                    u8 vin = v0_unidx(gf128_mul(idx, gf128_inv(a)), 0);
                    s0 += hlog[base[vout] + hit[vin]];
                }
                if (s0 > bestlo) { bestlo = s0; baidx = aidx; }
            }
            double besthi = -1e18; int bbidx = 0;
            for (int bidx = 0; bidx < 127; bidx++) {
                u8 b = (u8)(bidx + 1);
                double s1 = 0.0;
                for (int idx = 0; idx < 128; idx++) {
                    u8 vout = v0_unidx(idx, 1);
                    u8 vin = v0_unidx(gf128_mul(idx, gf128_inv(b)), 1);
                    s1 += hlog[base[vout] + hit[vin]];
                }
                if (s1 > besthi) { besthi = s1; bbidx = bidx; }
            }
            double net = (bestlo + besthi - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = (u32)baidx | ((u32)bbidx << 7); }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUE_GFMUL_B3 / VALUE_GFMUL_B4: same mechanism as VALUE_GFMUL, but
 * preserve bit 3 (resp. bit 4) instead of bit 0 as the group-select bit --
 * a different bit position picks which 7-bit GF(128) index the remaining
 * bits form, giving a distinct correlation structure than bit0/bit7
 * (VALUE_GFMULHI). */
static inline u8 vk_idx(u8 v, int k) {
    u8 lo = (u8)(v & ((1 << k) - 1));
    u8 hi = (u8)((v >> (k + 1)) << k);
    return (u8)(lo | hi);
}
static inline u8 vk_unidx(u8 idx, int gbit, int k) {
    u8 lo = (u8)(idx & ((1 << k) - 1));
    u8 hi = (u8)((idx >> k) << (k + 1));
    return (u8)(lo | hi | (u8)(gbit << k));
}
static void ap_valuegfmul_b3(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> 3) & 1;
        u8 idx = vk_idx(v, 3);
        u8 nidx = gf128_mul(idx, gbit ? b : a);
        d[i] = vk_unidx(nidx, gbit, 3);
    }
}
static void inv_valuegfmul_b3(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1);
    u8 ainv = gf128_inv(a), binv = gf128_inv(b);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> 3) & 1;
        u8 idx = vk_idx(v, 3);
        u8 nidx = gf128_mul(idx, gbit ? binv : ainv);
        d[i] = vk_unidx(nidx, gbit, 3);
    }
}
static double search_valuegfmul_b3(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(14.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            double bestlo = -1e18; int baidx = 0;
            for (int aidx = 0; aidx < 127; aidx++) {
                u8 a = (u8)(aidx + 1);
                double s0 = 0.0;
                for (int idx = 0; idx < 128; idx++) {
                    u8 vout = vk_unidx((u8)idx, 0, 3);
                    u8 vin = vk_unidx(gf128_mul((u8)idx, gf128_inv(a)), 0, 3);
                    s0 += hlog[base[vout] + hit[vin]];
                }
                if (s0 > bestlo) { bestlo = s0; baidx = aidx; }
            }
            double besthi = -1e18; int bbidx = 0;
            for (int bidx = 0; bidx < 127; bidx++) {
                u8 b = (u8)(bidx + 1);
                double s1 = 0.0;
                for (int idx = 0; idx < 128; idx++) {
                    u8 vout = vk_unidx((u8)idx, 1, 3);
                    u8 vin = vk_unidx(gf128_mul((u8)idx, gf128_inv(b)), 1, 3);
                    s1 += hlog[base[vout] + hit[vin]];
                }
                if (s1 > besthi) { besthi = s1; bbidx = bidx; }
            }
            double net = (bestlo + besthi - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = (u32)baidx | ((u32)bbidx << 7); }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}
static void ap_valuegfmul_b4(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> 4) & 1;
        u8 idx = vk_idx(v, 4);
        u8 nidx = gf128_mul(idx, gbit ? b : a);
        d[i] = vk_unidx(nidx, gbit, 4);
    }
}
static void inv_valuegfmul_b4(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1);
    u8 ainv = gf128_inv(a), binv = gf128_inv(b);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> 4) & 1;
        u8 idx = vk_idx(v, 4);
        u8 nidx = gf128_mul(idx, gbit ? binv : ainv);
        d[i] = vk_unidx(nidx, gbit, 4);
    }
}
static double search_valuegfmul_b4(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(14.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            double bestlo = -1e18; int baidx = 0;
            for (int aidx = 0; aidx < 127; aidx++) {
                u8 a = (u8)(aidx + 1);
                double s0 = 0.0;
                for (int idx = 0; idx < 128; idx++) {
                    u8 vout = vk_unidx((u8)idx, 0, 4);
                    u8 vin = vk_unidx(gf128_mul((u8)idx, gf128_inv(a)), 0, 4);
                    s0 += hlog[base[vout] + hit[vin]];
                }
                if (s0 > bestlo) { bestlo = s0; baidx = aidx; }
            }
            double besthi = -1e18; int bbidx = 0;
            for (int bidx = 0; bidx < 127; bidx++) {
                u8 b = (u8)(bidx + 1);
                double s1 = 0.0;
                for (int idx = 0; idx < 128; idx++) {
                    u8 vout = vk_unidx((u8)idx, 1, 4);
                    u8 vin = vk_unidx(gf128_mul((u8)idx, gf128_inv(b)), 1, 4);
                    s1 += hlog[base[vout] + hit[vin]];
                }
                if (s1 > besthi) { besthi = s1; bbidx = bidx; }
            }
            double net = (bestlo + besthi - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = (u32)baidx | ((u32)bbidx << 7); }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUE_GFMUL_B<K> family: completes the bit-position sweep for the
 * VALUE_GFMUL mechanism (B0=base VALUE_GFMUL, B7=VALUE_GFMULHI, B3/B4
 * already added) -- fills in preserved-bit positions 1,2,5,6. Generated
 * via macro since the mechanism (vk_idx/vk_unidx-based group split, GF(128)
 * multiply per group) is identical, only the fixed bit position differs. */
#define DEFINE_VALUE_GFMUL_BK(KVAL) \
static void ap_valuegfmul_b##KVAL(u8 *d, int n, int s, int p, u32 amp) { \
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int gbit = (v >> (KVAL)) & 1; \
        u8 idx = vk_idx(v, (KVAL)); \
        u8 nidx = gf128_mul(idx, gbit ? b : a); \
        d[i] = vk_unidx(nidx, gbit, (KVAL)); \
    } \
} \
static void inv_valuegfmul_b##KVAL(u8 *d, int n, int s, int p, u32 amp) { \
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1); \
    u8 ainv = gf128_inv(a), binv = gf128_inv(b); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int gbit = (v >> (KVAL)) & 1; \
        u8 idx = vk_idx(v, (KVAL)); \
        u8 nidx = gf128_mul(idx, gbit ? binv : ainv); \
        d[i] = vk_unidx(nidx, gbit, (KVAL)); \
    } \
} \
static double search_valuegfmul_b##KVAL(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(14.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            double bestlo = -1e18; int baidx = 0; \
            for (int aidx = 0; aidx < 127; aidx++) { \
                u8 a = (u8)(aidx + 1); \
                double s0 = 0.0; \
                for (int idx = 0; idx < 128; idx++) { \
                    u8 vout = vk_unidx((u8)idx, 0, (KVAL)); \
                    u8 vin = vk_unidx(gf128_mul((u8)idx, gf128_inv(a)), 0, (KVAL)); \
                    s0 += hlog[base[vout] + hit[vin]]; \
                } \
                if (s0 > bestlo) { bestlo = s0; baidx = aidx; } \
            } \
            double besthi = -1e18; int bbidx = 0; \
            for (int bidx = 0; bidx < 127; bidx++) { \
                u8 b = (u8)(bidx + 1); \
                double s1 = 0.0; \
                for (int idx = 0; idx < 128; idx++) { \
                    u8 vout = vk_unidx((u8)idx, 1, (KVAL)); \
                    u8 vin = vk_unidx(gf128_mul((u8)idx, gf128_inv(b)), 1, (KVAL)); \
                    s1 += hlog[base[vout] + hit[vin]]; \
                } \
                if (s1 > besthi) { besthi = s1; bbidx = bidx; } \
            } \
            double net = (bestlo + besthi - Sb) - oh; \
            if (net > best) { best = net; bs = s; bp = p; ba = (u32)baidx | ((u32)bbidx << 7); } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_VALUE_GFMUL_BK(1)
DEFINE_VALUE_GFMUL_BK(2)
DEFINE_VALUE_GFMUL_BK(5)
DEFINE_VALUE_GFMUL_BK(6)

/* ---- NIB_POW: independent nonlinear GF(16) power map per nibble
 * (v->v^e within GF(16)*, order 15=3x5, cyclic, generator 0x02 under
 * x^4+x+1) -- nonlinear sibling of NIB_GFMUL (exponentiation composes
 * multiplications nonlinearly, reaching correlations a single multiply
 * can't). phi(15)=8 valid exponents per nibble, a 3-bit index with no
 * waste. Pointwise bijection -> stride/phase. */
static u8 gf16_log[16], gf16_antilog[16];
static void init_gf16_log(void) {
    u8 x = 1;
    for (int i = 0; i < 15; i++) { gf16_antilog[i] = x; gf16_log[x] = (u8)i; x = gf16_mul(x, 0x02); }
}
static u8 nibpow_elist[8], nibpow_einv[8];
static int nibpow_ne;
static u8 nibpow_tab[8][16], nibpow_itab[8][16];
static inline u8 nibpow_raw(u8 v, int e) { if (v == 0) return 0; return gf16_antilog[(gf16_log[v] * e) % 15]; }
static void init_nibpow(void) {
    nibpow_ne = 0;
    for (int e = 1; e < 15; e++) {
        int a = e, b = 15, x0 = 1, x1 = 0;
        while (b != 0) { int q = a / b, t = b; b = a % b; a = t; t = x1; x1 = x0 - q * x1; x0 = t; }
        if (a != 1) continue;
        int einv = ((x0 % 15) + 15) % 15;
        int idx = nibpow_ne++;
        nibpow_elist[idx] = (u8)e; nibpow_einv[idx] = (u8)einv;
        for (int v = 0; v < 16; v++) { nibpow_tab[idx][v] = nibpow_raw((u8)v, e); nibpow_itab[idx][v] = nibpow_raw((u8)v, einv); }
    }
}
static void ap_nibpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *tlo = nibpow_tab[amp & 7], *thi = nibpow_tab[(amp >> 3) & 7];
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = tlo[v & 0xF], hi = thi[(v >> 4) & 0xF];
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *tlo = nibpow_itab[amp & 7], *thi = nibpow_itab[(amp >> 3) & 7];
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = tlo[v & 0xF], hi = thi[(v >> 4) & 0xF];
        d[i] = (u8)(lo | (hi << 4));
    }
}
static double search_nibpow(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(6.0, s);  /* idxlo(3) + idxhi(3) */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int ilo = 0; ilo < nibpow_ne; ilo++) {
                for (int ihi = 0; ihi < nibpow_ne; ihi++) {
                    const u8 *tlo = nibpow_tab[ilo], *thi = nibpow_tab[ihi];
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 lo = tlo[u & 0xF], hi = thi[(u >> 4) & 0xF];
                        rf[(u8)(lo | (hi << 4))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)ilo | ((u32)ihi << 3); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUE_ADD: sign bit (bit7) preserved; ADD-sibling of VALUE_XOR --
 * add alo/ahi mod 128 to the lo7 bits independently per sign group.
 * ADD's carries can't cross into bit7 since lo7 is already contiguous at
 * the bottom. Pointwise bijection -> stride/phase. */
static void ap_valueadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0x7F), ahi = (u8)((amp >> 7) & 0x7F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], top = (u8)(v & 0x80), lo7 = (u8)(v & 0x7F);
        d[i] = (u8)(top | ((lo7 + (top ? ahi : alo)) & 0x7F));
    }
}
static void inv_valueadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0x7F), ahi = (u8)((amp >> 7) & 0x7F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], top = (u8)(v & 0x80), lo7 = (u8)(v & 0x7F);
        d[i] = (u8)(top | ((lo7 - (top ? ahi : alo)) & 0x7F));
    }
}
static double search_valueadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(14.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            double bestlo = -1e18; int balo = 0;
            for (int a = 0; a < 128; a++) {
                double s0 = 0.0;
                for (int lo7 = 0; lo7 < 128; lo7++)
                    s0 += hlog[base[lo7] + hit[(lo7 - a) & 0x7F]];
                if (s0 > bestlo) { bestlo = s0; balo = a; }
            }
            double besthi = -1e18; int bahi = 0;
            for (int a = 0; a < 128; a++) {
                double s1 = 0.0;
                for (int lo7 = 0; lo7 < 128; lo7++)
                    s1 += hlog[base[0x80 | lo7] + hit[0x80 | ((lo7 - a) & 0x7F)]];
                if (s1 > besthi) { besthi = s1; bahi = a; }
            }
            double net = (bestlo + besthi - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = (u32)balo | ((u32)bahi << 7); }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUE_MUL: sign bit preserved; MUL-sibling of VALUE_ADD -- odd
 * multiply mod 128 to lo7 per sign group (bijection since 128=2^7, odd
 * is coprime). amp = aidx|(bidx<<6), a=2*aidx+1 (64 odd values each).
 * Pointwise bijection -> stride/phase. */
static void ap_valuemul(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(2 * (amp & 0x3F) + 1), ahi = (u8)(2 * ((amp >> 6) & 0x3F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], top = (u8)(v & 0x80), lo7 = (u8)(v & 0x7F);
        d[i] = (u8)(top | ((lo7 * (top ? ahi : alo)) & 0x7F));
    }
}
static void inv_valuemul(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(2 * (amp & 0x3F) + 1), ahi = (u8)(2 * ((amp >> 6) & 0x3F) + 1);
    u8 aloinv = (u8)(mul_inv256(alo) & 0x7F), ahiinv = (u8)(mul_inv256(ahi) & 0x7F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], top = (u8)(v & 0x80), lo7 = (u8)(v & 0x7F);
        d[i] = (u8)(top | ((lo7 * (top ? ahiinv : aloinv)) & 0x7F));
    }
}
static double search_valuemul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(12.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            double bestlo = -1e18; int balo = 0;
            for (int aidx = 0; aidx < 64; aidx++) {
                u8 a = (u8)(2 * aidx + 1), ainv = (u8)(mul_inv256(a) & 0x7F);
                double s0 = 0.0;
                for (int lo7 = 0; lo7 < 128; lo7++)
                    s0 += hlog[base[lo7] + hit[(lo7 * ainv) & 0x7F]];
                if (s0 > bestlo) { bestlo = s0; balo = aidx; }
            }
            double besthi = -1e18; int bahi = 0;
            for (int bidx = 0; bidx < 64; bidx++) {
                u8 b = (u8)(2 * bidx + 1), binv = (u8)(mul_inv256(b) & 0x7F);
                double s1 = 0.0;
                for (int lo7 = 0; lo7 < 128; lo7++)
                    s1 += hlog[base[0x80 | lo7] + hit[0x80 | ((lo7 * binv) & 0x7F)]];
                if (s1 > besthi) { besthi = s1; bahi = bidx; }
            }
            double net = (bestlo + besthi - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = (u32)balo | ((u32)bahi << 6); }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUEMAP4ADD: top 2 bits (quartile, never touched) pick one of 4
 * groups; ADD-sibling of VALMAP4_XOR -- add that group's own constant
 * mod 64 to the lower 6 bits. Pointwise bijection -> stride/phase. */
static void ap_valuemap4add(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4]; for (int g = 0; g < 4; g++) c[g] = (u8)((amp >> (g * 6)) & 0x3F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int g = (v >> 6) & 3;
        d[i] = (u8)((v & 0xC0) | (((v & 0x3F) + c[g]) & 0x3F));
    }
}
static void inv_valuemap4add(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4]; for (int g = 0; g < 4; g++) c[g] = (u8)((amp >> (g * 6)) & 0x3F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int g = (v >> 6) & 3;
        d[i] = (u8)((v & 0xC0) | (((v & 0x3F) - c[g]) & 0x3F));
    }
}
static double search_valuemap4add(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(24.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            u32 amp = 0;
            for (int g = 0; g < 4; g++) {
                double bestg = -1e18; int bc = 0;
                for (int c = 0; c < 64; c++) {
                    double sg = 0.0;
                    for (int lo = 0; lo < 64; lo++)
                        sg += hlog[base[(g << 6) | lo] + hit[(g << 6) | ((lo - c) & 0x3F)]];
                    if (sg > bestg) { bestg = sg; bc = c; }
                }
                amp |= (u32)bc << (g * 6);
            }
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) {
                int g = (u >> 6) & 3; u8 c = (u8)((amp >> (g * 6)) & 0x3F);
                rf[(u & 0xC0) | (((u & 0x3F) + c) & 0x3F)] += hit[u];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = amp; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUEMAP4MUL: top 2 bits preserved; MUL-sibling of VALMAP4_XOR --
 * odd multiply mod 64 (32 odd values) to the lower 6 bits per quartile.
 * Pointwise bijection -> stride/phase. */
static u8 mul_inv64(u8 a) {
    for (int c = 1; c < 64; c += 2) if ((u8)((a * c) & 0x3F) == 1) return (u8)c;
    return 1;
}
static void ap_valuemap4mul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4]; for (int g = 0; g < 4; g++) c[g] = (u8)(2 * ((amp >> (g * 5)) & 0x1F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int g = (v >> 6) & 3;
        d[i] = (u8)((v & 0xC0) | (((v & 0x3F) * c[g]) & 0x3F));
    }
}
static void inv_valuemap4mul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4]; for (int g = 0; g < 4; g++) c[g] = mul_inv64((u8)(2 * ((amp >> (g * 5)) & 0x1F) + 1));
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int g = (v >> 6) & 3;
        d[i] = (u8)((v & 0xC0) | (((v & 0x3F) * c[g]) & 0x3F));
    }
}
static double search_valuemap4mul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(20.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            u32 amp = 0;
            for (int g = 0; g < 4; g++) {
                double bestg = -1e18; int bcidx = 0;
                for (int cidx = 0; cidx < 32; cidx++) {
                    u8 c = (u8)(2 * cidx + 1), cinv = mul_inv64(c);
                    double sg = 0.0;
                    for (int lo = 0; lo < 64; lo++)
                        sg += hlog[base[(g << 6) | lo] + hit[(g << 6) | ((lo * cinv) & 0x3F)]];
                    if (sg > bestg) { bestg = sg; bcidx = cidx; }
                }
                amp |= (u32)bcidx << (g * 5);
            }
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) {
                int g = (u >> 6) & 3;
                u8 c = (u8)(2 * ((amp >> (g * 5)) & 0x1F) + 1);
                rf[(u & 0xC0) | (((u & 0x3F) * c) & 0x3F)] += hit[u];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = amp; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- CRMB_CADD: cross-crumb ADD mod 4, ADD-sibling of CRMB_CXOR --
 * crumb k += crumb j (j untouched, j!=k). Pointwise bijection -> stride/
 * phase. */
static void ap_crmbcadd(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3), ck = (u8)((v >> (2 * k)) & 3);
        u8 nck = (u8)((ck + cj) & 3);
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (nck << (2 * k)));
    }
}
static void inv_crmbcadd(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3), ck = (u8)((v >> (2 * k)) & 3);
        u8 nck = (u8)((ck - cj) & 3);
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (nck << (2 * k)));
    }
}
static double search_crmbcadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 cj = (u8)((u >> (2 * j)) & 3), ck = (u8)((u >> (2 * k)) & 3);
                        u8 w = (u8)((u & ~(u8)(3 << (2 * k))) | (((ck + cj) & 3) << (2 * k)));
                        rf[w] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)j | ((u32)k << 2); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- CRMB_IADD: independent ADD mod 4 per crumb (4 crumbs, each by its
 * own constant 0..3 -- unlike CRMB_GFMUL's multiplier codes, 0 is a
 * valid additive constant, no waste in the 2-bit field). Pointwise
 * bijection -> stride/phase. */
static void ap_crmbiadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4]; for (int g = 0; g < 4; g++) c[g] = (u8)((amp >> (g * 2)) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) {
            u8 crumb = (u8)((v >> (2 * g)) & 3);
            w = (u8)(w | (((crumb + c[g]) & 3) << (2 * g)));
        }
        d[i] = w;
    }
}
static void inv_crmbiadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4]; for (int g = 0; g < 4; g++) c[g] = (u8)((amp >> (g * 2)) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) {
            u8 crumb = (u8)((v >> (2 * g)) & 3);
            w = (u8)(w | (((crumb - c[g]) & 3) << (2 * g)));
        }
        d[i] = w;
    }
}
static double search_crmbiadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int amp = 0; amp < 256; amp++) {
                u8 c[4]; for (int g = 0; g < 4; g++) c[g] = (u8)((amp >> (g * 2)) & 3);
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    u8 w = 0;
                    for (int g = 0; g < 4; g++) {
                        u8 crumb = (u8)((u >> (2 * g)) & 3);
                        w = (u8)(w | (((crumb + c[g]) & 3) << (2 * g)));
                    }
                    rf[w] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)amp; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- NIB_MUL16: multiply lo/hi nibbles independently by odd constants
 * mod 16 (coprime to 16, bijection). Nibble-granularity sibling of
 * BYTE_MUL, the way ADD_NIBS is to STRIDE_ADD. Pointwise bijection ->
 * stride/phase. */
static u8 inv_mod16(u8 a) {
    for (int c = 1; c < 16; c += 2) if ((u8)((a * c) & 0xF) == 1) return (u8)c;
    return 1;
}
static void ap_nibmul16(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)(2 * (amp & 7) + 1), b = (u8)(2 * ((amp >> 3) & 7) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = (u8)(((v & 0xF) * a) & 0xF), hi = (u8)((((v >> 4) & 0xF) * b) & 0xF);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibmul16(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)(2 * (amp & 7) + 1), b = (u8)(2 * ((amp >> 3) & 7) + 1);
    u8 ainv = inv_mod16(a), binv = inv_mod16(b);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = (u8)(((v & 0xF) * ainv) & 0xF), hi = (u8)((((v >> 4) & 0xF) * binv) & 0xF);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static double search_nibmul16(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(6.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int amp = 0; amp < 64; amp++) {
                u8 a = (u8)(2 * (amp & 7) + 1), b = (u8)(2 * ((amp >> 3) & 7) + 1);
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    u8 lo = (u8)(((u & 0xF) * a) & 0xF), hi = (u8)((((u >> 4) & 0xF) * b) & 0xF);
                    rf[(u8)(lo | (hi << 4))] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)amp; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- NIB_CROSS_GFMUL: nibble k *= (nibble j, or 1 if nibble j==0), GF(16)
 * -- nibble-granularity sibling of CRMB_CROSSMUL, same zero-annihilator
 * fix as the crumb version. amp is just a 1-bit direction (only 2
 * positions: lo,hi). Pointwise bijection -> stride/phase. */
static void ap_nibcrossgfmul(u8 *d, int n, int s, int p, u32 amp) {
    int dir = (int)(amp & 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
        if (dir == 0) { u8 m = hi ? hi : 1; lo = gf16_mul(lo, m); }
        else          { u8 m = lo ? lo : 1; hi = gf16_mul(hi, m); }
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibcrossgfmul(u8 *d, int n, int s, int p, u32 amp) {
    int dir = (int)(amp & 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; u8 lo = (u8)(v & 0xF), hi = (u8)((v >> 4) & 0xF);
        if (dir == 0) { u8 m = hi ? hi : 1; lo = gf16_mul(lo, gf16_inv(m)); }
        else          { u8 m = lo ? lo : 1; hi = gf16_mul(hi, gf16_inv(m)); }
        d[i] = (u8)(lo | (hi << 4));
    }
}
static double search_nibcrossgfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(1.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int dir = 0; dir < 2; dir++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    u8 lo = (u8)(u & 0xF), hi = (u8)((u >> 4) & 0xF);
                    if (dir == 0) { u8 m = hi ? hi : 1; lo = gf16_mul(lo, m); }
                    else          { u8 m = lo ? lo : 1; hi = gf16_mul(hi, m); }
                    rf[(u8)(lo | (hi << 4))] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)dir; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- CRMB_CROSSMUL: cross-crumb sibling of CRMB_CADD/CRMB_CXOR -- crumb
 * k *= (crumb j, or 1 if crumb j==0), GF(4). Zero-annihilator fix: a raw
 * data-dependent multiplier isn't invertible when crumb j is 0, so
 * substitute 1 -- safe since crumb j is never written by this
 * transform, so decode recomputes the identical substitution. Pointwise
 * bijection -> stride/phase. */
static void ap_crmbcrossmul(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3);
        u8 mult = cj ? cj : 1;
        u8 ck = (u8)((v >> (2 * k)) & 3);
        u8 nck = gf4_mul(ck, mult);
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (nck << (2 * k)));
    }
}
static void inv_crmbcrossmul(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3);
        u8 mult = cj ? cj : 1;
        u8 multinv = gf4_inv(mult);
        u8 ck = (u8)((v >> (2 * k)) & 3);
        u8 nck = gf4_mul(ck, multinv);
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (nck << (2 * k)));
    }
}
static double search_crmbcrossmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 cj = (u8)((u >> (2 * j)) & 3);
                        u8 mult = cj ? cj : 1;
                        u8 ck = (u8)((u >> (2 * k)) & 3);
                        u8 w = (u8)((u & ~(u8)(3 << (2 * k))) | (gf4_mul(ck, mult) << (2 * k)));
                        rf[w] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)j | ((u32)k << 2); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- PRNG_ADD8: 8-stream xorshift16 ADD, stream picked by pos%8 --
 * aligns with 8-byte block-cipher boundaries (unlike PRNG_ADD4's 4
 * streams), whole-block. */
static void ap_prngadd8(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[8]; st[0] = ms; for (int k = 1; k < 8; k++) st[k] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + xs16_next(&st[i & 7]));
}
static void inv_prngadd8(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[8]; st[0] = ms; for (int k = 1; k < 8; k++) st[k] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - xs16_next(&st[i & 7]));
}
static double search_prngadd8(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[8]; s[0] = ms; for (int k = 1; k < 8; k++) s[k] = xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + xs16_next(&s[i & 7]))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- PRNG_XOR8: same 8-stream derivation as PRNG_ADD8, XOR instead of
 * ADD -- self-inverse, competes over the same seed space with a
 * different (bitwise vs arithmetic) bijection family. */
static void ap_prngxor8(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[8]; st[0] = ms; for (int k = 1; k < 8; k++) st[k] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] ^= xs16_next(&st[i & 7]);
}
static double search_prngxor8(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[8]; s[0] = ms; for (int k = 1; k < 8; k++) s[k] = xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ xs16_next(&s[i & 7]))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- BIT_SWAP2: swap bits at distance 2: (0,2)(1,3)(4,6)(5,7), a fixed
 * involution restricted to a (stride,phase) subset -- same shape as
 * BIT_ASWAP (adjacent, distance 1) but a different fixed permutation
 * (distance 2). Self-inverse. */
static inline u8 bitswap2_8(u8 v) {
    u8 out = 0;
    static const int pairmap[8] = { 2, 3, 0, 1, 6, 7, 4, 5 };
    for (int b = 0; b < 8; b++) out |= (u8)(((v >> pairmap[b]) & 1) << b);
    return out;
}
static void ap_bitswap2(u8 *d, int n, int s, int p, u32 amp) {
    (void)amp;
    for (int i = p; i < n; i += s) d[i] = bitswap2_8(d[i]);
}
static double search_bitswap2(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(0.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int rf[256];
            for (int v = 0; v < 256; v++) rf[v] = total[v] - hit[v];
            for (int v = 0; v < 256; v++) rf[bitswap2_8((u8)v)] += hit[v];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = 0;
    return best;
}

/* ---- VALUE_GFPOW: bit 0 preserved; nonlinear GF(128) power map on the
 * other 7 bits, independent exponent per group -- pow-sibling of
 * VALUE_GFMUL (exponentiation is nonlinear, reaches correlations a
 * single field multiply can't). GF(128)* has PRIME order 127, so every
 * exponent 1..126 is invertible, no gcd filtering needed. Pointwise
 * bijection -> stride/phase. Same independent-group decomposition as
 * VALUE_GFMUL/VALUE_XOR. */
static u8 gf128_log[128], gf128_antilog[128];
static void init_gf128_log(void) {
    u8 x = 1;
    for (int i = 0; i < 127; i++) { gf128_antilog[i] = x; gf128_log[x] = (u8)i; x = gf128_mul(x, 0x02); }
}
static u8 vgfpow_elist[126], vgfpow_einv[126];
static int vgfpow_ne;
static u8 vgfpow_tab[126][128], vgfpow_itab[126][128];
static inline u8 vgfpow_raw(u8 v, int e) { if (v == 0) return 0; return gf128_antilog[(gf128_log[v] * e) % 127]; }
static void init_vgfpow(void) {
    vgfpow_ne = 0;
    for (int e = 1; e < 127; e++) {
        int a = e, b = 127, x0 = 1, x1 = 0;
        while (b != 0) { int q = a / b, t = b; b = a % b; a = t; t = x1; x1 = x0 - q * x1; x0 = t; }
        int einv = ((x0 % 127) + 127) % 127;
        int idx = vgfpow_ne++;
        vgfpow_elist[idx] = (u8)e; vgfpow_einv[idx] = (u8)einv;
        for (int v = 0; v < 128; v++) { vgfpow_tab[idx][v] = vgfpow_raw((u8)v, e); vgfpow_itab[idx][v] = vgfpow_raw((u8)v, einv); }
    }
}
static void ap_valuegfpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *ta = vgfpow_tab[amp & 0x7F], *tb = vgfpow_tab[(amp >> 7) & 0x7F];
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = v & 1;
        u8 idx = v0_idx(v);
        u8 nidx = gbit ? tb[idx] : ta[idx];
        d[i] = v0_unidx(nidx, gbit);
    }
}
static void inv_valuegfpow(u8 *d, int n, int s, int p, u32 amp) {
    const u8 *ta = vgfpow_itab[amp & 0x7F], *tb = vgfpow_itab[(amp >> 7) & 0x7F];
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = v & 1;
        u8 idx = v0_idx(v);
        u8 nidx = gbit ? tb[idx] : ta[idx];
        d[i] = v0_unidx(nidx, gbit);
    }
}
static double search_valuegfpow(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(14.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            double bestlo = -1e18; int bea = 0;
            for (int ei = 0; ei < vgfpow_ne; ei++) {
                const u8 *inv_tab = vgfpow_itab[ei];
                double s0 = 0.0;
                for (int idx = 0; idx < 128; idx++)
                    s0 += hlog[base[v0_unidx((u8)idx, 0)] + hit[v0_unidx(inv_tab[idx], 0)]];
                if (s0 > bestlo) { bestlo = s0; bea = ei; }
            }
            double besthi = -1e18; int beb = 0;
            for (int ei = 0; ei < vgfpow_ne; ei++) {
                const u8 *inv_tab = vgfpow_itab[ei];
                double s1 = 0.0;
                for (int idx = 0; idx < 128; idx++)
                    s1 += hlog[base[v0_unidx((u8)idx, 1)] + hit[v0_unidx(inv_tab[idx], 1)]];
                if (s1 > besthi) { besthi = s1; beb = ei; }
            }
            double net = (bestlo + besthi - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = (u32)bea | ((u32)beb << 7); }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUEMAP4GFMUL: top 2 bits (quartile, untouched) pick one of 4
 * groups; GF(64)-multiply-sibling of VALMAP4_XOR/VALMAP4MUL on the
 * lower 6 bits. Reduction poly x^6+x+1 (0x03), verified irreducible via
 * Fermat's little theorem before use (a^63==1 for 30 random a). */
static u8 gf64_mul_tab[64][64];
static void init_gf64_mul_tab(void) {
    for (int av = 0; av < 64; av++) for (int bv = 0; bv < 64; bv++) {
        u8 a = (u8)av, b = (u8)bv, p = 0;
        for (int i = 0; i < 6; i++) {
            if (b & 1) p ^= a;
            u8 hi = (u8)(a & 0x20);
            a = (u8)((a << 1) & 0x3F);
            if (hi) a ^= 0x03;
            b = (u8)(b >> 1);
        }
        gf64_mul_tab[av][bv] = p;
    }
}
static inline u8 gf64_mul(u8 a, u8 b) { return gf64_mul_tab[a & 0x3F][b & 0x3F]; }
static u8 gf64_inv(u8 a) { for (int c = 1; c < 64; c++) if (gf64_mul(a, (u8)c) == 1) return (u8)c; return 1; }
static void ap_valuemap4gfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4]; for (int g = 0; g < 4; g++) c[g] = (u8)(((amp >> (g * 6)) & 0x3F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int g = (v >> 6) & 3;
        d[i] = (u8)((v & 0xC0) | gf64_mul((u8)(v & 0x3F), c[g]));
    }
}
static void inv_valuemap4gfmul(u8 *d, int n, int s, int p, u32 amp) {
    u8 c[4]; for (int g = 0; g < 4; g++) c[g] = gf64_inv((u8)(((amp >> (g * 6)) & 0x3F) + 1));
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int g = (v >> 6) & 3;
        d[i] = (u8)((v & 0xC0) | gf64_mul((u8)(v & 0x3F), c[g]));
    }
}
static double search_valuemap4gfmul(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(24.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            u32 amp = 0;
            for (int g = 0; g < 4; g++) {
                double bestg = -1e18; int bcidx = 0;
                for (int cidx = 0; cidx < 63; cidx++) {
                    u8 c = (u8)(cidx + 1), cinv = gf64_inv(c);
                    double sg = 0.0;
                    for (int lo = 0; lo < 64; lo++)
                        sg += hlog[base[(g << 6) | lo] + hit[(g << 6) | gf64_mul((u8)lo, cinv)]];
                    if (sg > bestg) { bestg = sg; bcidx = cidx; }
                }
                amp |= (u32)bcidx << (g * 6);
            }
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) {
                int g = (u >> 6) & 3; u8 c = (u8)(((amp >> (g * 6)) & 0x3F) + 1);
                rf[(u & 0xC0) | gf64_mul((u8)(u & 0x3F), c)] += hit[u];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = amp; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUEMAP4GFPOW: quartile-preserved nonlinear GF(64) power map --
 * pow-sibling of VM4GFMUL, same relationship as GF_POW to GF_MUL. GF(64)*
 * has order 63=9x7, so exponents coprime to 63 are valid (phi(63)=36). */
static u8 gf64_log[64], gf64_antilog[64];
static void init_gf64_log(void) {
    u8 x = 1;
    for (int i = 0; i < 63; i++) { gf64_antilog[i] = x; gf64_log[x] = (u8)i; x = gf64_mul(x, 0x02); }
}
static u8 vm4pow_elist[36], vm4pow_einv[36];
static int vm4pow_ne;
static u8 vm4pow_tab[36][64], vm4pow_itab[36][64];
static inline u8 vm4pow_raw(u8 v, int e) { if (v == 0) return 0; return gf64_antilog[(gf64_log[v] * e) % 63]; }
static void init_vm4pow(void) {
    vm4pow_ne = 0;
    for (int e = 1; e < 63; e++) {
        int a = e, b = 63, x0 = 1, x1 = 0;
        while (b != 0) { int q = a / b, t = b; b = a % b; a = t; t = x1; x1 = x0 - q * x1; x0 = t; }
        if (a != 1) continue;
        int einv = ((x0 % 63) + 63) % 63;
        int idx = vm4pow_ne++;
        vm4pow_elist[idx] = (u8)e; vm4pow_einv[idx] = (u8)einv;
        for (int v = 0; v < 64; v++) { vm4pow_tab[idx][v] = vm4pow_raw((u8)v, e); vm4pow_itab[idx][v] = vm4pow_raw((u8)v, einv); }
    }
}
static void ap_valuemap4gfpow(u8 *d, int n, int s, int p, u32 amp) {
    u8 e[4]; for (int g = 0; g < 4; g++) e[g] = (u8)(((amp >> (g * 6)) & 0x3F) % vm4pow_ne);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int g = (v >> 6) & 3;
        d[i] = (u8)((v & 0xC0) | vm4pow_tab[e[g]][v & 0x3F]);
    }
}
static void inv_valuemap4gfpow(u8 *d, int n, int s, int p, u32 amp) {
    u8 e[4]; for (int g = 0; g < 4; g++) e[g] = (u8)(((amp >> (g * 6)) & 0x3F) % vm4pow_ne);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int g = (v >> 6) & 3;
        d[i] = (u8)((v & 0xC0) | vm4pow_itab[e[g]][v & 0x3F]);
    }
}
static double search_valuemap4gfpow(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0 * 6.0, s);  /* 4 groups x 6-bit exponent index */
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            u32 amp = 0;
            for (int g = 0; g < 4; g++) {
                double bestg = -1e18; int bei = 0;
                for (int ei = 0; ei < vm4pow_ne; ei++) {
                    double sg = 0.0;
                    for (int lo = 0; lo < 64; lo++)
                        sg += hlog[base[(g << 6) | lo] + hit[(g << 6) | vm4pow_itab[ei][lo]]];
                    if (sg > bestg) { bestg = sg; bei = ei; }
                }
                amp |= (u32)bei << (g * 6);
            }
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) {
                int g = (u >> 6) & 3; int ei = (int)((amp >> (g * 6)) % vm4pow_ne);
                rf[(u & 0xC0) | vm4pow_tab[ei][u & 0x3F]] += hit[u];
            }
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = amp; }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- PATTERN_ADD: explicit 8-bit repeating pattern selects positions,
 * ADD-sibling of PATTERN_XOR -- amp packs pattern(8) + additive
 * constant(8) = 16 bits. Whole-block (pattern IS the positional
 * selector). */
static void ap_patadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 pat = (u8)(amp & 0xFF); u8 c = (u8)((amp >> 8) & 0xFF);
    for (int i = 0; i < n; i++) if ((pat >> (i % PATXA_L)) & 1) d[i] = (u8)(d[i] + c);
}
static void inv_patadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 pat = (u8)(amp & 0xFF); u8 c = (u8)((amp >> 8) & 0xFF);
    for (int i = 0; i < n; i++) if ((pat >> (i % PATXA_L)) & 1) d[i] = (u8)(d[i] - c);
}
static double search_patadd(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 ba = 0;
    for (int pat = 1; pat < 256; pat++) {
        int hit[256] = {0}, tot[256] = {0};
        for (int i = 0; i < n; i++) {
            tot[d[i]]++;
            if ((pat >> (i % PATXA_L)) & 1) hit[d[i]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v];
        for (int c = 1; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[(u + c) & 0xFF] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; ba = (u32)pat | ((u32)c << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- DELTA_XOR: d[i] ^= d[i-lag] (XOR instead of ADD), XOR-sibling of
 * DELTA. Same descending-apply/ascending-invert pattern; self-inverse
 * per step but full trajectory still needs proper direction handling. */
static void ap_deltaxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag = (int)amp;
    for (int i = n - 1; i >= lag; i--) d[i] ^= d[i - lag];
}
static void inv_deltaxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag = (int)amp;
    for (int i = lag; i < n; i++) d[i] ^= d[i - lag];
}
static double search_deltaxor(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 blag = 1;
    static u8 scr[BLOCK];
    for (int lag = 1; lag <= 64; lag++) {
        memcpy(scr, d, (size_t)n);
        for (int i = n - 1; i >= lag; i--) scr[i] ^= scr[i - lag];
        double net = (S_of(scr, n) - Sb) - oh_flat(6.0);
        if (net > best) { best = net; blag = (u32)lag; }
    }
    out->stride = 0; out->phase = 0; out->amp = blag;
    return best;
}

/* ---- WORD_ADD32: treat 4-byte groups as one big-endian 32-bit word, add
 * a constant mod 2^32 -- word-granularity sibling of WORD_ADD16 at
 * double the width; carry can propagate across all 4 bytes, genuine
 * cross-byte coupling escaping the whole-block trap the same way. amp
 * only gives 32 bits total for a 32-bit-domain constant, so the search
 * is restricted to a small candidate set (byte-repeated patterns) rather
 * than the full 2^32 space. */
static void ap_wordadd32(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i + 3 < n; i += 4) {
        u32 w = ((u32)d[i] << 24) | ((u32)d[i+1] << 16) | ((u32)d[i+2] << 8) | d[i+3];
        w = (u32)(w + amp);
        d[i] = (u8)(w >> 24); d[i+1] = (u8)(w >> 16); d[i+2] = (u8)(w >> 8); d[i+3] = (u8)w;
    }
}
static void inv_wordadd32(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i + 3 < n; i += 4) {
        u32 w = ((u32)d[i] << 24) | ((u32)d[i+1] << 16) | ((u32)d[i+2] << 8) | d[i+3];
        w = (u32)(w - amp);
        d[i] = (u8)(w >> 24); d[i+1] = (u8)(w >> 16); d[i+2] = (u8)(w >> 8); d[i+3] = (u8)w;
    }
}
static double search_wordadd32(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bc = 1;
    static u8 scr[BLOCK];
    for (u32 c = 1; c < 65536; c++) {
        u32 cc = c | (c << 16);  /* byte-repeated 32-bit candidate, keeps search tractable */
        for (int i = 0; i + 3 < n; i += 4) {
            u32 w = ((u32)d[i] << 24) | ((u32)d[i+1] << 16) | ((u32)d[i+2] << 8) | d[i+3];
            w = (u32)(w + cc);
            scr[i] = (u8)(w >> 24); scr[i+1] = (u8)(w >> 16); scr[i+2] = (u8)(w >> 8); scr[i+3] = (u8)w;
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(32.0);
        if (net > best) { best = net; bc = cc; }
    }
    out->stride = 0; out->phase = 0; out->amp = bc;
    return best;
}

/* ---- DIAG_ADD: ADD-sibling of DIAG_XOR -- diagonal selection in the
 * 64x64 grid, additive constant instead of XOR. */
static void ap_diagadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)n;
    int off = (int)(amp & 0x7F) - 63;
    u8 c = (u8)((amp >> 7) & 0xFF);
    for (int row = 0; row < GRID_N; row++) {
        int col = row - off;
        if (col >= 0 && col < GRID_N) d[row * GRID_N + col] = (u8)(d[row * GRID_N + col] + c);
    }
}
static void inv_diagadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)n;
    int off = (int)(amp & 0x7F) - 63;
    u8 c = (u8)((amp >> 7) & 0xFF);
    for (int row = 0; row < GRID_N; row++) {
        int col = row - off;
        if (col >= 0 && col < GRID_N) d[row * GRID_N + col] = (u8)(d[row * GRID_N + col] - c);
    }
}
static double search_diagadd(const u8 *d, int n, double Sb, Instr *out) {
    (void)n;
    double oh = oh_flat(15.0);
    double best = -1e18; u32 ba = 0;
    int total[256]; freq_of(d, BLOCK, total);
    for (int off = -63; off <= 63; off++) {
        int hit[256] = {0};
        for (int row = 0; row < GRID_N; row++) {
            int col = row - off;
            if (col >= 0 && col < GRID_N) hit[d[row * GRID_N + col]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        for (int c = 1; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[(u + c) & 0xFF] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; ba = (u32)(off + 63) | ((u32)c << 7); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- MIRROR_ADD: ADD-sibling of MIRROR_XOR -- mirror partner n-1-i
 * (untouched) still drives the correction, additive instead of XOR. */
static void ap_mirroradd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int i = 0; i < n / 2; i++) d[i] = (u8)(d[i] + gf_mul(d[n - 1 - i], a));
}
static void inv_mirroradd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int i = 0; i < n / 2; i++) d[i] = (u8)(d[i] - gf_mul(d[n - 1 - i], a));
}
static double search_mirroradd(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        for (int i = 0; i < n / 2; i++) scr[i] = (u8)(d[i] + gf_mul(d[n - 1 - i], (u8)a));
        for (int i = n / 2; i < n; i++) scr[i] = d[i];
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- FEISTEL_ADD: ADD-sibling of FEISTEL_HLF -- L += gf_mul(R,a)
 * instead of XOR, R (untouched) still serves as the key. */
static void ap_feisteladd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    int half = n / 2;
    for (int i = 0; i < half; i++) d[i] = (u8)(d[i] + gf_mul(d[half + i], a));
}
static void inv_feisteladd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    int half = n / 2;
    for (int i = 0; i < half; i++) d[i] = (u8)(d[i] - gf_mul(d[half + i], a));
}
static double search_feisteladd(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    int half = n / 2;
    for (int a = 1; a < 256; a++) {
        for (int i = 0; i < half; i++) scr[i] = (u8)(d[i] + gf_mul(d[half + i], (u8)a));
        memcpy(scr + half, d + half, (size_t)(n - half));
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- PATTERN4_XOR: explicit repeating pattern, period 4 instead of
 * PATTERN_XOR's period 8 -- a shorter, cheaper-per-position pattern
 * (fewer distinct pattern shapes, but each position gets touched at a
 * coarser granularity). amp packs pattern(4) + xor constant(8) = 12
 * bits. */
#define PAT4_L 4
static void ap_pattern4xor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 pat = (u8)(amp & 0xF); u8 c = (u8)((amp >> 4) & 0xFF);
    for (int i = 0; i < n; i++) if ((pat >> (i % PAT4_L)) & 1) d[i] ^= c;
}
static double search_pattern4xor(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(12.0);
    double best = -1e18; u32 ba = 0;
    for (int pat = 1; pat < 16; pat++) {
        int hit[256] = {0}, tot[256] = {0};
        for (int i = 0; i < n; i++) {
            tot[d[i]]++;
            if ((pat >> (i % PAT4_L)) & 1) hit[d[i]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v];
        for (int c = 1; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; ba = (u32)pat | ((u32)c << 4); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- DUALSHIFT_XOR: v ^= (v>>k1) ^ (v>>k2), two simultaneous self-shift
 * folds (generalizes SELFSHIFT's single-shift fold) -- verified
 * bijective for all k1!=k2 in 1..7 via exhaustive Python check before
 * implementing. Built via a direct lookup table (construct forward,
 * invert by table lookup) rather than a derived closed-form inverse --
 * safest option since the domain is only 256 elements. Pointwise
 * bijection -> stride/phase. k1,k2 restricted to a small set to keep
 * the table-building cost (done once per apply/invert call) and search
 * space reasonable. */
static const int DUALSHIFT_KSET[4] = { 1, 2, 3, 4 };
static void ap_dualshift(u8 *d, int n, int s, int p, u32 amp) {
    int k1 = DUALSHIFT_KSET[amp & 3], k2 = DUALSHIFT_KSET[(amp >> 2) & 3];
    for (int i = p; i < n; i += s) { u8 v = d[i]; d[i] = (u8)(v ^ (v >> k1) ^ (v >> k2)); }
}
static void inv_dualshift(u8 *d, int n, int s, int p, u32 amp) {
    int k1 = DUALSHIFT_KSET[amp & 3], k2 = DUALSHIFT_KSET[(amp >> 2) & 3];
    u8 inv_tab[256];
    for (int v = 0; v < 256; v++) inv_tab[(u8)(v ^ (v >> k1) ^ (v >> k2))] = (u8)v;
    for (int i = p; i < n; i += s) d[i] = inv_tab[d[i]];
}
static double search_dualshift(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k1i = 0; k1i < 4; k1i++) {
                for (int k2i = 0; k2i < 4; k2i++) {
                    if (k1i == k2i) continue;
                    int k1 = DUALSHIFT_KSET[k1i], k2 = DUALSHIFT_KSET[k2i];
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) rf[(u8)(u ^ (u >> k1) ^ (u >> k2))] += hit[u];
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)k1i | ((u32)k2i << 2); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- PATTERN16_XOR: explicit repeating pattern, period 16 -- longer
 * period sibling of PATTERN_XOR(8)/PATTERN4_XOR(4), catching structure
 * at a coarser positional granularity none of the shorter periods can
 * express. amp packs pattern(16) + xor constant(8) = 24 bits. */
#define PAT16_L 16
static void ap_pattern16xor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 pat = (u16)(amp & 0xFFFF); u8 c = (u8)((amp >> 16) & 0xFF);
    for (int i = 0; i < n; i++) if ((pat >> (i % PAT16_L)) & 1) d[i] ^= c;
}
static double search_pattern16xor(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(24.0);
    double best = -1e18; u32 ba = 0;
    for (int pat = 1; pat < 65536; pat++) {
        int hit[256] = {0}, tot[256] = {0};
        for (int i = 0; i < n; i++) {
            tot[d[i]]++;
            if ((pat >> (i % PAT16_L)) & 1) hit[d[i]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v];
        double bestc = -1e18; int bc = 0;
        for (int c = 1; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestc) { bestc = S; bc = c; }
        }
        double net = (bestc - Sb) - oh;
        if (net > best) { best = net; ba = (u32)pat | ((u32)bc << 16); }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- PATTERN<N>_XOR family: fills in every period from 2 to 24 (only
 * 2,4,8,16 existed before), same mechanism as PATTERN16_XOR (arbitrary
 * N-bit pattern selects which byte-positions-mod-N get XORed by a
 * shared constant). Two search strategies depending on period size:
 *  - EXH (period<=16): TRUE exhaustive search over all 2^period
 *    patterns, identical in kind to the existing PATTERN2/4/8/16_XOR.
 *  - SAMP (period 17..24): 2^period is too large to enumerate (2^24 is
 *    already 16 million), so this draws a FIXED budget of 65536 random
 *    candidate patterns (same cost ceiling as PATTERN16_XOR's full
 *    search) via a simple seeded LCG instead of trying all of them --
 *    a heuristic, not a proof of optimality, same tradeoff as
 *    AFFINE_FULL's pruning. Period 25..32 isn't included: pattern+
 *    constant no longer fits in the 32-bit amp field at full precision
 *    once period+8>32, and period 32 specifically already has the
 *    PATTERN32_RUN family (structured contiguous-run patterns) instead. */
#define DEFINE_PATTERN_XOR_EXH(SUF, PERIOD) \
static void ap_patternxor_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u32 pat = amp & ((1u << (PERIOD)) - 1); u8 c = (u8)((amp >> (PERIOD)) & 0xFF); \
    for (int i = 0; i < n; i++) if ((pat >> (i % (PERIOD))) & 1) d[i] ^= c; \
} \
static double search_patternxor_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double oh = oh_flat((double)((PERIOD) + 8)); \
    double best = -1e18; u32 ba = 0; \
    for (u32 pat = 1; pat < (1u << (PERIOD)); pat++) { \
        int hit[256] = {0}, tot[256] = {0}; \
        for (int i = 0; i < n; i++) { \
            tot[d[i]]++; \
            if ((pat >> (i % (PERIOD))) & 1) hit[d[i]]++; \
        } \
        int base[256]; \
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v]; \
        double bestc = -1e18; int bc = 0; \
        for (int c = 1; c < 256; c++) { \
            int rf[256]; memcpy(rf, base, sizeof rf); \
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u]; \
            double S = S_from_freq(rf); \
            if (S > bestc) { bestc = S; bc = c; } \
        } \
        double net = (bestc - Sb) - oh; \
        if (net > best) { best = net; ba = pat | ((u32)bc << (PERIOD)); } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
#define DEFINE_PATTERN_XOR_SAMP(SUF, PERIOD, NSAMP) \
static void ap_patternxor_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u32 pat = amp & ((1u << (PERIOD)) - 1); u8 c = (u8)((amp >> (PERIOD)) & 0xFF); \
    for (int i = 0; i < n; i++) if ((pat >> (i % (PERIOD))) & 1) d[i] ^= c; \
} \
static double search_patternxor_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double oh = oh_flat((double)((PERIOD) + 8)); \
    double best = -1e18; u32 ba = 1; \
    u32 rngstate = 0x9E3779B9u; \
    for (int k = 0; k < (NSAMP); k++) { \
        rngstate = rngstate * 1103515245u + 12345u; \
        u32 pat = 1u + (rngstate % (((u32)1 << (PERIOD)) - 1u)); \
        int hit[256] = {0}, tot[256] = {0}; \
        for (int i = 0; i < n; i++) { \
            tot[d[i]]++; \
            if ((pat >> (i % (PERIOD))) & 1) hit[d[i]]++; \
        } \
        int base[256]; \
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v]; \
        double bestc = -1e18; int bc = 0; \
        for (int c = 1; c < 256; c++) { \
            int rf[256]; memcpy(rf, base, sizeof rf); \
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u]; \
            double S = S_from_freq(rf); \
            if (S > bestc) { bestc = S; bc = c; } \
        } \
        double net = (bestc - Sb) - oh; \
        if (net > best) { best = net; ba = pat | ((u32)bc << (PERIOD)); } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_PATTERN_XOR_EXH(n3, 3)
DEFINE_PATTERN_XOR_EXH(n5, 5)
DEFINE_PATTERN_XOR_EXH(n6, 6)
DEFINE_PATTERN_XOR_EXH(n7, 7)
DEFINE_PATTERN_XOR_EXH(n9, 9)
DEFINE_PATTERN_XOR_EXH(n10, 10)
DEFINE_PATTERN_XOR_EXH(n11, 11)
DEFINE_PATTERN_XOR_EXH(n12, 12)
DEFINE_PATTERN_XOR_EXH(n13, 13)
DEFINE_PATTERN_XOR_EXH(n14, 14)
DEFINE_PATTERN_XOR_EXH(n15, 15)
DEFINE_PATTERN_XOR_SAMP(n17, 17, 65536)
DEFINE_PATTERN_XOR_SAMP(n18, 18, 65536)
DEFINE_PATTERN_XOR_SAMP(n19, 19, 65536)
DEFINE_PATTERN_XOR_SAMP(n20, 20, 65536)
DEFINE_PATTERN_XOR_SAMP(n21, 21, 65536)
DEFINE_PATTERN_XOR_SAMP(n22, 22, 65536)
DEFINE_PATTERN_XOR_SAMP(n23, 23, 65536)
DEFINE_PATTERN_XOR_SAMP(n24, 24, 65536)

/* ---- PATTERN32_RUN<K>/PATTERN64_RUN<K> family: PATTERN16_XOR won the
 * second-biggest layer in the layered pass on real data (+26.2 bits),
 * but its mechanism (fully exhaustive search over all 2^16 possible
 * 16-bit patterns) doesn't scale to period 32 or 64 -- 2^32/2^64
 * candidates is completely infeasible. Instead of arbitrary bitmasks,
 * this restricts to CONTIGUOUS runs of K positions within each period,
 * with the run's start position (rotation) searched exhaustively --
 * far cheaper (32 or 64 rotation candidates instead of billions) while
 * still covering a structured, meaningful family of periodic subsets
 * ("the first K out of every 32/64 bytes"). Self-inverse (XOR). K=1..16
 * at each of the two periods, 32 total. */
#define DEFINE_PATTERN_RUN(SUF, PERIOD, KVAL, ROTBITS) \
static void ap_patternrun_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    int rot = (int)(amp & ((1u << (ROTBITS)) - 1)); \
    u8 c = (u8)((amp >> (ROTBITS)) & 0xFF); \
    for (int i = 0; i < n; i++) { \
        int rel = ((i % (PERIOD)) - rot + (PERIOD)) % (PERIOD); \
        if (rel < (KVAL)) d[i] ^= c; \
    } \
} \
static double search_patternrun_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double oh = oh_flat((double)((ROTBITS) + 8)); \
    double best = -1e18; u32 ba = 0; \
    for (int rot = 0; rot < (PERIOD); rot++) { \
        int hit[256] = {0}, tot[256] = {0}; \
        for (int i = 0; i < n; i++) { \
            tot[d[i]]++; \
            int rel = ((i % (PERIOD)) - rot + (PERIOD)) % (PERIOD); \
            if (rel < (KVAL)) hit[d[i]]++; \
        } \
        int base[256]; \
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v]; \
        double bestc = -1e18; int bc = 0; \
        for (int c = 1; c < 256; c++) { \
            int rf[256]; memcpy(rf, base, sizeof rf); \
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u]; \
            double S = S_from_freq(rf); \
            if (S > bestc) { bestc = S; bc = c; } \
        } \
        double net = (bestc - Sb) - oh; \
        if (net > best) { best = net; ba = (u32)rot | ((u32)bc << (ROTBITS)); } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_PATTERN_RUN(p32k1, 32, 1, 5)
DEFINE_PATTERN_RUN(p32k2, 32, 2, 5)
DEFINE_PATTERN_RUN(p32k3, 32, 3, 5)
DEFINE_PATTERN_RUN(p32k4, 32, 4, 5)
DEFINE_PATTERN_RUN(p32k5, 32, 5, 5)
DEFINE_PATTERN_RUN(p32k6, 32, 6, 5)
DEFINE_PATTERN_RUN(p32k7, 32, 7, 5)
DEFINE_PATTERN_RUN(p32k8, 32, 8, 5)
DEFINE_PATTERN_RUN(p32k9, 32, 9, 5)
DEFINE_PATTERN_RUN(p32k10, 32, 10, 5)
DEFINE_PATTERN_RUN(p32k11, 32, 11, 5)
DEFINE_PATTERN_RUN(p32k12, 32, 12, 5)
DEFINE_PATTERN_RUN(p32k13, 32, 13, 5)
DEFINE_PATTERN_RUN(p32k14, 32, 14, 5)
DEFINE_PATTERN_RUN(p32k15, 32, 15, 5)
DEFINE_PATTERN_RUN(p32k16, 32, 16, 5)
DEFINE_PATTERN_RUN(p64k1, 64, 1, 6)
DEFINE_PATTERN_RUN(p64k2, 64, 2, 6)
DEFINE_PATTERN_RUN(p64k3, 64, 3, 6)
DEFINE_PATTERN_RUN(p64k4, 64, 4, 6)
DEFINE_PATTERN_RUN(p64k5, 64, 5, 6)
DEFINE_PATTERN_RUN(p64k6, 64, 6, 6)
DEFINE_PATTERN_RUN(p64k7, 64, 7, 6)
DEFINE_PATTERN_RUN(p64k8, 64, 8, 6)
DEFINE_PATTERN_RUN(p64k9, 64, 9, 6)
DEFINE_PATTERN_RUN(p64k10, 64, 10, 6)
DEFINE_PATTERN_RUN(p64k11, 64, 11, 6)
DEFINE_PATTERN_RUN(p64k12, 64, 12, 6)
DEFINE_PATTERN_RUN(p64k13, 64, 13, 6)
DEFINE_PATTERN_RUN(p64k14, 64, 14, 6)
DEFINE_PATTERN_RUN(p64k15, 64, 15, 6)
DEFINE_PATTERN_RUN(p64k16, 64, 16, 6)

/* ---- CRMB_POW: independent nonlinear GF(4) power map per crumb --
 * GF(4)* has order 3 (prime), so phi(3)=2 valid exponents {1,2} per
 * crumb (exponent 1 is identity, only exponent 2 is a genuine map --
 * marginal but completes the crumb-granularity operator family
 * alongside CRMB_GFMUL/CRMB_CROSSMUL). Pointwise bijection -> stride/
 * phase. */
static inline u8 gf4pow(u8 v, int e) { u8 r = 1; for (int i = 0; i < e; i++) r = gf4_mul(r, v); return v ? r : 0; }
static void ap_crmbpow(u8 *d, int n, int s, int p, u32 amp) {
    u8 e[4]; for (int g = 0; g < 4; g++) e[g] = (u8)(((amp >> (g * 2)) & 1) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) w = (u8)(w | (gf4pow((u8)((v >> (2 * g)) & 3), e[g]) << (2 * g)));
        d[i] = w;
    }
}
static void inv_crmbpow(u8 *d, int n, int s, int p, u32 amp) {
    u8 einv[4]; for (int g = 0; g < 4; g++) { int e = (int)((amp >> (g * 2)) & 1) + 1; einv[g] = (e == 1) ? 1 : 2; }
    for (int i = p; i < n; i += s) {
        u8 v = d[i], w = 0;
        for (int g = 0; g < 4; g++) w = (u8)(w | (gf4pow((u8)((v >> (2 * g)) & 3), einv[g]) << (2 * g)));
        d[i] = w;
    }
}
static double search_crmbpow(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int amp = 0; amp < 16; amp++) {
                u8 e[4]; for (int g = 0; g < 4; g++) e[g] = (u8)(((amp >> (g * 2)) & 1) + 1);
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    u8 w = 0;
                    for (int g = 0; g < 4; g++) w = (u8)(w | (gf4pow((u8)((u >> (2 * g)) & 3), e[g]) << (2 * g)));
                    rf[w] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)amp; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUE_GFMUL_HI: same mechanism as VALUE_GFMUL, but preserves bit 7
 * (the sign/top bit) instead of bit 0 -- tests a different hypothesis
 * about which bit carries the useful conditioning signal. Same
 * independent-group decomposition. */
static inline u8 v7_idx(u8 v) { return (u8)(v & 0x7F); }
static inline u8 v7_unidx(u8 idx, int gbit) { return (u8)(idx | (gbit << 7)); }
static void ap_valuegfmulhi(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> 7) & 1;
        u8 idx = v7_idx(v);
        u8 nidx = gf128_mul(idx, gbit ? b : a);
        d[i] = v7_unidx(nidx, gbit);
    }
}
static void inv_valuegfmulhi(u8 *d, int n, int s, int p, u32 amp) {
    u8 a = (u8)((amp & 0x7F) + 1), b = (u8)(((amp >> 7) & 0x7F) + 1);
    u8 ainv = gf128_inv(a), binv = gf128_inv(b);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = (v >> 7) & 1;
        u8 idx = v7_idx(v);
        u8 nidx = gf128_mul(idx, gbit ? binv : ainv);
        d[i] = v7_unidx(nidx, gbit);
    }
}
static double search_valuegfmulhi(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(14.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            double bestlo = -1e18; int baidx = 0;
            for (int aidx = 0; aidx < 127; aidx++) {
                u8 a = (u8)(aidx + 1), ainv = gf128_inv(a);
                double s0 = 0.0;
                for (int idx = 0; idx < 128; idx++)
                    s0 += hlog[base[v7_unidx((u8)idx, 0)] + hit[v7_unidx(gf128_mul((u8)idx, ainv), 0)]];
                if (s0 > bestlo) { bestlo = s0; baidx = aidx; }
            }
            double besthi = -1e18; int bbidx = 0;
            for (int bidx = 0; bidx < 127; bidx++) {
                u8 b = (u8)(bidx + 1), binv = gf128_inv(b);
                double s1 = 0.0;
                for (int idx = 0; idx < 128; idx++)
                    s1 += hlog[base[v7_unidx((u8)idx, 1)] + hit[v7_unidx(gf128_mul((u8)idx, binv), 1)]];
                if (s1 > besthi) { besthi = s1; bbidx = bidx; }
            }
            double net = (bestlo + besthi - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = (u32)baidx | ((u32)bbidx << 7); }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- WINDOW_ADD: ADD-sibling of WINDOW_XOR -- but ADD does NOT preserve
 * a window's XOR-sum invariant (carries break the XOR cancellation
 * trick), so this uses a DIFFERENT invariant: sum mod 256 of a W=256
 * byte window is preserved under uniformly ADDing any constant C to
 * every byte, because 256*C mod 256 == 0 for any C. That only works at
 * W=256 exactly (16 windows across the block), unlike WINDOW_XOR's W=16
 * (any even W works for the XOR version). Self-inverse pattern (own
 * apply/invert, ADD needs subtraction not XOR to undo). */
#define WADD_W 256
static void ap_windowadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int base = 0; base + WADD_W <= n; base += WADD_W) {
        int sum = 0; for (int j = 0; j < WADD_W; j++) sum += d[base + j];
        u8 c = gf_mul((u8)(sum & 0xFF), a);
        for (int j = 0; j < WADD_W; j++) d[base + j] = (u8)(d[base + j] + c);
    }
}
static void inv_windowadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int base = 0; base + WADD_W <= n; base += WADD_W) {
        int sum = 0; for (int j = 0; j < WADD_W; j++) sum += d[base + j];
        u8 c = gf_mul((u8)(sum & 0xFF), a);
        for (int j = 0; j < WADD_W; j++) d[base + j] = (u8)(d[base + j] - c);
    }
}
static double search_windowadd(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        for (int base = 0; base + WADD_W <= n; base += WADD_W) {
            int sum = 0; for (int j = 0; j < WADD_W; j++) sum += d[base + j];
            u8 c = gf_mul((u8)(sum & 0xFF), (u8)a);
            for (int j = 0; j < WADD_W; j++) scr[base + j] = (u8)(d[base + j] + c);
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- WINDOW_ADD_W512: same sum-mod-256-invariant mechanism as
 * WINDOW_ADD, but W=512 (8 windows across the block) instead of W=256 --
 * still exact since 512 is also a multiple of 256, so 512*C mod 256==0
 * for any C, but a much larger window changes which sum-derived key
 * gets shared across how many bytes. */
#define WADD_W512 512
static void ap_windowadd_w512(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int base = 0; base + WADD_W512 <= n; base += WADD_W512) {
        int sum = 0; for (int j = 0; j < WADD_W512; j++) sum += d[base + j];
        u8 c = gf_mul((u8)(sum & 0xFF), a);
        for (int j = 0; j < WADD_W512; j++) d[base + j] = (u8)(d[base + j] + c);
    }
}
static void inv_windowadd_w512(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int base = 0; base + WADD_W512 <= n; base += WADD_W512) {
        int sum = 0; for (int j = 0; j < WADD_W512; j++) sum += d[base + j];
        u8 c = gf_mul((u8)(sum & 0xFF), a);
        for (int j = 0; j < WADD_W512; j++) d[base + j] = (u8)(d[base + j] - c);
    }
}
static double search_windowadd_w512(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        for (int base = 0; base + WADD_W512 <= n; base += WADD_W512) {
            int sum = 0; for (int j = 0; j < WADD_W512; j++) sum += d[base + j];
            u8 c = gf_mul((u8)(sum & 0xFF), (u8)a);
            for (int j = 0; j < WADD_W512; j++) scr[base + j] = (u8)(d[base + j] + c);
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- WINDOW_ADD_W<N> family: same sum-mod-256-invariant mechanism as
 * WINDOW_ADD (W=256)/WINDOW_ADD_W512, filling in widths 1024,2048,4096
 * (all multiples of 256, so the invariant holds exactly; all divide 4096
 * evenly too). Generated via macro. */
#define DEFINE_WINDOWADD_WN(SUF, WIDTH) \
static void ap_windowadd_w##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 a = (u8)amp; \
    for (int base = 0; base + (WIDTH) <= n; base += (WIDTH)) { \
        int sum = 0; for (int j = 0; j < (WIDTH); j++) sum += d[base + j]; \
        u8 c = gf_mul((u8)(sum & 0xFF), a); \
        for (int j = 0; j < (WIDTH); j++) d[base + j] = (u8)(d[base + j] + c); \
    } \
} \
static void inv_windowadd_w##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 a = (u8)amp; \
    for (int base = 0; base + (WIDTH) <= n; base += (WIDTH)) { \
        int sum = 0; for (int j = 0; j < (WIDTH); j++) sum += d[base + j]; \
        u8 c = gf_mul((u8)(sum & 0xFF), a); \
        for (int j = 0; j < (WIDTH); j++) d[base + j] = (u8)(d[base + j] - c); \
    } \
} \
static double search_windowadd_w##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double best = -1e18; u32 ba = 1; \
    static u8 scr[BLOCK]; \
    for (int a = 1; a < 256; a++) { \
        for (int base = 0; base + (WIDTH) <= n; base += (WIDTH)) { \
            int sum = 0; for (int j = 0; j < (WIDTH); j++) sum += d[base + j]; \
            u8 c = gf_mul((u8)(sum & 0xFF), (u8)a); \
            for (int j = 0; j < (WIDTH); j++) scr[base + j] = (u8)(d[base + j] + c); \
        } \
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0); \
        if (net > best) { best = net; ba = (u32)a; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_WINDOWADD_WN(1024, 1024)
DEFINE_WINDOWADD_WN(2048, 2048)
DEFINE_WINDOWADD_WN(4096, 4096)

/* ---- SYNDROME_ADD: ADD-sibling of SYNDROME_XOR, using WINDOW_ADD's
 * W=256 sum-mod-256-invariant trick -- if the window sum's LSB is set,
 * add ahi, else alo. Binary threshold framing (like SYNDROME_XOR) rather
 * than WINDOW_ADD's data-derived multiplicative key. */
static void ap_syndromeadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    for (int base = 0; base + WADD_W <= n; base += WADD_W) {
        int sum = 0; for (int j = 0; j < WADD_W; j++) sum += d[base + j];
        u8 c = (sum & 1) ? ahi : alo;
        for (int j = 0; j < WADD_W; j++) d[base + j] = (u8)(d[base + j] + c);
    }
}
static void inv_syndromeadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    for (int base = 0; base + WADD_W <= n; base += WADD_W) {
        int sum = 0; for (int j = 0; j < WADD_W; j++) sum += d[base + j];
        u8 c = (sum & 1) ? ahi : alo;
        for (int j = 0; j < WADD_W; j++) d[base + j] = (u8)(d[base + j] - c);
    }
}
static double search_syndromeadd(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int alo = 0; alo < 256; alo++) {
        for (int ahi = 0; ahi < 256; ahi++) {
            for (int base = 0; base + WADD_W <= n; base += WADD_W) {
                int sum = 0; for (int j = 0; j < WADD_W; j++) sum += d[base + j];
                u8 c = (sum & 1) ? (u8)ahi : (u8)alo;
                for (int j = 0; j < WADD_W; j++) scr[base + j] = (u8)(d[base + j] + c);
            }
            double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
            if (net > best) { best = net; ba = (u32)alo | ((u32)ahi << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- ADAPT_LMS2: two-tap sign-sign LMS predictor -- predicts from TWO
 * priors (weighted w1,w2, both adapted from the shared residual sign)
 * instead of ADAPT_LMS's single tap. Same local-var-tracked-originals
 * pattern. */
static void ap_lms2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w1 = (int)(amp & 0xF) - 8, w2 = (int)((amp >> 4) & 0xF) - 8;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        int pred = (w1 * (int)pr1 + w2 * (int)pr2) >> 4;
        u8 residual = (u8)((int)orig_i - pred);
        d[i] = residual;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w1 += sgn; if (w1 > 7) w1 = 7; if (w1 < -8) w1 = -8;
        w2 += sgn; if (w2 > 7) w2 = 7; if (w2 < -8) w2 = -8;
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_lms2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w1 = (int)(amp & 0xF) - 8, w2 = (int)((amp >> 4) & 0xF) - 8;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 residual = d[i];
        int pred = (w1 * (int)pr1 + w2 * (int)pr2) >> 4;
        u8 orig_i = (u8)(residual + pred);
        d[i] = orig_i;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w1 += sgn; if (w1 > 7) w1 = 7; if (w1 < -8) w1 = -8;
        w2 += sgn; if (w2 > 7) w2 = 7; if (w2 < -8) w2 = -8;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_lms2(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0x88;
    static u8 scr[BLOCK];
    for (int w1_0 = 0; w1_0 < 16; w1_0++) {
        for (int w2_0 = 0; w2_0 < 16; w2_0++) {
            memcpy(scr, d, (size_t)n);
            u32 amp = (u32)w1_0 | ((u32)w2_0 << 4);
            ap_lms2(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- FEISTEL_QUARTER: split into 4 quarters; Q0 ^= gf_mul(Q2,a) --
 * pairs NON-adjacent quarters (0 and 2), a different structural
 * hypothesis than FEISTEL_HLF's half-split or MIRROR_XOR's reflective
 * pairing. Q1,Q3 untouched. Self-inverse. */
static void ap_feistelquarter(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    int q = n / 4;
    for (int i = 0; i < q; i++) d[i] ^= gf_mul(d[2 * q + i], a);
}
static double search_feistelquarter(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    int q = n / 4;
    for (int a = 1; a < 256; a++) {
        for (int i = 0; i < q; i++) scr[i] = (u8)(d[i] ^ gf_mul(d[2 * q + i], (u8)a));
        memcpy(scr + q, d + q, (size_t)(n - q));
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- PIECEWISE_XOR: XOR-sibling of PIECEWISE_ADD -- K=4 independently-
 * fit chunks, XOR instead of ADD. Self-inverse. Same greedy-independent-
 * fit-then-verify approximation. */
static void ap_piecewisexor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int chunk = n / PW_K;
    for (int c = 0; c < PW_K; c++) {
        u8 x = (u8)((amp >> (c * 8)) & 0xFF);
        int start = c * chunk, end = (c == PW_K - 1) ? n : start + chunk;
        for (int i = start; i < end; i++) d[i] ^= x;
    }
}
static double search_piecewisexor(const u8 *d, int n, double Sb, Instr *out) {
    int chunk = n / PW_K;
    int total[256]; freq_of(d, n, total);
    u32 amp = 0;
    for (int c = 0; c < PW_K; c++) {
        int start = c * chunk, end = (c == PW_K - 1) ? n : start + chunk;
        int hit[256] = {0};
        for (int i = start; i < end; i++) hit[d[i]]++;
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        double bestg = -1e18; int bx = 0;
        for (int x = 1; x < 256; x++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ x] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; bx = x; }
        }
        amp |= (u32)bx << (c * 8);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_piecewisexor(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(32.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

/* ---- VIGENERE_ADD: ADD-sibling of VIGENERE -- repeating L=4-byte key,
 * cycling by position mod 4, additive instead of XOR. Same greedy per-
 * residue independent fit. */
static void ap_vigenereadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 key[VIG_L]; for (int k = 0; k < VIG_L; k++) key[k] = (u8)((amp >> (k * 8)) & 0xFF);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + key[i % VIG_L]);
}
static void inv_vigenereadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 key[VIG_L]; for (int k = 0; k < VIG_L; k++) key[k] = (u8)((amp >> (k * 8)) & 0xFF);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - key[i % VIG_L]);
}
static double search_vigenereadd(const u8 *d, int n, double Sb, Instr *out) {
    u32 amp = 0;
    for (int r = 0; r < VIG_L; r++) {
        int hit[256] = {0}, tot[256] = {0};
        for (int i = 0; i < n; i++) {
            tot[d[i]]++;
            if (i % VIG_L == r) hit[d[i]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v];
        double bestg = -1e18; int bc = 0;
        for (int c = 1; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[(u + c) & 0xFF] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; bc = c; }
        }
        amp |= (u32)bc << (r * 8);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_vigenereadd(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(32.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

/* ---- BIT_MAJ5: target bit k (>=5) XORed with majority of 5 key bits
 * (0,1,2,3,4) -- extends BIT_MAJ3's 3-input majority to 5 inputs, still
 * self-inverse (key bits untouched, decoder recomputes the identical
 * majority). Pointwise bijection -> stride/phase. */
static inline int maj5(int b0, int b1, int b2, int b3, int b4) {
    int sum = b0 + b1 + b2 + b3 + b4;
    return sum >= 3;
}
static void ap_bitmaj5(u8 *d, int n, int s, int p, u32 amp) {
    int k = (int)amp;
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        int m = maj5(v & 1, (v >> 1) & 1, (v >> 2) & 1, (v >> 3) & 1, (v >> 4) & 1);
        d[i] = (u8)(v ^ (u8)(m << k));
    }
}
static double search_bitmaj5(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bk = 5;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(3.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k = 5; k <= 7; k++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) {
                    int m = maj5(u & 1, (u >> 1) & 1, (u >> 2) & 1, (u >> 3) & 1, (u >> 4) & 1);
                    rf[u ^ (m << k)] += hit[u];
                }
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bk = (u32)k; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = bk;
    return best;
}

/* ---- POPCNT_ADD / RANGE_ADD / MAGCLASS_ADD / QR_ADD / RANK_ADD:
 * ADD-mod-16-sibling of the corresponding _XOR conditional instructions
 * -- same conditioning source (hi nibble popcount/magnitude/QR-class/
 * frequency-rank, all untouched), lo nibble gets an ADD mod 16 instead
 * of XOR. ADD mod 16 is still a clean bijection on the lo nibble (no
 * cross-nibble carry since lo is masked to 4 bits before and after).
 * Pointwise bijection -> stride/phase, same tractable-search shape as
 * their _XOR counterparts. */
static void ap_popcntadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hipc = __builtin_popcount((unsigned)((v >> 4) & 0xF));
        u8 x = (hipc >= POPCXOR_T) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF));
    }
}
static void inv_popcntadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hipc = __builtin_popcount((unsigned)((v >> 4) & 0xF));
        u8 x = (hipc >= POPCXOR_T) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF));
    }
}
static double search_popcntadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    static int hipc_of[256]; static int hinit = 0;
    if (!hinit) { for (int v = 0; v < 256; v++) hipc_of[v] = __builtin_popcount((unsigned)((v >> 4) & 0xF)); hinit = 1; }
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (hipc_of[u] >= POPCXOR_T) ? (u8)ahi : (u8)alo;
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

static void ap_rangeadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (hi >= RANGE_T) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF));
    }
}
static void inv_rangeadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (hi >= RANGE_T) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF));
    }
}
static double search_rangeadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hi = (u >> 4) & 0xF;
                        u8 x = (hi >= RANGE_T) ? (u8)ahi : (u8)alo;
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

static void ap_magclassadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (bitlen4(hi) >= MAGCLASS_T) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF));
    }
}
static void inv_magclassadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (bitlen4(hi) >= MAGCLASS_T) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF));
    }
}
static double search_magclassadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    static int bl_of[256]; static int binit = 0;
    if (!binit) { for (int v = 0; v < 256; v++) bl_of[v] = bitlen4((v >> 4) & 0xF); binit = 1; }
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (bl_of[u] >= MAGCLASS_T) ? (u8)ahi : (u8)alo;
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- RANGE_ADD_T<t> / MAGCLASS_ADD_T<t> / POPCNT_ADD_T<t> family: fills
 * a gap -- the _XOR siblings (RANGE_XOR/MAGCLASS_XOR/POPCNT_XOR) each got
 * threshold-variant siblings (T2..T7) earlier, but the _ADD versions
 * never did. Same conditioning source, same valid-threshold constraints
 * (RANGE: 2,4,6,10,12,14; MAGCLASS: 1,2,4; POPCNT: 1,3,4 -- bases 8/3/2
 * already covered), ADD-mod-16 combine instead of XOR. */
#define DEFINE_RANGEADD_KT(SUF, TVAL) \
static void ap_rangeadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = (hi >= (TVAL)) ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF)); \
    } \
} \
static void inv_rangeadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = (hi >= (TVAL)) ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF)); \
    } \
} \
static double search_rangeadd_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int alo = 0; alo < 16; alo++) { \
                for (int ahi = 0; ahi < 16; ahi++) { \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        int hi = (u >> 4) & 0xF; \
                        u8 x = (hi >= (TVAL)) ? (u8)ahi : (u8)alo; \
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
#define DEFINE_MAGCLASSADD_KT(SUF, TVAL) \
static void ap_magclassadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = (bitlen4(hi) >= (TVAL)) ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF)); \
    } \
} \
static void inv_magclassadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = (bitlen4(hi) >= (TVAL)) ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF)); \
    } \
} \
static double search_magclassadd_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int alo = 0; alo < 16; alo++) { \
                for (int ahi = 0; ahi < 16; ahi++) { \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        u8 x = (bitlen4((u >> 4) & 0xF) >= (TVAL)) ? (u8)ahi : (u8)alo; \
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
#define DEFINE_POPCNTADD_KT(SUF, TVAL) \
static void ap_popcntadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hipc = __builtin_popcount((unsigned)((v >> 4) & 0xF)); \
        u8 x = (hipc >= (TVAL)) ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF)); \
    } \
} \
static void inv_popcntadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hipc = __builtin_popcount((unsigned)((v >> 4) & 0xF)); \
        u8 x = (hipc >= (TVAL)) ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF)); \
    } \
} \
static double search_popcntadd_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int alo = 0; alo < 16; alo++) { \
                for (int ahi = 0; ahi < 16; ahi++) { \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        int hipc = __builtin_popcount((unsigned)((u >> 4) & 0xF)); \
                        u8 x = (hipc >= (TVAL)) ? (u8)ahi : (u8)alo; \
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_RANGEADD_KT(t2, 2)
DEFINE_RANGEADD_KT(t3, 4)
DEFINE_RANGEADD_KT(t4, 6)
DEFINE_RANGEADD_KT(t5, 10)
DEFINE_RANGEADD_KT(t6, 12)
DEFINE_RANGEADD_KT(t7, 14)
DEFINE_MAGCLASSADD_KT(t2, 1)
DEFINE_MAGCLASSADD_KT(t3, 2)
DEFINE_MAGCLASSADD_KT(t4, 4)
DEFINE_POPCNTADD_KT(t4, 1)
DEFINE_POPCNTADD_KT(t5, 3)
DEFINE_POPCNTADD_KT(t6, 4)

/* ---- QR_XOR_MOD<M> / QR_ADD_MOD<M> family: same quadratic-residue split
 * mechanism as QR_XOR (mod5)/QR_XOR_MOD7/QR_ADD (mod5), extended to
 * moduli 11 and 13 -- each modulus gives a different partition of the 16
 * hi-nibble values into the two groups. Generated via macro (the
 * is_qr<M> table itself still needs its own init function per modulus). */
#define DEFINE_QR_MOD(SUF, MVAL) \
static int is_qr##SUF[16]; \
static void init_qr##SUF(void) { \
    int sq[MVAL] = { 0 }; \
    for (int x = 0; x < (MVAL); x++) sq[(x * x) % (MVAL)] = 1; \
    for (int h = 0; h < 16; h++) is_qr##SUF[h] = sq[h % (MVAL)]; \
} \
static void ap_qrxor_mod##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = is_qr##SUF[hi] ? ahi : alo; \
        d[i] = (u8)(v ^ x); \
    } \
} \
static double search_qrxor_mod##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int alo = 0; alo < 16; alo++) { \
                for (int ahi = 0; ahi < 16; ahi++) { \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        int hi = (u >> 4) & 0xF; \
                        u8 x = is_qr##SUF[hi] ? (u8)ahi : (u8)alo; \
                        rf[u ^ x] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
} \
static void ap_qradd_mod##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = is_qr##SUF[hi] ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF)); \
    } \
} \
static void inv_qradd_mod##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = is_qr##SUF[hi] ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF)); \
    } \
} \
static double search_qradd_mod##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int alo = 0; alo < 16; alo++) { \
                for (int ahi = 0; ahi < 16; ahi++) { \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        int hi = (u >> 4) & 0xF; \
                        u8 x = is_qr##SUF[hi] ? (u8)ahi : (u8)alo; \
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_QR_MOD(11, 11)
DEFINE_QR_MOD(13, 13)
DEFINE_QR_MOD(17, 17)
DEFINE_QR_MOD(19, 19)
DEFINE_QR_MOD(23, 23)
DEFINE_QR_MOD(29, 29)
DEFINE_QR_MOD(31, 31)
DEFINE_QR_MOD(37, 37)
DEFINE_QR_MOD(41, 41)
DEFINE_QR_MOD(43, 43)
DEFINE_QR_MOD(47, 47)
DEFINE_QR_MOD(53, 53)
DEFINE_QR_MOD(59, 59)
DEFINE_QR_MOD(61, 61)
DEFINE_QR_MOD(67, 67)
DEFINE_QR_MOD(71, 71)
DEFINE_QR_MOD(73, 73)
DEFINE_QR_MOD(79, 79)
DEFINE_QR_MOD(83, 83)
DEFINE_QR_MOD(89, 89)
DEFINE_QR_MOD(97, 97)
DEFINE_QR_MOD(101, 101)
DEFINE_QR_MOD(103, 103)
DEFINE_QR_MOD(107, 107)
DEFINE_QR_MOD(109, 109)
DEFINE_QR_MOD(113, 113)
DEFINE_QR_MOD(127, 127)
DEFINE_QR_MOD(131, 131)
DEFINE_QR_MOD(137, 137)
DEFINE_QR_MOD(139, 139)
DEFINE_QR_MOD(149, 149)
DEFINE_QR_MOD(151, 151)
DEFINE_QR_MOD(157, 157)
DEFINE_QR_MOD(163, 163)
DEFINE_QR_MOD(167, 167)
DEFINE_QR_MOD(173, 173)
DEFINE_QR_MOD(179, 179)
DEFINE_QR_MOD(181, 181)
DEFINE_QR_MOD(191, 191)
DEFINE_QR_MOD(193, 193)
DEFINE_QR_MOD(197, 197)
DEFINE_QR_MOD(199, 199)
DEFINE_QR_MOD(211, 211)
DEFINE_QR_MOD(223, 223)
DEFINE_QR_MOD(227, 227)
DEFINE_QR_MOD(229, 229)
DEFINE_QR_MOD(233, 233)
DEFINE_QR_MOD(239, 239)

/* ---- QR_ADD_MOD7: ADD-sibling of QR_XOR_MOD7, reusing its is_qr7
 * table (already init'd for QR_XOR_MOD7). */
static void ap_qradd_mod7(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = is_qr7[hi] ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF));
    }
}
static void inv_qradd_mod7(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = is_qr7[hi] ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF));
    }
}
static double search_qradd_mod7(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hi = (u >> 4) & 0xF;
                        u8 x = is_qr7[hi] ? (u8)ahi : (u8)alo;
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

static void ap_qradd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = is_qr5[hi] ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF));
    }
}
static void inv_qradd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = is_qr5[hi] ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF));
    }
}
static double search_qradd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = is_qr5[(u >> 4) & 0xF] ? (u8)ahi : (u8)alo;
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

static void ap_rankadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    int rank[16]; rankxor_compute_rank(d, n, rank);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (rank[hi] < RANKXOR_K) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF));
    }
}
static void inv_rankadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    int rank[16]; rankxor_compute_rank(d, n, rank);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (rank[hi] < RANKXOR_K) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF));
    }
}
static double search_rankadd(const u8 *d, int n, double Sb, Instr *out) {
    int rank[16]; rankxor_compute_rank(d, n, rank);
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (rank[(u >> 4) & 0xF] < RANKXOR_K) ? (u8)ahi : (u8)alo;
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUE_MUL_LO: bit0-preserved MUL-sibling of VALUE_MUL (which
 * preserves bit7/sign) -- odd multiply mod 128 on the OTHER 7 bits
 * (compacted via v0_idx/v0_unidx, reused from VALUE_GFMUL). Different
 * preserved-bit hypothesis, integer ring instead of GF(128) field. */
static void ap_valuemullo(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(2 * (amp & 0x3F) + 1), ahi = (u8)(2 * ((amp >> 6) & 0x3F) + 1);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = v & 1;
        u8 idx = v0_idx(v);
        u8 nidx = (u8)((idx * (gbit ? ahi : alo)) & 0x7F);
        d[i] = v0_unidx(nidx, gbit);
    }
}
static void inv_valuemullo(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(2 * (amp & 0x3F) + 1), ahi = (u8)(2 * ((amp >> 6) & 0x3F) + 1);
    u8 aloinv = (u8)(mul_inv256(alo) & 0x7F), ahiinv = (u8)(mul_inv256(ahi) & 0x7F);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int gbit = v & 1;
        u8 idx = v0_idx(v);
        u8 nidx = (u8)((idx * (gbit ? ahiinv : aloinv)) & 0x7F);
        d[i] = v0_unidx(nidx, gbit);
    }
}
static double search_valuemullo(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(12.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            double bestlo = -1e18; int baidx = 0;
            for (int aidx = 0; aidx < 64; aidx++) {
                u8 a = (u8)(2 * aidx + 1), ainv = (u8)(mul_inv256(a) & 0x7F);
                double s0 = 0.0;
                for (int idx = 0; idx < 128; idx++)
                    s0 += hlog[base[v0_unidx((u8)idx, 0)] + hit[v0_unidx((u8)((idx * ainv) & 0x7F), 0)]];
                if (s0 > bestlo) { bestlo = s0; baidx = aidx; }
            }
            double besthi = -1e18; int bbidx = 0;
            for (int bidx = 0; bidx < 64; bidx++) {
                u8 b = (u8)(2 * bidx + 1), binv = (u8)(mul_inv256(b) & 0x7F);
                double s1 = 0.0;
                for (int idx = 0; idx < 128; idx++)
                    s1 += hlog[base[v0_unidx((u8)idx, 1)] + hit[v0_unidx((u8)((idx * binv) & 0x7F), 1)]];
                if (s1 > besthi) { besthi = s1; bbidx = bidx; }
            }
            double net = (bestlo + besthi - Sb) - oh;
            if (net > best) { best = net; bs = s; bp = p; ba = (u32)baidx | ((u32)bbidx << 6); }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- VALUE_MUL_<mask> family: generalizes VALUE_MUL (preserves bit7)/
 * VALUE_MULLO (preserves bit0) to an ARBITRARY subset of preserved bits
 * (any bitmask), each of the 2^popcount(mask) groups getting its own
 * independent odd multiplier applied to the remaining (8-popcount(mask))
 * bits, mod 2^remaining -- same per-group-independent-optimization search
 * as the original (disjoint output classes per group, so best multiplier
 * per group can be found independently and summed). 32 variants: 6 new
 * single-bit positions (bit1..bit6, completing the bit0/bit7 sweep) plus
 * 26 distinct two-bit-selector partitions (all C(8,2)=28 pairs except
 * (5,7) and (6,7), to land exactly on 32 total). */
static inline u8 mask_group_val(u8 v, u8 mask) {
    u8 out = 0; int j = 0;
    for (int b = 0; b < 8; b++) if (mask & (1 << b)) { out |= (u8)(((v >> b) & 1) << j); j++; }
    return out;
}
static inline u8 mask_remain_val(u8 v, u8 mask) {
    u8 out = 0; int j = 0;
    for (int b = 0; b < 8; b++) if (!(mask & (1 << b))) { out |= (u8)(((v >> b) & 1) << j); j++; }
    return out;
}
static inline u8 mask_compose(u8 mask, u8 group_val, u8 remain_val) {
    u8 out = 0; int jg = 0, jr = 0;
    for (int b = 0; b < 8; b++) {
        if (mask & (1 << b)) { out |= (u8)(((group_val >> jg) & 1) << b); jg++; }
        else { out |= (u8)(((remain_val >> jr) & 1) << b); jr++; }
    }
    return out;
}
#define DEFINE_VALUE_MUL_MASK(SUF, MASKVAL, NGROUPS, REMBITS, IDXBITS) \
static void ap_valuemul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 a[NGROUPS]; for (int g = 0; g < (NGROUPS); g++) a[g] = (u8)(2 * ((amp >> (g * (IDXBITS))) & ((1u << (IDXBITS)) - 1)) + 1); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; u8 g = mask_group_val(v, (u8)(MASKVAL)); u8 idx = mask_remain_val(v, (u8)(MASKVAL)); \
        u8 nidx = (u8)((idx * a[g]) & ((1u << (REMBITS)) - 1)); \
        d[i] = mask_compose((u8)(MASKVAL), g, nidx); \
    } \
} \
static void inv_valuemul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 a[NGROUPS], ainv[NGROUPS]; \
    for (int g = 0; g < (NGROUPS); g++) { \
        a[g] = (u8)(2 * ((amp >> (g * (IDXBITS))) & ((1u << (IDXBITS)) - 1)) + 1); \
        ainv[g] = (u8)(mul_inv256(a[g]) & ((1u << (REMBITS)) - 1)); \
    } \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; u8 g = mask_group_val(v, (u8)(MASKVAL)); u8 idx = mask_remain_val(v, (u8)(MASKVAL)); \
        u8 nidx = (u8)((idx * ainv[g]) & ((1u << (REMBITS)) - 1)); \
        d[i] = mask_compose((u8)(MASKVAL), g, nidx); \
    } \
} \
static double search_valuemul_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided((double)((NGROUPS) * (IDXBITS)), s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            u32 amp = 0; double sumS = 0.0; \
            for (int g = 0; g < (NGROUPS); g++) { \
                double bestg = -1e18; int baidx = 0; \
                for (int aidx = 0; aidx < (1 << (IDXBITS)); aidx++) { \
                    u8 av = (u8)(2 * aidx + 1); \
                    u8 ainv = (u8)(mul_inv256(av) & ((1u << (REMBITS)) - 1)); \
                    double sg = 0.0; \
                    for (int idx = 0; idx < (1 << (REMBITS)); idx++) { \
                        u8 vout = mask_compose((u8)(MASKVAL), (u8)g, (u8)idx); \
                        u8 vin  = mask_compose((u8)(MASKVAL), (u8)g, (u8)((idx * ainv) & ((1u << (REMBITS)) - 1))); \
                        sg += hlog[base[vout] + hit[vin]]; \
                    } \
                    if (sg > bestg) { bestg = sg; baidx = aidx; } \
                } \
                amp |= (u32)baidx << (g * (IDXBITS)); \
                sumS += bestg; \
            } \
            double net = (sumS - Sb) - oh; \
            if (net > best) { best = net; bs = s; bp = p; ba = amp; } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_VALUE_MUL_MASK(b1, 0x02, 2, 7, 6)
DEFINE_VALUE_MUL_MASK(b2, 0x04, 2, 7, 6)
DEFINE_VALUE_MUL_MASK(b3, 0x08, 2, 7, 6)
DEFINE_VALUE_MUL_MASK(b4, 0x10, 2, 7, 6)
DEFINE_VALUE_MUL_MASK(b5, 0x20, 2, 7, 6)
DEFINE_VALUE_MUL_MASK(b6, 0x40, 2, 7, 6)
DEFINE_VALUE_MUL_MASK(p01, 0x03, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p02, 0x05, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p03, 0x09, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p04, 0x11, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p05, 0x21, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p06, 0x41, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p07, 0x81, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p12, 0x06, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p13, 0x0A, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p14, 0x12, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p15, 0x22, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p16, 0x42, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p17, 0x82, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p23, 0x0C, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p24, 0x14, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p25, 0x24, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p26, 0x44, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p27, 0x84, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p34, 0x18, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p35, 0x28, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p36, 0x48, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p37, 0x88, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p45, 0x30, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p46, 0x50, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p47, 0x90, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(p56, 0x60, 4, 6, 5)
DEFINE_VALUE_MUL_MASK(t012, 0x07, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t013, 0x0B, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t014, 0x13, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t015, 0x23, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t016, 0x43, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t017, 0x83, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t023, 0x0D, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t024, 0x15, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t025, 0x25, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t026, 0x45, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t027, 0x85, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t034, 0x19, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t035, 0x29, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t036, 0x49, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t037, 0x89, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t045, 0x31, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t046, 0x51, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t047, 0x91, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t056, 0x61, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t057, 0xA1, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t067, 0xC1, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t123, 0x0E, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t124, 0x16, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t125, 0x26, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t126, 0x46, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t127, 0x86, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t134, 0x1A, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t135, 0x2A, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t136, 0x4A, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t137, 0x8A, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t145, 0x32, 8, 5, 4)
DEFINE_VALUE_MUL_MASK(t146, 0x52, 8, 5, 4)

/* ---- NIB_POW_<mask> family: generalizes NIB_POW's fixed low4/high4
 * nibble split to an ARBITRARY 4-bit bipartition of the byte, reusing
 * the same GF(16) exponent tables (nibpow_tab/nibpow_itab) and the
 * mask_group_val/mask_remain_val/mask_compose helpers from the VALUE_MUL
 * mask family. Same JOINT (not independent) search as base NIB_POW,
 * since both halves are transformed on every byte simultaneously
 * (unlike a single-selector-bit split). 32 variants = all distinct
 * 4-bit bipartitions of the byte (C(8,4)/2=35 total, excluding the
 * base's low4/high4 split, taking 32 of the remaining 34). */
#define DEFINE_NIB_POW_MASK(SUF, MASKVAL) \
static void ap_nibpow_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    const u8 *ta = nibpow_tab[amp & 7], *tb = nibpow_tab[(amp >> 3) & 7]; \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; \
        u8 ga = mask_group_val(v, (u8)(MASKVAL)), gb = mask_remain_val(v, (u8)(MASKVAL)); \
        u8 na = ta[ga], nb = tb[gb]; \
        d[i] = mask_compose((u8)(MASKVAL), na, nb); \
    } \
} \
static void inv_nibpow_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    const u8 *ta = nibpow_itab[amp & 7], *tb = nibpow_itab[(amp >> 3) & 7]; \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; \
        u8 ga = mask_group_val(v, (u8)(MASKVAL)), gb = mask_remain_val(v, (u8)(MASKVAL)); \
        u8 na = ta[ga], nb = tb[gb]; \
        d[i] = mask_compose((u8)(MASKVAL), na, nb); \
    } \
} \
static double search_nibpow_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(6.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int ia = 0; ia < nibpow_ne; ia++) { \
                for (int ib = 0; ib < nibpow_ne; ib++) { \
                    const u8 *ta = nibpow_tab[ia], *tb = nibpow_tab[ib]; \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        u8 ga = mask_group_val((u8)u, (u8)(MASKVAL)), gb = mask_remain_val((u8)u, (u8)(MASKVAL)); \
                        u8 na = ta[ga], nb = tb[gb]; \
                        rf[mask_compose((u8)(MASKVAL), na, nb)] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)ia | ((u32)ib << 3); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_NIB_POW_MASK(m17, 0x17)
DEFINE_NIB_POW_MASK(m1b, 0x1B)
DEFINE_NIB_POW_MASK(m1d, 0x1D)
DEFINE_NIB_POW_MASK(m1e, 0x1E)
DEFINE_NIB_POW_MASK(m27, 0x27)
DEFINE_NIB_POW_MASK(m2b, 0x2B)
DEFINE_NIB_POW_MASK(m2d, 0x2D)
DEFINE_NIB_POW_MASK(m2e, 0x2E)
DEFINE_NIB_POW_MASK(m33, 0x33)
DEFINE_NIB_POW_MASK(m35, 0x35)
DEFINE_NIB_POW_MASK(m36, 0x36)
DEFINE_NIB_POW_MASK(m39, 0x39)
DEFINE_NIB_POW_MASK(m3a, 0x3A)
DEFINE_NIB_POW_MASK(m3c, 0x3C)
DEFINE_NIB_POW_MASK(m47, 0x47)
DEFINE_NIB_POW_MASK(m4b, 0x4B)
DEFINE_NIB_POW_MASK(m4d, 0x4D)
DEFINE_NIB_POW_MASK(m4e, 0x4E)
DEFINE_NIB_POW_MASK(m53, 0x53)
DEFINE_NIB_POW_MASK(m55, 0x55)
DEFINE_NIB_POW_MASK(m56, 0x56)
DEFINE_NIB_POW_MASK(m59, 0x59)
DEFINE_NIB_POW_MASK(m5a, 0x5A)
DEFINE_NIB_POW_MASK(m5c, 0x5C)
DEFINE_NIB_POW_MASK(m63, 0x63)
DEFINE_NIB_POW_MASK(m65, 0x65)
DEFINE_NIB_POW_MASK(m66, 0x66)
DEFINE_NIB_POW_MASK(m69, 0x69)
DEFINE_NIB_POW_MASK(m6a, 0x6A)
DEFINE_NIB_POW_MASK(m6c, 0x6C)
DEFINE_NIB_POW_MASK(m71, 0x71)
DEFINE_NIB_POW_MASK(m72, 0x72)
DEFINE_NIB_POW_MASK(m74, 0x74)
DEFINE_NIB_POW_MASK(m78, 0x78)

/* ---- NIB_CXOR_<mask> / NIB_CADD_<mask> family: generalizes NIB_CXOR/
 * NIB_CADD's fixed lo-nibble/hi-nibble split (mask=0x0F) to arbitrary
 * 4-bit bipartitions, same mask_group_val/mask_remain_val/mask_compose
 * mechanism as the NIB_POW mask family -- NIB_CXOR won a layer in the
 * real-data run (crumb^^nibble-level cross-conditioning has genuine
 * signal), but the mechanism only had ONE possible group pair (lo/hi) to
 * work with before; arbitrary masks give 34 more. Both directions
 * (group-A^=group-B vs group-B^=group-A, matching the existing `dir`
 * bit) searched per instruction, same as the originals. 32 masks each
 * (reusing the identical 32-mask set from NIB_POW_MASK's first batch)
 * for 64 total. */
#define DEFINE_NIB_CXOR_MASK(SUF, MASKVAL) \
static void ap_nibcxor_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    int dir = (int)(amp & 1); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; u8 ga = mask_group_val(v, (u8)(MASKVAL)), gb = mask_remain_val(v, (u8)(MASKVAL)); \
        if (dir == 0) { u8 nga = (u8)(ga ^ gb); d[i] = mask_compose((u8)(MASKVAL), nga, gb); } \
        else { u8 ngb = (u8)(gb ^ ga); d[i] = mask_compose((u8)(MASKVAL), ga, ngb); } \
    } \
} \
static double search_nibcxor_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(1.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int dir = 0; dir < 2; dir++) { \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int u = 0; u < 256; u++) { \
                    u8 ga = mask_group_val((u8)u, (u8)(MASKVAL)), gb = mask_remain_val((u8)u, (u8)(MASKVAL)); \
                    u8 w; \
                    if (dir == 0) { u8 nga = (u8)(ga ^ gb); w = mask_compose((u8)(MASKVAL), nga, gb); } \
                    else { u8 ngb = (u8)(gb ^ ga); w = mask_compose((u8)(MASKVAL), ga, ngb); } \
                    rf[w] += hit[u]; \
                } \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)dir; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
#define DEFINE_NIB_CADD_MASK(SUF, MASKVAL) \
static void ap_nibcadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    int dir = (int)(amp & 1); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; u8 ga = mask_group_val(v, (u8)(MASKVAL)), gb = mask_remain_val(v, (u8)(MASKVAL)); \
        if (dir == 0) { u8 nga = (u8)((ga + gb) & 0xF); d[i] = mask_compose((u8)(MASKVAL), nga, gb); } \
        else { u8 ngb = (u8)((gb + ga) & 0xF); d[i] = mask_compose((u8)(MASKVAL), ga, ngb); } \
    } \
} \
static void inv_nibcadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    int dir = (int)(amp & 1); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; u8 ga = mask_group_val(v, (u8)(MASKVAL)), gb = mask_remain_val(v, (u8)(MASKVAL)); \
        if (dir == 0) { u8 nga = (u8)((ga - gb) & 0xF); d[i] = mask_compose((u8)(MASKVAL), nga, gb); } \
        else { u8 ngb = (u8)((gb - ga) & 0xF); d[i] = mask_compose((u8)(MASKVAL), ga, ngb); } \
    } \
} \
static double search_nibcadd_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(1.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int dir = 0; dir < 2; dir++) { \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int u = 0; u < 256; u++) { \
                    u8 ga = mask_group_val((u8)u, (u8)(MASKVAL)), gb = mask_remain_val((u8)u, (u8)(MASKVAL)); \
                    u8 w; \
                    if (dir == 0) { u8 nga = (u8)((ga + gb) & 0xF); w = mask_compose((u8)(MASKVAL), nga, gb); } \
                    else { u8 ngb = (u8)((gb + ga) & 0xF); w = mask_compose((u8)(MASKVAL), ga, ngb); } \
                    rf[w] += hit[u]; \
                } \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)dir; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_NIB_CXOR_MASK(m17, 0x17)
DEFINE_NIB_CXOR_MASK(m1b, 0x1B)
DEFINE_NIB_CXOR_MASK(m1d, 0x1D)
DEFINE_NIB_CXOR_MASK(m1e, 0x1E)
DEFINE_NIB_CXOR_MASK(m27, 0x27)
DEFINE_NIB_CXOR_MASK(m2b, 0x2B)
DEFINE_NIB_CXOR_MASK(m2d, 0x2D)
DEFINE_NIB_CXOR_MASK(m2e, 0x2E)
DEFINE_NIB_CXOR_MASK(m33, 0x33)
DEFINE_NIB_CXOR_MASK(m35, 0x35)
DEFINE_NIB_CXOR_MASK(m36, 0x36)
DEFINE_NIB_CXOR_MASK(m39, 0x39)
DEFINE_NIB_CXOR_MASK(m3a, 0x3A)
DEFINE_NIB_CXOR_MASK(m3c, 0x3C)
DEFINE_NIB_CXOR_MASK(m47, 0x47)
DEFINE_NIB_CXOR_MASK(m4b, 0x4B)
DEFINE_NIB_CXOR_MASK(m4d, 0x4D)
DEFINE_NIB_CXOR_MASK(m4e, 0x4E)
DEFINE_NIB_CXOR_MASK(m53, 0x53)
DEFINE_NIB_CXOR_MASK(m55, 0x55)
DEFINE_NIB_CXOR_MASK(m56, 0x56)
DEFINE_NIB_CXOR_MASK(m59, 0x59)
DEFINE_NIB_CXOR_MASK(m5a, 0x5A)
DEFINE_NIB_CXOR_MASK(m5c, 0x5C)
DEFINE_NIB_CXOR_MASK(m63, 0x63)
DEFINE_NIB_CXOR_MASK(m65, 0x65)
DEFINE_NIB_CXOR_MASK(m66, 0x66)
DEFINE_NIB_CXOR_MASK(m69, 0x69)
DEFINE_NIB_CXOR_MASK(m6a, 0x6A)
DEFINE_NIB_CXOR_MASK(m6c, 0x6C)
DEFINE_NIB_CXOR_MASK(m71, 0x71)
DEFINE_NIB_CXOR_MASK(m72, 0x72)
DEFINE_NIB_CADD_MASK(m17, 0x17)
DEFINE_NIB_CADD_MASK(m1b, 0x1B)
DEFINE_NIB_CADD_MASK(m1d, 0x1D)
DEFINE_NIB_CADD_MASK(m1e, 0x1E)
DEFINE_NIB_CADD_MASK(m27, 0x27)
DEFINE_NIB_CADD_MASK(m2b, 0x2B)
DEFINE_NIB_CADD_MASK(m2d, 0x2D)
DEFINE_NIB_CADD_MASK(m2e, 0x2E)
DEFINE_NIB_CADD_MASK(m33, 0x33)
DEFINE_NIB_CADD_MASK(m35, 0x35)
DEFINE_NIB_CADD_MASK(m36, 0x36)
DEFINE_NIB_CADD_MASK(m39, 0x39)
DEFINE_NIB_CADD_MASK(m3a, 0x3A)
DEFINE_NIB_CADD_MASK(m3c, 0x3C)
DEFINE_NIB_CADD_MASK(m47, 0x47)
DEFINE_NIB_CADD_MASK(m4b, 0x4B)
DEFINE_NIB_CADD_MASK(m4d, 0x4D)
DEFINE_NIB_CADD_MASK(m4e, 0x4E)
DEFINE_NIB_CADD_MASK(m53, 0x53)
DEFINE_NIB_CADD_MASK(m55, 0x55)
DEFINE_NIB_CADD_MASK(m56, 0x56)
DEFINE_NIB_CADD_MASK(m59, 0x59)
DEFINE_NIB_CADD_MASK(m5a, 0x5A)
DEFINE_NIB_CADD_MASK(m5c, 0x5C)
DEFINE_NIB_CADD_MASK(m63, 0x63)
DEFINE_NIB_CADD_MASK(m65, 0x65)
DEFINE_NIB_CADD_MASK(m66, 0x66)
DEFINE_NIB_CADD_MASK(m69, 0x69)
DEFINE_NIB_CADD_MASK(m6a, 0x6A)
DEFINE_NIB_CADD_MASK(m6c, 0x6C)
DEFINE_NIB_CADD_MASK(m71, 0x71)
DEFINE_NIB_CADD_MASK(m72, 0x72)

/* ---- VALMAP_ADD_<mask> family: generalizes VALMAP4ADD's fixed top-2-bit
 * (mask=0xC0) group split to arbitrary bit-subsets, same mechanism as the
 * VALUE_MUL mask family but with per-group ADD (full residue range, not
 * just odd) instead of multiply -- reuses the same mask_group_val/
 * mask_remain_val/mask_compose helpers and per-group-independent search.
 * 27 new two-bit-mask partitions (all C(8,2)=28 pairs except the
 * existing {6,7}) plus 5 one-bit-mask partitions (VALUE_ADD's bit7 is
 * the existing popcount=1 case; b0-b4 fill in 5 more), for 32 total. */
#define DEFINE_VALMAP_ADD_MASK(SUF, MASKVAL, NGROUPS, REMBITS) \
static void ap_valmapadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 c[NGROUPS]; for (int g = 0; g < (NGROUPS); g++) c[g] = (u8)((amp >> (g * (REMBITS))) & ((1u << (REMBITS)) - 1)); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; u8 g = mask_group_val(v, (u8)(MASKVAL)); u8 idx = mask_remain_val(v, (u8)(MASKVAL)); \
        u8 nidx = (u8)((idx + c[g]) & ((1u << (REMBITS)) - 1)); \
        d[i] = mask_compose((u8)(MASKVAL), g, nidx); \
    } \
} \
static void inv_valmapadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 c[NGROUPS]; for (int g = 0; g < (NGROUPS); g++) c[g] = (u8)((amp >> (g * (REMBITS))) & ((1u << (REMBITS)) - 1)); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; u8 g = mask_group_val(v, (u8)(MASKVAL)); u8 idx = mask_remain_val(v, (u8)(MASKVAL)); \
        u8 nidx = (u8)((idx - c[g]) & ((1u << (REMBITS)) - 1)); \
        d[i] = mask_compose((u8)(MASKVAL), g, nidx); \
    } \
} \
static double search_valmapadd_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided((double)((NGROUPS) * (REMBITS)), s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            u32 amp = 0; double sumS = 0.0; \
            for (int g = 0; g < (NGROUPS); g++) { \
                double bestg = -1e18; int bc = 0; \
                for (int c = 0; c < (1 << (REMBITS)); c++) { \
                    double sg = 0.0; \
                    for (int idx = 0; idx < (1 << (REMBITS)); idx++) { \
                        u8 vout = mask_compose((u8)(MASKVAL), (u8)g, (u8)idx); \
                        u8 vin  = mask_compose((u8)(MASKVAL), (u8)g, (u8)((idx - c) & ((1u << (REMBITS)) - 1))); \
                        sg += hlog[base[vout] + hit[vin]]; \
                    } \
                    if (sg > bestg) { bestg = sg; bc = c; } \
                } \
                amp |= (u32)bc << (g * (REMBITS)); \
                sumS += bestg; \
            } \
            double net = (sumS - Sb) - oh; \
            if (net > best) { best = net; bs = s; bp = p; ba = amp; } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_VALMAP_ADD_MASK(p01, 0x03, 4, 6)
DEFINE_VALMAP_ADD_MASK(p02, 0x05, 4, 6)
DEFINE_VALMAP_ADD_MASK(p03, 0x09, 4, 6)
DEFINE_VALMAP_ADD_MASK(p04, 0x11, 4, 6)
DEFINE_VALMAP_ADD_MASK(p05, 0x21, 4, 6)
DEFINE_VALMAP_ADD_MASK(p06, 0x41, 4, 6)
DEFINE_VALMAP_ADD_MASK(p07, 0x81, 4, 6)
DEFINE_VALMAP_ADD_MASK(p12, 0x06, 4, 6)
DEFINE_VALMAP_ADD_MASK(p13, 0x0A, 4, 6)
DEFINE_VALMAP_ADD_MASK(p14, 0x12, 4, 6)
DEFINE_VALMAP_ADD_MASK(p15, 0x22, 4, 6)
DEFINE_VALMAP_ADD_MASK(p16, 0x42, 4, 6)
DEFINE_VALMAP_ADD_MASK(p17, 0x82, 4, 6)
DEFINE_VALMAP_ADD_MASK(p23, 0x0C, 4, 6)
DEFINE_VALMAP_ADD_MASK(p24, 0x14, 4, 6)
DEFINE_VALMAP_ADD_MASK(p25, 0x24, 4, 6)
DEFINE_VALMAP_ADD_MASK(p26, 0x44, 4, 6)
DEFINE_VALMAP_ADD_MASK(p27, 0x84, 4, 6)
DEFINE_VALMAP_ADD_MASK(p34, 0x18, 4, 6)
DEFINE_VALMAP_ADD_MASK(p35, 0x28, 4, 6)
DEFINE_VALMAP_ADD_MASK(p36, 0x48, 4, 6)
DEFINE_VALMAP_ADD_MASK(p37, 0x88, 4, 6)
DEFINE_VALMAP_ADD_MASK(p45, 0x30, 4, 6)
DEFINE_VALMAP_ADD_MASK(p46, 0x50, 4, 6)
DEFINE_VALMAP_ADD_MASK(p47, 0x90, 4, 6)
DEFINE_VALMAP_ADD_MASK(p56, 0x60, 4, 6)
DEFINE_VALMAP_ADD_MASK(p57, 0xA0, 4, 6)
DEFINE_VALMAP_ADD_MASK(b0, 0x01, 2, 7)
DEFINE_VALMAP_ADD_MASK(b1, 0x02, 2, 7)
DEFINE_VALMAP_ADD_MASK(b2, 0x04, 2, 7)
DEFINE_VALMAP_ADD_MASK(b3, 0x08, 2, 7)
DEFINE_VALMAP_ADD_MASK(b4, 0x10, 2, 7)

/* ---- AUTOPERIOD_ADD: ADD-sibling of AUTOPERIOD_XOR -- strides 65..256. */
static void ap_autoperiodadd(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] = (u8)(d[i] + amp);
}
static void inv_autoperiodadd(u8 *d, int n, int s, int p, u32 amp) {
    for (int i = p; i < n; i += s) d[i] = (u8)(d[i] - amp);
}
static double search_autoperiodadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = AUTOPERIOD_MIN, bp = 0; u32 ba = 1;
    for (int s = AUTOPERIOD_MIN; s <= AUTOPERIOD_MAX; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[(u + a) & 0xFF] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- RUNLEN2_XOR: same mechanism as RUNLEN_XOR, but K=2 (run of at
 * least 2 consecutive equal ORIGINAL bytes) instead of K=1 -- a
 * different, stricter run-length threshold. */
static void ap_runlen2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 prev = d[0]; int run = 0;
    for (int i = 1; i < n; i++) {
        u8 orig = d[i];
        int cond = (run >= 2);
        d[i] = (u8)(orig ^ (cond ? ahi : alo));
        run = (orig == prev) ? run + 1 : 0;
        prev = orig;
    }
}
static void inv_runlen2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    u8 prev = d[0]; int run = 0;
    for (int i = 1; i < n; i++) {
        int cond = (run >= 2);
        u8 orig = (u8)(d[i] ^ (cond ? ahi : alo));
        d[i] = orig;
        run = (orig == prev) ? run + 1 : 0;
        prev = orig;
    }
}
static double search_runlen2(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int alo = 0; alo < 256; alo++) {
        for (int ahi = 0; ahi < 256; ahi++) {
            u8 prev = d[0]; int run = 0;
            scr[0] = d[0];
            for (int i = 1; i < n; i++) {
                u8 orig = d[i];
                int cond = (run >= 2);
                scr[i] = (u8)(orig ^ ((cond ? ahi : alo)));
                run = (orig == prev) ? run + 1 : 0;
                prev = orig;
            }
            double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
            if (net > best) { best = net; ba = (u32)alo | ((u32)ahi << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- BITREV_IDX_ADD: ADD-sibling of BITREV_IDX -- bit-reversed index
 * subset selection, additive constant instead of XOR. */
static void ap_bitrevidxadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 c = (u8)amp;
    for (int i = 0; i < n; i++) if (bitrev12_tab[i] % s == (u16)p) d[i] = (u8)(d[i] + c);
}
static void inv_bitrevidxadd(u8 *d, int n, int s, int p, u32 amp) {
    u8 c = (u8)amp;
    for (int i = 0; i < n; i++) if (bitrev12_tab[i] % s == (u16)p) d[i] = (u8)(d[i] - c);
}
static double search_bitrevidxadd(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 1;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = 0; i < n; i++) if (bitrev12_tab[i] % s == (u16)p) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int a = 1; a < 256; a++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int u = 0; u < 256; u++) rf[(u + a) & 0xFF] += hit[u];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- BITREV_IDX_XOR_<range> / BITREV_IDX_ADD_<range> family: BITREV_IDX
 * reverses ALL 12 index bits before the mod-s subset selection;
 * BITREV_IDX_ADD won a real-data layer, but the mechanism only had that
 * one full-reversal permutation to offer. This generalizes it: reverse
 * only a CONTIGUOUS sub-range of the 12 bits (start..start+len-1),
 * leaving the rest of the index untouched -- a different, milder
 * "locality scramble" per (start,len) choice (66 possible ranges with
 * len>=2; len=1 would be a no-op). Still a pure permutation of the 4096
 * positions, so still needs stride/phase subsetting to matter, same as
 * the original. 32 ranges for XOR, a different 32 for ADD (64 total). */
#define DEFINE_BITREV_RANGE_XOR(SUF, START, LEN) \
static inline u16 bitrevrange_##SUF(int i) { \
    int seg = (i >> (START)) & ((1 << (LEN)) - 1); \
    int rev = 0; \
    for (int b = 0; b < (LEN); b++) rev |= ((seg >> b) & 1) << ((LEN) - 1 - b); \
    int hi = i & ~(((1 << (LEN)) - 1) << (START)); \
    return (u16)(hi | (rev << (START))); \
} \
static void ap_bitrevrange_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 c = (u8)amp; \
    for (int i = 0; i < n; i++) if (bitrevrange_##SUF(i) % s == (u16)p) d[i] ^= c; \
} \
static double search_bitrevrange_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 1; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = 0; i < n; i++) if (bitrevrange_##SUF(i) % s == (u16)p) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int a = 1; a < 256; a++) { \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int u = 0; u < 256; u++) rf[u ^ a] += hit[u]; \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
#define DEFINE_BITREV_RANGE_ADD(SUF, START, LEN) \
static inline u16 bitrevrangeadd_idx_##SUF(int i) { \
    int seg = (i >> (START)) & ((1 << (LEN)) - 1); \
    int rev = 0; \
    for (int b = 0; b < (LEN); b++) rev |= ((seg >> b) & 1) << ((LEN) - 1 - b); \
    int hi = i & ~(((1 << (LEN)) - 1) << (START)); \
    return (u16)(hi | (rev << (START))); \
} \
static void ap_bitrevrangeadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 c = (u8)amp; \
    for (int i = 0; i < n; i++) if (bitrevrangeadd_idx_##SUF(i) % s == (u16)p) d[i] = (u8)(d[i] + c); \
} \
static void inv_bitrevrangeadd_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 c = (u8)amp; \
    for (int i = 0; i < n; i++) if (bitrevrangeadd_idx_##SUF(i) % s == (u16)p) d[i] = (u8)(d[i] - c); \
} \
static double search_bitrevrangeadd_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 1; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = 0; i < n; i++) if (bitrevrangeadd_idx_##SUF(i) % s == (u16)p) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int a = 1; a < 256; a++) { \
                int rf[256]; memcpy(rf, base, sizeof rf); \
                for (int u = 0; u < 256; u++) rf[(u + a) & 0xFF] += hit[u]; \
                double net = (S_from_freq(rf) - Sb) - oh; \
                if (net > best) { best = net; bs = s; bp = p; ba = (u32)a; } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_BITREV_RANGE_XOR(s0l2, 0, 2)
DEFINE_BITREV_RANGE_XOR(s1l2, 1, 2)
DEFINE_BITREV_RANGE_XOR(s2l2, 2, 2)
DEFINE_BITREV_RANGE_XOR(s3l2, 3, 2)
DEFINE_BITREV_RANGE_XOR(s4l2, 4, 2)
DEFINE_BITREV_RANGE_XOR(s5l2, 5, 2)
DEFINE_BITREV_RANGE_XOR(s6l2, 6, 2)
DEFINE_BITREV_RANGE_XOR(s7l2, 7, 2)
DEFINE_BITREV_RANGE_XOR(s8l2, 8, 2)
DEFINE_BITREV_RANGE_XOR(s9l2, 9, 2)
DEFINE_BITREV_RANGE_XOR(s10l2, 10, 2)
DEFINE_BITREV_RANGE_XOR(s0l3, 0, 3)
DEFINE_BITREV_RANGE_XOR(s1l3, 1, 3)
DEFINE_BITREV_RANGE_XOR(s2l3, 2, 3)
DEFINE_BITREV_RANGE_XOR(s3l3, 3, 3)
DEFINE_BITREV_RANGE_XOR(s4l3, 4, 3)
DEFINE_BITREV_RANGE_XOR(s5l3, 5, 3)
DEFINE_BITREV_RANGE_XOR(s6l3, 6, 3)
DEFINE_BITREV_RANGE_XOR(s7l3, 7, 3)
DEFINE_BITREV_RANGE_XOR(s8l3, 8, 3)
DEFINE_BITREV_RANGE_XOR(s9l3, 9, 3)
DEFINE_BITREV_RANGE_XOR(s0l4, 0, 4)
DEFINE_BITREV_RANGE_XOR(s1l4, 1, 4)
DEFINE_BITREV_RANGE_XOR(s2l4, 2, 4)
DEFINE_BITREV_RANGE_XOR(s3l4, 3, 4)
DEFINE_BITREV_RANGE_XOR(s4l4, 4, 4)
DEFINE_BITREV_RANGE_XOR(s5l4, 5, 4)
DEFINE_BITREV_RANGE_XOR(s6l4, 6, 4)
DEFINE_BITREV_RANGE_XOR(s7l4, 7, 4)
DEFINE_BITREV_RANGE_XOR(s8l4, 8, 4)
DEFINE_BITREV_RANGE_XOR(s0l5, 0, 5)
DEFINE_BITREV_RANGE_XOR(s1l5, 1, 5)
DEFINE_BITREV_RANGE_ADD(s2l5, 2, 5)
DEFINE_BITREV_RANGE_ADD(s3l5, 3, 5)
DEFINE_BITREV_RANGE_ADD(s4l5, 4, 5)
DEFINE_BITREV_RANGE_ADD(s5l5, 5, 5)
DEFINE_BITREV_RANGE_ADD(s6l5, 6, 5)
DEFINE_BITREV_RANGE_ADD(s7l5, 7, 5)
DEFINE_BITREV_RANGE_ADD(s0l6, 0, 6)
DEFINE_BITREV_RANGE_ADD(s1l6, 1, 6)
DEFINE_BITREV_RANGE_ADD(s2l6, 2, 6)
DEFINE_BITREV_RANGE_ADD(s3l6, 3, 6)
DEFINE_BITREV_RANGE_ADD(s4l6, 4, 6)
DEFINE_BITREV_RANGE_ADD(s5l6, 5, 6)
DEFINE_BITREV_RANGE_ADD(s6l6, 6, 6)
DEFINE_BITREV_RANGE_ADD(s0l7, 0, 7)
DEFINE_BITREV_RANGE_ADD(s1l7, 1, 7)
DEFINE_BITREV_RANGE_ADD(s2l7, 2, 7)
DEFINE_BITREV_RANGE_ADD(s3l7, 3, 7)
DEFINE_BITREV_RANGE_ADD(s4l7, 4, 7)
DEFINE_BITREV_RANGE_ADD(s5l7, 5, 7)
DEFINE_BITREV_RANGE_ADD(s0l8, 0, 8)
DEFINE_BITREV_RANGE_ADD(s1l8, 1, 8)
DEFINE_BITREV_RANGE_ADD(s2l8, 2, 8)
DEFINE_BITREV_RANGE_ADD(s3l8, 3, 8)
DEFINE_BITREV_RANGE_ADD(s4l8, 4, 8)
DEFINE_BITREV_RANGE_ADD(s0l9, 0, 9)
DEFINE_BITREV_RANGE_ADD(s1l9, 1, 9)
DEFINE_BITREV_RANGE_ADD(s2l9, 2, 9)
DEFINE_BITREV_RANGE_ADD(s3l9, 3, 9)
DEFINE_BITREV_RANGE_ADD(s0l10, 0, 10)
DEFINE_BITREV_RANGE_ADD(s1l10, 1, 10)
DEFINE_BITREV_RANGE_ADD(s2l10, 2, 10)
DEFINE_BITREV_RANGE_ADD(s0l11, 0, 11)

/* ---- DELTA_LONGLAG: same mechanism as DELTA, but lag 65..256 instead of
 * 1..64 -- catches longer-range predictive structure DELTA's small-lag
 * search can't reach. Same descending-apply/ascending-invert pattern. */
static void ap_deltalonglag(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag = (int)amp;
    for (int i = n - 1; i >= lag; i--) d[i] = (u8)(d[i] - d[i - lag]);
}
static void inv_deltalonglag(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag = (int)amp;
    for (int i = lag; i < n; i++) d[i] = (u8)(d[i] + d[i - lag]);
}
static double search_deltalonglag(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 blag = 65;
    static u8 scr[BLOCK];
    for (int lag = 65; lag <= 256; lag++) {
        memcpy(scr, d, (size_t)n);
        for (int i = n - 1; i >= lag; i--) scr[i] = (u8)(scr[i] - scr[i - lag]);
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; blag = (u32)lag; }
    }
    out->stride = 0; out->phase = 0; out->amp = blag;
    return best;
}

/* ---- DELTA_XLONGLAG: same mechanism as DELTA_LONGLAG, but lag 257..768
 * instead of 65..256 -- reaches for even longer-range periodic structure
 * (e.g. a ~512-byte record stride) than DELTA_LONGLAG's range covers.
 * amp needs 9 bits (257..768 spans 512 values) to index the lag. */
static void ap_deltaxlonglag(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag = 257 + (int)(amp & 0x1FF);
    for (int i = n - 1; i >= lag; i--) d[i] = (u8)(d[i] - d[i - lag]);
}
static void inv_deltaxlonglag(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag = 257 + (int)(amp & 0x1FF);
    for (int i = lag; i < n; i++) d[i] = (u8)(d[i] + d[i - lag]);
}
static double search_deltaxlonglag(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int off = 0; off <= 511; off++) {
        int lag = 257 + off;
        if (lag >= n) break;
        memcpy(scr, d, (size_t)n);
        for (int i = n - 1; i >= lag; i--) scr[i] = (u8)(scr[i] - scr[i - lag]);
        double net = (S_of(scr, n) - Sb) - oh_flat(9.0);
        if (net > best) { best = net; ba = (u32)off; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- WORD_XOR32: bitwise XOR on 4-byte groups by a 32-bit constant --
 * decomposes exactly into 4 independent byte-level phase-XORs (no
 * carry), so like WORD_XOR16 this is predicted to be dominated by
 * running 4 independent XOR_PHASE(stride=4) picks; included for
 * empirical confirmation at this granularity too. Self-inverse. */
static void ap_wordxor32(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 c[4]; for (int k = 0; k < 4; k++) c[k] = (u8)((amp >> (k * 8)) & 0xFF);
    for (int i = 0; i + 3 < n; i += 4) for (int k = 0; k < 4; k++) d[i + k] ^= c[k];
}
static double search_wordxor32(const u8 *d, int n, double Sb, Instr *out) {
    u32 amp = 0;
    for (int k = 0; k < 4; k++) {
        int tot[256] = {0};
        for (int i = k; i < n; i += 4) tot[d[i]]++;
        double bestc = -1e18; int bc = 0;
        for (int c = 1; c < 256; c++) {
            int rf[256] = {0};
            for (int u = 0; u < 256; u++) rf[u ^ c] += tot[u];
            double S = 0.0; for (int v = 0; v < 256; v++) S += hlog[rf[v]];
            if (S > bestc) { bestc = S; bc = c; }
        }
        amp |= (u32)bc << (k * 8);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_wordxor32(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(32.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

/* ---- MULTINEIGH_ADD: ADD-sibling of MULTINEIGH_XOR -- d[i] += d[i-1] +
 * d[i-2] (both ORIGINAL, local-var tracked). Needs a genuine separate
 * inverse (same reasoning as MULTINEIGH_XOR: re-running apply would
 * capture transformed values into the tracker from i=3 onward). */
static void ap_multineighadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        if (i >= 2) d[i] = (u8)(orig_i + p1 + p2);
        p2 = p1; p1 = orig_i;
    }
}
static void inv_multineighadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i;
        if (i >= 2) { orig_i = (u8)(d[i] - p1 - p2); d[i] = orig_i; }
        else orig_i = d[i];
        p2 = p1; p1 = orig_i;
    }
}
static double search_multineighadd(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_multineighadd(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- ADAPT_LMS_FAST: same sign-sign LMS mechanism as ADAPT_LMS, but a
 * faster-reacting prediction shift (>>3 instead of >>4) -- a genuinely
 * different predictor dynamic (larger effective weight range per step),
 * not just a parameter tweak within the existing search space. */
static void ap_lmsfast(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w = (int)amp - 8;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)((w * (int)prev) >> 3);
        u8 residual = (u8)(orig_i - pred);
        d[i] = residual;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w += sgn; if (w > 7) w = 7; if (w < -8) w = -8;
        prev = orig_i;
    }
}
static void inv_lmsfast(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w = (int)amp - 8;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 residual = d[i];
        u8 pred = (u8)((w * (int)prev) >> 3);
        u8 orig_i = (u8)(residual + pred);
        d[i] = orig_i;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w += sgn; if (w > 7) w = 7; if (w < -8) w = -8;
        prev = orig_i;
    }
}
static double search_lmsfast(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bw0 = 8;
    static u8 scr[BLOCK];
    for (int w0 = 0; w0 < 16; w0++) {
        memcpy(scr, d, (size_t)n);
        ap_lmsfast(scr, n, 0, 0, (u32)w0);
        double net = (S_of(scr, n) - Sb) - oh_flat(4.0);
        if (net > best) { best = net; bw0 = (u32)w0; }
    }
    out->stride = 0; out->phase = 0; out->amp = bw0;
    return best;
}

/* ---- CTXTABLE_ADD: ADD-sibling of CTXTABLE_XOR -- context table predicts
 * via ADD instead of XOR combine. */
static void ap_ctxtableadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 table[256]; memset(table, 0, sizeof table);
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig = d[i];
        u8 pred = table[prev];
        d[i] = (u8)(orig - pred);
        table[prev] = orig;
        prev = orig;
    }
}
static void inv_ctxtableadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 table[256]; memset(table, 0, sizeof table);
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 pred = table[prev];
        u8 orig = (u8)(d[i] + pred);
        d[i] = orig;
        table[prev] = orig;
        prev = orig;
    }
}
static double search_ctxtableadd(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_ctxtableadd(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- HASH_ADD: ADD-sibling of HASH_XOR -- stateless per-position hash
 * keystream, additive combine instead of XOR. */
static void ap_hashadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + hash_byte((u32)i, amp));
}
static void inv_hashadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - hash_byte((u32)i, amp));
}
static double search_hashadd(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + hash_byte((u32)i, seed))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- HASH_XOR3 / HASH_ADD2 / HASH_ADD3: same stateless per-position
 * hash mechanism as HASH_XOR/HASH_XOR2/HASH_ADD, but pairing a THIRD
 * distinct finalizer (Chris Wellons' "lowbias32" mixer -- different
 * multiply constants and shift amounts than either the Murmur3-style
 * hash_byte or the SplitMix32-style hash_byte2) with both combine ops. */
static inline u8 hash_byte3(u32 i, u32 seed) {
    u32 x = i + seed * 0x2545F491u;
    x ^= x >> 16; x *= 0x7FEB352Du;
    x ^= x >> 15; x *= 0x846CA68Bu;
    x ^= x >> 16;
    return (u8)x;
}
static void ap_hashxor3(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] ^= hash_byte3((u32)i, amp);
}
static double search_hashxor3(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ hash_byte3((u32)i, seed))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}
static void ap_hashadd2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + hash_byte2((u32)i, amp));
}
static void inv_hashadd2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - hash_byte2((u32)i, amp));
}
static double search_hashadd2(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + hash_byte2((u32)i, seed))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}
static void ap_hashadd3(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + hash_byte3((u32)i, amp));
}
static void inv_hashadd3(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - hash_byte3((u32)i, amp));
}
static double search_hashadd3(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + hash_byte3((u32)i, seed))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- HASH_XOR4..HASH_XOR15 / HASH_ADD4..HASH_ADD15: same stateless
 * per-position keystream mechanism as HASH_XOR/HASH_ADD, crossed with 12
 * more distinct deterministic mixing functions -- varied constants,
 * shift amounts, rotate-vs-shift, and single/double/triple-round
 * structure (not just relabeled copies of hash_byte/2/3). Combine op
 * (XOR/ADD) is generated via a shared macro since that part is
 * identical regardless of which finalizer is plugged in. */
#define DEFINE_HASH_XOR_FOR(SUF, HFUNC) \
static void ap_hashxor##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    for (int i = 0; i < n; i++) d[i] ^= HFUNC((u32)i, amp); \
} \
static double search_hashxor##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double oh = oh_flat(16.0); \
    double best = -1e18; u32 bseed = 1; \
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) { \
        int f[256] = {0}; \
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ HFUNC((u32)i, seed))]++; \
        double net = (S_from_freq(f) - Sb) - oh; \
        if (net > best) { best = net; bseed = seed; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = bseed; \
    return best; \
}
#define DEFINE_HASH_ADD_FOR(SUF, HFUNC) \
static void ap_hashadd##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + HFUNC((u32)i, amp)); \
} \
static void inv_hashadd##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - HFUNC((u32)i, amp)); \
} \
static double search_hashadd##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double oh = oh_flat(16.0); \
    double best = -1e18; u32 bseed = 1; \
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) { \
        int f[256] = {0}; \
        for (int i = 0; i < n; i++) f[(u8)(d[i] + HFUNC((u32)i, seed))]++; \
        double net = (S_from_freq(f) - Sb) - oh; \
        if (net > best) { best = net; bseed = seed; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = bseed; \
    return best; \
}

static inline u8 hash_byte4(u32 i, u32 seed) {
    u32 x = i ^ seed;
    x ^= x >> 17; x *= 0xED5AD4BBu;
    x ^= x >> 13; x *= 0xAC4C1B51u;
    x ^= x >> 15;
    return (u8)x;
}
static inline u8 hash_byte5(u32 i, u32 seed) {
    u32 x = i + seed;
    x *= 0x2C1B3C6Du; x ^= x >> 8;
    x *= 0x297A2D39u; x ^= x >> 13;
    x *= 0x1B03738Bu; x ^= x >> 16;
    return (u8)x;
}
static inline u8 hash_byte6(u32 i, u32 seed) {
    u32 x = i * (seed | 1u);
    x ^= x >> 19;
    x *= 0x9E3779B9u;
    x ^= x >> 15;
    return (u8)x;
}
static inline u8 hash_byte7(u32 i, u32 seed) {
    u32 x = i ^ (seed * 0xC2B2AE35u);
    x = (x << 13) | (x >> 19);
    x *= 0x85EBCA6Du;
    x ^= (x << 7) | (x >> 25);
    return (u8)x;
}
static inline u8 hash_byte8(u32 i, u32 seed) {
    u32 x = i + seed;
    x += (x << 15); x ^= (x >> 10);
    x += (x << 3); x ^= (x >> 6);
    x += (x << 21);
    return (u8)(x ^ (x >> 8));
}
static inline u8 hash_byte9(u32 i, u32 seed) {
    u32 x = (i ^ seed) * 0xCC9E2D51u;
    x = (x << 15) | (x >> 17);
    x *= 0x1B873593u;
    x ^= x >> 13;
    return (u8)x;
}
static inline u8 hash_byte10(u32 i, u32 seed) {
    u32 x = (i + seed) * 0x61C88647u;
    x ^= x >> 15;
    return (u8)(x >> 4);
}
static inline u8 hash_byte11(u32 i, u32 seed) {
    u32 x = i ^ (seed << 1) ^ (seed >> 1);
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    return (u8)x;
}
static inline u8 hash_byte12(u32 i, u32 seed) {
    u32 x = i + seed;
    x ^= 0xDEADBEEFu;
    x = (x << 7) | (x >> 25); x += 0x9E3779B9u;
    x = (x << 11) | (x >> 21); x ^= 0x85EBCA6Bu;
    x = (x << 19) | (x >> 13);
    return (u8)x;
}
static inline u8 hash_byte13(u32 i, u32 seed) {
    u32 x = i * 2654435761u + seed * 40503u;
    x ^= x >> 16;
    x *= 0x045D9F3Bu;
    x ^= x >> 16;
    return (u8)x;
}
static inline u8 hash_byte14(u32 i, u32 seed) {
    u32 x = i ^ seed;
    x = ((x & 0xAAAAAAAAu) >> 1) | ((x & 0x55555555u) << 1);
    x *= 0x2545F491u;
    x ^= x >> 13;
    return (u8)x;
}
static inline u8 hash_byte15(u32 i, u32 seed) {
    u32 x = i;
    x += seed; x ^= seed >> 16;
    x *= 0x27220A95u;
    x ^= x >> 22;
    x *= 0x165667B1u;
    x ^= x >> 15;
    return (u8)x;
}
DEFINE_HASH_XOR_FOR(4, hash_byte4)   DEFINE_HASH_ADD_FOR(4, hash_byte4)
DEFINE_HASH_XOR_FOR(5, hash_byte5)   DEFINE_HASH_ADD_FOR(5, hash_byte5)
DEFINE_HASH_XOR_FOR(6, hash_byte6)   DEFINE_HASH_ADD_FOR(6, hash_byte6)
DEFINE_HASH_XOR_FOR(7, hash_byte7)   DEFINE_HASH_ADD_FOR(7, hash_byte7)
DEFINE_HASH_XOR_FOR(8, hash_byte8)   DEFINE_HASH_ADD_FOR(8, hash_byte8)
DEFINE_HASH_XOR_FOR(9, hash_byte9)   DEFINE_HASH_ADD_FOR(9, hash_byte9)
DEFINE_HASH_XOR_FOR(10, hash_byte10) DEFINE_HASH_ADD_FOR(10, hash_byte10)
DEFINE_HASH_XOR_FOR(11, hash_byte11) DEFINE_HASH_ADD_FOR(11, hash_byte11)
DEFINE_HASH_XOR_FOR(12, hash_byte12) DEFINE_HASH_ADD_FOR(12, hash_byte12)
DEFINE_HASH_XOR_FOR(13, hash_byte13) DEFINE_HASH_ADD_FOR(13, hash_byte13)
DEFINE_HASH_XOR_FOR(14, hash_byte14) DEFINE_HASH_ADD_FOR(14, hash_byte14)
DEFINE_HASH_XOR_FOR(15, hash_byte15) DEFINE_HASH_ADD_FOR(15, hash_byte15)

/* ---- HASH_ADD8_<S1>_<S6> family: same add-shift-xor-add-shift-xor-add
 * finalizer SHAPE as hash_byte8 (used by HASH_ADD8), but sweeping the
 * two outer shift amounts (S1, the first left-shift; S6, the final
 * right-shift used in the output fold) across 8x4=32 combinations while
 * keeping the middle shifts fixed at hash_byte8's original values
 * (10,3,6,21) -- even a single shift-amount change gives a genuinely
 * different avalanche/mixing behavior, not a relabeling. ADD combine
 * only (mirrors the specific triggered instruction). */
#define DEFINE_HASH_ADD8_VARIANT(SUF, S1, S6) \
static inline u8 hash8v_##SUF(u32 i, u32 seed) { \
    u32 x = i + seed; \
    x += (x << (S1)); x ^= (x >> 10); \
    x += (x << 3); x ^= (x >> 6); \
    x += (x << 21); \
    return (u8)(x ^ (x >> (S6))); \
}
DEFINE_HASH_ADD8_VARIANT(s11e4, 11, 4)
DEFINE_HASH_ADD8_VARIANT(s11e6, 11, 6)
DEFINE_HASH_ADD8_VARIANT(s11e8, 11, 8)
DEFINE_HASH_ADD8_VARIANT(s11e10, 11, 10)
DEFINE_HASH_ADD8_VARIANT(s13e4, 13, 4)
DEFINE_HASH_ADD8_VARIANT(s13e6, 13, 6)
DEFINE_HASH_ADD8_VARIANT(s13e8, 13, 8)
DEFINE_HASH_ADD8_VARIANT(s13e10, 13, 10)
DEFINE_HASH_ADD8_VARIANT(s15e4, 15, 4)
DEFINE_HASH_ADD8_VARIANT(s15e6, 15, 6)
DEFINE_HASH_ADD8_VARIANT(s15e8, 15, 8)
DEFINE_HASH_ADD8_VARIANT(s15e10, 15, 10)
DEFINE_HASH_ADD8_VARIANT(s17e4, 17, 4)
DEFINE_HASH_ADD8_VARIANT(s17e6, 17, 6)
DEFINE_HASH_ADD8_VARIANT(s17e8, 17, 8)
DEFINE_HASH_ADD8_VARIANT(s17e10, 17, 10)
DEFINE_HASH_ADD8_VARIANT(s19e4, 19, 4)
DEFINE_HASH_ADD8_VARIANT(s19e6, 19, 6)
DEFINE_HASH_ADD8_VARIANT(s19e8, 19, 8)
DEFINE_HASH_ADD8_VARIANT(s19e10, 19, 10)
DEFINE_HASH_ADD8_VARIANT(s21e4, 21, 4)
DEFINE_HASH_ADD8_VARIANT(s21e6, 21, 6)
DEFINE_HASH_ADD8_VARIANT(s21e8, 21, 8)
DEFINE_HASH_ADD8_VARIANT(s21e10, 21, 10)
DEFINE_HASH_ADD8_VARIANT(s23e4, 23, 4)
DEFINE_HASH_ADD8_VARIANT(s23e6, 23, 6)
DEFINE_HASH_ADD8_VARIANT(s23e8, 23, 8)
DEFINE_HASH_ADD8_VARIANT(s23e10, 23, 10)
DEFINE_HASH_ADD8_VARIANT(s25e4, 25, 4)
DEFINE_HASH_ADD8_VARIANT(s25e6, 25, 6)
DEFINE_HASH_ADD8_VARIANT(s25e8, 25, 8)
DEFINE_HASH_ADD8_VARIANT(s25e10, 25, 10)
DEFINE_HASH_ADD_FOR(8v_s11e4, hash8v_s11e4)
DEFINE_HASH_ADD_FOR(8v_s11e6, hash8v_s11e6)
DEFINE_HASH_ADD_FOR(8v_s11e8, hash8v_s11e8)
DEFINE_HASH_ADD_FOR(8v_s11e10, hash8v_s11e10)
DEFINE_HASH_ADD_FOR(8v_s13e4, hash8v_s13e4)
DEFINE_HASH_ADD_FOR(8v_s13e6, hash8v_s13e6)
DEFINE_HASH_ADD_FOR(8v_s13e8, hash8v_s13e8)
DEFINE_HASH_ADD_FOR(8v_s13e10, hash8v_s13e10)
DEFINE_HASH_ADD_FOR(8v_s15e4, hash8v_s15e4)
DEFINE_HASH_ADD_FOR(8v_s15e6, hash8v_s15e6)
DEFINE_HASH_ADD_FOR(8v_s15e8, hash8v_s15e8)
DEFINE_HASH_ADD_FOR(8v_s15e10, hash8v_s15e10)
DEFINE_HASH_ADD_FOR(8v_s17e4, hash8v_s17e4)
DEFINE_HASH_ADD_FOR(8v_s17e6, hash8v_s17e6)
DEFINE_HASH_ADD_FOR(8v_s17e8, hash8v_s17e8)
DEFINE_HASH_ADD_FOR(8v_s17e10, hash8v_s17e10)
DEFINE_HASH_ADD_FOR(8v_s19e4, hash8v_s19e4)
DEFINE_HASH_ADD_FOR(8v_s19e6, hash8v_s19e6)
DEFINE_HASH_ADD_FOR(8v_s19e8, hash8v_s19e8)
DEFINE_HASH_ADD_FOR(8v_s19e10, hash8v_s19e10)
DEFINE_HASH_ADD_FOR(8v_s21e4, hash8v_s21e4)
DEFINE_HASH_ADD_FOR(8v_s21e6, hash8v_s21e6)
DEFINE_HASH_ADD_FOR(8v_s21e8, hash8v_s21e8)
DEFINE_HASH_ADD_FOR(8v_s21e10, hash8v_s21e10)
DEFINE_HASH_ADD_FOR(8v_s23e4, hash8v_s23e4)
DEFINE_HASH_ADD_FOR(8v_s23e6, hash8v_s23e6)
DEFINE_HASH_ADD_FOR(8v_s23e8, hash8v_s23e8)
DEFINE_HASH_ADD_FOR(8v_s23e10, hash8v_s23e10)
DEFINE_HASH_ADD_FOR(8v_s25e4, hash8v_s25e4)
DEFINE_HASH_ADD_FOR(8v_s25e6, hash8v_s25e6)
DEFINE_HASH_ADD_FOR(8v_s25e8, hash8v_s25e8)
DEFINE_HASH_ADD_FOR(8v_s25e10, hash8v_s25e10)

/* ---- HASH_ADD8_M<S2>_<S4> family: same hash_byte8 finalizer shape as
 * HASH_ADD8/HASH_ADD8_S<n>E<n>, but this time sweeping the two MIDDLE
 * shift amounts (S2, S4) instead of the outer ones (S1, S6) -- one of
 * the earlier S1/S6 sweep variants (HASH_ADD8_S21E10) turned out to be
 * the single biggest winner across the whole layered pass on real data
 * (+70.3 bits, beating the original HASH_ADD8), so this explores a
 * different 2D slice of the same 6-shift space rather than assuming
 * S1/S6 were the only axes that mattered. S1=15,S3=3,S5=21,S6=8 held
 * at hash_byte8's original values. 8x4=32 combinations. */
#define DEFINE_HASH_ADD8_MID_VARIANT(SUF, S2, S4) \
static inline u8 hash8m_##SUF(u32 i, u32 seed) { \
    u32 x = i + seed; \
    x += (x << 15); x ^= (x >> (S2)); \
    x += (x << 3); x ^= (x >> (S4)); \
    x += (x << 21); \
    return (u8)(x ^ (x >> 8)); \
}
DEFINE_HASH_ADD8_MID_VARIANT(s6e2, 6, 2)
DEFINE_HASH_ADD8_MID_VARIANT(s6e4, 6, 4)
DEFINE_HASH_ADD8_MID_VARIANT(s6e6, 6, 6)
DEFINE_HASH_ADD8_MID_VARIANT(s6e8, 6, 8)
DEFINE_HASH_ADD8_MID_VARIANT(s8e2, 8, 2)
DEFINE_HASH_ADD8_MID_VARIANT(s8e4, 8, 4)
DEFINE_HASH_ADD8_MID_VARIANT(s8e6, 8, 6)
DEFINE_HASH_ADD8_MID_VARIANT(s8e8, 8, 8)
DEFINE_HASH_ADD8_MID_VARIANT(s10e2, 10, 2)
DEFINE_HASH_ADD8_MID_VARIANT(s10e4, 10, 4)
DEFINE_HASH_ADD8_MID_VARIANT(s10e6, 10, 6)
DEFINE_HASH_ADD8_MID_VARIANT(s10e8, 10, 8)
DEFINE_HASH_ADD8_MID_VARIANT(s12e2, 12, 2)
DEFINE_HASH_ADD8_MID_VARIANT(s12e4, 12, 4)
DEFINE_HASH_ADD8_MID_VARIANT(s12e6, 12, 6)
DEFINE_HASH_ADD8_MID_VARIANT(s12e8, 12, 8)
DEFINE_HASH_ADD8_MID_VARIANT(s14e2, 14, 2)
DEFINE_HASH_ADD8_MID_VARIANT(s14e4, 14, 4)
DEFINE_HASH_ADD8_MID_VARIANT(s14e6, 14, 6)
DEFINE_HASH_ADD8_MID_VARIANT(s14e8, 14, 8)
DEFINE_HASH_ADD8_MID_VARIANT(s16e2, 16, 2)
DEFINE_HASH_ADD8_MID_VARIANT(s16e4, 16, 4)
DEFINE_HASH_ADD8_MID_VARIANT(s16e6, 16, 6)
DEFINE_HASH_ADD8_MID_VARIANT(s16e8, 16, 8)
DEFINE_HASH_ADD8_MID_VARIANT(s18e2, 18, 2)
DEFINE_HASH_ADD8_MID_VARIANT(s18e4, 18, 4)
DEFINE_HASH_ADD8_MID_VARIANT(s18e6, 18, 6)
DEFINE_HASH_ADD8_MID_VARIANT(s18e8, 18, 8)
DEFINE_HASH_ADD8_MID_VARIANT(s20e2, 20, 2)
DEFINE_HASH_ADD8_MID_VARIANT(s20e4, 20, 4)
DEFINE_HASH_ADD8_MID_VARIANT(s20e6, 20, 6)
DEFINE_HASH_ADD8_MID_VARIANT(s20e8, 20, 8)
DEFINE_HASH_ADD_FOR(8m_s6e2, hash8m_s6e2)
DEFINE_HASH_ADD_FOR(8m_s6e4, hash8m_s6e4)
DEFINE_HASH_ADD_FOR(8m_s6e6, hash8m_s6e6)
DEFINE_HASH_ADD_FOR(8m_s6e8, hash8m_s6e8)
DEFINE_HASH_ADD_FOR(8m_s8e2, hash8m_s8e2)
DEFINE_HASH_ADD_FOR(8m_s8e4, hash8m_s8e4)
DEFINE_HASH_ADD_FOR(8m_s8e6, hash8m_s8e6)
DEFINE_HASH_ADD_FOR(8m_s8e8, hash8m_s8e8)
DEFINE_HASH_ADD_FOR(8m_s10e2, hash8m_s10e2)
DEFINE_HASH_ADD_FOR(8m_s10e4, hash8m_s10e4)
DEFINE_HASH_ADD_FOR(8m_s10e6, hash8m_s10e6)
DEFINE_HASH_ADD_FOR(8m_s10e8, hash8m_s10e8)
DEFINE_HASH_ADD_FOR(8m_s12e2, hash8m_s12e2)
DEFINE_HASH_ADD_FOR(8m_s12e4, hash8m_s12e4)
DEFINE_HASH_ADD_FOR(8m_s12e6, hash8m_s12e6)
DEFINE_HASH_ADD_FOR(8m_s12e8, hash8m_s12e8)
DEFINE_HASH_ADD_FOR(8m_s14e2, hash8m_s14e2)
DEFINE_HASH_ADD_FOR(8m_s14e4, hash8m_s14e4)
DEFINE_HASH_ADD_FOR(8m_s14e6, hash8m_s14e6)
DEFINE_HASH_ADD_FOR(8m_s14e8, hash8m_s14e8)
DEFINE_HASH_ADD_FOR(8m_s16e2, hash8m_s16e2)
DEFINE_HASH_ADD_FOR(8m_s16e4, hash8m_s16e4)
DEFINE_HASH_ADD_FOR(8m_s16e6, hash8m_s16e6)
DEFINE_HASH_ADD_FOR(8m_s16e8, hash8m_s16e8)
DEFINE_HASH_ADD_FOR(8m_s18e2, hash8m_s18e2)
DEFINE_HASH_ADD_FOR(8m_s18e4, hash8m_s18e4)
DEFINE_HASH_ADD_FOR(8m_s18e6, hash8m_s18e6)
DEFINE_HASH_ADD_FOR(8m_s18e8, hash8m_s18e8)
DEFINE_HASH_ADD_FOR(8m_s20e2, hash8m_s20e2)
DEFINE_HASH_ADD_FOR(8m_s20e4, hash8m_s20e4)
DEFINE_HASH_ADD_FOR(8m_s20e6, hash8m_s20e6)
DEFINE_HASH_ADD_FOR(8m_s20e8, hash8m_s20e8)

/* ---- CRC32_XOR / CRC32_ADD: per-position keystream via a genuine
 * table-driven CRC-32 (reflected, polynomial 0xEDB88320) computed over
 * the 4 bytes of the position index i, with `seed` as the CRC's initial
 * register value -- a fundamentally different mechanism than the
 * multiply-xor-shift hash family above (polynomial division in GF(2)[x]
 * rather than integer multiplicative mixing). */
static u32 crc32_tab[256];
static void init_crc32_tab(void) {
    for (u32 v = 0; v < 256; v++) {
        u32 c = v;
        for (int k = 0; k < 8; k++) c = (c & 1) ? (0xEDB88320u ^ (c >> 1)) : (c >> 1);
        crc32_tab[v] = c;
    }
}
static inline u8 crc32_byte(u32 i, u32 seed) {
    u32 crc = seed;
    for (int k = 0; k < 4; k++) {
        u8 b = (u8)(i >> (k * 8));
        crc = crc32_tab[(crc ^ b) & 0xFF] ^ (crc >> 8);
    }
    return (u8)crc;
}
DEFINE_HASH_XOR_FOR(_crc32, crc32_byte) DEFINE_HASH_ADD_FOR(_crc32, crc32_byte)

/* ---- LCG_XOR / LCG_ADD: per-position keystream via a linear
 * congruential generator evaluated in closed form at position i (not
 * chained/carried state like the PRNG_ADD/XOR family) -- x = i*A + seed*C,
 * classic glibc-style LCG constants, high bits taken as the output byte
 * (LCG low bits are known to be low-quality, so this deliberately reads
 * from the top). */
static inline u8 lcg_byte(u32 i, u32 seed) {
    u32 x = i * 1103515245u + seed * 12345u + 12345u;
    return (u8)(x >> 16);
}
DEFINE_HASH_XOR_FOR(_lcg, lcg_byte) DEFINE_HASH_ADD_FOR(_lcg, lcg_byte)

/* ---- WEYL_XOR / WEYL_ADD: per-position keystream via a raw Weyl/
 * low-discrepancy additive sequence (i * odd_constant, no avalanche
 * mixing at all) -- deliberately the simplest possible construction in
 * this family, testing whether a pure multiplicative quasi-random
 * sequence (no xor-shift rounds) correlates with the data on its own. */
static inline u8 weyl_byte(u32 i, u32 seed) {
    u32 k = seed | 1u;
    u32 x = i * k;
    return (u8)(x >> 24);
}
DEFINE_HASH_XOR_FOR(_weyl, weyl_byte) DEFINE_HASH_ADD_FOR(_weyl, weyl_byte)

/* ---- MSQUARE_XOR / MSQUARE_ADD: per-position keystream via von
 * Neumann's middle-square method -- combine i and seed, SQUARE the
 * result (32-bit overflow keeps the low 32 bits of the true square),
 * and read the middle bits. A squaring-based mechanism, structurally
 * unlike every multiply-xor-shift or polynomial-division finalizer
 * above. */
static inline u8 msquare_byte(u32 i, u32 seed) {
    u32 v = i ^ (seed * 0x9E3779B9u);
    u32 sq = v * v;
    return (u8)(sq >> 12);
}
DEFINE_HASH_XOR_FOR(_msquare, msquare_byte) DEFINE_HASH_ADD_FOR(_msquare, msquare_byte)

/* ---- FIXED_KS_ADD: fixed universal constant keystream, ADD instead of
 * XOR combine -- reuses FIXED_KS_XOR's table (still zero seed cost),
 * different combine operator. */
static void ap_fixedksadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + fixed_keystream[i]);
}
static void inv_fixedksadd(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - fixed_keystream[i]);
}
static double search_fixedksadd(const u8 *d, int n, double Sb, Instr *out) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[(u8)(d[i] + fixed_keystream[i])]++;
    double net = (S_from_freq(f) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- DIAG_MUL: GF(256)-multiply variant of the diagonal-selection
 * family -- 64x64 grid, select a diagonal, GF-multiply instead of XOR/
 * ADD. */
static void ap_diagmul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)n;
    int off = (int)(amp & 0x7F) - 63;
    u8 a = (u8)((amp >> 7) & 0xFF);
    for (int row = 0; row < GRID_N; row++) {
        int col = row - off;
        if (col >= 0 && col < GRID_N) { int idx = row * GRID_N + col; d[idx] = gf_mul(d[idx], a); }
    }
}
static void inv_diagmul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)n;
    int off = (int)(amp & 0x7F) - 63;
    u8 ainv = gf_inv((u8)((amp >> 7) & 0xFF));
    for (int row = 0; row < GRID_N; row++) {
        int col = row - off;
        if (col >= 0 && col < GRID_N) { int idx = row * GRID_N + col; d[idx] = gf_mul(d[idx], ainv); }
    }
}
static double search_diagmul(const u8 *d, int n, double Sb, Instr *out) {
    (void)n;
    double oh = oh_flat(15.0);
    double best = -1e18; u32 ba = 0;
    int total[256]; freq_of(d, BLOCK, total);
    for (int off = -63; off <= 63; off++) {
        int hit[256] = {0};
        for (int row = 0; row < GRID_N; row++) {
            int col = row - off;
            if (col >= 0 && col < GRID_N) hit[d[row * GRID_N + col]]++;
        }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        for (int a = 2; a < 256; a++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            const u8 *col_tab = gf_mul_tab[a];
            for (int u = 0; u < 256; u++) rf[col_tab[u]] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; ba = (u32)(off + 63) | ((u32)a << 7); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- DIAG_MUL_<grid>_<dir> family: same generalization as DIAG_ADD's
 * grid x direction sweep, applied to DIAG_MUL (GF(256)-multiply combine
 * instead of ADD). 11 grids x 2 directions minus the 1 existing
 * (64x64 main) = 21 new instructions. */
#define DEFINE_DIAG_MUL_GRID(SUF, WIDTH, HEIGHT, ANTI) \
static void ap_diagmul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)n; \
    int offmin = (ANTI) ? 0 : -((WIDTH) - 1); \
    int off = (int)(amp & 0xFFF) + offmin; \
    u8 a = (u8)((amp >> 12) & 0xFF); \
    for (int row = 0; row < (HEIGHT); row++) { \
        int col = (ANTI) ? (off - row) : (row - off); \
        if (col >= 0 && col < (WIDTH)) { int idx = row * (WIDTH) + col; d[idx] = gf_mul(d[idx], a); } \
    } \
} \
static void inv_diagmul_##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)n; \
    int offmin = (ANTI) ? 0 : -((WIDTH) - 1); \
    int off = (int)(amp & 0xFFF) + offmin; \
    u8 ainv = gf_inv((u8)((amp >> 12) & 0xFF)); \
    for (int row = 0; row < (HEIGHT); row++) { \
        int col = (ANTI) ? (off - row) : (row - off); \
        if (col >= 0 && col < (WIDTH)) { int idx = row * (WIDTH) + col; d[idx] = gf_mul(d[idx], ainv); } \
    } \
} \
static double search_diagmul_##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    (void)n; \
    double oh = oh_flat(20.0); \
    double best = -1e18; u32 ba = 0; \
    int total[256]; freq_of(d, BLOCK, total); \
    int offmin = (ANTI) ? 0 : -((WIDTH) - 1); \
    int offmax = (ANTI) ? ((WIDTH) + (HEIGHT) - 2) : ((HEIGHT) - 1); \
    for (int off = offmin; off <= offmax; off++) { \
        int hit[256] = {0}; \
        for (int row = 0; row < (HEIGHT); row++) { \
            int col = (ANTI) ? (off - row) : (row - off); \
            if (col >= 0 && col < (WIDTH)) hit[d[row * (WIDTH) + col]]++; \
        } \
        int base[256]; \
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
        for (int a = 2; a < 256; a++) { \
            int rf[256]; memcpy(rf, base, sizeof rf); \
            const u8 *col_tab = gf_mul_tab[a]; \
            for (int u = 0; u < 256; u++) rf[col_tab[u]] += hit[u]; \
            double net = (S_from_freq(rf) - Sb) - oh; \
            if (net > best) { best = net; ba = (u32)(off - offmin) | ((u32)a << 12); } \
        } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_DIAG_MUL_GRID(g2_main,    2, 2048, 0)
DEFINE_DIAG_MUL_GRID(g2_anti,    2, 2048, 1)
DEFINE_DIAG_MUL_GRID(g4_main,    4, 1024, 0)
DEFINE_DIAG_MUL_GRID(g4_anti,    4, 1024, 1)
DEFINE_DIAG_MUL_GRID(g8_main,    8, 512,  0)
DEFINE_DIAG_MUL_GRID(g8_anti,    8, 512,  1)
DEFINE_DIAG_MUL_GRID(g16_main,   16, 256, 0)
DEFINE_DIAG_MUL_GRID(g16_anti,   16, 256, 1)
DEFINE_DIAG_MUL_GRID(g32_main,   32, 128, 0)
DEFINE_DIAG_MUL_GRID(g32_anti,   32, 128, 1)
DEFINE_DIAG_MUL_GRID(g64_anti,   64, 64,  1)
DEFINE_DIAG_MUL_GRID(g128_main,  128, 32, 0)
DEFINE_DIAG_MUL_GRID(g128_anti,  128, 32, 1)
DEFINE_DIAG_MUL_GRID(g256_main,  256, 16, 0)
DEFINE_DIAG_MUL_GRID(g256_anti,  256, 16, 1)
DEFINE_DIAG_MUL_GRID(g512_main,  512, 8,  0)
DEFINE_DIAG_MUL_GRID(g512_anti,  512, 8,  1)
DEFINE_DIAG_MUL_GRID(g1024_main, 1024, 4, 0)
DEFINE_DIAG_MUL_GRID(g1024_anti, 1024, 4, 1)
DEFINE_DIAG_MUL_GRID(g2048_main, 2048, 2, 0)
DEFINE_DIAG_MUL_GRID(g2048_anti, 2048, 2, 1)

/* ---- MIRROR_MUL: GF-multiply variant of MIRROR_XOR -- mirror partner
 * n-1-i (untouched) still drives the correction, GF-multiply combine
 * instead of XOR (needs a nonzero-mapped multiplier since GF_mul by 0
 * annihilates). */
static void ap_mirrormul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int i = 0; i < n / 2; i++) {
        u8 partner = d[n - 1 - i];
        u8 mult = partner ? partner : 1;
        d[i] = gf_mul(d[i], gf_mul(mult, a) ? gf_mul(mult, a) : 1);
    }
}
static void inv_mirrormul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int i = 0; i < n / 2; i++) {
        u8 partner = d[n - 1 - i];
        u8 mult = partner ? partner : 1;
        u8 combined = gf_mul(mult, a) ? gf_mul(mult, a) : 1;
        d[i] = gf_mul(d[i], gf_inv(combined));
    }
}
static double search_mirrormul(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        for (int i = 0; i < n / 2; i++) {
            u8 partner = d[n - 1 - i];
            u8 mult = partner ? partner : 1;
            u8 combined = gf_mul(mult, (u8)a) ? gf_mul(mult, (u8)a) : 1;
            scr[i] = gf_mul(d[i], combined);
        }
        for (int i = n / 2; i < n; i++) scr[i] = d[i];
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- PRNG_GFMUL4: 4-stream xorshift16-driven GF(256) multiplier per
 * position (stream byte 0 mapped to 1 to avoid the zero-annihilator) --
 * a genuinely different combine operator than PRNG_ADD4/XOR4 (field
 * multiply vs arithmetic/bitwise), same seed-stream derivation. */
static void ap_prnggfmul4(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[4]; st[0] = ms; st[1] = xs16_next(&ms); st[2] = xs16_next(&ms); st[3] = xs16_next(&ms);
    for (int i = 0; i < n; i++) {
        u8 raw = xs16_next(&st[i & 3]);
        u8 mult = raw ? raw : 1;
        d[i] = gf_mul(d[i], mult);
    }
}
static void inv_prnggfmul4(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[4]; st[0] = ms; st[1] = xs16_next(&ms); st[2] = xs16_next(&ms); st[3] = xs16_next(&ms);
    for (int i = 0; i < n; i++) {
        u8 raw = xs16_next(&st[i & 3]);
        u8 mult = raw ? raw : 1;
        d[i] = gf_mul(d[i], gf_inv(mult));
    }
}
static double search_prnggfmul4(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[4]; s[0] = ms; s[1] = xs16_next(&ms); s[2] = xs16_next(&ms); s[3] = xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) {
            u8 raw = xs16_next(&s[i & 3]);
            u8 mult = raw ? raw : 1;
            f[gf_mul(d[i], mult)]++;
        }
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- RUNPARITY_MUL: GF-multiply variant of RUNPARITY_XOR -- running
 * XOR-so-far parity of original bytes selects a nonzero-mapped GF
 * multiplier instead of an XOR constant. */
static void ap_runparitymul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    if (!alo) alo = 1; if (!ahi) ahi = 1;
    u8 parity = 0;
    for (int i = 0; i < n; i++) {
        u8 orig = d[i];
        u8 mult = (parity & 1) ? ahi : alo;
        d[i] = gf_mul(orig, mult);
        parity ^= orig;
    }
}
static void inv_runparitymul(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 alo = (u8)(amp & 0xFF), ahi = (u8)((amp >> 8) & 0xFF);
    if (!alo) alo = 1; if (!ahi) ahi = 1;
    u8 aloinv = gf_inv(alo), ahiinv = gf_inv(ahi);
    u8 parity = 0;
    for (int i = 0; i < n; i++) {
        u8 mult = (parity & 1) ? ahiinv : aloinv;
        u8 orig = gf_mul(d[i], mult);
        d[i] = orig;
        parity ^= orig;
    }
}
static double search_runparitymul(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0x0101;
    static u8 scr[BLOCK];
    for (int alo = 1; alo < 256; alo++) {
        for (int ahi = 1; ahi < 256; ahi++) {
            u8 parity = 0;
            for (int i = 0; i < n; i++) {
                u8 orig = d[i];
                u8 mult = (parity & 1) ? (u8)ahi : (u8)alo;
                scr[i] = gf_mul(orig, mult);
                parity ^= orig;
            }
            double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
            if (net > best) { best = net; ba = (u32)alo | ((u32)ahi << 8); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- NIB_BITROT: rotate the lo and hi nibbles' bits INDEPENDENTLY (by
 * k1, k2 respectively, each within its own 4-bit window) -- distinct
 * from BYTE_ROT, which rotates across the nibble boundary. Pointwise
 * bijection -> stride/phase. */
static inline u8 nib_rot(u8 nib, int k) { return (u8)(((nib << k) | (nib >> (4 - k))) & 0xF); }
static void ap_nibbitrot(u8 *d, int n, int s, int p, u32 amp) {
    int k1 = (int)(amp & 3), k2 = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = nib_rot((u8)(v & 0xF), k1), hi = nib_rot((u8)((v >> 4) & 0xF), k2);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static void inv_nibbitrot(u8 *d, int n, int s, int p, u32 amp) {
    int k1 = (int)(amp & 3), k2 = (int)((amp >> 2) & 3);
    int ik1 = (4 - k1) & 3, ik2 = (4 - k2) & 3;
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 lo = nib_rot((u8)(v & 0xF), ik1), hi = nib_rot((u8)((v >> 4) & 0xF), ik2);
        d[i] = (u8)(lo | (hi << 4));
    }
}
static double search_nibbitrot(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k1 = 0; k1 < 4; k1++) {
                for (int k2 = 0; k2 < 4; k2++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 lo = nib_rot((u8)(u & 0xF), k1), hi = nib_rot((u8)((u >> 4) & 0xF), k2);
                        rf[(u8)(lo | (hi << 4))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)k1 | ((u32)k2 << 2); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- Predictor family (deltastuff.md): pred = f(already-known prior
 * bytes), residual = actual - pred (or XOR). Safe regardless of what f
 * is, since decode recomputes the identical f from already-recovered
 * data -- the predictor function itself never needs to be invertible,
 * only "subtract/XOR then add/XOR back" does, which always holds. */

/* ---- DELTA_OFFSET: pred = prev + c (c searched, -32..31) -- catches a
 * roughly-constant per-step increment/decrement (e.g. counter data)
 * that plain lag-1 DELTA (c=0 only) can't express. */
static void ap_deltaoffset(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int c = (int)(amp & 0x3F) - 32;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig = d[i];
        u8 pred = (u8)((int)prev + c);
        d[i] = (u8)(orig - pred);
        prev = orig;
    }
}
static void inv_deltaoffset(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int c = (int)(amp & 0x3F) - 32;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)((int)prev + c);
        u8 orig = (u8)(d[i] + pred);
        d[i] = orig;
        prev = orig;
    }
}
static double search_deltaoffset(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 32;
    static u8 scr[BLOCK];
    for (int ci = 0; ci < 64; ci++) {
        memcpy(scr, d, (size_t)n);
        ap_deltaoffset(scr, n, 0, 0, (u32)ci);
        double net = (S_of(scr, n) - Sb) - oh_flat(6.0);
        if (net > best) { best = net; ba = (u32)ci; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- DELTA_OFFSET_WIDE: same mechanism as DELTA_OFFSET, but c spans
 * the full -128..127 (8 bits) instead of -32..31 (6 bits) -- reaches
 * steeper constant per-step increments than the narrower range covers. */
static void ap_deltaoffset_wide(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int c = (int)(amp & 0xFF) - 128;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig = d[i];
        u8 pred = (u8)((int)prev + c);
        d[i] = (u8)(orig - pred);
        prev = orig;
    }
}
static void inv_deltaoffset_wide(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int c = (int)(amp & 0xFF) - 128;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)((int)prev + c);
        u8 orig = (u8)(d[i] + pred);
        d[i] = orig;
        prev = orig;
    }
}
static double search_deltaoffset_wide(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 128;
    static u8 scr[BLOCK];
    for (int ci = 0; ci < 256; ci++) {
        memcpy(scr, d, (size_t)n);
        ap_deltaoffset_wide(scr, n, 0, 0, (u32)ci);
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)ci; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- AVG2_PRED: pred = floor((prev+prev2)/2) -- classic averaging
 * predictor. No searched parameter. */
static void ap_avg2pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)(((int)pr1 + (int)pr2) / 2);
        d[i] = (u8)(orig_i - pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_avg2pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)(((int)pr1 + (int)pr2) / 2);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_avg2pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_avg2pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- MIN2_PRED / MAX2_PRED: pred = min(prev,prev2) / max(prev,prev2).
 * Same safety: the predictor function needn't be invertible, only the
 * subtract-then-add cycle around it. No searched parameter. */
static void ap_min2pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = pr1 < pr2 ? pr1 : pr2;
        d[i] = (u8)(orig_i - pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_min2pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 pred = pr1 < pr2 ? pr1 : pr2;
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_min2pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_min2pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}
static void ap_max2pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = pr1 > pr2 ? pr1 : pr2;
        d[i] = (u8)(orig_i - pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_max2pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 pred = pr1 > pr2 ? pr1 : pr2;
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_max2pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_max2pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- AND_PRED / OR_PRED: pred = prev & prev2 / prev | prev2. Neither
 * AND nor OR is invertible by itself -- doesn't matter, only the
 * subtract-then-add cycle needs to be, and it always is. */
static void ap_andpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)(pr1 & pr2);
        d[i] = (u8)(orig_i - pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_andpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)(pr1 & pr2);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_andpred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_andpred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}
static void ap_orpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)(pr1 | pr2);
        d[i] = (u8)(orig_i - pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_orpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)(pr1 | pr2);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_orpred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_orpred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- AVG3_PRED / AVG4_PRED: pred = average of last 3/4 ORIGINAL bytes,
 * tracked via a small shift register (hist[0]=most recent). Same safe
 * subtract-then-add pattern. */
static void ap_avg3pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[3] = { d[0], 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)(((int)h[0] + h[1] + h[2]) / 3);
        d[i] = (u8)(orig_i - pred);
        h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static void inv_avg3pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[3] = { d[0], 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)(((int)h[0] + h[1] + h[2]) / 3);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static double search_avg3pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_avg3pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}
static void ap_avg4pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[4] = { d[0], 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)(((int)h[0] + h[1] + h[2] + h[3]) / 4);
        d[i] = (u8)(orig_i - pred);
        h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static void inv_avg4pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[4] = { d[0], 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)(((int)h[0] + h[1] + h[2] + h[3]) / 4);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static double search_avg4pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_avg4pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- MIN4_PRED / MAX4_PRED: pred = min/max of last 4 original bytes. */
static void ap_min4pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[4] = { d[0], 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = h[0]; for (int k = 1; k < 4; k++) if (h[k] < pred) pred = h[k];
        d[i] = (u8)(orig_i - pred);
        h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static void inv_min4pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[4] = { d[0], 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = h[0]; for (int k = 1; k < 4; k++) if (h[k] < pred) pred = h[k];
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static double search_min4pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_min4pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}
static void ap_max4pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[4] = { d[0], 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = h[0]; for (int k = 1; k < 4; k++) if (h[k] > pred) pred = h[k];
        d[i] = (u8)(orig_i - pred);
        h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static void inv_max4pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[4] = { d[0], 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = h[0]; for (int k = 1; k < 4; k++) if (h[k] > pred) pred = h[k];
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static double search_max4pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_max4pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- MIN3_PRED / MAX3_PRED: pred = min/max of last 3 original bytes --
 * gap-filler between MIN2_PRED/MAX2_PRED and MIN4_PRED/MAX4_PRED. */
static void ap_min3pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[3] = { d[0], 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = h[0]; for (int k = 1; k < 3; k++) if (h[k] < pred) pred = h[k];
        d[i] = (u8)(orig_i - pred);
        h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static void inv_min3pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[3] = { d[0], 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = h[0]; for (int k = 1; k < 3; k++) if (h[k] < pred) pred = h[k];
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static double search_min3pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_min3pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}
static void ap_max3pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[3] = { d[0], 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = h[0]; for (int k = 1; k < 3; k++) if (h[k] > pred) pred = h[k];
        d[i] = (u8)(orig_i - pred);
        h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static void inv_max3pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[3] = { d[0], 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = h[0]; for (int k = 1; k < 3; k++) if (h[k] > pred) pred = h[k];
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static double search_max3pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_max3pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- AVG5_PRED: pred = average of last 5 original bytes -- extends
 * AVG3_PRED/AVG4_PRED one tap further. */
static void ap_avg5pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[5] = { d[0], 0, 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)(((int)h[0] + h[1] + h[2] + h[3] + h[4]) / 5);
        d[i] = (u8)(orig_i - pred);
        h[4] = h[3]; h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static void inv_avg5pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[5] = { d[0], 0, 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)(((int)h[0] + h[1] + h[2] + h[3] + h[4]) / 5);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        h[4] = h[3]; h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static double search_avg5pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_avg5pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- AVG<N>_PRED / MIN<N>_PRED / MAX<N>_PRED / MEDIAN<N>_PRED families:
 * extend the AVG2/3/4/5, MIN2/3/4, MAX2/3/4, MEDIAN3/5/7 tap-count
 * sweeps further -- AVG6/7/8, MIN5/6, MAX5/6, MEDIAN9/11. Generated via
 * macro: identical shift-register-of-originals mechanism, only the tap
 * count differs. */
#define DEFINE_AVGN_PRED(SUF, NTAPS) \
static void ap_avg##SUF##pred(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)amp; \
    u8 h[NTAPS]; h[0] = d[0]; for (int k = 1; k < (NTAPS); k++) h[k] = 0; \
    for (int i = 1; i < n; i++) { \
        u8 orig_i = d[i]; \
        int sum = 0; for (int k = 0; k < (NTAPS); k++) sum += h[k]; \
        u8 pred = (u8)(sum / (NTAPS)); \
        d[i] = (u8)(orig_i - pred); \
        for (int k = (NTAPS) - 1; k > 0; k--) h[k] = h[k - 1]; \
        h[0] = orig_i; \
    } \
} \
static void inv_avg##SUF##pred(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)amp; \
    u8 h[NTAPS]; h[0] = d[0]; for (int k = 1; k < (NTAPS); k++) h[k] = 0; \
    for (int i = 1; i < n; i++) { \
        int sum = 0; for (int k = 0; k < (NTAPS); k++) sum += h[k]; \
        u8 pred = (u8)(sum / (NTAPS)); \
        u8 orig_i = (u8)(d[i] + pred); \
        d[i] = orig_i; \
        for (int k = (NTAPS) - 1; k > 0; k--) h[k] = h[k - 1]; \
        h[0] = orig_i; \
    } \
} \
static double search_avg##SUF##pred(const u8 *d, int n, double Sb, Instr *out) { \
    static u8 scr[BLOCK]; \
    memcpy(scr, d, (size_t)n); \
    ap_avg##SUF##pred(scr, n, 0, 0, 0); \
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0); \
    out->stride = 0; out->phase = 0; out->amp = 0; \
    return net; \
}
#define DEFINE_MINN_PRED(SUF, NTAPS) \
static void ap_min##SUF##pred(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)amp; \
    u8 h[NTAPS]; h[0] = d[0]; for (int k = 1; k < (NTAPS); k++) h[k] = 0; \
    for (int i = 1; i < n; i++) { \
        u8 orig_i = d[i]; \
        u8 pred = h[0]; for (int k = 1; k < (NTAPS); k++) if (h[k] < pred) pred = h[k]; \
        d[i] = (u8)(orig_i - pred); \
        for (int k = (NTAPS) - 1; k > 0; k--) h[k] = h[k - 1]; \
        h[0] = orig_i; \
    } \
} \
static void inv_min##SUF##pred(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)amp; \
    u8 h[NTAPS]; h[0] = d[0]; for (int k = 1; k < (NTAPS); k++) h[k] = 0; \
    for (int i = 1; i < n; i++) { \
        u8 pred = h[0]; for (int k = 1; k < (NTAPS); k++) if (h[k] < pred) pred = h[k]; \
        u8 orig_i = (u8)(d[i] + pred); \
        d[i] = orig_i; \
        for (int k = (NTAPS) - 1; k > 0; k--) h[k] = h[k - 1]; \
        h[0] = orig_i; \
    } \
} \
static double search_min##SUF##pred(const u8 *d, int n, double Sb, Instr *out) { \
    static u8 scr[BLOCK]; \
    memcpy(scr, d, (size_t)n); \
    ap_min##SUF##pred(scr, n, 0, 0, 0); \
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0); \
    out->stride = 0; out->phase = 0; out->amp = 0; \
    return net; \
}
#define DEFINE_MAXN_PRED(SUF, NTAPS) \
static void ap_max##SUF##pred(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)amp; \
    u8 h[NTAPS]; h[0] = d[0]; for (int k = 1; k < (NTAPS); k++) h[k] = 0; \
    for (int i = 1; i < n; i++) { \
        u8 orig_i = d[i]; \
        u8 pred = h[0]; for (int k = 1; k < (NTAPS); k++) if (h[k] > pred) pred = h[k]; \
        d[i] = (u8)(orig_i - pred); \
        for (int k = (NTAPS) - 1; k > 0; k--) h[k] = h[k - 1]; \
        h[0] = orig_i; \
    } \
} \
static void inv_max##SUF##pred(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)amp; \
    u8 h[NTAPS]; h[0] = d[0]; for (int k = 1; k < (NTAPS); k++) h[k] = 0; \
    for (int i = 1; i < n; i++) { \
        u8 pred = h[0]; for (int k = 1; k < (NTAPS); k++) if (h[k] > pred) pred = h[k]; \
        u8 orig_i = (u8)(d[i] + pred); \
        d[i] = orig_i; \
        for (int k = (NTAPS) - 1; k > 0; k--) h[k] = h[k - 1]; \
        h[0] = orig_i; \
    } \
} \
static double search_max##SUF##pred(const u8 *d, int n, double Sb, Instr *out) { \
    static u8 scr[BLOCK]; \
    memcpy(scr, d, (size_t)n); \
    ap_max##SUF##pred(scr, n, 0, 0, 0); \
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0); \
    out->stride = 0; out->phase = 0; out->amp = 0; \
    return net; \
}
#define DEFINE_MEDIANN_PRED(SUF, NTAPS) \
static inline u8 median##SUF##_of(u8 *arr) { \
    u8 tmp[NTAPS]; \
    for (int i = 0; i < (NTAPS); i++) tmp[i] = arr[i]; \
    for (int i = 1; i < (NTAPS); i++) { u8 v = tmp[i]; int j = i - 1; while (j >= 0 && tmp[j] > v) { tmp[j+1] = tmp[j]; j--; } tmp[j+1] = v; } \
    return tmp[(NTAPS) / 2]; \
} \
static void ap_median##SUF##pred(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)amp; \
    u8 h[NTAPS]; h[0] = d[0]; for (int k = 1; k < (NTAPS); k++) h[k] = 0; \
    for (int i = 1; i < n; i++) { \
        u8 orig_i = d[i]; \
        u8 pred = median##SUF##_of(h); \
        d[i] = (u8)(orig_i - pred); \
        for (int k = (NTAPS) - 1; k > 0; k--) h[k] = h[k - 1]; \
        h[0] = orig_i; \
    } \
} \
static void inv_median##SUF##pred(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)amp; \
    u8 h[NTAPS]; h[0] = d[0]; for (int k = 1; k < (NTAPS); k++) h[k] = 0; \
    for (int i = 1; i < n; i++) { \
        u8 pred = median##SUF##_of(h); \
        u8 orig_i = (u8)(d[i] + pred); \
        d[i] = orig_i; \
        for (int k = (NTAPS) - 1; k > 0; k--) h[k] = h[k - 1]; \
        h[0] = orig_i; \
    } \
} \
static double search_median##SUF##pred(const u8 *d, int n, double Sb, Instr *out) { \
    static u8 scr[BLOCK]; \
    memcpy(scr, d, (size_t)n); \
    ap_median##SUF##pred(scr, n, 0, 0, 0); \
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0); \
    out->stride = 0; out->phase = 0; out->amp = 0; \
    return net; \
}
DEFINE_AVGN_PRED(6, 6)
DEFINE_AVGN_PRED(7, 7)
DEFINE_AVGN_PRED(8, 8)
DEFINE_MINN_PRED(5, 5)
DEFINE_MINN_PRED(6, 6)
DEFINE_MAXN_PRED(5, 5)
DEFINE_MAXN_PRED(6, 6)
DEFINE_MEDIANN_PRED(9, 9)
DEFINE_MEDIANN_PRED(11, 11)

/* ---- MEDIAN3_PRED / MEDIAN5_PRED: pred = median of last 3/5 original
 * bytes -- JPEG-LS-style, handles edges/discontinuities better than a
 * pure average. Small insertion-sort to find the median, cheap at this
 * size. */
static inline u8 median_of(u8 *arr, int k) {
    u8 tmp[5];
    for (int i = 0; i < k; i++) tmp[i] = arr[i];
    for (int i = 1; i < k; i++) { u8 v = tmp[i]; int j = i - 1; while (j >= 0 && tmp[j] > v) { tmp[j+1] = tmp[j]; j--; } tmp[j+1] = v; }
    return tmp[k / 2];
}
static void ap_median3pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[3] = { d[0], 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = median_of(h, 3);
        d[i] = (u8)(orig_i - pred);
        h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static void inv_median3pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[3] = { d[0], 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = median_of(h, 3);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static double search_median3pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_median3pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}
static void ap_median5pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[5] = { d[0], 0, 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = median_of(h, 5);
        d[i] = (u8)(orig_i - pred);
        h[4] = h[3]; h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static void inv_median5pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[5] = { d[0], 0, 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = median_of(h, 5);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        h[4] = h[3]; h[3] = h[2]; h[2] = h[1]; h[1] = h[0]; h[0] = orig_i;
    }
}
static double search_median5pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_median5pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- MEDIAN7_PRED: pred = median of last 7 original bytes -- extends
 * MEDIAN3_PRED/MEDIAN5_PRED one step further (own insertion-sort helper
 * since median_of's scratch buffer is sized for k<=5). */
static inline u8 median7_of(u8 *arr) {
    u8 tmp[7];
    for (int i = 0; i < 7; i++) tmp[i] = arr[i];
    for (int i = 1; i < 7; i++) { u8 v = tmp[i]; int j = i - 1; while (j >= 0 && tmp[j] > v) { tmp[j+1] = tmp[j]; j--; } tmp[j+1] = v; }
    return tmp[3];
}
static void ap_median7pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[7] = { d[0], 0, 0, 0, 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = median7_of(h);
        d[i] = (u8)(orig_i - pred);
        for (int k = 6; k > 0; k--) h[k] = h[k-1];
        h[0] = orig_i;
    }
}
static void inv_median7pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 h[7] = { d[0], 0, 0, 0, 0, 0, 0 };
    for (int i = 1; i < n; i++) {
        u8 pred = median7_of(h);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        for (int k = 6; k > 0; k--) h[k] = h[k-1];
        h[0] = orig_i;
    }
}
static double search_median7pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_median7pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- SHIFT_PRED: pred = (prev<<1) + prev2 -- a shift-mix predictor,
 * distinct combination shape from plain weighted-sum (LINPRED2). */
static void ap_shiftpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)(((int)pr1 << 1) + pr2);
        d[i] = (u8)(orig_i - pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_shiftpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)(((int)pr1 << 1) + pr2);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_shiftpred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_shiftpred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- MOMENTUM2_PRED: pred = prev + 2*(prev-prev2) = 3*prev - 2*prev2 --
 * doubled-momentum sibling of DELTA2's prev+(prev-prev2) (=2*prev-prev2). */
static void ap_momentum2pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)(3 * (int)pr1 - 2 * (int)pr2);
        d[i] = (u8)(orig_i - pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_momentum2pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 pred = (u8)(3 * (int)pr1 - 2 * (int)pr2);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_momentum2pred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_momentum2pred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- WAVG_PRED: named weighted-average configs (3a+b)/4, (7a+b)/8,
 * (5a+3b)/8, (a+b)/2 -- classic fixed weighting schemes (searched among
 * a small set), distinct from LINPRED2's generic coefficient search
 * since these specific ratios are named/common in audio/image codecs. */
static double search_wavgpred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    double best = -1e18; u32 ba = 0;
    for (int cfg = 0; cfg < 4; cfg++) {
        memcpy(scr, d, (size_t)n);
        u8 pr1 = scr[0], pr2 = 0;
        for (int i = 1; i < n; i++) {
            u8 orig_i = scr[i];
            int pred;
            switch (cfg) {
                case 0: pred = (3 * (int)pr1 + pr2) / 4; break;
                case 1: pred = (7 * (int)pr1 + pr2) / 8; break;
                case 2: pred = (5 * (int)pr1 + 3 * (int)pr2) / 8; break;
                default: pred = ((int)pr1 + pr2) / 2; break;
            }
            scr[i] = (u8)(orig_i - (u8)pred);
            pr2 = pr1; pr1 = orig_i;
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(2.0);
        if (net > best) { best = net; ba = (u32)cfg; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}
static void ap_wavgpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int cfg = (int)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        int pred;
        switch (cfg) {
            case 0: pred = (3 * (int)pr1 + pr2) / 4; break;
            case 1: pred = (7 * (int)pr1 + pr2) / 8; break;
            case 2: pred = (5 * (int)pr1 + 3 * (int)pr2) / 8; break;
            default: pred = ((int)pr1 + pr2) / 2; break;
        }
        d[i] = (u8)(orig_i - (u8)pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_wavgpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int cfg = (int)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        int pred;
        switch (cfg) {
            case 0: pred = (3 * (int)pr1 + pr2) / 4; break;
            case 1: pred = (7 * (int)pr1 + pr2) / 8; break;
            case 2: pred = (5 * (int)pr1 + 3 * (int)pr2) / 8; break;
            default: pred = ((int)pr1 + pr2) / 2; break;
        }
        u8 orig_i = (u8)(d[i] + (u8)pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}

/* ---- WAVG_PRED2: same weighted-average mechanism as WAVG_PRED, but a
 * different config set weighting prev2 MORE heavily than prev (the
 * mirror image of WAVG_PRED's prev-heavy ratios), plus one new 2:1
 * ratio -- (a+3b)/4, (a+7b)/8, (3a+5b)/8, (2a+b)/3. */
static double search_wavgpred2(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    double best = -1e18; u32 ba = 0;
    for (int cfg = 0; cfg < 4; cfg++) {
        memcpy(scr, d, (size_t)n);
        u8 pr1 = scr[0], pr2 = 0;
        for (int i = 1; i < n; i++) {
            u8 orig_i = scr[i];
            int pred;
            switch (cfg) {
                case 0: pred = ((int)pr1 + 3 * (int)pr2) / 4; break;
                case 1: pred = ((int)pr1 + 7 * (int)pr2) / 8; break;
                case 2: pred = (3 * (int)pr1 + 5 * (int)pr2) / 8; break;
                default: pred = (2 * (int)pr1 + pr2) / 3; break;
            }
            scr[i] = (u8)(orig_i - (u8)pred);
            pr2 = pr1; pr1 = orig_i;
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(2.0);
        if (net > best) { best = net; ba = (u32)cfg; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}
static void ap_wavgpred2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int cfg = (int)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        int pred;
        switch (cfg) {
            case 0: pred = ((int)pr1 + 3 * (int)pr2) / 4; break;
            case 1: pred = ((int)pr1 + 7 * (int)pr2) / 8; break;
            case 2: pred = (3 * (int)pr1 + 5 * (int)pr2) / 8; break;
            default: pred = (2 * (int)pr1 + pr2) / 3; break;
        }
        d[i] = (u8)(orig_i - (u8)pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_wavgpred2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int cfg = (int)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        int pred;
        switch (cfg) {
            case 0: pred = ((int)pr1 + 3 * (int)pr2) / 4; break;
            case 1: pred = ((int)pr1 + 7 * (int)pr2) / 8; break;
            case 2: pred = (3 * (int)pr1 + 5 * (int)pr2) / 8; break;
            default: pred = (2 * (int)pr1 + pr2) / 3; break;
        }
        u8 orig_i = (u8)(d[i] + (u8)pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}

/* ---- REVERSE_DELTA: predicts EACH byte from the NEXT one (pred =
 * x[i+1]) instead of the previous one -- a genuinely different topology:
 * encode must run ASCENDING (reads d[i+1], still original since ascending
 * hasn't reached it yet) while decode must run DESCENDING (needs d[i+1]
 * already recovered) -- the opposite direction pairing from DELTA's
 * descending-apply/ascending-invert. Last byte untouched (no successor). */
static void ap_reversedelta(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i < n - 1; i++) d[i] = (u8)(d[i] - d[i + 1]);
}
static void inv_reversedelta(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = n - 2; i >= 0; i--) d[i] = (u8)(d[i] + d[i + 1]);
}
static double search_reversedelta(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_reversedelta(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- BLOCK_BITSHIFT: rotate the ENTIRE 4096-byte block as one giant
 * bitstream by k bits (1..7), not per-byte like BYTE_ROT -- each output
 * byte draws bits across a byte boundary from two adjacent original
 * bytes, wrapping circularly. Snapshot-based (safe, same pattern as
 * PAETH2D) since neighbor lookups need untouched originals. */
static void ap_blockbitshift(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int k = (int)amp;
    static u8 orig[BLOCK];
    memcpy(orig, d, (size_t)n);
    for (int i = 0; i < n; i++) {
        int next_idx = (i + 1) % n;
        d[i] = (u8)((orig[i] << k) | (orig[next_idx] >> (8 - k)));
    }
}
static void inv_blockbitshift(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int k = (int)amp;
    static u8 cur[BLOCK];
    memcpy(cur, d, (size_t)n);
    for (int i = 0; i < n; i++) {
        int prev_idx = (i - 1 + n) % n;
        d[i] = (u8)((cur[i] >> k) | (cur[prev_idx] << (8 - k)));
    }
}
static double search_blockbitshift(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    double best = -1e18; u32 bk = 1;
    for (int k = 1; k <= 7; k++) {
        memcpy(scr, d, (size_t)n);
        ap_blockbitshift(scr, n, 0, 0, (u32)k);
        double net = (S_of(scr, n) - Sb) - oh_flat(3.0);
        if (net > best) { best = net; bk = (u32)k; }
    }
    out->stride = 0; out->phase = 0; out->amp = bk;
    return best;
}

/* ---- MED2D_PRED: JPEG-LS's median-edge-detector predictor -- reshape
 * 64x64, predict from left/up/upleft using MED logic (distinct
 * decision rule from PAETH2D's closest-of-three): if up is between
 * left and upleft, predict left+up-upleft; else predict whichever of
 * left/up is on the correct side. Same snapshot-based safety as
 * PAETH2D. No searched parameter. */
static inline int med_pred(int left, int up, int upleft) {
    int mn = left < up ? left : up, mx = left > up ? left : up;
    if (upleft >= mx) return mn;
    if (upleft <= mn) return mx;
    return left + up - upleft;
}
static void ap_med2d(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 orig[BLOCK];
    memcpy(orig, d, (size_t)n);
    for (int r = 0; r < P2D_N; r++) {
        for (int c = 0; c < P2D_N; c++) {
            int i = r * P2D_N + c;
            int left   = (c > 0) ? orig[r * P2D_N + (c - 1)] : 0;
            int up     = (r > 0) ? orig[(r - 1) * P2D_N + c] : 0;
            int upleft = (r > 0 && c > 0) ? orig[(r - 1) * P2D_N + (c - 1)] : 0;
            d[i] = (u8)(orig[i] - med_pred(left, up, upleft));
        }
    }
}
static void inv_med2d(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int r = 0; r < P2D_N; r++) {
        for (int c = 0; c < P2D_N; c++) {
            int i = r * P2D_N + c;
            int left   = (c > 0) ? d[r * P2D_N + (c - 1)] : 0;
            int up     = (r > 0) ? d[(r - 1) * P2D_N + c] : 0;
            int upleft = (r > 0 && c > 0) ? d[(r - 1) * P2D_N + (c - 1)] : 0;
            d[i] = (u8)(d[i] + med_pred(left, up, upleft));
        }
    }
}
static double search_med2d(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_med2d(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- MED2D_PRED_W32: same MED predictor as MED2D_PRED, reshaped 32x128
 * instead of 64x64 (paired with PAETH2D_W32's grid choice). */
static void ap_med2d_w32(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 orig[BLOCK];
    memcpy(orig, d, (size_t)n);
    for (int r = 0; r < P2D_H; r++) {
        for (int c = 0; c < P2D_W; c++) {
            int i = r * P2D_W + c;
            int left   = (c > 0) ? orig[r * P2D_W + (c - 1)] : 0;
            int up     = (r > 0) ? orig[(r - 1) * P2D_W + c] : 0;
            int upleft = (r > 0 && c > 0) ? orig[(r - 1) * P2D_W + (c - 1)] : 0;
            d[i] = (u8)(orig[i] - med_pred(left, up, upleft));
        }
    }
}
static void inv_med2d_w32(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int r = 0; r < P2D_H; r++) {
        for (int c = 0; c < P2D_W; c++) {
            int i = r * P2D_W + c;
            int left   = (c > 0) ? d[r * P2D_W + (c - 1)] : 0;
            int up     = (r > 0) ? d[(r - 1) * P2D_W + c] : 0;
            int upleft = (r > 0 && c > 0) ? d[(r - 1) * P2D_W + (c - 1)] : 0;
            d[i] = (u8)(d[i] + med_pred(left, up, upleft));
        }
    }
}
static double search_med2d_w32(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_med2d_w32(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- CTXTABLE2: 2-byte-context predictor (65536-entry table indexed by
 * the pair of previous bytes) -- richer sibling of CTXTABLE's single-
 * byte context. Same safe update-then-predict pattern; table is static
 * storage, not stack. No searched parameter. */
static void ap_ctxtable2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 table[65536];
    memset(table, 0, sizeof table);
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig = d[i];
        u16 ctx = (u16)(((u16)p2 << 8) | p1);
        u8 pred = table[ctx];
        d[i] = (u8)(orig - pred);
        table[ctx] = orig;
        p2 = p1; p1 = orig;
    }
}
static void inv_ctxtable2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 table[65536];
    memset(table, 0, sizeof table);
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u16 ctx = (u16)(((u16)p2 << 8) | p1);
        u8 pred = table[ctx];
        u8 orig = (u8)(d[i] + pred);
        d[i] = orig;
        table[ctx] = orig;
        p2 = p1; p1 = orig;
    }
}
static double search_ctxtable2(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_ctxtable2(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- DUAL_LAG_XOR: pred = x[i-lag1] XOR x[i-lag2], BOTH lags searched
 * independently (1..16 each) -- catches "x[i-8] xor x[i-1]"-style
 * predictors the fixed-adjacent-lag MULTINEIGH_XOR can't reach. Needs
 * both priors to still be ORIGINAL when read, so apply runs descending
 * (mirrors DELTA's safety argument, generalized to two lookback
 * distances) and invert runs ascending. */
static void ap_duallagxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag1 = (int)(amp & 0xF) + 1, lag2 = (int)((amp >> 4) & 0xF) + 1;
    int maxlag = lag1 > lag2 ? lag1 : lag2;
    for (int i = n - 1; i >= maxlag; i--) d[i] ^= d[i - lag1] ^ d[i - lag2];
}
static void inv_duallagxor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int lag1 = (int)(amp & 0xF) + 1, lag2 = (int)((amp >> 4) & 0xF) + 1;
    int maxlag = lag1 > lag2 ? lag1 : lag2;
    for (int i = maxlag; i < n; i++) d[i] ^= d[i - lag1] ^ d[i - lag2];
}
static double search_duallagxor(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int l1 = 0; l1 < 16; l1++) {
        for (int l2 = 0; l2 < 16; l2++) {
            if (l1 == l2) continue;
            memcpy(scr, d, (size_t)n);
            u32 amp = (u32)l1 | ((u32)l2 << 4);
            ap_duallagxor(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- SHIFT1_PRED: pred = prev<<k (dir=0) or prev>>k (dir=1), a bare
 * single-tap shift with no second operand -- the doc lists these as
 * standalone predictors distinct from the combined (prev<<1)+prev2
 * already built as SHIFT_PRED. */
static void ap_shift1pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int k = (int)(amp & 7), dir = (int)((amp >> 3) & 1);
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = dir ? (u8)(prev >> k) : (u8)(prev << k);
        d[i] = (u8)(orig_i - pred);
        prev = orig_i;
    }
}
static void inv_shift1pred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int k = (int)(amp & 7), dir = (int)((amp >> 3) & 1);
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 pred = dir ? (u8)(prev >> k) : (u8)(prev << k);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        prev = orig_i;
    }
}
static double search_shift1pred(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int k = 0; k < 8; k++) {
        for (int dir = 0; dir < 2; dir++) {
            memcpy(scr, d, (size_t)n);
            u32 amp = (u32)k | ((u32)dir << 3);
            ap_shift1pred(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh_flat(4.0);
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- ROTATE_PRED: pred = ROL(prev, k) -- predicts the current byte from
 * a bit-ROTATED version of the previous byte (searched k=1..7), not a
 * transform of the current byte's own value like BYTE_ROT. */
static void ap_rotatepred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int k = (int)amp;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = byterot_fwd(prev, k);
        d[i] = (u8)(orig_i - pred);
        prev = orig_i;
    }
}
static void inv_rotatepred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int k = (int)amp;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 pred = byterot_fwd(prev, k);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        prev = orig_i;
    }
}
static double search_rotatepred(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bk = 1;
    static u8 scr[BLOCK];
    for (int k = 1; k <= 7; k++) {
        memcpy(scr, d, (size_t)n);
        ap_rotatepred(scr, n, 0, 0, (u32)k);
        double net = (S_of(scr, n) - Sb) - oh_flat(3.0);
        if (net > best) { best = net; bk = (u32)k; }
    }
    out->stride = 0; out->phase = 0; out->amp = bk;
    return best;
}

/* ---- ROTATE_PRED_LAG2: same bit-rotate predictor mechanism as
 * ROTATE_PRED, but predicts from a ROTATED version of prev2 (lag 2)
 * instead of prev (lag 1) -- tests whether rotate-correlation shows up
 * one step further back than the immediate neighbor. */
static void ap_rotatepred_lag2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int k = (int)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = byterot_fwd(pr2, k);
        d[i] = (u8)(orig_i - pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_rotatepred_lag2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int k = (int)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 pred = byterot_fwd(pr2, k);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_rotatepred_lag2(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bk = 1;
    static u8 scr[BLOCK];
    for (int k = 1; k <= 7; k++) {
        memcpy(scr, d, (size_t)n);
        ap_rotatepred_lag2(scr, n, 0, 0, (u32)k);
        double net = (S_of(scr, n) - Sb) - oh_flat(3.0);
        if (net > best) { best = net; bk = (u32)k; }
    }
    out->stride = 0; out->phase = 0; out->amp = bk;
    return best;
}

/* ---- HALFMOMENTUM_PRED: pred = prev + (prev-prev2)/2 -- the doc's
 * literal half-strength momentum, distinct from MOMENTUM2_PRED's
 * doubled-strength (3*prev-2*prev2) variant already built. */
static void ap_halfmomentumpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        int diff = (int)pr1 - (int)pr2;
        u8 pred = (u8)((int)pr1 + diff / 2);
        d[i] = (u8)(orig_i - pred);
        pr2 = pr1; pr1 = orig_i;
    }
}
static void inv_halfmomentumpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 pr1 = d[0], pr2 = 0;
    for (int i = 1; i < n; i++) {
        int diff = (int)pr1 - (int)pr2;
        u8 pred = (u8)((int)pr1 + diff / 2);
        u8 orig_i = (u8)(d[i] + pred);
        d[i] = orig_i;
        pr2 = pr1; pr1 = orig_i;
    }
}
static double search_halfmomentumpred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_halfmomentumpred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- MAGNITUDE_PRED: sign bit (bit7, untouched) picked apart; predict
 * only the MAGNITUDE (lo7) from the previous byte's magnitude, sign
 * carried through unchanged -- the doc's "predict abs(value) only"
 * idea, applied per-byte rather than as a stream split. */
static void ap_magnitudepred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 prevmag = (u8)(d[0] & 0x7F);
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 sign = (u8)(orig_i & 0x80), mag = (u8)(orig_i & 0x7F);
        u8 newmag = (u8)((mag - prevmag) & 0x7F);
        d[i] = (u8)(sign | newmag);
        prevmag = mag;
    }
}
static void inv_magnitudepred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    u8 prevmag = (u8)(d[0] & 0x7F);
    for (int i = 1; i < n; i++) {
        u8 v = d[i];
        u8 sign = (u8)(v & 0x80), newmag = (u8)(v & 0x7F);
        u8 mag = (u8)((newmag + prevmag) & 0x7F);
        d[i] = (u8)(sign | mag);
        prevmag = mag;
    }
}
static double search_magnitudepred(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_magnitudepred(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- KALMAN_PRED: proper 2nd-order tracker (position + velocity state,
 * both updated from prediction error) -- the doc's actual Kalman-style
 * idea, distinct from EMA_BIAS's simpler 1st-order-only smoothing.
 * Fixed-point velocity (8 fractional bits) updated by a fraction of the
 * residual each step; gain gk searched over a small set. */
static const int KALMAN_GAIN[4] = { 2, 4, 8, 16 };
static void ap_kalmanpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int gain = KALMAN_GAIN[amp & 3];
    int pos = d[0] << 8, vel = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        int predpos = pos + vel;
        u8 pred = (u8)((predpos >> 8) & 0xFF);
        u8 residual = (u8)(orig_i - pred);
        d[i] = residual;
        int err = (int)(int8_t)residual;
        pos = predpos + ((err << 8) / gain);
        vel = vel + ((err << 8) / (gain * 4));
        if (vel > (32 << 8)) vel = 32 << 8; if (vel < -(32 << 8)) vel = -(32 << 8);
    }
}
static void inv_kalmanpred(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int gain = KALMAN_GAIN[amp & 3];
    int pos = d[0] << 8, vel = 0;
    for (int i = 1; i < n; i++) {
        int predpos = pos + vel;
        u8 pred = (u8)((predpos >> 8) & 0xFF);
        u8 residual = d[i];
        u8 orig_i = (u8)(residual + pred);
        d[i] = orig_i;
        int err = (int)(int8_t)residual;
        pos = predpos + ((err << 8) / gain);
        vel = vel + ((err << 8) / (gain * 4));
        if (vel > (32 << 8)) vel = 32 << 8; if (vel < -(32 << 8)) vel = -(32 << 8);
    }
}
static double search_kalmanpred(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int g = 0; g < 4; g++) {
        memcpy(scr, d, (size_t)n);
        ap_kalmanpred(scr, n, 0, 0, (u32)g);
        double net = (S_of(scr, n) - Sb) - oh_flat(2.0);
        if (net > best) { best = net; ba = (u32)g; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- KALMAN_PRED_ALT: same 2nd-order position+velocity tracker as
 * KALMAN_PRED, but a DIFFERENT fixed gain set {1,3,6,12} and a different
 * position/velocity gain ratio (velocity updates by err/(gain*2) instead
 * of err/(gain*4)) -- a more aggressive velocity-tracking variant that
 * reacts faster to trend changes than the original's smoother ratio. */
static const int KALMAN2_GAIN[4] = { 1, 3, 6, 12 };
static void ap_kalmanpred_alt(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int gain = KALMAN2_GAIN[amp & 3];
    int pos = d[0] << 8, vel = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        int predpos = pos + vel;
        u8 pred = (u8)((predpos >> 8) & 0xFF);
        u8 residual = (u8)(orig_i - pred);
        d[i] = residual;
        int err = (int)(int8_t)residual;
        pos = predpos + ((err << 8) / gain);
        vel = vel + ((err << 8) / (gain * 2));
        if (vel > (32 << 8)) vel = 32 << 8; if (vel < -(32 << 8)) vel = -(32 << 8);
    }
}
static void inv_kalmanpred_alt(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int gain = KALMAN2_GAIN[amp & 3];
    int pos = d[0] << 8, vel = 0;
    for (int i = 1; i < n; i++) {
        int predpos = pos + vel;
        u8 pred = (u8)((predpos >> 8) & 0xFF);
        u8 residual = d[i];
        u8 orig_i = (u8)(residual + pred);
        d[i] = orig_i;
        int err = (int)(int8_t)residual;
        pos = predpos + ((err << 8) / gain);
        vel = vel + ((err << 8) / (gain * 2));
        if (vel > (32 << 8)) vel = 32 << 8; if (vel < -(32 << 8)) vel = -(32 << 8);
    }
}
static double search_kalmanpred_alt(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int g = 0; g < 4; g++) {
        memcpy(scr, d, (size_t)n);
        ap_kalmanpred_alt(scr, n, 0, 0, (u32)g);
        double net = (S_of(scr, n) - Sb) - oh_flat(2.0);
        if (net > best) { best = net; ba = (u32)g; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- KALMAN_PRED_ALT2: same 2nd-order tracker as KALMAN_PRED/
 * KALMAN_PRED_ALT, but a THIRD fixed gain set {2,5,9,18} and a slower
 * velocity-tracking ratio (err/(gain*8) instead of /4 or /2) -- a more
 * conservative/smoother variant than either existing one. */
static const int KALMAN3_GAIN[4] = { 2, 5, 9, 18 };
static void ap_kalmanpred_alt2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int gain = KALMAN3_GAIN[amp & 3];
    int pos = d[0] << 8, vel = 0;
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        int predpos = pos + vel;
        u8 pred = (u8)((predpos >> 8) & 0xFF);
        u8 residual = (u8)(orig_i - pred);
        d[i] = residual;
        int err = (int)(int8_t)residual;
        pos = predpos + ((err << 8) / gain);
        vel = vel + ((err << 8) / (gain * 8));
        if (vel > (32 << 8)) vel = 32 << 8; if (vel < -(32 << 8)) vel = -(32 << 8);
    }
}
static void inv_kalmanpred_alt2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int gain = KALMAN3_GAIN[amp & 3];
    int pos = d[0] << 8, vel = 0;
    for (int i = 1; i < n; i++) {
        int predpos = pos + vel;
        u8 pred = (u8)((predpos >> 8) & 0xFF);
        u8 residual = d[i];
        u8 orig_i = (u8)(residual + pred);
        d[i] = orig_i;
        int err = (int)(int8_t)residual;
        pos = predpos + ((err << 8) / gain);
        vel = vel + ((err << 8) / (gain * 8));
        if (vel > (32 << 8)) vel = 32 << 8; if (vel < -(32 << 8)) vel = -(32 << 8);
    }
}
static double search_kalmanpred_alt2(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int g = 0; g < 4; g++) {
        memcpy(scr, d, (size_t)n);
        ap_kalmanpred_alt2(scr, n, 0, 0, (u32)g);
        double net = (S_of(scr, n) - Sb) - oh_flat(2.0);
        if (net > best) { best = net; ba = (u32)g; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- CTXTABLE3: order-3 context predictor (16.7M-entry table indexed by
 * the 3 previous bytes) -- the doc's explicit "context length 3: ABC ->
 * ?" example. Honest caveat: a 4096-byte block has far fewer
 * observations than the 256^3 context space, so most contexts are seen
 * at most once and this is expected to show little or no net gain here
 * -- included for the empirical test, not because it's likely to win. */
static void ap_ctxtable3(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 *table = NULL;
    if (!table) table = (u8 *)calloc(1u << 24, 1);
    else memset(table, 0, 1u << 24);
    u8 p1 = d[0], p2 = 0, p3 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig = d[i];
        u32 ctx = ((u32)p3 << 16) | ((u32)p2 << 8) | p1;
        u8 pred = table[ctx];
        d[i] = (u8)(orig - pred);
        table[ctx] = orig;
        p3 = p2; p2 = p1; p1 = orig;
    }
}
static void inv_ctxtable3(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 *table = NULL;
    if (!table) table = (u8 *)calloc(1u << 24, 1);
    else memset(table, 0, 1u << 24);
    u8 p1 = d[0], p2 = 0, p3 = 0;
    for (int i = 1; i < n; i++) {
        u32 ctx = ((u32)p3 << 16) | ((u32)p2 << 8) | p1;
        u8 pred = table[ctx];
        u8 orig = (u8)(d[i] + pred);
        d[i] = orig;
        table[ctx] = orig;
        p3 = p2; p2 = p1; p1 = orig;
    }
}
static double search_ctxtable3(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_ctxtable3(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- NIBBLE_DEINTERLEAVE_XOR: pack all hi-nibbles into the first half
 * of a conceptual stream and all lo-nibbles into the second half (like
 * BITPLANE_XOR's period-8 single-bit-plane pattern, but at nibble
 * granularity) -- selects whether to touch hi or lo nibbles based on a
 * period-2 position pattern, XORing a per-half constant. Self-inverse. */
static void ap_nibdeinterleave(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 chi = (u8)(amp & 0xF), clo = (u8)((amp >> 4) & 0xF);
    for (int i = 0; i < n; i++) {
        if (i & 1) d[i] = (u8)((d[i] & 0x0F) | (((((d[i] >> 4) & 0xF) ^ chi) & 0xF) << 4));
        else       d[i] = (u8)((d[i] & 0xF0) | (((d[i] & 0xF) ^ clo) & 0xF));
    }
}
static double search_nibdeinterleave(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(8.0);
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int chi = 0; chi < 16; chi++) {
        for (int clo = 0; clo < 16; clo++) {
            memcpy(scr, d, (size_t)n);
            u32 amp = (u32)chi | ((u32)clo << 4);
            ap_nibdeinterleave(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh;
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- POPCNT_XOR_T<T> family: fills in the remaining VALID popcount
 * thresholds (nibble popcount ranges 0..4, so only T=1,2,3,4 are
 * meaningful splits) not already covered -- base POPCNT_XOR and
 * POPCNT_XOR_T2 both use T=2, and POPCNT_XOR_T3 uses T=6 which is
 * unreachable for a 4-bit popcount (always picks alo, a pre-existing
 * degenerate instruction from an earlier phase, left as-is). This adds
 * genuinely distinct, reachable thresholds T=1,3,4. */
#define DEFINE_POPCXOR_KT(SUF, TVAL) \
static void ap_popcxor_v##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hipc = __builtin_popcount((unsigned)((v >> 4) & 0xF)); \
        u8 x = (hipc >= (TVAL)) ? ahi : alo; \
        d[i] = (u8)(v ^ x); \
    } \
} \
static double search_popcxor_v##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int alo = 0; alo < 16; alo++) { \
                for (int ahi = 0; ahi < 16; ahi++) { \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        int hipc = __builtin_popcount((unsigned)((u >> 4) & 0xF)); \
                        u8 x = (hipc >= (TVAL)) ? (u8)ahi : (u8)alo; \
                        rf[u ^ x] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_POPCXOR_KT(4, 1)
DEFINE_POPCXOR_KT(5, 3)
DEFINE_POPCXOR_KT(6, 4)

/* ---- RANGE_XOR_T<T> family: same hi-nibble-magnitude-threshold split
 * mechanism as RANGE_COND_XOR (T=8)/RANGE_XOR_T2 (T=4)/RANGE_XOR_T3
 * (T=12), filling in the remaining even thresholds 2,6,10,14. */
#define DEFINE_RANGEXOR_KT(SUF, TVAL) \
static void ap_rangexor_v##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = (hi >= (TVAL)) ? ahi : alo; \
        d[i] = (u8)(v ^ x); \
    } \
} \
static double search_rangexor_v##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int alo = 0; alo < 16; alo++) { \
                for (int ahi = 0; ahi < 16; ahi++) { \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        int hi = (u >> 4) & 0xF; \
                        u8 x = (hi >= (TVAL)) ? (u8)ahi : (u8)alo; \
                        rf[u ^ x] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_RANGEXOR_KT(4, 2)
DEFINE_RANGEXOR_KT(5, 6)
DEFINE_RANGEXOR_KT(6, 10)
DEFINE_RANGEXOR_KT(7, 14)

/* ---- MAGCLASS_XOR_T4: bitlen4 ranges 0..4, so T=1,2,3,4 are the only
 * valid thresholds; base/T2/T3 already cover 3,1,2 -- this fills the
 * last one, T=4 (only the top class, hi nibble in 8..15). */
static void ap_magclass_v4(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (bitlen4(hi) >= 4) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_magclass_v4(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (bitlen4((u >> 4) & 0xF) >= 4) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- Threshold-variant siblings: POPCNT_XOR/RANGE_XOR/MAGCLASS_XOR/
 * RANK_XOR all fixed their threshold constant to keep the search
 * tractable (T=8, RANGE_T=8, MAGCLASS_T=3, K=8) -- these are the same
 * mechanisms at DIFFERENT fixed thresholds, genuinely different
 * behavior (different position/group boundaries) even though the code
 * shape is identical. */
#define POPCXOR_T2 2
#define POPCXOR_T3 6
static void ap_popcxor_t2(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hipc = __builtin_popcount((unsigned)((v >> 4) & 0xF));
        u8 x = (hipc >= POPCXOR_T2) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_popcxor_t2(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hipc = __builtin_popcount((unsigned)((u >> 4) & 0xF));
                        u8 x = (hipc >= POPCXOR_T2) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}
static void ap_popcxor_t3(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hipc = __builtin_popcount((unsigned)((v >> 4) & 0xF));
        u8 x = (hipc >= POPCXOR_T3) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_popcxor_t3(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hipc = __builtin_popcount((unsigned)((u >> 4) & 0xF));
                        u8 x = (hipc >= POPCXOR_T3) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

#define RANGE_T2 4
#define RANGE_T3 12
static void ap_rangexor_t2(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (hi >= RANGE_T2) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_rangexor_t2(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hi = (u >> 4) & 0xF;
                        u8 x = (hi >= RANGE_T2) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}
static void ap_rangexor_t3(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (hi >= RANGE_T3) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_rangexor_t3(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        int hi = (u >> 4) & 0xF;
                        u8 x = (hi >= RANGE_T3) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

#define MAGCLASS_T2 1
#define MAGCLASS_T3 2
static void ap_magclass_t2(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (bitlen4(hi) >= MAGCLASS_T2) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_magclass_t2(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (bitlen4((u >> 4) & 0xF) >= MAGCLASS_T2) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}
static void ap_magclass_t3(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (bitlen4(hi) >= MAGCLASS_T3) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_magclass_t3(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (bitlen4((u >> 4) & 0xF) >= MAGCLASS_T3) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- RANK_XOR_K<T> / RANK_ADD_K<T> family: same rank<T threshold split
 * mechanism as RANK_XOR (T=8)/RANK_XOR_K2 (T=4)/RANK_XOR_K3 (T=12)/
 * RANK_ADD_K2 (T=4), filling in the remaining even thresholds 2,6,10,14
 * for both XOR and ADD combine. Generated via macro -- identical
 * mechanism, only the fixed rank threshold differs. */
#define DEFINE_RANK_XOR_KT(SUF, TVAL) \
static void ap_rankxor_k##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    int rank[16]; rankxor_compute_rank(d, n, rank); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = (rank[hi] < (TVAL)) ? ahi : alo; \
        d[i] = (u8)(v ^ x); \
    } \
} \
static double search_rankxor_k##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int rank[16]; rankxor_compute_rank(d, n, rank); \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int alo = 0; alo < 16; alo++) { \
                for (int ahi = 0; ahi < 16; ahi++) { \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        u8 x = (rank[(u >> 4) & 0xF] < (TVAL)) ? (u8)ahi : (u8)alo; \
                        rf[u ^ x] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
#define DEFINE_RANK_ADD_KT(SUF, TVAL) \
static void ap_rankadd_k##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    int rank[16]; rankxor_compute_rank(d, n, rank); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = (rank[hi] < (TVAL)) ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF)); \
    } \
} \
static void inv_rankadd_k##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF); \
    int rank[16]; rankxor_compute_rank(d, n, rank); \
    for (int i = p; i < n; i += s) { \
        u8 v = d[i]; int hi = (v >> 4) & 0xF; \
        u8 x = (rank[hi] < (TVAL)) ? ahi : alo; \
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF)); \
    } \
} \
static double search_rankadd_k##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int rank[16]; rankxor_compute_rank(d, n, rank); \
    int total[256]; freq_of(d, n, total); \
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0; \
    for (int s = 1; s <= MAX_STRIDE; s++) { \
        double oh = oh_strided(8.0, s); \
        for (int p = 0; p < s; p++) { \
            int hit[256] = {0}; \
            for (int i = p; i < n; i += s) hit[d[i]]++; \
            int base[256]; \
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
            for (int alo = 0; alo < 16; alo++) { \
                for (int ahi = 0; ahi < 16; ahi++) { \
                    int rf[256]; memcpy(rf, base, sizeof rf); \
                    for (int u = 0; u < 256; u++) { \
                        u8 x = (rank[(u >> 4) & 0xF] < (TVAL)) ? (u8)ahi : (u8)alo; \
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u]; \
                    } \
                    double net = (S_from_freq(rf) - Sb) - oh; \
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); } \
                } \
            } \
        } \
    } \
    out->stride = bs; out->phase = bp; out->amp = ba; \
    return best; \
}
DEFINE_RANK_XOR_KT(4, 2)
DEFINE_RANK_XOR_KT(5, 6)
DEFINE_RANK_XOR_KT(6, 10)
DEFINE_RANK_XOR_KT(7, 14)
DEFINE_RANK_ADD_KT(3, 12)
DEFINE_RANK_ADD_KT(4, 2)
DEFINE_RANK_ADD_KT(5, 6)
DEFINE_RANK_ADD_KT(6, 10)
DEFINE_RANK_ADD_KT(7, 14)
DEFINE_RANK_XOR_KT(o1, 1)
DEFINE_RANK_XOR_KT(o3, 3)
DEFINE_RANK_XOR_KT(o5, 5)
DEFINE_RANK_XOR_KT(o7, 7)
DEFINE_RANK_XOR_KT(o9, 9)
DEFINE_RANK_XOR_KT(o11, 11)
DEFINE_RANK_XOR_KT(o13, 13)
DEFINE_RANK_XOR_KT(o15, 15)
DEFINE_RANK_ADD_KT(o1, 1)
DEFINE_RANK_ADD_KT(o3, 3)
DEFINE_RANK_ADD_KT(o5, 5)
DEFINE_RANK_ADD_KT(o7, 7)
DEFINE_RANK_ADD_KT(o9, 9)
DEFINE_RANK_ADD_KT(o11, 11)
DEFINE_RANK_ADD_KT(o13, 13)
DEFINE_RANK_ADD_KT(o15, 15)

#define RANKXOR_K2 4
#define RANKXOR_K3 12
static void ap_rankxor_k2(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    int rank[16]; rankxor_compute_rank(d, n, rank);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (rank[hi] < RANKXOR_K2) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_rankxor_k2(const u8 *d, int n, double Sb, Instr *out) {
    int rank[16]; rankxor_compute_rank(d, n, rank);
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (rank[(u >> 4) & 0xF] < RANKXOR_K2) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}
static void ap_rankxor_k3(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    int rank[16]; rankxor_compute_rank(d, n, rank);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (rank[hi] < RANKXOR_K3) ? ahi : alo;
        d[i] = (u8)(v ^ x);
    }
}
static double search_rankxor_k3(const u8 *d, int n, double Sb, Instr *out) {
    int rank[16]; rankxor_compute_rank(d, n, rank);
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (rank[(u >> 4) & 0xF] < RANKXOR_K3) ? (u8)ahi : (u8)alo;
                        rf[u ^ x] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- RANK_ADD_K2: ADD-sibling of RANK_XOR_K2 -- same rank<RANKXOR_K2
 * threshold split, low-nibble modular ADD combine instead of XOR (RANK_ADD
 * only had a K=8 version before; this fills the K2=4 gap to match
 * RANK_XOR_K2/K3's coverage). */
static void ap_rankadd_k2(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    int rank[16]; rankxor_compute_rank(d, n, rank);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (rank[hi] < RANKXOR_K2) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) + x) & 0xF));
    }
}
static void inv_rankadd_k2(u8 *d, int n, int s, int p, u32 amp) {
    u8 alo = (u8)(amp & 0xF), ahi = (u8)((amp >> 4) & 0xF);
    int rank[16]; rankxor_compute_rank(d, n, rank);
    for (int i = p; i < n; i += s) {
        u8 v = d[i]; int hi = (v >> 4) & 0xF;
        u8 x = (rank[hi] < RANKXOR_K2) ? ahi : alo;
        d[i] = (u8)((v & 0xF0) | (((v & 0xF) - x) & 0xF));
    }
}
static double search_rankadd_k2(const u8 *d, int n, double Sb, Instr *out) {
    int rank[16]; rankxor_compute_rank(d, n, rank);
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int alo = 0; alo < 16; alo++) {
                for (int ahi = 0; ahi < 16; ahi++) {
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 x = (rank[(u >> 4) & 0xF] < RANKXOR_K2) ? (u8)ahi : (u8)alo;
                        rf[(u8)((u & 0xF0) | (((u & 0xF) + x) & 0xF))] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)alo | ((u32)ahi << 4); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

#define WXOR_W8 8
static void ap_winxor_w8(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int base = 0; base + WXOR_W8 <= n; base += WXOR_W8) {
        u8 sum = 0; for (int j = 0; j < WXOR_W8; j++) sum ^= d[base + j];
        u8 c = gf_mul(sum, a);
        for (int j = 0; j < WXOR_W8; j++) d[base + j] ^= c;
    }
}
static double search_winxor_w8(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        for (int base = 0; base + WXOR_W8 <= n; base += WXOR_W8) {
            u8 sum = 0; for (int j = 0; j < WXOR_W8; j++) sum ^= d[base + j];
            u8 c = gf_mul(sum, (u8)a);
            for (int j = 0; j < WXOR_W8; j++) scr[base + j] = (u8)(d[base + j] ^ c);
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}
#define WXOR_W32 32
static void ap_winxor_w32(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    for (int base = 0; base + WXOR_W32 <= n; base += WXOR_W32) {
        u8 sum = 0; for (int j = 0; j < WXOR_W32; j++) sum ^= d[base + j];
        u8 c = gf_mul(sum, a);
        for (int j = 0; j < WXOR_W32; j++) d[base + j] ^= c;
    }
}
static double search_winxor_w32(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        for (int base = 0; base + WXOR_W32 <= n; base += WXOR_W32) {
            u8 sum = 0; for (int j = 0; j < WXOR_W32; j++) sum ^= d[base + j];
            u8 c = gf_mul(sum, (u8)a);
            for (int j = 0; j < WXOR_W32; j++) scr[base + j] = (u8)(d[base + j] ^ c);
        }
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- WINDOW_XOR_W<N> family: same XOR-sum-invariant window mechanism as
 * WINDOW_XOR (W=16)/WINDOW_XOR_W8/WINDOW_XOR_W32, filling in widths
 * 4,64,128,256,512,1024,2048 -- the XOR-sum invariant holds for ANY even
 * window width, so all of these are valid. Generated via macro. */
#define DEFINE_WINXOR_WN(SUF, WIDTH) \
static void ap_winxor_w##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 a = (u8)amp; \
    for (int base = 0; base + (WIDTH) <= n; base += (WIDTH)) { \
        u8 sum = 0; for (int j = 0; j < (WIDTH); j++) sum ^= d[base + j]; \
        u8 c = gf_mul(sum, a); \
        for (int j = 0; j < (WIDTH); j++) d[base + j] ^= c; \
    } \
} \
static double search_winxor_w##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double best = -1e18; u32 ba = 1; \
    static u8 scr[BLOCK]; \
    for (int a = 1; a < 256; a++) { \
        for (int base = 0; base + (WIDTH) <= n; base += (WIDTH)) { \
            u8 sum = 0; for (int j = 0; j < (WIDTH); j++) sum ^= d[base + j]; \
            u8 c = gf_mul(sum, (u8)a); \
            for (int j = 0; j < (WIDTH); j++) scr[base + j] = (u8)(d[base + j] ^ c); \
        } \
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0); \
        if (net > best) { best = net; ba = (u32)a; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_WINXOR_WN(4, 4)
DEFINE_WINXOR_WN(64, 64)
DEFINE_WINXOR_WN(128, 128)
DEFINE_WINXOR_WN(256, 256)
DEFINE_WINXOR_WN(512, 512)
DEFINE_WINXOR_WN(1024, 1024)
DEFINE_WINXOR_WN(2048, 2048)

#define PW_K2 2
static void ap_piecewise_k2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int chunk = n / PW_K2;
    for (int c = 0; c < PW_K2; c++) {
        u8 add = (u8)((amp >> (c * 8)) & 0xFF);
        int start = c * chunk, end = (c == PW_K2 - 1) ? n : start + chunk;
        for (int i = start; i < end; i++) d[i] = (u8)(d[i] + add);
    }
}
static void inv_piecewise_k2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int chunk = n / PW_K2;
    for (int c = 0; c < PW_K2; c++) {
        u8 add = (u8)((amp >> (c * 8)) & 0xFF);
        int start = c * chunk, end = (c == PW_K2 - 1) ? n : start + chunk;
        for (int i = start; i < end; i++) d[i] = (u8)(d[i] - add);
    }
}
static double search_piecewise_k2(const u8 *d, int n, double Sb, Instr *out) {
    int chunk = n / PW_K2;
    int total[256]; freq_of(d, n, total);
    u32 amp = 0;
    for (int c = 0; c < PW_K2; c++) {
        int start = c * chunk, end = (c == PW_K2 - 1) ? n : start + chunk;
        int hit[256] = {0};
        for (int i = start; i < end; i++) hit[d[i]]++;
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        double bestg = -1e18; int bc = 0;
        for (int a = 0; a < 256; a++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[(u + a) & 255] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; bc = a; }
        }
        amp |= (u32)bc << (c * 8);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_piecewise_k2(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

#define PW_K8 8
static void ap_piecewise_k8(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int chunk = n / PW_K8;
    for (int c = 0; c < PW_K8; c++) {
        u8 add = (u8)((amp >> (c * 4)) & 0xF);
        int start = c * chunk, end = (c == PW_K8 - 1) ? n : start + chunk;
        for (int i = start; i < end; i++) d[i] = (u8)(d[i] + add);
    }
}
static void inv_piecewise_k8(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int chunk = n / PW_K8;
    for (int c = 0; c < PW_K8; c++) {
        u8 add = (u8)((amp >> (c * 4)) & 0xF);
        int start = c * chunk, end = (c == PW_K8 - 1) ? n : start + chunk;
        for (int i = start; i < end; i++) d[i] = (u8)(d[i] - add);
    }
}
static double search_piecewise_k8(const u8 *d, int n, double Sb, Instr *out) {
    int chunk = n / PW_K8;
    int total[256]; freq_of(d, n, total);
    u32 amp = 0;
    for (int c = 0; c < PW_K8; c++) {
        int start = c * chunk, end = (c == PW_K8 - 1) ? n : start + chunk;
        int hit[256] = {0};
        for (int i = start; i < end; i++) hit[d[i]]++;
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
        double bestg = -1e18; int bc = 0;
        for (int a = 0; a < 16; a++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[(u + a) & 255] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; bc = a; }
        }
        amp |= (u32)bc << (c * 4);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_piecewise_k8(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(32.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

/* ---- PIECEWISE_K<N> family: same equal-chunk-additive mechanism as
 * PIECEWISE (K4)/PIECEWISE_K2/PIECEWISE_K8, filling in K=3,16 -- add-
 * value bit-width per chunk shrinks as K grows to keep K*NBITS<=32 (K3
 * gets a full 8-bit add each since 3*8=24 fits; K16 gets 2 bits each). */
#define DEFINE_PIECEWISE_KN(SUF, KVAL, NBITS) \
static void ap_piecewise_k##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    int chunk = n / (KVAL); \
    for (int c = 0; c < (KVAL); c++) { \
        u8 add = (u8)((amp >> (c * (NBITS))) & ((1u << (NBITS)) - 1)); \
        int start = c * chunk, end = (c == (KVAL) - 1) ? n : start + chunk; \
        for (int i = start; i < end; i++) d[i] = (u8)(d[i] + add); \
    } \
} \
static void inv_piecewise_k##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    int chunk = n / (KVAL); \
    for (int c = 0; c < (KVAL); c++) { \
        u8 add = (u8)((amp >> (c * (NBITS))) & ((1u << (NBITS)) - 1)); \
        int start = c * chunk, end = (c == (KVAL) - 1) ? n : start + chunk; \
        for (int i = start; i < end; i++) d[i] = (u8)(d[i] - add); \
    } \
} \
static double search_piecewise_k##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int chunk = n / (KVAL); \
    int total[256]; freq_of(d, n, total); \
    u32 amp = 0; \
    for (int c = 0; c < (KVAL); c++) { \
        int start = c * chunk, end = (c == (KVAL) - 1) ? n : start + chunk; \
        int hit[256] = {0}; \
        for (int i = start; i < end; i++) hit[d[i]]++; \
        int base[256]; \
        for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v]; \
        double bestg = -1e18; int bc = 0; \
        for (int a = 0; a < (1 << (NBITS)); a++) { \
            int rf[256]; memcpy(rf, base, sizeof rf); \
            for (int u = 0; u < 256; u++) rf[(u + a) & 255] += hit[u]; \
            double S = S_from_freq(rf); \
            if (S > bestg) { bestg = S; bc = a; } \
        } \
        amp |= (u32)bc << (c * (NBITS)); \
    } \
    static u8 scr[BLOCK]; \
    memcpy(scr, d, (size_t)n); \
    ap_piecewise_k##SUF(scr, n, 0, 0, amp); \
    double net = (S_of(scr, n) - Sb) - oh_flat((double)((KVAL) * (NBITS))); \
    out->stride = 0; out->phase = 0; out->amp = amp; \
    return net; \
}
DEFINE_PIECEWISE_KN(3, 3, 8)
DEFINE_PIECEWISE_KN(16, 16, 2)

#define VIG_L2 2
static void ap_vigenere_l2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 key[VIG_L2]; for (int k = 0; k < VIG_L2; k++) key[k] = (u8)((amp >> (k * 8)) & 0xFF);
    for (int i = 0; i < n; i++) d[i] ^= key[i % VIG_L2];
}
static double search_vigenere_l2(const u8 *d, int n, double Sb, Instr *out) {
    u32 amp = 0;
    for (int r = 0; r < VIG_L2; r++) {
        int hit[256] = {0}, tot[256] = {0};
        for (int i = 0; i < n; i++) { tot[d[i]]++; if (i % VIG_L2 == r) hit[d[i]]++; }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v];
        double bestg = -1e18; int bc = 0;
        for (int c = 0; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; bc = c; }
        }
        amp |= (u32)bc << (r * 8);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_vigenere_l2(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(16.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

#define VIG_L8 8
static void ap_vigenere_l8(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 key[VIG_L8]; for (int k = 0; k < VIG_L8; k++) key[k] = (u8)(((amp >> (k * 4)) & 0xF) << 4);
    for (int i = 0; i < n; i++) d[i] ^= key[i % VIG_L8];
}
static double search_vigenere_l8(const u8 *d, int n, double Sb, Instr *out) {
    u32 amp = 0;
    for (int r = 0; r < VIG_L8; r++) {
        int hit[256] = {0}, tot[256] = {0};
        for (int i = 0; i < n; i++) { tot[d[i]]++; if (i % VIG_L8 == r) hit[d[i]]++; }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v];
        double bestg = -1e18; int bc = 0;
        for (int cidx = 0; cidx < 16; cidx++) {
            u8 c = (u8)(cidx << 4);
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; bc = cidx; }
        }
        amp |= (u32)bc << (r * 4);
    }
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_vigenere_l8(scr, n, 0, 0, amp);
    double net = (S_of(scr, n) - Sb) - oh_flat(32.0);
    out->stride = 0; out->phase = 0; out->amp = amp;
    return net;
}

/* ---- VIGENERE_L<N> family: same repeating-key XOR mechanism as
 * VIGENERE (L4)/VIGENERE_L2/VIGENERE_L8, filling in L=3,16,32 -- key
 * bit-width per position shrinks as L grows to keep L*NBITS<=32 (L3 gets
 * a full 8-bit key each since 3*8=24 fits; L16 gets 2 bits each
 * shifted into the top of the byte; L32 gets 1 bit each). */
#define DEFINE_VIGENERE_LN(SUF, LVAL, NBITS) \
static void ap_vigenere_l##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 key[LVAL]; for (int k = 0; k < (LVAL); k++) key[k] = (u8)(((amp >> (k * (NBITS))) & ((1u << (NBITS)) - 1)) << (8 - (NBITS))); \
    for (int i = 0; i < n; i++) d[i] ^= key[i % (LVAL)]; \
} \
static double search_vigenere_l##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    u32 amp = 0; \
    for (int r = 0; r < (LVAL); r++) { \
        int hit[256] = {0}, tot[256] = {0}; \
        for (int i = 0; i < n; i++) { tot[d[i]]++; if (i % (LVAL) == r) hit[d[i]]++; } \
        int base[256]; \
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v]; \
        double bestg = -1e18; int bc = 0; \
        for (int cidx = 0; cidx < (1 << (NBITS)); cidx++) { \
            u8 c = (u8)(cidx << (8 - (NBITS))); \
            int rf[256]; memcpy(rf, base, sizeof rf); \
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u]; \
            double S = S_from_freq(rf); \
            if (S > bestg) { bestg = S; bc = cidx; } \
        } \
        amp |= (u32)bc << (r * (NBITS)); \
    } \
    static u8 scr[BLOCK]; \
    memcpy(scr, d, (size_t)n); \
    ap_vigenere_l##SUF(scr, n, 0, 0, amp); \
    double net = (S_of(scr, n) - Sb) - oh_flat((double)((LVAL) * (NBITS))); \
    out->stride = 0; out->phase = 0; out->amp = amp; \
    return net; \
}
DEFINE_VIGENERE_LN(3, 3, 8)
DEFINE_VIGENERE_LN(16, 16, 2)
DEFINE_VIGENERE_LN(32, 32, 1)

#define PAT2_L 2
static void ap_pattern2xor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 pat = (u8)(amp & 3); u8 c = (u8)((amp >> 2) & 0xFF);
    for (int i = 0; i < n; i++) if ((pat >> (i % PAT2_L)) & 1) d[i] ^= c;
}
static double search_pattern2xor(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(10.0);
    double best = -1e18; u32 ba = 0;
    for (int pat = 1; pat < 4; pat++) {
        int hit[256] = {0}, tot[256] = {0};
        for (int i = 0; i < n; i++) { tot[d[i]]++; if ((pat >> (i % PAT2_L)) & 1) hit[d[i]]++; }
        int base[256];
        for (int v = 0; v < 256; v++) base[v] = tot[v] - hit[v];
        for (int c = 1; c < 256; c++) {
            int rf[256]; memcpy(rf, base, sizeof rf);
            for (int u = 0; u < 256; u++) rf[u ^ c] += hit[u];
            double net = (S_from_freq(rf) - Sb) - oh;
            if (net > best) { best = net; ba = (u32)pat | ((u32)c << 2); }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

#define AFFINE_A5 5
static void ap_affine_a5(u8 *d, int n, int s, int p, u32 amp) {
    u8 b = (u8)amp;
    for (int i = p; i < n; i += s) d[i] = (u8)(AFFINE_A5 * d[i] + b);
}
static void inv_affine_a5(u8 *d, int n, int s, int p, u32 amp) {
    u8 b = (u8)amp;
    u8 ainv = mul_inv256(AFFINE_A5);
    for (int i = p; i < n; i += s) d[i] = (u8)((d[i] - b) * ainv);
}
static double search_affine_a5(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bb = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            int hitm[256] = {0};
            for (int u = 0; u < 256; u++) hitm[(AFFINE_A5 * u) & 0xFF] += hit[u];
            for (int b = 0; b < 256; b++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int w = 0; w < 256; w++) rf[(w + b) & 0xFF] += hitm[w];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bb = (u32)b; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = bb;
    return best;
}

#define AFFINE_A7 7
static void ap_affine_a7(u8 *d, int n, int s, int p, u32 amp) {
    u8 b = (u8)amp;
    for (int i = p; i < n; i += s) d[i] = (u8)(AFFINE_A7 * d[i] + b);
}
static void inv_affine_a7(u8 *d, int n, int s, int p, u32 amp) {
    u8 b = (u8)amp;
    u8 ainv = mul_inv256(AFFINE_A7);
    for (int i = p; i < n; i += s) d[i] = (u8)((d[i] - b) * ainv);
}
static double search_affine_a7(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 bb = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(8.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            int hitm[256] = {0};
            for (int u = 0; u < 256; u++) hitm[(AFFINE_A7 * u) & 0xFF] += hit[u];
            for (int b = 0; b < 256; b++) {
                int rf[256]; memcpy(rf, base, sizeof rf);
                for (int w = 0; w < 256; w++) rf[(w + b) & 0xFF] += hitm[w];
                double net = (S_from_freq(rf) - Sb) - oh;
                if (net > best) { best = net; bs = s; bp = p; bb = (u32)b; }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = bb;
    return best;
}

static u8 fixed_keystream2[BLOCK];
static void init_fixed_keystream2(void) {
    u16 st = 0xBEEF;
    for (int i = 0; i < BLOCK; i++) fixed_keystream2[i] = xs16_next(&st);
}
static void ap_fixedks2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i < n; i++) d[i] ^= fixed_keystream2[i];
}
static double search_fixedks2(const u8 *d, int n, double Sb, Instr *out) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[(u8)(d[i] ^ fixed_keystream2[i])]++;
    double net = (S_from_freq(f) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

static u8 fixed_keystream3[BLOCK];
static void init_fixed_keystream3(void) {
    u16 st = 0x1337;
    for (int i = 0; i < BLOCK; i++) fixed_keystream3[i] = xs16_next(&st);
}
static void ap_fixedks3(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i < n; i++) d[i] ^= fixed_keystream3[i];
}
static double search_fixedks3(const u8 *d, int n, double Sb, Instr *out) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[(u8)(d[i] ^ fixed_keystream3[i])]++;
    double net = (S_from_freq(f) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- FIXED_KS<N> family: same fixed-seed xorshift16 keystream mechanism
 * as FIXED_KS/KS2/KS3, just more distinct fixed seeds -- zero amp bits
 * (no searched parameter), so each is essentially free to test whether
 * that particular fixed pattern happens to correlate with the data. */
#define DEFINE_FIXEDKS_N(SUF, SEEDHEX) \
static u8 fixed_keystream##SUF[BLOCK]; \
static void init_fixed_keystream##SUF(void) { \
    u16 st = (SEEDHEX); \
    for (int i = 0; i < BLOCK; i++) fixed_keystream##SUF[i] = xs16_next(&st); \
} \
static void ap_fixedks##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; (void)amp; \
    for (int i = 0; i < n; i++) d[i] ^= fixed_keystream##SUF[i]; \
} \
static double search_fixedks##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    int f[256] = {0}; \
    for (int i = 0; i < n; i++) f[(u8)(d[i] ^ fixed_keystream##SUF[i])]++; \
    double net = (S_from_freq(f) - Sb) - oh_flat(0.0); \
    out->stride = 0; out->phase = 0; out->amp = 0; \
    return net; \
}
DEFINE_FIXEDKS_N(4, 0xACE1)
DEFINE_FIXEDKS_N(5, 0xF00D)
DEFINE_FIXEDKS_N(6, 0xCAFE)
DEFINE_FIXEDKS_N(7, 0xD00D)
DEFINE_FIXEDKS_N(8, 0x5EED)
DEFINE_FIXEDKS_N(9, 0x9E37)
DEFINE_FIXEDKS_N(10, 0x1234)

static const int DUALSHIFT_KSET2[4] = { 1, 5, 6, 7 };
static void ap_dualshift2(u8 *d, int n, int s, int p, u32 amp) {
    int k1 = DUALSHIFT_KSET2[amp & 3], k2 = DUALSHIFT_KSET2[(amp >> 2) & 3];
    for (int i = p; i < n; i += s) { u8 v = d[i]; d[i] = (u8)(v ^ (v >> k1) ^ (v >> k2)); }
}
static void inv_dualshift2(u8 *d, int n, int s, int p, u32 amp) {
    int k1 = DUALSHIFT_KSET2[amp & 3], k2 = DUALSHIFT_KSET2[(amp >> 2) & 3];
    u8 inv_tab[256];
    for (int v = 0; v < 256; v++) inv_tab[(u8)(v ^ (v >> k1) ^ (v >> k2))] = (u8)v;
    for (int i = p; i < n; i += s) d[i] = inv_tab[d[i]];
}
static double search_dualshift2(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int k1i = 0; k1i < 4; k1i++) {
                for (int k2i = 0; k2i < 4; k2i++) {
                    if (k1i == k2i) continue;
                    int k1 = DUALSHIFT_KSET2[k1i], k2 = DUALSHIFT_KSET2[k2i];
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) rf[(u8)(u ^ (u >> k1) ^ (u >> k2))] += hit[u];
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)k1i | ((u32)k2i << 2); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

#define REGION_K2 2
static void ap_regionalanchor_k2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    int rlen = n / REGION_K2;
    for (int r = 0; r < REGION_K2; r++) {
        int start = r * rlen, end = (r == REGION_K2 - 1) ? n : start + rlen;
        u8 c = gf_mul(d[start], a);
        for (int i = start + 1; i < end; i++) d[i] ^= c;
    }
}
static double search_regionalanchor_k2(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        memcpy(scr, d, (size_t)n);
        ap_regionalanchor_k2(scr, n, 0, 0, (u32)a);
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

#define REGION_K8 8
static void ap_regionalanchor_k8(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 a = (u8)amp;
    int rlen = n / REGION_K8;
    for (int r = 0; r < REGION_K8; r++) {
        int start = r * rlen, end = (r == REGION_K8 - 1) ? n : start + rlen;
        u8 c = gf_mul(d[start], a);
        for (int i = start + 1; i < end; i++) d[i] ^= c;
    }
}
static double search_regionalanchor_k8(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 1;
    static u8 scr[BLOCK];
    for (int a = 1; a < 256; a++) {
        memcpy(scr, d, (size_t)n);
        ap_regionalanchor_k8(scr, n, 0, 0, (u32)a);
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
        if (net > best) { best = net; ba = (u32)a; }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- REGIONAL_ANCHOR_K<N> family: same per-region self-anchor XOR
 * mechanism as REGIONAL_ANCHOR (K4)/K2/K8, filling in K=3,5,6,16 -- the
 * multiplier `a` is shared across all regions (8 bits total regardless
 * of K), so extending K is a pure region-count change. */
#define DEFINE_REGIONALANCHOR_KN(SUF, KVAL) \
static void ap_regionalanchor_k##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 a = (u8)amp; \
    int rlen = n / (KVAL); \
    for (int r = 0; r < (KVAL); r++) { \
        int start = r * rlen, end = (r == (KVAL) - 1) ? n : start + rlen; \
        u8 c = gf_mul(d[start], a); \
        for (int i = start + 1; i < end; i++) d[i] ^= c; \
    } \
} \
static double search_regionalanchor_k##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double best = -1e18; u32 ba = 1; \
    static u8 scr[BLOCK]; \
    for (int a = 1; a < 256; a++) { \
        memcpy(scr, d, (size_t)n); \
        ap_regionalanchor_k##SUF(scr, n, 0, 0, (u32)a); \
        double net = (S_of(scr, n) - Sb) - oh_flat(8.0); \
        if (net > best) { best = net; ba = (u32)a; } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_REGIONALANCHOR_KN(3, 3)
DEFINE_REGIONALANCHOR_KN(5, 5)
DEFINE_REGIONALANCHOR_KN(6, 6)
DEFINE_REGIONALANCHOR_KN(16, 16)

#define SPARSE_K1 1
static void ap_sparsexor_k1(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)n;
    u8 c = (u8)(amp & 0xFF);
    int pos0 = (int)((amp >> 8) & 0xFFF);
    d[pos0] ^= c;
}
static double search_sparsexor_k1(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; u32 ba = 0;
    for (int c = 1; c < 256; c++) {
        double bestg = -1e18; int bp0 = 0;
        for (int i0 = 0; i0 < n; i0++) {
            int rf[256]; memcpy(rf, total, sizeof rf);
            rf[d[i0]]--; rf[d[i0] ^ c]++;
            double S = S_from_freq(rf);
            if (S > bestg) { bestg = S; bp0 = i0; }
        }
        double net = (bestg - Sb) - oh_flat(20.0);
        if (net > best) { best = net; ba = (u32)c | ((u32)bp0 << 8); }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- SPARSE_XOR_K3: three-position explicit selection. 3 positions x
 * 12 bits + 8-bit constant = 44 bits, doesn't fit u32 -- reduced to one
 * searched anchor position plus two positions at fixed offsets (+1,+2)
 * from it, to keep amp within 32 bits while still testing a 3-position
 * hypothesis. */
#define SPARSE_K3 3
static void ap_sparsexor_k3fixed(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 c = (u8)(amp & 0xFF);
    int pos0 = (int)((amp >> 8) & 0xFFF) % n;
    int pos1 = (pos0 + 1) % n, pos2 = (pos0 + 2) % n;
    d[pos0] ^= c; d[pos1] ^= c; d[pos2] ^= c;
}
static double search_sparsexor_k3(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int c = 1; c < 256; c++) {
        for (int pos0 = 0; pos0 < n; pos0++) {
            memcpy(scr, d, (size_t)n);
            u32 amp = (u32)c | ((u32)pos0 << 8);
            ap_sparsexor_k3fixed(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh_flat(20.0);
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- SPARSE_XOR_K4 / SPARSE_XOR_K5: same fixed-offset-anchor mechanism
 * as SPARSE_XOR_K3, extended to 4 (resp. 5) positions -- anchor pos0 plus
 * fixed offsets +1..+3 (K4) or +1..+4 (K5) from it, all sharing the same
 * XOR constant. Still self-inverse regardless of how the fixed positions
 * are chosen (each byte gets the same constant XORed twice on a second
 * apply). */
#define SPARSE_K4 4
static void ap_sparsexor_k4fixed(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 c = (u8)(amp & 0xFF);
    int pos0 = (int)((amp >> 8) & 0xFFF) % n;
    int pos1 = (pos0 + 1) % n, pos2 = (pos0 + 2) % n, pos3 = (pos0 + 3) % n;
    d[pos0] ^= c; d[pos1] ^= c; d[pos2] ^= c; d[pos3] ^= c;
}
static double search_sparsexor_k4(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int c = 1; c < 256; c++) {
        for (int pos0 = 0; pos0 < n; pos0++) {
            memcpy(scr, d, (size_t)n);
            u32 amp = (u32)c | ((u32)pos0 << 8);
            ap_sparsexor_k4fixed(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh_flat(20.0);
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}
#define SPARSE_K5 5
static void ap_sparsexor_k5fixed(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u8 c = (u8)(amp & 0xFF);
    int pos0 = (int)((amp >> 8) & 0xFFF) % n;
    int pos1 = (pos0 + 1) % n, pos2 = (pos0 + 2) % n, pos3 = (pos0 + 3) % n, pos4 = (pos0 + 4) % n;
    d[pos0] ^= c; d[pos1] ^= c; d[pos2] ^= c; d[pos3] ^= c; d[pos4] ^= c;
}
static double search_sparsexor_k5(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int c = 1; c < 256; c++) {
        for (int pos0 = 0; pos0 < n; pos0++) {
            memcpy(scr, d, (size_t)n);
            u32 amp = (u32)c | ((u32)pos0 << 8);
            ap_sparsexor_k5fixed(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh_flat(20.0);
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- SPARSE_XOR_K<N> family: same fixed-offset-anchor mechanism as
 * SPARSE_XOR_K1/K3/K4/K5, filling in K=2,6,7,8 -- anchor pos0 plus fixed
 * offsets +1..+(N-1) from it, all sharing the same XOR constant. Self-
 * inverse regardless of N or the fixed offsets chosen. */
#define DEFINE_SPARSE_XOR_KN(SUF, KCOUNT) \
static void ap_sparsexor_k##SUF(u8 *d, int n, int s, int p, u32 amp) { \
    (void)s; (void)p; \
    u8 c = (u8)(amp & 0xFF); \
    int pos0 = (int)((amp >> 8) & 0xFFF) % n; \
    for (int k = 0; k < (KCOUNT); k++) d[(pos0 + k) % n] ^= c; \
} \
static double search_sparsexor_k##SUF(const u8 *d, int n, double Sb, Instr *out) { \
    double best = -1e18; u32 ba = 0; \
    static u8 scr[BLOCK]; \
    for (int c = 1; c < 256; c++) { \
        for (int pos0 = 0; pos0 < n; pos0++) { \
            memcpy(scr, d, (size_t)n); \
            u32 amp = (u32)c | ((u32)pos0 << 8); \
            ap_sparsexor_k##SUF(scr, n, 0, 0, amp); \
            double net = (S_of(scr, n) - Sb) - oh_flat(20.0); \
            if (net > best) { best = net; ba = amp; } \
        } \
    } \
    out->stride = 0; out->phase = 0; out->amp = ba; \
    return best; \
}
DEFINE_SPARSE_XOR_KN(2, 2)
DEFINE_SPARSE_XOR_KN(6, 6)
DEFINE_SPARSE_XOR_KN(7, 7)
DEFINE_SPARSE_XOR_KN(8, 8)

/* ---- LINPRED2_WIDE: same mechanism as LINPRED2, but coefficients range
 * -8..7 (4 bits each) instead of -4..3 (3 bits each) -- reaches steeper
 * predictor slopes at the cost of a slightly larger amp field. */
static void ap_linpred2wide(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int a = (int)(amp & 0xF) - 8, b = (int)((amp >> 4) & 0xF) - 8;
    for (int i = n - 1; i >= 2; i--) {
        int pred = (a * (int)d[i - 1] + b * (int)d[i - 2]) >> 2;
        d[i] = (u8)((int)d[i] - pred);
    }
}
static void inv_linpred2wide(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int a = (int)(amp & 0xF) - 8, b = (int)((amp >> 4) & 0xF) - 8;
    for (int i = 2; i < n; i++) {
        int pred = (a * (int)d[i - 1] + b * (int)d[i - 2]) >> 2;
        d[i] = (u8)((int)d[i] + pred);
    }
}
static double search_linpred2wide(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 ba = 0;
    static u8 scr[BLOCK];
    for (int aidx = 0; aidx < 16; aidx++) {
        for (int bidx = 0; bidx < 16; bidx++) {
            memcpy(scr, d, (size_t)n);
            u32 amp = (u32)aidx | ((u32)bidx << 4);
            ap_linpred2wide(scr, n, 0, 0, amp);
            double net = (S_of(scr, n) - Sb) - oh_flat(8.0);
            if (net > best) { best = net; ba = amp; }
        }
    }
    out->stride = 0; out->phase = 0; out->amp = ba;
    return best;
}

/* ---- ADAPT_LMS_SHIFT2 / ADAPT_LMS_SHIFT5: same sign-sign LMS mechanism
 * as ADAPT_LMS/ADAPT_LMS_FAST, at prediction shifts 2 and 5 -- rounds
 * out the shift range (2,3,4,5 now all covered by a distinct instruction
 * each). */
static void ap_lmsshift2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w = (int)amp - 8;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)((w * (int)prev) >> 2);
        u8 residual = (u8)(orig_i - pred);
        d[i] = residual;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w += sgn; if (w > 7) w = 7; if (w < -8) w = -8;
        prev = orig_i;
    }
}
static void inv_lmsshift2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w = (int)amp - 8;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 residual = d[i];
        u8 pred = (u8)((w * (int)prev) >> 2);
        u8 orig_i = (u8)(residual + pred);
        d[i] = orig_i;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w += sgn; if (w > 7) w = 7; if (w < -8) w = -8;
        prev = orig_i;
    }
}
static double search_lmsshift2(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bw0 = 8;
    static u8 scr[BLOCK];
    for (int w0 = 0; w0 < 16; w0++) {
        memcpy(scr, d, (size_t)n);
        ap_lmsshift2(scr, n, 0, 0, (u32)w0);
        double net = (S_of(scr, n) - Sb) - oh_flat(4.0);
        if (net > best) { best = net; bw0 = (u32)w0; }
    }
    out->stride = 0; out->phase = 0; out->amp = bw0;
    return best;
}
static void ap_lmsshift5(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w = (int)amp - 8;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 orig_i = d[i];
        u8 pred = (u8)((w * (int)prev) >> 5);
        u8 residual = (u8)(orig_i - pred);
        d[i] = residual;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w += sgn; if (w > 7) w = 7; if (w < -8) w = -8;
        prev = orig_i;
    }
}
static void inv_lmsshift5(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    int w = (int)amp - 8;
    u8 prev = d[0];
    for (int i = 1; i < n; i++) {
        u8 residual = d[i];
        u8 pred = (u8)((w * (int)prev) >> 5);
        u8 orig_i = (u8)(residual + pred);
        d[i] = orig_i;
        int r = (int)residual;
        int sgn = (r == 0) ? 0 : (r < 128 ? 1 : -1);
        w += sgn; if (w > 7) w = 7; if (w < -8) w = -8;
        prev = orig_i;
    }
}
static double search_lmsshift5(const u8 *d, int n, double Sb, Instr *out) {
    double best = -1e18; u32 bw0 = 8;
    static u8 scr[BLOCK];
    for (int w0 = 0; w0 < 16; w0++) {
        memcpy(scr, d, (size_t)n);
        ap_lmsshift5(scr, n, 0, 0, (u32)w0);
        double net = (S_of(scr, n) - Sb) - oh_flat(4.0);
        if (net > best) { best = net; bw0 = (u32)w0; }
    }
    out->stride = 0; out->phase = 0; out->amp = bw0;
    return best;
}

/* ---- PRNG_ADD2 / PRNG_XOR2: 2-stream sibling of PRNG_ADD4/XOR4, stream
 * picked by pos%2 -- coarser interleave than the 4-stream version. */
static void ap_prngadd2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[2]; st[0] = ms; st[1] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + xs16_next(&st[i & 1]));
}
static void inv_prngadd2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[2]; st[0] = ms; st[1] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - xs16_next(&st[i & 1]));
}
static double search_prngadd2(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[2]; s[0] = ms; s[1] = xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + xs16_next(&s[i & 1]))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}
static void ap_prngxor2(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[2]; st[0] = ms; st[1] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] ^= xs16_next(&st[i & 1]);
}
static double search_prngxor2(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[2]; s[0] = ms; s[1] = xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ xs16_next(&s[i & 1]))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- PRNG_ADD16 / PRNG_XOR16: 16-stream sibling, stream picked by
 * pos%16 -- finer interleave than the 8-stream version. */
static void ap_prngadd16(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[16]; st[0] = ms; for (int k = 1; k < 16; k++) st[k] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] + xs16_next(&st[i & 15]));
}
static void inv_prngadd16(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[16]; st[0] = ms; for (int k = 1; k < 16; k++) st[k] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] = (u8)(d[i] - xs16_next(&st[i & 15]));
}
static double search_prngadd16(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[16]; s[0] = ms; for (int k = 1; k < 16; k++) s[k] = xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] + xs16_next(&s[i & 15]))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}
static void ap_prngxor16(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p;
    u16 ms = (u16)(amp & 0xFFFF);
    u16 st[16]; st[0] = ms; for (int k = 1; k < 16; k++) st[k] = xs16_next(&ms);
    for (int i = 0; i < n; i++) d[i] ^= xs16_next(&st[i & 15]);
}
static double search_prngxor16(const u8 *d, int n, double Sb, Instr *out) {
    double oh = oh_flat(16.0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < PRNG_SEEDS; seed++) {
        u16 ms = (u16)seed;
        u16 s[16]; s[0] = ms; for (int k = 1; k < 16; k++) s[k] = xs16_next(&ms);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ xs16_next(&s[i & 15]))]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    out->stride = 0; out->phase = 0; out->amp = bseed;
    return best;
}

/* ---- BLOCKDIFF2_ALT: same mechanism as BLOCKDIFF2, DIFFERENT fixed 2x2
 * GF(256) matrix (m=[[3,2],[1,3]] instead of [[2,1],[1,1]]) -- a
 * genuinely different fixed diffusion, not just a relabeling (det
 * differs: gf_mul(3,3)^gf_mul(2,1) vs the original's gf_mul(2,1)^
 * gf_mul(1,1), both nonzero/valid but distinct mixing). */
#define BD2_M00 3
#define BD2_M01 2
#define BD2_M10 1
#define BD2_M11 3
static u8 bd2_inv00, bd2_inv01, bd2_inv10, bd2_inv11;
static void init_blockdiff2alt(void) {
    u8 det = (u8)(gf_mul(BD2_M00, BD2_M11) ^ gf_mul(BD2_M01, BD2_M10));
    u8 di = gf_inv(det);
    bd2_inv00 = gf_mul(di, BD2_M11); bd2_inv01 = gf_mul(di, BD2_M01);
    bd2_inv10 = gf_mul(di, BD2_M10); bd2_inv11 = gf_mul(di, BD2_M00);
}
static void ap_blockdiff2alt(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i + 1 < n; i += 2) {
        u8 b0 = d[i], b1 = d[i + 1];
        d[i]     = (u8)(gf_mul(BD2_M00, b0) ^ gf_mul(BD2_M01, b1));
        d[i + 1] = (u8)(gf_mul(BD2_M10, b0) ^ gf_mul(BD2_M11, b1));
    }
}
static void inv_blockdiff2alt(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int i = 0; i + 1 < n; i += 2) {
        u8 b0 = d[i], b1 = d[i + 1];
        d[i]     = (u8)(gf_mul(bd2_inv00, b0) ^ gf_mul(bd2_inv01, b1));
        d[i + 1] = (u8)(gf_mul(bd2_inv10, b0) ^ gf_mul(bd2_inv11, b1));
    }
}
static double search_blockdiff2alt(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_blockdiff2alt(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- HADAMARD4_ALT: same 2-stage butterfly structure as HADAMARD4, but
 * reusing BLOCKDIFF2_ALT's different fixed matrix instead. */
static void hadamard4alt_stage(u8 *a, u8 *b) {
    u8 na = (u8)(gf_mul(BD2_M00, *a) ^ gf_mul(BD2_M01, *b));
    u8 nb = (u8)(gf_mul(BD2_M10, *a) ^ gf_mul(BD2_M11, *b));
    *a = na; *b = nb;
}
static void hadamard4alt_stage_inv(u8 *a, u8 *b) {
    u8 na = (u8)(gf_mul(bd2_inv00, *a) ^ gf_mul(bd2_inv01, *b));
    u8 nb = (u8)(gf_mul(bd2_inv10, *a) ^ gf_mul(bd2_inv11, *b));
    *a = na; *b = nb;
}
static void ap_hadamard4alt(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int base = 0; base + 4 <= n; base += 4) {
        hadamard4alt_stage(&d[base + 0], &d[base + 1]);
        hadamard4alt_stage(&d[base + 2], &d[base + 3]);
        hadamard4alt_stage(&d[base + 0], &d[base + 2]);
        hadamard4alt_stage(&d[base + 1], &d[base + 3]);
    }
}
static void inv_hadamard4alt(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    for (int base = 0; base + 4 <= n; base += 4) {
        hadamard4alt_stage_inv(&d[base + 0], &d[base + 2]);
        hadamard4alt_stage_inv(&d[base + 1], &d[base + 3]);
        hadamard4alt_stage_inv(&d[base + 0], &d[base + 1]);
        hadamard4alt_stage_inv(&d[base + 2], &d[base + 3]);
    }
}
static double search_hadamard4alt(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_hadamard4alt(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ---- LATIN_CROSS2: cross-crumb operator using a DIFFERENT 4x4 Latin
 * square than LATIN_CROSS (this one derived from GF(4) addition rather
 * than an arbitrary non-group table). */
static const u8 latin4b[4][4] = {
    { 0, 1, 2, 3 },
    { 1, 0, 3, 2 },
    { 2, 3, 0, 1 },
    { 3, 2, 1, 0 },
};
static u8 latin4b_inv[4][4];
static void init_latin4b(void) {
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 4; k++)
            latin4b_inv[latin4b[k][j]][j] = (u8)k;
}
static void ap_latincross2(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3);
        u8 ck = (u8)((v >> (2 * k)) & 3);
        u8 nck = latin4b[ck][cj];
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (nck << (2 * k)));
    }
}
static void inv_latincross2(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3);
        u8 nck = (u8)((v >> (2 * k)) & 3);
        u8 ck = latin4b_inv[nck][cj];
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (ck << (2 * k)));
    }
}
static double search_latincross2(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 cj = (u8)((u >> (2 * j)) & 3);
                        u8 ck = (u8)((u >> (2 * k)) & 3);
                        u8 w = (u8)((u & ~(u8)(3 << (2 * k))) | (latin4b[ck][cj] << (2 * k)));
                        rf[w] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)j | ((u32)k << 2); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- LATIN_CROSS3: same cross-crumb mechanism as LATIN_CROSS/
 * LATIN_CROSS2, but a THIRD Latin square -- the cyclic Z4 addition
 * table {0,1,2,3}/{1,2,3,0}/{2,3,0,1}/{3,0,1,2}, a genuinely different
 * (fully cyclic, no fixed points off the diagonal) structure than
 * either of the other two. */
static const u8 latin4c[4][4] = {
    { 0, 1, 2, 3 },
    { 1, 2, 3, 0 },
    { 2, 3, 0, 1 },
    { 3, 0, 1, 2 },
};
static u8 latin4c_inv[4][4];
static void init_latin4c(void) {
    for (int j = 0; j < 4; j++)
        for (int k = 0; k < 4; k++)
            latin4c_inv[latin4c[k][j]][j] = (u8)k;
}
static void ap_latincross3(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3);
        u8 ck = (u8)((v >> (2 * k)) & 3);
        u8 nck = latin4c[ck][cj];
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (nck << (2 * k)));
    }
}
static void inv_latincross3(u8 *d, int n, int s, int p, u32 amp) {
    int j = (int)(amp & 3), k = (int)((amp >> 2) & 3);
    for (int i = p; i < n; i += s) {
        u8 v = d[i];
        u8 cj = (u8)((v >> (2 * j)) & 3);
        u8 nck = (u8)((v >> (2 * k)) & 3);
        u8 ck = latin4c_inv[nck][cj];
        d[i] = (u8)((v & ~(u8)(3 << (2 * k))) | (ck << (2 * k)));
    }
}
static double search_latincross3(const u8 *d, int n, double Sb, Instr *out) {
    int total[256]; freq_of(d, n, total);
    double best = -1e18; int bs = 1, bp = 0; u32 ba = 0;
    for (int s = 1; s <= MAX_STRIDE; s++) {
        double oh = oh_strided(4.0, s);
        for (int p = 0; p < s; p++) {
            int hit[256] = {0};
            for (int i = p; i < n; i += s) hit[d[i]]++;
            int base[256];
            for (int v = 0; v < 256; v++) base[v] = total[v] - hit[v];
            for (int j = 0; j < 4; j++) {
                for (int k = 0; k < 4; k++) {
                    if (j == k) continue;
                    int rf[256]; memcpy(rf, base, sizeof rf);
                    for (int u = 0; u < 256; u++) {
                        u8 cj = (u8)((u >> (2 * j)) & 3);
                        u8 ck = (u8)((u >> (2 * k)) & 3);
                        u8 w = (u8)((u & ~(u8)(3 << (2 * k))) | (latin4c[ck][cj] << (2 * k)));
                        rf[w] += hit[u];
                    }
                    double net = (S_from_freq(rf) - Sb) - oh;
                    if (net > best) { best = net; bs = s; bp = p; ba = (u32)j | ((u32)k << 2); }
                }
            }
        }
    }
    out->stride = bs; out->phase = bp; out->amp = ba;
    return best;
}

/* ---- CTXTABLE2_XOR: XOR-sibling of CTXTABLE2 -- 2-byte context table,
 * XOR combine instead of subtract. */
static void ap_ctxtable2xor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 table[65536];
    memset(table, 0, sizeof table);
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u8 orig = d[i];
        u16 ctx = (u16)(((u16)p2 << 8) | p1);
        u8 pred = table[ctx];
        d[i] = (u8)(orig ^ pred);
        table[ctx] = orig;
        p2 = p1; p1 = orig;
    }
}
static void inv_ctxtable2xor(u8 *d, int n, int s, int p, u32 amp) {
    (void)s; (void)p; (void)amp;
    static u8 table[65536];
    memset(table, 0, sizeof table);
    u8 p1 = d[0], p2 = 0;
    for (int i = 1; i < n; i++) {
        u16 ctx = (u16)(((u16)p2 << 8) | p1);
        u8 pred = table[ctx];
        u8 orig = (u8)(d[i] ^ pred);
        d[i] = orig;
        table[ctx] = orig;
        p2 = p1; p1 = orig;
    }
}
static double search_ctxtable2xor(const u8 *d, int n, double Sb, Instr *out) {
    static u8 scr[BLOCK];
    memcpy(scr, d, (size_t)n);
    ap_ctxtable2xor(scr, n, 0, 0, 0);
    double net = (S_of(scr, n) - Sb) - oh_flat(0.0);
    out->stride = 0; out->phase = 0; out->amp = 0;
    return net;
}

/* ============================================================ *
 *  registry -- add a new instruction by appending one line here *
 * ============================================================ */

typedef double (*SearchFn)(const u8 *, int, double, Instr *);
typedef void   (*XformFn)(u8 *, int, int, int, u32);

typedef struct { const char *name; SearchFn search; XformFn apply; XformFn invert; } InstrDesc;

static const InstrDesc REGISTRY[] = {
    { "XOR_PHASE",   search_xorp,       ap_xorp,       ap_xorp        }, /* self-inverse */
    { "STRIDE_ADD",  search_strideadd,  ap_strideadd,  inv_strideadd  },
    { "PRNG_ADD4",   search_prngadd4,   ap_prngadd4,   inv_prngadd4   },
    { "PRNG_XOR4",   search_prngxor4,   ap_prngxor4,   ap_prngxor4    }, /* self-inverse */
    { "VALUE_XOR",   search_valuexor,   ap_valuexor,   ap_valuexor    }, /* self-inverse */
    { "POPCNT_XOR",  search_popcxor,    ap_popcxor,    ap_popcxor     }, /* self-inverse */
    { "NIB_CXOR",    search_nibcxor,    ap_nibcxor,    ap_nibcxor     }, /* self-inverse */
    { "NIB_CADD",    search_nibcadd,    ap_nibcadd,    inv_nibcadd    },
    { "SELFSHIFT",   search_selfshift,  ap_selfshift,  inv_selfshift  },
    { "BIT_MAJ3",    search_bitmaj3,    ap_bitmaj3,    ap_bitmaj3     }, /* self-inverse */
    { "DELTA",       search_delta,      ap_delta,      inv_delta      },
    { "WINDOW_XOR",  search_winxor,     ap_winxor,     ap_winxor      }, /* self-inverse */
    { "RUNPARITY",   search_runparity,  ap_runparity,  inv_runparity  },
    { "ADAPT_LMS",   search_lms,        ap_lms,        inv_lms        },
    { "WORD_ADD16",  search_wordadd,    ap_wordadd,    inv_wordadd    },
    { "FEISTEL_HLF", search_feistel,    ap_feistel,    ap_feistel     }, /* self-inverse */
    { "PATTERN_XOR", search_patxor,     ap_patxor,     ap_patxor      }, /* self-inverse */
    { "VALMAP4_XOR", search_valuemap4,  ap_valuemap4,  ap_valuemap4   }, /* self-inverse */
    { "RANGE_XOR",   search_rangexor,   ap_rangexor,   ap_rangexor    }, /* self-inverse */
    { "MAGCLASS_XOR",search_magclass,   ap_magclass,   ap_magclass    }, /* self-inverse */
    { "CRMB_CXOR",   search_crmbcxor,   ap_crmbcxor,   ap_crmbcxor    }, /* self-inverse */
    { "BIT_CXOR",    search_bitcxor,    ap_bitcxor,    ap_bitcxor     }, /* self-inverse */
    { "LINPRED2",    search_linpred2,   ap_linpred2,   inv_linpred2   },
    { "MIRROR_XOR",  search_mirrorxor,  ap_mirrorxor,  ap_mirrorxor   }, /* self-inverse */
    { "ERRDIFF_XOR", search_errdiff,    ap_errdiff,    inv_errdiff    },
    { "DETREND",     search_detrend,    ap_detrend,    inv_detrend    },
    { "SINTREND",    search_sintrend,   ap_sintrend,   inv_sintrend   },
    { "PRNG_BIT",    search_prngbit,    ap_prngbit,    ap_prngbit     }, /* self-inverse */
    { "PRNG_SUBSET", search_prngsubset, ap_prngsubset, ap_prngsubset  }, /* self-inverse */
    { "HASH_XOR",    search_hashxor,    ap_hashxor,    ap_hashxor     }, /* self-inverse */
    { "EMA_BIAS",    search_emabias,    ap_emabias,    inv_emabias    },
    { "BITPLANE_XOR",search_bitplane,   ap_bitplane,   ap_bitplane    }, /* self-inverse */
    { "GF_POW",      search_gfpow,      ap_gfpow,      inv_gfpow      },
    { "AFFINE",      search_affine,     ap_affine,     inv_affine     },
    { "RANK_XOR",    search_rankxor,    ap_rankxor,    ap_rankxor     }, /* self-inverse */
    { "PRNG_PERM",   search_prngperm,   ap_prngperm,   inv_prngperm   },
    { "QR_XOR",      search_qrxor,      ap_qrxor,      ap_qrxor       }, /* self-inverse */
    { "GF2_LINEAR",  search_gf2linear,  ap_gf2linear,  inv_gf2linear  },
    { "LATIN_CROSS", search_latincross, ap_latincross, inv_latincross },
    { "WORD_GFMUL",  search_wordgfmul,  ap_wordgfmul,  inv_wordgfmul  },
    { "PRIME257",    search_prime257,   ap_prime257,   inv_prime257   },
    { "BITREV_IDX",  search_bitrevidx,  ap_bitrevidx,  ap_bitrevidx   }, /* self-inverse */
    { "BITREV_STRIDE", search_bitrevstride, ap_bitrevstride, ap_bitrevstride }, /* self-inverse */
    { "LINPRED3",    search_linpred3,   ap_linpred3,   inv_linpred3   },
    { "REGIONAL_ANCHOR", search_regionalanchor, ap_regionalanchor, ap_regionalanchor }, /* self-inverse */
    { "NIB_SWAP_STRIDE", search_nibswapstride, ap_nibswapstride, ap_nibswapstride }, /* self-inverse */
    { "BIT_ASWAP",   search_bitaswap,   ap_bitaswap,   ap_bitaswap    }, /* self-inverse */
    { "CRMB_GFMUL",  search_crmbgfmul,  ap_crmbgfmul,  inv_crmbgfmul  },
    { "NIB_GFMUL",   search_nibgfmul,   ap_nibgfmul,   inv_nibgfmul   },
    { "VALUE_GFMUL", search_valuegfmul, ap_valuegfmul, inv_valuegfmul },
    { "NIB_POW",     search_nibpow,     ap_nibpow,     inv_nibpow     },
    { "VALUE_ADD",   search_valueadd,   ap_valueadd,   inv_valueadd   },
    { "VALUE_MUL",   search_valuemul,   ap_valuemul,   inv_valuemul   },
    { "VALMAP4ADD",  search_valuemap4add, ap_valuemap4add, inv_valuemap4add },
    { "VALMAP4MUL",  search_valuemap4mul, ap_valuemap4mul, inv_valuemap4mul },
    { "CRMB_CADD",   search_crmbcadd,   ap_crmbcadd,   inv_crmbcadd   },
    { "CRMB_IADD",   search_crmbiadd,   ap_crmbiadd,   inv_crmbiadd   },
    { "NIB_MUL16",   search_nibmul16,   ap_nibmul16,   inv_nibmul16   },
    { "NIBXGFMUL",   search_nibcrossgfmul, ap_nibcrossgfmul, inv_nibcrossgfmul },
    { "CRMBXMUL",    search_crmbcrossmul, ap_crmbcrossmul, inv_crmbcrossmul },
    { "PRNG_ADD8",   search_prngadd8,   ap_prngadd8,   inv_prngadd8   },
    { "PRNG_XOR8",   search_prngxor8,   ap_prngxor8,   ap_prngxor8    }, /* self-inverse */
    { "BIT_SWAP2",   search_bitswap2,   ap_bitswap2,   ap_bitswap2    }, /* self-inverse */
    { "VALUE_GFPOW", search_valuegfpow, ap_valuegfpow, inv_valuegfpow },
    { "VM4GFMUL",    search_valuemap4gfmul, ap_valuemap4gfmul, inv_valuemap4gfmul },
    { "PATTERN_ADD", search_patadd,     ap_patadd,     inv_patadd     },
    { "DELTA_XOR",   search_deltaxor,   ap_deltaxor,   inv_deltaxor   },
    { "DIAG_ADD",    search_diagadd,    ap_diagadd,    inv_diagadd    },
    { "MIRROR_ADD",  search_mirroradd,  ap_mirroradd,  inv_mirroradd  },
    { "FEISTEL_ADD", search_feisteladd, ap_feisteladd, inv_feisteladd },
    { "PATTERN4_XOR", search_pattern4xor, ap_pattern4xor, ap_pattern4xor }, /* self-inverse */
    { "DUALSHIFT_XOR", search_dualshift, ap_dualshift, inv_dualshift },
    { "PATTERN16_XOR", search_pattern16xor, ap_pattern16xor, ap_pattern16xor }, /* self-inverse */
    { "CRMB_POW",    search_crmbpow,    ap_crmbpow,    inv_crmbpow    },
    { "VALUE_GFMULHI", search_valuegfmulhi, ap_valuegfmulhi, inv_valuegfmulhi },
    { "WINDOW_ADD",  search_windowadd,  ap_windowadd,  inv_windowadd  },
    { "SYNDROME_ADD", search_syndromeadd, ap_syndromeadd, inv_syndromeadd },
    { "FEISTEL_QTR", search_feistelquarter, ap_feistelquarter, ap_feistelquarter }, /* self-inverse */
    { "BIT_MAJ5",    search_bitmaj5,    ap_bitmaj5,    ap_bitmaj5     }, /* self-inverse */
    { "POPCNT_ADD",  search_popcntadd,  ap_popcntadd,  inv_popcntadd  },
    { "RANGE_ADD",   search_rangeadd,   ap_rangeadd,   inv_rangeadd   },
    { "MAGCLASS_ADD", search_magclassadd, ap_magclassadd, inv_magclassadd },
    { "QR_ADD",      search_qradd,      ap_qradd,      inv_qradd      },
    { "RANK_ADD",    search_rankadd,    ap_rankadd,    inv_rankadd    },
    { "VALUE_MULLO", search_valuemullo, ap_valuemullo, inv_valuemullo },
    { "BITREV_IDX_ADD", search_bitrevidxadd, ap_bitrevidxadd, inv_bitrevidxadd },
    { "DELTA_LONGLAG", search_deltalonglag, ap_deltalonglag, inv_deltalonglag },
    { "HASH_ADD",    search_hashadd,    ap_hashadd,    inv_hashadd    },
    { "MIRROR_MUL",  search_mirrormul,  ap_mirrormul,  inv_mirrormul  },
    { "PRNG_GFMUL4", search_prnggfmul4, ap_prnggfmul4, inv_prnggfmul4 },
    { "RUNPARITY_MUL", search_runparitymul, ap_runparitymul, inv_runparitymul },
    { "NIB_BITROT",  search_nibbitrot,  ap_nibbitrot,  inv_nibbitrot  },
    { "AVG2_PRED",   search_avg2pred,   ap_avg2pred,   inv_avg2pred   },
    { "MIN2_PRED",   search_min2pred,   ap_min2pred,   inv_min2pred   },
    { "MAX2_PRED",   search_max2pred,   ap_max2pred,   inv_max2pred   },
    { "AND_PRED",    search_andpred,    ap_andpred,    inv_andpred    },
    { "OR_PRED",     search_orpred,     ap_orpred,     inv_orpred     },
    { "MIN4_PRED",   search_min4pred,   ap_min4pred,   inv_min4pred   },
    { "MAX4_PRED",   search_max4pred,   ap_max4pred,   inv_max4pred   },
    { "MEDIAN3_PRED", search_median3pred, ap_median3pred, inv_median3pred },
    { "MEDIAN5_PRED", search_median5pred, ap_median5pred, inv_median5pred },
    { "MOMENTUM2_PRED", search_momentum2pred, ap_momentum2pred, inv_momentum2pred },
    { "WAVG_PRED",   search_wavgpred,   ap_wavgpred,   inv_wavgpred   },
    { "BLOCK_BITSHIFT", search_blockbitshift, ap_blockbitshift, inv_blockbitshift },
    { "MED2D_PRED",  search_med2d,      ap_med2d,      inv_med2d      },
    { "DUAL_LAG_XOR", search_duallagxor, ap_duallagxor, inv_duallagxor },
    { "SHIFT1_PRED", search_shift1pred, ap_shift1pred, inv_shift1pred },
    { "HALFMOMENTUM_PRED", search_halfmomentumpred, ap_halfmomentumpred, inv_halfmomentumpred },
    { "KALMAN_PRED", search_kalmanpred, ap_kalmanpred, inv_kalmanpred },
    { "NIB_DEINTERLEAVE", search_nibdeinterleave, ap_nibdeinterleave, ap_nibdeinterleave }, /* self-inverse */
    { "POPCNT_XOR_T2", search_popcxor_t2, ap_popcxor_t2, ap_popcxor_t2 }, /* self-inverse */
    { "POPCNT_XOR_T3", search_popcxor_t3, ap_popcxor_t3, ap_popcxor_t3 }, /* self-inverse */
    { "RANGE_XOR_T2", search_rangexor_t2, ap_rangexor_t2, ap_rangexor_t2 }, /* self-inverse */
    { "RANGE_XOR_T3", search_rangexor_t3, ap_rangexor_t3, ap_rangexor_t3 }, /* self-inverse */
    { "MAGCLASS_XOR_T2", search_magclass_t2, ap_magclass_t2, ap_magclass_t2 }, /* self-inverse */
    { "MAGCLASS_XOR_T3", search_magclass_t3, ap_magclass_t3, ap_magclass_t3 }, /* self-inverse */
    { "RANK_XOR_K2", search_rankxor_k2, ap_rankxor_k2, ap_rankxor_k2 }, /* self-inverse */
    { "RANK_XOR_K3", search_rankxor_k3, ap_rankxor_k3, ap_rankxor_k3 }, /* self-inverse */
    { "WINDOW_XOR_W8", search_winxor_w8, ap_winxor_w8, ap_winxor_w8 }, /* self-inverse */
    { "WINDOW_XOR_W32", search_winxor_w32, ap_winxor_w32, ap_winxor_w32 }, /* self-inverse */
    { "PIECEWISE_K2", search_piecewise_k2, ap_piecewise_k2, inv_piecewise_k2 },
    { "PATTERN2_XOR", search_pattern2xor, ap_pattern2xor, ap_pattern2xor }, /* self-inverse */
    { "AFFINE_A5",   search_affine_a5,  ap_affine_a5,  inv_affine_a5  },
    { "AFFINE_A7",   search_affine_a7,  ap_affine_a7,  inv_affine_a7  },
    { "FIXED_KS3",   search_fixedks3,   ap_fixedks3,   ap_fixedks3    }, /* self-inverse */
    { "DUALSHIFT_XOR2", search_dualshift2, ap_dualshift2, inv_dualshift2 },
    { "REGIONAL_ANCHOR_K2", search_regionalanchor_k2, ap_regionalanchor_k2, ap_regionalanchor_k2 }, /* self-inverse */
    { "REGIONAL_ANCHOR_K8", search_regionalanchor_k8, ap_regionalanchor_k8, ap_regionalanchor_k8 }, /* self-inverse */
    { "LINPRED2_WIDE", search_linpred2wide, ap_linpred2wide, inv_linpred2wide },
    { "PRNG_ADD2",   search_prngadd2,   ap_prngadd2,   inv_prngadd2   },
    { "PRNG_XOR2",   search_prngxor2,   ap_prngxor2,   ap_prngxor2    }, /* self-inverse */
    { "PRNG_ADD16",  search_prngadd16,  ap_prngadd16,  inv_prngadd16  },
    { "PRNG_XOR16",  search_prngxor16,  ap_prngxor16,  ap_prngxor16   }, /* self-inverse */
    { "LATIN_CROSS2", search_latincross2, ap_latincross2, inv_latincross2 },
    { "VALUE_GFMUL_B3", search_valuegfmul_b3, ap_valuegfmul_b3, inv_valuegfmul_b3 },
    { "VALUE_GFMUL_B4", search_valuegfmul_b4, ap_valuegfmul_b4, inv_valuegfmul_b4 },
    { "PAETH2D_W32", search_paeth2d_w32, ap_paeth2d_w32, inv_paeth2d_w32 },
    { "EMA_BIAS_WIDE", search_emabias_wide, ap_emabias_wide, inv_emabias_wide },
    { "WINDOW_ADD_W512", search_windowadd_w512, ap_windowadd_w512, inv_windowadd_w512 },
    { "BIT_MAJ3_HI", search_bitmaj3_hi, ap_bitmaj3_hi, ap_bitmaj3_hi }, /* self-inverse */
    { "QR_XOR_MOD7", search_qrxor_mod7, ap_qrxor_mod7, ap_qrxor_mod7 }, /* self-inverse */
    { "MIN3_PRED",   search_min3pred,   ap_min3pred,   inv_min3pred   },
    { "DELTA_XLONGLAG", search_deltaxlonglag, ap_deltaxlonglag, inv_deltaxlonglag },
    { "MIRROR_XOR_HALF", search_mirrorxor_half, ap_mirrorxor_half, ap_mirrorxor_half }, /* self-inverse */
    { "WAVG_PRED2",  search_wavgpred2,  ap_wavgpred2,  inv_wavgpred2  },
    { "HASH_XOR2",   search_hashxor2,   ap_hashxor2,   ap_hashxor2    }, /* self-inverse */
    { "RANK_ADD_K2", search_rankadd_k2, ap_rankadd_k2, inv_rankadd_k2 },
    { "ROTATE_PRED_LAG2", search_rotatepred_lag2, ap_rotatepred_lag2, inv_rotatepred_lag2 },
    { "AFFINE_A9",   search_affine_v9,  ap_affine_v9,  inv_affine_v9  },
    { "AFFINE_A11",  search_affine_v11, ap_affine_v11, inv_affine_v11 },
    { "AFFINE_A13",  search_affine_v13, ap_affine_v13, inv_affine_v13 },
    { "AFFINE_A15",  search_affine_v15, ap_affine_v15, inv_affine_v15 },
    { "AFFINE_A17",  search_affine_v17, ap_affine_v17, inv_affine_v17 },
    { "AFFINE_A19",  search_affine_v19, ap_affine_v19, inv_affine_v19 },
    { "AFFINE_A21",  search_affine_v21, ap_affine_v21, inv_affine_v21 },
    { "AFFINE_A23",  search_affine_v23, ap_affine_v23, inv_affine_v23 },
    { "AFFINE_A25",  search_affine_v25, ap_affine_v25, inv_affine_v25 },
    { "AFFINE_A27",  search_affine_v27, ap_affine_v27, inv_affine_v27 },
    { "AFFINE_A29",  search_affine_v29, ap_affine_v29, inv_affine_v29 },
    { "AFFINE_A31",  search_affine_v31, ap_affine_v31, inv_affine_v31 },
    { "AFFINE_A37",  search_affine_v37, ap_affine_v37, inv_affine_v37 },
    { "AFFINE_A41",  search_affine_v41, ap_affine_v41, inv_affine_v41 },
    { "AFFINE_A43",  search_affine_v43, ap_affine_v43, inv_affine_v43 },
    { "AFFINE_A47",  search_affine_v47, ap_affine_v47, inv_affine_v47 },
    { "AFFINE_A53",  search_affine_v53, ap_affine_v53, inv_affine_v53 },
    { "AFFINE_A59",  search_affine_v59, ap_affine_v59, inv_affine_v59 },
    { "AFFINE_A61",  search_affine_v61, ap_affine_v61, inv_affine_v61 },
    { "AFFINE_A63",  search_affine_v63, ap_affine_v63, inv_affine_v63 },
    { "AFFINE_A33",  search_affine_v33,  ap_affine_v33,  inv_affine_v33  },
    { "AFFINE_A35",  search_affine_v35,  ap_affine_v35,  inv_affine_v35  },
    { "AFFINE_A39",  search_affine_v39,  ap_affine_v39,  inv_affine_v39  },
    { "AFFINE_A45",  search_affine_v45,  ap_affine_v45,  inv_affine_v45  },
    { "AFFINE_A49",  search_affine_v49,  ap_affine_v49,  inv_affine_v49  },
    { "AFFINE_A51",  search_affine_v51,  ap_affine_v51,  inv_affine_v51  },
    { "AFFINE_A55",  search_affine_v55,  ap_affine_v55,  inv_affine_v55  },
    { "AFFINE_A57",  search_affine_v57,  ap_affine_v57,  inv_affine_v57  },
    { "AFFINE_A65",  search_affine_v65,  ap_affine_v65,  inv_affine_v65  },
    { "AFFINE_A67",  search_affine_v67,  ap_affine_v67,  inv_affine_v67  },
    { "AFFINE_A69",  search_affine_v69,  ap_affine_v69,  inv_affine_v69  },
    { "AFFINE_A71",  search_affine_v71,  ap_affine_v71,  inv_affine_v71  },
    { "AFFINE_A73",  search_affine_v73,  ap_affine_v73,  inv_affine_v73  },
    { "AFFINE_A75",  search_affine_v75,  ap_affine_v75,  inv_affine_v75  },
    { "AFFINE_A77",  search_affine_v77,  ap_affine_v77,  inv_affine_v77  },
    { "AFFINE_A79",  search_affine_v79,  ap_affine_v79,  inv_affine_v79  },
    { "AFFINE_A81",  search_affine_v81,  ap_affine_v81,  inv_affine_v81  },
    { "AFFINE_A83",  search_affine_v83,  ap_affine_v83,  inv_affine_v83  },
    { "AFFINE_A85",  search_affine_v85,  ap_affine_v85,  inv_affine_v85  },
    { "AFFINE_A87",  search_affine_v87,  ap_affine_v87,  inv_affine_v87  },
    { "AFFINE_A89",  search_affine_v89,  ap_affine_v89,  inv_affine_v89  },
    { "AFFINE_A91",  search_affine_v91,  ap_affine_v91,  inv_affine_v91  },
    { "AFFINE_A93",  search_affine_v93,  ap_affine_v93,  inv_affine_v93  },
    { "AFFINE_A95",  search_affine_v95,  ap_affine_v95,  inv_affine_v95  },
    { "AFFINE_A97",  search_affine_v97,  ap_affine_v97,  inv_affine_v97  },
    { "AFFINE_A99",  search_affine_v99,  ap_affine_v99,  inv_affine_v99  },
    { "AFFINE_A101", search_affine_v101, ap_affine_v101, inv_affine_v101 },
    { "AFFINE_A103", search_affine_v103, ap_affine_v103, inv_affine_v103 },
    { "AFFINE_A105", search_affine_v105, ap_affine_v105, inv_affine_v105 },
    { "AFFINE_A107", search_affine_v107, ap_affine_v107, inv_affine_v107 },
    { "AFFINE_A109", search_affine_v109, ap_affine_v109, inv_affine_v109 },
    { "AFFINE_A111", search_affine_v111, ap_affine_v111, inv_affine_v111 },
    { "VALUE_GFMUL_B1", search_valuegfmul_b1, ap_valuegfmul_b1, inv_valuegfmul_b1 },
    { "VALUE_GFMUL_B2", search_valuegfmul_b2, ap_valuegfmul_b2, inv_valuegfmul_b2 },
    { "VALUE_GFMUL_B5", search_valuegfmul_b5, ap_valuegfmul_b5, inv_valuegfmul_b5 },
    { "VALUE_GFMUL_B6", search_valuegfmul_b6, ap_valuegfmul_b6, inv_valuegfmul_b6 },
    { "RANK_XOR_K4", search_rankxor_k4, ap_rankxor_k4, ap_rankxor_k4 }, /* self-inverse */
    { "RANK_XOR_K5", search_rankxor_k5, ap_rankxor_k5, ap_rankxor_k5 }, /* self-inverse */
    { "RANK_XOR_K6", search_rankxor_k6, ap_rankxor_k6, ap_rankxor_k6 }, /* self-inverse */
    { "RANK_XOR_K7", search_rankxor_k7, ap_rankxor_k7, ap_rankxor_k7 }, /* self-inverse */
    { "RANK_ADD_K3", search_rankadd_k3, ap_rankadd_k3, inv_rankadd_k3 },
    { "RANK_ADD_K4", search_rankadd_k4, ap_rankadd_k4, inv_rankadd_k4 },
    { "RANK_ADD_K5", search_rankadd_k5, ap_rankadd_k5, inv_rankadd_k5 },
    { "RANK_ADD_K6", search_rankadd_k6, ap_rankadd_k6, inv_rankadd_k6 },
    { "RANK_ADD_K7", search_rankadd_k7, ap_rankadd_k7, inv_rankadd_k7 },
    { "POPCNT_XOR_T4", search_popcxor_v4, ap_popcxor_v4, ap_popcxor_v4 }, /* self-inverse */
    { "POPCNT_XOR_T5", search_popcxor_v5, ap_popcxor_v5, ap_popcxor_v5 }, /* self-inverse */
    { "POPCNT_XOR_T6", search_popcxor_v6, ap_popcxor_v6, ap_popcxor_v6 }, /* self-inverse */
    { "RANGE_XOR_T4", search_rangexor_v4, ap_rangexor_v4, ap_rangexor_v4 }, /* self-inverse */
    { "RANGE_XOR_T5", search_rangexor_v5, ap_rangexor_v5, ap_rangexor_v5 }, /* self-inverse */
    { "RANGE_XOR_T6", search_rangexor_v6, ap_rangexor_v6, ap_rangexor_v6 }, /* self-inverse */
    { "RANGE_XOR_T7", search_rangexor_v7, ap_rangexor_v7, ap_rangexor_v7 }, /* self-inverse */
    { "MAGCLASS_XOR_T4", search_magclass_v4, ap_magclass_v4, ap_magclass_v4 }, /* self-inverse */
    { "PRNG_ADD3",   search_prngadd3,   ap_prngadd3,   inv_prngadd3   },
    { "PRNG_XOR3",   search_prngxor3,   ap_prngxor3,   ap_prngxor3    }, /* self-inverse */
    { "PRNG_ADD5",   search_prngadd5,   ap_prngadd5,   inv_prngadd5   },
    { "PRNG_XOR5",   search_prngxor5,   ap_prngxor5,   ap_prngxor5    }, /* self-inverse */
    { "PRNG_ADD6",   search_prngadd6,   ap_prngadd6,   inv_prngadd6   },
    { "PRNG_XOR6",   search_prngxor6,   ap_prngxor6,   ap_prngxor6    }, /* self-inverse */
    { "PRNG_ADD7",   search_prngadd7,   ap_prngadd7,   inv_prngadd7   },
    { "PRNG_XOR7",   search_prngxor7,   ap_prngxor7,   ap_prngxor7    }, /* self-inverse */
    { "PRNG_ADD32",  search_prngadd32,  ap_prngadd32,  inv_prngadd32  },
    { "PRNG_XOR32",  search_prngxor32,  ap_prngxor32,  ap_prngxor32   }, /* self-inverse */
    { "AVG8_PRED",   search_avg8pred,   ap_avg8pred,   inv_avg8pred   },
    { "MIN5_PRED",   search_min5pred,   ap_min5pred,   inv_min5pred   },
    { "MAX5_PRED",   search_max5pred,   ap_max5pred,   inv_max5pred   },
    { "MAX6_PRED",   search_max6pred,   ap_max6pred,   inv_max6pred   },
    { "WINDOW_XOR_W4", search_winxor_w4, ap_winxor_w4, ap_winxor_w4 }, /* self-inverse */
    { "WINDOW_XOR_W64", search_winxor_w64, ap_winxor_w64, ap_winxor_w64 }, /* self-inverse */
    { "WINDOW_XOR_W128", search_winxor_w128, ap_winxor_w128, ap_winxor_w128 }, /* self-inverse */
    { "WINDOW_XOR_W256", search_winxor_w256, ap_winxor_w256, ap_winxor_w256 }, /* self-inverse */
    { "WINDOW_XOR_W512", search_winxor_w512, ap_winxor_w512, ap_winxor_w512 }, /* self-inverse */
    { "WINDOW_XOR_W1024", search_winxor_w1024, ap_winxor_w1024, ap_winxor_w1024 }, /* self-inverse */
    { "WINDOW_XOR_W2048", search_winxor_w2048, ap_winxor_w2048, ap_winxor_w2048 }, /* self-inverse */
    { "WINDOW_ADD_W1024", search_windowadd_w1024, ap_windowadd_w1024, inv_windowadd_w1024 },
    { "WINDOW_ADD_W2048", search_windowadd_w2048, ap_windowadd_w2048, inv_windowadd_w2048 },
    { "REGIONAL_ANCHOR_K3", search_regionalanchor_k3, ap_regionalanchor_k3, ap_regionalanchor_k3 }, /* self-inverse */
    { "REGIONAL_ANCHOR_K5", search_regionalanchor_k5, ap_regionalanchor_k5, ap_regionalanchor_k5 }, /* self-inverse */
    { "REGIONAL_ANCHOR_K6", search_regionalanchor_k6, ap_regionalanchor_k6, ap_regionalanchor_k6 }, /* self-inverse */
    { "REGIONAL_ANCHOR_K16", search_regionalanchor_k16, ap_regionalanchor_k16, ap_regionalanchor_k16 }, /* self-inverse */
    { "FIXED_KS6",   search_fixedks6,   ap_fixedks6,   ap_fixedks6    }, /* self-inverse */
    { "FIXED_KS8",   search_fixedks8,   ap_fixedks8,   ap_fixedks8    }, /* self-inverse */
    { "FIXED_KS10",  search_fixedks10,  ap_fixedks10,  ap_fixedks10   }, /* self-inverse */
    { "QR_XOR_MOD11", search_qrxor_mod11, ap_qrxor_mod11, ap_qrxor_mod11 }, /* self-inverse */
    { "QR_XOR_MOD13", search_qrxor_mod13, ap_qrxor_mod13, ap_qrxor_mod13 }, /* self-inverse */
    { "QR_ADD_MOD7", search_qradd_mod7, ap_qradd_mod7, inv_qradd_mod7 },
    { "QR_ADD_MOD11", search_qradd_mod11, ap_qradd_mod11, inv_qradd_mod11 },
    { "QR_ADD_MOD13", search_qradd_mod13, ap_qradd_mod13, inv_qradd_mod13 },
    { "BIT_MAJ3_K1", search_bitmaj3_k1, ap_bitmaj3_k1, ap_bitmaj3_k1 }, /* self-inverse */
    { "BIT_MAJ3_K2", search_bitmaj3_k2, ap_bitmaj3_k2, ap_bitmaj3_k2 }, /* self-inverse */
    { "BIT_MAJ3_K4", search_bitmaj3_k4, ap_bitmaj3_k4, ap_bitmaj3_k4 }, /* self-inverse */
    { "WORD_GFMUL_ODD", search_wordgfmul_odd, ap_wordgfmul_odd, inv_wordgfmul_odd },
    { "HASH_XOR3",   search_hashxor3,   ap_hashxor3,   ap_hashxor3    }, /* self-inverse */
    { "HASH_ADD2",   search_hashadd2,   ap_hashadd2,   inv_hashadd2   },
    { "HASH_ADD3",   search_hashadd3,   ap_hashadd3,   inv_hashadd3   },
    { "LATIN_CROSS3", search_latincross3, ap_latincross3, inv_latincross3 },
    { "KALMAN_PRED_ALT2", search_kalmanpred_alt2, ap_kalmanpred_alt2, inv_kalmanpred_alt2 },
    { "MIRROR_XOR_QUARTER", search_mirrorxor_quarter, ap_mirrorxor_quarter, ap_mirrorxor_quarter }, /* self-inverse */
    { "WORD_GFMUL_POLY2", search_wordgfmul_poly2, ap_wordgfmul_poly2, inv_wordgfmul_poly2 },
    { "WORD_GFMUL_POLY2_ODD", search_wordgfmul_poly2_odd, ap_wordgfmul_poly2_odd, inv_wordgfmul_poly2_odd },
    { "WORD_GFMUL_POLY3", search_wordgfmul_poly3, ap_wordgfmul_poly3, inv_wordgfmul_poly3 },
    { "WORD_GFMUL_POLY3_ODD", search_wordgfmul_poly3_odd, ap_wordgfmul_poly3_odd, inv_wordgfmul_poly3_odd },
    { "WORD_GFMUL_LE", search_wordgfmul_le, ap_wordgfmul_le, inv_wordgfmul_le },
    { "WORD_GFMUL_ODD_LE", search_wordgfmul_odd_le, ap_wordgfmul_odd_le, inv_wordgfmul_odd_le },
    { "WORD_GFMUL_SPARSE_EVEN", search_wordgfmul_sparse_even, ap_wordgfmul_sparse_even, inv_wordgfmul_sparse_even },
    { "WORD_GFMUL_SPARSE_ODD", search_wordgfmul_sparse_odd, ap_wordgfmul_sparse_odd, inv_wordgfmul_sparse_odd },
    { "HASH_XOR4",   search_hashxor4,   ap_hashxor4,   ap_hashxor4    }, /* self-inverse */
    { "HASH_ADD4",   search_hashadd4,   ap_hashadd4,   inv_hashadd4   },
    { "HASH_XOR5",   search_hashxor5,   ap_hashxor5,   ap_hashxor5    }, /* self-inverse */
    { "HASH_ADD5",   search_hashadd5,   ap_hashadd5,   inv_hashadd5   },
    { "HASH_XOR6",   search_hashxor6,   ap_hashxor6,   ap_hashxor6    }, /* self-inverse */
    { "HASH_ADD6",   search_hashadd6,   ap_hashadd6,   inv_hashadd6   },
    { "HASH_XOR7",   search_hashxor7,   ap_hashxor7,   ap_hashxor7    }, /* self-inverse */
    { "HASH_ADD7",   search_hashadd7,   ap_hashadd7,   inv_hashadd7   },
    { "HASH_XOR8",   search_hashxor8,   ap_hashxor8,   ap_hashxor8    }, /* self-inverse */
    { "HASH_ADD8",   search_hashadd8,   ap_hashadd8,   inv_hashadd8   },
    { "HASH_XOR9",   search_hashxor9,   ap_hashxor9,   ap_hashxor9    }, /* self-inverse */
    { "HASH_ADD9",   search_hashadd9,   ap_hashadd9,   inv_hashadd9   },
    { "HASH_XOR10",  search_hashxor10,  ap_hashxor10,  ap_hashxor10   }, /* self-inverse */
    { "HASH_ADD10",  search_hashadd10,  ap_hashadd10,  inv_hashadd10  },
    { "HASH_ADD11",  search_hashadd11,  ap_hashadd11,  inv_hashadd11  },
    { "HASH_XOR13",  search_hashxor13,  ap_hashxor13,  ap_hashxor13   }, /* self-inverse */
    { "HASH_ADD13",  search_hashadd13,  ap_hashadd13,  inv_hashadd13  },
    { "HASH_XOR14",  search_hashxor14,  ap_hashxor14,  ap_hashxor14   }, /* self-inverse */
    { "HASH_ADD14",  search_hashadd14,  ap_hashadd14,  inv_hashadd14  },
    { "HASH_XOR15",  search_hashxor15,  ap_hashxor15,  ap_hashxor15   }, /* self-inverse */
    { "HASH_ADD15",  search_hashadd15,  ap_hashadd15,  inv_hashadd15  },
    { "CRC32_ADD",   search_hashadd_crc32, ap_hashadd_crc32, inv_hashadd_crc32 },
    { "LCG_XOR",     search_hashxor_lcg, ap_hashxor_lcg, ap_hashxor_lcg }, /* self-inverse */
    { "WEYL_XOR",    search_hashxor_weyl, ap_hashxor_weyl, ap_hashxor_weyl }, /* self-inverse */
    { "WEYL_ADD",    search_hashadd_weyl, ap_hashadd_weyl, inv_hashadd_weyl },
    { "MSQUARE_XOR", search_hashxor_msquare, ap_hashxor_msquare, ap_hashxor_msquare }, /* self-inverse */
    { "MSQUARE_ADD", search_hashadd_msquare, ap_hashadd_msquare, inv_hashadd_msquare },
    { "QR_XOR_MOD17", search_qrxor_mod17, ap_qrxor_mod17, ap_qrxor_mod17 }, /* self-inverse */
    { "QR_ADD_MOD17", search_qradd_mod17, ap_qradd_mod17, inv_qradd_mod17 },
    { "QR_XOR_MOD19", search_qrxor_mod19, ap_qrxor_mod19, ap_qrxor_mod19 }, /* self-inverse */
    { "QR_ADD_MOD19", search_qradd_mod19, ap_qradd_mod19, inv_qradd_mod19 },
    { "QR_XOR_MOD23", search_qrxor_mod23, ap_qrxor_mod23, ap_qrxor_mod23 }, /* self-inverse */
    { "QR_ADD_MOD23", search_qradd_mod23, ap_qradd_mod23, inv_qradd_mod23 },
    { "QR_XOR_MOD29", search_qrxor_mod29, ap_qrxor_mod29, ap_qrxor_mod29 }, /* self-inverse */
    { "QR_ADD_MOD29", search_qradd_mod29, ap_qradd_mod29, inv_qradd_mod29 },
    { "QR_XOR_MOD31", search_qrxor_mod31, ap_qrxor_mod31, ap_qrxor_mod31 }, /* self-inverse */
    { "QR_ADD_MOD31", search_qradd_mod31, ap_qradd_mod31, inv_qradd_mod31 },
    { "QR_XOR_MOD37", search_qrxor_mod37, ap_qrxor_mod37, ap_qrxor_mod37 }, /* self-inverse */
    { "QR_ADD_MOD37", search_qradd_mod37, ap_qradd_mod37, inv_qradd_mod37 },
    { "QR_XOR_MOD41", search_qrxor_mod41, ap_qrxor_mod41, ap_qrxor_mod41 }, /* self-inverse */
    { "QR_ADD_MOD41", search_qradd_mod41, ap_qradd_mod41, inv_qradd_mod41 },
    { "QR_XOR_MOD43", search_qrxor_mod43, ap_qrxor_mod43, ap_qrxor_mod43 }, /* self-inverse */
    { "QR_ADD_MOD43", search_qradd_mod43, ap_qradd_mod43, inv_qradd_mod43 },
    { "QR_XOR_MOD47", search_qrxor_mod47, ap_qrxor_mod47, ap_qrxor_mod47 }, /* self-inverse */
    { "QR_ADD_MOD47", search_qradd_mod47, ap_qradd_mod47, inv_qradd_mod47 },
    { "QR_XOR_MOD53", search_qrxor_mod53, ap_qrxor_mod53, ap_qrxor_mod53 }, /* self-inverse */
    { "QR_ADD_MOD53", search_qradd_mod53, ap_qradd_mod53, inv_qradd_mod53 },
    { "QR_XOR_MOD59", search_qrxor_mod59, ap_qrxor_mod59, ap_qrxor_mod59 }, /* self-inverse */
    { "QR_ADD_MOD59", search_qradd_mod59, ap_qradd_mod59, inv_qradd_mod59 },
    { "QR_XOR_MOD61", search_qrxor_mod61, ap_qrxor_mod61, ap_qrxor_mod61 }, /* self-inverse */
    { "QR_ADD_MOD61", search_qradd_mod61, ap_qradd_mod61, inv_qradd_mod61 },
    { "QR_XOR_MOD67", search_qrxor_mod67, ap_qrxor_mod67, ap_qrxor_mod67 }, /* self-inverse */
    { "QR_ADD_MOD67", search_qradd_mod67, ap_qradd_mod67, inv_qradd_mod67 },
    { "QR_XOR_MOD71", search_qrxor_mod71, ap_qrxor_mod71, ap_qrxor_mod71 }, /* self-inverse */
    { "QR_ADD_MOD71", search_qradd_mod71, ap_qradd_mod71, inv_qradd_mod71 },
    { "QR_XOR_MOD73", search_qrxor_mod73, ap_qrxor_mod73, ap_qrxor_mod73 }, /* self-inverse */
    { "QR_ADD_MOD73", search_qradd_mod73, ap_qradd_mod73, inv_qradd_mod73 },
    { "QR_XOR_MOD79", search_qrxor_mod79, ap_qrxor_mod79, ap_qrxor_mod79 }, /* self-inverse */
    { "QR_ADD_MOD79", search_qradd_mod79, ap_qradd_mod79, inv_qradd_mod79 },
    { "VALUE_MUL_B1", search_valuemul_b1, ap_valuemul_b1, inv_valuemul_b1 },
    { "VALUE_MUL_B2", search_valuemul_b2, ap_valuemul_b2, inv_valuemul_b2 },
    { "VALUE_MUL_B3", search_valuemul_b3, ap_valuemul_b3, inv_valuemul_b3 },
    { "VALUE_MUL_B4", search_valuemul_b4, ap_valuemul_b4, inv_valuemul_b4 },
    { "VALUE_MUL_B5", search_valuemul_b5, ap_valuemul_b5, inv_valuemul_b5 },
    { "VALUE_MUL_B6", search_valuemul_b6, ap_valuemul_b6, inv_valuemul_b6 },
    { "VALUE_MUL_P01", search_valuemul_p01, ap_valuemul_p01, inv_valuemul_p01 },
    { "VALUE_MUL_P02", search_valuemul_p02, ap_valuemul_p02, inv_valuemul_p02 },
    { "VALUE_MUL_P03", search_valuemul_p03, ap_valuemul_p03, inv_valuemul_p03 },
    { "VALUE_MUL_P04", search_valuemul_p04, ap_valuemul_p04, inv_valuemul_p04 },
    { "VALUE_MUL_P05", search_valuemul_p05, ap_valuemul_p05, inv_valuemul_p05 },
    { "VALUE_MUL_P06", search_valuemul_p06, ap_valuemul_p06, inv_valuemul_p06 },
    { "VALUE_MUL_P07", search_valuemul_p07, ap_valuemul_p07, inv_valuemul_p07 },
    { "VALUE_MUL_P12", search_valuemul_p12, ap_valuemul_p12, inv_valuemul_p12 },
    { "VALUE_MUL_P13", search_valuemul_p13, ap_valuemul_p13, inv_valuemul_p13 },
    { "VALUE_MUL_P14", search_valuemul_p14, ap_valuemul_p14, inv_valuemul_p14 },
    { "VALUE_MUL_P15", search_valuemul_p15, ap_valuemul_p15, inv_valuemul_p15 },
    { "VALUE_MUL_P16", search_valuemul_p16, ap_valuemul_p16, inv_valuemul_p16 },
    { "VALUE_MUL_P17", search_valuemul_p17, ap_valuemul_p17, inv_valuemul_p17 },
    { "VALUE_MUL_P23", search_valuemul_p23, ap_valuemul_p23, inv_valuemul_p23 },
    { "VALUE_MUL_P24", search_valuemul_p24, ap_valuemul_p24, inv_valuemul_p24 },
    { "VALUE_MUL_P25", search_valuemul_p25, ap_valuemul_p25, inv_valuemul_p25 },
    { "VALUE_MUL_P26", search_valuemul_p26, ap_valuemul_p26, inv_valuemul_p26 },
    { "VALUE_MUL_P27", search_valuemul_p27, ap_valuemul_p27, inv_valuemul_p27 },
    { "VALUE_MUL_P34", search_valuemul_p34, ap_valuemul_p34, inv_valuemul_p34 },
    { "VALUE_MUL_P35", search_valuemul_p35, ap_valuemul_p35, inv_valuemul_p35 },
    { "VALUE_MUL_P36", search_valuemul_p36, ap_valuemul_p36, inv_valuemul_p36 },
    { "VALUE_MUL_P37", search_valuemul_p37, ap_valuemul_p37, inv_valuemul_p37 },
    { "VALUE_MUL_P45", search_valuemul_p45, ap_valuemul_p45, inv_valuemul_p45 },
    { "VALUE_MUL_P46", search_valuemul_p46, ap_valuemul_p46, inv_valuemul_p46 },
    { "VALUE_MUL_P47", search_valuemul_p47, ap_valuemul_p47, inv_valuemul_p47 },
    { "VALUE_MUL_P56", search_valuemul_p56, ap_valuemul_p56, inv_valuemul_p56 },
    { "PRIME263",  search_primeP_263,  ap_primeP_263,  inv_primeP_263  },
    { "PRIME269",  search_primeP_269,  ap_primeP_269,  inv_primeP_269  },
    { "PRIME271",  search_primeP_271,  ap_primeP_271,  inv_primeP_271  },
    { "PRIME277",  search_primeP_277,  ap_primeP_277,  inv_primeP_277  },
    { "PRIME281",  search_primeP_281,  ap_primeP_281,  inv_primeP_281  },
    { "PRIME283",  search_primeP_283,  ap_primeP_283,  inv_primeP_283  },
    { "PRIME293",  search_primeP_293,  ap_primeP_293,  inv_primeP_293  },
    { "PRIME307",  search_primeP_307,  ap_primeP_307,  inv_primeP_307  },
    { "PRIME311",  search_primeP_311,  ap_primeP_311,  inv_primeP_311  },
    { "PRIME313",  search_primeP_313,  ap_primeP_313,  inv_primeP_313  },
    { "PRIME317",  search_primeP_317,  ap_primeP_317,  inv_primeP_317  },
    { "PRIME331",  search_primeP_331,  ap_primeP_331,  inv_primeP_331  },
    { "PRIME337",  search_primeP_337,  ap_primeP_337,  inv_primeP_337  },
    { "PRIME347",  search_primeP_347,  ap_primeP_347,  inv_primeP_347  },
    { "PRIME349",  search_primeP_349,  ap_primeP_349,  inv_primeP_349  },
    { "PRIME353",  search_primeP_353,  ap_primeP_353,  inv_primeP_353  },
    { "PRIME359",  search_primeP_359,  ap_primeP_359,  inv_primeP_359  },
    { "PRIME367",  search_primeP_367,  ap_primeP_367,  inv_primeP_367  },
    { "PRIME373",  search_primeP_373,  ap_primeP_373,  inv_primeP_373  },
    { "PRIME379",  search_primeP_379,  ap_primeP_379,  inv_primeP_379  },
    { "PRIME383",  search_primeP_383,  ap_primeP_383,  inv_primeP_383  },
    { "PRIME389",  search_primeP_389,  ap_primeP_389,  inv_primeP_389  },
    { "PRIME397",  search_primeP_397,  ap_primeP_397,  inv_primeP_397  },
    { "PRIME401",  search_primeP_401,  ap_primeP_401,  inv_primeP_401  },
    { "PRIME409",  search_primeP_409,  ap_primeP_409,  inv_primeP_409  },
    { "PRIME419",  search_primeP_419,  ap_primeP_419,  inv_primeP_419  },
    { "PRIME421",  search_primeP_421,  ap_primeP_421,  inv_primeP_421  },
    { "PRIME431",  search_primeP_431,  ap_primeP_431,  inv_primeP_431  },
    { "PRIME433",  search_primeP_433,  ap_primeP_433,  inv_primeP_433  },
    { "PRIME439",  search_primeP_439,  ap_primeP_439,  inv_primeP_439  },
    { "PRIME443",  search_primeP_443,  ap_primeP_443,  inv_primeP_443  },
    { "PRIME449",  search_primeP_449,  ap_primeP_449,  inv_primeP_449  },
    { "NIB_POW_M17", search_nibpow_m17, ap_nibpow_m17, inv_nibpow_m17 },
    { "NIB_POW_M1B", search_nibpow_m1b, ap_nibpow_m1b, inv_nibpow_m1b },
    { "NIB_POW_M1D", search_nibpow_m1d, ap_nibpow_m1d, inv_nibpow_m1d },
    { "NIB_POW_M1E", search_nibpow_m1e, ap_nibpow_m1e, inv_nibpow_m1e },
    { "NIB_POW_M27", search_nibpow_m27, ap_nibpow_m27, inv_nibpow_m27 },
    { "NIB_POW_M2B", search_nibpow_m2b, ap_nibpow_m2b, inv_nibpow_m2b },
    { "NIB_POW_M2D", search_nibpow_m2d, ap_nibpow_m2d, inv_nibpow_m2d },
    { "NIB_POW_M2E", search_nibpow_m2e, ap_nibpow_m2e, inv_nibpow_m2e },
    { "NIB_POW_M33", search_nibpow_m33, ap_nibpow_m33, inv_nibpow_m33 },
    { "NIB_POW_M35", search_nibpow_m35, ap_nibpow_m35, inv_nibpow_m35 },
    { "NIB_POW_M36", search_nibpow_m36, ap_nibpow_m36, inv_nibpow_m36 },
    { "NIB_POW_M39", search_nibpow_m39, ap_nibpow_m39, inv_nibpow_m39 },
    { "NIB_POW_M3A", search_nibpow_m3a, ap_nibpow_m3a, inv_nibpow_m3a },
    { "NIB_POW_M3C", search_nibpow_m3c, ap_nibpow_m3c, inv_nibpow_m3c },
    { "NIB_POW_M47", search_nibpow_m47, ap_nibpow_m47, inv_nibpow_m47 },
    { "NIB_POW_M4B", search_nibpow_m4b, ap_nibpow_m4b, inv_nibpow_m4b },
    { "NIB_POW_M4D", search_nibpow_m4d, ap_nibpow_m4d, inv_nibpow_m4d },
    { "NIB_POW_M4E", search_nibpow_m4e, ap_nibpow_m4e, inv_nibpow_m4e },
    { "NIB_POW_M53", search_nibpow_m53, ap_nibpow_m53, inv_nibpow_m53 },
    { "NIB_POW_M55", search_nibpow_m55, ap_nibpow_m55, inv_nibpow_m55 },
    { "NIB_POW_M56", search_nibpow_m56, ap_nibpow_m56, inv_nibpow_m56 },
    { "NIB_POW_M59", search_nibpow_m59, ap_nibpow_m59, inv_nibpow_m59 },
    { "NIB_POW_M5A", search_nibpow_m5a, ap_nibpow_m5a, inv_nibpow_m5a },
    { "NIB_POW_M5C", search_nibpow_m5c, ap_nibpow_m5c, inv_nibpow_m5c },
    { "NIB_POW_M63", search_nibpow_m63, ap_nibpow_m63, inv_nibpow_m63 },
    { "NIB_POW_M65", search_nibpow_m65, ap_nibpow_m65, inv_nibpow_m65 },
    { "NIB_POW_M66", search_nibpow_m66, ap_nibpow_m66, inv_nibpow_m66 },
    { "NIB_POW_M69", search_nibpow_m69, ap_nibpow_m69, inv_nibpow_m69 },
    { "NIB_POW_M6A", search_nibpow_m6a, ap_nibpow_m6a, inv_nibpow_m6a },
    { "NIB_POW_M6C", search_nibpow_m6c, ap_nibpow_m6c, inv_nibpow_m6c },
    { "NIB_POW_M71", search_nibpow_m71, ap_nibpow_m71, inv_nibpow_m71 },
    { "NIB_POW_M72", search_nibpow_m72, ap_nibpow_m72, inv_nibpow_m72 },
    { "BIT_CXOR2_P01", search_bitcxor2_p01, ap_bitcxor2_p01, ap_bitcxor2_p01 }, /* self-inverse */
    { "BIT_CXOR2_P02", search_bitcxor2_p02, ap_bitcxor2_p02, ap_bitcxor2_p02 }, /* self-inverse */
    { "BIT_CXOR2_P03", search_bitcxor2_p03, ap_bitcxor2_p03, ap_bitcxor2_p03 }, /* self-inverse */
    { "BIT_CXOR2_P04", search_bitcxor2_p04, ap_bitcxor2_p04, ap_bitcxor2_p04 }, /* self-inverse */
    { "BIT_CXOR2_P05", search_bitcxor2_p05, ap_bitcxor2_p05, ap_bitcxor2_p05 }, /* self-inverse */
    { "BIT_CXOR2_P06", search_bitcxor2_p06, ap_bitcxor2_p06, ap_bitcxor2_p06 }, /* self-inverse */
    { "BIT_CXOR2_P07", search_bitcxor2_p07, ap_bitcxor2_p07, ap_bitcxor2_p07 }, /* self-inverse */
    { "BIT_CXOR2_P12", search_bitcxor2_p12, ap_bitcxor2_p12, ap_bitcxor2_p12 }, /* self-inverse */
    { "BIT_CXOR2_P13", search_bitcxor2_p13, ap_bitcxor2_p13, ap_bitcxor2_p13 }, /* self-inverse */
    { "BIT_CXOR2_P14", search_bitcxor2_p14, ap_bitcxor2_p14, ap_bitcxor2_p14 }, /* self-inverse */
    { "BIT_CXOR2_P15", search_bitcxor2_p15, ap_bitcxor2_p15, ap_bitcxor2_p15 }, /* self-inverse */
    { "BIT_CXOR2_P16", search_bitcxor2_p16, ap_bitcxor2_p16, ap_bitcxor2_p16 }, /* self-inverse */
    { "BIT_CXOR2_P17", search_bitcxor2_p17, ap_bitcxor2_p17, ap_bitcxor2_p17 }, /* self-inverse */
    { "BIT_CXOR2_P23", search_bitcxor2_p23, ap_bitcxor2_p23, ap_bitcxor2_p23 }, /* self-inverse */
    { "BIT_CXOR2_P24", search_bitcxor2_p24, ap_bitcxor2_p24, ap_bitcxor2_p24 }, /* self-inverse */
    { "BIT_CXOR2_P25", search_bitcxor2_p25, ap_bitcxor2_p25, ap_bitcxor2_p25 }, /* self-inverse */
    { "BIT_CXOR2_P26", search_bitcxor2_p26, ap_bitcxor2_p26, ap_bitcxor2_p26 }, /* self-inverse */
    { "BIT_CXOR2_P27", search_bitcxor2_p27, ap_bitcxor2_p27, ap_bitcxor2_p27 }, /* self-inverse */
    { "BIT_CXOR2_P34", search_bitcxor2_p34, ap_bitcxor2_p34, ap_bitcxor2_p34 }, /* self-inverse */
    { "BIT_CXOR2_P35", search_bitcxor2_p35, ap_bitcxor2_p35, ap_bitcxor2_p35 }, /* self-inverse */
    { "BIT_CXOR2_P36", search_bitcxor2_p36, ap_bitcxor2_p36, ap_bitcxor2_p36 }, /* self-inverse */
    { "BIT_CXOR2_P37", search_bitcxor2_p37, ap_bitcxor2_p37, ap_bitcxor2_p37 }, /* self-inverse */
    { "BIT_CXOR2_P45", search_bitcxor2_p45, ap_bitcxor2_p45, ap_bitcxor2_p45 }, /* self-inverse */
    { "BIT_CXOR2_P46", search_bitcxor2_p46, ap_bitcxor2_p46, ap_bitcxor2_p46 }, /* self-inverse */
    { "BIT_CXOR2_P47", search_bitcxor2_p47, ap_bitcxor2_p47, ap_bitcxor2_p47 }, /* self-inverse */
    { "BIT_CXOR2_P56", search_bitcxor2_p56, ap_bitcxor2_p56, ap_bitcxor2_p56 }, /* self-inverse */
    { "BIT_CXOR2_P57", search_bitcxor2_p57, ap_bitcxor2_p57, ap_bitcxor2_p57 }, /* self-inverse */
    { "BIT_CXOR2_P67", search_bitcxor2_p67, ap_bitcxor2_p67, ap_bitcxor2_p67 }, /* self-inverse */
    { "BIT_CXOR3_T012", search_bitcxor3_t012, ap_bitcxor3_t012, ap_bitcxor3_t012 }, /* self-inverse */
    { "BIT_CXOR3_T345", search_bitcxor3_t345, ap_bitcxor3_t345, ap_bitcxor3_t345 }, /* self-inverse */
    { "BIT_CXOR3_T135", search_bitcxor3_t135, ap_bitcxor3_t135, ap_bitcxor3_t135 }, /* self-inverse */
    { "BIT_CXOR3_T246", search_bitcxor3_t246, ap_bitcxor3_t246, ap_bitcxor3_t246 }, /* self-inverse */
    { "VALMAP_ADD_P01", search_valmapadd_p01, ap_valmapadd_p01, inv_valmapadd_p01 },
    { "VALMAP_ADD_P02", search_valmapadd_p02, ap_valmapadd_p02, inv_valmapadd_p02 },
    { "VALMAP_ADD_P03", search_valmapadd_p03, ap_valmapadd_p03, inv_valmapadd_p03 },
    { "VALMAP_ADD_P04", search_valmapadd_p04, ap_valmapadd_p04, inv_valmapadd_p04 },
    { "VALMAP_ADD_P05", search_valmapadd_p05, ap_valmapadd_p05, inv_valmapadd_p05 },
    { "VALMAP_ADD_P06", search_valmapadd_p06, ap_valmapadd_p06, inv_valmapadd_p06 },
    { "VALMAP_ADD_P07", search_valmapadd_p07, ap_valmapadd_p07, inv_valmapadd_p07 },
    { "VALMAP_ADD_P12", search_valmapadd_p12, ap_valmapadd_p12, inv_valmapadd_p12 },
    { "VALMAP_ADD_P13", search_valmapadd_p13, ap_valmapadd_p13, inv_valmapadd_p13 },
    { "VALMAP_ADD_P14", search_valmapadd_p14, ap_valmapadd_p14, inv_valmapadd_p14 },
    { "VALMAP_ADD_P15", search_valmapadd_p15, ap_valmapadd_p15, inv_valmapadd_p15 },
    { "VALMAP_ADD_P16", search_valmapadd_p16, ap_valmapadd_p16, inv_valmapadd_p16 },
    { "VALMAP_ADD_P17", search_valmapadd_p17, ap_valmapadd_p17, inv_valmapadd_p17 },
    { "VALMAP_ADD_P23", search_valmapadd_p23, ap_valmapadd_p23, inv_valmapadd_p23 },
    { "VALMAP_ADD_P24", search_valmapadd_p24, ap_valmapadd_p24, inv_valmapadd_p24 },
    { "VALMAP_ADD_P25", search_valmapadd_p25, ap_valmapadd_p25, inv_valmapadd_p25 },
    { "VALMAP_ADD_P26", search_valmapadd_p26, ap_valmapadd_p26, inv_valmapadd_p26 },
    { "VALMAP_ADD_P27", search_valmapadd_p27, ap_valmapadd_p27, inv_valmapadd_p27 },
    { "VALMAP_ADD_P34", search_valmapadd_p34, ap_valmapadd_p34, inv_valmapadd_p34 },
    { "VALMAP_ADD_P35", search_valmapadd_p35, ap_valmapadd_p35, inv_valmapadd_p35 },
    { "VALMAP_ADD_P36", search_valmapadd_p36, ap_valmapadd_p36, inv_valmapadd_p36 },
    { "VALMAP_ADD_P37", search_valmapadd_p37, ap_valmapadd_p37, inv_valmapadd_p37 },
    { "VALMAP_ADD_P45", search_valmapadd_p45, ap_valmapadd_p45, inv_valmapadd_p45 },
    { "VALMAP_ADD_P46", search_valmapadd_p46, ap_valmapadd_p46, inv_valmapadd_p46 },
    { "VALMAP_ADD_P47", search_valmapadd_p47, ap_valmapadd_p47, inv_valmapadd_p47 },
    { "VALMAP_ADD_P56", search_valmapadd_p56, ap_valmapadd_p56, inv_valmapadd_p56 },
    { "VALMAP_ADD_P57", search_valmapadd_p57, ap_valmapadd_p57, inv_valmapadd_p57 },
    { "VALMAP_ADD_B0", search_valmapadd_b0, ap_valmapadd_b0, inv_valmapadd_b0 },
    { "VALMAP_ADD_B1", search_valmapadd_b1, ap_valmapadd_b1, inv_valmapadd_b1 },
    { "VALMAP_ADD_B2", search_valmapadd_b2, ap_valmapadd_b2, inv_valmapadd_b2 },
    { "VALMAP_ADD_B3", search_valmapadd_b3, ap_valmapadd_b3, inv_valmapadd_b3 },
    { "VALMAP_ADD_B4", search_valmapadd_b4, ap_valmapadd_b4, inv_valmapadd_b4 },
    { "PRNG_PERM_BIT1", search_prngperm_bit1, ap_prngperm_bit1, inv_prngperm_bit1 },
    { "PRNG_PERM_BIT2", search_prngperm_bit2, ap_prngperm_bit2, inv_prngperm_bit2 },
    { "PRNG_PERM_BIT3", search_prngperm_bit3, ap_prngperm_bit3, inv_prngperm_bit3 },
    { "PRNG_PERM_BIT4", search_prngperm_bit4, ap_prngperm_bit4, inv_prngperm_bit4 },
    { "PRNG_PERM_BIT5", search_prngperm_bit5, ap_prngperm_bit5, inv_prngperm_bit5 },
    { "PRNG_PERM_BIT6", search_prngperm_bit6, ap_prngperm_bit6, inv_prngperm_bit6 },
    { "PRNG_PERM_BIT7", search_prngperm_bit7, ap_prngperm_bit7, inv_prngperm_bit7 },
    { "PRNG_PERM_E1", search_prngperm_e1, ap_prngperm_e1, inv_prngperm_e1 },
    { "PRNG_PERM_E2", search_prngperm_e2, ap_prngperm_e2, inv_prngperm_e2 },
    { "PRNG_PERM_E3", search_prngperm_e3, ap_prngperm_e3, inv_prngperm_e3 },
    { "PRNG_PERM_E4", search_prngperm_e4, ap_prngperm_e4, inv_prngperm_e4 },
    { "PRNG_PERM_E5", search_prngperm_e5, ap_prngperm_e5, inv_prngperm_e5 },
    { "PRNG_PERM_E6", search_prngperm_e6, ap_prngperm_e6, inv_prngperm_e6 },
    { "PRNG_PERM_E7", search_prngperm_e7, ap_prngperm_e7, inv_prngperm_e7 },
    { "PRNG_PERM_S1", search_prngperm_s1, ap_prngperm_s1, inv_prngperm_s1 },
    { "PRNG_PERM_S2", search_prngperm_s2, ap_prngperm_s2, inv_prngperm_s2 },
    { "PRNG_PERM_S3", search_prngperm_s3, ap_prngperm_s3, inv_prngperm_s3 },
    { "PRNG_PERM_S4", search_prngperm_s4, ap_prngperm_s4, inv_prngperm_s4 },
    { "PRNG_PERM_S5", search_prngperm_s5, ap_prngperm_s5, inv_prngperm_s5 },
    { "PRNG_PERM_S6", search_prngperm_s6, ap_prngperm_s6, inv_prngperm_s6 },
    { "PRNG_PERM_S7", search_prngperm_s7, ap_prngperm_s7, inv_prngperm_s7 },
    { "PRNG_PERM_S8", search_prngperm_s8, ap_prngperm_s8, inv_prngperm_s8 },
    { "PRNG_PERM_S9", search_prngperm_s9, ap_prngperm_s9, inv_prngperm_s9 },
    { "PRNG_PERM_S10", search_prngperm_s10, ap_prngperm_s10, inv_prngperm_s10 },
    { "PRNG_PERM_S11", search_prngperm_s11, ap_prngperm_s11, inv_prngperm_s11 },
    { "PRNG_PERM_S12", search_prngperm_s12, ap_prngperm_s12, inv_prngperm_s12 },
    { "PRNG_PERM_S13", search_prngperm_s13, ap_prngperm_s13, inv_prngperm_s13 },
    { "PRNG_PERM_S14", search_prngperm_s14, ap_prngperm_s14, inv_prngperm_s14 },
    { "PRNG_PERM_S15", search_prngperm_s15, ap_prngperm_s15, inv_prngperm_s15 },
    { "PRNG_PERM_Q1", search_prngperm_q1, ap_prngperm_q1, inv_prngperm_q1 },
    { "PRNG_PERM_Q2", search_prngperm_q2, ap_prngperm_q2, inv_prngperm_q2 },
    { "PRNG_PERM_Q3", search_prngperm_q3, ap_prngperm_q3, inv_prngperm_q3 },
    { "HASH_ADD8_S11E4", search_hashadd8v_s11e4, ap_hashadd8v_s11e4, inv_hashadd8v_s11e4 },
    { "HASH_ADD8_S11E6", search_hashadd8v_s11e6, ap_hashadd8v_s11e6, inv_hashadd8v_s11e6 },
    { "HASH_ADD8_S11E8", search_hashadd8v_s11e8, ap_hashadd8v_s11e8, inv_hashadd8v_s11e8 },
    { "HASH_ADD8_S11E10", search_hashadd8v_s11e10, ap_hashadd8v_s11e10, inv_hashadd8v_s11e10 },
    { "HASH_ADD8_S13E4", search_hashadd8v_s13e4, ap_hashadd8v_s13e4, inv_hashadd8v_s13e4 },
    { "HASH_ADD8_S13E6", search_hashadd8v_s13e6, ap_hashadd8v_s13e6, inv_hashadd8v_s13e6 },
    { "HASH_ADD8_S13E8", search_hashadd8v_s13e8, ap_hashadd8v_s13e8, inv_hashadd8v_s13e8 },
    { "HASH_ADD8_S13E10", search_hashadd8v_s13e10, ap_hashadd8v_s13e10, inv_hashadd8v_s13e10 },
    { "HASH_ADD8_S15E4", search_hashadd8v_s15e4, ap_hashadd8v_s15e4, inv_hashadd8v_s15e4 },
    { "HASH_ADD8_S15E6", search_hashadd8v_s15e6, ap_hashadd8v_s15e6, inv_hashadd8v_s15e6 },
    { "HASH_ADD8_S15E8", search_hashadd8v_s15e8, ap_hashadd8v_s15e8, inv_hashadd8v_s15e8 },
    { "HASH_ADD8_S15E10", search_hashadd8v_s15e10, ap_hashadd8v_s15e10, inv_hashadd8v_s15e10 },
    { "HASH_ADD8_S17E4", search_hashadd8v_s17e4, ap_hashadd8v_s17e4, inv_hashadd8v_s17e4 },
    { "HASH_ADD8_S17E6", search_hashadd8v_s17e6, ap_hashadd8v_s17e6, inv_hashadd8v_s17e6 },
    { "HASH_ADD8_S17E8", search_hashadd8v_s17e8, ap_hashadd8v_s17e8, inv_hashadd8v_s17e8 },
    { "HASH_ADD8_S17E10", search_hashadd8v_s17e10, ap_hashadd8v_s17e10, inv_hashadd8v_s17e10 },
    { "HASH_ADD8_S19E4", search_hashadd8v_s19e4, ap_hashadd8v_s19e4, inv_hashadd8v_s19e4 },
    { "HASH_ADD8_S19E6", search_hashadd8v_s19e6, ap_hashadd8v_s19e6, inv_hashadd8v_s19e6 },
    { "HASH_ADD8_S19E8", search_hashadd8v_s19e8, ap_hashadd8v_s19e8, inv_hashadd8v_s19e8 },
    { "HASH_ADD8_S19E10", search_hashadd8v_s19e10, ap_hashadd8v_s19e10, inv_hashadd8v_s19e10 },
    { "HASH_ADD8_S21E4", search_hashadd8v_s21e4, ap_hashadd8v_s21e4, inv_hashadd8v_s21e4 },
    { "HASH_ADD8_S21E6", search_hashadd8v_s21e6, ap_hashadd8v_s21e6, inv_hashadd8v_s21e6 },
    { "HASH_ADD8_S21E8", search_hashadd8v_s21e8, ap_hashadd8v_s21e8, inv_hashadd8v_s21e8 },
    { "HASH_ADD8_S21E10", search_hashadd8v_s21e10, ap_hashadd8v_s21e10, inv_hashadd8v_s21e10 },
    { "HASH_ADD8_S23E4", search_hashadd8v_s23e4, ap_hashadd8v_s23e4, inv_hashadd8v_s23e4 },
    { "HASH_ADD8_S23E6", search_hashadd8v_s23e6, ap_hashadd8v_s23e6, inv_hashadd8v_s23e6 },
    { "HASH_ADD8_S23E8", search_hashadd8v_s23e8, ap_hashadd8v_s23e8, inv_hashadd8v_s23e8 },
    { "HASH_ADD8_S23E10", search_hashadd8v_s23e10, ap_hashadd8v_s23e10, inv_hashadd8v_s23e10 },
    { "HASH_ADD8_S25E4", search_hashadd8v_s25e4, ap_hashadd8v_s25e4, inv_hashadd8v_s25e4 },
    { "HASH_ADD8_S25E6", search_hashadd8v_s25e6, ap_hashadd8v_s25e6, inv_hashadd8v_s25e6 },
    { "HASH_ADD8_S25E8", search_hashadd8v_s25e8, ap_hashadd8v_s25e8, inv_hashadd8v_s25e8 },
    { "HASH_ADD8_S25E10", search_hashadd8v_s25e10, ap_hashadd8v_s25e10, inv_hashadd8v_s25e10 },
    { "AFFINE_FULL", search_affinefull, ap_affinefull, inv_affinefull },
    { "HASH_ADD8_M6E2", search_hashadd8m_s6e2, ap_hashadd8m_s6e2, inv_hashadd8m_s6e2 },
    { "HASH_ADD8_M6E4", search_hashadd8m_s6e4, ap_hashadd8m_s6e4, inv_hashadd8m_s6e4 },
    { "HASH_ADD8_M6E6", search_hashadd8m_s6e6, ap_hashadd8m_s6e6, inv_hashadd8m_s6e6 },
    { "HASH_ADD8_M6E8", search_hashadd8m_s6e8, ap_hashadd8m_s6e8, inv_hashadd8m_s6e8 },
    { "HASH_ADD8_M8E2", search_hashadd8m_s8e2, ap_hashadd8m_s8e2, inv_hashadd8m_s8e2 },
    { "HASH_ADD8_M8E4", search_hashadd8m_s8e4, ap_hashadd8m_s8e4, inv_hashadd8m_s8e4 },
    { "HASH_ADD8_M8E6", search_hashadd8m_s8e6, ap_hashadd8m_s8e6, inv_hashadd8m_s8e6 },
    { "HASH_ADD8_M8E8", search_hashadd8m_s8e8, ap_hashadd8m_s8e8, inv_hashadd8m_s8e8 },
    { "HASH_ADD8_M10E2", search_hashadd8m_s10e2, ap_hashadd8m_s10e2, inv_hashadd8m_s10e2 },
    { "HASH_ADD8_M10E4", search_hashadd8m_s10e4, ap_hashadd8m_s10e4, inv_hashadd8m_s10e4 },
    { "HASH_ADD8_M10E6", search_hashadd8m_s10e6, ap_hashadd8m_s10e6, inv_hashadd8m_s10e6 },
    { "HASH_ADD8_M10E8", search_hashadd8m_s10e8, ap_hashadd8m_s10e8, inv_hashadd8m_s10e8 },
    { "HASH_ADD8_M12E2", search_hashadd8m_s12e2, ap_hashadd8m_s12e2, inv_hashadd8m_s12e2 },
    { "HASH_ADD8_M12E4", search_hashadd8m_s12e4, ap_hashadd8m_s12e4, inv_hashadd8m_s12e4 },
    { "HASH_ADD8_M12E6", search_hashadd8m_s12e6, ap_hashadd8m_s12e6, inv_hashadd8m_s12e6 },
    { "HASH_ADD8_M12E8", search_hashadd8m_s12e8, ap_hashadd8m_s12e8, inv_hashadd8m_s12e8 },
    { "HASH_ADD8_M14E2", search_hashadd8m_s14e2, ap_hashadd8m_s14e2, inv_hashadd8m_s14e2 },
    { "HASH_ADD8_M14E4", search_hashadd8m_s14e4, ap_hashadd8m_s14e4, inv_hashadd8m_s14e4 },
    { "HASH_ADD8_M14E6", search_hashadd8m_s14e6, ap_hashadd8m_s14e6, inv_hashadd8m_s14e6 },
    { "HASH_ADD8_M14E8", search_hashadd8m_s14e8, ap_hashadd8m_s14e8, inv_hashadd8m_s14e8 },
    { "HASH_ADD8_M16E2", search_hashadd8m_s16e2, ap_hashadd8m_s16e2, inv_hashadd8m_s16e2 },
    { "HASH_ADD8_M16E4", search_hashadd8m_s16e4, ap_hashadd8m_s16e4, inv_hashadd8m_s16e4 },
    { "HASH_ADD8_M16E6", search_hashadd8m_s16e6, ap_hashadd8m_s16e6, inv_hashadd8m_s16e6 },
    { "HASH_ADD8_M16E8", search_hashadd8m_s16e8, ap_hashadd8m_s16e8, inv_hashadd8m_s16e8 },
    { "HASH_ADD8_M18E2", search_hashadd8m_s18e2, ap_hashadd8m_s18e2, inv_hashadd8m_s18e2 },
    { "HASH_ADD8_M18E4", search_hashadd8m_s18e4, ap_hashadd8m_s18e4, inv_hashadd8m_s18e4 },
    { "HASH_ADD8_M18E6", search_hashadd8m_s18e6, ap_hashadd8m_s18e6, inv_hashadd8m_s18e6 },
    { "HASH_ADD8_M18E8", search_hashadd8m_s18e8, ap_hashadd8m_s18e8, inv_hashadd8m_s18e8 },
    { "HASH_ADD8_M20E2", search_hashadd8m_s20e2, ap_hashadd8m_s20e2, inv_hashadd8m_s20e2 },
    { "HASH_ADD8_M20E4", search_hashadd8m_s20e4, ap_hashadd8m_s20e4, inv_hashadd8m_s20e4 },
    { "HASH_ADD8_M20E6", search_hashadd8m_s20e6, ap_hashadd8m_s20e6, inv_hashadd8m_s20e6 },
    { "HASH_ADD8_M20E8", search_hashadd8m_s20e8, ap_hashadd8m_s20e8, inv_hashadd8m_s20e8 },
    { "PRIME457", search_primeP_457, ap_primeP_457, inv_primeP_457 },
    { "PRIME461", search_primeP_461, ap_primeP_461, inv_primeP_461 },
    { "PRIME463", search_primeP_463, ap_primeP_463, inv_primeP_463 },
    { "PRIME467", search_primeP_467, ap_primeP_467, inv_primeP_467 },
    { "PRIME479", search_primeP_479, ap_primeP_479, inv_primeP_479 },
    { "PRIME487", search_primeP_487, ap_primeP_487, inv_primeP_487 },
    { "PRIME491", search_primeP_491, ap_primeP_491, inv_primeP_491 },
    { "PRIME499", search_primeP_499, ap_primeP_499, inv_primeP_499 },
    { "PRIME503", search_primeP_503, ap_primeP_503, inv_primeP_503 },
    { "PRIME509", search_primeP_509, ap_primeP_509, inv_primeP_509 },
    { "PRIME521", search_primeP_521, ap_primeP_521, inv_primeP_521 },
    { "PRIME523", search_primeP_523, ap_primeP_523, inv_primeP_523 },
    { "PRIME541", search_primeP_541, ap_primeP_541, inv_primeP_541 },
    { "PRIME547", search_primeP_547, ap_primeP_547, inv_primeP_547 },
    { "PRIME557", search_primeP_557, ap_primeP_557, inv_primeP_557 },
    { "PRIME563", search_primeP_563, ap_primeP_563, inv_primeP_563 },
    { "PRIME569", search_primeP_569, ap_primeP_569, inv_primeP_569 },
    { "PRIME571", search_primeP_571, ap_primeP_571, inv_primeP_571 },
    { "PRIME577", search_primeP_577, ap_primeP_577, inv_primeP_577 },
    { "PRIME587", search_primeP_587, ap_primeP_587, inv_primeP_587 },
    { "PRIME593", search_primeP_593, ap_primeP_593, inv_primeP_593 },
    { "PRIME599", search_primeP_599, ap_primeP_599, inv_primeP_599 },
    { "PRIME601", search_primeP_601, ap_primeP_601, inv_primeP_601 },
    { "PRIME607", search_primeP_607, ap_primeP_607, inv_primeP_607 },
    { "PRIME613", search_primeP_613, ap_primeP_613, inv_primeP_613 },
    { "PRIME617", search_primeP_617, ap_primeP_617, inv_primeP_617 },
    { "PRIME619", search_primeP_619, ap_primeP_619, inv_primeP_619 },
    { "PRIME631", search_primeP_631, ap_primeP_631, inv_primeP_631 },
    { "PRIME641", search_primeP_641, ap_primeP_641, inv_primeP_641 },
    { "PRIME643", search_primeP_643, ap_primeP_643, inv_primeP_643 },
    { "PRIME647", search_primeP_647, ap_primeP_647, inv_primeP_647 },
    { "PRIME653", search_primeP_653, ap_primeP_653, inv_primeP_653 },
    { "PRNG_PERM_W1", search_prngperm_w1, ap_prngperm_w1, inv_prngperm_w1 },
    { "PRNG_PERM_W2", search_prngperm_w2, ap_prngperm_w2, inv_prngperm_w2 },
    { "PRNG_PERM_W3", search_prngperm_w3, ap_prngperm_w3, inv_prngperm_w3 },
    { "PRNG_PERM_W4", search_prngperm_w4, ap_prngperm_w4, inv_prngperm_w4 },
    { "PRNG_PERM_W5", search_prngperm_w5, ap_prngperm_w5, inv_prngperm_w5 },
    { "PRNG_PERM_W6", search_prngperm_w6, ap_prngperm_w6, inv_prngperm_w6 },
    { "PRNG_PERM_W7", search_prngperm_w7, ap_prngperm_w7, inv_prngperm_w7 },
    { "PRNG_PERM_W8", search_prngperm_w8, ap_prngperm_w8, inv_prngperm_w8 },
    { "PRNG_PERM_W9", search_prngperm_w9, ap_prngperm_w9, inv_prngperm_w9 },
    { "PRNG_PERM_W10", search_prngperm_w10, ap_prngperm_w10, inv_prngperm_w10 },
    { "PRNG_PERM_W11", search_prngperm_w11, ap_prngperm_w11, inv_prngperm_w11 },
    { "PRNG_PERM_W12", search_prngperm_w12, ap_prngperm_w12, inv_prngperm_w12 },
    { "PRNG_PERM_W13", search_prngperm_w13, ap_prngperm_w13, inv_prngperm_w13 },
    { "PRNG_PERM_W14", search_prngperm_w14, ap_prngperm_w14, inv_prngperm_w14 },
    { "PRNG_PERM_W15", search_prngperm_w15, ap_prngperm_w15, inv_prngperm_w15 },
    { "PRNG_PERM_W16", search_prngperm_w16, ap_prngperm_w16, inv_prngperm_w16 },
    { "PRNG_PERM_X1", search_prngperm_x1, ap_prngperm_x1, inv_prngperm_x1 },
    { "PRNG_PERM_X2", search_prngperm_x2, ap_prngperm_x2, inv_prngperm_x2 },
    { "PRNG_PERM_X3", search_prngperm_x3, ap_prngperm_x3, inv_prngperm_x3 },
    { "PRNG_PERM_X4", search_prngperm_x4, ap_prngperm_x4, inv_prngperm_x4 },
    { "PRNG_PERM_X5", search_prngperm_x5, ap_prngperm_x5, inv_prngperm_x5 },
    { "PRNG_PERM_X6", search_prngperm_x6, ap_prngperm_x6, inv_prngperm_x6 },
    { "PRNG_PERM_X7", search_prngperm_x7, ap_prngperm_x7, inv_prngperm_x7 },
    { "PRNG_PERM_X8", search_prngperm_x8, ap_prngperm_x8, inv_prngperm_x8 },
    { "PRNG_PERM_X9", search_prngperm_x9, ap_prngperm_x9, inv_prngperm_x9 },
    { "PRNG_PERM_X10", search_prngperm_x10, ap_prngperm_x10, inv_prngperm_x10 },
    { "PRNG_PERM_X11", search_prngperm_x11, ap_prngperm_x11, inv_prngperm_x11 },
    { "PRNG_PERM_X12", search_prngperm_x12, ap_prngperm_x12, inv_prngperm_x12 },
    { "PRNG_PERM_X13", search_prngperm_x13, ap_prngperm_x13, inv_prngperm_x13 },
    { "PRNG_PERM_X14", search_prngperm_x14, ap_prngperm_x14, inv_prngperm_x14 },
    { "PRNG_PERM_X15", search_prngperm_x15, ap_prngperm_x15, inv_prngperm_x15 },
    { "PRNG_PERM_X16", search_prngperm_x16, ap_prngperm_x16, inv_prngperm_x16 },
    { "PATTERN32_RUN1", search_patternrun_p32k1, ap_patternrun_p32k1, ap_patternrun_p32k1 }, /* self-inverse */
    { "PATTERN32_RUN2", search_patternrun_p32k2, ap_patternrun_p32k2, ap_patternrun_p32k2 }, /* self-inverse */
    { "PATTERN32_RUN3", search_patternrun_p32k3, ap_patternrun_p32k3, ap_patternrun_p32k3 }, /* self-inverse */
    { "PATTERN32_RUN4", search_patternrun_p32k4, ap_patternrun_p32k4, ap_patternrun_p32k4 }, /* self-inverse */
    { "PATTERN32_RUN5", search_patternrun_p32k5, ap_patternrun_p32k5, ap_patternrun_p32k5 }, /* self-inverse */
    { "PATTERN32_RUN6", search_patternrun_p32k6, ap_patternrun_p32k6, ap_patternrun_p32k6 }, /* self-inverse */
    { "PATTERN32_RUN7", search_patternrun_p32k7, ap_patternrun_p32k7, ap_patternrun_p32k7 }, /* self-inverse */
    { "PATTERN32_RUN8", search_patternrun_p32k8, ap_patternrun_p32k8, ap_patternrun_p32k8 }, /* self-inverse */
    { "PATTERN32_RUN9", search_patternrun_p32k9, ap_patternrun_p32k9, ap_patternrun_p32k9 }, /* self-inverse */
    { "PATTERN32_RUN10", search_patternrun_p32k10, ap_patternrun_p32k10, ap_patternrun_p32k10 }, /* self-inverse */
    { "PATTERN32_RUN11", search_patternrun_p32k11, ap_patternrun_p32k11, ap_patternrun_p32k11 }, /* self-inverse */
    { "PATTERN32_RUN12", search_patternrun_p32k12, ap_patternrun_p32k12, ap_patternrun_p32k12 }, /* self-inverse */
    { "PATTERN32_RUN13", search_patternrun_p32k13, ap_patternrun_p32k13, ap_patternrun_p32k13 }, /* self-inverse */
    { "PATTERN32_RUN14", search_patternrun_p32k14, ap_patternrun_p32k14, ap_patternrun_p32k14 }, /* self-inverse */
    { "PATTERN32_RUN15", search_patternrun_p32k15, ap_patternrun_p32k15, ap_patternrun_p32k15 }, /* self-inverse */
    { "PATTERN32_RUN16", search_patternrun_p32k16, ap_patternrun_p32k16, ap_patternrun_p32k16 }, /* self-inverse */
    { "PATTERN64_RUN2", search_patternrun_p64k2, ap_patternrun_p64k2, ap_patternrun_p64k2 }, /* self-inverse */
    { "PATTERN64_RUN3", search_patternrun_p64k3, ap_patternrun_p64k3, ap_patternrun_p64k3 }, /* self-inverse */
    { "PATTERN64_RUN4", search_patternrun_p64k4, ap_patternrun_p64k4, ap_patternrun_p64k4 }, /* self-inverse */
    { "PATTERN64_RUN5", search_patternrun_p64k5, ap_patternrun_p64k5, ap_patternrun_p64k5 }, /* self-inverse */
    { "PATTERN64_RUN6", search_patternrun_p64k6, ap_patternrun_p64k6, ap_patternrun_p64k6 }, /* self-inverse */
    { "PATTERN64_RUN7", search_patternrun_p64k7, ap_patternrun_p64k7, ap_patternrun_p64k7 }, /* self-inverse */
    { "PATTERN64_RUN8", search_patternrun_p64k8, ap_patternrun_p64k8, ap_patternrun_p64k8 }, /* self-inverse */
    { "PATTERN64_RUN9", search_patternrun_p64k9, ap_patternrun_p64k9, ap_patternrun_p64k9 }, /* self-inverse */
    { "PATTERN64_RUN10", search_patternrun_p64k10, ap_patternrun_p64k10, ap_patternrun_p64k10 }, /* self-inverse */
    { "PATTERN64_RUN11", search_patternrun_p64k11, ap_patternrun_p64k11, ap_patternrun_p64k11 }, /* self-inverse */
    { "PATTERN64_RUN12", search_patternrun_p64k12, ap_patternrun_p64k12, ap_patternrun_p64k12 }, /* self-inverse */
    { "PATTERN64_RUN13", search_patternrun_p64k13, ap_patternrun_p64k13, ap_patternrun_p64k13 }, /* self-inverse */
    { "PATTERN64_RUN14", search_patternrun_p64k14, ap_patternrun_p64k14, ap_patternrun_p64k14 }, /* self-inverse */
    { "PATTERN64_RUN15", search_patternrun_p64k15, ap_patternrun_p64k15, ap_patternrun_p64k15 }, /* self-inverse */
    { "PATTERN64_RUN16", search_patternrun_p64k16, ap_patternrun_p64k16, ap_patternrun_p64k16 }, /* self-inverse */
    { "AFFINE_A113",  search_affine_v113,  ap_affine_v113,  inv_affine_v113  },
    { "AFFINE_A115",  search_affine_v115,  ap_affine_v115,  inv_affine_v115  },
    { "AFFINE_A117",  search_affine_v117,  ap_affine_v117,  inv_affine_v117  },
    { "AFFINE_A119",  search_affine_v119,  ap_affine_v119,  inv_affine_v119  },
    { "AFFINE_A121",  search_affine_v121,  ap_affine_v121,  inv_affine_v121  },
    { "AFFINE_A123",  search_affine_v123,  ap_affine_v123,  inv_affine_v123  },
    { "AFFINE_A125",  search_affine_v125,  ap_affine_v125,  inv_affine_v125  },
    { "AFFINE_A127",  search_affine_v127,  ap_affine_v127,  inv_affine_v127  },
    { "AFFINE_A129",  search_affine_v129,  ap_affine_v129,  inv_affine_v129  },
    { "AFFINE_A131",  search_affine_v131,  ap_affine_v131,  inv_affine_v131  },
    { "AFFINE_A133",  search_affine_v133,  ap_affine_v133,  inv_affine_v133  },
    { "AFFINE_A135",  search_affine_v135,  ap_affine_v135,  inv_affine_v135  },
    { "AFFINE_A137",  search_affine_v137,  ap_affine_v137,  inv_affine_v137  },
    { "AFFINE_A139",  search_affine_v139,  ap_affine_v139,  inv_affine_v139  },
    { "AFFINE_A141",  search_affine_v141,  ap_affine_v141,  inv_affine_v141  },
    { "AFFINE_A143",  search_affine_v143,  ap_affine_v143,  inv_affine_v143  },
    { "AFFINE_A145",  search_affine_v145,  ap_affine_v145,  inv_affine_v145  },
    { "AFFINE_A147",  search_affine_v147,  ap_affine_v147,  inv_affine_v147  },
    { "AFFINE_A149",  search_affine_v149,  ap_affine_v149,  inv_affine_v149  },
    { "AFFINE_A151",  search_affine_v151,  ap_affine_v151,  inv_affine_v151  },
    { "AFFINE_A153",  search_affine_v153,  ap_affine_v153,  inv_affine_v153  },
    { "AFFINE_A155",  search_affine_v155,  ap_affine_v155,  inv_affine_v155  },
    { "AFFINE_A157",  search_affine_v157,  ap_affine_v157,  inv_affine_v157  },
    { "AFFINE_A159",  search_affine_v159,  ap_affine_v159,  inv_affine_v159  },
    { "AFFINE_A161",  search_affine_v161,  ap_affine_v161,  inv_affine_v161  },
    { "AFFINE_A163",  search_affine_v163,  ap_affine_v163,  inv_affine_v163  },
    { "AFFINE_A165",  search_affine_v165,  ap_affine_v165,  inv_affine_v165  },
    { "AFFINE_A167",  search_affine_v167,  ap_affine_v167,  inv_affine_v167  },
    { "AFFINE_A169",  search_affine_v169,  ap_affine_v169,  inv_affine_v169  },
    { "AFFINE_A171",  search_affine_v171,  ap_affine_v171,  inv_affine_v171  },
    { "AFFINE_A173",  search_affine_v173,  ap_affine_v173,  inv_affine_v173  },
    { "AFFINE_A175",  search_affine_v175,  ap_affine_v175,  inv_affine_v175  },
    { "AFFINE_A177",  search_affine_v177,  ap_affine_v177,  inv_affine_v177  },
    { "AFFINE_A179",  search_affine_v179,  ap_affine_v179,  inv_affine_v179  },
    { "AFFINE_A181",  search_affine_v181,  ap_affine_v181,  inv_affine_v181  },
    { "AFFINE_A183",  search_affine_v183,  ap_affine_v183,  inv_affine_v183  },
    { "AFFINE_A185",  search_affine_v185,  ap_affine_v185,  inv_affine_v185  },
    { "AFFINE_A187",  search_affine_v187,  ap_affine_v187,  inv_affine_v187  },
    { "AFFINE_A189",  search_affine_v189,  ap_affine_v189,  inv_affine_v189  },
    { "AFFINE_A191",  search_affine_v191,  ap_affine_v191,  inv_affine_v191  },
    { "AFFINE_A193",  search_affine_v193,  ap_affine_v193,  inv_affine_v193  },
    { "AFFINE_A195",  search_affine_v195,  ap_affine_v195,  inv_affine_v195  },
    { "AFFINE_A197",  search_affine_v197,  ap_affine_v197,  inv_affine_v197  },
    { "AFFINE_A199",  search_affine_v199,  ap_affine_v199,  inv_affine_v199  },
    { "AFFINE_A201",  search_affine_v201,  ap_affine_v201,  inv_affine_v201  },
    { "AFFINE_A203",  search_affine_v203,  ap_affine_v203,  inv_affine_v203  },
    { "AFFINE_A205",  search_affine_v205,  ap_affine_v205,  inv_affine_v205  },
    { "AFFINE_A207",  search_affine_v207,  ap_affine_v207,  inv_affine_v207  },
    { "AFFINE_A209",  search_affine_v209,  ap_affine_v209,  inv_affine_v209  },
    { "AFFINE_A211",  search_affine_v211,  ap_affine_v211,  inv_affine_v211  },
    { "AFFINE_A213",  search_affine_v213,  ap_affine_v213,  inv_affine_v213  },
    { "AFFINE_A215",  search_affine_v215,  ap_affine_v215,  inv_affine_v215  },
    { "AFFINE_A217",  search_affine_v217,  ap_affine_v217,  inv_affine_v217  },
    { "AFFINE_A219",  search_affine_v219,  ap_affine_v219,  inv_affine_v219  },
    { "AFFINE_A221",  search_affine_v221,  ap_affine_v221,  inv_affine_v221  },
    { "AFFINE_A223",  search_affine_v223,  ap_affine_v223,  inv_affine_v223  },
    { "AFFINE_A225",  search_affine_v225,  ap_affine_v225,  inv_affine_v225  },
    { "AFFINE_A227",  search_affine_v227,  ap_affine_v227,  inv_affine_v227  },
    { "AFFINE_A229",  search_affine_v229,  ap_affine_v229,  inv_affine_v229  },
    { "AFFINE_A231",  search_affine_v231,  ap_affine_v231,  inv_affine_v231  },
    { "AFFINE_A233",  search_affine_v233,  ap_affine_v233,  inv_affine_v233  },
    { "AFFINE_A235",  search_affine_v235,  ap_affine_v235,  inv_affine_v235  },
    { "AFFINE_A237",  search_affine_v237,  ap_affine_v237,  inv_affine_v237  },
    { "AFFINE_A239",  search_affine_v239,  ap_affine_v239,  inv_affine_v239  },
    { "AFFINE_A241",  search_affine_v241,  ap_affine_v241,  inv_affine_v241  },
    { "AFFINE_A243",  search_affine_v243,  ap_affine_v243,  inv_affine_v243  },
    { "AFFINE_A245",  search_affine_v245,  ap_affine_v245,  inv_affine_v245  },
    { "AFFINE_A247",  search_affine_v247,  ap_affine_v247,  inv_affine_v247  },
    { "AFFINE_A249",  search_affine_v249,  ap_affine_v249,  inv_affine_v249  },
    { "AFFINE_A251",  search_affine_v251,  ap_affine_v251,  inv_affine_v251  },
    { "AFFINE_A253",  search_affine_v253,  ap_affine_v253,  inv_affine_v253  },
    { "AFFINE_A255",  search_affine_v255,  ap_affine_v255,  inv_affine_v255  },
    { "PRIME659", search_primeP_659, ap_primeP_659, inv_primeP_659 },
    { "PRIME661", search_primeP_661, ap_primeP_661, inv_primeP_661 },
    { "PRIME673", search_primeP_673, ap_primeP_673, inv_primeP_673 },
    { "PRIME677", search_primeP_677, ap_primeP_677, inv_primeP_677 },
    { "PRIME683", search_primeP_683, ap_primeP_683, inv_primeP_683 },
    { "PRIME691", search_primeP_691, ap_primeP_691, inv_primeP_691 },
    { "PRIME701", search_primeP_701, ap_primeP_701, inv_primeP_701 },
    { "PRIME709", search_primeP_709, ap_primeP_709, inv_primeP_709 },
    { "PRIME719", search_primeP_719, ap_primeP_719, inv_primeP_719 },
    { "PRIME727", search_primeP_727, ap_primeP_727, inv_primeP_727 },
    { "PRIME733", search_primeP_733, ap_primeP_733, inv_primeP_733 },
    { "PRIME739", search_primeP_739, ap_primeP_739, inv_primeP_739 },
    { "PRIME743", search_primeP_743, ap_primeP_743, inv_primeP_743 },
    { "PRIME751", search_primeP_751, ap_primeP_751, inv_primeP_751 },
    { "PRIME757", search_primeP_757, ap_primeP_757, inv_primeP_757 },
    { "PRIME761", search_primeP_761, ap_primeP_761, inv_primeP_761 },
    { "PRIME769", search_primeP_769, ap_primeP_769, inv_primeP_769 },
    { "PRIME773", search_primeP_773, ap_primeP_773, inv_primeP_773 },
    { "PRIME787", search_primeP_787, ap_primeP_787, inv_primeP_787 },
    { "PRIME797", search_primeP_797, ap_primeP_797, inv_primeP_797 },
    { "PRIME809", search_primeP_809, ap_primeP_809, inv_primeP_809 },
    { "PRIME811", search_primeP_811, ap_primeP_811, inv_primeP_811 },
    { "PRIME821", search_primeP_821, ap_primeP_821, inv_primeP_821 },
    { "PRIME823", search_primeP_823, ap_primeP_823, inv_primeP_823 },
    { "PRIME827", search_primeP_827, ap_primeP_827, inv_primeP_827 },
    { "PRIME829", search_primeP_829, ap_primeP_829, inv_primeP_829 },
    { "PRIME839", search_primeP_839, ap_primeP_839, inv_primeP_839 },
    { "PRIME853", search_primeP_853, ap_primeP_853, inv_primeP_853 },
    { "PRIME857", search_primeP_857, ap_primeP_857, inv_primeP_857 },
    { "PRIME859", search_primeP_859, ap_primeP_859, inv_primeP_859 },
    { "PRIME863", search_primeP_863, ap_primeP_863, inv_primeP_863 },
    { "PRIME877", search_primeP_877, ap_primeP_877, inv_primeP_877 },
    { "PRIME881", search_primeP_881, ap_primeP_881, inv_primeP_881 },
    { "PRIME883", search_primeP_883, ap_primeP_883, inv_primeP_883 },
    { "PRIME887", search_primeP_887, ap_primeP_887, inv_primeP_887 },
    { "PRIME907", search_primeP_907, ap_primeP_907, inv_primeP_907 },
    { "PRIME911", search_primeP_911, ap_primeP_911, inv_primeP_911 },
    { "PRIME919", search_primeP_919, ap_primeP_919, inv_primeP_919 },
    { "PRIME929", search_primeP_929, ap_primeP_929, inv_primeP_929 },
    { "PRIME937", search_primeP_937, ap_primeP_937, inv_primeP_937 },
    { "PRIME941", search_primeP_941, ap_primeP_941, inv_primeP_941 },
    { "PRIME947", search_primeP_947, ap_primeP_947, inv_primeP_947 },
    { "PRIME953", search_primeP_953, ap_primeP_953, inv_primeP_953 },
    { "PRIME967", search_primeP_967, ap_primeP_967, inv_primeP_967 },
    { "PRIME971", search_primeP_971, ap_primeP_971, inv_primeP_971 },
    { "PRIME977", search_primeP_977, ap_primeP_977, inv_primeP_977 },
    { "PRIME983", search_primeP_983, ap_primeP_983, inv_primeP_983 },
    { "PRIME991", search_primeP_991, ap_primeP_991, inv_primeP_991 },
    { "PRIME997", search_primeP_997, ap_primeP_997, inv_primeP_997 },
    { "PRIME1009", search_primeP_1009, ap_primeP_1009, inv_primeP_1009 },
    { "PRIME1013", search_primeP_1013, ap_primeP_1013, inv_primeP_1013 },
    { "PRIME1019", search_primeP_1019, ap_primeP_1019, inv_primeP_1019 },
    { "PRIME1021", search_primeP_1021, ap_primeP_1021, inv_primeP_1021 },
    { "PRIME1031", search_primeP_1031, ap_primeP_1031, inv_primeP_1031 },
    { "PRIME1033", search_primeP_1033, ap_primeP_1033, inv_primeP_1033 },
    { "PRIME1039", search_primeP_1039, ap_primeP_1039, inv_primeP_1039 },
    { "PRIME1049", search_primeP_1049, ap_primeP_1049, inv_primeP_1049 },
    { "PRIME1051", search_primeP_1051, ap_primeP_1051, inv_primeP_1051 },
    { "PRIME1061", search_primeP_1061, ap_primeP_1061, inv_primeP_1061 },
    { "PRIME1063", search_primeP_1063, ap_primeP_1063, inv_primeP_1063 },
    { "QR_XOR_MOD83", search_qrxor_mod83, ap_qrxor_mod83, ap_qrxor_mod83 }, /* self-inverse */
    { "QR_ADD_MOD83", search_qradd_mod83, ap_qradd_mod83, inv_qradd_mod83 },
    { "QR_XOR_MOD89", search_qrxor_mod89, ap_qrxor_mod89, ap_qrxor_mod89 }, /* self-inverse */
    { "QR_ADD_MOD89", search_qradd_mod89, ap_qradd_mod89, inv_qradd_mod89 },
    { "QR_XOR_MOD97", search_qrxor_mod97, ap_qrxor_mod97, ap_qrxor_mod97 }, /* self-inverse */
    { "QR_ADD_MOD97", search_qradd_mod97, ap_qradd_mod97, inv_qradd_mod97 },
    { "QR_XOR_MOD101", search_qrxor_mod101, ap_qrxor_mod101, ap_qrxor_mod101 }, /* self-inverse */
    { "QR_ADD_MOD101", search_qradd_mod101, ap_qradd_mod101, inv_qradd_mod101 },
    { "QR_XOR_MOD103", search_qrxor_mod103, ap_qrxor_mod103, ap_qrxor_mod103 }, /* self-inverse */
    { "QR_ADD_MOD103", search_qradd_mod103, ap_qradd_mod103, inv_qradd_mod103 },
    { "QR_XOR_MOD107", search_qrxor_mod107, ap_qrxor_mod107, ap_qrxor_mod107 }, /* self-inverse */
    { "QR_ADD_MOD107", search_qradd_mod107, ap_qradd_mod107, inv_qradd_mod107 },
    { "QR_XOR_MOD109", search_qrxor_mod109, ap_qrxor_mod109, ap_qrxor_mod109 }, /* self-inverse */
    { "QR_ADD_MOD109", search_qradd_mod109, ap_qradd_mod109, inv_qradd_mod109 },
    { "QR_XOR_MOD113", search_qrxor_mod113, ap_qrxor_mod113, ap_qrxor_mod113 }, /* self-inverse */
    { "QR_ADD_MOD113", search_qradd_mod113, ap_qradd_mod113, inv_qradd_mod113 },
    { "QR_XOR_MOD127", search_qrxor_mod127, ap_qrxor_mod127, ap_qrxor_mod127 }, /* self-inverse */
    { "QR_ADD_MOD127", search_qradd_mod127, ap_qradd_mod127, inv_qradd_mod127 },
    { "QR_XOR_MOD131", search_qrxor_mod131, ap_qrxor_mod131, ap_qrxor_mod131 }, /* self-inverse */
    { "QR_ADD_MOD131", search_qradd_mod131, ap_qradd_mod131, inv_qradd_mod131 },
    { "QR_XOR_MOD137", search_qrxor_mod137, ap_qrxor_mod137, ap_qrxor_mod137 }, /* self-inverse */
    { "QR_ADD_MOD137", search_qradd_mod137, ap_qradd_mod137, inv_qradd_mod137 },
    { "QR_XOR_MOD139", search_qrxor_mod139, ap_qrxor_mod139, ap_qrxor_mod139 }, /* self-inverse */
    { "QR_ADD_MOD139", search_qradd_mod139, ap_qradd_mod139, inv_qradd_mod139 },
    { "QR_XOR_MOD149", search_qrxor_mod149, ap_qrxor_mod149, ap_qrxor_mod149 }, /* self-inverse */
    { "QR_ADD_MOD149", search_qradd_mod149, ap_qradd_mod149, inv_qradd_mod149 },
    { "QR_XOR_MOD151", search_qrxor_mod151, ap_qrxor_mod151, ap_qrxor_mod151 }, /* self-inverse */
    { "QR_ADD_MOD151", search_qradd_mod151, ap_qradd_mod151, inv_qradd_mod151 },
    { "QR_XOR_MOD157", search_qrxor_mod157, ap_qrxor_mod157, ap_qrxor_mod157 }, /* self-inverse */
    { "QR_ADD_MOD157", search_qradd_mod157, ap_qradd_mod157, inv_qradd_mod157 },
    { "QR_XOR_MOD163", search_qrxor_mod163, ap_qrxor_mod163, ap_qrxor_mod163 }, /* self-inverse */
    { "QR_ADD_MOD163", search_qradd_mod163, ap_qradd_mod163, inv_qradd_mod163 },
    { "QR_XOR_MOD167", search_qrxor_mod167, ap_qrxor_mod167, ap_qrxor_mod167 }, /* self-inverse */
    { "QR_ADD_MOD167", search_qradd_mod167, ap_qradd_mod167, inv_qradd_mod167 },
    { "QR_XOR_MOD173", search_qrxor_mod173, ap_qrxor_mod173, ap_qrxor_mod173 }, /* self-inverse */
    { "QR_ADD_MOD173", search_qradd_mod173, ap_qradd_mod173, inv_qradd_mod173 },
    { "QR_XOR_MOD179", search_qrxor_mod179, ap_qrxor_mod179, ap_qrxor_mod179 }, /* self-inverse */
    { "QR_ADD_MOD179", search_qradd_mod179, ap_qradd_mod179, inv_qradd_mod179 },
    { "QR_XOR_MOD181", search_qrxor_mod181, ap_qrxor_mod181, ap_qrxor_mod181 }, /* self-inverse */
    { "QR_ADD_MOD181", search_qradd_mod181, ap_qradd_mod181, inv_qradd_mod181 },
    { "QR_XOR_MOD191", search_qrxor_mod191, ap_qrxor_mod191, ap_qrxor_mod191 }, /* self-inverse */
    { "QR_ADD_MOD191", search_qradd_mod191, ap_qradd_mod191, inv_qradd_mod191 },
    { "QR_XOR_MOD193", search_qrxor_mod193, ap_qrxor_mod193, ap_qrxor_mod193 }, /* self-inverse */
    { "QR_ADD_MOD193", search_qradd_mod193, ap_qradd_mod193, inv_qradd_mod193 },
    { "QR_XOR_MOD197", search_qrxor_mod197, ap_qrxor_mod197, ap_qrxor_mod197 }, /* self-inverse */
    { "QR_ADD_MOD197", search_qradd_mod197, ap_qradd_mod197, inv_qradd_mod197 },
    { "QR_XOR_MOD199", search_qrxor_mod199, ap_qrxor_mod199, ap_qrxor_mod199 }, /* self-inverse */
    { "QR_ADD_MOD199", search_qradd_mod199, ap_qradd_mod199, inv_qradd_mod199 },
    { "QR_XOR_MOD211", search_qrxor_mod211, ap_qrxor_mod211, ap_qrxor_mod211 }, /* self-inverse */
    { "QR_ADD_MOD211", search_qradd_mod211, ap_qradd_mod211, inv_qradd_mod211 },
    { "QR_XOR_MOD223", search_qrxor_mod223, ap_qrxor_mod223, ap_qrxor_mod223 }, /* self-inverse */
    { "QR_ADD_MOD223", search_qradd_mod223, ap_qradd_mod223, inv_qradd_mod223 },
    { "QR_XOR_MOD227", search_qrxor_mod227, ap_qrxor_mod227, ap_qrxor_mod227 }, /* self-inverse */
    { "QR_ADD_MOD227", search_qradd_mod227, ap_qradd_mod227, inv_qradd_mod227 },
    { "QR_XOR_MOD229", search_qrxor_mod229, ap_qrxor_mod229, ap_qrxor_mod229 }, /* self-inverse */
    { "QR_ADD_MOD229", search_qradd_mod229, ap_qradd_mod229, inv_qradd_mod229 },
    { "QR_XOR_MOD233", search_qrxor_mod233, ap_qrxor_mod233, ap_qrxor_mod233 }, /* self-inverse */
    { "QR_ADD_MOD233", search_qradd_mod233, ap_qradd_mod233, inv_qradd_mod233 },
    { "QR_XOR_MOD239", search_qrxor_mod239, ap_qrxor_mod239, ap_qrxor_mod239 }, /* self-inverse */
    { "QR_ADD_MOD239", search_qradd_mod239, ap_qradd_mod239, inv_qradd_mod239 },
    { "VALUE_MUL_T012", search_valuemul_t012, ap_valuemul_t012, inv_valuemul_t012 },
    { "VALUE_MUL_T013", search_valuemul_t013, ap_valuemul_t013, inv_valuemul_t013 },
    { "VALUE_MUL_T014", search_valuemul_t014, ap_valuemul_t014, inv_valuemul_t014 },
    { "VALUE_MUL_T015", search_valuemul_t015, ap_valuemul_t015, inv_valuemul_t015 },
    { "VALUE_MUL_T016", search_valuemul_t016, ap_valuemul_t016, inv_valuemul_t016 },
    { "VALUE_MUL_T017", search_valuemul_t017, ap_valuemul_t017, inv_valuemul_t017 },
    { "VALUE_MUL_T023", search_valuemul_t023, ap_valuemul_t023, inv_valuemul_t023 },
    { "VALUE_MUL_T024", search_valuemul_t024, ap_valuemul_t024, inv_valuemul_t024 },
    { "VALUE_MUL_T025", search_valuemul_t025, ap_valuemul_t025, inv_valuemul_t025 },
    { "VALUE_MUL_T026", search_valuemul_t026, ap_valuemul_t026, inv_valuemul_t026 },
    { "VALUE_MUL_T027", search_valuemul_t027, ap_valuemul_t027, inv_valuemul_t027 },
    { "VALUE_MUL_T034", search_valuemul_t034, ap_valuemul_t034, inv_valuemul_t034 },
    { "VALUE_MUL_T035", search_valuemul_t035, ap_valuemul_t035, inv_valuemul_t035 },
    { "VALUE_MUL_T036", search_valuemul_t036, ap_valuemul_t036, inv_valuemul_t036 },
    { "VALUE_MUL_T037", search_valuemul_t037, ap_valuemul_t037, inv_valuemul_t037 },
    { "VALUE_MUL_T045", search_valuemul_t045, ap_valuemul_t045, inv_valuemul_t045 },
    { "VALUE_MUL_T046", search_valuemul_t046, ap_valuemul_t046, inv_valuemul_t046 },
    { "VALUE_MUL_T047", search_valuemul_t047, ap_valuemul_t047, inv_valuemul_t047 },
    { "VALUE_MUL_T056", search_valuemul_t056, ap_valuemul_t056, inv_valuemul_t056 },
    { "VALUE_MUL_T057", search_valuemul_t057, ap_valuemul_t057, inv_valuemul_t057 },
    { "VALUE_MUL_T067", search_valuemul_t067, ap_valuemul_t067, inv_valuemul_t067 },
    { "VALUE_MUL_T123", search_valuemul_t123, ap_valuemul_t123, inv_valuemul_t123 },
    { "VALUE_MUL_T124", search_valuemul_t124, ap_valuemul_t124, inv_valuemul_t124 },
    { "VALUE_MUL_T125", search_valuemul_t125, ap_valuemul_t125, inv_valuemul_t125 },
    { "VALUE_MUL_T126", search_valuemul_t126, ap_valuemul_t126, inv_valuemul_t126 },
    { "VALUE_MUL_T127", search_valuemul_t127, ap_valuemul_t127, inv_valuemul_t127 },
    { "VALUE_MUL_T134", search_valuemul_t134, ap_valuemul_t134, inv_valuemul_t134 },
    { "VALUE_MUL_T135", search_valuemul_t135, ap_valuemul_t135, inv_valuemul_t135 },
    { "VALUE_MUL_T136", search_valuemul_t136, ap_valuemul_t136, inv_valuemul_t136 },
    { "VALUE_MUL_T137", search_valuemul_t137, ap_valuemul_t137, inv_valuemul_t137 },
    { "VALUE_MUL_T145", search_valuemul_t145, ap_valuemul_t145, inv_valuemul_t145 },
    { "VALUE_MUL_T146", search_valuemul_t146, ap_valuemul_t146, inv_valuemul_t146 },
    { "BIT_CXOR3_T013", search_bitcxor3_t013, ap_bitcxor3_t013, ap_bitcxor3_t013 }, /* self-inverse */
    { "BIT_CXOR3_T014", search_bitcxor3_t014, ap_bitcxor3_t014, ap_bitcxor3_t014 }, /* self-inverse */
    { "BIT_CXOR3_T015", search_bitcxor3_t015, ap_bitcxor3_t015, ap_bitcxor3_t015 }, /* self-inverse */
    { "BIT_CXOR3_T016", search_bitcxor3_t016, ap_bitcxor3_t016, ap_bitcxor3_t016 }, /* self-inverse */
    { "BIT_CXOR3_T017", search_bitcxor3_t017, ap_bitcxor3_t017, ap_bitcxor3_t017 }, /* self-inverse */
    { "BIT_CXOR3_T023", search_bitcxor3_t023, ap_bitcxor3_t023, ap_bitcxor3_t023 }, /* self-inverse */
    { "BIT_CXOR3_T024", search_bitcxor3_t024, ap_bitcxor3_t024, ap_bitcxor3_t024 }, /* self-inverse */
    { "BIT_CXOR3_T026", search_bitcxor3_t026, ap_bitcxor3_t026, ap_bitcxor3_t026 }, /* self-inverse */
    { "BIT_CXOR3_T027", search_bitcxor3_t027, ap_bitcxor3_t027, ap_bitcxor3_t027 }, /* self-inverse */
    { "BIT_CXOR3_T034", search_bitcxor3_t034, ap_bitcxor3_t034, ap_bitcxor3_t034 }, /* self-inverse */
    { "BIT_CXOR3_T035", search_bitcxor3_t035, ap_bitcxor3_t035, ap_bitcxor3_t035 }, /* self-inverse */
    { "BIT_CXOR3_T036", search_bitcxor3_t036, ap_bitcxor3_t036, ap_bitcxor3_t036 }, /* self-inverse */
    { "BIT_CXOR3_T037", search_bitcxor3_t037, ap_bitcxor3_t037, ap_bitcxor3_t037 }, /* self-inverse */
    { "BIT_CXOR3_T046", search_bitcxor3_t046, ap_bitcxor3_t046, ap_bitcxor3_t046 }, /* self-inverse */
    { "BIT_CXOR3_T047", search_bitcxor3_t047, ap_bitcxor3_t047, ap_bitcxor3_t047 }, /* self-inverse */
    { "BIT_CXOR3_T056", search_bitcxor3_t056, ap_bitcxor3_t056, ap_bitcxor3_t056 }, /* self-inverse */
    { "BIT_CXOR3_T057", search_bitcxor3_t057, ap_bitcxor3_t057, ap_bitcxor3_t057 }, /* self-inverse */
    { "BIT_CXOR3_T067", search_bitcxor3_t067, ap_bitcxor3_t067, ap_bitcxor3_t067 }, /* self-inverse */
    { "BIT_CXOR3_T123", search_bitcxor3_t123, ap_bitcxor3_t123, ap_bitcxor3_t123 }, /* self-inverse */
    { "BIT_CXOR3_T124", search_bitcxor3_t124, ap_bitcxor3_t124, ap_bitcxor3_t124 }, /* self-inverse */
    { "BIT_CXOR3_T125", search_bitcxor3_t125, ap_bitcxor3_t125, ap_bitcxor3_t125 }, /* self-inverse */
    { "BIT_CXOR3_T126", search_bitcxor3_t126, ap_bitcxor3_t126, ap_bitcxor3_t126 }, /* self-inverse */
    { "BIT_CXOR3_T127", search_bitcxor3_t127, ap_bitcxor3_t127, ap_bitcxor3_t127 }, /* self-inverse */
    { "BIT_CXOR3_T134", search_bitcxor3_t134, ap_bitcxor3_t134, ap_bitcxor3_t134 }, /* self-inverse */
    { "BIT_CXOR3_T136", search_bitcxor3_t136, ap_bitcxor3_t136, ap_bitcxor3_t136 }, /* self-inverse */
    { "BIT_CXOR3_T137", search_bitcxor3_t137, ap_bitcxor3_t137, ap_bitcxor3_t137 }, /* self-inverse */
    { "BIT_CXOR3_T145", search_bitcxor3_t145, ap_bitcxor3_t145, ap_bitcxor3_t145 }, /* self-inverse */
    { "BIT_CXOR3_T146", search_bitcxor3_t146, ap_bitcxor3_t146, ap_bitcxor3_t146 }, /* self-inverse */
    { "BIT_CXOR3_T147", search_bitcxor3_t147, ap_bitcxor3_t147, ap_bitcxor3_t147 }, /* self-inverse */
    { "BIT_CXOR3_T156", search_bitcxor3_t156, ap_bitcxor3_t156, ap_bitcxor3_t156 }, /* self-inverse */
    { "RANK_XOR_KO1", search_rankxor_ko1, ap_rankxor_ko1, ap_rankxor_ko1 }, /* self-inverse */
    { "RANK_XOR_KO3", search_rankxor_ko3, ap_rankxor_ko3, ap_rankxor_ko3 }, /* self-inverse */
    { "RANK_XOR_KO5", search_rankxor_ko5, ap_rankxor_ko5, ap_rankxor_ko5 }, /* self-inverse */
    { "RANK_XOR_KO7", search_rankxor_ko7, ap_rankxor_ko7, ap_rankxor_ko7 }, /* self-inverse */
    { "RANK_XOR_KO9", search_rankxor_ko9, ap_rankxor_ko9, ap_rankxor_ko9 }, /* self-inverse */
    { "RANK_XOR_KO11", search_rankxor_ko11, ap_rankxor_ko11, ap_rankxor_ko11 }, /* self-inverse */
    { "RANK_XOR_KO13", search_rankxor_ko13, ap_rankxor_ko13, ap_rankxor_ko13 }, /* self-inverse */
    { "RANK_XOR_KO15", search_rankxor_ko15, ap_rankxor_ko15, ap_rankxor_ko15 }, /* self-inverse */
    { "RANK_ADD_KO1", search_rankadd_ko1, ap_rankadd_ko1, inv_rankadd_ko1 },
    { "RANK_ADD_KO3", search_rankadd_ko3, ap_rankadd_ko3, inv_rankadd_ko3 },
    { "RANK_ADD_KO5", search_rankadd_ko5, ap_rankadd_ko5, inv_rankadd_ko5 },
    { "RANK_ADD_KO7", search_rankadd_ko7, ap_rankadd_ko7, inv_rankadd_ko7 },
    { "RANK_ADD_KO9", search_rankadd_ko9, ap_rankadd_ko9, inv_rankadd_ko9 },
    { "RANK_ADD_KO11", search_rankadd_ko11, ap_rankadd_ko11, inv_rankadd_ko11 },
    { "RANK_ADD_KO13", search_rankadd_ko13, ap_rankadd_ko13, inv_rankadd_ko13 },
    { "RANK_ADD_KO15", search_rankadd_ko15, ap_rankadd_ko15, inv_rankadd_ko15 },
    { "RANGE_ADD_T2", search_rangeadd_t2, ap_rangeadd_t2, inv_rangeadd_t2 },
    { "RANGE_ADD_T3", search_rangeadd_t3, ap_rangeadd_t3, inv_rangeadd_t3 },
    { "RANGE_ADD_T4", search_rangeadd_t4, ap_rangeadd_t4, inv_rangeadd_t4 },
    { "RANGE_ADD_T5", search_rangeadd_t5, ap_rangeadd_t5, inv_rangeadd_t5 },
    { "RANGE_ADD_T6", search_rangeadd_t6, ap_rangeadd_t6, inv_rangeadd_t6 },
    { "RANGE_ADD_T7", search_rangeadd_t7, ap_rangeadd_t7, inv_rangeadd_t7 },
    { "MAGCLASS_ADD_T2", search_magclassadd_t2, ap_magclassadd_t2, inv_magclassadd_t2 },
    { "MAGCLASS_ADD_T3", search_magclassadd_t3, ap_magclassadd_t3, inv_magclassadd_t3 },
    { "MAGCLASS_ADD_T4", search_magclassadd_t4, ap_magclassadd_t4, inv_magclassadd_t4 },
    { "POPCNT_ADD_T4", search_popcntadd_t4, ap_popcntadd_t4, inv_popcntadd_t4 },
    { "POPCNT_ADD_T5", search_popcntadd_t5, ap_popcntadd_t5, inv_popcntadd_t5 },
    { "POPCNT_ADD_T6", search_popcntadd_t6, ap_popcntadd_t6, inv_popcntadd_t6 },
    { "NIB_POW_M74", search_nibpow_m74, ap_nibpow_m74, inv_nibpow_m74 },
    { "NIB_POW_M78", search_nibpow_m78, ap_nibpow_m78, inv_nibpow_m78 },
    { "PATTERN3_XOR", search_patternxor_n3, ap_patternxor_n3, ap_patternxor_n3 }, /* self-inverse */
    { "PATTERN5_XOR", search_patternxor_n5, ap_patternxor_n5, ap_patternxor_n5 }, /* self-inverse */
    { "PATTERN6_XOR", search_patternxor_n6, ap_patternxor_n6, ap_patternxor_n6 }, /* self-inverse */
    { "PATTERN7_XOR", search_patternxor_n7, ap_patternxor_n7, ap_patternxor_n7 }, /* self-inverse */
    { "PATTERN9_XOR", search_patternxor_n9, ap_patternxor_n9, ap_patternxor_n9 }, /* self-inverse */
    { "PATTERN10_XOR", search_patternxor_n10, ap_patternxor_n10, ap_patternxor_n10 }, /* self-inverse */
    { "PATTERN11_XOR", search_patternxor_n11, ap_patternxor_n11, ap_patternxor_n11 }, /* self-inverse */
    { "PATTERN12_XOR", search_patternxor_n12, ap_patternxor_n12, ap_patternxor_n12 }, /* self-inverse */
    { "PATTERN13_XOR", search_patternxor_n13, ap_patternxor_n13, ap_patternxor_n13 }, /* self-inverse */
    { "PATTERN14_XOR", search_patternxor_n14, ap_patternxor_n14, ap_patternxor_n14 }, /* self-inverse */
    { "PATTERN15_XOR", search_patternxor_n15, ap_patternxor_n15, ap_patternxor_n15 }, /* self-inverse */
    { "PATTERN17_XOR", search_patternxor_n17, ap_patternxor_n17, ap_patternxor_n17 }, /* self-inverse */
    { "PATTERN18_XOR", search_patternxor_n18, ap_patternxor_n18, ap_patternxor_n18 }, /* self-inverse */
    { "PATTERN19_XOR", search_patternxor_n19, ap_patternxor_n19, ap_patternxor_n19 }, /* self-inverse */
    { "PATTERN20_XOR", search_patternxor_n20, ap_patternxor_n20, ap_patternxor_n20 }, /* self-inverse */
    { "PATTERN21_XOR", search_patternxor_n21, ap_patternxor_n21, ap_patternxor_n21 }, /* self-inverse */
    { "PATTERN22_XOR", search_patternxor_n22, ap_patternxor_n22, ap_patternxor_n22 }, /* self-inverse */
    { "PATTERN23_XOR", search_patternxor_n23, ap_patternxor_n23, ap_patternxor_n23 }, /* self-inverse */
    { "PATTERN24_XOR", search_patternxor_n24, ap_patternxor_n24, ap_patternxor_n24 }, /* self-inverse */
    { "PRIME1069", search_primeP_1069, ap_primeP_1069, inv_primeP_1069 },
    { "PRIME1087", search_primeP_1087, ap_primeP_1087, inv_primeP_1087 },
    { "PRIME1091", search_primeP_1091, ap_primeP_1091, inv_primeP_1091 },
    { "PRIME1093", search_primeP_1093, ap_primeP_1093, inv_primeP_1093 },
    { "PRIME1097", search_primeP_1097, ap_primeP_1097, inv_primeP_1097 },
    { "PRIME1103", search_primeP_1103, ap_primeP_1103, inv_primeP_1103 },
    { "PRIME1109", search_primeP_1109, ap_primeP_1109, inv_primeP_1109 },
    { "PRIME1117", search_primeP_1117, ap_primeP_1117, inv_primeP_1117 },
    { "PRIME1123", search_primeP_1123, ap_primeP_1123, inv_primeP_1123 },
    { "PRIME1129", search_primeP_1129, ap_primeP_1129, inv_primeP_1129 },
    { "PRIME1151", search_primeP_1151, ap_primeP_1151, inv_primeP_1151 },
    { "PRIME1153", search_primeP_1153, ap_primeP_1153, inv_primeP_1153 },
    { "PRIME1163", search_primeP_1163, ap_primeP_1163, inv_primeP_1163 },
    { "PRIME1171", search_primeP_1171, ap_primeP_1171, inv_primeP_1171 },
    { "PRIME1181", search_primeP_1181, ap_primeP_1181, inv_primeP_1181 },
    { "PRIME1187", search_primeP_1187, ap_primeP_1187, inv_primeP_1187 },
    { "PRIME1193", search_primeP_1193, ap_primeP_1193, inv_primeP_1193 },
    { "PRIME1201", search_primeP_1201, ap_primeP_1201, inv_primeP_1201 },
    { "PRIME1213", search_primeP_1213, ap_primeP_1213, inv_primeP_1213 },
    { "PRIME1217", search_primeP_1217, ap_primeP_1217, inv_primeP_1217 },
    { "PRIME1223", search_primeP_1223, ap_primeP_1223, inv_primeP_1223 },
    { "PRIME1229", search_primeP_1229, ap_primeP_1229, inv_primeP_1229 },
    { "PRIME1231", search_primeP_1231, ap_primeP_1231, inv_primeP_1231 },
    { "PRIME1237", search_primeP_1237, ap_primeP_1237, inv_primeP_1237 },
    { "PRIME1249", search_primeP_1249, ap_primeP_1249, inv_primeP_1249 },
    { "PRIME1259", search_primeP_1259, ap_primeP_1259, inv_primeP_1259 },
    { "PRIME1277", search_primeP_1277, ap_primeP_1277, inv_primeP_1277 },
    { "PRIME1279", search_primeP_1279, ap_primeP_1279, inv_primeP_1279 },
    { "PRIME1283", search_primeP_1283, ap_primeP_1283, inv_primeP_1283 },
    { "PRIME1289", search_primeP_1289, ap_primeP_1289, inv_primeP_1289 },
    { "PRIME1291", search_primeP_1291, ap_primeP_1291, inv_primeP_1291 },
    { "PRIME1297", search_primeP_1297, ap_primeP_1297, inv_primeP_1297 },
    { "PRIME1301", search_primeP_1301, ap_primeP_1301, inv_primeP_1301 },
    { "PRIME1303", search_primeP_1303, ap_primeP_1303, inv_primeP_1303 },
    { "PRIME1307", search_primeP_1307, ap_primeP_1307, inv_primeP_1307 },
    { "PRIME1319", search_primeP_1319, ap_primeP_1319, inv_primeP_1319 },
    { "PRIME1321", search_primeP_1321, ap_primeP_1321, inv_primeP_1321 },
    { "PRIME1327", search_primeP_1327, ap_primeP_1327, inv_primeP_1327 },
    { "PRIME1361", search_primeP_1361, ap_primeP_1361, inv_primeP_1361 },
    { "PRIME1367", search_primeP_1367, ap_primeP_1367, inv_primeP_1367 },
    { "PRIME1373", search_primeP_1373, ap_primeP_1373, inv_primeP_1373 },
    { "PRIME1381", search_primeP_1381, ap_primeP_1381, inv_primeP_1381 },
    { "PRIME1399", search_primeP_1399, ap_primeP_1399, inv_primeP_1399 },
    { "PRIME1409", search_primeP_1409, ap_primeP_1409, inv_primeP_1409 },
    { "PRIME1423", search_primeP_1423, ap_primeP_1423, inv_primeP_1423 },
    { "PRIME1427", search_primeP_1427, ap_primeP_1427, inv_primeP_1427 },
    { "PRIME1429", search_primeP_1429, ap_primeP_1429, inv_primeP_1429 },
    { "PRIME1433", search_primeP_1433, ap_primeP_1433, inv_primeP_1433 },
    { "PRIME1439", search_primeP_1439, ap_primeP_1439, inv_primeP_1439 },
    { "PRIME1447", search_primeP_1447, ap_primeP_1447, inv_primeP_1447 },
    { "PRIME1451", search_primeP_1451, ap_primeP_1451, inv_primeP_1451 },
    { "PRIME1453", search_primeP_1453, ap_primeP_1453, inv_primeP_1453 },
    { "PRIME1459", search_primeP_1459, ap_primeP_1459, inv_primeP_1459 },
    { "PRIME1471", search_primeP_1471, ap_primeP_1471, inv_primeP_1471 },
    { "PRIME1481", search_primeP_1481, ap_primeP_1481, inv_primeP_1481 },
    { "PRIME1483", search_primeP_1483, ap_primeP_1483, inv_primeP_1483 },
    { "PRIME1487", search_primeP_1487, ap_primeP_1487, inv_primeP_1487 },
    { "PRIME1489", search_primeP_1489, ap_primeP_1489, inv_primeP_1489 },
    { "PRIME1493", search_primeP_1493, ap_primeP_1493, inv_primeP_1493 },
    { "PRIME1499", search_primeP_1499, ap_primeP_1499, inv_primeP_1499 },
    { "PRIME1511", search_primeP_1511, ap_primeP_1511, inv_primeP_1511 },
    { "PRIME1523", search_primeP_1523, ap_primeP_1523, inv_primeP_1523 },
    { "PRIME1531", search_primeP_1531, ap_primeP_1531, inv_primeP_1531 },
    { "PRIME1543", search_primeP_1543, ap_primeP_1543, inv_primeP_1543 },
    { "PRNG_PERM_Y3", search_prngperm_y3, ap_prngperm_y3, inv_prngperm_y3 },
    { "PRNG_PERM_Y4", search_prngperm_y4, ap_prngperm_y4, inv_prngperm_y4 },
    { "PRNG_PERM_Y5", search_prngperm_y5, ap_prngperm_y5, inv_prngperm_y5 },
    { "PRNG_PERM_Y6", search_prngperm_y6, ap_prngperm_y6, inv_prngperm_y6 },
    { "PRNG_PERM_Y7", search_prngperm_y7, ap_prngperm_y7, inv_prngperm_y7 },
    { "PRNG_PERM_Y8", search_prngperm_y8, ap_prngperm_y8, inv_prngperm_y8 },
    { "PRNG_PERM_Y9", search_prngperm_y9, ap_prngperm_y9, inv_prngperm_y9 },
    { "PRNG_PERM_Y10", search_prngperm_y10, ap_prngperm_y10, inv_prngperm_y10 },
    { "PRNG_PERM_Y11", search_prngperm_y11, ap_prngperm_y11, inv_prngperm_y11 },
    { "PRNG_PERM_Y12", search_prngperm_y12, ap_prngperm_y12, inv_prngperm_y12 },
    { "PRNG_PERM_Y13", search_prngperm_y13, ap_prngperm_y13, inv_prngperm_y13 },
    { "PRNG_PERM_Y14", search_prngperm_y14, ap_prngperm_y14, inv_prngperm_y14 },
    { "PRNG_PERM_Y15", search_prngperm_y15, ap_prngperm_y15, inv_prngperm_y15 },
    { "PRNG_PERM_Y16", search_prngperm_y16, ap_prngperm_y16, inv_prngperm_y16 },
    { "PRNG_PERM_Y17", search_prngperm_y17, ap_prngperm_y17, inv_prngperm_y17 },
    { "PRNG_PERM_Y18", search_prngperm_y18, ap_prngperm_y18, inv_prngperm_y18 },
    { "PRNG_PERM_Y19", search_prngperm_y19, ap_prngperm_y19, inv_prngperm_y19 },
    { "PRNG_PERM_Y20", search_prngperm_y20, ap_prngperm_y20, inv_prngperm_y20 },
    { "PRNG_PERM_Y21", search_prngperm_y21, ap_prngperm_y21, inv_prngperm_y21 },
    { "PRNG_PERM_Y22", search_prngperm_y22, ap_prngperm_y22, inv_prngperm_y22 },
    { "PRNG_PERM_Y23", search_prngperm_y23, ap_prngperm_y23, inv_prngperm_y23 },
    { "PRNG_PERM_Y24", search_prngperm_y24, ap_prngperm_y24, inv_prngperm_y24 },
    { "PRNG_PERM_Y25", search_prngperm_y25, ap_prngperm_y25, inv_prngperm_y25 },
    { "PRNG_PERM_Y26", search_prngperm_y26, ap_prngperm_y26, inv_prngperm_y26 },
    { "PRNG_PERM_Y27", search_prngperm_y27, ap_prngperm_y27, inv_prngperm_y27 },
    { "PRNG_PERM_Y28", search_prngperm_y28, ap_prngperm_y28, inv_prngperm_y28 },
    { "PRNG_PERM_Y29", search_prngperm_y29, ap_prngperm_y29, inv_prngperm_y29 },
    { "PRNG_PERM_Y30", search_prngperm_y30, ap_prngperm_y30, inv_prngperm_y30 },
    { "PRNG_PERM_Y31", search_prngperm_y31, ap_prngperm_y31, inv_prngperm_y31 },
    { "PRNG_PERM_Y32", search_prngperm_y32, ap_prngperm_y32, inv_prngperm_y32 },
    { "PRNG_PERM_Y33", search_prngperm_y33, ap_prngperm_y33, inv_prngperm_y33 },
    { "PRNG_PERM_Y34", search_prngperm_y34, ap_prngperm_y34, inv_prngperm_y34 },
    { "PRNG_PERM_Y35", search_prngperm_y35, ap_prngperm_y35, inv_prngperm_y35 },
    { "PRNG_PERM_Y36", search_prngperm_y36, ap_prngperm_y36, inv_prngperm_y36 },
    { "PRNG_PERM_Y37", search_prngperm_y37, ap_prngperm_y37, inv_prngperm_y37 },
    { "PRNG_PERM_Y38", search_prngperm_y38, ap_prngperm_y38, inv_prngperm_y38 },
    { "PRNG_PERM_Y39", search_prngperm_y39, ap_prngperm_y39, inv_prngperm_y39 },
    { "PRNG_PERM_Y40", search_prngperm_y40, ap_prngperm_y40, inv_prngperm_y40 },
    { "PRNG_PERM_Y41", search_prngperm_y41, ap_prngperm_y41, inv_prngperm_y41 },
    { "PRNG_PERM_Y42", search_prngperm_y42, ap_prngperm_y42, inv_prngperm_y42 },
    { "PRNG_PERM_Y43", search_prngperm_y43, ap_prngperm_y43, inv_prngperm_y43 },
    { "PRNG_PERM_Y44", search_prngperm_y44, ap_prngperm_y44, inv_prngperm_y44 },
    { "PRNG_PERM_Y45", search_prngperm_y45, ap_prngperm_y45, inv_prngperm_y45 },
    { "PRNG_PERM_Y46", search_prngperm_y46, ap_prngperm_y46, inv_prngperm_y46 },
    { "PRNG_PERM_Y47", search_prngperm_y47, ap_prngperm_y47, inv_prngperm_y47 },
    { "PRNG_PERM_Y48", search_prngperm_y48, ap_prngperm_y48, inv_prngperm_y48 },
    { "PRNG_PERM_Y49", search_prngperm_y49, ap_prngperm_y49, inv_prngperm_y49 },
    { "PRNG_PERM_Y50", search_prngperm_y50, ap_prngperm_y50, inv_prngperm_y50 },
    { "PRNG_PERM_Y51", search_prngperm_y51, ap_prngperm_y51, inv_prngperm_y51 },
    { "PRNG_PERM_Y52", search_prngperm_y52, ap_prngperm_y52, inv_prngperm_y52 },
    { "PRNG_PERM_Y53", search_prngperm_y53, ap_prngperm_y53, inv_prngperm_y53 },
    { "PRNG_PERM_Y54", search_prngperm_y54, ap_prngperm_y54, inv_prngperm_y54 },
    { "PRNG_PERM_Y55", search_prngperm_y55, ap_prngperm_y55, inv_prngperm_y55 },
    { "PRNG_PERM_Y56", search_prngperm_y56, ap_prngperm_y56, inv_prngperm_y56 },
    { "PRNG_PERM_Y57", search_prngperm_y57, ap_prngperm_y57, inv_prngperm_y57 },
    { "PRNG_PERM_Y58", search_prngperm_y58, ap_prngperm_y58, inv_prngperm_y58 },
    { "PRNG_PERM_Y59", search_prngperm_y59, ap_prngperm_y59, inv_prngperm_y59 },
    { "PRNG_PERM_Y60", search_prngperm_y60, ap_prngperm_y60, inv_prngperm_y60 },
    { "PRNG_PERM_Y61", search_prngperm_y61, ap_prngperm_y61, inv_prngperm_y61 },
    { "PRNG_PERM_Y62", search_prngperm_y62, ap_prngperm_y62, inv_prngperm_y62 },
    { "PRNG_PERM_Y63", search_prngperm_y63, ap_prngperm_y63, inv_prngperm_y63 },
    { "PRNG_PERM_Y64", search_prngperm_y64, ap_prngperm_y64, inv_prngperm_y64 },
    { "NIB_CXOR_M17", search_nibcxor_m17, ap_nibcxor_m17, ap_nibcxor_m17 }, /* self-inverse */
    { "NIB_CXOR_M1B", search_nibcxor_m1b, ap_nibcxor_m1b, ap_nibcxor_m1b }, /* self-inverse */
    { "NIB_CXOR_M1D", search_nibcxor_m1d, ap_nibcxor_m1d, ap_nibcxor_m1d }, /* self-inverse */
    { "NIB_CXOR_M1E", search_nibcxor_m1e, ap_nibcxor_m1e, ap_nibcxor_m1e }, /* self-inverse */
    { "NIB_CXOR_M27", search_nibcxor_m27, ap_nibcxor_m27, ap_nibcxor_m27 }, /* self-inverse */
    { "NIB_CXOR_M2B", search_nibcxor_m2b, ap_nibcxor_m2b, ap_nibcxor_m2b }, /* self-inverse */
    { "NIB_CXOR_M2D", search_nibcxor_m2d, ap_nibcxor_m2d, ap_nibcxor_m2d }, /* self-inverse */
    { "NIB_CXOR_M2E", search_nibcxor_m2e, ap_nibcxor_m2e, ap_nibcxor_m2e }, /* self-inverse */
    { "NIB_CXOR_M33", search_nibcxor_m33, ap_nibcxor_m33, ap_nibcxor_m33 }, /* self-inverse */
    { "NIB_CXOR_M35", search_nibcxor_m35, ap_nibcxor_m35, ap_nibcxor_m35 }, /* self-inverse */
    { "NIB_CXOR_M36", search_nibcxor_m36, ap_nibcxor_m36, ap_nibcxor_m36 }, /* self-inverse */
    { "NIB_CXOR_M39", search_nibcxor_m39, ap_nibcxor_m39, ap_nibcxor_m39 }, /* self-inverse */
    { "NIB_CXOR_M3A", search_nibcxor_m3a, ap_nibcxor_m3a, ap_nibcxor_m3a }, /* self-inverse */
    { "NIB_CXOR_M3C", search_nibcxor_m3c, ap_nibcxor_m3c, ap_nibcxor_m3c }, /* self-inverse */
    { "NIB_CXOR_M47", search_nibcxor_m47, ap_nibcxor_m47, ap_nibcxor_m47 }, /* self-inverse */
    { "NIB_CXOR_M4B", search_nibcxor_m4b, ap_nibcxor_m4b, ap_nibcxor_m4b }, /* self-inverse */
    { "NIB_CXOR_M4D", search_nibcxor_m4d, ap_nibcxor_m4d, ap_nibcxor_m4d }, /* self-inverse */
    { "NIB_CXOR_M4E", search_nibcxor_m4e, ap_nibcxor_m4e, ap_nibcxor_m4e }, /* self-inverse */
    { "NIB_CXOR_M53", search_nibcxor_m53, ap_nibcxor_m53, ap_nibcxor_m53 }, /* self-inverse */
    { "NIB_CXOR_M55", search_nibcxor_m55, ap_nibcxor_m55, ap_nibcxor_m55 }, /* self-inverse */
    { "NIB_CXOR_M56", search_nibcxor_m56, ap_nibcxor_m56, ap_nibcxor_m56 }, /* self-inverse */
    { "NIB_CXOR_M59", search_nibcxor_m59, ap_nibcxor_m59, ap_nibcxor_m59 }, /* self-inverse */
    { "NIB_CXOR_M5A", search_nibcxor_m5a, ap_nibcxor_m5a, ap_nibcxor_m5a }, /* self-inverse */
    { "NIB_CXOR_M5C", search_nibcxor_m5c, ap_nibcxor_m5c, ap_nibcxor_m5c }, /* self-inverse */
    { "NIB_CXOR_M63", search_nibcxor_m63, ap_nibcxor_m63, ap_nibcxor_m63 }, /* self-inverse */
    { "NIB_CXOR_M65", search_nibcxor_m65, ap_nibcxor_m65, ap_nibcxor_m65 }, /* self-inverse */
    { "NIB_CXOR_M66", search_nibcxor_m66, ap_nibcxor_m66, ap_nibcxor_m66 }, /* self-inverse */
    { "NIB_CXOR_M69", search_nibcxor_m69, ap_nibcxor_m69, ap_nibcxor_m69 }, /* self-inverse */
    { "NIB_CXOR_M6A", search_nibcxor_m6a, ap_nibcxor_m6a, ap_nibcxor_m6a }, /* self-inverse */
    { "NIB_CXOR_M6C", search_nibcxor_m6c, ap_nibcxor_m6c, ap_nibcxor_m6c }, /* self-inverse */
    { "NIB_CXOR_M71", search_nibcxor_m71, ap_nibcxor_m71, ap_nibcxor_m71 }, /* self-inverse */
    { "NIB_CXOR_M72", search_nibcxor_m72, ap_nibcxor_m72, ap_nibcxor_m72 }, /* self-inverse */
    { "NIB_CADD_M17", search_nibcadd_m17, ap_nibcadd_m17, inv_nibcadd_m17 },
    { "NIB_CADD_M1B", search_nibcadd_m1b, ap_nibcadd_m1b, inv_nibcadd_m1b },
    { "NIB_CADD_M1D", search_nibcadd_m1d, ap_nibcadd_m1d, inv_nibcadd_m1d },
    { "NIB_CADD_M1E", search_nibcadd_m1e, ap_nibcadd_m1e, inv_nibcadd_m1e },
    { "NIB_CADD_M27", search_nibcadd_m27, ap_nibcadd_m27, inv_nibcadd_m27 },
    { "NIB_CADD_M2B", search_nibcadd_m2b, ap_nibcadd_m2b, inv_nibcadd_m2b },
    { "NIB_CADD_M2D", search_nibcadd_m2d, ap_nibcadd_m2d, inv_nibcadd_m2d },
    { "NIB_CADD_M2E", search_nibcadd_m2e, ap_nibcadd_m2e, inv_nibcadd_m2e },
    { "NIB_CADD_M33", search_nibcadd_m33, ap_nibcadd_m33, inv_nibcadd_m33 },
    { "NIB_CADD_M35", search_nibcadd_m35, ap_nibcadd_m35, inv_nibcadd_m35 },
    { "NIB_CADD_M36", search_nibcadd_m36, ap_nibcadd_m36, inv_nibcadd_m36 },
    { "NIB_CADD_M39", search_nibcadd_m39, ap_nibcadd_m39, inv_nibcadd_m39 },
    { "NIB_CADD_M3A", search_nibcadd_m3a, ap_nibcadd_m3a, inv_nibcadd_m3a },
    { "NIB_CADD_M3C", search_nibcadd_m3c, ap_nibcadd_m3c, inv_nibcadd_m3c },
    { "NIB_CADD_M47", search_nibcadd_m47, ap_nibcadd_m47, inv_nibcadd_m47 },
    { "NIB_CADD_M4B", search_nibcadd_m4b, ap_nibcadd_m4b, inv_nibcadd_m4b },
    { "NIB_CADD_M4D", search_nibcadd_m4d, ap_nibcadd_m4d, inv_nibcadd_m4d },
    { "NIB_CADD_M4E", search_nibcadd_m4e, ap_nibcadd_m4e, inv_nibcadd_m4e },
    { "NIB_CADD_M53", search_nibcadd_m53, ap_nibcadd_m53, inv_nibcadd_m53 },
    { "NIB_CADD_M55", search_nibcadd_m55, ap_nibcadd_m55, inv_nibcadd_m55 },
    { "NIB_CADD_M56", search_nibcadd_m56, ap_nibcadd_m56, inv_nibcadd_m56 },
    { "NIB_CADD_M59", search_nibcadd_m59, ap_nibcadd_m59, inv_nibcadd_m59 },
    { "NIB_CADD_M5A", search_nibcadd_m5a, ap_nibcadd_m5a, inv_nibcadd_m5a },
    { "NIB_CADD_M5C", search_nibcadd_m5c, ap_nibcadd_m5c, inv_nibcadd_m5c },
    { "NIB_CADD_M63", search_nibcadd_m63, ap_nibcadd_m63, inv_nibcadd_m63 },
    { "NIB_CADD_M65", search_nibcadd_m65, ap_nibcadd_m65, inv_nibcadd_m65 },
    { "NIB_CADD_M66", search_nibcadd_m66, ap_nibcadd_m66, inv_nibcadd_m66 },
    { "NIB_CADD_M69", search_nibcadd_m69, ap_nibcadd_m69, inv_nibcadd_m69 },
    { "NIB_CADD_M6A", search_nibcadd_m6a, ap_nibcadd_m6a, inv_nibcadd_m6a },
    { "NIB_CADD_M6C", search_nibcadd_m6c, ap_nibcadd_m6c, inv_nibcadd_m6c },
    { "NIB_CADD_M71", search_nibcadd_m71, ap_nibcadd_m71, inv_nibcadd_m71 },
    { "NIB_CADD_M72", search_nibcadd_m72, ap_nibcadd_m72, inv_nibcadd_m72 },
    { "BITREV_IDX_XOR_S0L2", search_bitrevrange_s0l2, ap_bitrevrange_s0l2, ap_bitrevrange_s0l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S1L2", search_bitrevrange_s1l2, ap_bitrevrange_s1l2, ap_bitrevrange_s1l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S2L2", search_bitrevrange_s2l2, ap_bitrevrange_s2l2, ap_bitrevrange_s2l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S3L2", search_bitrevrange_s3l2, ap_bitrevrange_s3l2, ap_bitrevrange_s3l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S4L2", search_bitrevrange_s4l2, ap_bitrevrange_s4l2, ap_bitrevrange_s4l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S5L2", search_bitrevrange_s5l2, ap_bitrevrange_s5l2, ap_bitrevrange_s5l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S6L2", search_bitrevrange_s6l2, ap_bitrevrange_s6l2, ap_bitrevrange_s6l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S7L2", search_bitrevrange_s7l2, ap_bitrevrange_s7l2, ap_bitrevrange_s7l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S8L2", search_bitrevrange_s8l2, ap_bitrevrange_s8l2, ap_bitrevrange_s8l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S9L2", search_bitrevrange_s9l2, ap_bitrevrange_s9l2, ap_bitrevrange_s9l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S10L2", search_bitrevrange_s10l2, ap_bitrevrange_s10l2, ap_bitrevrange_s10l2 }, /* self-inverse */
    { "BITREV_IDX_XOR_S0L3", search_bitrevrange_s0l3, ap_bitrevrange_s0l3, ap_bitrevrange_s0l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S1L3", search_bitrevrange_s1l3, ap_bitrevrange_s1l3, ap_bitrevrange_s1l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S2L3", search_bitrevrange_s2l3, ap_bitrevrange_s2l3, ap_bitrevrange_s2l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S3L3", search_bitrevrange_s3l3, ap_bitrevrange_s3l3, ap_bitrevrange_s3l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S4L3", search_bitrevrange_s4l3, ap_bitrevrange_s4l3, ap_bitrevrange_s4l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S5L3", search_bitrevrange_s5l3, ap_bitrevrange_s5l3, ap_bitrevrange_s5l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S6L3", search_bitrevrange_s6l3, ap_bitrevrange_s6l3, ap_bitrevrange_s6l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S7L3", search_bitrevrange_s7l3, ap_bitrevrange_s7l3, ap_bitrevrange_s7l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S8L3", search_bitrevrange_s8l3, ap_bitrevrange_s8l3, ap_bitrevrange_s8l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S9L3", search_bitrevrange_s9l3, ap_bitrevrange_s9l3, ap_bitrevrange_s9l3 }, /* self-inverse */
    { "BITREV_IDX_XOR_S0L4", search_bitrevrange_s0l4, ap_bitrevrange_s0l4, ap_bitrevrange_s0l4 }, /* self-inverse */
    { "BITREV_IDX_XOR_S1L4", search_bitrevrange_s1l4, ap_bitrevrange_s1l4, ap_bitrevrange_s1l4 }, /* self-inverse */
    { "BITREV_IDX_XOR_S2L4", search_bitrevrange_s2l4, ap_bitrevrange_s2l4, ap_bitrevrange_s2l4 }, /* self-inverse */
    { "BITREV_IDX_XOR_S3L4", search_bitrevrange_s3l4, ap_bitrevrange_s3l4, ap_bitrevrange_s3l4 }, /* self-inverse */
    { "BITREV_IDX_XOR_S4L4", search_bitrevrange_s4l4, ap_bitrevrange_s4l4, ap_bitrevrange_s4l4 }, /* self-inverse */
    { "BITREV_IDX_XOR_S5L4", search_bitrevrange_s5l4, ap_bitrevrange_s5l4, ap_bitrevrange_s5l4 }, /* self-inverse */
    { "BITREV_IDX_XOR_S6L4", search_bitrevrange_s6l4, ap_bitrevrange_s6l4, ap_bitrevrange_s6l4 }, /* self-inverse */
    { "BITREV_IDX_XOR_S7L4", search_bitrevrange_s7l4, ap_bitrevrange_s7l4, ap_bitrevrange_s7l4 }, /* self-inverse */
    { "BITREV_IDX_XOR_S8L4", search_bitrevrange_s8l4, ap_bitrevrange_s8l4, ap_bitrevrange_s8l4 }, /* self-inverse */
    { "BITREV_IDX_XOR_S0L5", search_bitrevrange_s0l5, ap_bitrevrange_s0l5, ap_bitrevrange_s0l5 }, /* self-inverse */
    { "BITREV_IDX_XOR_S1L5", search_bitrevrange_s1l5, ap_bitrevrange_s1l5, ap_bitrevrange_s1l5 }, /* self-inverse */
    { "BITREV_IDX_ADD_S2L5", search_bitrevrangeadd_s2l5, ap_bitrevrangeadd_s2l5, inv_bitrevrangeadd_s2l5 },
    { "BITREV_IDX_ADD_S3L5", search_bitrevrangeadd_s3l5, ap_bitrevrangeadd_s3l5, inv_bitrevrangeadd_s3l5 },
    { "BITREV_IDX_ADD_S4L5", search_bitrevrangeadd_s4l5, ap_bitrevrangeadd_s4l5, inv_bitrevrangeadd_s4l5 },
    { "BITREV_IDX_ADD_S5L5", search_bitrevrangeadd_s5l5, ap_bitrevrangeadd_s5l5, inv_bitrevrangeadd_s5l5 },
    { "BITREV_IDX_ADD_S6L5", search_bitrevrangeadd_s6l5, ap_bitrevrangeadd_s6l5, inv_bitrevrangeadd_s6l5 },
    { "BITREV_IDX_ADD_S7L5", search_bitrevrangeadd_s7l5, ap_bitrevrangeadd_s7l5, inv_bitrevrangeadd_s7l5 },
    { "BITREV_IDX_ADD_S0L6", search_bitrevrangeadd_s0l6, ap_bitrevrangeadd_s0l6, inv_bitrevrangeadd_s0l6 },
    { "BITREV_IDX_ADD_S1L6", search_bitrevrangeadd_s1l6, ap_bitrevrangeadd_s1l6, inv_bitrevrangeadd_s1l6 },
    { "BITREV_IDX_ADD_S2L6", search_bitrevrangeadd_s2l6, ap_bitrevrangeadd_s2l6, inv_bitrevrangeadd_s2l6 },
    { "BITREV_IDX_ADD_S3L6", search_bitrevrangeadd_s3l6, ap_bitrevrangeadd_s3l6, inv_bitrevrangeadd_s3l6 },
    { "BITREV_IDX_ADD_S4L6", search_bitrevrangeadd_s4l6, ap_bitrevrangeadd_s4l6, inv_bitrevrangeadd_s4l6 },
    { "BITREV_IDX_ADD_S5L6", search_bitrevrangeadd_s5l6, ap_bitrevrangeadd_s5l6, inv_bitrevrangeadd_s5l6 },
    { "BITREV_IDX_ADD_S6L6", search_bitrevrangeadd_s6l6, ap_bitrevrangeadd_s6l6, inv_bitrevrangeadd_s6l6 },
    { "BITREV_IDX_ADD_S0L7", search_bitrevrangeadd_s0l7, ap_bitrevrangeadd_s0l7, inv_bitrevrangeadd_s0l7 },
    { "BITREV_IDX_ADD_S1L7", search_bitrevrangeadd_s1l7, ap_bitrevrangeadd_s1l7, inv_bitrevrangeadd_s1l7 },
    { "BITREV_IDX_ADD_S2L7", search_bitrevrangeadd_s2l7, ap_bitrevrangeadd_s2l7, inv_bitrevrangeadd_s2l7 },
    { "BITREV_IDX_ADD_S3L7", search_bitrevrangeadd_s3l7, ap_bitrevrangeadd_s3l7, inv_bitrevrangeadd_s3l7 },
    { "BITREV_IDX_ADD_S4L7", search_bitrevrangeadd_s4l7, ap_bitrevrangeadd_s4l7, inv_bitrevrangeadd_s4l7 },
    { "BITREV_IDX_ADD_S5L7", search_bitrevrangeadd_s5l7, ap_bitrevrangeadd_s5l7, inv_bitrevrangeadd_s5l7 },
    { "BITREV_IDX_ADD_S0L8", search_bitrevrangeadd_s0l8, ap_bitrevrangeadd_s0l8, inv_bitrevrangeadd_s0l8 },
    { "BITREV_IDX_ADD_S1L8", search_bitrevrangeadd_s1l8, ap_bitrevrangeadd_s1l8, inv_bitrevrangeadd_s1l8 },
    { "BITREV_IDX_ADD_S2L8", search_bitrevrangeadd_s2l8, ap_bitrevrangeadd_s2l8, inv_bitrevrangeadd_s2l8 },
    { "BITREV_IDX_ADD_S3L8", search_bitrevrangeadd_s3l8, ap_bitrevrangeadd_s3l8, inv_bitrevrangeadd_s3l8 },
    { "BITREV_IDX_ADD_S4L8", search_bitrevrangeadd_s4l8, ap_bitrevrangeadd_s4l8, inv_bitrevrangeadd_s4l8 },
    { "BITREV_IDX_ADD_S0L9", search_bitrevrangeadd_s0l9, ap_bitrevrangeadd_s0l9, inv_bitrevrangeadd_s0l9 },
    { "BITREV_IDX_ADD_S1L9", search_bitrevrangeadd_s1l9, ap_bitrevrangeadd_s1l9, inv_bitrevrangeadd_s1l9 },
    { "BITREV_IDX_ADD_S2L9", search_bitrevrangeadd_s2l9, ap_bitrevrangeadd_s2l9, inv_bitrevrangeadd_s2l9 },
    { "BITREV_IDX_ADD_S3L9", search_bitrevrangeadd_s3l9, ap_bitrevrangeadd_s3l9, inv_bitrevrangeadd_s3l9 },
    { "BITREV_IDX_ADD_S0L10", search_bitrevrangeadd_s0l10, ap_bitrevrangeadd_s0l10, inv_bitrevrangeadd_s0l10 },
    { "BITREV_IDX_ADD_S1L10", search_bitrevrangeadd_s1l10, ap_bitrevrangeadd_s1l10, inv_bitrevrangeadd_s1l10 },
    { "BITREV_IDX_ADD_S2L10", search_bitrevrangeadd_s2l10, ap_bitrevrangeadd_s2l10, inv_bitrevrangeadd_s2l10 },
    { "BITREV_IDX_ADD_S0L11", search_bitrevrangeadd_s0l11, ap_bitrevrangeadd_s0l11, inv_bitrevrangeadd_s0l11 },
};
#define NREG ((int)(sizeof(REGISTRY) / sizeof(REGISTRY[0])))

static void apply_instr(u8 *d, int n, Instr t)  { REGISTRY[t.type].apply(d, n, t.stride, t.phase, t.amp); }
static void invert_instr(u8 *d, int n, Instr t) { REGISTRY[t.type].invert(d, n, t.stride, t.phase, t.amp); }

/* ============================================================ *
 *  selftest -- apply/invert round-trip for every registered type *
 *  (run this after adding a new instruction, before trusting it)  *
 * ============================================================ */

/* only >= 0: test just that one registry index (fast -- used while adding
 * instructions one at a time, so we don't re-run every prior type's
 * search() again on each new addition). only < 0: test everything. */
static int selftest(int only) {
    srand(12345);
    u8 orig[BLOCK], work[BLOCK];
    for (int i = 0; i < BLOCK; i++) orig[i] = (u8)rand();
    int fails = 0;
    int r0 = (only >= 0) ? only : 0;
    int r1 = (only >= 0) ? only : NREG - 1;
    for (int r = r0; r <= r1; r++) {
        memcpy(work, orig, BLOCK);
        double Sb = S_of(work, BLOCK);
        Instr t = {0};
        REGISTRY[r].search(work, BLOCK, Sb, &t);
        t.type = r;
        apply_instr(work, BLOCK, t);
        invert_instr(work, BLOCK, t);
        int ok = memcmp(work, orig, BLOCK) == 0;
        if (!ok) fails++;
        printf("  %-10s  %s\n", REGISTRY[r].name, ok ? "OK" : "FAIL");
    }
    printf("selftest: %d/%d types OK\n", (r1 - r0 + 1) - fails, r1 - r0 + 1);
    return fails;
}

/* ============================================================ *
 *  one pass -- run every instruction's search once, report all  *
 * ============================================================ */

#define MAX_LAYERS 200

static void run_one_pass(const u8 *d, int n) {
    u8 orig[BLOCK];
    memcpy(orig, d, (size_t)n);
    u8 work[BLOCK];
    memcpy(work, d, (size_t)n);

    static Instr all_t[NREG];
    static double all_net[NREG];
    Instr layers[MAX_LAYERS];
    double layer_net[MAX_LAYERS];
    double layer_overhead[MAX_LAYERS];
    int nlayers = 0;
    double total_net = 0.0;
    double total_overhead = 0.0;
    double total_raw = 0.0;
    double final_Sb = 0.0;

    /* Tracks, per instruction, how many layers it had a POSITIVE candidate
     * net in (not just how many layers it actually WON -- only one
     * instruction wins each layer, but many others may also have scored
     * positive that round). Instructions with profit_count==0 after the
     * whole pass never once looked useful on any layer of this data;
     * instructions with a small nonzero count were only occasionally
     * useful. Both are candidates for pruning from the registry. */
    static int profit_count[NREG];
    memset(profit_count, 0, sizeof profit_count);
    int layers_evaluated = 0;

    for (int layer = 0; layer < MAX_LAYERS; layer++) {
        double Sb = S_of(work, n);
        final_Sb = Sb;
        int best_r = 0; double best_net = -1e18; Instr best_t = {0};

        /* Parallelized across instructions: each REGISTRY[r].search() call only
         * reads `work`/`n`/`Sb` and writes to its own local `cand`/`all_t[r]`/
         * `all_net[r]` slot -- no two threads ever touch the same instruction
         * (each r runs on exactly one thread, exactly once per layer), and
         * every lookup table the search functions read (gf_mul_tab, hlog,
         * nibpow_elist, etc.) was populated once by the init_*() calls in
         * main() before any threading starts, so it's read-only from here on.
         * Verified via grep that no file-scope mutable array is written by
         * more than one function. The live "best so far" progress line from
         * before doesn't translate cleanly to parallel execution (results
         * arrive out of order), so this reports a simple completion counter
         * instead and reveals the actual best only after the region joins. */
        int progress_done = 0;
        #pragma omp parallel for schedule(dynamic, 1)
        for (int r = 0; r < NREG; r++) {
            Instr cand = {0};
            double net = REGISTRY[r].search(work, n, Sb, &cand);
            cand.type = r;
            all_t[r] = cand;
            all_net[r] = net;
            int done;
            #pragma omp atomic capture
            done = ++progress_done;
            if (done % 8 == 0 || done == NREG) {
                #pragma omp critical
                {
                    printf("\r[layer %2d] [%4d/%4d] searching (parallel, %d threads)...    ",
                           layer + 1, done, NREG, omp_get_max_threads());
                    fflush(stdout);
                }
            }
        }
        layers_evaluated++;
        for (int r = 0; r < NREG; r++) {
            if (all_net[r] > 0.0) profit_count[r]++;
            if (all_net[r] > best_net) { best_net = all_net[r]; best_t = all_t[r]; best_r = r; }
        }
        printf("\n  best this layer: %s net=%+.1f\n", REGISTRY[best_r].name, best_net);

        if (best_net <= 0.0) {
            printf("no further improvement (best candidate net=%+.1f) -- stopping after %d layer(s)\n",
                   best_net, nlayers);
            break;
        }

        apply_instr(work, n, best_t);
        double S_after = S_of(work, n);
        double raw_gain = S_after - Sb;
        double overhead = raw_gain - best_net;
        double nlog2n = (double)n * log2((double)n);
        double entropy_before = nlog2n - Sb;
        double entropy_after = nlog2n - S_after;
        layers[nlayers] = best_t;
        layer_net[nlayers] = best_net;
        layer_overhead[nlayers] = overhead;
        total_net += best_net;
        total_overhead += overhead;
        total_raw += raw_gain;
        nlayers++;

        printf("layer %d: %s (stride=%d phase=%d amp=%u) net=%+.1f bits (raw=%+.1f overhead=%.1f)  (cumulative=%+.1f bits)\n",
               nlayers, REGISTRY[best_t.type].name, best_t.stride, best_t.phase, best_t.amp,
               best_net, raw_gain, overhead, total_net);
        printf("  entropy: %.1f bits (%.4f bps) -> %.1f bits (%.4f bps)\n",
               entropy_before, entropy_before / n, entropy_after, entropy_after / n);

        if (nlayers >= MAX_LAYERS) {
            printf("reached max layer cap (%d) -- stopping\n", MAX_LAYERS);
            break;
        }
    }

    u8 verify[BLOCK];
    memcpy(verify, work, (size_t)n);
    for (int i = nlayers - 1; i >= 0; i--) invert_instr(verify, n, layers[i]);
    printf("\nlayered round-trip (%d layer%s, invert in reverse order): %s\n",
           nlayers, nlayers == 1 ? "" : "s", memcmp(verify, orig, (size_t)n) == 0 ? "OK" : "FAIL");

    {
        const char *outpath = "instlaboutput.bin";
        FILE *outf = fopen(outpath, "wb");
        if (!outf) {
            fprintf(stderr, "warning: could not open %s for writing\n", outpath);
        } else {
            size_t written = fwrite(work, 1, (size_t)n, outf);
            fclose(outf);
            if (written == (size_t)n) printf("wrote %d bytes of layered output to %s\n", n, outpath);
            else fprintf(stderr, "warning: short write to %s (%zu/%d bytes)\n", outpath, written, n);
        }
    }

    printf("\n%d layer(s) applied, total net=%+.1f bits saved (raw gain=%+.1f, total overhead=%.1f)\n",
           nlayers, total_net, total_raw, total_overhead);
    for (int i = 0; i < nlayers; i++) {
        Instr *t = &layers[i];
        printf("  layer %2d: %-20s stride=%-4d phase=%-4d amp=%-10u net=%+8.1f  overhead=%6.1f\n",
               i + 1, REGISTRY[t->type].name, t->stride, t->phase, t->amp, layer_net[i], layer_overhead[i]);
    }

    int order[NREG];
    for (int r = 0; r < NREG; r++) order[r] = r;
    for (int i = 0; i < NREG; i++) {
        int best_j = i;
        for (int j = i + 1; j < NREG; j++) if (all_net[order[j]] > all_net[order[best_j]]) best_j = j;
        int tmp = order[i]; order[i] = order[best_j]; order[best_j] = tmp;
    }

    int topn = (NREG < 15) ? NREG : 15;
    printf("\ntop %d instructions on the FINAL layer's data (all non-improving now):\n", topn);
    printf("  %-4s  %-18s  %+9s  %7s  %7s  %10s  %8s\n", "rank", "type", "net", "stride", "phase", "amp", "overhead");
    for (int i = 0; i < topn; i++) {
        int r = order[i];
        Instr *t = &all_t[r];
        u8 scratch[BLOCK];
        memcpy(scratch, work, (size_t)n);
        apply_instr(scratch, n, *t);
        double raw_gain = S_of(scratch, n) - final_Sb;
        double overhead = raw_gain - all_net[r];
        printf("  %-4d  %-18s  %+9.1f  %7d  %7d  %10u  %8.1f\n",
               i + 1, REGISTRY[r].name, all_net[r], t->stride, t->phase, t->amp, overhead);
    }

    int never_count = 0, rare_count = 0;
    for (int r = 0; r < NREG; r++) {
        if (profit_count[r] == 0) never_count++;
        else if (profit_count[r] <= 3) rare_count++;
    }
    printf("\nnever profitable (0/%d layers, candidates for removal): %d instruction(s)\n",
           layers_evaluated, never_count);
    for (int r = 0; r < NREG; r++) {
        if (profit_count[r] == 0) printf("  %s\n", REGISTRY[r].name);
    }
    printf("\nrarely profitable (1-3/%d layers, marginal): %d instruction(s)\n",
           layers_evaluated, rare_count);
    for (int r = 0; r < NREG; r++) {
        if (profit_count[r] >= 1 && profit_count[r] <= 3) printf("  %-20s (%d/%d layers)\n", REGISTRY[r].name, profit_count[r], layers_evaluated);
    }
}

/* ============================================================ *
 *  main                                                          *
 * ============================================================ */

int main(int argc, char **argv) {
    init_hlog();
    init_log2_tab();
    init_gf_mul_tab();
    init_fixed_keystream();
    init_blockdiff();
    init_sin256();
    init_gf_log();
    init_gfpow();
    init_qr5();
    init_latin4();
    init_prime257();
    init_compand();
    init_bitrev12();
    init_fixed_keystream2();
    init_fixed_keystream3();
    init_fixed_keystream4();
    init_fixed_keystream5();
    init_fixed_keystream6();
    init_fixed_keystream7();
    init_fixed_keystream8();
    init_fixed_keystream9();
    init_fixed_keystream10();
    init_hilbert();
    init_gf16_mul_tab();
    init_gf128_mul_tab();
    init_gf16_log();
    init_nibpow();
    init_gf128_log();
    init_vgfpow();
    init_gf64_mul_tab();
    init_gf64_log();
    init_vm4pow();
    init_blockdiff2alt();
    init_latin4b();
    init_compand2();
    init_compand3();
    init_compand4();
    init_compand5();
    init_compand6();
    init_compand7();
    init_compand8();
    init_qr11();
    init_qr13();
    init_qr17();
    init_qr19();
    init_qr23();
    init_qr29();
    init_qr31();
    init_qr37();
    init_qr41();
    init_qr43();
    init_qr47();
    init_qr53();
    init_qr59();
    init_qr61();
    init_qr67();
    init_qr71();
    init_qr73();
    init_qr79();
    init_qr83();
    init_qr89();
    init_qr97();
    init_qr101();
    init_qr103();
    init_qr107();
    init_qr109();
    init_qr113();
    init_qr127();
    init_qr131();
    init_qr137();
    init_qr139();
    init_qr149();
    init_qr151();
    init_qr157();
    init_qr163();
    init_qr167();
    init_qr173();
    init_qr179();
    init_qr181();
    init_qr191();
    init_qr193();
    init_qr197();
    init_qr199();
    init_qr211();
    init_qr223();
    init_qr227();
    init_qr229();
    init_qr233();
    init_qr239();
    init_primeP_263();
    init_primeP_269();
    init_primeP_271();
    init_primeP_277();
    init_primeP_281();
    init_primeP_283();
    init_primeP_293();
    init_primeP_307();
    init_primeP_311();
    init_primeP_313();
    init_primeP_317();
    init_primeP_331();
    init_primeP_337();
    init_primeP_347();
    init_primeP_349();
    init_primeP_353();
    init_primeP_359();
    init_primeP_367();
    init_primeP_373();
    init_primeP_379();
    init_primeP_383();
    init_primeP_389();
    init_primeP_397();
    init_primeP_401();
    init_primeP_409();
    init_primeP_419();
    init_primeP_421();
    init_primeP_431();
    init_primeP_433();
    init_primeP_439();
    init_primeP_443();
    init_primeP_449();
    init_primeP_457();
    init_primeP_461();
    init_primeP_463();
    init_primeP_467();
    init_primeP_479();
    init_primeP_487();
    init_primeP_491();
    init_primeP_499();
    init_primeP_503();
    init_primeP_509();
    init_primeP_521();
    init_primeP_523();
    init_primeP_541();
    init_primeP_547();
    init_primeP_557();
    init_primeP_563();
    init_primeP_569();
    init_primeP_571();
    init_primeP_577();
    init_primeP_587();
    init_primeP_593();
    init_primeP_599();
    init_primeP_601();
    init_primeP_607();
    init_primeP_613();
    init_primeP_617();
    init_primeP_619();
    init_primeP_631();
    init_primeP_641();
    init_primeP_643();
    init_primeP_647();
    init_primeP_653();
    init_primeP_659();
    init_primeP_661();
    init_primeP_673();
    init_primeP_677();
    init_primeP_683();
    init_primeP_691();
    init_primeP_701();
    init_primeP_709();
    init_primeP_719();
    init_primeP_727();
    init_primeP_733();
    init_primeP_739();
    init_primeP_743();
    init_primeP_751();
    init_primeP_757();
    init_primeP_761();
    init_primeP_769();
    init_primeP_773();
    init_primeP_787();
    init_primeP_797();
    init_primeP_809();
    init_primeP_811();
    init_primeP_821();
    init_primeP_823();
    init_primeP_827();
    init_primeP_829();
    init_primeP_839();
    init_primeP_853();
    init_primeP_857();
    init_primeP_859();
    init_primeP_863();
    init_primeP_877();
    init_primeP_881();
    init_primeP_883();
    init_primeP_887();
    init_primeP_907();
    init_primeP_911();
    init_primeP_919();
    init_primeP_929();
    init_primeP_937();
    init_primeP_941();
    init_primeP_947();
    init_primeP_953();
    init_primeP_967();
    init_primeP_971();
    init_primeP_977();
    init_primeP_983();
    init_primeP_991();
    init_primeP_997();
    init_primeP_1009();
    init_primeP_1013();
    init_primeP_1019();
    init_primeP_1021();
    init_primeP_1031();
    init_primeP_1033();
    init_primeP_1039();
    init_primeP_1049();
    init_primeP_1051();
    init_primeP_1061();
    init_primeP_1063();
    init_primeP_1069();
    init_primeP_1087();
    init_primeP_1091();
    init_primeP_1093();
    init_primeP_1097();
    init_primeP_1103();
    init_primeP_1109();
    init_primeP_1117();
    init_primeP_1123();
    init_primeP_1129();
    init_primeP_1151();
    init_primeP_1153();
    init_primeP_1163();
    init_primeP_1171();
    init_primeP_1181();
    init_primeP_1187();
    init_primeP_1193();
    init_primeP_1201();
    init_primeP_1213();
    init_primeP_1217();
    init_primeP_1223();
    init_primeP_1229();
    init_primeP_1231();
    init_primeP_1237();
    init_primeP_1249();
    init_primeP_1259();
    init_primeP_1277();
    init_primeP_1279();
    init_primeP_1283();
    init_primeP_1289();
    init_primeP_1291();
    init_primeP_1297();
    init_primeP_1301();
    init_primeP_1303();
    init_primeP_1307();
    init_primeP_1319();
    init_primeP_1321();
    init_primeP_1327();
    init_primeP_1361();
    init_primeP_1367();
    init_primeP_1373();
    init_primeP_1381();
    init_primeP_1399();
    init_primeP_1409();
    init_primeP_1423();
    init_primeP_1427();
    init_primeP_1429();
    init_primeP_1433();
    init_primeP_1439();
    init_primeP_1447();
    init_primeP_1451();
    init_primeP_1453();
    init_primeP_1459();
    init_primeP_1471();
    init_primeP_1481();
    init_primeP_1483();
    init_primeP_1487();
    init_primeP_1489();
    init_primeP_1493();
    init_primeP_1499();
    init_primeP_1511();
    init_primeP_1523();
    init_primeP_1531();
    init_primeP_1543();
    init_latin4c();
    init_crc32_tab();
    init_qr7();

    if (argc > 1 && strcmp(argv[1], "selftest") == 0) {
        if (argc > 3) {
            int r0 = atoi(argv[2]), r1 = atoi(argv[3]);
            if (r1 > NREG - 1) {
                printf("WARNING: requested r1=%d but NREG=%d -- clamping to %d\n", r1, NREG, NREG - 1);
                r1 = NREG - 1;
            }
            int fails = 0, tested = 0;
            srand(12345);
            u8 orig[BLOCK], work[BLOCK];
            for (int i = 0; i < BLOCK; i++) orig[i] = (u8)rand();
            for (int r = r0; r <= r1 && r < NREG; r++) {
                tested++;
                memcpy(work, orig, BLOCK);
                double Sb = S_of(work, BLOCK);
                Instr t = {0};
                REGISTRY[r].search(work, BLOCK, Sb, &t);
                t.type = r;
                apply_instr(work, BLOCK, t);
                invert_instr(work, BLOCK, t);
                int ok = memcmp(work, orig, BLOCK) == 0;
                if (!ok) fails++;
                printf("  %-10s  %s\n", REGISTRY[r].name, ok ? "OK" : "FAIL");
            }
            printf("selftest range [%d,%d]: %d/%d types OK\n", r0, r1, tested - fails, tested);
            return fails ? 2 : 0;
        }
        int only = (argc > 2) ? atoi(argv[2]) : -1;
        return selftest(only) ? 2 : 0;
    }

    static u8 data[BLOCK];
    FILE *f = fopen(INPUT_FILE, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", INPUT_FILE); return 1; }
    size_t got = fread(data, 1, BLOCK, f);
    fclose(f);
    if (got < BLOCK) { fprintf(stderr, "file too small (%zu bytes, need %d)\n", got, BLOCK); return 1; }

    printf("input entropy: %.4f bps\n\n", entropy_bits(data, BLOCK) / BLOCK);
    run_one_pass(data, BLOCK);
    return 0;
}
