/* bitinstructionlab.c -- instructionlab.c's sibling, but every candidate
 * operates on the RAW BITSTREAM (individual bits, crossing byte boundaries
 * freely) instead of whole bytes. The scoring objective stays byte-level
 * (the real Dirichlet(alpha) code length reduce2.c/compressor3.c actually
 * pay for, same ctab/lgamma machinery as instructionlab.c) because that is
 * still what the downstream coder charges -- only the TRANSFORM mechanism
 * moves to bit granularity here, not the cost model. Screened against real
 * residual data, never fresh random data, same "residual-data trap"
 * discipline as instructionlab.c. ALWAYS reads from compresseddata.bin (the
 * real project target, not a scratch file) -- see DATAFILE below.
 *
 * WHY THIS IS A GENUINELY NEW AXIS, NOT JUST MORE OF THE SAME:
 *   Every instruction in instructionlab.c/reduce2.c reads and writes whole
 *   bytes -- it can never express "the 5th bit of the whole stream", only
 *   "the 5th BYTE" or "bit k of THIS byte". A period like 5, 7, 9, 11, 13...
 *   does not divide 8, so a stride-5 BIT pattern cuts across byte boundaries
 *   with a repeat length of lcm(5,8)=40 bits (5 bytes) -- a structure no
 *   byte-oriented instruction can represent, let alone exploit, because it
 *   only ever sees each byte as one atomic 256-ary symbol.
 *
 * A NOTE ON WHY "FIXED, UNSEARCHED, WHOLE-BLOCK" IS DEAD BUT THIS ISN'T:
 *   instructionlab.c/reduce2.c killed REFLECT/GRAY_CODE/BIT_REV forever --
 *   applying one fixed bijection to every byte, uniformly, is a pure symbol
 *   relabelling, and order-0 entropy is exactly invariant under any such
 *   relabelling. PHASE-based bit patterns below are NOT that: XORing a
 *   DIFFERENT bit depending on position (mod S) is position-dependent and
 *   non-uniform -- it can and does reshape the byte histogram.
 *
 * Candidate families: ROTATE, BITLAG, PHASEDIRECT, PHASEPRNG, BIGNUM_ADD,
 * BIGNUM_SUB -- see the per-family comments below for what each does.
 * Every structural parameter (period, lag, width) is explicitly billed
 * (TAGB + log2(#variants) + payload), per this project's hard-won
 * honest-accounting lesson (letting "which variant" ride free under a flat
 * tag silently inflates apparent gains).
 *
 * BLOCK SIZE IS NOW RUNTIME-VARIABLE (not a compile-time constant): every
 * scratch buffer that used to be a fixed-size `u8 buf[BLOCK]` is now
 * malloc'd to the actual block length passed in, so any candidate family
 * can be evaluated at any block size, from a few bytes to however much RAM
 * allows. This exists specifically for `sizesweep` mode below.
 *
 * Modes:
 *   (default)          -- screen: rank all candidates at a fixed block size.
 *   selftest           -- apply+invert round-trip check for every family.
 *   sizesweep [..]      -- sweep block size = 2^x bytes (x=xmin..xmax) and
 *                         report which size yields the most net bits saved
 *                         PER BYTE (so different sizes are comparable) --
 *                         answers "what's the best chunk size to use?".
 *                         Auto-stops once a size exceeds the file or a
 *                         memory/compute budget (2^64 is not literally
 *                         achievable; the file's real size caps it well
 *                         before then, and the sweep reports why it stopped).
 *   layered [..]       -- real multi-round greedy loop (plain greedy, no
 *                         recursive lookahead -- a natural extension, not
 *                         implemented here).
 *
 * Build: gcc -O2 -fopenmp -o bitinstructionlab bitinstructionlab.c -lm
 * Run:   ./bitinstructionlab [NB] [alpha] [seedlim]
 *        ./bitinstructionlab selftest
 *        ./bitinstructionlab sizesweep [NB_per_size] [alpha] [seedlim] [xmin] [xmax]
 *        ./bitinstructionlab layered [NB] [alpha] [seedlim] [maxlayers]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

#define DATAFILE "compresseddata.bin"   /* always this file -- the real project target */
#define LN2 0.69314718055994530942
#define TAGB 6.0          /* flat placeholder type-tag cost, matches instructionlab.c's convention */
#define MAXPERIOD 256
#define DEFAULT_BLOCK 4096

/* ============================================================ *
 *  objective: real Dirichlet(alpha) code length over the BYTE  *
 *  histogram -- identical to instructionlab.c/reduce2.c. Only  *
 *  the transform mechanism is bit-native in this file; the     *
 *  cost model stays byte-level because that's what actually    *
 *  gets paid downstream. ctab is now sized dynamically to      *
 *  whatever block size is currently in use (see init_ctab).    *
 * ============================================================ */
static double g_alpha = 32.0;
static double *ctab = NULL;
static int ctab_cap = 0;

static void init_ctab(int maxcount) {
    if (ctab_cap < maxcount + 1) {
        free(ctab);
        ctab = malloc((size_t)(maxcount + 1) * sizeof(double));
        ctab_cap = maxcount + 1;
    }
    ctab[0] = 0.0;
    double la = lgamma(g_alpha);
    for (int k = 1; k <= maxcount; k++)
        ctab[k] = (lgamma((double)k + g_alpha) - la) / LN2;
}
static double S_from_freq(const int *f) {
    double s = 0.0;
    for (int v = 0; v < 256; v++) s += ctab[f[v]];
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

/* ============================================================ *
 *  keystream generator (identical to reduce2.c/instructionlab.c's xs16) *
 * ============================================================ */
static inline u8 xs16_next(u16 *s) {
    u16 x = *s;
    x ^= x << 7; x ^= x >> 9; x ^= x << 8;
    *s = x;
    return (u8)(x ^ (x >> 8));
}

/* ============================================================ *
 *  bit-level primitives: bit 0 of a byte is its MSB, bits are   *
 *  numbered globally across the whole block (bit 8*i+b = bit b  *
 *  of byte i). All candidate families operate through these.    *
 * ============================================================ */
static inline int bit_get(const u8 *d, int i) { return (d[i >> 3] >> (7 - (i & 7))) & 1; }
static inline void bit_put(u8 *d, int i, int v) {
    u8 m = (u8)(0x80u >> (i & 7));
    if (v) d[i >> 3] |= m; else d[i >> 3] &= (u8)~m;
}
static inline void bit_flip(u8 *d, int i) { d[i >> 3] ^= (u8)(0x80u >> (i & 7)); }

static int igcd(int a, int b) { while (b) { int t = a % b; a = b; b = t; } return a; }

/* per-candidate seed budget: keeps O(seedlim * n) families bounded in total
 * work regardless of how large n (the block size) gets, by shrinking the
 * seed count for big blocks rather than letting runtime explode. */
static u32 eff_seedlim(u32 seedlim, long long n) {
    long long budget = 20000000LL;
    long long cap = budget / (n > 0 ? n : 1);
    if (cap < 2) cap = 2;
    return (seedlim < (u32)cap) ? seedlim : (u32)cap;
}

/* ============================================================ *
 *  ROTATE -- whole-block cyclic bit-rotation by k in 1..7.      *
 *  Re-aligns byte boundaries against the underlying bit content *
 *  (byte i of the output is a splice of input bytes i and i+1). *
 * ============================================================ */
static void apply_rotate(u8 *d, int n, int k) {
    if (!k) return;
    u8 *orig = malloc((size_t)n); memcpy(orig, d, n);
    for (int i = 0; i < n; i++)
        d[i] = (u8)((orig[i] << k) | (orig[(i + 1) % n] >> (8 - k)));
    free(orig);
}
static void invert_rotate(u8 *d, int n, int k) {
    if (!k) return;
    u8 *orig = malloc((size_t)n); memcpy(orig, d, n);
    for (int i = 0; i < n; i++)
        d[i] = (u8)((orig[i] >> k) | (orig[(i - 1 + n) % n] << (8 - k)));
    free(orig);
}
static double eval_rotate(const u8 *d, int n, double Sb, double oh, int k) {
    u8 *t = malloc((size_t)n); memcpy(t, d, n);
    apply_rotate(t, n, k);
    double net = (S_of(t, n) - Sb) - oh;
    free(t);
    return net;
}
static const int NROTATE = 7;

/* ============================================================ *
 *  BITLAG -- bit[i] ^= bit[i-k] (bit-level forward-difference / *
 *  "xor delta", lag k need not be a byte-aligned amount). Encode*
 *  reads from a FROZEN copy (each output bit depends on two      *
 *  original bits); decode runs the recurrence in place, low to  *
 *  high, since old[i-k] is already recovered by the time bit i  *
 *  is processed (i-k < i) -- classic delta encode/decode split. *
 * ============================================================ */
static const int BITLAGS[] = {
    1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,17,19,23,24,31,32,40,48,56,63,64,96,128,192,256
};
#define NBITLAG ((int)(sizeof(BITLAGS)/sizeof(int)))

static void apply_bitlagxor(u8 *d, int n, int k) {
    u8 *orig = malloc((size_t)n); memcpy(orig, d, n);
    for (int i = k; i < n * 8; i++) {
        int v = bit_get(orig, i) ^ bit_get(orig, i - k);
        bit_put(d, i, v);
    }
    free(orig);
}
static void invert_bitlagxor(u8 *d, int n, int k) {
    for (int i = k; i < n * 8; i++) {
        int v = bit_get(d, i) ^ bit_get(d, i - k);
        bit_put(d, i, v);
    }
}
static double eval_bitlagxor(const u8 *d, int n, double Sb, double oh, int k) {
    u8 *t = malloc((size_t)n); memcpy(t, d, n);
    apply_bitlagxor(t, n, k);
    double net = (S_of(t, n) - Sb) - oh;
    free(t);
    return net;
}

/* ============================================================ *
 *  PHASEDIRECT -- period S, one XOR-bit decision per residue     *
 *  class, chosen by direct greedy search against the REAL        *
 *  Dirichlet objective. Self-inverse (XOR). Implemented via       *
 *  incremental frequency-table updates (no full O(n) rescan per   *
 *  residue) and a sparse touched-byte list (at most 8 distinct     *
 *  byte-cycle positions carry a nonzero mask per residue, since     *
 *  one L-byte mask cycle spans exactly 8/gcd(S,8) occurrences of    *
 *  residue r) -- this keeps the whole S-residue search well under   *
 *  O(S*n) for large S, which matters once block sizes get big.       *
 * ============================================================ */
/* pruned 2026-07-21: empirically screened against 500 real residual blocks;
 * periods 2,4,8,12,16,17,20,23,24,31,32,40,48,56,64 never fired once (0/500)
 * and were removed -- only periods that cleared net>0 at least once survive. */
static const int PHASE_PERIODS_DIRECT[] = {
    3,5,6,7,9,10,11,13,19
};
#define NPHASE_DIRECT ((int)(sizeof(PHASE_PERIODS_DIRECT)/sizeof(int)))

static void apply_phasepattern(u8 *d, int n, int S, u64 pat) {
    for (int i = 0; i < n * 8; i++)
        if ((pat >> (i % S)) & 1ULL) bit_flip(d, i);
}
/* invert == apply again: XOR with the same fixed pattern is self-inverse */

static double eval_phasedirect(const u8 *d, int n, double Sb, double oh, int S, u64 *pat_out) {
    u8 *work = malloc((size_t)n); memcpy(work, d, n);
    int f[256]; freq_of(work, n, f);
    double curS = Sb;
    u64 pat = 0;
    int L = S / igcd(S, 8);
    for (int r = 0; r < S; r++) {
        int touched_pos[8]; u8 touched_mask[8]; int T = 0;
        for (int bidx = 0; bidx < L && T < 8; bidx++) {
            u8 m = 0;
            for (int b = 0; b < 8; b++) {
                int gpos = bidx * 8 + b;
                if (gpos % S == r) m |= (u8)(0x80 >> b);
            }
            if (m) { touched_pos[T] = bidx; touched_mask[T] = m; T++; }
        }
        double deltaS = 0.0;
        for (int t = 0; t < T; t++) {
            u8 m = touched_mask[t];
            for (int i = touched_pos[t]; i < n; i += L) {
                u8 oldv = work[i], newv = (u8)(oldv ^ m);
                deltaS += ctab[f[oldv]-1] + ctab[f[newv]+1] - ctab[f[oldv]] - ctab[f[newv]];
                f[oldv]--; f[newv]++;
                work[i] = newv;
            }
        }
        if (deltaS > 0.0) {
            curS += deltaS;
            pat |= (1ULL << r);
        } else {
            for (int t = 0; t < T; t++) {
                u8 m = touched_mask[t];
                for (int i = touched_pos[t]; i < n; i += L) {
                    u8 oldv = work[i], newv = (u8)(oldv ^ m);
                    f[oldv]--; f[newv]++;
                    work[i] = newv;
                }
            }
        }
    }
    free(work);
    if (pat_out) *pat_out = pat;
    return (curS - Sb) - oh;
}

/* ============================================================ *
 *  PHASEPRNG -- same repeating-XOR-pattern mechanism, but the    *
 *  pattern comes from a 16-bit PRNG seed (searched over seedlim)  *
 *  instead of being derived directly from the data. Cost is fixed *
 *  at seed_bits no matter how large S is. Applied via a per-BYTE   *
 *  mask cycle of length L=S/gcd(S,8) (derived once per seed) so    *
 *  the seed search runs at O(n) per trial, not O(8n).              *
 * ============================================================ */
static const int PHASE_PERIODS_PRNG[] = {
    3,5,6,7,9,11,13,17,19,23,29,31,37,40,53,64,80,96,120,128
};
#define NPHASE_PRNG ((int)(sizeof(PHASE_PERIODS_PRNG)/sizeof(int)))

static void gen_phase_pattern_bits(u32 seed, int S, u8 *pat) {
    u16 s = (u16)seed;
    for (int r = 0; r < S; r++) pat[r] = (u8)(xs16_next(&s) & 1);
}
static void gen_phase_bytemask(u32 seed, int S, u8 *bytemask, int *L_out) {
    u8 pat[MAXPERIOD]; gen_phase_pattern_bits(seed, S, pat);
    int L = S / igcd(S, 8);
    for (int bidx = 0; bidx < L; bidx++) {
        u8 m = 0;
        for (int b = 0; b < 8; b++) {
            int gpos = bidx * 8 + b;
            if (pat[gpos % S]) m |= (u8)(0x80 >> b);
        }
        bytemask[bidx] = m;
    }
    *L_out = L;
}
static void apply_phaseprng(u8 *d, int n, int S, u32 seed) {
    u8 bytemask[MAXPERIOD]; int L;
    gen_phase_bytemask(seed, S, bytemask, &L);
    for (int i = 0; i < n; i++) d[i] ^= bytemask[i % L];
}
/* invert == apply again (same seed): byte-mask XOR is self-inverse */

static double eval_phaseprng(const u8 *d, int n, double Sb, double oh, int S, u32 seedlim, u32 *bseed_out) {
    double best = -1e18; u32 bseed = 1;
    u8 bytemask[MAXPERIOD]; int L;
    for (u32 seed = 1; seed < seedlim; seed++) {
        gen_phase_bytemask(seed, S, bytemask, &L);
        int f[256] = {0};
        for (int i = 0; i < n; i++) f[(u8)(d[i] ^ bytemask[i % L])]++;
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    *bseed_out = bseed;
    return best;
}

/* ============================================================ *
 *  BIGNUM_ADD / BIGNUM_SUB -- the whole block as ONE big-endian  *
 *  unsigned integer; add/subtract a PRNG-seeded W-bit delta       *
 *  (placed in the low W bits) with full carry/borrow propagation   *
 *  mod 2^(8*n). Reversible exactly (it's arithmetic in a group):    *
 *  SUB undoes ADD and vice versa, same seed/width.                  *
 * ============================================================ */
/* pruned 2026-07-21: widths 16-4096 never fired once (0/500) against real
 * residual data -- a small delta is too locally-confined to matter, only a
 * near-full-width delta with real carry propagation ever wins. */
static const int BIGNUM_WBITS[] = { 16384,32768 };
#define NBIGW ((int)(sizeof(BIGNUM_WBITS)/sizeof(int)))

static void gen_delta(u32 seed, int n, int Wbits, u8 *delta) {
    memset(delta, 0, n);
    int Wbytes = (Wbits + 7) / 8; if (Wbytes > n) Wbytes = n;
    u16 s = (u16)seed;
    for (int i = 0; i < Wbytes; i++) delta[n - 1 - i] = xs16_next(&s);
}
static void bignum_add_ip(u8 *d, int n, const u8 *delta) {
    int carry = 0;
    for (int i = n - 1; i >= 0; i--) { int t = d[i] + delta[i] + carry; d[i] = (u8)t; carry = t >> 8; }
}
static void bignum_sub_ip(u8 *d, int n, const u8 *delta) {
    int borrow = 0;
    for (int i = n - 1; i >= 0; i--) {
        int t = d[i] - delta[i] - borrow;
        if (t < 0) { t += 256; borrow = 1; } else borrow = 0;
        d[i] = (u8)t;
    }
}
static void apply_bignumadd(u8 *d, int n, int Wbits, u32 seed)  { u8 *delta = malloc((size_t)n); gen_delta(seed, n, Wbits, delta); bignum_add_ip(d, n, delta); free(delta); }
static void invert_bignumadd(u8 *d, int n, int Wbits, u32 seed) { u8 *delta = malloc((size_t)n); gen_delta(seed, n, Wbits, delta); bignum_sub_ip(d, n, delta); free(delta); }
static void apply_bignumsub(u8 *d, int n, int Wbits, u32 seed)  { u8 *delta = malloc((size_t)n); gen_delta(seed, n, Wbits, delta); bignum_sub_ip(d, n, delta); free(delta); }
static void invert_bignumsub(u8 *d, int n, int Wbits, u32 seed) { u8 *delta = malloc((size_t)n); gen_delta(seed, n, Wbits, delta); bignum_add_ip(d, n, delta); free(delta); }

static double eval_bignumadd(const u8 *d, int n, double Sb, double oh, int Wbits, u32 seedlim, u32 *bseed_out) {
    double best = -1e18; u32 bseed = 1;
    u8 *delta = malloc((size_t)n), *work = malloc((size_t)n);
    for (u32 seed = 1; seed < seedlim; seed++) {
        gen_delta(seed, n, Wbits, delta);
        memcpy(work, d, n);
        bignum_add_ip(work, n, delta);
        double net = (S_of(work, n) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    free(delta); free(work);
    *bseed_out = bseed;
    return best;
}
static double eval_bignumsub(const u8 *d, int n, double Sb, double oh, int Wbits, u32 seedlim, u32 *bseed_out) {
    double best = -1e18; u32 bseed = 1;
    u8 *delta = malloc((size_t)n), *work = malloc((size_t)n);
    for (u32 seed = 1; seed < seedlim; seed++) {
        gen_delta(seed, n, Wbits, delta);
        memcpy(work, d, n);
        bignum_sub_ip(work, n, delta);
        double net = (S_of(work, n) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    free(delta); free(work);
    *bseed_out = bseed;
    return best;
}

/* ============================================================ *
 *  PRNGXOR8 -- full-block, byte-granular PRNG-driven XOR: the   *
 *  "prng xor" idea directly. Generating 8 bits at once (one PRNG *
 *  call per byte) and XORing them in is exactly equivalent to     *
 *  flipping each of those 8 bits independently -- so this covers   *
 *  the bit-level dense PRNG-XOR mechanism at 8x the search speed,    *
 *  with no loss of generality. genid picks a DEDICATED generator      *
 *  family per the "prngs dedicated for this type of binary blocks"     *
 *  idea: 0=xs16 (this project's standard), 1=xs32 (Marsaglia,           *
 *  different period/mixing), 2=splitmix64-style (multiplicative          *
 *  avalanche, structurally unrelated to any xorshift), 3=LCG              *
 *  (deliberately lower-quality control, known low-bit weaknesses).         *
 * ============================================================ */
static inline u32 xs32_next(u32 *s) { u32 x = *s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; *s = x; return x; }
static inline u8 xs32_byte(u32 *s) { u32 x = xs32_next(s); return (u8)(x ^ (x >> 16)); }
static inline u8 splitmix_byte(u64 *s) {
    u64 z = (*s += 0x9E3779B97F4A7C15ULL);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    return (u8)(z ^ (z >> 32));
}
static inline u8 lcg_byte(u32 *s) { *s = (*s) * 1103515245u + 12345u; return (u8)((*s) >> 16); }

static void apply_prngxor8(u8 *d, int n, int genid, u32 seed) {
    switch (genid) {
        case 0: { u16 s = (u16)seed; for (int i = 0; i < n; i++) d[i] ^= xs16_next(&s); break; }
        case 1: { u32 s = seed;      for (int i = 0; i < n; i++) d[i] ^= xs32_byte(&s); break; }
        case 2: { u64 s = seed;      for (int i = 0; i < n; i++) d[i] ^= splitmix_byte(&s); break; }
        default:{ u32 s = seed;      for (int i = 0; i < n; i++) d[i] ^= lcg_byte(&s); break; }
    }
}
/* invert == apply again (same genid/seed): the keystream is deterministic and XOR is self-inverse */
static double eval_prngxor8(const u8 *d, int n, double Sb, double oh, int genid, u32 seedlim, u32 *bseed_out) {
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < seedlim; seed++) {
        int f[256] = {0};
        switch (genid) {
            case 0: { u16 s = (u16)seed; for (int i = 0; i < n; i++) f[(u8)(d[i] ^ xs16_next(&s))]++; break; }
            case 1: { u32 s = seed;      for (int i = 0; i < n; i++) f[(u8)(d[i] ^ xs32_byte(&s))]++; break; }
            case 2: { u64 s = seed;      for (int i = 0; i < n; i++) f[(u8)(d[i] ^ splitmix_byte(&s))]++; break; }
            default:{ u32 s = seed;      for (int i = 0; i < n; i++) f[(u8)(d[i] ^ lcg_byte(&s))]++; break; }
        }
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    *bseed_out = bseed;
    return best;
}

/* ============================================================ *
 *  PRNGXOR8_MASKED -- density-gated version of PRNGXOR8: a       *
 *  searched selector keystream decides WHICH bytes get touched     *
 *  at all (5 curated densities), each touched byte XORed with an    *
 *  independent value-keystream byte. Reads "small areas" as SPARSE   *
 *  scattered coverage rather than spatial locality (see CHUNK below   *
 *  for the spatial-locality reading).                                  *
 * ============================================================ */
/* pruned 2026-07-21: the ~10% density (D4) never fired (0/500); only the
 * denser 50/40/30/20% masks ever win. */
static const u8 PXMASK_TH[4] = { 128,102,77,51 }; /* ~50/40/30/20% */
static void apply_prngxor8_masked(u8 *d, int n, int dens, u32 seed) {
    u8 th = PXMASK_TH[dens];
    u16 sel = (u16)seed, val = (u16)(seed ^ 0x5A5Au);
    for (int i = 0; i < n; i++) {
        u8 v = xs16_next(&val);
        u8 sbyte = xs16_next(&sel);
        if (sbyte < th) d[i] ^= v;
    }
}
static double eval_prngxor8_masked(const u8 *d, int n, double Sb, double oh, int dens, u32 seedlim, u32 *bseed_out) {
    u8 th = PXMASK_TH[dens];
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < seedlim; seed++) {
        u16 sel = (u16)seed, val = (u16)(seed ^ 0x5A5Au);
        int f[256] = {0};
        for (int i = 0; i < n; i++) {
            u8 v = xs16_next(&val);
            u8 sbyte = xs16_next(&sel);
            f[(sbyte < th) ? (u8)(d[i] ^ v) : d[i]]++;
        }
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    *bseed_out = bseed;
    return best;
}

/* ============================================================ *
 *  PRNGXOR8_CHUNK -- "small areas" read as SPATIAL locality:      *
 *  full/dense PRNG-XOR (like PRNGXOR8) but confined to ONE          *
 *  contiguous chunk (1/K of the block, index 0..K-1, last chunk      *
 *  absorbs any remainder). Mirrors the OLD instrlab.c project's       *
 *  proven SPLIT_PATTERN_ADD result (a self-contained chunk-confined    *
 *  instruction won a real layer there): local structure a whole-block   *
 *  search averages away can still be cheap to encode (K + chunk index)   *
 *  even though coverage is partial.                                       *
 * ============================================================ */
/* pruned 2026-07-21: K=8,16,32,64 never fired at ANY chunk index (0/500 each)
 * -- only the two largest/least-confined chunk sizes (K=2,4) ever win. */
static const int CHUNK_K[] = { 2,4 };
#define NCHUNKK ((int)(sizeof(CHUNK_K)/sizeof(int)))

static void apply_prngxor8_chunk(u8 *d, int n, int K, int idx, u32 seed) {
    int chunklen = n / K;
    int start = idx * chunklen;
    int end = (idx == K - 1) ? n : start + chunklen;
    u16 s = (u16)seed;
    for (int i = start; i < end; i++) d[i] ^= xs16_next(&s);
}
/* invert == apply again (same K/idx/seed): XOR is self-inverse */
static double eval_prngxor8_chunk(const u8 *d, int n, double Sb, double oh, int K, int idx, u32 seedlim, u32 *bseed_out) {
    int chunklen = n / K;
    int start = idx * chunklen;
    int end = (idx == K - 1) ? n : start + chunklen;
    int f0[256]; freq_of(d, n, f0);
    double best = -1e18; u32 bseed = 1;
    for (u32 seed = 1; seed < seedlim; seed++) {
        int f[256]; memcpy(f, f0, sizeof f);
        u16 s = (u16)seed;
        for (int i = start; i < end; i++) {
            u8 oldv = d[i], newv = (u8)(oldv ^ xs16_next(&s));
            f[oldv]--; f[newv]++;
        }
        double net = (S_from_freq(f) - Sb) - oh;
        if (net > best) { best = net; bseed = seed; }
    }
    *bseed_out = bseed;
    return best;
}

typedef struct { int kind; int a; int b; } Cand; /* kind: 0=ROTATE 1=BITLAG 2=PHASEDIRECT 3=PHASEPRNG 4=BIGADD 5=BIGSUB 6=PRNGXOR8 7=PRNGXOR8_MASKED 8=PRNGXOR8_CHUNK */

static void name_for_cand(Cand c, char *out) {
    static const char *genn[4] = { "XS16","XS32","SPLITMIX","LCG" };
    switch (c.kind) {
        case 0: snprintf(out, 40, "ROTATE_K%d", c.a); break;
        case 1: snprintf(out, 40, "BITLAG_%d", c.a); break;
        case 2: snprintf(out, 40, "PHASEDIRECT_S%d", c.a); break;
        case 3: snprintf(out, 40, "PHASEPRNG_S%d", c.a); break;
        case 4: snprintf(out, 40, "BIGNUM_ADD_W%d", c.a); break;
        case 5: snprintf(out, 40, "BIGNUM_SUB_W%d", c.a); break;
        case 6: snprintf(out, 40, "PRNGXOR8_%s", genn[c.a & 3]); break;
        case 7: snprintf(out, 40, "PRNGXOR8_MASKED_D%d", c.a); break;
        case 8: snprintf(out, 40, "PRNGXOR8_CHUNK_K%d_I%d", c.a, c.b); break;
    }
}

/* ============================================================ *
 *  selftest -- apply then invert on random data, every family,  *
 *  every parameter, check byte-exact round trip. Uses a fixed    *
 *  small test size (independent of any sweep's block size).      *
 * ============================================================ */
static int run_selftest(void) {
    srand(12345);
    const int N = DEFAULT_BLOCK;
    u8 *orig = malloc((size_t)N), *work = malloc((size_t)N);
    for (int i = 0; i < N; i++) orig[i] = (u8)rand();
    int pass = 0, fail = 0;

    for (int k = 1; k <= 7; k++) {
        memcpy(work, orig, N);
        apply_rotate(work, N, k);
        invert_rotate(work, N, k);
        if (memcmp(orig, work, N) == 0) pass++; else { fail++; printf("FAIL ROTATE k=%d\n", k); }
    }
    for (int li = 0; li < NBITLAG; li++) {
        int k = BITLAGS[li];
        memcpy(work, orig, N);
        apply_bitlagxor(work, N, k);
        invert_bitlagxor(work, N, k);
        if (memcmp(orig, work, N) == 0) pass++; else { fail++; printf("FAIL BITLAG k=%d\n", k); }
    }
    for (int pi = 0; pi < NPHASE_DIRECT; pi++) {
        int S = PHASE_PERIODS_DIRECT[pi];
        u64 pat = 0xD3ADBEEF1234ULL & ((S < 64) ? ((1ULL << S) - 1) : ~0ULL);
        memcpy(work, orig, N);
        apply_phasepattern(work, N, S, pat);
        apply_phasepattern(work, N, S, pat);
        if (memcmp(orig, work, N) == 0) pass++; else { fail++; printf("FAIL PHASEDIRECT S=%d\n", S); }
    }
    for (int pi = 0; pi < NPHASE_PRNG; pi++) {
        int S = PHASE_PERIODS_PRNG[pi];
        memcpy(work, orig, N);
        apply_phaseprng(work, N, S, 777);
        apply_phaseprng(work, N, S, 777);
        if (memcmp(orig, work, N) == 0) pass++; else { fail++; printf("FAIL PHASEPRNG S=%d\n", S); }
    }
    for (int wi = 0; wi < NBIGW; wi++) {
        int W = BIGNUM_WBITS[wi];
        memcpy(work, orig, N);
        apply_bignumadd(work, N, W, 4242);
        invert_bignumadd(work, N, W, 4242);
        if (memcmp(orig, work, N) == 0) pass++; else { fail++; printf("FAIL BIGNUM_ADD W=%d\n", W); }

        memcpy(work, orig, N);
        apply_bignumsub(work, N, W, 4242);
        invert_bignumsub(work, N, W, 4242);
        if (memcmp(orig, work, N) == 0) pass++; else { fail++; printf("FAIL BIGNUM_SUB W=%d\n", W); }
    }
    for (int g = 0; g < 4; g++) {
        memcpy(work, orig, N);
        apply_prngxor8(work, N, g, 999);
        apply_prngxor8(work, N, g, 999);
        if (memcmp(orig, work, N) == 0) pass++; else { fail++; printf("FAIL PRNGXOR8 genid=%d\n", g); }
    }
    for (int dd = 0; dd < 4; dd++) {
        memcpy(work, orig, N);
        apply_prngxor8_masked(work, N, dd, 888);
        apply_prngxor8_masked(work, N, dd, 888);
        if (memcmp(orig, work, N) == 0) pass++; else { fail++; printf("FAIL PRNGXOR8_MASKED dens=%d\n", dd); }
    }
    for (int ki = 0; ki < NCHUNKK; ki++) {
        int K = CHUNK_K[ki];
        for (int j = 0; j < K; j++) {
            memcpy(work, orig, N);
            apply_prngxor8_chunk(work, N, K, j, 555);
            apply_prngxor8_chunk(work, N, K, j, 555);
            if (memcmp(orig, work, N) == 0) pass++; else { fail++; printf("FAIL PRNGXOR8_CHUNK K=%d idx=%d\n", K, j); }
        }
    }
    printf("selftest: %d passed, %d failed\n", pass, fail);
    free(orig); free(work);
    return fail != 0;
}

/* ============================================================ *
 *  screen mode: rank every candidate over NB blocks of a fixed   *
 *  size, read from DATAFILE.                                     *
 * ============================================================ */
typedef struct {
    char name[40];
    double sum_net;
    double best_net;
    int    fires;
    int    nblocks;
} Result;

static int cmp_result(const void *a, const void *b) {
    const Result *ra = a, *rb = b;
    double na = ra->fires ? ra->sum_net / ra->fires : -1e18;
    double nb = rb->fires ? rb->sum_net / rb->fires : -1e18;
    if (rb->fires != ra->fires) return rb->fires - ra->fires;
    return (na < nb) ? 1 : (na > nb ? -1 : 0);
}

static int run_screen_mode(int blocksize, int NB, u32 seedlim) {
    init_ctab(blocksize);
    FILE *f = fopen(DATAFILE, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", DATAFILE); return 1; }
    u8 *all = malloc((size_t)NB * blocksize);
    NB = (int)(fread(all, 1, (size_t)NB * blocksize, f) / blocksize);
    fclose(f);
    if (NB == 0) { fprintf(stderr, "file too small\n"); return 1; }

    double seed_bits = log2((double)seedlim);
    double oh_rotate  = TAGB + log2((double)NROTATE);
    double oh_bitlag  = TAGB + log2((double)NBITLAG);
    double oh_pdirect_base = TAGB + log2((double)NPHASE_DIRECT); /* + S added per-candidate */
    double oh_pprng   = TAGB + log2((double)NPHASE_PRNG) + seed_bits;
    double oh_bigadd  = TAGB + log2((double)NBIGW) + seed_bits;
    double oh_bigsub  = oh_bigadd;
    double oh_pxor8   = TAGB + log2(4.0) + seed_bits;
    double oh_pxor8m  = TAGB + log2(4.0) + seed_bits;

    printf("bit-level screening on RESIDUAL data: %s (%d blocks x %d bytes), alpha=%g, seeds=%u (%.1f-bit)\n\n",
           DATAFILE, NB, blocksize, g_alpha, seedlim, seed_bits);

    int nchunk = 0; for (int i = 0; i < NCHUNKK; i++) nchunk += CHUNK_K[i];
    int ncand = NROTATE + NBITLAG + NPHASE_DIRECT + NPHASE_PRNG + NBIGW + NBIGW + 4 + 5 + nchunk;
    Result *res = calloc((size_t)ncand, sizeof(Result));
    Cand   *cl  = malloc((size_t)ncand * sizeof(Cand));
    int idx = 0;
    for (int k = 1; k <= NROTATE; k++) { cl[idx] = (Cand){0,k,0}; name_for_cand(cl[idx], res[idx].name); idx++; }
    for (int i = 0; i < NBITLAG; i++) { cl[idx] = (Cand){1,BITLAGS[i],0}; name_for_cand(cl[idx], res[idx].name); idx++; }
    for (int i = 0; i < NPHASE_DIRECT; i++) { cl[idx] = (Cand){2,PHASE_PERIODS_DIRECT[i],0}; name_for_cand(cl[idx], res[idx].name); idx++; }
    for (int i = 0; i < NPHASE_PRNG; i++) { cl[idx] = (Cand){3,PHASE_PERIODS_PRNG[i],0}; name_for_cand(cl[idx], res[idx].name); idx++; }
    for (int i = 0; i < NBIGW; i++) { cl[idx] = (Cand){4,BIGNUM_WBITS[i],0}; name_for_cand(cl[idx], res[idx].name); idx++; }
    for (int i = 0; i < NBIGW; i++) { cl[idx] = (Cand){5,BIGNUM_WBITS[i],0}; name_for_cand(cl[idx], res[idx].name); idx++; }
    for (int g = 0; g < 4; g++) { cl[idx] = (Cand){6,g,0}; name_for_cand(cl[idx], res[idx].name); idx++; }
    for (int dd = 0; dd < 4; dd++) { cl[idx] = (Cand){7,dd,0}; name_for_cand(cl[idx], res[idx].name); idx++; }
    for (int ki = 0; ki < NCHUNKK; ki++)
        for (int j = 0; j < CHUNK_K[ki]; j++) { cl[idx] = (Cand){8,CHUNK_K[ki],j}; name_for_cand(cl[idx], res[idx].name); idx++; }
    ncand = idx;

    #pragma omp parallel for schedule(dynamic)
    for (int c = 0; c < ncand; c++) {
        res[c].fires = 0; res[c].sum_net = 0.0; res[c].best_net = -1e18; res[c].nblocks = NB;
        double oh_chunk = TAGB + log2((double)NCHUNKK) + log2((double)cl[c].a) + seed_bits;
        for (int b = 0; b < NB; b++) {
            const u8 *d = all + (size_t)b * blocksize;
            double Sb = S_of(d, blocksize);
            double net; u32 dummy; u64 dummy64;
            switch (cl[c].kind) {
                case 0: net = eval_rotate(d, blocksize, Sb, oh_rotate, cl[c].a); break;
                case 1: net = eval_bitlagxor(d, blocksize, Sb, oh_bitlag, cl[c].a); break;
                case 2: net = eval_phasedirect(d, blocksize, Sb, oh_pdirect_base + cl[c].a, cl[c].a, &dummy64); break;
                case 3: net = eval_phaseprng(d, blocksize, Sb, oh_pprng, cl[c].a, seedlim, &dummy); break;
                case 4: net = eval_bignumadd(d, blocksize, Sb, oh_bigadd, cl[c].a, seedlim, &dummy); break;
                case 5: net = eval_bignumsub(d, blocksize, Sb, oh_bigsub, cl[c].a, seedlim, &dummy); break;
                case 6: net = eval_prngxor8(d, blocksize, Sb, oh_pxor8, cl[c].a, seedlim, &dummy); break;
                case 7: net = eval_prngxor8_masked(d, blocksize, Sb, oh_pxor8m, cl[c].a, seedlim, &dummy); break;
                default: net = eval_prngxor8_chunk(d, blocksize, Sb, oh_chunk, cl[c].a, cl[c].b, seedlim, &dummy); break;
            }
            if (net > 0.0) { res[c].fires++; res[c].sum_net += net; }
            if (net > res[c].best_net) res[c].best_net = net;
        }
    }

    qsort(res, (size_t)ncand, sizeof(Result), cmp_result);
    printf("%-20s %6s  %8s  %8s\n", "candidate", "fires", "avg net", "top net");
    printf("%-20s %6s  %8s  %8s\n", "---------", "-----", "-------", "-------");
    int shown = 0;
    for (int i = 0; i < ncand; i++) {
        if (res[i].fires == 0) continue;
        printf("%-20s %3d/%-3d %+8.2f  %+8.2f\n", res[i].name, res[i].fires, res[i].nblocks,
               res[i].sum_net / res[i].fires, res[i].best_net);
        shown++;
    }
    if (shown == 0) printf("(nothing cleared net>0 in this sample)\n");
    int nfired = 0;
    for (int i = 0; i < ncand; i++) if (res[i].fires) nfired++;
    printf("\n%d/%d candidates fired at least once (all shown above)\n", nfired, ncand);

    printf("\n--- NEVER fired (0/%d), candidates for removal ---\n", NB);
    int nzero = 0;
    for (int i = 0; i < ncand; i++) {
        if (res[i].fires != 0) continue;
        printf("%s\n", res[i].name);
        nzero++;
    }
    printf("\n%d/%d candidates never fired\n", nzero, ncand);

    free(res); free(cl); free(all);
    return 0;
}

/* ============================================================ *
 *  sizesweep mode -- loop block size = 2^x bytes over x=xmin..  *
 *  xmax, report the single best-firing candidate at each size    *
 *  and its net bits saved PER BYTE (normalized so sizes are       *
 *  comparable), then report the overall winning size. Every size  *
 *  reads the SAME leading portion of DATAFILE (rewound each time) *
 *  so the comparison isn't confounded by which bytes got tested.  *
 *  Auto-stops once 2^x exceeds the file size or a memory/compute   *
 *  budget -- 2^64 bytes is not an achievable block size on any     *
 *  real machine, so the loop reports why it stopped rather than     *
 *  trying to honor an impossible x literally.                       *
 * ============================================================ */
static int run_sizesweep(int NB_per_size, double alpha, u32 seedlim, int xmin, int xmax) {
    g_alpha = alpha;
    FILE *f = fopen(DATAFILE, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", DATAFILE); return 1; }
    fseek(f, 0, SEEK_END);
    long long filesize = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (filesize <= 0) { fprintf(stderr, "cannot stat %s\n", DATAFILE); fclose(f); return 1; }

    const long long MEMCAP = 512LL * 1024 * 1024; /* don't read more than 512MB total per size */

    printf("bit-level BLOCK-SIZE sweep on %s (%lld bytes), alpha=%g, seeds<=%u, x=%d..%d (block=2^x bytes)\n\n",
           DATAFILE, filesize, alpha, seedlim, xmin, xmax);
    printf("%-4s %14s %10s  %-20s %11s %14s\n", "x", "block(bytes)", "NB", "best candidate", "avgnet/blk", "net-bits/KB");
    printf("%-4s %14s %10s  %-20s %11s %14s\n", "-", "------------", "--", "---------------", "----------", "-----------");

    double best_perbyte = -1e300; int best_x = -1; long long best_block = 0; char best_name[40] = "(none)";

    for (int x = xmin; x <= xmax; x++) {
        if (x >= 62) { printf("x=%d: 2^%d bytes is not an achievable allocation on any real machine -- stopping sweep\n", x, x); break; }
        long long block = 1LL << x;
        if (block > filesize) {
            printf("x=%d: block size 2^%d=%lld bytes exceeds the file (%lld bytes) -- stopping sweep\n", x, x, block, filesize);
            break;
        }
        long long NBx = filesize / block;
        if (NBx > NB_per_size) NBx = NB_per_size;
        if (NBx * block > MEMCAP) NBx = MEMCAP / block;
        if (NBx < 1) { printf("x=%d: block too large for the %lldMB per-size read budget -- stopping sweep\n", x, MEMCAP / (1024*1024)); break; }

        u8 *all = malloc((size_t)(NBx * block));
        if (!all) { printf("x=%d: allocation of %lld bytes failed -- stopping sweep\n", x, NBx * block); break; }
        fseek(f, 0, SEEK_SET);
        size_t got = fread(all, 1, (size_t)(NBx * block), f);
        NBx = (long long)(got / block);
        if (NBx < 1) { free(all); printf("x=%d: short read -- stopping sweep\n", x); break; }

        init_ctab((int)block);
        u32 esl = eff_seedlim(seedlim, block);
        double seed_bits = log2((double)esl);
        long long blockbits = block * 8;

        int nchunk_here = 0; for (int i = 0; i < NCHUNKK; i++) if (CHUNK_K[i] <= block) nchunk_here += CHUNK_K[i];
        int cap_n = NROTATE + NBITLAG + NPHASE_DIRECT + NPHASE_PRNG + NBIGW + 1 + NBIGW + 1 + 4 + 5 + nchunk_here;
        Cand *cl = malloc((size_t)cap_n * sizeof(Cand));
        int ncand = 0;
        for (int k = 1; k <= NROTATE; k++) cl[ncand++] = (Cand){0,k,0};
        for (int i = 0; i < NBITLAG; i++) if (BITLAGS[i] < blockbits) cl[ncand++] = (Cand){1,BITLAGS[i],0};
        for (int i = 0; i < NPHASE_DIRECT; i++) if (PHASE_PERIODS_DIRECT[i] < blockbits) cl[ncand++] = (Cand){2,PHASE_PERIODS_DIRECT[i],0};
        for (int i = 0; i < NPHASE_PRNG; i++) if (PHASE_PERIODS_PRNG[i] < blockbits) cl[ncand++] = (Cand){3,PHASE_PERIODS_PRNG[i],0};
        { int seen = 0;
          for (int i = 0; i < NBIGW; i++) {
              long long w = BIGNUM_WBITS[i];
              if (w >= blockbits) { if (seen) continue; w = blockbits; seen = 1; }
              cl[ncand++] = (Cand){4,(int)w,0};
          }
          if (!seen) cl[ncand++] = (Cand){4,(int)blockbits,0};
        }
        { int seen = 0;
          for (int i = 0; i < NBIGW; i++) {
              long long w = BIGNUM_WBITS[i];
              if (w >= blockbits) { if (seen) continue; w = blockbits; seen = 1; }
              cl[ncand++] = (Cand){5,(int)w,0};
          }
          if (!seen) cl[ncand++] = (Cand){5,(int)blockbits,0};
        }
        for (int g = 0; g < 4; g++) cl[ncand++] = (Cand){6,g,0};
        for (int dd = 0; dd < 4; dd++) cl[ncand++] = (Cand){7,dd,0};
        for (int ki = 0; ki < NCHUNKK; ki++)
            if (CHUNK_K[ki] <= block)
                for (int j = 0; j < CHUNK_K[ki]; j++) cl[ncand++] = (Cand){8,CHUNK_K[ki],j};

        double oh_rotate  = TAGB + log2((double)NROTATE);
        double oh_bitlag  = TAGB + log2((double)NBITLAG);
        double oh_pdirect_base = TAGB + log2((double)NPHASE_DIRECT);
        double oh_pprng   = TAGB + log2((double)NPHASE_PRNG) + seed_bits;
        double oh_big     = TAGB + log2((double)(NBIGW + 1)) + seed_bits;
        double oh_pxor8   = TAGB + log2(4.0) + seed_bits;
        double oh_pxor8m  = TAGB + log2(4.0) + seed_bits;

        double *sum_net = calloc((size_t)ncand, sizeof(double));
        int    *fires   = calloc((size_t)ncand, sizeof(int));

        #pragma omp parallel for schedule(dynamic)
        for (int c = 0; c < ncand; c++) {
            double oh_chunk = TAGB + log2((double)NCHUNKK) + log2((double)cl[c].a) + seed_bits;
            for (long long b = 0; b < NBx; b++) {
                const u8 *d = all + (size_t)b * block;
                double Sb = S_of(d, (int)block);
                double net; u32 dummy; u64 dummy64;
                switch (cl[c].kind) {
                    case 0: net = eval_rotate(d, (int)block, Sb, oh_rotate, cl[c].a); break;
                    case 1: net = eval_bitlagxor(d, (int)block, Sb, oh_bitlag, cl[c].a); break;
                    case 2: net = eval_phasedirect(d, (int)block, Sb, oh_pdirect_base + cl[c].a, cl[c].a, &dummy64); break;
                    case 3: net = eval_phaseprng(d, (int)block, Sb, oh_pprng, cl[c].a, esl, &dummy); break;
                    case 4: net = eval_bignumadd(d, (int)block, Sb, oh_big, cl[c].a, esl, &dummy); break;
                    case 5: net = eval_bignumsub(d, (int)block, Sb, oh_big, cl[c].a, esl, &dummy); break;
                    case 6: net = eval_prngxor8(d, (int)block, Sb, oh_pxor8, cl[c].a, esl, &dummy); break;
                    case 7: net = eval_prngxor8_masked(d, (int)block, Sb, oh_pxor8m, cl[c].a, esl, &dummy); break;
                    default: net = eval_prngxor8_chunk(d, (int)block, Sb, oh_chunk, cl[c].a, cl[c].b, esl, &dummy); break;
                }
                if (net > 0.0) { sum_net[c] += net; fires[c]++; }
            }
        }

        int best_local = -1; double best_avg = -1e300;
        for (int c = 0; c < ncand; c++) {
            if (!fires[c]) continue;
            double avg = sum_net[c] / fires[c];
            if (avg > best_avg) { best_avg = avg; best_local = c; }
        }
        char name[40] = "(none fired)";
        double perbyte = 0.0;
        if (best_local >= 0) {
            name_for_cand(cl[best_local], name);
            perbyte = (best_avg / (double)block) * 1024.0; /* bits saved per KB of input, for a readable scale */
        }
        printf("%-4d %14lld %10lld  %-20s %11.3f %14.4f\n", x, block, NBx, name, (best_local >= 0 ? best_avg : 0.0), perbyte);
        if (best_local >= 0 && perbyte > best_perbyte) {
            best_perbyte = perbyte; best_x = x; best_block = block; strncpy(best_name, name, 39);
        }

        free(sum_net); free(fires); free(cl); free(all);
    }
    fclose(f);

    if (best_x >= 0)
        printf("\nBEST BLOCK SIZE: 2^%d = %lld bytes (winning candidate %s, %.4f net bits/KB)\n",
               best_x, best_block, best_name, best_perbyte);
    else
        printf("\nnothing fired at any block size tried\n");
    return 0;
}

/* ============================================================ *
 *  layered mode: real multi-round greedy loop over all bit-level *
 *  candidates, mirroring reduce2's own compress() -- plain greedy *
 *  (depth=1) only; recursive lookahead like instructionlab.c's    *
 *  layered mode is a natural extension, not implemented here.     *
 * ============================================================ */
#define NKINDS 9
static const char *KIND_NAME[NKINDS] = {
    "ROTATE","BITLAG","PHASEDIRECT","PHASEPRNG","BIGNUM_ADD","BIGNUM_SUB",
    "PRNGXOR8","PRNGXOR8_MASK","PRNGXOR8_CHUNK"
};

static int build_lcand_list(Cand *cl) {
    int idx = 0;
    for (int k = 1; k <= NROTATE; k++) cl[idx++] = (Cand){0,k,0};
    for (int i = 0; i < NBITLAG; i++) cl[idx++] = (Cand){1,BITLAGS[i],0};
    for (int i = 0; i < NPHASE_DIRECT; i++) cl[idx++] = (Cand){2,PHASE_PERIODS_DIRECT[i],0};
    for (int i = 0; i < NPHASE_PRNG; i++) cl[idx++] = (Cand){3,PHASE_PERIODS_PRNG[i],0};
    for (int i = 0; i < NBIGW; i++) cl[idx++] = (Cand){4,BIGNUM_WBITS[i],0};
    for (int i = 0; i < NBIGW; i++) cl[idx++] = (Cand){5,BIGNUM_WBITS[i],0};
    for (int g = 0; g < 4; g++) cl[idx++] = (Cand){6,g,0};
    for (int dd = 0; dd < 4; dd++) cl[idx++] = (Cand){7,dd,0};
    for (int ki = 0; ki < NCHUNKK; ki++)
        for (int j = 0; j < CHUNK_K[ki]; j++) cl[idx++] = (Cand){8,CHUNK_K[ki],j};
    return idx;
}
static double oh_for_kind(int kind, int a, double seed_bits) {
    switch (kind) {
        case 0: return TAGB + log2((double)NROTATE);
        case 1: return TAGB + log2((double)NBITLAG);
        case 2: return TAGB + log2((double)NPHASE_DIRECT) + a;   /* a == S, real pattern payload */
        case 3: return TAGB + log2((double)NPHASE_PRNG) + seed_bits;
        case 4: case 5: return TAGB + log2((double)NBIGW) + seed_bits;   /* BIGNUM_ADD / BIGNUM_SUB */
        case 6: return TAGB + log2(4.0) + seed_bits;                     /* PRNGXOR8: which generator */
        case 7: return TAGB + log2(4.0) + seed_bits;                     /* PRNGXOR8_MASKED: which density */
        default: return TAGB + log2((double)NCHUNKK) + log2((double)a) + seed_bits; /* PRNGXOR8_CHUNK: which K + which index */
    }
}
static double eval_lcand(const u8 *d, int n, double Sb, Cand c, double seed_bits, u32 seedlim, u64 *amp_out) {
    double oh = oh_for_kind(c.kind, c.a, seed_bits);
    u32 seed32;
    double net;
    switch (c.kind) {
        case 0: net = eval_rotate(d, n, Sb, oh, c.a); if (amp_out) *amp_out = 0; return net;
        case 1: net = eval_bitlagxor(d, n, Sb, oh, c.a); if (amp_out) *amp_out = 0; return net;
        case 2: { u64 pat; net = eval_phasedirect(d, n, Sb, oh, c.a, &pat); if (amp_out) *amp_out = pat; return net; }
        case 3: net = eval_phaseprng(d, n, Sb, oh, c.a, seedlim, &seed32); if (amp_out) *amp_out = seed32; return net;
        case 4: net = eval_bignumadd(d, n, Sb, oh, c.a, seedlim, &seed32); if (amp_out) *amp_out = seed32; return net;
        case 5: net = eval_bignumsub(d, n, Sb, oh, c.a, seedlim, &seed32); if (amp_out) *amp_out = seed32; return net;
        case 6: net = eval_prngxor8(d, n, Sb, oh, c.a, seedlim, &seed32); if (amp_out) *amp_out = seed32; return net;
        case 7: net = eval_prngxor8_masked(d, n, Sb, oh, c.a, seedlim, &seed32); if (amp_out) *amp_out = seed32; return net;
        default: net = eval_prngxor8_chunk(d, n, Sb, oh, c.a, c.b, seedlim, &seed32); if (amp_out) *amp_out = seed32; return net;
    }
}
static void apply_lcand(u8 *d, int n, Cand c, u64 amp) {
    switch (c.kind) {
        case 0: apply_rotate(d, n, c.a); break;
        case 1: apply_bitlagxor(d, n, c.a); break;
        case 2: apply_phasepattern(d, n, c.a, amp); break;
        case 3: apply_phaseprng(d, n, c.a, (u32)amp); break;
        case 4: apply_bignumadd(d, n, c.a, (u32)amp); break;
        case 5: apply_bignumsub(d, n, c.a, (u32)amp); break;
        case 6: apply_prngxor8(d, n, c.a, (u32)amp); break;
        case 7: apply_prngxor8_masked(d, n, c.a, (u32)amp); break;
        default: apply_prngxor8_chunk(d, n, c.a, c.b, (u32)amp); break;
    }
}

static int run_layered_mode(int blocksize, int NB, double alpha, u32 seedlim, int maxlayers) {
    g_alpha = alpha;
    init_ctab(blocksize);
    FILE *f = fopen(DATAFILE, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", DATAFILE); return 1; }
    u8 *all = malloc((size_t)NB * blocksize);
    NB = (int)(fread(all, 1, (size_t)NB * blocksize, f) / blocksize);
    fclose(f);
    if (NB == 0) { fprintf(stderr, "file too small\n"); return 1; }

    double seed_bits = log2((double)seedlim);
    int nchunk = 0; for (int i = 0; i < NCHUNKK; i++) nchunk += CHUNK_K[i];
    int cap = NROTATE + NBITLAG + NPHASE_DIRECT + NPHASE_PRNG + NBIGW + NBIGW + 4 + 5 + nchunk;
    Cand *lcl = malloc((size_t)cap * sizeof(Cand));
    int ncand = build_lcand_list(lcl);
    printf("bit-level layered mode on %s: %d blocks x %d bytes, alpha=%g, seeds=%u (%.1f-bit), maxlayers=%d, %d candidates\n",
           DATAFILE, NB, blocksize, alpha, seedlim, seed_bits, maxlayers, ncand);

    double *allnet = malloc((size_t)ncand * sizeof(double));
    u64    *allamp = malloc((size_t)ncand * sizeof(u64));

    long long total_layers = 0;
    double total_net_all = 0.0;
    int kind_fires[NKINDS] = {0};
    double kind_net[NKINDS] = {0};

    for (int b = 0; b < NB; b++) {
        u8 *work = malloc((size_t)blocksize);
        memcpy(work, all + (size_t)b * blocksize, blocksize);
        double block_net = 0.0;
        int layer;
        for (layer = 0; layer < maxlayers; layer++) {
            double Sb = S_of(work, blocksize);
            #pragma omp parallel for schedule(dynamic)
            for (int c = 0; c < ncand; c++) {
                u64 amp;
                double net = eval_lcand(work, blocksize, Sb, lcl[c], seed_bits, seedlim, &amp);
                allnet[c] = net; allamp[c] = amp;
            }
            double best_net = -1e18; int best_i = -1;
            for (int c = 0; c < ncand; c++)
                if (allnet[c] > best_net) { best_net = allnet[c]; best_i = c; }
            if (best_net <= 0.0) break;
            apply_lcand(work, blocksize, lcl[best_i], allamp[best_i]);
            block_net += best_net;
            kind_fires[lcl[best_i].kind]++;
            kind_net[lcl[best_i].kind] += best_net;
            if (NB <= 5)
                printf("  block %d layer %d: %-12s net=%+.2f\n", b, layer, KIND_NAME[lcl[best_i].kind], best_net);
        }
        total_layers += layer;
        total_net_all += block_net;
        printf("block %2d: %d layers, total net=%+.2f\n", b, layer, block_net);
        fflush(stdout);
        free(work);
    }
    printf("\n=== bit-level layered summary: %d blocks ===\n", NB);
    printf("avg layers/block: %.2f   avg net/block: %+.2f\n", (double)total_layers / NB, total_net_all / NB);
    printf("%-12s %6s %10s\n", "kind", "fires", "avg net");
    printf("%-12s %6s %10s\n", "----", "-----", "-------");
    for (int k = 0; k < NKINDS; k++)
        if (kind_fires[k])
            printf("%-12s %6d %+10.2f\n", KIND_NAME[k], kind_fires[k], kind_net[k] / kind_fires[k]);

    free(allnet); free(allamp); free(lcl); free(all);
    return 0;
}

/* ============================================================ *
 *  main -- ALWAYS reads DATAFILE (compresseddata.bin); no file    *
 *  argument accepted in any mode.                                  *
 * ============================================================ */
int main(int argc, char **argv) {
    if (argc > 1 && strcmp(argv[1], "selftest") == 0)
        return run_selftest();
    if (argc > 1 && strcmp(argv[1], "sizesweep") == 0) {
        int NB_per_size = (argc > 2) ? atoi(argv[2]) : 8;
        double alpha = (argc > 3) ? atof(argv[3]) : 32.0;
        u32 seedlim = (argc > 4) ? (u32)atoi(argv[4]) : 256;
        int xmin = (argc > 5) ? atoi(argv[5]) : 2;
        int xmax = (argc > 6) ? atoi(argv[6]) : 64;
        return run_sizesweep(NB_per_size, alpha, seedlim, xmin, xmax);
    }
    if (argc > 1 && strcmp(argv[1], "layered") == 0) {
        int NB = (argc > 2) ? atoi(argv[2]) : 10;
        double alpha = (argc > 3) ? atof(argv[3]) : 32.0;
        u32 seedlim = (argc > 4) ? (u32)atoi(argv[4]) : 256;
        int maxlayers = (argc > 5) ? atoi(argv[5]) : 20;
        return run_layered_mode(DEFAULT_BLOCK, NB, alpha, seedlim, maxlayers);
    }
    int NB = (argc > 1) ? atoi(argv[1]) : 20;
    if (argc > 2) g_alpha = atof(argv[2]);
    u32 seedlim = (argc > 3) ? (u32)atoi(argv[3]) : 256;
    return run_screen_mode(DEFAULT_BLOCK, NB, seedlim);
}
