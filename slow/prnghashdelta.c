/* entropylowerer.c — per-chunk hash-seed search to minimize delta magnitude.
 *
 * For each CHUNKSIZE-byte chunk of the input file: add an avalanche-hash
 * byte to every byte in the chunk (one hash byte per (seed,position) pair,
 * looked up from a precomputed SEEDSPACE*CHUNKSIZE table -- unlike a
 * stream PRNG, an avalanche hash has no sequential state, so every
 * (seed,pos) result is independent and gets cached up front instead of
 * recomputed on every search trial), then delta-encode the hashed
 * bytes: delta[0] = hashed[0] (stored as-is, not differenced -- this is the
 * anchor, for free, inside the 8 output bytes), delta[i] = hashed[i]-
 * hashed[i-1] for i>0, mod 256. No wraparound: hashed[0] is never
 * subtracted from anything, so nothing cancels and the transform is a
 * lossless bijection -- delta[0] alone always recovers the DC level, no
 * extra anchor byte required (see the invert_delta_layer comment for why
 * an earlier wraparound version needed one).
 * DELTALAYERS controls how many times that delta is applied in a row --
 * layer 2 is the delta of layer 1's output, layer 3 the delta of that,
 * etc. Score a seed by the summed unsigned byte values (0..255 each, no
 * signed reinterpretation -- a delta of 255 counts as 255, not -1) of the
 * LAST layer only. Exhaustively search the SEEDBITS-bit seed space
 * (0..2^SEEDBITS-1) per chunk and keep whichever seed minimizes that score.
 *
 * NUMOFCHUNKS is the chunk count directly (not derived from bytes-read/
 * CHUNKSIZE) -- the file is read as exactly NUMOFCHUNKS*CHUNKSIZE bytes,
 * and only falls back to a smaller chunk count if the file can't supply
 * that much. Cost is NUMOFCHUNKS * SEEDSPACE * CHUNKSIZE byte-ops, so
 * running this over the whole 198MB compresseddata.bin (~24.8M chunks)
 * is not feasible in one pass. Raise NUMOFCHUNKS for a bigger sample, at
 * roughly linear cost; SEEDBITS is the other knob on that same cost.
 *
 * Build: gcc -O2 -o entropylowerer entropylowerer.c
 */
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>

typedef unsigned char u8;
typedef uint16_t      u16;
typedef uint32_t      u32;
typedef uint64_t      u64;

#define INPUT_PATH  "C:\\Users\\lukac\\Documents\\compressor\\compresseddata.bin"
#define CHUNKSIZE   8
#define NUMOFCHUNKS 4096
#define SEEDBITS 8

/* Bits of seed searched per chunk (seed range 0..2^SEEDBITS-1). Valid from
 * 1 to 32 -- the hash is a 32-bit avalanche mix, so 32 bits is the hard
 * ceiling. Lower SEEDBITS = cheaper to store per chunk but a smaller,
 * coarser search space (weaker best-case reduction); higher SEEDBITS costs
 * proportionally more search time (2x per extra bit) AND more memory for
 * the cached hash table (SEEDSPACE*CHUNKSIZE bytes -- keep this in mind
 * before raising SEEDBITS much past ~24, where the table hits ~128MB). */
#if SEEDBITS < 1 || SEEDBITS > 32
#error "SEEDBITS must be between 1 and 32"
#endif
#define SEEDSPACE ((u64)1 << SEEDBITS)

/* Hard safety cap on delta layers per seed -- NOT the driving parameter
 * anymore. The search now runs each seed until it finds a genuine repeat
 * (checked against every state visited so far for that seed, not just the
 * first one -- see chunk_score_layers), and stops there on its own. A
 * sweep already proved the true ceiling for CHUNKSIZE=8 is exactly 1024
 * (see the file's earlier history/derivation: D=I-N is nilpotent-based and
 * invertible, so orbits are pure cycles, and Kummer's theorem on
 * C(k,j) mod 256 puts the guaranteed full-return point at 2^10). This is
 * sized well above that (4x margin) purely as a defensive stop in case
 * that assumption is ever wrong; main() warns loudly if it's ever hit. */
#define DELTALAYERS 1024
#if DELTALAYERS < 1
#error "DELTALAYERS must be >= 1"
#endif

/* Avalanche hash (MurmurHash3-style finalizer): a single fixed-cost mix of
 * (seed,pos) into a well-scattered byte, no sequential state. Position is
 * folded in before mixing so every (seed,pos) pair gets an independent,
 * uncorrelated result -- unlike the old xorshift32 stream, this can be
 * evaluated for any (seed,pos) in isolation, which is what makes caching
 * the whole table up front possible. */
static inline u8 avalanche_hash(u32 seed, int pos) {
    u32 h = seed + (u32)pos * 0x9E3779B1u;
    h ^= h >> 16;
    h *= 0x85ebca6bu;
    h ^= h >> 13;
    h *= 0xc2b2ae35u;
    h ^= h >> 16;
    return (u8)h;
}

/* Precomputed hash_table[seed*CHUNKSIZE + pos] = avalanche_hash(seed, pos),
 * built once in main() before any search starts. Turns every hash lookup
 * in the search hot loop into a single memory read instead of a handful of
 * multiply/shift ops. */
static u8 *hash_table = NULL;

static void build_hash_table(void) {
    hash_table = malloc((size_t)SEEDSPACE * CHUNKSIZE);
    if (!hash_table) {
        fprintf(stderr, "hash table alloc failed: SEEDSPACE=%llu * CHUNKSIZE=%d = %llu bytes -- lower SEEDBITS\n",
                (unsigned long long)SEEDSPACE, CHUNKSIZE, (unsigned long long)SEEDSPACE * CHUNKSIZE);
        exit(1);
    }
    for (u64 seed = 0; seed < SEEDSPACE; seed++)
        for (int pos = 0; pos < CHUNKSIZE; pos++)
            hash_table[seed * CHUNKSIZE + pos] = avalanche_hash((u32)seed, pos);
}

/* Hash a chunk via the cached table: hashed_out[i] = chunk[i] + hash(seed,i). */
static void hash_chunk(const u8 *chunk, u32 seed, u8 *hashed_out) {
    const u8 *row = hash_table + (u64)seed * CHUNKSIZE;
    for (int i = 0; i < CHUNKSIZE; i++)
        hashed_out[i] = (u8)(chunk[i] + row[i]);
}

/* One delta pass, no wraparound: out[0] = in[0] (stored directly -- this
 * is what makes the transform bijective, see the file header comment),
 * out[i] = in[i]-in[i-1] for i>0. in and out may not alias. */
static void apply_delta_layer(const u8 *in, u8 *out) {
    out[0] = in[0];
    for (int i = 1; i < CHUNKSIZE; i++)
        out[i] = (u8)(in[i] - in[i - 1]);
}

/* Hash every byte of the chunk via the cached table, then keep applying
 * delta passes back-to-back for as long as it takes to find a repeat --
 * NOT bounded by any fixed layer count except the DELTALAYERS safety cap.
 * scores_out[k-1] gets the summed unsigned byte value after k layers
 * (k=1..*ncomputed_out).
 *
 * Loop detection here is general: each new state is checked against EVERY
 * state visited so far for this seed (not just the very first one), so a
 * repeat is caught whenever/wherever it actually happens, not assumed.
 * (Separately: apply_delta_layer is provably a bijection on the
 * CHUNKSIZE-byte state -- I-N for nilpotent shift N, invertible via
 * I+N+N^2+...+N^7, see invert_delta_layer -- and a bijection of a finite
 * set only has pure cycles, no "rho tail", so in theory any repeat must be
 * the very first state. This general check doesn't assume that theory
 * holds; it just looks.) Once a repeat is found, every later layer is a
 * verbatim repeat of what's already in scores_out, so the search stops
 * there. *ncomputed_out receives how many layers were actually computed.
 * Hot path used by the seed search. */
static void chunk_score_layers(const u8 *chunk, u32 seed, u32 scores_out[DELTALAYERS], int *ncomputed_out) {
    static u8 states[DELTALAYERS + 1][CHUNKSIZE];
    hash_chunk(chunk, seed, states[0]);

    int layer;
    int found_repeat = 0;
    for (layer = 0; layer < DELTALAYERS; layer++) {
        apply_delta_layer(states[layer], states[layer + 1]);
        u32 score = 0;
        for (int i = 0; i < CHUNKSIZE; i++) score += states[layer + 1][i];
        scores_out[layer] = score;

        for (int k = 0; k <= layer; k++) {
            if (memcmp(states[layer + 1], states[k], CHUNKSIZE) == 0) { found_repeat = 1; break; }
        }
        if (found_repeat) { layer++; break; }
    }
    /* NOTE: layer==DELTALAYERS is not itself a failure signal -- a repeat
     * found on the very last allowed iteration also lands here via the
     * layer++ above. Only found_repeat==0 means the cap was hit for real. */
    if (!found_repeat)
        fprintf(stderr, "WARNING: seed=%u hit the DELTALAYERS safety cap (%d) with no repeat found -- "
                         "the pure-cycle assumption may be wrong, or DELTALAYERS needs raising\n", seed, DELTALAYERS);
    *ncomputed_out = layer;
}

/* Same as chunk_score_layers but with NO hash applied at all -- the "no
 * transform" baseline. Deliberately NOT the same as chunk_score_layers
 * with seed=0: an avalanche hash's seed=0 output is not the identity (only
 * position 0 lands on 0; positions 1..7 still get real avalanche-mixed
 * bytes), unlike the old xorshift32 stream where state=0 was a genuine
 * all-zero no-op. This keeps the baseline honestly hash-free regardless of
 * what the hash function does at seed=0. Same general loop detection. */
static void raw_score_layers(const u8 *chunk, u32 scores_out[DELTALAYERS], int *ncomputed_out) {
    static u8 states[DELTALAYERS + 1][CHUNKSIZE];
    for (int i = 0; i < CHUNKSIZE; i++) states[0][i] = chunk[i];

    int layer;
    int found_repeat = 0;
    for (layer = 0; layer < DELTALAYERS; layer++) {
        apply_delta_layer(states[layer], states[layer + 1]);
        u32 score = 0;
        for (int i = 0; i < CHUNKSIZE; i++) score += states[layer + 1][i];
        scores_out[layer] = score;

        for (int k = 0; k <= layer; k++) {
            if (memcmp(states[layer + 1], states[k], CHUNKSIZE) == 0) { found_repeat = 1; break; }
        }
        if (found_repeat) { layer++; break; }
    }
    if (!found_repeat)
        fprintf(stderr, "WARNING: raw baseline hit the DELTALAYERS safety cap (%d) with no repeat found\n", DELTALAYERS);
    *ncomputed_out = layer;
}

/* Look up the score at a 1-indexed layer depth that may be beyond what was
 * actually computed, using the detected cycle period `ncomputed` to wrap
 * around: layer P (the period itself) repeats layer P's own score being
 * the closure point, layer P+1 repeats layer 1's score, etc. Valid because
 * once a cycle closes, everything from there on is a verbatim repeat of
 * layers 1..ncomputed (see chunk_score_layers). */
static u32 periodic_score_at(const u32 *scores, int ncomputed, int want_layer) {
    int idx = (want_layer - 1) % ncomputed;
    return scores[idx];
}

/* Same computation as chunk_score, but keeps every intermediate layer for
 * inspection/printing. layers_out must have DELTALAYERS+1 rows of
 * CHUNKSIZE bytes: row 0 = hashed, row k (1<=k<=DELTALAYERS) = the k-th
 * delta layer. Not used in the search hot path -- only for detail dumps. */
static u32 chunk_transform_layers(const u8 *chunk, u32 seed, u8 layers_out[][CHUNKSIZE]) {
    hash_chunk(chunk, seed, layers_out[0]);

    for (int layer = 1; layer <= DELTALAYERS; layer++)
        apply_delta_layer(layers_out[layer - 1], layers_out[layer]);

    u32 score = 0;
    for (int i = 0; i < CHUNKSIZE; i++) score += layers_out[DELTALAYERS][i];
    return score;
}

/* Same as chunk_transform_layers but with NO hash applied (see
 * raw_score_layers for why this can't just be chunk_transform_layers with
 * seed=0 anymore). */
static void raw_transform_layers(const u8 *chunk, u8 layers_out[][CHUNKSIZE]) {
    for (int i = 0; i < CHUNKSIZE; i++) layers_out[0][i] = chunk[i];

    for (int layer = 1; layer <= DELTALAYERS; layer++)
        apply_delta_layer(layers_out[layer - 1], layers_out[layer]);
}

/* Inverse of one apply_delta_layer call: delta[0] IS in[0] directly (no
 * wraparound, so nothing to separately supply), and in[i] = in[i-1] +
 * delta[i] for i>0. Fully self-contained -- no external anchor byte
 * needed. delta and out may not alias. */
static void invert_delta_layer(const u8 *delta, u8 *out) {
    out[0] = delta[0];
    for (int i = 1; i < CHUNKSIZE; i++)
        out[i] = (u8)(out[i - 1] + delta[i]);
}

/* Full inverse of chunk_transform_layers run to depth `layer`: given just
 * the delta bytes at that depth and the seed, reconstruct the original
 * CHUNKSIZE raw bytes. No anchors required (see invert_delta_layer). */
static void chunk_inverse(const u8 *final_layer, int layer, u32 seed, u8 *raw_out) {
    u8 cur[CHUNKSIZE];
    for (int i = 0; i < CHUNKSIZE; i++) cur[i] = final_layer[i];

    for (int L = layer; L >= 1; L--) {
        u8 prev[CHUNKSIZE];
        invert_delta_layer(cur, prev);
        for (int i = 0; i < CHUNKSIZE; i++) cur[i] = prev[i];
    }
    /* cur now holds the hashed values (layer 0); reverse the hash-add */
    const u8 *row = hash_table + (u64)seed * CHUNKSIZE;
    for (int i = 0; i < CHUNKSIZE; i++)
        raw_out[i] = (u8)(cur[i] - row[i]);
}

static void print_bytes(const char *label, const u8 *b) {
    printf("%s", label);
    for (int i = 0; i < CHUNKSIZE; i++) printf(" %3u", b[i]);
    printf("\n");
}

int main(void) {
    build_hash_table();
    printf("hash table: %llu bytes cached (SEEDSPACE=%llu * CHUNKSIZE=%d)\n",
           (unsigned long long)SEEDSPACE * CHUNKSIZE, (unsigned long long)SEEDSPACE, CHUNKSIZE);

    FILE *f = fopen(INPUT_PATH, "rb");
    if (!f) { perror(INPUT_PATH); return 1; }

    size_t nchunks = NUMOFCHUNKS;
    size_t want = nchunks * CHUNKSIZE;
    u8 *buf = malloc(want);
    if (!buf) { fprintf(stderr, "malloc failed\n"); fclose(f); return 1; }
    size_t got = fread(buf, 1, want, f);
    fclose(f);

    if (got < want) {
        nchunks = got / CHUNKSIZE;
        fprintf(stderr, "warning: file supplied only %zu of %zu requested bytes; "
                         "using %zu chunks instead of %d\n", got, want, nchunks, NUMOFCHUNKS);
    }

    printf("input:  %s\n", INPUT_PATH);
    printf("chunks: %zu (%d bytes each, %zu bytes total)\n", nchunks, CHUNKSIZE, nchunks * (size_t)CHUNKSIZE);
    printf("seedbits: %d (seedspace=%llu)\n", SEEDBITS, (unsigned long long)SEEDSPACE);

    u32 *best_seed  = malloc(nchunks * sizeof(u32));
    u32 *best_score = malloc(nchunks * sizeof(u32));
    int *best_layer = malloc(nchunks * sizeof(int));
    u32 *zero_score = malloc(nchunks * sizeof(u32));  /* no-hash score at the WINNING layer, same-layer comparison */

    u64 total_best = 0, total_zero = 0;
    u64 cyc_sum = 0, cyc_count = 0;
    int cyc_min = DELTALAYERS, cyc_max = 0;
    u32 scores[DELTALAYERS];
    for (size_t c = 0; c < nchunks; c++) {
        const u8 *chunk = buf + c * CHUNKSIZE;
        u32 bestv = 0xFFFFFFFFu;
        u32 bests = 0;
        int bestlayer = 1;
        for (u64 seed = 0; seed < SEEDSPACE; seed++) {
            int ncomputed;
            chunk_score_layers(chunk, (u32)seed, scores, &ncomputed);
            for (int layer = 0; layer < ncomputed; layer++) {
                if (scores[layer] < bestv) { bestv = scores[layer]; bests = (u32)seed; bestlayer = layer + 1; }
            }
            cyc_sum += (u64)ncomputed; cyc_count++;
            if (ncomputed < cyc_min) cyc_min = ncomputed;
            if (ncomputed > cyc_max) cyc_max = ncomputed;
        }
        best_seed[c]  = bests;
        best_score[c] = bestv;
        best_layer[c] = bestlayer;

        int raw_ncomputed;
        raw_score_layers(chunk, scores, &raw_ncomputed);
        zero_score[c] = periodic_score_at(scores, raw_ncomputed, bestlayer);

        total_best += bestv;
        total_zero += zero_score[c];
    }

    printf("delta cycle length: min=%d max=%d avg=%.1f (over %llu seed trials, DELTALAYERS cap=%d%s)\n",
           cyc_min, cyc_max, cyc_count ? (double)cyc_sum / (double)cyc_count : 0.0,
           (unsigned long long)cyc_count, DELTALAYERS, cyc_max == DELTALAYERS ? ", cap WAS hit" : ", cap never hit");

    /* Winning-layer distribution across ALL chunks (not just the first 20
     * printed below) -- this is the practically useful number for picking
     * a DELTALAYERS cap: what fraction of chunks would still find their
     * best result if the search stopped at various depths. */
    if (nchunks > 0) {
        int wl_min = best_layer[0], wl_max = best_layer[0];
        u64 wl_sum = 0;
        for (size_t c = 0; c < nchunks; c++) {
            if (best_layer[c] < wl_min) wl_min = best_layer[c];
            if (best_layer[c] > wl_max) wl_max = best_layer[c];
            wl_sum += (u64)best_layer[c];
        }
        printf("winning layer:      min=%d max=%d avg=%.1f (across %zu chunks)\n",
               wl_min, wl_max, (double)wl_sum / (double)nchunks, nchunks);

        static const int THRESH[] = {8, 16, 32, 64, 128, 256, 512, 1024};
        printf("coverage if DELTALAYERS were capped at:\n");
        for (size_t t = 0; t < sizeof(THRESH) / sizeof(THRESH[0]); t++) {
            size_t covered = 0;
            for (size_t c = 0; c < nchunks; c++) if (best_layer[c] <= THRESH[t]) covered++;
            printf("  %5d: %5zu/%-5zu chunks (%.1f%%) would still find their best result\n",
                   THRESH[t], covered, nchunks, 100.0 * (double)covered / (double)nchunks);
        }
    }

    printf("\nno-hash baseline (same layer as winner) total delta-magnitude: %llu\n", (unsigned long long)total_zero);
    printf("best (seed,layer)                       total delta-magnitude: %llu\n", (unsigned long long)total_best);
    if (total_zero > 0)
        printf("reduction: %.2f%%\n", 100.0 * (double)(total_zero - total_best) / (double)total_zero);

    /* Round-trip verification: with no wraparound, delta[0] carries the DC
     * level directly, so the final delta bytes + seed are self-contained --
     * no separate anchor bytes needed. Reconstruct every chunk and confirm
     * it matches the original exactly. */
    size_t roundtrip_ok = 0;
    for (size_t c = 0; c < nchunks; c++) {
        const u8 *chunk = buf + c * CHUNKSIZE;
        u8 layers[DELTALAYERS + 1][CHUNKSIZE];
        chunk_transform_layers(chunk, best_seed[c], layers);

        u8 recon[CHUNKSIZE];
        chunk_inverse(layers[best_layer[c]], best_layer[c], best_seed[c], recon);

        if (memcmp(chunk, recon, CHUNKSIZE) == 0) roundtrip_ok++;
    }
    printf("\nround-trip: %zu/%zu chunks reconstruct exactly from (final delta bytes + %d-bit seed, no anchors)\n",
           roundtrip_ok, nchunks, SEEDBITS);

    size_t show = nchunks < 20 ? nchunks : 20;
    printf("\nfirst %zu chunks:\n", show);
    for (size_t c = 0; c < show; c++)
        printf("  chunk %4zu: seed=%6u layer=%d/%d  score=%u (no-hash L%d score=%u)\n",
               c, best_seed[c], best_layer[c], DELTALAYERS, best_score[c], best_layer[c], zero_score[c]);

    if (nchunks > 0) {
        const u8 *chunk0 = buf;
        u8 layers0[DELTALAYERS + 1][CHUNKSIZE];
        u8 layersB[DELTALAYERS + 1][CHUNKSIZE];
        raw_transform_layers(chunk0, layers0);
        chunk_transform_layers(chunk0, best_seed[0], layersB);

        u32 scores0[DELTALAYERS], scoresB[DELTALAYERS];
        int ncomputed0, ncomputedB;
        raw_score_layers(chunk0, scores0, &ncomputed0);
        chunk_score_layers(chunk0, best_seed[0], scoresB, &ncomputedB);

        printf("\nchunk 0 detail (%d delta layer%s cap, best found at layer %d; no-hash cycle=%d, best-seed cycle=%d):\n",
               DELTALAYERS, DELTALAYERS == 1 ? "" : "s", best_layer[0], ncomputed0, ncomputedB);
        print_bytes("  raw              :", chunk0);

        /* Cycle lengths can run to ~1000+ layers now that the search isn't
         * capped -- printing every one would be unreadable, so show only a
         * short window (first DETAIL_CAP layers) plus the specific winning
         * layer's row, rather than the full dump. */
        const int DETAIL_CAP = 15;

        printf("  no-hash baseline score, first %d of %d layers:", DETAIL_CAP < ncomputed0 ? DETAIL_CAP : ncomputed0, ncomputed0);
        for (int layer = 1; layer <= ncomputed0 && layer <= DETAIL_CAP; layer++)
            printf(" L%d=%u", layer, periodic_score_at(scores0, ncomputed0, layer));
        printf("\n");
        print_bytes("    unhashed       :", layers0[0]);
        for (int layer = 1; layer <= ncomputed0 && layer <= DETAIL_CAP; layer++) {
            char label[32];
            snprintf(label, sizeof(label), "    delta L%-2d      :", layer);
            print_bytes(label, layers0[layer]);
        }
        if (ncomputed0 > DETAIL_CAP) printf("    ... (%d more layers, cycle closes at L%d)\n", ncomputed0 - DETAIL_CAP, ncomputed0);

        printf("\n  seed=%u (best, layer %d, score=%u), first %d of %d layers:",
               best_seed[0], best_layer[0], best_score[0], DETAIL_CAP < ncomputedB ? DETAIL_CAP : ncomputedB, ncomputedB);
        for (int layer = 1; layer <= ncomputedB && layer <= DETAIL_CAP; layer++)
            printf(" L%d=%u", layer, periodic_score_at(scoresB, ncomputedB, layer));
        printf("\n");
        print_bytes("    hashed         :", layersB[0]);
        for (int layer = 1; layer <= ncomputedB && layer <= DETAIL_CAP; layer++) {
            char label[32];
            snprintf(label, sizeof(label), "    delta L%-2d      :", layer);
            print_bytes(label, layersB[layer]);
        }
        if (ncomputedB > DETAIL_CAP) {
            printf("    ... (%d more layers, cycle closes at L%d)\n", ncomputedB - DETAIL_CAP, ncomputedB);
            if (best_layer[0] > DETAIL_CAP) {
                char label[48];
                snprintf(label, sizeof(label), "    delta L%-2d (WINNER):", best_layer[0]);
                print_bytes(label, layersB[best_layer[0]]);
            }
        }

        u8 recon0[CHUNKSIZE];
        chunk_inverse(layersB[best_layer[0]], best_layer[0], best_seed[0], recon0);

        printf("\n  round-trip using seed=%u, layer=%d, no anchors needed (%s):\n",
               best_seed[0], best_layer[0],
               memcmp(chunk0, recon0, CHUNKSIZE) == 0 ? "MATCH" : "MISMATCH");
        print_bytes("    reconstructed  :", recon0);
        print_bytes("    original       :", chunk0);
    }

    free(best_seed); free(best_score); free(best_layer); free(zero_score); free(buf); free(hash_table);
    return 0;
}
