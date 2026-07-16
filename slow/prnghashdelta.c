/* entropylowerer.c — per-chunk hash-seed search to minimize delta magnitude.
 *
 * For each CHUNKSIZE-byte chunk of the input file: add a xorshift32-derived
 * hash byte to every byte in the chunk (one hash byte per position, all
 * derived from a single seed, SEEDBITS wide), then delta-encode the hashed
 * bytes with wraparound: delta[0] = hashed[0]-hashed[CHUNKSIZE-1] (last byte
 * of the same chunk), delta[i] = hashed[i]-hashed[i-1] for i>0, both mod 256.
 * DELTALAYERS controls how many times that wraparound delta is applied in a
 * row -- layer 2 is the delta of layer 1's output, layer 3 the delta of
 * that, etc. Score a seed by the summed unsigned byte values (0..255 each,
 * no wraparound reinterpretation -- a delta of 255 counts as 255, not -1)
 * of the LAST layer only. Exhaustively search the SEEDBITS-bit seed space
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
#define NUMOFCHUNKS 4
#define SEEDBITS 8

/* Bits of seed searched per chunk (seed range 0..2^SEEDBITS-1). Valid from
 * 1 to 32 -- the hash stream is a 32-bit xorshift, so 32 bits is the hard
 * ceiling. Lower SEEDBITS = cheaper to store per chunk but a smaller,
 * coarser search space (weaker best-case reduction); higher SEEDBITS costs
 * proportionally more search time (2x per extra bit). Left at 16 for now. */
#if SEEDBITS < 1 || SEEDBITS > 32
#error "SEEDBITS must be between 1 and 32"
#endif
#define SEEDSPACE ((u64)1 << SEEDBITS)

/* Number of times the wraparound delta is applied back-to-back after
 * hashing. 1 = the original single-layer behavior. Each extra layer costs
 * one more CHUNKSIZE-byte pass per seed tried (roughly +DELTALAYERS to the
 * per-chunk search cost). */
#define DELTALAYERS 32
#if DELTALAYERS < 1
#error "DELTALAYERS must be >= 1"
#endif

/* xorshift32 stream byte, state advanced per call */
static inline u8 xs32_next(u32 *s) {
    u32 x = *s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    *s = x;
    return (u8)x;
}

/* One wraparound delta pass: out[0] = in[0]-in[CHUNKSIZE-1] (wraps to the
 * last byte), out[i] = in[i]-in[i-1] for i>0. in and out may not alias. */
static void apply_delta_layer(const u8 *in, u8 *out) {
    u8 prev = in[CHUNKSIZE - 1];
    for (int i = 0; i < CHUNKSIZE; i++) {
        out[i] = (u8)(in[i] - prev);
        prev = in[i];
    }
}

/* Hash every byte of the chunk with the seed's xorshift32 stream, then walk
 * DELTALAYERS wraparound delta passes back-to-back, filling scores_out[k-1]
 * with the summed unsigned byte value after k layers (k=1..DELTALAYERS).
 * One hash pass covers every layer depth, so the search can pick whichever
 * layer a given seed does best at, not just the last one. Hot path used by
 * the seed search. */
static void chunk_score_layers(const u8 *chunk, u32 seed, u32 scores_out[DELTALAYERS]) {
    u32 s = seed;
    u8 cur[CHUNKSIZE];
    for (int i = 0; i < CHUNKSIZE; i++)
        cur[i] = (u8)(chunk[i] + xs32_next(&s));

    for (int layer = 0; layer < DELTALAYERS; layer++) {
        u8 next[CHUNKSIZE];
        apply_delta_layer(cur, next);
        for (int i = 0; i < CHUNKSIZE; i++) cur[i] = next[i];
        u32 score = 0;
        for (int i = 0; i < CHUNKSIZE; i++) score += cur[i];
        scores_out[layer] = score;
    }
}

/* Same computation as chunk_score, but keeps every intermediate layer for
 * inspection/printing. layers_out must have DELTALAYERS+1 rows of
 * CHUNKSIZE bytes: row 0 = hashed, row k (1<=k<=DELTALAYERS) = the k-th
 * delta layer. Not used in the search hot path -- only for detail dumps. */
static u32 chunk_transform_layers(const u8 *chunk, u32 seed, u8 layers_out[][CHUNKSIZE]) {
    u32 s = seed;
    for (int i = 0; i < CHUNKSIZE; i++)
        layers_out[0][i] = (u8)(chunk[i] + xs32_next(&s));

    for (int layer = 1; layer <= DELTALAYERS; layer++)
        apply_delta_layer(layers_out[layer - 1], layers_out[layer]);

    u32 score = 0;
    for (int i = 0; i < CHUNKSIZE; i++) score += layers_out[DELTALAYERS][i];
    return score;
}

/* Inverse of one apply_delta_layer call: given delta[] and the ORIGINAL
 * value that was at position 0 before that delta was taken (the "anchor",
 * i.e. in[0] from the forward pass -- NOT part of delta[] itself, since
 * delta[0]=in[0]-in[CHUNKSIZE-1] mixes both), reconstruct in[] exactly via
 * the recurrence in[i] = in[i-1] + delta[i]. delta and out may not alias. */
static void invert_delta_layer(const u8 *delta, u8 anchor, u8 *out) {
    out[0] = anchor;
    for (int i = 1; i < CHUNKSIZE; i++)
        out[i] = (u8)(out[i - 1] + delta[i]);
}

/* Full inverse of chunk_transform_layers run to depth `layer`: given the
 * delta bytes at that depth, the anchors used at each layer (anchors[k] =
 * layers_out[k][0] from the forward pass, i.e. the value fed INTO delta
 * layer k+1 -- for k=0..layer-1), and the seed, reconstruct the original
 * CHUNKSIZE raw bytes. Un-invertible without those `layer` anchor bytes:
 * the delta bytes alone are not enough (see comment on apply_delta_layer /
 * the DC-loss argument above main()). */
static void chunk_inverse(const u8 *final_layer, const u8 *anchors, int layer, u32 seed, u8 *raw_out) {
    u8 cur[CHUNKSIZE];
    for (int i = 0; i < CHUNKSIZE; i++) cur[i] = final_layer[i];

    for (int L = layer; L >= 1; L--) {
        u8 prev[CHUNKSIZE];
        invert_delta_layer(cur, anchors[L - 1], prev);
        for (int i = 0; i < CHUNKSIZE; i++) cur[i] = prev[i];
    }
    /* cur now holds the hashed values (layer 0); reverse the hash-add */
    u32 s = seed;
    for (int i = 0; i < CHUNKSIZE; i++)
        raw_out[i] = (u8)(cur[i] - xs32_next(&s));
}

static void print_bytes(const char *label, const u8 *b) {
    printf("%s", label);
    for (int i = 0; i < CHUNKSIZE; i++) printf(" %3u", b[i]);
    printf("\n");
}

int main(void) {
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
    u32 *zero_score = malloc(nchunks * sizeof(u32));  /* seed=0 score at the WINNING layer, same-layer comparison */

    u64 total_best = 0, total_zero = 0;
    u32 scores[DELTALAYERS];
    for (size_t c = 0; c < nchunks; c++) {
        const u8 *chunk = buf + c * CHUNKSIZE;
        u32 bestv = 0xFFFFFFFFu;
        u32 bests = 0;
        int bestlayer = 1;
        for (u64 seed = 0; seed < SEEDSPACE; seed++) {
            chunk_score_layers(chunk, (u32)seed, scores);
            for (int layer = 0; layer < DELTALAYERS; layer++) {
                if (scores[layer] < bestv) { bestv = scores[layer]; bests = (u32)seed; bestlayer = layer + 1; }
            }
        }
        best_seed[c]  = bests;
        best_score[c] = bestv;
        best_layer[c] = bestlayer;

        chunk_score_layers(chunk, 0, scores);
        zero_score[c] = scores[bestlayer - 1];

        total_best += bestv;
        total_zero += zero_score[c];
    }

    printf("\nseed=0 (no-op hash, same layer as winner) total delta-magnitude: %llu\n", (unsigned long long)total_zero);
    printf("best (seed,layer)                         total delta-magnitude: %llu\n", (unsigned long long)total_best);
    if (total_zero > 0)
        printf("reduction: %.2f%%\n", 100.0 * (double)(total_zero - total_best) / (double)total_zero);

    /* Round-trip verification: each layer of delta throws away that layer's
     * DC level (see apply_delta_layer/invert_delta_layer comments), so the
     * final delta bytes ALONE cannot reconstruct the original chunk -- one
     * anchor byte per delta layer used is also required. Reconstruct every
     * chunk from (final delta bytes + seed + those anchors) and confirm it
     * matches the original exactly, and report the anchor cost the score
     * above doesn't account for. */
    size_t roundtrip_ok = 0;
    u64 total_anchor_bytes = 0;
    for (size_t c = 0; c < nchunks; c++) {
        const u8 *chunk = buf + c * CHUNKSIZE;
        u8 layers[DELTALAYERS + 1][CHUNKSIZE];
        chunk_transform_layers(chunk, best_seed[c], layers);

        u8 anchors[DELTALAYERS];
        for (int k = 0; k < best_layer[c]; k++) anchors[k] = layers[k][0];

        u8 recon[CHUNKSIZE];
        chunk_inverse(layers[best_layer[c]], anchors, best_layer[c], best_seed[c], recon);

        if (memcmp(chunk, recon, CHUNKSIZE) == 0) roundtrip_ok++;
        total_anchor_bytes += (u64)best_layer[c];
    }
    printf("\nround-trip: %zu/%zu chunks reconstruct exactly from (final delta bytes + %d-bit seed + anchors)\n",
           roundtrip_ok, nchunks, SEEDBITS);
    printf("anchor overhead: %llu bytes total (avg %.2f bytes/chunk of %d) -- NOT counted in the score above\n",
           (unsigned long long)total_anchor_bytes,
           nchunks ? (double)total_anchor_bytes / (double)nchunks : 0.0, CHUNKSIZE);

    size_t show = nchunks < 20 ? nchunks : 20;
    printf("\nfirst %zu chunks:\n", show);
    for (size_t c = 0; c < show; c++)
        printf("  chunk %4zu: seed=%6u layer=%d/%d  score=%u (seed=0 L%d score=%u)\n",
               c, best_seed[c], best_layer[c], DELTALAYERS, best_score[c], best_layer[c], zero_score[c]);

    if (nchunks > 0) {
        const u8 *chunk0 = buf;
        u8 layers0[DELTALAYERS + 1][CHUNKSIZE];
        u8 layersB[DELTALAYERS + 1][CHUNKSIZE];
        chunk_transform_layers(chunk0, 0, layers0);
        chunk_transform_layers(chunk0, best_seed[0], layersB);

        u32 scores0[DELTALAYERS], scoresB[DELTALAYERS];
        chunk_score_layers(chunk0, 0, scores0);
        chunk_score_layers(chunk0, best_seed[0], scoresB);

        printf("\nchunk 0 detail (%d delta layer%s, best found at layer %d):\n",
               DELTALAYERS, DELTALAYERS == 1 ? "" : "s", best_layer[0]);
        print_bytes("  raw              :", chunk0);

        printf("  seed=0 (no-op) score per layer:");
        for (int layer = 1; layer <= DELTALAYERS; layer++) printf(" L%d=%u", layer, scores0[layer - 1]);
        printf("\n");
        print_bytes("    hashed         :", layers0[0]);
        for (int layer = 1; layer <= DELTALAYERS; layer++) {
            char label[32];
            snprintf(label, sizeof(label), "    delta L%-2d      :", layer);
            print_bytes(label, layers0[layer]);
        }

        printf("  seed=%u (best, layer %d, score=%u) score per layer:", best_seed[0], best_layer[0], best_score[0]);
        for (int layer = 1; layer <= DELTALAYERS; layer++) printf(" L%d=%u", layer, scoresB[layer - 1]);
        printf("\n");
        print_bytes("    hashed         :", layersB[0]);
        for (int layer = 1; layer <= DELTALAYERS; layer++) {
            char label[32];
            snprintf(label, sizeof(label), "    delta L%-2d      :", layer);
            print_bytes(label, layersB[layer]);
        }

        u8 anchorsB[DELTALAYERS];
        for (int k = 0; k < best_layer[0]; k++) anchorsB[k] = layersB[k][0];
        u8 recon0[CHUNKSIZE];
        chunk_inverse(layersB[best_layer[0]], anchorsB, best_layer[0], best_seed[0], recon0);

        printf("\n  round-trip using seed=%u, layer=%d, and %d anchor byte%s (%s):\n",
               best_seed[0], best_layer[0], best_layer[0], best_layer[0] == 1 ? "" : "s",
               memcmp(chunk0, recon0, CHUNKSIZE) == 0 ? "MATCH" : "MISMATCH");
        print_bytes("    reconstructed  :", recon0);
        print_bytes("    original       :", chunk0);
    }

    free(best_seed); free(best_score); free(best_layer); free(zero_score); free(buf);
    return 0;
}
