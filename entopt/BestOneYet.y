//DO NOT USE THIS CODE IF YOU DO IT WILL BREAK YOUR COMPUTER DONT EVEN TRY LEARNING ABOUT IT IT WILL RESUALT IN AIS BREAKING IDK WHY JUST DONT TRY IT
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint8_t  u8;
typedef uint32_t u32;
typedef uint64_t u64;

// xorshift64 — data generation only, kept completely separate from search PRNG
static inline u64 xs64(u64 s) { s ^= s<<13; s ^= s>>7; s ^= s<<17; return s; }

// xorshift32 — the search PRNG, linear over GF(2) so we can do matrix math on it
static inline u32 xs32(u32 s) { s ^= s<<13; s ^= s>>17; s ^= s<<5; return s; }

static float byteEntropy(const u8 *d, int n) {
    int freq[256] = {0};
    for (int i = 0; i < n; i++) freq[d[i]]++;
    double e = 0.0;
    for (int i = 0; i < 256; i++) {
        if (!freq[i]) continue;
        double p = (double)freq[i] / n;
        e -= p * log2(p);
    }
    return (float)e;
}

// Build M[i] — a 32-bit mask where bit j = 1 means "seed bit j affects bit 7
// of the i-th xs32 output". Computed by running each of the 32 basis vectors
// through the PRNG and recording which bit-7s light up.
static void buildM(u32 *M, int N) {
    memset(M, 0, N * sizeof(u32));
    for (int j = 0; j < 32; j++) {
        u32 s = (1u << j);
        for (int i = 0; i < N; i++) {
            s = xs32(s);
            if ((s >> 7) & 1) M[i] |= (1u << j);
        }
    }
}

// GF(2) Gaussian elimination: solve A * x = b (mod 2).
// A[0..31] are the 32 rows; b[0..31] are the RHS bits (0 or 1).
// Returns 1 with solution in *x_out if A has full rank, else 0.
static int gf2_solve(u32 A[32], const u8 b[32], u32 *x_out) {
    // Augmented matrix: LHS in bits 0-31, RHS in bit 32
    u64 aug[32];
    for (int i = 0; i < 32; i++)
        aug[i] = (u64)A[i] | ((u64)b[i] << 32);

    int pivot_col[32];
    int pr = 0;  // next pivot row

    for (int col = 0; col < 32 && pr < 32; col++) {
        // Find a row at or below pr that has a 1 in this column
        int found = -1;
        for (int r = pr; r < 32; r++)
            if ((aug[r] >> col) & 1) { found = r; break; }
        if (found < 0) continue;  // no pivot here, column is all zeros

        // Swap found row into pivot position
        u64 tmp = aug[pr]; aug[pr] = aug[found]; aug[found] = tmp;
        pivot_col[pr] = col;

        // Eliminate this column from every other row (full RREF)
        for (int r = 0; r < 32; r++)
            if (r != pr && ((aug[r] >> col) & 1))
                aug[r] ^= aug[pr];
        pr++;
    }

    if (pr < 32) return 0;  // rank < 32, no unique solution

    // Read off the solution: each pivot row r tells us x[pivot_col[r]]
    u32 x = 0;
    for (int r = 0; r < 32; r++)
        if ((aug[r] >> 32) & 1)
            x |= (1u << pivot_col[r]);
    *x_out = x;
    return 1;
}

// Count how many of the N xs32 outputs (starting from seed) have bit 7 == data_bits[i]
static int evaluate(u32 seed, const u8 *data_bits, int N) {
    u32 s = seed;
    int m = 0;
    for (int i = 0; i < N; i++) {
        s = xs32(s);
        m += (((s >> 7) & 1) == data_bits[i]);
    }
    return m;
}

int main() {
    const int DATA_SEED = 42;
    const int SIZE      = 1048;
    // Each candidate costs O(32^3) to solve + O(SIZE) to evaluate.
    // Much cheaper per good candidate than random sampling.
    const int N_CANDS   = 200000;

    // --- Generate data with xorshift64 ---
    u64 ds = (u64)DATA_SEED;
    u8 *data = malloc(SIZE);
    for (int i = 0; i < SIZE; i++) { ds = xs64(ds); data[i] = (u8)(ds & 0xFF); }

    u8 *data_bits = malloc(SIZE);
    int data_ones = 0;
    for (int i = 0; i < SIZE; i++) {
        data_bits[i] = (data[i] < 128) ? 1 : 0;
        data_ones += data_bits[i];
    }

    float base_ent = byteEntropy(data, SIZE);
    printf("Data seed: %d (xorshift64) | Size: %d bytes\n", DATA_SEED, SIZE);
    printf("Bytes < 128:  %d / %d (%.1f%%)\n", data_ones, SIZE, 100.0f * data_ones / SIZE);
    printf("Byte entropy (before): %.6f / 8.000000 bits\n\n", base_ent);

    // --- Build the GF(2) matrix M (SIZE x 32) for xs32 ---
    printf("Building GF(2) matrix (%d x 32)...\n", SIZE);
    u32 *M = malloc(SIZE * sizeof(u32));
    buildM(M, SIZE);

    // --- Information Set Decoding ---
    // Pick N_CANDS random subsets of 32 positions.
    // Each subset gives a 32x32 GF(2) system whose exact solution is the unique
    // seed that perfectly matches those 32 positions. Then we check the full SIZE
    // positions and keep the best overall seed.
    // About 29% of random 32x32 GF(2) matrices are full rank, so ~29% solve.
    printf("Solving %d random GF(2) systems (32x32)...\n\n", N_CANDS);

    u64 sampler  = 0xFEDCBA9876543210ULL;
    u32 best_seed  = 0;
    int best_match = 0;
    int n_solved   = 0;

    for (int c = 0; c < N_CANDS; c++) {
        // Pick 32 distinct random positions from [0, SIZE)
        int idx[32], n = 0;
        while (n < 32) {
            sampler = xs64(sampler);
            int pos = (int)((sampler >> 33) % (u64)SIZE);
            int dup = 0;
            for (int j = 0; j < n; j++) if (idx[j] == pos) { dup = 1; break; }
            if (!dup) idx[n++] = pos;
        }

        // Extract the 32x32 submatrix and corresponding data bits
        u32 A[32]; u8 b[32];
        for (int i = 0; i < 32; i++) { A[i] = M[idx[i]]; b[i] = data_bits[idx[i]]; }

        u32 seed_cand;
        if (!gf2_solve(A, b, &seed_cand)) continue;
        n_solved++;

        int m = evaluate(seed_cand, data_bits, SIZE);
        if (m > best_match) { best_match = m; best_seed = seed_cand; }
    }

    printf("Systems solved (full rank): %d / %d  (~%.0f%% as expected for random GF2)\n",
           n_solved, N_CANDS, 100.0f * n_solved / N_CANDS);
    printf("Best seed:   0x%08X\n", best_seed);
    printf("Bit matches: %d / %d (%.2f%%)\n\n", best_match, SIZE, 100.0f * best_match / SIZE);

    // --- Apply correction: where xs32 output has bit 7 = 1, add 128 to data byte ---
    u32 state = best_seed;
    u8 *corrected = malloc(SIZE);
    memcpy(corrected, data, SIZE);
    for (int i = 0; i < SIZE; i++) {
        state = xs32(state);
        if ((state >> 7) & 1) {
            int v = corrected[i] + 128;
            corrected[i] = (v > 255) ? 255 : (u8)v;
        }
    }

    int small_after = 0;
    for (int i = 0; i < SIZE; i++) if (corrected[i] < 128) small_after++;

    printf("Small bytes before: %d / %d (%.1f%%)\n", data_ones,   SIZE, 100.0f * data_ones   / SIZE);
    printf("Small bytes after:  %d / %d (%.1f%%)\n", small_after, SIZE, 100.0f * small_after  / SIZE);
    printf("Byte entropy before: %.6f / 8.000000 bits\n", base_ent);
    printf("Byte entropy after:  %.6f / 8.000000 bits\n", byteEntropy(corrected, SIZE));

    free(M); free(data); free(data_bits); free(corrected);
    return 0;
}
