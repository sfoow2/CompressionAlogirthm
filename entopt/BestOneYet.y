//DO NOT USE THIS CODE IF YOU DO IT WILL BREAK YOUR COMPUTER DONT EVEN TRY LEARNING ABOUT IT IT WILL RESUALT IN AIS BREAKING IDK WHY JUST DONT TRY IT
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint8_t  u8;
typedef uint64_t u64;

// xorshift64: data generation only
static inline u64 xs64(u64 s) { s^=s<<13; s^=s>>7; s^=s<<17; return s; }

// xoshiro256 without output transform = 100% linear over GF(2), 256-bit state
// All ops are XOR, shift, or rotate — all linear over GF(2)
static void xs256_step(u64 s[4]) {
    u64 t  = s[1] << 17;
    s[2] ^= s[0];  s[3] ^= s[1];
    s[1] ^= s[2];  s[0] ^= s[3];
    s[2] ^= t;
    s[3]  = (s[3] << 45) | (s[3] >> 19);  // rotate = linear
}

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

// Build M[i][0..3]: bit-packed 256-bit row where bit j = 1 means
// "seed bit j affects bit 7 of s[0] at output position i".
// Done by running each of the 256 basis vectors through xs256 and recording.
static void buildM(u64 (*M)[4], int N) {
    memset(M, 0, N * 4 * sizeof(u64));
    for (int j = 0; j < 256; j++) {
        u64 s[4] = {0,0,0,0};
        s[j>>6] = 1ULL << (j&63);      // basis vector e_j
        for (int i = 0; i < N; i++) {
            xs256_step(s);
            if ((s[0] >> 7) & 1)        // bit 7 of s[0] lit up
                M[i][j>>6] |= 1ULL << (j&63);
        }
    }
}

// GF(2) Gaussian elimination on a 256x256 system (full RREF).
// A[0..255][0..3] = bit-packed rows, b[0..255] = RHS bits (0 or 1).
// Returns 1 with solution packed into x_out[4] if full rank, else 0.
static int gf2_solve256(u64 A[256][4], const u8 b[256], u64 x_out[4]) {
    // Augmented [A | b]: b stored as bit 0 of word [4]
    u64 aug[256][5];
    for (int i = 0; i < 256; i++) {
        aug[i][0]=A[i][0]; aug[i][1]=A[i][1];
        aug[i][2]=A[i][2]; aug[i][3]=A[i][3];
        aug[i][4]=b[i];    // 0 or 1
    }

    int pivot_col[256], pr = 0;

    for (int col = 0; col < 256 && pr < 256; col++) {
        // Find pivot row
        int found = -1;
        for (int r = pr; r < 256; r++)
            if ((aug[r][col>>6] >> (col&63)) & 1) { found = r; break; }
        if (found < 0) continue;

        // Swap rows pr <-> found
        u64 tmp[5];
        memcpy(tmp,        aug[pr],    5*sizeof(u64));
        memcpy(aug[pr],    aug[found], 5*sizeof(u64));
        memcpy(aug[found], tmp,        5*sizeof(u64));
        pivot_col[pr] = col;

        // Eliminate col from every other row (full RREF, not just lower)
        for (int r = 0; r < 256; r++) {
            if (r == pr) continue;
            if (!((aug[r][col>>6] >> (col&63)) & 1)) continue;
            aug[r][0]^=aug[pr][0]; aug[r][1]^=aug[pr][1];
            aug[r][2]^=aug[pr][2]; aug[r][3]^=aug[pr][3];
            aug[r][4]^=aug[pr][4];
        }
        pr++;
    }
    if (pr < 256) return 0;  // singular, no unique solution

    // Read solution: pivot row r tells us x[pivot_col[r]] = aug[r][4] & 1
    x_out[0]=x_out[1]=x_out[2]=x_out[3]=0;
    for (int r = 0; r < 256; r++)
        if (aug[r][4] & 1)
            x_out[pivot_col[r]>>6] |= 1ULL << (pivot_col[r]&63);
    return 1;
}

// Count how many of the N xs256 outputs (from seed) have bit 7 of s[0] == data_bits[i]
static int evaluate256(const u64 seed[4], const u8 *data_bits, int N) {
    u64 s[4] = {seed[0], seed[1], seed[2], seed[3]};
    int m = 0;
    for (int i = 0; i < N; i++) {
        xs256_step(s);
        m += (((s[0]>>7)&1) == data_bits[i]);
    }
    return m;
}

int main() {
    const int DATA_SEED = 42;
    const int SIZE      = 1048;
    // 256-bit seed guarantees 256 exact matches per solved system vs 32 for 32-bit,
    // so fewer candidates needed for a similar best-of distribution.
    const int N_CANDS   = 20000;

    // Generate data
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

    // Build GF(2) matrix: 1048 rows, each 256 bits wide (4 x u64)
    printf("Building GF(2) matrix (%d x 256 bits)...\n", SIZE);
    u64 (*M)[4] = malloc(SIZE * 4 * sizeof(u64));
    buildM(M, SIZE);

    printf("Solving %d random 256x256 GF(2) systems...\n", N_CANDS);
    printf("(~29%% will be full rank — same as 32-bit case)\n\n");

    u64 sampler     = 0xFEDCBA9876543210ULL;
    u64 best_seed[4] = {0,0,0,0};
    int best_match   = 0, n_solved = 0;

    // Allocate work buffers once outside the loop
    u8  *used  = calloc(SIZE, 1);
    int *idx   = malloc(256 * sizeof(int));
    u64 (*A_sub)[4] = malloc(256 * 4 * sizeof(u64));
    u8  *b_sub = malloc(256);

    for (int c = 0; c < N_CANDS; c++) {
        // Pick 256 distinct random positions from [0, SIZE)
        int n = 0;
        while (n < 256) {
            sampler = xs64(sampler);
            int pos = (int)((sampler >> 33) % (u64)SIZE);
            if (!used[pos]) { used[pos] = 1; idx[n++] = pos; }
        }

        // Build 256x256 submatrix and reset used[] in one pass
        for (int i = 0; i < 256; i++) {
            A_sub[i][0]=M[idx[i]][0]; A_sub[i][1]=M[idx[i]][1];
            A_sub[i][2]=M[idx[i]][2]; A_sub[i][3]=M[idx[i]][3];
            b_sub[i] = data_bits[idx[i]];
            used[idx[i]] = 0;
        }

        u64 seed_cand[4];
        if (!gf2_solve256(A_sub, b_sub, seed_cand)) continue;
        n_solved++;

        int m = evaluate256(seed_cand, data_bits, SIZE);
        if (m > best_match) {
            best_match = m;
            memcpy(best_seed, seed_cand, 4*sizeof(u64));
        }
    }

    printf("Systems solved: %d / %d (~%.0f%%)\n",
           n_solved, N_CANDS, 100.0f * n_solved / N_CANDS);
    printf("Best seed: %016llX %016llX\n",
           (unsigned long long)best_seed[0], (unsigned long long)best_seed[1]);
    printf("           %016llX %016llX\n",
           (unsigned long long)best_seed[2], (unsigned long long)best_seed[3]);
    printf("Bit matches: %d / %d (%.2f%%)\n\n",
           best_match, SIZE, 100.0f * best_match / SIZE);

    // Apply correction: where xs256 output has bit 7 of s[0] = 1, add 128
    u64 state[4]; memcpy(state, best_seed, sizeof(state));
    u8 *corrected = malloc(SIZE);
    memcpy(corrected, data, SIZE);
    for (int i = 0; i < SIZE; i++) {
        xs256_step(state);
        if ((state[0] >> 7) & 1) {
            int v = corrected[i] + 128;
            corrected[i] = (v > 255) ? 255 : (u8)v;
        }
    }

    int small_after = 0;
    for (int i = 0; i < SIZE; i++) if (corrected[i] < 128) small_after++;

    printf("Small bytes before: %d / %d (%.1f%%)\n", data_ones,   SIZE, 100.0f*data_ones/SIZE);
    printf("Small bytes after:  %d / %d (%.1f%%)\n", small_after, SIZE, 100.0f*small_after/SIZE);
    printf("Byte entropy before: %.6f / 8.000000 bits\n", base_ent);
    printf("Byte entropy after:  %.6f / 8.000000 bits\n", byteEntropy(corrected, SIZE));

    printf("\nSeed overhead: 32 bytes (256-bit seed)\n");
    float before_bytes = (base_ent            * SIZE) / 8.0f;
    float after_bytes  = (byteEntropy(corrected, SIZE) * SIZE) / 8.0f;
    printf("Ideal bytes before: %.2f\n", before_bytes);
    printf("Ideal bytes after:  %.2f\n", after_bytes);
    printf("Net savings:        %.2f bytes\n", before_bytes - after_bytes - 32.0f);

    free(M); free(data); free(data_bits); free(corrected);
    free(used); free(idx); free(A_sub); free(b_sub);
    return 0;
}
