#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>
#include <windows.h>
#include <bcrypt.h>

typedef uint8_t  u8;
typedef uint64_t u64;

/* ---- disparity-controlled PRNG ------------------------------------ */
/* thresh=0 → pure LCG (no balance constraint)                        */
/* thresh>0 → force a corrective bit when imbalance exceeds threshold */

typedef struct { u64 state; int disp; } DispGen;

static inline int disp_next_t(DispGen *g, int thresh) {
    g->state = g->state * 6364136223846793005ULL + 1442695040888963407ULL;
    int b = (int)((g->state >> 33) & 1);
    if (thresh > 0) {
        if (g->disp >=  thresh) b = 0;
        if (g->disp <= -thresh) b = 1;
    }
    g->disp += b ? 1 : -1;
    return b;
}

/* thresholds to try during the search phase — 0 = pure LCG          */
/* the winner gets hardcoded; nothing extra is stored in the output   */
static const int THRESHOLDS[] = { 0, 1, 2, 3 };
#define N_THRESH (int)(sizeof(THRESHOLDS)/sizeof(THRESHOLDS[0]))
/* ------------------------------------------------------------------ */

static float lowerBitsEntropy(const u8 *d, int n) {
    int freq[128] = {0};
    for (int i = 0; i < n; i++) freq[d[i] & 0x7F]++;
    double e = 0.0;
    for (int i = 0; i < 128; i++) {
        if (!freq[i]) continue;
        double p = (double)freq[i] / n;
        e -= p * log2(p);
    }
    return (float)e;
}

#define MAX_SB     512
#define MAX_PASSES  32

typedef struct { int sb; int thresh; u8 seeds[MAX_SB]; } PassParams;

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

int main(int argc, char **argv) {
    const char *OUT_PATH = "C:/Users/lukac/Documents/compressor/transformed.bin";

    int size = argc > 1 ? atoi(argv[1]) : 65536;
    u8 *data = malloc(size);
    BCryptGenRandom(NULL, data, size, BCRYPT_USE_SYSTEM_PREFERRED_RNG);

    float orig_entropy = byteEntropy(data, size);
    printf("Original entropy = %.6f\n", orig_entropy);

    int bits_len = (size + 7) / 8;
    u8 *bits = malloc(bits_len);
    u8 *work = malloc(size);
    u8 *temp = malloc(size);
    memcpy(work, data, size);

    u8 cand_seeds[4][MAX_SB];
    u8 best_seeds[MAX_SB];

    PassParams passes[MAX_PASSES];
    u8        *pass_bits[MAX_PASSES];
    int        n_passes = 0;

    int   total_header = 0;
    float prev_entropy = orig_entropy;

    for (int pass = 0; ; pass++) {
        int ones = 0;
        memset(bits, 0, bits_len);
        for (int i = 0; i < size; i++) {
            if (work[i] < 128) {
                bits[i / 8] |= (1 << (7 - (i % 8)));
                ones++;
            }
        }
        if (ones == 0) break;

        int max_sb = ones < MAX_SB ? ones : MAX_SB;

        int   best_sb         = 1;
        int   best_net        = -(size + 1);
        int   best_gain       = 0;
        float best_ent_after  = prev_entropy;
        int   best_thresh_val = 0;
        int   stagnant        = 0;

        for (int sb = 1; sb <= max_sb; sb++) {
            int64_t totals[4] = {0, 0, 0, 0};

            #pragma omp parallel for schedule(static)
            for (int seg = 0; seg < sb; seg++) {
                int seg_start = (int)((int64_t)seg       * size / sb);
                int seg_end   = (int)((int64_t)(seg + 1) * size / sb);

                for (int ti = 0; ti < N_THRESH; ti++) {
                    int thresh = THRESHOLDS[ti];
                    int best_s = 0, best_m = -1;
                    for (int s = 0; s < 256; s++) {
                        DispGen g = { (u64)s * 2654435761ULL + 1442695040888963407ULL, 0 };
                        int m = 0;
                        for (int i = seg_start; i < seg_end; i++) {
                            int bit = (bits[i / 8] >> (7 - (i % 8))) & 1;
                            if (disp_next_t(&g, thresh) == bit) m++;
                        }
                        if (m > best_m) { best_m = m; best_s = s; }
                    }
                    cand_seeds[ti][seg] = (u8)best_s;
                    #pragma omp atomic
                    totals[ti] += (int64_t)best_m;
                }
            }

            int best_ti = 0;
            for (int ti = 1; ti < N_THRESH; ti++)
                if (totals[ti] > totals[best_ti]) best_ti = ti;

            memcpy(temp, work, size);
            int t = THRESHOLDS[best_ti];
            for (int seg = 0; seg < sb; seg++) {
                int seg_start = (int)((int64_t)seg       * size / sb);
                int seg_end   = (int)((int64_t)(seg + 1) * size / sb);
                DispGen g = { (u64)cand_seeds[best_ti][seg] * 2654435761ULL
                              + 1442695040888963407ULL, 0 };
                for (int i = seg_start; i < seg_end; i++) {
                    if (disp_next_t(&g, t) == 1)
                        temp[i] |= 0x80;
                }
            }

            float ent = byteEntropy(temp, size);
            int gain  = (int)((prev_entropy - ent) * size / 8.0f + 0.5f);
            int pcost = 1 + sb;
            int net   = gain - pcost;

            if (net > best_net) {
                best_net        = net;
                best_gain       = gain;
                best_sb         = sb;
                best_ent_after  = ent;
                best_thresh_val = t;
                memcpy(best_seeds, cand_seeds[best_ti], sb);
                stagnant = 0;
            } else if (++stagnant >= 10) {
                break;
            }
        }

        int pcost = 1 + best_sb;
        total_header += pcost;

        if (best_net <= 0) {
            printf("Pass %-2d  gain: %4dB  header: %3d  net: %+5d  [stopped]\n",
                   pass + 1, best_gain, pcost, best_net);
            total_header -= pcost;
            break;
        }

        for (int seg = 0; seg < best_sb; seg++) {
            int seg_start = (int)((int64_t)seg       * size / best_sb);
            int seg_end   = (int)((int64_t)(seg + 1) * size / best_sb);
            DispGen g = { (u64)best_seeds[seg] * 2654435761ULL
                          + 1442695040888963407ULL, 0 };
            for (int i = seg_start; i < seg_end; i++) {
                if (disp_next_t(&g, best_thresh_val) == 1)
                    work[i] |= 0x80;
            }
        }

        passes[n_passes].sb     = best_sb;
        passes[n_passes].thresh = best_thresh_val;
        memcpy(passes[n_passes].seeds, best_seeds, best_sb);
        pass_bits[n_passes] = malloc(bits_len);
        memcpy(pass_bits[n_passes], bits, bits_len);
        n_passes++;

        printf("Pass %-2d  gain: %4dB  header: %3d  net: %+5d  (sb=%d)\n",
               pass + 1, best_gain, pcost, best_net, best_sb);
        prev_entropy = best_ent_after;
    }

    /* decode: replay passes in reverse; undo |= 0x80 only where the bitstream
       said byte was originally < 128 (PRNG=1 on >=128 bytes was a no-op) */
    u8 *decoded = malloc(size);
    memcpy(decoded, work, size);
    for (int p = n_passes - 1; p >= 0; p--) {
        int sb     = passes[p].sb;
        int thresh = passes[p].thresh;
        for (int seg = 0; seg < sb; seg++) {
            int seg_start = (int)((int64_t)seg       * size / sb);
            int seg_end   = (int)((int64_t)(seg + 1) * size / sb);
            DispGen g = { (u64)passes[p].seeds[seg] * 2654435761ULL
                          + 1442695040888963407ULL, 0 };
            for (int i = seg_start; i < seg_end; i++) {
                int was_below = (pass_bits[p][i / 8] >> (7 - (i % 8))) & 1;
                if (disp_next_t(&g, thresh) == 1 && was_below)
                    decoded[i] &= ~0x80;
            }
        }
        free(pass_bits[p]);
    }
    int mismatches = 0;
    for (int i = 0; i < size; i++)
        if (decoded[i] != data[i]) mismatches++;
    printf("\nDecode: %s", mismatches == 0 ? "OK — round-trip verified\n" : "FAIL\n");
    if (mismatches) printf("  %d mismatches\n", mismatches);
    free(decoded);

    int total_gain = (int)((orig_entropy - prev_entropy) * size / 8.0f + 0.5f);
    printf("\nTotal profit: %+d bytes  (header: %d)\n",
           total_gain - total_header, total_header);
    printf("Final entropy: %.6f  (was %.6f)\n", prev_entropy, orig_entropy);

    FILE *fout = fopen(OUT_PATH, "wb");
    if (fout) { fwrite(work, 1, size, fout); fclose(fout); }
    else       { fprintf(stderr, "Cannot write %s\n", OUT_PATH); }

    const char *OH_PATH = "C:/Users/lukac/Documents/compressor/overhead.bin";
    FILE *foh = fopen(OH_PATH, "wb");
    if (foh) {
        uint8_t  np  = (uint8_t)n_passes;
        fwrite(&np, 1, 1, foh);
        for (int p = 0; p < n_passes; p++) {
            uint16_t sb16  = (uint16_t)passes[p].sb;
            uint8_t  th8   = (uint8_t) passes[p].thresh;
            fwrite(&sb16, 2, 1,             foh);
            fwrite(&th8,  1, 1,             foh);
            fwrite(passes[p].seeds, 1, passes[p].sb, foh);
        }
        long oh_size = ftell(foh);
        fclose(foh);
        printf("overhead.bin: %ld bytes\n", oh_size);
    } else {
        fprintf(stderr, "Cannot write %s\n", OH_PATH);
    }

    free(bits);
    free(work);
    free(temp);
    free(data);
    return 0;
}
