#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <omp.h>

typedef uint8_t u8;

#define ScrambeSizeBits     4
#define ScanSizeBits        4
#define AmplifierSizeBits   7
#define MAX_SCRAMBLE_PASSES 8
#define BLOCK_SIZE          (64 * 1024)
#define BWT_OVERHEAD_BITS   32

int ScrambleSize     = (1 << ScrambeSizeBits);
int ScanSize         = (1 << ScanSizeBits);
int AmpliferScan     = (1 << AmplifierSizeBits);
int ScanSizeBitsG    = ScanSizeBits;    // runtime-tunable
int ScrambeSizeBitsG = ScrambeSizeBits; // runtime-tunable

int FindNextBetter(u8 *Data, int size, u8 *amp, u8 *op);

// GF(256) tables — primitive polynomial 0x1B (x^8+x^4+x^3+x+1, AES field).
// gf_exp[i] = generator^i; gf_log[x] = discrete log; gf_inv[x] = 1/x.
// Initialized once by init_gf256() before any compression.
static u8 gf_exp[512]; // doubled to avoid mod-255 in multiply
static u8 gf_log[256];
static u8 gf_inv[256];

static void init_gf256(void) {
    u8 x = 1;
    for (int i = 0; i < 255; i++) {
        gf_exp[i] = x;
        gf_log[x] = (u8)i;
        u8 xhi = (x & 0x80) ? ((x << 1) ^ 0x1B) : (x << 1); // x * 2
        x = xhi ^ x; // x * 3 (generator)
    }
    for (int i = 255; i < 512; i++) gf_exp[i] = gf_exp[i - 255];
    gf_log[0] = 0; // undefined; never used in mul (zero stays zero)
    gf_inv[0] = 0;
    gf_inv[1] = 1;
    for (int a = 2; a < 256; a++) gf_inv[a] = gf_exp[255 - gf_log[a]];
}

static inline u8 gf_mul(u8 a, u8 b) {
    if (!a || !b) return 0;
    return gf_exp[(int)gf_log[a] + gf_log[b]];
}

static const char *op_names[] = {"ADD", "XOR", "ROL", "MUL", "ADDHI", "ADDLO", "SWXOR", "GFMUL"};

typedef struct { u8 seed, scan, amp, op; } ScrambleStep;

// ── Core helpers ──────────────────────────────────────────────────────────────

static double byteEntropy(const u8 *d, int n) {
    int freq[256] = {0};
    for (int i = 0; i < n; i++) freq[d[i]]++;
    double e = 0.0;
    for (int i = 0; i < 256; i++) {
        if (!freq[i]) continue;
        double p = (double)freq[i] / n;
        e -= p * log2(p);
    }
    return e;
}

// Prints histogram shape: zero buckets, peak height, and top-5 most frequent values.
// Reveals how "collisions" concentrate the distribution.
static void printHistStats(const u8 *d, int n, const char *label) {
    int freq[256] = {0};
    for (int i = 0; i < n; i++) freq[d[i]]++;

    int zeros = 0, maxFreq = 0;
    for (int i = 0; i < 256; i++) {
        if (!freq[i]) zeros++;
        if (freq[i] > maxFreq) maxFreq = freq[i];
    }

    int used[256] = {0};
    printf("  hist[%s] zeros=%d  peak=%d(%.2fx mean)  top5:", label, zeros, maxFreq,
           maxFreq * 256.0 / n);
    for (int k = 0; k < 5; k++) {
        int best = 0, bestF = -1;
        for (int i = 0; i < 256; i++)
            if (!used[i] && freq[i] > bestF) { bestF = freq[i]; best = i; }
        printf(" %02X:%d", best, bestF);
        used[best] = 1;
    }
    printf("\n");
}

static u8 mul_inverse_byte(u8 a) {
    for (int b = 1; b <= 255; b += 2)
        if ((u8)(a * b) == 1) return (u8)b;
    return 1;
}

static void apply_op_inplace(u8 *Data, int size, int stride, u8 amp, u8 op) {
    for (int p = 0; p < size; p += stride) {
        switch (op) {
            case 0: Data[p] += amp; break;
            case 1: Data[p] ^= amp; break;
            case 2: Data[p] = (u8)((Data[p] << amp) | (Data[p] >> (8 - amp))); break;
            case 3: Data[p] = (u8)(Data[p] * amp); break;
            case 4: Data[p] = (u8)((((Data[p] >> 4) + amp) & 0xF) << 4 | (Data[p] & 0x0F)); break;
            case 5: Data[p] = (u8)((Data[p] & 0xF0) | ((Data[p] + amp) & 0x0F)); break;
            case 6: { u8 s = (Data[p] << 4) | (Data[p] >> 4); Data[p] = s ^ amp; } break;
            case 7: Data[p] = gf_mul(Data[p], amp); break;
        }
    }
}

static void undo_op_inplace(u8 *Data, int size, int stride, u8 amp, u8 op) {
    u8 inv = (op == 3) ? mul_inverse_byte(amp) : (op == 7) ? gf_inv[amp] : 0;
    for (int p = 0; p < size; p += stride) {
        switch (op) {
            case 0: Data[p] -= amp; break;
            case 1: Data[p] ^= amp; break;
            case 2: Data[p] = (u8)((Data[p] >> amp) | (Data[p] << (8 - amp))); break;
            case 3: Data[p] = (u8)(Data[p] * inv); break;
            case 4: Data[p] = (u8)((((Data[p] >> 4) - amp) & 0xF) << 4 | (Data[p] & 0x0F)); break;
            case 5: Data[p] = (u8)((Data[p] & 0xF0) | ((Data[p] - amp) & 0x0F)); break;
            case 6: { u8 t = Data[p] ^ amp; Data[p] = (t << 4) | (t >> 4); } break;
            case 7: Data[p] = gf_mul(Data[p], inv); break;
        }
    }
}

static int op_overhead(u8 op) {
    // Opcode is now 3 bits (8 ops).
    // ADD/XOR: 8-bit amp; ROL: 3-bit; MUL: 7-bit (odd 3-255);
    // ADDHI/ADDLO: 4-bit amp (1-15); SWXOR: 8-bit (0-255); GFMUL: 8-bit (2-255).
    static const int amp_bits[] = {8, 8, 3, AmplifierSizeBits, 4, 4, 8, 8};
    return ScanSizeBitsG + 3 + amp_bits[op];
}

// ── Scramble ──────────────────────────────────────────────────────────────────

void scramble_bytes(u8 *data, int size, u8 seed) {
    if (!data || size <= 1) return;
    srand(seed);
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        u8 tmp = data[i]; data[i] = data[j]; data[j] = tmp;
    }
}

void unscramble_bytes(u8 *data, int size, u8 seed) {
    if (!data || size <= 1) return;
    int *swaps = malloc(size * sizeof(int));
    if (!swaps) return;
    srand(seed);
    for (int i = size - 1; i > 0; i--) swaps[i] = rand() % (i + 1);
    for (int i = 1; i < size; i++) {
        u8 tmp = data[i]; data[i] = data[swaps[i]]; data[swaps[i]] = tmp;
    }
    free(swaps);
}

// ── Transform search ──────────────────────────────────────────────────────────

void FindNextBestScramble(u8 *Data, int size, u8 *seed, u8 *Scan, u8 *amp, u8 *op) {
    u8 *Data2 = malloc(size);
    if (!Data2) { *seed = *Scan = *amp = *op = 0; return; }

    double Topempt = byteEntropy(Data, size);
    u8 BestSeed = 0, BestScan = 0, BestAmp = 0, BestOp = 0;

    for (int si = 0; si < ScrambleSize; si++) {
        for (int x = 0; x < size; x++) Data2[x] = Data[x];
        scramble_bytes(Data2, size, (u8)si);

        u8 Lastamp = 0, Lastop = 0;
        u8 LastScan = FindNextBetter(Data2, size, &Lastamp, &Lastop);

        if (LastScan > 0) apply_op_inplace(Data2, size, LastScan, Lastamp, Lastop);

        double empt = byteEntropy(Data2, size);
        if (empt < Topempt) {
            Topempt  = empt;
            BestSeed = (u8)si;
            BestScan = LastScan;
            BestAmp  = Lastamp;
            BestOp   = Lastop;
        }
    }

    *seed = BestSeed; *Scan = BestScan; *amp = BestAmp; *op = BestOp;
    free(Data2);
}

int FindNextBetter(u8 *Data, int size, u8 *amp, u8 *op) {
    double BestEmpt = byteEntropy(Data, size);
    int BestScan = 0, BestAmp = 0, BestOp = 0;

    #pragma omp parallel
    {
        u8 *Data2 = malloc(size);
        double localBestEmpt = BestEmpt;
        int localScan = 0, localAmp = 0, localOp = 0;

        if (Data2) {
            #pragma omp for schedule(dynamic, 1)
            for (int x = 1; x <= ScanSize; x++) {
                for (int s = 0; s < size; s++) Data2[s] = Data[s];

                for (int a = 1; a <= 255; a++) {
                    for (int p = 0; p < size; p += x) Data2[p] = (u8)(Data[p] + a);
                    double empt = byteEntropy(Data2, size);
                    if (empt < localBestEmpt) { localBestEmpt = empt; localScan = x; localAmp = a; localOp = 0; }

                    for (int p = 0; p < size; p += x) Data2[p] = Data[p] ^ (u8)a;
                    empt = byteEntropy(Data2, size);
                    if (empt < localBestEmpt) { localBestEmpt = empt; localScan = x; localAmp = a; localOp = 1; }
                }

                for (int r = 1; r <= 7; r++) {
                    for (int p = 0; p < size; p += x)
                        Data2[p] = (u8)((Data[p] << r) | (Data[p] >> (8 - r)));
                    double empt = byteEntropy(Data2, size);
                    if (empt < localBestEmpt) { localBestEmpt = empt; localScan = x; localAmp = r; localOp = 2; }
                }

                for (int a = 3; a <= 255; a += 2) {
                    for (int p = 0; p < size; p += x)
                        Data2[p] = (u8)(Data[p] * a);
                    double empt = byteEntropy(Data2, size);
                    if (empt < localBestEmpt) { localBestEmpt = empt; localScan = x; localAmp = a; localOp = 3; }
                }

                // Op 4: ADDHI — add amp (1-15) to high nibble only
                for (int a = 1; a <= 15; a++) {
                    for (int p = 0; p < size; p += x)
                        Data2[p] = (u8)((((Data[p] >> 4) + a) & 0xF) << 4 | (Data[p] & 0x0F));
                    double empt = byteEntropy(Data2, size);
                    if (empt < localBestEmpt) { localBestEmpt = empt; localScan = x; localAmp = a; localOp = 4; }
                }

                // Op 5: ADDLO — add amp (1-15) to low nibble only
                for (int a = 1; a <= 15; a++) {
                    for (int p = 0; p < size; p += x)
                        Data2[p] = (u8)((Data[p] & 0xF0) | ((Data[p] + a) & 0x0F));
                    double empt = byteEntropy(Data2, size);
                    if (empt < localBestEmpt) { localBestEmpt = empt; localScan = x; localAmp = a; localOp = 5; }
                }

                // Op 6: SWXOR — swap nibbles then XOR amp (0-255); amp=0 is pure nibble swap
                for (int a = 0; a <= 255; a++) {
                    for (int p = 0; p < size; p += x)
                        Data2[p] = (u8)(((Data[p] << 4) | (Data[p] >> 4)) ^ a);
                    double empt = byteEntropy(Data2, size);
                    if (empt < localBestEmpt) { localBestEmpt = empt; localScan = x; localAmp = a; localOp = 6; }
                }

                // Op 7: GFMUL — GF(256) multiply by amp (2-255, skip 1=identity)
                {
                    u8 lut[256];
                    for (int a = 2; a <= 255; a++) {
                        lut[0] = 0;
                        for (int v = 1; v < 256; v++)
                            lut[v] = gf_exp[(int)gf_log[v] + gf_log[a]];
                        for (int p = 0; p < size; p += x) Data2[p] = lut[Data[p]];
                        double empt = byteEntropy(Data2, size);
                        if (empt < localBestEmpt) { localBestEmpt = empt; localScan = x; localAmp = a; localOp = 7; }
                    }
                }
            }
            free(Data2);
        }

        #pragma omp critical
        {
            if (localBestEmpt < BestEmpt) {
                BestEmpt = localBestEmpt;
                BestScan = localScan;
                BestAmp  = localAmp;
                BestOp   = localOp;
            }
        }
    }

    *amp = (u8)BestAmp; *op = (u8)BestOp;
    return BestScan;
}

// ── BWT ──────────────────────────────────────────────────────────────────────
// Cyclic suffix array via prefix doubling (O(n log^2 n)).
// Verified correct on "abcd" and "banana".

typedef struct { int orig, r0, r1; } SuffixRank;

static int cmp_sr(const void *a, const void *b) {
    const SuffixRank *x = a, *y = b;
    if (x->r0 != y->r0) return (x->r0 > y->r0) - (x->r0 < y->r0);
    return (x->r1 > y->r1) - (x->r1 < y->r1);
}

static int *build_cyclic_suffix_array(const u8 *s, int n) {
    if (n <= 0) return NULL;
    if (n == 1) {
        int *sa = malloc(sizeof(int));
        if (sa) sa[0] = 0;
        return sa;
    }

    int *rank = malloc(n * sizeof(int));
    int *tmp  = malloc(n * sizeof(int));
    SuffixRank *sr = malloc(n * sizeof(SuffixRank));
    if (!rank || !tmp || !sr) { free(rank); free(tmp); free(sr); return NULL; }

    for (int i = 0; i < n; i++) rank[i] = s[i];

    for (int k = 1; k < n; k <<= 1) {
        for (int i = 0; i < n; i++) {
            sr[i].orig = i;
            sr[i].r0   = rank[i];
            sr[i].r1   = rank[(i + k) % n];
        }
        qsort(sr, n, sizeof(SuffixRank), cmp_sr);

        tmp[sr[0].orig] = 0;
        for (int i = 1; i < n; i++)
            tmp[sr[i].orig] = tmp[sr[i-1].orig] +
                ((sr[i].r0 != sr[i-1].r0 || sr[i].r1 != sr[i-1].r1) ? 1 : 0);
        for (int i = 0; i < n; i++) rank[i] = tmp[i];
        if (rank[sr[n-1].orig] == n - 1) break;
    }

    int *sa = malloc(n * sizeof(int));
    if (sa) for (int i = 0; i < n; i++) sa[i] = sr[i].orig;

    free(rank); free(tmp); free(sr);
    return sa;
}

// Replaces data[] with BWT last column; sets *primary_index to the row where
// the original string starts (needed for inverse).
static void bwt_forward(u8 *data, int n, int *primary_index) {
    if (n <= 0) return;
    if (n == 1) { *primary_index = 0; return; }

    int *sa = build_cyclic_suffix_array(data, n);
    u8  *L  = malloc(n);
    if (!sa || !L) { free(sa); free(L); return; }

    *primary_index = 0;
    for (int i = 0; i < n; i++) {
        L[i] = data[(sa[i] + n - 1) % n];
        if (sa[i] == 0) *primary_index = i;
    }
    memcpy(data, L, n);
    free(sa); free(L);
}

// LF-mapping inverse: reconstructs original from BWT last column + primary_index.
static void bwt_inverse(u8 *data, int n, int primary_index) {
    if (n <= 1) return;

    int freq[256] = {0};
    for (int i = 0; i < n; i++) freq[data[i]]++;

    int first[256], acc = 0;
    for (int c = 0; c < 256; c++) { first[c] = acc; acc += freq[c]; }

    int *LF  = malloc(n * sizeof(int));
    u8  *out = malloc(n);
    if (!LF || !out) { free(LF); free(out); return; }

    int occ[256] = {0};
    for (int i = 0; i < n; i++)
        LF[i] = first[data[i]] + occ[data[i]]++;

    int row = primary_index;
    for (int i = n - 1; i >= 0; i--) {
        out[i] = data[row];
        row = LF[row];
    }
    memcpy(data, out, n);
    free(LF); free(out);
}

// ── MTF ──────────────────────────────────────────────────────────────────────

static void mtf_forward(u8 *data, int n) {
    u8 list[256];
    for (int i = 0; i < 256; i++) list[i] = (u8)i;
    for (int i = 0; i < n; i++) {
        int pos = 0;
        while (list[pos] != data[i]) pos++;
        data[i] = (u8)pos;
        u8 sym = list[pos];
        memmove(list + 1, list, pos);
        list[0] = sym;
    }
}

static void mtf_inverse(u8 *data, int n) {
    u8 list[256];
    for (int i = 0; i < 256; i++) list[i] = (u8)i;
    for (int i = 0; i < n; i++) {
        int pos = data[i];
        u8  sym = list[pos];
        data[i] = sym;
        memmove(list + 1, list, pos);
        list[0] = sym;
    }
}

// ── Parameter tuning ─────────────────────────────────────────────────────────
// Probes ScanBits and ScrambeBits on the first block to find the combination
// that yields the most net gain. Runs once at startup; sets ScanSizeBitsG,
// ScrambeSizeBitsG, ScanSize, ScrambleSize for the whole file.

static void tuneParams(const u8 *data, int size) {
    // Phase 1 sample: BLOCK_SIZE/4 (16 KB) — large enough to capture compounding,
    // small enough for 20 FindNextBetter calls to finish in a few seconds.
    int sample = BLOCK_SIZE / 4;
    if (sample > size) sample = size;
    if (sample < 1024) sample = 1024;

    // Phase 2 sample: BLOCK_SIZE/8 (8 KB) — ScrambleSize search is slower per call.
    int scrSample = BLOCK_SIZE / 8;
    if (scrSample > size) scrSample = size;
    if (scrSample < 512) scrSample = 512;

    printf("Tuning on %d bytes (scramble probe: %d bytes)...\n", sample, scrSample);

    // Build BWT+MTF base once — all Phase 1 probes start from the same state.
    u8 *base = malloc(sample);
    if (!base) return;
    memcpy(base, data, sample);
    {
        int pi = 0;
        bwt_forward(base, sample, &pi);
        mtf_forward(base, sample);
    }
    double baseE = byteEntropy(base, sample);

    // ── Phase 1: tune ScanBits with 5-iteration multi-step probe ─────────────
    // A single-step probe is noisy: 1-bit difference in first-step gain means
    // nothing. Running 5 iterations captures the compounding effect — a larger
    // ScanSize unlocks more strides per block and its advantage grows with each
    // subsequent FindNextBetter call.
    int bestScanBits = ScanSizeBitsG;
    double bestScanGain = -1e18;

    printf("  ScanSize (5-iter probe, BWT+MTF base):\n");
    for (int sb = 2; sb <= 5; sb++) {
        u8 *copy = malloc(sample);
        if (!copy) continue;
        memcpy(copy, base, sample);

        ScanSizeBitsG = sb;
        ScanSize      = 1 << sb;

        double curE   = baseE;
        int    totalOH = 0;

        for (int iter = 0; iter < 5; iter++) {
            u8 amp = 0, op = 0;
            int v = FindNextBetter(copy, sample, &amp, &op);
            if (v == 0) break;
            apply_op_inplace(copy, sample, v, amp, op);
            double newE = byteEntropy(copy, sample);
            int    oh   = op_overhead(op);
            if ((curE - newE) * sample <= oh) {
                undo_op_inplace(copy, sample, v, amp, op);
                break;
            }
            totalOH += oh;
            curE     = newE;
        }

        double netGain = (baseE - curE) * sample - totalOH;
        printf("    ScanSize=%2d (bits=%d): net_gain=%.1f bits%s\n",
               1 << sb, sb, netGain, sb == ScanSizeBits ? " [default]" : "");
        if (netGain > bestScanGain) { bestScanGain = netGain; bestScanBits = sb; }
        free(copy);
    }
    ScanSizeBitsG = bestScanBits;
    ScanSize      = 1 << bestScanBits;
    printf("    => ScanSize=%d (bits=%d)\n", ScanSize, ScanSizeBitsG);
    free(base);

    // ── Phase 2: tune ScrambeBits ─────────────────────────────────────────────
    // Build BWT+MTF, apply up to 5 transform steps with the tuned ScanBits to
    // get a realistic post-transform state, then probe each ScrambleSize.
    u8 *post = malloc(scrSample);
    if (post) {
        memcpy(post, data, scrSample);
        {
            int pi = 0;
            bwt_forward(post, scrSample, &pi);
            mtf_forward(post, scrSample);
        }
        for (int iter = 0; iter < 5; iter++) {
            u8 amp = 0, op = 0;
            int v = FindNextBetter(post, scrSample, &amp, &op);
            if (v == 0) break;
            double eBefore = byteEntropy(post, scrSample);
            apply_op_inplace(post, scrSample, v, amp, op);
            double eAfter  = byteEntropy(post, scrSample);
            if ((eBefore - eAfter) * scrSample <= op_overhead(op)) {
                undo_op_inplace(post, scrSample, v, amp, op);
                break;
            }
        }
        double postE = byteEntropy(post, scrSample);

        int    bestScrBits = ScrambeSizeBitsG;
        double bestScrGain = -1e18;

        printf("  ScrambleSize (post-transform state):\n");
        for (int scb = 2; scb <= 5; scb++) {
            ScrambeSizeBitsG = scb;
            ScrambleSize     = 1 << scb;

            u8 *scopy = malloc(scrSample);
            if (!scopy) continue;
            memcpy(scopy, post, scrSample);

            u8 seed, scan, scrAmp, scrOp;
            FindNextBestScramble(scopy, scrSample, &seed, &scan, &scrAmp, &scrOp);
            scramble_bytes(scopy, scrSample, seed);
            if (scan > 0) apply_op_inplace(scopy, scrSample, scan, scrAmp, scrOp);

            double scrE = byteEntropy(scopy, scrSample);
            int    oh   = ScrambeSizeBitsG + (scan > 0 ? op_overhead(scrOp) : 0);
            double gain = (postE - scrE) * scrSample - oh;

            printf("    ScrambleSize=%2d (bits=%d): gain=%.1f bits%s\n",
                   1 << scb, scb, gain, scb == ScrambeSizeBits ? " [default]" : "");
            if (gain > bestScrGain) { bestScrGain = gain; bestScrBits = scb; }
            free(scopy);
        }
        ScrambeSizeBitsG = bestScrBits;
        ScrambleSize     = 1 << bestScrBits;
        printf("    => ScrambleSize=%d (bits=%d)\n", ScrambleSize, ScrambeSizeBitsG);
        free(post);
    }

    printf("Tuned: ScanSize=%d (bits=%d)  ScrambleSize=%d (bits=%d)\n\n",
           ScanSize, ScanSizeBitsG, ScrambleSize, ScrambeSizeBitsG);
}

// ── Block processing ──────────────────────────────────────────────────────────

static double compressBlock(u8 *data, int bsize, int blockIdx) {
    double baseEntropy = byteEntropy(data, bsize);
    printf("\n=== Block %d (%d bytes)  base=%.4f bits/byte ===\n",
           blockIdx, bsize, baseEntropy);

    // BWT + MTF as pre-processing step
    printHistStats(data, bsize, "base");
    int primary_index = 0;
    bwt_forward(data, bsize, &primary_index);
    double afterBWT = byteEntropy(data, bsize);
    mtf_forward(data, bsize);
    double afterMTF = byteEntropy(data, bsize);
    printHistStats(data, bsize, "BWT+MTF");

    printf("  BWT: %.4f -> %.4f   MTF: -> %.4f   gain=%.1f bits  overhead=%d bits\n",
           baseEntropy, afterBWT, afterMTF,
           (baseEntropy - afterMTF) * bsize, BWT_OVERHEAD_BITS);

    int totalOverhead         = BWT_OVERHEAD_BITS;
    int totalScrambleOverhead = 0;
    double LastEntropy        = afterMTF;
    int done = 0;

    while (!done) {
        u8 amp = 0, op = 0;
        int v = FindNextBetter(data, bsize, &amp, &op);

        // 0 = no transform improved entropy at all; mandatory scramble-or-stop.
        // 1 = transform is unprofitable after overhead; mandatory scramble-or-stop.
        // 2 = profitable but small gain; try scramble as bonus, keep going even if fails.
        // 3 = profitable and large gain; no scramble needed, loop immediately.
        int scrambleMode = 3;

        if (v == 0) {
            printf("  no better transform\n");
            scrambleMode = 0;
        } else {
            apply_op_inplace(data, bsize, v, amp, op);
            double empt = byteEntropy(data, bsize);
            int overhead = op_overhead(op);
            double Difference = (LastEntropy - empt) * bsize - overhead;
            printf("  [%s stride=%d amp=%d] entropy=%.4f  profit=%.1f\n",
                   op_names[op], v, amp, empt, Difference);
            printHistStats(data, bsize, op_names[op]);

            if (Difference <= 0) {
                undo_op_inplace(data, bsize, v, amp, op);
                scrambleMode = 1;  // mandatory
            } else {
                totalOverhead += overhead;
                LastEntropy = empt;
                // Small gain: try scramble as bonus — don't stop if it fails
                if (Difference < overhead) scrambleMode = 2;
            }
        }

        if (scrambleMode <= 2) {
            double before = byteEntropy(data, bsize);
            ScrambleStep passes[MAX_SCRAMBLE_PASSES];
            int nPasses = 0;
            double currentEntropy = before;

            while (nPasses < MAX_SCRAMBLE_PASSES) {
                u8 seed, scan, scrAmp, scrOp;
                FindNextBestScramble(data, bsize, &seed, &scan, &scrAmp, &scrOp);
                scramble_bytes(data, bsize, seed);
                if (scan > 0) apply_op_inplace(data, bsize, scan, scrAmp, scrOp);
                double scrEmpt = byteEntropy(data, bsize);
                int passOverhead = ScrambeSizeBitsG + (scan > 0 ? op_overhead(scrOp) : 0);
                double passGain  = (currentEntropy - scrEmpt) * bsize;

                if (passGain <= passOverhead) {
                    if (scan > 0) undo_op_inplace(data, bsize, scan, scrAmp, scrOp);
                    unscramble_bytes(data, bsize, seed);
                    break;
                }
                passes[nPasses++] = (ScrambleStep){seed, scan, scrAmp, scrOp};
                currentEntropy = scrEmpt;
            }

            if (nPasses == 0) {
                printf("  scramble not profitable\n");
                // Only stop if we had no profitable direct transform (modes 0 and 1).
                // Mode 2 means a profitable transform already ran — keep looping.
                if (scrambleMode <= 1) done = 1;
            } else {
                int chainOverhead = 0;
                for (int i = 0; i < nPasses; i++)
                    chainOverhead += ScrambeSizeBitsG +
                        (passes[i].scan > 0 ? op_overhead(passes[i].op) : 0);
                double ScrDiff = (before - currentEntropy) * bsize - chainOverhead;
                printf("  Scramble x%d  ScrDiff=%.1f bits\n", nPasses, ScrDiff);
                for (int i = 0; i < nPasses; i++)
                    printf("    pass %d: seed=%d [%s stride=%d amp=%d]\n", i + 1,
                           passes[i].seed,
                           passes[i].scan > 0 ? op_names[passes[i].op] : "NONE",
                           passes[i].scan, passes[i].amp);
                totalScrambleOverhead += chainOverhead;
                LastEntropy = currentEntropy;
            }
        }
    }

    double ending  = byteEntropy(data, bsize);
    double netGain = (baseEntropy - ending) * bsize - totalOverhead - totalScrambleOverhead;
    printHistStats(data, bsize, "final");
    printf("  Block %d done: %.4f -> %.4f  net_gain=%.1f bits"
           "  (overhead %d + %d scramble = %d bits)\n",
           blockIdx, baseEntropy, ending, netGain,
           totalOverhead, totalScrambleOverhead,
           totalOverhead + totalScrambleOverhead);
    return netGain;
}

int main(void) {
    init_gf256();
    srand(5);

    int size = 1024 * 1024;
    u8 *Data = malloc(size);
    if (!Data) return 1;
    for (int x = 0; x < size; x++) Data[x] = rand() % 256;

    double BaseEntropy = byteEntropy(Data, size);
    printf("Data: %d bytes  global entropy=%.4f bits/byte\n", size, BaseEntropy);

    int nThreads = omp_get_max_threads() - 1;
    if (nThreads < 1) nThreads = 1;
    omp_set_num_threads(nThreads);
    printf("Using %d threads\n", nThreads);

    tuneParams(Data, size);

    int numBlocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    printf("Blocks: %d x %d bytes\n", numBlocks, BLOCK_SIZE);

    double totalBitsBefore = 0.0, totalBitsAfter = 0.0, totalNetGain = 0.0;

    for (int b = 0; b < numBlocks; b++) {
        int offset = b * BLOCK_SIZE;
        int bsize  = (offset + BLOCK_SIZE <= size) ? BLOCK_SIZE : (size - offset);
        u8 *block  = Data + offset;

        totalBitsBefore += byteEntropy(block, bsize) * bsize;
        totalNetGain    += compressBlock(block, bsize, b);
        totalBitsAfter  += byteEntropy(block, bsize) * bsize;
    }

    double rawDrop = totalBitsBefore - totalBitsAfter;
    double overhead = rawDrop - totalNetGain;
    double withInstructions = totalBitsAfter + overhead;
    double sz = (double)size;
    printf("\n=== Summary ===\n");
    printf("  %-32s %10.0f bits  (%10.2f bytes)  %.6f bits/byte\n",
           "Original (no reduction):", totalBitsBefore, totalBitsBefore / 8.0,
           totalBitsBefore / sz);
    printf("  %-32s %10.0f bits  (%10.2f bytes)  %.6f bits/byte\n",
           "Reduced (data only):", totalBitsAfter, totalBitsAfter / 8.0,
           totalBitsAfter / sz);
    printf("  %-32s %10.0f bits  (%10.2f bytes)  %.6f bits/byte\n",
           "Reduced + instructions:", withInstructions, withInstructions / 8.0,
           withInstructions / sz);
    printf("  %-32s %10.0f bits  (%10.2f bytes)\n",
           "Net saved:", totalNetGain, totalNetGain / 8.0);

    free(Data);
    return 0;
}
