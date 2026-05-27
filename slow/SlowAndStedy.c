#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <omp.h>

typedef uint8_t u8;

#define ScrambeSizeBits    4
#define ScanSizeBits       4
#define AmplifierSizeBits  7
#define MAX_SCRAMBLE_PASSES 8

int ScrambleSize = (1 << ScrambeSizeBits);
int ScanSize     = (1 << ScanSizeBits);
int AmpliferScan = (1 << AmplifierSizeBits);

int FindNextBetter(u8 *Data, int size, u8 *amp, u8 *op);

static const char *op_names[] = {"ADD", "XOR", "ROL", "MUL"};

typedef struct { u8 seed, scan, amp, op; } ScrambleStep;

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

// finds b such that (a * b) % 256 == 1; a must be odd
static u8 mul_inverse_byte(u8 a) {
    for (int b = 1; b <= 255; b += 2) {
        if ((u8)(a * b) == 1) return (u8)b;
    }
    return 1;
}

static void apply_op_inplace(u8 *Data, int size, int stride, u8 amp, u8 op) {
    for (int p = 0; p < size; p += stride) {
        switch (op) {
            case 0: Data[p] += amp; break;
            case 1: Data[p] ^= amp; break;
            case 2: Data[p] = (u8)((Data[p] << amp) | (Data[p] >> (8 - amp))); break; // ROL
            case 3: Data[p] = (u8)(Data[p] * amp); break;                              // MUL
        }
    }
}

static void undo_op_inplace(u8 *Data, int size, int stride, u8 amp, u8 op) {
    u8 inv = (op == 3) ? mul_inverse_byte(amp) : 0; // precompute once
    for (int p = 0; p < size; p += stride) {
        switch (op) {
            case 0: Data[p] -= amp; break;
            case 1: Data[p] ^= amp; break;
            case 2: Data[p] = (u8)((Data[p] >> amp) | (Data[p] << (8 - amp))); break; // ROR
            case 3: Data[p] = (u8)(Data[p] * inv); break;
        }
    }
}

// bits needed to encode one transform step in the bitstream
static int op_overhead(u8 op) {
    static const int amp_bits[] = {AmplifierSizeBits, AmplifierSizeBits, 3, AmplifierSizeBits};
    return ScanSizeBits + 2 + amp_bits[op]; // 2 bits for op selector
}

void scramble_bytes(u8 *data, int size, u8 seed) {
    if (!data || size <= 1) return;
    srand(seed);
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        u8 tmp = data[i];
        data[i] = data[j];
        data[j] = tmp;
    }
}

void unscramble_bytes(u8 *data, int size, u8 seed) {
    if (!data || size <= 1) return;
    int *swaps = malloc(size * sizeof(int));
    srand(seed);
    for (int i = size - 1; i > 0; i--)
        swaps[i] = rand() % (i + 1);
    for (int i = 1; i < size; i++) {
        u8 tmp = data[i];
        data[i] = data[swaps[i]];
        data[swaps[i]] = tmp;
    }
    free(swaps);
}

void FindNextBestScramble(u8 *Data, int size, u8 *seed, u8 *Scan, u8 *amp, u8 *op) {
    u8 *Data2 = malloc(size);

    double Topempt = byteEntropy(Data, size);
    u8 BestSeed = 0;
    u8 BestScan = 0;
    u8 BestAmp  = 0;
    u8 BestOp   = 0;

    for (int si = 0; si < ScrambleSize; si++) {
        for (int x = 0; x < size; x++) Data2[x] = Data[x];
        scramble_bytes(Data2, size, (u8)si);

        u8 Lastamp = 0;
        u8 Lastop  = 0;
        u8 LastScan = FindNextBetter(Data2, size, &Lastamp, &Lastop);

        if (LastScan > 0) {
            apply_op_inplace(Data2, size, LastScan, Lastamp, Lastop);
        }

        double empt = byteEntropy(Data2, size);

        if (empt < Topempt) {
            Topempt  = empt;
            BestSeed = (u8)si;
            BestScan = LastScan;
            BestAmp  = Lastamp;
            BestOp   = Lastop;
        }
    }

    *seed = BestSeed;
    *Scan = BestScan;
    *amp  = BestAmp;
    *op   = BestOp;

    free(Data2);
}

int FindNextBetter(u8 *Data, int size, u8 *amp, u8 *op) {
    double BestEmpt = byteEntropy(Data, size);
    int BestScan = 0;
    int BestAmp  = 0;
    int BestOp   = 0;

    #pragma omp parallel
    {
        u8 *Data2 = malloc(size);

        // each thread tracks its own best; merged into globals at the end
        double localBestEmpt = BestEmpt;
        int localScan = 0, localAmp = 0, localOp = 0;

        if (Data2) {
            #pragma omp for schedule(dynamic, 1)
            for (int x = 1; x <= ScanSize; x++) {

                // one copy per stride; all ops reuse by overwriting only strided positions
                for (int s = 0; s < size; s++) Data2[s] = Data[s];

                // ADD + XOR share the same amplitude range and the same copy
                for (int a = 1; a <= AmpliferScan; a++) {

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
            }

            free(Data2);
        }

        // serialize the merge: one thread at a time updates the shared best
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

    *amp = (u8)BestAmp;
    *op  = (u8)BestOp;
    return BestScan;
}

int main(void) {
    srand(5);

    int size = 1024 * 1024;
    u8 *Data = malloc(size);
    if (!Data) return 1;

    for (int x = 0; x < size; x++) Data[x] = rand() % 256;

    double BaseEntropy = byteEntropy(Data, size);
    printf("Base empt = %lf\n", BaseEntropy);

    int done = 0;
    int totalOverhead = 0;
    int totalScrambleOverhead = 0;

    double Starting    = BaseEntropy;
    double LastEntropy = Starting;

    int nThreads = omp_get_max_threads() - 1;
    if (nThreads < 1) nThreads = 1;
    omp_set_num_threads(nThreads);
    printf("Using %d threads\n", nThreads);

    srand(88);
    int ScrambleCount = 0;

    while (!done) {
        u8 amp = 0;
        u8 op  = 0;
        int v = FindNextBetter(Data, size, &amp, &op);

        if (v == 0) {
            printf("\nfound nothing\n");
            done = 1;
        } else {
            apply_op_inplace(Data, size, v, amp, op);

            double empt = byteEntropy(Data, size);
            printf("new empt = %lf  [%s stride=%d amp=%d]\n", empt, op_names[op], v, amp);

            int overhead = op_overhead(op);
            double Difference = ((LastEntropy * size) - (empt * size) - overhead);
            printf("profit of %lf bits\n", Difference);

            int tryScramble = 0;

            if (Difference <= 0) {
                undo_op_inplace(Data, size, v, amp, op);
                empt = LastEntropy;
                tryScramble = 1;
            } else {
                totalOverhead += overhead;
                LastEntropy = empt;
                if (Difference <= overhead) {
                    tryScramble = 1;
                }
            }

            if (tryScramble) {
                double before = empt;
                ScrambleStep passes[MAX_SCRAMBLE_PASSES];
                int nPasses = 0;
                double currentEntropy = before;

                while (nPasses < MAX_SCRAMBLE_PASSES) {
                    u8 seed, scan, scrAmp, scrOp;
                    FindNextBestScramble(Data, size, &seed, &scan, &scrAmp, &scrOp);
                    scramble_bytes(Data, size, seed);
                    if (scan > 0) apply_op_inplace(Data, size, scan, scrAmp, scrOp);

                    double scrEmpt = byteEntropy(Data, size);

                    if (scrEmpt >= currentEntropy) {
                        if (scan > 0) undo_op_inplace(Data, size, scan, scrAmp, scrOp);
                        unscramble_bytes(Data, size, seed);
                        break;
                    }

                    passes[nPasses++] = (ScrambleStep){seed, scan, scrAmp, scrOp};
                    currentEntropy = scrEmpt;
                }

                if (nPasses == 0) {
                    printf("\nnot profitable\n");
                    done = 1;
                } else {
                    int chainOverhead = 0;
                    for (int i = 0; i < nPasses; i++)
                        chainOverhead += ScrambeSizeBits + (passes[i].scan > 0 ? op_overhead(passes[i].op) : 0);
                    double ScrDiff = (((before * size) - (currentEntropy * size)) - chainOverhead);
                    printf("Scramble chain x%d  ScrDiff=%lf\n", nPasses, ScrDiff);
                    for (int i = 0; i < nPasses; i++)
                        printf("  pass %d: seed=%d [%s stride=%d amp=%d]\n", i + 1,
                               passes[i].seed,
                               passes[i].scan > 0 ? op_names[passes[i].op] : "NONE",
                               passes[i].scan, passes[i].amp);
                    ScrambleCount += nPasses;
                    totalScrambleOverhead += chainOverhead;
                    LastEntropy = currentEntropy;
                }
            }
        }
    }

    double ending     = byteEntropy(Data, size);
    double Difference = (((Starting * size) - (ending * size)) - totalOverhead - totalScrambleOverhead);
    printf("Total Difference of %lf bits\n", Difference);

    free(Data);
    return 0;
}
