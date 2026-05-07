*/
DO NOT USE THIS CODE IT WILL BREAK YOUR SYSTEM
  /*

#include <stdio.h>
#include <stdint.h>
#include <string.h>

typedef uint8_t u8;
typedef uint8_t u4;
typedef uint8_t u3;

#define SEED_COUNT  16
#define OFFSET_MAX   8
#define SYM_COUNT   16

// Simple 4-bit permutation via repeated XOR-rotate mixing
// Each seed produces a different permutation of 0–15
static u4 perm[SEED_COUNT][SYM_COUNT];

void build_permutations() {
    for (int s = 0; s < SEED_COUNT; s++) {
        // Start with identity
        u4 p[SYM_COUNT];
        for (int i = 0; i < SYM_COUNT; i++) p[i] = i;

        // Fisher-Yates using a tiny LCG keyed on seed
        uint32_t lcg = (uint32_t)s * 2654435761u ^ 0xDEADBEEF;
        for (int i = SYM_COUNT - 1; i > 0; i--) {
            lcg = lcg * 1664525u + 1013904223u;
            int j = (lcg >> 28) % (i + 1);  // 4-bit index
            u4 tmp = p[i]; p[i] = p[j]; p[j] = tmp;
        }

        for (int i = 0; i < SYM_COUNT; i++)
            perm[s][i] = p[i];
    }
}

// pos=0 is the "first output" (the a matcher)
// pos=1..7 is the window for b
static inline u4 get_output(u4 seed, u8 pos) {
    return perm[seed][pos % SYM_COUNT];
}

u8 FindSolution(u4 a, u4 b, u4 *out_seed, u3 *out_offset) {
    for (u4 seed = 0; seed < SEED_COUNT; seed++) {
        if (get_output(seed, 0) == a) {
            for (u3 z = 0; z < OFFSET_MAX; z++) {
                if (get_output(seed, z + 1) == b) {
                    *out_seed   = seed;
                    *out_offset = z;
                    return 0;
                }
            }
        }
    }
    return 1;
}

int main() {
    build_permutations();

    // Debug: print permutations so we can see diversity
    printf("Permutations:\n");
    for (int s = 0; s < SEED_COUNT; s++) {
        printf("seed %2d: ", s);
        for (int i = 0; i < SYM_COUNT; i++)
            printf("%2d ", perm[s][i]);
        printf("\n");
    }

    int sum = 0;
    for (u4 a = 0; a < 16; a++) {
        for (u4 b = 0; b < 16; b++) {
            u4 fs; u3 fo;
            if (FindSolution(a, b, &fs, &fo) == 0)
                sum++;
        }
    }

    printf("\nfound %d / 256  (%.4f%%)\n", sum, 100.0 * sum / 256);
    return 0;
}
