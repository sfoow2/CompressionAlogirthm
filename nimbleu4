*/
DO NOT USE THIS CODE IT WILL BREAK EVERYTHING AND IS UNSTABEL
/*

#include <stdio.h>
#include <stdint.h>
#include "Markov.c"

typedef uint8_t u8;
typedef uint8_t u4;  // values 0–15 only
typedef uint8_t u3;  // values 0–7 only

#define NIBBLE(x)   ((x) & 0x0F)
#define SEED_COUNT  16   // u4 seed space
#define OFFSET_MAX   8   // u3 offset space (0–7)
#define BUF_SIZE     8   // only need 8 draws after first

MarkovPRNG rng;

void SetSeed(u4 seed) {
    markov_seed(&rng, seed);  // seed is already 0–15
}

u4 getNext() {
    return NIBBLE(markov_next(&rng));  // truncate to 4 bits
}

// Returns offset (0–7) on success, 255 on failure
// Encodes: first output == a, then b appears within OFFSET_MAX steps
u8 FindSolution(u4 a, u4 b, u4 *out_seed, u3 *out_offset) {
    u4 buffer[OFFSET_MAX];

    for (u4 seed = 0; seed < SEED_COUNT; seed++) {
        SetSeed(seed);

        if (getNext() == a) {
            for (u3 t = 0; t < OFFSET_MAX; t++) {
                buffer[t] = getNext();
            }

            for (u3 z = 0; z < OFFSET_MAX; z++) {
                if (buffer[z] == b) {
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
    int sum = 0;
    int total = 16 * 16;  // all u4 x u4 pairs

    for (u4 a = 0; a < 16; a++) {
        for (u4 b = 0; b < 16; b++) {
            u4 found_seed;
            u3 found_offset;
            if (FindSolution(a, b, &found_seed, &found_offset) == 0) {
                sum++;
                // \\printf("(%2u,%2u) -> seed=%u offset=%u\n", a, b, found_seed, found_offset);
            }
        }
    }

    printf("found %d / %d  (%.4f%%)\n", sum, total, 100.0 * sum / total);
    return 0;
}
