*/
DO NOT USE THIS CODE IT WILL BREAK YOUR SYSTEM
  /*
  
#include <stdio.h>
#include <stdint.h>

typedef uint8_t u8;

/* ── Optimized table (128/256 = 50% coverage) ─────────────────────────────── */
static const u8 OPT_TABLE[16][9] = {
    {  6,  1,  8,  7,  4,  3, 15,  0, 12 },  /* seed  0: a= 6 */
    {  5, 11,  9,  5, 10,  1, 15,  4, 14 },  /* seed  1: a= 5 */
    { 10, 13,  9,  4, 14,  0, 12, 11,  5 },  /* seed  2: a=10 */
    {  9,  9, 13, 14,  3, 11,  2,  7,  6 },  /* seed  3: a= 9 */
    { 14, 13,  3,  2,  8,  7, 15,  9, 10 },  /* seed  4: a=14 */
    {  4,  0, 12,  4,  3, 15,  1, 14,  5 },  /* seed  5: a= 4 */
    { 13,  6,  5, 12, 13,  0, 10,  8,  1 },  /* seed  6: a=13 */
    { 15,  6,  1, 11,  9, 15,  0,  5, 12 },  /* seed  7: a=15 */
    {  1, 15,  7,  4,  6, 13, 14,  2, 12 },  /* seed  8: a= 1 */
    {  0,  6,  3,  2,  4, 13, 11, 14,  7 },  /* seed  9: a= 0 */
    {  8,  5,  0, 14, 15,  1,  9, 13, 11 },  /* seed 10: a= 8 */
    { 12, 11,  7, 14,  5,  1,  6, 10, 13 },  /* seed 11: a=12 */
    {  3, 11,  4, 15, 14,  9,  7,  2,  0 },  /* seed 12: a= 3 */
    { 11,  3,  2, 12, 15,  9,  0,  1,  6 },  /* seed 13: a=11 */
    {  2, 11,  8,  9, 15, 14,  2,  0,  6 },  /* seed 14: a= 2 */
    {  7,  5, 14, 11, 15,  2,  0,  1,  8 }   /* seed 15: a= 7 */
};

/* ── Encode: find seed+offset for (a,b) pair ─────────────────────────────── */
/* returns 1 on success, 0 on failure                                         */
/* seed = u4 (0-15), offset = u3 (0-7)                                        */
int FindSolution(u8 a, u8 b, u8 *out_seed, u8 *out_offset) {
    for (int s = 0; s < 16; s++) {
        if (OPT_TABLE[s][0] == a) {
            for (int z = 1; z < 9; z++) {
                if (OPT_TABLE[s][z] == b) {
                    *out_seed   = (u8)s;
                    *out_offset = (u8)(z - 1);
                    return 1;
                }
            }
        }
    }
    return 0;
}

/* ── Decode: recover (a,b) from seed+offset ─────────────────────────────── */
void Decode(u8 seed, u8 offset, u8 *a, u8 *b) {
    *a = OPT_TABLE[seed][0];
    *b = OPT_TABLE[seed][offset + 1];
}

/* ── Coverage test ───────────────────────────────────────────────────────── */
int main() {
    int sum = 0;

    for (int a = 0; a < 16; a++) {
        for (int b = 0; b < 16; b++) {
            u8 fs, fo;
            if (FindSolution((u8)a, (u8)b, &fs, &fo) == 1) {
                sum++;

                /* sanity check: decode must round-trip */
                u8 ra, rb;
                Decode(fs, fo, &ra, &rb);
                if (ra != a || rb != b) {
                    printf("ROUNDTRIP FAIL: (%d,%d) -> seed=%d off=%d -> (%d,%d)\n",
                           a, b, fs, fo, ra, rb);
                }
            }
        }
    }

    printf("found %d / 256  (%.4f%%)\n", sum, 100.0 * sum / 256);
    return 0;
}
