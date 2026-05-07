*/
The following code is experimental is should really really really really really really not be used at all, it might provoke random crashes
/*


#include <stdio.h>
#include <stdint.h>
#include "Markov.c"

typedef uint8_t u8;

MarkovPRNG rng;

void SetSeed(unsigned char seed){
    markov_seed(&rng, seed);
}

unsigned char getNext(){
    return markov_next(&rng);
}


u8 FindSolution(uint8_t a, uint8_t b) {
    u8 buffer[128];

    for (uint16_t x = 0; x < 256; x++) {
        SetSeed((u8)x);

        if (getNext() == a) {
            
            for (u8 t = 0; t < 128; t++){
                buffer[t] = getNext();
            }

            for (u8 z = 0; z < 128; z++) {
                if (buffer[z] == b) {
                    printf("found b at offset %d for seed %d\n", z, x);
                    return 0;
                }
            }
        }
    }
    return 1;
}


int main() {

    int sum = 0;

    for (u8 x = 0; x < 255; x++){
        for (u8 z = 0; z < 255; z++){
            if (FindSolution(x,z) == 0){
                sum++;
            }
        }
    }

    printf("found %d / %d",sum,256*256);

    return 0;
}
