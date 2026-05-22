
//DO NOT USE THIS CODE YOU WILL BREAK YOUR COMPUTER THIS BREAKS THE LAWS OF MATTER



#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>

typedef uint8_t u8;

float getEntropy(u8* Data, int size) {
    int freq[256] = {0};
    for (int i = 0; i < size; i++)
        freq[Data[i]]++;

    double entropy = 0.0;
    for (int i = 0; i < 256; i++) {
        if (freq[i] == 0) continue;
        double p = (double)freq[i] / size;
        entropy -= p * log2(p);
    }

    return entropy;
}


u8 getRandomNum(int seed, u8 thresh){
    srand(seed);

    u8 val = 0;

    for (int x = 0; x < 8; x++){
        val = val | (((rand() % 100) >= thresh) << x);
    }


    /*printf("0b");
    for (int x = 0; x<8;x++){
        printf("%d",((val >> x) & 1) == 1);
    }*/

    return val;

}



int main(){
    srand(523);

    int size = 1048 * 16;
    u8 *Data = malloc(size);
    u8 *Data2 = malloc(size);
    for (int x = 0; x < size; x++){
        Data[x] = rand() % 255;
        Data2[x] = Data[x];
    }

    int freq[256] = {0};
    for (int i = 0; i < size; i++) {
        freq[Data[i]]++;
    }

    int TopElement = 0;
    int TopCount = 0;

    for (int x = 0; x < 255; x++){ 
        if (freq[x] > TopCount){
            TopElement = x;
            TopCount = freq[x];
        }
    }

    printf("Top Element is = %d with %d show",TopElement,TopCount);

    float BaseEmpt = getEntropy(Data,size);

    float TopEmpt = 8;
    int TopSeed = 0; 
    u8 TopThresh = 0; 
    
    for (int seed = 0; seed < 65536; seed++){

        for (int tr = 1; tr < 100; tr++){

            for (int x = 0; x < size; x++){
                Data2[x] = Data[x];
            }   
            
            for (int x = 0; x < size; x++){
                Data2[x] = Data2[x] = getRandomNum(seed + x,tr + x);
            }

            float empt = getEntropy(Data2,size);

            if (empt <= TopEmpt){
                TopEmpt = empt;
                TopSeed = seed;
                TopThresh = tr;
            }

        }
    }

    printf("\nStart = %lf, end = %lf, seed = %d, thre = %d",BaseEmpt,TopEmpt,TopSeed,TopThresh);


    // Encode original data with best parameters found
    u8 *Encoded = malloc(size);
    for (int x = 0; x < size; x++) {
        Encoded[x] = Data[x] ^ getRandomNum(TopSeed + x, TopThresh + x);
    }


    // Reverse: XOR with same key again (XOR is its own inverse)
    u8 *Recovered = malloc(size);
    for (int x = 0; x < size; x++) {
        Recovered[x] = Encoded[x] ^ getRandomNum(TopSeed + x, TopThresh + x);
    }

    // Check if recovered matches original
    int match = 1;
    for (int x = 0; x < size; x++) {
        if (Recovered[x] != Data[x]) {
            match = 0;
            printf("\n  Mismatch at index %d: original=%d, recovered=%d", x, Data[x], Recovered[x]);
        }
    }
    printf("\nReversed data %s the original", match ? "matches" : "does NOT match");

    free(Encoded);
    free(Recovered);

    return 0;
    
}
