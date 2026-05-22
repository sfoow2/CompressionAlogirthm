
//DO NOT USE THIS CODE YOU WILL BREAK YOUR COMPUTER THIS BREAKS THE LAWS OF MATTER



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


u8 getRandomNum(u8 seed, unsigned short thresh){
    uint32_t s = (uint32_t)seed | 0x12340000u;

    u8 val = 0;

    for (int x = 0; x < 8; x++){
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        val |= (u8)(((s >> 16) >= thresh) << x);
    }

    return val;

}



int main(){
    int StartSeed = 523;

    srand(StartSeed);

    int size = 256;
    u8 *Data = malloc(size);
    u8 *DataBack = malloc(size);
    for (int x = 0; x < size; x++){
        Data[x] = rand() % 255;
        DataBack[x] = Data[x];
    }

    float BaseEmpt = getEntropy(Data,size);

    float TopEmpt = BaseEmpt;
    unsigned short TopTr = 0;
    u8 Topseed = 0;

    for (u8 seed = 0; seed < 255; seed++){

        for (unsigned short tr = 1; tr < 65535; tr++){
            
            for (int x = 0; x < size; x++){
                Data[x] = DataBack[x];
            }

            for (int x = 0; x < size; x++){
                Data[x] = Data[x] ^ getRandomNum(seed - x, tr + x);
            }

            float empt = getEntropy(Data,size);

            if (empt <= TopEmpt){
                Topseed = seed;
                TopTr = tr;
                TopEmpt = empt;
            }
        }   
    }

    u8 same = 0;
    for (int x = 0; x < size; x++){
        if (DataBack[x] == Data[x]){
            same++;
        }
    }
    if (same > 0){
        printf("has %d thats same", same);
    }


    printf("Base = %lf, other = %lf, seed = %u, tr = %u",BaseEmpt,TopEmpt,Topseed,TopTr);

    return 0;
    
}
