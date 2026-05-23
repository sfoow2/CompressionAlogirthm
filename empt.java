
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


u8 getRandomNum(u8 seed, u8 thresh){
    uint32_t s = (uint32_t)seed | 0x12340000u;

    u8 val = 0;

    for (int x = 0; x < 8; x++){
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        val |= (u8)((((s >> 24) & 0xFF) >= thresh) << x);
    }

    return val;

}

float LastTopEmpt = 0;
u8 LastTr = 0;
u8 LastSeed = 0;

u8 TryToComp(u8* Data, int size){

    u8* DataBack = malloc(size);
    for (int x = 0; x < size; x++){
        DataBack[x] = Data[x];
    }

    float BaseEmpt = getEntropy(Data,size);

    float TopEmpt = BaseEmpt;
    u8 TopTr = 0;
    u8 Topseed = 0;

    for (u8 seed = 0; seed < 255; seed++){

        for (u8 tr = 1; tr < 255; tr++){
            
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

    for (int x = 0; x < size; x++){
        Data[x] = DataBack[x] ^ getRandomNum(Topseed - x, TopTr + x);//should write the values
    }

    LastTr = TopTr;
    LastSeed = Topseed;
    LastTopEmpt = TopEmpt; 

    if (BaseEmpt <= TopEmpt){
        return 1; //cant do more
    } else {

        float SizeBase = (size * BaseEmpt) / 8;
        float SizeAfter = (size * TopEmpt) / 8;
        
        float Diff = SizeBase - SizeAfter;

        if (Diff > 2){
            return 0;
        } else {
            return 3;
        }
    }
}




int main(){
    srand(452);


    FILE *fptr = fopen("before.bin","w");

    int size = 256;
    u8 *Data = malloc(size);
    for (int x = 0; x < size; x++){
        Data[x] = rand() % 256;
        fprintf(fptr, "%c", Data[x]);
    }

    float BaseEmpt = getEntropy(Data,size);

    printf("Base = %lf",BaseEmpt);

    for (int count = 0; count < 255; count++){
        u8 prot = TryToComp(Data,size);

        if (prot == 0){
            printf("\nNew Layer: %d empt = %lf Seed = %d, tresh = %d",count,LastTopEmpt,LastSeed,LastTr);
        } else if (prot == 1){
            printf("\nFound Max at %d, empt = %lf", count, LastTopEmpt);
            break;
        } else if (prot == 3){
            printf("\n Found one but too low at %d, empt = %lf", count, LastTopEmpt);
            break;
        }
    }    

    FILE *after = fopen("after.bin","w");
    
    for (int x = 0; x < size; x++){
        fprintf(after, "%c", Data[x]);
    }


    fclose(fptr);
    fclose(after);
    return 0;

    
}
