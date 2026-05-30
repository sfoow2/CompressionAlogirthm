#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <bcrypt.h>
#include <omp.h>

typedef uint8_t u8;

#define BLOCK_SIZE  (4 * 1024)
#define MAX_STRIDE  64
#define N_OPS       6
#define N_PATTERNS  23

// ── GF(256) ───────────────────────────────────────────────────────────────────
static u8 gf_exp[512], gf_log[256];
static void init_gf256(void) {
    u8 x=1;
    for(int i=0;i<255;i++){gf_exp[i]=x;gf_log[x]=(u8)i;u8 h=(x&0x80)?((x<<1)^0x1B):(x<<1);x=h^x;}
    for(int i=255;i<512;i++) gf_exp[i]=gf_exp[i-255];
    gf_log[0]=0;
}
static inline u8 gf_mul(u8 a,u8 b){return(!a||!b)?0:gf_exp[gf_log[a]+gf_log[b]];}

// ── Core helpers ──────────────────────────────────────────────────────────────
static inline u8 lcg_byte(uint32_t *s){*s=*s*1664525u+1013904223u;return(u8)(*s>>24);}
static inline u8 op_byte(u8 v,u8 amp,int op){
    switch(op){
        case 0:return(u8)(v+amp);
        case 1:return(u8)(v^amp);
        case 2:return(u8)(v*(amp|1));
        case 3:return(u8)((v&0xF0)|((v+amp)&0x0F));
        case 4:{u8 s=(u8)((v<<4)|(v>>4));return s^amp;}
        case 5:{u8 a=amp<2?2:amp;return gf_mul(v,a);}
    }return v;
}
static double entropy_from_hist(const int f[256],int n){
    double e=0.0;for(int i=0;i<256;i++){if(!f[i])continue;double p=(double)f[i]/n;e-=p*log2(p);}return e;
}
static double byte_entropy(const u8*d,int n){int f[256]={0};for(int i=0;i<n;i++)f[d[i]]++;return entropy_from_hist(f,n);}
static int fill_random(u8*buf,int n){return BCryptGenRandom(NULL,(PUCHAR)buf,(ULONG)n,BCRYPT_USE_SYSTEM_PREFERRED_RNG)==0;}
static void hist_transform(const int in[256],int out[256],int op,int amp){
    memset(out,0,256*sizeof(int));for(int v=0;v<256;v++)if(in[v])out[op_byte((u8)v,(u8)amp,op)]+=in[v];
}
// ops: 0=ADD  1=XOR  2=MUL  3=ADDLO  4=SWXOR  5=GFMUL
static const char *opname[]={"ADD","XOR","MUL","ADDLO","SWXOR","GFMUL"};


// ── Search result struct ──────────────────────────────────────────────────────
typedef struct {
    int    id;
    char   name[48];
    double entropy;
    int    overhead;
    int    p[8];
    u8     amps[256];  // large enough for COND-PREV k=8 (256 groups)
} SR;

// Overhead (bits) = 5 (pattern type ID) + parameter bits
//  0  STRIDE-CONST:    stride(6)+op(3)+amp(8)          = 22
//  1  STRIDE-PRNG-AMP: stride(6)+seed(8)+op(3)         = 22
//  2  DUAL-STRIDE:     stride(6)+op(3)+a1(8)+a2(8)     = 30
//  3  PRNG-SELECT:     seed(8)+amp(8)+op(3)+thr(3)      = 27
//  4  GLOBAL-SHAPE:    shape(4)+op(3)+amp(8)            = 20
//  5  PERIODIC-SHAPE:  P(6)+shape(3)+op(3)+amp(8)       = 25
//  6  SPARSE-INDEX:    set_id(5)+op(3)+amp(8)           = 21
//  7  PRNG+GRADIENT:   shape(4)+seed(8)+op(3)           = 20
//  8  NWAY-STRIDE:     N(3)+op(3)+(N*8 for amps)        = 11+N*8 (dynamic)
//  9  PRNG-JUMP:       seed(9)+maxstep(4)+op(3)+amp(8)  = 29  (seed 0..511)
// 10  MODULAR-MASK:    P(3)+mask(8)+op(3)+amp(8)        = 27
// 11  STRIDE-NEIGHBOR: stride(6)+op(3)                  = 14
// 11  DELTA:           op_type(1)                       =  8  (5 ID + 3 op_type: SUB or XOR)
// 12  COND-PREV:       k(3)+op(3)+2^k*8 amps            = 27 min (k=1), up to 2059 (k=8)
// 13  COND-NEXT:       k(2)+op(3)+2^k*8 amps            = 26 min (k=1, dynamic)
// 14  COND-DELTA:      k(2)+op(3)+2^k*8 amps            = 26 min (k=1, dynamic)
// 15  STRIDE-DELTA:    N(6)+op_type(1)                  = 12  (stride-N difference/XOR)
// 16  THRESH-MAP:      T(8)+amp(8)                      = 21  (rotate [T..255] by amp positions)
// 17  BIT-ROTATE:      k(3)                             =  8  (rotate each byte's bits by k)
// 18  GRAY:            dir(1)                           =  6  (Gray encode or Gray decode)
// 19  DELTA-2:         (none)                           =  5  (second-order SUB delta: b[i]-2b[i-1]+b[i-2])
// 20  PRNG-WALK:       seed(8)+op(3)+amp(8)             = 24  (full LCG byte as step, ~n/128 visits)
// 21  MIN-SUB:         type(3)+params+min(8)              = 21 min (stride/walk/prng-sel/mod-mask/sparse)
// 22  POS-ADD:         step(8)                           = 13
// 23  POS-XOR:         step(8)                           = 13
// 24  NIBBLE-DELTA:    (none)                            =  5
// 25  ACCUM-XOR:       seed(8)                           = 13
// 26  ACCUM-ADD:       seed(8)                           = 13
// 27  HI-NIB-ADD:      amp(4)                            =  9
// 28  LO-NIB-ADD:      amp(4)                            =  9
// 29  HI-NIB-XOR:      amp(4)                            =  9
// 30  LO-NIB-XOR:      amp(4)                            =  9
// 31  BIT-REVERSE:     (none)                            =  5
// 32  PRNG-BITROT:     seed(8)                           = 13  per-byte PRNG rotation amount
// 33  PRNG-NIBSWAP:    seed(8)+thr(3)                    = 16  PRNG-gated nibble swap
// 34  PRNG-GRAY:       seed(8)+thr(3)+dir(1)             = 17  PRNG-gated Gray encode/decode
// 35  PRNG-HINIB-ADD:  seed(8)+thr(3)+amp(4)             = 20  PRNG-gated high-nibble rotate
// 36  PRNG-LONIB-ADD:  seed(8)+thr(3)+amp(4)             = 20  PRNG-gated low-nibble rotate
// 37  PRNG-BITREV:     seed(8)+thr(3)                    = 16  PRNG-gated bit reversal
// 38  PRNG-DELTA-GATE: seed(8)+thr(3)                    = 16  PRNG-gated delta subtraction
// 39  PRNG-HINIB-XOR:  seed(8)+thr(3)+amp(4)             = 20  PRNG-gated high-nibble XOR
// 40  PRNG-LONIB-XOR:  seed(8)+thr(3)+amp(4)             = 20  PRNG-gated low-nibble XOR
// 41  NWAY-PRNG-AMP:    N(3)+seed(8)+op(3)               = 19  N interleaved per-channel PRNGs
// 42  PRNG-STRD-DELTA:  N(6)+seed(8)+thr(3)             = 22  PRNG-gated stride delta
// 43  PRNG-DUAL-STRIDE: stride(6)+seed(8)+op(3)         = 22  dual stride with PRNG amps
// 44  PRNG-NIB-DELTA:   seed(8)+thr(3)                  = 16  PRNG-gated nibble delta
// 45  BLOCK-FOLD-XOR:   (none)                          =  5  XOR first half with second half
// 46  STRIDE-DELTA-2:   N(6)                            = 11  second-order stride delta
// 47  ROLLING-XOR-K:    k(6)                            = 11  b[i]^=b[i-k]^b[i-2k]
// 48  STRIDE-ACCUM:     stride(6)+seed(8)               = 19  ACCUM-XOR within each stride group
// 49  NIB-CROSS-DELTA:  (none)                          =  5  cross-delta between nibble streams
// 50  PRNG-CTXT-PREV:   k(3)+op(3)+seed(8)             = 19  PRNG-amp context-keyed by prev byte
// 51  PRNG-GFMUL-STRD:  stride(6)+seed(8)              = 19  PRNG GF256 coeff at stride positions
// 52  STRIDE-ACCUM-ADD: stride(6)+seed(8)              = 19  ACCUM-ADD within each stride group
// 53  POS-MOD-OP:       P(6)+op(3)                    = 14  amp=i%P, apply op(b[i], i%P)
// 54  PRNG-COMPLEMENT:  seed(8)+thr(3)                 = 16  PRNG-gated byte complement (255-v)
// 55  PRNG-STRD-XDELTA: N(6)+seed(8)+thr(3)           = 22  PRNG-gated XOR stride delta
// 56  PRNG-GFMUL-ALL:   seed(8)                        = 13  PRNG GF256 coeff per byte
// 57  STRIDE-NEG:       stride(6)                      = 11  complement (255-v) at stride positions
// 58  GF-STRIDE-FIXED:  stride(6)+alpha(8)             = 19  fixed-alpha GF256 at stride positions
// 59  PRNG-XOR-PREV:    seed(8)+thr(3)                 = 16  PRNG-gated XOR-with-previous-byte
// 60  DUAL-DELTA:       (none)                         =  5  b[i]-=b[i-1]+b[i-2], right-to-left
// 61  STRIDE-CROSS-XOR: stride(6)                      = 11  XOR pairs within stride-2s groups
// 62  STRIDE-NIBSWAP:   stride(6)                      = 11  nibble-swap at stride positions
// 63  GF-RAMP:          alpha(8)                        = 13  multiply b[i] by alpha^i in GF256
// 64  PRNG-PERIOD-AMP:  P(6)+seed(8)+op(3)             = 22  PRNG-derived per-position amps (periodic)
// 65  MOD-COMPLEMENT:   P(3)+r(3)                      = 11  complement bytes where i%P==r
// 66  VALUE-BKT-ROTATE: log2M(3)+amp(8)                = 16  rotate value within M-size buckets
// 67  BIT-SWAP-PAIRS:   (none)                         =  5  swap adjacent bit pairs in every byte
// 68  BIT-ROT-STRIDE:   stride(6)+k(3)                 = 14  fixed bit-rotation at stride positions
// 69  PRNG-HASH-CTXT:   seed(8)+op(3)                  = 16  context=b[i-1]^b[i-2], PRNG-amp
// 70  QUAD-DELTA:       (none)                         =  5  b[i]-=b[i-1]-b[i-2]+b[i-3]
// 71  PRNG-STRD-BITROT: stride(6)+seed(8)              = 19  per-position PRNG bit-rotation at stride
// 72  TWO-STRD-DELTA:   N1(6)+N2(6)                   = 17  b[i]-=b[i-N1]+b[i-N2]
// 73  STRIDED-GRAY:     stride(6)+dir(1)               = 12  Gray encode/decode at stride positions
// 74  NIBBLE-ACCUM:     seed(8)                        = 13  XOR nibble accumulator (self-inverse)
// 75  PRNG-POS-SCALE:   seed(8)                        = 13  b[i]+=(prng_i*(u8)i)%256
// 76  PRNG-STRD-COMP:   stride(6)+seed(8)+thr(3)       = 22  PRNG-gated complement at stride pos
// 77  PRNG-SPARSE-PRNG: set_id(5)+seed(8)+op(3)        = 21  sparse selection + PRNG amp per pos
// 78  STRD-COMPL-ALT:   stride(6)                      = 11  complement alternating stride groups
// 79  PRNG-BITSWAP:     seed(8)+thr(3)                 = 16  PRNG-gated bit-swap-pairs
// 80  STRIDE-GRP-AMP:   stride(6)+op(3)               = 14  amp=(i/stride)%256, group-index amp
// 81  DUAL-GRAY:        (none)                         =  5  encode first half, decode second half
// 82  FIBONACCI-DELTA:  (none)                         =  5  delta along Fibonacci index sequence
// 83  CHUNK-ACCUM:      chunk(6)                       = 11  ACCUM-XOR within each fixed-size chunk
// 84  PRNG-CTXT-NEXT-K: k(3)+op(3)+seed(8)            = 19  PRNG amps for next-byte context
// 85  STAIRCASE-ADD:    step(6)+scale(8)               = 19  b[i]+=(i/step)*scale
// 86  PARITY-XOR:       amp(8)                         = 13  XOR at even-popcount-index positions
// 87  TWO-ACCUM:        seed(8)                        = 13  separate ACCUM-XOR for even/odd pos
// 88  STAIRCASE-XOR:    step(6)                        = 11  b[i]^=(i/step)&0xFF, self-inverse
// 89  PRNG-TWIN-ACCUM:  seed1(8)+seed2(8)              = 21  two interleaved ACCUM-XOR streams
// 90  PRNG-CHUNK-ACCUM: chunk(6)+seed(8)               = 19  per-chunk PRNG-seeded ACCUM-XOR
// 91  HALF-DELTA:       (none)                         =  5  delta on first half of block only
// 92  XOR-ALTERNATING:  amp(8)                         = 13  even^amp, odd^(255-amp)
// 93  GF-STRIDE-HALF:   stride(6)+alpha(8)             = 19  alpha at even groups, inv_alpha at odd
// 94  PRNG-FIB-AMP:     seed(8)+op(3)                  = 16  PRNG amp at Fibonacci positions
// 95  PRNG-CHUNK-COMP:  chunk(6)+seed(8)+thr(3)        = 22  PRNG-gated complement, reset per chunk
// 96  BIT-INTERLEAVE:   (none)                         =  5  swap bit planes between consecutive pairs
// 97  PERIOD-TWO-OP:    op1(3)+op2(3)+amp(8)           = 19  alternate two ops with shared amp
// 98  STRIDE-GF-RAMP:   stride(6)+alpha(8)             = 19  GF-ramp at stride positions only
// 99  PRNG-HALF-ACCUM:  seed1(8)+seed2(8)              = 21  separate ACCUM-XOR per block half
static const int PAT_OH[] = {22,22,30,27,20,25,21,20,11,29,27, 8,27,26,26,12,21,8,6,5,24,21,
                              13,13,5,13,13,9,9,9,9,5,
                              13,16,17,20,20,16,16,20,20,19,
                              22,22,16,5,11,11,19,5,19,19,
                              19,14,16,22,13,11,19,16,5,11,
                              11,13,22,11,16,5,14,16,5,19,
                              17,12,13,13,22,21,11,16,
                              14,5,5,11,19,19,13,13,11,21,
                              19,5,13,19,16,22,5,19,19,21};

// ── Apply a SearchResult in-place ─────────────────────────────────────────────
static void apply_sr(u8 *blk, int n, const SR *r) {
    int x,op,amp,seed,P;
    u8 *orig=NULL;
    if(r->id==12||r->id==14){  // COND-PREV and COND-DELTA need orig
        orig=malloc(n); if(orig) memcpy(orig,blk,n);
    }
    switch(r->id){
        case 1: x=r->p[0];seed=r->p[1];op=r->p[2]; // STRIDE-PRNG-AMP
            {uint32_t st=(uint32_t)seed; for(int p=0;p<n;p+=x) blk[p]=op_byte(blk[p],lcg_byte(&st),op);} break;
        case 3: seed=r->p[0];amp=r->p[1];op=r->p[2]; // PRNG-SELECT
            {int thr_vals[]={25,64,128,192,230};u8 thr=(u8)thr_vals[r->p[3]];
             uint32_t st=(uint32_t)seed;
             for(int i=0;i<n;i++) if(lcg_byte(&st)>=thr) blk[i]=op_byte(blk[i],(u8)amp,op);} break;
        case 9: seed=r->p[0];{int ms=r->p[1];op=r->p[2];amp=r->p[3]; // PRNG-JUMP
            uint32_t st=(uint32_t)seed;int pos=0;
            while(pos<n){blk[pos]=op_byte(blk[pos],(u8)amp,op);pos+=(int)(lcg_byte(&st)%ms)+1;}} break;
        case 10: P=r->p[0];{int mask=r->p[1];op=r->p[2];amp=r->p[3]; // MODULAR-MASK
            for(int i=0;i<n;i++) if((mask>>(i%P))&1) blk[i]=op_byte(blk[i],(u8)amp,op);} break;
        case 12: {int k=r->p[0];op=r->p[1]; // COND-PREV: amp keyed by high k bits of previous byte
            if(orig) for(int i=1;i<n;i++) blk[i]=op_byte(blk[i],r->amps[orig[i-1]>>(8-k)],op);} break;
        case 13: {int k=r->p[0];op=r->p[1]; // COND-NEXT: amp keyed by high k bits of next byte
            for(int i=0;i<n-1;i++) blk[i]=op_byte(blk[i],r->amps[blk[i+1]>>(8-k)],op);} break;
        case 14: {int k=r->p[0];op=r->p[1]; // COND-DELTA: amp keyed by high k bits of (prev XOR pprev)
            if(orig) for(int i=2;i<n;i++) blk[i]=op_byte(blk[i],r->amps[(orig[i-1]^orig[i-2])>>(8-k)],op);} break;
        case 36: { // PRNG-LONIB-ADD: PRNG-gated low-nibble rotate, inv: subtract amp
            int tv[]={25,64,128,192,230}; u8 thr=(u8)tv[r->p[1]];
            int amp=r->p[2]; uint32_t st=(uint32_t)r->p[0];
            for(int i=0;i<n;i++){if(lcg_byte(&st)>=thr)
                blk[i]=(u8)((blk[i]&0xF0)|((blk[i]+amp)&0xF));}} break;
        case 39: { // PRNG-HINIB-XOR: PRNG-gated high-nibble XOR, self-inverse
            int tv[]={25,64,128,192,230}; u8 thr=(u8)tv[r->p[1]];
            int amp=r->p[2]; uint32_t st=(uint32_t)r->p[0];
            for(int i=0;i<n;i++){if(lcg_byte(&st)>=thr)
                blk[i]=(u8)((blk[i]&0x0F)|(((blk[i]>>4)^amp)<<4));}} break;
        case 40: { // PRNG-LONIB-XOR: PRNG-gated low-nibble XOR, self-inverse
            int tv[]={25,64,128,192,230}; u8 thr=(u8)tv[r->p[1]];
            int amp=r->p[2]; uint32_t st=(uint32_t)r->p[0];
            for(int i=0;i<n;i++){if(lcg_byte(&st)>=thr)
                blk[i]=(u8)((blk[i]&0xF0)|((blk[i]^amp)&0x0F));}} break;
        case 42: { // PRNG-STRD-DELTA: apply stride-N delta at PRNG-gated positions, right-to-left
            int N=r->p[0]; int tv[]={25,64,128,192,230}; u8 thr=(u8)tv[r->p[2]];
            uint32_t st=(uint32_t)r->p[1];
            u8 *gate=malloc(n); if(!gate) break;
            for(int i=0;i<n;i++) gate[i]=(lcg_byte(&st)>=thr)?1:0;
            for(int i=n-1;i>=N;i--) if(gate[i]) blk[i]=(u8)(blk[i]-blk[i-N]);
            free(gate);} break;
        case 43: { // PRNG-DUAL-STRIDE: dual sub-stride with per-element PRNG amps
            // Even sub-stride uses seed; odd sub-stride uses seed^255.
            int s=r->p[0]; int op=r->p[2];
            uint32_t st_e=(uint32_t)r->p[1], st_o=(uint32_t)(r->p[1]^255);
            for(int p=0;p<n;p+=2*s) blk[p]=op_byte(blk[p],lcg_byte(&st_e),op);
            for(int p=s;p<n;p+=2*s) blk[p]=op_byte(blk[p],lcg_byte(&st_o),op);} break;
        case 95: {int C=r->p[0]; u8 seed=(u8)r->p[1]; // PRNG-CHUNK-COMP
            int tv[]={25,64,128,192,230}; u8 thr=(u8)tv[r->p[2]];
            // PRNG resets per chunk; gate selects complement. Self-inverse.
            for(int base_=0;base_<n;base_+=C){
                uint32_t st=(uint32_t)seed;
                for(int k=0;k<C&&base_+k<n;k++)
                    if(lcg_byte(&st)>=thr) blk[base_+k]=(u8)(255-blk[base_+k]);}} break;
        case 98: {int s=r->p[0]; u8 alpha=(u8)r->p[1]; if(alpha<2)alpha=2; // STRIDE-GF-RAMP
            // At stride positions, multiply by successive powers of alpha (alpha^0, alpha^1, ...).
            // Inverse: multiply by powers of gf_inv(alpha).
            u8 acc=1;
            for(int p=0;p<n;p+=s){blk[p]=gf_mul(blk[p],acc);acc=gf_mul(acc,alpha);}} break;
        case 73: { // STRIDED-GRAY: Gray encode(dir=0) or decode(dir=1) at stride positions
            int s=r->p[0]; int dir=r->p[1];
            for(int p=0;p<n;p+=s){
                if(!dir)blk[p]^=(blk[p]>>1);
                else{u8 v=blk[p];v^=(v>>4);v^=(v>>2);v^=(v>>1);blk[p]=v;}}} break;
        case 76: { // PRNG-STRD-COMP: PRNG-gated complement at stride positions
            int s=r->p[0]; int tv[]={25,64,128,192,230}; u8 thr=(u8)tv[r->p[2]];
            uint32_t st=(uint32_t)(u8)r->p[1];
            for(int p=0;p<n;p+=s){if(lcg_byte(&st)>=thr) blk[p]=(u8)(255-blk[p]);}} break;
        case 62: {int s=r->p[0]; // STRIDE-NIBSWAP: nibble-swap at stride positions, self-inverse
            for(int p=0;p<n;p+=s){u8 v=blk[p];blk[p]=(u8)((v>>4)|(v<<4));}} break;
        case 68: {int s=r->p[0]; int k=r->p[1]; // BIT-ROT-STRIDE: fixed bit-rotation at stride
            for(int p=0;p<n;p+=s) blk[p]=(u8)((blk[p]<<k)|(blk[p]>>(8-k)));} break;
        case 71: {int s=r->p[0]; uint32_t st=(uint32_t)(u8)r->p[1]; // PRNG-STRD-BITROT
            for(int p=0;p<n;p+=s){int k=(int)(lcg_byte(&st)%7)+1;
                blk[p]=(u8)((blk[p]<<k)|(blk[p]>>(8-k)));}} break;
        case 55: { // PRNG-STRD-XDELTA: PRNG-gated XOR stride delta, right-to-left, self-inverse
            int N=r->p[0]; int tv[]={25,64,128,192,230}; u8 thr=(u8)tv[r->p[2]];
            uint32_t st=(uint32_t)r->p[1];
            u8 *gate=malloc(n); if(!gate) break;
            for(int i=0;i<n;i++) gate[i]=(lcg_byte(&st)>=thr)?1:0;
            for(int i=n-1;i>=N;i--) if(gate[i]) blk[i]^=blk[i-N];
            free(gate);} break;
        case 57: {int s=r->p[0]; // STRIDE-NEG: complement (255-v) at stride positions, self-inverse
            for(int p=0;p<n;p+=s) blk[p]=(u8)(255-blk[p]);} break;
        case 58: {int s=r->p[0]; u8 alpha=(u8)r->p[1]; if(alpha<2)alpha=2; // GF-STRIDE-FIXED
            // Inverse: multiply by gf_exp[255-gf_log[alpha]] at same positions.
            for(int p=0;p<n;p+=s) blk[p]=gf_mul(blk[p],alpha);} break;
        case 51: { // PRNG-GFMUL-STRD: PRNG GF256 coeff at stride positions, inv: GF256 inverse coeff
            int s=r->p[0]; uint32_t st=(uint32_t)(u8)r->p[1];
            for(int p=0;p<n;p+=s){u8 alpha=lcg_byte(&st); if(alpha<2)alpha=2;
                blk[p]=gf_mul(blk[p],alpha);}} break;
    }
    free(orig);
}

// ── Helpers for search functions ──────────────────────────────────────────────
// MUL(2) needs odd amp; ADDLO(3) amp 0-15; GFMUL(5) amp >= 2
#define OP_RANGE(op,alo,ahi) int alo=1,ahi=255; \
    if(op==2)alo=3; if(op==3)ahi=15; if(op==5)alo=2;
#define SKIP_OP(op,amp) if(op==2&&(amp&1)==0) continue;

// ── Search functions ──────────────────────────────────────────────────────────

static SR search_stride_prng_amp(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=1; r.entropy=base; r.overhead=PAT_OH[1];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    double be=base;int bx=0,bop=0,bseed=0;
    #pragma omp parallel
    {double le=base;int lx=0,lop=0,ls=0;
     #pragma omp for schedule(dynamic,1)
     for(int x=1;x<=MAX_STRIDE;x++){
         int sf[256]={0};for(int p=0;p<n;p+=x)sf[blk[p]]++;
         int nsf[256];for(int v=0;v<256;v++)nsf[v]=bfreq[v]-sf[v];
         for(int seed=0;seed<256;seed++)for(int op=0;op<N_OPS;op++){
             int tf[256]={0};uint32_t st=(uint32_t)seed;
             for(int p=0;p<n;p+=x)tf[op_byte(blk[p],lcg_byte(&st),op)]++;
             int ff[256];for(int v=0;v<256;v++)ff[v]=nsf[v]+tf[v];
             double e=entropy_from_hist(ff,n);if(e<le){le=e;lx=x;lop=op;ls=seed;}}
     }
     #pragma omp critical
     if(le<be){be=le;bx=lx;bop=lop;bseed=ls;}}
    r.entropy=be;r.p[0]=bx;r.p[1]=bseed;r.p[2]=bop;
    snprintf(r.name,sizeof(r.name),"STRIDE-PRNG-AMP %s s=%d seed=%d",opname[bop],bx,bseed);
    return r;
}

static SR search_prng_select(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=3; r.entropy=base; r.overhead=PAT_OH[3];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    int thr_vals[]={25,64,128,192,230};
    double be=base;int bseed=0,bop=0,bamp=0,bthi=0;
    #pragma omp parallel
    {double le=base;int ls=0,lop=0,la=0,lti=0;
     #pragma omp for schedule(dynamic,1)
     for(int si=0;si<256;si++){
         for(int ti=0;ti<5;ti++){
             u8 thr=(u8)thr_vals[ti];
             int sel[256]={0};uint32_t st=(uint32_t)si;
             for(int i=0;i<n;i++)if(lcg_byte(&st)>=thr)sel[blk[i]]++;
             int unsel[256];for(int v=0;v<256;v++)unsel[v]=bfreq[v]-sel[v];
             for(int op=0;op<N_OPS;op++){OP_RANGE(op,alo,ahi)
                 for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
                     int tf[256],ff[256];hist_transform(sel,tf,op,amp);
                     for(int v=0;v<256;v++)ff[v]=unsel[v]+tf[v];
                     double e=entropy_from_hist(ff,n);if(e<le){le=e;ls=si;lop=op;la=amp;lti=ti;}}}}}
     #pragma omp critical
     if(le<be){be=le;bseed=ls;bop=lop;bamp=la;bthi=lti;}}
    r.entropy=be;r.p[0]=bseed;r.p[1]=bamp;r.p[2]=bop;r.p[3]=bthi;
    snprintf(r.name,sizeof(r.name),"PRNG-SELECT %s seed=%d a=%d",opname[bop],bseed,bamp);
    return r;
}

static SR search_prng_jump(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=9; r.entropy=base; r.overhead=PAT_OH[9];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    double be=base;int bseed=0,bstep=0,bop=0,bamp=0;
    #pragma omp parallel
    {double le=base;int ls=0,lst=0,lop=0,la=0;u8 *vm=malloc(n);
     if(vm){
     #pragma omp for schedule(dynamic,1)
     for(int seed=0;seed<512;seed++)for(int ms=1;ms<=16;ms++){
         memset(vm,0,n);uint32_t st=(uint32_t)seed;int pos=0;
         while(pos<n){vm[pos]=1;pos+=(int)(lcg_byte(&st)%ms)+1;}
         int vis[256]={0},unvis[256];
         for(int i=0;i<n;i++)if(vm[i])vis[blk[i]]++;
         for(int v=0;v<256;v++)unvis[v]=bfreq[v]-vis[v];
         for(int op=0;op<N_OPS;op++){OP_RANGE(op,alo,ahi)
             for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
                 int tf[256],ff[256];hist_transform(vis,tf,op,amp);
                 for(int v=0;v<256;v++)ff[v]=unvis[v]+tf[v];
                 double e=entropy_from_hist(ff,n);if(e<le){le=e;ls=seed;lst=ms;lop=op;la=amp;}}}}
     free(vm);}
     #pragma omp critical
     if(le<be){be=le;bseed=ls;bstep=lst;bop=lop;bamp=la;}}
    r.entropy=be;r.p[0]=bseed;r.p[1]=bstep;r.p[2]=bop;r.p[3]=bamp;
    snprintf(r.name,sizeof(r.name),"PRNG-JUMP %s s=%d ms=%d a=%d",opname[bop],bseed,bstep,bamp);
    return r;
}

static SR search_modular_mask(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=10; r.entropy=base; r.overhead=PAT_OH[10];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    double be=base;int bP=0,bmask=0,bop=0,bamp=0;
    #pragma omp parallel
    {double le=base;int lP=0,lm=0,lop=0,la=0;
     #pragma omp for schedule(dynamic,1)
     for(int P=2;P<=8;P++){
         int sf[8][256];memset(sf,0,sizeof(sf));
         for(int i=0;i<n;i++)sf[i%P][blk[i]]++;
         int fm=(1<<P)-1;
         for(int M=1;M<fm;M++){
             int sel[256]={0},unsel[256]={0};
             for(int q=0;q<P;q++){int*dst=(M>>(q))&1?sel:unsel;for(int v=0;v<256;v++)dst[v]+=sf[q][v];}
             for(int op=0;op<N_OPS;op++){OP_RANGE(op,alo,ahi)
                 for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
                     int tf[256],ff[256];hist_transform(sel,tf,op,amp);
                     for(int v=0;v<256;v++)ff[v]=unsel[v]+tf[v];
                     double e=entropy_from_hist(ff,n);if(e<le){le=e;lP=P;lm=M;lop=op;la=amp;}}}}}
     #pragma omp critical
     if(le<be){be=le;bP=lP;bmask=lm;bop=lop;bamp=la;}}
    r.entropy=be;r.p[0]=bP;r.p[1]=bmask;r.p[2]=bop;r.p[3]=bamp;
    snprintf(r.name,sizeof(r.name),"MOD-MASK %s P=%d m=0x%02X a=%d",opname[bop],bP,bmask,bamp);
    return r;
}

// Shared greedy search core used by all COND-* patterns.
// gh[g][0..255] = input histogram for group g.  N = number of groups.
// unchanged_val = byte value of the one byte that has no context (counted outside groups).
static double cond_greedy(const int (*gh)[256],int N,int n,int unchanged_val,
                          int op,u8 *amps_out){
    OP_RANGE(op,alo,ahi)
    u8 amps[256]; for(int g=0;g<N;g++) amps[g]=(u8)alo;
    int combined[256]={0}; combined[unchanged_val]++;
    for(int g=0;g<N;g++){int tf[256];hist_transform(gh[g],tf,op,amps[g]);for(int v=0;v<256;v++)combined[v]+=tf[v];}
    for(int pass=0;pass<2;pass++) for(int g=0;g<N;g++){
        int tc[256],wk[256]; hist_transform(gh[g],tc,op,amps[g]);
        for(int v=0;v<256;v++) wk[v]=combined[v]-tc[v];
        double bge=1e30; u8 bga=amps[g];
        for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
            int tf[256],ff[256]; hist_transform(gh[g],tf,op,amp);
            for(int v=0;v<256;v++) ff[v]=wk[v]+tf[v];
            double e=entropy_from_hist(ff,n); if(e<bge){bge=e;bga=(u8)amp;}}
        int tn[256]; hist_transform(gh[g],tn,op,bga);
        for(int v=0;v<256;v++) combined[v]=wk[v]+tn[v];
        amps[g]=bga;}
    for(int g=0;g<N;g++) amps_out[g]=amps[g];
    return entropy_from_hist(combined,n);
}

static SR search_cond_prev(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=12; r.entropy=base; r.overhead=PAT_OH[12];
    double be=base; int bk=1,bop=0; u8 bamps[256]={0};
    // k=5..8 have overhead 267..2059 bits — never profitable on 4KB blocks
    for(int k=1;k<=4;k++){
        int N=1<<k;
        int (*gh)[256]=malloc(N*256*sizeof(int)); if(!gh) continue;
        memset(gh,0,N*256*sizeof(int));
        for(int i=1;i<n;i++) gh[blk[i-1]>>(8-k)][blk[i]]++;
        for(int op=0;op<N_OPS;op++){
            u8 amps[256]={0};
            double e=cond_greedy(gh,N,n,blk[0],op,amps);
            if(e<be){be=e;bk=k;bop=op;memcpy(bamps,amps,N*sizeof(u8));}}
        free(gh);}
    r.entropy=be; r.p[0]=bk; r.p[1]=bop;
    {int N=1<<bk; memcpy(r.amps,bamps,N*sizeof(u8)); r.overhead=5+3+3+N*8;}
    snprintf(r.name,sizeof(r.name),"COND-PREV k=%d %s",bk,opname[bop]);
    return r;
}

// PRNG-LONIB-ADD: PRNG-gated low-nibble rotation (add amp mod 16).
// Inverse: same gate, low-nibble subtract amp.
static SR search_prng_lo_nib_add(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=36; r.entropy=base; r.overhead=PAT_OH[36];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    int tv[]={25,64,128,192,230};
    double be=base; int bseed=0,bti=0,bamp=0;
    #pragma omp parallel
    {double le=base; int ls=0,lti=0,la=0;
     #pragma omp for schedule(dynamic,1)
     for(int si=0;si<256;si++) for(int ti=0;ti<5;ti++){
         u8 thr=(u8)tv[ti]; uint32_t st=(uint32_t)si;
         int sel[256]={0};
         for(int i=0;i<n;i++) if(lcg_byte(&st)>=thr) sel[blk[i]]++;
         for(int amp=1;amp<=15;amp++){
             int ff[256]={0};
             for(int v=0;v<256;v++){
                 ff[v]+=bfreq[v]-sel[v];
                 ff[(v&0xF0)|((v+amp)&0xF)]+=sel[v];}
             double e=entropy_from_hist(ff,n); if(e<le){le=e;ls=si;lti=ti;la=amp;}}}
     #pragma omp critical
     if(le<be){be=le;bseed=ls;bti=lti;bamp=la;}}
    r.entropy=be; r.p[0]=bseed; r.p[1]=bti; r.p[2]=bamp;
    snprintf(r.name,sizeof(r.name),"PRNG-LONIB-ADD a=%d s=%d",bamp,bseed);
    return r;
}

// PRNG-HINIB-XOR: PRNG-gated high-nibble XOR. Self-inverse.
static SR search_prng_hi_nib_xor(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=39; r.entropy=base; r.overhead=PAT_OH[39];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    int tv[]={25,64,128,192,230};
    double be=base; int bseed=0,bti=0,bamp=0;
    #pragma omp parallel
    {double le=base; int ls=0,lti=0,la=0;
     #pragma omp for schedule(dynamic,1)
     for(int si=0;si<256;si++) for(int ti=0;ti<5;ti++){
         u8 thr=(u8)tv[ti]; uint32_t st=(uint32_t)si;
         int sel[256]={0};
         for(int i=0;i<n;i++) if(lcg_byte(&st)>=thr) sel[blk[i]]++;
         for(int amp=1;amp<=15;amp++){
             int ff[256]={0};
             for(int v=0;v<256;v++){
                 ff[v]+=bfreq[v]-sel[v];
                 ff[(v&0x0F)|(((v>>4)^amp)<<4)]+=sel[v];}
             double e=entropy_from_hist(ff,n); if(e<le){le=e;ls=si;lti=ti;la=amp;}}}
     #pragma omp critical
     if(le<be){be=le;bseed=ls;bti=lti;bamp=la;}}
    r.entropy=be; r.p[0]=bseed; r.p[1]=bti; r.p[2]=bamp;
    snprintf(r.name,sizeof(r.name),"PRNG-HINIB-XOR a=%d s=%d",bamp,bseed);
    return r;
}

// PRNG-LONIB-XOR: PRNG-gated low-nibble XOR. Self-inverse.
static SR search_prng_lo_nib_xor(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=40; r.entropy=base; r.overhead=PAT_OH[40];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    int tv[]={25,64,128,192,230};
    double be=base; int bseed=0,bti=0,bamp=0;
    #pragma omp parallel
    {double le=base; int ls=0,lti=0,la=0;
     #pragma omp for schedule(dynamic,1)
     for(int si=0;si<256;si++) for(int ti=0;ti<5;ti++){
         u8 thr=(u8)tv[ti]; uint32_t st=(uint32_t)si;
         int sel[256]={0};
         for(int i=0;i<n;i++) if(lcg_byte(&st)>=thr) sel[blk[i]]++;
         for(int amp=1;amp<=15;amp++){
             int ff[256]={0};
             for(int v=0;v<256;v++){
                 ff[v]+=bfreq[v]-sel[v];
                 ff[(v&0xF0)|((v^amp)&0x0F)]+=sel[v];}
             double e=entropy_from_hist(ff,n); if(e<le){le=e;ls=si;lti=ti;la=amp;}}}
     #pragma omp critical
     if(le<be){be=le;bseed=ls;bti=lti;bamp=la;}}
    r.entropy=be; r.p[0]=bseed; r.p[1]=bti; r.p[2]=bamp;
    snprintf(r.name,sizeof(r.name),"PRNG-LONIB-XOR a=%d s=%d",bamp,bseed);
    return r;
}

// PRNG-CHUNK-COMP: PRNG-gated complement per chunk; PRNG resets each chunk. Self-inverse.
static SR search_prng_chunk_comp(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=95; r.entropy=base; r.overhead=PAT_OH[95];
    int tv[]={25,64,128,192,230};
    double be=base; int bC=2,bseed=0,bti=0;
    #pragma omp parallel
    {double le=base; int lC=2,lseed=0,lti=0;
     #pragma omp for schedule(dynamic,1)
     for(int C=2;C<=MAX_STRIDE;C++) for(int si=0;si<256;si++) for(int ti=0;ti<5;ti++){
         u8 thr=(u8)tv[ti]; u8 seed=(u8)si;
         int ff[256]={0};
         for(int base_=0;base_<n;base_+=C){
             uint32_t st=(uint32_t)seed;
             for(int k=0;k<C&&base_+k<n;k++)
                 ff[lcg_byte(&st)>=thr?(u8)(255-blk[base_+k]):blk[base_+k]]++;}
         double e=entropy_from_hist(ff,n); if(e<le){le=e;lC=C;lseed=si;lti=ti;}}
     #pragma omp critical
     if(le<be){be=le;bC=lC;bseed=lseed;bti=lti;}}
    r.entropy=be; r.p[0]=bC; r.p[1]=bseed; r.p[2]=bti;
    snprintf(r.name,sizeof(r.name),"PRNG-CHUNK-COMP C=%d s=%d",bC,bseed);
    return r;
}

// STRIDE-GF-RAMP: at stride positions, multiply by successive powers of alpha.
// k-th stride position gets alpha^k. Inverse: multiply by (inv_alpha)^k.
static SR search_stride_gf_ramp(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=98; r.entropy=base; r.overhead=PAT_OH[98];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    double be=base; int bs=1,balpha=2;
    #pragma omp parallel
    {double le=base; int ls=1,la=2;
     #pragma omp for schedule(dynamic,1)
     for(int s=1;s<=MAX_STRIDE;s++) for(int alpha=2;alpha<=255;alpha++){
         int ff[256]; memcpy(ff,bfreq,sizeof(ff));
         u8 acc=1;
         for(int p=0;p<n;p+=s){ff[blk[p]]--;ff[gf_mul(blk[p],acc)]++;acc=gf_mul(acc,(u8)alpha);}
         double e=entropy_from_hist(ff,n); if(e<le){le=e;ls=s;la=alpha;}}
     #pragma omp critical
     if(le<be){be=le;bs=ls;balpha=la;}}
    r.entropy=be; r.p[0]=bs; r.p[1]=balpha;
    snprintf(r.name,sizeof(r.name),"STRIDE-GF-RAMP s=%d a=%d",bs,balpha);
    return r;
}

// COND-NEXT: amp keyed by high k bits of the NEXT byte (reads original, left-to-right scan).
// Inverse: same context (next byte already decoded, going right-to-left) with inverse op.
static SR search_cond_next(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=13; r.entropy=base; r.overhead=PAT_OH[13];
    double be=base; int bk=1,bop=0; u8 bamps[256]={0};
    for(int k=1;k<=4;k++){
        int N=1<<k;
        int (*gh)[256]=malloc(N*256*sizeof(int)); if(!gh) continue;
        memset(gh,0,N*256*sizeof(int));
        for(int i=0;i<n-1;i++) gh[blk[i+1]>>(8-k)][blk[i]]++;
        for(int op=0;op<N_OPS;op++){
            u8 amps[256]={0};
            double e=cond_greedy(gh,N,n,blk[n-1],op,amps);
            if(e<be){be=e;bk=k;bop=op;memcpy(bamps,amps,N*sizeof(u8));}}
        free(gh);}
    r.entropy=be; r.p[0]=bk; r.p[1]=bop;
    {int N=1<<bk; memcpy(r.amps,bamps,N*sizeof(u8)); r.overhead=5+3+3+N*8;}
    snprintf(r.name,sizeof(r.name),"COND-NEXT k=%d %s",bk,opname[bop]);
    return r;
}

// COND-DELTA: amp keyed by high k bits of (b[i-1] XOR b[i-2]) — local difference as context.
// Inverse: same context (b[i-1], b[i-2] already decoded) with inverse op, left-to-right.
static SR search_cond_delta(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=14; r.entropy=base; r.overhead=PAT_OH[14];
    double be=base; int bk=1,bop=0; u8 bamps[256]={0};
    for(int k=1;k<=4;k++){
        int N=1<<k;
        int (*gh)[256]=malloc(N*256*sizeof(int)); if(!gh) continue;
        memset(gh,0,N*256*sizeof(int));
        for(int i=2;i<n;i++) gh[(blk[i-1]^blk[i-2])>>(8-k)][blk[i]]++;
        for(int op=0;op<N_OPS;op++){
            u8 amps[256]={0};
            double e=cond_greedy(gh,N,n,blk[0],op,amps);
            if(e<be){be=e;bk=k;bop=op;memcpy(bamps,amps,N*sizeof(u8));}}
        free(gh);}
    r.entropy=be; r.p[0]=bk; r.p[1]=bop;
    {int N=1<<bk; memcpy(r.amps,bamps,N*sizeof(u8)); r.overhead=5+3+3+N*8;}
    snprintf(r.name,sizeof(r.name),"COND-DELTA k=%d %s",bk,opname[bop]);
    return r;
}

// STRIDED-GRAY: apply Gray encode(dir=0) or Gray decode(dir=1) at stride positions.
// Inverse: apply opposite direction at same positions.
static SR search_strided_gray(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=73; r.entropy=base; r.overhead=PAT_OH[73];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    u8 genc[256],gdec[256];
    for(int v=0;v<256;v++) genc[v]=(u8)(v^(v>>1));
    for(int v=0;v<256;v++){u8 x=(u8)v;x^=(x>>4);x^=(x>>2);x^=(x>>1);gdec[v]=x;}
    double be=base; int bs=1,bdir=0;
    for(int s=1;s<=MAX_STRIDE;s++) for(int dir=0;dir<2;dir++){
        u8 *tab=dir?gdec:genc;
        int sf[256]={0}; for(int p=0;p<n;p+=s) sf[blk[p]]++;
        int ff[256]={0};
        for(int v=0;v<256;v++){ff[v]+=bfreq[v]-sf[v]; ff[tab[v]]+=sf[v];}
        double e=entropy_from_hist(ff,n); if(e<be){be=e;bs=s;bdir=dir;}}
    r.entropy=be; r.p[0]=bs; r.p[1]=bdir;
    snprintf(r.name,sizeof(r.name),"STRIDED-GRAY s=%d %s",bs,bdir?"dec":"enc");
    return r;
}

// PRNG-STRD-COMP: PRNG-gated complement at stride-selected positions. Self-inverse.
static SR search_prng_stride_comp(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=76; r.entropy=base; r.overhead=PAT_OH[76];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    int tv[]={25,64,128,192,230};
    double be=base; int bs=1,bseed=0,bti=0;
    #pragma omp parallel
    {double le=base; int ls_=1,lseed=0,lti=0;
     #pragma omp for schedule(dynamic,1)
     for(int s=1;s<=MAX_STRIDE;s++) for(int si=0;si<256;si++) for(int ti=0;ti<5;ti++){
         u8 thr=(u8)tv[ti]; uint32_t st=(uint32_t)si;
         int sel[256]={0};
         for(int p=0;p<n;p+=s) if(lcg_byte(&st)>=thr) sel[blk[p]]++;
         int ff[256]={0};
         for(int v=0;v<256;v++){ff[v]+=bfreq[v]-sel[v]; ff[255-v]+=sel[v];}
         double e=entropy_from_hist(ff,n); if(e<le){le=e;ls_=s;lseed=si;lti=ti;}}
     #pragma omp critical
     if(le<be){be=le;bs=ls_;bseed=lseed;bti=lti;}}
    r.entropy=be; r.p[0]=bs; r.p[1]=bseed; r.p[2]=bti;
    snprintf(r.name,sizeof(r.name),"PRNG-STRD-COMP s=%d seed=%d",bs,bseed);
    return r;
}

// STRIDE-NIBSWAP: swap high and low nibbles at stride-selected positions. Self-inverse.
static SR search_stride_nibswap(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=62; r.entropy=base; r.overhead=PAT_OH[62];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    double be=base; int bs=1;
    for(int s=1;s<=MAX_STRIDE;s++){
        int sf[256]={0}; for(int p=0;p<n;p+=s) sf[blk[p]]++;
        int ff[256]={0};
        for(int v=0;v<256;v++){ff[v]+=bfreq[v]-sf[v]; ff[(u8)((v>>4)|(v<<4))]+=sf[v];}
        double e=entropy_from_hist(ff,n); if(e<be){be=e;bs=s;}}
    r.entropy=be; r.p[0]=bs;
    snprintf(r.name,sizeof(r.name),"STRIDE-NIBSWAP s=%d",bs);
    return r;
}

// BIT-ROT-STRIDE: fixed bit-rotation k (1..7) at stride-selected positions.
// Inverse: rotate by (8-k) at same positions.
static SR search_bit_rot_stride(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=68; r.entropy=base; r.overhead=PAT_OH[68];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    double be=base; int bs=1,bk=1;
    for(int s=1;s<=MAX_STRIDE;s++){
        int sf[256]={0}; for(int p=0;p<n;p+=s) sf[blk[p]]++;
        for(int k=1;k<=7;k++){
            int ff[256]={0};
            for(int v=0;v<256;v++){ff[v]+=bfreq[v]-sf[v];
                ff[(u8)((v<<k)|(v>>(8-k)))]+=sf[v];}
            double e=entropy_from_hist(ff,n); if(e<be){be=e;bs=s;bk=k;}}}
    r.entropy=be; r.p[0]=bs; r.p[1]=bk;
    snprintf(r.name,sizeof(r.name),"BIT-ROT-STRIDE s=%d k=%d",bs,bk);
    return r;
}

// PRNG-STRD-BITROT: per-position PRNG-determined bit rotation at stride positions.
// Inverse: replay same PRNG, rotate each position right by same k.
static SR search_prng_stride_bitrot(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=71; r.entropy=base; r.overhead=PAT_OH[71];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    double be=base; int bs=1,bseed=0;
    #pragma omp parallel
    {double le=base; int ls_=1,lseed=0;
     #pragma omp for schedule(dynamic,1)
     for(int s=1;s<=MAX_STRIDE;s++) for(int seed=0;seed<256;seed++){
         int ff[256]; memcpy(ff,bfreq,sizeof(ff));
         uint32_t st=(uint32_t)seed;
         for(int p=0;p<n;p+=s){int k=(int)(lcg_byte(&st)%7)+1;
             ff[blk[p]]--; ff[(u8)((blk[p]<<k)|(blk[p]>>(8-k)))]++;}
         double e=entropy_from_hist(ff,n); if(e<le){le=e;ls_=s;lseed=seed;}}
     #pragma omp critical
     if(le<be){be=le;bs=ls_;bseed=lseed;}}
    r.entropy=be; r.p[0]=bs; r.p[1]=bseed;
    snprintf(r.name,sizeof(r.name),"PRNG-STRD-BITROT s=%d seed=%d",bs,bseed);
    return r;
}

// PRNG-STRD-XDELTA: PRNG-gated XOR stride delta. Self-inverse (XOR, same gate).
static SR search_prng_stride_xor_delta(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=55; r.entropy=base; r.overhead=PAT_OH[55];
    int tv[]={25,64,128,192,230};
    double be=base; int bN=1,bseed=0,bti=0;
    #pragma omp parallel
    {double le=base; int lN=1,ls=0,lti=0;
     u8 *tmp=malloc(n); u8 *gate=malloc(n);
     if(tmp&&gate){
     #pragma omp for schedule(dynamic,1)
     for(int N=1;N<=MAX_STRIDE;N++) for(int si=0;si<256;si++) for(int ti=0;ti<5;ti++){
         u8 thr=(u8)tv[ti]; uint32_t st=(uint32_t)si;
         for(int i=0;i<n;i++) gate[i]=(lcg_byte(&st)>=thr)?1:0;
         memcpy(tmp,blk,n);
         for(int i=n-1;i>=N;i--) if(gate[i]) tmp[i]^=tmp[i-N];
         int ff[256]={0}; for(int i=0;i<n;i++) ff[tmp[i]]++;
         double e=entropy_from_hist(ff,n); if(e<le){le=e;lN=N;ls=si;lti=ti;}}}
     free(tmp); free(gate);
     #pragma omp critical
     if(le<be){be=le;bN=lN;bseed=ls;bti=lti;}}
    r.entropy=be; r.p[0]=bN; r.p[1]=bseed; r.p[2]=bti;
    snprintf(r.name,sizeof(r.name),"PRNG-STRD-XDELTA N=%d s=%d",bN,bseed);
    return r;
}

// STRIDE-NEG: complement (255-v) at stride positions. Self-inverse.
static SR search_stride_neg(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=57; r.entropy=base; r.overhead=PAT_OH[57];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    double be=base; int bs=1;
    for(int s=1;s<=MAX_STRIDE;s++){
        int sf[256]={0}; for(int p=0;p<n;p+=s) sf[blk[p]]++;
        int ff[256]={0};
        for(int v=0;v<256;v++){ff[v]+=bfreq[v]-sf[v]; ff[255-v]+=sf[v];}
        double e=entropy_from_hist(ff,n); if(e<be){be=e;bs=s;}}
    r.entropy=be; r.p[0]=bs;
    snprintf(r.name,sizeof(r.name),"STRIDE-NEG s=%d",bs);
    return r;
}

// GF-STRIDE-FIXED: multiply stride-selected bytes by fixed GF256 coefficient.
// Inverse: multiply by GF256 inverse of alpha.
static SR search_gf_stride_fixed(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=58; r.entropy=base; r.overhead=PAT_OH[58];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    double be=base; int bs=1,balpha=2;
    #pragma omp parallel
    {double le=base; int ls=1,la=2;
     #pragma omp for schedule(dynamic,1)
     for(int s=1;s<=MAX_STRIDE;s++) for(int alpha=2;alpha<=255;alpha++){
         int sf[256]={0}; for(int p=0;p<n;p+=s) sf[blk[p]]++;
         int ff[256]={0};
         for(int v=0;v<256;v++){ff[v]+=bfreq[v]-sf[v]; ff[gf_mul((u8)v,(u8)alpha)]+=sf[v];}
         double e=entropy_from_hist(ff,n); if(e<le){le=e;ls=s;la=alpha;}}
     #pragma omp critical
     if(le<be){be=le;bs=ls;balpha=la;}}
    r.entropy=be; r.p[0]=bs; r.p[1]=balpha;
    snprintf(r.name,sizeof(r.name),"GF-STRIDE-FIXED s=%d a=%d",bs,balpha);
    return r;
}

// PRNG-STRD-DELTA: apply stride-N delta only at PRNG-selected positions.
// Gate generated left-to-right; delta applied right-to-left (b[i-N] always original).
static SR search_prng_stride_delta(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=42; r.entropy=base; r.overhead=PAT_OH[42];
    int tv[]={25,64,128,192,230};
    double be=base; int bN=1,bseed=0,bti=0;
    #pragma omp parallel
    {double le=base; int lN=1,ls=0,lti=0;
     u8 *tmp=malloc(n); u8 *gate=malloc(n);
     if(tmp&&gate){
     #pragma omp for schedule(dynamic,1)
     for(int N=1;N<=MAX_STRIDE;N++) for(int si=0;si<256;si++) for(int ti=0;ti<5;ti++){
         u8 thr=(u8)tv[ti]; uint32_t st=(uint32_t)si;
         for(int i=0;i<n;i++) gate[i]=(lcg_byte(&st)>=thr)?1:0;
         memcpy(tmp,blk,n);
         for(int i=n-1;i>=N;i--) if(gate[i]) tmp[i]=(u8)(tmp[i]-tmp[i-N]);
         int ff[256]={0}; for(int i=0;i<n;i++) ff[tmp[i]]++;
         double e=entropy_from_hist(ff,n); if(e<le){le=e;lN=N;ls=si;lti=ti;}}}
     free(tmp); free(gate);
     #pragma omp critical
     if(le<be){be=le;bN=lN;bseed=ls;bti=lti;}}
    r.entropy=be; r.p[0]=bN; r.p[1]=bseed; r.p[2]=bti;
    snprintf(r.name,sizeof(r.name),"PRNG-STRD-DELTA N=%d s=%d",bN,bseed);
    return r;
}

// PRNG-DUAL-STRIDE: dual sub-stride at period 2s, each element gets its own PRNG amp.
// Even sub-stride positions use LCG(seed); odd positions use LCG(seed^255).
// Inverse: replay same PRNGs with inverse op.
static SR search_prng_dual_stride(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=43; r.entropy=base; r.overhead=PAT_OH[43];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    double be=base; int bs=1,bseed=0,bop=0;
    #pragma omp parallel
    {double le=base; int ls_=1,lseed=0,lop=0;
     #pragma omp for schedule(dynamic,1)
     for(int s=1;s<=MAX_STRIDE;s++) for(int seed=0;seed<256;seed++) for(int op=0;op<N_OPS;op++){
         int ff[256]; memcpy(ff,bfreq,sizeof(ff));
         uint32_t st_e=(uint32_t)seed, st_o=(uint32_t)(seed^255);
         for(int p=0;p<n;p+=2*s){ff[blk[p]]--; ff[op_byte(blk[p],lcg_byte(&st_e),op)]++;}
         for(int p=s;p<n;p+=2*s){ff[blk[p]]--; ff[op_byte(blk[p],lcg_byte(&st_o),op)]++;}
         double e=entropy_from_hist(ff,n); if(e<le){le=e;ls_=s;lseed=seed;lop=op;}}
     #pragma omp critical
     if(le<be){be=le;bs=ls_;bseed=lseed;bop=lop;}}
    r.entropy=be; r.p[0]=bs; r.p[1]=bseed; r.p[2]=bop;
    snprintf(r.name,sizeof(r.name),"PRNG-DUAL-STRIDE s=%d %s seed=%d",bs,opname[bop],bseed);
    return r;
}

// PRNG-GFMUL-STRD: at stride positions, multiply by a PRNG-derived GF256 coefficient (>=2).
// Inverse: replay same PRNG, multiply by GF256 inverse of each coefficient.
static SR search_prng_gfmul_stride(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=51; r.entropy=base; r.overhead=PAT_OH[51];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    double be=base; int bs=1,bseed=0;
    #pragma omp parallel
    {double le=base; int ls_=1,lseed=0;
     #pragma omp for schedule(dynamic,1)
     for(int s=1;s<=MAX_STRIDE;s++) for(int seed=0;seed<256;seed++){
         int ff[256]; memcpy(ff,bfreq,sizeof(ff));
         uint32_t st=(uint32_t)seed;
         for(int p=0;p<n;p+=s){u8 alpha=lcg_byte(&st); if(alpha<2)alpha=2;
             ff[blk[p]]--; ff[gf_mul(blk[p],alpha)]++;}
         double e=entropy_from_hist(ff,n); if(e<le){le=e;ls_=s;lseed=seed;}}
     #pragma omp critical
     if(le<be){be=le;bs=ls_;bseed=lseed;}}
    r.entropy=be; r.p[0]=bs; r.p[1]=bseed;
    snprintf(r.name,sizeof(r.name),"PRNG-GFMUL-STRD s=%d seed=%d",bs,bseed);
    return r;
}

// ── Run all searches ──────────────────────────────────────────────────────────
static void run_all(const u8 *blk, int n, double base, SR *out) {
    out[0]  = search_stride_prng_amp     (blk,n,base);
    out[1]  = search_prng_select         (blk,n,base);
    out[2]  = search_prng_jump           (blk,n,base);
    out[3]  = search_modular_mask        (blk,n,base);
    out[4]  = search_cond_prev           (blk,n,base);
    out[5]  = search_prng_lo_nib_add     (blk,n,base);
    out[6]  = search_prng_hi_nib_xor     (blk,n,base);
    out[7]  = search_prng_lo_nib_xor     (blk,n,base);
    out[8]  = search_prng_stride_delta   (blk,n,base);
    out[9]  = search_prng_dual_stride    (blk,n,base);
    out[10] = search_prng_gfmul_stride   (blk,n,base);
    out[11] = search_prng_stride_xor_delta(blk,n,base);
    out[12] = search_stride_neg          (blk,n,base);
    out[13] = search_gf_stride_fixed     (blk,n,base);
    out[14] = search_stride_nibswap      (blk,n,base);
    out[15] = search_bit_rot_stride      (blk,n,base);
    out[16] = search_prng_stride_bitrot  (blk,n,base);
    out[17] = search_cond_next           (blk,n,base);
    out[18] = search_cond_delta          (blk,n,base);
    out[19] = search_strided_gray        (blk,n,base);
    out[20] = search_prng_stride_comp    (blk,n,base);
    out[21] = search_prng_chunk_comp     (blk,n,base);
    out[22] = search_stride_gf_ramp      (blk,n,base);
}


// ── Per-operation statistics ───────────────────────────────────────────────────
typedef struct {
    char   name[32];   // base op name (no params), fixed at startup
    long   count;      // total times this slot won across all blocks
    double sum_net;
    double min_net;
    double max_net;
} OpStat;

// ── Main ──────────────────────────────────────────────────────────────────────
#define N_STATS 50   // number of random 4KB blocks to run

int main(void) {
    init_gf256();
    int nt=omp_get_max_threads()-1; if(nt<1)nt=1;
    omp_set_num_threads(nt);

    // Seed stat names with one dummy block; strip params so the base name is stable.
    OpStat stats[N_PATTERNS];
    memset(stats,0,sizeof(stats));
    for(int i=0;i<N_PATTERNS;i++) stats[i].min_net=1e30;
    {
        u8 *dummy=malloc(BLOCK_SIZE);
        if(dummy&&fill_random(dummy,BLOCK_SIZE)){
            SR tmp[N_PATTERNS];
            run_all(dummy,BLOCK_SIZE,byte_entropy(dummy,BLOCK_SIZE),tmp);
            for(int i=0;i<N_PATTERNS;i++){
                strncpy(stats[i].name,tmp[i].name,31); stats[i].name[31]=0;
                char *sp=strchr(stats[i].name,' '); if(sp)*sp=0; // keep first word only
            }
        }
        free(dummy);
    }

    printf("Running %d blocks x %d bytes  (threads=%d)\n\n",
           N_STATS, BLOCK_SIZE, nt);

    for(int bi=0;bi<N_STATS;bi++){
        u8 *blk=malloc(BLOCK_SIZE);
        if(!blk||!fill_random(blk,BLOCK_SIZE)){free(blk);continue;}

        double base=byte_entropy(blk,BLOCK_SIZE);
        double start_entropy=base, total_net=0.0;
        SR results[N_PATTERNS];
        int passes_taken=0;

        while(1){
            run_all(blk,BLOCK_SIZE,base,results);

            int best_idx=-1; double best_net=-1e30;
            for(int i=0;i<N_PATTERNS;i++){
                double net=(base-results[i].entfropy)*BLOCK_SIZE-results[i].overhead;
                if(net>best_net){best_net=net;best_idx=i;}
            }
            if(best_net<=0||best_idx<0) break;

            // Accumulate stats — name stays as the clean base name set above
            OpStat *s=&stats[best_idx];
            s->count++;
            s->sum_net+=best_net;
            if(best_net<s->min_net) s->min_net=best_net;
            if(best_net>s->max_net) s->max_net=best_net;

            apply_sr(blk,BLOCK_SIZE,&results[best_idx]);
            base=byte_entropy(blk,BLOCK_SIZE);
            total_net+=best_net;
            passes_taken++;
        }

        // One line per block — just progress, not per-pass noise
        printf("Block %02d: %d pass(es)  H %.4f -> %.4f  net=%+.0f bits\n",
               bi+1, passes_taken, start_entropy, base, total_net);
        fflush(stdout);
        free(blk);
    }

    // ── Count table ───────────────────────────────────────────────────────────
    // Sort by count descending so useful ops bubble to the top, zeros sink to bottom
    int order[N_PATTERNS];
    for(int i=0;i<N_PATTERNS;i++) order[i]=i;
    for(int i=1;i<N_PATTERNS;i++){          // insertion sort
        int k=order[i]; int j=i-1;
        while(j>=0&&stats[order[j]].count<stats[k].count){order[j+1]=order[j];j--;}
        order[j+1]=k;
    }

    // Count how many were never triggered
    int n_zero=0;
    for(int i=0;i<N_PATTERNS;i++) if(stats[i].count==0) n_zero++;

    printf("\n=== OPERATION USE COUNT (%d blocks) ===\n", N_STATS);
    printf("  %6s  %8s  %8s  %8s  %-28s\n",
           "Count", "Avg Net", "Min Net", "Max Net", "Operation");
    printf("  %6s  %8s  %8s  %8s  %-28s\n",
           "------","--------","--------","--------","--------");

    for(int i=0;i<N_PATTERNS;i++){
        int ii=order[i];
        if(stats[ii].count==0) break;   // zeros sorted to bottom — stop printing
        double avg=stats[ii].sum_net/stats[ii].count;
        printf("  %6ld  %8.1f  %8.1f  %8.1f  %s\n",
               stats[ii].count, avg, stats[ii].min_net, stats[ii].max_net,
               stats[ii].name);
    }

    printf("\n--- NEVER TRIGGERED (%d / %d) — candidates to remove: ---\n",
           n_zero, N_PATTERNS);
    for(int i=0;i<N_PATTERNS;i++){
        int ii=order[i];
        if(stats[ii].count>0) continue;
        printf("  slot%03d  %s\n", ii, stats[ii].name);
    }

    return 0;
}
