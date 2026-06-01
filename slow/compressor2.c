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
#define N_OPS       16
#define INPUT_FILE   "C:/Users/lukac/Documents/compressor/compressor.c"
#define OUTPUT_FILE  INPUT_FILE ".cmp2"
#define MAX_TRANSFORMS 16
#define N_PATTERNS  3

// â"€â"€ GF(256) â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
static u8 gf_exp[512], gf_log[256];
static void init_gf256(void) {
    u8 x=1;
    for(int i=0;i<255;i++){gf_exp[i]=x;gf_log[x]=(u8)i;u8 h=(x&0x80)?((x<<1)^0x1B):(x<<1);x=h^x;}
    for(int i=255;i<512;i++) gf_exp[i]=gf_exp[i-255];
    gf_log[0]=0;
}
static inline u8 gf_mul(u8 a,u8 b){return(!a||!b)?0:gf_exp[gf_log[a]+gf_log[b]];}

// â"€â"€ Core helpers â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
static inline u8 lcg_byte(uint32_t *s){*s=*s*1664525u+1013904223u;return(u8)(*s>>24);}
static inline u8 op_byte(u8 v,u8 amp,int op){
    switch(op){
        case 0:return(u8)(v+amp);
        case 1:return(u8)(v^amp);
        case 2:return(u8)(v*(amp|1));
        case 3:return(u8)((v&0xF0)|((v+amp)&0x0F));
        case 4:{u8 s=(u8)((v<<4)|(v>>4));return s^amp;}
        case 5:{u8 a=amp<2?2:amp;return gf_mul(v,a);}
        case 6:{u8 a=amp&7;if(!a)a=1;return(u8)((v<<a)|(v>>(8-a)));}  // ROL
        case 7:return(u8)((((v>>4)+amp)&0xF)<<4|(v&0x0F));             // ADDHI
        case 8:{u8 w=(u8)(v+amp);return w^(w>>1);}                     // GRAY: Gray(v+amp), decode=gray_inv(r)-amp
        case 9: return(u8)(((((v>>4)^(v&0xF)^(amp&0xF))&0xF)<<4)|(v&0x0F));// XORNIBBLE
        case 10:return(u8)((v&0xF0)|((amp-v)&0x0F));                          // SUBLO
        case 11:return(u8)((((amp-(v>>4))&0x0F)<<4)|(v&0x0F));               // SUBHI
        case 12:return(u8)(amp-v);                                             // NEGADD: amp-v mod 256
        case 13:return(u8)((v&0xF0)|((v^amp)&0x0F));                         // XORLO
        case 14:return(u8)((v&0x0F)|((((v>>4)^amp)&0x0F)<<4));              // XORHI
        case 15:{u8 a=(u8)(amp&7);if(!a)a=1;return(u8)((v>>a)|(v<<(8-a)));} // ROR
    }return v;
}
static double entropy_from_hist(const int f[256],int n){
    double e=0.0;for(int i=0;i<256;i++){if(!f[i])continue;double p=(double)f[i]/n;e-=p*log2(p);}return e;
}
// Fast entropy using a precomputed -p*log2(p) table indexed by count.
// Call init_entropy_table(BLOCK_SIZE) once before use.
static float g_ent_tab[BLOCK_SIZE+1];
static void init_entropy_table(int n){
    g_ent_tab[0]=0.0f;
    for(int c=1;c<=n;c++){double p=(double)c/n;g_ent_tab[c]=(float)(-p*log2(p));}
}
static inline double entropy_fast(const int f[256]){
    double e=0.0;for(int i=0;i<256;i++)e+=g_ent_tab[f[i]];return e;
}
static double byte_entropy(const u8*d,int n){int f[256]={0};for(int i=0;i<n;i++)f[d[i]]++;return entropy_from_hist(f,n);}
static int fill_random(u8*buf,int n){return BCryptGenRandom(NULL,(PUCHAR)buf,(ULONG)n,BCRYPT_USE_SYSTEM_PREFERRED_RNG)==0;}
static void hist_transform(const int in[256],int out[256],int op,int amp){
    memset(out,0,256*sizeof(int));for(int v=0;v<256;v++)if(in[v])out[op_byte((u8)v,(u8)amp,op)]+=in[v];
}
// ops: 0=ADD  1=XOR  2=MUL  3=ADDLO  4=SWXOR  5=GFMUL  6=ROL  7=ADDHI  8=GRAY  9=XORNIBBLE
//      10=SUBLO  11=SUBHI  12=NEGADD  13=XORLO  14=XORHI  15=ROR
// Removed from search (never chosen on test data): ADDHI(7) XORLO(13) XORHI(14) ROR(15)
static const int SEARCH_OPS[]  = {0,1,2,3,4,5,6,8,9,10,11,12};
#define N_SEARCH_OPS 12


// â"€â"€ Search result struct â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
typedef struct {
    int    id;
    char   name[48];
    double entropy;
    int    overhead;
    int    p[8];
    u8     amps[256];     // per-group amplitude (2^k entries used)
    u8     grp_ops[256];  // per-group op (2^k entries used)
} SR;

// Overhead bits = 5 (ID) + parameter bits.  -F suffix = per-group op mode.
// Phase 1 (always applied first):
// 12  COND-PREV:       k(3) + 2^k*(op(3)+amp(8))       = 30..184 bits
// 13  COND-NEXT:       k(3) + 2^k*(op(3)+amp(8))       = 30..184 bits
// 14  COND-DELTA:      k(3) + 2^k*(op(3)+amp(8))       = 30..184 bits  context=b[i-1]^b[i-2]
// Phase 2 (refinement on top of phase 1 result, looped until no gain):
// Phase 1: 2 bits (which of 3 transforms) + per-group params. No shared ID field needed for Phase 2.
// 25  PRNG-GATE-PRNG-AMP:   seed(16)+thr(5)+op(4)                                    = 25 bits
// 26  PRNG-GATE-FIXED-AMP:  seed(10)+thr(5)+op(4)+amp(8)                             = 27 bits
// 27  BIT-ROTATE:           rotation_amount(15, 0..32767 bit positions)              = 15 bits
static const int PAT_OH[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,  // 0-11 unused
    30, 30, 30,                 // 12 COND-PREV, 13 COND-NEXT, 14 COND-DELTA
    30, 45, 0, 0, 0, 0, 0, 0, 0,  // 15 COND-POS, 16 COND-PREV2, 17-24 unused (8 entries)
    25,                          // 25 PRNG-GATE-PRNG-AMP
    27,                          // 26 PRNG-GATE-FIXED-AMP
    15,                          // 27 BIT-ROTATE
};

// Map a PRNG byte to a valid (bijective) amplitude for the given op.
static inline u8 prng_amp(u8 r, int op) {
    switch(op){
        case 2: {u8 v=(u8)(r|1);return v<3?(u8)3:v;} // MUL: odd 3..255 (matches OP_RANGE alo=3)
        case 3: return (u8)(r&0x0F);         // ADDLO: 0..15
        case 5: return r<2?(u8)2:r;          // GFMUL: 2..255
        case 6: return (u8)((r%7)+1);        // ROL: 1..7
        case 7: return (u8)(r&0x0F);         // ADDHI: 0..15
        case 9:  return (u8)(r&0x0F);        // XORNIBBLE: 0..15
        case 10: return (u8)(r&0x0F);        // SUBLO: 0..15
        case 11: return (u8)(r&0x0F);        // SUBHI: 0..15
        case 12: return r;                    // NEGADD: 0..255
        case 13: return (u8)(r&0x0F);        // XORLO: 0..15
        case 14: return (u8)(r&0x0F);        // XORHI: 0..15
        case 15: return (u8)((r%7)+1);       // ROR: 1..7
        default: return r;                    // ADD, XOR, SWXOR, GRAY: 0..255
    }
}

// â"€â"€ Apply a SearchResult in-place â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
static void apply_sr(u8 *blk, int n, const SR *r) {
    u8 *orig=NULL;
    if(r->id==12||r->id==14||r->id==16){  // COND-PREV, COND-DELTA, COND-PREV2 need orig snapshot
        orig=malloc(n); if(orig) memcpy(orig,blk,n);
    }
    switch(r->id){
        case 12: {int k=r->p[0],flex=r->p[2],op=r->p[1]; // COND-PREV
            if(orig) for(int i=1;i<n;i++){int g=orig[i-1]>>(8-k);
                blk[i]=op_byte(blk[i],r->amps[g],flex?r->grp_ops[g]:op);}} break;
        case 13: {int k=r->p[0],flex=r->p[2],op=r->p[1]; // COND-NEXT
            for(int i=0;i<n-1;i++){int g=blk[i+1]>>(8-k);
                blk[i]=op_byte(blk[i],r->amps[g],flex?r->grp_ops[g]:op);}} break;
        case 14: {int k=r->p[0],flex=r->p[2],op=r->p[1]; // COND-DELTA: context=b[i-1]^b[i-2]
            if(orig) for(int i=2;i<n;i++){int g=(orig[i-1]^orig[i-2])>>(8-k);
                blk[i]=op_byte(blk[i],r->amps[g],flex?r->grp_ops[g]:op);}} break;
        case 15: {int k=r->p[0],flex=r->p[2],op=r->p[1]; // COND-POS: context=i%(1<<k)
            for(int i=0;i<n;i++){int g=i%(1<<k);
                blk[i]=op_byte(blk[i],r->amps[g],flex?r->grp_ops[g]:op);}} break;
        case 16: {int j=r->p[0],l=r->p[1],flex=r->p[2],op=r->p[3]; // COND-PREV2
            if(orig) for(int i=2;i<n;i++){
                int g=((orig[i-1]>>(8-j))<<l)|(orig[i-2]>>(8-l));
                blk[i]=op_byte(blk[i],r->amps[g],flex?r->grp_ops[g]:op);}} break;
        case 25: { // PRNG-GATE-PRNG-AMP
            uint32_t st=(uint32_t)r->p[0]; u8 thr=(u8)r->p[1]; int op=r->p[2];
            for(int i=0;i<n;i++){u8 gate=lcg_byte(&st);u8 ra=lcg_byte(&st);
                if(gate>=thr) blk[i]=op_byte(blk[i],prng_amp(ra,op),op);}} break;
        case 26: { // PRNG-GATE-FIXED-AMP: one gate LCG per position, fixed op+amp
            uint32_t st=(uint32_t)r->p[0]; u8 thr=(u8)r->p[1];
            int op=r->p[2]; u8 amp=(u8)r->p[3];
            for(int i=0;i<n;i++){u8 gate=lcg_byte(&st);
                if(gate>=thr) blk[i]=op_byte(blk[i],amp,op);}} break;
        case 27: { // BIT-ROTATE: cyclic left-rotation of entire block by rot bits
            int rot=r->p[0]; if(!rot) break;
            int bs=rot/8, bp=rot%8;
            u8 *tmp=malloc(n); if(!tmp) break;
            int j=bs%n, j1=(bs+1)%n;
            for(int i=0;i<n;i++){
                u8 a=blk[j], b=blk[j1];
                tmp[i]=bp?(u8)((a<<bp)|(b>>(8-bp))):a;
                if(++j>=n)j=0; if(++j1>=n)j1=0;}
            memcpy(blk,tmp,n); free(tmp);} break;
    }
    free(orig);
}

// â"€â"€ Helpers for search functions â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
// MUL(2): odd 3..255; ADDLO(3)/ADDHI(7)/XORNIBBLE(9): amp 0..15; GFMUL(5): amp>=2; ROL(6): 1..7; GRAY(8): 0..255
#define OP_RANGE(op,alo,ahi) int alo=1,ahi=255; \
    if(op==2)alo=3; if(op==3)ahi=15; if(op==5)alo=2; if(op==6)ahi=7; if(op==7)ahi=15; \
    if(op==8)alo=0; if(op==9){alo=0;ahi=15;} \
    if(op==10){alo=0;ahi=15;} if(op==11){alo=0;ahi=15;} if(op==12)alo=0; \
    if(op==13)ahi=15; if(op==14)ahi=15; if(op==15)ahi=7;
#define SKIP_OP(op,amp) if(op==2&&(amp&1)==0) continue;

// â"€â"€ Search functions â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

// Shared-op greedy: all groups use the same op, 2 refinement passes.
// Overhead when using this: 5 + 3 + 3 + N*8 bits (k + op + N*amp).
static double cond_greedy(const int (*gh)[256], int N, int n, int unchanged_val,
                           int op, u8 *amps_out) {
    OP_RANGE(op,alo,ahi)
    u8 amps[256]; for(int g=0;g<N;g++) amps[g]=(u8)alo;
    int combined[256]={0}; if(unchanged_val>=0) combined[unchanged_val]++;
    for(int g=0;g<N;g++){int tf[256];hist_transform(gh[g],tf,op,amps[g]);for(int v=0;v<256;v++)combined[v]+=tf[v];}
    for(int pass=0;pass<2;pass++) for(int g=0;g<N;g++){
        int tc[256],wk[256]; hist_transform(gh[g],tc,op,amps[g]);
        for(int v=0;v<256;v++) wk[v]=combined[v]-tc[v];
        double bge=1e30; u8 bga=amps[g]; int best_tf[256]; memcpy(best_tf,tc,256*sizeof(int));
        for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
            int tf[256],ff[256]; hist_transform(gh[g],tf,op,amp);
            for(int v=0;v<256;v++) ff[v]=wk[v]+tf[v];
            double e=entropy_fast(ff); if(e<bge){bge=e;bga=(u8)amp;memcpy(best_tf,tf,256*sizeof(int));}}
        for(int v=0;v<256;v++) combined[v]=wk[v]+best_tf[v];
        amps[g]=bga;}
    for(int g=0;g<N;g++) amps_out[g]=amps[g];
    return entropy_fast(combined);
}

// Per-group-op greedy: each group picks its own (op, amp), 3 refinement passes.
// Overhead when using this: 5 + 3 + N*11 bits (k + N*(op+amp)).
// Strictly more general than cond_greedy â€" wins on structured data where groups differ.
static double cond_greedy_flex(const int (*gh)[256], int N, int n, int unchanged_val,
                                u8 *amps_out, u8 *ops_out) {
    u8 amps[256], ops[256];
    for(int g=0;g<N;g++){amps[g]=1; ops[g]=0;}  // start: ADD, amp=1
    int combined[256]={0}; if(unchanged_val>=0) combined[unchanged_val]++;
    for(int g=0;g<N;g++){int tf[256];hist_transform(gh[g],tf,ops[g],amps[g]);for(int v=0;v<256;v++)combined[v]+=tf[v];}
    for(int pass=0;pass<3;pass++) for(int g=0;g<N;g++){
        int tc[256],wk[256]; hist_transform(gh[g],tc,ops[g],amps[g]);
        for(int v=0;v<256;v++) wk[v]=combined[v]-tc[v];
        double bge=1e30; u8 bga=amps[g]; u8 bgo=ops[g]; int best_tf[256]; memcpy(best_tf,tc,256*sizeof(int));
        for(int si=0;si<N_SEARCH_OPS;si++){int op=SEARCH_OPS[si];OP_RANGE(op,alo,ahi)
            for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
                int tf[256],ff[256]; hist_transform(gh[g],tf,op,amp);
                for(int v=0;v<256;v++) ff[v]=wk[v]+tf[v];
                double e=entropy_fast(ff); if(e<bge){bge=e;bga=(u8)amp;bgo=(u8)op;memcpy(best_tf,tf,256*sizeof(int));}}}
        for(int v=0;v<256;v++) combined[v]=wk[v]+best_tf[v];
        amps[g]=bga; ops[g]=bgo;}
    for(int g=0;g<N;g++){amps_out[g]=amps[g]; ops_out[g]=ops[g];}
    return entropy_fast(combined);
}

static SR search_cond_prev(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=12; r.entropy=base; r.overhead=PAT_OH[12];
    double best_net=-1e30; int bk=1,bflex=0; u8 bamps[256]={0}, bops[256]={0};
    int (*gh)[256]=malloc(32*256*sizeof(int));
    if(gh) for(int k=1;k<=5;k++){
        int N=1<<k;
        memset(gh,0,N*256*sizeof(int));
        for(int i=1;i<n;i++) gh[blk[i-1]>>(8-k)][blk[i]]++;
        // shared op (cheaper overhead)
        for(int op=0;op<N_OPS;op++){
            u8 amps[256]={0}; int oh=5+3+3+N*8;
            double e=cond_greedy(gh,N,n,blk[0],op,amps);
            double net=(base-e)*n-oh;
            if(net>best_net){best_net=net;bk=k;bflex=0;r.entropy=e;r.overhead=oh;r.p[1]=op;
                memcpy(bamps,amps,N);}}
        // per-group ops (higher overhead, better fit on structured data)
        {u8 amps[256]={0},ops[256]={0}; int oh=5+3+N*11;
         double e=cond_greedy_flex(gh,N,n,blk[0],amps,ops);
         double net=(base-e)*n-oh;
         if(net>best_net){best_net=net;bk=k;bflex=1;r.entropy=e;r.overhead=oh;
             memcpy(bamps,amps,N);memcpy(bops,ops,N);}}}
    free(gh);
    r.p[0]=bk; r.p[2]=bflex;
    {int N=1<<bk; memcpy(r.amps,bamps,N); if(bflex) memcpy(r.grp_ops,bops,N);}
    snprintf(r.name,sizeof(r.name),"COND-PREV k=%d%s",bk,bflex?"-F":"");
    return r;
}

// COND-NEXT: op+amp keyed by high k bits of b[i+1] (unmodified next byte).
static SR search_cond_next(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=13; r.entropy=base; r.overhead=PAT_OH[13];
    double best_net=-1e30; int bk=1,bflex=0; u8 bamps[256]={0}, bops[256]={0};
    int (*gh)[256]=malloc(32*256*sizeof(int));
    if(gh) for(int k=1;k<=5;k++){
        int N=1<<k;
        memset(gh,0,N*256*sizeof(int));
        for(int i=0;i<n-1;i++) gh[blk[i+1]>>(8-k)][blk[i]]++;
        for(int op=0;op<N_OPS;op++){
            u8 amps[256]={0}; int oh=5+3+3+N*8;
            double e=cond_greedy(gh,N,n,blk[n-1],op,amps);
            double net=(base-e)*n-oh;
            if(net>best_net){best_net=net;bk=k;bflex=0;r.entropy=e;r.overhead=oh;r.p[1]=op;
                memcpy(bamps,amps,N);}}
        {u8 amps[256]={0},ops[256]={0}; int oh=5+3+N*11;
         double e=cond_greedy_flex(gh,N,n,blk[n-1],amps,ops);
         double net=(base-e)*n-oh;
         if(net>best_net){best_net=net;bk=k;bflex=1;r.entropy=e;r.overhead=oh;
             memcpy(bamps,amps,N);memcpy(bops,ops,N);}}}
    free(gh);
    r.p[0]=bk; r.p[2]=bflex;
    {int N=1<<bk; memcpy(r.amps,bamps,N); if(bflex) memcpy(r.grp_ops,bops,N);}
    snprintf(r.name,sizeof(r.name),"COND-NEXT k=%d%s",bk,bflex?"-F":"");
    return r;
}

// COND-DELTA: op+amp keyed by high k bits of (b[i-1] XOR b[i-2]).
static SR search_cond_delta(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=14; r.entropy=base; r.overhead=PAT_OH[14];
    double best_net=-1e30; int bk=1,bflex=0; u8 bamps[256]={0}, bops[256]={0};
    int (*gh)[256]=malloc(32*256*sizeof(int));
    if(gh) for(int k=1;k<=5;k++){
        int N=1<<k;
        memset(gh,0,N*256*sizeof(int));
        for(int i=2;i<n;i++) gh[(blk[i-1]^blk[i-2])>>(8-k)][blk[i]]++;
        for(int op=0;op<N_OPS;op++){
            u8 amps[256]={0}; int oh=5+3+3+N*8;
            double e=cond_greedy(gh,N,n,blk[0],op,amps);
            double net=(base-e)*n-oh;
            if(net>best_net){best_net=net;bk=k;bflex=0;r.entropy=e;r.overhead=oh;r.p[1]=op;
                memcpy(bamps,amps,N);}}
        {u8 amps[256]={0},ops[256]={0}; int oh=5+3+N*11;
         double e=cond_greedy_flex(gh,N,n,blk[0],amps,ops);
         double net=(base-e)*n-oh;
         if(net>best_net){best_net=net;bk=k;bflex=1;r.entropy=e;r.overhead=oh;
             memcpy(bamps,amps,N);memcpy(bops,ops,N);}}}
    free(gh);
    r.p[0]=bk; r.p[2]=bflex;
    {int N=1<<bk; memcpy(r.amps,bamps,N); if(bflex) memcpy(r.grp_ops,bops,N);}
    snprintf(r.name,sizeof(r.name),"COND-DELTA k=%d%s",bk,bflex?"-F":"");
    return r;
}

// COND-PREV2: 2nd-order Markov, context = top-j bits of b[i-1] || top-l bits of b[i-2].
// Parallelised over (j,l) pairs. bpk[4] optionally receives best SR per k=4..7.
static SR search_cond_prev2(const u8*blk,int n,double base, SR *bpk){
    SR r; memset(&r,0,sizeof(r)); r.id=16; r.entropy=base; r.overhead=PAT_OH[16];
    if(bpk) for(int i=0;i<4;i++){memset(&bpk[i],0,sizeof(SR));bpk[i].id=16;bpk[i].entropy=base;}
    double best_net=-1e30, bnet_k[4]={-1e30,-1e30,-1e30,-1e30};

    int pairs[64][2],npairs=0;
    for(int j=1;j<=6;j++) for(int l=1;l<=6;l++)
        if(j+l>=5&&j+l<=6){pairs[npairs][0]=j;pairs[npairs][1]=l;npairs++;}

    #pragma omp parallel for schedule(dynamic,1)
    for(int p=0;p<npairs;p++){
        int j=pairs[p][0],l=pairs[p][1],ki=j+l-4;
        int N=(1<<j)*(1<<l);
        int (*gh)[256]=malloc(N*256*sizeof(int));
        if(!gh) continue;
        memset(gh,0,N*256*sizeof(int));
        for(int i=2;i<n;i++) gh[((blk[i-1]>>(8-j))<<l)|(blk[i-2]>>(8-l))][blk[i]]++;
        for(int si=0;si<N_SEARCH_OPS;si++){int op=SEARCH_OPS[si];
            u8 amps[256]={0};
            int ab=(op==6||op==15)?3:(op==3||op==7||op==9||op==10||op==11||op==13||op==14)?4:(op==2)?7:8;
            int oh=11+N*(1+ab);  // 3(j)+3(l)+1(flex)+4(op)+N(bitmask)+N*ab
            double e=cond_greedy(gh,N,n,blk[0],op,amps);
            double net=(base-e)*n-oh;
            #pragma omp critical
            {if(net>best_net){best_net=net;r.entropy=e;r.overhead=oh;
                 r.p[0]=j;r.p[1]=l;r.p[2]=0;r.p[3]=op;memcpy(r.amps,amps,N);
                 snprintf(r.name,sizeof(r.name),"COND-PREV2 j=%d l=%d",j,l);}
             if(bpk&&ki>=0&&ki<4&&net>bnet_k[ki]){bnet_k[ki]=net;
                 bpk[ki].entropy=e;bpk[ki].overhead=oh;
                 bpk[ki].p[0]=j;bpk[ki].p[1]=l;bpk[ki].p[2]=0;bpk[ki].p[3]=op;
                 memcpy(bpk[ki].amps,amps,N);
                 snprintf(bpk[ki].name,sizeof(bpk[ki].name),"COND-PREV2 j=%d l=%d",j,l);}}}
        {u8 amps[256]={0},ops[256]={0}; int oh=7+N*13;  // 3+3+1(header)+N*(1+4+8) worst-case flex
         double e=cond_greedy_flex(gh,N,n,blk[0],amps,ops);
         double net=(base-e)*n-oh;
         #pragma omp critical
         {if(net>best_net){best_net=net;r.entropy=e;r.overhead=oh;
              r.p[0]=j;r.p[1]=l;r.p[2]=1;memcpy(r.amps,amps,N);memcpy(r.grp_ops,ops,N);
              snprintf(r.name,sizeof(r.name),"COND-PREV2 j=%d l=%d-F",j,l);}
          if(bpk&&ki>=0&&ki<4&&net>bnet_k[ki]){bnet_k[ki]=net;
              bpk[ki].entropy=e;bpk[ki].overhead=oh;
              bpk[ki].p[0]=j;bpk[ki].p[1]=l;bpk[ki].p[2]=1;
              memcpy(bpk[ki].amps,amps,N);memcpy(bpk[ki].grp_ops,ops,N);
              snprintf(bpk[ki].name,sizeof(bpk[ki].name),"COND-PREV2 j=%d l=%d-F",j,l);}}}
        free(gh);
    }
    return r;
}

// â"€â"€ Phase 2 search functions â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€

// COND-POS: context = position mod 2^k (k=1..4). Groups bytes by their index
// modulo 2/4/8/16. Captures periodic structure: even vs odd bytes, etc.
// All n bytes are covered (no "unchanged" position), so unchanged_val = -1.
static SR search_cond_pos(const u8 *blk, int n, double base){
    SR r; memset(&r,0,sizeof(r)); r.id=15; r.entropy=base; r.overhead=PAT_OH[15];
    double best_net=-1e30; int bk=1,bflex=0; u8 bamps[256]={0}, bops[256]={0};
    int (*gh)[256]=malloc(32*256*sizeof(int));
    if(gh) for(int k=1;k<=4;k++){
        int N=1<<k;
        memset(gh,0,N*256*sizeof(int));
        for(int i=0;i<n;i++) gh[i%N][blk[i]]++;  // context = position mod N
        for(int op=0;op<N_OPS;op++){
            u8 amps[256]={0}; int oh=5+3+3+N*8;
            double e=cond_greedy(gh,N,n,-1,op,amps);  // -1: all positions covered
            double net=(base-e)*n-oh;
            if(net>best_net){best_net=net;bk=k;bflex=0;r.entropy=e;r.overhead=oh;r.p[1]=op;
                memcpy(bamps,amps,N);}}
        {u8 amps[256]={0},ops[256]={0}; int oh=5+3+N*11;
         double e=cond_greedy_flex(gh,N,n,-1,amps,ops);
         double net=(base-e)*n-oh;
         if(net>best_net){best_net=net;bk=k;bflex=1;r.entropy=e;r.overhead=oh;
             memcpy(bamps,amps,N);memcpy(bops,ops,N);}}}
    free(gh);
    r.p[0]=bk; r.p[2]=bflex;
    {int N=1<<bk; memcpy(r.amps,bamps,N); if(bflex) memcpy(r.grp_ops,bops,N);}
    snprintf(r.name,sizeof(r.name),"COND-POS k=%d%s",bk,bflex?"-F":"");
    return r;
}

// ID=25  PRNG-GATE-PRNG-AMP: 2 LCG/pos (gate, amp), fixed op brute-forced, thr swept 1..255.
// Sweep algorithm: counting-sort positions by gate value once per seed, then slide thr in O(n) total.
static SR search_prng_gate_prng_amp(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=25; r.entropy=base; r.overhead=PAT_OH[25];
    double be=base; int bseed=0,bop=0; u8 bthr=128;
    #pragma omp parallel
    {double le=base; int ls=0,lop=0; u8 lth=128;
     u8 *gate=malloc(n),*ra=malloc(n);
     int *spos=malloc(n*sizeof(int));
     #pragma omp for schedule(dynamic,32)
     for(int seed=0;seed<65536;seed++){  // 16-bit seed: 0..65535
         if(!gate||!ra||!spos) continue;
         uint32_t st=(uint32_t)seed;
         for(int i=0;i<n;i++){gate[i]=lcg_byte(&st);ra[i]=lcg_byte(&st);}
         // Counting sort: sort positions by gate value for O(1) deselection per thr step
         int gcnt[256]={0};
         for(int i=0;i<n;i++) gcnt[gate[i]]++;
         int gst[256]; gst[0]=0;
         for(int g=1;g<256;g++) gst[g]=gst[g-1]+gcnt[g-1];
         {int sc[256]; memcpy(sc,gst,sizeof(gst));
          for(int i=0;i<n;i++) spos[sc[gate[i]]++]=i;}
         for(int op=0;op<N_OPS;op++){
             // Precompute transformed output for this op
             u8 trf[BLOCK_SIZE];
             for(int i=0;i<n;i++) trf[i]=op_byte(blk[i],prng_amp(ra[i],op),op);
             // Build initial ff for thr=224: gate>=224 → transformed, gate<224 → original
             int ff[256]={0};
             for(int i=0;i<n;i++) ff[gate[i]>=224?trf[i]:blk[i]]++;
             // Sweep thr 224..255 (5 bits, stored as thr-224, 32 values): deselect gate[i]==thr
             for(int g=224;g<=254;g++){
                 double e=entropy_fast(ff);
                 if(e<le){le=e;ls=seed;lop=op;lth=(u8)g;}
                 for(int k=gst[g];k<gst[g]+gcnt[g];k++){
                     int i=spos[k]; ff[trf[i]]--; ff[blk[i]]++;}}
             double e=entropy_fast(ff);
             if(e<le){le=e;ls=seed;lop=op;lth=255;}}
     }
     free(gate); free(ra); free(spos);
     #pragma omp critical
     if(le<be){be=le;bseed=ls;bop=lop;bthr=lth;}}
    r.entropy=be; r.p[0]=bseed; r.p[1]=(int)bthr; r.p[2]=bop;
    snprintf(r.name,sizeof(r.name),"PRNG-GATE-PRNG-AMP s=%d thr=%d op%d",bseed,(int)bthr,bop);
    return r;
}


// â"€â"€ Phase runners â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€â"€
// ID=26 PRNG-GATE-FIXED-AMP: PRNG gate (1 LCG per position), single brute-forced op+amp.
// Uses histogram trick: sel_hist built once per (seed,thr), all (op,amp) tried in O(256).
// Overhead: seed(10)+thr(5)+op(4)+amp(8) = 27 bits. Seeds: 0..1023 (10-bit).
static SR search_prng_gate_fixed_amp(const u8 *blk, int n, double base){
    SR r; memset(&r,0,sizeof(r)); r.id=26; r.entropy=base; r.overhead=PAT_OH[26];
    int gh[256]={0}; for(int i=0;i<n;i++) gh[blk[i]]++;
    double be=base; int bseed=0,bop=0,bamp=1; u8 bthr=224;
    #pragma omp parallel
    {double le=base; int ls=0,lop=0,lamp=1; u8 lth=224;
     int *spos=malloc(n*sizeof(int));
     #pragma omp for schedule(dynamic,16)
     for(int seed=0;seed<1024;seed++){
         if(!spos) continue;
         uint32_t st=(uint32_t)seed;
         u8 gate[BLOCK_SIZE];
         for(int i=0;i<n;i++) gate[i]=lcg_byte(&st);
         // Counting sort positions by gate value
         int gcnt[256]={0};
         for(int i=0;i<n;i++) gcnt[gate[i]]++;
         int gst[256]; gst[0]=0;
         for(int g=1;g<256;g++) gst[g]=gst[g-1]+gcnt[g-1];
         {int sc[256]; memcpy(sc,gst,sizeof(gst));
          for(int i=0;i<n;i++) spos[sc[gate[i]]++]=i;}
         // Build sel_hist for thr=224 (positions with gate[i]>=224)
         int sel[256]={0};
         for(int g=224;g<256;g++)
             for(int k=gst[g];k<gst[g]+gcnt[g];k++) sel[blk[spos[k]]]++;
         int non[256]; for(int v=0;v<256;v++) non[v]=gh[v]-sel[v];
         // Sweep thr 224..255: try all (op,amp) at each threshold
         for(int thr=224;thr<=254;thr++){
             for(int op=0;op<N_OPS;op++){
                 OP_RANGE(op,alo,ahi)
                 for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
                     int tf[256]; hist_transform(sel,tf,op,amp);
                     int combined[256]; for(int v=0;v<256;v++) combined[v]=non[v]+tf[v];
                     double e=entropy_fast(combined);
                     if(e<le){le=e;ls=seed;lop=op;lamp=amp;lth=(u8)thr;}
                 }
             }
             // Slide: deselect positions with gate[i]==thr
             for(int k=gst[thr];k<gst[thr]+gcnt[thr];k++){
                 sel[blk[spos[k]]]--; non[blk[spos[k]]]++;}
         }
         // Check thr=255
         for(int op=0;op<N_OPS;op++){
             OP_RANGE(op,alo,ahi)
             for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
                 int tf[256]; hist_transform(sel,tf,op,amp);
                 int combined[256]; for(int v=0;v<256;v++) combined[v]=non[v]+tf[v];
                 double e=entropy_fast(combined);
                 if(e<le){le=e;ls=seed;lop=op;lamp=amp;lth=255;}
             }
         }
     }
     free(spos);
     #pragma omp critical
     if(le<be){be=le;bseed=ls;bop=lop;bamp=lamp;bthr=lth;}}
    r.entropy=be; r.p[0]=bseed; r.p[1]=(int)bthr; r.p[2]=bop; r.p[3]=bamp;
    snprintf(r.name,sizeof(r.name),"PRNG-GATE-FIXED s=%d thr=%d op%d amp=%d",
             bseed,(int)bthr,bop,bamp);
    return r;
}

// ID=27 BIT-ROTATE: cyclic left-rotation of entire block by rot bits (rot=0..8n-1).
// Tries all 32768 positions. Overhead: 15 bits (log2(n*8) for n=4096).
// Inverse: rotate left by (n*8 - rot) bits, which is rotate right by rot.
static SR search_bit_rotate(const u8 *blk, int n, double base){
    SR r; memset(&r,0,sizeof(r)); r.id=27; r.entropy=base; r.overhead=PAT_OH[27];
    u8 *tmp=malloc(n); if(!tmp) return r;
    double be=base; int brot=0;
    // For a fixed bit-phase bp, all byte-offsets bs produce identical histograms:
    // incrementing bs by 1 cyclically shifts tmp by one byte, which preserves entropy.
    // So only bp=1..7 need testing; bp=0 (byte-aligned) never changes the histogram.
    for(int bp=1;bp<=7;bp++){
        for(int i=0;i<n-1;i++)
            tmp[i]=(u8)((blk[i]<<bp)|(blk[i+1]>>(8-bp)));
        tmp[n-1]=(u8)((blk[n-1]<<bp)|(blk[0]>>(8-bp)));
        double e=byte_entropy(tmp,n);
        if(e<be){be=e;brot=bp;}
    }
    free(tmp);
    r.entropy=be; r.p[0]=brot;
    snprintf(r.name,sizeof(r.name),"BIT-ROTATE rot=%d",brot);
    return r;
}

#define N_P1 1
#define N_P2 3
static void run_phase1(const u8 *blk, int n, double base, SR *out) {
    out[0] = search_cond_prev2(blk,n,base,NULL);
}
static void run_phase2(const u8 *blk, int n, double base, SR *out) {
    out[0] = search_prng_gate_prng_amp  (blk,n,base);
    out[1] = search_prng_gate_fixed_amp (blk,n,base);
    out[2] = search_bit_rotate          (blk,n,base);
}


// Returns 1 if a profitable transform was found and applied, 0 otherwise.
// applied_out: if non-NULL, receives a copy of the applied SR.
static int run_phase_apply(u8 *blk, int n, double *base, double *total_net,
                           SR *results, int np, int *total_oh, SR *applied_out) {
    int best=-1; double best_net=-1e30;
    for(int i=0;i<np;i++){
        double net=((*base)-results[i].entropy)*n-results[i].overhead;
        if(net>best_net){best_net=net;best=i;}
    }
    if(best<0||best_net<=0) return 0;
    if(applied_out) *applied_out=results[best];
    apply_sr(blk,n,&results[best]);
    *base=byte_entropy(blk,n);
    *total_net+=best_net;
    if(total_oh) *total_oh+=results[best].overhead;
    return 1;
}

// ── BitBuf: bit-level I/O for the instruction stream ─────────────────────────
typedef struct { uint8_t *buf; int bit_pos; int cap; } BitBuf;
static void bb_init(BitBuf *b, uint8_t *buf, int cap){
    b->buf=buf; b->bit_pos=0; b->cap=cap; memset(buf,0,cap);}
static void bb_write(BitBuf *b, uint32_t val, int bits){
    for(int i=bits-1;i>=0;i--){
        if(b->bit_pos<b->cap*8)
            b->buf[b->bit_pos>>3]|=(uint8_t)(((val>>i)&1)<<(7-(b->bit_pos&7)));
        b->bit_pos++;}}
static uint32_t bb_read(BitBuf *b, int bits){
    uint32_t v=0;
    for(int i=0;i<bits;i++){
        v<<=1;
        if(b->bit_pos<b->cap*8) v|=(b->buf[b->bit_pos>>3]>>(7-(b->bit_pos&7)))&1;
        b->bit_pos++;}
    return v;}

// ── Global frequency table (rANS precision M=4096) ───────────────────────────
#define RANS_M 4096
#define RANS_L ((uint32_t)(RANS_M) << 11)   // lower bound = 2^23
// Normalize raw histogram to sum=RANS_M; each non-zero raw[i] → norm[i]≥1.
// Requires RANS_M ≥ number of non-zero symbols (always true: 4096 >> 256).
static void freq_normalize(const int raw[256], uint16_t norm[256]){
    int total=0;
    for(int i=0;i<256;i++) total+=raw[i];
    if(!total){memset(norm,0,256*sizeof(uint16_t));norm[0]=RANS_M;return;}
    int sum=0;
    double frac[256];
    for(int i=0;i<256;i++){
        if(!raw[i]){norm[i]=0;frac[i]=0.0;continue;}
        double s=(double)raw[i]*RANS_M/total;
        int v=(int)s; if(v<1)v=1;
        norm[i]=(uint16_t)v; frac[i]=s-v; sum+=v;
    }
    // Trim: remove excess from largest buckets (rare edge case)
    while(sum>RANS_M){
        int bi=-1,bv=1;
        for(int i=0;i<256;i++) if(norm[i]>bv){bv=norm[i];bi=i;}
        if(bi<0) break;
        norm[bi]--; sum--;
    }
    // Grow: add remainder to symbols with largest fractional deficit
    while(sum<RANS_M){
        double best=-1.0; int bi=-1;
        for(int i=0;i<256;i++) if(raw[i]&&frac[i]>best){best=frac[i];bi=i;}
        if(bi<0) break;
        norm[bi]++; frac[bi]-=1.0; sum++;
    }
}

static void print_compression_estimate(const uint16_t norm[256],
                                        int n_blocks, int raw_bits_total,
                                        int payload_bits_total, int instr_bits_total){
    double H_table=0.0; int nz=0;
    for(int i=0;i<256;i++){
        if(!norm[i]) continue; nz++;
        double p=(double)norm[i]/RANS_M; H_table-=p*log2(p);
    }
    printf("=== Global Frequency Table ===\n");
    printf("  Non-zero symbols:   %d / 256\n", nz);
    printf("  Table entropy:      %.4f bits/symbol\n", H_table);
    printf("  Table storage:      512 bytes (written once per file)\n\n");

    int table_bits = 512*8;
    int total_compressed = payload_bits_total + instr_bits_total + table_bits;
    printf("=== Compression Estimate (%d blocks) ===\n", n_blocks);
    printf("  rANS payload  (H*n lower bound):  %8d bits  (%6.2f KB)\n",
           payload_bits_total, payload_bits_total/8192.0);
    printf("  Instructions  (bit-packed):       %8d bits  (%6.2f bytes)\n",
           instr_bits_total, instr_bits_total/8.0);
    printf("  Freq table    (amortized):        %8d bits  (%6.2f bytes)\n",
           table_bits, table_bits/8.0);
    printf("  Total compressed:                 %8d bits  (%6.2f KB)\n",
           total_compressed, total_compressed/8192.0);
    printf("  Total raw:                        %8d bits  (%6.2f KB)\n",
           raw_bits_total, raw_bits_total/8192.0);
    printf("  Ratio: %.4f  (%.2f%% of original)\n\n",
           (double)total_compressed/raw_bits_total,
           100.0*total_compressed/raw_bits_total);
    printf("  Note: instructions are bit-packed to their information-theoretic\n");
    printf("  minimum. ANS/Huffman on ~%d bytes of instruction data would add\n",
           instr_bits_total/8+1);
    printf("  coder overhead exceeding savings. Bit-packing is optimal here.\n\n");
}

// ── rANS encoder ─────────────────────────────────────────────────────────────
static void rans_build_cum(const uint16_t freq[256], uint32_t cum[257]) {
    cum[0]=0;
    for(int i=0;i<256;i++) cum[i+1]=cum[i]+freq[i];
}

// Encodes src[n] backwards into buf[0..buf_cap-1].
// Returns pointer to start of encoded data; *out_len = number of bytes.
static u8 *rans_encode_blk(const u8 *src, int n,
                            const uint16_t freq[256], const uint32_t cum[257],
                            u8 *buf, int buf_cap, int *out_len) {
    u8 *ptr = buf + buf_cap;
    uint32_t x = RANS_L;
    for(int i=n-1; i>=0; i--) {
        u8  s  = src[i];
        uint32_t fs = freq[s] ? (uint32_t)freq[s] : 1u;
        uint32_t cs = cum[s];
        uint32_t xmax = ((RANS_L / RANS_M) << 8) * fs;
        while(x >= xmax) { *--ptr = (u8)(x & 0xFF); x >>= 8; }
        x = (x / fs) * RANS_M + cs + (x % fs);
    }
    // Write final state LE (4 bytes) — decoder reads this first
    ptr -= 4;
    ptr[0]=(u8)x; ptr[1]=(u8)(x>>8); ptr[2]=(u8)(x>>16); ptr[3]=(u8)(x>>24);
    *out_len = (int)(buf + buf_cap - ptr);
    return ptr;
}

// ── Instruction stream serialization ─────────────────────────────────────────
// Returns 1 if op+amp is a no-op (identity transform).
// Only ADD/XOR/ADDLO/ADDHI/XORLO/XORHI have an identity at amp=0.
static int is_identity(int op, u8 amp){
    if(amp!=0) return 0;
    return op==0||op==1||op==3||op==7||op==13||op==14;
}

// COND-PREV2 layout: j-1(3) l-1(3) flex(1)
//   shared: op(4) bitmask(N) active*(amp_bits(op))
//   flex:   bitmask(N) active*(op(4)+amp_bits(op))
// Inactive groups (identity amp) are marked 0 in the bitmask and skipped.
// amp_bits: ROL/ROR→3, nibble-ops→4, MUL→7, all others→8
static int amp_bits(int op){
    if(op==6||op==15) return 3;
    if(op==3||op==7||op==9||op==10||op==11||op==13||op==14) return 4;
    if(op==2) return 7;
    return 8;
}
static void write_amp(BitBuf *bb, int op, u8 amp){
    if(op==6||op==15){ bb_write(bb,(uint32_t)(amp-1),3); return; }
    if(op==3||op==7||op==9||op==10||op==11||op==13||op==14){ bb_write(bb,amp,4); return; }
    if(op==2){ bb_write(bb,(uint32_t)((amp-3)/2),7); return; }
    bb_write(bb,amp,8);
}
static u8 read_amp(BitBuf *bb, int op){
    if(op==6||op==15) return (u8)(bb_read(bb,3)+1);
    if(op==3||op==7||op==9||op==10||op==11||op==13||op==14) return (u8)bb_read(bb,4);
    if(op==2) return (u8)(bb_read(bb,7)*2+3);
    return (u8)bb_read(bb,8);
}
static void serialize_sr(BitBuf *bb, const SR *r) {
    if(r->id!=16) return;
    int j=r->p[0],l=r->p[1],flex=r->p[2],op=r->p[3];
    int N=(1<<j)*(1<<l);
    bb_write(bb,(uint32_t)(j-1),3);
    bb_write(bb,(uint32_t)(l-1),3);
    bb_write(bb,(uint32_t)flex,1);
    if(flex){
        // bitmask first, then active group op+amp pairs
        for(int g=0;g<N;g++)
            bb_write(bb,(uint32_t)(!is_identity(r->grp_ops[g],r->amps[g])),1);
        for(int g=0;g<N;g++){
            if(is_identity(r->grp_ops[g],r->amps[g])) continue;
            bb_write(bb,(uint32_t)r->grp_ops[g],4);
            write_amp(bb,r->grp_ops[g],r->amps[g]);
        }
    } else {
        // shared op first, then bitmask, then active amps
        bb_write(bb,(uint32_t)op,4);
        for(int g=0;g<N;g++)
            bb_write(bb,(uint32_t)(!is_identity(op,r->amps[g])),1);
        for(int g=0;g<N;g++){
            if(is_identity(op,r->amps[g])) continue;
            write_amp(bb,op,r->amps[g]);
        }
    }
}

// Actual rANS coding cost in bits for a block under a given frequency table.
static double rans_cost_bits(const u8 *blk, int n, const uint16_t freq[256]){
    double cost=0.0;
    for(int i=0;i<n;i++){uint16_t f=freq[blk[i]];if(!f)f=1;cost+=log2((double)RANS_M/f);}
    return cost;
}
// Count bits serialize_sr would write for a given SR.
static int sr_bit_count(const SR *r){
    u8 tmp[512]; BitBuf bb; bb_init(&bb,tmp,sizeof(tmp));
    serialize_sr(&bb,r); return bb.bit_pos;
}

// ── Block storage (pass 1 results) ───────────────────────────────────────────
typedef struct {
    u8  blk[BLOCK_SIZE];
    SR  applied[MAX_TRANSFORMS];
    int n_applied;
    int orig_bytes;
} BlockData;


int main(void) {
    init_gf256();
    init_entropy_table(BLOCK_SIZE);
    int nt=omp_get_max_threads();
    omp_set_num_threads(nt);
    // ── Allocate block storage ────────────────────────────────────────────────
    int blocks_cap = 1024;
    BlockData *blocks = malloc((size_t)blocks_cap * sizeof(BlockData));
    if(!blocks){ fprintf(stderr,"OOM\n"); return 1; }

    // ── Pass 1: read file, apply transforms, collect stats ────────────────────
    FILE *fin = fopen(INPUT_FILE,"rb");
    if(!fin){ fprintf(stderr,"Cannot open: %s\n", INPUT_FILE); free(blocks); return 1; }
    printf("Input:  %s\n", INPUT_FILE);
    printf("Output: %s\n", OUTPUT_FILE);
    printf("Block:  %d bytes  threads=%d\n\n", BLOCK_SIZE, nt);
    printf("%-4s  %-7s %-7s  %-6s  %-3s  %-3s %-3s  %-7s\n","Blk","startH","finalH","net","T","j","l","act/tot");
    printf("----  ------- -------  ------  ---  --- ---  -------\n");

    uint64_t total_input_bytes=0;
    int n_blocks=0;
    long total_act=0, total_tot=0;
    int op_counts[N_OPS]={0};

    for(;;) {
        if(n_blocks >= blocks_cap) {
            blocks_cap *= 2;
            BlockData *tmp = realloc(blocks, (size_t)blocks_cap * sizeof(BlockData));
            if(!tmp){ fprintf(stderr,"OOM\n"); fclose(fin); free(blocks); return 1; }
            blocks = tmp;
        }
        BlockData *bd = &blocks[n_blocks];
        int got = (int)fread(bd->blk, 1, BLOCK_SIZE, fin);
        if(got <= 0) break;
        bd->orig_bytes = got;
        bd->n_applied  = 0;
        total_input_bytes += (uint64_t)got;
        memset(bd->blk + got, 0, BLOCK_SIZE - got);

        double base = byte_entropy(bd->blk, BLOCK_SIZE);
        double start_h = base, total_net = 0.0;

        int diag_j=0,diag_l=0,diag_act=0,diag_tot=0;
        {
            SR best=search_cond_prev2(bd->blk,BLOCK_SIZE,base,NULL);
            double best_net=(base-best.entropy)*(double)BLOCK_SIZE-best.overhead;
            if(best_net>0){
                apply_sr(bd->blk,BLOCK_SIZE,&best);
                base=byte_entropy(bd->blk,BLOCK_SIZE);
                total_net+=best_net;
                bd->applied[bd->n_applied++]=best;
                diag_j=best.p[0]; diag_l=best.p[1];
                diag_tot=(1<<diag_j)*(1<<diag_l);
                for(int g=0;g<diag_tot;g++){
                    int op=best.p[2]?best.grp_ops[g]:best.p[3];
                    if(!is_identity(op,best.amps[g])){diag_act++;op_counts[op]++;}
                }
            }
        }

        total_act += diag_act;
        total_tot += diag_tot;
        if(diag_tot>0)
            printf("%4d  %.4f %.4f  %+6.0f  %3d  %3d %3d  %3d/%-3d\n",
                   n_blocks+1, start_h, base, total_net, bd->n_applied,
                   diag_j, diag_l, diag_act, diag_tot);
        else
            printf("%4d  %.4f %.4f  %+6.0f  %3d  ---  --  ---/---\n",
                   n_blocks+1, start_h, base, total_net, bd->n_applied);
        n_blocks++;
    }
    fclose(fin);
    printf("\n%d blocks read (%.2f KB)\n", n_blocks, total_input_bytes/1024.0);

    if(total_tot>0){
        long unused = total_tot - total_act;
        printf("\n=== Group Slot Usage ===\n");
        printf("  Used   (non-identity amp): %5ld / %ld  (%.1f%%)\n",
               total_act, total_tot, 100.0*total_act/total_tot);
        printf("  Unused (identity amp):     %5ld / %ld  (%.1f%%)\n",
               unused, total_tot, 100.0*unused/total_tot);
    }

    {
        static const char *op_names[N_OPS]={
            "ADD","XOR","MUL","ADDLO","SWXOR","GFMUL",
            "ROL","ADDHI","GRAY","XORNIBBLE","SUBLO","SUBHI",
            "NEGADD","XORLO","XORHI","ROR"
        };
        printf("\n=== Op Usage (active group slots) ===\n");
        for(int i=0;i<N_OPS;i++){
            if(op_counts[i]>0)
                printf("  %-10s  %6d\n", op_names[i], op_counts[i]);
            else
                printf("  %-10s  (never used)\n", op_names[i]);
        }
    }

    // ── Pass 2: write compressed file (per-block freq tables) ─────────────────
    FILE *fout = fopen(OUTPUT_FILE,"wb");
    if(!fout){ fprintf(stderr,"Cannot create: %s\n", OUTPUT_FILE); free(blocks); return 1; }

    // Header: "CMP2"(4) ver(2) orig_size(8,LE) block_size(4,LE) n_blocks(4,LE)
    fwrite("CMP2",1,4,fout);
    fputc(0x02,fout);
    for(int i=0;i<8;i++) fputc((int)((total_input_bytes>>(i*8))&0xFF),fout);
    uint32_t bsz=BLOCK_SIZE;
    for(int i=0;i<4;i++) fputc((bsz>>(i*8))&0xFF,fout);
    uint32_t nb=(uint32_t)n_blocks;
    for(int i=0;i<4;i++) fputc((nb>>(i*8))&0xFF,fout);

    // Per-block: local_freq(512) n_applied(1) instr_size(2,LE) instr_bytes rans_size(3,LE) rans_bytes
    u8 *rans_buf = malloc(2*BLOCK_SIZE+64);
    u8 instr_buf[2048];
    int instr_hist[256]={0};
    long instr_total_bits=0, instr_total_bytes=0;
    long rans_total_bytes=0;
    for(int bi=0; bi<n_blocks; bi++) {
        BlockData *bd = &blocks[bi];

        // Per-block frequency table — rANS always achieves local entropy
        int local_hist[256]={0};
        for(int i=0;i<BLOCK_SIZE;i++) local_hist[bd->blk[i]]++;
        uint16_t local_freq[256]; freq_normalize(local_hist,local_freq);
        uint32_t local_cum[257]; rans_build_cum(local_freq,local_cum);
        for(int i=0;i<256;i++){ fputc(local_freq[i]&0xFF,fout); fputc((local_freq[i]>>8)&0xFF,fout); }

        // Serialize instruction stream
        BitBuf bb; bb_init(&bb, instr_buf, sizeof(instr_buf));
        for(int j=0;j<bd->n_applied;j++) serialize_sr(&bb, &bd->applied[j]);
        int instr_bytes = (bb.bit_pos + 7) / 8;
        instr_total_bits += bb.bit_pos;
        instr_total_bytes += instr_bytes;
        for(int j=0;j<instr_bytes;j++) instr_hist[instr_buf[j]]++;

        // rANS encode with local table
        int rans_len = 0;
        u8 *rans_ptr = rans_encode_blk(bd->blk, BLOCK_SIZE, local_freq, local_cum,
                                        rans_buf, 2*BLOCK_SIZE+64, &rans_len);
        rans_total_bytes += rans_len;

        fputc((u8)bd->n_applied, fout);
        fputc(instr_bytes & 0xFF, fout);
        fputc((instr_bytes >> 8) & 0xFF, fout);
        fwrite(instr_buf, 1, instr_bytes, fout);
        fputc(rans_len & 0xFF, fout);
        fputc((rans_len >> 8) & 0xFF, fout);
        fputc((rans_len >> 16) & 0xFF, fout);
        fwrite(rans_ptr, 1, rans_len, fout);
    }
    free(rans_buf);
    printf("\n  rANS payload:    %ld bytes  (%.2f KB)\n", rans_total_bytes, rans_total_bytes/1024.0);
    printf("  Per-block tables: %d bytes  (%.2f KB)\n", n_blocks*512, n_blocks*512/1024.0);

    double instr_H = instr_total_bytes>0 ? entropy_from_hist(instr_hist,(int)instr_total_bytes) : 0.0;
    double instr_ideal_bits = instr_H * instr_total_bytes;

    long out_bytes = ftell(fout);
    fclose(fout);
    free(blocks);

    printf("\n=== Instruction Stream ===\n");
    printf("  Raw bits packed:   %ld  (%.2f KB)\n", instr_total_bits, instr_total_bits/8192.0);
    printf("  Padded bytes:      %ld  (%.2f KB)\n", instr_total_bytes, instr_total_bytes/1024.0);
    printf("  Entropy:           %.4f bits/byte\n", instr_H);
    printf("  Ideal (H*n):       %.0f bits  (%.2f KB)\n", instr_ideal_bits, instr_ideal_bits/8192.0);
    printf("  Compressibility:   %.2f%%  (%.0f bits saveable)\n",
           instr_total_bytes>0 ? 100.0*(1.0-instr_H/8.0) : 0.0,
           instr_total_bits - instr_ideal_bits);

    printf("\n=== Compression Result ===\n");
    printf("  Input:   %llu bytes  (%.2f KB)\n",
           (unsigned long long)total_input_bytes, total_input_bytes/1024.0);
    printf("  Output:  %ld bytes  (%.2f KB)\n", out_bytes, out_bytes/1024.0);
    printf("  Ratio:   %.4f  (%.2f%%)\n",
           (double)out_bytes/total_input_bytes,
           100.0*out_bytes/total_input_bytes);
    return 0;
}
