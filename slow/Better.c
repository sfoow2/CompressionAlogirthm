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
#define N_BLOCKS    3
#define N_OPS       16
#define N_PATTERNS  3

// â”€â”€ GF(256) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
static u8 gf_exp[512], gf_log[256];
static void init_gf256(void) {
    u8 x=1;
    for(int i=0;i<255;i++){gf_exp[i]=x;gf_log[x]=(u8)i;u8 h=(x&0x80)?((x<<1)^0x1B):(x<<1);x=h^x;}
    for(int i=255;i<512;i++) gf_exp[i]=gf_exp[i-255];
    gf_log[0]=0;
}
static inline u8 gf_mul(u8 a,u8 b){return(!a||!b)?0:gf_exp[gf_log[a]+gf_log[b]];}

// â”€â”€ Core helpers â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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


// â”€â”€ Search result struct â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
// 25  PRNG-GATE-PRNG-AMP:  seed(16)+thr(5, stored as thr-224, range 224..255)+op(4)  = 25 bits
static const int PAT_OH[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,  // 0-11 unused
    30, 30, 30,                 // 12 COND-PREV, 13 COND-NEXT, 14 COND-DELTA
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0,  // 15-24 unused (10 entries)
    25,                          // 25 PRNG-GATE-PRNG-AMP
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

// â”€â”€ Apply a SearchResult in-place â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
static void apply_sr(u8 *blk, int n, const SR *r) {
    u8 *orig=NULL;
    if(r->id==12||r->id==14){  // COND-PREV, COND-DELTA need orig snapshot
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
        case 25: { // PRNG-GATE-PRNG-AMP: 2 LCG/pos (gate, amp); fixed op; thr stored as raw byte
            uint32_t st=(uint32_t)r->p[0]; u8 thr=(u8)r->p[1]; int op=r->p[2];
            for(int i=0;i<n;i++){u8 gate=lcg_byte(&st);u8 ra=lcg_byte(&st);
                if(gate>=thr) blk[i]=op_byte(blk[i],prng_amp(ra,op),op);}} break;
    }
    free(orig);
}

// â”€â”€ Helpers for search functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
// MUL(2): odd 3..255; ADDLO(3)/ADDHI(7)/XORNIBBLE(9): amp 0..15; GFMUL(5): amp>=2; ROL(6): 1..7; GRAY(8): 0..255
#define OP_RANGE(op,alo,ahi) int alo=1,ahi=255; \
    if(op==2)alo=3; if(op==3)ahi=15; if(op==5)alo=2; if(op==6)ahi=7; if(op==7)ahi=15; \
    if(op==8)alo=0; if(op==9){alo=0;ahi=15;} \
    if(op==10){alo=0;ahi=15;} if(op==11){alo=0;ahi=15;} if(op==12)alo=0; \
    if(op==13)ahi=15; if(op==14)ahi=15; if(op==15)ahi=7;
#define SKIP_OP(op,amp) if(op==2&&(amp&1)==0) continue;

// â”€â”€ Search functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

// Shared-op greedy: all groups use the same op, 2 refinement passes.
// Overhead when using this: 5 + 3 + 3 + N*8 bits (k + op + N*amp).
static double cond_greedy(const int (*gh)[256], int N, int n, int unchanged_val,
                           int op, u8 *amps_out) {
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

// Per-group-op greedy: each group picks its own (op, amp), 3 refinement passes.
// Overhead when using this: 5 + 3 + N*11 bits (k + N*(op+amp)).
// Strictly more general than cond_greedy â€” wins on structured data where groups differ.
static double cond_greedy_flex(const int (*gh)[256], int N, int n, int unchanged_val,
                                u8 *amps_out, u8 *ops_out) {
    u8 amps[256], ops[256];
    for(int g=0;g<N;g++){amps[g]=1; ops[g]=0;}  // start: ADD, amp=1
    int combined[256]={0}; combined[unchanged_val]++;
    for(int g=0;g<N;g++){int tf[256];hist_transform(gh[g],tf,ops[g],amps[g]);for(int v=0;v<256;v++)combined[v]+=tf[v];}
    for(int pass=0;pass<3;pass++) for(int g=0;g<N;g++){
        int tc[256],wk[256]; hist_transform(gh[g],tc,ops[g],amps[g]);
        for(int v=0;v<256;v++) wk[v]=combined[v]-tc[v];
        double bge=1e30; u8 bga=amps[g]; u8 bgo=ops[g];
        for(int op=0;op<N_OPS;op++){OP_RANGE(op,alo,ahi)
            for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
                int tf[256],ff[256]; hist_transform(gh[g],tf,op,amp);
                for(int v=0;v<256;v++) ff[v]=wk[v]+tf[v];
                double e=entropy_from_hist(ff,n); if(e<bge){bge=e;bga=(u8)amp;bgo=(u8)op;}}}
        int tn[256]; hist_transform(gh[g],tn,bgo,bga);
        for(int v=0;v<256;v++) combined[v]=wk[v]+tn[v];
        amps[g]=bga; ops[g]=bgo;}
    for(int g=0;g<N;g++){amps_out[g]=amps[g]; ops_out[g]=ops[g];}
    return entropy_from_hist(combined,n);
}

static SR search_cond_prev(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=12; r.entropy=base; r.overhead=PAT_OH[12];
    double best_net=-1e30; int bk=1,bflex=0; u8 bamps[256]={0}, bops[256]={0};
    for(int k=1;k<=4;k++){
        int N=1<<k;
        int (*gh)[256]=malloc(N*256*sizeof(int)); if(!gh) continue;
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
             memcpy(bamps,amps,N);memcpy(bops,ops,N);}}
        free(gh);}
    r.p[0]=bk; r.p[2]=bflex;
    {int N=1<<bk; memcpy(r.amps,bamps,N); if(bflex) memcpy(r.grp_ops,bops,N);}
    snprintf(r.name,sizeof(r.name),"COND-PREV k=%d%s",bk,bflex?"-F":"");
    return r;
}

// COND-NEXT: op+amp keyed by high k bits of b[i+1] (unmodified next byte).
static SR search_cond_next(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=13; r.entropy=base; r.overhead=PAT_OH[13];
    double best_net=-1e30; int bk=1,bflex=0; u8 bamps[256]={0}, bops[256]={0};
    for(int k=1;k<=4;k++){
        int N=1<<k;
        int (*gh)[256]=malloc(N*256*sizeof(int)); if(!gh) continue;
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
             memcpy(bamps,amps,N);memcpy(bops,ops,N);}}
        free(gh);}
    r.p[0]=bk; r.p[2]=bflex;
    {int N=1<<bk; memcpy(r.amps,bamps,N); if(bflex) memcpy(r.grp_ops,bops,N);}
    snprintf(r.name,sizeof(r.name),"COND-NEXT k=%d%s",bk,bflex?"-F":"");
    return r;
}

// COND-DELTA: op+amp keyed by high k bits of (b[i-1] XOR b[i-2]).
static SR search_cond_delta(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=14; r.entropy=base; r.overhead=PAT_OH[14];
    double best_net=-1e30; int bk=1,bflex=0; u8 bamps[256]={0}, bops[256]={0};
    for(int k=1;k<=4;k++){
        int N=1<<k;
        int (*gh)[256]=malloc(N*256*sizeof(int)); if(!gh) continue;
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
             memcpy(bamps,amps,N);memcpy(bops,ops,N);}}
        free(gh);}
    r.p[0]=bk; r.p[2]=bflex;
    {int N=1<<bk; memcpy(r.amps,bamps,N); if(bflex) memcpy(r.grp_ops,bops,N);}
    snprintf(r.name,sizeof(r.name),"COND-DELTA k=%d%s",bk,bflex?"-F":"");
    return r;
}

// â”€â”€ Phase 2 search functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

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


// â”€â”€ Phase runners â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
#define N_P1 3
#define N_P2 1
static void run_phase1(const u8 *blk, int n, double base, SR *out) {
    out[0] = search_cond_prev  (blk,n,base);
    out[1] = search_cond_next  (blk,n,base);
    out[2] = search_cond_delta (blk,n,base);
}
static void run_phase2(const u8 *blk, int n, double base, SR *out) {
    out[0] = search_prng_gate_prng_amp (blk,n,base);
}


static void print_bytes(const u8 *blk, int n, const char *label) {
    int show = 512 < n ? 512 : n;
    printf("%s [%d bytes, first %d]:\n", label, n, show);
    for(int i=0;i<show;i++){
        printf("%3d ", blk[i]);
        if((i&15)==15) printf("\n");  // 16 values per row
    }
    printf("\n");
}

// -- Phase 2 statistics -------------------------------------------------------
typedef struct {
    int thr_count[256];  // applied threshold histogram (224..255 range)
    int op_count[N_OPS]; // applied op histogram
    int total_applied;   // total transforms applied across all blocks/iters
} P2Stats;

// -- Main ----------------------------------------------------------------------
// Returns 1 if a transform was applied, 0 if nothing was profitable.
// stats: if non-NULL and applied transform is ID=25, records thr+op.
static int run_phase_print(u8 *blk, int n, double *base, double *total_net,
                            SR *results, int np, const char *phase_label,
                            P2Stats *stats) {
    double nets[256]; int best=-1; double best_net=-1e30;
    for(int i=0;i<np;i++){
        nets[i]=((*base)-results[i].entropy)*n-results[i].overhead;
        if(nets[i]>best_net){best_net=nets[i];best=i;}
    }
    printf("=== %s ===\n", phase_label);
    for(int i=0;i<np;i++)
        printf("  %-30s  net=%+.0f bits%s\n",
               results[i].name, nets[i], i==best&&best_net>0?" <--":"");
    if(best<0||best_net<=0){
        printf("  (no profitable transform)\n\n"); return 0;}
    printf("Applied: %-28s  H %.4f -> %.4f\n\n",
           results[best].name, *base, results[best].entropy);
    apply_sr(blk,n,&results[best]);
    *base=byte_entropy(blk,n);
    *total_net+=best_net;
    if(stats && results[best].id==25){
        int thr=results[best].p[1], op=results[best].p[2];
        if(thr>=0&&thr<256) stats->thr_count[thr]++;
        if(op>=0&&op<N_OPS) stats->op_count[op]++;
        stats->total_applied++;
    }
    return 1;
}

static void print_p2_stats(const P2Stats *s) {
    static const char *op_names[]={"ADD","XOR","MUL","ADDLO","SWXOR","GFMUL","ROL","ADDHI","GRAY","XORNIBBLE",
                                    "SUBLO","SUBHI","NEGADD","XORLO","XORHI","ROR"};
    int total=s->total_applied;
    printf("\n=== Phase 2 Statistics (%d transforms applied) ===\n\n", total);
    if(!total){printf("  (none applied)\n");return;}

    // -- Threshold stats (224..255) -------------------------------------------
    int tmin=256,tmax=-1,tnever=0; double tsum=0;
    for(int t=224;t<=255;t++){
        if(s->thr_count[t]){
            if(t<tmin)tmin=t; if(t>tmax)tmax=t;
            tsum+=t*s->thr_count[t];
        } else tnever++;
    }
    printf("Threshold (224..255):\n");
    printf("  min=%-3d  max=%-3d  avg=%.1f  never-used=%d/32\n",
           tmin,tmax,tsum/total,tnever);
    int omax_t=0;
    for(int t=224;t<=255;t++) if(s->thr_count[t]>omax_t) omax_t=s->thr_count[t];
    printf("  Non-zero values:\n");
    for(int t=224;t<=255;t++){
        if(!s->thr_count[t]) continue;
        int c=s->thr_count[t];
        int bar=omax_t?(int)(30.0*c/omax_t+0.5):0;
        printf("    %3d: %5d (%5.1f%%)  |", t, c, 100.0*c/total);
        for(int b=0;b<bar;b++) printf("#");
        printf("\n");
    }
    if(tnever){
        printf("  Never used:");
        for(int t=224;t<=255;t++) if(!s->thr_count[t]) printf(" %d",t);
        printf("\n");
    }

    // -- Op stats -------------------------------------------------------------
    printf("\nOps:\n");
    int omax_op=0;
    for(int op=0;op<N_OPS;op++) if(s->op_count[op]>omax_op) omax_op=s->op_count[op];
    for(int op=0;op<N_OPS;op++){
        int c=s->op_count[op];
        int bar=omax_op?(int)(30.0*c/omax_op+0.5):0;
        printf("  op%d %-10s %5d (%5.1f%%)  |", op, op_names[op], c, 100.0*c/total);
        for(int b=0;b<bar;b++) printf("#");
        if(!c) printf(" [NEVER USED]");
        printf("\n");
    }
    printf("\n");
}

int main(void) {
    init_gf256();
    init_entropy_table(BLOCK_SIZE);
    int nt=omp_get_max_threads()-1; if(nt<1)nt=1;
    omp_set_num_threads(nt);

    printf("Block size: %d bytes  threads=%d  N_BLOCKS=%d\n\n", BLOCK_SIZE, nt, N_BLOCKS);

    P2Stats stats; memset(&stats,0,sizeof(stats));

    double bst_min=1e30,bst_max=-1e30,bst_sum=0;  // start entropy
    double bfn_min=1e30,bfn_max=-1e30,bfn_sum=0;  // final entropy
    double bnt_min=1e30,bnt_max=-1e30,bnt_sum=0;  // net bits
    int    n_done=0;

    for(int blk_idx=0; blk_idx<N_BLOCKS; blk_idx++) {
        u8 *blk=malloc(BLOCK_SIZE);
        if(!blk||!fill_random(blk,BLOCK_SIZE)){free(blk);continue;}

        printf("\n=== Block %d/%d ===\n\n", blk_idx+1, N_BLOCKS);

        double base=byte_entropy(blk,BLOCK_SIZE);
        double start_h=base, total_net=0.0;
        int p1_iter=0, p2_iter=0;
        int p2_before=stats.total_applied;

        // Interleaved loop: one Phase 1 step, then Phase 2 loop, repeat until both fail.
        for(;;) {
            int p1_applied, p2_applied=0;

            p1_iter++;
            char lbl1[32]; snprintf(lbl1,sizeof(lbl1),"Phase 1 iter %d",p1_iter);
            SR p1[N_P1]; run_phase1(blk,BLOCK_SIZE,base,p1);
            p1_applied = run_phase_print(blk,BLOCK_SIZE,&base,&total_net,p1,N_P1,lbl1,NULL);

            for(;;) {
                p2_iter++;
                char lbl2[32]; snprintf(lbl2,sizeof(lbl2),"Phase 2 iter %d",p2_iter);
                SR p2[N_P2]; run_phase2(blk,BLOCK_SIZE,base,p2);
                if(!run_phase_print(blk,BLOCK_SIZE,&base,&total_net,p2,N_P2,lbl2,&stats)) break;
                p2_applied=1;
            }

            if(!p1_applied && !p2_applied) break;
        }

        printf("Block %d: H %.4f -> %.4f  net=%+.0f bits  p1_iters=%d  p2_applied=%d\n\n",
               blk_idx+1, start_h, base, total_net, p1_iter, stats.total_applied-p2_before);

        // Accumulate per-block stats
        if(start_h<bst_min) bst_min=start_h; if(start_h>bst_max) bst_max=start_h; bst_sum+=start_h;
        if(base   <bfn_min) bfn_min=base;    if(base   >bfn_max) bfn_max=base;    bfn_sum+=base;
        if(total_net<bnt_min) bnt_min=total_net; if(total_net>bnt_max) bnt_max=total_net; bnt_sum+=total_net;
        n_done++;

        free(blk);
    }

    // Per-block summary
    if(n_done>0){
        printf("=== Block Summary (%d blocks) ===\n", n_done);
        printf("               min       max       avg\n");
        printf("  start H:  %7.4f   %7.4f   %7.4f\n", bst_min, bst_max, bst_sum/n_done);
        printf("  final H:  %7.4f   %7.4f   %7.4f\n", bfn_min, bfn_max, bfn_sum/n_done);
        printf("  net bits: %7.0f   %7.0f   %7.0f\n", bnt_min, bnt_max, bnt_sum/n_done);
        printf("\n");
    }

    print_p2_stats(&stats);
    return 0;
}
