#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <bcrypt.h>
#include <omp.h>

typedef uint8_t u8;

#define BLOCK_SIZE  (64 * 1024)
#define MAX_STRIDE  64
#define N_OPS       6
#define N_GSHAPES   10
#define N_PSHAPES   8
#define N_PATTERNS  19

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

// ── Amplitude shapes ──────────────────────────────────────────────────────────
static inline u8 global_amp(int s,int i,int n,int A){
    long long v=0,d,h;
    switch(s){
        case 0:v=(long long)i*A/(n-1);break;
        case 1:v=A-(long long)i*A/(n-1);break;
        case 2:v=(long long)i*i*A/((long long)(n-1)*(n-1));break;
        case 3:v=A-(long long)i*i*A/((long long)(n-1)*(n-1));break;
        case 4:d=(i<n/2)?i:(n-1-i);v=d*2*A/(n-1);break;
        case 5:d=(i<n/2)?i:(n-1-i);v=A-d*2*A/(n-1);break;
        case 6:v=(long long)(i/(n/4))*A/3;break;
        case 7:v=A-(long long)(i/(n/4))*A/3;break;
        case 8:h=(n-1)/2;d=i-h;v=A-d*d*A/(h*h+1);break;
        case 9:{long long t=(long long)i*256/(n-1);v=t*t*(3*256-2*t)*A/(256LL*256*256);}break;
    }
    if(v<0)v=0;if(v>255)v=255;return(u8)v;
}
static const char *gshape_name[]={"ramp-up","ramp-dn","quad-up","quad-dn","pyramid","valley","stair-up","stair-dn","gauss","s-curve"};
static inline u8 periodic_amp(int s,int q,int P,int A){
    int v=0,h=P/2?P/2:1;
    switch(s){
        case 0:v=(P>1)?q*A/(P-1):0;break;
        case 1:v=(P>1)?A-q*A/(P-1):A;break;
        case 2:v=(q<h)?(q*A/h):((P-1-q)*A/h);break;
        case 3:v=(q<h)?A:0;break;
        case 4:v=(q<P/4)?A:0;break;
        case 5:v=(q==0)?A:0;break;
        case 6:v=(q==h)?A:0;break;
        case 7:v=(int)(A*sin(M_PI*(double)q/P)+0.5);break;
    }
    if(v<0)v=0;if(v>255)v=255;return(u8)v;
}
static const char *pshape_name[]={"saw-up","saw-dn","triangle","sq-50%","sq-25%","impulse-0","impulse-mid","half-sine"};

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
//  9  PRNG-JUMP:       seed(8)+maxstep(4)+op(3)+amp(8)  = 28
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
static const int PAT_OH[] = {22,22,30,27,20,25,21,20,11,28,27, 8,27,26,26,12,21,8,6};

// ── Apply a SearchResult in-place ─────────────────────────────────────────────
static void apply_sr(u8 *blk, int n, const SR *r) {
    int x,op,amp,seed,P,N,shape;
    u8 *orig=NULL;
    if(r->id==12||r->id==14){  // COND-PREV and COND-DELTA need orig
        orig=malloc(n); if(orig) memcpy(orig,blk,n);
    }
    switch(r->id){
        case 0: x=r->p[0];op=r->p[1];amp=r->p[2]; // STRIDE-CONST
            for(int p=0;p<n;p+=x) blk[p]=op_byte(blk[p],(u8)amp,op); break;
        case 1: x=r->p[0];seed=r->p[1];op=r->p[2]; // STRIDE-PRNG-AMP
            {uint32_t st=(uint32_t)seed; for(int p=0;p<n;p+=x) blk[p]=op_byte(blk[p],lcg_byte(&st),op);} break;
        case 2: x=r->p[0];op=r->p[1];{int a1=r->p[2],a2=r->p[3]; // DUAL-STRIDE
            for(int p=0;p<n;p+=2*x) blk[p]=op_byte(blk[p],(u8)a1,op);
            for(int p=x;p<n;p+=2*x) blk[p]=op_byte(blk[p],(u8)a2,op);} break;
        case 3: seed=r->p[0];amp=r->p[1];op=r->p[2]; // PRNG-SELECT
            {int thr_vals[]={25,64,128,192,230};u8 thr=(u8)thr_vals[r->p[3]];
             uint32_t st=(uint32_t)seed;
             for(int i=0;i<n;i++) if(lcg_byte(&st)>=thr) blk[i]=op_byte(blk[i],(u8)amp,op);} break;
        case 4: shape=r->p[0];op=r->p[1];amp=r->p[2]; // GLOBAL-SHAPE
            for(int i=0;i<n;i++) blk[i]=op_byte(blk[i],global_amp(shape,i,n,amp),op); break;
        case 5: P=r->p[0];shape=r->p[1];op=r->p[2];amp=r->p[3]; // PERIODIC-SHAPE
            for(int i=0;i<n;i++) blk[i]=op_byte(blk[i],periodic_amp(shape,i%P,P,amp),op); break;
        case 6: {int set_id=r->p[0];op=r->p[1];amp=r->p[2]; // SPARSE-INDEX
            u8*mask=calloc(n,1);if(!mask)break;
            if(set_id==0){int a=0,b=1;while(b<n){mask[b]=1;int c=a+b;a=b;b=c;}}
            else if(set_id==1){for(int p=1;p<n;p<<=1)mask[p]=1;}
            else if(set_id==2){u8*sv=calloc(n,1);for(int i=2;i<n;i++)sv[i]=1;
                for(int i=2;(long long)i*i<n;i++)if(sv[i])for(int j=i*i;j<n;j+=i)sv[j]=0;
                for(int i=2;i<n;i++)if(sv[i])mask[i]=1;free(sv);}
            else{int pc=set_id-3;for(int i=0;i<n;i++)if(__builtin_popcount(i)==pc)mask[i]=1;}
            for(int i=0;i<n;i++)if(mask[i])blk[i]=op_byte(blk[i],(u8)amp,op);
            free(mask);} break;
        case 7: shape=r->p[0];seed=r->p[1];op=r->p[2]; // PRNG+GRADIENT
            {uint32_t st=(uint32_t)seed;
             for(int i=0;i<n;i++){u8 a=(u8)(global_amp(shape,i,n,128)+lcg_byte(&st));blk[i]=op_byte(blk[i],a,op);}} break;
        case 8: N=r->p[0];op=r->p[1]; // NWAY-STRIDE
            for(int i=0;i<n;i++) blk[i]=op_byte(blk[i],r->amps[i%N],op); break;
        case 9: seed=r->p[0];{int ms=r->p[1];op=r->p[2];amp=r->p[3]; // PRNG-JUMP
            uint32_t st=(uint32_t)seed;int pos=0;
            while(pos<n){blk[pos]=op_byte(blk[pos],(u8)amp,op);pos+=(int)(lcg_byte(&st)%ms)+1;}} break;
        case 10: P=r->p[0];{int mask=r->p[1];op=r->p[2];amp=r->p[3]; // MODULAR-MASK
            for(int i=0;i<n;i++) if((mask>>(i%P))&1) blk[i]=op_byte(blk[i],(u8)amp,op);} break;
        case 11: {int ot=r->p[0]; // DELTA: byte[i] -= byte[i-1], applied right-to-left
            if(ot==0) for(int i=n-1;i>=1;i--) blk[i]=(u8)(blk[i]-blk[i-1]);
            else      for(int i=n-1;i>=1;i--) blk[i]^=blk[i-1];} break;
        case 12: {int k=r->p[0];op=r->p[1]; // COND-PREV: amp keyed by high k bits of previous byte
            if(orig) for(int i=1;i<n;i++) blk[i]=op_byte(blk[i],r->amps[orig[i-1]>>(8-k)],op);} break;
        case 13: {int k=r->p[0];op=r->p[1]; // COND-NEXT: amp keyed by high k bits of next byte
            for(int i=0;i<n-1;i++) blk[i]=op_byte(blk[i],r->amps[blk[i+1]>>(8-k)],op);} break;
        case 14: {int k=r->p[0];op=r->p[1]; // COND-DELTA: amp keyed by high k bits of (prev XOR pprev)
            if(orig) for(int i=2;i<n;i++) blk[i]=op_byte(blk[i],r->amps[(orig[i-1]^orig[i-2])>>(8-k)],op);} break;
        case 15: {int N=r->p[0];int ot=r->p[1]; // STRIDE-DELTA: b'[i] = b[i] OP b[i-N], right-to-left
            if(ot==0) for(int i=n-1;i>=N;i--) blk[i]=(u8)(blk[i]-blk[i-N]);
            else       for(int i=n-1;i>=N;i--) blk[i]^=blk[i-N];} break;
        case 16: {int T=r->p[0]; int amp=r->p[1]; int M=256-T; // THRESH-MAP: rotate [T..255] by amp
            // Reversible: inverse applies rotation by M-(amp%M) within [T..255]
            for(int i=0;i<n;i++)
                if(blk[i]>=(u8)T) blk[i]=(u8)(T+(blk[i]-T+amp)%M);} break;
        case 17: {int k=r->p[0]; // BIT-ROTATE: rotate all bytes left by k bits
            // Reversible: inverse rotates by (8-k)
            for(int i=0;i<n;i++) blk[i]=(u8)((blk[i]<<k)|(blk[i]>>(8-k)));} break;
        case 18: {int dir=r->p[0]; // GRAY: dir=0 encode (b^(b>>1)), dir=1 decode
            // Encode and decode are each other's inverse
            if(dir==0){for(int i=0;i<n;i++) blk[i]^=(blk[i]>>1);}
            else{for(int i=0;i<n;i++){u8 v=blk[i];v^=(v>>4);v^=(v>>2);v^=(v>>1);blk[i]=v;}}
            } break;
    }
    free(orig);
}

// ── Helpers for search functions ──────────────────────────────────────────────
// MUL(2) needs odd amp; ADDLO(3) amp 0-15; GFMUL(5) amp >= 2
#define OP_RANGE(op,alo,ahi) int alo=1,ahi=255; \
    if(op==2)alo=3; if(op==3)ahi=15; if(op==5)alo=2;
#define SKIP_OP(op,amp) if(op==2&&(amp&1)==0) continue;

// ── Search functions ──────────────────────────────────────────────────────────

static SR search_stride_const(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=0; strcpy(r.name,"STRIDE-CONST"); r.entropy=base; r.overhead=PAT_OH[0];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    double be=base;int bx=0,bop=0,bamp=0;
    #pragma omp parallel
    {double le=base;int lx=0,lop=0,la=0;
     #pragma omp for schedule(dynamic,1)
     for(int x=1;x<=MAX_STRIDE;x++){
         int sf[256]={0};for(int p=0;p<n;p+=x)sf[blk[p]]++;
         int nsf[256];for(int v=0;v<256;v++)nsf[v]=bfreq[v]-sf[v];
         for(int op=0;op<N_OPS;op++){OP_RANGE(op,alo,ahi)
             for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
                 int tf[256],ff[256];hist_transform(sf,tf,op,amp);
                 for(int v=0;v<256;v++)ff[v]=nsf[v]+tf[v];
                 double e=entropy_from_hist(ff,n);if(e<le){le=e;lx=x;lop=op;la=amp;}}}
     }
     #pragma omp critical
     if(le<be){be=le;bx=lx;bop=lop;bamp=la;}}
    r.entropy=be;r.p[0]=bx;r.p[1]=bop;r.p[2]=bamp;
    snprintf(r.name,sizeof(r.name),"STRIDE-CONST %s s=%d a=%d",opname[bop],bx,bamp);
    return r;
}

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

static SR search_dual_stride(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=2; r.entropy=base; r.overhead=PAT_OH[2];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    double be=base;int bx=0,bop=0,ba1=0,ba2=0;
    #pragma omp parallel
    {double le=base;int lx=0,lop=0,la1=0,la2=0;
     #pragma omp for schedule(dynamic,1)
     for(int x=1;x<=MAX_STRIDE;x++){
         int ef[256]={0},of_[256]={0};
         for(int p=0;p<n;p+=2*x)ef[blk[p]]++;
         for(int p=x;p<n;p+=2*x)of_[blk[p]]++;
         int nf[256];for(int v=0;v<256;v++)nf[v]=bfreq[v]-ef[v]-of_[v];
         for(int op=0;op<N_OPS;op++){OP_RANGE(op,alo,ahi)
             double p1b=le;int p1a=0;
             for(int a1=alo;a1<=ahi;a1++){SKIP_OP(op,a1)
                 int tf[256],ff[256];hist_transform(ef,tf,op,a1);
                 for(int v=0;v<256;v++)ff[v]=nf[v]+of_[v]+tf[v];
                 double e=entropy_from_hist(ff,n);if(e<p1b){p1b=e;p1a=a1;}}
             if(!p1a)continue;
             int eft[256];hist_transform(ef,eft,op,p1a);
             for(int a2=alo;a2<=ahi;a2++){SKIP_OP(op,a2)
                 int tf[256],ff[256];hist_transform(of_,tf,op,a2);
                 for(int v=0;v<256;v++)ff[v]=nf[v]+eft[v]+tf[v];
                 double e=entropy_from_hist(ff,n);if(e<le){le=e;lx=x;lop=op;la1=p1a;la2=a2;}}}}
     #pragma omp critical
     if(le<be){be=le;bx=lx;bop=lop;ba1=la1;ba2=la2;}}
    r.entropy=be;r.p[0]=bx;r.p[1]=bop;r.p[2]=ba1;r.p[3]=ba2;
    snprintf(r.name,sizeof(r.name),"DUAL-STRIDE %s s=%d a1=%d a2=%d",opname[bop],bx,ba1,ba2);
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

static SR search_global_shapes(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=4; r.entropy=base; r.overhead=PAT_OH[4];
    double be=base;int bsh=0,bop=0,bamp=0;
    #pragma omp parallel
    {double le=base;int lsh=0,lop=0,la=0;int ff[256];
     #pragma omp for schedule(dynamic,1)
     for(int sh=0;sh<N_GSHAPES;sh++)for(int A=1;A<=255;A++)for(int op=0;op<N_OPS;op++){
         memset(ff,0,sizeof(ff));
         for(int i=0;i<n;i++)ff[op_byte(blk[i],global_amp(sh,i,n,A),op)]++;
         double e=entropy_from_hist(ff,n);if(e<le){le=e;lsh=sh;lop=op;la=A;}}
     #pragma omp critical
     if(le<be){be=le;bsh=lsh;bop=lop;bamp=la;}}
    r.entropy=be;r.p[0]=bsh;r.p[1]=bop;r.p[2]=bamp;
    snprintf(r.name,sizeof(r.name),"GLOBAL-SHAPE %s %s A=%d",gshape_name[bsh],opname[bop],bamp);
    return r;
}

static SR search_periodic_shapes(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=5; r.entropy=base; r.overhead=PAT_OH[5];
    double be=base;int bP=0,bsh=0,bop=0,bamp=0;
    #pragma omp parallel
    {double le=base;int lP=0,lsh=0,lop=0,la=0;
     #pragma omp for schedule(dynamic,1)
     for(int P=2;P<=MAX_STRIDE;P++){
         int (*sf)[256]=malloc(P*256*sizeof(int));if(!sf)continue;
         memset(sf,0,P*256*sizeof(int));
         for(int i=0;i<n;i++)sf[i%P][blk[i]]++;
         for(int A=1;A<=255;A++)for(int sh=0;sh<N_PSHAPES;sh++)for(int op=0;op<N_OPS;op++){
             int ff[256]={0};
             for(int q=0;q<P;q++){u8 amp=periodic_amp(sh,q,P,A);int tf[256];
                 hist_transform(sf[q],tf,op,amp);for(int v=0;v<256;v++)ff[v]+=tf[v];}
             double e=entropy_from_hist(ff,n);if(e<le){le=e;lP=P;lsh=sh;lop=op;la=A;}}
         free(sf);}
     #pragma omp critical
     if(le<be){be=le;bP=lP;bsh=lsh;bop=lop;bamp=la;}}
    r.entropy=be;r.p[0]=bP;r.p[1]=bsh;r.p[2]=bop;r.p[3]=bamp;
    snprintf(r.name,sizeof(r.name),"PERIODIC %s %s P=%d A=%d",pshape_name[bsh],opname[bop],bP,bamp);
    return r;
}

static SR search_sparse_indexed(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=6; r.entropy=base; r.overhead=PAT_OH[6];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    int fib[64],nfib=0;{int a=0,b=1;while(b<n&&nfib<64){fib[nfib++]=b;int c=a+b;a=b;b=c;}}
    int p2[20],np2=0;{for(int p=1;p<n;p<<=1)p2[np2++]=p;}
    u8 *sv=calloc(n,1);int *primes=NULL;int nprimes=0;
    if(sv){for(int i=2;i<n;i++)sv[i]=1;
        for(int i=2;(long long)i*i<n;i++)if(sv[i])for(int j=i*i;j<n;j+=i)sv[j]=0;
        for(int i=2;i<n;i++)if(sv[i])nprimes++;
        primes=malloc(nprimes*sizeof(int));int j=0;for(int i=2;i<n;i++)if(sv[i])primes[j++]=i;
        free(sv);}
    int *pcg[17];int pcn[17]={0};
    for(int k=0;k<=16;k++){pcg[k]=malloc((n+1)*sizeof(int));if(!pcg[k]){for(int m=0;m<k;m++)free(pcg[m]);free(primes);r.entropy=base;return r;}}
    for(int i=0;i<n;i++){int pc=__builtin_popcount(i);if(pc<=16)pcg[pc][pcn[pc]++]=i;}

    double be=base;int bset=0,bop=0,bamp=0;

    #define TRY(set_id,idx_arr,cnt) do{ \
        if((cnt)<=0)break; \
        int sel[256]={0},unsel[256]; \
        for(int j=0;j<(cnt);j++)sel[blk[(idx_arr)[j]]]++; \
        for(int v=0;v<256;v++)unsel[v]=bfreq[v]-sel[v]; \
        for(int op=0;op<N_OPS;op++){OP_RANGE(op,alo,ahi) \
            for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp) \
                int tf[256],ff[256];hist_transform(sel,tf,op,amp); \
                for(int v=0;v<256;v++)ff[v]=unsel[v]+tf[v]; \
                double e=entropy_from_hist(ff,n); \
                if(e<be){be=e;bset=(set_id);bop=op;bamp=amp;}}} \
    }while(0)
    TRY(0,fib,nfib); TRY(1,p2,np2); TRY(2,primes,nprimes);
    for(int k=0;k<=16;k++) TRY(3+k,pcg[k],pcn[k]);
    #undef TRY

    r.entropy=be;r.p[0]=bset;r.p[1]=bop;r.p[2]=bamp;
    const char *snames[]={"fib","pow2","primes","pc0","pc1","pc2","pc3","pc4","pc5","pc6","pc7","pc8","pc9","pc10","pc11","pc12","pc13","pc14","pc15","pc16"};
    snprintf(r.name,sizeof(r.name),"SPARSE %s %s a=%d",snames[bset<20?bset:0],opname[bop],bamp);
    free(primes);for(int k=0;k<=16;k++)free(pcg[k]);
    return r;
}

static SR search_prng_gradient(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=7; r.entropy=base; r.overhead=PAT_OH[7];
    double be=base;int bsh=0,bseed=0,bop=0;
    #pragma omp parallel
    {double le=base;int lsh=0,ls=0,lop=0;int ff[256];
     #pragma omp for schedule(dynamic,1)
     for(int sh=0;sh<N_GSHAPES;sh++)for(int seed=0;seed<256;seed++)for(int op=0;op<N_OPS;op++){
         memset(ff,0,sizeof(ff));uint32_t st=(uint32_t)seed;
         for(int i=0;i<n;i++){u8 a=(u8)(global_amp(sh,i,n,128)+lcg_byte(&st));ff[op_byte(blk[i],a,op)]++;}
         double e=entropy_from_hist(ff,n);if(e<le){le=e;lsh=sh;ls=seed;lop=op;}}
     #pragma omp critical
     if(le<be){be=le;bsh=lsh;bseed=ls;bop=lop;}}
    r.entropy=be;r.p[0]=bsh;r.p[1]=bseed;r.p[2]=bop;
    snprintf(r.name,sizeof(r.name),"PRNG+GRAD %s+n %s seed=%d",gshape_name[bsh],opname[bop],bseed);
    return r;
}

static SR search_nway_stride(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=8; r.entropy=base; r.overhead=PAT_OH[8];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    double be=base;int bN=0,bop=0;u8 bamps[8]={0};
    #pragma omp parallel
    {double le=base;int lN=0,lop=0;u8 la[8]={0};
     #pragma omp for schedule(dynamic,1)
     for(int N=3;N<=8;N++){
         int ch[8][256];memset(ch,0,sizeof(ch));
         for(int i=0;i<n;i++)ch[i%N][blk[i]]++;
         for(int op=0;op<N_OPS;op++){OP_RANGE(op,alo,ahi)
             u8 amps[8];for(int k=0;k<N;k++)amps[k]=(u8)alo;
             int combined[256]={0};
             for(int k=0;k<N;k++){int tf[256];hist_transform(ch[k],tf,op,amps[k]);for(int v=0;v<256;v++)combined[v]+=tf[v];}
             for(int pass=0;pass<2;pass++)for(int k=0;k<N;k++){
                 int tc[256],wk[256];hist_transform(ch[k],tc,op,amps[k]);
                 for(int v=0;v<256;v++)wk[v]=combined[v]-tc[v];
                 double bke=1e30;u8 bka=amps[k];
                 for(int amp=alo;amp<=ahi;amp++){SKIP_OP(op,amp)
                     int tf[256],ff[256];hist_transform(ch[k],tf,op,amp);
                     for(int v=0;v<256;v++)ff[v]=wk[v]+tf[v];
                     double e=entropy_from_hist(ff,n);if(e<bke){bke=e;bka=(u8)amp;}}
                 int tn[256];hist_transform(ch[k],tn,op,bka);
                 for(int v=0;v<256;v++)combined[v]=wk[v]+tn[v];amps[k]=bka;}
             double e=entropy_from_hist(combined,n);
             if(e<le){le=e;lN=N;lop=op;for(int k=0;k<N;k++)la[k]=amps[k];}}}
     #pragma omp critical
     if(le<be){be=le;bN=lN;bop=lop;for(int k=0;k<bN;k++)bamps[k]=la[k];}}
    r.entropy=be;r.p[0]=bN;r.p[1]=bop;for(int k=0;k<bN;k++)r.amps[k]=bamps[k];
    r.overhead=5+(11+bN*8);
    snprintf(r.name,sizeof(r.name),"NWAY N=%d %s",bN,opname[bop]);
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
     for(int seed=0;seed<256;seed++)for(int ms=1;ms<=16;ms++){
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

// DELTA: encode b[i] as (b[i] - b[i-1]) mod 256, or (b[i] XOR b[i-1]).
// Apply right-to-left so b[i-1] is always the original value when read.
// Invert left-to-right (standard scan reconstruction).
// For smooth/correlated data (audio, images) consecutive bytes are similar,
// so differences cluster near 0 — causing real collisions in the output histogram.
static SR search_delta(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=11; r.entropy=base; r.overhead=PAT_OH[11];
    double be=base; int bot=0;
    // try SUB delta
    {int ff[256]={0}; ff[blk[0]]++;
     for(int i=1;i<n;i++) ff[(u8)(blk[i]-blk[i-1])]++;
     double e=entropy_from_hist(ff,n); if(e<be){be=e;bot=0;}}
    // try XOR delta
    {int ff[256]={0}; ff[blk[0]]++;
     for(int i=1;i<n;i++) ff[blk[i]^blk[i-1]]++;
     double e=entropy_from_hist(ff,n); if(e<be){be=e;bot=1;}}
    r.entropy=be; r.p[0]=bot;
    snprintf(r.name,sizeof(r.name),"DELTA %s",bot?"XOR":"SUB");
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
    // try k=1..8: 2,4,...,256 groups keyed on high k bits of previous byte
    for(int k=1;k<=8;k++){
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

// COND-NEXT: amp keyed on high k bits of the NEXT byte.
// Apply left→right (next byte untouched when processing i). Invert right→left.
static SR search_cond_next(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=13; r.entropy=base; r.overhead=PAT_OH[13];
    double be=base; int bk=1,bop=0; u8 bamps[8]={0};
    for(int k=1;k<=3;k++){
        int N=1<<k;
        int gh[8][256]; memset(gh,0,sizeof(gh));
        for(int i=0;i<n-1;i++) gh[blk[i+1]>>(8-k)][blk[i]]++;
        for(int op=0;op<N_OPS;op++){
            u8 amps[8]={0};
            // byte n-1 has no context, stays unchanged
            double e=cond_greedy((const int(*)[256])gh,N,n,blk[n-1],op,amps);
            if(e<be){be=e;bk=k;bop=op;memcpy(bamps,amps,N*sizeof(u8));}}
    }
    r.entropy=be; r.p[0]=bk; r.p[1]=bop;
    {int N=1<<bk; memcpy(r.amps,bamps,N*sizeof(u8)); r.overhead=5+2+3+N*8;}
    snprintf(r.name,sizeof(r.name),"COND-NEXT k=%d %s",bk,opname[bop]);
    return r;
}

// COND-DELTA: amp keyed on high k bits of (byte[i-1] XOR byte[i-2]).
// Context = rate of change between the two previous bytes.
// Reversible: during decompression forward pass, i-1 and i-2 are already decoded.
static SR search_cond_delta(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=14; r.entropy=base; r.overhead=PAT_OH[14];
    double be=base; int bk=1,bop=0; u8 bamps[8]={0};
    for(int k=1;k<=3;k++){
        int N=1<<k;
        int gh[8][256]; memset(gh,0,sizeof(gh));
        for(int i=2;i<n;i++) gh[(blk[i-1]^blk[i-2])>>(8-k)][blk[i]]++;
        for(int op=0;op<N_OPS;op++){OP_RANGE(op,alo,ahi)
            u8 amps[8]; for(int g=0;g<N;g++) amps[g]=(u8)alo;
            // bytes 0 and 1 have no context, stay unchanged
            int combined[256]={0}; combined[blk[0]]++; combined[blk[1]]++;
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
            double e=entropy_from_hist(combined,n);
            if(e<be){be=e;bk=k;bop=op;memcpy(bamps,amps,N*sizeof(u8));}}}
    r.entropy=be; r.p[0]=bk; r.p[1]=bop;
    {int N=1<<bk; memcpy(r.amps,bamps,N*sizeof(u8)); r.overhead=5+2+3+N*8;}
    snprintf(r.name,sizeof(r.name),"COND-DELTA k=%d %s",bk,opname[bop]);
    return r;
}

// STRIDE-DELTA: b'[i] = b[i] - b[i-N]  (or XOR), applied right-to-left.
// Right-to-left guarantees b[i-N] is always original when read — no orig[] needed.
// Invert: left-to-right scan reconstruction (b[i] += b[i-N] or b[i] ^= b[i-N]).
// Equivalent to: deinterleave N channels, apply DELTA within each channel, leave interleaved.
// For real multi-channel data (stereo N=2, RGB N=3, RGBA N=4, etc.) this finds
// within-channel correlations that stride-1 DELTA misses.
static SR search_stride_delta(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=15; r.entropy=base; r.overhead=PAT_OH[15];
    double be=base; int bN=2,bot=0;
    for(int N=2;N<=MAX_STRIDE;N++){
        // SUB delta at stride N
        {int ff[256]={0};
         for(int i=0;i<N&&i<n;i++) ff[blk[i]]++;
         for(int i=N;i<n;i++) ff[(u8)(blk[i]-blk[i-N])]++;
         double e=entropy_from_hist(ff,n); if(e<be){be=e;bN=N;bot=0;}}
        // XOR delta at stride N
        {int ff[256]={0};
         for(int i=0;i<N&&i<n;i++) ff[blk[i]]++;
         for(int i=N;i<n;i++) ff[blk[i]^blk[i-N]]++;
         double e=entropy_from_hist(ff,n); if(e<be){be=e;bN=N;bot=1;}}
    }
    r.entropy=be; r.p[0]=bN; r.p[1]=bot;
    snprintf(r.name,sizeof(r.name),"STRIDE-DELTA N=%d %s",bN,bot?"XOR":"SUB");
    return r;
}

// THRESH-MAP: rotate the value range [T..255] by amp positions (mod 256-T).
// Bytes < T are unchanged. Bytes >= T are mapped to T + (v-T+amp)%(256-T).
// This is always a bijection: rotation within a closed set.
// Inverse: rotation by (256-T - amp%(256-T)).
static SR search_thresh_map(const u8*blk, int n, double base){
    SR r; memset(&r,0,sizeof(r)); r.id=16; r.entropy=base; r.overhead=PAT_OH[16];
    int bfreq[256]={0}; for(int i=0;i<n;i++) bfreq[blk[i]]++;
    double be=base; int bT=0,bamp=0;
    #pragma omp parallel
    {double le=base; int lT=0,lamp=0;
     #pragma omp for schedule(dynamic,4)
     for(int T=0;T<=254;T++){
         int M=256-T;
         int ff[256];
         for(int amp=1;amp<M;amp++){
             for(int v=0;v<T;v++) ff[v]=bfreq[v];
             for(int v=T;v<256;v++) ff[T+(v-T+amp)%M]=bfreq[v];
             double e=entropy_from_hist(ff,n);
             if(e<le){le=e;lT=T;lamp=amp;}
         }
     }
     #pragma omp critical
     if(le<be){be=le;bT=lT;bamp=lamp;}}
    r.entropy=be; r.p[0]=bT; r.p[1]=bamp;
    snprintf(r.name,sizeof(r.name),"THRESH-MAP T=%d a=%d",bT,bamp);
    return r;
}

// BIT-ROTATE: rotate every byte left by k bits (k=1..7).
// Inverse: rotate right by k, i.e. rotate left by (8-k).
static SR search_bit_rotate(const u8*blk, int n, double base){
    SR r; memset(&r,0,sizeof(r)); r.id=17; r.entropy=base; r.overhead=PAT_OH[17];
    double be=base; int bk=0;
    for(int k=1;k<=7;k++){
        int ff[256]={0};
        for(int i=0;i<n;i++) ff[(u8)((blk[i]<<k)|(blk[i]>>(8-k)))]++;
        double e=entropy_from_hist(ff,n); if(e<be){be=e;bk=k;}
    }
    r.entropy=be; r.p[0]=bk;
    snprintf(r.name,sizeof(r.name),"BIT-ROTATE k=%d",bk);
    return r;
}

// GRAY: apply Gray encoding (b^(b>>1)) or Gray decoding to every byte.
// Encode and decode are each other's inverse, so both are self-consistently reversible.
static SR search_gray(const u8*blk, int n, double base){
    SR r; memset(&r,0,sizeof(r)); r.id=18; r.entropy=base; r.overhead=PAT_OH[18];
    double be=base; int bdir=0;
    {int ff[256]={0};
     for(int i=0;i<n;i++) ff[blk[i]^(blk[i]>>1)]++;
     double e=entropy_from_hist(ff,n); if(e<be){be=e;bdir=0;}}
    {int ff[256]={0};
     for(int i=0;i<n;i++){u8 v=blk[i];v^=(v>>4);v^=(v>>2);v^=(v>>1);ff[v]++;}
     double e=entropy_from_hist(ff,n); if(e<be){be=e;bdir=1;}}
    r.entropy=be; r.p[0]=bdir;
    snprintf(r.name,sizeof(r.name),"GRAY %s",bdir?"decode":"encode");
    return r;
}

// ── Run all searches ──────────────────────────────────────────────────────────
static void run_all(const u8 *blk, int n, double base, SR *out) {
    out[0]  = search_stride_const    (blk,n,base);
    out[1]  = search_stride_prng_amp (blk,n,base);
    out[2]  = search_dual_stride     (blk,n,base);
    out[3]  = search_prng_select     (blk,n,base);
    out[4]  = search_global_shapes   (blk,n,base);
    out[5]  = search_periodic_shapes (blk,n,base);
    out[6]  = search_sparse_indexed  (blk,n,base);
    out[7]  = search_prng_gradient   (blk,n,base);
    out[8]  = search_nway_stride     (blk,n,base);
    out[9]  = search_prng_jump       (blk,n,base);
    out[10] = search_modular_mask    (blk,n,base);
    out[11] = search_delta           (blk,n,base);
    out[12] = search_cond_prev       (blk,n,base);
    out[13] = search_cond_next       (blk,n,base);
    out[14] = search_cond_delta      (blk,n,base);
    out[15] = search_stride_delta    (blk,n,base);
    out[16] = search_thresh_map      (blk,n,base);
    out[17] = search_bit_rotate      (blk,n,base);
    out[18] = search_gray            (blk,n,base);
}

// ── Byte dump ────────────────────────────────────────────────────────────────
static void print_bytes(const u8 *data, int len, const char *label) {
    printf("\n=== %s (%d bytes) ===\n", label, len);
    for(int i=0;i<len;i++) printf("%d ", data[i]);
    printf("\n");
}

// ── BWT fallback ──────────────────────────────────────────────────────────────
// Called when every regular transform gives negative net.
// Applies Burrows-Wheeler Transform and re-runs all searches on the result.
// BWT groups identical contexts together, making COND-* and DELTA patterns
// far more effective on structured data that stumped the per-byte searches.
// Overhead: 5 (pattern id) + 16 (row index, since BLOCK_SIZE = 2^16) = 21 bits.
#define BWT_OH 21

static const u8 *g_bwt_src;
static int       g_bwt_n;
static int bwt_cmp(const void *a, const void *b){
    int ia=*(const int*)a, ib=*(const int*)b;
    const u8 *s=g_bwt_src; int n=g_bwt_n;
    for(int len=n;len--;){
        int d=(int)s[ia]-(int)s[ib]; if(d) return d;
        if(++ia==n) ia=0; if(++ib==n) ib=0;
    }
    return 0;
}

// Forward BWT of blk[0..n-1] in-place using cyclic rotations.
// Returns the row index of the original string (needed for inverse), or -1 on alloc failure.
static int bwt_forward(u8 *blk, int n){
    int *sa=malloc(n*sizeof(int)); u8 *out=malloc(n);
    if(!sa||!out){free(sa);free(out);return -1;}
    for(int i=0;i<n;i++) sa[i]=i;
    g_bwt_src=blk; g_bwt_n=n;
    qsort(sa,n,sizeof(int),bwt_cmp);
    int idx=-1;
    for(int i=0;i<n;i++){out[i]=blk[sa[i]?sa[i]-1:n-1]; if(!sa[i]) idx=i;}
    memcpy(blk,out,n); free(sa); free(out);
    return idx;
}

static int try_bwt(u8*blk,int n,double*base,int pass,double*tnet){
    printf("  (stuck — trying BWT)\n"); fflush(stdout);
    u8 *tmp=malloc(n); if(!tmp) return 0;
    memcpy(tmp,blk,n);
    int idx=bwt_forward(tmp,n);
    if(idx<0){free(tmp);return 0;}
    SR res[N_PATTERNS];
    run_all(tmp,n,*base,res);
    int best_i=-1; double best_net=0;
    for(int i=0;i<N_PATTERNS;i++){
        double net=(*base-res[i].entropy)*n-BWT_OH-res[i].overhead;
        if(net>best_net){best_net=net;best_i=i;}
    }
    if(best_i<0){printf("  (BWT did not improve — done)\n");free(tmp);return 0;}
    memcpy(blk,tmp,n); free(tmp);
    double before=*base;
    apply_sr(blk,n,&res[best_i]);
    *base=byte_entropy(blk,n);
    *tnet+=best_net;
    printf("Pass %d: BWT(idx=%d)+%s  entropy %.6f->%.6f  (net=%.1f)\n",
           pass,idx,res[best_i].name,before,*base,best_net);
    fflush(stdout);
    return 1;
}

// ── Main ──────────────────────────────────────────────────────────────────────
#define N_STATS 1

int main(void) {
    init_gf256();
    int nt=omp_get_max_threads()-1; if(nt<1)nt=1;
    omp_set_num_threads(nt);

    printf("Running %d block x %d bytes  (threads=%d)\n",
           N_STATS, BLOCK_SIZE, nt);

    for(int bi=0;bi<N_STATS;bi++){
        u8 *blk=malloc(BLOCK_SIZE);
        if(!blk||!fill_random(blk,BLOCK_SIZE)){free(blk);continue;}

        print_bytes(blk, 256 * 5, "BEFORE transforms");

        double base=byte_entropy(blk,BLOCK_SIZE);
        double start_entropy=base;
        double total_net=0.0;
        SR results[N_PATTERNS];
        int pass=0;

        while(1){
            pass++;
            run_all(blk,BLOCK_SIZE,base,results);

            int best_idx=-1; double best_net=-1e30;
            for(int i=0;i<N_PATTERNS;i++){
                double net=(base-results[i].entropy)*BLOCK_SIZE-results[i].overhead;
                if(net>best_net){best_net=net;best_idx=i;}
            }
            if(best_net<=0||best_idx<0){
                if(!try_bwt(blk,BLOCK_SIZE,&base,pass,&total_net)) break;
                continue;  // loop: run_all again on BWT'd data
            }

            double before=base;
            apply_sr(blk,BLOCK_SIZE,&results[best_idx]);
            base=byte_entropy(blk,BLOCK_SIZE);
            total_net+=best_net;
            printf("Pass %d: %s  entropy %.6f -> %.6f  (net=%.1f)\n",
                   pass, results[best_idx].name, before, base, best_net);
            fflush(stdout);
        }

        printf("\nSummary: entropy %.6f -> %.6f  total net=%.1f bits\n",
               start_entropy, base, total_net);

        print_bytes(blk, 256 * 5, "AFTER transforms");
        free(blk);
    }

    return 0;
}
