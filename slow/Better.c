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
#define N_OPS       8
#define N_GSHAPES   10
#define N_PSHAPES   8
#define N_PATTERNS  20

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
        case 2:{int r=amp&7;if(!r)r=1;return(u8)((v<<r)|(v>>(8-r)));}
        case 3:return(u8)(v*(amp|1));
        case 4:return(u8)((((v>>4)+amp)&0xF)<<4|(v&0x0F));
        case 5:return(u8)((v&0xF0)|((v+amp)&0x0F));
        case 6:{u8 s=(u8)((v<<4)|(v>>4));return s^amp;}
        case 7:{u8 a=amp<2?2:amp;return gf_mul(v,a);}
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
static const char *opname[]={"ADD","XOR","ROL","MUL","ADDHI","ADDLO","SWXOR","GFMUL"};

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
// Stores everything needed to report AND replay a transform.
typedef struct {
    int    id;          // pattern index
    char   name[48];    // human-readable
    double entropy;     // achieved entropy
    int    overhead;    // bits needed to store the transform parameters
    int    p[8];        // integer params (stride, op, amp, seed, period, etc.)
    u8     amps[8];     // byte params (NWAY per-channel amps)
} SR;

// Overhead (bits) = 5 (pattern type ID) + parameter bits
// 0  STRIDE-CONST:     stride(6)+op(3)+amp(8)          = 22
// 1  STRIDE-PRNG-AMP:  stride(6)+seed(8)+op(3)         = 22
// 2  DUAL-STRIDE:      stride(6)+op(3)+a1(8)+a2(8)     = 30
// 3  PRNG-SELECT:      seed(8)+amp(8)+op(3)+thr(3)      = 27
// 4  FULL-PRNG:        seed(8)+op(3)                    = 16
// 5  GLOBAL-SHAPE:     shape(4)+op(3)+amp(8)            = 20
// 6  PERIODIC-SHAPE:   P(6)+shape(3)+op(3)+amp(8)       = 25
// 7  STEP-BLOCKS:      chunk_idx(4)+step(8)+op(3)+dir(1)= 21
// 8  SPARSE-INDEX:     set_id(5)+op(3)+amp(8)           = 21
// 9  PRNG+GRADIENT:    shape(4)+seed(8)+op(3)           = 20
// 10 NWAY-STRIDE:      N(3)+op(3)+(N*8 for amps)        = 11+N*8 (dynamic)
// 11 ROLLING-DIFF:     op(3)+K(6)                       = 14
// 12 NEIGHBOR-OP:      op(3)+K(5)+C(8)                  = 21
// 13 LOCAL-DELTA:      mode(1)+K(6)                     = 12
// 14 POSITION-HASH:    prime_idx(5)+seed(8)+op(3)       = 21
// 15 PAIR-CROSS:       op(3)+D(6)                       = 14
// 16 PRNG-JUMP:        seed(8)+maxstep(4)+op(3)+amp(8)  = 28
// 17 PRNG-KEYED-POS:   seed(8)+op(3)                    = 16
// 18 MODULAR-MASK:     P(3)+mask(8)+op(3)+amp(8)        = 27
// 19 STRIDE-NEIGHBOR:  stride(6)+op(3)                  = 14
static const int PAT_OH[] = {22,22,30,27,16,20,25,21,21,20,11,14,21,12,21,14,28,16,27,14};

// ── Apply a SearchResult in-place ─────────────────────────────────────────────
static void apply_sr(u8 *blk, int n, const SR *r) {
    int x,op,amp,seed,K,C,D,P,N,shape;
    u8 *orig=NULL;
    // Patterns that use a neighbor value need the original data
    if(r->id==11||r->id==12||r->id==13||r->id==19){
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
        case 4: seed=r->p[0];op=r->p[1]; // FULL-PRNG
            {uint32_t st=(uint32_t)seed; for(int i=0;i<n;i++) blk[i]=op_byte(blk[i],lcg_byte(&st),op);} break;
        case 5: shape=r->p[0];op=r->p[1];amp=r->p[2]; // GLOBAL-SHAPE
            for(int i=0;i<n;i++) blk[i]=op_byte(blk[i],global_amp(shape,i,n,amp),op); break;
        case 6: P=r->p[0];shape=r->p[1];op=r->p[2];amp=r->p[3]; // PERIODIC-SHAPE
            for(int i=0;i<n;i++) blk[i]=op_byte(blk[i],periodic_amp(shape,i%P,P,amp),op); break;
        case 7: {int C2=r->p[0],step=r->p[1];op=r->p[2];int dir=r->p[3]; // STEP-BLOCKS
            int nc=n/C2;
            for(int k=0;k<nc;k++){u8 a=(u8)(((dir?nc-1-k:k)*step)&0xFF);for(int i=k*C2;i<(k+1)*C2;i++) blk[i]=op_byte(blk[i],a,op);}
            for(int i=(n/C2)*C2;i<n;i++) blk[i]=blk[i];} break;
        case 8: {int set_id=r->p[0];op=r->p[1];amp=r->p[2]; // SPARSE-INDEX
            u8*mask=calloc(n,1);if(!mask)break;
            if(set_id==0){int a=0,b=1;while(b<n){mask[b]=1;int c=a+b;a=b;b=c;}}
            else if(set_id==1){for(int p=1;p<n;p<<=1)mask[p]=1;}
            else if(set_id==2){u8*sv=calloc(n,1);for(int i=2;i<n;i++)sv[i]=1;
                for(int i=2;(long long)i*i<n;i++)if(sv[i])for(int j=i*i;j<n;j+=i)sv[j]=0;
                for(int i=2;i<n;i++)if(sv[i])mask[i]=1;free(sv);}
            else{int pc=set_id-3;for(int i=0;i<n;i++)if(__builtin_popcount(i)==pc)mask[i]=1;}
            for(int i=0;i<n;i++)if(mask[i])blk[i]=op_byte(blk[i],(u8)amp,op);
            free(mask);} break;
        case 9: shape=r->p[0];seed=r->p[1];op=r->p[2]; // PRNG+GRADIENT
            {uint32_t st=(uint32_t)seed;
             for(int i=0;i<n;i++){u8 a=(u8)(global_amp(shape,i,n,128)+lcg_byte(&st));blk[i]=op_byte(blk[i],a,op);}} break;
        case 10: N=r->p[0];op=r->p[1]; // NWAY-STRIDE
            for(int i=0;i<n;i++) blk[i]=op_byte(blk[i],r->amps[i%N],op); break;
        case 11: K=r->p[0];op=r->p[1]; // ROLLING-DIFF (needs orig)
            if(orig){for(int i=K;i<n;i++) blk[i]=op_byte(orig[i],orig[i-K],op);} break;
        case 12: K=r->p[0];C=r->p[1];op=r->p[2]; // NEIGHBOR-OP (needs orig)
            if(orig){for(int i=K;i<n;i++) blk[i]=op_byte(orig[i],(u8)(orig[i-K]+C),op);} break;
        case 13: {int xm=r->p[0];K=r->p[1]; // LOCAL-DELTA (needs orig)
            if(orig){for(int i=0;i<n;i++){
                if(i%K==0) blk[i]=orig[i];
                else{u8 d=xm?(orig[i]^orig[i-1]):(u8)(orig[i]-orig[i-1]);blk[i]=d;}}}} break;
        case 14: {int prime=r->p[0];seed=r->p[1];op=r->p[2]; // POSITION-HASH
            for(int i=0;i<n;i++) blk[i]=op_byte(blk[i],(u8)((i*prime+seed)&0xFF),op);} break;
        case 15: op=r->p[0];D=r->p[1]; // PAIR-CROSS
            {int pair=2*D,fp=(n/pair)*pair;
             for(int bi=0;bi<fp;bi+=pair)
                for(int j=0;j<D;j++) blk[bi+j]=op_byte(blk[bi+j],blk[bi+D+j],op);} break;
        case 16: seed=r->p[0];{int ms=r->p[1];op=r->p[2];amp=r->p[3]; // PRNG-JUMP
            uint32_t st=(uint32_t)seed;int pos=0;
            while(pos<n){blk[pos]=op_byte(blk[pos],(u8)amp,op);pos+=(int)(lcg_byte(&st)%ms)+1;}} break;
        case 17: seed=r->p[0];op=r->p[1]; // PRNG-KEYED-POS
            for(int i=0;i<n;i++){uint32_t h=(uint32_t)(i^(seed*2654435761u));h^=h>>16;h*=0x45d9f3b;h^=h>>16;
                blk[i]=op_byte(blk[i],(u8)(h>>24),op);} break;
        case 18: P=r->p[0];{int mask=r->p[1];op=r->p[2];amp=r->p[3]; // MODULAR-MASK
            for(int i=0;i<n;i++) if((mask>>(i%P))&1) blk[i]=op_byte(blk[i],(u8)amp,op);} break;
        case 19: x=r->p[0];op=r->p[1]; // STRIDE-NEIGHBOR (needs orig)
            if(orig){int cnt=0;for(int p=0;p<n;p+=x)cnt++;
                int *pos=malloc(cnt*sizeof(int));if(!pos)break;
                int k=0;for(int p=0;p<n;p+=x)pos[k++]=p;
                for(int j=0;j<cnt;j++) blk[pos[j]]=op_byte(orig[pos[j]],orig[pos[(j+1)%cnt]],op);
                free(pos);} break;
    }
    free(orig);
}

// ── Helpers for search functions ──────────────────────────────────────────────
#define OP_RANGE(op,alo,ahi) int alo=1,ahi=255; \
    if(op==2)ahi=7; if(op==3)alo=3; if(op==4||op==5)ahi=15; if(op==7)alo=2;
#define SKIP_OP(op,amp) if(op==3&&(amp&1)==0) continue;

// ── Search functions (each returns SR) ───────────────────────────────────────

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

static SR search_full_prng(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=4; r.entropy=base; r.overhead=PAT_OH[4];
    double be=base;int bseed=0,bop=0;
    #pragma omp parallel
    {double le=base;int ls=0,lop=0;int ff[256];
     #pragma omp for schedule(dynamic,1)
     for(int seed=0;seed<256;seed++)for(int op=0;op<N_OPS;op++){
         memset(ff,0,sizeof(ff));uint32_t st=(uint32_t)seed;
         for(int i=0;i<n;i++)ff[op_byte(blk[i],lcg_byte(&st),op)]++;
         double e=entropy_from_hist(ff,n);if(e<le){le=e;ls=seed;lop=op;}}
     #pragma omp critical
     if(le<be){be=le;bseed=ls;bop=lop;}}
    r.entropy=be;r.p[0]=bseed;r.p[1]=bop;
    snprintf(r.name,sizeof(r.name),"FULL-PRNG %s seed=%d",opname[bop],bseed);
    return r;
}

static SR search_global_shapes(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=5; r.entropy=base; r.overhead=PAT_OH[5];
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
    SR r; memset(&r,0,sizeof(r)); r.id=6; r.entropy=base; r.overhead=PAT_OH[6];
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

static SR search_step_blocks(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=7; r.entropy=base; r.overhead=PAT_OH[7];
    int csizes[]={64,128,256,512,1024,2048,4096,8192,16384};
    double be=base;int bcsize=0,bstep=0,bop=0,bdir=0;
    #pragma omp parallel
    {double le=base;int lc=0,ls=0,lop=0,ld=0;
     #pragma omp for schedule(dynamic,1)
     for(int ci=0;ci<9;ci++){
         int C=csizes[ci],nc=n/C;
         int(*cf)[256]=malloc(nc*256*sizeof(int));if(!cf)continue;
         memset(cf,0,nc*256*sizeof(int));
         for(int i=0;i<n;i++)cf[i/C][blk[i]]++;
         for(int step=1;step<=255;step++)for(int dir=0;dir<2;dir++)for(int op=0;op<N_OPS;op++){
             int ff[256]={0};
             for(int k=0;k<nc;k++){u8 amp=(u8)(((dir?nc-1-k:k)*step)&0xFF);
                 int tf[256];hist_transform(cf[k],tf,op,amp);for(int v=0;v<256;v++)ff[v]+=tf[v];}
             double e=entropy_from_hist(ff,n);if(e<le){le=e;lc=C;ls=step;lop=op;ld=dir;}}
         free(cf);}
     #pragma omp critical
     if(le<be){be=le;bcsize=lc;bstep=ls;bop=lop;bdir=ld;}}
    r.entropy=be;r.p[0]=bcsize;r.p[1]=bstep;r.p[2]=bop;r.p[3]=bdir;
    snprintf(r.name,sizeof(r.name),"STEP-BLOCKS %s C=%d st=%d %s",opname[bop],bcsize,bstep,bdir?"fall":"rise");
    return r;
}

static SR search_sparse_indexed(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=8; r.entropy=base; r.overhead=PAT_OH[8];
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

    // Try each set: 0=fib,1=pow2,2=primes,3..19=popcount 0..16
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
    SR r; memset(&r,0,sizeof(r)); r.id=9; r.entropy=base; r.overhead=PAT_OH[9];
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
    SR r; memset(&r,0,sizeof(r)); r.id=10; r.entropy=base; r.overhead=PAT_OH[10];
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
    r.overhead=5+(11+bN*8); // dynamic overhead
    snprintf(r.name,sizeof(r.name),"NWAY N=%d %s",bN,opname[bop]);
    return r;
}

static SR search_rolling_diff(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=11; r.entropy=base; r.overhead=PAT_OH[11];
    double be=base;int bK=0,bop=0;
    #pragma omp parallel
    {double le=base;int lK=0,lop=0;int ff[256];
     #pragma omp for schedule(dynamic,1)
     for(int op=0;op<N_OPS;op++)for(int K=1;K<=MAX_STRIDE;K++){
         memset(ff,0,sizeof(ff));
         for(int i=0;i<K;i++)ff[blk[i]]++;
         for(int i=K;i<n;i++)ff[op_byte(blk[i],blk[i-K],op)]++;
         double e=entropy_from_hist(ff,n);if(e<le){le=e;lK=K;lop=op;}}
     #pragma omp critical
     if(le<be){be=le;bK=lK;bop=lop;}}
    r.entropy=be;r.p[0]=bK;r.p[1]=bop;
    snprintf(r.name,sizeof(r.name),"ROLLING-DIFF %s K=%d",opname[bop],bK);
    return r;
}

static SR search_neighbor_op(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=12; r.entropy=base; r.overhead=PAT_OH[12];
    double be=base;int bK=0,bC=0,bop=0;
    #pragma omp parallel
    {double le=base;int lK=0,lC=0,lop=0;int ff[256];
     #pragma omp for schedule(dynamic,1)
     for(int K=1;K<=32;K++)for(int C=0;C<256;C++)for(int op=0;op<N_OPS;op++){
         memset(ff,0,sizeof(ff));
         for(int i=0;i<K;i++)ff[blk[i]]++;
         for(int i=K;i<n;i++)ff[op_byte(blk[i],(u8)(blk[i-K]+C),op)]++;
         double e=entropy_from_hist(ff,n);if(e<le){le=e;lK=K;lC=C;lop=op;}}
     #pragma omp critical
     if(le<be){be=le;bK=lK;bC=lC;bop=lop;}}
    r.entropy=be;r.p[0]=bK;r.p[1]=bC;r.p[2]=bop;
    snprintf(r.name,sizeof(r.name),"NEIGHBOR-OP %s K=%d C=%d",opname[bop],bK,bC);
    return r;
}

static SR search_local_delta(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=13; r.entropy=base; r.overhead=PAT_OH[13];
    double be=base;int bK=0,bxm=0;
    for(int xm=0;xm<2;xm++)for(int K=2;K<=64;K++){
        int ff[256]={0};
        for(int i=0;i<n;i++){
            if(i%K==0)ff[blk[i]]++;
            else{u8 d=xm?(blk[i]^blk[i-1]):(u8)(blk[i]-blk[i-1]);ff[d]++;}}
        double e=entropy_from_hist(ff,n);if(e<be){be=e;bK=K;bxm=xm;}}
    r.entropy=be;r.p[0]=bxm;r.p[1]=bK;
    snprintf(r.name,sizeof(r.name),"LOCAL-DELTA %s K=%d",bxm?"XOR":"SUB",bK);
    return r;
}

static SR search_position_hash(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=14; r.entropy=base; r.overhead=PAT_OH[14];
    int primes[]={3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97};
    double be=base;int bprime=0,bseed=0,bop=0;
    #pragma omp parallel
    {double le=base;int lp=0,ls=0,lop=0;int ff[256];
     #pragma omp for schedule(dynamic,1)
     for(int pi=0;pi<24;pi++)for(int seed=0;seed<256;seed++)for(int op=0;op<N_OPS;op++){
         int prime=primes[pi];
         memset(ff,0,sizeof(ff));
         for(int i=0;i<n;i++)ff[op_byte(blk[i],(u8)((i*prime+seed)&0xFF),op)]++;
         double e=entropy_from_hist(ff,n);if(e<le){le=e;lp=prime;ls=seed;lop=op;}}
     #pragma omp critical
     if(le<be){be=le;bprime=lp;bseed=ls;bop=lop;}}
    r.entropy=be;r.p[0]=bprime;r.p[1]=bseed;r.p[2]=bop;
    snprintf(r.name,sizeof(r.name),"POS-HASH %s p=%d s=%d",opname[bop],bprime,bseed);
    return r;
}

static SR search_pair_cross(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=15; r.entropy=base; r.overhead=PAT_OH[15];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    double be=base;int bD=0,bop=0;
    for(int op=0;op<N_OPS;op++)for(int D=1;D<=64;D++){
        int ff[256]={0};int pair=2*D,fp=(n/pair)*pair;
        for(int bi=0;bi<fp;bi+=pair){
            for(int j=0;j<D;j++)ff[op_byte(blk[bi+j],blk[bi+D+j],op)]++;
            for(int j=D;j<pair;j++)ff[blk[bi+j]]++;}
        for(int i=fp;i<n;i++)ff[blk[i]]++;
        double e=entropy_from_hist(ff,n);if(e<be){be=e;bD=D;bop=op;}}
    r.entropy=be;r.p[0]=bop;r.p[1]=bD;
    snprintf(r.name,sizeof(r.name),"PAIR-CROSS %s D=%d",opname[bop],bD);
    return r;
}

static SR search_prng_jump(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=16; r.entropy=base; r.overhead=PAT_OH[16];
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

static SR search_prng_keyed_pos(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=17; r.entropy=base; r.overhead=PAT_OH[17];
    double be=base;int bseed=0,bop=0;
    #pragma omp parallel
    {double le=base;int ls=0,lop=0;int ff[256];
     #pragma omp for schedule(dynamic,1)
     for(int seed=0;seed<256;seed++)for(int op=0;op<N_OPS;op++){
         memset(ff,0,sizeof(ff));
         for(int i=0;i<n;i++){uint32_t h=(uint32_t)(i^(seed*2654435761u));h^=h>>16;h*=0x45d9f3b;h^=h>>16;
             ff[op_byte(blk[i],(u8)(h>>24),op)]++;}
         double e=entropy_from_hist(ff,n);if(e<le){le=e;ls=seed;lop=op;}}
     #pragma omp critical
     if(le<be){be=le;bseed=ls;bop=lop;}}
    r.entropy=be;r.p[0]=bseed;r.p[1]=bop;
    snprintf(r.name,sizeof(r.name),"PRNG-KEYED-POS %s s=%d",opname[bop],bseed);
    return r;
}

static SR search_modular_mask(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=18; r.entropy=base; r.overhead=PAT_OH[18];
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

static SR search_stride_neighbor(const u8*blk,int n,double base){
    SR r; memset(&r,0,sizeof(r)); r.id=19; r.entropy=base; r.overhead=PAT_OH[19];
    int bfreq[256]={0};for(int i=0;i<n;i++)bfreq[blk[i]]++;
    double be=base;int bx=0,bop=0;
    for(int op=0;op<N_OPS;op++)for(int x=1;x<=MAX_STRIDE;x++){
        int cnt=0;for(int p=0;p<n;p+=x)cnt++;
        int *pos=malloc(cnt*sizeof(int));if(!pos)continue;
        int k=0;for(int p=0;p<n;p+=x)pos[k++]=p;
        int sf[256]={0},sfo[256]={0};
        for(int j=0;j<cnt;j++){sf[op_byte(blk[pos[j]],blk[pos[(j+1)%cnt]],op)]++;sfo[blk[pos[j]]]++;}
        int ff[256];for(int v=0;v<256;v++)ff[v]=(bfreq[v]-sfo[v])+sf[v];
        free(pos);
        double e=entropy_from_hist(ff,n);if(e<be){be=e;bx=x;bop=op;}}
    r.entropy=be;r.p[0]=bx;r.p[1]=bop;
    snprintf(r.name,sizeof(r.name),"STRIDE-NBR %s s=%d",opname[bop],bx);
    return r;
}

// ── Extract op index from any SR ─────────────────────────────────────────────
static int sr_get_op(const SR *r) {
    switch(r->id){
        case 0: return r->p[1];
        case 1: return r->p[2];
        case 2: return r->p[1];
        case 3: return r->p[2];
        case 4: return r->p[1];
        case 5: return r->p[1];
        case 6: return r->p[2];
        case 7: return r->p[2];
        case 8: return r->p[1];
        case 9: return r->p[2];
        case 10: return r->p[1];
        case 11: return r->p[1];
        case 12: return r->p[2];
        case 13: return r->p[0] ? 1 : 0; // xor_mode → map to XOR(1) or ADD(0)
        case 14: return r->p[2];
        case 15: return r->p[0];
        case 16: return r->p[2];
        case 17: return r->p[1];
        case 18: return r->p[2];
        case 19: return r->p[1];
    }
    return 0;
}

static const char *pat_label[N_PATTERNS] = {
    "STRIDE-CONST","STRIDE-PRNG-AMP","DUAL-STRIDE","PRNG-SELECT","FULL-PRNG",
    "GLOBAL-SHAPE","PERIODIC-SHAPE","STEP-BLOCKS","SPARSE-INDEX","PRNG+GRADIENT",
    "NWAY-STRIDE","ROLLING-DIFF","NEIGHBOR-OP","LOCAL-DELTA","POSITION-HASH",
    "PAIR-CROSS","PRNG-JUMP","PRNG-KEYED-POS","MODULAR-MASK","STRIDE-NEIGHBOR"
};

// ── Run all searches and return array ────────────────────────────────────────
static void run_all(const u8 *blk, int n, double base, SR *out) {
    out[0]  = search_stride_const    (blk,n,base);
    out[1]  = search_stride_prng_amp (blk,n,base);
    out[2]  = search_dual_stride     (blk,n,base);
    out[3]  = search_prng_select     (blk,n,base);
    out[4]  = search_full_prng       (blk,n,base);
    out[5]  = search_global_shapes   (blk,n,base);
    out[6]  = search_periodic_shapes (blk,n,base);
    out[7]  = search_step_blocks     (blk,n,base);
    out[8]  = search_sparse_indexed  (blk,n,base);
    out[9]  = search_prng_gradient   (blk,n,base);
    out[10] = search_nway_stride     (blk,n,base);
    out[11] = search_rolling_diff    (blk,n,base);
    out[12] = search_neighbor_op     (blk,n,base);
    out[13] = search_local_delta     (blk,n,base);
    out[14] = search_position_hash   (blk,n,base);
    out[15] = search_pair_cross      (blk,n,base);
    out[16] = search_prng_jump       (blk,n,base);
    out[17] = search_prng_keyed_pos  (blk,n,base);
    out[18] = search_modular_mask    (blk,n,base);
    out[19] = search_stride_neighbor (blk,n,base);
}

// ── Main: stats run over N_STATS blocks ──────────────────────────────────────
#define N_STATS 100

int main(void) {
    init_gf256();
    int nt=omp_get_max_threads()-1; if(nt<1)nt=1;
    omp_set_num_threads(nt);

    // Per-pattern stats
    int   pat_wins[N_PATTERNS]={0};    // total times applied (any pass)
    int   pat_p1[N_PATTERNS]  ={0};    // times won pass 1 specifically
    double pat_net[N_PATTERNS] ={0.0}; // cumulative net gain
    // Per-op stats
    int   op_wins[N_OPS]      ={0};

    int total_applies=0, total_passes=0;

    printf("Running %d blocks x %d bytes each  (threads=%d)\n\n",
           N_STATS,BLOCK_SIZE,nt);

    for(int bi=0;bi<N_STATS;bi++){
        u8 *blk=malloc(BLOCK_SIZE);
        if(!blk||!fill_random(blk,BLOCK_SIZE)){free(blk);continue;}

        double base=byte_entropy(blk,BLOCK_SIZE);
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
            if(best_net<=0||best_idx<0) break;

            pat_wins[best_idx]++;
            if(pass==1) pat_p1[best_idx]++;
            pat_net[best_idx]+=best_net;
            int op=sr_get_op(&results[best_idx]);
            if(op>=0&&op<N_OPS) op_wins[op]++;
            total_applies++;

            apply_sr(blk,BLOCK_SIZE,&results[best_idx]);
            base=byte_entropy(blk,BLOCK_SIZE);
        }
        total_passes+=pass-1;
        free(blk);

        if((bi+1)%50==0||bi==N_STATS-1)
            printf("  %d/%d blocks done...\n",bi+1,N_STATS);
    }

    // ── Sort patterns by win count (descending) ───────────────────────────────
    int order[N_PATTERNS];
    for(int i=0;i<N_PATTERNS;i++) order[i]=i;
    for(int i=1;i<N_PATTERNS;i++){
        int k=order[i],j=i-1;
        while(j>=0&&pat_wins[order[j]]<pat_wins[k]){order[j+1]=order[j];j--;}
        order[j+1]=k;
    }

    printf("\n=== Pattern frequency  (%d blocks, %d total transforms applied) ===\n",
           N_STATS,total_applies);
    printf("%-4s %-20s %7s %6s %7s %7s  %s\n",
           "Rank","Pattern","Applied","%tot","P1wins","AvgNET","Note");
    printf("%-4s %-20s %7s %6s %7s %7s\n",
           "----","-------------------","-------","------","------","------");
    for(int r=0;r<N_PATTERNS;r++){
        int i=order[r];
        double avg=pat_wins[i]>0?pat_net[i]/pat_wins[i]:0.0;
        double pct=100.0*pat_wins[i]/N_STATS;
        printf("%3d. %-20s %7d %5.1f%% %7d %+7.1f",
               r+1,pat_label[i],pat_wins[i],pct,pat_p1[i],avg);
        if(pat_wins[i]==0)             printf("  ← NEVER USED — remove");
        else if(pct<2.0)               printf("  ← rarely used (<2%%)");
        else if(pct<5.0)               printf("  ← seldom used (<5%%)");
        printf("\n");
    }

    // ── Sort ops by frequency ─────────────────────────────────────────────────
    int op_order[N_OPS]; for(int i=0;i<N_OPS;i++) op_order[i]=i;
    for(int i=1;i<N_OPS;i++){
        int k=op_order[i],j=i-1;
        while(j>=0&&op_wins[op_order[j]]<op_wins[k]){op_order[j+1]=op_order[j];j--;}
        op_order[j+1]=k;
    }
    printf("\n=== Op frequency across all winning transforms ===\n");
    for(int r=0;r<N_OPS;r++){
        int i=op_order[r];
        double pct=total_applies>0?100.0*op_wins[i]/total_applies:0.0;
        printf("  %-6s %6d  %5.1f%%",opname[i],op_wins[i],pct);
        if(pct<2.0) printf("  ← candidate for removal");
        printf("\n");
    }

    printf("\n=== Suggested pruning ===\n");
    printf("Patterns never or rarely applied (remove to reduce OH and search time):\n");
    for(int r=N_PATTERNS-1;r>=0;r--){
        int i=order[r];
        if(100.0*pat_wins[i]/N_STATS<5.0)
            printf("  %-20s  %d/%d applications (%.1f%%)\n",
                   pat_label[i],pat_wins[i],N_STATS,100.0*pat_wins[i]/N_STATS);
        else break;
    }
    printf("Ops with <2%% share (could remove from each pattern's search):\n");
    for(int r=N_OPS-1;r>=0;r--){
        int i=op_order[r];
        double pct=total_applies>0?100.0*op_wins[i]/total_applies:0.0;
        if(pct<2.0) printf("  %-6s  %d wins (%.1f%%)\n",opname[i],op_wins[i],pct);
        else break;
    }
    printf("\nAvg passes per block: %.2f\n",(double)total_passes/N_STATS);
    return 0;
}
