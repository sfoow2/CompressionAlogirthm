/* compressor3.c — Dirichlet-smoothed adaptive arithmetic coder, binary alphabet.
 *
 * Assumes the input only ever contains at most 2 distinct byte values
 * (verified at runtime, not just assumed) — the model is sized for that
 * alphabet instead of the full 256, which removes almost all of the
 * "learning cost" of discovering an alphabet that never gets used.
 *
 * Format: [2B N][1B alpha][1B nsym][nsym B symbol table][arithmetic-coded data]
 *
 * Build: gcc -O2 -o compressor3 compressor3.c -lm
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef unsigned char  u8;
typedef uint32_t       u32;
typedef uint64_t       u64;

typedef struct { u8 *buf; size_t pos; u8 acc; int bits; } BW;
static void bw_init(BW *w, u8 *b)  { w->buf=b; w->pos=0; w->acc=0; w->bits=0; }
static void bw_bit(BW *w, int b)   { w->acc=(u8)((w->acc<<1)|(b&1)); if(++w->bits==8){w->buf[w->pos++]=w->acc;w->acc=0;w->bits=0;} }
static size_t bw_done(BW *w)       { if(w->bits)w->buf[w->pos++]=(u8)(w->acc<<(8-w->bits)); return w->pos; }

typedef struct { const u8 *buf; size_t pos; u8 acc; int bits; } BR;
static void br_init(BR *r, const u8 *b) { r->buf=b; r->pos=0; r->acc=0; r->bits=0; }
static int  br_bit(BR *r)               { if(!r->bits){r->acc=r->buf[r->pos++];r->bits=8;} return (r->acc>>(--r->bits))&1; }

#define AC_TOP  0xFFFFFFFFUL
#define AC_HALF 0x80000000UL
#define AC_QTR  0x40000000UL
#define AC_3QT  0xC0000000UL

typedef struct { u32 lo,hi; long pend; BW *w; } AEnc;
static void ae_init(AEnc *e, BW *w) { e->lo=0; e->hi=AC_TOP; e->pend=0; e->w=w; }
static void ae_emit(AEnc *e, int b) { bw_bit(e->w,b); for(;e->pend>0;e->pend--) bw_bit(e->w,!b); }
static void ae_sym(AEnc *e, u32 cum, u32 freq, u32 total) {
    u64 r=(u64)(e->hi-e->lo)+1;
    e->hi=e->lo+(u32)((r*(u64)(cum+freq))/total)-1;
    e->lo=e->lo+(u32)((r*(u64)cum)/total);
    for(;;){
        if      (e->hi<AC_HALF)               { ae_emit(e,0); e->lo<<=1; e->hi=(e->hi<<1)|1; }
        else if (e->lo>=AC_HALF)              { ae_emit(e,1); e->lo=(e->lo-AC_HALF)<<1; e->hi=((e->hi-AC_HALF)<<1)|1; }
        else if (e->lo>=AC_QTR&&e->hi<AC_3QT){ e->pend++; e->lo=(e->lo-AC_QTR)<<1; e->hi=((e->hi-AC_QTR)<<1)|1; }
        else break;
    }
}
static void ae_done(AEnc *e) { e->pend++; ae_emit(e,(e->lo>=AC_QTR)?1:0); }

typedef struct { u32 lo,hi,code; BR *r; } ADec;
static void ad_init(ADec *d, BR *r) {
    d->lo=0; d->hi=AC_TOP; d->code=0; d->r=r;
    for (int i=0;i<32;i++) d->code=(d->code<<1)|br_bit(r);
}
static int ad_sym(ADec *d, const u32 *cum, u32 total, int nsym) {
    u64 range=(u64)(d->hi-d->lo)+1;
    u32 val=(u32)(((u64)(d->code-d->lo)*total)/range);
    int s=0;
    while (s+1<nsym && cum[s+1]<=val) s++;
    u32 c=cum[s], f=cum[s+1]-c;
    d->hi=d->lo+(u32)((range*(u64)(c+f))/total)-1;
    d->lo=d->lo+(u32)((range*(u64)c)/total);
    for(;;){
        if      (d->hi<AC_HALF)               { d->lo<<=1; d->hi=(d->hi<<1)|1; d->code=(d->code<<1)|br_bit(d->r); }
        else if (d->lo>=AC_HALF)              { d->lo=(d->lo-AC_HALF)<<1; d->hi=((d->hi-AC_HALF)<<1)|1; d->code=((d->code-AC_HALF)<<1)|br_bit(d->r); }
        else if (d->lo>=AC_QTR&&d->hi<AC_3QT){ d->lo=(d->lo-AC_QTR)<<1; d->hi=((d->hi-AC_QTR)<<1)|1; d->code=((d->code-AC_QTR)<<1)|br_bit(d->r); }
        else break;
    }
    return s;
}

/* cnt/total are sized for at most 2 symbols — the actual alphabet in use. */
typedef struct { u32 cnt[2]; u32 total; int nsym; } DirModel;

static void dm_init(DirModel *m, u32 alpha, int nsym) {
    m->nsym=nsym;
    for (int v=0;v<nsym;v++) m->cnt[v]=alpha;
    m->total=(u32)nsym*alpha;
}

static void dm_cum(const DirModel *m, u32 *cum) {
    cum[0]=0;
    for (int v=0;v<m->nsym;v++) cum[v+1]=cum[v]+m->cnt[v];
}

static void dm_update(DirModel *m, int v) { m->cnt[v]++; m->total++; }

static double dm_code_len(const u8 *idx, size_t n, u32 alpha, int nsym) {
    DirModel m; dm_init(&m, alpha, nsym);
    double L=0.0;
    for (size_t i=0;i<n;i++) {
        int v=idx[i];
        L -= log2((double)m.cnt[v] / m.total);
        dm_update(&m, v);
    }
    return L;
}

static u32 find_best_alpha(const u8 *idx, size_t n, int nsym) {
    u32 best_a=1; double best_L=1e18;
    for (u32 a=1; a<=255; a++) {
        double L=dm_code_len(idx, n, a, nsym);
        if (L<best_L) { best_L=L; best_a=a; }
    }
    return best_a;
}

/* Format: [2B N|nsym-bit][1B alpha][nsym B symbol table][arithmetic-coded data]
 * N's top bit (which N itself never needs below 32768) doubles as nsym-1,
 * so the symbol count no longer needs a byte of its own. */
size_t compress_block(const u8 *idx, size_t n, u8 *out, u32 alpha, int nsym, const u8 *sym) {
    out[0]=(u8)((n>>8)|((nsym-1)<<7)); out[1]=(u8)n;
    out[2]=(u8)alpha;
    for (int i=0;i<nsym;i++) out[3+i]=sym[i];
    size_t hdr=(size_t)3+nsym;

    BW bw; bw_init(&bw, out+hdr);
    AEnc enc; ae_init(&enc, &bw);
    DirModel m; dm_init(&m, alpha, nsym);
    u32 cum[3];

    for (size_t i=0;i<n;i++) {
        int v=idx[i];
        dm_cum(&m, cum);
        ae_sym(&enc, cum[v], m.cnt[v], m.total);
        dm_update(&m, v);
    }
    ae_done(&enc);
    return hdr+bw_done(&bw);
}

size_t decompress_block(const u8 *in, u8 *out) {
    size_t n=(((size_t)in[0]&0x7F)<<8)|(size_t)in[1];
    int nsym=((in[0]&0x80)?2:1);
    u32 alpha=in[2];
    u8 sym[2];
    for (int i=0;i<nsym;i++) sym[i]=in[3+i];
    size_t hdr=(size_t)3+nsym;

    BR br; br_init(&br, in+hdr);
    ADec dec; ad_init(&dec, &br);
    DirModel m; dm_init(&m, alpha, nsym);
    u32 cum[3];

    for (size_t i=0;i<n;i++) {
        dm_cum(&m, cum);
        int v=ad_sym(&dec, cum, m.total, nsym);
        out[i]=sym[v];
        dm_update(&m, v);
    }
    return n;
}

static double entropy(const u8 *d, size_t n) {
    u32 f[256]={0}; for(size_t i=0;i<n;i++) f[d[i]]++;
    double H=0.0;
    for(int v=0;v<256;v++) if(f[v]){double p=(double)f[v]/n; H-=p*log2(p);}
    return H;
}
static u8 *read_file(const char *path, size_t *len) {
    FILE *f=fopen(path,"rb"); if(!f){perror(path);return NULL;}
    fseek(f,0,SEEK_END); *len=(size_t)ftell(f); rewind(f);
    u8 *buf=malloc(*len+1); fread(buf,1,*len,f); fclose(f);
    return buf;
}

#define INPUT_PATH  "C:\\Users\\lukac\\Documents\\compressor\\aftercheat7.bin"
#define OUTPUT_PATH "C:\\Users\\lukac\\Documents\\compressor\\output.bin"

int main(void) {
    size_t n; u8 *in=read_file(INPUT_PATH,&n); if(!in) return 1;

    /* Build the symbol table by scanning the file — asserts the "at most
       2 distinct byte values" assumption instead of just trusting it. */
    u8 sym[2]; int nsym=0;
    u8 map[256]; memset(map,0xFF,sizeof map);
    for (size_t i=0;i<n;i++) {
        u8 v=in[i];
        if (map[v]==0xFF) {
            if (nsym==2) {
                fprintf(stderr,"error: input has more than 2 distinct byte values at offset %zu (value %u) — binary-alphabet assumption violated\n", i, v);
                free(in); return 1;
            }
            map[v]=(u8)nsym; sym[nsym]=v; nsym++;
        }
    }
    if (nsym==0) { nsym=1; sym[0]=0; }
    if (n>=32768) {
        fprintf(stderr,"error: input is %zu bytes, exceeds 32767-byte limit (N's top bit is reserved for nsym)\n", n);
        free(in); return 1;
    }

    u8 *idx=malloc(n?n:1);
    for (size_t i=0;i<n;i++) idx[i]=map[in[i]];

    u32 alpha=find_best_alpha(idx,n,nsym);
    double best_L=dm_code_len(idx,n,alpha,nsym)/8.0;
    u8 *out=malloc(n+1024);
    size_t clen=compress_block(idx,n,out,alpha,nsym,sym);
    double H=entropy(in,n);
    printf("input:      %zu bytes  H=%.6f bps  (nsym=%d, symbols=", n, H, nsym);
    for (int i=0;i<nsym;i++) printf("%s%u", i?",":"", sym[i]);
    printf(")\n");
    printf("alpha:      %u  (model code-len=%.1f bytes, overhead=%.1f bytes)\n",
           alpha, best_L, best_L-H*n/8.0);
    printf("compressed: %zu bytes\n", clen);
    if (clen<n) printf("saved:      %ld bytes (%.3f%%)\n",(long)n-(long)clen,100.0*(n-clen)/n);
    else        printf("expanded:   %ld bytes\n",(long)clen-(long)n);

    u8 *back=malloc(n?n:1);
    size_t dn=decompress_block(out, back);
    int ok = (dn==n) && (n==0 || memcmp(in,back,n)==0);
    printf("roundtrip:  %s\n", ok?"OK":"FAIL");

    FILE *fo=fopen(OUTPUT_PATH,"wb"); fwrite(out,1,clen,fo); fclose(fo);
    free(out); free(in); free(idx); free(back);
    return ok?0:1;
}
