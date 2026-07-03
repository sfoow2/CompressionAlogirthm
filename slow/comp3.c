/* compressor3.c — Dirichlet-smoothed adaptive arithmetic coder.
 *
 * Key idea: instead of a static model (requires transmitting a 155-byte freq
 * table) or a pure adaptive model (Shtarkov regret ~191 bytes for k=256),
 * use a Dirichlet(α) prior: initialise every symbol count to α pseudo-counts,
 * then update counts normally as symbols are encoded/decoded.
 *
 * P(v | past) = (count[v] + α) / (total + 256α)
 *
 * With α = N/256 (balanced prior):
 *   - starts as uniform (no table to transmit)
 *   - adapts toward the actual distribution as it sees data
 *   - overhead vs entropy ≈ (k-1)/2 × log₂(1 + N/kα) = 127 bits ≈ 16 bytes
 *     (much less than static-model rice-table cost of ~155 bytes)
 *
 * For BCrypt after reduce2.c (H≈7.87 bps, N=4096):
 *   savings = (8-7.87)×4096/8 = 66 bytes
 *   overhead ≈ 16 bytes
 *   net ≈ 50 bytes saved per block (before instruction-stream overhead)
 *
 * Format: [4B N] [2B alpha] [arithmetic-coded data]
 *
 * Build: gcc -O2 -o compressor3 compressor3.c -lm
 * Usage: compressor3 compress   <input> <output>
 *        compressor3 decompress <input> <output>
 *        compressor3 roundtrip  <input>
 *        compressor3 scan       <input>    (tries multiple alpha values)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

typedef unsigned char  u8;
typedef uint32_t       u32;
typedef uint64_t       u64;

/* ================================================================
 * Bit reader / writer
 * ================================================================ */
typedef struct { u8 *buf; size_t pos; u8 acc; int bits; } BW;
static void bw_init(BW *w, u8 *b)  { w->buf=b; w->pos=0; w->acc=0; w->bits=0; }
static void bw_bit(BW *w, int b)   { w->acc=(u8)((w->acc<<1)|(b&1)); if(++w->bits==8){w->buf[w->pos++]=w->acc;w->acc=0;w->bits=0;} }
static size_t bw_done(BW *w)       { if(w->bits)w->buf[w->pos++]=(u8)(w->acc<<(8-w->bits)); return w->pos; }

typedef struct { const u8 *buf; size_t pos; u8 acc; int bits; } BR;
static void br_init(BR *r, const u8 *b) { r->buf=b; r->pos=0; r->acc=0; r->bits=0; }
static int  br_bit(BR *r)               { if(!r->bits){r->acc=r->buf[r->pos++];r->bits=8;} return (r->acc>>(--r->bits))&1; }

/* ================================================================
 * Range-coder (32-bit Elias interval)
 * ================================================================ */
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
static int ad_sym(ADec *d, const u32 *cum, u32 total) {
    u64 range=(u64)(d->hi-d->lo)+1;
    u32 val=(u32)(((u64)(d->code-d->lo)*total)/range);
    int lo=0, hi=255;
    while (lo<hi) { int m=(lo+hi+1)/2; if(cum[m]<=val) lo=m; else hi=m-1; }
    int s=lo;
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

/* ================================================================
 * Dirichlet model: counts[v] start at alpha, increment after each symbol.
 * Arithmetic coder sees prob[v] = counts[v] / total.
 * ================================================================ */
typedef struct {
    u32 cnt[256];  /* counts[v] = alpha + how many times v seen so far */
    u32 total;     /* = 256*alpha + symbols seen so far                 */
} DirModel;

static void dm_init(DirModel *m, u32 alpha) {
    for (int v=0;v<256;v++) m->cnt[v]=alpha;
    m->total=256u*alpha;
}

/* Build prefix-sum into cum[0..256] from m->cnt. O(k) per call. */
static void dm_cum(const DirModel *m, u32 *cum) {
    cum[0]=0;
    for (int v=0;v<256;v++) cum[v+1]=cum[v]+m->cnt[v];
}

static void dm_update(DirModel *m, int v) {
    m->cnt[v]++;
    m->total++;
}

/* Simulate and return code length in bits (for alpha selection). */
static double dm_code_len(const u8 *data, size_t n, u32 alpha) {
    DirModel m; dm_init(&m, alpha);
    double L=0.0;
    for (size_t i=0;i<n;i++) {
        int v=data[i];
        L -= log2((double)m.cnt[v] / m.total);
        dm_update(&m, v);
    }
    return L;
}

/* Find best alpha by exhaustive search over 1..255 (fits in 1 byte). */
static u32 find_best_alpha(const u8 *data, size_t n) {
    u32 best_a = 1; double best_L = 1e18;
    for (u32 a = 1; a <= 255; a++) {
        double L = dm_code_len(data, n, a);
        if (L < best_L) { best_L = L; best_a = a; }
    }
    return best_a;
}

/* ================================================================
 * Compress
 * Format: [2B N as u16][1B alpha][arithmetic-coded data]
 * ================================================================ */
size_t compress_block(const u8 *in, size_t n, u8 *out, u32 alpha) {
    /* Header: 3 bytes */
    out[0]=(u8)(n>>8); out[1]=(u8)n;
    out[2]=(u8)alpha;

    BW bw; bw_init(&bw, out+3);
    AEnc enc; ae_init(&enc, &bw);
    DirModel m; dm_init(&m, alpha);
    u32 cum[257];

    for (size_t i=0;i<n;i++) {
        int v=in[i];
        dm_cum(&m, cum);
        ae_sym(&enc, cum[v], m.cnt[v], m.total);
        dm_update(&m, v);
    }
    ae_done(&enc);
    return 3+bw_done(&bw);
}

/* ================================================================
 * Decompress
 * ================================================================ */
size_t decompress_block(const u8 *in, u8 *out) {
    size_t n = ((size_t)in[0]<<8)|(size_t)in[1];
    u32 alpha = in[2];

    BR br; br_init(&br, in+3);
    ADec dec; ad_init(&dec, &br);
    DirModel m; dm_init(&m, alpha);
    u32 cum[257];

    for (size_t i=0;i<n;i++) {
        dm_cum(&m, cum);
        int v = ad_sym(&dec, cum, m.total);
        out[i]=(u8)v;
        dm_update(&m, v);
    }
    return n;
}

/* ================================================================
 * Utility
 * ================================================================ */
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

/* Hard-coded paths — change these to point at your files. */
#define INPUT_PATH  "C:\\Users\\lukac\\Documents\\compressor\\compressed.bin"
#define OUTPUT_PATH "C:\\Users\\lukac\\Documents\\compressor\\output.bin"

int main(void) {
    size_t n; u8 *in=read_file(INPUT_PATH,&n); if(!in) return 1;
    u32 alpha=find_best_alpha(in,n);
    double best_L = dm_code_len(in,n,alpha)/8.0;
    u8 *out=malloc(n+1024);
    size_t clen=compress_block(in,n,out,alpha);
    double H=entropy(in,n);
    printf("input:      %zu bytes  H=%.6f bps\n", n, H);
    printf("alpha:      %u  (model code-len=%.1f bytes, overhead=%.1f bytes)\n",
           alpha, best_L, best_L-H*n/8.0);
    printf("compressed: %zu bytes\n", clen);
    if (clen<n) printf("saved:      %ld bytes (%.3f%%)\n",(long)n-(long)clen,100.0*(n-clen)/n);
    else        printf("expanded:   %ld bytes\n",(long)clen-(long)n);
    FILE *fo=fopen(OUTPUT_PATH,"wb"); fwrite(out,1,clen,fo); fclose(fo);

    u8 *back=malloc(n+1);
    size_t dlen=decompress_block(out, back);
    int ok = (dlen==n) && (memcmp(in,back,n)==0);
    printf("roundtrip:  %s (decompressed %zu bytes)\n", ok?"OK, bit-exact":"MISMATCH", dlen);
    free(back);

    free(out); free(in);
    return 0;
}
