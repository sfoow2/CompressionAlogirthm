// pipeline.c — Iterative rANS + Polar-code bias-reduction compressor
// Compile: gcc -O2 -o pipeline pipeline.c -lm
//
// Each layer:
//   1. rANS-compress current data  → payload
//   2. Polar SC bias-reduce payload MSBs in-place  → corrected payload
//   3. Store {rANS header, polar info bits} and loop on corrected payload
//
// File format:
//   [ 2 bytes              ] number of layers
//   [ 1036 × layers        ] LayerHdrBase records (rANS info + n_info_bytes each)
//   [ variable × layers    ] packed polar info bits for each layer
//   [ remaining            ] final corrected payload
//
// Decompression order per layer:
//   undo_polar_bias(payload, info_bits) → rans_decompress

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>

typedef uint8_t  u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;

// ── Tunable ────────────────────────────────────────────────────────────────
static const char *INPUT_PATH  = "C:/Users/lukac/Documents/compressor/test.exe";
static const char *OUTPUT_PATH = "C:/Users/lukac/Documents/compressor/pipeline.bin";

#define POLAR_N     1024
#define POLAR_K     65
#define POLAR_D     0.37
#define MAX_LAYERS  65536   // hard cap on iterations

// ── xorshift64 ────────────────────────────────────────────────────────────
static inline u64 xs64(u64 s) { s^=s<<13; s^=s>>7; s^=s<<17; return s; }

// ── Byte entropy ────────────────────────────────────────────────────────────
static float byte_entropy(const u8 *d, int n) {
    int cnt[256] = {0};
    for (int i = 0; i < n; i++) cnt[d[i]]++;
    double e = 0.0;
    for (int i = 0; i < 256; i++) {
        if (!cnt[i]) continue;
        double p = (double)cnt[i] / n;
        e -= p * log2(p);
    }
    return (float)e;
}

// ── Polar: Bhattacharyya frozen-set design ───────────────────────────────────
// (Korada & Urbanke 2010 — optimal for lossy source coding over BSC)
typedef struct { double z; int idx; } zpair;
static int cmp_zpair(const void *a, const void *b) {
    double za = ((const zpair *)a)->z, zb = ((const zpair *)b)->z;
    if (za < zb) return -1;
    if (za > zb) return  1;
    return ((const zpair *)a)->idx - ((const zpair *)b)->idx;
}
static int cmp_int(const void *a, const void *b) {
    int ia = *(const int*)a, ib = *(const int*)b;
    return (ia > ib) - (ia < ib);
}

// Returns malloc'd info_set[K]: K indices with smallest Bhattacharyya Z.
// Fills is_frozen[N] (1=frozen, 0=info).
static int *build_info_set(int N, int K, double D, u8 *is_frozen) {
    int logN = 0;
    while ((1 << logN) < N) logN++;

    double *z = malloc(N * sizeof(double));
    z[0] = 2.0 * sqrt(D * (1.0 - D));
    int cur = 1;
    for (int s = 0; s < logN; s++) {
        for (int j = cur - 1; j >= 0; j--) {
            double zj = z[j];
            double zm = 2.0*zj - zj*zj;
            double zp = zj*zj;
            if (zm > 1.0) zm = 1.0;
            if (zm < 0.0) zm = 0.0;
            z[2*j]   = zm;
            z[2*j+1] = zp;
        }
        cur *= 2;
    }

    zpair *pairs = malloc(N * sizeof(zpair));
    for (int i = 0; i < N; i++) { pairs[i].z = z[i]; pairs[i].idx = i; }
    qsort(pairs, N, sizeof(zpair), cmp_zpair);

    int *info = malloc(K * sizeof(int));
    memset(is_frozen, 1, N);
    for (int i = 0; i < K; i++) {
        info[i] = pairs[i].idx;
        is_frozen[pairs[i].idx] = 0;
    }
    qsort(info, K, sizeof(int), cmp_int);

    free(z); free(pairs);
    return info;
}

// ── Polar: butterfly transform  x = u * G_N  (in-place on bits) ─────────────
static void polar_transform(u8 *bits, int N) {
    for (int step = 1; step < N; step <<= 1)
        for (int i = 0; i < N; i += (step << 1))
            for (int j = 0; j < step; j++)
                bits[i+j] ^= bits[i+j+step];
}

// ── Polar: SC encoder LLR operations ────────────────────────────────────────
static inline double f_op(double a, double b) {
    double t = tanh(a * 0.5) * tanh(b * 0.5);
    if (t >  0.999999999999) t =  0.999999999999;
    if (t < -0.999999999999) t = -0.999999999999;
    return 2.0 * atanh(t);
}
static inline double g_op(double a, double b, u8 u) {
    return u ? (b - a) : (b + a);
}

// ── Polar: recursive SC encoder ──────────────────────────────────────────────
// Source bit y_i in {0,1} → LLR = ±log((1-D)/D).
// Info positions: MAP decision (LLR < 0 → u=1).
// Frozen positions: use shared frozen_bits[i].
// x_out returns the partial-sum codeword bits for the parent g-step.
// u_bits[offset..offset+len-1] receives the u-bit decisions.
static void sc_encode(double *llrs, u8 *x_out, int len,
                      const u8 *is_frozen, const u8 *frozen_bits,
                      u8 *u_bits, int offset)
{
    if (len == 1) {
        u8 u = is_frozen[offset] ? frozen_bits[offset]
                                 : ((llrs[0] < 0.0) ? 1 : 0);
        u_bits[offset] = u;
        x_out[0] = u;
        return;
    }
    int h = len / 2;

    double *L = malloc(h * sizeof(double));
    for (int i = 0; i < h; i++) L[i] = f_op(llrs[i], llrs[i+h]);
    u8 *xL = malloc(h);
    sc_encode(L, xL, h, is_frozen, frozen_bits, u_bits, offset);

    double *R = malloc(h * sizeof(double));
    for (int i = 0; i < h; i++) R[i] = g_op(llrs[i], llrs[i+h], xL[i]);
    u8 *xR = malloc(h);
    sc_encode(R, xR, h, is_frozen, frozen_bits, u_bits, offset + h);

    for (int i = 0; i < h; i++) {
        x_out[i]   = xL[i] ^ xR[i];
        x_out[i+h] = xR[i];
    }
    free(L); free(R); free(xL); free(xR);
}

// ── Polar: compress one block of N source bits → K info bits ────────────────
static void compress_block(const u8 *src, int N, double D,
                           const u8 *is_frozen, const u8 *frozen_bits,
                           const int *info_set, int K,
                           u8 *payload_bits)
{
    double L0 = log((1.0 - D) / D);
    double *llrs  = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) llrs[i] = src[i] ? -L0 : +L0;
    u8 *x_dummy = malloc(N);
    u8 *u_bits  = malloc(N);
    sc_encode(llrs, x_dummy, N, is_frozen, frozen_bits, u_bits, 0);
    for (int k = 0; k < K; k++) payload_bits[k] = u_bits[info_set[k]];
    free(llrs); free(x_dummy); free(u_bits);
}

// ── Polar: decompress K info bits → N reconstruction bits ───────────────────
static void decompress_block(const u8 *payload_bits, int K,
                             const u8 *is_frozen, const u8 *frozen_bits,
                             const int *info_set, int N,
                             u8 *recon)
{
    u8 *u = malloc(N);
    for (int i = 0; i < N; i++) u[i] = is_frozen[i] ? frozen_bits[i] : 0;
    for (int k = 0; k < K; k++) u[info_set[k]] = payload_bits[k];
    memcpy(recon, u, N);
    polar_transform(recon, N);
    free(u);
}

// ── Global polar state (computed once; shared frozen seq = fixed seed) ───────
static u8  *g_is_frozen   = NULL;
static u8  *g_frozen_bits = NULL;
static int *g_info_set    = NULL;

static void polar_init(void) {
    g_is_frozen   = malloc(POLAR_N);
    g_info_set    = build_info_set(POLAR_N, POLAR_K, POLAR_D, g_is_frozen);
    g_frozen_bits = calloc(POLAR_N, 1);
    u64 fs = 0xC0FFEEULL;
    for (int i = 0; i < POLAR_N; i++)
        if (g_is_frozen[i]) { fs = xs64(fs); g_frozen_bits[i] = (fs >> 11) & 1; }
}

// ── Polar bias reduction ─────────────────────────────────────────────────────
// src_block[i] = 1 if data[i] < 128 (low byte), 0 otherwise.
// Polar encodes which bytes are low → K info bits → reconstruct → XOR 0x80 on
// predicted-low bytes to push them above 128. Decompressor mirrors this exactly.
//
// Returns match count (correct predictions, out of n).
// *n_polar_ones_out = how many bytes polar predicted to flip (recon=1).
// *n_true_pos_out   = how many of those were actually low (precision numerator).
static int polar_bias_reduce(u8 *data, int n, u8 **info_out, int *n_info_bytes_out,
                              int *n_polar_ones_out, int *n_true_pos_out) {
    int n_blocks        = (n + POLAR_N - 1) / POLAR_N;
    int total_info_bits = n_blocks * POLAR_K;
    int n_info_bytes    = (total_info_bits + 7) / 8;

    u8 *all_info    = calloc(n_info_bytes, 1);
    u8 *src_block   = malloc(POLAR_N);
    u8 *pay_block   = malloc(POLAR_K);
    u8 *recon       = malloc(POLAR_N);
    int matches     = 0;
    int n_polar_ones = 0;
    int n_true_pos  = 0;
    int bit_cur     = 0;

    for (int b = 0; b < n_blocks; b++) {
        int start = b * POLAR_N;
        int blen  = (start + POLAR_N <= n) ? POLAR_N : (n - start);

        // 1 = low byte (< 128), 0 = high byte; zero-pad partial last block
        memset(src_block, 0, POLAR_N);
        for (int i = 0; i < blen; i++)
            src_block[i] = (data[start+i] >> 7) ^ 1;

        compress_block(src_block, POLAR_N, POLAR_D,
                       g_is_frozen, g_frozen_bits, g_info_set, POLAR_K,
                       pay_block);
        decompress_block(pay_block, POLAR_K,
                         g_is_frozen, g_frozen_bits, g_info_set, POLAR_N,
                         recon);

        for (int i = 0; i < blen; i++) {
            matches      += (src_block[i] == recon[i]);
            n_polar_ones += recon[i];
            n_true_pos   += (src_block[i] & recon[i]);
            if (recon[i]) data[start+i] ^= 0x80;
        }

        // Pack K info bits (LSB-first)
        for (int k = 0; k < POLAR_K; k++) {
            if (pay_block[k]) all_info[bit_cur >> 3] |= 1u << (bit_cur & 7);
            bit_cur++;
        }
    }

    free(src_block); free(pay_block); free(recon);
    *info_out          = all_info;
    *n_info_bytes_out  = n_info_bytes;
    *n_polar_ones_out  = n_polar_ones;
    *n_true_pos_out    = n_true_pos;
    return matches;
}

// ── rANS ───────────────────────────────────────────────────────────────────
#define RANS_SCALE  14
#define RANS_M      (1u << RANS_SCALE)
#define RANS_L      (1u << 23)
#define RANS_HDR    (4 + 256*4 + 4)   // n(4) + freq(1024) + state(4) = 1032

typedef struct { u32 freq[256]; u32 cum[257]; u8 sym[RANS_M]; } RT;

static void rt_build(RT *t, const u8 *d, int n) {
    u32 raw[256]={0}; for(int i=0;i<n;i++) raw[d[i]]++;
    u32 sum=0;
    for(int s=0;s<256;s++){
        if(!raw[s]){t->freq[s]=0;continue;}
        t->freq[s]=(u32)(((u64)raw[s]*RANS_M)/n);
        if(!t->freq[s]) t->freq[s]=1;
        sum+=t->freq[s];
    }
    while(sum<RANS_M){ int b=0; for(int s=1;s<256;s++) if(t->freq[s]>t->freq[b]) b=s; t->freq[b]++;sum++;}
    while(sum>RANS_M){ int b=-1; for(int s=0;s<256;s++) if(t->freq[s]>1&&(b<0||t->freq[s]>t->freq[b]))b=s; t->freq[b]--;sum--;}
    t->cum[0]=0; for(int s=0;s<256;s++) t->cum[s+1]=t->cum[s]+t->freq[s];
    for(int s=0;s<256;s++) for(u32 j=t->cum[s];j<t->cum[s+1];j++) t->sym[j]=(u8)s;
}

static void rt_from_freq(RT *t, const u32 freq[256]) {
    memcpy(t->freq,freq,1024);
    t->cum[0]=0; for(int s=0;s<256;s++) t->cum[s+1]=t->cum[s]+t->freq[s];
    for(int s=0;s<256;s++) for(u32 j=t->cum[s];j<t->cum[s+1];j++) t->sym[j]=(u8)s;
}

static inline void renc(u32 *x, u8 **p, const RT *t, int s) {
    u32 f=t->freq[s], xm=((RANS_L>>RANS_SCALE)<<8)*f;
    while(*x>=xm){*--(*p)=(u8)(*x&0xFF);*x>>=8;}
    *x=((*x/f)<<RANS_SCALE)+t->cum[s]+(*x%f);
}

static inline int rdec(u32 *x, const u8 **p, const u8 *end, const RT *t) {
    u32 sl=*x&(RANS_M-1); int s=t->sym[sl];
    *x=t->freq[s]*(*x>>RANS_SCALE)+sl-t->cum[s];
    while(*x<RANS_L&&*p<end){*x=(*x<<8)|**p;(*p)++;}
    return s;
}

// Returns malloc'd [RANS_HDR + payload]; sets *out_len.
static u8 *rans_compress(const u8 *in, int n, int *out_len) {
    RT t; rt_build(&t,in,n);
    int cap=2*n+4096; u8 *buf=malloc(cap), *p=buf+cap; u32 x=RANS_L;
    for(int i=n-1;i>=0;i--) renc(&x,&p,&t,in[i]);
    int pay=(int)((buf+cap)-p);
    *out_len=RANS_HDR+pay;
    u8 *out=malloc(*out_len), *q=out;
    memcpy(q,&n,4);q+=4; memcpy(q,t.freq,1024);q+=1024; memcpy(q,&x,4);q+=4;
    memcpy(q,p,pay);
    free(buf); return out;
}

// ── Layer header: fixed part stored for every layer ──────────────────────────
// Decompression order per layer:
//   read n_info_bytes of polar info → undo_polar_bias(payload) → rans_decompress
typedef struct {
    u32 rans_n;           // number of symbols rANS encoded
    u32 freq[256];        // rANS frequency table
    u32 rans_state;       // rANS initial decoder state
    u32 n_info_bytes;     // bytes of polar info that follow (in the info section)
} LayerHdrBase;           // 4 + 1024 + 4 + 4 = 1036 bytes

#define LAYER_HDR_BASE  ((int)sizeof(LayerHdrBase))   // 1036

// ── Pipeline compressor ─────────────────────────────────────────────────────
u8 *pipeline_compress(const u8 *in, int n, int *out_len) {
    u8 *cur = malloc(n);
    memcpy(cur, in, n);
    int cur_n = n;

    LayerHdrBase *base_hdrs    = NULL;
    u8          **layer_info   = NULL;
    int          *layer_ibytes = NULL;
    int           nlayers      = 0;
    int           total_info   = 0;

    printf("Input: %d bytes  entropy %.4f bpb\n\n", n, byte_entropy(in, n));

    for (int lay = 0; lay < MAX_LAYERS; lay++) {
        if (cur_n < 512) {
            printf("Layer %d: payload too small (%d bytes), stopping.\n", lay, cur_n);
            break;
        }

        printf("─── Layer %d  [%d bytes, %.4f bpb] ───\n",
               lay, cur_n, byte_entropy(cur, cur_n));

        int clen;
        u8 *comp    = rans_compress(cur, cur_n, &clen);
        int pay_len = clen - RANS_HDR;

        printf("  rANS:  %d → %d bytes payload  (%.2f%%)\n",
               cur_n, pay_len, 100.0f * pay_len / cur_n);

        // Check before the expensive SC pass: estimate polar info overhead
        int n_blocks_est = (pay_len + POLAR_N - 1) / POLAR_N;
        int info_est     = (n_blocks_est * POLAR_K + 7) / 8;
        if (LAYER_HDR_BASE + info_est + pay_len >= cur_n) {
            printf("  Header (%d) + info (%d) + payload (%d) = %d >= input (%d). Stopping.\n",
                   LAYER_HDR_BASE, info_est, pay_len,
                   LAYER_HDR_BASE + info_est + pay_len, cur_n);
            free(comp);
            break;
        }

        // Parse rANS header into layer record
        LayerHdrBase hdr;
        memcpy(&hdr.rans_n,     comp,           4);
        memcpy(hdr.freq,        comp + 4,        1024);
        memcpy(&hdr.rans_state, comp + 4 + 1024, 4);

        u8 *payload = malloc(pay_len);
        memcpy(payload, comp + RANS_HDR, pay_len);
        free(comp);

        int low_before = 0;
        for (int i = 0; i < pay_len; i++) low_before += (payload[i] < 128);
        float ent_before = byte_entropy(payload, pay_len);
        printf("  Bias:  entropy %.4f  low %d/%d (%.1f%%) →",
               ent_before, low_before, pay_len, 100.0f * low_before / pay_len);
        fflush(stdout);

        // Polar bias reduction
        u8 *info_bits;
        int n_info_bytes, n_polar_ones, n_true_pos;
        int match = polar_bias_reduce(payload, pay_len, &info_bits, &n_info_bytes,
                                      &n_polar_ones, &n_true_pos);
        hdr.n_info_bytes = (u32)n_info_bytes;

        int low_after = 0;
        for (int i = 0; i < pay_len; i++) low_after += (payload[i] < 128);
        float ent_after = byte_entropy(payload, pay_len);

        float prec = n_polar_ones > 0 ? 100.0f * n_true_pos / n_polar_ones : 0.0f;
        float recl = low_before   > 0 ? 100.0f * n_true_pos / low_before   : 0.0f;

        printf(" %.4f bpb  (Δ%.4f)  low %d/%d (%.1f%%)  match %.1f%%\n",
               ent_after, ent_before - ent_after,
               low_after, pay_len, 100.0f * low_after / pay_len,
               100.0f * match / pay_len);
        printf("         polar flipped %d/%d (%.1f%%)  precision %.1f%%  recall %.1f%%\n\n",
               n_polar_ones, pay_len, 100.0f * n_polar_ones / pay_len, prec, recl);

        // Commit layer
        base_hdrs    = realloc(base_hdrs,    (nlayers+1) * sizeof(LayerHdrBase));
        base_hdrs[nlayers] = hdr;
        layer_info   = realloc(layer_info,   (nlayers+1) * sizeof(u8*));
        layer_info[nlayers] = info_bits;
        layer_ibytes = realloc(layer_ibytes, (nlayers+1) * sizeof(int));
        layer_ibytes[nlayers] = n_info_bytes;
        nlayers++;
        total_info += n_info_bytes;

        free(cur);
        cur   = payload;
        cur_n = pay_len;
    }

    // Assemble output:
    //   [2 bytes nlayers] [LayerHdrBase × nlayers] [info bits each] [final payload]
    int hdr_bytes = 2 + nlayers * LAYER_HDR_BASE;
    *out_len = hdr_bytes + total_info + cur_n;
    u8 *out = malloc(*out_len), *p = out;

    u16 nl16 = (u16)nlayers;
    memcpy(p, &nl16, 2); p += 2;
    for (int i = 0; i < nlayers; i++) { memcpy(p, &base_hdrs[i], LAYER_HDR_BASE); p += LAYER_HDR_BASE; }
    for (int i = 0; i < nlayers; i++) { memcpy(p, layer_info[i], layer_ibytes[i]); p += layer_ibytes[i]; free(layer_info[i]); }
    memcpy(p, cur, cur_n);

    printf("Layers: %d   Header: %d bytes   Polar info: %d bytes\n",
           nlayers, hdr_bytes, total_info);
    printf("Output: %d bytes  (%.2f%% of input)\n", *out_len, 100.0f * (*out_len) / n);

    free(cur); free(base_hdrs); free(layer_info); free(layer_ibytes);
    return out;
}

// ── Main ───────────────────────────────────────────────────────────────────
int main(void) {
    polar_init();

    FILE *f = fopen(INPUT_PATH, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", INPUT_PATH); return 1; }
    fseek(f, 0, SEEK_END); int n = (int)ftell(f); rewind(f);
    u8 *data = malloc(n);
    fread(data, 1, n, f); fclose(f);

    int out_len;
    u8 *out = pipeline_compress(data, n, &out_len);

    FILE *fo = fopen(OUTPUT_PATH, "wb");
    if (fo) { fwrite(out, 1, out_len, fo); fclose(fo); printf("Saved → %s\n", OUTPUT_PATH); }
    else fprintf(stderr, "Could not write %s\n", OUTPUT_PATH);

    free(data); free(out);
    free(g_is_frozen); free(g_frozen_bits); free(g_info_set);
    return 0;
}
