
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <windows.h>
#include <bcrypt.h>

typedef uint8_t u8;

#define BLOCK_SIZE 4096

// ── GF(256) ───────────────────────────────────────────────────────────────────
static u8 gf_exp[512], gf_log[256];
static void init_gf256(void) {
    u8 x = 1;
    for (int i = 0; i < 255; i++) {
        gf_exp[i] = x; gf_log[x] = (u8)i;
        u8 h = (x & 0x80) ? ((x << 1) ^ 0x1B) : (x << 1); x = h ^ x;
    }
    for (int i = 255; i < 512; i++) gf_exp[i] = gf_exp[i - 255];
    gf_log[0] = 0;
}
static inline u8 gf_mul(u8 a, u8 b) { return (!a || !b) ? 0 : gf_exp[gf_log[a] + gf_log[b]]; }

// ── Byte operations ───────────────────────────────────────────────────────────
static inline u8 op_byte(u8 v, u8 amp, int op) {
    switch (op) {
        case  0: return (u8)(v + amp);
        case  1: return (u8)(v ^ amp);
        case  2: return (u8)(v * (amp | 1));
        case  3: return (u8)((v & 0xF0) | ((v + amp) & 0x0F));
        case  4: { u8 s = (u8)((v << 4) | (v >> 4)); return s ^ amp; }
        case  5: { u8 a = amp < 2 ? 2 : amp; return gf_mul(v, a); }
        case  6: { u8 a = amp & 7; if (!a) a = 1; return (u8)((v << a) | (v >> (8 - a))); }
        case  7: return (u8)((((v >> 4) + amp) & 0xF) << 4 | (v & 0x0F));
        case  8: { u8 w = (u8)(v + amp); return w ^ (w >> 1); }
        case  9: return (u8)(((((v >> 4) ^ (v & 0xF) ^ (amp & 0xF)) & 0xF) << 4) | (v & 0x0F));
        case 10: return (u8)((v & 0xF0) | ((amp - v) & 0x0F));
        case 11: return (u8)((((amp - (v >> 4)) & 0x0F) << 4) | (v & 0x0F));
        case 12: return (u8)(amp - v);
        case 13: return (u8)((v & 0xF0) | ((v ^ amp) & 0x0F));
        case 14: return (u8)((v & 0x0F) | ((((v >> 4) ^ amp) & 0x0F) << 4));
        case 15: { u8 a = (u8)(amp & 7); if (!a) a = 1; return (u8)((v >> a) | (v << (8 - a))); }
    }
    return v;
}

// ── Entropy ───────────────────────────────────────────────────────────────────
static double entropyFromFreq(const int *freq, int n) {
    double e = 0.0;
    for (int i = 0; i < 256; i++) {
        if (!freq[i]) continue;
        double p = (double)freq[i] / n; e -= p * log2(p);
    }
    return e;
}

static float g_ent_tab[BLOCK_SIZE + 1];
static void init_entropy_table(void) {
    g_ent_tab[0] = 0.0f;
    for (int c = 1; c <= BLOCK_SIZE; c++) {
        double p = (double)c / BLOCK_SIZE; g_ent_tab[c] = (float)(-p * log2(p));
    }
}
static inline double entropy_fast(const int f[256]) {
    double e = 0.0; for (int i = 0; i < 256; i++) e += g_ent_tab[f[i]]; return e;
}

// ── Histogram transform ───────────────────────────────────────────────────────
static void hist_transform(const int in[256], int out[256], int op, int amp) {
    memset(out, 0, 256 * sizeof(int));
    for (int v = 0; v < 256; v++) if (in[v]) out[op_byte((u8)v, (u8)amp, op)] += in[v];
}

static const int SEARCH_OPS[] = {0,1,2,3,4,5,6,8,9,10,11,12};
#define N_SEARCH_OPS 12

#define OP_RANGE(op,alo,ahi) int alo=1,ahi=255; \
    if(op==2)alo=3; if(op==3)ahi=15; if(op==5)alo=2; if(op==6)ahi=7; if(op==7)ahi=15; \
    if(op==8)alo=0; if(op==9){alo=0;ahi=15;} \
    if(op==10){alo=0;ahi=15;} if(op==11){alo=0;ahi=15;} if(op==12)alo=0; \
    if(op==13)ahi=15; if(op==14)ahi=15; if(op==15)ahi=7;
#define SKIP_OP(op,amp) if(op==2&&(amp&1)==0) continue;

// ── SR struct ─────────────────────────────────────────────────────────────────
typedef struct {
    int    id;
    double entropy;
    int    overhead;
    int    p[8];        // p[0]=j, p[1]=l, p[2]=flex, p[3]=op
    u8     amps[256];
    u8     grp_ops[256];
} SR;

// ── Greedy amplitude search (shared op) ──────────────────────────────────────
static double cond_greedy(const int (*gh)[256], int N, int unchanged_val,
                           int op, u8 *amps_out) {
    OP_RANGE(op, alo, ahi)
    u8 amps[256]; for (int g = 0; g < N; g++) amps[g] = (u8)alo;
    int combined[256] = {0}; if (unchanged_val >= 0) combined[unchanged_val]++;
    for (int g = 0; g < N; g++) { int tf[256]; hist_transform(gh[g],tf,op,amps[g]); for (int v=0;v<256;v++) combined[v]+=tf[v]; }
    for (int pass = 0; pass < 2; pass++) for (int g = 0; g < N; g++) {
        int tc[256], wk[256]; hist_transform(gh[g], tc, op, amps[g]);
        for (int v = 0; v < 256; v++) wk[v] = combined[v] - tc[v];
        double bge = 1e30; u8 bga = amps[g]; int best_tf[256]; memcpy(best_tf, tc, 256*sizeof(int));
        for (int amp = alo; amp <= ahi; amp++) { SKIP_OP(op, amp)
            int tf[256], ff[256]; hist_transform(gh[g], tf, op, amp);
            for (int v=0;v<256;v++) ff[v]=wk[v]+tf[v];
            double e = entropy_fast(ff); if (e < bge) { bge=e; bga=(u8)amp; memcpy(best_tf,tf,256*sizeof(int)); } }
        for (int v = 0; v < 256; v++) combined[v] = wk[v] + best_tf[v];
        amps[g] = bga; }
    for (int g = 0; g < N; g++) amps_out[g] = amps[g];
    return entropy_fast(combined);
}

// ── Greedy amplitude search (per-group op) ────────────────────────────────────
static double cond_greedy_flex(const int (*gh)[256], int N, int unchanged_val,
                                u8 *amps_out, u8 *ops_out) {
    u8 amps[256], ops[256];
    for (int g = 0; g < N; g++) { amps[g] = 1; ops[g] = 0; }
    int combined[256] = {0}; if (unchanged_val >= 0) combined[unchanged_val]++;
    for (int g = 0; g < N; g++) { int tf[256]; hist_transform(gh[g],tf,ops[g],amps[g]); for (int v=0;v<256;v++) combined[v]+=tf[v]; }
    for (int pass = 0; pass < 3; pass++) for (int g = 0; g < N; g++) {
        int tc[256], wk[256]; hist_transform(gh[g], tc, ops[g], amps[g]);
        for (int v = 0; v < 256; v++) wk[v] = combined[v] - tc[v];
        double bge = 1e30; u8 bga = amps[g], bgo = ops[g]; int best_tf[256]; memcpy(best_tf, tc, 256*sizeof(int));
        for (int si = 0; si < N_SEARCH_OPS; si++) { int op = SEARCH_OPS[si]; OP_RANGE(op, alo, ahi)
            for (int amp = alo; amp <= ahi; amp++) { SKIP_OP(op, amp)
                int tf[256], ff[256]; hist_transform(gh[g], tf, op, amp);
                for (int v=0;v<256;v++) ff[v]=wk[v]+tf[v];
                double e = entropy_fast(ff); if (e < bge) { bge=e; bga=(u8)amp; bgo=(u8)op; memcpy(best_tf,tf,256*sizeof(int)); } } }
        for (int v = 0; v < 256; v++) combined[v] = wk[v] + best_tf[v];
        amps[g] = bga; ops[g] = bgo; }
    for (int g = 0; g < N; g++) { amps_out[g] = amps[g]; ops_out[g] = ops[g]; }
    return entropy_fast(combined);
}

// ── COND-PREV2: 2nd-order Markov conditional transform search ─────────────────
static SR search_cond_prev2(const u8 *blk, int n, double base) {
    SR r; memset(&r, 0, sizeof(r)); r.id = 16; r.entropy = base; r.overhead = 0;
    double best_net = -1e30;
    int bj = 1, bl = 1, bflex = 0, bop = 0;
    u8 bamps[256] = {0}, bops[256] = {0};

    int pairs[64][2], npairs = 0;
    for (int j = 1; j <= 6; j++) for (int l = 1; l <= 6; l++)
        if (j + l >= 5 && j + l <= 6) { pairs[npairs][0] = j; pairs[npairs][1] = l; npairs++; }

    for (int p = 0; p < npairs; p++) {
        int j = pairs[p][0], l = pairs[p][1];
        int N = (1 << j) * (1 << l);
        int (*gh)[256] = malloc(N * 256 * sizeof(int));
        if (!gh) continue;
        memset(gh, 0, N * 256 * sizeof(int));
        for (int i = 2; i < n; i++)
            gh[((blk[i-1] >> (8-j)) << l) | (blk[i-2] >> (8-l))][blk[i]]++;

        for (int si = 0; si < N_SEARCH_OPS; si++) { int op = SEARCH_OPS[si];
            int ab = (op==6||op==15)?3 : (op==3||op==7||op==9||op==10||op==11||op==13||op==14)?4 : (op==2)?7 : 8;
            int oh = 11 + N * (1 + ab);
            u8 amps[256] = {0};
            double e = cond_greedy(gh, N, blk[0], op, amps);
            double net = (base - e) * n - oh;
            if (net > best_net) { best_net=net; r.entropy=e; r.overhead=oh; bj=j; bl=l; bflex=0; bop=op; memcpy(bamps,amps,N); } }

        { u8 amps[256] = {0}, ops[256] = {0}; int oh = 7 + N * 13;
          double e = cond_greedy_flex(gh, N, blk[0], amps, ops);
          double net = (base - e) * n - oh;
          if (net > best_net) { best_net=net; r.entropy=e; r.overhead=oh; bj=j; bl=l; bflex=1; memcpy(bamps,amps,N); memcpy(bops,ops,N); } }

        free(gh);
    }

    r.p[0] = bj; r.p[1] = bl; r.p[2] = bflex; r.p[3] = bop;
    memcpy(r.amps, bamps, sizeof(bamps));
    if (bflex) memcpy(r.grp_ops, bops, sizeof(bops));
    return r;
}

// ── Apply COND-PREV2 ──────────────────────────────────────────────────────────
static void apply_cond_prev2(u8 *blk, int n, const SR *r) {
    int j = r->p[0], l = r->p[1], flex = r->p[2], op = r->p[3];
    u8 *orig = malloc(n); if (!orig) return;
    memcpy(orig, blk, n);
    for (int i = 2; i < n; i++) {
        int g = ((orig[i-1] >> (8-j)) << l) | (orig[i-2] >> (8-l));
        blk[i] = op_byte(blk[i], r->amps[g], flex ? r->grp_ops[g] : op);
    }
    free(orig);
}

int main(void) {
    init_gf256();
    init_entropy_table();

    int n_blocks = 16;
    int size = BLOCK_SIZE * n_blocks;
    u8 *data = malloc(size);
    if (!data) { fprintf(stderr, "malloc failed\n"); return 1; }

    if (BCryptGenRandom(NULL, (PUCHAR)data, size, BCRYPT_USE_SYSTEM_PREFERRED_RNG) != 0) {
        fprintf(stderr, "BCryptGenRandom failed\n"); return 1;
    }

    { int freq[256] = {0}; for (int i = 0; i < size; i++) freq[data[i]]++;
      printf("Original entropy: %.6f\n\n", entropyFromFreq(freq, size)); }

    int n_improved = 0;
    double sum_before = 0.0, sum_after = 0.0;

    for (int b = 0; b < n_blocks; b++) {
        u8 *blk = data + b * BLOCK_SIZE;
        int freq[256] = {0};
        for (int i = 0; i < BLOCK_SIZE; i++) freq[blk[i]]++;
        double before = entropyFromFreq(freq, BLOCK_SIZE);
        sum_before += before;

        SR r = search_cond_prev2(blk, BLOCK_SIZE, before);
        double net = (before - r.entropy) * BLOCK_SIZE - r.overhead;
        if (net > 0) { apply_cond_prev2(blk, BLOCK_SIZE, &r); n_improved++; }

        int fq[256] = {0};
        for (int i = 0; i < BLOCK_SIZE; i++) fq[blk[i]]++;
        double after = entropyFromFreq(fq, BLOCK_SIZE);
        sum_after += after;

        printf("Block %3d: %.4f -> %.4f  (net %+.0f bits, j=%d l=%d %s)%s\n",
               b, before, after, (before - after) * BLOCK_SIZE,
               r.p[0], r.p[1], r.p[2] ? "flex" : "shared",
               net > 0 ? " *" : "");
    }

    printf("\nAvg entropy: %.4f -> %.4f  (%d/%d blocks improved)\n",
           sum_before / n_blocks, sum_after / n_blocks, n_improved, n_blocks);

    free(data);
    return 0;
}
