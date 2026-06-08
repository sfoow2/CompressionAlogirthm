/*
 * simple.c — greedy entropy compressor
 *
 * Batch same-stride: after findBest picks the winner, every other phase at
 * that stride gets its own evalPair call and is applied if net > 0.  Phases
 * are disjoint, so this is exact and saves ~10x findBest calls.
 *
 * 16 instruction types:
 *   0  XOR_PHASE     1  ADD_PHASE     2  MUL_ODD       3  COND_LO_XOR
 *   4  ADD_NIBS      5  DUAL_XOR      6  DUAL_ADD      7  ROT_PHASE
 *   8  COND_HI_XOR   9  COND_HI_ADD  10  COND_LO_ADD  11  QUAD_XOR
 *  12  QUAD_ADD     13  VALUE_SWAP   14  OCTET_ADD
 *     VALUE_SWAP: swap values A↔B in phase group; amp = A|(B<<8); cost = 16
 *     OCTET_ADD:  8-group ADD; amp = a0..a3, mask = a4..a7; cost = 40
 *  (removed: DELTA_PHASE, DUAL_XOR_ADD, MUL_NIBS — never fired, save 4-bit type)
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include <windows.h>
#include <bcrypt.h>

typedef uint8_t u8;

#define BLOCK_SIZE 4096
#define NUM_BLOCKS 1

/* ── log table ───────────────────────────────────────────────────────────── */
static double hlog[BLOCK_SIZE + 1];
static void init_hlog(void) {
    hlog[0] = 0.0;
    for (int i = 1; i <= BLOCK_SIZE; i++) hlog[i] = i * log2(i);
}

/* ── entropy ─────────────────────────────────────────────────────────────── */
static double entropy(const u8 *data, int n) {
    int f[256] = {0};
    for (int i = 0; i < n; i++) f[data[i]]++;
    double s = 0.0;
    for (int i = 0; i < 256; i++) s += hlog[f[i]];
    return log2(n) - s / n;
}

/* ── data ────────────────────────────────────────────────────────────────── */
static void fill_random(u8 *buf, int n) {
    BCryptGenRandom(NULL, buf, (ULONG)n, BCRYPT_USE_SYSTEM_PREFERRED_RNG);
}

/* ── inverse tables mod 256 and mod 16 ──────────────────────────────────── */
static u8 mul_inv[256];    /* mod-256 inverse for odd values */

static void init_inv_tables(void) {
    for (int k = 1; k < 256; k += 2)
        for (int inv = 1; inv < 256; inv += 2)
            if (((k * inv) & 0xFF) == 1) { mul_inv[k] = (u8)inv; break; }
}

/* ── instructions ────────────────────────────────────────────────────────── */
typedef enum {
    XOR_PHASE    =  0, /* data[i] ^= amp */
    ADD_PHASE    =  1, /* data[i] += amp */
    MUL_ODD      =  2, /* data[i] *= amp (amp odd 3-255) */
    COND_LO_XOR  =  3, /* if hi-nib==(amp>>4): lo-nib ^= (amp&0xF) */
    ADD_NIBS     =  4, /* lo+=(amp&0xF), hi+=(amp>>4) mod 16 */
    DUAL_XOR     =  5, /* even^=lo, odd^=hi; amp=lo|(hi<<8) */
    DUAL_ADD     =  6, /* even+=lo, odd+=hi */
    ROT_PHASE    =  7, /* ROL(data[i], amp) for amp=1..7 */
    COND_HI_XOR  =  8, /* if lo-nib==(amp&0xF): hi-nib ^= ((amp>>4)<<4) */
    COND_HI_ADD  =  9, /* if lo-nib==(amp&0xF): hi-nib += (amp>>4) mod 16 */
    COND_LO_ADD  = 10, /* if hi-nib==(amp>>4): lo-nib += (amp&0xF) mod 16 */
    QUAD_XOR     = 11, /* 4-group XOR; amp=a0|(a1<<8)|(a2<<16)|(a3<<24) */
    QUAD_ADD     = 12, /* 4-group ADD */
    VALUE_SWAP   = 13, /* swap values A↔B in phase group; amp = A|(B<<8) */
    OCTET_ADD    = 14, /* 8-group ADD cycling k%8; amp=a0..a3, mask=a4..a7 */
    NUM_INSTR_TYPES
} InstrType;

static const char *INSTR_NAMES[NUM_INSTR_TYPES] = {
    "XOR_PHASE",  "ADD_PHASE",  "MUL_ODD",    "COND_LO_XOR",
    "ADD_NIBS",   "DUAL_XOR",   "DUAL_ADD",   "ROT_PHASE",
    "COND_HI_XOR","COND_HI_ADD","COND_LO_ADD","QUAD_XOR",
    "QUAD_ADD",   "VALUE_SWAP", "OCTET_ADD"
};

/* overhead bits per instruction type (used in net = entropy_gain - overhead) */
static const double INSTR_COST[NUM_INSTR_TYPES] = {
    16, 16, 16, 16,   /* XOR ADD MUL COND_LO_XOR */
    16, 24, 24,  8,   /* ADD_NIBS DUAL_XOR DUAL_ADD ROT */
    16, 16, 16, 32,   /* COND_HI_XOR COND_HI_ADD COND_LO_ADD QUAD_XOR */
    32, 16, 40        /* QUAD_ADD VALUE_SWAP OCTET_ADD */
};

typedef struct { InstrType type; int stride, phase, amp; unsigned mask; } Instr;

/* ── apply ───────────────────────────────────────────────────────────────── */
static void applyInstr(u8 *data, int n, Instr t) {
    int i;
    switch (t.type) {
        case XOR_PHASE:
            for (i = t.phase; i < n; i += t.stride) data[i] ^= (u8)t.amp;
            break;
        case ADD_PHASE:
            for (i = t.phase; i < n; i += t.stride) data[i] += (u8)t.amp;
            break;
        case MUL_ODD:
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)((data[i] * (u8)t.amp) & 0xFF);
            break;
        case COND_LO_XOR: {
            int nc = (t.amp >> 4) & 0xF, xv = t.amp & 0xF;
            for (i = t.phase; i < n; i += t.stride)
                if ((data[i] >> 4) == nc) data[i] ^= (u8)xv;
            break;
        }
        case ADD_NIBS: {
            u8 lo = (u8)(t.amp & 0xF), hi = (u8)((t.amp >> 4) & 0xF);
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)(((data[i] + lo) & 0xF) | ((((data[i]>>4) + hi) & 0xF) << 4));
            break;
        }
        case DUAL_XOR: {
            int lo = t.amp & 0xFF, hi = (t.amp >> 8) & 0xFF, k = 0;
            for (i = t.phase; i < n; i += t.stride, k++)
                data[i] ^= (u8)(k & 1 ? hi : lo);
            break;
        }
        case DUAL_ADD: {
            int lo = t.amp & 0xFF, hi = (t.amp >> 8) & 0xFF, k = 0;
            for (i = t.phase; i < n; i += t.stride, k++)
                data[i] += (u8)(k & 1 ? hi : lo);
            break;
        }
        case ROT_PHASE: {
            int k = t.amp & 7;
            for (i = t.phase; i < n; i += t.stride)
                data[i] = (u8)(((unsigned)data[i] << k) | ((unsigned)data[i] >> (8-k)));
            break;
        }
        case COND_HI_XOR: {
            int nc = t.amp & 0xF, xv = (t.amp >> 4) & 0xF;
            for (i = t.phase; i < n; i += t.stride)
                if ((data[i] & 0xF) == nc) data[i] ^= (u8)(xv << 4);
            break;
        }
        case COND_HI_ADD: {
            int nc = t.amp & 0xF, av = (t.amp >> 4) & 0xF;
            for (i = t.phase; i < n; i += t.stride)
                if ((data[i] & 0xF) == nc)
                    data[i] = (u8)((data[i] & 0x0F) | (((data[i]>>4) + av) & 0xF) << 4);
            break;
        }
        case COND_LO_ADD: {
            int nc = (t.amp >> 4) & 0xF, av = t.amp & 0xF;
            for (i = t.phase; i < n; i += t.stride)
                if ((data[i] >> 4) == nc)
                    data[i] = (u8)((data[i] & 0xF0) | ((data[i] + av) & 0xF));
            break;
        }
        case QUAD_XOR: {
            unsigned ua = (unsigned)t.amp;
            int a[4] = {ua&0xFF, (ua>>8)&0xFF, (ua>>16)&0xFF, (ua>>24)&0xFF};
            int k = 0;
            for (i = t.phase; i < n; i += t.stride, k++) data[i] ^= (u8)a[k & 3];
            break;
        }
        case QUAD_ADD: {
            unsigned ua = (unsigned)t.amp;
            int a[4] = {ua&0xFF, (ua>>8)&0xFF, (ua>>16)&0xFF, (ua>>24)&0xFF};
            int k = 0;
            for (i = t.phase; i < n; i += t.stride, k++) data[i] += (u8)a[k & 3];
            break;
        }
        case VALUE_SWAP: {
            u8 A = (u8)(t.amp & 0xFF), B = (u8)((t.amp >> 8) & 0xFF);
            for (i = t.phase; i < n; i += t.stride)
                if (data[i] == A) data[i] = B; else if (data[i] == B) data[i] = A;
            break;
        }
        case OCTET_ADD: {
            unsigned ua = (unsigned)t.amp, ub = t.mask;
            u8 a[8] = {ua&0xFF,(ua>>8)&0xFF,(ua>>16)&0xFF,ua>>24,
                       ub&0xFF,(ub>>8)&0xFF,(ub>>16)&0xFF,ub>>24};
            int k = 0;
            for (i = t.phase; i < n; i += t.stride, k++) data[i] += a[k % 8];
            break;
        }
        default: break;
    }
}

/*
 * ── evalPair ──────────────────────────────────────────────────────────────
 * Evaluate all 16 instruction types at one (stride, phase) pair.
 * Updates *bestNet/*best if a better instruction is found.
 */
static void evalPair(const u8 *data, int n, const int *tot, double Sbase,
                     int stride, int phase, double *bestNet, Instr *best) {
    int phF[256]; memset(phF, 0, sizeof phF);
    for (int i = phase; i < n; i += stride) phF[data[i]]++;

    /* XOR_PHASE + ADD_PHASE */
    for (int amp = 1; amp < 256; amp++) {
        double Sx = 0.0, Sa = 0.0;
        for (int v = 0; v < 256; v++) {
            Sx += hlog[tot[v] - phF[v] + phF[v ^ amp]];
            Sa += hlog[tot[v] - phF[v] + phF[(v - amp) & 0xFF]];
        }
        double nx = (Sx - Sbase) - 16.0, na = (Sa - Sbase) - 16.0;
        if (nx > *bestNet) { *bestNet = nx; *best = (Instr){XOR_PHASE, stride, phase, amp}; }
        if (na > *bestNet) { *bestNet = na; *best = (Instr){ADD_PHASE, stride, phase, amp}; }
    }

    /* MUL_ODD */
    for (int amp = 3; amp < 256; amp += 2) {
        double Sm = 0.0;
        for (int v = 0; v < 256; v++)
            Sm += hlog[tot[v] - phF[v] + phF[(v * mul_inv[amp]) & 0xFF]];
        double nm = (Sm - Sbase) - 16.0;
        if (nm > *bestNet) { *bestNet = nm; *best = (Instr){MUL_ODD, stride, phase, amp}; }
    }

    /* COND_LO_XOR: if hi-nib==nc: lo-nib ^= xv */
    for (int nc = 0; nc < 16; nc++) {
        for (int xv = 1; xv < 16; xv++) {
            double delta = 0.0;
            for (int lo = 0; lo < 16; lo++) {
                int v = (nc<<4)|lo, v2 = (nc<<4)|(lo^xv);
                delta += hlog[tot[v]-phF[v]+phF[v2]] - hlog[tot[v]];
            }
            double nd = delta - 16.0;
            if (nd > *bestNet) { *bestNet = nd; *best = (Instr){COND_LO_XOR, stride, phase, (nc<<4)|xv}; }
        }
    }

    /* ADD_NIBS */
    for (int amp = 1; amp < 256; amp++) {
        int la = amp & 0xF, ha = (amp >> 4) & 0xF;
        double Sn = 0.0;
        for (int v = 0; v < 256; v++) {
            int ov = ((v-la)&0xF) | (((v>>4)-ha)&0xF)<<4;
            Sn += hlog[tot[v]-phF[v]+phF[ov]];
        }
        double nn = (Sn - Sbase) - 16.0;
        if (nn > *bestNet) { *bestNet = nn; *best = (Instr){ADD_NIBS, stride, phase, amp}; }
    }

    /* ROT_PHASE: bit rotation k=1..7 (k=4 replaces old NIB_SWAP) */
    for (int k = 1; k <= 7; k++) {
        double Sr = 0.0;
        for (int v = 0; v < 256; v++) {
            /* inverse of ROL(v,k) is ROR(v,k) */
            int rv = (int)(((unsigned)v >> k) | ((unsigned)v << (8-k))) & 0xFF;
            Sr += hlog[tot[v] - phF[v] + phF[rv]];
        }
        double nr = (Sr - Sbase) - 8.0;  /* tiny amp field → cheap */
        if (nr > *bestNet) { *bestNet = nr; *best = (Instr){ROT_PHASE, stride, phase, k}; }
    }

    /* COND_HI_XOR: if lo-nib==nc: hi-nib ^= xv */
    for (int nc = 0; nc < 16; nc++) {
        for (int xv = 1; xv < 16; xv++) {
            double delta = 0.0;
            for (int hi = 0; hi < 16; hi++) {
                int v = (hi<<4)|nc, v2 = ((hi^xv)<<4)|nc;
                delta += hlog[tot[v]-phF[v]+phF[v2]] - hlog[tot[v]];
            }
            double nd = delta - 16.0;
            if (nd > *bestNet) { *bestNet = nd; *best = (Instr){COND_HI_XOR, stride, phase, nc|(xv<<4)}; }
        }
    }

    /* COND_HI_ADD: if lo-nib==nc: hi-nib += av */
    for (int nc = 0; nc < 16; nc++) {
        for (int av = 1; av < 16; av++) {
            double delta = 0.0;
            for (int hi = 0; hi < 16; hi++) {
                int v = (hi<<4)|nc, v2 = (((hi+av)&0xF)<<4)|nc;
                delta += hlog[tot[v]-phF[v]+phF[v2]] - hlog[tot[v]];
            }
            double nd = delta - 16.0;
            if (nd > *bestNet) { *bestNet = nd; *best = (Instr){COND_HI_ADD, stride, phase, nc|(av<<4)}; }
        }
    }

    /* COND_LO_ADD: if hi-nib==nc: lo-nib += av */
    for (int nc = 0; nc < 16; nc++) {
        for (int av = 1; av < 16; av++) {
            double delta = 0.0;
            for (int lo = 0; lo < 16; lo++) {
                int v = (nc<<4)|lo, v2 = (nc<<4)|((lo+av)&0xF);
                delta += hlog[tot[v]-phF[v]+phF[v2]] - hlog[tot[v]];
            }
            double nd = delta - 16.0;
            if (nd > *bestNet) { *bestNet = nd; *best = (Instr){COND_LO_ADD, stride, phase, (nc<<4)|av}; }
        }
    }

    /* VALUE_SWAP: swap two byte values A↔B in phase group */
    for (int A = 0; A < 255; A++) {
        if (!phF[A]) continue;
        for (int B = A+1; B < 256; B++) {
            if (!phF[B]) continue;
            double d = hlog[tot[A]-phF[A]+phF[B]] + hlog[tot[B]-phF[B]+phF[A]]
                     - hlog[tot[A]] - hlog[tot[B]];
            double nd = d - 16.0;
            if (nd > *bestNet) {
                *bestNet = nd;
                *best = (Instr){VALUE_SWAP, stride, phase, A|(B<<8), 0};
            }
        }
    }

    /* --- dual / quad / octet section: build even-subset phFe once, share below --- */
    int phFe[256]; memset(phFe, 0, sizeof phFe);
    { int k = 0; for (int i = phase; i < n; i += stride, k++) if (!(k&1)) phFe[data[i]]++; }

    /* DUAL_XOR */
    {
        int blo = 1; double bS = -1e30;
        for (int amp = 1; amp < 256; amp++) {
            double S = 0.0;
            for (int v = 0; v < 256; v++) S += hlog[tot[v]-phFe[v]+phFe[v^amp]];
            if (S > bS) { bS = S; blo = amp; }
        }
        int t2[256]; for (int v=0;v<256;v++) t2[v]=tot[v]-phFe[v]+phFe[v^blo];
        int bhi = 1; double bS2 = -1e30;
        for (int amp = 1; amp < 256; amp++) {
            double S = 0.0;
            for (int v = 0; v < 256; v++)
                S += hlog[t2[v]-(phF[v]-phFe[v])+(phF[v^amp]-phFe[v^amp])];
            if (S > bS2) { bS2 = S; bhi = amp; }
        }
        double nd = (bS2 - Sbase) - 24.0;
        if (nd > *bestNet) { *bestNet = nd; *best = (Instr){DUAL_XOR, stride, phase, blo|(bhi<<8)}; }
    }

    /* DUAL_ADD */
    {
        int blo = 1; double bS = -1e30;
        for (int amp = 1; amp < 256; amp++) {
            double S = 0.0;
            for (int v = 0; v < 256; v++) S += hlog[tot[v]-phFe[v]+phFe[(v-amp)&0xFF]];
            if (S > bS) { bS = S; blo = amp; }
        }
        int t2[256]; for (int v=0;v<256;v++) t2[v]=tot[v]-phFe[v]+phFe[(v-blo)&0xFF];
        int bhi = 1; double bS2 = -1e30;
        for (int amp = 1; amp < 256; amp++) {
            double S = 0.0;
            for (int v = 0; v < 256; v++)
                S += hlog[t2[v]-(phF[v]-phFe[v])+(phF[(v-amp)&0xFF]-phFe[(v-amp)&0xFF])];
            if (S > bS2) { bS2 = S; bhi = amp; }
        }
        double nd = (bS2 - Sbase) - 24.0;
        if (nd > *bestNet) { *bestNet = nd; *best = (Instr){DUAL_ADD, stride, phase, blo|(bhi<<8)}; }
    }

    /* QUAD_XOR + QUAD_ADD + OCTET_ADD: 4/4/8-group independent transforms */
    {
        int p0[256]={0}, p1[256]={0}, p2[256]={0}, p3[256]={0};
        { int k=0; for (int i=phase;i<n;i+=stride,k++)
            switch(k&3){case 0:p0[data[i]]++;break;case 1:p1[data[i]]++;break;
                         case 2:p2[data[i]]++;break;case 3:p3[data[i]]++;break;} }

        /* ── QUAD_XOR ── */
        int qx0=1; double qS0=-1e30;
        for (int a=1;a<256;a++) { double S=0.0; for(int v=0;v<256;v++) S+=hlog[tot[v]-p0[v]+p0[v^a]]; if(S>qS0){qS0=S;qx0=a;} }
        int tx1[256]; for(int v=0;v<256;v++) tx1[v]=tot[v]-p0[v]+p0[v^qx0];
        int qx1=1; double qS1=-1e30;
        for (int a=1;a<256;a++) { double S=0.0; for(int v=0;v<256;v++) S+=hlog[tx1[v]-p1[v]+p1[v^a]]; if(S>qS1){qS1=S;qx1=a;} }
        int tx2[256]; for(int v=0;v<256;v++) tx2[v]=tx1[v]-p1[v]+p1[v^qx1];
        int qx2=1; double qS2=-1e30;
        for (int a=1;a<256;a++) { double S=0.0; for(int v=0;v<256;v++) S+=hlog[tx2[v]-p2[v]+p2[v^a]]; if(S>qS2){qS2=S;qx2=a;} }
        int tx3[256]; for(int v=0;v<256;v++) tx3[v]=tx2[v]-p2[v]+p2[v^qx2];
        int qx3=1; double qS3=-1e30;
        for (int a=1;a<256;a++) { double S=0.0; for(int v=0;v<256;v++) S+=hlog[tx3[v]-p3[v]+p3[v^a]]; if(S>qS3){qS3=S;qx3=a;} }
        {
            double nd=(qS3-Sbase)-32.0;
            if (nd>*bestNet) {
                unsigned ua=(unsigned)qx0|((unsigned)qx1<<8)|((unsigned)qx2<<16)|((unsigned)qx3<<24);
                int packed; memcpy(&packed,&ua,4);
                *bestNet=nd; *best=(Instr){QUAD_XOR,stride,phase,packed};
            }
        }

        /* ── QUAD_ADD ── */
        int qa0=1; double qaS0=-1e30;
        for (int a=1;a<256;a++) { double S=0.0; for(int v=0;v<256;v++) S+=hlog[tot[v]-p0[v]+p0[(v-a)&0xFF]]; if(S>qaS0){qaS0=S;qa0=a;} }
        int ta1[256]; for(int v=0;v<256;v++) ta1[v]=tot[v]-p0[v]+p0[(v-qa0)&0xFF];
        int qa1=1; double qaS1=-1e30;
        for (int a=1;a<256;a++) { double S=0.0; for(int v=0;v<256;v++) S+=hlog[ta1[v]-p1[v]+p1[(v-a)&0xFF]]; if(S>qaS1){qaS1=S;qa1=a;} }
        int ta2[256]; for(int v=0;v<256;v++) ta2[v]=ta1[v]-p1[v]+p1[(v-qa1)&0xFF];
        int qa2=1; double qaS2=-1e30;
        for (int a=1;a<256;a++) { double S=0.0; for(int v=0;v<256;v++) S+=hlog[ta2[v]-p2[v]+p2[(v-a)&0xFF]]; if(S>qaS2){qaS2=S;qa2=a;} }
        int ta3[256]; for(int v=0;v<256;v++) ta3[v]=ta2[v]-p2[v]+p2[(v-qa2)&0xFF];
        int qa3=1; double qaS3=-1e30;
        for (int a=1;a<256;a++) { double S=0.0; for(int v=0;v<256;v++) S+=hlog[ta3[v]-p3[v]+p3[(v-a)&0xFF]]; if(S>qaS3){qaS3=S;qa3=a;} }
        {
            double nd=(qaS3-Sbase)-32.0;
            if (nd>*bestNet) {
                unsigned ua=(unsigned)qa0|((unsigned)qa1<<8)|((unsigned)qa2<<16)|((unsigned)qa3<<24);
                int packed; memcpy(&packed,&ua,4);
                *bestNet=nd; *best=(Instr){QUAD_ADD,stride,phase,packed};
            }
        }
    }

    /* OCTET_ADD: 8-group coordinate descent ADD */
    {
        int p[8][256]; memset(p, 0, sizeof p);
        { int k=0; for(int i=phase;i<n;i+=stride,k++) p[k%8][data[i]]++; }

        int a[8]; int rt[256]; memcpy(rt, tot, 256*sizeof(int)); double fS=Sbase;
        for(int g=0; g<8; g++) {
            int ba=1; fS=-1e30;
            for(int av=1;av<256;av++) {
                double S=0.0;
                for(int v=0;v<256;v++) S+=hlog[rt[v]-p[g][v]+p[g][(v-av)&0xFF]];
                if(S>fS){fS=S;ba=av;}
            }
            a[g]=ba;
            for(int v=0;v<256;v++) rt[v]=rt[v]-p[g][v]+p[g][(v-a[g])&0xFF];
        }
        double nd=(fS-Sbase)-40.0;
        if(nd>*bestNet) {
            unsigned lo=(unsigned)a[0]|((unsigned)a[1]<<8)|((unsigned)a[2]<<16)|((unsigned)a[3]<<24);
            unsigned hi=(unsigned)a[4]|((unsigned)a[5]<<8)|((unsigned)a[6]<<16)|((unsigned)a[7]<<24);
            int packed; memcpy(&packed,&lo,4);
            *bestNet=nd; *best=(Instr){OCTET_ADD,stride,phase,packed,hi};
        }
    }
}

/* ── find best instruction across all (stride, phase) pairs ─────────────── */
static Instr findBest(const u8 *data, int n, double *netOut) {
    int tot[256] = {0};
    for (int i = 0; i < n; i++) tot[data[i]]++;
    double Sbase = 0.0;
    for (int v = 0; v < 256; v++) Sbase += hlog[tot[v]];

    double bestNet = 0.0;
    Instr best = {XOR_PHASE, 2, 0, 1};

    for (int stride = 1; stride <= 64; stride++)
        for (int phase = 0; phase < stride; phase++)
            evalPair(data, n, tot, Sbase, stride, phase, &bestNet, &best);

    if (netOut) *netOut = bestNet;
    return best;
}

/* ── scramble helpers ────────────────────────────────────────────────────── */
static void interleave_stride(const u8 *src, u8 *dst, int n, int s) {
    int w = n / s;
    for (int i = 0; i < n; i++) dst[(i % s) * w + (i / s)] = src[i];
}
static void bit_plane_sep(const u8 *src, u8 *dst, int n) {
    int ps = n / 8; memset(dst, 0, n);
    for (int i = 0; i < n; i++)
        for (int b = 0; b < 8; b++)
            if ((src[i] >> b) & 1) dst[b*ps + i/8] |= (u8)(1 << (i%8));
}
static void nib_plane_sep(const u8 *src, u8 *dst, int n) {
    int h = n / 2;
    for (int i = 0; i < h; i++) {
        dst[i]   = (u8)((src[2*i] & 0xF) | ((src[2*i+1] & 0xF) << 4));
        dst[i+h] = (u8)((src[2*i] >> 4)  | ((src[2*i+1] >> 4)  << 4));
    }
}
static void block_reverse(const u8 *src, u8 *dst, int n) {
    for (int i = 0; i < n; i++) dst[i] = src[n-1-i];
}
static void xor_fold_scramble(const u8 *src, u8 *dst, int n) {
    int h = n / 2;
    for (int i = 0; i < h; i++) dst[i] = src[i] ^ src[i+h];
    memcpy(dst+h, src+h, h);
}

/* ── greedy compress ─────────────────────────────────────────────────────── */
static double compress(u8 *data, int n, int verbose, int *counts) {
    double total_net = 0.0;
    int findBest_calls = 0;

    for (;;) {
        /* ── primary greedy loop ────────────────────────────────────────── */
        for (;;) {
            double net;
            Instr t = findBest(data, n, &net);
            findBest_calls++;
            if (net <= 0.0) break;

            double e0 = entropy(data, n);
            applyInstr(data, n, t);
            total_net += net;
            counts[t.type]++;
            if (verbose)
                printf("  %-14s s%-2d p%-2d a%-10d  %.6f -> %.6f  net=%.1f\n",
                       INSTR_NAMES[t.type], t.stride, t.phase, t.amp,
                       e0, entropy(data, n), net);

            /* Batch: apply best instruction at every other phase of t.stride.
             * Phases cover disjoint positions — evaluation is exact.
             * Rebuild tot[] after each apply to stay accurate. */
            {
                int tot[256] = {0};
                for (int i = 0; i < n; i++) tot[data[i]]++;

                for (int p = 0; p < t.stride; p++) {
                    if (p == t.phase) continue;

                    double Sbase = 0.0;
                    for (int v = 0; v < 256; v++) Sbase += hlog[tot[v]];

                    double bNet = 0.0;
                    Instr b = {XOR_PHASE, t.stride, p, 1};
                    evalPair(data, n, tot, Sbase, t.stride, p, &bNet, &b);

                    if (bNet > 0) {
                        double e0b = entropy(data, n);
                        applyInstr(data, n, b);
                        total_net += bNet;
                        counts[b.type]++;
                        if (verbose)
                            printf("  + %-12s s%-2d p%-2d a%-10d  %.6f -> %.6f  net=%.1f\n",
                                   INSTR_NAMES[b.type], b.stride, b.phase, b.amp,
                                   e0b, entropy(data, n), bNet);

                        memset(tot, 0, sizeof tot);
                        for (int i = 0; i < n; i++) tot[data[i]]++;
                    }
                }
            }
        }

        /* ── scramble-restart ────────────────────────────────────────────── */
        int found = 0;
        double E0 = entropy(data, n);
        u8 *temp  = malloc(n);
        u8 *after = malloc(n);

#define TRY_SCRAMBLE(label, cond, scramble_call) \
        if (!found && (cond)) { \
            scramble_call; \
            double net2; Instr t2 = findBest(temp, n, &net2); findBest_calls++; \
            memcpy(after, temp, n); applyInstr(after, n, t2); \
            double true_net = (E0 - entropy(after, n)) * n - 8.0; \
            if (true_net > 0) { \
                memcpy(data, temp, n); \
                double e0 = entropy(data, n); \
                applyInstr(data, n, t2); \
                total_net += true_net; counts[t2.type]++; found = 1; \
                if (verbose) \
                    printf("  [%-14s] %-14s s%-2d p%-2d a%-10d  %.6f -> %.6f  net=%.1f\n", \
                           label, INSTR_NAMES[t2.type], t2.stride, t2.phase, t2.amp, \
                           e0, entropy(data, n), true_net); \
            } \
        }

        TRY_SCRAMBLE("INTERLEAVE-2",  n%2==0,  interleave_stride(data, temp, n, 2))
        TRY_SCRAMBLE("INTERLEAVE-4",  n%4==0,  interleave_stride(data, temp, n, 4))
        TRY_SCRAMBLE("INTERLEAVE-8",  n%8==0,  interleave_stride(data, temp, n, 8))
        TRY_SCRAMBLE("INTERLEAVE-16", n%16==0, interleave_stride(data, temp, n, 16))
        TRY_SCRAMBLE("INTERLEAVE-32", n%32==0, interleave_stride(data, temp, n, 32))
        TRY_SCRAMBLE("INTERLEAVE-64", n%64==0, interleave_stride(data, temp, n, 64))
        TRY_SCRAMBLE("BIT-PLANE-SEP", n%8==0,  bit_plane_sep(data, temp, n))
        TRY_SCRAMBLE("NIB-PLANE-SEP", n%2==0,  nib_plane_sep(data, temp, n))
        TRY_SCRAMBLE("BLOCK-REVERSE", 1,       block_reverse(data, temp, n))
        TRY_SCRAMBLE("XOR-FOLD-SCR",  n%2==0,  xor_fold_scramble(data, temp, n))

#undef TRY_SCRAMBLE

        free(after);
        free(temp);
        if (!found) break;
    }

    if (verbose)
        printf("  [findBest calls: %d]\n", findBest_calls);

    return total_net;
}

/* ── main ────────────────────────────────────────────────────────────────── */
int main(void) {
    init_hlog();
    init_inv_tables();
    int counts[NUM_INSTR_TYPES] = {0};
    double sum = 0.0;

    for (int b = 0; b < NUM_BLOCKS; b++) {
        u8 *data = malloc(BLOCK_SIZE);
        fill_random(data, BLOCK_SIZE);
        double e0 = entropy(data, BLOCK_SIZE);
        { int f[256]={0}; for(int i=0;i<BLOCK_SIZE;i++) f[data[i]]++;
          int mv=0; for(int i=1;i<256;i++) if(f[i]>f[mv]) mv=i;
          printf("block %2d start: most common = 0x%02X (count %d)\n", b, mv, f[mv]); }
        double net = compress(data, BLOCK_SIZE, b == 0, counts);
        double e1 = entropy(data, BLOCK_SIZE);
        { int f[256]={0}; for(int i=0;i<BLOCK_SIZE;i++) f[data[i]]++;
          int mv=0; for(int i=1;i<256;i++) if(f[i]>f[mv]) mv=i;
          printf("block %2d end:   most common = 0x%02X (count %d)\n", b, mv, f[mv]); }
        printf("block %2d: %.6f -> %.6f  net=%.1f bits\n", b, e0, e1, net);
        sum += net;
        free(data);
    }

    printf("\navg net: %.1f bits/block over %d blocks\n", sum / NUM_BLOCKS, NUM_BLOCKS);
    printf("\ninstruction usage counts:\n");
    for (int i = 0; i < NUM_INSTR_TYPES; i++)
        printf("  %-14s %d\n", INSTR_NAMES[i], counts[i]);

    return 0;
}
