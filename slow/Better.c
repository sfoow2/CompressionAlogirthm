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
#define N_OPS       6
#define N_PATTERNS  3

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


// ── Search result struct ──────────────────────────────────────────────────────
typedef struct {
    int    id;
    char   name[48];
    double entropy;
    int    overhead;
    int    p[8];
    u8     amps[256];     // per-group amplitude (2^k entries used)
    u8     grp_ops[256];  // per-group op (2^k entries used)
} SR;

// Overhead bits = 5 (ID) + parameter bits.  Per-group ops: each group stores (op 3b + amp 8b) = 11b.
// 12  COND-PREV:  k(3) + 2^k*(op(3)+amp(8))  = 30 min (k=1,N=2)  ..  184 (k=4,N=16)
//     Context = high k bits of b[i-1].  Each of the 2^k groups has its own op+amp.
// 13  COND-NEXT:  k(3) + 2^k*(op(3)+amp(8))  = 30 min (k=1)
//     Context = high k bits of b[i+1].  Uses original unmodified next byte.
// 14  COND-DELTA: k(3) + 2^k*(op(3)+amp(8))  = 30 min (k=1)
//     Context = high k bits of (b[i-1] XOR b[i-2]).  Tracks local difference trend.
//
// Per-group ops: instead of one shared op for all groups, cond_greedy_flex() independently
// picks the best (op, amp) per group.  A group whose sub-histogram responds to XOR gets XOR;
// another gets ADD; etc.  Net gain improves significantly with minimal overhead increase.
static const int PAT_OH[] = {
    0,0,0,0,0,0,0,0,0,0,0,0,  // IDs 0-11 unused
    30,                         // 12 COND-PREV  (k=1 min; dynamic per best k found)
    30,                         // 13 COND-NEXT
    30,                         // 14 COND-DELTA
};

// ── Apply a SearchResult in-place ─────────────────────────────────────────────
static void apply_sr(u8 *blk, int n, const SR *r) {
    u8 *orig=NULL;
    if(r->id==12||r->id==14){  // COND-PREV and COND-DELTA need a snapshot of the input
        orig=malloc(n); if(orig) memcpy(orig,blk,n);
    }
    switch(r->id){
        case 12: {int k=r->p[0],flex=r->p[2],op=r->p[1]; // COND-PREV
            if(orig) for(int i=1;i<n;i++){int g=orig[i-1]>>(8-k);
                blk[i]=op_byte(blk[i],r->amps[g],flex?r->grp_ops[g]:op);}} break;
        case 13: {int k=r->p[0],flex=r->p[2],op=r->p[1]; // COND-NEXT
            for(int i=0;i<n-1;i++){int g=blk[i+1]>>(8-k);
                blk[i]=op_byte(blk[i],r->amps[g],flex?r->grp_ops[g]:op);}} break;
        case 14: {int k=r->p[0],flex=r->p[2],op=r->p[1]; // COND-DELTA
            if(orig) for(int i=2;i<n;i++){int g=(orig[i-1]^orig[i-2])>>(8-k);
                blk[i]=op_byte(blk[i],r->amps[g],flex?r->grp_ops[g]:op);}} break;
    }
    free(orig);
}

// ── Helpers for search functions ──────────────────────────────────────────────
// MUL(2) needs odd amp; ADDLO(3) amp 0-15; GFMUL(5) amp >= 2
#define OP_RANGE(op,alo,ahi) int alo=1,ahi=255; \
    if(op==2)alo=3; if(op==3)ahi=15; if(op==5)alo=2;
#define SKIP_OP(op,amp) if(op==2&&(amp&1)==0) continue;

// ── Search functions ──────────────────────────────────────────────────────────

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
// Strictly more general than cond_greedy — wins on structured data where groups differ.
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

// ── Run all searches ──────────────────────────────────────────────────────────
static void run_all(const u8 *blk, int n, double base, SR *out) {
    out[0] = search_cond_prev  (blk,n,base);
    out[1] = search_cond_next  (blk,n,base);
    out[2] = search_cond_delta (blk,n,base);
}


// ── Per-operation statistics ───────────────────────────────────────────────────
typedef struct {
    char   name[32];   // base op name (no params), fixed at startup
    long   count;      // total times this slot won across all blocks
    long   opt_count;  // times in top-3 profitable but not chosen
    double sum_net;
    double min_net;
    double max_net;
} OpStat;

// ── Main ──────────────────────────────────────────────────────────────────────
#define N_STATS 30   // number of random 4KB blocks to run

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

            double nets[N_PATTERNS];
            int best_idx=-1; double best_net=-1e30;
            for(int i=0;i<N_PATTERNS;i++){
                nets[i]=(base-results[i].entropy)*BLOCK_SIZE-results[i].overhead;
                if(nets[i]>best_net){best_net=nets[i];best_idx=i;}
            }
            if(best_net<=0||best_idx<0) break;

            // Winner stats
            OpStat *s=&stats[best_idx];
            s->count++;
            s->sum_net+=best_net;
            if(best_net<s->min_net) s->min_net=best_net;
            if(best_net>s->max_net) s->max_net=best_net;

            // Top-3 optional: 2nd and 3rd best with positive net (not the winner)
            {
                int opt1=-1, opt2=-1; double on1=-1e30, on2=-1e30;
                for(int i=0;i<N_PATTERNS;i++){
                    if(i==best_idx||nets[i]<=0) continue;
                    if(nets[i]>on1){on2=on1;opt2=opt1;on1=nets[i];opt1=i;}
                    else if(nets[i]>on2){on2=nets[i];opt2=i;}
                }
                if(opt1>=0) stats[opt1].opt_count++;
                if(opt2>=0) stats[opt2].opt_count++;
            }

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
    // Sort by count desc, then opt_count desc as tiebreak
    int order[N_PATTERNS];
    for(int i=0;i<N_PATTERNS;i++) order[i]=i;
    for(int i=1;i<N_PATTERNS;i++){
        int k=order[i]; int j=i-1;
        while(j>=0 && (stats[order[j]].count < stats[k].count ||
                       (stats[order[j]].count == stats[k].count &&
                        stats[order[j]].opt_count < stats[k].opt_count))){
            order[j+1]=order[j]; j--;
        }
        order[j+1]=k;
    }

    int n_never=0;  // neither won nor runner-up
    for(int i=0;i<N_PATTERNS;i++)
        if(stats[i].count==0 && stats[i].opt_count==0) n_never++;

    printf("\n=== OPERATION USE COUNT (%d blocks) ===\n", N_STATS);
    printf("  %-28s  %6s  %6s  %8s  %8s  %8s\n",
           "Operation", "Count", "Opt", "Avg Net", "Min Net", "Max Net");
    printf("  %-28s  %6s  %6s  %8s  %8s  %8s\n",
           "----------------------------","------","------","--------","--------","--------");

    for(int i=0;i<N_PATTERNS;i++){
        int ii=order[i];
        if(stats[ii].count==0 && stats[ii].opt_count==0) break;
        double avg = stats[ii].count ? stats[ii].sum_net/stats[ii].count : 0.0;
        // Mark runner-up-only rows with "opt" tag
        const char *tag = (stats[ii].count==0) ? " [opt-only]" : "";
        printf("  %-28s  %6ld  %6ld  %8.1f  %8.1f  %8.1f%s\n",
               stats[ii].name, stats[ii].count, stats[ii].opt_count,
               avg,
               stats[ii].count ? stats[ii].min_net : 0.0,
               stats[ii].max_net,
               tag);
    }

    printf("\n--- NEVER TRIGGERED (%d / %d) — candidates to remove: ---\n",
           n_never, N_PATTERNS);
    for(int i=0;i<N_PATTERNS;i++){
        int ii=order[i];
        if(stats[ii].count>0 || stats[ii].opt_count>0) continue;
        printf("  slot%03d  %s\n", ii, stats[ii].name);
    }

    return 0;
}
