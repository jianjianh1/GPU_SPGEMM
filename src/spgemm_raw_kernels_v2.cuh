// spgemm_raw_kernels_v2.cuh — v2: full warp per tile, 2 threads per row
// Split A-row scan across thread pairs, shuffle-reduce to combine.
#pragma once
#include "spgemm_raw_kernels.cuh"

#define V2_WARPS_PER_BLOCK 8
#define V2_BLOCK_THREADS (V2_WARPS_PER_BLOCK * 32)  // 256
#define V2_MIN_BLOCKS 8  // compiler uses this for register pressure; ptxas warning is harmless

__global__ void __launch_bounds__(V2_BLOCK_THREADS)
step3_numeric_kernel_v2(
    const int* __restrict__ tile_ptrA, const int* __restrict__ tile_colidxA,
    const int* __restrict__ tile_nnzA, const TVAL* __restrict__ tile_valA,
    const unsigned char* __restrict__ tile_colA, const unsigned char* __restrict__ tile_ptrRowA,
    const int* __restrict__ csc_tile_ptrB, const int* __restrict__ csc_tile_rowidxB,
    const int* __restrict__ tile_nnzB, const TVAL* __restrict__ tile_valB,
    const unsigned char* __restrict__ tile_colB, const unsigned char* __restrict__ tile_ptrRowB,
    const int* __restrict__ tile_rowidxC, const int* __restrict__ tile_colidxC,
    const int* __restrict__ tile_nnzC_prefix, const unsigned short* __restrict__ maskC,
    const unsigned char* __restrict__ ptrRowC,
    TVAL* __restrict__ tile_valC, unsigned char* __restrict__ tile_colC,
    const int* __restrict__ spec_cnt, const int* __restrict__ spec_off,
    const int* __restrict__ spec_posa, const int* __restrict__ spec_posb,
    int numtileC)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int ti = gid >> 5;
    int lane = gid & 31;
    if (ti >= numtileC) return;

    int nnzcstart = tile_nnzC_prefix[ti];
    int nnzctotal = tile_nnzC_prefix[ti + 1] - nnzcstart;
    if (!nnzctotal) return;

    int bi = tile_rowidxC[ti], bj = tile_colidxC[ti];
    int row = lane >> 1;       // 0..15
    int sub = lane & 1;        // 0 or 1: which half of A-row entries

    int matchedcnt = spec_cnt[ti];
    int local_warp = threadIdx.x >> 5;
    __shared__ int s_posa[V2_WARPS_PER_BLOCK * MAX_MATCHED];
    __shared__ int s_posb[V2_WARPS_PER_BLOCK * MAX_MATCHED];
    int* my_posa = s_posa + local_warp * MAX_MATCHED;
    int* my_posb = s_posb + local_warp * MAX_MATCHED;

    if (matchedcnt <= MAX_MATCHED) {
        int soff = spec_off[ti];
        for (int i = lane; i < matchedcnt; i += 32) {
            my_posa[i] = spec_posa[soff + i];
            my_posb[i] = spec_posb[soff + i];
        }
    }

    TVAL r0=0,r1=0,r2=0,r3=0,r4=0,r5=0,r6=0,r7=0,
         r8=0,r9=0,r10=0,r11=0,r12=0,r13=0,r14=0,r15=0;

    #define ACCUM(col, val) \
        switch(col) { \
            case 0:  r0  += val; break; case 1:  r1  += val; break; \
            case 2:  r2  += val; break; case 3:  r3  += val; break; \
            case 4:  r4  += val; break; case 5:  r5  += val; break; \
            case 6:  r6  += val; break; case 7:  r7  += val; break; \
            case 8:  r8  += val; break; case 9:  r9  += val; break; \
            case 10: r10 += val; break; case 11: r11 += val; break; \
            case 12: r12 += val; break; case 13: r13 += val; break; \
            case 14: r14 += val; break; case 15: r15 += val; break; \
        }

    auto do_multiply = [&](int aidx, int bidx) {
        int nnzAs = __ldg(&tile_nnzA[aidx]);
        int nnzAtot = __ldg(&tile_nnzA[aidx + 1]) - nnzAs;
        int nnzBs = __ldg(&tile_nnzB[bidx]);
        int nnzBtot = __ldg(&tile_nnzB[bidx + 1]) - nnzBs;

        int aRowS = __ldg(&tile_ptrRowA[aidx * TS + row]);
        int aRowE = (row == TS - 1) ? nnzAtot : (int)__ldg(&tile_ptrRowA[aidx * TS + row + 1]);
        const unsigned char* __restrict__ cA = tile_colA + nnzAs;
        const TVAL* __restrict__ vA = tile_valA + nnzAs;
        const unsigned char* __restrict__ cB = tile_colB + nnzBs;
        const TVAL* __restrict__ vB = tile_valB + nnzBs;
        const unsigned char* __restrict__ pRB = tile_ptrRowB + bidx * TS;

        for (int a = aRowS + sub; a < aRowE; a += 2) {
            int k = __ldg(&cA[a]) & 0xf;
            TVAL va = __ldg(&vA[a]);

            int bPs = __ldg(&pRB[k]);
            int bPe = (k == TS - 1) ? nnzBtot : (int)__ldg(&pRB[k + 1]);

            for (int b = bPs; b < bPe; b++) {
                TVAL prod = va * __ldg(&vB[b]);
                ACCUM(__ldg(&cB[b]), prod)
            }
        }
    };

    if (matchedcnt <= MAX_MATCHED) {
        for (int mi = 0; mi < matchedcnt; mi++)
            do_multiply(my_posa[mi], my_posb[mi]);
    } else {
        int astart = tile_ptrA[bi], astop = tile_ptrA[bi + 1];
        int bstart = csc_tile_ptrB[bj], bstop = csc_tile_ptrB[bj + 1];
        int ai = astart, bii = bstart;
        while (ai < astop && bii < bstop) {
            int ka = __ldg(&tile_colidxA[ai]), kb = __ldg(&csc_tile_rowidxB[bii]);
            if (ka == kb) { do_multiply(ai, bii); ai++; bii++; }
            else if (ka < kb) ai++; else bii++;
        }
    }

    // Reduce: combine partner's registers via shuffle
    #define SHFL_REDUCE(r) r += __shfl_xor_sync(0xffffffff, r, 1)
    SHFL_REDUCE(r0);  SHFL_REDUCE(r1);  SHFL_REDUCE(r2);  SHFL_REDUCE(r3);
    SHFL_REDUCE(r4);  SHFL_REDUCE(r5);  SHFL_REDUCE(r6);  SHFL_REDUCE(r7);
    SHFL_REDUCE(r8);  SHFL_REDUCE(r9);  SHFL_REDUCE(r10); SHFL_REDUCE(r11);
    SHFL_REDUCE(r12); SHFL_REDUCE(r13); SHFL_REDUCE(r14); SHFL_REDUCE(r15);
    #undef SHFL_REDUCE

    // Only sub==0 writes (it has the full row result)
    if (sub == 0) {
        //grab mask for my row.

        //mc can be calculated
        unsigned short mc = maskC[ti * TS + row];

        //ptr start can be calculated as prefix sum of popcnt(mc)
        int ptrStart = ptrRowC[ti * TS + row];
        TVAL racc[TS] = {r0,r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15};
        int cnt = 0;
        for (int c = 0; c < TS; c++) {
            if ((mc >> (TS - 1 - c)) & 1) {
                tile_valC[nnzcstart + ptrStart + cnt] = racc[c];
                tile_colC[nnzcstart + ptrStart + cnt] = (unsigned char)((row << 4) | c);
                cnt++;
            }
        }
    }
    #undef ACCUM
}
