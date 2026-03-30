// spgemm_raw_kernels.cuh — raw CUDA kernels for tile SpGEMM (no Andes)
#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
namespace cg = cooperative_groups;

#define TS 16
#define MAX_MATCHED 32
#define HALFWARP 16
#define WARPS_PER_BLOCK 8
#define BLOCK_THREADS (HALFWARP * WARPS_PER_BLOCK)  // 128
#define TILES_PER_HALFWARP 8

// ---- Binary search ----
__device__ __forceinline__ int raw_bsearch(const int* arr, int lo, int hi, int key) {
    while (lo <= hi) { int m = (lo+hi)/2; int v = __ldg(&arr[m]); if (v==key) return m; if (v<key) lo=m+1; else hi=m-1; }
    return -1;
}

// ---- Step1: count tiles per tile-row of C (one warp per tile-row) ----
__global__ void step1_count_kernel(
    const int* __restrict__ tile_ptrA, const int* __restrict__ tile_colidxA, int tilemA,
    const int* __restrict__ csc_tile_ptrB, const int* __restrict__ csc_tile_rowidxB, int tilenB,
    int* tile_ptrC)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = gid >> 5;
    int lane = gid & 31;
    if (warp_id >= tilemA) return;

    int nmasks = (tilenB + 31) / 32;
    // Use dynamic shared memory for bitmask
    extern __shared__ unsigned int s_bm[];
    int local_warp = (threadIdx.x >> 5);
    unsigned int* bm = s_bm + local_warp * nmasks;

    for (int i = lane; i < nmasks; i += 32) bm[i] = 0;
    __syncwarp();

    for (int i = tile_ptrA[warp_id]; i < tile_ptrA[warp_id + 1]; i++) {
        int k = __ldg(&tile_colidxA[i]);
        int bstart = __ldg(&csc_tile_ptrB[k]);
        int bstop = __ldg(&csc_tile_ptrB[k + 1]);
        for (int j = bstart + lane; j < bstop; j += 32) {
            int col = __ldg(&csc_tile_rowidxB[j]);
            atomicOr(&bm[col / 32], 1u << (31 - col % 32));
        }
    }
    __syncwarp();

    int cnt = 0;
    for (int i = lane; i < nmasks; i += 32) cnt += __popc(bm[i]);
    for (int d = 16; d > 0; d >>= 1) cnt += __shfl_xor_sync(0xffffffff, cnt, d);
    if (lane == 0) tile_ptrC[warp_id] = cnt;
}

// ---- Step1b: fill tile row/col indices of C ----
__global__ void step1_fill_kernel(
    const int* __restrict__ tile_ptrA, const int* __restrict__ tile_colidxA, int tilemA,
    const int* __restrict__ csc_tile_ptrB, const int* __restrict__ csc_tile_rowidxB, int tilenB,
    const int* __restrict__ tile_ptrC, int* tile_rowidxC, int* tile_colidxC)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = gid >> 5;
    int lane = gid & 31;
    if (warp_id >= tilemA) return;

    int nmasks = (tilenB + 31) / 32;
    extern __shared__ unsigned int s_bm[];
    int local_warp = (threadIdx.x >> 5);
    unsigned int* bm = s_bm + local_warp * nmasks;

    for (int i = lane; i < nmasks; i += 32) bm[i] = 0;
    __syncwarp();

    for (int i = tile_ptrA[warp_id]; i < tile_ptrA[warp_id + 1]; i++) {
        int k = __ldg(&tile_colidxA[i]);
        int bstart = __ldg(&csc_tile_ptrB[k]);
        int bstop = __ldg(&csc_tile_ptrB[k + 1]);
        for (int j = bstart + lane; j < bstop; j += 32) {
            int col = __ldg(&csc_tile_rowidxB[j]);
            atomicOr(&bm[col / 32], 1u << (31 - col % 32));
        }
    }
    __syncwarp();

    if (lane == 0) {
        int cbase = tile_ptrC[warp_id], pos = 0;
        for (int i = 0; i < nmasks; i++) {
            unsigned int bits = bm[i];
            while (bits) {
                int bit = __clz(bits);
                tile_rowidxC[cbase + pos] = warp_id;
                tile_colidxC[cbase + pos] = i * 32 + bit;
                pos++;
                bits &= ~(1u << (31 - bit));
            }
        }
    }
}

// ---- Step2: symbolic — compute nnz per C tile + cache intersection ----
__global__ void step2_symbolic_kernel(
    const int* __restrict__ tile_ptrA, const int* __restrict__ tile_colidxA,
    const int* __restrict__ tile_nnzA, const unsigned char* __restrict__ tile_colA,
    const unsigned char* __restrict__ tile_ptrRowA, const unsigned short* __restrict__ maskB,
    const int* __restrict__ csc_tile_ptrB, const int* __restrict__ csc_tile_rowidxB,
    const int* __restrict__ tile_rowidxC, const int* __restrict__ tile_colidxC,
    int numtileC,
    int* tile_nnzC, unsigned short* maskC,
    int* spec_cnt, int* spec_off, int* spec_posa, int* spec_posb, int* alloc_counter)
{
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    int hwarp = gid >> 4;  // half-warp id
    int lane = gid & 15;
    int local_hwarp = (threadIdx.x >> 4);

    int tile_start = hwarp * TILES_PER_HALFWARP;
    if (tile_start >= numtileC) return;
    int tile_end = min(tile_start + TILES_PER_HALFWARP, numtileC);

    cg::thread_block_tile<16> team = cg::tiled_partition<16>(cg::this_thread_block());

    __shared__ int s_posa[WARPS_PER_BLOCK * MAX_MATCHED];
    __shared__ int s_posb[WARPS_PER_BLOCK * MAX_MATCHED];
    __shared__ int s_cnt[WARPS_PER_BLOCK];
    __shared__ unsigned short s_maskb[WARPS_PER_BLOCK * TS];

    int* my_posa = s_posa + local_hwarp * MAX_MATCHED;
    int* my_posb = s_posb + local_hwarp * MAX_MATCHED;
    int* my_cnt = s_cnt + local_hwarp;
    unsigned short* my_maskb = s_maskb + local_hwarp * TS;

    for (int ti = tile_start; ti < tile_end; ti++) {
        int bi = tile_rowidxC[ti], bj = tile_colidxC[ti];
        unsigned int rmaskc = 0;
        if (lane == 0) *my_cnt = 0;
        team.sync();

        int astart = tile_ptrA[bi], astop = tile_ptrA[bi + 1];
        int bstart = csc_tile_ptrB[bj], bstop = csc_tile_ptrB[bj + 1];
        int lena = astop - astart, lenb = bstop - bstart;

        if (lena <= lenb) {
            for (int i = lane; i < lena; i += HALFWARP) {
                int ka = __ldg(&tile_colidxA[astart + i]);
                int p = raw_bsearch(csc_tile_rowidxB, bstart, bstop - 1, ka);
                if (p >= 0) { int s = atomicAdd(my_cnt, 1); if (s < MAX_MATCHED) { my_posa[s] = astart + i; my_posb[s] = p; } }
            }
        } else {
            for (int i = lane; i < lenb; i += HALFWARP) {
                int kb = __ldg(&csc_tile_rowidxB[bstart + i]);
                int p = raw_bsearch(tile_colidxA, astart, astop - 1, kb);
                if (p >= 0) { int s = atomicAdd(my_cnt, 1); if (s < MAX_MATCHED) { my_posa[s] = p; my_posb[s] = bstart + i; } }
            }
        }
        team.sync();
        int matchedcnt = *my_cnt;

        int soff = -1;
        if (matchedcnt >= 1 && matchedcnt <= MAX_MATCHED) {
            if (lane == 0) soff = atomicAdd(alloc_counter, matchedcnt);
            soff = team.shfl(soff, 0);
            for (int i = lane; i < matchedcnt; i += HALFWARP) {
                spec_posa[soff + i] = my_posa[i];
                spec_posb[soff + i] = my_posb[i];
            }
        }
        if (lane == 0) { spec_cnt[ti] = matchedcnt; spec_off[ti] = soff; }

        auto do_sym = [&](int aidx, int bidx) {
            my_maskb[lane] = __ldg(&maskB[bidx * TS + lane]);
            team.sync();
            int nnzAs = __ldg(&tile_nnzA[aidx]);
            int nnzAe = __ldg(&tile_nnzA[aidx + 1]);
            int aRowS = __ldg(&tile_ptrRowA[aidx * TS + lane]);
            int aRowE = (lane == TS-1) ? (nnzAe - nnzAs) : (int)__ldg(&tile_ptrRowA[aidx * TS + lane + 1]);
            const unsigned char* __restrict__ cA = tile_colA + nnzAs;
            for (int a = aRowS; a < aRowE; a++)
                rmaskc |= my_maskb[__ldg(&cA[a]) & 0xf];
            team.sync();
        };

        if (matchedcnt <= MAX_MATCHED) {
            for (int mi = 0; mi < matchedcnt; mi++) do_sym(my_posa[mi], my_posb[mi]);
        } else {
            int ai = astart, bii = bstart;
            while (ai < astop && bii < bstop) {
                int ka = __ldg(&tile_colidxA[ai]), kb = __ldg(&csc_tile_rowidxB[bii]);
                if (ka == kb) { do_sym(ai, bii); ai++; bii++; }
                else if (ka < kb) ai++; else bii++;
            }
        }

        maskC[ti * TS + lane] = (unsigned short)rmaskc;
        int my_cnt_val = __popc(rmaskc);
        my_cnt_val = cg::reduce(team, my_cnt_val, cg::plus<int>());
        if (lane == 0) tile_nnzC[ti] = my_cnt_val;
    }
}

// ---- Step2b: build ptrRowC (exclusive scan per tile) ----
__global__ void step2_build_ptrRowC_kernel(
    const unsigned short* __restrict__ maskC, unsigned char* ptrRowC, int numtileC)
{
    int ti = blockIdx.x * blockDim.x + threadIdx.x;
    if (ti >= numtileC) return;
    unsigned char s = 0;
    for (int r = 0; r < TS; r++) {
        ptrRowC[ti * TS + r] = s;
        s += (unsigned char)__popc(maskC[ti * TS + r]);
    }
}

// template <uint32_t register_ID>
// void accumulate(left_row, right_row){

//     constexpr if (register_ID == 0){

//         asm volatile("add.u32 rax %1 %2")
//     }

// }


// for (i in my_nz){

//     for (match in left_row, right_row){
//         racc[i] += left[match], right[match];
//     }

// }


//modified loop order - i->j->k
// for each active nnz in c
// grab i,j in CSR
// grab row i,-
// grab row j,_

//pointer trawl both lists - 
// increment smaller k value - ++ and mult on match.

//rowstart, rowend,
// if (colA[i]==colB[j]){
//     racc[nnz_current] = vals[i]*vals[j];
// }

//both matrices in CSR

//parallelize over the threads in the thread block

//run over all pairs that match

//when you do a pair of tiles, read current value from shared sparse accumulator

//thread picks an output, reads current value, sums up all results in register, writes back to smem

//explicit naming of acc register and uses control flow bypass.

//TODOS: Implement new loop order and vary tile size 4->32

// ---- Step3: numeric — one half-warp per tile ----
__global__ void step3_numeric_kernel(
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
    int hwarp = gid >> 4;
    int lane = gid & 15;
    if (hwarp >= numtileC) return;

    cg::thread_block_tile<16> team = cg::tiled_partition<16>(cg::this_thread_block());

    int ti = hwarp;
    int nnzcstart = tile_nnzC_prefix[ti];
    int nnzctotal = tile_nnzC_prefix[ti + 1] - nnzcstart;
    if (!nnzctotal) return;

    int bi = tile_rowidxC[ti], bj = tile_colidxC[ti];
    int row = lane;

    int matchedcnt = spec_cnt[ti];

    int local_hwarp = (threadIdx.x >> 4);
    __shared__ int s_posa[WARPS_PER_BLOCK * MAX_MATCHED];
    __shared__ int s_posb[WARPS_PER_BLOCK * MAX_MATCHED];
    int* my_posa = s_posa + local_hwarp * MAX_MATCHED;
    int* my_posb = s_posb + local_hwarp * MAX_MATCHED;

    if (matchedcnt <= MAX_MATCHED) {
        int soff = spec_off[ti];
        for (int i = lane; i < matchedcnt; i += HALFWARP) {
            my_posa[i] = spec_posa[soff + i];
            my_posb[i] = spec_posb[soff + i];
        }
        team.sync();
    }

    // Register accumulator
    TVAL racc[TS];
    for (int c = 0; c < TS; c++) racc[c] = 0.0f;

    // lambda to multiply two tiles
    auto do_multiply = [&](int aidx, int bidx) {

        //load tile components
        int nnzAs = __ldg(&tile_nnzA[aidx]);
        int nnzAtot = __ldg(&tile_nnzA[aidx+1]) - nnzAs;
        int nnzBs = __ldg(&tile_nnzB[bidx]);
        int nnzBtot = __ldg(&tile_nnzB[bidx+1]) - nnzBs;

        //row Start / row End of A
        int aRowS = __ldg(&tile_ptrRowA[aidx * TS + row]);
        int aRowE = (row == TS-1) ? nnzAtot : (int)__ldg(&tile_ptrRowA[aidx * TS + row+1]);
        const unsigned char* __restrict__ cA = tile_colA + nnzAs;
        const TVAL* __restrict__ vA = tile_valA + nnzAs;
        const unsigned char* __restrict__ cB = tile_colB + nnzBs;
        const TVAL* __restrict__ vB = tile_valB + nnzBs;
        const unsigned char* __restrict__ pRB = tile_ptrRowB + bidx * TS;

        //for each item in row of K
        for (int a = aRowS; a < aRowE; a++) {
            int k = __ldg(&cA[a]) & 0xf;
            TVAL va = __ldg(&vA[a]);

            //identify matching items in colB - start and end
            int bPs = __ldg(&pRB[k]);
            int bPe = (k == TS-1) ? nnzBtot : (int)__ldg(&pRB[k+1]);

            //for each output - sum to register accumulator.
            for (int b = bPs; b < bPe; b++)
                racc[__ldg(&cB[b])] += va * __ldg(&vB[b]);
        }
    };

    if (matchedcnt <= MAX_MATCHED) {
        for (int mi = 0; mi < matchedcnt; mi++)
            do_multiply(my_posa[mi], my_posb[mi]);
    } else {
        int astart = tile_ptrA[bi], astop = tile_ptrA[bi+1];
        int bstart = csc_tile_ptrB[bj], bstop = csc_tile_ptrB[bj+1];
        int ai = astart, bii = bstart;
        while (ai < astop && bii < bstop) {
            int ka = __ldg(&tile_colidxA[ai]), kb = __ldg(&csc_tile_rowidxB[bii]);
            if (ka == kb) { do_multiply(ai, bii); ai++; bii++; }
            else if (ka < kb) ai++; else bii++;
        }
    }

    // Writeback
    unsigned short mc = maskC[ti * TS + row];
    int ptrStart = ptrRowC[ti * TS + row];
    int cnt = 0;
    for (int c = 0; c < TS; c++) {
        if ((mc >> (TS - 1 - c)) & 1) {
            tile_valC[nnzcstart + ptrStart + cnt] = racc[c];
            tile_colC[nnzcstart + ptrStart + cnt] = (unsigned char)((row << 4) | c);
            cnt++;
        }
    }
}
