// Raw CUDA kernel tile SpGEMM v2 — output-nnz-centric loop order
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <unordered_map>
#include <sys/time.h>
#include <argparse/argparse.hpp>
#include "tile_format.h"
#include "spgemm_raw_kernels_v2.cuh"

#include <gpu_error/progress_bar.cuh>

struct TileDevPtrs {
    int *tile_ptr, *tile_colidx, *tile_nnz;
    float *tile_val; unsigned char *tile_col, *tile_ptrRow;
    unsigned short *mask;
    int *csc_tile_ptr, *csc_tile_rowidx;
    // Within-tile CSC (for B v2)
    unsigned char *tile_csc_Ptr, *tile_csc_Row;
    float *tile_csc_Val;
};

TileDevPtrs upload_tiles(const TileMatrix& T, bool is_B) {
    TileDevPtrs d;
    auto up = [](auto* src, int n, auto*& dst) { cudaMalloc(&dst, n * sizeof(*src)); cudaMemcpy(dst, src, n * sizeof(*src), cudaMemcpyHostToDevice); };
    up(T.tile_ptr, T.tilem + 1, d.tile_ptr);
    up(T.tile_columnidx, T.numtile, d.tile_colidx);
    int nt = is_B ? T.numtileB_csc : T.numtile;
    up(T.tile_nnz, nt + 1, d.tile_nnz);
    up(T.tile_csr_Val, T.tile_nnz[nt], d.tile_val);
    up(T.tile_csr_Col, T.tile_nnz[nt], d.tile_col);
    up(T.tile_csr_Ptr, nt * TILE_SIZE, d.tile_ptrRow);
    up(T.mask, nt * TILE_SIZE, d.mask);
    d.csc_tile_ptr = nullptr; d.csc_tile_rowidx = nullptr;
    d.tile_csc_Ptr = nullptr; d.tile_csc_Row = nullptr; d.tile_csc_Val = nullptr;
    if (is_B && T.csc_tile_ptr) {
        up(T.csc_tile_ptr, T.tilen + 1, d.csc_tile_ptr);
        up(T.csc_tile_rowidx, T.numtileB_csc, d.csc_tile_rowidx);
    }
    if (is_B && T.tile_csc_Ptr) {
        up(T.tile_csc_Ptr, T.numtileB_csc * TILE_SIZE, d.tile_csc_Ptr);
        up(T.tile_csc_Row, T.tile_nnz[T.numtileB_csc], d.tile_csc_Row);
        up(T.tile_csc_Val, T.tile_nnz[T.numtileB_csc], d.tile_csc_Val);
    }
    return d;
}

void free_dev(TileDevPtrs& d) {
    cudaFree(d.tile_ptr); cudaFree(d.tile_colidx); cudaFree(d.tile_nnz);
    cudaFree(d.tile_val); cudaFree(d.tile_col); cudaFree(d.tile_ptrRow);
    cudaFree(d.mask);
    if (d.csc_tile_ptr) cudaFree(d.csc_tile_ptr);
    if (d.csc_tile_rowidx) cudaFree(d.csc_tile_rowidx);
    if (d.tile_csc_Ptr) cudaFree(d.tile_csc_Ptr);
    if (d.tile_csc_Row) cudaFree(d.tile_csc_Row);
    if (d.tile_csc_Val) cudaFree(d.tile_csc_Val);
}

int main(int argc, char** argv) {
    argparse::ArgumentParser program("spgemm_raw_v2");
    program.add_argument("--matrix", "-m").required();
    program.add_argument("--aat", "-a").default_value(0).scan<'i', int>();
    program.add_argument("--device", "-d").default_value(0).scan<'i', int>();
    try { program.parse_args(argc, argv); }
    catch (const std::exception& e) { std::cerr << e.what() << "\n" << program; return 1; }

    cudaSetDevice(program.get<int>("--device"));
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, program.get<int>("--device"));
    printf("Device [%d] %s\n", program.get<int>("--device"), prop.name);

    auto fname = program.get<std::string>("--matrix");
    printf("Loading %s\n", fname.c_str());
    HostCSR hA = load_mtx(fname.c_str());
    printf("A: %d x %d, nnz = %d\n", hA.m, hA.n, hA.nnz);

    HostCSR hB;
    if (program.get<int>("--aat")) {
        hB.m = hA.n; hB.n = hA.m; hB.nnz = hA.nnz;
        hB.row_ptr.resize(hB.m + 1, 0); hB.col_idx.resize(hB.nnz); hB.val.resize(hB.nnz);
        for (int i = 0; i < hA.nnz; i++) hB.row_ptr[hA.col_idx[i] + 1]++;
        for (int i = 0; i < hB.m; i++) hB.row_ptr[i + 1] += hB.row_ptr[i];
        std::vector<int> off(hB.row_ptr.begin(), hB.row_ptr.end());
        for (int r = 0; r < hA.m; r++)
            for (int j = hA.row_ptr[r]; j < hA.row_ptr[r + 1]; j++) {
                int c = hA.col_idx[j]; int p = off[c]++;
                hB.col_idx[p] = r; hB.val[p] = hA.val[j];
            }
    } else {
        if (hA.m != hA.n) { printf("A^2 requires square\n"); return 1; }
        hB = hA;
    }

    unsigned long long flops = 0;
    for (int i = 0; i < hA.m; i++)
        for (int j = hA.row_ptr[i]; j < hA.row_ptr[i + 1]; j++)
            flops += hB.row_ptr[hA.col_idx[j] + 1] - hB.row_ptr[hA.col_idx[j]];
    printf("SpGEMM flops = %llu\n", flops);

    TileMatrix tA = csr2tile(hA);
    TileMatrix tB = csr2tile(hB);
    build_csc_tiles(tB, hB);

    TileDevPtrs dA = upload_tiles(tA, false);
    TileDevPtrs dB = upload_tiles(tB, true);

    int tilemA = tA.tilem, tilenB = tB.tilen;
    int nmasks = (tilenB + 31) / 32;

    // ---- Step 1: tile structure of C ----
    int* d_tile_ptrC;
    cudaMalloc(&d_tile_ptrC, (tilemA + 1) * sizeof(int));

    cudaEvent_t ev0, ev1;
    cudaEventCreate(&ev0); cudaEventCreate(&ev1);
    cudaDeviceSynchronize();

    // Step1a: count
    {
        int warps_per_block = 4;
        int threads = warps_per_block * 32;
        int blocks = (tilemA + warps_per_block - 1) / warps_per_block;
        int smem = warps_per_block * nmasks * sizeof(unsigned int);
        cudaEventRecord(ev0);
        step1_count_kernel<<<blocks, threads, smem>>>(
            dA.tile_ptr, dA.tile_colidx, tilemA,
            dB.tile_ptr, dB.tile_colidx, tilenB, d_tile_ptrC);
    }

    // Step1b: exclusive scan on tile_ptrC
    {
        void* d_temp = nullptr; size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_tile_ptrC, d_tile_ptrC, tilemA + 1);
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_tile_ptrC, d_tile_ptrC, tilemA + 1);
        cudaFree(d_temp);
    }

    int numtileC;
    cudaMemcpy(&numtileC, d_tile_ptrC + tilemA, sizeof(int), cudaMemcpyDeviceToHost);

    // Step1c: fill indices
    int *d_rowidxC, *d_colidxC;
    cudaMalloc(&d_rowidxC, numtileC * sizeof(int));
    cudaMalloc(&d_colidxC, numtileC * sizeof(int));
    {
        int warps_per_block = 4;
        int threads = warps_per_block * 32;
        int blocks = (tilemA + warps_per_block - 1) / warps_per_block;
        int smem = warps_per_block * nmasks * sizeof(unsigned int);
        step1_fill_kernel<<<blocks, threads, smem>>>(
            dA.tile_ptr, dA.tile_colidx, tilemA,
            dB.tile_ptr, dB.tile_colidx, tilenB,
            d_tile_ptrC, d_rowidxC, d_colidxC);
    }
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms1; cudaEventElapsedTime(&ms1, ev0, ev1);

    // ---- Step 2: symbolic ----
    int *d_tile_nnzC;
    unsigned short *d_maskC;
    int *d_spec_cnt, *d_spec_off, *d_spec_posa, *d_spec_posb, *d_alloc_counter;
    cudaMalloc(&d_tile_nnzC, (numtileC + 1) * sizeof(int));
    cudaMalloc(&d_maskC, numtileC * TS * sizeof(unsigned short));
    cudaMalloc(&d_spec_cnt, numtileC * sizeof(int));
    cudaMalloc(&d_spec_off, numtileC * sizeof(int));
    cudaMalloc(&d_alloc_counter, sizeof(int));

    long long spec_cap = (long long)numtileC * 8;
    cudaMalloc(&d_spec_posa, spec_cap * sizeof(int));
    cudaMalloc(&d_spec_posb, spec_cap * sizeof(int));

    auto launch_symbolic = [&]() {
        cudaMemset(d_alloc_counter, 0, sizeof(int));
        int hwarps_per_block = WARPS_PER_BLOCK;
        int threads = hwarps_per_block * HALFWARP;
        int n_hwarps = (numtileC + TILES_PER_HALFWARP - 1) / TILES_PER_HALFWARP;
        int blocks = (n_hwarps + hwarps_per_block - 1) / hwarps_per_block;
        step2_symbolic_kernel<<<blocks, threads>>>(
            dA.tile_ptr, dA.tile_colidx, dA.tile_nnz, dA.tile_col, dA.tile_ptrRow, dB.mask,
            dB.csc_tile_ptr, dB.csc_tile_rowidx,
            d_rowidxC, d_colidxC, numtileC,
            d_tile_nnzC, d_maskC,
            d_spec_cnt, d_spec_off, d_spec_posa, d_spec_posb, d_alloc_counter);
    };

    cudaEventRecord(ev0);
    launch_symbolic();

    int total_matched;
    cudaMemcpy(&total_matched, d_alloc_counter, sizeof(int), cudaMemcpyDeviceToHost);
    if (total_matched > spec_cap) {
        printf("  spec realloc: %lld -> %d\n", spec_cap, total_matched);
        cudaFree(d_spec_posa); cudaFree(d_spec_posb);
        spec_cap = total_matched;
        cudaMalloc(&d_spec_posa, spec_cap * sizeof(int));
        cudaMalloc(&d_spec_posb, spec_cap * sizeof(int));
        launch_symbolic();
    }

    {
        void* d_temp = nullptr; size_t temp_bytes = 0;
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_tile_nnzC, d_tile_nnzC, numtileC + 1);
        cudaMalloc(&d_temp, temp_bytes);
        cub::DeviceScan::ExclusiveSum(d_temp, temp_bytes, d_tile_nnzC, d_tile_nnzC, numtileC + 1);
        cudaFree(d_temp);
    }

    int nnzC;
    cudaMemcpy(&nnzC, d_tile_nnzC + numtileC, sizeof(int), cudaMemcpyDeviceToHost);

    unsigned char* d_ptrRowC;
    cudaMalloc(&d_ptrRowC, numtileC * TS * sizeof(unsigned char));
    {
        int threads = 256;
        int blocks = (numtileC + threads - 1) / threads;
        step2_build_ptrRowC_kernel<<<blocks, threads>>>(d_maskC, d_ptrRowC, numtileC);
    }
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms2; cudaEventElapsedTime(&ms2, ev0, ev1);

    // ---- Step 3: numeric (v2 kernel) ----
    float *d_valC; unsigned char *d_colC;
    cudaMalloc(&d_valC, nnzC * sizeof(float));
    cudaMalloc(&d_colC, nnzC * sizeof(unsigned char));

    cudaEventRecord(ev0);
    {
        int blocks = (numtileC + V2_WARPS_PER_BLOCK - 1) / V2_WARPS_PER_BLOCK;
        step3_numeric_kernel_v2<<<blocks, V2_BLOCK_THREADS>>>(
            dA.tile_ptr, dA.tile_colidx, dA.tile_nnz, dA.tile_val, dA.tile_col, dA.tile_ptrRow,
            dB.csc_tile_ptr, dB.csc_tile_rowidx, dB.tile_nnz, dB.tile_val, dB.tile_col, dB.tile_ptrRow,
            d_rowidxC, d_colidxC, d_tile_nnzC, d_maskC, d_ptrRowC,
            d_valC, d_colC,
            d_spec_cnt, d_spec_off, d_spec_posa, d_spec_posb,
            numtileC);
    }
    cudaEventRecord(ev1); cudaEventSynchronize(ev1);
    float ms3; cudaEventElapsedTime(&ms3, ev0, ev1);

    auto err = cudaGetLastError();
    if (err != cudaSuccess) printf("CUDA error: %s\n", cudaGetErrorString(err));

    float ms = ms1 + ms2 + ms3;
    printf("\nRaw Tile SpGEMM v2: tiles_C = %d, nnzC = %d\n", numtileC, nnzC);
    printf("  Step1 (tile structure) = %.2f ms\n", ms1);
    printf("  Step2 (symbolic+scan)  = %.2f ms\n", ms2);
    printf("  Step3 (numeric v2)     = %.2f ms\n", ms3);
    printf("GPU time = %.2f ms, GFlops = %.2f\n", ms, 2.0 * flops / (ms * 1e6));

    // ---- Correctness verification ----
    {
        printf("\n=== Correctness Verification ===\n");
        int M = hA.m, N = hB.n;

        std::vector<int> h_tile_ptrC(tilemA + 1), h_rowidxC(numtileC), h_colidxC(numtileC);
        std::vector<int> h_tile_nnzC(numtileC + 1);
        std::vector<unsigned char> h_colC(nnzC);
        std::vector<float> h_valC(nnzC);
        cudaMemcpy(h_tile_ptrC.data(), d_tile_ptrC, (tilemA+1)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_rowidxC.data(), d_rowidxC, numtileC*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_colidxC.data(), d_colidxC, numtileC*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_tile_nnzC.data(), d_tile_nnzC, (numtileC+1)*sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_colC.data(), d_colC, nnzC*sizeof(unsigned char), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_valC.data(), d_valC, nnzC*sizeof(float), cudaMemcpyDeviceToHost);

        std::vector<std::vector<std::pair<int,float>>> gpu_C(M);
        for (int ti = 0; ti < numtileC; ti++) {
            int tr = h_rowidxC[ti], tc = h_colidxC[ti];
            int base_nnz = h_tile_nnzC[ti];
            int tile_total = h_tile_nnzC[ti+1] - base_nnz;
            for (int j = 0; j < tile_total; j++) {
                unsigned char packed = h_colC[base_nnz + j];
                int local_row = (packed >> 4) & 0xf;
                int local_col = packed & 0xf;
                int global_row = tr * TS + local_row;
                int global_col = tc * TS + local_col;
                if (global_row < M && global_col < N)
                    gpu_C[global_row].push_back({global_col, h_valC[base_nnz + j]});
            }
        }
        for (int i = 0; i < M; i++)
            std::sort(gpu_C[i].begin(), gpu_C[i].end());

        int missing_gpu = 0, extra_gpu = 0, total_ref = 0;
        double frob_diff2 = 0.0, frob_ref2 = 0.0;

        std::vector<double> acc(N, 0.0);
        std::vector<bool> used(N, false);
        std::vector<int> cols;

        {

        gpu_error::progress_bar bar("CPU Correctness Check", M, .001);

        for (int i = 0; i < M; i++) {
            cols.clear();
            for (int ja = hA.row_ptr[i]; ja < hA.row_ptr[i+1]; ja++) {
                int k = hA.col_idx[ja];
                float va = hA.val[ja];
                for (int jb = hB.row_ptr[k]; jb < hB.row_ptr[k+1]; jb++) {
                    int c = hB.col_idx[jb];
                    if (!used[c]) { used[c] = true; cols.push_back(c); }
                    acc[c] += va * hB.val[jb];
                }
            }
            std::sort(cols.begin(), cols.end());
            total_ref += (int)cols.size();

            size_t gi = 0;
            for (int ci = 0; ci < (int)cols.size(); ci++) {
                int c = cols[ci];
                double rv = acc[c];
                frob_ref2 += rv * rv;
                while (gi < gpu_C[i].size() && gpu_C[i][gi].first < c) {
                    extra_gpu++;
                    double gv = gpu_C[i][gi].second;
                    frob_diff2 += gv * gv;
                    gi++;
                }
                if (gi < gpu_C[i].size() && gpu_C[i][gi].first == c) {
                    double d = rv - gpu_C[i][gi].second;
                    frob_diff2 += d * d;
                    gi++;
                } else {
                    missing_gpu++;
                    frob_diff2 += rv * rv;
                }
            }
            while (gi < gpu_C[i].size()) {
                extra_gpu++;
                double gv = gpu_C[i][gi].second;
                frob_diff2 += gv * gv;
                gi++;
            }

            for (int c : cols) { acc[c] = 0.0; used[c] = false; }

            bar.increment();
        }

        }

        int gpu_nnz_total = 0;
        for (int i = 0; i < M; i++) gpu_nnz_total += (int)gpu_C[i].size();

        double rel_frob = sqrt(frob_diff2 / frob_ref2);
        printf("  Reference nnz:  %d\n", total_ref);
        printf("  GPU nnz:        %d (tile nnzC = %d)\n", gpu_nnz_total, nnzC);
        printf("  Missing in GPU: %d\n", missing_gpu);
        printf("  Extra in GPU:   %d\n", extra_gpu);
        printf("  ||C_ref - C_gpu||_F / ||C_ref||_F = %.6e\n", rel_frob);

        if (missing_gpu == 0 && extra_gpu == 0 && rel_frob < 1e-5)
            printf("  *** PASS ***\n");
        else
            printf("  *** FAIL ***\n");
    }

    // Cleanup
    cudaFree(d_tile_ptrC); cudaFree(d_rowidxC); cudaFree(d_colidxC);
    cudaFree(d_tile_nnzC); cudaFree(d_maskC); cudaFree(d_ptrRowC);
    cudaFree(d_valC); cudaFree(d_colC);
    cudaFree(d_spec_cnt); cudaFree(d_spec_off); cudaFree(d_spec_posa); cudaFree(d_spec_posb);
    cudaFree(d_alloc_counter);
    free_dev(dA); free_dev(dB);
    free_tile_matrix(tA); free_tile_matrix(tB);
    cudaEventDestroy(ev0); cudaEventDestroy(ev1);
    printf("Done.\n");
    return 0;
}
