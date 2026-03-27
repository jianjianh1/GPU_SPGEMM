// tile_format.h — CSR-to-tile conversion (host side) + tile data structures
// Adapted from TileSpGEMM's csr2tile.h for use with Andes
#pragma once
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <numeric>

#define TILE_SIZE 16

using TVAL = float;

struct TileMatrix {
    int m, n, nnz;
    int tilem, tilen, numtile;
    // CSR-level
    int* rowpointer;
    int* columnindex;
    TVAL* value;
    // Tile-level
    int* tile_ptr;       // tilem+1: tile row pointers
    int* tile_columnidx; // numtile: tile column index
    int* tile_nnz;       // numtile+1: prefix sum of nnz per tile
    TVAL* tile_csr_Val;  // nnz: values packed per tile
    unsigned char* tile_csr_Col; // nnz: packed (row<<4|col) per nonzero
    unsigned char* tile_csr_Ptr; // numtile*TILE_SIZE: per-row nnz offset within tile
    unsigned short* mask;        // numtile*TILE_SIZE: bitmask per row of each tile
    // CSC tile-level (for B)
    int* csc_tile_ptr;
    int* csc_tile_rowidx;
    int numtileB_csc;  // number of tiles in CSC (transposed) tiling
    // Within-tile CSC for B (v2): tiles in CSC tile order, entries within each tile in CSC order
    unsigned char* tile_csc_Ptr;  // numtileB_csc * TILE_SIZE: per-column offset within tile
    unsigned char* tile_csc_Row;  // totalNnz: local row index (k % TS) per entry
    TVAL* tile_csc_Val;           // totalNnz: values in within-tile CSC order
    // Within-tile CSC for A (v3 outer product): tiles in CSR tile order, entries in CSC order
    unsigned char* tile_a_csc_Ptr;  // numtile * TILE_SIZE: per-column offset within tile
    unsigned char* tile_a_csc_Row;  // totalNnz: local row index per entry
    TVAL* tile_a_csc_Val;           // totalNnz: values in within-tile CSC order
};

struct HostCSR {
    int m, n, nnz;
    std::vector<int> row_ptr, col_idx;
    std::vector<TVAL> val;
};

// ---- MTX loader ----
inline HostCSR load_mtx(const char* filename) {
    FILE* f = fopen(filename, "r");
    if (!f) { printf("Cannot open %s\n", filename); exit(1); }
    char line[1024];
    bool is_sym = false, is_pat = false, is_complex = false;
    if (fgets(line, sizeof(line), f)) {
        if (strstr(line, "symmetric")) is_sym = true;
        if (strstr(line, "pattern")) is_pat = true;
        if (strstr(line, "complex")) is_complex = true;
    }
    while (fgets(line, sizeof(line), f)) { if (line[0] != '%') break; }
    int m, n, nf;
    sscanf(line, "%d %d %d", &m, &n, &nf);
    std::vector<int> rows, cols; std::vector<TVAL> vals;
    for (int i = 0; i < nf; i++) {
        int r, c; double v = 1.0, vi = 0.0;
        if (is_pat) fscanf(f, "%d %d", &r, &c);
        else if (is_complex) fscanf(f, "%d %d %lf %lf", &r, &c, &v, &vi);
        else fscanf(f, "%d %d %lf", &r, &c, &v);
        r--; c--;
        rows.push_back(r); cols.push_back(c); vals.push_back((TVAL)v);
        if (is_sym && r != c) {
            rows.push_back(c); cols.push_back(r); vals.push_back((TVAL)v);
        }
    }
    fclose(f);
    int nnz_total = (int)rows.size();
    std::vector<int> ord(nnz_total); std::iota(ord.begin(), ord.end(), 0);
    std::sort(ord.begin(), ord.end(), [&](int a, int b) {
        return rows[a] < rows[b] || (rows[a] == rows[b] && cols[a] < cols[b]);
    });
    HostCSR csr; csr.m = m; csr.n = n; csr.nnz = nnz_total;
    csr.row_ptr.resize(m + 1, 0); csr.col_idx.resize(nnz_total); csr.val.resize(nnz_total);
    for (int i = 0; i < nnz_total; i++) {
        csr.col_idx[i] = cols[ord[i]]; csr.val[i] = vals[ord[i]];
        csr.row_ptr[rows[ord[i]] + 1]++;
    }
    for (int i = 0; i < m; i++) csr.row_ptr[i + 1] += csr.row_ptr[i];
    return csr;
}

// ---- CSR to tile (row-major) ----
inline TileMatrix csr2tile(const HostCSR& csr) {
    TileMatrix T;
    T.m = csr.m; T.n = csr.n; T.nnz = csr.nnz;
    T.tilem = (T.m + TILE_SIZE - 1) / TILE_SIZE;
    T.tilen = (T.n + TILE_SIZE - 1) / TILE_SIZE;

    // Step 1: count tiles per tile-row
    T.tile_ptr = (int*)calloc(T.tilem + 1, sizeof(int));
    for (int bi = 0; bi < T.tilem; bi++) {
        std::vector<char> flag(T.tilen, 0);
        int rstart = bi * TILE_SIZE;
        int rend = std::min(rstart + TILE_SIZE, T.m);
        for (int j = csr.row_ptr[rstart]; j < csr.row_ptr[rend]; j++) {
            int tc = csr.col_idx[j] / TILE_SIZE;
            if (!flag[tc]) { flag[tc] = 1; T.tile_ptr[bi + 1]++; }
        }
    }
    // Prefix sum
    for (int i = 0; i < T.tilem; i++) T.tile_ptr[i + 1] += T.tile_ptr[i];
    T.numtile = T.tile_ptr[T.tilem];

    // Step 2: fill tile column indices + count nnz per tile
    T.tile_columnidx = (int*)calloc(T.numtile, sizeof(int));
    T.tile_nnz = (int*)calloc(T.numtile + 1, sizeof(int));
    T.tile_csr_Ptr = (unsigned char*)calloc(T.numtile * TILE_SIZE, sizeof(unsigned char));

    for (int bi = 0; bi < T.tilem; bi++) {
        std::vector<char> col_flag(T.tilen, 0);
        std::vector<int> nnz_per_tc(T.tilen, 0);
        std::vector<std::vector<unsigned char>> ptr_per_tc(T.tilen, std::vector<unsigned char>(TILE_SIZE, 0));
        int rstart = bi * TILE_SIZE;
        int rend = std::min(rstart + TILE_SIZE, T.m);
        int rowlen = rend - rstart;

        for (int ri = 0; ri < rowlen; ri++)
            for (int j = csr.row_ptr[rstart + ri]; j < csr.row_ptr[rstart + ri + 1]; j++) {
                int tc = csr.col_idx[j] / TILE_SIZE;
                col_flag[tc] = 1; nnz_per_tc[tc]++; ptr_per_tc[tc][ri]++;
            }

        int cnt = 0;
        int base = T.tile_ptr[bi];
        for (int tc = 0; tc < T.tilen; tc++) {
            if (col_flag[tc]) {
                T.tile_columnidx[base + cnt] = tc;
                T.tile_nnz[base + cnt + 1] = nnz_per_tc[tc];
                for (int ri = 0; ri < TILE_SIZE; ri++)
                    T.tile_csr_Ptr[(base + cnt) * TILE_SIZE + ri] = ptr_per_tc[tc][ri];
                cnt++;
            }
        }
    }

    // Prefix sum on tile_nnz
    for (int i = 0; i < T.numtile; i++) T.tile_nnz[i + 1] += T.tile_nnz[i];

    // Convert tile_csr_Ptr to exclusive scan per tile
    for (int t = 0; t < T.numtile; t++) {
        unsigned char* p = T.tile_csr_Ptr + t * TILE_SIZE;
        unsigned char sum = 0;
        for (int r = 0; r < TILE_SIZE; r++) { unsigned char v = p[r]; p[r] = sum; sum += v; }
    }

    // Step 3: fill values + packed col indices + masks
    T.tile_csr_Val = (TVAL*)calloc(T.nnz, sizeof(TVAL));
    T.tile_csr_Col = (unsigned char*)calloc(T.nnz, sizeof(unsigned char));
    T.mask = (unsigned short*)calloc(T.numtile * TILE_SIZE, sizeof(unsigned short));

    for (int bi = 0; bi < T.tilem; bi++) {
        int rstart = bi * TILE_SIZE;
        int rend = std::min(rstart + TILE_SIZE, T.m);
        int rowlen = rend - rstart;
        int ntiles = T.tile_ptr[bi + 1] - T.tile_ptr[bi];
        int tbase = T.tile_ptr[bi];

        std::vector<int> tile_count(ntiles, 0);
        // Temp storage
        int max_nnz = T.tile_nnz[tbase + ntiles] - T.tile_nnz[tbase];
        std::vector<unsigned char> tmp_col(max_nnz, 0);
        std::vector<TVAL> tmp_val(max_nnz, 0);

        for (int j = csr.row_ptr[rstart]; j < csr.row_ptr[rend]; j++) {
            int tc = csr.col_idx[j] / TILE_SIZE;
            // Find which local tile
            for (int ti = 0; ti < ntiles; ti++) {
                if (T.tile_columnidx[tbase + ti] == tc) {
                    int pre = T.tile_nnz[tbase + ti] - T.tile_nnz[tbase];
                    tmp_val[pre + tile_count[ti]] = csr.val[j];
                    tmp_col[pre + tile_count[ti]] = (unsigned char)(csr.col_idx[j] - tc * TILE_SIZE);
                    tile_count[ti]++;
                    break;
                }
            }
        }

        // Now reorder into row-major within each tile and build packed col + mask
        for (int ti = 0; ti < ntiles; ti++) {
            int tid = tbase + ti;
            int tilennz = T.tile_nnz[tid + 1] - T.tile_nnz[tid];
            int offset = T.tile_nnz[tid];
            int pre = T.tile_nnz[tid] - T.tile_nnz[tbase];
            unsigned char* ptr = T.tile_csr_Ptr + tid * TILE_SIZE;

            for (int ri = 0; ri < rowlen; ri++) {
                int start = ptr[ri];
                int stop = (ri == rowlen - 1) ? tilennz : ptr[ri + 1];
                for (int k = start; k < stop; k++) {
                    unsigned char colidx = tmp_col[pre + k];
                    T.tile_csr_Val[offset + k] = tmp_val[pre + k];
                    T.tile_csr_Col[offset + k] = (ri << 4) | colidx;
                    T.mask[tid * TILE_SIZE + ri] |= (unsigned short)(1 << (TILE_SIZE - 1 - colidx));
                }
            }
        }
    }

    T.rowpointer = nullptr; T.columnindex = nullptr; T.value = nullptr;
    T.csc_tile_ptr = nullptr; T.csc_tile_rowidx = nullptr; T.numtileB_csc = 0;
    T.tile_csc_Ptr = nullptr; T.tile_csc_Row = nullptr; T.tile_csc_Val = nullptr;
    T.tile_a_csc_Ptr = nullptr; T.tile_a_csc_Row = nullptr; T.tile_a_csc_Val = nullptr;
    return T;
}

// ---- CSR to tile with parameterized tile size ----
inline TileMatrix csr2tile_ts(const HostCSR& csr, int ts) {
    TileMatrix T;
    T.m = csr.m; T.n = csr.n; T.nnz = csr.nnz;
    T.tilem = (T.m + ts - 1) / ts;
    T.tilen = (T.n + ts - 1) / ts;

    T.tile_ptr = (int*)calloc(T.tilem + 1, sizeof(int));
    for (int bi = 0; bi < T.tilem; bi++) {
        std::vector<char> flag(T.tilen, 0);
        int rstart = bi * ts, rend = std::min(rstart + ts, T.m);
        for (int j = csr.row_ptr[rstart]; j < csr.row_ptr[rend]; j++) {
            int tc = csr.col_idx[j] / ts;
            if (!flag[tc]) { flag[tc] = 1; T.tile_ptr[bi + 1]++; }
        }
    }
    for (int i = 0; i < T.tilem; i++) T.tile_ptr[i + 1] += T.tile_ptr[i];
    T.numtile = T.tile_ptr[T.tilem];

    T.tile_columnidx = (int*)calloc(T.numtile, sizeof(int));
    T.tile_nnz = (int*)calloc(T.numtile + 1, sizeof(int));
    T.tile_csr_Ptr = (unsigned char*)calloc(T.numtile * ts, sizeof(unsigned char));

    for (int bi = 0; bi < T.tilem; bi++) {
        std::vector<char> col_flag(T.tilen, 0);
        std::vector<int> nnz_per_tc(T.tilen, 0);
        std::vector<std::vector<unsigned char>> ptr_per_tc(T.tilen, std::vector<unsigned char>(ts, 0));
        int rstart = bi * ts, rend = std::min(rstart + ts, T.m);
        int rowlen = rend - rstart;
        for (int ri = 0; ri < rowlen; ri++)
            for (int j = csr.row_ptr[rstart + ri]; j < csr.row_ptr[rstart + ri + 1]; j++) {
                int tc = csr.col_idx[j] / ts;
                col_flag[tc] = 1; nnz_per_tc[tc]++; ptr_per_tc[tc][ri]++;
            }
        int cnt = 0, base = T.tile_ptr[bi];
        for (int tc = 0; tc < T.tilen; tc++) {
            if (col_flag[tc]) {
                T.tile_columnidx[base + cnt] = tc;
                T.tile_nnz[base + cnt + 1] = nnz_per_tc[tc];
                for (int ri = 0; ri < ts; ri++)
                    T.tile_csr_Ptr[(base + cnt) * ts + ri] = ptr_per_tc[tc][ri];
                cnt++;
            }
        }
    }
    for (int i = 0; i < T.numtile; i++) T.tile_nnz[i + 1] += T.tile_nnz[i];
    for (int t = 0; t < T.numtile; t++) {
        unsigned char* p = T.tile_csr_Ptr + t * ts;
        unsigned char sum = 0;
        for (int r = 0; r < ts; r++) { unsigned char v = p[r]; p[r] = sum; sum += v; }
    }

    T.tile_csr_Val = (TVAL*)calloc(T.nnz, sizeof(TVAL));
    T.tile_csr_Col = (unsigned char*)calloc(T.nnz, sizeof(unsigned char));
    T.mask = (unsigned short*)calloc(T.numtile * ts, sizeof(unsigned short));

    for (int bi = 0; bi < T.tilem; bi++) {
        int rstart = bi * ts, rend = std::min(rstart + ts, T.m);
        int rowlen = rend - rstart;
        int ntiles = T.tile_ptr[bi + 1] - T.tile_ptr[bi];
        int tbase = T.tile_ptr[bi];
        std::vector<int> tile_count(ntiles, 0);
        int max_nnz = T.tile_nnz[tbase + ntiles] - T.tile_nnz[tbase];
        std::vector<unsigned char> tmp_col(max_nnz, 0);
        std::vector<TVAL> tmp_val(max_nnz, 0);
        for (int j = csr.row_ptr[rstart]; j < csr.row_ptr[rend]; j++) {
            int tc = csr.col_idx[j] / ts;
            for (int ti = 0; ti < ntiles; ti++) {
                if (T.tile_columnidx[tbase + ti] == tc) {
                    int pre = T.tile_nnz[tbase + ti] - T.tile_nnz[tbase];
                    tmp_val[pre + tile_count[ti]] = csr.val[j];
                    tmp_col[pre + tile_count[ti]] = (unsigned char)(csr.col_idx[j] - tc * ts);
                    tile_count[ti]++;
                    break;
                }
            }
        }
        for (int ti = 0; ti < ntiles; ti++) {
            int tid = tbase + ti;
            int tilennz = T.tile_nnz[tid + 1] - T.tile_nnz[tid];
            int offset = T.tile_nnz[tid];
            int pre = T.tile_nnz[tid] - T.tile_nnz[tbase];
            unsigned char* ptr = T.tile_csr_Ptr + tid * ts;
            for (int ri = 0; ri < rowlen; ri++) {
                int start = ptr[ri];
                int stop = (ri == rowlen - 1) ? tilennz : ptr[ri + 1];
                for (int k = start; k < stop; k++) {
                    unsigned char colidx = tmp_col[pre + k];
                    T.tile_csr_Val[offset + k] = tmp_val[pre + k];
                    T.tile_csr_Col[offset + k] = (ri << 4) | colidx;
                    T.mask[tid * ts + ri] |= (unsigned short)(1 << (ts - 1 - colidx));
                }
            }
        }
    }
    T.rowpointer = nullptr; T.columnindex = nullptr; T.value = nullptr;
    T.csc_tile_ptr = nullptr; T.csc_tile_rowidx = nullptr; T.numtileB_csc = 0;
    T.tile_csc_Ptr = nullptr; T.tile_csc_Row = nullptr; T.tile_csc_Val = nullptr;
    T.tile_a_csc_Ptr = nullptr; T.tile_a_csc_Row = nullptr; T.tile_a_csc_Val = nullptr;
    return T;
}

// ---- Build CSC tile structure with parameterized tile size ----
inline void build_csc_tiles_ts(TileMatrix& B, const HostCSR& hB, int ts) {
    int m = hB.m, n = hB.n, nnz = hB.nnz;
    int tilem = B.tilem, tilen = B.tilen;
    std::vector<int> cscColPtr(n + 1, 0), cscRowIdx(nnz);
    std::vector<TVAL> cscVal(nnz);
    for (int j = 0; j < nnz; j++) cscColPtr[hB.col_idx[j] + 1]++;
    for (int i = 0; i < n; i++) cscColPtr[i + 1] += cscColPtr[i];
    std::vector<int> cscOff(cscColPtr.begin(), cscColPtr.end());
    for (int r = 0; r < m; r++)
        for (int j = hB.row_ptr[r]; j < hB.row_ptr[r + 1]; j++) {
            int c = hB.col_idx[j]; int p = cscOff[c]++;
            cscRowIdx[p] = r; cscVal[p] = hB.val[j];
        }
    B.csc_tile_ptr = (int*)calloc(tilen + 1, sizeof(int));
    for (int bti = 0; bti < tilen; bti++) {
        std::vector<char> flag(tilem, 0);
        int rstart = bti * ts, rend = std::min(rstart + ts, n);
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int tc = cscRowIdx[j] / ts;
                if (!flag[tc]) { flag[tc] = 1; B.csc_tile_ptr[bti + 1]++; }
            }
    }
    for (int i = 0; i < tilen; i++) B.csc_tile_ptr[i + 1] += B.csc_tile_ptr[i];
    int numtileBT = B.csc_tile_ptr[tilen];
    B.numtileB_csc = numtileBT;
    B.csc_tile_rowidx = (int*)calloc(numtileBT, sizeof(int));
    for (int bti = 0; bti < tilen; bti++) {
        std::vector<char> flag(tilem, 0);
        int rstart = bti * ts, rend = std::min(rstart + ts, n);
        int pos = B.csc_tile_ptr[bti];
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int tc = cscRowIdx[j] / ts;
                if (!flag[tc]) { flag[tc] = 1; B.csc_tile_rowidx[pos++] = tc; }
            }
        std::sort(B.csc_tile_rowidx + B.csc_tile_ptr[bti], B.csc_tile_rowidx + B.csc_tile_ptr[bti + 1]);
    }
    free(B.tile_nnz);
    B.tile_nnz = (int*)calloc(numtileBT + 1, sizeof(int));
    for (int bti = 0; bti < tilen; bti++) {
        int rstart = bti * ts, rend = std::min(rstart + ts, n);
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int tc = cscRowIdx[j] / ts;
                int* base = B.csc_tile_rowidx + B.csc_tile_ptr[bti];
                int* endp = B.csc_tile_rowidx + B.csc_tile_ptr[bti + 1];
                int tidx = B.csc_tile_ptr[bti] + (int)(std::lower_bound(base, endp, tc) - base);
                B.tile_nnz[tidx + 1]++;
            }
    }
    for (int i = 0; i < numtileBT; i++) B.tile_nnz[i + 1] += B.tile_nnz[i];
    int totalNnz = B.tile_nnz[numtileBT];
    free(B.tile_csr_Ptr); B.tile_csr_Ptr = (unsigned char*)calloc(numtileBT * ts, sizeof(unsigned char));
    free(B.tile_csr_Val); B.tile_csr_Val = (TVAL*)calloc(totalNnz, sizeof(TVAL));
    free(B.tile_csr_Col); B.tile_csr_Col = (unsigned char*)calloc(totalNnz, sizeof(unsigned char));
    free(B.mask); B.mask = (unsigned short*)calloc(numtileBT * ts, sizeof(unsigned short));
    std::vector<int> tileRowCnt(numtileBT * ts, 0);
    for (int bti = 0; bti < tilen; bti++) {
        int rstart = bti * ts, rend = std::min(rstart + ts, n);
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int brow = cscRowIdx[j], tc = brow / ts, lr = brow % ts;
                int* base = B.csc_tile_rowidx + B.csc_tile_ptr[bti];
                int* endp = B.csc_tile_rowidx + B.csc_tile_ptr[bti + 1];
                int tidx = B.csc_tile_ptr[bti] + (int)(std::lower_bound(base, endp, tc) - base);
                tileRowCnt[tidx * ts + lr]++;
            }
    }
    for (int t = 0; t < numtileBT; t++) {
        unsigned char s = 0;
        for (int r = 0; r < ts; r++) { B.tile_csr_Ptr[t * ts + r] = s; s += (unsigned char)tileRowCnt[t * ts + r]; }
    }
    std::vector<int> tileRowOff(numtileBT * ts, 0);
    for (int bti = 0; bti < tilen; bti++) {
        int rstart = bti * ts, rend = std::min(rstart + ts, n);
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int brow = cscRowIdx[j], bcol = ri;
                int lc = bcol % ts, tc = brow / ts, lr = brow % ts;
                int* base = B.csc_tile_rowidx + B.csc_tile_ptr[bti];
                int* endp = B.csc_tile_rowidx + B.csc_tile_ptr[bti + 1];
                int tidx = B.csc_tile_ptr[bti] + (int)(std::lower_bound(base, endp, tc) - base);
                int off = B.tile_nnz[tidx] + B.tile_csr_Ptr[tidx * ts + lr] + tileRowOff[tidx * ts + lr];
                B.tile_csr_Val[off] = cscVal[j];
                B.tile_csr_Col[off] = (unsigned char)lc;
                B.mask[tidx * ts + lr] |= (unsigned short)(1 << (ts - 1 - lc));
                tileRowOff[tidx * ts + lr]++;
            }
    }
}

// ---- Build CSC tile structure for B (matching TileSpGEMM exactly) ----
// Tiles are ordered by column (CSC order), but within each tile the data
// is stored row-major: tile_csr_Ptr[tile*16+r] gives offset for B's row r,
// tile_csr_Col stores just the local column (0-15, NOT packed row<<4|col),
// mask[tile*16+r] is bitmask of columns in row r.
inline void build_csc_tiles(TileMatrix& B, const HostCSR& hB) {
    int m = hB.m, n = hB.n, nnz = hB.nnz;
    int tilem = B.tilem, tilen = B.tilen;
    // Transpose hB to CSC
    std::vector<int> cscColPtr(n + 1, 0), cscRowIdx(nnz);
    std::vector<TVAL> cscVal(nnz);
    for (int j = 0; j < nnz; j++) cscColPtr[hB.col_idx[j] + 1]++;
    for (int i = 0; i < n; i++) cscColPtr[i + 1] += cscColPtr[i];
    std::vector<int> cscOff(cscColPtr.begin(), cscColPtr.end());
    for (int r = 0; r < m; r++)
        for (int j = hB.row_ptr[r]; j < hB.row_ptr[r + 1]; j++) {
            int c = hB.col_idx[j]; int p = cscOff[c]++;
            cscRowIdx[p] = r; cscVal[p] = hB.val[j];
        }
    // Count tiles per tile-column of B (= tile-row of BT)
    B.csc_tile_ptr = (int*)calloc(tilen + 1, sizeof(int));
    for (int bti = 0; bti < tilen; bti++) {
        std::vector<char> flag(tilem, 0);
        int rstart = bti * TILE_SIZE, rend = std::min(rstart + TILE_SIZE, n);
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int tc = cscRowIdx[j] / TILE_SIZE;
                if (!flag[tc]) { flag[tc] = 1; B.csc_tile_ptr[bti + 1]++; }
            }
    }
    for (int i = 0; i < tilen; i++) B.csc_tile_ptr[i + 1] += B.csc_tile_ptr[i];
    int numtileBT = B.csc_tile_ptr[tilen];
    B.numtileB_csc = numtileBT;
    // Fill tile-row indices (B's tile-row for each CSC tile)
    B.csc_tile_rowidx = (int*)calloc(numtileBT, sizeof(int));
    for (int bti = 0; bti < tilen; bti++) {
        std::vector<char> flag(tilem, 0);
        int rstart = bti * TILE_SIZE, rend = std::min(rstart + TILE_SIZE, n);
        int pos = B.csc_tile_ptr[bti];
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int tc = cscRowIdx[j] / TILE_SIZE;
                if (!flag[tc]) { flag[tc] = 1; B.csc_tile_rowidx[pos++] = tc; }
            }
        std::sort(B.csc_tile_rowidx + B.csc_tile_ptr[bti],
                  B.csc_tile_rowidx + B.csc_tile_ptr[bti + 1]);
    }
    // Count nnz per tile
    free(B.tile_nnz);
    B.tile_nnz = (int*)calloc(numtileBT + 1, sizeof(int));
    for (int bti = 0; bti < tilen; bti++) {
        int rstart = bti * TILE_SIZE, rend = std::min(rstart + TILE_SIZE, n);
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int tc = cscRowIdx[j] / TILE_SIZE;
                int* base = B.csc_tile_rowidx + B.csc_tile_ptr[bti];
                int* endp = B.csc_tile_rowidx + B.csc_tile_ptr[bti + 1];
                int tidx = B.csc_tile_ptr[bti] + (int)(std::lower_bound(base, endp, tc) - base);
                B.tile_nnz[tidx + 1]++;
            }
    }
    for (int i = 0; i < numtileBT; i++) B.tile_nnz[i + 1] += B.tile_nnz[i];
    int totalNnz = B.tile_nnz[numtileBT];
    // Allocate tile data arrays
    free(B.tile_csr_Ptr); B.tile_csr_Ptr = (unsigned char*)calloc(numtileBT * TILE_SIZE, sizeof(unsigned char));
    free(B.tile_csr_Val); B.tile_csr_Val = (TVAL*)calloc(totalNnz, sizeof(TVAL));
    free(B.tile_csr_Col); B.tile_csr_Col = (unsigned char*)calloc(totalNnz, sizeof(unsigned char));
    free(B.mask); B.mask = (unsigned short*)calloc(numtileBT * TILE_SIZE, sizeof(unsigned short));
    // Count per-row nnz within each tile
    std::vector<int> tileRowCnt(numtileBT * TILE_SIZE, 0);
    for (int bti = 0; bti < tilen; bti++) {
        int rstart = bti * TILE_SIZE, rend = std::min(rstart + TILE_SIZE, n);
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int brow = cscRowIdx[j], tc = brow / TILE_SIZE, lr = brow % TILE_SIZE;
                int* base = B.csc_tile_rowidx + B.csc_tile_ptr[bti];
                int* endp = B.csc_tile_rowidx + B.csc_tile_ptr[bti + 1];
                int tidx = B.csc_tile_ptr[bti] + (int)(std::lower_bound(base, endp, tc) - base);
                tileRowCnt[tidx * TILE_SIZE + lr]++;
            }
    }
    // Build tile_csr_Ptr (exclusive scan per tile)
    for (int t = 0; t < numtileBT; t++) {
        unsigned char s = 0;
        for (int r = 0; r < TILE_SIZE; r++) {
            B.tile_csr_Ptr[t * TILE_SIZE + r] = s;
            s += (unsigned char)tileRowCnt[t * TILE_SIZE + r];
        }
    }
    // Fill values, col indices, masks
    std::vector<int> tileRowOff(numtileBT * TILE_SIZE, 0);
    for (int bti = 0; bti < tilen; bti++) {
        int rstart = bti * TILE_SIZE, rend = std::min(rstart + TILE_SIZE, n);
        for (int ri = rstart; ri < rend; ri++)
            for (int j = cscColPtr[ri]; j < cscColPtr[ri + 1]; j++) {
                int brow = cscRowIdx[j], bcol = ri;
                int lc = bcol % TILE_SIZE, tc = brow / TILE_SIZE, lr = brow % TILE_SIZE;
                int* base = B.csc_tile_rowidx + B.csc_tile_ptr[bti];
                int* endp = B.csc_tile_rowidx + B.csc_tile_ptr[bti + 1];
                int tidx = B.csc_tile_ptr[bti] + (int)(std::lower_bound(base, endp, tc) - base);
                int off = B.tile_nnz[tidx] + B.tile_csr_Ptr[tidx * TILE_SIZE + lr] + tileRowOff[tidx * TILE_SIZE + lr];
                B.tile_csr_Val[off] = cscVal[j];
                B.tile_csr_Col[off] = (unsigned char)lc;
                B.mask[tidx * TILE_SIZE + lr] |= (unsigned short)(1 << (TILE_SIZE - 1 - lc));
                tileRowOff[tidx * TILE_SIZE + lr]++;
            }
    }
    // Build within-tile CSC arrays: transpose each tile's CSR to CSC
    B.tile_csc_Ptr = (unsigned char*)calloc(numtileBT * TILE_SIZE, sizeof(unsigned char));
    B.tile_csc_Row = (unsigned char*)calloc(totalNnz, sizeof(unsigned char));
    B.tile_csc_Val = (TVAL*)calloc(totalNnz, sizeof(TVAL));
    // Count nnz per local column j within each tile
    for (int t = 0; t < numtileBT; t++) {
        int base = B.tile_nnz[t];
        int tnnz = B.tile_nnz[t + 1] - base;
        unsigned char colCnt[TILE_SIZE] = {};
        for (int e = 0; e < tnnz; e++)
            colCnt[B.tile_csr_Col[base + e]]++;
        // Exclusive scan into tile_csc_Ptr
        unsigned char s = 0;
        for (int j = 0; j < TILE_SIZE; j++) {
            B.tile_csc_Ptr[t * TILE_SIZE + j] = s;
            s += colCnt[j];
        }
    }
    // Fill CSC entries (row index = k, sorted by k within each column)
    {
        std::vector<int> colOff(numtileBT * TILE_SIZE, 0);
        for (int t = 0; t < numtileBT; t++) {
            int base = B.tile_nnz[t];
            int tnnz = B.tile_nnz[t + 1] - base;
            // Walk CSR rows in order so k values are naturally sorted within each column
            for (int r = 0; r < TILE_SIZE; r++) {
                int rS = B.tile_csr_Ptr[t * TILE_SIZE + r];
                int rE = (r == TILE_SIZE - 1) ? tnnz : (int)B.tile_csr_Ptr[t * TILE_SIZE + r + 1];
                for (int e = rS; e < rE; e++) {
                    int lc = B.tile_csr_Col[base + e];
                    int dst = base + B.tile_csc_Ptr[t * TILE_SIZE + lc] + colOff[t * TILE_SIZE + lc];
                    B.tile_csc_Row[dst] = (unsigned char)r;
                    B.tile_csc_Val[dst] = B.tile_csr_Val[base + e];
                    colOff[t * TILE_SIZE + lc]++;
                }
            }
        }
    }
}

// Build within-tile CSC for A tiles (CSR tile order, CSC within each tile)
// Column k of A-tile gives all rows i that have an entry at column k.
inline void build_a_tile_csc(TileMatrix& A) {
    int nt = A.numtile;
    int totalNnz = A.tile_nnz[nt];
    A.tile_a_csc_Ptr = (unsigned char*)calloc(nt * TILE_SIZE, sizeof(unsigned char));
    A.tile_a_csc_Row = (unsigned char*)calloc(totalNnz, sizeof(unsigned char));
    A.tile_a_csc_Val = (TVAL*)calloc(totalNnz, sizeof(TVAL));
    // Count nnz per local column k within each tile
    for (int t = 0; t < nt; t++) {
        int base = A.tile_nnz[t];
        int tnnz = A.tile_nnz[t + 1] - base;
        unsigned char colCnt[TILE_SIZE] = {};
        for (int e = 0; e < tnnz; e++)
            colCnt[A.tile_csr_Col[base + e] & 0xf]++;
        unsigned char s = 0;
        for (int k = 0; k < TILE_SIZE; k++) {
            A.tile_a_csc_Ptr[t * TILE_SIZE + k] = s;
            s += colCnt[k];
        }
    }
    // Fill CSC entries: walk CSR rows in order so row indices are sorted per column
    {
        std::vector<int> colOff(nt * TILE_SIZE, 0);
        for (int t = 0; t < nt; t++) {
            int base = A.tile_nnz[t];
            int tnnz = A.tile_nnz[t + 1] - base;
            for (int r = 0; r < TILE_SIZE; r++) {
                int rS = A.tile_csr_Ptr[t * TILE_SIZE + r];
                int rE = (r == TILE_SIZE - 1) ? tnnz : (int)A.tile_csr_Ptr[t * TILE_SIZE + r + 1];
                for (int e = rS; e < rE; e++) {
                    int lc = A.tile_csr_Col[base + e] & 0xf;
                    int dst = base + A.tile_a_csc_Ptr[t * TILE_SIZE + lc] + colOff[t * TILE_SIZE + lc];
                    A.tile_a_csc_Row[dst] = (unsigned char)r;
                    A.tile_a_csc_Val[dst] = A.tile_csr_Val[base + e];
                    colOff[t * TILE_SIZE + lc]++;
                }
            }
        }
    }
}
inline void build_bitmasks(const TileMatrix& A, const TileMatrix& B,
                          unsigned int*& bmA, unsigned int*& bmB, int& bm_len) {
    bm_len = (A.tilen + 31) / 32;
    long long lenA = (long long)A.tilem * bm_len;
    long long lenB = (long long)B.tilen * bm_len;
    bmA = (unsigned int*)calloc(lenA, sizeof(unsigned int));
    bmB = (unsigned int*)calloc(lenB, sizeof(unsigned int));
    for (int i = 0; i < A.tilem; i++)
        for (int j = A.tile_ptr[i]; j < A.tile_ptr[i + 1]; j++) {
            int idx = A.tile_columnidx[j];
            bmA[(long long)i * bm_len + idx / 32] |= 1u << (31 - idx % 32);
        }
    for (int i = 0; i < B.tilen; i++)
        for (int j = B.csc_tile_ptr[i]; j < B.csc_tile_ptr[i + 1]; j++) {
            int idx = B.csc_tile_rowidx[j];
            bmB[(long long)i * bm_len + idx / 32] |= 1u << (31 - idx % 32);
        }
}

inline void free_tile_matrix(TileMatrix& T) {
    free(T.tile_ptr); free(T.tile_columnidx); free(T.tile_nnz);
    free(T.tile_csr_Val); free(T.tile_csr_Col); free(T.tile_csr_Ptr);
    free(T.mask);
    if (T.csc_tile_ptr) free(T.csc_tile_ptr);
    if (T.csc_tile_rowidx) free(T.csc_tile_rowidx);
    if (T.tile_csc_Ptr) free(T.tile_csc_Ptr);
    if (T.tile_csc_Row) free(T.tile_csc_Row);
    if (T.tile_csc_Val) free(T.tile_csc_Val);
    if (T.tile_a_csc_Ptr) free(T.tile_a_csc_Ptr);
    if (T.tile_a_csc_Row) free(T.tile_a_csc_Row);
    if (T.tile_a_csc_Val) free(T.tile_a_csc_Val);
}
