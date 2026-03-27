#ifndef GPU_LOADS
#define GPU_LOADS

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <random>

#include <filesystem>
#include <iostream>
#include <fstream>

#include <gpu_error/progress_bar.cuh>

#include "assert.h"
#include "stdio.h"


namespace gpu_error {

namespace load {


template <typename T>
struct chunk {
    static constexpr uint size = 16 / sizeof(T);
    T data[size];
    
    __device__ T& operator[](uint i) { return data[i]; }
    __device__ const T& operator[](uint i) const { return data[i]; }
};

template <typename T>
__device__ chunk<T> load_16_bytes(const T* src) {
    chunk<T> buffer;
    
    const uint4* src_u4 = reinterpret_cast<const uint4*>(src);
    uint4* dst_u4 = reinterpret_cast<uint4*>(buffer.data);
    
    asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];"
                : "=r"(dst_u4->x), "=r"(dst_u4->y), "=r"(dst_u4->z), "=r"(dst_u4->w)
                : "l"(src_u4));
    
    return buffer;
}

//split a load among multiple threads in a tile
// This does not check for safety of misaligned loads - you need to do that.
// this will terminate the loop early if there are less bytes to load than threads available to load them.
template <typename T, typename LoadT, uint tile_size>
class tile_load_iterator {

    static constexpr uint objs_per_load = 16 / sizeof(T);
    static constexpr uint objs_per_tiled_load = objs_per_load*tile_size;

    const T* src;
    uint n_iters;
    const uint total_size;
    uint thread_offset;
    
public:
    __device__ tile_load_iterator(cooperative_groups::thread_block_tile<tile_size>& tile, const T* src_ptr, uint total_bytes=sizeof(T)) 
        : src(src_ptr),n_iters(0), total_size(total_bytes), thread_offset(tile.thread_rank() * 16 / sizeof(LoadT)) {}
    
    __device__ auto load_next() {
        const LoadT* load_src = reinterpret_cast<const LoadT*>(src);
        uint offset = n_iters * objs_per_tiled_load + thread_offset*objs_per_load;
        n_iters++;
        return load_16_bytes(load_src + offset);
    }
    
    __device__ bool done() const {
        return n_iters * objs_per_tiled_load + thread_offset * objs_per_load >= total_size;
    }
};

}

// Custom load/store operators for different data sizes
template<typename T>
__device__ __forceinline__ T load_once(const T* ptr) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, 
                  "Unsupported size: only 1,2,4,8,16 bytes supported");
    T val;
    if constexpr (sizeof(T) == 1) {
        asm volatile("ld.global.cv.u8 %0, [%1];" : "=r"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("ld.global.cv.u16 %0, [%1];" : "=h"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("ld.global.cv.u32 %0, [%1];" : "=r"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("ld.global.cv.u64 %0, [%1];" : "=l"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 16) {
        uint4* val_u4 = reinterpret_cast<uint4*>(&val);
        const uint4* ptr_u4 = reinterpret_cast<const uint4*>(ptr);
        asm volatile("ld.global.cv.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(val_u4->x), "=r"(val_u4->y), "=r"(val_u4->z), "=r"(val_u4->w)
                    : "l"(ptr_u4));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T load_cached(const T* ptr) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, 
                  "Unsupported size: only 1,2,4,8,16 bytes supported");
    T val;
    if constexpr (sizeof(T) == 1) {
        asm volatile("ld.global.ca.u8 %0, [%1];" : "=r"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("ld.global.ca.u16 %0, [%1];" : "=h"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("ld.global.ca.u64 %0, [%1];" : "=l"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 16) {
        uint4* val_u4 = reinterpret_cast<uint4*>(&val);
        const uint4* ptr_u4 = reinterpret_cast<const uint4*>(ptr);
        asm volatile("ld.global.ca.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(val_u4->x), "=r"(val_u4->y), "=r"(val_u4->z), "=r"(val_u4->w)
                    : "l"(ptr_u4));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T load_acquire(const T* ptr) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, 
                  "Unsupported size: only 1,2,4,8,16 bytes supported");
    T val;
    if constexpr (sizeof(T) == 1) {
        asm volatile("ld.global.acquire.gpu.u8 %0, [%1];" : "=r"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("ld.global.acquire.gpu.u16 %0, [%1];" : "=h"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("ld.global.acquire.gpu.u32 %0, [%1];" : "=r"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("ld.global.acquire.gpu.u64 %0, [%1];" : "=l"(val) : "l"(ptr));
    } else if constexpr (sizeof(T) == 16) {
        uint4* val_u4 = reinterpret_cast<uint4*>(&val);
        const uint4* ptr_u4 = reinterpret_cast<const uint4*>(ptr);
        asm volatile("ld.global.acquire.gpu.v4.u32 {%0,%1,%2,%3}, [%4];"
                    : "=r"(val_u4->x), "=r"(val_u4->y), "=r"(val_u4->z), "=r"(val_u4->w)
                    : "l"(ptr_u4));
    }
    return val;
}

template<typename T>
__device__ __forceinline__ void store_release(T* ptr, T val) {
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 16, 
                  "Unsupported size: only 1,2,4,8,16 bytes supported");
    if constexpr (sizeof(T) == 1) {
        asm volatile("st.global.release.gpu.u8 [%0], %1;" :: "l"(ptr), "r"(val));
    } else if constexpr (sizeof(T) == 2) {
        asm volatile("st.global.release.gpu.u16 [%0], %1;" :: "l"(ptr), "h"(val));
    } else if constexpr (sizeof(T) == 4) {
        asm volatile("st.global.release.gpu.u32 [%0], %1;" :: "l"(ptr), "r"(val));
    } else if constexpr (sizeof(T) == 8) {
        asm volatile("st.global.release.gpu.u64 [%0], %1;" :: "l"(ptr), "l"(val));
    } else if constexpr (sizeof(T) == 16) {
        const uint4* val_u4 = reinterpret_cast<const uint4*>(&val);
        uint4* ptr_u4 = reinterpret_cast<uint4*>(ptr);
        asm volatile("st.global.release.gpu.v4.u32 [%0], {%1,%2,%3,%4};"
                    :: "l"(ptr_u4), "r"(val_u4->x), "r"(val_u4->y), "r"(val_u4->z), "r"(val_u4->w));
    }
}

}

#endif  // GPU_LOADS