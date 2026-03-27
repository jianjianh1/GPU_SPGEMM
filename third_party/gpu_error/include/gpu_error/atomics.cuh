#ifndef GPU_ATOMICS
#define GPU_ATOMICS

#include <cuda.h>
#include <cuda_runtime_api.h>

namespace gpu_error {

// Generic typed atomicCAS for 16, 32, and 64-bit types
template<typename T>
__device__ __forceinline__ T typed_atomic_cas(T* addr, T compare, T val) {
    static_assert(sizeof(T) == 2 || sizeof(T) == 4 || sizeof(T) == 8,
                  "typed_atomic_cas: only 2, 4, and 8 byte types supported");
    if constexpr (sizeof(T) == 2) {
        unsigned short result = atomicCAS(
            reinterpret_cast<unsigned short*>(addr),
            __builtin_bit_cast(unsigned short, compare),
            __builtin_bit_cast(unsigned short, val));
        return __builtin_bit_cast(T, result);
    } else if constexpr (sizeof(T) == 4) {
        unsigned int result = atomicCAS(
            reinterpret_cast<unsigned int*>(addr),
            __builtin_bit_cast(unsigned int, compare),
            __builtin_bit_cast(unsigned int, val));
        return __builtin_bit_cast(T, result);
    } else if constexpr (sizeof(T) == 8) {
        unsigned long long result = atomicCAS(
            reinterpret_cast<unsigned long long*>(addr),
            __builtin_bit_cast(unsigned long long, compare),
            __builtin_bit_cast(unsigned long long, val));
        return __builtin_bit_cast(T, result);
    }
}

// Generic typed atomicAdd for 32 and 64-bit types
template<typename T>
__device__ __forceinline__ T typed_atomic_add(T* addr, T val) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "typed_atomic_add: only 4 and 8 byte types supported");
    if constexpr (sizeof(T) == 4) {
        unsigned int result = atomicAdd(
            reinterpret_cast<unsigned int*>(addr),
            __builtin_bit_cast(unsigned int, val));
        return __builtin_bit_cast(T, result);
    } else if constexpr (sizeof(T) == 8) {
        unsigned long long result = atomicAdd(
            reinterpret_cast<unsigned long long*>(addr),
            __builtin_bit_cast(unsigned long long, val));
        return __builtin_bit_cast(T, result);
    }
}

// Generic typed atomicMin for 32 and 64-bit types
template<typename T>
__device__ __forceinline__ T typed_atomic_min(T* addr, T val) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "typed_atomic_min: only 4 and 8 byte types supported");
    if constexpr (sizeof(T) == 4) {
        unsigned int result = atomicMin(
            reinterpret_cast<unsigned int*>(addr),
            __builtin_bit_cast(unsigned int, val));
        return __builtin_bit_cast(T, result);
    } else if constexpr (sizeof(T) == 8) {
        unsigned long long result = atomicMin(
            reinterpret_cast<unsigned long long*>(addr),
            __builtin_bit_cast(unsigned long long, val));
        return __builtin_bit_cast(T, result);
    }
}

// Generic typed atomicMax for 32 and 64-bit types
template<typename T>
__device__ __forceinline__ T typed_atomic_max(T* addr, T val) {
    static_assert(sizeof(T) == 4 || sizeof(T) == 8,
                  "typed_atomic_max: only 4 and 8 byte types supported");
    if constexpr (sizeof(T) == 4) {
        unsigned int result = atomicMax(
            reinterpret_cast<unsigned int*>(addr),
            __builtin_bit_cast(unsigned int, val));
        return __builtin_bit_cast(T, result);
    } else if constexpr (sizeof(T) == 8) {
        unsigned long long result = atomicMax(
            reinterpret_cast<unsigned long long*>(addr),
            __builtin_bit_cast(unsigned long long, val));
        return __builtin_bit_cast(T, result);
    }
}

} // namespace gpu_error

#endif // GPU_ATOMICS
