#ifndef GPU_ERR_HELPERS
#define GPU_ERR_HELPERS

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cmath>

#include "assert.h"
#include "stdio.h"

namespace gpu_error {

  //get the smallest inclusive power of 2
  __device__ __forceinline__ uint32_t next_pow2_bit(uint32_t x) {
      return x <= 1 ? 0 : 32 - __clz(x - 1);
  }


}  // namespace gpu_error
#endif  // GPU_BLOCK_