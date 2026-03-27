#ifndef GPU_ERROR_TID
#define GPU_ERROR_TID
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
// sell copies of the Software, and to permit persons to whom the Software is
// furnished to do so,
// subject to the following conditions:

// The above copyright notice and this permission notice shall be included in
// all copies or substantial
// portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR
// IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

// Helpers to select individual threads and teams of threads


#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

namespace gpu_error {

  __device__ inline uint64_t get_tid() {
    return ((uint64_t)threadIdx.x) + ((uint64_t)blockIdx.x) * ((uint64_t) blockDim.x);
  }

  template <uint team_size>
  __device__ inline uint64_t get_team_tid(cg::thread_block_tile<team_size> team) {

    uint64_t block_id = blockIdx.x;

    uint64_t team_id = team.meta_group_rank();

    uint64_t team_meta_size = team.meta_group_size();

    return team_id + team_meta_size*block_id;

  }

}



#endif  // End of guard
