#ifndef GLOBAL_ALLOC_SINGLETON
#define GLOBAL_ALLOC_SINGLETON

#include <gallatin/allocators/gallatin.cuh>
#include <gpu_error/gpu_singleton.cuh>

namespace gpu_error {

/*__host__ void init_global_allocator_host(uint64_t num_bytes, uint64_t seed,
bool print_info=true, bool running_calloc=false){

  global_allocator_type * local_copy =
global_allocator_type::generate_on_device_host(num_bytes, seed, print_info,
running_calloc);

  cudaMemcpyToSymbol(global_host_gallatin, &local_copy,
sizeof(global_allocator_type *));

  cudaDeviceSynchronize();

}*/

using gallatin_type =
    gallatin::allocators::Gallatin<16ULL * 1024 * 1024, 16ULL, 4096ULL>;
using host_mem_allocator_type = gpu_singleton<gallatin_type *>;

static __host__ void init_log_allocator(uint64_t n_bytes) {
  gallatin_type *allocator = gallatin_type::generate_on_device_host(
      n_bytes, 15321666607839316903ULL, false, false);

  host_mem_allocator_type::write_instance(allocator);
}

static __host__ void free_log_allocator() {
  gallatin_type *allocator = host_mem_allocator_type::read_instance();

  host_mem_allocator_type::write_instance(nullptr);

  gallatin_type::free_on_device(allocator);
}

static __device__ void *hmalloc(uint64_t n_bytes) {
  return host_mem_allocator_type::instance()->malloc(n_bytes);
}

static __device__ void hfree(void *allocation) {
  return host_mem_allocator_type::instance()->free(allocation);
}

}  // namespace gpu_error

#endif  // End of singleton