#ifndef GPU_ERR_SINGLETON
#define GPU_ERR_SINGLETON

namespace gpu_error {

template <typename singleton, typename T>
__global__ void write_singleton_kernel(T *device_version) {
  singleton::instance() = device_version[0];
}

template <typename singleton, typename T>
__global__ void read_singleton_kernel(T *device_version) {
  device_version[0] = singleton::instance();
}

template <typename T>
struct gpu_singleton {
  using internal_type = T;

  using my_type = gpu_singleton<T>;

  __device__ static T &instance() {
    static T s;
    return s;
  }  // instance

  static __host__ T read_instance() {
    T *device_version;

    cudaMallocManaged((void **)&device_version, sizeof(T));

    read_singleton_kernel<my_type, T><<<1, 1>>>(device_version);

    cudaDeviceSynchronize();

    T output = device_version[0];

    cudaFree(device_version);

    return output;
  }

  static __host__ void write_instance(T write) {
    T *device_version;

    cudaMallocManaged((void **)&device_version, sizeof(T));

    device_version[0] = write;

    write_singleton_kernel<my_type, T><<<1, 1>>>(device_version);

    cudaDeviceSynchronize();

    cudaFree(device_version);
  }

  gpu_singleton(const gpu_singleton &) = delete;
  gpu_singleton &operator=(const gpu_singleton &) = delete;

 private:
  gpu_singleton() {}
  ~gpu_singleton() {}

};  // struct singleton

}  // namespace gpu_error

#endif  // End of singleton