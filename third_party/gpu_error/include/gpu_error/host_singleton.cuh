#ifndef GPU_ERR_HOST_SINGLETON
#define GPU_ERR_HOST_SINGLETON

namespace gpu_error {

template <typename T, uint counter>
struct host_singleton {
  using internal_type = T;

  using my_type = host_singleton<T, counter>;

  __host__ static T &instance() {
    static T s;
    return s;
  }  // instance

  host_singleton(const host_singleton &) = delete;
  host_singleton &operator=(const host_singleton &) = delete;

 private:
  host_singleton() {}
  ~host_singleton() {}

};  // struct singleton

}  // namespace gpu_error

#endif  // End of singleton