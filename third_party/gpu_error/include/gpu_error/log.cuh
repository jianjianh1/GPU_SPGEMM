#ifndef GPU_ERROR_LOG
#define GPU_ERROR_LOG

#include <cuda.h>
#include <cuda_runtime_api.h>

#include <fstream>
#include <iterator>

// alloc utils needed for easy host_device transfer
// and the global allocator
#include <gallatin/allocators/alloc_utils.cuh>
#include <gallatin/allocators/global_allocator.cuh>
#include <gpu_error/custring.cuh>
#include <gpu_error/fixed_vector.cuh>
#include <gpu_error/global_alloc_singleton.cuh>
#include <gpu_error/gpu_singleton.cuh>
#include <gpu_error/progress_bar.cuh>
#include <string>

// This is a logger for cuda! Uses a queue structure to record
//  entries with unbounded length, up to the maximum device memory

// entries maintain TID, logID, and a message generated via a cuda string
// implementation.

namespace gpu_error {

template <typename log_vector>
__global__ void free_log_strings(log_vector *logs, uint64_t nitems) {
  uint64_t tid = gallatin::utils::get_tid();

  if (tid >= nitems) return;

  auto log_entry = logs[0][tid];

  auto log_string = log_entry.message;

  log_string.release_string();
};

struct log_entry {
  uint64_t tid;

  custring message;

  __device__ log_entry(custring new_message) {
    tid = gallatin::utils::get_tid();
    message = new_message;
  }

  __device__ log_entry(uint64_t ext_tid, custring new_message) {
    tid = ext_tid;
    message = new_message;
  }

  log_entry() = default;

  // template<typename ... Args>
  // __device__ log_entry(Args...all_args){

  // 	tid = gallatin::utils::get_tid();

  // 	message = make_string<on_host, Args...>(all_args...);

  // }

  // assumes log host is on host but custring is maybe not.
  __host__ std::string export_log() {
#ifdef GPU_NDEBUG

    std::string my_string;

#else

    std::string my_string(message.data());

#endif

    // std::cout << "String is " << my_string << std::endl;
    return my_string;
  }
};

// define vector
using log_vector_type = fixed_vector<log_entry, 128ULL, 1000000000ULL>;

using log_singleton = gpu_singleton<log_vector_type *>;

using lw_log_singleton = gpu_singleton<uint64_t *>;

static __host__ void init_gpu_log(uint64_t n_bytes = 8589934592ULL,
                                  uint64_t n_lw_logs = 32) {
  // initialize allocator.

#ifndef GPU_NDEBUG

  init_log_allocator(n_bytes);

  log_vector_type *log = log_vector_type::generate_on_device();

  log_singleton::write_instance(log);

  uint64_t *lw_log;

  cudaMallocManaged((void **)&lw_log, sizeof(uint64_t) * (n_lw_logs + 1));

  lw_log[0] = n_lw_logs;
  for (uint64_t i = 0; i < n_lw_logs; i++) {
    lw_log[i + 1] = 0;
  }

  lw_log_singleton::write_instance(lw_log);
#endif
}

static __host__ void free_gpu_log() {
#ifndef GPU_NDEBUG

  log_vector_type *log = log_singleton::read_instance();

  log_singleton::write_instance(nullptr);

  log_vector_type::free_on_device(log);

  free_log_allocator();

  cudaFree(lw_log_singleton::read_instance());

  lw_log_singleton::write_instance(nullptr);

#endif
}

static __host__ std::vector<std::string> export_log() {
#ifdef GPU_NDEBUG

  std::vector<std::string> empty_logs;
  return empty_logs;

#else

  auto vector_log =
      log_vector_type::export_to_host(log_singleton::read_instance());

  std::vector<std::string> output_strings;

  // printf("Vector log has %lu items\n", vector_log.size());

  printf("Log exported, writing out...\n");

  gpu_error::progress_bar bar("Writing log", vector_log.size(), .01);

  for (uint64_t i = 0; i < vector_log.size(); i++) {
    auto log = vector_log[i];

    // std::cout << log.tid << " " << log.message.length << std::endl;

    // if (log.message.length < 10){
    // 	std::cout << "log " << i << " is busted\n";
    // }

    output_strings.push_back(vector_log[i].export_log());

    bar.increment();
  }

  return output_strings;

#endif
}

static __host__ void print_events() {
#if GPU_NDEBUG

  std::cout << "Events are not being tracked due to GPU_NDEBUG flag. "
            << std::endl;
  return;

#else

  std::cout << "Event counts: " << std::endl;
  std::cout << "==========================" << std::endl;

  uint64_t *event_logs = lw_log_singleton::read_instance();

  uint64_t n_events = event_logs[0];

  for (uint64_t i = 0; i < n_events; i++) {
    if (event_logs[i + 1] != 0) {
      std::cout << "[EVENT " << i << "]: occurred " << event_logs[i + 1]
                << " times" << std::endl;
    }
  }

  std::cout << "==========================" << std::endl;

#endif
}

static __host__ uint64_t count_events() {
#if GPU_NDEBUG

  std::cout << "Events are not being tracked due to GPU_NDEBUG flag. "
            << std::endl;
  return 0;

#else

  uint64_t *event_logs = lw_log_singleton::read_instance();

  uint64_t n_events = event_logs[0];

  uint64_t count = 0;

  for (uint64_t i = 0; i < n_events; i++) {
    count += event_logs[i + 1];
  }

  return count;

#endif
}

static __host__ void clear_events() {
#if GPU_NDEBUG

  return;

#else

  uint64_t *event_logs = lw_log_singleton::read_instance();

  uint64_t n_events = event_logs[0];

  for (uint64_t i = 0; i < n_events; i++) {
    event_logs[i + 1] = 0;
  }

#endif
}

template <typename... Args>
static __device__ void log(Args... all_args) {
  auto string_message = make_string<Args...>(all_args...);

  log_entry new_log_entry(string_message);

  log_singleton::instance()->insert(new_log_entry);
}

static __device__ void lw_log(uint64_t log_value) {
  atomicAdd(
      (unsigned long long int *)&lw_log_singleton::instance()[log_value + 1],
      (unsigned long long int)1);
}

template <typename... Args>
static __device__ void print_error(Args... all_args) {
#ifdef LOG_GPU_ERRORS

  log<Args...>(all_args...);

#else

  auto string_message = make_string<Args...>(all_args...);

  string_message.print_string_device();

  __threadfence();

  asm volatile("trap;");

#endif
}

template <typename... Args>
static __device__ void print_assertion(bool assertion, Args... all_args) {


  if (assertion) return;

  if (host_mem_allocator_type::instance() == nullptr){
    //If you trapped here, you failed to init the allocator.
    // call "init_gpu_log();"
    asm volatile("trap;");
  }



#ifdef LOG_GPU_ERRORS

  log<Args...>(all_args...);

#else

  auto string_message = make_string<Args...>(all_args...);

  string_message.print_string_device();

  __threadfence();

  asm volatile("trap;");

#endif
}

// //implementation uses fixed_vector now as it is embarassingly parallel and
// much easier to write struct error_log {

// 	using my_type = error_log;

// 	using log_type = log_entry;

// 	using custring_type = custring;

// 	//10 trillion log entries
// 	//if you try to hit this it will break.
// 	using vector_type = fixed_vector<log_type, 128ULL, 100000000000ULL,
// on_host>;

// 	vector_type * storage_vector;

// 	static __host__ my_type * generate_on_device(){

// 		my_type * host_version =
// gallatin::utils::get_host_version<my_type>();

// 		host_version->storage_vector = vector_type::get_device_vector();

// 		return gallatin::utils::move_to_device(host_version);

// 	}

// 	static __host__ void free_on_device(my_type * device_log){

// 		auto host_version = gallatin::utils::move_to_host(device_log);

// 		//non destructive copy so that the vector components can be
// freed 		auto host_vector =
// gallatin::utils::copy_to_host(host_version->storage_vector);

// 		uint64_t n_logs = host_vector->size;

// 		free_log_strings<vector_type><<<(n_logs-1)/256+1,
// 256>>>(host_version->storage_vector, n_logs);

// 		cudaDeviceSynchronize();

// 		vector_type::free_device_vector(host_version->storage_vector);

// 		cudaFreeHost(host_vector);

// 		cudaFreeHost(host_version);

// 	}

// 	template <typename ... Args>
// 	__device__ void add_log(Args...all_args){

// 		auto string_message = make_string<Args...>(all_args...);

// 		log_entry new_log_entry(string_message);

// 		storage_vector->insert(new_log_entry);

// 	}

// 	//dump log to a host vector for easy export.
// 	//general steps:
// 	static __host__ std::vector<std::string> export_log(my_type *
// device_version){

// 		my_type * host_version =
// gallatin::utils::move_to_host(device_version);

// 		auto vector_log =
// vector_type::export_to_host(host_version->storage_vector);

// 		std::vector<std::string> output_strings;

// 		//printf("Vector log has %lu items\n", vector_log.size());

// 		for (uint64_t i = 0; i < vector_log.size(); i++){

// 			auto log = vector_log[i];

// 			//std::cout << log.tid << " " << log.message.length <<
// std::endl;

// 			// if (log.message.length < 10){
// 			// 	std::cout << "log " << i << " is busted\n";
// 			// }

// 			output_strings.push_back(vector_log[i].export_log());

// 		}

// 		//printf("Ouput strings has %lu items", output_strings.size());

// 		return output_strings;

// 	}

// 	//generate output vector and write to host file.
// 	//for the moment this uses host buffer.
// 	__host__ void dump_to_file(std::string filename){

// 		auto log_strings = my_type::export_log(this);

// 		std::cout << "Writing " << log_strings.size() << " logs to file
// " << filename << std::endl;

// 		// for (uint64_t i = 0; i < log_strings.size(); i++){

// 		// 	if (log_strings[i].size() < 10){
// 		// 		printf("Small string %lu, length is %lu\n", i,
// log_strings[i].size());
// 		// 		std::cout << log_strings[i] << std::endl;
// 		// 	}

// 		// 	if (log_strings[i].find("\n") != std::string::npos) {
// 		// 		std::cout << "found in " << i << '\n';
// 		// 	}

// 		// }

// 		std::ofstream output_file(filename);
// 		std::ostream_iterator<std::string> output_iterator(output_file,
// "\n"); 		std::copy(log_strings.begin(), log_strings.end(), output_iterator);

// 	}

// };

}  // namespace gpu_error

#ifdef GPU_NDEBUG

#define count_event(...) ((void)0)

#define gpu_log(...) ((void)0)

#define gpu_error(...) ((void)0)

#define gpu_assert(...) ((void)0)

#else

#define count_event(x) gpu_error::lw_log(x)

#define gpu_log(...) gpu_error::log(__VA_ARGS__)

#define gpu_error(...)                                                      \
  gpu_error::print_error("\033[1;31mError in file ", __FILE__, " at line ", \
                         __LINE__, ": ", __VA_ARGS__, "\033[0m")

#define gpu_assert(BOOL_FLAG, ...)                                             \
  gpu_error::print_assertion(BOOL_FLAG, "\033[1;31mAssertion failed in file ", \
                             __FILE__, " at line ", __LINE__, ": ",            \
                             __VA_ARGS__, "\033[0m")

#endif

#endif  // end of log name guard