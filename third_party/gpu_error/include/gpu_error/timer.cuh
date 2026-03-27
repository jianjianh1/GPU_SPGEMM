#ifndef GPU_ERROR_TIMER
#define GPU_ERROR_TIMER
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

// Timer is a helper tool for timing the runtime of a kernel, provides an upper bound on time taken.
// Times are recorded with cuda events, and a helper template abstraction is used to allow for
// automatic timer initialization and registration. To use, pick an ID you havent used before
// and call

// inlcudes
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <gpu_error/host_singleton.cuh>

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <map>         // std::map (ordered registry)
#include <string>      // std::string
#include <mutex>       // std::mutex, std::lock_guard

using namespace std::chrono;

namespace gpu_error {


  struct timer_interface {
    virtual void print() const = 0;
    virtual uint get_id() const = 0;
    virtual void delete_events() const = 0;
    virtual ~timer_interface() = default;
  };

  struct cuda_timer : public timer_interface {

    cudaEvent_t start_event;
    cudaEvent_t stop_event;
    float total_time_ms = 0.0f;  // Accumulated time
    int calls = 0;               // Number of timing intervals

    int init = 0;;
    std::string event_name;

    cuda_timer() {
        cudaEventCreate(&start_event);
        cudaEventCreate(&stop_event);
    }

    void delete_events() const override{
        cudaEventDestroy(start_event);
        cudaEventDestroy(stop_event);
    }

    ~cuda_timer() {
    }

    // Begin timing interval
    void start(cudaStream_t stream = 0) {
        cudaEventRecord(start_event, stream);
    }

    // End timing interval and accumulate result
    void stop(cudaStream_t stream = 0) {
        cudaEventRecord(stop_event, stream);
        cudaEventSynchronize(stop_event); // ensure event completion

        float elapsed = 0.0f;
        cudaEventElapsedTime(&elapsed, start_event, stop_event);
        total_time_ms += elapsed;
        ++calls;
    }

    // Returns total time accumulated
    float total_time() const {
        return total_time_ms;
    }

    // Returns average time per timing interval
    float average_time() const {
        return calls > 0 ? total_time_ms / calls : 0.0f;
    }

    // Reset timer state
    void reset() {
        total_time_ms = 0.0f;
        calls = 0;
    }

    uint id = 0;

    void set_id(uint new_id) {
        id = new_id;
    }

    uint get_id() const override {
        return id;
    }

    void set_name(std::string name){

      if (init == 0){
        event_name = name;
        init = 1;
      }
      
    }


    void print() const override{
      std::cout << "\033[0;32mTimed event " << id <<": " << event_name << "\033[0m" << std::endl;
      std::cout << "-- total time: " << total_time() << " ms" << std::endl;
      std::cout << "-- average time: " << average_time() << " ms" << " over " << calls << " calls" << std::endl;
    }

};


//registry to hold timers in use
class timer_registry {
public:
    static timer_registry& instance() {
        static timer_registry inst;
        return inst;
    }

    void register_timer(timer_interface* timer) {
        std::lock_guard<std::mutex> lock(mutex_);
        timers[timer->get_id()] = timer;
    }

    void print_all() {
        for (const auto& [id, timer] : timers) {
            if (timer) timer->print();
        }
    }

    void clear(){
      std::lock_guard<std::mutex> lock(mutex_);

      for (auto& [id, timer] : timers) {
        timer->delete_events();
      }
      timers.clear();
    }

private:
    std::map<uint, timer_interface*> timers;
    std::mutex mutex_;
};

template <uint id>
struct static_timer {
    using singleton_type = gpu_error::host_singleton<cuda_timer, id>;

    static void start(std::string name, cudaStream_t stream = 0) {
        auto& timer = singleton_type::instance();
        static bool initialized = false;
        if (!initialized) {
            timer.set_id(id);
            timer.set_name(name);
            gpu_error::timer_registry::instance().register_timer(&timer);
            initialized = true;
        }
        timer.start(stream);
    }

    static void stop(cudaStream_t stream = 0) {
        singleton_type::instance().stop(stream);
    }

    static void print() {
        singleton_type::instance().print();
    }
};


static void print_all_timers(){
  gpu_error::timer_registry::instance().print_all();
  gpu_error::timer_registry::instance().clear();
}



}



#endif  // End of guard
