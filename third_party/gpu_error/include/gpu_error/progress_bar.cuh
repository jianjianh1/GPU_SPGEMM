#ifndef PROGRESS
#define PROGRESS

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime_api.h>

#include <cmath>

#include "assert.h"
#include "stdio.h"

namespace gpu_error {

struct progress_bar {
  std::string task_name;
  uint64_t n_items;
  uint64_t current_item;
  double update_interval;

  uint64_t next_interval;
  uint64_t interval_amount;

  int barwidth;

  __host__ progress_bar(std::string ext_task_name, uint64_t ext_n_items,
                        double ext_update_interval, int ext_barwidth = 50) {
    task_name = ext_task_name;
    n_items = ext_n_items;
    current_item = 0;
    update_interval = ext_update_interval;
    barwidth = ext_barwidth;

    next_interval = 0;
    interval_amount = n_items * update_interval;
    if (interval_amount == 0) interval_amount = 1;
  }

  __host__ void increment(uint64_t amount = 1) {
    current_item += amount;

    if (current_item >= next_interval) {
      next_interval += interval_amount;

      while (current_item >= next_interval) {
        next_interval += interval_amount;
      }

      show_progress();
    }
  }

  __host__ void show_progress() {
    std::cout << " " << task_name << ": [";
    int progress = barwidth * current_item / n_items;

    for (int j = 0; j < barwidth; j++) {
      if (j < progress)
        std::cout << "=";
      else if (j == progress)
        std::cout << ">";
      else
        std::cout << " ";
    }

    double percent = 100.0 * current_item / n_items;
    std::cout << "] " << current_item << "/" << n_items << " " << percent
              << "%\r";
    std::cout.flush();
  }

  ~progress_bar() {
    show_progress();
    std::cout.flush();
    std::cout << "\n\n";
    std::cout.flush();
  }
};

}  // namespace gpu_error
#endif  // GPU_BLOCK_