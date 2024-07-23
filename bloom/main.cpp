#include "bloom_filter.hpp"
#include <iostream>

int main() {
  auto future = TimerAsync(1min, []() { std::cout << "Timer finished\n"; });

  while (true) {
    std::cout << "Processing\n";
    std::this_thread::sleep_for(1s);
    auto status = future.wait_for(1ms);
    if (status == std::future_status::ready)
      break;
  }
  std::cout << "Finished\n";
}
