#pragma once

// Small concurrency helpers: a 1-byte spinlock and a parallel_for.

#include <algorithm>
#include <atomic>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

namespace strata {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(_M_X64)
  _mm_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

// One per graph node. std::mutex is 64 bytes on macOS (40 on glibc), so a 1M-node index would
// spend 64 MB on locks; this costs 1 MB. Critical sections are tiny (copying <= 2*M ints), so
// spinning is cheaper than parking the thread.
class SpinLock {
 public:
  void lock() noexcept {
    unsigned spins = 0;
    for (;;) {
      if (!flag_.exchange(true, std::memory_order_acquire)) return;
      while (flag_.load(std::memory_order_relaxed)) {
        if (++spins < 64) {
          cpu_relax();
        } else {
          std::this_thread::yield();
        }
      }
    }
  }
  bool try_lock() noexcept {
    return !flag_.load(std::memory_order_relaxed) && !flag_.exchange(true, std::memory_order_acquire);
  }
  void unlock() noexcept { flag_.store(false, std::memory_order_release); }

 private:
  std::atomic<bool> flag_{false};
};
static_assert(sizeof(SpinLock) == 1, "SpinLock should be a single byte");

inline int resolve_threads(int requested) {
  if (requested > 0) return requested;
  unsigned hc = std::thread::hardware_concurrency();
  return hc ? static_cast<int>(hc) : 1;
}

// Runs fn(i) for i in [0, n) on up to num_threads threads (<= 0 means all cores).
// Work is handed out one index at a time from an atomic counter, which balances
// uneven per-item cost (HNSW inserts vary a lot). The first exception is rethrown.
template <class Fn>
void parallel_for(size_t n, int num_threads, Fn&& fn) {
  if (n == 0) return;
  const size_t threads = std::min<size_t>(static_cast<size_t>(resolve_threads(num_threads)), n);
  if (threads <= 1) {
    for (size_t i = 0; i < n; ++i) fn(i);
    return;
  }
  std::atomic<size_t> next{0};
  std::exception_ptr error;
  std::mutex error_mu;
  auto worker = [&] {
    for (;;) {
      const size_t i = next.fetch_add(1, std::memory_order_relaxed);
      if (i >= n) return;
      try {
        fn(i);
      } catch (...) {
        std::lock_guard<std::mutex> g(error_mu);
        if (!error) error = std::current_exception();
        next.store(n);
        return;
      }
    }
  };
  std::vector<std::thread> pool;
  pool.reserve(threads - 1);
  for (size_t t = 0; t + 1 < threads; ++t) pool.emplace_back(worker);
  worker();
  for (auto& th : pool) th.join();
  if (error) std::rethrow_exception(error);
}

}  // namespace strata
