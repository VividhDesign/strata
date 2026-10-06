// Query latency while a large batch is being inserted.
//
// Builds an index of `base` vectors, then inserts `extra` more in one add() call on one thread
// while another thread issues single queries back to back. Reports how many queries were
// answered during the insert and their latency percentiles.
//
//   ./build/bench_ingest [base=100000] [extra=100000] [dim=96] [insert_threads=3]

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <thread>
#include <vector>

#include "strata/hnsw.h"

using namespace strata;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const size_t base = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 100000;
  const size_t extra = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 100000;
  const size_t dim = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 96;
  const int insert_threads = argc > 4 ? std::atoi(argv[4]) : 3;

  // Clustered data (64 Gaussian blobs), closer to real embeddings than i.i.d. noise.
  std::mt19937 rng(7);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> centers(64 * dim);
  for (auto& c : centers) c = 3.f * g(rng);
  std::vector<float> data((base + extra) * dim);
  for (size_t i = 0; i < base + extra; ++i) {
    const float* c = centers.data() + (i % 64) * dim;
    for (size_t d = 0; d < dim; ++d) data[i * dim + d] = c[d] + g(rng);
  }
  std::vector<label_t> labels(base + extra);
  for (size_t i = 0; i < labels.size(); ++i) labels[i] = i;

  HNSWIndex index(dim, Metric::L2, {}, base + extra);
  index.add(data.data(), labels.data(), base, 0);

  std::atomic<bool> inserting{true};
  std::vector<double> lat_us;
  std::thread reader([&] {
    size_t i = 0;
    while (inserting.load(std::memory_order_relaxed)) {
      const float* q = data.data() + ((i++ * 7919) % base) * dim;
      const auto t0 = Clock::now();
      const auto res = index.search(q, 10, 64);
      lat_us.push_back(std::chrono::duration<double, std::micro>(Clock::now() - t0).count());
      if (res.empty()) std::abort();
    }
  });
  const auto t0 = Clock::now();
  index.add(data.data() + base * dim, labels.data() + base, extra, insert_threads);
  const double insert_s = std::chrono::duration<double>(Clock::now() - t0).count();
  inserting.store(false);
  reader.join();

  std::sort(lat_us.begin(), lat_us.end());
  auto pct = [&](double p) { return lat_us.empty() ? 0.0 : lat_us[std::min(lat_us.size() - 1, size_t(p * lat_us.size()))]; };
  std::printf("{\"base\": %zu, \"extra\": %zu, \"dim\": %zu, \"insert_threads\": %d, \"insert_seconds\": %.3f, "
              "\"queries_during_insert\": %zu, \"p50_us\": %.1f, \"p99_us\": %.1f, \"max_us\": %.1f}\n",
              base, extra, dim, insert_threads, insert_s, lat_us.size(), pct(0.5), pct(0.99),
              lat_us.empty() ? 0.0 : lat_us.back());
  return 0;
}
