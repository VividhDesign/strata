// Filtered search: recall@10 and QPS against the selectivity of an allow-list filter.
//
//   ./build/bench_filter [n=200000] [dim=96] [queries=500]
//
// Labels are assigned to a random "category"; the filter allows one category, whose share of
// the data is the selectivity. Ground truth is exact search over the allowed vectors.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <set>
#include <vector>

#include "strata/flat.h"
#include "strata/hnsw.h"

using namespace strata;
using Clock = std::chrono::steady_clock;

int main(int argc, char** argv) {
  const size_t n = argc > 1 ? std::strtoull(argv[1], nullptr, 10) : 200000;
  const size_t dim = argc > 2 ? std::strtoull(argv[2], nullptr, 10) : 96;
  const size_t nq = argc > 3 ? std::strtoull(argv[3], nullptr, 10) : 500;

  std::mt19937 rng(11);
  std::normal_distribution<float> g(0.f, 1.f);
  std::vector<float> centers(64 * dim);
  for (auto& c : centers) c = 3.f * g(rng);
  auto sample = [&](float* out) {
    const float* c = centers.data() + (rng() % 64) * dim;
    for (size_t d = 0; d < dim; ++d) out[d] = c[d] + g(rng);
  };
  std::vector<float> data(n * dim), queries(nq * dim);
  for (size_t i = 0; i < n; ++i) sample(data.data() + i * dim);
  for (size_t i = 0; i < nq; ++i) sample(queries.data() + i * dim);
  std::vector<label_t> labels(n);
  for (size_t i = 0; i < n; ++i) labels[i] = i;

  HNSWIndex index(dim, Metric::L2, {}, n);
  index.add(data.data(), labels.data(), n, 0);

  std::uniform_real_distribution<double> u(0, 1);
  std::printf("mode,selectivity,ef,recall,qps,brute_force_qps\n");
  for (double sel : {0.5, 0.2, 0.1, 0.05, 0.02, 0.01}) {
    std::vector<label_t> allowed;
    for (size_t i = 0; i < n; ++i) {
      if (u(rng) < sel) allowed.push_back(i);
    }
    const LabelFilter filter(allowed.data(), allowed.size(), LabelFilter::Mode::Allow);
    // Exact answers: flat search over the allowed subset.
    FlatIndex exact(dim, Metric::L2);
    {
      std::vector<float> sub(allowed.size() * dim);
      for (size_t j = 0; j < allowed.size(); ++j)
        std::copy(data.begin() + allowed[j] * dim, data.begin() + (allowed[j] + 1) * dim, sub.begin() + j * dim);
      exact.add(sub.data(), allowed.data(), allowed.size());
    }
    std::vector<std::set<label_t>> truth(nq);
    const auto tb = Clock::now();
    for (size_t i = 0; i < nq; ++i)
      for (const auto& r : exact.search(queries.data() + i * dim, 10)) truth[i].insert(r.label);
    const double bf_qps = nq / std::chrono::duration<double>(Clock::now() - tb).count();
    for (const char* mode : {"walk", "planner"}) {
      // walk: always the filtered graph walk; planner: the default brute-force/walk choice
      index.set_flat_search_cutoff(mode[0] == 'w' ? 0 : 2048);
      for (size_t ef : {32, 64, 128, 256}) {
        size_t hits = 0;
        const auto t0 = Clock::now();
        for (size_t i = 0; i < nq; ++i) {
          for (const auto& r : index.search(queries.data() + i * dim, 10, ef, &filter)) hits += truth[i].count(r.label);
        }
        const double qps = nq / std::chrono::duration<double>(Clock::now() - t0).count();
        std::printf("%s,%.2f,%zu,%.4f,%.0f,%.0f\n", mode, sel, ef, hits / (10.0 * nq), qps, bf_qps);
      }
    }
  }
  return 0;
}
