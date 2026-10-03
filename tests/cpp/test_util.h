#pragma once

#include <unistd.h>

#include <filesystem>
#include <random>
#include <set>
#include <string>
#include <vector>

#include "strata/flat.h"
#include "strata/hnsw.h"

namespace testutil {

inline std::vector<float> random_vectors(size_t n, size_t dim, uint32_t seed) {
  std::mt19937 rng(seed);
  std::normal_distribution<float> dist(0.f, 1.f);
  std::vector<float> v(n * dim);
  for (auto& x : v) x = dist(rng);
  return v;
}

inline std::vector<strata::label_t> iota_labels(size_t n, strata::label_t start = 0) {
  std::vector<strata::label_t> l(n);
  for (size_t i = 0; i < n; ++i) l[i] = start + i;
  return l;
}

// Mean recall@k of the HNSW index against exact search.
inline double recall_at_k(const strata::HNSWIndex& index, const strata::FlatIndex& exact, const std::vector<float>& queries,
                          size_t nq, size_t k, size_t ef, const strata::LabelFilter* filter = nullptr) {
  const size_t dim = index.dim();
  double hits = 0, total = 0;
  for (size_t i = 0; i < nq; ++i) {
    const auto truth = exact.search(queries.data() + i * dim, k, filter);
    const auto got = index.search(queries.data() + i * dim, k, ef, filter);
    std::set<strata::label_t> t;
    for (const auto& r : truth) t.insert(r.label);
    for (const auto& r : got) hits += t.count(r.label);
    total += static_cast<double>(truth.size());
  }
  return total > 0 ? hits / total : 1.0;
}

struct TempDir {
  std::filesystem::path path;
  TempDir() {
    static int counter = 0;
    path = std::filesystem::temp_directory_path() /
           ("strata_test_" + std::to_string(::getpid()) + "_" + std::to_string(counter++));
    std::filesystem::remove_all(path);
    std::filesystem::create_directories(path);
  }
  ~TempDir() { std::filesystem::remove_all(path); }
  std::string str(const std::string& name = "") const { return (path / name).string(); }
};

}  // namespace testutil
