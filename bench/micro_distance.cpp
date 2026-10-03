// Micro-benchmark: hand-written SIMD distance kernels vs. scalar loops.
// Prints ns per distance computation for common embedding sizes.

#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "strata/distance.h"

using namespace strata;

namespace {

volatile float g_sink;

double ns_per_call(DistanceFn fn, const std::vector<float>& data, size_t dim, size_t n_vectors, size_t calls) {
  float acc = 0.f;
  const float* q = data.data();
  // warm-up
  for (size_t i = 0; i < n_vectors; ++i) acc += fn(q, data.data() + i * dim, dim);
  const auto t0 = std::chrono::steady_clock::now();
  for (size_t c = 0; c < calls; ++c) acc += fn(q, data.data() + (c % n_vectors) * dim, dim);
  const auto t1 = std::chrono::steady_clock::now();
  g_sink = acc;
  return std::chrono::duration<double, std::nano>(t1 - t0).count() / static_cast<double>(calls);
}

}  // namespace

int main() {
  std::printf("SIMD backend: %s\n", kernels::simd_backend());
  std::printf("%6s  %-6s %12s %12s %9s\n", "dim", "kernel", "scalar ns", "simd ns", "speedup");
  std::mt19937 rng(1);
  std::uniform_real_distribution<float> u(-1.f, 1.f);
  for (size_t dim : {128, 384, 768, 1536}) {
    const size_t n_vectors = 4096;  // stays in L2 cache: measures compute, not memory
    std::vector<float> data(n_vectors * dim);
    for (auto& x : data) x = u(rng);
    const size_t calls = 20'000'000 / (dim / 64);
    for (Metric m : {Metric::L2, Metric::InnerProduct}) {
      const double scalar = ns_per_call(distance_function(m, false), data, dim, n_vectors, calls);
      const double simd = ns_per_call(distance_function(m, true), data, dim, n_vectors, calls);
      std::printf("%6zu  %-6s %12.2f %12.2f %8.1fx\n", dim, m == Metric::L2 ? "l2" : "dot", scalar, simd, scalar / simd);
    }
  }
  return 0;
}
