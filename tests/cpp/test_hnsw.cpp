#include <doctest.h>

#include <algorithm>
#include <atomic>
#include <set>
#include <random>
#include <thread>
#include <unordered_map>

#include "strata/flat.h"
#include "strata/hnsw.h"
#include "test_util.h"

using namespace strata;

namespace {

struct Fixture {
  size_t n, dim, nq;
  std::vector<float> data, queries;
  std::vector<label_t> labels;
  Fixture(size_t n_, size_t dim_, size_t nq_ = 200)
      : n(n_), dim(dim_), nq(nq_), data(testutil::random_vectors(n_, dim_, 1)),
        queries(testutil::random_vectors(nq_, dim_, 2)), labels(testutil::iota_labels(n_)) {}
};

}  // namespace

TEST_CASE("HNSW reaches high recall for every metric") {
  Fixture f(5000, 32);
  for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
    CAPTURE(metric_name(m));
    HNSWIndex index(f.dim, m, {16, 200, 42, true});
    index.add(f.data.data(), f.labels.data(), f.n, 4);
    FlatIndex exact(f.dim, m);
    exact.add(f.data.data(), f.labels.data(), f.n);
    CHECK(index.size() == f.n);
    const double recall = testutil::recall_at_k(index, exact, f.queries, f.nq, 10, 128);
    CHECK(recall >= 0.95);
  }
}

TEST_CASE("single-threaded and parallel builds are both accurate") {
  Fixture f(8000, 24);
  FlatIndex exact(f.dim, Metric::L2);
  exact.add(f.data.data(), f.labels.data(), f.n);
  for (int threads : {1, 8}) {
    CAPTURE(threads);
    HNSWIndex index(f.dim, Metric::L2);
    index.add(f.data.data(), f.labels.data(), f.n, threads);
    CHECK(testutil::recall_at_k(index, exact, f.queries, f.nq, 10, 100) >= 0.95);
    const HNSWStats s = index.stats();
    CHECK(s.mean_degree_level0 > 4.0);
    CHECK(s.max_level >= 1);
  }
}

TEST_CASE("incremental adds grow capacity and keep recall") {
  Fixture f(6000, 16);
  HNSWIndex index(f.dim, Metric::L2, {}, 16);  // force many reallocations
  for (size_t start = 0; start < f.n; start += 500) {
    index.add(f.data.data() + start * f.dim, f.labels.data() + start, 500, 4);
  }
  FlatIndex exact(f.dim, Metric::L2);
  exact.add(f.data.data(), f.labels.data(), f.n);
  CHECK(index.size() == f.n);
  CHECK(testutil::recall_at_k(index, exact, f.queries, f.nq, 10, 100) >= 0.95);
}

TEST_CASE("results are sorted, exact match comes first, k larger than size is handled") {
  Fixture f(50, 8);
  HNSWIndex index(f.dim, Metric::L2);
  index.add(f.data.data(), f.labels.data(), f.n);
  const auto res = index.search(f.data.data() + 7 * f.dim, 100, 200);
  REQUIRE(res.size() == f.n);
  CHECK(res[0].label == 7);
  CHECK(res[0].distance == doctest::Approx(0.0).epsilon(1e-6));
  for (size_t i = 1; i < res.size(); ++i) CHECK(res[i - 1].distance <= res[i].distance);
}

TEST_CASE("empty index and k = 0") {
  HNSWIndex index(4, Metric::L2);
  const float q[4] = {0, 0, 0, 0};
  CHECK(index.search(q, 5).empty());
  const float v[4] = {1, 2, 3, 4};
  const label_t l = 9;
  index.add(v, &l, 1);
  CHECK(index.search(q, 0).empty());
  CHECK(index.search(q, 3).size() == 1);
}

TEST_CASE("upsert replaces the vector for an existing label") {
  HNSWIndex index(2, Metric::L2);
  const float a[2] = {0, 0};
  const float b[2] = {100, 100};
  const label_t l = 5;
  index.add(a, &l, 1);
  index.add(b, &l, 1);
  CHECK(index.size() == 1);
  CHECK(index.stats().deleted == 1);
  const auto res = index.search(b, 1);
  REQUIRE(res.size() == 1);
  CHECK(res[0].label == 5);
  CHECK(res[0].distance == doctest::Approx(0.0));
  float out[2];
  REQUIRE(index.get_vector(5, out));
  CHECK(out[0] == 100.f);
}

TEST_CASE("duplicate labels in one batch: the last one wins") {
  HNSWIndex index(1, Metric::L2);
  const float v[3] = {1, 2, 3};
  const label_t l[3] = {4, 4, 4};
  index.add(v, l, 3);
  CHECK(index.size() == 1);
  float out;
  REQUIRE(index.get_vector(4, &out));
  CHECK(out == 3.f);
}

TEST_CASE("removed labels are never returned and recall over the rest stays high") {
  Fixture f(4000, 16);
  HNSWIndex index(f.dim, Metric::L2);
  index.add(f.data.data(), f.labels.data(), f.n, 4);
  std::vector<label_t> kept;
  std::vector<float> kept_vecs;
  for (size_t i = 0; i < f.n; ++i) {
    if (i % 3 == 0) {
      CHECK(index.remove(i));
    } else {
      kept.push_back(i);
      kept_vecs.insert(kept_vecs.end(), f.data.begin() + i * f.dim, f.data.begin() + (i + 1) * f.dim);
    }
  }
  CHECK_FALSE(index.remove(0));  // already gone
  CHECK_FALSE(index.contains(3));
  CHECK(index.contains(4));
  CHECK(index.size() == kept.size());

  FlatIndex exact(f.dim, Metric::L2);
  exact.add(kept_vecs.data(), kept.data(), kept.size());
  for (size_t i = 0; i < f.nq; ++i) {
    for (const auto& r : index.search(f.queries.data() + i * f.dim, 10, 64)) CHECK(r.label % 3 != 0);
  }
  CHECK(testutil::recall_at_k(index, exact, f.queries, f.nq, 10, 100) >= 0.93);

  index.compact(4);
  CHECK(index.stats().deleted == 0);
  CHECK(index.size() == kept.size());
  CHECK(testutil::recall_at_k(index, exact, f.queries, f.nq, 10, 100) >= 0.95);
}

TEST_CASE("allow and deny filters") {
  Fixture f(3000, 16);
  HNSWIndex index(f.dim, Metric::L2);
  index.add(f.data.data(), f.labels.data(), f.n, 4);
  FlatIndex exact(f.dim, Metric::L2);
  exact.add(f.data.data(), f.labels.data(), f.n);

  std::vector<label_t> even;
  for (label_t i = 0; i < f.n; i += 2) even.push_back(i);
  LabelFilter allow(even.data(), even.size(), LabelFilter::Mode::Allow);
  LabelFilter deny(even.data(), even.size(), LabelFilter::Mode::Deny);

  SUBCASE("graph path") {
    index.set_flat_search_cutoff(0);
    for (size_t i = 0; i < 50; ++i) {
      for (const auto& r : index.search(f.queries.data() + i * f.dim, 10, 64, &allow)) CHECK(r.label % 2 == 0);
      for (const auto& r : index.search(f.queries.data() + i * f.dim, 10, 64, &deny)) CHECK(r.label % 2 == 1);
    }
    CHECK(testutil::recall_at_k(index, exact, f.queries, f.nq, 10, 100, &allow) >= 0.93);
  }
  SUBCASE("brute-force path is exact for small allow-lists") {
    const std::vector<label_t> few = {5, 17, 900, 2999};
    LabelFilter small(few.data(), few.size(), LabelFilter::Mode::Allow);
    CHECK(testutil::recall_at_k(index, exact, f.queries, f.nq, 3, 10, &small) == 1.0);
  }
}

TEST_CASE("search_batch matches individual searches") {
  Fixture f(2000, 16, 64);
  HNSWIndex index(f.dim, Metric::Cosine);
  index.add(f.data.data(), f.labels.data(), f.n, 4);
  const size_t k = 5;
  std::vector<label_t> labels(f.nq * k);
  std::vector<float> dists(f.nq * k);
  index.search_batch(f.queries.data(), f.nq, k, 50, 8, nullptr, labels.data(), dists.data());
  for (size_t i = 0; i < f.nq; ++i) {
    const auto single = index.search(f.queries.data() + i * f.dim, k, 50);
    for (size_t j = 0; j < k; ++j) CHECK(labels[i * k + j] == single[j].label);
  }
}

TEST_CASE("concurrent readers while the index is idle") {
  Fixture f(3000, 16, 400);
  HNSWIndex index(f.dim, Metric::L2);
  index.add(f.data.data(), f.labels.data(), f.n, 8);
  std::vector<std::thread> readers;
  std::atomic<size_t> bad{0};
  for (int t = 0; t < 8; ++t) {
    readers.emplace_back([&, t] {
      for (size_t i = static_cast<size_t>(t); i < f.nq; i += 8) {
        if (index.search(f.queries.data() + i * f.dim, 10, 64).size() != 10) ++bad;
      }
    });
  }
  // A writer interleaving with the readers must not corrupt anything.
  const auto extra = testutil::random_vectors(500, f.dim, 99);
  const auto extra_labels = testutil::iota_labels(500, 100000);
  index.add(extra.data(), extra_labels.data(), 500, 4);
  for (auto& th : readers) th.join();
  CHECK(bad == 0);
  CHECK(index.size() == f.n + 500);
}

TEST_CASE("plain neighbour selection works too (ablation switch)") {
  Fixture f(3000, 16);
  HNSWParams p;
  p.use_heuristic = false;
  HNSWIndex index(f.dim, Metric::L2, p);
  index.add(f.data.data(), f.labels.data(), f.n, 4);
  FlatIndex exact(f.dim, Metric::L2);
  exact.add(f.data.data(), f.labels.data(), f.n);
  CHECK(testutil::recall_at_k(index, exact, f.queries, f.nq, 10, 128) >= 0.9);
}

TEST_CASE("searches run while a batch is being inserted, and see consistent results") {
  const size_t dim = 24, base = 3000, extra = 20000;
  const auto data = testutil::random_vectors(base + extra, dim, 61);
  const auto labels = testutil::iota_labels(base + extra);
  for (Quantization qz : {Quantization::None, Quantization::SQ8}) {
    CAPTURE(static_cast<int>(qz));
    HNSWParams p;
    p.quantization = qz;
    HNSWIndex index(dim, Metric::L2, p);
    index.add(data.data(), labels.data(), base, 4);

    std::atomic<bool> adding{true};
    std::atomic<size_t> during{0}, bad{0};
    std::vector<std::thread> readers;
    for (int t = 0; t < 3; ++t) {
      readers.emplace_back([&, t] {
        size_t i = static_cast<size_t>(t);
        while (adding.load()) {
          // The base vectors are always present: each must find itself.
          const size_t probe = (i * 7919) % base;
          const auto res = index.search(data.data() + probe * dim, 5, 64);
          if (res.empty() || res[0].label != probe) bad.fetch_add(1);
          for (const auto& r : res) {
            if (r.label >= base + extra) bad.fetch_add(1);
          }
          (void)index.stats();
          during.fetch_add(1);
          ++i;
        }
      });
    }
    index.add(data.data() + base * dim, labels.data() + base, extra, 2);
    adding.store(false);
    for (auto& r : readers) r.join();
    CHECK(during.load() > 0);  // readers made progress while the batch was linking
    CHECK(bad.load() == 0);
    CHECK(index.size() == base + extra);
    FlatIndex exact(dim, Metric::L2);
    exact.add(data.data(), labels.data(), base + extra);
    const auto queries = testutil::random_vectors(100, dim, 62);
    CHECK(testutil::recall_at_k(index, exact, queries, 100, 10, 128) >= 0.9);
  }
}

TEST_CASE("LabelMap matches std::unordered_map under random inserts, overwrites and erases") {
  std::mt19937_64 rng(5);
  LabelMap map;
  std::unordered_map<label_t, node_t> ref;
  for (int step = 0; step < 200000; ++step) {
    const label_t key = rng() % 5000 + (step % 3 == 0 ? (label_t{1} << 40) : 0);  // collisions + spread
    const int op = static_cast<int>(rng() % 10);
    if (op < 6) {
      const auto v = static_cast<node_t>(rng() % 1000000);
      map.set(key, v);
      ref[key] = v;
    } else if (op < 9) {
      CHECK(map.erase(key) == (ref.erase(key) == 1));
    } else {
      auto it = ref.find(key);
      CHECK(map.find(key) == (it == ref.end() ? kInvalidNode : it->second));
    }
    if (step % 20000 == 0) {
      REQUIRE(map.size() == ref.size());
      size_t seen = 0;
      map.for_each([&](label_t k, node_t v) {
        ++seen;
        CHECK(ref.at(k) == v);
      });
      CHECK(seen == ref.size());
    }
  }
  for (const auto& [k, v] : ref) CHECK(map.find(k) == v);
}

TEST_CASE("filter planner: selective allow-lists are answered exactly, small indexes are scanned") {
  Fixture f(20000, 16, 50);
  HNSWIndex index(f.dim, Metric::L2);
  index.add(f.data.data(), f.labels.data(), f.n, 4);
  // 3% selectivity: above the fixed floor (lowered to 100 here), below the cost-based
  // crossover sqrt(M0/2 * ef * n) ~ 4.5k -> answered by exact brute force.
  index.set_flat_search_cutoff(100);
  std::vector<label_t> allowed;
  for (label_t i = 0; i < f.n; i += 33) allowed.push_back(i);
  REQUIRE(allowed.size() > index.flat_search_cutoff());
  const LabelFilter allow(allowed.data(), allowed.size(), LabelFilter::Mode::Allow);
  FlatIndex exact(f.dim, Metric::L2);
  exact.add(f.data.data(), f.labels.data(), f.n);
  CHECK(testutil::recall_at_k(index, exact, f.queries, f.nq, 10, 64, &allow) == doctest::Approx(1.0));

  // ef >= live vectors: every vector is reachable, whatever the graph looks like.
  HNSWIndex tiny(f.dim, Metric::L2, {8, 32, 42, true});
  tiny.add(f.data.data(), f.labels.data(), 100, 8);
  CHECK(tiny.search(f.queries.data(), 100, 128).size() == 100);
}
