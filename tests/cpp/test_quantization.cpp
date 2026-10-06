#include <doctest.h>

#include <cmath>
#include <filesystem>
#include <set>

#include "strata/distance.h"
#include "strata/flat.h"
#include "strata/hnsw.h"
#include "test_util.h"

using namespace strata;

namespace {

HNSWParams sq8(bool rerank) {
  HNSWParams p;
  p.quantization = Quantization::SQ8;
  p.rerank = rerank;
  return p;
}

}  // namespace

TEST_CASE("SQ8 SIMD kernels match scalar kernels for every dimension remainder") {
  for (size_t dim = 1; dim <= 300; ++dim) {
    const auto q = testutil::random_vectors(1, dim, static_cast<uint32_t>(dim));
    const auto w0 = testutil::random_vectors(1, dim, static_cast<uint32_t>(dim) + 7);
    std::vector<float> w(dim);
    std::vector<uint8_t> c(dim);
    for (size_t i = 0; i < dim; ++i) {
      w[i] = std::abs(w0[i]);
      c[i] = static_cast<uint8_t>((i * 37 + dim * 11) % 256);
    }
    CHECK(kernels::sq8_l2(q.data(), w.data(), c.data(), dim) ==
          doctest::Approx(kernels::sq8_l2_scalar(q.data(), w.data(), c.data(), dim)).epsilon(1e-4));
    CHECK(kernels::sq8_dot(q.data(), c.data(), dim) ==
          doctest::Approx(kernels::sq8_dot_scalar(q.data(), c.data(), dim)).epsilon(1e-3).scale(1.0));
  }
}

TEST_CASE("quantization names round-trip") {
  for (Quantization q : {Quantization::None, Quantization::SQ8}) CHECK(parse_quantization(quantization_name(q)) == q);
  CHECK(parse_quantization("INT8") == Quantization::SQ8);
  CHECK_THROWS_AS(parse_quantization("pq"), Error);
}

TEST_CASE("SQ8 with rerank keeps the recall of the float index for every metric") {
  const size_t n = 5000, dim = 32, nq = 200;
  const auto data = testutil::random_vectors(n, dim, 1);
  const auto queries = testutil::random_vectors(nq, dim, 2);
  const auto labels = testutil::iota_labels(n);
  for (Metric m : {Metric::L2, Metric::InnerProduct, Metric::Cosine}) {
    CAPTURE(metric_name(m));
    FlatIndex exact(dim, m);
    exact.add(data.data(), labels.data(), n);
    HNSWIndex index(dim, m, sq8(true));
    index.add(data.data(), labels.data(), n, 4);
    CHECK(testutil::recall_at_k(index, exact, queries, nq, 10, 128) >= 0.95);
    // Re-ranked distances are exact.
    const auto got = index.search(queries.data(), 5, 128);
    const auto truth = exact.search(queries.data(), 5);
    REQUIRE(got.size() == 5);
    CHECK(got[0].label == truth[0].label);
    CHECK(got[0].distance == doctest::Approx(truth[0].distance).epsilon(1e-4));
  }
}

TEST_CASE("SQ8 without rerank: approximate but accurate, and the graph block is ~4x smaller") {
  const size_t n = 5000, dim = 64, nq = 200;
  const auto data = testutil::random_vectors(n, dim, 11);
  const auto queries = testutil::random_vectors(nq, dim, 12);
  const auto labels = testutil::iota_labels(n);
  for (Metric m : {Metric::L2, Metric::Cosine}) {
    CAPTURE(metric_name(m));
    FlatIndex exact(dim, m);
    exact.add(data.data(), labels.data(), n);
    HNSWIndex f32(dim, m, {}, n);
    f32.add(data.data(), labels.data(), n, 4);
    HNSWIndex q(dim, m, sq8(false), n);
    q.add(data.data(), labels.data(), n, 4);
    CHECK(testutil::recall_at_k(q, exact, queries, nq, 10, 128) >= 0.85);
    CHECK(q.stats().memory_bytes < f32.stats().memory_bytes * 6 / 10);
    // Decoded vectors are within half a quantization step of the originals (after normalisation).
    std::vector<float> got(dim), want(data.begin() + 3 * dim, data.begin() + 4 * dim);
    if (m == Metric::Cosine) normalize_inplace(want.data(), dim);
    REQUIRE(q.get_vector(3, got.data()));
    double err = 0;
    for (size_t d = 0; d < dim; ++d) err = std::max(err, static_cast<double>(std::abs(got[d] - want[d])));
    CHECK(err < (m == Metric::Cosine ? 0.01 : 0.05));
  }
}

TEST_CASE("SQ8 supports later batches, upserts, deletes, filters and compaction") {
  const size_t n = 4000, dim = 24, nq = 100;
  const auto data = testutil::random_vectors(n, dim, 21);
  const auto queries = testutil::random_vectors(nq, dim, 22);
  const auto labels = testutil::iota_labels(n);
  for (bool rerank : {true, false}) {
    CAPTURE(rerank);
    HNSWIndex index(dim, Metric::L2, sq8(rerank), 16);
    for (size_t s = 0; s < n; s += 1000) index.add(data.data() + s * dim, labels.data() + s, 1000, 4);
    for (label_t l = 0; l < 400; ++l) index.remove(l);
    CHECK(index.size() == n - 400);
    const auto res = index.search(queries.data(), 50, 200);
    for (const auto& r : res) CHECK(r.label >= 400);

    std::vector<label_t> allow;
    for (label_t l = 1000; l < 1100; ++l) allow.push_back(l);
    const LabelFilter f(allow.data(), allow.size(), LabelFilter::Mode::Allow);
    for (size_t cutoff : {size_t{0}, size_t{2048}}) {  // graph walk and brute-force paths
      index.set_flat_search_cutoff(cutoff);
      const auto fr = index.search(queries.data(), 10, 64, &f);
      CHECK(fr.size() == 10);
      for (const auto& r : fr) CHECK((r.label >= 1000 && r.label < 1100));
    }

    index.compact(4);
    CHECK(index.stats().deleted == 0);
    std::vector<float> live(data.begin() + 400 * dim, data.end());
    FlatIndex exact(dim, Metric::L2);
    exact.add(live.data(), labels.data() + 400, n - 400);
    CHECK(testutil::recall_at_k(index, exact, queries, nq, 10, 128) >= (rerank ? 0.95 : 0.85));
  }
}

TEST_CASE("SQ8 index save/load round-trip returns identical results") {
  testutil::TempDir tmp;
  const size_t n = 3000, dim = 20;
  const auto data = testutil::random_vectors(n, dim, 3);
  const auto labels = testutil::iota_labels(n, 1000);
  const auto queries = testutil::random_vectors(50, dim, 4);
  for (bool rerank : {true, false}) {
    CAPTURE(rerank);
    HNSWIndex index(dim, Metric::Cosine, sq8(rerank));
    index.add(data.data(), labels.data(), n, 4);
    index.remove(1005);
    index.save(tmp.str("q.bin"));
    auto loaded = HNSWIndex::load(tmp.str("q.bin"));
    CHECK(loaded->params().quantization == Quantization::SQ8);
    CHECK(loaded->params().rerank == rerank);
    CHECK(loaded->size() == index.size());
    for (size_t i = 0; i < 50; ++i) {
      const auto a = index.search(queries.data() + i * dim, 10);
      const auto b = loaded->search(queries.data() + i * dim, 10);
      REQUIRE(a.size() == b.size());
      for (size_t j = 0; j < a.size(); ++j) {
        CHECK(a[j].label == b[j].label);
        CHECK(a[j].distance == b[j].distance);
      }
    }
    const auto extra = testutil::random_vectors(10, dim, 5);
    const auto extra_labels = testutil::iota_labels(10, 50000);
    loaded->add(extra.data(), extra_labels.data(), 10);
    CHECK(loaded->size() == index.size() + 10);
  }
}

TEST_CASE("SQ8 ranges widen while the index is small, so one-by-one inserts stay accurate") {
  const size_t n = 2000, dim = 16, nq = 100;
  const auto data = testutil::random_vectors(n, dim, 31);
  const auto queries = testutil::random_vectors(nq, dim, 32);
  const auto labels = testutil::iota_labels(n);
  FlatIndex exact(dim, Metric::L2);
  exact.add(data.data(), labels.data(), n);
  for (bool rerank : {true, false}) {
    CAPTURE(rerank);
    HNSWIndex index(dim, Metric::L2, sq8(rerank));
    for (size_t i = 0; i < n; ++i) index.add(data.data() + i * dim, labels.data() + i, 1);
    CHECK(testutil::recall_at_k(index, exact, queries, nq, 10, 128) >= (rerank ? 0.95 : 0.85));
  }
}

TEST_CASE("SQ8 index with memory-mapped re-rank vectors gives identical results and stays writable") {
  testutil::TempDir tmp;
  const size_t n = 3000, dim = 40;
  const auto data = testutil::random_vectors(n, dim, 41);
  const auto labels = testutil::iota_labels(n);
  const auto queries = testutil::random_vectors(50, dim, 42);
  HNSWIndex index(dim, Metric::L2, sq8(true));
  index.add(data.data(), labels.data(), n, 4);
  index.remove(17);
  index.save(tmp.str("m.bin"));

  auto mapped = HNSWIndex::load(tmp.str("m.bin"), /*mmap_vectors=*/true);
  // The float vectors are no longer counted in RAM.
  CHECK(mapped->stats().memory_bytes + n * dim * sizeof(float) <= index.stats().memory_bytes);
  for (size_t i = 0; i < 50; ++i) {
    const auto a = index.search(queries.data() + i * dim, 10);
    const auto b = mapped->search(queries.data() + i * dim, 10);
    REQUIRE(a.size() == b.size());
    for (size_t j = 0; j < a.size(); ++j) {
      CHECK(a[j].label == b[j].label);
      CHECK(a[j].distance == b[j].distance);
    }
  }
  std::vector<float> v(dim);
  REQUIRE(mapped->get_vector(5, v.data()));
  for (size_t d = 0; d < dim; ++d) CHECK(v[d] == data[5 * dim + d]);

  // Re-saving from the mapping, then writing (which copies the vectors into memory), both work.
  mapped->save(tmp.str("m2.bin"));
  const auto extra = testutil::random_vectors(10, dim, 43);
  const auto extra_labels = testutil::iota_labels(10, 90000);
  mapped->add(extra.data(), extra_labels.data(), 10);
  CHECK(mapped->size() == index.size() + 10);
  CHECK(mapped->search(extra.data(), 1)[0].label == 90000);
  auto again = HNSWIndex::load(tmp.str("m2.bin"), true);
  CHECK(again->size() == index.size());
  again->compact(2);
  CHECK(again->size() == index.size());
  CHECK(again->search(data.data() + 5 * dim, 1)[0].label == 5);

  // A truncated file is rejected in mmap mode too.
  std::filesystem::resize_file(tmp.str("m.bin"), std::filesystem::file_size(tmp.str("m.bin")) - 100);
  CHECK_THROWS_AS(HNSWIndex::load(tmp.str("m.bin"), true), Error);
}
