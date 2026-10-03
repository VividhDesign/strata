#include <doctest.h>

#include <cmath>

#include "strata/distance.h"
#include "strata/io.h"
#include "test_util.h"

using namespace strata;

TEST_CASE("SIMD kernels match scalar kernels for every dimension remainder") {
  for (size_t dim = 1; dim <= 300; ++dim) {
    const auto a = testutil::random_vectors(1, dim, static_cast<uint32_t>(dim));
    const auto b = testutil::random_vectors(1, dim, static_cast<uint32_t>(dim) + 1000);
    const float l2s = kernels::l2sq_scalar(a.data(), b.data(), dim);
    const float l2v = kernels::l2sq(a.data(), b.data(), dim);
    const float dps = kernels::dot_scalar(a.data(), b.data(), dim);
    const float dpv = kernels::dot(a.data(), b.data(), dim);
    CHECK(l2v == doctest::Approx(l2s).epsilon(1e-4));
    CHECK(dpv == doctest::Approx(dps).epsilon(1e-3).scale(1.0));
  }
}

TEST_CASE("distance functions follow the documented conventions") {
  const float a[3] = {1, 0, 0};
  const float b[3] = {0, 1, 0};
  CHECK(distance_function(Metric::L2)(a, b, 3) == doctest::Approx(2.0));
  CHECK(distance_function(Metric::InnerProduct)(a, a, 3) == doctest::Approx(0.0));
  CHECK(distance_function(Metric::Cosine)(a, b, 3) == doctest::Approx(1.0));
}

TEST_CASE("normalize_inplace produces unit vectors and leaves zero vectors alone") {
  float v[4] = {3, 4, 0, 0};
  normalize_inplace(v, 4);
  CHECK(v[0] == doctest::Approx(0.6));
  CHECK(v[1] == doctest::Approx(0.8));
  float z[2] = {0, 0};
  normalize_inplace(z, 2);
  CHECK(z[0] == 0.f);
}

TEST_CASE("metric names round-trip") {
  CHECK(parse_metric("L2") == Metric::L2);
  CHECK(parse_metric("cosine") == Metric::Cosine);
  CHECK(parse_metric("inner_product") == Metric::InnerProduct);
  CHECK_THROWS_AS(parse_metric("manhattan"), Error);
}

TEST_CASE("crc32 matches the standard check value") {
  const char* s = "123456789";
  CHECK(crc32_update(0, s, 9) == 0xCBF43926u);
  // Incremental updates give the same result as one pass.
  const uint32_t part = crc32_update(0, s, 4);
  CHECK(crc32_update(part, s + 4, 5) == 0xCBF43926u);
}
