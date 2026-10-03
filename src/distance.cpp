#include "strata/distance.h"

#include <algorithm>
#include <cctype>
#include <cmath>

#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#define STRATA_NEON 1
#elif defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define STRATA_AVX2 1
#endif

#if defined(__clang__)
#define STRATA_NO_VECTORIZE _Pragma("clang loop vectorize(disable) interleave(disable)")
#else
#define STRATA_NO_VECTORIZE
#endif

namespace strata {

const char* metric_name(Metric m) {
  switch (m) {
    case Metric::L2:
      return "l2";
    case Metric::InnerProduct:
      return "ip";
    case Metric::Cosine:
      return "cosine";
  }
  return "unknown";
}

Metric parse_metric(const std::string& name) {
  std::string s = name;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  if (s == "l2" || s == "euclidean") return Metric::L2;
  if (s == "ip" || s == "inner_product" || s == "dot") return Metric::InnerProduct;
  if (s == "cosine" || s == "angular") return Metric::Cosine;
  throw Error("unknown metric '" + name + "' (expected l2, ip or cosine)");
}

namespace kernels {

float l2sq_scalar(const float* a, const float* b, size_t dim) {
  float sum = 0.f;
  STRATA_NO_VECTORIZE
  for (size_t i = 0; i < dim; ++i) {
    const float t = a[i] - b[i];
    sum += t * t;
  }
  return sum;
}

float dot_scalar(const float* a, const float* b, size_t dim) {
  float sum = 0.f;
  STRATA_NO_VECTORIZE
  for (size_t i = 0; i < dim; ++i) sum += a[i] * b[i];
  return sum;
}

#if defined(STRATA_NEON)

// Four independent accumulators hide the 3-4 cycle FMA latency.
float l2sq(const float* a, const float* b, size_t dim) {
  float32x4_t s0 = vdupq_n_f32(0.f), s1 = s0, s2 = s0, s3 = s0;
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    const float32x4_t d1 = vsubq_f32(vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    const float32x4_t d2 = vsubq_f32(vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    const float32x4_t d3 = vsubq_f32(vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
    s0 = vfmaq_f32(s0, d0, d0);
    s1 = vfmaq_f32(s1, d1, d1);
    s2 = vfmaq_f32(s2, d2, d2);
    s3 = vfmaq_f32(s3, d3, d3);
  }
  for (; i + 4 <= dim; i += 4) {
    const float32x4_t d0 = vsubq_f32(vld1q_f32(a + i), vld1q_f32(b + i));
    s0 = vfmaq_f32(s0, d0, d0);
  }
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) {
    const float t = a[i] - b[i];
    sum += t * t;
  }
  return sum;
}

float dot(const float* a, const float* b, size_t dim) {
  float32x4_t s0 = vdupq_n_f32(0.f), s1 = s0, s2 = s0, s3 = s0;
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  for (; i + 4 <= dim; i += 4) s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
  float sum = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < dim; ++i) sum += a[i] * b[i];
  return sum;
}

const char* simd_backend() { return "neon"; }

#elif defined(STRATA_AVX2)

static inline float hsum256(__m256 v) {
  __m128 lo = _mm256_castps256_ps128(v);
  const __m128 hi = _mm256_extractf128_ps(v, 1);
  lo = _mm_add_ps(lo, hi);
  __m128 shuf = _mm_movehdup_ps(lo);
  __m128 sums = _mm_add_ps(lo, shuf);
  shuf = _mm_movehl_ps(shuf, sums);
  sums = _mm_add_ss(sums, shuf);
  return _mm_cvtss_f32(sums);
}

float l2sq(const float* a, const float* b, size_t dim) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    const __m256 d1 = _mm256_sub_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
    s1 = _mm256_fmadd_ps(d1, d1, s1);
  }
  for (; i + 8 <= dim; i += 8) {
    const __m256 d0 = _mm256_sub_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i));
    s0 = _mm256_fmadd_ps(d0, d0, s0);
  }
  float sum = hsum256(_mm256_add_ps(s0, s1));
  for (; i < dim; ++i) {
    const float t = a[i] - b[i];
    sum += t * t;
  }
  return sum;
}

float dot(const float* a, const float* b, size_t dim) {
  __m256 s0 = _mm256_setzero_ps(), s1 = _mm256_setzero_ps();
  size_t i = 0;
  for (; i + 16 <= dim; i += 16) {
    s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
    s1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i + 8), _mm256_loadu_ps(b + i + 8), s1);
  }
  for (; i + 8 <= dim; i += 8) s0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i), s0);
  float sum = hsum256(_mm256_add_ps(s0, s1));
  for (; i < dim; ++i) sum += a[i] * b[i];
  return sum;
}

const char* simd_backend() { return "avx2"; }

#else

float l2sq(const float* a, const float* b, size_t dim) { return l2sq_scalar(a, b, dim); }
float dot(const float* a, const float* b, size_t dim) { return dot_scalar(a, b, dim); }
const char* simd_backend() { return "scalar"; }

#endif

}  // namespace kernels

namespace {
float ip_distance_simd(const float* a, const float* b, size_t dim) { return 1.f - kernels::dot(a, b, dim); }
float ip_distance_scalar(const float* a, const float* b, size_t dim) {
  return 1.f - kernels::dot_scalar(a, b, dim);
}
}  // namespace

DistanceFn distance_function(Metric metric, bool use_simd) {
  if (metric == Metric::L2) return use_simd ? kernels::l2sq : kernels::l2sq_scalar;
  return use_simd ? ip_distance_simd : ip_distance_scalar;
}

void normalize_inplace(float* v, size_t dim) {
  const float norm = std::sqrt(kernels::dot(v, v, dim));
  if (norm > 0.f) {
    const float inv = 1.f / norm;
    for (size_t i = 0; i < dim; ++i) v[i] *= inv;
  }
}

}  // namespace strata
