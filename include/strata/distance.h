#pragma once

#include "strata/common.h"

namespace strata {

using DistanceFn = float (*)(const float*, const float*, size_t);

namespace kernels {

// Plain loops with auto-vectorisation disabled: the baseline for the SIMD speedup numbers.
float l2sq_scalar(const float* a, const float* b, size_t dim);
float dot_scalar(const float* a, const float* b, size_t dim);

// Hand-written SIMD kernels (NEON on ARM64, AVX2+FMA on x86-64), scalar fallback otherwise.
float l2sq(const float* a, const float* b, size_t dim);
float dot(const float* a, const float* b, size_t dim);

// 8-bit scalar quantization (SQ8). `c` holds one uint8 code per dimension.
//   sq8_l2:  sum_i w[i] * (q[i] - c[i])^2   (q already shifted/scaled into code space)
//   sq8_dot: sum_i q[i] * c[i]
float sq8_l2_scalar(const float* q, const float* w, const uint8_t* c, size_t dim);
float sq8_dot_scalar(const float* q, const uint8_t* c, size_t dim);
float sq8_l2(const float* q, const float* w, const uint8_t* c, size_t dim);
float sq8_dot(const float* q, const uint8_t* c, size_t dim);

// "neon", "avx2" or "scalar".
const char* simd_backend();

}  // namespace kernels

// Distance used inside the index (smaller is closer):
//   L2           -> squared Euclidean distance
//   InnerProduct -> 1 - dot(a, b)
//   Cosine       -> 1 - dot(a, b) on normalised vectors
DistanceFn distance_function(Metric metric, bool use_simd = true);

void normalize_inplace(float* v, size_t dim);

}  // namespace strata
