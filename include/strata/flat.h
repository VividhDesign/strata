#pragma once

#include <vector>

#include "strata/common.h"
#include "strata/distance.h"
#include "strata/filter.h"

namespace strata {

// Exact brute-force search. Used for ground truth, as a baseline, and in tests.
// Not safe for concurrent add() + search().
class FlatIndex {
 public:
  FlatIndex(size_t dim, Metric metric);

  void add(const float* vectors, const label_t* labels, size_t n);
  std::vector<SearchResult> search(const float* query, size_t k,
                                   const LabelFilter* filter = nullptr) const;
  void search_batch(const float* queries, size_t nq, size_t k, int num_threads,
                    const LabelFilter* filter, label_t* out_labels, float* out_distances) const;

  size_t size() const { return labels_.size(); }
  size_t dim() const { return dim_; }
  Metric metric() const { return metric_; }

 private:
  size_t dim_;
  Metric metric_;
  DistanceFn dist_;
  std::vector<float> data_;
  std::vector<label_t> labels_;
};

}  // namespace strata
