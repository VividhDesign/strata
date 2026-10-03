#pragma once

#include <algorithm>
#include <vector>

#include "strata/common.h"

namespace strata {

// Restricts search results to (Allow) or away from (Deny) a set of labels.
// Filtered nodes are still traversed by the graph search, so connectivity is preserved;
// they just never enter the result set.
//
// Labels are kept as a sorted array and probed with binary search. Building a hash set
// per query cost more than the search itself for large filters (measured: 20k-id filters
// ran at 3.5k req/s through the server with a hash set).
class LabelFilter {
 public:
  enum class Mode { Allow, Deny };

  LabelFilter(const label_t* labels, size_t n, Mode mode) : labels_(labels, labels + n), mode_(mode) {
    if (!std::is_sorted(labels_.begin(), labels_.end())) std::sort(labels_.begin(), labels_.end());
    labels_.erase(std::unique(labels_.begin(), labels_.end()), labels_.end());
  }
  LabelFilter(std::vector<label_t>&& sorted_unique, Mode mode) : labels_(std::move(sorted_unique)), mode_(mode) {}

  bool allows(label_t label) const {
    return std::binary_search(labels_.begin(), labels_.end(), label) == (mode_ == Mode::Allow);
  }
  Mode mode() const { return mode_; }
  size_t size() const { return labels_.size(); }
  const std::vector<label_t>& labels() const { return labels_; }

 private:
  std::vector<label_t> labels_;
  Mode mode_;
};

}  // namespace strata
