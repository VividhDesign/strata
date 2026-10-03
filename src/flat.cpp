#include "strata/flat.h"

#include <limits>
#include <queue>

#include "strata/sync.h"

namespace strata {

FlatIndex::FlatIndex(size_t dim, Metric metric) : dim_(dim), metric_(metric), dist_(distance_function(metric)) {
  if (dim_ == 0) throw Error("dim must be > 0");
}

void FlatIndex::add(const float* vectors, const label_t* labels, size_t n) {
  const size_t offset = data_.size();
  data_.insert(data_.end(), vectors, vectors + n * dim_);
  labels_.insert(labels_.end(), labels, labels + n);
  if (metric_ == Metric::Cosine) {
    for (size_t i = 0; i < n; ++i) normalize_inplace(data_.data() + offset + i * dim_, dim_);
  }
}

std::vector<SearchResult> FlatIndex::search(const float* query, size_t k, const LabelFilter* filter) const {
  std::vector<float> normalized;
  const float* q = query;
  if (metric_ == Metric::Cosine) {
    normalized.assign(query, query + dim_);
    normalize_inplace(normalized.data(), dim_);
    q = normalized.data();
  }
  std::priority_queue<std::pair<float, size_t>> top;
  for (size_t i = 0; i < labels_.size(); ++i) {
    if (filter && !filter->allows(labels_[i])) continue;
    const float d = dist_(q, data_.data() + i * dim_, dim_);
    if (top.size() < k) {
      top.emplace(d, i);
    } else if (k > 0 && d < top.top().first) {
      top.pop();
      top.emplace(d, i);
    }
  }
  std::vector<SearchResult> out(top.size());
  for (size_t i = top.size(); i-- > 0;) {
    out[i] = {labels_[top.top().second], top.top().first};
    top.pop();
  }
  return out;
}

void FlatIndex::search_batch(const float* queries, size_t nq, size_t k, int num_threads,
                             const LabelFilter* filter, label_t* out_labels, float* out_distances) const {
  parallel_for(nq, num_threads, [&](size_t i) {
    const auto res = search(queries + i * dim_, k, filter);
    for (size_t j = 0; j < k; ++j) {
      const bool has = j < res.size();
      out_labels[i * k + j] = has ? res[j].label : kNoLabel;
      out_distances[i * k + j] = has ? res[j].distance : std::numeric_limits<float>::infinity();
    }
  });
}

}  // namespace strata
