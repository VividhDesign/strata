#pragma once

// Hierarchical Navigable Small World graph index.
// Malkov & Yashunin, "Efficient and robust approximate nearest neighbor search using
// Hierarchical Navigable Small World graphs", IEEE TPAMI 2018 (arXiv:1603.09320).

#include <atomic>
#include <memory>
#include <mutex>
#include <queue>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "strata/common.h"
#include "strata/distance.h"
#include "strata/filter.h"
#include "strata/sync.h"
#include "strata/visited.h"

namespace strata {

struct HNSWParams {
  size_t M = 16;                 // max neighbours per node on levels >= 1 (2*M on level 0)
  size_t ef_construction = 200;  // beam width while building
  uint64_t seed = 42;            // node levels are a pure function of (seed, label)
  bool use_heuristic = true;     // diversity heuristic (paper Alg. 4) vs. plain M-closest
};

struct HNSWStats {
  size_t size = 0;
  size_t nodes = 0;
  size_t deleted = 0;
  size_t capacity = 0;
  int max_level = -1;
  double mean_degree_level0 = 0.0;
  size_t memory_bytes = 0;
};

// Thread safety:
//   * add()/remove()/compact() take the index-wide lock exclusively. add() then inserts in
//     parallel internally, coordinating through 1-byte per-node spinlocks.
//   * search()/search_batch() take the lock shared, so any number of queries run in parallel
//     and read the graph without per-node locking.
class HNSWIndex {
 public:
  HNSWIndex(size_t dim, Metric metric, HNSWParams params = {}, size_t initial_capacity = 1024);
  ~HNSWIndex();
  HNSWIndex(const HNSWIndex&) = delete;
  HNSWIndex& operator=(const HNSWIndex&) = delete;

  // Inserts n vectors (row-major, n x dim). An existing label is replaced (upsert): the old
  // node becomes a tombstone. Within one batch the last occurrence of a label wins.
  void add(const float* vectors, const label_t* labels, size_t n, int num_threads = 0);

  // k nearest neighbours, closest first. ef = 0 uses ef_search().
  std::vector<SearchResult> search(const float* query, size_t k, size_t ef = 0,
                                   const LabelFilter* filter = nullptr) const;

  // nq queries in parallel. Outputs are nq x k; missing results are kNoLabel / +inf.
  void search_batch(const float* queries, size_t nq, size_t k, size_t ef, int num_threads,
                    const LabelFilter* filter, label_t* out_labels, float* out_distances) const;

  bool remove(label_t label);  // tombstone; returns false if the label is unknown
  bool contains(label_t label) const;
  bool get_vector(label_t label, float* out) const;  // stored (normalised for cosine) vector
  std::vector<label_t> labels() const;

  // Rebuilds the graph without tombstones.
  void compact(int num_threads = 0);

  // Binary snapshot with a CRC-32 footer, written to a temp file and renamed into place.
  void save(const std::string& path) const;
  static std::unique_ptr<HNSWIndex> load(const std::string& path);

  size_t size() const;  // live vectors
  size_t dim() const { return dim_; }
  Metric metric() const { return metric_; }
  const HNSWParams& params() const { return params_; }
  HNSWStats stats() const;

  size_t ef_search() const { return ef_search_.load(); }
  void set_ef_search(size_t ef) { ef_search_.store(ef ? ef : 1); }
  // Allow-filters with at most this many labels are answered by exact brute force over the
  // allowed set instead of a filtered graph walk (which degrades when few nodes qualify).
  size_t flat_search_cutoff() const { return flat_cutoff_.load(); }
  void set_flat_search_cutoff(size_t n) { flat_cutoff_.store(n); }

 private:
  using Candidate = std::pair<float, node_t>;
  using MaxHeap = std::priority_queue<Candidate>;
  using MinHeap = std::priority_queue<Candidate, std::vector<Candidate>, std::greater<Candidate>>;

  // Storage layout. Level 0 holds one fixed-size block per node, so the links and the vector
  // of a node share cache lines:
  //   [count:u32][neighbour ids:u32 x 2M][pad to 16B][vector:f32 x dim][pad to 16B]
  // Upper levels (rarely visited) live in a separate per-node array:
  //   level l (>= 1): [count:u32][neighbour ids:u32 x M]
  uint32_t* links0(node_t n) const {
    return reinterpret_cast<uint32_t*>(level0_ + static_cast<size_t>(n) * size_per_element_);
  }
  uint32_t* links(node_t n, int level) const {
    return level == 0 ? links0(n) : upper_[n].get() + static_cast<size_t>(level - 1) * (maxM_ + 1);
  }
  const float* vec(node_t n) const {
    return reinterpret_cast<const float*>(level0_ + static_cast<size_t>(n) * size_per_element_ +
                                          data_offset_);
  }
  float* mutable_vec(node_t n) {
    return reinterpret_cast<float*>(level0_ + static_cast<size_t>(n) * size_per_element_ + data_offset_);
  }

  void grow_to(size_t new_capacity);
  int random_level(label_t label) const;
  void insert_one(const float* vector, label_t label);

  template <bool kLock>
  void greedy_search(const float* q, node_t& ep, float& ep_dist, int level) const;
  MaxHeap search_layer_build(const float* q, node_t ep, int level) const;
  template <bool kFiltered>
  void search_layer_query(const float* q, node_t ep, size_t ef, const LabelFilter* filter,
                          MaxHeap& top) const;
  std::vector<Candidate> select_neighbors(MaxHeap& candidates, size_t m) const;
  node_t connect(node_t cur, MaxHeap& candidates, int level);

  std::vector<SearchResult> search_unlocked(const float* query, size_t k, size_t ef,
                                            const LabelFilter* filter) const;
  std::vector<SearchResult> brute_force(const float* q, size_t k, const LabelFilter& allow) const;
  bool is_result(node_t n, const LabelFilter* filter) const {
    return !deleted_[n] && (filter == nullptr || filter->allows(labels_[n]));
  }
  void swap_storage(HNSWIndex& other);

  // Configuration
  size_t dim_;
  Metric metric_;
  HNSWParams params_;
  size_t maxM_;
  size_t maxM0_;
  double level_mult_;
  DistanceFn dist_;
  size_t data_offset_;
  size_t size_per_element_;
  std::atomic<size_t> ef_search_{64};
  std::atomic<size_t> flat_cutoff_{2048};

  // Storage
  size_t capacity_ = 0;
  std::atomic<size_t> count_{0};  // nodes ever inserted (incl. tombstones)
  char* level0_ = nullptr;
  std::vector<int32_t> levels_;
  std::vector<std::unique_ptr<uint32_t[]>> upper_;
  std::vector<label_t> labels_;
  std::vector<uint8_t> deleted_;
  size_t num_deleted_ = 0;
  std::unique_ptr<SpinLock[]> node_locks_;
  std::unordered_map<label_t, node_t> label_to_node_;

  // Graph entry point
  node_t entry_ = kInvalidNode;
  int max_level_ = -1;

  mutable std::shared_mutex rw_;  // readers: queries; writer: add/remove/compact
  std::mutex label_mu_;           // label map, during parallel inserts
  std::mutex entry_mu_;           // entry point, during parallel inserts
  mutable VisitedPool visited_;
};

}  // namespace strata
