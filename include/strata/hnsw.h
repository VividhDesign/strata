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
#include "strata/label_map.h"
#include "strata/sync.h"
#include "strata/visited.h"

namespace strata {

class MappedFile;

enum class Quantization : uint32_t {
  None = 0,  // float32 vectors in the graph
  SQ8 = 1,   // 8-bit scalar quantization: one byte per dimension (per-dimension min/max)
};

const char* quantization_name(Quantization q);
Quantization parse_quantization(const std::string& name);

struct HNSWParams {
  size_t M = 16;                 // max neighbours per node on levels >= 1 (2*M on level 0)
  size_t ef_construction = 200;  // beam width while building
  uint64_t seed = 42;            // node levels are a pure function of (seed, label)
  bool use_heuristic = true;     // diversity heuristic (paper Alg. 4) vs. plain M-closest
  Quantization quantization = Quantization::None;
  // With quantization: also keep the float32 vectors (outside the graph blocks). They are used
  // to build the graph at full precision and to re-rank the ef candidates of a query, so recall
  // matches the float index while traversal reads 4x less memory. false drops them: the index
  // is ~3x smaller, the graph is built on codes and distances are approximate.
  bool rerank = true;
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
//   * Writers (add/remove/compact) are serialised by a writer mutex.
//   * add() holds the index-wide lock exclusively only for a short first phase (allocate nodes,
//     store vectors, update the label map). It then links the new nodes into the graph in
//     parallel under the *shared* lock, coordinating through 1-byte per-node spinlocks, so
//     searches keep running while a batch is being inserted. A new vector is visible to
//     filtered brute-force search and get_vector() after the first phase, and to graph search
//     once it is linked.
//   * search()/search_batch() take the lock shared. When no insert is linking, they read the
//     graph without per-node locking; during an insert they copy neighbour lists under the
//     node spinlocks.
//   * remove() and compact() take the lock exclusively (remove is O(1); compact rebuilds).
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
  // mmap_vectors: for a quantized index with rerank, memory-map the float32 vectors from the
  // file instead of reading them into RAM. Only the graph and the SQ8 codes stay resident;
  // the floats are paged in for the final re-rank. The first add() copies them into memory.
  static std::unique_ptr<HNSWIndex> load(const std::string& path, bool mmap_vectors = false);

  size_t size() const;  // live vectors
  size_t dim() const { return dim_; }
  Metric metric() const { return metric_; }
  const HNSWParams& params() const { return params_; }
  HNSWStats stats() const;

  size_t ef_search() const { return ef_search_.load(); }
  void set_ef_search(size_t ef) { ef_search_.store(ef ? ef : 1); }
  // Allow-filters are answered by exact brute force over the allowed set instead of a filtered
  // graph walk when that is cheaper: the walk must visit ~ef/selectivity nodes to fill its
  // results, so brute force wins when |allowed| <= sqrt(M * ef * n) (calibrated, see
  // bench/bench_filter). At least this many labels always use brute force; 0 disables brute
  // force entirely (always walk the graph).
  size_t flat_search_cutoff() const { return flat_cutoff_.load(); }
  void set_flat_search_cutoff(size_t n) { flat_cutoff_.store(n); }


 private:
  using Candidate = std::pair<float, node_t>;

  // A prepared distance computation against stored nodes. `exact` queries compare float32
  // vectors; the others compare against SQ8 codes using the transformed query `t`.
  struct Query {
    const float* raw = nullptr;
    bool exact = true;
    std::vector<float> t;
    float bias = 0.f;
    std::vector<float> decoded;  // scratch for node-to-node distances without float vectors
  };
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
  // Stored element of a node: a float32 vector, or SQ8 codes.
  const char* data(node_t n) const { return level0_ + static_cast<size_t>(n) * size_per_element_ + data_offset_; }
  const float* vec(node_t n) const { return reinterpret_cast<const float*>(data(n)); }
  float* mutable_vec(node_t n) { return const_cast<float*>(vec(n)); }
  const uint8_t* code(node_t n) const { return reinterpret_cast<const uint8_t*>(data(n)); }
  // Full-precision vector of a node, or nullptr when only codes are kept.
  const float* full(node_t n) const {
    if (!sq_) return vec(n);
    return sq_keep_raw_ ? raw_ptr_ + static_cast<size_t>(n) * dim_ : nullptr;
  }
  bool has_full() const { return !sq_ || sq_keep_raw_; }

  // Quantization
  void train_sq(const float* vectors, const std::vector<size_t>& rows);
  void set_sq_ranges(const float* vmin, const float* vmax);
  void release_raw();
  void materialize_raw();  // copies memory-mapped float32 vectors into raw_
  void encode(const float* v, uint8_t* out) const;
  void decode(node_t n, float* out) const;
  void prepare(const float* q, bool exact, Query& out) const;
  void prepare_node(node_t n, Query& out) const;  // node-to-node distances
  float qdist(const Query& q, node_t n) const {
    if (q.exact) return dist_(q.raw, full(n), dim_);
    if (metric_ == Metric::L2) return kernels::sq8_l2(q.t.data(), sq_w_.data(), code(n), dim_);
    return 1.f - q.bias - kernels::sq8_dot(q.t.data(), code(n), dim_);
  }

  void grow_to(size_t new_capacity);
  int random_level(label_t label) const;
  node_t place(label_t label);                        // first phase of add(), serial
  void store_vector(node_t cur, const float* vector);  // first phase of add(), parallel
  void link(node_t cur, const float* vector);          // second phase of add(), parallel

  template <bool kLock>
  void greedy_search(const Query& q, node_t& ep, float& ep_dist, int level) const;
  MaxHeap search_layer_build(const Query& q, node_t ep, int level) const;
  template <bool kFiltered, bool kLock>
  void search_layer_query(const Query& q, node_t ep, size_t ef, const LabelFilter* filter,
                          MaxHeap& top) const;

  std::vector<Candidate> select_neighbors(MaxHeap& candidates, size_t m) const;
  node_t connect(node_t cur, MaxHeap& candidates, int level);

  std::vector<SearchResult> search_unlocked(const float* query, size_t k, size_t ef,
                                            const LabelFilter* filter) const;
  std::vector<SearchResult> brute_force(const Query& q, size_t k, const LabelFilter& allow) const;
  std::vector<SearchResult> scan_all(const Query& q, size_t k, const LabelFilter* filter) const;
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
  bool sq_ = false;
  bool sq_trained_ = false;
  // raw_ is kept with rerank, and also without it while the quantizer ranges are provisional
  // (small index), so re-encoding after a range change is lossless.
  bool sq_keep_raw_ = false;
  std::vector<float> sq_min_, sq_max_, sq_scale_, sq_inv_scale_, sq_w_;  // per dimension; w = scale^2

  // Storage
  size_t capacity_ = 0;
  std::atomic<size_t> count_{0};  // nodes ever inserted (incl. tombstones)
  char* level0_ = nullptr;
  std::vector<float> raw_;  // float32 vectors of a quantized index, see sq_keep_raw_ (capacity x dim)
  std::unique_ptr<MappedFile> raw_map_;  // set when the float32 vectors are memory-mapped
  const float* raw_ptr_ = nullptr;       // raw_.data() or a pointer into raw_map_
  std::vector<int32_t> levels_;
  std::vector<std::unique_ptr<uint32_t[]>> upper_;
  std::vector<label_t> labels_;
  std::vector<uint8_t> deleted_;
  size_t num_deleted_ = 0;
  std::unique_ptr<SpinLock[]> node_locks_;
  LabelMap label_to_node_;

  // Graph entry point
  node_t entry_ = kInvalidNode;
  int max_level_ = -1;

  mutable std::shared_mutex rw_;  // readers: queries and add()'s linking phase; writer: see above
  mutable std::mutex write_mu_;   // serialises writers (and save) with each other
  mutable std::mutex entry_mu_;   // entry point, during parallel inserts
  std::atomic<bool> linking_{false};  // add() is linking nodes: readers must lock neighbour lists
  mutable VisitedPool visited_;
};

}  // namespace strata
