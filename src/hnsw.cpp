#include "strata/hnsw.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <limits>
#include <new>

#include "strata/io.h"

namespace strata {

namespace {

constexpr char kMagic[8] = {'S', 'T', 'R', 'A', 'T', 'A', 'I', 'X'};
constexpr uint32_t kFormatVersion = 1;       // float32 index
constexpr uint32_t kFormatVersionQuant = 2;  // adds quantization parameters (and codes)
constexpr int kMaxLevel = 32;
// SQ8 ranges keep widening to cover new batches until the index holds this many vectors, so
// small first batches (e.g. one-by-one upserts) do not freeze a degenerate range. After that
// the ranges are fixed and out-of-range values are clamped.
constexpr size_t kSqFreezeAt = 1000;
constexpr std::align_val_t kAlign{64};

size_t round_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ull;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

inline void prefetch(const void* p) { __builtin_prefetch(p, 0, 3); }

}  // namespace

const char* quantization_name(Quantization q) {
  switch (q) {
    case Quantization::None:
      return "none";
    case Quantization::SQ8:
      return "sq8";
  }
  return "unknown";
}

Quantization parse_quantization(const std::string& name) {
  std::string s = name;
  std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
  if (s.empty() || s == "none" || s == "f32" || s == "float32") return Quantization::None;
  if (s == "sq8" || s == "int8" || s == "uint8") return Quantization::SQ8;
  throw Error("unknown quantization '" + name + "' (expected none or sq8)");
}

HNSWIndex::HNSWIndex(size_t dim, Metric metric, HNSWParams params, size_t initial_capacity)
    : dim_(dim), metric_(metric), params_(params) {
  if (dim_ == 0) throw Error("dim must be > 0");
  if (params_.M < 2 || params_.M > 1024) throw Error("M must be in [2, 1024]");
  if (params_.ef_construction == 0) throw Error("ef_construction must be > 0");
  maxM_ = params_.M;
  maxM0_ = 2 * params_.M;
  level_mult_ = 1.0 / std::log(static_cast<double>(params_.M));
  dist_ = distance_function(metric_);
  if (params_.quantization != Quantization::None && params_.quantization != Quantization::SQ8) {
    throw Error("unknown quantization");
  }
  sq_ = params_.quantization == Quantization::SQ8;
  sq_keep_raw_ = sq_;
  data_offset_ = round_up(sizeof(uint32_t) * (1 + maxM0_), 16);
  size_per_element_ = round_up(data_offset_ + (sq_ ? sizeof(uint8_t) : sizeof(float)) * dim_, 16);
  grow_to(std::max<size_t>(initial_capacity, 16));
}

HNSWIndex::~HNSWIndex() {  // out of line: MappedFile is incomplete in the header
  if (level0_) ::operator delete(level0_, kAlign);
}

void HNSWIndex::grow_to(size_t new_capacity) {
  if (new_capacity <= capacity_) return;
  auto* block = static_cast<char*>(::operator new(new_capacity * size_per_element_, kAlign));
  if (level0_) {
    std::memcpy(block, level0_, count_.load() * size_per_element_);
    ::operator delete(level0_, kAlign);
  }
  level0_ = block;
  if (sq_keep_raw_) {
    materialize_raw();
    raw_.resize(new_capacity * dim_);
    raw_ptr_ = raw_.data();
  }
  levels_.resize(new_capacity, 0);
  upper_.resize(new_capacity);
  labels_.resize(new_capacity, 0);
  deleted_.resize(new_capacity, 0);
  node_locks_ = std::make_unique<SpinLock[]>(new_capacity);
  capacity_ = new_capacity;
  visited_.resize(new_capacity);
}

// Level ~ floor(-ln(U) * mL) with mL = 1/ln(M): each level holds ~1/M of the level below.
// U is derived from a hash of (seed, label) instead of a shared RNG, so it is thread-safe
// and the same data always produces the same level assignment.
int HNSWIndex::random_level(label_t label) const {
  const uint64_t h = splitmix64(params_.seed ^ splitmix64(label));
  const double u = (static_cast<double>(h >> 11) + 0.5) * (1.0 / 9007199254740992.0);  // (0, 1)
  const int level = static_cast<int>(-std::log(u) * level_mult_);
  return std::min(level, kMaxLevel);
}

// ---------------------------------------------------------------------------------------------
// Scalar quantization
// ---------------------------------------------------------------------------------------------

// Per-dimension min/max over the training rows (united with the current ranges while the index
// is small); each dimension's range is split into 255 steps. If the ranges change, existing
// codes are re-encoded: exactly from the float32 vectors when kept, else from their decoding.
void HNSWIndex::train_sq(const float* vectors, const std::vector<size_t>& rows) {
  std::vector<float> vmin(dim_, std::numeric_limits<float>::max());
  std::vector<float> vmax(dim_, std::numeric_limits<float>::lowest());
  if (sq_trained_) {
    vmin = sq_min_;
    vmax = sq_max_;
  }
  const std::vector<float> old_min = vmin, old_max = vmax;
  std::vector<float> tmp(dim_);
  for (const size_t r : rows) {
    const float* v = vectors + r * dim_;
    if (metric_ == Metric::Cosine) {
      std::memcpy(tmp.data(), v, dim_ * sizeof(float));
      normalize_inplace(tmp.data(), dim_);
      v = tmp.data();
    }
    for (size_t d = 0; d < dim_; ++d) {
      vmin[d] = std::min(vmin[d], v[d]);
      vmax[d] = std::max(vmax[d], v[d]);
    }
  }
  if (sq_trained_ && vmin == old_min && vmax == old_max) return;
  const size_t n = count_.load();
  std::vector<float> decoded;
  if (!has_full() && n > 0) {
    decoded.resize(n * dim_);
    for (node_t i = 0; i < n; ++i) decode(i, decoded.data() + static_cast<size_t>(i) * dim_);
  }
  set_sq_ranges(vmin.data(), vmax.data());
  for (node_t i = 0; i < n; ++i) {
    const float* v = has_full() ? full(i) : decoded.data() + static_cast<size_t>(i) * dim_;
    encode(v, reinterpret_cast<uint8_t*>(const_cast<char*>(data(i))));
  }
}

void HNSWIndex::set_sq_ranges(const float* vmin, const float* vmax) {
  sq_min_.assign(vmin, vmin + dim_);
  sq_max_.assign(vmax, vmax + dim_);
  sq_scale_.resize(dim_);
  sq_inv_scale_.resize(dim_);
  sq_w_.resize(dim_);
  for (size_t d = 0; d < dim_; ++d) {
    const float range = vmax[d] - vmin[d];
    const float scale = range > 0.f && std::isfinite(range) ? range / 255.f : 1.f;
    sq_scale_[d] = scale;
    sq_inv_scale_[d] = 1.f / scale;
    sq_w_[d] = scale * scale;
  }
  sq_trained_ = true;
}

void HNSWIndex::encode(const float* v, uint8_t* out) const {
  for (size_t d = 0; d < dim_; ++d) {
    const float x = std::nearbyint((v[d] - sq_min_[d]) * sq_inv_scale_[d]);
    out[d] = static_cast<uint8_t>(std::min(255.f, std::max(0.f, x)));
  }
}

void HNSWIndex::decode(node_t n, float* out) const {
  const uint8_t* c = code(n);
  for (size_t d = 0; d < dim_; ++d) out[d] = sq_min_[d] + sq_scale_[d] * static_cast<float>(c[d]);
}

// For codes x = min + scale * c:
//   L2:     |q - x|^2   = sum_d scale_d^2 * ((q_d - min_d) / scale_d - c_d)^2
//   IP/cos: 1 - <q, x>  = 1 - <q, min> - sum_d (q_d * scale_d) * c_d
void HNSWIndex::prepare(const float* q, bool exact, Query& out) const {
  out.raw = q;
  out.exact = exact || !sq_;
  if (out.exact) return;
  out.t.resize(dim_);
  if (metric_ == Metric::L2) {
    for (size_t d = 0; d < dim_; ++d) out.t[d] = (q[d] - sq_min_[d]) * sq_inv_scale_[d];
    out.bias = 0.f;
  } else {
    for (size_t d = 0; d < dim_; ++d) out.t[d] = q[d] * sq_scale_[d];
    out.bias = kernels::dot(q, sq_min_.data(), dim_);
  }
}

void HNSWIndex::prepare_node(node_t n, Query& out) const {
  if (has_full()) {
    prepare(full(n), true, out);
    return;
  }
  out.decoded.resize(dim_);
  decode(n, out.decoded.data());
  prepare(out.decoded.data(), false, out);
}

// ---------------------------------------------------------------------------------------------
// Insertion (paper Alg. 1)
// ---------------------------------------------------------------------------------------------

void HNSWIndex::add(const float* vectors, const label_t* labels, size_t n, int num_threads) {
  if (n == 0) return;
  for (size_t i = 0; i < n; ++i) {
    if (labels[i] == kNoLabel) throw Error("label 2^64-1 is reserved");
  }
  std::unique_lock<std::shared_mutex> write_lock(rw_);
  materialize_raw();

  // Last occurrence of a label in the batch wins.
  std::vector<size_t> order;
  order.reserve(n);
  if (n == 1) {
    order.push_back(0);
  } else {
    std::unordered_map<label_t, size_t> last;
    last.reserve(n * 2);
    for (size_t i = 0; i < n; ++i) last[labels[i]] = i;
    for (size_t i = 0; i < n; ++i) {
      if (last[labels[i]] == i) order.push_back(i);
    }
  }

  // Upsert: the previous version of a label becomes a tombstone. It stays in the graph as a
  // waypoint (removing it would disconnect neighbours) but is never returned.
  for (size_t i : order) {
    auto it = label_to_node_.find(labels[i]);
    if (it != label_to_node_.end()) {
      deleted_[it->second] = 1;
      ++num_deleted_;
      label_to_node_.erase(it);
    }
  }

  if (sq_ && (!sq_trained_ || count_.load() < kSqFreezeAt)) train_sq(vectors, order);

  const size_t needed = count_.load() + order.size();
  if (needed > static_cast<size_t>(std::numeric_limits<node_t>::max()) - 1) throw Error("index is full");
  if (needed > capacity_) grow_to(std::max(needed, capacity_ * 2));
  label_to_node_.reserve(label_to_node_.size() + order.size());

  parallel_for(order.size(), num_threads,
               [&](size_t j) { insert_one(vectors + order[j] * dim_, labels[order[j]]); });
  if (sq_keep_raw_ && !params_.rerank && count_.load() >= kSqFreezeAt) release_raw();
}

// Quantizer ranges are now fixed: without rerank the float32 vectors are no longer needed.
void HNSWIndex::release_raw() {
  sq_keep_raw_ = false;
  raw_.clear();
  raw_.shrink_to_fit();
  raw_map_.reset();
  raw_ptr_ = nullptr;
}

void HNSWIndex::materialize_raw() {
  if (!raw_map_) return;
  std::vector<float> owned(capacity_ * dim_);
  std::memcpy(owned.data(), raw_ptr_, count_.load() * dim_ * sizeof(float));
  raw_.swap(owned);
  raw_map_.reset();
  raw_ptr_ = raw_.data();
}

void HNSWIndex::insert_one(const float* vector, label_t label) {
  const auto cur = static_cast<node_t>(count_.fetch_add(1, std::memory_order_relaxed));
  labels_[cur] = label;
  deleted_[cur] = 0;
  // The full-precision (normalised for cosine) vector: stored in the graph block, in raw_, or
  // only in a scratch buffer when just the codes are kept.
  std::vector<float> scratch;
  float* dst;
  if (!sq_) {
    dst = mutable_vec(cur);
  } else if (sq_keep_raw_) {
    dst = raw_.data() + static_cast<size_t>(cur) * dim_;
  } else {
    scratch.resize(dim_);
    dst = scratch.data();
  }
  std::memcpy(dst, vector, dim_ * sizeof(float));
  if (metric_ == Metric::Cosine) normalize_inplace(dst, dim_);
  if (sq_) encode(dst, reinterpret_cast<uint8_t*>(const_cast<char*>(data(cur))));

  const int level = random_level(label);
  levels_[cur] = level;
  links0(cur)[0] = 0;
  if (level > 0) upper_[cur].reset(new uint32_t[static_cast<size_t>(level) * (maxM_ + 1)]());
  {
    std::lock_guard<std::mutex> g(label_mu_);
    label_to_node_[label] = cur;
  }

  // Hold our own node lock for the whole insertion: other threads that discover `cur` through
  // a reverse link will wait until its own neighbour list is complete.
  std::lock_guard<SpinLock> node_guard(node_locks_[cur]);

  // If this node will become the new top of the hierarchy, keep the entry lock until the
  // end so nobody else updates the entry point concurrently.
  std::unique_lock<std::mutex> entry_lock(entry_mu_);
  const int top_level = max_level_;
  node_t ep = entry_;
  if (level <= top_level) entry_lock.unlock();

  if (ep == kInvalidNode) {  // first node
    entry_ = cur;
    max_level_ = level;
    return;
  }

  // Build distances use float32 vectors whenever they are kept, so a quantized index with
  // rerank has the same graph as a float32 one.
  Query q;
  prepare(dst, has_full(), q);
  float ep_dist = qdist(q, ep);
  // Phase 1: greedy descent through the levels above the new node's level.
  for (int l = top_level; l > level; --l) greedy_search<true>(q, ep, ep_dist, l);
  // Phase 2: on each level the node lives on, find ef_construction candidates and link.
  for (int l = std::min(level, top_level); l >= 0; --l) {
    MaxHeap candidates = search_layer_build(q, ep, l);
    ep = connect(cur, candidates, l);
  }

  if (level > top_level) {
    entry_ = cur;
    max_level_ = level;
  }
}

template <bool kLock>
void HNSWIndex::greedy_search(const Query& q, node_t& ep, float& ep_dist, int level) const {
  std::vector<uint32_t> nbrs;
  bool changed = true;
  while (changed) {
    changed = false;
    const uint32_t* list;
    uint32_t count;
    if constexpr (kLock) {
      std::lock_guard<SpinLock> g(node_locks_[ep]);
      const uint32_t* l = links(ep, level);
      nbrs.assign(l + 1, l + 1 + l[0]);
      list = nbrs.data();
      count = static_cast<uint32_t>(nbrs.size());
    } else {
      const uint32_t* l = links(ep, level);
      list = l + 1;
      count = l[0];
    }
    for (uint32_t j = 0; j < count; ++j) {
      const node_t cand = list[j];
      const float d = qdist(q, cand);
      if (d < ep_dist) {
        ep_dist = d;
        ep = cand;
        changed = true;
      }
    }
  }
}

// Beam search on one level during construction (paper Alg. 2). Neighbour lists are copied
// under the node's spinlock because other threads may be rewriting them.
HNSWIndex::MaxHeap HNSWIndex::search_layer_build(const Query& q, node_t ep, int level) const {
  VisitedHandle handle(visited_);
  VisitedList& visited = *handle;
  const size_t ef = params_.ef_construction;

  MaxHeap top;
  MinHeap frontier;
  const float d0 = qdist(q, ep);
  top.emplace(d0, ep);
  frontier.emplace(d0, ep);
  visited.visit(ep);
  float worst = d0;

  std::vector<uint32_t> nbrs;
  nbrs.reserve(maxM0_);
  while (!frontier.empty()) {
    const auto [cd, c] = frontier.top();
    if (cd > worst && top.size() >= ef) break;
    frontier.pop();
    {
      std::lock_guard<SpinLock> g(node_locks_[c]);
      const uint32_t* l = links(c, level);
      nbrs.assign(l + 1, l + 1 + l[0]);
    }
    for (const node_t nb : nbrs) {
      if (!visited.visit(nb)) continue;
      const float d = qdist(q, nb);
      if (top.size() < ef || d < worst) {
        frontier.emplace(d, nb);
        top.emplace(d, nb);
        if (top.size() > ef) top.pop();
        worst = top.top().first;
      }
    }
  }
  return top;
}

// Neighbour selection (paper Alg. 4, without extendCandidates/keepPruned, as in hnswlib).
// Walk candidates from closest to farthest and keep c only if it is closer to the base
// point than to every neighbour kept so far. This spreads edges in different directions,
// which keeps clustered data navigable. Returns the kept candidates, closest first.
std::vector<HNSWIndex::Candidate> HNSWIndex::select_neighbors(MaxHeap& candidates, size_t m) const {
  std::vector<Candidate> sorted;
  sorted.reserve(candidates.size());
  while (!candidates.empty()) {
    sorted.push_back(candidates.top());
    candidates.pop();
  }
  std::reverse(sorted.begin(), sorted.end());
  if (sorted.size() < m) return sorted;
  if (!params_.use_heuristic) {
    sorted.resize(m);
    return sorted;
  }
  std::vector<Candidate> kept;
  kept.reserve(m);
  Query cq;
  for (const Candidate& c : sorted) {
    if (kept.size() >= m) break;
    if (!kept.empty()) prepare_node(c.second, cq);
    bool good = true;
    for (const Candidate& s : kept) {
      if (qdist(cq, s.second) < c.first) {
        good = false;
        break;
      }
    }
    if (good) kept.push_back(c);
  }
  return kept;
}

// Links `cur` to its selected neighbours and adds the reverse edges. A neighbour whose list
// is full re-runs neighbour selection over (old neighbours + cur) to stay within its budget.
node_t HNSWIndex::connect(node_t cur, MaxHeap& candidates, int level) {
  const size_t max_links = level == 0 ? maxM0_ : maxM_;
  const std::vector<Candidate> selected = select_neighbors(candidates, params_.M);

  uint32_t* own = links(cur, level);  // caller holds cur's lock
  own[0] = static_cast<uint32_t>(selected.size());
  for (size_t i = 0; i < selected.size(); ++i) own[1 + i] = selected[i].second;

  for (const auto& [d, nb] : selected) {
    std::lock_guard<SpinLock> g(node_locks_[nb]);
    uint32_t* l = links(nb, level);
    const uint32_t count = l[0];
    if (count < max_links) {
      l[1 + count] = cur;
      l[0] = count + 1;
      continue;
    }
    MaxHeap pool;
    pool.emplace(d, cur);
    Query nq;
    prepare_node(nb, nq);
    for (uint32_t j = 1; j <= count; ++j) pool.emplace(qdist(nq, l[j]), l[j]);
    const std::vector<Candidate> kept = select_neighbors(pool, max_links);
    l[0] = static_cast<uint32_t>(kept.size());
    for (size_t i = 0; i < kept.size(); ++i) l[1 + i] = kept[i].second;
  }
  return selected.front().second;
}

// ---------------------------------------------------------------------------------------------
// Search (paper Alg. 5)
// ---------------------------------------------------------------------------------------------

// Query-time beam search on level 0. Runs under the shared lock, so the graph is immutable
// and no per-node locking is needed. Tombstones and filtered-out nodes are traversed but
// never enter `top`; in that case we keep exploring until `top` holds ef valid results.
template <bool kFiltered>
void HNSWIndex::search_layer_query(const Query& q, node_t ep, size_t ef, const LabelFilter* filter,
                                   MaxHeap& top) const {
  VisitedHandle handle(visited_);
  VisitedList& visited = *handle;
  const bool must_fill = kFiltered || num_deleted_ > 0;

  MinHeap frontier;
  const float d0 = qdist(q, ep);
  float worst = std::numeric_limits<float>::max();
  if (is_result(ep, filter)) {
    top.emplace(d0, ep);
    worst = d0;
  }
  frontier.emplace(d0, ep);
  visited.visit(ep);

  while (!frontier.empty()) {
    const auto [cd, c] = frontier.top();
    if (cd > worst && (top.size() >= ef || !must_fill)) break;
    frontier.pop();

    const uint32_t* l = links0(c);
    const uint32_t count = l[0];
    if (count > 0) prefetch(data(l[1]));
    for (uint32_t j = 1; j <= count; ++j) {
      const node_t nb = l[j];
      if (j < count) prefetch(data(l[j + 1]));  // hide the cache miss of the next vector
      if (!visited.visit(nb)) continue;
      const float d = qdist(q, nb);
      if (top.size() < ef || d < worst) {
        frontier.emplace(d, nb);
        if (is_result(nb, filter)) {
          top.emplace(d, nb);
          if (top.size() > ef) top.pop();
        }
        if (!top.empty()) worst = top.top().first;
      }
    }
  }
}

std::vector<SearchResult> HNSWIndex::search(const float* query, size_t k, size_t ef,
                                            const LabelFilter* filter) const {
  std::shared_lock<std::shared_mutex> read_lock(rw_);
  return search_unlocked(query, k, ef, filter);
}

std::vector<SearchResult> HNSWIndex::search_unlocked(const float* query, size_t k, size_t ef,
                                                     const LabelFilter* filter) const {
  std::vector<SearchResult> out;
  if (k == 0 || entry_ == kInvalidNode || count_.load() == num_deleted_) return out;

  std::vector<float> normalized;
  const float* q = query;
  if (metric_ == Metric::Cosine) {
    normalized.assign(query, query + dim_);
    normalize_inplace(normalized.data(), dim_);
    q = normalized.data();
  }

  if (filter && filter->mode() == LabelFilter::Mode::Allow && filter->size() <= flat_cutoff_.load()) {
    Query bq;
    prepare(q, has_full(), bq);
    return brute_force(bq, k, *filter);
  }

  ef = std::max(ef ? ef : ef_search_.load(), k);
  Query pq;
  prepare(q, false, pq);  // SQ8 codes when quantized
  node_t ep = entry_;
  float ep_dist = qdist(pq, ep);
  for (int l = max_level_; l > 0; --l) greedy_search<false>(pq, ep, ep_dist, l);

  MaxHeap top;
  if (filter) {
    search_layer_query<true>(pq, ep, ef, filter, top);
  } else {
    search_layer_query<false>(pq, ep, ef, nullptr, top);
  }
  if (sq_ && params_.rerank) {
    // Re-rank the ef candidates with exact distances on the float32 vectors.
    MaxHeap exact;
    while (!top.empty()) {
      const node_t n = top.top().second;
      top.pop();
      exact.emplace(dist_(q, full(n), dim_), n);
      if (exact.size() > k) exact.pop();
    }
    top.swap(exact);
  }
  while (top.size() > k) top.pop();
  out.resize(top.size());
  for (size_t i = top.size(); i-- > 0;) {
    out[i] = {labels_[top.top().second], top.top().first};
    top.pop();
  }
  return out;
}

std::vector<SearchResult> HNSWIndex::brute_force(const Query& q, size_t k, const LabelFilter& allow) const {
  std::priority_queue<std::pair<float, label_t>> top;
  for (const label_t label : allow.labels()) {
    auto it = label_to_node_.find(label);
    if (it == label_to_node_.end()) continue;
    const float d = qdist(q, it->second);
    if (top.size() < k) {
      top.emplace(d, label);
    } else if (d < top.top().first) {
      top.pop();
      top.emplace(d, label);
    }
  }
  std::vector<SearchResult> out(top.size());
  for (size_t i = top.size(); i-- > 0;) {
    out[i] = {top.top().second, top.top().first};
    top.pop();
  }
  return out;
}

void HNSWIndex::search_batch(const float* queries, size_t nq, size_t k, size_t ef, int num_threads,
                             const LabelFilter* filter, label_t* out_labels, float* out_distances) const {
  std::shared_lock<std::shared_mutex> read_lock(rw_);
  parallel_for(nq, num_threads, [&](size_t i) {
    const auto res = search_unlocked(queries + i * dim_, k, ef, filter);
    for (size_t j = 0; j < k; ++j) {
      const bool has = j < res.size();
      out_labels[i * k + j] = has ? res[j].label : kNoLabel;
      out_distances[i * k + j] = has ? res[j].distance : std::numeric_limits<float>::infinity();
    }
  });
}

// ---------------------------------------------------------------------------------------------
// Deletes, accessors, compaction
// ---------------------------------------------------------------------------------------------

bool HNSWIndex::remove(label_t label) {
  std::unique_lock<std::shared_mutex> write_lock(rw_);
  auto it = label_to_node_.find(label);
  if (it == label_to_node_.end()) return false;
  deleted_[it->second] = 1;
  ++num_deleted_;
  label_to_node_.erase(it);
  return true;
}

bool HNSWIndex::contains(label_t label) const {
  std::shared_lock<std::shared_mutex> read_lock(rw_);
  return label_to_node_.count(label) != 0;
}

bool HNSWIndex::get_vector(label_t label, float* out) const {
  std::shared_lock<std::shared_mutex> read_lock(rw_);
  auto it = label_to_node_.find(label);
  if (it == label_to_node_.end()) return false;
  if (has_full()) {
    std::memcpy(out, full(it->second), dim_ * sizeof(float));
  } else {
    decode(it->second, out);
  }
  return true;
}

std::vector<label_t> HNSWIndex::labels() const {
  std::shared_lock<std::shared_mutex> read_lock(rw_);
  std::vector<label_t> out;
  out.reserve(label_to_node_.size());
  for (const auto& kv : label_to_node_) out.push_back(kv.first);
  std::sort(out.begin(), out.end());
  return out;
}

size_t HNSWIndex::size() const {
  std::shared_lock<std::shared_mutex> read_lock(rw_);
  return count_.load() - num_deleted_;
}

HNSWStats HNSWIndex::stats() const {
  std::shared_lock<std::shared_mutex> read_lock(rw_);
  HNSWStats s;
  const size_t n = count_.load();
  s.nodes = n;
  s.deleted = num_deleted_;
  s.size = n - num_deleted_;
  s.capacity = capacity_;
  s.max_level = max_level_;
  size_t degree_sum = 0, upper_bytes = 0;
  for (node_t i = 0; i < n; ++i) {
    if (!deleted_[i]) degree_sum += links0(i)[0];
    if (levels_[i] > 0) upper_bytes += static_cast<size_t>(levels_[i]) * (maxM_ + 1) * sizeof(uint32_t);
  }
  s.mean_degree_level0 = s.size ? static_cast<double>(degree_sum) / static_cast<double>(s.size) : 0.0;
  const size_t per_node_overhead = sizeof(int32_t) + sizeof(label_t) + sizeof(uint8_t) + sizeof(SpinLock) +
                                   sizeof(std::unique_ptr<uint32_t[]>);
  s.memory_bytes = capacity_ * (size_per_element_ + per_node_overhead) + upper_bytes +
                   raw_.size() * sizeof(float) + sq_min_.size() * 5 * sizeof(float) +
                   label_to_node_.size() * (sizeof(label_t) + sizeof(node_t) + 2 * sizeof(void*));
  return s;
}

void HNSWIndex::compact(int num_threads) {
  std::unique_lock<std::shared_mutex> write_lock(rw_);
  if (num_deleted_ == 0) return;
  const size_t n = count_.load();
  std::vector<label_t> live_labels;
  std::vector<float> live_vectors;
  live_labels.reserve(n - num_deleted_);
  live_vectors.reserve((n - num_deleted_) * dim_);
  std::vector<float> tmp(dim_);
  for (node_t i = 0; i < n; ++i) {
    if (deleted_[i]) continue;
    live_labels.push_back(labels_[i]);
    const float* v = full(i);
    if (v == nullptr) {
      decode(i, tmp.data());
      v = tmp.data();
    }
    live_vectors.insert(live_vectors.end(), v, v + dim_);
  }
  HNSWIndex fresh(dim_, metric_, params_, std::max<size_t>(live_labels.size(), 16));
  if (sq_trained_) {  // keep the trained quantizer: re-encoding decoded codes is then lossless
    fresh.set_sq_ranges(sq_min_.data(), sq_max_.data());
  }
  if (!live_labels.empty()) fresh.add(live_vectors.data(), live_labels.data(), live_labels.size(), num_threads);
  swap_storage(fresh);
}

void HNSWIndex::swap_storage(HNSWIndex& other) {
  std::swap(level0_, other.level0_);
  raw_.swap(other.raw_);
  raw_map_.swap(other.raw_map_);
  std::swap(raw_ptr_, other.raw_ptr_);
  std::swap(sq_keep_raw_, other.sq_keep_raw_);
  std::swap(capacity_, other.capacity_);
  const size_t c = count_.load();
  count_.store(other.count_.load());
  other.count_.store(c);
  levels_.swap(other.levels_);
  upper_.swap(other.upper_);
  labels_.swap(other.labels_);
  deleted_.swap(other.deleted_);
  std::swap(num_deleted_, other.num_deleted_);
  node_locks_.swap(other.node_locks_);
  label_to_node_.swap(other.label_to_node_);
  std::swap(entry_, other.entry_);
  std::swap(max_level_, other.max_level_);
  visited_.resize(capacity_);
}

// ---------------------------------------------------------------------------------------------
// Persistence
// ---------------------------------------------------------------------------------------------

void HNSWIndex::save(const std::string& path) const {
  std::shared_lock<std::shared_mutex> read_lock(rw_);
  const std::string tmp = path + ".tmp";
  {
    FileWriter w(tmp);
    w.write(kMagic, sizeof(kMagic));
    w.put<uint32_t>(sq_ ? kFormatVersionQuant : kFormatVersion);
    w.put<uint32_t>(static_cast<uint32_t>(dim_));
    w.put<uint32_t>(static_cast<uint32_t>(metric_));
    w.put<uint64_t>(params_.M);
    w.put<uint64_t>(params_.ef_construction);
    w.put<uint64_t>(params_.seed);
    w.put<uint8_t>(params_.use_heuristic ? 1 : 0);
    if (sq_) {
      w.put<uint32_t>(static_cast<uint32_t>(params_.quantization));
      w.put<uint8_t>(params_.rerank ? 1 : 0);
      w.put<uint8_t>(sq_trained_ ? 1 : 0);
      w.put<uint8_t>(sq_keep_raw_ ? 1 : 0);
      if (sq_trained_) {
        w.write(sq_min_.data(), dim_ * sizeof(float));
        w.write(sq_max_.data(), dim_ * sizeof(float));
      }
    }
    w.put<uint64_t>(ef_search_.load());
    w.put<uint64_t>(flat_cutoff_.load());
    const uint64_t n = count_.load();
    w.put<uint64_t>(n);
    w.put<uint64_t>(num_deleted_);
    w.put<uint32_t>(entry_);
    w.put<int32_t>(max_level_);
    w.put<uint64_t>(size_per_element_);
    w.write(level0_, n * size_per_element_);
    w.write(levels_.data(), n * sizeof(int32_t));
    w.write(labels_.data(), n * sizeof(label_t));
    w.write(deleted_.data(), n);
    for (node_t i = 0; i < n; ++i) {
      if (levels_[i] > 0) {
        w.write(upper_[i].get(), static_cast<size_t>(levels_[i]) * (maxM_ + 1) * sizeof(uint32_t));
      }
    }
    if (sq_keep_raw_) {  // last, page-friendly aligned, so load() can memory-map it
      w.pad_to(64);
      w.write(raw_ptr_, n * dim_ * sizeof(float));
    }
    w.write_crc();
    w.sync_and_close();
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) throw Error("rename " + tmp + " -> " + path + ": " + ec.message());
  const auto parent = std::filesystem::path(path).parent_path();
  fsync_dir(parent.empty() ? "." : parent.string());
}

std::unique_ptr<HNSWIndex> HNSWIndex::load(const std::string& path, bool mmap_vectors) {
  FileReader r(path);
  char magic[sizeof(kMagic)];
  r.read(magic, sizeof(magic));
  if (std::memcmp(magic, kMagic, sizeof(kMagic)) != 0) throw Error(path + " is not a Strata index file");
  const auto version = r.get<uint32_t>();
  if (version != kFormatVersion && version != kFormatVersionQuant) {
    throw Error("unsupported index format version " + std::to_string(version));
  }
  const auto dim = r.get<uint32_t>();
  const auto metric = static_cast<Metric>(r.get<uint32_t>());
  HNSWParams params;
  params.M = r.get<uint64_t>();
  params.ef_construction = r.get<uint64_t>();
  params.seed = r.get<uint64_t>();
  params.use_heuristic = r.get<uint8_t>() != 0;
  std::vector<float> sq_min, sq_max;
  bool keep_raw = false;
  if (version == kFormatVersionQuant) {
    params.quantization = static_cast<Quantization>(r.get<uint32_t>());
    if (params.quantization != Quantization::SQ8) throw Error("corrupted index: bad quantization");
    params.rerank = r.get<uint8_t>() != 0;
    const bool trained = r.get<uint8_t>() != 0;
    keep_raw = r.get<uint8_t>() != 0;
    if (trained) {
      sq_min.resize(dim);
      sq_max.resize(dim);
      r.read(sq_min.data(), dim * sizeof(float));
      r.read(sq_max.data(), dim * sizeof(float));
    }
  }
  const auto ef_search = r.get<uint64_t>();
  const auto flat_cutoff = r.get<uint64_t>();
  const auto n = r.get<uint64_t>();
  const auto num_deleted = r.get<uint64_t>();
  const auto entry = r.get<uint32_t>();
  const auto max_level = r.get<int32_t>();
  const auto spe = r.get<uint64_t>();

  auto index = std::make_unique<HNSWIndex>(dim, metric, params, std::max<size_t>(n, 16));
  if (spe != index->size_per_element_) throw Error("index file layout mismatch");
  index->set_ef_search(ef_search);
  index->set_flat_search_cutoff(flat_cutoff);
  if (index->sq_ && !keep_raw) index->release_raw();
  if (!sq_min.empty()) index->set_sq_ranges(sq_min.data(), sq_max.data());
  r.read(index->level0_, n * spe);
  r.read(index->levels_.data(), n * sizeof(int32_t));
  r.read(index->labels_.data(), n * sizeof(label_t));
  r.read(index->deleted_.data(), n);
  for (node_t i = 0; i < n; ++i) {
    const int32_t lvl = index->levels_[i];
    if (lvl < 0 || lvl > kMaxLevel) throw Error("corrupted index: bad level");
    if (lvl > 0) {
      const size_t words = static_cast<size_t>(lvl) * (index->maxM_ + 1);
      index->upper_[i].reset(new uint32_t[words]);
      r.read(index->upper_[i].get(), words * sizeof(uint32_t));
    }
  }
  if (index->sq_keep_raw_) {
    r.skip_pad(64);
    const size_t bytes = n * dim * sizeof(float);
    if (mmap_vectors && bytes > 0) {
      auto map = std::make_unique<MappedFile>(path);
      if (map->size() < r.position() + bytes + sizeof(uint32_t)) throw Error("unexpected end of file: " + path);
      const char* p = map->data() + r.position();
      r.absorb(p, bytes);
      index->raw_.clear();
      index->raw_.shrink_to_fit();
      index->raw_ptr_ = reinterpret_cast<const float*>(p);
      index->raw_map_ = std::move(map);
    } else {
      r.read(index->raw_.data(), bytes);
    }
  }
  r.verify_crc();
  if (index->raw_map_) index->raw_map_->advise_random_and_release();

  index->count_.store(n);
  index->num_deleted_ = num_deleted;
  index->entry_ = entry;
  index->max_level_ = max_level;
  index->label_to_node_.reserve(n - num_deleted);
  for (node_t i = 0; i < n; ++i) {
    if (!index->deleted_[i]) index->label_to_node_[index->labels_[i]] = i;
  }
  return index;
}

}  // namespace strata
