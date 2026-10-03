#include "strata/collection.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <nlohmann/json.hpp>

#include "strata/io.h"

namespace strata {

namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

// --- WAL payload encoding --------------------------------------------------------------------

class ByteWriter {
 public:
  template <class T>
  void put(const T& v) {
    const auto* p = reinterpret_cast<const char*>(&v);
    buf_.append(p, sizeof(T));
  }
  void raw(const void* p, size_t n) { buf_.append(static_cast<const char*>(p), n); }
  std::string take() { return std::move(buf_); }

 private:
  std::string buf_;
};

class ByteReader {
 public:
  explicit ByteReader(const std::string& s) : s_(s) {}
  template <class T>
  T get() {
    T v;
    raw(&v, sizeof(T));
    return v;
  }
  void raw(void* p, size_t n) {
    if (pos_ + n > s_.size()) throw Error("malformed WAL payload");
    std::memcpy(p, s_.data() + pos_, n);
    pos_ += n;
  }

 private:
  const std::string& s_;
  size_t pos_ = 0;
};

std::string encode_upsert(const label_t* ids, const float* vectors, size_t n, size_t dim,
                          const std::vector<std::string>& metadata) {
  ByteWriter w;
  w.put<uint64_t>(n);
  w.put<uint32_t>(static_cast<uint32_t>(dim));
  for (size_t i = 0; i < n; ++i) {
    w.put<uint64_t>(ids[i]);
    w.raw(vectors + i * dim, dim * sizeof(float));
    const std::string& md = metadata.empty() ? std::string() : metadata[i];
    w.put<uint32_t>(static_cast<uint32_t>(md.size()));
    w.raw(md.data(), md.size());
  }
  return w.take();
}

std::string encode_delete(const label_t* ids, size_t n) {
  ByteWriter w;
  w.put<uint64_t>(n);
  w.raw(ids, n * sizeof(label_t));
  return w.take();
}

std::string snapshot_name(uint64_t seq) {
  char buf[32];
  std::snprintf(buf, sizeof(buf), "snap-%020" PRIu64, seq);
  return buf;
}

json config_to_json(const CollectionConfig& c) {
  return {{"name", c.name},
          {"dim", c.dim},
          {"metric", metric_name(c.metric)},
          {"M", c.params.M},
          {"ef_construction", c.params.ef_construction},
          {"seed", c.params.seed},
          {"use_heuristic", c.params.use_heuristic},
          {"checkpoint_wal_bytes", c.checkpoint_wal_bytes}};
}

CollectionConfig config_from_json(const json& j) {
  CollectionConfig c;
  c.name = j.at("name").get<std::string>();
  c.dim = j.at("dim").get<size_t>();
  c.metric = parse_metric(j.at("metric").get<std::string>());
  c.params.M = j.at("M").get<size_t>();
  c.params.ef_construction = j.at("ef_construction").get<size_t>();
  c.params.seed = j.at("seed").get<uint64_t>();
  c.params.use_heuristic = j.value("use_heuristic", true);
  c.checkpoint_wal_bytes = j.value("checkpoint_wal_bytes", c.checkpoint_wal_bytes);
  return c;
}

}  // namespace

Collection::Collection(std::string dir, CollectionConfig cfg) : dir_(std::move(dir)), cfg_(std::move(cfg)) {}

Collection::~Collection() = default;

std::unique_ptr<Collection> Collection::create(const std::string& dir, const CollectionConfig& cfg) {
  if (cfg.dim == 0) throw Error("dim must be > 0");
  if (fs::exists(fs::path(dir) / "config.json")) throw Error("collection already exists at " + dir);
  fs::create_directories(dir);
  atomic_write_file((fs::path(dir) / "config.json").string(), config_to_json(cfg).dump(2));
  return open(dir, cfg.sync_wal);
}

std::unique_ptr<Collection> Collection::open(const std::string& dir, bool sync_wal) {
  const fs::path cfg_path = fs::path(dir) / "config.json";
  if (!fs::exists(cfg_path)) throw Error("no collection at " + dir);
  CollectionConfig cfg = config_from_json(json::parse(read_file(cfg_path.string())));
  cfg.sync_wal = sync_wal;
  std::unique_ptr<Collection> c(new Collection(dir, std::move(cfg)));
  c->recover();
  return c;
}

void Collection::recover() {
  const fs::path root(dir_);
  const fs::path current = root / "CURRENT";
  if (fs::exists(current)) {
    std::string name = read_file(current.string());
    while (!name.empty() && (name.back() == '\n' || name.back() == '\r')) name.pop_back();
    const fs::path snap = root / name;
    index_ = HNSWIndex::load((snap / "index.bin").string());
    FileReader r((snap / "meta.bin").string());
    meta_.load(r);
    r.verify_crc();
    snapshot_seq_ = std::stoull(name.substr(5));
  } else {
    index_ = std::make_unique<HNSWIndex>(cfg_.dim, cfg_.metric, cfg_.params);
  }
  seq_ = snapshot_seq_;
  wal_ = std::make_unique<WriteAheadLog>((root / "wal.log").string(), cfg_.sync_wal);
  wal_->replay([&](const WriteAheadLog::Record& rec) {
    if (rec.seq <= snapshot_seq_) return;  // already contained in the snapshot
    apply(rec);
    seq_ = rec.seq;
  });
}

void Collection::apply(const WriteAheadLog::Record& rec) {
  ByteReader r(rec.payload);
  const auto n = r.get<uint64_t>();
  if (rec.op == WriteAheadLog::Op::Upsert) {
    const auto dim = r.get<uint32_t>();
    if (dim != cfg_.dim) throw Error("WAL dimension mismatch");
    std::vector<label_t> ids(n);
    std::vector<float> vectors(n * dim);
    std::vector<std::string> metadata(n);
    for (uint64_t i = 0; i < n; ++i) {
      ids[i] = r.get<uint64_t>();
      r.raw(vectors.data() + i * dim, dim * sizeof(float));
      metadata[i].resize(r.get<uint32_t>());
      r.raw(metadata[i].data(), metadata[i].size());
    }
    index_->add(vectors.data(), ids.data(), n);
    for (uint64_t i = 0; i < n; ++i) meta_.put(ids[i], metadata[i]);
  } else if (rec.op == WriteAheadLog::Op::Delete) {
    for (uint64_t i = 0; i < n; ++i) {
      const auto id = r.get<uint64_t>();
      index_->remove(id);
      meta_.erase(id);
    }
  } else {
    throw Error("unknown WAL op");
  }
}

void Collection::upsert(const label_t* ids, const float* vectors, size_t n, const std::vector<std::string>& metadata) {
  if (n == 0) return;
  if (!metadata.empty() && metadata.size() != n) throw Error("metadata must have one entry per vector");
  // Validate before logging so a bad request never reaches the WAL.
  for (const auto& md : metadata) {
    if (md.empty()) continue;
    const json doc = json::parse(md, nullptr, false);
    if (doc.is_discarded() || !doc.is_object()) throw Error("metadata must be a JSON object");
  }
  for (size_t i = 0; i < n; ++i) {
    if (ids[i] == kNoLabel) throw Error("id 2^64-1 is reserved");
  }
  std::unique_lock<std::shared_mutex> lock(mu_);
  WriteAheadLog::Record rec{seq_ + 1, WriteAheadLog::Op::Upsert, encode_upsert(ids, vectors, n, cfg_.dim, metadata)};
  wal_->append(rec.seq, rec.op, rec.payload);
  apply(rec);
  seq_ = rec.seq;
  if (wal_->size_bytes() > cfg_.checkpoint_wal_bytes) checkpoint_locked();
}

size_t Collection::remove(const label_t* ids, size_t n) {
  if (n == 0) return 0;
  std::unique_lock<std::shared_mutex> lock(mu_);
  size_t existing = 0;
  for (size_t i = 0; i < n; ++i) existing += index_->contains(ids[i]) ? 1 : 0;
  WriteAheadLog::Record rec{seq_ + 1, WriteAheadLog::Op::Delete, encode_delete(ids, n)};
  wal_->append(rec.seq, rec.op, rec.payload);
  apply(rec);
  seq_ = rec.seq;
  if (wal_->size_bytes() > cfg_.checkpoint_wal_bytes) checkpoint_locked();
  return existing;
}

std::vector<std::vector<Hit>> Collection::query(const float* queries, size_t nq, size_t k, size_t ef,
                                                const std::string& filter_json) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  std::unique_ptr<LabelFilter> filter;
  if (!filter_json.empty()) {
    const json f = json::parse(filter_json, nullptr, false);
    if (f.is_discarded()) throw Error("filter is not valid JSON");
    if (!f.is_null() && !(f.is_object() && f.empty())) {
      filter = std::make_unique<LabelFilter>(meta_.evaluate(f), LabelFilter::Mode::Allow);  // already sorted
    }
  }
  std::vector<std::vector<Hit>> out(nq);
  if (filter && filter->size() == 0) return out;
  for (size_t i = 0; i < nq; ++i) {
    for (const SearchResult& r : index_->search(queries + i * cfg_.dim, k, ef, filter.get())) {
      const std::string* md = meta_.get(r.label);
      out[i].push_back({r.label, r.distance, md ? *md : std::string("{}")});
    }
  }
  return out;
}

bool Collection::get(label_t id, std::vector<float>* vector, std::string* metadata) const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  if (!index_->contains(id)) return false;
  if (vector) {
    vector->resize(cfg_.dim);
    index_->get_vector(id, vector->data());
  }
  if (metadata) {
    const std::string* md = meta_.get(id);
    *metadata = md ? *md : "{}";
  }
  return true;
}

void Collection::checkpoint() {
  std::unique_lock<std::shared_mutex> lock(mu_);
  checkpoint_locked();
}

// 1. write snap-<seq>.tmp/{index,meta}.bin and fsync them
// 2. rename the directory to snap-<seq>                (atomic)
// 3. point CURRENT at it via temp file + rename        (atomic, the commit point)
// 4. truncate the WAL and delete older snapshots
// A crash before step 3 leaves the previous snapshot + full WAL; after step 3 the WAL
// records are <= the snapshot seq and are skipped on replay.
void Collection::checkpoint_locked() {
  const fs::path root(dir_);
  const std::string name = snapshot_name(seq_);
  const fs::path tmp = root / (name + ".tmp");
  const fs::path snap = root / name;
  fs::remove_all(tmp);
  fs::create_directories(tmp);
  index_->save((tmp / "index.bin").string());
  {
    FileWriter w((tmp / "meta.bin").string());
    meta_.save(w);
    w.write_crc();
    w.sync_and_close();
  }
  fsync_dir(tmp.string());
  if (fs::exists(snap)) fs::remove_all(snap);
  fs::rename(tmp, snap);
  fsync_dir(root.string());
  atomic_write_file((root / "CURRENT").string(), name + "\n");
  wal_->truncate();
  snapshot_seq_ = seq_;
  for (const auto& entry : fs::directory_iterator(root)) {
    const std::string fname = entry.path().filename().string();
    if (entry.is_directory() && fname.rfind("snap-", 0) == 0 && fname != name) fs::remove_all(entry.path());
  }
}

size_t Collection::size() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  return index_->size();
}

void Collection::set_ef_search(size_t ef) { index_->set_ef_search(ef); }

std::string Collection::stats_json() const {
  std::shared_lock<std::shared_mutex> lock(mu_);
  const HNSWStats s = index_->stats();
  json j = config_to_json(cfg_);
  j["size"] = s.size;
  j["deleted"] = s.deleted;
  j["max_level"] = s.max_level;
  j["mean_degree_level0"] = s.mean_degree_level0;
  j["memory_bytes"] = s.memory_bytes;
  j["ef_search"] = index_->ef_search();
  j["last_seq"] = seq_;
  j["snapshot_seq"] = snapshot_seq_;
  j["wal_bytes"] = wal_->size_bytes();
  return j.dump();
}

}  // namespace strata
