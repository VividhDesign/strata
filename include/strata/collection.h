#pragma once

#include <atomic>
#include <memory>
#include <shared_mutex>
#include <string>
#include <vector>

#include "strata/hnsw.h"
#include "strata/metadata.h"
#include "strata/wal.h"

namespace strata {

struct CollectionConfig {
  std::string name;
  size_t dim = 0;
  Metric metric = Metric::Cosine;
  HNSWParams params;
  bool sync_wal = true;                         // fsync every WAL append
  uint64_t checkpoint_wal_bytes = 64ull << 20;  // auto-checkpoint once the WAL exceeds this
};

struct Hit {
  label_t id;
  float distance;
  std::string metadata;  // JSON text
};

// A durable collection: HNSW index + metadata + write-ahead log.
//
// On-disk layout of a collection directory:
//   config.json              immutable settings
//   CURRENT                  name of the latest complete snapshot directory
//   snap-<seq>/index.bin     HNSW snapshot (CRC-checked)
//   snap-<seq>/meta.bin      metadata snapshot (CRC-checked)
//   wal.log                  operations after the snapshot
//
// Writes go to the WAL first, then memory. Recovery loads the snapshot named by CURRENT and
// replays WAL records with seq > snapshot seq, so a crash at any point loses nothing that
// was acknowledged (with sync_wal = true).
class Collection {
 public:
  static std::unique_ptr<Collection> create(const std::string& dir, const CollectionConfig& cfg);
  static std::unique_ptr<Collection> open(const std::string& dir, bool sync_wal = true);
  ~Collection();

  // metadata: empty, or one JSON object string per vector.
  void upsert(const label_t* ids, const float* vectors, size_t n,
              const std::vector<std::string>& metadata = {});
  size_t remove(const label_t* ids, size_t n);

  // filter_json: "" for no filter, else a MetadataStore filter expression.
  std::vector<std::vector<Hit>> query(const float* queries, size_t nq, size_t k, size_t ef = 0,
                                      const std::string& filter_json = "") const;
  bool get(label_t id, std::vector<float>* vector, std::string* metadata) const;

  void checkpoint();

  size_t size() const;
  const CollectionConfig& config() const { return cfg_; }
  const std::string& dir() const { return dir_; }
  std::string stats_json() const;
  void set_ef_search(size_t ef);

 private:
  Collection(std::string dir, CollectionConfig cfg);
  void recover();
  void apply(const WriteAheadLog::Record& rec);
  void checkpoint_locked();

  std::string dir_;
  CollectionConfig cfg_;
  std::unique_ptr<HNSWIndex> index_;
  MetadataStore meta_;
  std::unique_ptr<WriteAheadLog> wal_;
  // Writers (upsert, remove, checkpoint) are serialised by write_mu_ and hold mu_ exclusively
  // only to update the metadata store and to delete. Vector inserts run under the index's own
  // concurrency control, so queries are not blocked while a batch is being indexed.
  std::atomic<uint64_t> seq_{0};
  std::atomic<uint64_t> snapshot_seq_{0};
  std::atomic<uint64_t> wal_bytes_{0};
  mutable std::mutex write_mu_;
  mutable std::shared_mutex mu_;  // guards meta_ (and index deletes)
};

}  // namespace strata
