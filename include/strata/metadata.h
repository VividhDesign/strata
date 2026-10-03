#pragma once

#include <mutex>
#include <nlohmann/json_fwd.hpp>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "strata/common.h"
#include "strata/io.h"

namespace strata {

// JSON metadata per vector plus an inverted index over top-level scalar fields
// (and scalar elements of top-level arrays), used to evaluate filters like
//   {"source": "fiqa", "year": {"$in": [2020, 2021]}, "$or": [{...}, {...}]}
// Supported operators: implicit equality, $eq, $ne, $in, $nin, $and, $or.
class MetadataStore {
 public:
  void put(label_t id, const std::string& json_text);  // "" means {}
  void erase(label_t id);
  const std::string* get(label_t id) const;
  size_t size() const { return docs_.size(); }

  // Ids matching `filter`, sorted ascending. Throws Error on malformed filters.
  std::vector<label_t> evaluate(const nlohmann::json& filter) const;

  void save(FileWriter& w) const;
  void load(FileReader& r);

 private:
  // Ids sharing one (field, value). The hash set gives O(1) updates; the sorted copy (what
  // filters need for merging) is rebuilt lazily on the first read after a change.
  struct Posting {
    std::unordered_set<label_t> ids;
    mutable std::vector<label_t> sorted;
    mutable bool dirty = true;
  };

  std::vector<label_t> match_value(const std::string& field, const nlohmann::json& value) const;
  std::vector<label_t> all_ids() const;
  void index_doc(label_t id, const nlohmann::json& doc, bool add);

  std::unordered_map<label_t, std::string> docs_;
  // field -> canonical value -> ids
  std::unordered_map<std::string, std::unordered_map<std::string, Posting>> inverted_;
  mutable std::vector<label_t> all_sorted_;
  mutable bool all_dirty_ = true;
  mutable std::mutex cache_mu_;  // concurrent readers may refresh the sorted caches
};

}  // namespace strata
