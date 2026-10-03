#include "strata/metadata.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <nlohmann/json.hpp>

namespace strata {

using json = nlohmann::json;

namespace {

// Canonical key of a scalar so that 3 and 3.0 match.
std::string value_key(const json& v) {
  if (v.is_number_float()) {
    const double d = v.get<double>();
    if (std::floor(d) == d && std::fabs(d) < 9e15) return std::to_string(static_cast<int64_t>(d));
  }
  return v.dump();
}

bool is_scalar(const json& v) { return v.is_string() || v.is_number() || v.is_boolean() || v.is_null(); }

std::vector<label_t> set_intersection(const std::vector<label_t>& a, const std::vector<label_t>& b) {
  std::vector<label_t> out;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

std::vector<label_t> set_union(const std::vector<label_t>& a, const std::vector<label_t>& b) {
  std::vector<label_t> out;
  std::set_union(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

std::vector<label_t> set_difference(const std::vector<label_t>& a, const std::vector<label_t>& b) {
  std::vector<label_t> out;
  std::set_difference(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(out));
  return out;
}

}  // namespace

void MetadataStore::index_doc(label_t id, const json& doc, bool add) {
  for (auto it = doc.begin(); it != doc.end(); ++it) {
    const std::string& field = it.key();
    const json& value = it.value();
    auto touch = [&](const json& v) {
      if (!is_scalar(v)) return;
      const std::string key = value_key(v);
      Posting& p = inverted_[field][key];
      p.dirty = true;
      if (add) {
        p.ids.insert(id);
      } else {
        p.ids.erase(id);
        if (p.ids.empty()) inverted_[field].erase(key);
      }
    };
    if (value.is_array()) {
      for (const auto& el : value) touch(el);
    } else {
      touch(value);
    }
  }
}

void MetadataStore::put(label_t id, const std::string& json_text) {
  json doc = json_text.empty() ? json::object() : json::parse(json_text, nullptr, false);
  if (doc.is_discarded() || !doc.is_object()) throw Error("metadata must be a JSON object");
  erase(id);
  index_doc(id, doc, true);
  docs_[id] = doc.dump();
  all_dirty_ = true;
}

void MetadataStore::erase(label_t id) {
  auto it = docs_.find(id);
  if (it == docs_.end()) return;
  index_doc(id, json::parse(it->second), false);
  docs_.erase(it);
  all_dirty_ = true;
}

const std::string* MetadataStore::get(label_t id) const {
  auto it = docs_.find(id);
  return it == docs_.end() ? nullptr : &it->second;
}

std::vector<label_t> MetadataStore::all_ids() const {
  std::lock_guard<std::mutex> g(cache_mu_);
  if (all_dirty_) {
    all_sorted_.clear();
    all_sorted_.reserve(docs_.size());
    for (const auto& kv : docs_) all_sorted_.push_back(kv.first);
    std::sort(all_sorted_.begin(), all_sorted_.end());
    all_dirty_ = false;
  }
  return all_sorted_;
}

std::vector<label_t> MetadataStore::match_value(const std::string& field, const json& value) const {
  if (!is_scalar(value)) throw Error("filter values must be scalars (field '" + field + "')");
  auto f = inverted_.find(field);
  if (f == inverted_.end()) return {};
  auto v = f->second.find(value_key(value));
  if (v == f->second.end()) return {};
  const Posting& p = v->second;
  std::lock_guard<std::mutex> g(cache_mu_);
  if (p.dirty) {
    p.sorted.assign(p.ids.begin(), p.ids.end());
    std::sort(p.sorted.begin(), p.sorted.end());
    p.dirty = false;
  }
  return p.sorted;
}

std::vector<label_t> MetadataStore::evaluate(const json& filter) const {
  if (!filter.is_object()) throw Error("filter must be a JSON object");
  std::vector<std::vector<label_t>> parts;

  for (const auto& [key, value] : filter.items()) {
    if (key == "$and" || key == "$or") {
      if (!value.is_array() || value.empty()) throw Error(key + " expects a non-empty array");
      std::vector<label_t> acc = evaluate(value[0]);
      for (size_t i = 1; i < value.size(); ++i) {
        acc = key == "$and" ? set_intersection(acc, evaluate(value[i])) : set_union(acc, evaluate(value[i]));
      }
      parts.push_back(std::move(acc));
    } else if (!key.empty() && key[0] == '$') {
      throw Error("unsupported operator " + key);
    } else if (value.is_object()) {
      for (const auto& [op, arg] : value.items()) {
        if (op == "$eq") {
          parts.push_back(match_value(key, arg));
        } else if (op == "$ne") {
          parts.push_back(set_difference(all_ids(), match_value(key, arg)));
        } else if (op == "$in" || op == "$nin") {
          if (!arg.is_array()) throw Error(op + " expects an array");
          std::vector<label_t> acc;
          for (const auto& v : arg) acc = set_union(acc, match_value(key, v));
          parts.push_back(op == "$in" ? acc : set_difference(all_ids(), acc));
        } else {
          throw Error("unsupported operator " + op);
        }
      }
    } else {
      parts.push_back(match_value(key, value));
    }
  }

  if (parts.empty()) return all_ids();
  std::vector<label_t> result = std::move(parts[0]);
  for (size_t i = 1; i < parts.size(); ++i) result = set_intersection(result, parts[i]);
  return result;
}

void MetadataStore::save(FileWriter& w) const {
  w.put<uint64_t>(docs_.size());
  for (const auto& [id, text] : docs_) {
    w.put<uint64_t>(id);
    w.put_string(text);
  }
}

void MetadataStore::load(FileReader& r) {
  docs_.clear();
  inverted_.clear();
  all_dirty_ = true;
  const auto n = r.get<uint64_t>();
  for (uint64_t i = 0; i < n; ++i) {
    const auto id = r.get<uint64_t>();
    put(id, r.get_string());
  }
}

}  // namespace strata
