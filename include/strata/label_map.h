#pragma once

// Label -> node id map: open addressing with linear probing and backward-shift deletion.
// 16 bytes per slot at a load factor <= 0.5 (vs ~40-60 bytes per entry for a node-based
// std::unordered_map), one cache miss per lookup, and lookups can be prefetched in batches.

#include <cstdint>
#include <utility>
#include <vector>

#include "strata/common.h"

namespace strata {

class LabelMap {
 public:
  // kInvalidNode when absent.
  node_t find(label_t key) const {
    if (size_ == 0) return kInvalidNode;
    for (size_t i = slot(key);; i = (i + 1) & mask_) {
      const Slot& s = slots_[i];
      if (s.value == kInvalidNode) return kInvalidNode;
      if (s.key == key) return s.value;
    }
  }
  bool contains(label_t key) const { return find(key) != kInvalidNode; }
  void prefetch(label_t key) const {
    if (!slots_.empty()) __builtin_prefetch(&slots_[slot(key)], 0, 1);
  }

  // Inserts or overwrites. value must not be kInvalidNode.
  void set(label_t key, node_t value) {
    if ((size_ + 1) * 2 > slots_.size()) rehash(slots_.empty() ? 16 : slots_.size() * 2);
    for (size_t i = slot(key);; i = (i + 1) & mask_) {
      Slot& s = slots_[i];
      if (s.value == kInvalidNode) {
        s = {key, value};
        ++size_;
        return;
      }
      if (s.key == key) {
        s.value = value;
        return;
      }
    }
  }

  bool erase(label_t key) {
    if (size_ == 0) return false;
    size_t i = slot(key);
    for (;; i = (i + 1) & mask_) {
      if (slots_[i].value == kInvalidNode) return false;
      if (slots_[i].key == key) break;
    }
    // Backward-shift: pull later entries of the probe chain into the hole, so lookups never
    // need tombstones.
    for (size_t j = i;;) {
      j = (j + 1) & mask_;
      if (slots_[j].value == kInvalidNode) break;
      const size_t home = slot(slots_[j].key);
      const bool stays = (i <= j) ? (i < home && home <= j) : (i < home || home <= j);
      if (!stays) {
        slots_[i] = slots_[j];
        i = j;
      }
    }
    slots_[i].value = kInvalidNode;
    --size_;
    return true;
  }

  void reserve(size_t n) {
    size_t cap = 16;
    while (cap < n * 2) cap *= 2;
    if (cap > slots_.size()) rehash(cap);
  }
  size_t size() const { return size_; }
  size_t memory_bytes() const { return slots_.size() * sizeof(Slot); }
  void swap(LabelMap& o) noexcept {
    slots_.swap(o.slots_);
    std::swap(size_, o.size_);
    std::swap(mask_, o.mask_);
  }
  template <class Fn>
  void for_each(Fn&& fn) const {
    for (const Slot& s : slots_) {
      if (s.value != kInvalidNode) fn(s.key, s.value);
    }
  }

 private:
  struct Slot {
    label_t key = 0;
    node_t value = kInvalidNode;
  };

  size_t slot(label_t key) const {
    uint64_t x = key + 0x9e3779b97f4a7c15ull;  // splitmix64 finaliser: sequential ids spread well
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ull;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebull;
    return static_cast<size_t>(x ^ (x >> 31)) & mask_;
  }

  void rehash(size_t cap) {
    std::vector<Slot> old(cap);
    old.swap(slots_);
    mask_ = cap - 1;
    size_ = 0;
    for (const Slot& s : old) {
      if (s.value != kInvalidNode) set(s.key, s.value);
    }
  }

  std::vector<Slot> slots_;
  size_t size_ = 0;
  size_t mask_ = 0;
};

}  // namespace strata
