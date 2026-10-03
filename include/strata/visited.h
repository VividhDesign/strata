#pragma once

#include <algorithm>
#include <cstdint>
#include <memory>
#include <mutex>
#include <vector>

namespace strata {

// "Have I seen node i during this search?" in O(1) without clearing an array per query.
// Each list stores a uint16 tag per node; a node counts as visited when its tag equals the
// current epoch. reset() just bumps the epoch, and only every 65535 searches do we pay for
// a full clear.
class VisitedList {
 public:
  explicit VisitedList(size_t n) : marks_(n, 0) {}

  void reset() {
    if (++epoch_ == 0) {
      std::fill(marks_.begin(), marks_.end(), 0);
      epoch_ = 1;
    }
  }
  // Marks `id` visited; returns true if it had not been visited yet.
  bool visit(uint32_t id) {
    if (marks_[id] == epoch_) return false;
    marks_[id] = epoch_;
    return true;
  }
  size_t capacity() const { return marks_.size(); }

 private:
  std::vector<uint16_t> marks_;
  uint16_t epoch_ = 0;
};

// Pool of visited lists so concurrent searches never allocate on the hot path.
class VisitedPool {
 public:
  explicit VisitedPool(size_t n = 0) : n_(n) {}

  std::unique_ptr<VisitedList> acquire() {
    std::unique_ptr<VisitedList> list;
    size_t n;
    {
      std::lock_guard<std::mutex> g(mu_);
      n = n_;
      if (!free_.empty()) {
        list = std::move(free_.back());
        free_.pop_back();
      }
    }
    if (!list || list->capacity() < n) list = std::make_unique<VisitedList>(n);
    list->reset();
    return list;
  }
  void release(std::unique_ptr<VisitedList> list) {
    std::lock_guard<std::mutex> g(mu_);
    free_.push_back(std::move(list));
  }
  void resize(size_t n) {
    std::lock_guard<std::mutex> g(mu_);
    n_ = n;
    free_.clear();
  }

 private:
  std::mutex mu_;
  std::vector<std::unique_ptr<VisitedList>> free_;
  size_t n_;
};

class VisitedHandle {
 public:
  explicit VisitedHandle(VisitedPool& pool) : pool_(pool), list_(pool.acquire()) {}
  ~VisitedHandle() { pool_.release(std::move(list_)); }
  VisitedHandle(const VisitedHandle&) = delete;
  VisitedHandle& operator=(const VisitedHandle&) = delete;
  VisitedList& operator*() { return *list_; }

 private:
  VisitedPool& pool_;
  std::unique_ptr<VisitedList> list_;
};

}  // namespace strata
