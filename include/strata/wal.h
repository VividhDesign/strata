#pragma once

#include <functional>
#include <string>

#include "strata/common.h"

namespace strata {

// Append-only write-ahead log.
// Record framing: [body_len:u32][crc32(body):u32][body], body = [seq:u64][op:u8][payload].
// A crash can leave a torn record at the tail; replay() stops at the first record whose
// length or checksum is invalid and truncates the file there.
class WriteAheadLog {
 public:
  enum class Op : uint8_t { Upsert = 1, Delete = 2 };
  struct Record {
    uint64_t seq;
    Op op;
    std::string payload;
  };

  WriteAheadLog(std::string path, bool sync_every_write);
  ~WriteAheadLog();
  WriteAheadLog(const WriteAheadLog&) = delete;
  WriteAheadLog& operator=(const WriteAheadLog&) = delete;

  void append(uint64_t seq, Op op, const std::string& payload);
  size_t replay(const std::function<void(const Record&)>& fn);
  void truncate();
  uint64_t size_bytes() const { return size_; }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
  int fd_ = -1;
  bool sync_;
  uint64_t size_ = 0;
};

}  // namespace strata
