#include "strata/wal.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <vector>

#include "strata/io.h"

namespace strata {

namespace {

constexpr size_t kHeader = 2 * sizeof(uint32_t);
constexpr size_t kBodyPrefix = sizeof(uint64_t) + sizeof(uint8_t);
constexpr uint32_t kMaxRecord = 1u << 30;

void write_all(int fd, const char* data, size_t n, const std::string& path) {
  while (n > 0) {
    const ssize_t w = ::write(fd, data, n);
    if (w < 0) {
      if (errno == EINTR) continue;
      throw Error("WAL write failed (" + path + "): " + std::strerror(errno));
    }
    data += w;
    n -= static_cast<size_t>(w);
  }
}

}  // namespace

WriteAheadLog::WriteAheadLog(std::string path, bool sync_every_write) : path_(std::move(path)), sync_(sync_every_write) {
  fd_ = ::open(path_.c_str(), O_RDWR | O_CREAT | O_APPEND, 0644);
  if (fd_ < 0) throw Error("cannot open WAL " + path_ + ": " + std::strerror(errno));
  struct stat st {};
  if (::fstat(fd_, &st) == 0) size_ = static_cast<uint64_t>(st.st_size);
}

WriteAheadLog::~WriteAheadLog() {
  if (fd_ >= 0) ::close(fd_);
}

void WriteAheadLog::append(uint64_t seq, Op op, const std::string& payload) {
  const size_t body_len = kBodyPrefix + payload.size();
  if (body_len > kMaxRecord) throw Error("WAL record too large");
  std::vector<char> buf(kHeader + body_len);
  char* body = buf.data() + kHeader;
  std::memcpy(body, &seq, sizeof(seq));
  body[sizeof(seq)] = static_cast<char>(op);
  std::memcpy(body + kBodyPrefix, payload.data(), payload.size());
  const auto len = static_cast<uint32_t>(body_len);
  const uint32_t crc = crc32_update(0, body, body_len);
  std::memcpy(buf.data(), &len, sizeof(len));
  std::memcpy(buf.data() + sizeof(len), &crc, sizeof(crc));

  // One write() per record; a crash mid-write leaves a torn tail that replay() detects.
  write_all(fd_, buf.data(), buf.size(), path_);
  if (sync_ && full_fsync(fd_) != 0) throw Error("WAL fsync failed: " + std::string(std::strerror(errno)));
  size_ += buf.size();
}

size_t WriteAheadLog::replay(const std::function<void(const Record&)>& fn) {
  const std::string data = read_file(path_);
  size_t pos = 0, records = 0;
  while (pos + kHeader <= data.size()) {
    uint32_t len, crc;
    std::memcpy(&len, data.data() + pos, sizeof(len));
    std::memcpy(&crc, data.data() + pos + sizeof(len), sizeof(crc));
    if (len < kBodyPrefix || len > kMaxRecord || pos + kHeader + len > data.size()) break;  // torn
    const char* body = data.data() + pos + kHeader;
    if (crc32_update(0, body, len) != crc) break;  // corrupted
    Record rec;
    std::memcpy(&rec.seq, body, sizeof(rec.seq));
    rec.op = static_cast<Op>(static_cast<uint8_t>(body[sizeof(rec.seq)]));
    rec.payload.assign(body + kBodyPrefix, len - kBodyPrefix);
    fn(rec);
    ++records;
    pos += kHeader + len;
  }
  if (pos != data.size()) {
    // Drop the damaged tail so new records are appended after the last good one.
    if (::ftruncate(fd_, static_cast<off_t>(pos)) != 0) throw Error("cannot truncate WAL tail");
    full_fsync(fd_);
  }
  size_ = pos;
  return records;
}

void WriteAheadLog::truncate() {
  if (::ftruncate(fd_, 0) != 0) throw Error("cannot truncate WAL: " + std::string(std::strerror(errno)));
  full_fsync(fd_);
  size_ = 0;
}

}  // namespace strata
