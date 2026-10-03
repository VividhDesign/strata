#pragma once

#include <cstdio>
#include <string>

#include "strata/common.h"

namespace strata {

// CRC-32 (IEEE 802.3, same as zlib). Uses the ARMv8 CRC32 instructions when available.
uint32_t crc32_update(uint32_t crc, const void* data, size_t n);

// fsync that really reaches the disk: on macOS plain fsync() only flushes to the drive's
// cache, F_FULLFSYNC asks the drive to flush too.
int full_fsync(int fd);
void fsync_dir(const std::string& dir);

// Writes `contents` to `path` via temp file + fsync + rename, so readers see either the old
// or the new file, never a half-written one.
void atomic_write_file(const std::string& path, const std::string& contents);
std::string read_file(const std::string& path);

// Buffered binary writer that keeps a running CRC of everything written.
class FileWriter {
 public:
  explicit FileWriter(const std::string& path);
  ~FileWriter();
  FileWriter(const FileWriter&) = delete;
  FileWriter& operator=(const FileWriter&) = delete;

  void write(const void* data, size_t n);
  template <class T>
  void put(const T& v) {
    write(&v, sizeof(T));
  }
  void put_string(const std::string& s) {
    put<uint32_t>(static_cast<uint32_t>(s.size()));
    write(s.data(), s.size());
  }
  // Appends the CRC of all bytes written so far (the CRC itself is not checksummed).
  void write_crc();
  void sync_and_close();

 private:
  std::FILE* f_ = nullptr;
  std::string path_;
  uint32_t crc_ = 0;
};

class FileReader {
 public:
  explicit FileReader(const std::string& path);
  ~FileReader();
  FileReader(const FileReader&) = delete;
  FileReader& operator=(const FileReader&) = delete;

  void read(void* data, size_t n);
  template <class T>
  T get() {
    T v;
    read(&v, sizeof(T));
    return v;
  }
  std::string get_string();
  // Reads the stored CRC and throws if it does not match the bytes read so far.
  void verify_crc();

 private:
  std::FILE* f_ = nullptr;
  std::string path_;
  uint32_t crc_ = 0;
};

}  // namespace strata
