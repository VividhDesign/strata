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
  // Writes zero bytes until the file offset is a multiple of `alignment`.
  void pad_to(size_t alignment);
  uint64_t position() const { return pos_; }
  // Appends the CRC of all bytes written so far (the CRC itself is not checksummed).
  void write_crc();
  void sync_and_close();

 private:
  std::FILE* f_ = nullptr;
  std::string path_;
  uint32_t crc_ = 0;
  uint64_t pos_ = 0;
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
  // Skips padding written by FileWriter::pad_to (the bytes still enter the CRC).
  void skip_pad(size_t alignment);
  // Adds the next n bytes of the file, already available in memory at `data` (e.g. through a
  // memory map), to the CRC and moves past them without copying.
  void absorb(const void* data, size_t n);
  uint64_t position() const { return pos_; }
  // Reads the stored CRC and throws if it does not match the bytes read so far.
  void verify_crc();

 private:
  std::FILE* f_ = nullptr;
  std::string path_;
  uint32_t crc_ = 0;
  uint64_t pos_ = 0;
};

// Read-only memory map of a whole file.
class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;

  const char* data() const { return static_cast<const char*>(base_); }
  size_t size() const { return size_; }
  // Hints that pages will be read in random order, and drops already-touched pages from this
  // process's resident set (they stay in the OS page cache).
  void advise_random_and_release() const;

 private:
  void* base_ = nullptr;
  size_t size_ = 0;
};

}  // namespace strata
