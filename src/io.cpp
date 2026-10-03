#include "strata/io.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>

#if defined(__ARM_FEATURE_CRC32)
#include <arm_acle.h>
#endif

namespace strata {

namespace {

std::string errno_message() { return std::strerror(errno); }

#if !defined(__ARM_FEATURE_CRC32)
struct Crc32Table {
  uint32_t t[256];
  Crc32Table() {
    for (uint32_t i = 0; i < 256; ++i) {
      uint32_t c = i;
      for (int k = 0; k < 8; ++k) c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
      t[i] = c;
    }
  }
};
const Crc32Table& crc_table() {
  static const Crc32Table table;
  return table;
}
#endif

}  // namespace

uint32_t crc32_update(uint32_t crc, const void* data, size_t n) {
  const auto* p = static_cast<const uint8_t*>(data);
  uint32_t c = ~crc;
#if defined(__ARM_FEATURE_CRC32)
  while (n >= 8) {
    uint64_t v;
    std::memcpy(&v, p, 8);
    c = __crc32d(c, v);
    p += 8;
    n -= 8;
  }
  while (n--) c = __crc32b(c, *p++);
#else
  const uint32_t* t = crc_table().t;
  while (n--) c = t[(c ^ *p++) & 0xFFu] ^ (c >> 8);
#endif
  return ~c;
}

int full_fsync(int fd) {
#if defined(__APPLE__) && defined(F_FULLFSYNC)
  if (::fcntl(fd, F_FULLFSYNC) == 0) return 0;
#endif
  return ::fsync(fd);
}

void fsync_dir(const std::string& dir) {
  const int fd = ::open(dir.c_str(), O_RDONLY);
  if (fd < 0) return;
  ::fsync(fd);
  ::close(fd);
}

void atomic_write_file(const std::string& path, const std::string& contents) {
  const std::string tmp = path + ".tmp";
  {
    FileWriter w(tmp);
    w.write(contents.data(), contents.size());
    w.sync_and_close();
  }
  std::error_code ec;
  std::filesystem::rename(tmp, path, ec);
  if (ec) throw Error("rename " + tmp + " -> " + path + ": " + ec.message());
  fsync_dir(std::filesystem::path(path).parent_path().string());
}

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw Error("cannot open " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

FileWriter::FileWriter(const std::string& path) : path_(path) {
  f_ = std::fopen(path.c_str(), "wb");
  if (!f_) throw Error("cannot open " + path + " for writing: " + errno_message());
  std::setvbuf(f_, nullptr, _IOFBF, 1 << 20);
}

FileWriter::~FileWriter() {
  if (f_) std::fclose(f_);
}

void FileWriter::write(const void* data, size_t n) {
  if (n == 0) return;
  if (std::fwrite(data, 1, n, f_) != n) throw Error("write failed: " + path_ + ": " + errno_message());
  crc_ = crc32_update(crc_, data, n);
}

void FileWriter::write_crc() {
  const uint32_t c = crc_;
  if (std::fwrite(&c, sizeof(c), 1, f_) != 1) throw Error("write failed: " + path_);
}

void FileWriter::sync_and_close() {
  if (std::fflush(f_) != 0) throw Error("flush failed: " + path_ + ": " + errno_message());
  full_fsync(fileno(f_));
  std::fclose(f_);
  f_ = nullptr;
}

FileReader::FileReader(const std::string& path) : path_(path) {
  f_ = std::fopen(path.c_str(), "rb");
  if (!f_) throw Error("cannot open " + path + ": " + errno_message());
  std::setvbuf(f_, nullptr, _IOFBF, 1 << 20);
}

FileReader::~FileReader() {
  if (f_) std::fclose(f_);
}

void FileReader::read(void* data, size_t n) {
  if (n == 0) return;
  if (std::fread(data, 1, n, f_) != n) throw Error("unexpected end of file: " + path_);
  crc_ = crc32_update(crc_, data, n);
}

std::string FileReader::get_string() {
  const auto n = get<uint32_t>();
  std::string s(n, '\0');
  read(s.data(), n);
  return s;
}

void FileReader::verify_crc() {
  uint32_t stored = 0;
  if (std::fread(&stored, sizeof(stored), 1, f_) != 1) throw Error("missing checksum: " + path_);
  if (stored != crc_) throw Error("checksum mismatch: " + path_ + " is corrupted");
}

}  // namespace strata
