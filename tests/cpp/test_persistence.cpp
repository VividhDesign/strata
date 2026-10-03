#include <doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>

#include "strata/hnsw.h"
#include "strata/wal.h"
#include "test_util.h"

using namespace strata;

TEST_CASE("index save/load round-trip returns identical results") {
  testutil::TempDir tmp;
  const size_t n = 3000, dim = 20;
  const auto data = testutil::random_vectors(n, dim, 3);
  const auto labels = testutil::iota_labels(n, 1000);
  const auto queries = testutil::random_vectors(50, dim, 4);

  HNSWIndex index(dim, Metric::Cosine, {12, 100, 7, true});
  index.add(data.data(), labels.data(), n, 4);
  index.remove(1005);
  index.set_ef_search(77);
  index.save(tmp.str("idx.bin"));

  auto loaded = HNSWIndex::load(tmp.str("idx.bin"));
  CHECK(loaded->size() == index.size());
  CHECK(loaded->dim() == dim);
  CHECK(loaded->metric() == Metric::Cosine);
  CHECK(loaded->params().M == 12);
  CHECK(loaded->ef_search() == 77);
  CHECK_FALSE(loaded->contains(1005));
  for (size_t i = 0; i < 50; ++i) {
    const auto a = index.search(queries.data() + i * dim, 10);
    const auto b = loaded->search(queries.data() + i * dim, 10);
    REQUIRE(a.size() == b.size());
    for (size_t j = 0; j < a.size(); ++j) {
      CHECK(a[j].label == b[j].label);
      CHECK(a[j].distance == b[j].distance);
    }
  }
  // The loaded index stays writable.
  const auto extra = testutil::random_vectors(10, dim, 5);
  const auto extra_labels = testutil::iota_labels(10, 50000);
  loaded->add(extra.data(), extra_labels.data(), 10);
  CHECK(loaded->size() == index.size() + 10);
}

TEST_CASE("corrupted or truncated index files are rejected") {
  testutil::TempDir tmp;
  const auto data = testutil::random_vectors(500, 8, 3);
  const auto labels = testutil::iota_labels(500);
  HNSWIndex index(8, Metric::L2);
  index.add(data.data(), labels.data(), 500);
  const std::string path = tmp.str("idx.bin");
  index.save(path);
  const auto size = std::filesystem::file_size(path);

  SUBCASE("flipped byte") {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(static_cast<std::streamoff>(size / 2));
    char c;
    f.read(&c, 1);
    f.seekp(static_cast<std::streamoff>(size / 2));
    c = static_cast<char>(c ^ 0x5A);
    f.write(&c, 1);
    f.close();
    CHECK_THROWS_AS(HNSWIndex::load(path), Error);
  }
  SUBCASE("truncated") {
    std::filesystem::resize_file(path, size - 100);
    CHECK_THROWS_AS(HNSWIndex::load(path), Error);
  }
  SUBCASE("not an index") {
    std::ofstream(path, std::ios::trunc) << "hello";
    CHECK_THROWS_AS(HNSWIndex::load(path), Error);
  }
}

TEST_CASE("WAL replays intact records and drops a torn tail") {
  testutil::TempDir tmp;
  const std::string path = tmp.str("wal.log");
  {
    WriteAheadLog wal(path, true);
    wal.append(1, WriteAheadLog::Op::Upsert, "first");
    wal.append(2, WriteAheadLog::Op::Delete, std::string(1000, 'x'));
    wal.append(3, WriteAheadLog::Op::Upsert, "third");
  }
  // Simulate a crash in the middle of writing record 3.
  std::filesystem::resize_file(path, std::filesystem::file_size(path) - 3);

  WriteAheadLog wal(path, true);
  std::vector<uint64_t> seqs;
  CHECK(wal.replay([&](const WriteAheadLog::Record& r) { seqs.push_back(r.seq); }) == 2);
  CHECK(seqs == std::vector<uint64_t>{1, 2});

  // New records go after the last good one.
  wal.append(3, WriteAheadLog::Op::Upsert, "again");
  WriteAheadLog reopened(path, true);
  std::vector<std::string> payloads;
  reopened.replay([&](const WriteAheadLog::Record& r) { payloads.push_back(r.payload); });
  REQUIRE(payloads.size() == 3);
  CHECK(payloads[2] == "again");
}

TEST_CASE("WAL stops at a corrupted record") {
  testutil::TempDir tmp;
  const std::string path = tmp.str("wal.log");
  {
    WriteAheadLog wal(path, false);
    wal.append(1, WriteAheadLog::Op::Upsert, "aaaa");
    wal.append(2, WriteAheadLog::Op::Upsert, "bbbb");
  }
  {
    std::fstream f(path, std::ios::in | std::ios::out | std::ios::binary);
    f.seekp(-2, std::ios::end);
    f.put('Z');
  }
  WriteAheadLog wal(path, false);
  CHECK(wal.replay([](const WriteAheadLog::Record&) {}) == 1);
}
