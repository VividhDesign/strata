#include <doctest.h>

#include <filesystem>
#include <nlohmann/json.hpp>

#include "strata/collection.h"
#include "test_util.h"

using namespace strata;
using json = nlohmann::json;

namespace {

CollectionConfig small_config(size_t dim) {
  CollectionConfig c;
  c.name = "test";
  c.dim = dim;
  c.metric = Metric::L2;
  c.params.M = 8;
  c.params.ef_construction = 64;
  return c;
}

std::vector<std::string> make_metadata(size_t n) {
  std::vector<std::string> md;
  for (size_t i = 0; i < n; ++i) {
    md.push_back(json{{"parity", i % 2 == 0 ? "even" : "odd"},
                      {"bucket", static_cast<int>(i % 5)},
                      {"tags", {"all", i < 10 ? "first-ten" : "rest"}}}
                     .dump());
  }
  return md;
}

std::set<label_t> ids_of(const std::vector<Hit>& hits) {
  std::set<label_t> s;
  for (const auto& h : hits) s.insert(h.id);
  return s;
}

}  // namespace

TEST_CASE("metadata filters: equality, $in, $ne, $nin, $or, arrays") {
  testutil::TempDir tmp;
  auto col = Collection::create(tmp.str("c"), small_config(4));
  const size_t n = 100;
  const auto vecs = testutil::random_vectors(n, 4, 1);
  const auto ids = testutil::iota_labels(n);
  col->upsert(ids.data(), vecs.data(), n, make_metadata(n));
  const float* q = vecs.data();

  auto all = [&](const std::string& filter) { return col->query(q, 1, n, 400, filter)[0]; };

  for (const auto& h : all(R"({"parity": "even"})")) CHECK(h.id % 2 == 0);
  CHECK(all(R"({"parity": "even"})").size() == 50);
  CHECK(all(R"({"bucket": {"$in": [1, 2]}})").size() == 40);
  CHECK(all(R"({"bucket": {"$ne": 0}})").size() == 80);
  CHECK(all(R"({"bucket": {"$nin": [0, 1]}})").size() == 60);
  CHECK(all(R"({"parity": "odd", "bucket": 3})").size() == 10);
  CHECK(all(R"({"$or": [{"bucket": 0}, {"parity": "odd"}]})").size() == 60);
  CHECK(all(R"({"tags": "first-ten"})").size() == 10);
  CHECK(all(R"({"bucket": 3.0})").size() == 20);  // 3 and 3.0 are the same value
  CHECK(all(R"({"parity": "nope"})").empty());
  CHECK(all("").size() == n);
  CHECK(all("{}").size() == n);
  CHECK_THROWS_AS(all(R"({"bucket": {"$gt": 1}})"), Error);
  CHECK_THROWS_AS(all("not json"), Error);

  const auto hits = all(R"({"parity": "odd"})");
  REQUIRE_FALSE(hits.empty());
  CHECK(json::parse(hits[0].metadata)["parity"] == "odd");
}

TEST_CASE("upsert updates metadata and the inverted index") {
  testutil::TempDir tmp;
  auto col = Collection::create(tmp.str("c"), small_config(2));
  const float v[2] = {1, 1};
  const label_t id = 42;
  col->upsert(&id, v, 1, {R"({"color": "red"})"});
  col->upsert(&id, v, 1, {R"({"color": "blue"})"});
  CHECK(col->query(v, 1, 5, 0, R"({"color": "red"})")[0].empty());
  CHECK(col->query(v, 1, 5, 0, R"({"color": "blue"})")[0].size() == 1);
  CHECK(col->size() == 1);
  CHECK_THROWS_AS(col->upsert(&id, v, 1, {"[1,2]"}), Error);
}

TEST_CASE("crash recovery from the WAL alone, then from snapshot + WAL") {
  testutil::TempDir tmp;
  const std::string dir = tmp.str("c");
  const size_t n = 300;
  const auto vecs = testutil::random_vectors(n, 6, 7);
  const auto ids = testutil::iota_labels(n);
  const auto md = make_metadata(n);

  {
    auto col = Collection::create(dir, small_config(6));
    col->upsert(ids.data(), vecs.data(), 200, std::vector<std::string>(md.begin(), md.begin() + 200));
    const label_t gone[2] = {3, 4};
    col->remove(gone, 2);
    // Destroyed without a checkpoint: simulates a crash. Only the WAL survives.
  }
  {
    auto col = Collection::open(dir);
    CHECK(col->size() == 198);
    std::string meta;
    CHECK_FALSE(col->get(3, nullptr, &meta));
    REQUIRE(col->get(10, nullptr, &meta));
    CHECK(json::parse(meta)["tags"][1] == "rest");
    col->checkpoint();
    CHECK(json::parse(col->stats_json())["wal_bytes"] == 0);
    col->upsert(ids.data() + 200, vecs.data() + 200 * 6, 100, std::vector<std::string>(md.begin() + 200, md.end()));
  }
  {
    auto col = Collection::open(dir);
    CHECK(col->size() == 298);
    const json stats = json::parse(col->stats_json());
    CHECK(stats["snapshot_seq"] == 2);
    CHECK(stats["last_seq"] == 3);
    std::vector<float> v;
    REQUIRE(col->get(250, &v, nullptr));
    CHECK(v[0] == vecs[250 * 6]);
    const auto hits = col->query(vecs.data() + 250 * 6, 1, 1)[0];
    REQUIRE(hits.size() == 1);
    CHECK(hits[0].id == 250);
  }
}

TEST_CASE("a torn WAL tail loses only the unacknowledged write") {
  testutil::TempDir tmp;
  const std::string dir = tmp.str("c");
  const auto vecs = testutil::random_vectors(3, 4, 9);
  {
    auto col = Collection::create(dir, small_config(4));
    for (label_t i = 0; i < 3; ++i) col->upsert(&i, vecs.data() + i * 4, 1);
  }
  const auto wal = std::filesystem::path(dir) / "wal.log";
  std::filesystem::resize_file(wal, std::filesystem::file_size(wal) - 5);
  auto col = Collection::open(dir);
  CHECK(col->size() == 2);
  CHECK(json::parse(col->stats_json())["last_seq"] == 2);
}

TEST_CASE("create refuses to overwrite and open requires a collection") {
  testutil::TempDir tmp;
  Collection::create(tmp.str("c"), small_config(3));
  CHECK_THROWS_AS(Collection::create(tmp.str("c"), small_config(3)), Error);
  CHECK_THROWS_AS(Collection::open(tmp.str("missing")), Error);
}

TEST_CASE("automatic checkpoint when the WAL grows past the threshold") {
  testutil::TempDir tmp;
  auto cfg = small_config(8);
  cfg.checkpoint_wal_bytes = 4096;
  const std::string dir = tmp.str("c");
  {
    auto col = Collection::create(dir, cfg);
    const auto vecs = testutil::random_vectors(200, 8, 11);
    for (label_t i = 0; i < 200; ++i) col->upsert(&i, vecs.data() + i * 8, 1);
    CHECK(json::parse(col->stats_json())["snapshot_seq"].get<uint64_t>() > 0);
  }
  CHECK(std::filesystem::exists(std::filesystem::path(dir) / "CURRENT"));
  CHECK(Collection::open(dir)->size() == 200);
}
