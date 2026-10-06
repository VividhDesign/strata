// strata-server: REST API over durable Strata collections.
//
//   strata-server --data-dir ./strata-data --port 8080
//
//   POST   /collections                       {"name", "dim", "metric"?, "M"?, "ef_construction"?,
//                                               "quantization"? ("none" | "sq8"), "rerank"?}
//   GET    /collections                       list collections with stats
//   GET    /collections/{name}                stats
//   DELETE /collections/{name}                drop collection (removes its directory)
//   POST   /collections/{name}/upsert         {"ids": [...], "vectors": [[...]], "metadata"?: [{...}]}
//   POST   /collections/{name}/query          {"vector" | "vectors", "k"?, "ef"?, "filter"?, "include_metadata"?}
//   POST   /collections/{name}/delete         {"ids": [...]}
//   GET    /collections/{name}/points/{id}    vector + metadata
//   POST   /collections/{name}/checkpoint     snapshot + truncate WAL
//   GET    /health, GET /metrics (Prometheus text format)

#include <httplib.h>

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <regex>
#include <shared_mutex>
#include <string>
#include <thread>

#include "strata/collection.h"

namespace fs = std::filesystem;
using json = nlohmann::json;
using strata::Collection;
using strata::CollectionConfig;

namespace {

std::atomic<bool> g_stop{false};
void on_signal(int) { g_stop = true; }

struct HttpError {
  int status;
  std::string message;
};

// ---- metrics ------------------------------------------------------------------------------

constexpr double kBucketsMs[] = {0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 250, 1000};
constexpr size_t kNumBuckets = sizeof(kBucketsMs) / sizeof(kBucketsMs[0]);

struct RouteMetrics {
  std::atomic<uint64_t> requests{0}, errors{0}, micros{0};
  std::atomic<uint64_t> buckets[kNumBuckets + 1] = {};
};

class Metrics {
 public:
  RouteMetrics& route(const std::string& name) {
    std::lock_guard<std::mutex> g(mu_);
    auto& p = routes_[name];
    if (!p) p = std::make_unique<RouteMetrics>();
    return *p;
  }
  std::string render() {
    std::lock_guard<std::mutex> g(mu_);
    std::string out =
        "# TYPE strata_requests_total counter\n# TYPE strata_errors_total counter\n"
        "# TYPE strata_request_duration_ms histogram\n";
    for (const auto& [name, m] : routes_) {
      const std::string lbl = "{route=\"" + name + "\"";
      out += "strata_requests_total" + lbl + "} " + std::to_string(m->requests.load()) + "\n";
      out += "strata_errors_total" + lbl + "} " + std::to_string(m->errors.load()) + "\n";
      uint64_t cumulative = 0;
      for (size_t i = 0; i < kNumBuckets; ++i) {
        cumulative += m->buckets[i].load();
        char le[32];
        std::snprintf(le, sizeof(le), "%g", kBucketsMs[i]);
        out += "strata_request_duration_ms_bucket" + lbl + ",le=\"" + le + "\"} " + std::to_string(cumulative) + "\n";
      }
      cumulative += m->buckets[kNumBuckets].load();
      out += "strata_request_duration_ms_bucket" + lbl + ",le=\"+Inf\"} " + std::to_string(cumulative) + "\n";
      out += "strata_request_duration_ms_sum" + lbl + "} " + std::to_string(m->micros.load() / 1000.0) + "\n";
      out += "strata_request_duration_ms_count" + lbl + "} " + std::to_string(m->requests.load()) + "\n";
    }
    return out;
  }

 private:
  std::mutex mu_;
  std::map<std::string, std::unique_ptr<RouteMetrics>> routes_;
};

// ---- collection registry ------------------------------------------------------------------

class Registry {
 public:
  Registry(fs::path root, bool sync_wal) : root_(std::move(root)), sync_wal_(sync_wal) {
    fs::create_directories(root_);
    for (const auto& entry : fs::directory_iterator(root_)) {
      if (entry.is_directory() && fs::exists(entry.path() / "config.json")) {
        const std::string name = entry.path().filename().string();
        collections_[name] = std::shared_ptr<Collection>(Collection::open(entry.path().string(), sync_wal_));
        std::cerr << "opened collection '" << name << "' (" << collections_[name]->size() << " vectors)\n";
      }
    }
  }

  std::shared_ptr<Collection> get(const std::string& name) {
    std::shared_lock<std::shared_mutex> g(mu_);
    auto it = collections_.find(name);
    if (it == collections_.end()) throw HttpError{404, "collection '" + name + "' not found"};
    return it->second;
  }

  std::shared_ptr<Collection> create(CollectionConfig cfg) {
    static const std::regex valid("[A-Za-z0-9_-]{1,64}");
    if (!std::regex_match(cfg.name, valid)) throw HttpError{400, "name must match [A-Za-z0-9_-]{1,64}"};
    std::unique_lock<std::shared_mutex> g(mu_);
    if (collections_.count(cfg.name)) throw HttpError{409, "collection '" + cfg.name + "' already exists"};
    cfg.sync_wal = sync_wal_;
    auto c = std::shared_ptr<Collection>(Collection::create((root_ / cfg.name).string(), cfg));
    collections_[cfg.name] = c;
    return c;
  }

  void drop(const std::string& name) {
    std::unique_lock<std::shared_mutex> g(mu_);
    auto it = collections_.find(name);
    if (it == collections_.end()) throw HttpError{404, "collection '" + name + "' not found"};
    collections_.erase(it);  // in-flight requests keep their shared_ptr alive
    fs::remove_all(root_ / name);
  }

  std::vector<std::shared_ptr<Collection>> all() {
    std::shared_lock<std::shared_mutex> g(mu_);
    std::vector<std::shared_ptr<Collection>> out;
    for (const auto& kv : collections_) out.push_back(kv.second);
    return out;
  }

 private:
  fs::path root_;
  bool sync_wal_;
  std::shared_mutex mu_;
  std::map<std::string, std::shared_ptr<Collection>> collections_;
};

// ---- request helpers ----------------------------------------------------------------------

json parse_body(const httplib::Request& req) {
  json body = json::parse(req.body, nullptr, false);
  if (body.is_discarded() || !body.is_object()) throw HttpError{400, "request body must be a JSON object"};
  return body;
}

std::vector<float> parse_vectors(const json& rows, size_t dim) {
  if (!rows.is_array()) throw HttpError{400, "vectors must be an array of arrays"};
  std::vector<float> out;
  out.reserve(rows.size() * dim);
  for (const auto& row : rows) {
    if (!row.is_array() || row.size() != dim) {
      throw HttpError{400, "every vector must be an array of " + std::to_string(dim) + " numbers"};
    }
    for (const auto& x : row) {
      if (!x.is_number()) throw HttpError{400, "vector entries must be numbers"};
      out.push_back(x.get<float>());
    }
  }
  return out;
}

std::vector<strata::label_t> parse_ids(const json& ids) {
  if (!ids.is_array()) throw HttpError{400, "ids must be an array of non-negative integers"};
  std::vector<strata::label_t> out;
  out.reserve(ids.size());
  for (const auto& id : ids) {
    if (!id.is_number_unsigned() && !(id.is_number_integer() && id.get<int64_t>() >= 0)) {
      throw HttpError{400, "ids must be non-negative integers"};
    }
    out.push_back(id.get<strata::label_t>());
  }
  return out;
}

void send_json(httplib::Response& res, const json& body, int status = 200) {
  res.status = status;
  res.set_content(body.dump(), "application/json");
}

}  // namespace

int main(int argc, char** argv) {
  std::string host = "127.0.0.1", data_dir = "./strata-data";
  // cpp-httplib serves each keep-alive connection on one pool thread, so the pool must be
  // larger than the number of concurrent clients or connections queue (seen as p99 spikes).
  int port = 8080, threads = std::max(64, 4 * static_cast<int>(std::thread::hardware_concurrency()));
  bool sync_wal = true;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto next = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "missing value for " << a << "\n";
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--host") host = next();
    else if (a == "--port") port = std::stoi(next());
    else if (a == "--data-dir") data_dir = next();
    else if (a == "--threads") threads = std::stoi(next());
    else if (a == "--no-fsync") sync_wal = false;
    else {
      std::cerr << "usage: strata-server [--host H] [--port P] [--data-dir D] [--threads N] [--no-fsync]\n";
      return a == "--help" ? 0 : 2;
    }
  }

  Registry registry(data_dir, sync_wal);
  Metrics metrics;
  httplib::Server server;
  server.new_task_queue = [threads] { return new httplib::ThreadPool(static_cast<size_t>(threads)); };
  server.set_payload_max_length(512ull << 20);
  server.set_keep_alive_max_count(100000);  // default (5) forces a reconnect every 5 requests
  server.set_keep_alive_timeout(30);

  // Wraps a handler with error handling + latency metrics.
  auto route = [&](const std::string& name, auto handler) {
    RouteMetrics* m = &metrics.route(name);
    return [m, handler](const httplib::Request& req, httplib::Response& res) {
      const auto t0 = std::chrono::steady_clock::now();
      try {
        handler(req, res);
      } catch (const HttpError& e) {
        send_json(res, {{"error", e.message}}, e.status);
      } catch (const strata::Error& e) {
        send_json(res, {{"error", e.what()}}, 400);
      } catch (const json::exception& e) {
        send_json(res, {{"error", std::string("bad JSON: ") + e.what()}}, 400);
      } catch (const std::exception& e) {
        send_json(res, {{"error", e.what()}}, 500);
      }
      const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      m->requests++;
      if (res.status >= 400) m->errors++;
      m->micros += static_cast<uint64_t>(ms * 1000);
      size_t b = 0;
      while (b < kNumBuckets && ms > kBucketsMs[b]) ++b;
      m->buckets[b]++;
    };
  };

  const std::string C = R"(/collections/([A-Za-z0-9_\-]+))";

  server.Get("/health", route("health", [](const httplib::Request&, httplib::Response& res) {
               send_json(res, {{"status", "ok"}, {"version", strata::kVersion}, {"simd", strata::kernels::simd_backend()}});
             }));

  server.Get("/metrics", [&](const httplib::Request&, httplib::Response& res) {
    res.set_content(metrics.render(), "text/plain; version=0.0.4");
  });

  server.Get("/collections", route("list", [&](const httplib::Request&, httplib::Response& res) {
               json out = json::array();
               for (const auto& c : registry.all()) out.push_back(json::parse(c->stats_json()));
               send_json(res, {{"collections", out}});
             }));

  server.Post("/collections", route("create", [&](const httplib::Request& req, httplib::Response& res) {
                const json body = parse_body(req);
                CollectionConfig cfg;
                cfg.name = body.at("name").get<std::string>();
                cfg.dim = body.at("dim").get<size_t>();
                cfg.metric = strata::parse_metric(body.value("metric", "cosine"));
                cfg.params.M = body.value("M", static_cast<size_t>(16));
                cfg.params.ef_construction = body.value("ef_construction", static_cast<size_t>(200));
                cfg.params.quantization = strata::parse_quantization(body.value("quantization", "none"));
                cfg.params.rerank = body.value("rerank", true);
                auto c = registry.create(cfg);
                send_json(res, json::parse(c->stats_json()), 201);
              }));

  server.Get(C, route("stats", [&](const httplib::Request& req, httplib::Response& res) {
               send_json(res, json::parse(registry.get(req.matches[1])->stats_json()));
             }));

  server.Delete(C, route("drop", [&](const httplib::Request& req, httplib::Response& res) {
                  registry.drop(req.matches[1]);
                  send_json(res, {{"dropped", std::string(req.matches[1])}});
                }));

  server.Post(C + "/upsert", route("upsert", [&](const httplib::Request& req, httplib::Response& res) {
                auto c = registry.get(req.matches[1]);
                const json body = parse_body(req);
                const auto ids = parse_ids(body.at("ids"));
                const auto vectors = parse_vectors(body.at("vectors"), c->config().dim);
                if (vectors.size() != ids.size() * c->config().dim) throw HttpError{400, "ids and vectors differ in length"};
                std::vector<std::string> metadata;
                if (body.contains("metadata") && !body["metadata"].is_null()) {
                  const json& md = body["metadata"];
                  if (!md.is_array() || md.size() != ids.size()) throw HttpError{400, "metadata must have one object per id"};
                  for (const auto& m : md) metadata.push_back(m.dump());
                }
                c->upsert(ids.data(), vectors.data(), ids.size(), metadata);
                send_json(res, {{"upserted", ids.size()}});
              }));

  server.Post(C + "/query", route("query", [&](const httplib::Request& req, httplib::Response& res) {
                auto c = registry.get(req.matches[1]);
                const json body = parse_body(req);
                const size_t dim = c->config().dim;
                const bool single = body.contains("vector");
                const std::vector<float> queries =
                    single ? parse_vectors(json::array({body["vector"]}), dim) : parse_vectors(body.at("vectors"), dim);
                const size_t k = body.value("k", static_cast<size_t>(10));
                const size_t ef = body.value("ef", static_cast<size_t>(0));
                if (k == 0 || k > 10000) throw HttpError{400, "k must be in [1, 10000]"};
                const std::string filter = body.contains("filter") ? body["filter"].dump() : "";
                const bool with_md = body.value("include_metadata", true);
                const auto hits = c->query(queries.data(), queries.size() / dim, k, ef, filter);
                json results = json::array();
                for (const auto& row : hits) {
                  json r = json::array();
                  for (const auto& h : row) {
                    json item = {{"id", h.id}, {"distance", h.distance}};
                    if (with_md) item["metadata"] = json::parse(h.metadata);
                    r.push_back(std::move(item));
                  }
                  results.push_back(std::move(r));
                }
                send_json(res, single ? json{{"results", results[0]}} : json{{"results", results}});
              }));

  server.Post(C + "/delete", route("delete", [&](const httplib::Request& req, httplib::Response& res) {
                auto c = registry.get(req.matches[1]);
                const auto ids = parse_ids(parse_body(req).at("ids"));
                send_json(res, {{"deleted", c->remove(ids.data(), ids.size())}});
              }));

  server.Get(C + R"(/points/(\d+))", route("get", [&](const httplib::Request& req, httplib::Response& res) {
               auto c = registry.get(req.matches[1]);
               const auto id = std::stoull(req.matches[2]);
               std::vector<float> vec;
               std::string md;
               if (!c->get(id, &vec, &md)) throw HttpError{404, "id not found"};
               send_json(res, {{"id", id}, {"vector", vec}, {"metadata", json::parse(md)}});
             }));

  server.Post(C + "/checkpoint", route("checkpoint", [&](const httplib::Request& req, httplib::Response& res) {
                auto c = registry.get(req.matches[1]);
                c->checkpoint();
                send_json(res, json::parse(c->stats_json()));
              }));

  std::signal(SIGINT, on_signal);
  std::signal(SIGTERM, on_signal);
  std::thread watcher([&] {
    while (!g_stop) std::this_thread::sleep_for(std::chrono::milliseconds(100));
    server.stop();
  });

  std::cerr << "strata-server " << strata::kVersion << " (" << strata::kernels::simd_backend() << ") listening on "
            << host << ":" << port << ", data in " << fs::absolute(data_dir) << ", " << threads << " threads"
            << (sync_wal ? "" : ", fsync disabled") << "\n";
  if (!server.listen(host, port)) {
    std::cerr << "failed to bind " << host << ":" << port << "\n";
    g_stop = true;
    watcher.join();
    return 1;
  }
  g_stop = true;
  watcher.join();
  std::cerr << "shutting down: checkpointing collections...\n";
  for (const auto& c : registry.all()) c->checkpoint();
  std::cerr << "bye\n";
  return 0;
}
