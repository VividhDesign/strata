// pybind11 bindings. NumPy arrays are passed to C++ without copying, and the GIL is
// released during builds and searches so Python threads can run alongside.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <string>
#include <vector>

#include "strata/collection.h"
#include "strata/flat.h"
#include "strata/hnsw.h"

namespace py = pybind11;
using namespace strata;

namespace {

using FloatArray = py::array_t<float, py::array::c_style | py::array::forcecast>;
using IdArray = py::array_t<int64_t, py::array::c_style | py::array::forcecast>;

struct Matrix {
  const float* data;
  size_t rows;
};

Matrix as_matrix(const FloatArray& a, size_t dim, const char* what) {
  if (a.ndim() == 1 && static_cast<size_t>(a.shape(0)) == dim) return {a.data(), 1};
  if (a.ndim() == 2 && static_cast<size_t>(a.shape(1)) == dim) return {a.data(), static_cast<size_t>(a.shape(0))};
  throw py::value_error(std::string(what) + " must have shape (n, " + std::to_string(dim) + ") or (" +
                        std::to_string(dim) + ",)");
}

std::vector<label_t> as_labels(const py::object& ids, size_t n, label_t auto_start) {
  std::vector<label_t> out(n);
  if (ids.is_none()) {
    for (size_t i = 0; i < n; ++i) out[i] = auto_start + i;
    return out;
  }
  const IdArray arr = py::cast<IdArray>(ids);
  if (arr.ndim() != 1 || static_cast<size_t>(arr.shape(0)) != n) throw py::value_error("ids must be a 1-D array with one id per vector");
  for (size_t i = 0; i < n; ++i) {
    if (arr.data()[i] < 0) throw py::value_error("ids must be non-negative");
    out[i] = static_cast<label_t>(arr.data()[i]);
  }
  return out;
}

std::unique_ptr<LabelFilter> make_filter(const py::object& allow, const py::object& deny) {
  if (!allow.is_none() && !deny.is_none()) throw py::value_error("pass either filter_ids or exclude_ids, not both");
  const py::object& src = allow.is_none() ? deny : allow;
  if (src.is_none()) return nullptr;
  const IdArray arr = py::cast<IdArray>(src);
  std::vector<label_t> ids(arr.data(), arr.data() + arr.size());
  return std::make_unique<LabelFilter>(ids.data(), ids.size(),
                                       allow.is_none() ? LabelFilter::Mode::Deny : LabelFilter::Mode::Allow);
}

// Converts (labels, distances) buffers into NumPy arrays; kNoLabel becomes -1.
py::tuple to_numpy(std::vector<label_t>& labels, std::vector<float>& dists, size_t nq, size_t k) {
  py::array_t<int64_t> l({nq, k});
  py::array_t<float> d({nq, k});
  auto* lp = l.mutable_data();
  auto* dp = d.mutable_data();
  for (size_t i = 0; i < nq * k; ++i) {
    lp[i] = labels[i] == kNoLabel ? -1 : static_cast<int64_t>(labels[i]);
    dp[i] = dists[i];
  }
  return py::make_tuple(l, d);
}

}  // namespace

PYBIND11_MODULE(_strata, m) {
  m.doc() = "Strata: an HNSW vector database written from scratch in C++";
  m.attr("__version__") = STRATA_VERSION;
  m.def("simd_backend", &kernels::simd_backend, "SIMD kernel set compiled in: neon, avx2 or scalar");

  py::register_exception<Error>(m, "StrataError", PyExc_RuntimeError);

  py::class_<HNSWIndex>(m, "Index")
      .def(py::init([](size_t dim, const std::string& metric, size_t M, size_t ef_construction, uint64_t seed,
                       bool use_heuristic, size_t capacity, const std::string& quantization, bool rerank) {
             HNSWParams p{M, ef_construction, seed, use_heuristic};
             p.quantization = parse_quantization(quantization);
             p.rerank = rerank;
             return std::make_unique<HNSWIndex>(dim, parse_metric(metric), p, capacity);
           }),
           py::arg("dim"), py::arg("metric") = "l2", py::arg("M") = 16, py::arg("ef_construction") = 200,
           py::arg("seed") = 42, py::arg("use_heuristic") = true, py::arg("capacity") = 1024,
           py::arg("quantization") = "none", py::arg("rerank") = true)
      .def(
          "add",
          [](HNSWIndex& self, const FloatArray& vectors, const py::object& ids, int num_threads) {
            const Matrix mat = as_matrix(vectors, self.dim(), "vectors");
            const auto labels = as_labels(ids, mat.rows, self.stats().nodes);
            py::gil_scoped_release release;
            self.add(mat.data, labels.data(), mat.rows, num_threads);
          },
          py::arg("vectors"), py::arg("ids") = py::none(), py::arg("num_threads") = 0,
          "Insert (or replace) vectors. ids default to consecutive integers.")
      .def(
          "search",
          [](const HNSWIndex& self, const FloatArray& queries, size_t k, size_t ef, int num_threads,
             const py::object& filter_ids, const py::object& exclude_ids) {
            const Matrix mat = as_matrix(queries, self.dim(), "queries");
            const auto filter = make_filter(filter_ids, exclude_ids);
            std::vector<label_t> labels(mat.rows * k);
            std::vector<float> dists(mat.rows * k);
            {
              py::gil_scoped_release release;
              self.search_batch(mat.data, mat.rows, k, ef, num_threads, filter.get(), labels.data(), dists.data());
            }
            return to_numpy(labels, dists, mat.rows, k);
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 0, py::arg("num_threads") = 0,
          py::arg("filter_ids") = py::none(), py::arg("exclude_ids") = py::none(),
          "Returns (ids, distances), each of shape (n_queries, k). Missing results are -1 / inf.")
      .def(
          "remove",
          [](HNSWIndex& self, const IdArray& ids) {
            size_t removed = 0;
            for (py::ssize_t i = 0; i < ids.size(); ++i) removed += self.remove(static_cast<label_t>(ids.data()[i]));
            return removed;
          },
          py::arg("ids"), "Delete ids (tombstones). Returns how many existed.")
      .def("contains", &HNSWIndex::contains)
      .def("__contains__", &HNSWIndex::contains)
      .def("__len__", &HNSWIndex::size)
      .def(
          "get_vector",
          [](const HNSWIndex& self, label_t id) {
            py::array_t<float> out(self.dim());
            if (!self.get_vector(id, out.mutable_data())) throw py::key_error(std::to_string(id));
            return out;
          },
          py::arg("id"))
      .def("ids", [](const HNSWIndex& self) {
        const auto l = self.labels();
        return py::array_t<int64_t>(l.size(), reinterpret_cast<const int64_t*>(l.data()));
      })
      .def(
          "compact",
          [](HNSWIndex& self, int num_threads) {
            py::gil_scoped_release release;
            self.compact(num_threads);
          },
          py::arg("num_threads") = 0)
      .def(
          "save",
          [](const HNSWIndex& self, const std::string& path) {
            py::gil_scoped_release release;
            self.save(path);
          },
          py::arg("path"))
      .def_static(
          "load",
          [](const std::string& path, bool mmap_vectors) {
            py::gil_scoped_release release;
            return HNSWIndex::load(path, mmap_vectors);
          },
          py::arg("path"), py::arg("mmap_vectors") = false,
          "Load a saved index. mmap_vectors=True keeps the float32 re-rank vectors of an SQ8 index "
          "on disk (memory-mapped) instead of in RAM.")
      .def_property("ef_search", &HNSWIndex::ef_search, &HNSWIndex::set_ef_search)
      .def_property("flat_search_cutoff", &HNSWIndex::flat_search_cutoff, &HNSWIndex::set_flat_search_cutoff)
      .def_property_readonly("dim", &HNSWIndex::dim)
      .def_property_readonly("metric", [](const HNSWIndex& self) { return metric_name(self.metric()); })
      .def_property_readonly("M", [](const HNSWIndex& self) { return self.params().M; })
      .def_property_readonly("ef_construction", [](const HNSWIndex& self) { return self.params().ef_construction; })
      .def_property_readonly("quantization",
                             [](const HNSWIndex& self) { return quantization_name(self.params().quantization); })
      .def_property_readonly("rerank", [](const HNSWIndex& self) { return self.params().rerank; })
      .def("stats", [](const HNSWIndex& self) {
        const HNSWStats s = self.stats();
        py::dict d;
        d["size"] = s.size;
        d["nodes"] = s.nodes;
        d["deleted"] = s.deleted;
        d["capacity"] = s.capacity;
        d["max_level"] = s.max_level;
        d["mean_degree_level0"] = s.mean_degree_level0;
        d["memory_bytes"] = s.memory_bytes;
        return d;
      })
      .def("__repr__", [](const HNSWIndex& self) {
        return "<strata.Index dim=" + std::to_string(self.dim()) + " metric=" + metric_name(self.metric()) +
               " size=" + std::to_string(self.size()) + " M=" + std::to_string(self.params().M) + ">";
      });

  py::class_<FlatIndex>(m, "FlatIndex")
      .def(py::init([](size_t dim, const std::string& metric) {
             return std::make_unique<FlatIndex>(dim, parse_metric(metric));
           }),
           py::arg("dim"), py::arg("metric") = "l2")
      .def(
          "add",
          [](FlatIndex& self, const FloatArray& vectors, const py::object& ids) {
            const Matrix mat = as_matrix(vectors, self.dim(), "vectors");
            const auto labels = as_labels(ids, mat.rows, self.size());
            self.add(mat.data, labels.data(), mat.rows);
          },
          py::arg("vectors"), py::arg("ids") = py::none())
      .def(
          "search",
          [](const FlatIndex& self, const FloatArray& queries, size_t k, int num_threads, const py::object& filter_ids,
             const py::object& exclude_ids) {
            const Matrix mat = as_matrix(queries, self.dim(), "queries");
            const auto filter = make_filter(filter_ids, exclude_ids);
            std::vector<label_t> labels(mat.rows * k);
            std::vector<float> dists(mat.rows * k);
            {
              py::gil_scoped_release release;
              self.search_batch(mat.data, mat.rows, k, num_threads, filter.get(), labels.data(), dists.data());
            }
            return to_numpy(labels, dists, mat.rows, k);
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("num_threads") = 0, py::arg("filter_ids") = py::none(),
          py::arg("exclude_ids") = py::none())
      .def("__len__", &FlatIndex::size)
      .def_property_readonly("dim", &FlatIndex::dim);

  py::class_<Collection>(m, "_Collection")
      .def_static(
          "create",
          [](const std::string& path, size_t dim, const std::string& metric, size_t M, size_t ef_construction,
             bool sync_wal, const std::string& quantization, bool rerank) {
            CollectionConfig cfg;
            cfg.name = path;
            cfg.dim = dim;
            cfg.metric = parse_metric(metric);
            cfg.params.M = M;
            cfg.params.ef_construction = ef_construction;
            cfg.params.quantization = parse_quantization(quantization);
            cfg.params.rerank = rerank;
            cfg.sync_wal = sync_wal;
            return Collection::create(path, cfg);
          },
          py::arg("path"), py::arg("dim"), py::arg("metric") = "cosine", py::arg("M") = 16,
          py::arg("ef_construction") = 200, py::arg("sync_wal") = true, py::arg("quantization") = "none",
          py::arg("rerank") = true)
      .def_static("open", &Collection::open, py::arg("path"), py::arg("sync_wal") = true)
      .def(
          "upsert",
          [](Collection& self, const IdArray& ids, const FloatArray& vectors, const std::vector<std::string>& metadata) {
            const Matrix mat = as_matrix(vectors, self.config().dim, "vectors");
            const auto labels = as_labels(ids, mat.rows, 0);
            py::gil_scoped_release release;
            self.upsert(labels.data(), mat.data, mat.rows, metadata);
          },
          py::arg("ids"), py::arg("vectors"), py::arg("metadata"))
      .def(
          "remove",
          [](Collection& self, const IdArray& ids) {
            const auto labels = as_labels(ids, static_cast<size_t>(ids.size()), 0);
            return self.remove(labels.data(), labels.size());
          },
          py::arg("ids"))
      .def(
          "query",
          [](const Collection& self, const FloatArray& queries, size_t k, size_t ef, const std::string& filter) {
            const Matrix mat = as_matrix(queries, self.config().dim, "queries");
            std::vector<std::vector<Hit>> hits;
            {
              py::gil_scoped_release release;
              hits = self.query(mat.data, mat.rows, k, ef, filter);
            }
            py::list out;
            for (const auto& row : hits) {
              py::list r;
              for (const auto& h : row) r.append(py::make_tuple(static_cast<int64_t>(h.id), h.distance, h.metadata));
              out.append(r);
            }
            return out;
          },
          py::arg("queries"), py::arg("k") = 10, py::arg("ef") = 0, py::arg("filter") = "")
      .def(
          "get",
          [](const Collection& self, label_t id) -> py::object {
            std::vector<float> v;
            std::string md;
            if (!self.get(id, &v, &md)) return py::none();
            return py::make_tuple(py::array_t<float>(v.size(), v.data()), md);
          },
          py::arg("id"))
      .def("checkpoint", &Collection::checkpoint, py::call_guard<py::gil_scoped_release>())
      .def("stats_json", &Collection::stats_json)
      .def("set_ef_search", &Collection::set_ef_search)
      .def("__len__", &Collection::size)
      .def_property_readonly("dim", [](const Collection& self) { return self.config().dim; });
}
