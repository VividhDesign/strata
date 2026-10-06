#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata {

constexpr const char* kVersion = "0.2.0";

// External, user-facing identifier of a vector.
using label_t = uint64_t;
// Internal, dense identifier of a graph node (index into storage arrays).
using node_t = uint32_t;

constexpr node_t kInvalidNode = std::numeric_limits<node_t>::max();
// Returned in place of a label when fewer than k results exist (-1 when viewed as int64).
constexpr label_t kNoLabel = std::numeric_limits<label_t>::max();

enum class Metric : uint32_t {
  L2 = 0,            // squared Euclidean distance
  InnerProduct = 1,  // distance = 1 - <a, b>
  Cosine = 2,        // vectors are L2-normalised on insert/query, then 1 - <a, b>
};

const char* metric_name(Metric m);
Metric parse_metric(const std::string& name);

struct SearchResult {
  label_t label;
  float distance;
};

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

}  // namespace strata
