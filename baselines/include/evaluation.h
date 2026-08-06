#pragma once

#include <algorithm>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(_WIN32)
#include <windows.h>
#include <psapi.h>
#elif defined(__APPLE__) && defined(__MACH__)
#include <mach/mach.h>
#include <sys/resource.h>
#elif defined(__linux__)
#include <sys/resource.h>
#include <unistd.h>
#elif defined(__unix__) || defined(__unix) || defined(unix)
#include <sys/resource.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

// CMake replaces this with a SHA-256 over the runner, shared evaluation support, and
// effective locally patched upstream sources compiled into each executable.
// Keep a fallback so the header remains usable outside the provided build.
#ifndef TDVS_SOURCE_FINGERPRINT
#define TDVS_SOURCE_FINGERPRINT "unavailable"
#endif

namespace tdvs_mips {

using SteadyClock = std::chrono::steady_clock;
inline constexpr unsigned kDefaultBuildThreads = 64;

inline size_t ParseDecimalSize(const std::string& text,
                               const std::string& description) {
  if (text.empty()) {
    throw std::invalid_argument("Invalid integer value for " + description +
                                ": " + text);
  }
  size_t value = 0;
  for (unsigned char character : text) {
    if (!std::isdigit(character)) {
      throw std::invalid_argument("Invalid integer value for " + description +
                                  ": " + text);
    }
    const size_t digit = static_cast<size_t>(character - '0');
    if (value > (std::numeric_limits<size_t>::max() - digit) / 10) {
      throw std::invalid_argument("Integer value for " + description +
                                  " is too large: " + text);
    }
    value = value * 10 + digit;
  }
  return value;
}

class Args {
 public:
  Args(int argc, char** argv) {
    for (int i = 1; i < argc; ++i) {
      std::string key(argv[i]);
      if (key.rfind("--", 0) != 0) {
        throw std::invalid_argument("Unexpected positional argument: " + key);
      }
      key.erase(0, 2);
      if (key.empty()) {
        throw std::invalid_argument("Empty option name");
      }
      if (i + 1 < argc && std::string(argv[i + 1]).rfind("--", 0) != 0) {
        values_[key] = argv[++i];
      } else {
        values_[key] = "true";
      }
    }
  }

  bool Has(const std::string& key) const { return values_.count(key) != 0; }

  std::string Require(const std::string& key) const {
    const auto it = values_.find(key);
    if (it == values_.end()) {
      throw std::invalid_argument("Missing required option --" + key);
    }
    return it->second;
  }

  std::string Get(const std::string& key, const std::string& fallback) const {
    const auto it = values_.find(key);
    return it == values_.end() ? fallback : it->second;
  }

  size_t GetSize(const std::string& key, size_t fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    return ParseSize(key, it->second);
  }

  size_t RequireSize(const std::string& key) const {
    return ParseSize(key, Require(key));
  }

  unsigned GetUnsigned(const std::string& key, unsigned fallback) const {
    const size_t value = GetSize(key, fallback);
    if (value > std::numeric_limits<unsigned>::max()) {
      throw std::invalid_argument("--" + key + " is too large");
    }
    return static_cast<unsigned>(value);
  }

  double GetDouble(const std::string& key, double fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    size_t consumed = 0;
    const double value = std::stod(it->second, &consumed);
    if (consumed != it->second.size() || !std::isfinite(value)) {
      throw std::invalid_argument("Invalid numeric value for --" + key + ": " +
                                  it->second);
    }
    return value;
  }

  bool GetBool(const std::string& key, bool fallback) const {
    const auto it = values_.find(key);
    if (it == values_.end()) return fallback;
    std::string value = it->second;
    std::transform(value.begin(), value.end(), value.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "1" || value == "true" || value == "yes" || value == "on") return true;
    if (value == "0" || value == "false" || value == "no" || value == "off") return false;
    throw std::invalid_argument("Invalid boolean value for --" + key + ": " + value);
  }

 private:
  static size_t ParseSize(const std::string& key, const std::string& text) {
    return ParseDecimalSize(text, "--" + key);
  }

  std::map<std::string, std::string> values_;
};

inline uint64_t FileSize(const std::string& path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) throw std::runtime_error("Cannot open file: " + path);
  const std::streampos end = input.tellg();
  if (end < 0) throw std::runtime_error("Cannot determine file size: " + path);
  return static_cast<uint64_t>(end);
}

using IndexMetadata = std::map<std::string, std::string>;

inline void WriteIndexMetadata(const std::string& path, const std::string& magic,
                               const IndexMetadata& fields) {
  std::ofstream output(path, std::ios::trunc);
  if (!output) throw std::runtime_error("Cannot write index metadata: " + path);
  output << magic << '\n';
  for (const auto& [key, value] : fields) output << key << '=' << value << '\n';
  output.flush();
  if (!output) throw std::runtime_error("Failed to write index metadata: " + path);
}

inline IndexMetadata ReadIndexMetadata(const std::string& path,
                                       const std::string& expected_magic) {
  std::ifstream input(path);
  if (!input) throw std::runtime_error("Cannot open index metadata: " + path);
  std::string line;
  if (!std::getline(input, line) || line != expected_magic) {
    throw std::runtime_error("Invalid index metadata header: " + path);
  }
  IndexMetadata fields;
  while (std::getline(input, line)) {
    if (line.empty()) continue;
    const size_t equals = line.find('=');
    if (equals == std::string::npos || equals == 0) {
      throw std::runtime_error("Malformed index metadata line in " + path);
    }
    fields[line.substr(0, equals)] = line.substr(equals + 1);
  }
  return fields;
}

inline void RequireIndexMetadataValue(const IndexMetadata& fields,
                                      const std::string& key,
                                      const std::string& expected) {
  const auto found = fields.find(key);
  if (found == fields.end() || found->second != expected) {
    throw std::runtime_error("Index metadata mismatch for " + key);
  }
}

inline void RequireIndexMetadataSize(const IndexMetadata& fields,
                                     const std::string& key, size_t expected) {
  const auto found = fields.find(key);
  if (found == fields.end()) {
    throw std::runtime_error("Missing index metadata field: " + key);
  }
  size_t value = 0;
  try {
    value = ParseDecimalSize(found->second, "index metadata field " + key);
  } catch (const std::invalid_argument&) {
    throw std::runtime_error("Invalid index metadata field: " + key);
  }
  if (value != expected) {
    throw std::runtime_error("Index metadata mismatch for " + key);
  }
}

struct KnngBuildTiming {
  bool known = false;
  std::string source = "unknown";
  std::string manifest_path;
  double graph_build_seconds = 0.0;
  double graph_extract_seconds = 0.0;
  double graph_write_seconds = 0.0;
  double total_seconds = 0.0;
};

inline bool ParseJsonNumberField(const std::string& text, const std::string& key,
                                 double* value) {
  const std::string quoted_key = "\"" + key + "\"";
  const size_t key_pos = text.find(quoted_key);
  if (key_pos == std::string::npos) return false;
  const size_t colon = text.find(':', key_pos + quoted_key.size());
  if (colon == std::string::npos) return false;
  const char* begin = text.c_str() + colon + 1;
  char* end = nullptr;
  const double parsed = std::strtod(begin, &end);
  if (end == begin || !std::isfinite(parsed)) return false;
  *value = parsed;
  return true;
}

inline bool ParseJsonStringField(const std::string& text,
                                 const std::string& key,
                                 std::string* value) {
  const std::string quoted_key = "\"" + key + "\"";
  const size_t key_pos = text.find(quoted_key);
  if (key_pos == std::string::npos) return false;
  const size_t colon = text.find(':', key_pos + quoted_key.size());
  if (colon == std::string::npos) return false;
  const size_t begin = text.find('"', colon + 1);
  if (begin == std::string::npos) return false;
  const size_t end = text.find('"', begin + 1);
  if (end == std::string::npos) return false;
  *value = text.substr(begin + 1, end - begin - 1);
  return true;
}

// The shared kNN graph is a required preprocessing stage for MAG and PSP.
// Prefer an explicit legacy override when supplied; otherwise automatically
// consume the manifest written by build_knng.py.  This keeps the
// algorithm-specific core timer separate while making the end-to-end index
// construction time impossible to omit accidentally in normal runs.
inline KnngBuildTiming ResolveKnngBuildTiming(const Args& args,
                                              const std::string& knng_path) {
  KnngBuildTiming timing;
  if (args.Has("knng-build-seconds")) {
    timing.total_seconds = args.GetDouble("knng-build-seconds", 0.0);
    if (timing.total_seconds < 0.0) {
      throw std::invalid_argument("--knng-build-seconds must be non-negative");
    }
    timing.known = true;
    timing.source = "cli_override";
    return timing;
  }

  timing.manifest_path = args.Get("knng-manifest", knng_path + ".json");
  std::ifstream input(timing.manifest_path);
  if (!input) return timing;
  std::ostringstream buffer;
  buffer << input.rdbuf();
  const std::string text = buffer.str();
  std::string method;
  double output_bytes = 0.0;
  double manifest_total_seconds = 0.0;
  if (!ParseJsonStringField(text, "method", &method) ||
      !ParseJsonNumberField(text, "graph_build_seconds",
                            &timing.graph_build_seconds) ||
      !ParseJsonNumberField(text, "graph_extract_seconds",
                            &timing.graph_extract_seconds) ||
      !ParseJsonNumberField(text, "graph_write_seconds",
                            &timing.graph_write_seconds) ||
      !ParseJsonNumberField(text, "total_seconds", &manifest_total_seconds) ||
      !ParseJsonNumberField(text, "output_bytes", &output_bytes)) {
    throw std::runtime_error("Malformed kNN graph timing manifest: " +
                             timing.manifest_path);
  }
  if (method != "faiss_nndescent") {
    throw std::runtime_error(
        "Unsupported kNN graph generator in manifest (expected "
        "faiss_nndescent): " + timing.manifest_path);
  }
  if (timing.graph_build_seconds < 0.0 ||
      timing.graph_extract_seconds < 0.0 ||
      timing.graph_write_seconds < 0.0 || manifest_total_seconds < 0.0 ||
      output_bytes < 0.0) {
    throw std::runtime_error("Negative value in kNN graph timing manifest: " +
                             timing.manifest_path);
  }
  const uint64_t manifest_bytes = static_cast<uint64_t>(output_bytes);
  if (manifest_bytes != FileSize(knng_path)) {
    throw std::runtime_error(
        "kNN graph timing manifest output_bytes does not match graph file");
  }
  timing.total_seconds = timing.graph_build_seconds +
                         timing.graph_extract_seconds +
                         timing.graph_write_seconds;
  const double total_tolerance =
      1e-9 * std::max(1.0, std::abs(manifest_total_seconds));
  if (std::abs(timing.total_seconds - manifest_total_seconds) >
      total_tolerance) {
    throw std::runtime_error(
        "kNN graph timing manifest total_seconds does not match its phases");
  }
  timing.known = true;
  timing.source = "adjacent_manifest";
  return timing;
}

// These process-level measurements intentionally live in the shared evaluation layer so
// every method uses the same OS API and byte units.  A footprint run must be a
// fresh, load-only process: current RSS is otherwise affected by allocator
// history from an earlier build or another loaded index.
inline uint64_t CurrentRSSBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS info;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) return 0;
  return static_cast<uint64_t>(info.WorkingSetSize);
#elif defined(__APPLE__) && defined(__MACH__)
  mach_task_basic_info info;
  mach_msg_type_number_t count = MACH_TASK_BASIC_INFO_COUNT;
  if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO,
                reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS) {
    return 0;
  }
  return static_cast<uint64_t>(info.resident_size);
#elif defined(__linux__)
  FILE* input = std::fopen("/proc/self/statm", "r");
  if (input == nullptr) return 0;
  long resident_pages = 0;
  const int scanned = std::fscanf(input, "%*s%ld", &resident_pages);
  std::fclose(input);
  if (scanned != 1 || resident_pages < 0) return 0;
  const long page_bytes = sysconf(_SC_PAGESIZE);
  if (page_bytes <= 0) return 0;
  return static_cast<uint64_t>(resident_pages) *
         static_cast<uint64_t>(page_bytes);
#else
  return 0;
#endif
}

inline uint64_t PeakRSSBytes() {
#if defined(_WIN32)
  PROCESS_MEMORY_COUNTERS info;
  if (!GetProcessMemoryInfo(GetCurrentProcess(), &info, sizeof(info))) return 0;
  return static_cast<uint64_t>(info.PeakWorkingSetSize);
#elif defined(__APPLE__) && defined(__MACH__)
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
  // Darwin reports ru_maxrss in bytes.
  return static_cast<uint64_t>(usage.ru_maxrss);
#elif defined(__unix__) || defined(__unix) || defined(unix) || defined(__linux__)
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) != 0) return 0;
  // Linux and the other supported Unix implementations report KiB.
  return static_cast<uint64_t>(usage.ru_maxrss) * 1024ULL;
#else
  return 0;
#endif
}

inline uint64_t RSSDeltaBytes(uint64_t after, uint64_t before) {
  return after >= before ? after - before : 0;
}

inline void ReadExact(std::ifstream& input, char* destination, uint64_t bytes,
                      const std::string& path) {
  constexpr uint64_t kChunk = 256ULL * 1024ULL * 1024ULL;
  uint64_t offset = 0;
  while (offset < bytes) {
    const uint64_t remaining = bytes - offset;
    const std::streamsize count = static_cast<std::streamsize>(std::min(kChunk, remaining));
    input.read(destination + offset, count);
    if (input.gcount() != count) {
      throw std::runtime_error("Short read from " + path);
    }
    offset += static_cast<uint64_t>(count);
  }
}

struct FloatMatrix {
  size_t rows = 0;
  size_t dim = 0;
  size_t stride = 0;
  std::vector<float> values;

  float* Row(size_t row) { return values.data() + row * stride; }
  const float* Row(size_t row) const { return values.data() + row * stride; }
};

inline FloatMatrix LoadRawFloatMatrix(const std::string& path, size_t dim,
                                      size_t expected_rows = 0) {
  if (dim == 0) throw std::invalid_argument("Vector dimension must be positive");
  const uint64_t bytes = FileSize(path);
  const uint64_t row_bytes = static_cast<uint64_t>(dim) * sizeof(float);
  if (bytes == 0 || bytes % row_bytes != 0) {
    throw std::runtime_error(path + " is not a headerless float32 matrix with dim=" +
                             std::to_string(dim));
  }
  const size_t rows = static_cast<size_t>(bytes / row_bytes);
  if (expected_rows != 0 && rows != expected_rows) {
    throw std::runtime_error(path + " has " + std::to_string(rows) +
                             " rows; expected " + std::to_string(expected_rows));
  }

  FloatMatrix matrix;
  matrix.rows = rows;
  matrix.dim = dim;
  matrix.stride = (dim + 7U) & ~size_t(7U);
  if (rows > std::numeric_limits<size_t>::max() / matrix.stride) {
    throw std::runtime_error("Matrix is too large: " + path);
  }
  matrix.values.assign(rows * matrix.stride, 0.0f);

  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open file: " + path);
  if (matrix.stride == dim) {
    ReadExact(input, reinterpret_cast<char*>(matrix.values.data()), bytes, path);
  } else {
    for (size_t row = 0; row < rows; ++row) {
      input.read(reinterpret_cast<char*>(matrix.Row(row)),
                 static_cast<std::streamsize>(row_bytes));
      if (input.gcount() != static_cast<std::streamsize>(row_bytes)) {
        throw std::runtime_error("Short read from " + path + " at row " +
                                 std::to_string(row));
      }
    }
  }
  return matrix;
}

inline void NormalizeRows(FloatMatrix& matrix) {
  for (size_t row = 0; row < matrix.rows; ++row) {
    double squared_norm = 0.0;
    float* vector = matrix.Row(row);
    for (size_t d = 0; d < matrix.dim; ++d) {
      squared_norm += static_cast<double>(vector[d]) * vector[d];
    }
    if (!(squared_norm > 0.0) || !std::isfinite(squared_norm)) {
      throw std::runtime_error("Cannot normalize row " + std::to_string(row));
    }
    const float inverse_norm = static_cast<float>(1.0 / std::sqrt(squared_norm));
    for (size_t d = 0; d < matrix.dim; ++d) vector[d] *= inverse_norm;
  }
}

struct GroundTruth {
  size_t rows = 0;
  size_t width = 0;
  std::vector<uint64_t> ids;

  const uint64_t* Row(size_t row) const { return ids.data() + row * width; }
};

inline GroundTruth LoadGroundTruth(const std::string& path, size_t rows, size_t width,
                                   const std::string& type, size_t base_rows) {
  if (rows == 0 || width == 0) throw std::invalid_argument("Invalid ground-truth shape");
  const size_t element_bytes = type == "int64" ? sizeof(int64_t)
                              : type == "uint32" ? sizeof(uint32_t)
                                                   : 0;
  if (element_bytes == 0) {
    throw std::invalid_argument("--gt-type must be int64 or uint32");
  }
  const uint64_t expected = static_cast<uint64_t>(rows) * width * element_bytes;
  if (FileSize(path) != expected) {
    throw std::runtime_error(path + " size does not match " + std::to_string(rows) + "x" +
                             std::to_string(width) + " " + type);
  }

  GroundTruth gt;
  gt.rows = rows;
  gt.width = width;
  gt.ids.resize(rows * width);
  std::ifstream input(path, std::ios::binary);
  if (!input) throw std::runtime_error("Cannot open file: " + path);
  if (type == "int64") {
    std::vector<int64_t> source(rows * width);
    ReadExact(input, reinterpret_cast<char*>(source.data()), expected, path);
    for (size_t i = 0; i < source.size(); ++i) {
      if (source[i] < 0 || static_cast<uint64_t>(source[i]) >= base_rows) {
        throw std::runtime_error("Ground-truth ID out of range at element " +
                                 std::to_string(i));
      }
      gt.ids[i] = static_cast<uint64_t>(source[i]);
    }
  } else {
    std::vector<uint32_t> source(rows * width);
    ReadExact(input, reinterpret_cast<char*>(source.data()), expected, path);
    for (size_t i = 0; i < source.size(); ++i) {
      if (source[i] >= base_rows) {
        throw std::runtime_error("Ground-truth ID out of range at element " +
                                 std::to_string(i));
      }
      gt.ids[i] = source[i];
    }
  }
  return gt;
}

struct KnngInfo {
  size_t rows = 0;
  unsigned degree = 0;
  uint64_t bytes = 0;
};

// MAG and PSP use repeated fixed-width rows: [uint32 degree][degree uint32 IDs].
// There is no separate global header; the first uint32 is row 0's degree.
inline KnngInfo ValidateKnng(const std::string& path, size_t expected_rows) {
  const uint64_t bytes = FileSize(path);
  if (bytes < sizeof(uint32_t)) throw std::runtime_error("Empty kNN graph: " + path);
  std::ifstream input(path, std::ios::binary);
  uint32_t degree = 0;
  input.read(reinterpret_cast<char*>(&degree), sizeof(degree));
  if (!input || degree == 0) throw std::runtime_error("Invalid kNN degree in " + path);
  const uint64_t row_words = static_cast<uint64_t>(degree) + 1;
  const uint64_t row_bytes = row_words * sizeof(uint32_t);
  if (bytes % row_bytes != 0) {
    throw std::runtime_error("kNN graph size is inconsistent with its row degree: " + path);
  }
  const uint64_t row_count = bytes / row_bytes;
  if (row_count > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("kNN graph row count exceeds the supported range: " + path);
  }
  const size_t rows = static_cast<size_t>(row_count);
  if (rows != expected_rows) {
    throw std::runtime_error("kNN graph has " + std::to_string(rows) +
                             " rows; expected " + std::to_string(expected_rows));
  }

  if (row_words > std::numeric_limits<size_t>::max()) {
    throw std::runtime_error("kNN graph degree exceeds the supported range: " + path);
  }
  input.clear();
  input.seekg(0);
  if (!input) throw std::runtime_error("Cannot seek in kNN graph: " + path);

  constexpr uint64_t kTargetChunkBytes = 64ULL * 1024ULL * 1024ULL;
  const size_t words_per_row = static_cast<size_t>(row_words);
  const size_t rows_per_chunk = static_cast<size_t>(
      std::max<uint64_t>(1, kTargetChunkBytes / row_bytes));
  if (rows_per_chunk > std::numeric_limits<size_t>::max() / words_per_row) {
    throw std::runtime_error("kNN graph validation buffer is too large: " + path);
  }
  std::vector<uint32_t> chunk(rows_per_chunk * words_per_row);
  for (size_t row_begin = 0; row_begin < rows;) {
    const size_t chunk_rows = std::min(rows_per_chunk, rows - row_begin);
    const uint64_t chunk_bytes =
        static_cast<uint64_t>(chunk_rows) * row_bytes;
    ReadExact(input, reinterpret_cast<char*>(chunk.data()), chunk_bytes, path);
    for (size_t local_row = 0; local_row < chunk_rows; ++local_row) {
      const uint32_t* row = chunk.data() + local_row * words_per_row;
      const size_t global_row = row_begin + local_row;
      if (row[0] != degree) {
        throw std::runtime_error("Malformed kNN graph row " +
                                 std::to_string(global_row));
      }
      for (size_t neighbor = 1; neighbor < words_per_row; ++neighbor) {
        if (row[neighbor] >= rows) {
          throw std::runtime_error("Out-of-range kNN ID at row " +
                                   std::to_string(global_row));
        }
      }
    }
    row_begin += chunk_rows;
  }
  return KnngInfo{rows, degree, bytes};
}

inline std::vector<size_t> ParseSizeList(const std::string& text) {
  if (text.empty() || text.front() == ',' || text.back() == ',' ||
      text.find(",,") != std::string::npos) {
    throw std::invalid_argument("Invalid search list: " + text);
  }
  std::vector<size_t> values;
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    size_t value = 0;
    try {
      value = ParseDecimalSize(token, "search list element");
    } catch (const std::invalid_argument&) {
      throw std::invalid_argument("Invalid search list: " + text);
    }
    if (value == 0) throw std::invalid_argument("Invalid search list: " + text);
    values.push_back(value);
  }
  if (values.empty()) throw std::invalid_argument("Search list is empty");
  return values;
}

inline void ConfigureOpenMP(unsigned threads) {
  if (threads == 0) throw std::invalid_argument("Thread count must be positive");
#ifdef _OPENMP
  omp_set_dynamic(0);
  omp_set_num_threads(static_cast<int>(threads));
#else
  if (threads != 1) {
    throw std::runtime_error(
        "Requested multi-thread construction, but this binary was built "
        "without OpenMP. Install/configure OpenMP or pass --build-threads 1 "
        "for an explicitly serial run.");
  }
#endif
}

inline double RecallAtK(const GroundTruth& gt, const std::vector<unsigned>& results,
                        size_t k) {
  if (gt.rows * k != results.size()) throw std::logic_error("Result shape mismatch");
  uint64_t hits = 0;
  for (size_t row = 0; row < gt.rows; ++row) {
    std::unordered_set<uint64_t> expected;
    expected.reserve(k * 2);
    for (size_t j = 0; j < k; ++j) expected.insert(gt.Row(row)[j]);
    std::unordered_set<unsigned> seen;
    seen.reserve(k * 2);
    for (size_t j = 0; j < k; ++j) {
      const unsigned id = results[row * k + j];
      if (seen.insert(id).second && expected.count(id) != 0) ++hits;
    }
  }
  return static_cast<double>(hits) / static_cast<double>(gt.rows * k);
}

inline void ValidateSearchResultIds(const unsigned* results, size_t count,
                                    size_t base_rows) {
  std::unordered_set<unsigned> unique_ids;
  unique_ids.reserve(count);
  for (size_t position = 0; position < count; ++position) {
    const unsigned id = results[position];
    if (static_cast<size_t>(id) >= base_rows) {
      throw std::runtime_error(
          "Search returned an out-of-range result ID at rank " +
          std::to_string(position));
    }
    if (!unique_ids.insert(id).second) {
      throw std::runtime_error("Search returned a duplicate result ID at rank " +
                               std::to_string(position));
    }
  }
}

inline double Percentile(std::vector<double> values, double p) {
  if (values.empty()) return 0.0;
  std::sort(values.begin(), values.end());
  const double position = p * static_cast<double>(values.size() - 1);
  const size_t lower = static_cast<size_t>(std::floor(position));
  const size_t upper = static_cast<size_t>(std::ceil(position));
  const double weight = position - static_cast<double>(lower);
  return values[lower] * (1.0 - weight) + values[upper] * weight;
}

struct EvaluationConfig {
  size_t k = 10;
  size_t repeats = 3;
  // Search budgets are algorithm-specific and may span the full feasible
  // range. Individual drivers can impose a lower method-specific limit.
  size_t max_search_value = std::numeric_limits<size_t>::max();
  unsigned seed = 42;
  std::vector<size_t> search_values;
  std::string method;
  std::string method_commit;
  std::string source_fingerprint = TDVS_SOURCE_FINGERPRINT;
  std::string index_path;
  std::string search_param_name = "search_L";
  bool index_contains_base = false;
  std::string timed_scope = "serial_search_and_topk_materialization";
  std::string distance_count_definition = "all_query_to_data_distance_calls";
};

// A driver may provide an inner duration to exclude method-independent setup
// (parameter assignment and counter reset).  The measured interval itself must
// cover both native search and top-k ID materialization for every method.
struct SearchMeasurement {
  int computations = 0;
  double elapsed_seconds = -1.0;
};

template <typename SearchFunction>
SearchMeasurement InvokeSearch(SearchFunction& search, const float* query,
                               size_t search_value, unsigned* result) {
  auto value = search(query, search_value, result);
  if constexpr (std::is_same_v<std::decay_t<decltype(value)>, SearchMeasurement>) {
    return value;
  } else {
    return SearchMeasurement{static_cast<int>(value), -1.0};
  }
}

struct EvaluationRecord {
  std::string method;
  std::string method_commit;
  std::string source_fingerprint;
  std::string search_param_name = "search_L";
  size_t search_param_value = 0;
  size_t base_rows = 0;
  size_t dim = 0;
  size_t queries = 0;
  size_t gt_width = 0;
  size_t k = 0;
  size_t warmup_queries = 0;
  size_t repeats = 0;
  unsigned query_threads = 1;
  double recall = 0.0;
  double qps = 0.0;
  double mean_ms = 0.0;
  double p50_ms = 0.0;
  double p95_ms = 0.0;
  double p99_ms = 0.0;
  double avg_distance_computations = 0.0;
  uint64_t graph_index_bytes = 0;
  uint64_t resident_base_bytes = 0;
  std::string timed_scope = "serial_search_and_topk_materialization";
  std::string distance_count_definition = "all_query_to_data_distance_calls";
};

template <typename SearchFunction>
std::vector<EvaluationRecord> RunSearchEvaluation(const FloatMatrix& base,
                                                 const FloatMatrix& queries,
                                                 const GroundTruth& gt,
                                                 const EvaluationConfig& config,
                                                 SearchFunction search) {
  // Reported latency runs are strictly single-threaded. The outer loop below
  // is serial; forcing the OpenMP runtime to one thread also prevents a future
  // upstream search implementation from silently creating worker threads.
  ConfigureOpenMP(1);
  if (config.k == 0 || config.k > gt.width) {
    throw std::invalid_argument("k must be in [1, gt-width]");
  }
  if (base.rows > std::numeric_limits<unsigned>::max()) {
    throw std::invalid_argument(
        "Baseline result IDs require base_rows <= UINT_MAX");
  }
  if (config.repeats == 0) throw std::invalid_argument("repeats must be positive");
  std::vector<EvaluationRecord> records;
  std::vector<unsigned> results(queries.rows * config.k);

  std::cerr << "Evaluation provenance: upstream_commit=" << config.method_commit
            << ", source_fingerprint=" << config.source_fingerprint << '\n';
  std::cout << "method,search_param_name,search_param_value,recall,qps,mean_ms,p50_ms,"
               "p95_ms,p99_ms,avg_distance_computations\n";
  for (size_t search_value : config.search_values) {
    if (search_value < config.k || search_value >= base.rows ||
        search_value > config.max_search_value) {
      throw std::invalid_argument("Each " + config.search_param_name +
                                  " must satisfy k <= value <= " +
                                  std::to_string(config.max_search_value) +
                                  " and value < N");
    }
    uint64_t total_computations = 0;
    double total_seconds = 0.0;
    double recall_sum = 0.0;
    std::vector<double> latencies_ms;
    latencies_ms.reserve(queries.rows * config.repeats);

    for (size_t repeat = 0; repeat < config.repeats; ++repeat) {
      std::srand(config.seed + static_cast<unsigned>(repeat));
      for (size_t row = 0; row < queries.rows; ++row) {
        unsigned* row_results = results.data() + row * config.k;
        std::fill(row_results, row_results + config.k,
                  std::numeric_limits<unsigned>::max());
        const auto query_start = SteadyClock::now();
        const SearchMeasurement measurement = InvokeSearch(
            search, queries.Row(row), search_value, row_results);
        const auto query_end = SteadyClock::now();
        ValidateSearchResultIds(row_results, config.k, base.rows);
        if (measurement.computations < 0) {
          throw std::runtime_error("Negative computation count");
        }
        const double outer_seconds =
            std::chrono::duration<double>(query_end - query_start).count();
        const double elapsed_seconds = measurement.elapsed_seconds >= 0.0
                                           ? measurement.elapsed_seconds
                                           : outer_seconds;
        if (!std::isfinite(elapsed_seconds) || elapsed_seconds < 0.0) {
          throw std::runtime_error("Invalid measured search duration");
        }
        total_computations += static_cast<uint64_t>(measurement.computations);
        total_seconds += elapsed_seconds;
        latencies_ms.push_back(elapsed_seconds * 1000.0);
      }
      recall_sum += RecallAtK(gt, results, config.k);
    }

    EvaluationRecord record;
    record.method = config.method;
    record.method_commit = config.method_commit;
    record.source_fingerprint = config.source_fingerprint;
    record.search_param_name = config.search_param_name;
    record.search_param_value = search_value;
    record.base_rows = base.rows;
    record.dim = base.dim;
    record.queries = queries.rows;
    record.gt_width = gt.width;
    record.k = config.k;
    // Keep warmup_queries in the CSV as an explicit protocol audit field. The
    // evaluation performs no untimed warmup: every query in every repeat
    // contributes to latency, QPS, recall, and computation statistics.
    record.warmup_queries = 0;
    record.repeats = config.repeats;
    record.recall = recall_sum / config.repeats;
    const double samples = static_cast<double>(queries.rows * config.repeats);
    record.qps = samples / total_seconds;
    record.mean_ms = 1000.0 / record.qps;
    record.p50_ms = Percentile(latencies_ms, 0.50);
    record.p95_ms = Percentile(latencies_ms, 0.95);
    record.p99_ms = Percentile(latencies_ms, 0.99);
    record.avg_distance_computations = total_computations / samples;
    record.graph_index_bytes = FileSize(config.index_path);
    record.resident_base_bytes = config.index_contains_base
                                     ? 0
                                     : static_cast<uint64_t>(base.values.capacity()) *
                                           sizeof(float);
    record.distance_count_definition = config.distance_count_definition;
    record.timed_scope = config.timed_scope;
    records.push_back(record);

    std::cout << record.method << ',' << record.search_param_name << ','
              << record.search_param_value << ',' << std::fixed << std::setprecision(6)
              << record.recall << ',' << std::setprecision(2) << record.qps << ','
              << std::setprecision(6) << record.mean_ms << ',' << record.p50_ms << ','
              << record.p95_ms << ',' << record.p99_ms << ','
              << std::setprecision(2) << record.avg_distance_computations << '\n';
  }
  return records;
}

inline std::string CsvEscape(const std::string& value) {
  if (value.find_first_of(",\"\n") == std::string::npos) return value;
  std::string escaped = "\"";
  for (char c : value) escaped += c == '\"' ? "\"\"" : std::string(1, c);
  return escaped + "\"";
}

struct FootprintRecord {
  std::string method;
  std::string method_commit;
  std::string source_fingerprint = TDVS_SOURCE_FINGERPRINT;
  size_t base_rows = 0;
  size_t dim = 0;
  size_t stored_stride = 0;
  bool index_contains_base = false;
  uint64_t serialized_index_bytes = 0;
  uint64_t external_base_file_bytes = 0;
  uint64_t external_base_allocation_bytes = 0;
  uint64_t serving_artifact_bytes = 0;
  uint64_t runtime_auxiliary_known_bytes = 0;
  uint64_t rss_baseline_bytes = 0;
  uint64_t rss_after_load_bytes = 0;
  uint64_t rss_delta_load_bytes = 0;
  uint64_t peak_rss_bytes = 0;
  std::string footprint_scope = "fresh_process_load_only_no_query_gt";
};

inline uint64_t CheckedAddBytes(uint64_t left, uint64_t right,
                                const char* description) {
  if (right > std::numeric_limits<uint64_t>::max() - left) {
    throw std::overflow_error(std::string(description) + " byte count overflow");
  }
  return left + right;
}

inline uint64_t CheckedMulBytes(uint64_t count, uint64_t element_bytes,
                                const char* description) {
  if (element_bytes != 0 &&
      count > std::numeric_limits<uint64_t>::max() / element_bytes) {
    throw std::overflow_error(std::string(description) + " byte count overflow");
  }
  return count * element_bytes;
}

inline void FinalizeFootprintRecord(FootprintRecord& record) {
  record.serving_artifact_bytes = CheckedAddBytes(
      record.serialized_index_bytes, record.external_base_file_bytes,
      "serving artifact");
  record.rss_delta_load_bytes =
      RSSDeltaBytes(record.rss_after_load_bytes, record.rss_baseline_bytes);
}

inline void PrintFootprintSummary(const FootprintRecord& record) {
  std::cout << "method=" << record.method << '\n'
            << "method_commit=" << record.method_commit << '\n'
            << "source_fingerprint=" << record.source_fingerprint << '\n'
            << "footprint_scope=" << record.footprint_scope << '\n'
            << "base_rows=" << record.base_rows << '\n'
            << "dim=" << record.dim << '\n'
            << "stored_stride=" << record.stored_stride << '\n'
            << "index_contains_base="
            << (record.index_contains_base ? "true" : "false") << '\n'
            << "serialized_index_bytes=" << record.serialized_index_bytes << '\n'
            << "external_base_file_bytes=" << record.external_base_file_bytes << '\n'
            << "external_base_allocation_bytes="
            << record.external_base_allocation_bytes << '\n'
            << "serving_artifact_bytes=" << record.serving_artifact_bytes << '\n'
            << "runtime_auxiliary_known_bytes="
            << record.runtime_auxiliary_known_bytes << '\n'
            << "rss_baseline_bytes=" << record.rss_baseline_bytes << '\n'
            << "rss_after_load_bytes=" << record.rss_after_load_bytes << '\n'
            << "rss_delta_load_bytes=" << record.rss_delta_load_bytes << '\n'
            << "peak_rss_bytes=" << record.peak_rss_bytes << '\n';
}

inline void WriteFootprintCsv(const std::string& path,
                              const FootprintRecord& record) {
  if (path.empty()) return;
  std::ofstream output(path);
  if (!output) throw std::runtime_error("Cannot write CSV: " + path);
  output << "method,method_commit,source_fingerprint,base_rows,dim,stored_stride,"
            "index_contains_base,serialized_index_bytes,external_base_file_bytes,"
            "external_base_allocation_bytes,serving_artifact_bytes,"
            "runtime_auxiliary_known_bytes,rss_baseline_bytes,rss_after_load_bytes,"
            "rss_delta_load_bytes,peak_rss_bytes,footprint_scope\n";
  output << CsvEscape(record.method) << ',' << CsvEscape(record.method_commit) << ','
         << CsvEscape(record.source_fingerprint) << ',' << record.base_rows << ','
         << record.dim << ',' << record.stored_stride << ','
         << (record.index_contains_base ? 1 : 0) << ','
         << record.serialized_index_bytes << ',' << record.external_base_file_bytes
         << ',' << record.external_base_allocation_bytes << ','
         << record.serving_artifact_bytes << ','
         << record.runtime_auxiliary_known_bytes << ',' << record.rss_baseline_bytes
         << ',' << record.rss_after_load_bytes << ',' << record.rss_delta_load_bytes
         << ',' << record.peak_rss_bytes << ','
         << CsvEscape(record.footprint_scope) << '\n';
}

inline void EmitFootprint(const std::string& output_csv,
                          FootprintRecord record) {
  FinalizeFootprintRecord(record);
  PrintFootprintSummary(record);
  WriteFootprintCsv(output_csv, record);
}

inline void WriteCsv(const std::string& path, const std::vector<EvaluationRecord>& records) {
  if (path.empty()) return;
  std::ofstream output(path);
  if (!output) throw std::runtime_error("Cannot write CSV: " + path);
  output << "method,method_commit,source_fingerprint,search_param_name,search_param_value,base_rows,dim,"
            "queries,gt_width,k,warmup_queries,repeats,query_threads,recall,qps,mean_ms,"
            "p50_ms,p95_ms,p99_ms,avg_distance_computations,distance_count_definition,"
            "graph_index_bytes,resident_base_bytes,online_bytes,timed_scope\n";
  output << std::setprecision(10);
  for (const auto& row : records) {
    output << CsvEscape(row.method) << ',' << CsvEscape(row.method_commit) << ','
           << CsvEscape(row.source_fingerprint) << ',' << row.search_param_name << ','
           << row.search_param_value << ',' << row.base_rows
           << ',' << row.dim << ',' << row.queries << ',' << row.gt_width << ',' << row.k
           << ',' << row.warmup_queries << ',' << row.repeats << ',' << row.query_threads
           << ',' << row.recall << ',' << row.qps << ',' << row.mean_ms << ',' << row.p50_ms
           << ',' << row.p95_ms << ',' << row.p99_ms << ','
           << row.avg_distance_computations << ','
           << CsvEscape(row.distance_count_definition) << ',' << row.graph_index_bytes << ','
           << row.resident_base_bytes << ','
           << (row.graph_index_bytes + row.resident_base_bytes) << ','
           << CsvEscape(row.timed_scope) << '\n';
  }
}

inline void PrintInputSummary(const FloatMatrix& base, const FloatMatrix* queries = nullptr) {
  std::cerr << "Loaded base: N=" << base.rows << ", dim=" << base.dim
            << ", aligned_stride=" << base.stride << '\n';
  if (queries != nullptr) {
    std::cerr << "Loaded queries: Q=" << queries->rows << ", dim=" << queries->dim
              << ", aligned_stride=" << queries->stride << '\n';
  }
}

}  // namespace tdvs_mips
