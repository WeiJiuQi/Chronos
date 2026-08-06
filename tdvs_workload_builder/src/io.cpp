#include "io.h"

#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>

namespace chronos {

#ifndef CHRONOS_GIT_REVISION
#define CHRONOS_GIT_REVISION "unknown"
#endif
#ifndef CHRONOS_GIT_DIRTY
#define CHRONOS_GIT_DIRTY 0
#endif

const char* source_revision() { return CHRONOS_GIT_REVISION; }
bool source_tree_was_dirty_at_configure() { return CHRONOS_GIT_DIRTY != 0; }

namespace {

constexpr std::size_t kFbinHeaderBytes = 2 * sizeof(std::uint32_t);

bool read_fbin_header(FILE* fp, int expected_dim, std::size_t& rows) {
    std::uint32_t header[2] = {0, 0};
    if (std::fseek(fp, 0, SEEK_SET) != 0 ||
        std::fread(header, sizeof(std::uint32_t), 2, fp) != 2) {
        return false;
    }
    if (header[0] == 0 || header[1] != static_cast<std::uint32_t>(expected_dim)) return false;
    rows = static_cast<std::size_t>(header[0]);
    return true;
}

bool seek_to_row(FILE* fp, std::size_t header_bytes, std::size_t index, int dim) {
    const std::size_t row_bytes = static_cast<std::size_t>(dim) * sizeof(float);
    if (index > (std::numeric_limits<std::size_t>::max() - header_bytes) / row_bytes) return false;
    const std::size_t offset = header_bytes + index * row_bytes;
    if (offset > static_cast<std::size_t>(std::numeric_limits<long>::max())) return false;
    return std::fseek(fp, static_cast<long>(offset), SEEK_SET) == 0;
}

}  // namespace

const char* input_vector_format_name(InputVectorFormat format) {
    switch (format) {
        case InputVectorFormat::RawFloat32: return "raw";
        case InputVectorFormat::Fvecs: return "fvecs";
        case InputVectorFormat::Fbin: return "fbin";
    }
    return "unknown";
}

std::size_t file_size_bytes(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) return 0;
    return static_cast<std::size_t>(f.tellg());
}

std::size_t count_vectors_in_file(const std::string& path, int dim, InputVectorFormat fmt) {
    if (dim <= 0) return 0;
    std::size_t sz = file_size_bytes(path);
    if (sz == 0) return 0;
    if (fmt == InputVectorFormat::RawFloat32) {
        std::size_t row = static_cast<std::size_t>(dim) * sizeof(float);
        if (sz % row != 0) return 0;
        return sz / row;
    }
    if (fmt == InputVectorFormat::Fbin) {
        FILE* fp = std::fopen(path.c_str(), "rb");
        if (!fp) return 0;
        std::size_t rows = 0;
        const bool valid_header = read_fbin_header(fp, dim, rows);
        std::fclose(fp);
        if (!valid_header) return 0;
        const std::size_t row = static_cast<std::size_t>(dim) * sizeof(float);
        if (rows > (std::numeric_limits<std::size_t>::max() - kFbinHeaderBytes) / row) return 0;
        return sz == kFbinHeaderBytes + rows * row ? rows : 0;
    }
    // Fvecs: validate the exact fixed record count and every row header.
    const std::size_t record_bytes = (static_cast<std::size_t>(dim) + 1) * sizeof(std::uint32_t);
    if (sz % record_bytes != 0) return 0;
    const std::size_t expected_rows = sz / record_bytes;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return 0;
    std::size_t cnt = 0;
    for (;;) {
        int d_file = 0;
        if (std::fread(&d_file, sizeof(int), 1, fp) != 1) break;
        if (d_file != dim) {
            std::fclose(fp);
            return 0;
        }
        if (std::fseek(fp, static_cast<long>(dim) * static_cast<long>(sizeof(float)), SEEK_CUR) !=
            0) {
            std::fclose(fp);
            return 0;
        }
        ++cnt;
    }
    std::fclose(fp);
    return cnt == expected_rows ? cnt : 0;
}

bool read_vectors_prefix(const std::string& path, int dim, InputVectorFormat fmt,
                         std::size_t max_read, std::vector<float>& out, std::size_t& out_count) {
    out.clear();
    out_count = 0;
    if (dim <= 0 || max_read == 0) return true;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    if (fmt == InputVectorFormat::RawFloat32 || fmt == InputVectorFormat::Fbin) {
        std::size_t available = 0;
        if (fmt == InputVectorFormat::Fbin) {
            if (!read_fbin_header(fp, dim, available)) {
                std::fclose(fp);
                return false;
            }
        } else {
            const std::size_t bytes = file_size_bytes(path);
            const std::size_t row = static_cast<std::size_t>(dim) * sizeof(float);
            if (bytes == 0 || bytes % row != 0) {
                std::fclose(fp);
                return false;
            }
            available = bytes / row;
        }
        const std::size_t to_read = std::min(max_read, available);
        out.resize(max_read * static_cast<std::size_t>(dim));
        const std::size_t values = to_read * static_cast<std::size_t>(dim);
        const std::size_t read_values = std::fread(out.data(), sizeof(float), values, fp);
        if (read_values != values) {
            out.clear();
            std::fclose(fp);
            return false;
        }
        out.resize(values);
        out_count = to_read;
        std::fclose(fp);
        return to_read != 0;
    }

    // Fvecs
    out.resize(max_read * static_cast<std::size_t>(dim));
    std::size_t read_vecs = 0;
    for (; read_vecs < max_read; ++read_vecs) {
        int d_file = 0;
        if (std::fread(&d_file, sizeof(int), 1, fp) != 1) break;
        if (d_file != dim) {
            out.resize(read_vecs * static_cast<std::size_t>(dim));
            out_count = read_vecs;
            std::fclose(fp);
            return read_vecs > 0;
        }
        float* row = out.data() + read_vecs * static_cast<std::size_t>(dim);
        if (std::fread(row, sizeof(float), static_cast<std::size_t>(dim), fp) !=
            static_cast<std::size_t>(dim)) {
            out.resize(read_vecs * static_cast<std::size_t>(dim));
            out_count = read_vecs;
            std::fclose(fp);
            return read_vecs > 0;
        }
    }
    out.resize(read_vecs * static_cast<std::size_t>(dim));
    out_count = read_vecs;
    std::fclose(fp);
    return read_vecs > 0;
}

static bool fseek_fvecs_record(FILE* fp, std::size_t index) {
    if (std::fseek(fp, 0, SEEK_SET) != 0) return false;
    for (std::size_t i = 0; i < index; ++i) {
        int d_file = 0;
        if (std::fread(&d_file, sizeof(int), 1, fp) != 1) return false;
        long skip = static_cast<long>(d_file) * static_cast<long>(sizeof(float));
        if (std::fseek(fp, skip, SEEK_CUR) != 0) return false;
    }
    return true;
}

bool read_vectors_sorted_indices(const std::string& path, int dim, InputVectorFormat fmt,
                                 const std::vector<std::size_t>& indices, std::vector<float>& out) {
    const std::size_t n = indices.size();
    out.resize(n * static_cast<std::size_t>(dim));
    if (n == 0) return true;
    for (std::size_t i = 1; i < n; ++i) {
        if (indices[i] <= indices[i - 1]) return false;
    }
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    if (fmt == InputVectorFormat::RawFloat32 || fmt == InputVectorFormat::Fbin) {
        const std::size_t row = static_cast<std::size_t>(dim) * sizeof(float);
        const std::size_t file_sz = file_size_bytes(path);
        std::size_t total_vecs = 0;
        std::size_t header_bytes = 0;
        if (fmt == InputVectorFormat::Fbin) {
            if (!read_fbin_header(fp, dim, total_vecs)) {
                std::fclose(fp);
                return false;
            }
            header_bytes = kFbinHeaderBytes;
            if (total_vecs > (std::numeric_limits<std::size_t>::max() - header_bytes) / row ||
                file_sz != header_bytes + total_vecs * row) {
                std::fclose(fp);
                return false;
            }
        } else if (file_sz == 0 || file_sz % row != 0) {
            std::fclose(fp);
            return false;
        } else {
            total_vecs = file_sz / row;
        }
        if (indices.back() >= total_vecs) {
            std::fclose(fp);
            return false;
        }

        bool contiguous = true;
        for (std::size_t i = 1; i < n; ++i) {
            if (indices[i] != indices[i - 1] + 1) {
                contiguous = false;
                break;
            }
        }
        if (contiguous) {
            if (!seek_to_row(fp, header_bytes, indices.front(), dim)) {
                std::fclose(fp);
                return false;
            }
            std::size_t nf = n * static_cast<std::size_t>(dim);
            if (std::fread(out.data(), sizeof(float), nf, fp) != nf) {
                std::fclose(fp);
                return false;
            }
            std::fclose(fp);
            return true;
        }

        for (std::size_t begin = 0; begin < n;) {
            std::size_t end = begin + 1;
            while (end < n && indices[end] == indices[end - 1] + 1) ++end;
            if (!seek_to_row(fp, header_bytes, indices[begin], dim)) {
                std::fclose(fp);
                return false;
            }
            const std::size_t values = (end - begin) * static_cast<std::size_t>(dim);
            float* destination = out.data() + begin * static_cast<std::size_t>(dim);
            if (std::fread(destination, sizeof(float), values, fp) != values) {
                std::fclose(fp);
                return false;
            }
            begin = end;
        }
        std::fclose(fp);
        return true;
    }

    // Fvecs: single forward pass, early exit when all indices collected
    std::size_t ki = 0;
    for (std::size_t vid = 0; ki < n; ++vid) {
        int d_file = 0;
        if (std::fread(&d_file, sizeof(int), 1, fp) != 1) {
            std::fclose(fp);
            return false;
        }
        if (d_file != dim) {
            std::fclose(fp);
            return false;
        }
        if (vid == indices[ki]) {
            float* rowptr = out.data() + ki * static_cast<std::size_t>(dim);
            if (std::fread(rowptr, sizeof(float), static_cast<std::size_t>(dim), fp) !=
                static_cast<std::size_t>(dim)) {
                std::fclose(fp);
                return false;
            }
            ++ki;
        } else {
            long skip = static_cast<long>(dim) * static_cast<long>(sizeof(float));
            if (std::fseek(fp, skip, SEEK_CUR) != 0) {
                std::fclose(fp);
                return false;
            }
        }
    }
    std::fclose(fp);
    return true;
}

bool read_vectors_at_indices(const std::string& path, int dim, InputVectorFormat fmt,
                             const std::vector<std::size_t>& indices, std::vector<float>& out) {
    out.resize(indices.size() * static_cast<std::size_t>(dim));
    if (indices.empty()) return true;
    FILE* fp = std::fopen(path.c_str(), "rb");
    if (!fp) return false;

    if (fmt == InputVectorFormat::RawFloat32 || fmt == InputVectorFormat::Fbin) {
        std::size_t header_bytes = 0;
        std::size_t total_rows = 0;
        if (fmt == InputVectorFormat::Fbin) {
            if (!read_fbin_header(fp, dim, total_rows)) {
                std::fclose(fp);
                return false;
            }
            header_bytes = kFbinHeaderBytes;
        } else {
            total_rows = count_vectors_in_file(path, dim, fmt);
        }
        for (std::size_t r = 0; r < indices.size(); ++r) {
            std::size_t idx = indices[r];
            if (idx >= total_rows || !seek_to_row(fp, header_bytes, idx, dim)) {
                std::fclose(fp);
                return false;
            }
            float* row = out.data() + r * static_cast<std::size_t>(dim);
            if (std::fread(row, sizeof(float), static_cast<std::size_t>(dim), fp) !=
                static_cast<std::size_t>(dim)) {
                std::fclose(fp);
                return false;
            }
        }
        std::fclose(fp);
        return true;
    }

    // Fvecs: seek from the beginning for each requested row. Callers with a
    // sorted index list should use read_vectors_sorted_indices instead.
    for (std::size_t r = 0; r < indices.size(); ++r) {
        if (!fseek_fvecs_record(fp, indices[r])) {
            std::fclose(fp);
            return false;
        }
        int d_file = 0;
        if (std::fread(&d_file, sizeof(int), 1, fp) != 1 || d_file != dim) {
            std::fclose(fp);
            return false;
        }
        float* row = out.data() + r * static_cast<std::size_t>(dim);
        if (std::fread(row, sizeof(float), static_cast<std::size_t>(dim), fp) !=
            static_cast<std::size_t>(dim)) {
            std::fclose(fp);
            return false;
        }
    }
    std::fclose(fp);
    return true;
}

bool write_raw_vectors(const std::string& path, const float* data, std::size_t n, int dim) {
    if (!data || n == 0 || dim <= 0) return false;
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    std::size_t total = n * static_cast<std::size_t>(dim);
    std::size_t w = std::fwrite(data, sizeof(float), total, fp);
    std::fclose(fp);
    return w == total;
}

bool write_source_row_ids(const std::string& path, const std::vector<std::size_t>& indices) {
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    for (std::size_t index : indices) {
        if (index > static_cast<std::size_t>(std::numeric_limits<std::uint64_t>::max())) {
            std::fclose(fp);
            return false;
        }
        const std::uint64_t value = static_cast<std::uint64_t>(index);
        if (std::fwrite(&value, sizeof(value), 1, fp) != 1) {
            std::fclose(fp);
            return false;
        }
    }
    std::fclose(fp);
    return true;
}

bool write_groundtruth(const std::string& path, const GroundTruthId* flat, std::size_t num_queries,
                       int k) {
    if (!flat || num_queries == 0 || k <= 0) return false;
    FILE* fp = std::fopen(path.c_str(), "wb");
    if (!fp) return false;
    for (std::size_t q = 0; q < num_queries; ++q) {
        const GroundTruthId* row = flat + q * static_cast<std::size_t>(k);
        std::size_t w = std::fwrite(row, sizeof(GroundTruthId), static_cast<std::size_t>(k), fp);
        if (w != static_cast<std::size_t>(k)) {
            std::fclose(fp);
            return false;
        }
    }
    std::fclose(fp);
    return true;
}

bool write_text_file(const std::string& path, const std::string& content) {
    FILE* fp = std::fopen(path.c_str(), "w");
    if (!fp) return false;
    std::fwrite(content.data(), 1, content.size(), fp);
    std::fclose(fp);
    return true;
}

}  // namespace chronos
