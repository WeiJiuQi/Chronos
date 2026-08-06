#pragma once

// MAG and PSP only use construction, operator[] and reset() from
// boost::dynamic_bitset. Keeping this small compatibility implementation in
// the shared include tree avoids pulling the full Boost distribution into minimal builds.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace boost {

template <typename Block = unsigned long, typename Allocator = std::allocator<Block>>
class dynamic_bitset {
 public:
  class reference {
   public:
    reference(uint64_t& block, uint64_t mask) : block_(block), mask_(mask) {}
    reference& operator=(bool value) {
      if (value) {
        block_ |= mask_;
      } else {
        block_ &= ~mask_;
      }
      return *this;
    }
    reference& operator=(const reference& other) {
      return *this = static_cast<bool>(other);
    }
    operator bool() const { return (block_ & mask_) != 0; }

   private:
    uint64_t& block_;
    uint64_t mask_;
  };

  explicit dynamic_bitset(size_t size = 0, unsigned long value = 0)
      : size_(size), blocks_((size + 63U) / 64U, value == 0 ? 0U : ~uint64_t{0}) {
    TrimLastBlock();
  }

  reference operator[](size_t position) {
    if (position >= size_) throw std::out_of_range("dynamic_bitset index");
    return reference(blocks_[position >> 6U], uint64_t{1} << (position & 63U));
  }

  bool operator[](size_t position) const {
    if (position >= size_) throw std::out_of_range("dynamic_bitset index");
    return (blocks_[position >> 6U] & (uint64_t{1} << (position & 63U))) != 0;
  }

  dynamic_bitset& reset() {
    std::fill(blocks_.begin(), blocks_.end(), uint64_t{0});
    return *this;
  }

  size_t size() const { return size_; }

 private:
  void TrimLastBlock() {
    if (blocks_.empty() || size_ % 64U == 0) return;
    blocks_.back() &= (uint64_t{1} << (size_ % 64U)) - 1U;
  }

  size_t size_;
  std::vector<uint64_t> blocks_;
};

}  // namespace boost
