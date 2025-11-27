#pragma once

#include "Division.hpp"
#include "Karatsuba.hpp"
#include <algorithm>
#include <array>
#include <charconv>
#include <cstring>
#include <immintrin.h>
#include <iomanip>
#include <limits>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Type Aliases
using u8 = unsigned char;
using u16 = unsigned short;
using u64 = unsigned long long;
using u128 = unsigned __int128;

#define BUILTIN_EXPECT(x, y) (__builtin_expect(!!(x), y))
#define FORCE_INLINE inline __attribute__((always_inline))

template <u8 N> class alignas(64) UInt;

// Fixed-size arbitrary precision integer class.
//
// @tparam N Number of 64-bit limbs.
//
// Design Goals:
// - High performance for small to medium sizes (N <= 64).
// - Stack allocation (no dynamic memory).
// - Constant time operations where possible (though many are data-dependent for speed).
// - Modern C++20/23 features.
template <u8 N> class alignas(64) UInt {
  static_assert(N > 0, "Limbs must be positive");

public:
  static constexpr u8 TotalBlocks = N;
  std::array<u64, N> bits{}; // initialize all elements to zero

private:
  FORCE_INLINE constexpr u8 charToValue(char c) const {
    constexpr auto lut = []() { // lookup table
      std::array<u8, 256> table;
      table.fill(255);
      for (u8 i = 0; i < 10; ++i)
        table['0' + i] = i;
      for (u8 i = 0; i < 6; ++i) {
        table['a' + i] = 10 + i;
        table['A' + i] = 10 + i;
      }
      return table;
    }();
    u8 value = lut[static_cast<u8>(c)];
    if (BUILTIN_EXPECT(value > 15, 0))
      throw std::invalid_argument("Invalid character");
    return value;
  }

  [[nodiscard]] FORCE_INLINE u8 num_limbs() const noexcept {
    for (int i = N - 1; i >= 0; --i) {
      if (bits[i] != 0)
        return i + 1;
    }
    return 0;
  }

public:
  constexpr UInt() noexcept = default;

  constexpr UInt(const u64 value) noexcept { bits[0] = value; }

  constexpr UInt(const UInt &o) noexcept = default;

  constexpr UInt(UInt &&o) noexcept = default;

  constexpr explicit UInt(std::string_view sv) {
    bits.fill(0);
    if (sv.empty())
      return;
    bool signal = (sv.front() == '-');
    if (signal || sv.front() == '+')
      sv.remove_prefix(1);
    if (sv.empty())
      return;
    const bool is_hex = sv.starts_with("0x") || sv.starts_with("0X");
    if (is_hex) {
      sv.remove_prefix(2);
      if (sv.empty())
        return;
      if (sv.size() > (N * 16))
        throw std::out_of_range("Hex string too long");
      u16 size = sv.size();
      for (u8 i = 0; i < N && size > 0; ++i) {
        u64 value = 0;
        int chars_to_read = std::min(16, (int)size);
        for (int j = 0; j < chars_to_read; ++j) {
          value |= static_cast<u64>(charToValue(sv[--size])) << (4 * j);
        }
        bits[i] = value;
      }
    } else {
      constexpr u64 CHUNK_POW = 1000000000000000000ull;
      constexpr int CHUNK_SIZE = 18;
      *this = 0;
      const char *p = sv.data();
      const char *const end = p + sv.size();
      u16 first_chunk_len = sv.size() % CHUNK_SIZE;
      if (first_chunk_len == 0 && !sv.empty())
        first_chunk_len = CHUNK_SIZE;
      u64 chunk_val = 0;
      const char *const end_first_chunk = p + first_chunk_len;
      auto res = std::from_chars(p, end_first_chunk, chunk_val);
      if (BUILTIN_EXPECT(res.ec != std::errc(), 0))
        throw std::invalid_argument("Invalid decimal character");
      *this = chunk_val;
      p = end_first_chunk;
      while (p < end) {
        const char *const end_chunk = p + CHUNK_SIZE;
        res = std::from_chars(p, end_chunk, chunk_val);
        if (BUILTIN_EXPECT(res.ec != std::errc(), 0))
          throw std::invalid_argument("Invalid decimal character");
        *this *= CHUNK_POW;
        *this += chunk_val;
        p = end_chunk;
      }
    }
    if (signal)
      throw std::invalid_argument("Negative values not supported in UInt");
  }

  constexpr UInt &operator=(const UInt &o) noexcept = default;
  constexpr UInt &operator=(UInt &&o) noexcept = default;
  constexpr UInt &operator=(u64 o) noexcept {
    bits.fill(0);
    bits[0] = o;
    return *this;
  }

  [[nodiscard]] FORCE_INLINE constexpr bool operator==(const UInt &o) const noexcept {
    return bits == o.bits;
  }
  [[nodiscard]] FORCE_INLINE constexpr bool operator!=(const UInt &o) const noexcept {
    return !(*this == o);
  }
  [[nodiscard]] FORCE_INLINE constexpr bool operator==(u64 o) const noexcept {
    if (bits[0] != o)
      return false;
    for (u8 i = 1; i < N; ++i)
      if (bits[i] != 0)
        return false;
    return true;
  }
  [[nodiscard]] FORCE_INLINE constexpr bool operator<(const UInt &o) const noexcept {
    for (int i = N - 1; i >= 0; --i) {
      if (bits[i] < o.bits[i])
        return true;
      if (bits[i] > o.bits[i])
        return false;
    }
    return false;
  }
  [[nodiscard]] FORCE_INLINE constexpr bool operator>(const UInt &o) const noexcept {
    return o < *this;
  }
  [[nodiscard]] FORCE_INLINE constexpr bool operator<=(const UInt &o) const noexcept {
    return !(o < *this);
  }
  [[nodiscard]] FORCE_INLINE constexpr bool operator>=(const UInt &o) const noexcept {
    return !(*this < o);
  }

  FORCE_INLINE constexpr UInt &operator++() noexcept {
    if (++bits[0] != 0) [[likely]]
      return *this;
    u8 index = 1;
    while (index < N) {
      if (++bits[index] != 0) [[likely]]
        return *this; // Carry absorvido, retorna imediatamente
      ++index;
    }
    return *this;
  }

  FORCE_INLINE constexpr UInt &operator--() noexcept {
    if (--bits[0] != 0) [[likely]]
      return *this;
    u8 index = 1;
    while (index < N) {
      if (--bits[index] != 0) [[likely]]
        return *this; // Borrow absorvido, retorna imediatamente
      ++index;
    }
    return *this;
  }
  FORCE_INLINE constexpr UInt operator++(int) noexcept {
    UInt t = *this;
    ++*this;
    return t;
  }
  FORCE_INLINE constexpr UInt operator--(int) noexcept {
    UInt t = *this;
    --*this;
    return t;
  }

  // --- Unrolling Helpers ---
  template <u8 I>
  struct UnrollAdd {
    static FORCE_INLINE void step(unsigned char &carry, u64 *dst, const u64 *src) {
      carry = _addcarry_u64(carry, dst[I], src[I], &dst[I]);
      if constexpr (I + 1 < N)
        UnrollAdd<I + 1>::step(carry, dst, src);
    }
  };

  template <u8 I>
  struct UnrollSub {
    static FORCE_INLINE void step(unsigned char &borrow, u64 *dst, const u64 *src) {
      borrow = _subborrow_u64(borrow, dst[I], src[I], &dst[I]);
      if constexpr (I + 1 < N)
        UnrollSub<I + 1>::step(borrow, dst, src);
    }
  };

  FORCE_INLINE UInt &operator+=(const UInt &other) noexcept {
    if (std::is_constant_evaluated()) {
      u64 c = 0;
      for (u8 i = 0; i < N; ++i) {
        u128 t = (u128)bits[i] + other.bits[i] + c;
        bits[i] = (u64)t;
        c = t >> 64;
      }
    } else if constexpr (N <= 56) {
      unsigned char carry = 0;
      UnrollAdd<0>::step(carry, bits.data(), other.bits.data());
    } else {
      unsigned char carry = 0;
      u64 *dst = bits.data();
      const u64 *src = other.bits.data();
      size_t i = 0;
      for (; i + 4 <= N; i += 4) {
        carry = _addcarry_u64(carry, dst[i], src[i], &dst[i]);
        carry = _addcarry_u64(carry, dst[i + 1], src[i + 1], &dst[i + 1]);
        carry = _addcarry_u64(carry, dst[i + 2], src[i + 2], &dst[i + 2]);
        carry = _addcarry_u64(carry, dst[i + 3], src[i + 3], &dst[i + 3]);
      }
      for (; i < N; ++i) {
        carry = _addcarry_u64(carry, dst[i], src[i], &dst[i]);
      }
    }
    return *this;
  }

  FORCE_INLINE UInt &operator+=(u64 val) noexcept {
    u64 c = val;
    u8 i = 0;
    while (c > 0 && i < N) {
      u128 t = (u128)bits[i] + c;
      bits[i] = (u64)t;
      c = t >> 64;
      i++;
    }
    return *this;
  }

  FORCE_INLINE UInt &operator-=(const UInt &other) noexcept {
    if (std::is_constant_evaluated()) {
      u64 b = 0;
      for (u8 i = 0; i < N; ++i) {
        u128 t = (u128)bits[i] - other.bits[i] - b;
        bits[i] = (u64)t;
        b = (t >> 64) & 1;
      }
    } else if constexpr (N <= 56) {
      unsigned char borrow = 0;
      UnrollSub<0>::step(borrow, bits.data(), other.bits.data());
    } else {
      unsigned char borrow = 0;
      u64 *dst = bits.data();
      const u64 *src = other.bits.data();
      size_t i = 0;

      for (; i + 4 <= N; i += 4) {
        borrow = _subborrow_u64(borrow, dst[i], src[i], &dst[i]);
        borrow = _subborrow_u64(borrow, dst[i + 1], src[i + 1], &dst[i + 1]);
        borrow = _subborrow_u64(borrow, dst[i + 2], src[i + 2], &dst[i + 2]);
        borrow = _subborrow_u64(borrow, dst[i + 3], src[i + 3], &dst[i + 3]);
      }

      for (; i < N; ++i) {
        borrow = _subborrow_u64(borrow, dst[i], src[i], &dst[i]);
      }
    }
    return *this;
  }
  FORCE_INLINE UInt &operator*=(const UInt &other) noexcept;
  FORCE_INLINE UInt &operator*=(u64 val) noexcept;
  // Division
  [[nodiscard]] static std::pair<UInt<N>, UInt<N>> divmod(UInt<N> u, UInt<N> v);
  [[nodiscard]] FORCE_INLINE UInt<N> operator/(const UInt<N> &other) const;
  [[nodiscard]] FORCE_INLINE UInt<N> operator%(const UInt<N> &other) const;
  FORCE_INLINE UInt<N> &operator/=(const UInt<N> &other) {
    *this = divmod(*this, other).first;
    return *this;
  }
  FORCE_INLINE UInt<N> &operator%=(const UInt<N> &other) {
    *this = divmod(*this, other).second;
    return *this;
  }
  FORCE_INLINE constexpr UInt &operator&=(const UInt &other) noexcept {
    for (u8 i = 0; i < N; ++i)
      bits[i] &= other.bits[i];
    return *this;
  }
  FORCE_INLINE constexpr UInt &operator|=(const UInt &other) noexcept {
    for (u8 i = 0; i < N; ++i)
      bits[i] |= other.bits[i];
    return *this;
  }
  FORCE_INLINE constexpr UInt &operator^=(const UInt &other) noexcept {
    for (u8 i = 0; i < N; ++i)
      bits[i] ^= other.bits[i];
    return *this;
  }
  FORCE_INLINE constexpr UInt &operator<<=(u16 n) noexcept;
  FORCE_INLINE constexpr UInt &operator>>=(u16 n) noexcept;



  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] std::string to_hex_string() const;
  [[nodiscard]] FORCE_INLINE constexpr bool is_zero() const noexcept {
    for (u64 limb : bits)
      if (limb != 0)
        return false;
    return true;
  }
  [[nodiscard]] FORCE_INLINE constexpr bool bt(const u16 index) const noexcept {
    if (BUILTIN_EXPECT(index >= N * 64, 0))
      return false;
    return (bits[index / 64] >> (index % 64)) & 1;
  }
  FORCE_INLINE constexpr void bts(const u16 index) noexcept {
    if (BUILTIN_EXPECT(index < N * 64, 1))
      bits[index / 64] |= (1ull << (index % 64));
  }
};

// --- Method Implementations ---

template <u8 N>
FORCE_INLINE UInt<N> &UInt<N>::operator*=(const UInt<N> &other) noexcept {
  UInt<N> self_copy = *this;
  this->bits.fill(0);

  if constexpr (N <= 6) {
    // Zone 1: Schoolbook Unrolled (Template Recursion)
    // Fastest for N = 1 to 6
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      auto outer_step = [&](auto i_const) {
        constexpr u8 i = i_const;
        u64 y = self_copy.bits[i];
        if (y == 0)
          return;

        u128 carry = 0;
        [&]<std::size_t... J>(std::index_sequence<J...>) {
          auto inner_step = [&](auto j_const) {
            constexpr u8 j = j_const;
            if constexpr (i + j < N) {
              u64 hi, lo;
              lo = _mulx_u64(other.bits[j], y, &hi);

              u64 carry_lo = (u64)carry;
              u64 carry_hi = (u64)(carry >> 64);

              unsigned char c1 = _addcarry_u64(0, lo, carry_lo, &lo);
              unsigned char c2 =
                  _addcarry_u64(0, lo, this->bits[i + j], &this->bits[i + j]);
              
              carry = (u128)hi + carry_hi + c1 + c2;
            }
          };
          (inner_step(std::integral_constant<std::size_t, J>{}), ...);
        }(std::make_index_sequence<N - i>{});
      };
      (outer_step(std::integral_constant<std::size_t, I>{}), ...);
    }(std::make_index_sequence<N>{});

  } else if constexpr (N <= 32) {
    // Zone 2: Schoolbook Looped
    // Fastest for N = 7 to 32 (Extended based on benchmark)
    // Truncated schoolbook is O(N^2/2), very competitive against Karatsuba O(N^1.58) for small N.
    for (u8 i = 0; i < N; ++i) {
      u64 y = self_copy.bits[i];
      if (y == 0)
        continue;

      u128 carry = 0;
      for (u8 j = 0; j < N - i; ++j) {
        u128 temp = (u128)other.bits[j] * y + this->bits[i + j] + carry;
        this->bits[i + j] = (u64)temp;
        carry = temp >> 64;
      }
    }
  } else {
    // Zone 3: Karatsuba
    // Fastest for N >= 33
    // Buffer size: 8n + 1000 safety
    alignas(64) u64 tmp[8 * N + 1000];  // 64-byte aligned for cache efficiency
    
    if (this == &other) {
        Karatsuba::square_truncated_fixed<N>(this->bits.data(), self_copy.bits.data(), tmp);
    } else {
        Karatsuba::mul_truncated_fixed<N>(this->bits.data(), self_copy.bits.data(),
                                 other.bits.data(), tmp);
    }
  }
  return *this;
}

template <u8 N> FORCE_INLINE UInt<N> &UInt<N>::operator*=(u64 val) noexcept {
  u64 c = 0;
  for (u8 i = 0; i < N; ++i) {
    u128 temp = (u128)bits[i] * val + c;
    bits[i] = (u64)temp;
    c = temp >> 64;
  }
  return *this;
}

template <u8 N>
FORCE_INLINE constexpr UInt<N> &UInt<N>::operator<<=(u16 n) noexcept {
  if (BUILTIN_EXPECT(n == 0, 1))
    return *this;
  if (n >= N * 64) {
    bits.fill(0);
    return *this;
  }
  const u16 block_shift = n / 64;
  const u16 bit_shift = n % 64;
  if (block_shift > 0) {
    for (int i = N - 1; i >= (int)block_shift; --i)
      bits[i] = bits[i - block_shift];
    std::fill(bits.begin(), bits.begin() + block_shift, 0ull);
  }
  if (bit_shift > 0) {
    u64 carry = 0;
    for (u8 i = 0; i < N; ++i) {
      u64 next_carry = bits[i] >> (64 - bit_shift);
      bits[i] = (bits[i] << bit_shift) | carry;
      carry = next_carry;
    }
  }
  return *this;
}

template <u8 N>
FORCE_INLINE constexpr UInt<N> &UInt<N>::operator>>=(u16 n) noexcept {
  if (BUILTIN_EXPECT(n == 0, 1))
    return *this;
  if (n >= N * 64) {
    bits.fill(0);
    return *this;
  }
  const u16 block_shift = n / 64;
  const u16 bit_shift = n % 64;
  if (block_shift > 0) {
    for (u8 i = 0; i < N - block_shift; ++i)
      bits[i] = bits[i + block_shift];
    std::fill(bits.begin() + N - block_shift, bits.end(), 0ull);
  }
  if (bit_shift > 0) {
    u64 carry = 0;
    for (int i = N - 1; i >= 0; --i) {
      u64 next_carry = bits[i] << (64 - bit_shift);
      bits[i] = (bits[i] >> bit_shift) | carry;
      carry = next_carry;
    }
  }
  return *this;
}

// Hybrid division: fast path for single-limb divisor, correct slow path for
// others.
template <u8 N>
std::pair<UInt<N>, UInt<N>> UInt<N>::divmod(UInt<N> u, UInt<N> v) {
  if (v.is_zero())
    throw std::domain_error("Division by zero");
  if (u < v)
    return {UInt<N>(0), u};

  // FAST PATH for single-limb divisor
  u8 n = v.num_limbs();
  if (n == 1) {
    u64 rem = 0;
    UInt<N> quo = {};
    u64 d = v.bits[0];
    for (int i = u.num_limbs() - 1; i >= 0; --i) {
      u128 temp = ((u128)rem << 64) | u.bits[i];
      quo.bits[i] = temp / d;
      rem = temp % d;
    }
    return {quo, UInt<N>(rem)};
  }
  


  // Use Division.hpp implementation
  UInt<N> q{}, r{};

  if constexpr (N < 34) {
    // Zone 1: Knuth D (Iterative)
    Division::div_knuth_impl<N>(q.bits.data(), r.bits.data(), u.bits.data(),
                             u.num_limbs(), v.bits.data(), v.num_limbs());
  } else {
    // Zone 2: Burnikel-Ziegler (Recursive)
    // Runtime check: if divisor is small, avoid recursion overhead
    // Runtime check: if difference in size is small, use Knuth.
    // BZ is efficient for 2n / n. If u and v are close in size, Knuth is faster (O(N * (N-M))).
    int u_len = u.num_limbs();
    int v_len = v.num_limbs();
    if (u_len - v_len < 32) {
      // Use templated implementation with stack buffer size N
      // This is safe because u_len <= N.
      Division::div_knuth_impl<N>(q.bits.data(), r.bits.data(), u.bits.data(),
                               u_len, v.bits.data(), v_len);
    } else {
      // Zone 2: Burnikel-Ziegler (Recursive)
      // We must normalize v so that the most significant bit is 1.
      // And we must provide a 2N buffer for u.
      
      // 1. Calculate shift to normalize v
      // We know v.num_limbs() >= 34 (from the check above).
      // Find the most significant limb.
      int n_limbs = v.num_limbs();
      int shift_limbs = N - n_limbs;
      int shift_bits = __builtin_clzll(v.bits[n_limbs - 1]);
      
      // Total shift needed to make MSB of v at bit (N*64 - 1)
      // v_norm = v << (shift_limbs * 64 + shift_bits)
      
      // Stack buffers
      // u_norm: 2N limbs (required by div_2n_1n)
      u64 u_norm[2 * N];
      u64 v_norm[N];
      // Stack buffer for recursion
      // Size estimation: Need ~5.5N for structure + Karatsuba scratch.
      // Increased to 20*N + 1000 to be absolutely safe against stack smashing.
      u64 tmp[20 * N + 1000]; // Scratch
      
      // 2. Normalize v
      // We can use the Division helpers if we make them public or duplicate logic.
      // For now, let's do it manually or use local helpers.
      // Actually, we can just use the shift operators of UInt if we cast/copy.
      // But we are inside UInt, so we can use private helpers? 
      // Let's use a simple loop for normalization.
      
      // Normalize v into v_norm
      if (shift_limbs > 0 || shift_bits > 0) {
          // Shift left by shift_limbs blocks
          for(int i = N - 1; i >= shift_limbs; --i)
              v_norm[i] = v.bits[i - shift_limbs];
          std::fill_n(v_norm, shift_limbs, 0);
          
          // Shift left by shift_bits
          if (shift_bits > 0) {
              u64 carry = 0;
              for(int i = 0; i < N; ++i) {
                  u64 val = v_norm[i];
                  v_norm[i] = (val << shift_bits) | carry;
                  carry = val >> (64 - shift_bits);
              }
          }
      } else {
          std::copy_n(v.bits.data(), N, v_norm);
      }
      
      // 3. Normalize u into u_norm (size 2N)
      // u is size N. We place it in u_norm, shifted.
      std::fill_n(u_norm, 2 * N, 0);
      
      // Copy u to u_norm with block shift
      // u_norm has 2N. u has N.
      // We shift u left by shift_limbs blocks.
      // u fits in lower N+shift_limbs? No, u is N.
      // u_norm is 2N.
      for(int i = 0; i < N; ++i) {
          u_norm[i + shift_limbs] = u.bits[i];
      }
      
      // Bit shift u_norm
      if (shift_bits > 0) {
          u64 carry = 0;
          // We only need to shift up to N + shift_limbs + 1?
          // Just shift all 2N to be safe and simple.
          for(int i = 0; i < 2 * N; ++i) {
              u64 val = u_norm[i];
              u_norm[i] = (val << shift_bits) | carry;
              carry = val >> (64 - shift_bits);
          }
      }
      
      // 4. Call Recursive Division
      Division::div_recursive(q.bits.data(), r.bits.data(), u_norm,
                              v_norm, N, tmp);
                              
      // 5. Unnormalize Remainder
      // r = r >> shift
      if (shift_bits > 0) {
          u64 carry = 0;
          for(int i = N - 1; i >= 0; --i) {
              u64 val = r.bits[i];
              r.bits[i] = (val >> shift_bits) | carry;
              carry = (val << (64 - shift_bits));
          }
      }
      if (shift_limbs > 0) {
          for(int i = 0; i < N - shift_limbs; ++i)
              r.bits[i] = r.bits[i + shift_limbs];
          std::fill_n(r.bits.data() + N - shift_limbs, shift_limbs, 0);
      }
    }
  }

  return {q, r};
}

template <u8 N> std::string UInt<N>::to_string() const {
  if (is_zero())
    return "0";
  UInt<N> temp = *this;
  constexpr u64 CHUNK_POW = 1000000000000000000ull;
  constexpr int CHUNK_SIZE = 18;
  
  // Max chunks needed: N * 64 bits * log10(2) / 18 approx N * 1.1
  // For N=255, ~280 chunks. Safe buffer 300.
  u64 chunks[300];
  int chunk_count = 0;
  
  while (!temp.is_zero()) {
    auto [quotient, remainder_uint] = divmod(temp, UInt<N>(CHUNK_POW));
    chunks[chunk_count++] = remainder_uint.bits[0];
    temp = std::move(quotient);
  }
  std::ostringstream oss;
  oss << chunks[chunk_count - 1];
  for (int i = chunk_count - 2; i >= 0; --i) {
    oss << std::setw(CHUNK_SIZE) << std::setfill('0') << chunks[i];
  }
  return oss.str();
}

template <u8 N> std::string UInt<N>::to_hex_string() const {
  if (is_zero())
    return "0x0";
  int msb_idx = N - 1;
  while (msb_idx > 0 && bits[msb_idx] == 0)
    --msb_idx;
  std::ostringstream oss;
  oss << "0x" << std::hex << std::nouppercase;
  oss << bits[msb_idx];
  for (int i = msb_idx - 1; i >= 0; --i)
    oss << std::setw(16) << std::setfill('0') << bits[i];
  return oss.str();
}

// --- Free Operators ---
template <u8 N>
[[nodiscard]] FORCE_INLINE constexpr UInt<N> operator+(UInt<N> lhs,
                                                 const UInt<N> &rhs) noexcept {
  return lhs += rhs;
}
template <u8 N>
[[nodiscard]] FORCE_INLINE constexpr UInt<N> operator-(UInt<N> lhs,
                                                 const UInt<N> &rhs) noexcept {
  return lhs -= rhs;
}
template <u8 N>
[[nodiscard]] FORCE_INLINE UInt<N> operator*(UInt<N> lhs,
                                       const UInt<N> &rhs) noexcept {
  return lhs *= rhs;
}
template <u8 N>
[[nodiscard]] FORCE_INLINE UInt<N> operator/(UInt<N> lhs, const UInt<N> &rhs) {
  return lhs /= rhs;
}
template <u8 N>
[[nodiscard]] FORCE_INLINE UInt<N> operator%(UInt<N> lhs, const UInt<N> &rhs) {
  return lhs %= rhs;
}
template <u8 N>
[[nodiscard]] FORCE_INLINE constexpr UInt<N> operator<<(UInt<N> lhs,
                                                  u16 rhs) noexcept {
  return lhs <<= rhs;
}
template <u8 N>
[[nodiscard]] FORCE_INLINE constexpr UInt<N> operator>>(UInt<N> lhs,
                                                  u16 rhs) noexcept {
  return lhs >>= rhs;
}
