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
#define INLINE inline __attribute__((always_inline))

template <const u16 B> class alignas(64) UInt;

// Main Class Definition
template <const u16 B> class alignas(64) UInt {
  static_assert(B > 0 && B % 64 == 0, "Bits must be a positive multiple of 64");

public:
  static constexpr u16 TotalBlocks = B / 64;
  std::array<u64, TotalBlocks> bits{}; // initialize all elements to zero

private:
  INLINE constexpr u8 charToValue(char c) const {
    static constexpr auto lut = []() { // lookup table
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

  [[nodiscard]] INLINE u16 num_limbs() const noexcept {
    for (int i = TotalBlocks - 1; i >= 0; --i) {
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
      if (sv.size() > (TotalBlocks * 16))
        throw std::out_of_range("Hex string too long");
      u16 size = sv.size();
      for (u16 i = 0; i < TotalBlocks && size > 0; ++i) {
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

  [[nodiscard]] INLINE constexpr bool operator==(const UInt &o) const noexcept {
    return bits == o.bits;
  }
  [[nodiscard]] INLINE constexpr bool operator!=(const UInt &o) const noexcept {
    return !(*this == o);
  }
  [[nodiscard]] INLINE constexpr bool operator==(u64 o) const noexcept {
    if (bits[0] != o)
      return false;
    for (u16 i = 1; i < TotalBlocks; ++i)
      if (bits[i] != 0)
        return false;
    return true;
  }
  [[nodiscard]] INLINE constexpr bool operator<(const UInt &o) const noexcept {
    for (int i = TotalBlocks - 1; i >= 0; --i) {
      if (bits[i] < o.bits[i])
        return true;
      if (bits[i] > o.bits[i])
        return false;
    }
    return false;
  }
  [[nodiscard]] INLINE constexpr bool operator>(const UInt &o) const noexcept {
    return o < *this;
  }
  [[nodiscard]] INLINE constexpr bool operator<=(const UInt &o) const noexcept {
    return !(o < *this);
  }
  [[nodiscard]] INLINE constexpr bool operator>=(const UInt &o) const noexcept {
    return !(*this < o);
  }

  INLINE constexpr UInt &operator++() noexcept {
    if (++bits[0] != 0) [[likely]]
      return *this;
    u16 index = 1;
    while (index < TotalBlocks) {
      if (++bits[index] != 0) [[likely]]
        return *this; // Carry absorvido, retorna imediatamente
      ++index;
    }
    return *this;
  }

  INLINE constexpr UInt &operator--() noexcept {
    if (--bits[0] != 0) [[likely]]
      return *this;
    u16 index = 1;
    while (index < TotalBlocks) {
      if (--bits[index] != 0) [[likely]]
        return *this; // Borrow absorvido, retorna imediatamente
      ++index;
    }
    return *this;
  }
  INLINE constexpr UInt operator++(int) noexcept {
    UInt t = *this;
    ++*this;
    return t;
  }
  INLINE constexpr UInt operator--(int) noexcept {
    UInt t = *this;
    --*this;
    return t;
  }

  INLINE UInt &operator+=(const UInt &other) noexcept {
    if consteval {
      // Versão constexpr (compatibilidade)
      u64 c = 0;
      for (u16 i = 0; i < TotalBlocks; ++i) {
        u128 t = (u128)bits[i] + other.bits[i] + c;
        bits[i] = (u64)t;
        c = t >> 64;
      }
    } else {
      if constexpr (TotalBlocks == 1) {
        unsigned char c = _addcarry_u64(0, bits[0], other.bits[0],
                                        (unsigned long long *)&bits[0]);
        (void)c;
      } else if constexpr (TotalBlocks == 2) {
        unsigned char c = _addcarry_u64(0, bits[0], other.bits[0],
                                        (unsigned long long *)&bits[0]);
        _addcarry_u64(c, bits[1], other.bits[1],
                      (unsigned long long *)&bits[1]);
      } else if constexpr (TotalBlocks == 3) {
        unsigned char c = _addcarry_u64(0, bits[0], other.bits[0],
                                        (unsigned long long *)&bits[0]);
        c = _addcarry_u64(c, bits[1], other.bits[1],
                          (unsigned long long *)&bits[1]);
        _addcarry_u64(c, bits[2], other.bits[2],
                      (unsigned long long *)&bits[2]);
      } else if constexpr (TotalBlocks == 4) {
        unsigned char c = _addcarry_u64(0, bits[0], other.bits[0],
                                        (unsigned long long *)&bits[0]);
        c = _addcarry_u64(c, bits[1], other.bits[1],
                          (unsigned long long *)&bits[1]);
        c = _addcarry_u64(c, bits[2], other.bits[2],
                          (unsigned long long *)&bits[2]);
        _addcarry_u64(c, bits[3], other.bits[3],
                      (unsigned long long *)&bits[3]);
      } else if constexpr (TotalBlocks <= 64) {
        unsigned char c = _addcarry_u64(0, bits[0], other.bits[0],
                                        (unsigned long long *)&bits[0]);
        auto unroll_step = [&](auto i) {
          c = _addcarry_u64(c, bits[i], other.bits[i],
                            (unsigned long long *)&bits[i]);
        };
        [&]<std::size_t... I>(std::index_sequence<I...>) {
          (unroll_step(std::integral_constant<std::size_t, I + 1>{}), ...);
        }(std::make_index_sequence<TotalBlocks - 1>{});
      } else {
        unsigned char carry = _addcarry_u64(0, bits[0], other.bits[0],
                                            (unsigned long long *)&bits[0]);
#pragma GCC unroll 16
        for (u16 i = 1; i < TotalBlocks; ++i) {
          carry = _addcarry_u64(carry, bits[i], other.bits[i],
                                (unsigned long long *)&bits[i]);
        }
      }
    }
    return *this;
  }
  INLINE UInt &operator+=(u64 val) noexcept {
    u64 c = val;
    u16 i = 0;
    while (c > 0 && i < TotalBlocks) {
      u128 t = (u128)bits[i] + c;
      bits[i] = (u64)t;
      c = t >> 64;
      i++;
    }
    return *this;
  }
  INLINE UInt &operator-=(const UInt &other) noexcept {
    if consteval {
      // Versão constexpr (compatibilidade)
      u64 b = 0;
      for (u16 i = 0; i < TotalBlocks; ++i) {
        u128 t = (u128)bits[i] - other.bits[i] - b;
        bits[i] = (u64)t;
        b = (t >> 64) & 1;
      }
    } else {
      if constexpr (TotalBlocks == 1) {
        unsigned char b = _subborrow_u64(0, bits[0], other.bits[0],
                                         (unsigned long long *)&bits[0]);
        (void)b;
      } else if constexpr (TotalBlocks == 2) {
        unsigned char b = _subborrow_u64(0, bits[0], other.bits[0],
                                         (unsigned long long *)&bits[0]);
        _subborrow_u64(b, bits[1], other.bits[1],
                       (unsigned long long *)&bits[1]);
      } else if constexpr (TotalBlocks == 3) {
        unsigned char b = _subborrow_u64(0, bits[0], other.bits[0],
                                         (unsigned long long *)&bits[0]);
        b = _subborrow_u64(b, bits[1], other.bits[1],
                           (unsigned long long *)&bits[1]);
        _subborrow_u64(b, bits[2], other.bits[2],
                       (unsigned long long *)&bits[2]);
      } else if constexpr (TotalBlocks == 4) {
        unsigned char b = _subborrow_u64(0, bits[0], other.bits[0],
                                         (unsigned long long *)&bits[0]);
        b = _subborrow_u64(b, bits[1], other.bits[1],
                           (unsigned long long *)&bits[1]);
        b = _subborrow_u64(b, bits[2], other.bits[2],
                           (unsigned long long *)&bits[2]);
        _subborrow_u64(b, bits[3], other.bits[3],
                       (unsigned long long *)&bits[3]);
      } else if constexpr (TotalBlocks <= 64) {
        unsigned char b = _subborrow_u64(0, bits[0], other.bits[0],
                                         (unsigned long long *)&bits[0]);
        auto unroll_step = [&](auto i) {
          b = _subborrow_u64(b, bits[i], other.bits[i],
                             (unsigned long long *)&bits[i]);
        };
        [&]<std::size_t... I>(std::index_sequence<I...>) {
          (unroll_step(std::integral_constant<std::size_t, I + 1>{}), ...);
        }(std::make_index_sequence<TotalBlocks - 1>{});
      } else {
        // Versão com loop e hint de unrolling
        unsigned char borrow = _subborrow_u64(0, bits[0], other.bits[0],
                                              (unsigned long long *)&bits[0]);
#pragma GCC unroll 8
        for (u16 i = 1; i < TotalBlocks; ++i) {
          borrow = _subborrow_u64(borrow, bits[i], other.bits[i],
                                  (unsigned long long *)&bits[i]);
        }
      }
    }
    return *this;
  }
  INLINE UInt &operator*=(const UInt &other) noexcept;
  INLINE UInt &operator*=(u64 val) noexcept;
  INLINE UInt &operator/=(const UInt &other) {
    *this = divmod(*this, other).first;
    return *this;
  }
  INLINE UInt &operator%=(const UInt &other) {
    *this = divmod(*this, other).second;
    return *this;
  }
  INLINE constexpr UInt &operator&=(const UInt &other) noexcept {
    for (u16 i = 0; i < TotalBlocks; ++i)
      bits[i] &= other.bits[i];
    return *this;
  }
  INLINE constexpr UInt &operator|=(const UInt &other) noexcept {
    for (u16 i = 0; i < TotalBlocks; ++i)
      bits[i] |= other.bits[i];
    return *this;
  }
  INLINE constexpr UInt &operator^=(const UInt &other) noexcept {
    for (u16 i = 0; i < TotalBlocks; ++i)
      bits[i] ^= other.bits[i];
    return *this;
  }
  INLINE constexpr UInt &operator<<=(u16 n) noexcept;
  INLINE constexpr UInt &operator>>=(u16 n) noexcept;

  [[nodiscard]] static std::pair<UInt, UInt> divmod(UInt u, UInt v);

  [[nodiscard]] std::string to_string() const;
  [[nodiscard]] std::string to_hex_string() const;
  [[nodiscard]] INLINE constexpr bool is_zero() const noexcept {
    for (u64 limb : bits)
      if (limb != 0)
        return false;
    return true;
  }
  [[nodiscard]] INLINE constexpr bool bt(const u16 index) const noexcept {
    if (BUILTIN_EXPECT(index >= B, 0))
      return false;
    return (bits[index / 64] >> (index % 64)) & 1;
  }
  INLINE constexpr void bts(const u16 index) noexcept {
    if (BUILTIN_EXPECT(index < B, 1))
      bits[index / 64] |= (1ull << (index % 64));
  }
};

// --- Method Implementations ---

template <u16 B>
INLINE UInt<B> &UInt<B>::operator*=(const UInt<B> &other) noexcept {
  UInt<B> self_copy = *this;
  this->bits.fill(0);

  if constexpr (TotalBlocks <= 2) {
    // Small sizes: Use intrinsics manually for best latency
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      auto outer_step = [&](auto i_const) {
        constexpr u16 i = i_const;
        u64 y = self_copy.bits[i];
        if (y == 0)
          return;

        u128 carry = 0;
        [&]<std::size_t... J>(std::index_sequence<J...>) {
          auto inner_step = [&](auto j_const) {
            constexpr u16 j = j_const;
            if constexpr (i + j < TotalBlocks) {
              u64 hi, lo;
              lo = _mulx_u64(other.bits[j], y, &hi);

              u64 c_in = (u64)carry;
              u64 c_extra = (u64)(carry >> 64);

              unsigned char c1 =
                  _addcarry_u64(0, this->bits[i + j], lo, &this->bits[i + j]);
              unsigned char c2 =
                  _addcarry_u64(0, this->bits[i + j], c_in, &this->bits[i + j]);

              carry = (u128)hi + c1 + c2 + c_extra;
            }
          };
          (inner_step(std::integral_constant<std::size_t, J>{}), ...);
        }(std::make_index_sequence<TotalBlocks - i>{});
      };
      (outer_step(std::integral_constant<std::size_t, I>{}), ...);
    }(std::make_index_sequence<TotalBlocks>{});
  } else if constexpr (TotalBlocks <= 32) {
    // Medium sizes: Use u128 arithmetic, letting compiler schedule carries.
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      auto outer_step = [&](auto i_const) {
        constexpr u16 i = i_const;
        u64 y = self_copy.bits[i];
        if (y == 0)
          return;

        u64 c = 0;
        [&]<std::size_t... J>(std::index_sequence<J...>) {
          auto inner_step = [&](auto j_const) {
            constexpr u16 j = j_const;
            if constexpr (i + j < TotalBlocks) {
              u128 temp = (u128)other.bits[j] * y + this->bits[i + j] + c;
              this->bits[i + j] = (u64)temp;
              c = temp >> 64;
            }
          };
          (inner_step(std::integral_constant<std::size_t, J>{}), ...);
        }(std::make_index_sequence<TotalBlocks - i>{});
      };
      (outer_step(std::integral_constant<std::size_t, I>{}), ...);
    }(std::make_index_sequence<TotalBlocks>{});
  } else {
    // Large sizes: Use Karatsuba
    // Allocate temp buffer once. Size needed is approx 8*N.
    // For N=4096 bits (64 u64s), 8*64 = 512 u64s = 4KB. Small enough for stack?
    // Maybe not for deep recursion or very large N.
    // Let's use a vector for now, but it's ONE allocation per operator*= call,
    // not recursive. Ideally we'd use a thread_local scratch buffer or a custom
    // allocator. For now, std::vector is fine as it's O(1) allocation count vs
    // O(log N) before. Actually, before it was 1 allocation per call too (in
    // mul_truncated). But we can optimize this further later.

    size_t n = TotalBlocks;
    // Buffer size: 8n + 1000 safety
    std::vector<u64> tmp(8 * n + 1000);
    Karatsuba::mul_truncated(this->bits.data(), self_copy.bits.data(),
                             other.bits.data(), TotalBlocks, tmp.data());
  }
  return *this;
}

template <u16 B> INLINE UInt<B> &UInt<B>::operator*=(u64 val) noexcept {
  u64 c = 0;
  for (u16 i = 0; i < TotalBlocks; ++i) {
    u128 temp = (u128)bits[i] * val + c;
    bits[i] = (u64)temp;
    c = temp >> 64;
  }
  return *this;
}

template <u16 B>
INLINE constexpr UInt<B> &UInt<B>::operator<<=(u16 n) noexcept {
  if (BUILTIN_EXPECT(n == 0, 1))
    return *this;
  if (n >= B) {
    bits.fill(0);
    return *this;
  }
  const u16 block_shift = n / 64;
  const u16 bit_shift = n % 64;
  if (block_shift > 0) {
    for (int i = TotalBlocks - 1; i >= (int)block_shift; --i)
      bits[i] = bits[i - block_shift];
    std::fill(bits.begin(), bits.begin() + block_shift, 0ull);
  }
  if (bit_shift > 0) {
    u64 carry = 0;
    for (u16 i = 0; i < TotalBlocks; ++i) {
      u64 next_carry = bits[i] >> (64 - bit_shift);
      bits[i] = (bits[i] << bit_shift) | carry;
      carry = next_carry;
    }
  }
  return *this;
}

template <u16 B>
INLINE constexpr UInt<B> &UInt<B>::operator>>=(u16 n) noexcept {
  if (BUILTIN_EXPECT(n == 0, 1))
    return *this;
  if (n >= B) {
    bits.fill(0);
    return *this;
  }
  const u16 block_shift = n / 64;
  const u16 bit_shift = n % 64;
  if (block_shift > 0) {
    for (u16 i = 0; i < TotalBlocks - block_shift; ++i)
      bits[i] = bits[i + block_shift];
    std::fill(bits.begin() + TotalBlocks - block_shift, bits.end(), 0ull);
  }
  if (bit_shift > 0) {
    u64 carry = 0;
    for (int i = TotalBlocks - 1; i >= 0; --i) {
      u64 next_carry = bits[i] << (64 - bit_shift);
      bits[i] = (bits[i] >> bit_shift) | carry;
      carry = next_carry;
    }
  }
  return *this;
}

// Hybrid division: fast path for single-limb divisor, correct slow path for
// others.
template <u16 B>
std::pair<UInt<B>, UInt<B>> UInt<B>::divmod(UInt<B> u, UInt<B> v) {
  if (v.is_zero())
    throw std::domain_error("Division by zero");
  if (u < v)
    return {UInt<B>(0), u};

  // FAST PATH for single-limb divisor
  u16 n = v.num_limbs();
  if (n == 1) {
    u64 rem = 0;
    UInt<B> quo = {};
    u64 d = v.bits[0];
    for (int i = u.num_limbs() - 1; i >= 0; --i) {
      u128 temp = ((u128)rem << 64) | u.bits[i];
      quo.bits[i] = temp / d;
      rem = temp % d;
    }
    return {quo, UInt<B>(rem)};
  }

  // Knuth's Algorithm D (The Art of Computer Programming, Vol 2, 4.3.1)
  u16 m = u.num_limbs();

  // D1: Normalize - shift v left so its MSB is >= 2^63
  const int shift = __builtin_clzll(v.bits[n - 1]);

  // Allocate normalized dividend (m+1 limbs needed)
  std::array<u64, TotalBlocks + 1> u_norm{};

  if (__builtin_expect(shift > 0, 1)) {
    // Normalize divisor and dividend
    v <<= shift;
    u64 carry = 0;
    for (u16 i = 0; i < m; ++i) {
      u64 val = u.bits[i];
      u_norm[i] = (val << shift) | carry;
      carry = val >> (64 - shift);
    }
    u_norm[m] = carry;
  } else {
    // No normalization needed - fast path
    for (u16 i = 0; i < m; ++i)
      u_norm[i] = u.bits[i];
  }

  UInt<B> q{};
  const u64 v_high = v.bits[n - 1];
  const u64 v_next = (n > 1) ? v.bits[n - 2] : 0;

  // D2-D7: Main loop - compute quotient digits
  for (int j = m - n; j >= 0; --j) {
    // D3: Calculate trial quotient digit
    u64 u_high = u_norm[j + n];
    u64 u_mid = u_norm[j + n - 1];
    u64 q_hat, r_hat;

    if (u_high == v_high) {
      q_hat = ~0ULL;
      r_hat = u_mid + v_high;
    } else {
      u128 dividend = ((u128)u_high << 64) | u_mid;
      q_hat = dividend / v_high;
      r_hat = dividend % v_high;
    }

    // D3 continued: Refine q_hat to ensure it's not too large
    while (q_hat > 0) {
      u128 lhs = (u128)q_hat * v_next;
      u128 rhs = ((u128)r_hat << 64) | u_norm[j + n - 2];
      if (lhs <= rhs)
        break;
      q_hat--;
      r_hat += v_high;
      if (r_hat < v_high)
        break; // Overflow
    }

    // D4: Multiply and subtract - u_norm[j..j+n] -= q_hat * v[0..n-1]
    u64 mult_carry = 0;
    unsigned char sub_borrow = 0;

    // Unroll inner loop for small N (common case in recursive division or just
    // small numbers) But n is runtime variable here (divisor size). We can't
    // use constexpr if for n. But we can use a Duff's device or just rely on
    // compiler loop unrolling. Let's try to help compiler by splitting the
    // loop.

    u16 i = 0;
    // Main loop unrolled by 4
    for (; i + 4 <= n; i += 4) {
      u64 hi, lo;
      unsigned char c;

      // 0
      lo = _mulx_u64(q_hat, v.bits[i], &hi);
      c = _addcarry_u64(0, lo, mult_carry, &lo);
      mult_carry = hi + c;
      sub_borrow =
          _subborrow_u64(sub_borrow, u_norm[j + i], lo, &u_norm[j + i]);

      // 1
      lo = _mulx_u64(q_hat, v.bits[i + 1], &hi);
      c = _addcarry_u64(0, lo, mult_carry, &lo);
      mult_carry = hi + c;
      sub_borrow =
          _subborrow_u64(sub_borrow, u_norm[j + i + 1], lo, &u_norm[j + i + 1]);

      // 2
      lo = _mulx_u64(q_hat, v.bits[i + 2], &hi);
      c = _addcarry_u64(0, lo, mult_carry, &lo);
      mult_carry = hi + c;
      sub_borrow =
          _subborrow_u64(sub_borrow, u_norm[j + i + 2], lo, &u_norm[j + i + 2]);

      // 3
      lo = _mulx_u64(q_hat, v.bits[i + 3], &hi);
      c = _addcarry_u64(0, lo, mult_carry, &lo);
      mult_carry = hi + c;
      sub_borrow =
          _subborrow_u64(sub_borrow, u_norm[j + i + 3], lo, &u_norm[j + i + 3]);
    }

    // Handle remaining
    for (; i < n; ++i) {
      u64 hi, lo;
      lo = _mulx_u64(q_hat, v.bits[i], &hi);

      unsigned char c = _addcarry_u64(0, lo, mult_carry, &lo);
      mult_carry = hi + c;

      sub_borrow =
          _subborrow_u64(sub_borrow, u_norm[j + i], lo, &u_norm[j + i]);
    }

    u64 old_high = u_norm[j + n];
    u_norm[j + n] = old_high - mult_carry - sub_borrow;
    sub_borrow = (u_norm[j + n] > old_high) ||
                 ((mult_carry || sub_borrow) && u_norm[j + n] >= old_high);

    // D5 & D6: Test and add back if needed (rare - happens ~1/2^64)
    if (__builtin_expect(sub_borrow, 0)) {
      q_hat--;
      u64 add_carry = 0;
      for (u16 i = 0; i < n; ++i) {
        u128 sum = (u128)u_norm[j + i] + v.bits[i] + add_carry;
        u_norm[j + i] = (u64)sum;
        add_carry = sum >> 64;
      }
      u_norm[j + n] += add_carry;
    }

    q.bits[j] = q_hat;
  }

  // D8: Unnormalize remainder
  UInt<B> r{};
  if (__builtin_expect(shift > 0, 1)) {
    u64 carry = 0;
    for (int i = n - 1; i >= 0; --i) {
      u64 val = u_norm[i];
      r.bits[i] = (val >> shift) | carry;
      carry = (val & ((1ULL << shift) - 1)) << (64 - shift);
    }
  } else {
    // No denormalization needed - fast path
    for (u16 i = 0; i < n; ++i)
      r.bits[i] = u_norm[i];
  }

  return {q, r};
}

template <u16 B> std::string UInt<B>::to_string() const {
  if (is_zero())
    return "0";
  UInt<B> temp = *this;
  constexpr u64 CHUNK_POW = 1000000000000000000ull;
  constexpr int CHUNK_SIZE = 18;
  std::vector<u64> chunks;
  chunks.reserve((B * 301LLU / 1000) / 18 + 2);
  while (!temp.is_zero()) {
    auto [quotient, remainder_uint] = divmod(temp, UInt<B>(CHUNK_POW));
    chunks.push_back(remainder_uint.bits[0]);
    temp = std::move(quotient);
  }
  std::ostringstream oss;
  oss << chunks.back();
  for (auto it = chunks.rbegin() + 1; it != chunks.rend(); ++it) {
    oss << std::setw(CHUNK_SIZE) << std::setfill('0') << *it;
  }
  return oss.str();
}

template <u16 B> std::string UInt<B>::to_hex_string() const {
  if (is_zero())
    return "0x0";
  int msb_idx = TotalBlocks - 1;
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
template <u16 B>
[[nodiscard]] INLINE constexpr UInt<B> operator+(UInt<B> lhs,
                                                 const UInt<B> &rhs) noexcept {
  return lhs += rhs;
}
template <u16 B>
[[nodiscard]] INLINE constexpr UInt<B> operator-(UInt<B> lhs,
                                                 const UInt<B> &rhs) noexcept {
  return lhs -= rhs;
}
template <u16 B>
[[nodiscard]] INLINE UInt<B> operator*(UInt<B> lhs,
                                       const UInt<B> &rhs) noexcept {
  return lhs *= rhs;
}
template <u16 B>
[[nodiscard]] INLINE UInt<B> operator/(UInt<B> lhs, const UInt<B> &rhs) {
  return lhs /= rhs;
}
template <u16 B>
[[nodiscard]] INLINE UInt<B> operator%(UInt<B> lhs, const UInt<B> &rhs) {
  return lhs %= rhs;
}
template <u16 B>
[[nodiscard]] INLINE constexpr UInt<B> operator<<(UInt<B> lhs,
                                                  u16 rhs) noexcept {
  return lhs <<= rhs;
}
template <u16 B>
[[nodiscard]] INLINE constexpr UInt<B> operator>>(UInt<B> lhs,
                                                  u16 rhs) noexcept {
  return lhs >>= rhs;
}
