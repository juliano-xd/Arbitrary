#pragma once

#include <array>
#include <charconv>
#include <cstring>
#include <iomanip>
#include <immintrin.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <utility>
#include <stdexcept>
#include <algorithm>
#include <limits>

// Type Aliases
using u8 = unsigned char;
using u16 = unsigned short;
using u64 = unsigned long long;
using u128 = unsigned __int128;

#define BUILTIN_EXPECT(x, y) (__builtin_expect(!!(x), y))
#define INLINE inline __attribute__((always_inline))

template <const u16 B> class alignas(64) UInt;

// Main Class Definition
template <const u16 B>
class alignas(64) UInt {
    static_assert(B > 0 && B % 64 == 0, "Bits must be a positive multiple of 64");

public:
    static constexpr u16 TotalBlocks = B / 64;
    std::array<u64, TotalBlocks> bits{};

private:
    INLINE constexpr u8 charToValue(char c) const {
        static constexpr auto lut = []() {
            std::array<u8, 256> table; table.fill(255);
            for (u8 i = 0; i < 10; ++i) table['0' + i] = i;
            for (u8 i = 0; i < 6; ++i) { table['a' + i] = 10 + i; table['A' + i] = 10 + i; }
            return table;
        }();
        u8 value = lut[static_cast<u8>(c)];
        if (BUILTIN_EXPECT(value > 15, 0)) throw std::invalid_argument("Invalid character");
        return value;
    }

    [[nodiscard]] INLINE u16 num_limbs() const noexcept {
        for (int i = TotalBlocks - 1; i >= 0; --i) {
            if (bits[i] != 0) return i + 1;
        }
        return 0;
    }

public:
    constexpr UInt() noexcept = default;
    constexpr explicit UInt(u64 value) noexcept : bits{value} {}
    constexpr UInt(const UInt& o) noexcept = default;
    constexpr UInt(UInt&& o) noexcept = default;

    constexpr explicit UInt(std::string_view sv) {
        bits.fill(0);
        if (sv.empty()) return;
        bool signal = (sv.front() == '-');
        if (signal || sv.front() == '+') sv.remove_prefix(1);
        if (sv.empty()) return;
        const bool is_hex = sv.starts_with("0x") || sv.starts_with("0X");
        if (is_hex) {
            sv.remove_prefix(2);
            if (sv.empty()) return;
            if (sv.size() > (TotalBlocks * 16)) throw std::out_of_range("Hex string too long");
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
            const char* p = sv.data();
            const char* const end = p + sv.size();
            u16 first_chunk_len = sv.size() % CHUNK_SIZE;
            if (first_chunk_len == 0 && !sv.empty()) first_chunk_len = CHUNK_SIZE;
            u64 chunk_val = 0;
            const char* const end_first_chunk = p + first_chunk_len;
            auto res = std::from_chars(p, end_first_chunk, chunk_val);
            if (BUILTIN_EXPECT(res.ec != std::errc(), 0)) throw std::invalid_argument("Invalid decimal character");
            *this = chunk_val;
            p = end_first_chunk;
            while (p < end) {
                const char* const end_chunk = p + CHUNK_SIZE;
                res = std::from_chars(p, end_chunk, chunk_val);
                if (BUILTIN_EXPECT(res.ec != std::errc(), 0)) throw std::invalid_argument("Invalid decimal character");
                *this *= CHUNK_POW;
                *this += chunk_val;
                p = end_chunk;
            }
        }
        if (signal) throw std::invalid_argument("Negative values not supported in UInt");
    }

    constexpr UInt& operator=(const UInt& o) noexcept = default;
    constexpr UInt& operator=(UInt&& o) noexcept = default;
    constexpr UInt& operator=(u64 o) noexcept { bits.fill(0); bits[0] = o; return *this; }

    [[nodiscard]] INLINE constexpr bool operator==(const UInt& o) const noexcept { return bits == o.bits; }
    [[nodiscard]] INLINE constexpr bool operator!=(const UInt& o) const noexcept { return !(*this == o); }
    [[nodiscard]] INLINE constexpr bool operator==(u64 o) const noexcept { if (bits[0] != o) return false; for (u16 i = 1; i < TotalBlocks; ++i) if (bits[i] != 0) return false; return true; }
    [[nodiscard]] INLINE constexpr bool operator<(const UInt& o) const noexcept { for (int i=TotalBlocks-1; i>=0; --i) { if(bits[i]<o.bits[i]) return true; if(bits[i]>o.bits[i]) return false; } return false; }
    [[nodiscard]] INLINE constexpr bool operator>(const UInt& o) const noexcept { return o < *this; }
    [[nodiscard]] INLINE constexpr bool operator<=(const UInt& o) const noexcept { return !(o < *this); }
    [[nodiscard]] INLINE constexpr bool operator>=(const UInt& o) const noexcept { return !(*this < o); }

    INLINE constexpr UInt& operator++() noexcept { u16 i=0; while(i<TotalBlocks && ++bits[i]==0) {++i;} return *this; }
    INLINE constexpr UInt& operator--() noexcept { u16 i=0; while(i<TotalBlocks && --bits[i]==~0ull) {++i;} return *this; }
    INLINE constexpr UInt operator++(int) noexcept { UInt t=*this; ++*this; return t; }
    INLINE constexpr UInt operator--(int) noexcept { UInt t=*this; --*this; return t; }

    INLINE UInt& operator+=(const UInt& other) noexcept { u64 c=0; for(u16 i=0;i<TotalBlocks;++i){u128 t=(u128)bits[i]+other.bits[i]+c; bits[i]=(u64)t; c=t>>64;} return *this; }
    INLINE UInt& operator+=(u64 val) noexcept { u64 c=val; u16 i=0; while(c>0 && i<TotalBlocks){ u128 t=(u128)bits[i]+c; bits[i]=(u64)t; c=t>>64; i++;} return *this; }
    INLINE UInt& operator-=(const UInt& other) noexcept { u64 b=0; for(u16 i=0;i<TotalBlocks;++i){u128 t=(u128)bits[i]-other.bits[i]-b; bits[i]=(u64)t; b=(t>>64)&1;} return *this; }
    INLINE UInt& operator*=(const UInt& other) noexcept;
    INLINE UInt& operator*=(u64 val) noexcept;
    INLINE UInt& operator/=(const UInt& other) { *this = divmod(*this, other).first; return *this; }
    INLINE UInt& operator%=(const UInt& other) { *this = divmod(*this, other).second; return *this; }
    INLINE constexpr UInt& operator&=(const UInt& other) noexcept { for(u16 i=0;i<TotalBlocks;++i) bits[i]&=other.bits[i]; return *this; }
    INLINE constexpr UInt& operator|=(const UInt& other) noexcept { for(u16 i=0;i<TotalBlocks;++i) bits[i]|=other.bits[i]; return *this; }
    INLINE constexpr UInt& operator^=(const UInt& other) noexcept { for(u16 i=0;i<TotalBlocks;++i) bits[i]^=other.bits[i]; return *this; }
    INLINE constexpr UInt& operator<<=(u16 n) noexcept;
    INLINE constexpr UInt& operator>>=(u16 n) noexcept;

    [[nodiscard]] static std::pair<UInt, UInt> divmod(UInt u, UInt v);

    [[nodiscard]] std::string to_string() const;
    [[nodiscard]] std::string to_hex_string() const;
    [[nodiscard]] INLINE constexpr bool is_zero() const noexcept { for(u64 limb : bits) if(limb!=0) return false; return true; }
    [[nodiscard]] INLINE constexpr bool bt(const u16 index) const noexcept { if (BUILTIN_EXPECT(index >= B, 0)) return false; return (bits[index / 64] >> (index % 64)) & 1; }
    INLINE constexpr void bts(const u16 index) noexcept { if (BUILTIN_EXPECT(index < B, 1)) bits[index / 64] |= (1ull << (index % 64)); }
};

// --- Method Implementations ---

template<u16 B> INLINE UInt<B>& UInt<B>::operator*=(const UInt<B>& other) noexcept {
    UInt<B> self_copy = *this;
    this->bits.fill(0);
    for (u16 i = 0; i < TotalBlocks; ++i) {
        if (self_copy.bits[i] == 0) continue;
        u64 c = 0;
        for (u16 j = 0; j < TotalBlocks - i; ++j) {
            u128 temp = (u128)other.bits[j] * self_copy.bits[i] + this->bits[i + j] + c;
            this->bits[i + j] = (u64)temp;
            c = temp >> 64;
        }
    }
    return *this;
}

template<u16 B> INLINE UInt<B>& UInt<B>::operator*=(u64 val) noexcept {
    u64 c = 0;
    for (u16 i = 0; i < TotalBlocks; ++i) {
        u128 temp = (u128)bits[i] * val + c;
        bits[i] = (u64)temp;
        c = temp >> 64;
    }
    return *this;
}

template<u16 B> INLINE constexpr UInt<B>& UInt<B>::operator<<=(u16 n) noexcept {
    if (BUILTIN_EXPECT(n == 0, 1)) return *this;
    if (n >= B) { bits.fill(0); return *this; }
    const u16 block_shift = n / 64; const u16 bit_shift = n % 64;
    if (block_shift > 0) {
        for (int i = TotalBlocks - 1; i >= (int)block_shift; --i) bits[i] = bits[i - block_shift];
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

template<u16 B> INLINE constexpr UInt<B>& UInt<B>::operator>>=(u16 n) noexcept {
    if (BUILTIN_EXPECT(n == 0, 1)) return *this;
    if (n >= B) { bits.fill(0); return *this; }
    const u16 block_shift = n / 64; const u16 bit_shift = n % 64;
    if (block_shift > 0) {
        for (u16 i = 0; i < TotalBlocks - block_shift; ++i) bits[i] = bits[i + block_shift];
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

// Hybrid division: fast path for single-limb divisor, correct slow path for others.
template<u16 B> std::pair<UInt<B>, UInt<B>> UInt<B>::divmod(UInt<B> u, UInt<B> v) {
    if (v.is_zero()) throw std::domain_error("Division by zero");
    if (u < v) return {UInt<B>(0), u};

    // FAST PATH for single-limb divisor (huge speedup for to_string)
    if (v.num_limbs() == 1) {
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

    // SLOW BUT CORRECT restoring binary division for multi-limb divisors
    UInt<B> q = {};
    UInt<B> r = {};
    for (int i = B - 1; i >= 0; --i) {
        r <<= 1;
        if (u.bt(i)) r.bits[0] |= 1;
        if (r >= v) {
            r -= v;
            q.bts(i);
        }
    }
    return {q, r};
}

template<u16 B> std::string UInt<B>::to_string() const {
    if (is_zero()) return "0";
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

template<u16 B> std::string UInt<B>::to_hex_string() const {
    if (is_zero()) return "0x0";
    int msb_idx = TotalBlocks - 1;
    while (msb_idx > 0 && bits[msb_idx] == 0) --msb_idx;
    std::ostringstream oss; oss << "0x" << std::hex << std::nouppercase;
    oss << bits[msb_idx];
    for (int i = msb_idx - 1; i >= 0; --i) oss << std::setw(16) << std::setfill('0') << bits[i];
    return oss.str();
}

// --- Free Operators ---
template<u16 B> [[nodiscard]] INLINE constexpr UInt<B> operator+(UInt<B> lhs, const UInt<B>& rhs) noexcept { return lhs += rhs; }
template<u16 B> [[nodiscard]] INLINE constexpr UInt<B> operator-(UInt<B> lhs, const UInt<B>& rhs) noexcept { return lhs -= rhs; }
template<u16 B> [[nodiscard]] INLINE UInt<B> operator*(UInt<B> lhs, const UInt<B>& rhs) noexcept { return lhs *= rhs; }
template<u16 B> [[nodiscard]] INLINE UInt<B> operator/(UInt<B> lhs, const UInt<B>& rhs) { return lhs /= rhs; }
template<u16 B> [[nodiscard]] INLINE UInt<B> operator%(UInt<B> lhs, const UInt<B>& rhs) { return lhs %= rhs; }
template<u16 B> [[nodiscard]] INLINE constexpr UInt<B> operator<<(UInt<B> lhs, u16 rhs) noexcept { return lhs <<= rhs; }
template<u16 B> [[nodiscard]] INLINE constexpr UInt<B> operator>>(UInt<B> lhs, u16 rhs) noexcept { return lhs >>= rhs; }
