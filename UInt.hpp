#pragma once

#include <algorithm>
#include <array>
#include <immintrin.h>
#include <stdexcept>
#include <string>
#include <utility>
#include <sstream>
#include <iomanip>
#include <ios>
#include <charconv>

// Core modules
#include "Division.hpp"
#include "Multiplication.hpp"

using namespace std;

namespace Arbitrary {
    // Type Aliases
    using u8 = unsigned char;
    using u16 = unsigned short;
    using u64 = unsigned long long;
    using u128 = unsigned __int128;

    #define BUILTIN_EXPECT(x, y) (__builtin_expect(!!(x), y))
    #define FORCE_INLINE inline __attribute__((always_inline))

    template <u8 N> class alignas(64) UInt {
        static_assert(N > 0, "Limbs must be positive");

    public:
        static constexpr u8 TotalBlocks = N;

        // using Storage = conditional_t<N == 1, u64, conditional_t<N == 2, u128, array<u64, N>>>;
        // Storage bits {}; // inicialize all elements to zero
        array<u64, N> bits;

    private:
        static FORCE_INLINE constexpr uint64_t rand(u64& state) noexcept {
            state += 0x9e3779b97f4a7c15ULL;
            uint64_t result = state;
            result = (result ^ (result >> 30)) * 0xbf58476d1ce4e5b9ULL;
            result = (result ^ (result >> 27)) * 0x94d049bb133111ebULL;
            return result ^ (result >> 31);
        }

        FORCE_INLINE constexpr u8 charToValue(char c) const {
            constexpr auto lut = []() { // lookup table
                array<u8, 256> table;
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
                throw invalid_argument("Invalid character");
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

        constexpr UInt(const UInt &o) noexcept = default;

        constexpr UInt(UInt &&o) noexcept = default;

        constexpr explicit UInt(string_view sv) {
            if (sv.empty()) return;
            if (sv.front() == '-') throw invalid_argument("Negative values not supported in UInt");
            if (sv.front() == '+') sv.remove_prefix(1);
            if (sv.empty()) return;

            const bool is_hex = sv.starts_with("0x") || sv.starts_with("0X");
            if (is_hex) {
                sv.remove_prefix(2);
                if (sv.empty()) return;
                if (sv.size() > (N * 16)) throw out_of_range("Hex string too long");
                u16 size = sv.size();
                for (u8 i = 0; i < N && size > 0; ++i) {
                    u64 value = 0;
                    int chars_to_read = min(16, (int)size);
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
                auto res = from_chars(p, end_first_chunk, chunk_val);
                if (BUILTIN_EXPECT(res.ec != errc(), 0))
                    throw invalid_argument("Invalid decimal character");
                *this = chunk_val;
                p = end_first_chunk;
                while (p < end) {
                    const char *const end_chunk = p + CHUNK_SIZE;
                    res = from_chars(p, end_chunk, chunk_val);
                    if (BUILTIN_EXPECT(res.ec != errc(), 0))
                        throw invalid_argument("Invalid decimal character");
                    *this *= CHUNK_POW;
                    *this += chunk_val;
                    p = end_chunk;
                }
            }
        }

        template <typename... Args> requires (sizeof...(Args) <= N)
        constexpr UInt(Args... args) noexcept {
            if constexpr (N == 1){
                bits = (static_cast<u64>(args), ...);
            }else if constexpr(N == 2){
                u64 parts[2] = { static_cast<u64>(args)... };
                bits = (static_cast<u128>(parts[1]) << 64) | parts[0];
            }else{
                bits = {static_cast<u64>(args)...};
            }
        }


        constexpr UInt &operator=(const UInt &o) noexcept = default;
        constexpr UInt &operator=(UInt &&o) noexcept = default;
        constexpr FORCE_INLINE UInt &operator=(u64 o) noexcept {
            if constexpr (N == 1) {
                bits = o;
            } else if constexpr (N == 2) {
                bits = (u128)o;
            } else {
                bits.fill(0);
                bits[0] = o;
            }
            return *this;
        }

        // comparators
        [[nodiscard]] FORCE_INLINE constexpr bool operator==(const UInt &o) const noexcept {
            return bits == o.bits;
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator!=(const UInt &o) const noexcept {
            return bits != o.bits;
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator<(const UInt &other) const noexcept {
            for (int i = N - 1; i >= 0; --i) {
                if (bits[i] < other.bits[i]) return true;
                if (bits[i] > other.bits[i]) return false;
            }
            return false; // Equal
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator>(const UInt &other) const noexcept {
            return other < *this;
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator<=(const UInt &other) const noexcept {
            return !(*this > other);
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator>=(const UInt &other) const noexcept {
            return !(*this < other);
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator==(const u64 o) const noexcept {
            if(bits[0] != o) return false;
            else if constexpr (N > 1){
                for (u8 i = 1; i < N; ++i){
                    if (bits[i] != 0) return false;
                }
            }
            return true;
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator!=(const u64 o) const noexcept {
                return !(*this == o);
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator<(const u64 o) const noexcept {
            if (bits[0] >= o) return false;
            else if constexpr(N > 1){
                for (u8 i = 1; i < N; ++i) {
                    if (bits[i] != 0) return false;
                }
            }
            return true;
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator>(const u64 o) const noexcept{
            if (bits[0] <= o) return false;
            else if constexpr (N > 1) {
                for (u8 i = 1; i < N; ++i) {
                    if (bits[i] != 0) return false;
                }
            }
            return true;
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator<=(const u64 o) const noexcept{
            return *this < o || *this == o;
        }
        [[nodiscard]] FORCE_INLINE constexpr bool operator>=(const u64 o) const noexcept{
            return *this > o || *this == o;
        }
        FORCE_INLINE constexpr UInt &operator++() noexcept { // melhor forma. testado!
            u8 i = 0;
            do {if (++bits[i] == 0) [[unlikely]] ++i;
                else return *this;
            }while (i < N);
            return *this;
        }
        FORCE_INLINE constexpr UInt &operator--() noexcept { // melhor forma. testado!
            u8 i = 0;
            do {if (--bits[i] == -1ull) [[unlikely]] ++i;
                else return *this;
            }while (i < N);
            return *this;
        }
        FORCE_INLINE constexpr UInt operator++(int) noexcept {// melhor forma. testado!
            UInt t = *this;
            ++*this;
            return t;
        }
        FORCE_INLINE constexpr UInt operator--(int) noexcept {// melhor forma. testado!
            UInt t = *this;
            --*this;
            return t;
        }

        constexpr FORCE_INLINE UInt &operator+=(const UInt &other) noexcept { // melhor forma. testado!
            if consteval {
                u64 carry = 0;
                for (u8 i = 0; i < N; ++i) {
                    u128 sum = static_cast<u128>(bits[i]) + other.bits[i] + carry;
                    bits[i] = static_cast<u64>(sum);
                    carry = sum >> 64;
                }
            } else {
                asm volatile(R"(
                    .set offset, 0
                    movq offset(%[src]), %%rax
                    addq %%rax, offset(%[dst])
                    .set offset, offset+8
                    .rept %c[count]
                        movq offset(%[src]), %%rax
                        adcq %%rax, offset(%[dst])
                        .set offset, offset+8
                    .endr
                )"
                    : "+m"(bits)
                    : [dst]   "r"(bits.data()),
                      [src]   "r"(other.bits.data()),
                      [count] "n"(N - 1),
                      "m"(other.bits)
                    : "rax", "cc"
                );
            }
            return *this;
        }

        constexpr FORCE_INLINE UInt &operator+=(u64 val) noexcept { // melhor forma. testado!
            if ((bits[0] += val) < val)[[unlikely]]{
                for (u8 i = 1; i < N; ++i) {
                    if (++bits[i] != 0) [[likely]] return *this;
                }
            }
            return *this;
        }

        constexpr FORCE_INLINE UInt &operator-=(const UInt &other) noexcept { // melhor forma. testado!
            if consteval {
                u64 carry = 0;
                for (u8 i = 0; i < N; ++i) {
                    u128 sub = static_cast<u128>(bits[i]) - other.bits[i] - carry;
                    bits[i] = static_cast<u64>(sub);
                    carry = sub >> 64;
                }
            } else {
                asm volatile(R"(
                    .set offset, 0
                    movq offset(%[src]), %%rax
                    subq %%rax, offset(%[dst])
                    .set offset, offset+8
                    .rept %c[count]
                        movq offset(%[src]), %%rax
                        sbbq %%rax, offset(%[dst])
                        .set offset, offset+8
                    .endr
                )"
                    : "+m"(bits)
                    : [dst]   "r"(bits.data()),
                      [src]   "r"(other.bits.data()),
                      [count] "n"(N - 1),
                      "m"(other.bits)
                    : "rax", "cc"
                );
            }
            return *this;
        }

        constexpr FORCE_INLINE UInt &operator-=(const u64 val) noexcept { //melhor forma. testado!
            if ((bits[0] -= val) > val)[[unlikely]] {
                for (u8 i = 1; i < N; ++i) {
                    if (--bits[i] != -1ull) [[likely]] return *this;
                }
            }
            return *this;
        }

        constexpr FORCE_INLINE UInt &operator*=(const UInt &other) noexcept {
            if consteval {
                UInt<N> self_copy = *this;
                this->bits.fill(0);
                for (u8 i = 0; i < N; ++i) {
                    u64 y = self_copy.bits[i];
                    if (y == 0) continue;
                    u128 carry = 0;
                    for (u8 j = 0; j < N - i; ++j) {
                        u128 temp = (u128)other.bits[j] * y + this->bits[i + j] + carry;
                        this->bits[i + j] = (u64)temp;
                        carry = temp >> 64;
                    }
                }
                return *this;
            }
            if constexpr (N == 1) {
                bits[0] *= other.bits[0];
                return *this;
            }
            else if constexpr (N <= 8) {
                u64 p_res[N];
                if constexpr (N == 1) {
                    bits[0] *= other.bits[0];
                } else if constexpr (N == 2) {
                    u64 r1 = bits[1] * other.bits[0];
                    u64 r2 = bits[0] * other.bits[1];
                    u128 r3 = static_cast<u128>(bits[0]) * other.bits[0];
                    bits[0] = r3;
                    bits[1] = (r1 + r2 + static_cast<u64>(r3>>64));
                } else if constexpr (N == 3) {
                    Multiplication::mul_schoolbook_truncated_fixed_3x3(&bits[0], &bits[0], &other.bits[0]);
                } else if constexpr (N == 8) {
                     Multiplication::mul_split_truncated_fixed_8(p_res, &bits[0], &other.bits[0]);
                     copy_n(p_res, N, &bits[0]);
                } else if constexpr (N >= 5 && N <= 6) {
                     Multiplication::mul_comba_truncated_fixed<N>(p_res, &bits[0], &other.bits[0]);
                     copy_n(p_res, N, &bits[0]);
                } else {
                     Multiplication::mul_schoolbook_truncated_fixed<N>(p_res, &bits[0], &other.bits[0]);
                     copy_n(p_res, N, &bits[0]);
                }

            } else if constexpr (N <= 16) {
                if (this == &other) {
                     Multiplication::square_schoolbook_truncated_fixed<N>(&bits[0], &bits[0]);
                } else {
                     Multiplication::mul_schoolbook_truncated_fixed<N>(&bits[0], &bits[0],
                                             &other.bits[0]);
                }
            } else {
                alignas(64) u64 tmp[8 * N + 1000];
                if (this == &other) {
                    Multiplication::square_truncated_fixed<N>(&bits[0], &bits[0], tmp);
                } else {
                    Multiplication::mul_truncated_fixed<N>(&bits[0], &bits[0],
                                            &other.bits[0], tmp);
                }
            }
            return *this;
        }

        constexpr FORCE_INLINE UInt &operator*=(u64 val) noexcept {
            u64 c = 0;
            for (u8 i = 0; i < N; ++i) {
                u128 temp = (u128)bits[i] * val + c;
                bits[i] = (u64)temp;
                c = temp >> 64;
            }
            return *this;
        }

        FORCE_INLINE UInt &operator/=(u64 val) {
            if (val == 0) throw domain_error("Division by zero");
            u128 rem = 0;
            for (int i = N - 1; i >= 0; --i) {
                u128 cur = bits[i] | (rem << 64);
                bits[i] = (u64)(cur / val);
                rem = cur % val;
            }
            return *this;
        }

        FORCE_INLINE UInt &operator%=(u64 val) {
            if (val == 0) throw domain_error("Division by zero");
            u128 rem = 0;
            for (int i = N - 1; i >= 0; --i) {
                u128 cur = bits[i] | (rem << 64);
                rem = cur % val;
            }
            fill(bits.begin(), bits.end(), 0);
            bits[0] = (u64)rem;
            return *this;
        }

        [[nodiscard]] FORCE_INLINE constexpr static UInt<N> random(const u64 seed = 0) noexcept;
        [[nodiscard]] static constexpr pair<UInt<N>, UInt<N>> divmod(UInt<N> u, UInt<N> v);

        constexpr FORCE_INLINE UInt<N> &operator/=(const UInt<N> &other) {
            *this = divmod(*this, other).first;
            return *this;
        }
        constexpr FORCE_INLINE UInt<N> &operator%=(const UInt<N> &other) {
            if constexpr(N == 1){
                bits %= other.bits;
            }else {
                *this = divmod(*this, other).second;
            }
            return *this;
        }
        [[nodiscard]] constexpr FORCE_INLINE UInt<N> operator%(const UInt<N> &other) const {
            return divmod(*this, other).second;
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

        [[nodiscard]] string to_string() const;
        [[nodiscard]] string to_hex_string() const;
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

    template <u8 N>
    FORCE_INLINE constexpr UInt<N> &UInt<N>::operator<<=(const u16 n) noexcept {
        if (n == 0)[[likely]] return *this;
        else if (n >= N * 64){
            bits.fill(0);
            return *this;
        }
        const u16 block_shift = n >> 6;
        const u16 bit_shift = n & 63;
        if (block_shift > 0) {
            for (int i = N - 1; i >= (int)block_shift; --i)
                bits[i] = bits[i - block_shift];
            fill(bits.begin(), bits.begin() + block_shift, 0ull);
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
            fill(bits.begin() + N - block_shift, bits.end(), 0ull);
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

    template <u8 N>
    constexpr pair<UInt<N>, UInt<N>> UInt<N>::divmod(UInt<N> u, UInt<N> v) {
        if (v.is_zero())
            throw domain_error("Division by zero");

        if consteval {
            if (u < v) return {UInt<N>(0), u};
            UInt<N> q(0);
            UInt<N> r(0);
            for (int i = N * 64 - 1; i >= 0; --i) {
                r <<= 1;
                if ((u.bits[i / 64] >> (i % 64)) & 1) {
                    r.bits[0] |= 1;
                }
                if (r >= v) {
                    r -= v;
                    q.bits[i / 64] |= (1ULL << (i % 64));
                }
            }
            return {q, r};
        }

        if (u < v)
            return {UInt<N>(0), u};

        u8 n_v = v.num_limbs();
        u8 n_u = u.num_limbs(); // We know u >= v, so n_u >= n_v

        // Optimization for small quotients (common in random distribution)
        if constexpr (N == 2 || N == 3 || N == 5 || N == 8) {
            if (n_u == n_v) {
                // q is likely small. Try finding it by subtraction.
                UInt<N> q(0);
                u -= v;
                q.bits[0] = 1;

                // Check if done
                if (u < v) return {q, u};

                // Try one more time (q=2)
                u -= v;
                q.bits[0]++;
                if (u < v) return {q, u};

                for (int k = 0; k < 10; ++k) {
                     u -= v;
                     q.bits[0]++;
                     if (u < v) return {q, u};
                     if (u.num_limbs() < n_v) return {q, u}; // Optimization
                }
            }
        }

        u8 n = n_v;
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

        UInt<N> q{}, r{};

        if constexpr (N <= 16) {
             Division::DivisionFixed<N>::div(q.bits.data(), r.bits.data(), u.bits.data(), v.bits.data());
        } else if constexpr (N <= 64) {
            Division::div_knuth_impl<N>(q.bits.data(), r.bits.data(), u.bits.data(),
                                    u.num_limbs(), v.bits.data(), v.num_limbs());
        } else {
            int u_len = u.num_limbs();
            int v_len = v.num_limbs();
            if (u_len - v_len < 32) {
                Division::div_knuth_impl<N>(q.bits.data(), r.bits.data(), u.bits.data(),
                                        u_len, v.bits.data(), v_len);
            } else {
                int n_limbs = v.num_limbs();
                int shift_limbs = N - n_limbs;
                int shift_bits = __builtin_clzll(v.bits[n_limbs - 1]);

                u64 u_norm[2 * N];
                u64 v_norm[N];
                u64 tmp[20 * N + 1000];

                if (shift_limbs > 0 || shift_bits > 0) {
                    for(int i = N - 1; i >= shift_limbs; --i)
                        v_norm[i] = v.bits[i - shift_limbs];
                    fill_n(v_norm, shift_limbs, 0);

                    if (shift_bits > 0) {
                        u64 carry = 0;
                        for(int i = 0; i < N; ++i) {
                            u64 val = v_norm[i];
                            v_norm[i] = (val << shift_bits) | carry;
                            carry = val >> (64 - shift_bits);
                        }
                    }
                } else {
                    copy_n(v.bits.data(), N, v_norm);
                }

                fill_n(u_norm, 2 * N, 0);

                for(int i = 0; i < N; ++i) {
                    u_norm[i + shift_limbs] = u.bits[i];
                }

                if (shift_bits > 0) {
                    u64 carry = 0;
                    for(int i = 0; i < 2 * N; ++i) {
                        u64 val = u_norm[i];
                        u_norm[i] = (val << shift_bits) | carry;
                        carry = val >> (64 - shift_bits);
                    }
                }

                Division::div_recursive(q.bits.data(), r.bits.data(), u_norm,
                                        v_norm, N, tmp);

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
                    fill_n(r.bits.data() + N - shift_limbs, shift_limbs, 0);
                }
            }
        }

        return {q, r};
    }

    template<u8 N>
    constexpr UInt<N> UInt<N>::random(const u64 seed) noexcept {
        UInt<N> number;
        u64 state = seed;

        if (state == 0) {
            if consteval {
                state = 0xCAFEBABE + N;
            } else {
                state = reinterpret_cast<uintptr_t>(&number) ^ 0x9e3779b97f4a7c15ULL;
            }
        }

        for (u64 &block : number.bits) {
            block = rand(state);
        }
        return number;
    }

    template <u8 N> string UInt<N>::to_string() const {
        if (is_zero())
            return "0";
        UInt<N> temp = *this;
        constexpr u64 CHUNK_POW = 1000000000000000000ull;
        constexpr int CHUNK_SIZE = 18;

        u64 chunks[300];
        int chunk_count = 0;

        while (!temp.is_zero()) {
            auto [quotient, remainder_uint] = divmod(temp, UInt<N>(CHUNK_POW));
            chunks[chunk_count++] = remainder_uint.bits[0];
            temp = move(quotient);
        }
        ostringstream oss;
        oss << chunks[chunk_count - 1];
        for (int i = chunk_count - 2; i >= 0; --i) {
            oss << setw(CHUNK_SIZE) << setfill('0') << chunks[i];
        }
        return oss.str();
    }

    template <u8 N> string UInt<N>::to_hex_string() const {
        if (is_zero())
            return "0x0";
        int msb_idx = N - 1;
        while (msb_idx > 0 && bits[msb_idx] == 0)
            --msb_idx;
        ostringstream oss;
        oss << "0x" << hex << nouppercase;
        oss << bits[msb_idx];
        for (int i = msb_idx - 1; i >= 0; --i)
            oss << setw(16) << setfill('0') << bits[i];
        return oss.str();
    }

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
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator+(UInt<N> lhs, u64 rhs) noexcept {
        return lhs += rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator+(u64 lhs, UInt<N> rhs) noexcept {
        return rhs += lhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator-(UInt<N> lhs, u64 rhs) noexcept {
        return lhs -= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator-(u64 lhs, const UInt<N>& rhs) noexcept {
        return UInt<N>(lhs) -= rhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator*(UInt<N> lhs, u64 rhs) noexcept {
        return lhs *= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator*(u64 lhs, UInt<N> rhs) noexcept {
        return rhs *= lhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator/(UInt<N> lhs, u64 rhs) {
        return lhs /= rhs;
    }

    template <u8 N>
    [[nodiscard]] FORCE_INLINE UInt<N> operator%(UInt<N> lhs, u64 rhs) {
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

    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator&(UInt<N> lhs, const UInt<N> &rhs) noexcept {
        return lhs &= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator|(UInt<N> lhs, const UInt<N> &rhs) noexcept {
        return lhs |= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator^(UInt<N> lhs, const UInt<N> &rhs) noexcept {
        return lhs ^= rhs;
    }
    template <u8 N>
    [[nodiscard]] FORCE_INLINE constexpr UInt<N> operator~(UInt<N> val) noexcept {
        for (auto &limb : val.bits) limb = ~limb;
        return val;
    }
}
