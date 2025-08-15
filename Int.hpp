#pragma once

#include <array>
#include <cstddef>
#include <compare>
#include <algorithm>
#include <string_view>
#include <charconv>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <ostream>
#include <utility>
#include <vector>
#include <sstream>
#include <iomanip>
#include <immintrin.h>
#include <cstring>
#include <limits>
#include <bit>
#include <assert.h>

#if defined(__GNUC__) || defined(__clang__)
#define BUILTIN_EXPECT(x, y) (__builtin_expect(!!(x), y))
#define INLINE inline __attribute__((always_inline))
#elif defined(_MSC_VER)
#define BUILTIN_EXPECT(x, y) (x)
#define INLINE inline
#include <intrin.h>
#else
#define BUILTIN_EXPECT(x, y) (x)
#define INLINE inline
#endif

#if defined(__AVX2__)
#define HAS_AVX2 1
#else
#define HAS_AVX2 0
#endif

#if defined(__AVX512F__)
#define HAS_AVX512 1
#else
#define HAS_AVX512 0
#endif

using u64 = unsigned long long;
using i64 = long long;
using u128 = unsigned __int128;

// Helper para unrolling de loops
template <size_t... Is>
struct indices {};

template <size_t N, size_t... Is>
struct build_indices : build_indices<N-1, N-1, Is...> {};

template <size_t... Is>
struct build_indices<0, Is...> : indices<Is...> {};

template <size_t B = 256>
class alignas(64) Int {
    static_assert(B > 0 && B % 64 == 0, "Bits must be a positive multiple of 64");
public:
    static constexpr size_t NumBlocks = B / 64;
    static constexpr size_t Bits = B;

    using array_t = std::array<u64, NumBlocks>;

private:
    array_t bits{};

    // Helper para multiplicação 64x64->128
    static INLINE void mul_u64(u64 a, u64 b, u64* low, u64* high) {
        #if defined(__GNUC__) || defined(__clang__)
            u128 p = static_cast<u128>(a) * b;
            *low = static_cast<u64>(p);
            *high = static_cast<u64>(p >> 64);
        #elif defined(_MSC_VER) && defined(_M_X64)
            *low = _umul128(a, b, high);
        #else
            u64 a_low = a & 0xFFFFFFFF;
            u64 a_high = a >> 32;
            u64 b_low = b & 0xFFFFFFFF;
            u64 b_high = b >> 32;

            u64 p0 = a_low * b_low;
            u64 p1 = a_low * b_high;
            u64 p2 = a_high * b_low;
            u64 p3 = a_high * b_high;

            u64 carry = ((p0 >> 32) + (p1 & 0xFFFFFFFF) + (p2 & 0xFFFFFFFF)) >> 32;
            *low = p0 + (p1 << 32) + (p2 << 32);
            *high = p3 + (p1 >> 32) + (p2 >> 32) + carry;
        #endif
    }

    [[nodiscard]] static constexpr bool unsigned_less(const Int& a, const Int& b) noexcept {
        for (size_t i = NumBlocks; i-- > 0;) {
            if (a.bits[i] != b.bits[i])
                return a.bits[i] < b.bits[i];
        }
        return false;
    }

    // Unrolled addition
    template <size_t... I>
    INLINE constexpr void add_unrolled(const Int& o, indices<I...>) noexcept {
        unsigned char carry = 0;
        (..., (carry = _addcarry_u64(carry, bits[I], o.bits[I], &bits[I])));
    }

    // Unrolled subtraction
    template <size_t... I>
    INLINE constexpr void sub_unrolled(const Int& o, indices<I...>) noexcept {
        unsigned char borrow = 0;
        (..., (borrow = _subborrow_u64(borrow, bits[I], o.bits[I], &bits[I])));
    }

    // Multiplicação base (escolar)
    INLINE void multiply_base(const Int& a, const Int& b) noexcept {
        constexpr size_t temp_size = 2 * NumBlocks;
        std::array<u64, temp_size> temp = {0};

        for (size_t i = 0; i < NumBlocks; i++) {
            u64 carry = 0;
            for (size_t j = 0; j < NumBlocks; j++) {
                size_t idx = i + j;
                if (idx >= temp_size) continue;

                u64 hi, lo;
                mul_u64(a.bits[i], b.bits[j], &lo, &hi);

                u128 sum = static_cast<u128>(temp[idx]) + lo + carry;
                temp[idx] = static_cast<u64>(sum);
                carry = hi + (sum >> 64);
            }

            // Propagação de carry
            for (size_t idx = i + NumBlocks; idx < temp_size && carry; idx++) {
                u128 sum = static_cast<u128>(temp[idx]) + carry;
                temp[idx] = static_cast<u64>(sum);
                carry = sum >> 64;
            }
        }

        // Copiar resultado com truncamento
        for (size_t i = 0; i < NumBlocks; i++) {
            bits[i] = temp[i];
        }
    }

    // Multiplicação Karatsuba
    static Int karatsuba_mul_unsigned(const Int& a, const Int& b) {
        // Caso base para NumBlocks pequeno
        if constexpr (NumBlocks <= 1) {
            Int result;
            result.multiply_base(a, b);
            return result;
        }

        constexpr size_t half = NumBlocks / 2;
        constexpr size_t half_bits = half * 64;

        // Base case: if numbers are small enough, use base multiplication.
        bool a_small = std::all_of(a.bits.begin() + half, a.bits.end(), [](u64 val) { return val == 0; });
        bool b_small = std::all_of(b.bits.begin() + half, b.bits.end(), [](u64 val) { return val == 0; });

        if (a_small && b_small) {
            Int result;
            result.multiply_base(a, b);
            return result;
        }

        // Split operands into high and low halves
        Int al, ah, bl, bh;
        for (size_t i = 0; i < half; i++) {
            al.bits[i] = a.bits[i];
            ah.bits[i] = a.bits[i + half];
            bl.bits[i] = b.bits[i];
            bh.bits[i] = b.bits[i + half];
        }

        // Recursive steps
        Int p1 = karatsuba_mul_unsigned(ah, bh); // p1 = ah*bh
        Int p2 = karatsuba_mul_unsigned(al, bl); // p2 = al*bl

        // Calculate (ah+al) and (bh+bl) with carries
        Int ah_al = ah;
        unsigned char carry_a = 0;
        for (size_t i = 0; i < half; i++) {
            carry_a = _addcarry_u64(carry_a, ah_al.bits[i], al.bits[i], &ah_al.bits[i]);
        }

        Int bh_bl = bh;
        unsigned char carry_b = 0;
        for (size_t i = 0; i < half; i++) {
            carry_b = _addcarry_u64(carry_b, bh_bl.bits[i], bl.bits[i], &bh_bl.bits[i]);
        }

        Int p3 = karatsuba_mul_unsigned(ah_al, bh_bl);

        // Calculate the middle term: mid = (ah+al)(bh+bl) - p1 - p2
        Int mid_term = p3 - p1 - p2;
        if (carry_a) mid_term += bh_bl;
        if (carry_b) mid_term += ah_al;

        // Recombination for fixed-width result: result = p2 + (mid << half_bits)
        Int result = p2;
        result.add_shifted(mid_term, half_bits);

        return result;
    }

    // Public Karatsuba multiplication handler for signed numbers
    static Int karatsuba_mul(const Int& a, const Int& b) {
        // Handle trivial cases first
        if (a == Int(0) || b == Int(0)) return Int(0);
        if (a == Int(1)) return b;
        if (b == Int(1)) return a;
        if (a == Int(-1)) return -b;
        if (b == Int(-1)) return -a;

        const bool result_is_neg = a.isNegative() ^ b.isNegative();

        Int abs_a = a.abs();
        Int abs_b = b.abs();

        Int result = karatsuba_mul_unsigned(abs_a, abs_b);

        return result_is_neg ? -result : result;
    }

    template<size_t Start, size_t Count>
    [[nodiscard]] INLINE Int slice() const noexcept {
        static_assert(Start + Count <= NumBlocks, "Slice out of bounds");
        Int result;
        for (size_t i = 0; i < Count; ++i) {
            result.bits[i] = bits[Start + i];
        }
        return result;
    }

    INLINE void add_shifted(const Int& other, unsigned shift) noexcept {
        const size_t word_shift = shift / 64;
        const size_t bit_shift = shift % 64;

        if (word_shift >= NumBlocks) return;

        u128 carry = 0;
        for (size_t i = 0; i < NumBlocks - word_shift; ++i) {
            u128 val = static_cast<u128>(other.bits[i]);
            u128 shifted = (val << bit_shift) | carry;

            u128 sum = static_cast<u128>(bits[i + word_shift]) + (shifted & 0xFFFFFFFFFFFFFFFF);
            bits[i + word_shift] = static_cast<u64>(sum);
            carry = (sum >> 64) | (shifted >> 64);
        }
    }

    // Newton-Raphson para divisão
    INLINE Int newton_div(const Int& divisor) const {
        if (divisor == Int(0))
            throw std::domain_error("division by zero");

        // Handle special overflow case
        if (divisor == Int(-1)) {
            bool is_min_val = (bits[NumBlocks - 1] == 0x8000000000000000);
            for (size_t i = 0; i < NumBlocks - 1; ++i) {
                if (bits[i] != 0) {
                    is_min_val = false;
                    break;
                }
            }
            if (is_min_val) {
                return *this;
            }
        }

        const bool result_sign = isNegative() != divisor.isNegative();
        Int num = abs();
        Int den = divisor.abs();

        // Fast path for small divisors
        if (den.bits[NumBlocks-1] == 0) {
            for (size_t i = NumBlocks - 1; i > 0; i--) {
                if (den.bits[i] != 0) break;
                if (i == 1) {
                    u64 d = den.bits[0];
                    u128 rem = 0;
                    for (size_t i = NumBlocks; i-- > 0;) {
                        rem = (rem << 64) | num.bits[i];
                        num.bits[i] = static_cast<u64>(rem / d);
                        rem %= d;
                    }
                    return result_sign ? -num : num;
                }
            }
        }

        // Normalization
        const int lead_zeros = den.bits[NumBlocks-1] == 0 ? 64 : std::countl_zero(den.bits[NumBlocks-1]);
        den <<= lead_zeros;
        num <<= lead_zeros;

        // Initial approximation
        u128 d_top = (static_cast<u128>(den.bits[NumBlocks-1]) << 64);
        if (NumBlocks > 1) d_top |= den.bits[NumBlocks-2];

        if (d_top == 0) d_top = 1; // Prevent division by zero
        u128 approx = (~static_cast<u128>(0)) / d_top;

        Int approx_int = Int::from_u128(approx);
        constexpr int num_iters = (B >= 512) ? 6 : (B >= 256) ? 5 : 4;

        for (int i = 0; i < num_iters; ++i) {
            Int product = den * approx_int;
            Int two = Int(2) << (B - 1);
            Int diff = two - product;
            approx_int = approx_int * diff;
            approx_int >>= (B - 1);
        }

        // Final quotient
        Int quotient = num * approx_int;
        quotient >>= (B + 64 - lead_zeros);

        // Adjust for sign and return
        return result_sign ? -quotient : quotient;
    }

    // Helper para logical right shift
    INLINE constexpr void logical_right_shift(unsigned k) noexcept {
        const size_t word_shift = k / 64;
        const size_t bit_shift = k % 64;

        if (word_shift >= NumBlocks) {
            bits.fill(0);
            return;
        }

        for (size_t i = 0; i < NumBlocks - word_shift; ++i) {
            u64 low_part = bits[i + word_shift] >> bit_shift;
            u64 high_part = (i + word_shift + 1) < NumBlocks ?
                bits[i + word_shift + 1] << (64 - bit_shift) : 0;
            bits[i] = low_part | high_part;
        }

        // Fill upper words with zeros
        std::fill(bits.begin() + (NumBlocks - word_shift), bits.end(), 0);
    }

public:
    // --- Construtores ---
    constexpr Int() noexcept = default;
    constexpr Int(const Int&) noexcept = default;
    constexpr Int& operator=(const Int&) noexcept = default;

    constexpr Int(int v) noexcept : Int(static_cast<i64>(v)) {}
    constexpr Int(i64 v) noexcept {
        bits.fill((v < 0) ? ~0ull : 0ull);
        bits[0] = static_cast<u64>(v);
    }

    template <typename... Args>
        requires (std::is_unsigned_v<std::decay_t<Args>> && ...)
    constexpr Int(Args... args) noexcept {
        static_assert(sizeof...(args) <= NumBlocks, "Too many initializers for Int");
        bits.fill(0);
        size_t i = 0;
        ((bits[i++] = static_cast<u64>(args)), ...);
    }

    constexpr explicit Int(std::string_view sv);

    // --- Métodos de Conversão ---
    static constexpr Int from_u128(u128 value) noexcept {
        static_assert(NumBlocks >= 2, "from_u128 requires at least 128 bits");
        Int result;
        result.bits[0] = static_cast<u64>(value);
        result.bits[1] = static_cast<u64>(value >> 64);
        return result;
    }

    // --- Funções Públicas ---
    [[nodiscard]] INLINE constexpr Int abs() const noexcept {
        return isNegative() ? -*this : *this;
    }

    [[nodiscard]] INLINE constexpr bool isNegative() const noexcept {
        return (bits[NumBlocks - 1] >> 63) != 0;
    }

    [[nodiscard]] std::string to_string() const;

    [[nodiscard]] INLINE constexpr u64& operator[](size_t i) noexcept {
        assert(i < NumBlocks);
        return bits[i];
    }

    [[nodiscard]] INLINE constexpr const u64& operator[](size_t i) const noexcept {
        assert(i < NumBlocks);
        return bits[i];
    }

    // --- Operadores Aritméticos ---
    INLINE constexpr Int& operator+=(const Int& o) noexcept {
        add_unrolled(o, build_indices<NumBlocks>{});
        return *this;
    }

    INLINE constexpr Int& operator-=(const Int& o) noexcept {
        sub_unrolled(o, build_indices<NumBlocks>{});
        return *this;
    }

    INLINE constexpr Int& operator*=(const Int& o) noexcept {
        if constexpr (NumBlocks >= 4) {
            *this = karatsuba_mul(*this, o);
        } else {
            multiply_base(*this, o);
        }
        return *this;
    }

    INLINE Int& operator/=(const Int& divisor) {
        *this = newton_div(divisor);
        return *this;
    }

    INLINE Int& operator%=(const Int& divisor) {
        Int quotient = *this / divisor;
        *this -= quotient * divisor;
        return *this;
    }

    // --- Operadores Bitwise ---
    INLINE constexpr Int& operator|=(const Int& o) noexcept {
        #if HAS_AVX2 && defined(__AVX2__)
        if constexpr (NumBlocks % 4 == 0) {
            for (size_t i = 0; i < NumBlocks; i += 4) {
                __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&bits[i]));
                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&o.bits[i]));
                __m256i res = _mm256_or_si256(a, b);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&bits[i]), res);
            }
            return *this;
        }
        #endif

        for (size_t i = 0; i < NumBlocks; ++i)
            bits[i] |= o.bits[i];
        return *this;
    }

    INLINE constexpr Int& operator&=(const Int& o) noexcept {
        #if HAS_AVX2 && defined(__AVX2__)
        if constexpr (NumBlocks % 4 == 0) {
            for (size_t i = 0; i < NumBlocks; i += 4) {
                __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&bits[i]));
                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&o.bits[i]));
                __m256i res = _mm256_and_si256(a, b);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&bits[i]), res);
            }
            return *this;
        }
        #endif

        for (size_t i = 0; i < NumBlocks; ++i)
            bits[i] &= o.bits[i];
        return *this;
    }

    INLINE constexpr Int& operator^=(const Int& o) noexcept {
        #if HAS_AVX2 && defined(__AVX2__)
        if constexpr (NumBlocks % 4 == 0) {
            for (size_t i = 0; i < NumBlocks; i += 4) {
                __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&bits[i]));
                __m256i b = _mm256_load_si256(reinterpret_cast<const __m256i*>(&o.bits[i]));
                __m256i res = _mm256_xor_si256(a, b);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&bits[i]), res);
            }
            return *this;
        }
        #endif

        for (size_t i = 0; i < NumBlocks; ++i)
            bits[i] ^= o.bits[i];
        return *this;
    }

    // --- Shifts Otimizados ---
    [[nodiscard]] INLINE constexpr Int rotate_left(unsigned k) const noexcept {
        k %= Bits;
        if (k == 0) return *this;
        Int left = *this << k;
        Int right = *this;
        right.logical_right_shift(Bits - k);
        return left | right;
    }

    INLINE constexpr Int& operator<<=(unsigned k) noexcept {
        if (BUILTIN_EXPECT(k == 0, 1)) return *this;
        if (BUILTIN_EXPECT(k >= Bits, 0)) { bits.fill(0); return *this; }

        #if HAS_AVX2 && defined(__AVX2__)
        if constexpr (NumBlocks % 4 == 0) {
            // Implementação AVX2 para grandes blocos
            const size_t word_shift = k / 64;
            const size_t bit_shift = k % 64;

            if (word_shift > 0) {
                for (ssize_t i = NumBlocks - 1; i >= static_cast<ssize_t>(word_shift); --i) {
                    bits[i] = bits[i - word_shift];
                }
                std::fill(bits.begin(), bits.begin() + word_shift, 0);
            }

            if (bit_shift > 0) {
                u64 carry = 0;
                for (size_t i = word_shift; i < NumBlocks; ++i) {
                    u64 val = bits[i];
                    bits[i] = (val << bit_shift) | carry;
                    carry = safe_shift_carry(val, bit_shift);
                }
            }
            return *this;
        }
        #endif

        // Fallback escalar
        const size_t word_shift = k / 64;
        const size_t bit_shift = k % 64;

        if (word_shift > 0) {
            for (ssize_t i = NumBlocks - 1; i >= static_cast<ssize_t>(word_shift); --i) {
                bits[i] = bits[i - word_shift];
            }
            std::fill(bits.begin(), bits.begin() + word_shift, 0);
        }

        if (bit_shift > 0) {
            u64 carry = 0;
            for (size_t i = word_shift; i < NumBlocks; ++i) {
                u64 val = bits[i];
                bits[i] = (val << bit_shift) | carry;
                carry = val >> (64 - bit_shift);
            }
        }

        return *this;
    }
    INLINE u64 safe_shift_carry(u64 val, unsigned bit_shift) noexcept {
        return (bit_shift > 0) ? (val >> (64 - bit_shift)) : 0;
    }

    INLINE constexpr Int& operator>>=(unsigned k) noexcept {
        if (BUILTIN_EXPECT(k == 0, 1)) return *this;
        if (BUILTIN_EXPECT(k >= Bits, 0)) {
            bits.fill(isNegative() ? ~0ull : 0ull);
            return *this;
        }

        #if HAS_AVX2 && defined(__AVX2__)
        if constexpr (NumBlocks % 4 == 0) {
            const size_t word_shift = k / 64;
            const size_t bit_shift = k % 64;
            const u64 sign_fill = isNegative() ? ~0ull : 0ull;

            if (bit_shift == 0) {
                for (size_t i = 0; i < NumBlocks - word_shift; ++i) {
                    bits[i] = bits[i + word_shift];
                }
            } else {
                for (size_t i = 0; i < NumBlocks - word_shift; ++i) {
                    u64 low_part = bits[i + word_shift] >> bit_shift;
                    u64 high_part = ((i + word_shift + 1) < NumBlocks)
                                ? safe_shift_carry(bits[i + word_shift + 1], 64 - bit_shift)
                                : (sign_fill << (64 - bit_shift));
                    bits[i] = low_part | high_part;
                }
            }

            if (word_shift > 0) {
                std::fill(bits.begin() + (NumBlocks - word_shift), bits.end(), sign_fill);
            }

            return *this;
        }
        #endif

        // Fallback escalar
        const size_t word_shift = k / 64;
        const size_t bit_shift = k % 64;
        const u64 sign_fill = isNegative() ? ~0ull : 0ull;

        if (bit_shift == 0) {
            for (size_t i = 0; i < NumBlocks - word_shift; ++i) {
                bits[i] = bits[i + word_shift];
            }
        } else {
            for (size_t i = 0; i < NumBlocks - word_shift; ++i) {
                u64 low_part = bits[i + word_shift] >> bit_shift;
                u64 high_part = ((i + word_shift + 1) < NumBlocks)
                            ? (bits[i + word_shift + 1] << (64 - bit_shift))
                            : (sign_fill << (64 - bit_shift));
                bits[i] = low_part | high_part;
            }
        }

        if (word_shift > 0) {
            std::fill(bits.begin() + (NumBlocks - word_shift), bits.end(), sign_fill);
        }

        return *this;
    }

    // --- Incremento/Decremento ---
    INLINE constexpr Int& operator++() noexcept { return *this += Int(1); }
    INLINE constexpr Int& operator--() noexcept { return *this -= Int(1); }
    INLINE constexpr Int operator++(int) noexcept { Int t = *this; ++*this; return t; }
    INLINE constexpr Int operator--(int) noexcept { Int t = *this; --*this; return t; }

    // --- Negação e Complemento ---
    [[nodiscard]] INLINE constexpr Int operator-() const noexcept {
        return ~(*this) + Int(1);
    }

    [[nodiscard]] INLINE constexpr Int operator~() const noexcept {
        Int r = *this;
        #if HAS_AVX2 && defined(__AVX2__)
        if constexpr (NumBlocks % 4 == 0) {
            for (size_t i = 0; i < NumBlocks; i += 4) {
                __m256i a = _mm256_load_si256(reinterpret_cast<const __m256i*>(&bits[i]));
                __m256i ones = _mm256_set1_epi64x(-1);
                __m256i res = _mm256_xor_si256(a, ones);
                _mm256_store_si256(reinterpret_cast<__m256i*>(&r.bits[i]), res);
            }
            return r;
        }
        #endif

        for (auto& b : r.bits) b = ~b;
        return r;
    }

    // --- Comparações ---
    [[nodiscard]] friend constexpr bool operator==(const Int& a, const Int& b) noexcept {
        return std::equal(a.bits.begin(), a.bits.end(), b.bits.begin());
    }

    [[nodiscard]] friend constexpr std::strong_ordering operator<=>(const Int& a, const Int& b) noexcept {
        const bool sa = a.isNegative(), sb = b.isNegative();
        if (sa != sb) {
            return sa ? std::strong_ordering::less : std::strong_ordering::greater;
        }

        if (sa) { // Ambos negativos: maior magnitude é menor
            return unsigned_less(b, a) ? std::strong_ordering::less :
                   unsigned_less(a, b) ? std::strong_ordering::greater :
                   std::strong_ordering::equal;
        }

        return unsigned_less(a, b) ? std::strong_ordering::less :
               unsigned_less(b, a) ? std::strong_ordering::greater :
               std::strong_ordering::equal;
    }
};

// Implementações de Métodos
template<size_t B>
constexpr Int<B>::Int(std::string_view sv) {
    bits.fill(0);
    if (sv.empty()) return;

    bool is_neg = (sv.front() == '-');
    if (is_neg || sv.front() == '+') {
        sv.remove_prefix(1);
    }
    if (sv.empty()) return;

    const bool is_hex = sv.starts_with("0x") || sv.starts_with("0X");
    if (is_hex) {
        sv.remove_prefix(2);
        if (sv.empty()) return;

        size_t start_idx = 0;
        size_t end_idx = sv.size();
        size_t bit_pos = 0;
        size_t block_idx = 0;

        // Processar de trás para frente (LSB primeiro)
        while (end_idx > start_idx && block_idx < NumBlocks) {
            size_t digits_to_process = std::min<size_t>(16, end_idx - start_idx);
            size_t chunk_start = end_idx - digits_to_process;
            std::string_view chunk = sv.substr(chunk_start, digits_to_process);

            u64 block_val = 0;
            auto res = std::from_chars(chunk.data(), chunk.data() + chunk.size(), block_val, 16);
            if (res.ec != std::errc()) {
                throw std::invalid_argument("Invalid hex character");
            }

            bits[block_idx] |= block_val << bit_pos;
            if (bit_pos > 0 && (block_idx + 1) < NumBlocks) {
                bits[block_idx + 1] = block_val >> (64 - bit_pos);
            }

            bit_pos = (bit_pos + digits_to_process * 4) % 64;
            if (bit_pos == 0) {
                block_idx++;
            }

            end_idx = chunk_start;
        }
    } else {
        constexpr size_t max_digits = ((B - 1) * 301LLU / 1000) + 2;
        if (sv.size() > max_digits) {
            throw std::out_of_range("Decimal string too long");
        }
        constexpr u64 CHUNK_POW = 1000000000000000000ull; // 10^18
        constexpr int CHUNK_SIZE = 18;

        const char* p = sv.data();
        const char* const end = p + sv.size();

        size_t first_chunk_len = sv.size() % CHUNK_SIZE;
        if (first_chunk_len == 0 && !sv.empty()) {
            first_chunk_len = CHUNK_SIZE;
        }

        u64 chunk_val = 0;
        const char* const end_first_chunk = p + first_chunk_len;
        auto res = std::from_chars(p, end_first_chunk, chunk_val);
        if (res.ec != std::errc()) {
            throw std::invalid_argument("Invalid decimal character");
        }
        *this = chunk_val;
        p = end_first_chunk;

        while (p < end) {
            const char* const end_chunk = p + CHUNK_SIZE;
            res = std::from_chars(p, end_chunk, chunk_val);
            if (res.ec != std::errc()) {
                throw std::invalid_argument("Invalid decimal character");
            }
            *this *= CHUNK_POW;
            *this += chunk_val;
            p = end_chunk;
        }
    }

    if (is_neg) {
        *this = -*this;
    }

    // Overflow check para decimal
    if (!is_hex) {
        const bool final_sign = isNegative();
        if (is_neg && !final_sign && *this != Int<B>(0)) {
            throw std::out_of_range("Negative value too large for signed Int");
        }
        if (!is_neg && final_sign) {
            throw std::out_of_range("Positive value too large for signed Int");
        }
    }
}

template<size_t B>
std::string Int<B>::to_string() const {
    if (*this == Int<B>(0)) return "0";

    Int temp = abs();
    constexpr u64 CHUNK_POW = 1000000000000000000ull; // 10^18
    constexpr int CHUNK_SIZE = 18;

    std::vector<u64> chunks;
    while (temp != Int<B>(0)) {
        Int quotient;
        u64 remainder = 0;

        for (size_t i = NumBlocks; i-- > 0;) {
            u128 dividend = (static_cast<u128>(remainder) << 64) | temp.bits[i];
            quotient.bits[i] = static_cast<u64>(dividend / CHUNK_POW);
            remainder = static_cast<u64>(dividend % CHUNK_POW);
        }

        chunks.push_back(remainder);
        temp = quotient;
    }

    std::ostringstream oss;
    if (isNegative()) oss << '-';

    oss << chunks.back();
    for (auto it = chunks.rbegin() + 1; it != chunks.rend(); ++it) {
        oss << std::setw(CHUNK_SIZE) << std::setfill('0') << *it;
    }

    return oss.str();
}

// Operadores Não-Membros
template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator+(Int<B> a, const Int<B>& b) noexcept { return a += b; }
template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator-(Int<B> a, const Int<B>& b) noexcept { return a -= b; }
template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator*(Int<B> a, const Int<B>& b) noexcept { return a *= b; }
template<size_t B> [[nodiscard]] INLINE Int<B> operator/(Int<B> a, const Int<B>& b) { return a /= b; }
template<size_t B> [[nodiscard]] INLINE Int<B> operator%(Int<B> a, const Int<B>& b) { return a %= b; }
template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator|(Int<B> a, const Int<B>& b) noexcept { return a |= b; }
template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator&(Int<B> a, const Int<B>& b) noexcept { return a &= b; }
template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator^(Int<B> a, const Int<B>& b) noexcept { return a ^= b; }
template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator<<(Int<B> a, unsigned k) noexcept { return a <<= k; }
template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator>>(Int<B> a, unsigned k) noexcept { return a >>= k; }

// Operadores com Tipos Nativos
#define DEFINE_INT_NATIVE_OPS(T) \
    template<size_t B> [[nodiscard]] INLINE constexpr bool operator==(const Int<B>& a, T b) noexcept { return a == Int<B>(b); } \
    template<size_t B> [[nodiscard]] INLINE constexpr std::strong_ordering operator<=>(const Int<B>& a, T b) noexcept { return a <=> Int<B>(b); } \
    template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator+(const Int<B>& a, T b) noexcept { return a + Int<B>(b); } \
    template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator-(const Int<B>& a, T b) noexcept { return a - Int<B>(b); } \
    template<size_t B> [[nodiscard]] INLINE constexpr Int<B> operator*(const Int<B>& a, T b) noexcept { return a * Int<B>(b); } \
    template<size_t B> [[nodiscard]] INLINE Int<B> operator/(const Int<B>& a, T b) { return a / Int<B>(b); } \
    template<size_t B> [[nodiscard]] INLINE Int<B> operator%(const Int<B>& a, T b) { return a % Int<B>(b); }

DEFINE_INT_NATIVE_OPS(i64)
DEFINE_INT_NATIVE_OPS(u64)

// I/O e Literais
template<size_t B>
std::ostream& operator<<(std::ostream& os, const Int<B>& val) {
    return os << val.to_string();
}

#define DEFINE_INT_LITERAL(bits) \
    [[nodiscard]] consteval Int<bits> operator""_i##bits(const char* s, size_t) { return Int<bits>(s); }

inline namespace int_literals {
    DEFINE_INT_LITERAL(128)
    DEFINE_INT_LITERAL(256)
    DEFINE_INT_LITERAL(512)
    DEFINE_INT_LITERAL(1024)
    DEFINE_INT_LITERAL(2048)
}

#undef DEFINE_INT_LITERAL

// Aliases
using Int128 = Int<128>;
using Int256 = Int<256>;
using Int512 = Int<512>;
using Int1024 = Int<1024>;
using Int2048 = Int<2048>;
