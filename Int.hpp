#pragma once

#include <array>
#include <cstddef>
#include <compare>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <linux/types.h>
#include <string_view>
#include <charconv>
#include <stdexcept>
#include <sys/types.h>
#include <tuple>
#include <type_traits>
#include <string>
#include <ostream>
#include <utility>
#include <vector>
#include <sstream>
#include <iomanip>
#include <immintrin.h>
#include <cstring>
#include <bit>
#include <assert.h>
#include <x86intrin.h>

// compilação com g++ C++ 26
#define BUILTIN_EXPECT(x, y) (__builtin_expect(!!(x), y))
#define INLINE inline __attribute__((always_inline))
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

using u8 = unsigned char;
using u16 = unsigned short;
using u64 = unsigned long long;
using i64 = signed long long;
using u128 = unsigned __int128;

// Helper para unrolling de loops
template <u8... Is>
struct indices {};

template <u8 N, u8... Is>
struct build_indices : build_indices<N-1, N-1, Is...> {};

template <u8... Is>
struct build_indices<0, Is...> : indices<Is...> {};

template <u8 B = 64>
class alignas(64) Int {
    static_assert(B % 64 == 0, "Bits must be a positive multiple of 64");
    public:
        static constexpr u8 NumBlocks = B / 64;
        static constexpr u16 Bits = B;

    private:
        using array_t = std::array<u64, NumBlocks>;
        array_t bits{};

        static INLINE void swap(u64& a, u64& b) noexcept {
            u64 t = a;
            a = b;
            b = t;
        }

        static INLINE void swap(u64* src, u64* dst, const u8 q) noexcept {
            u64 temp;
            for (u8 i = 0; i < q; i++) {
                temp = dst[i];
                dst[i] = src[i];
                dst[i] = temp;
            }
        }

        [[nodiscard]] static constexpr bool unsigned_less(const Int& a, const Int& b) noexcept {
            for (u8 i = NumBlocks; i-- > 0;) {
                if (a.bits[i] != b.bits[i])
                    return a.bits[i] < b.bits[i];
            }
            return false;
        }

        INLINE constexpr u64 mul(u64 carry, u64 a, u64 b, u64& product) noexcept {
            u64 hi, lo = _mulx_u64(a, b, &hi);
            unsigned char cf = _addcarryx_u64(0, lo, carry, &lo);
            cf = _addcarryx_u64(cf, hi, 0, &hi);
            product = lo;
            return hi;
        }

        // Unrolled addition
        template <u8... I>
        INLINE constexpr void add_unrolled(const Int& o, indices<I...>) noexcept {
            unsigned char carry = 0;
            (..., (carry = _addcarry_u64(carry, bits[I], o.bits[I], &bits[I])));
        }

        // Unrolled subtraction
        template <u8... I>
        INLINE constexpr void sub_unrolled(const Int& o, indices<I...>) noexcept {
            unsigned char borrow = 0;
            (..., (borrow = _subborrow_u64(borrow, bits[I], o.bits[I], &bits[I])));
        }
        template <u8... I>
        INLINE constexpr void mul_unrolled(const u64 o, indices<I...>) noexcept {
            u64 carry = 0;
            (..., (carry = mul(carry, bits[I], o, bits[I])));
        }

        INLINE constexpr Int& singleMul(Int& a, const u64 o) const noexcept{
            u64 carry = 0;
            for (u8 i = 0; i < NumBlocks; i++) {
                auto [hi, lo] = mul(a[i], o);
                a.bits[i] = lo + carry;
                carry = hi;
            }
            return a;
        }

        INLINE constexpr Int& BMR(u8 i) noexcept{
            if (i >= NumBlocks) *this = Int();
            else memmove(this->bits[i], this->bits[0], i);
            return *this;
        }
        INLINE constexpr Int& BML(u8 i) noexcept{
            if (i >= NumBlocks) *this = Int();
            else if (i != 0){
                memmove(&bits[0], &bits[i], (NumBlocks - i) * sizeof(u64));
                memset(&bits[NumBlocks - i], 0, i * sizeof(u64));
            }
            return *this;
        }


        // Auxiliares novos (adicionados como private na classe Int<B>)
        INLINE u8 leading_bit() const noexcept {
            for (u8 i = NumBlocks; i-- > 0;) {
                if (bits[i]) {
                    return i * 64 + (63 - std::countl_zero(bits[i]));
                }
            }
            return 0;  // Para zero, consideramos 0 bits
        }

        INLINE bool test_bit(u8 bit) const noexcept {
            u8 i = 64 & 63;
            return (bits[bit >> 6]) >> (bit & 63);
        }

        INLINE void set_bit(u8 bit) noexcept {
            if (bit >= Bits) return;
            u8 word = bit / 64;
            u8 pos = bit % 64;
            bits[word] |= (1ull << pos);
        }


        [[nodiscard]] std::string toBase16() const;
        [[nodiscard]] std::string to_string() const;

    public:

        // --- Construtores ---
        constexpr Int() noexcept = default;
        constexpr Int(const Int&) noexcept = default;
        constexpr Int& operator=(const Int&) noexcept = default;
        constexpr explicit Int(std::string_view sv);

        constexpr Int(unsigned v) noexcept : Int(static_cast<signed>(v)) {}
        constexpr Int(signed v) noexcept {
            bits.fill((v < 0) ? ~0ull : 0ull);
            bits[0] = static_cast<unsigned>(v);
        }

        template <typename... Args> requires (std::is_unsigned_v<std::decay_t<Args>> && ...)
        constexpr Int(Args... args) noexcept {
            static_assert(sizeof...(args) <= NumBlocks, "Too many initializers for Int");
            bits.fill(0);
            u8 i = 0;
            ((bits[i++] = static_cast<u64>(args)), ...);
        }


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

        INLINE constexpr bool isBaseTwo() const noexcept {
            return popcnt() == 1;
        }

        INLINE constexpr bool isOdd() const noexcept {
            return bits[0] & 1;
        }

        INLINE constexpr bool isEven() const noexcept {
            return !isOdd();
        }


        [[nodiscard]] INLINE constexpr u64& operator[](u8 i) noexcept { // read & write
            assert(i < NumBlocks);
            return bits[i];
        }
        [[nodiscard]] INLINE constexpr const u64& operator[](u8 i) const noexcept { // only read
            assert(i < NumBlocks);
            return bits[i];
        }

        // --- Operadores Aritméticos ---
        INLINE constexpr Int& operator+=(const Int& o) noexcept {
            if constexpr (NumBlocks == 1) this->bits[0] += o.bits[0];
            else add_unrolled(o, build_indices<NumBlocks>{});
            return *this;
        }

        INLINE constexpr Int& operator-=(const Int& o) noexcept {
            if constexpr (NumBlocks == 1) this->bits[0] -= o.bits[0];
            else sub_unrolled(o, build_indices<NumBlocks>{});
            return *this;
        }

        INLINE constexpr Int& operator*=(const u64& o) noexcept{
            if constexpr (NumBlocks == 1) bits[0] *= o;
            else if constexpr (NumBlocks == 2){
                u128 value = (u128) bits[0] * o;
                bits[0] = value;
                bits[1] = (bits[1] * o) + (value >> 64);
            }
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

        // --- Operadores Bitwise ---
        INLINE constexpr Int& operator|=(const Int& o) noexcept {
            for (u8 i = 0; i < NumBlocks; ++i)
                bits[i] |= o.bits[i];
            return *this;
        }

        INLINE constexpr Int& operator&=(const Int& o) noexcept {
            for (u8 i = 0; i < NumBlocks; ++i)
                bits[i] &= o.bits[i];
            return *this;
        }

        INLINE constexpr Int& operator^=(const Int& o) noexcept {
            for (u8 i = 0; i < NumBlocks; ++i)
                bits[i] ^= o.bits[i];
            return *this;
        }

        // --- Shifts Otimizados ---
        [[nodiscard]] INLINE constexpr Int rotl(unsigned k) const noexcept {
            if (k %= Bits == 0) return *this;
            Int temp = *this;
            return *this <<= k |= temp >>= (Bits - k);
        }

        [[nodiscard]] INLINE constexpr Int rotr(unsigned k) const noexcept {
            if (k %= Bits == 0) return *this;
            Int temp = *this;
            return *this >>= k |= temp <<= (Bits - k);
        }

        INLINE constexpr Int& operator<<=(u8 k) noexcept {
            if (k) [[likely]] {// baixa probabilidade de k ser 0
                if (k >= Bits) { bits.fill(0); return *this; }
                if constexpr (NumBlocks > 1){
                    const u8 blocks = k >> 6; // antigo: k / 64
                    if (blocks) { // antes (blocks != 0), isso é a mesma coisa no binario, apenas escrito de outra forma
                        memmove(&bits[blocks], &bits[0], (NumBlocks - blocks) * sizeof(u64));
                        memzero(&bits[0], blocks);
                    }
                    if (k &= 63){
                        u8 index = NumBlocks;
                        while (index-- > 1)
                            bits[index] = bits[index] << k | bits[index - 1] >> (64 - k);
                        // for (u8 i = NumBlocks; i-- > 1; bits[i] = bits[i] << k | bits[i - 1] >> (64 - k));
                    }
                }
                bits[0] <<= k;
            }
            return *this;
        }

        // Move blocos de 64 bits para a esquerda
        INLINE constexpr Int& SLB(const u8 k) noexcept {
            if (BUILTIN_EXPECT(k == 0, 1)) return *this;
            if (BUILTIN_EXPECT(k >= NumBlocks, 0)) { bits.fill(0); return *this; }
            memmove(&bits[k], &bits[0], (NumBlocks - k) * sizeof(u64));
            memset(&bits[0], 0, k * sizeof(u64)); //anterior
        }

        INLINE Int& operator>>=(u8 k) noexcept {
            if constexpr (NumBlocks > 1){
                const u8 block = k >> 6;
                // const u8 block = k / 64; // antigo
                if (block != 0){
                    memmove(&bits[0], &bits[block], (NumBlocks - block) * sizeof(u64));
                    memzero(&bits[NumBlocks - block], block);
                    // memset(&bits[NumBlocks - block], 0, block * sizeof(u64)); // anterior
                }
                if (k %= 64){
                    for (u8 i = 0; i < NumBlocks-1; i++) {
                        (bits[i] >>= k) |= bits[i + 1] << (64 - k);
                        // bits[i] = bits[i] >> k | bits[i + 1] << (64 - k);
                    }
                }
            }
            bits[NumBlocks-1] >>= k;
            return *this;
        }

        // Move blocos de 64 bits para a direita
        INLINE constexpr Int& SRB(const u8 k) noexcept {
            if (BUILTIN_EXPECT(k == 0, 1)) return *this;
            if (BUILTIN_EXPECT(k >= NumBlocks, 0)) {
                bits.fill(0);
                return *this;
            }
            memmove(&bits[0], &bits[k], (NumBlocks - k) * sizeof(u64));
            memzero(&bits[NumBlocks - k], k);
            // memset(&bits[NumBlocks - k], 0, k * sizeof(u64)); // anterior
        }

        // --- Incremento/Decremento ---
        INLINE constexpr Int& operator++() noexcept {
            [[maybe_unused]] bool carry = ++bits[0] == 0;
            if constexpr (NumBlocks > 1){
                if (carry) [[unlikely]] {
                    u8 index = 1;
                    inc: if (++bits[index] == 0) {
                        [[unlikely]] if (++index < NumBlocks) goto inc;
                    }
                }
            }
            return *this;
        }

        // outras operações binarias.

        // retorna a contagem de numeros de bits setados como 1
        INLINE constexpr u8 popcnt() const noexcept {
            u8 count = 0;
            for (auto word : bits) {
                count += __builtin_popcountll(word);
            }
            return count;
        }

        INLINE constexpr Int& operator--() noexcept {
            [[maybe_unused]] bool carry = --bits[0] == -1ull;
            if constexpr (NumBlocks > 1){
                if (carry) [[unlikely]] {
                    u8 index = 1;
                    inc: if (--bits[index] == -1ull) {
                        [[unlikely]] if (++index < NumBlocks) goto inc;
                    }
                }
            }



            // u8 i = 0;
            // dec: if (--bits[i++] == (u64)-1) [[unlikely]] goto dec;
            return *this;
        }
        INLINE constexpr Int operator++(int) noexcept { Int t = *this; ++*this; return t; }
        INLINE constexpr Int operator--(int) noexcept { Int t = *this; --*this; return t; }

        // --- Negação e Complemento ---
        INLINE constexpr Int operator~() const noexcept {
            Int<Bits> temp;
            for (u8 i = 0; i < NumBlocks; i++) temp.bits[i] = ~bits[i];
            return temp;
        }

        [[nodiscard]] INLINE constexpr Int operator-() const noexcept { return ~(*this) + Int(1); }

        // --- Comparações ---
        [[nodiscard]] friend constexpr bool operator==(const Int& a, const u64 b) noexcept {
            if constexpr (NumBlocks == 1) return a.bits[0] == b;
            else return a.bits[0] == b && std::all_of(a.bits.begin() + 1, a.bits.end(), [](u64 x){ return x == 0; });
        }
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
template<u8 B>
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

        u8 start_idx = 0;
        u8 end_idx = sv.size();
        u8 bit_pos = 0;
        u8 block_idx = 0;

        // Processar de trás para frente (LSB primeiro)
        while (end_idx > start_idx && block_idx < NumBlocks) {
            u8 digits_to_process = std::min<u8>(16, end_idx - start_idx);
            u8 chunk_start = end_idx - digits_to_process;
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
        constexpr u8 max_digits = ((B - 1) * 301LLU / 1000) + 2;
        if (sv.size() > max_digits) {
            throw std::out_of_range("Decimal string too long");
        }
        constexpr u64 CHUNK_POW = 1000000000000000000ull; // 10^18
        constexpr int CHUNK_SIZE = 18;

        const char* p = sv.data();
        const char* const end = p + sv.size();

        u8 first_chunk_len = sv.size() % CHUNK_SIZE;
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

template<u8 B>
std::string Int<B>::toBase16() const {
    if (*this == 0) return "0x0";
    Int temp = abs();
    int msb_idx = static_cast<int>(NumBlocks) - 1;
    while (msb_idx > 0 && temp.bits[msb_idx] == 0) --msb_idx;
    std::ostringstream oss;
    if (isNegative()) oss << '-';
    oss << "0x";
    oss << std::hex << std::nouppercase << temp.bits[msb_idx];
    for (int i = msb_idx - 1; i >= 0; --i) {
        oss << std::setw(16) << std::setfill('0') << temp.bits[i];
    }
    return oss.str();
}

template<u8 B>
std::string Int<B>::to_string() const {
    if (*this == Int<B>(0)) return "0";

    Int temp = abs();
    constexpr u64 CHUNK_POW = 1000000000000000000ull; // 10^18
    constexpr int CHUNK_SIZE = 18;

    std::vector<u64> chunks;
    while (temp != Int<B>(0)) {
        Int quotient;
        u64 remainder = 0;

        for (u8 i = NumBlocks; i-- > 0;) {
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
template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator+(const Int<B>& a, const Int<B>& b) noexcept {
    Int<B> temp = a;
    temp += b;
    return temp;
}
template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator-(const Int<B>& a, const Int<B>& b) noexcept {
    Int<B> temp = a;
    temp -= b;
    return temp;
}
template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator*(const Int<B>& a, const Int<B>& b) noexcept {
    Int<B> temp = a;
    temp *= b;
    return temp;
}
template<u8 B> [[nodiscard]] INLINE Int<B> operator/(Int<B> a, const Int<B>& b) { return a /= b; }
template<u8 B> [[nodiscard]] INLINE Int<B> operator%(Int<B> a, const Int<B>& b) { return a %= b; }
template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator|(Int<B> a, const Int<B>& b) noexcept { return a |= b; }
template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator&(Int<B> a, const Int<B>& b) noexcept { return a &= b; }
template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator^(Int<B> a, const Int<B>& b) noexcept { return a ^= b; }
template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator<<(Int<B> a, unsigned k) noexcept { return a <<= k; }
template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator>>(Int<B> a, unsigned k) noexcept { return a >>= k; }

// Operadores com Tipos Nativos
#define DEFINE_INT_NATIVE_OPS(T) \
    template<u8 B> [[nodiscard]] INLINE constexpr bool operator==(const Int<B>& a, T b) noexcept { return a == Int<B>(b); } \
    template<u8 B> [[nodiscard]] INLINE constexpr std::strong_ordering operator<=>(const Int<B>& a, T b) noexcept { return a <=> Int<B>(b); } \
    template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator+(const Int<B>& a, T b) noexcept { return a + Int<B>(b); } \
    template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator-(const Int<B>& a, T b) noexcept { return a - Int<B>(b); } \
    template<u8 B> [[nodiscard]] INLINE constexpr Int<B> operator*(const Int<B>& a, T b) noexcept { return a * Int<B>(b); } \
    template<u8 B> [[nodiscard]] INLINE Int<B> operator/(const Int<B>& a, T b) { return a / Int<B>(b); } \
    template<u8 B> [[nodiscard]] INLINE Int<B> operator%(const Int<B>& a, T b) { return a % Int<B>(b); }

DEFINE_INT_NATIVE_OPS(i64)
DEFINE_INT_NATIVE_OPS(u64)

// I/O e Literais
template<u8 B>
std::ostream& operator<<(std::ostream& os, const Int<B>& val) {
    return os << val.to_string();
}

// #define DEFINE_INT_LITERAL(bits) \
//     [[nodiscard]] consteval Int<bits> operator""_i##bits(const char* s, u8) { return Int<bits>(s); }

// inline namespace int_literals {
//     DEFINE_INT_LITERAL(128)
//     DEFINE_INT_LITERAL(256)
//     DEFINE_INT_LITERAL(512)
// //     DEFINE_INT_LITERAL(1024)
// //     DEFINE_INT_LITERAL(2048)
// // }

// #undef DEFINE_INT_LITERAL

// // Aliases
// using Int128 = Int<128>;
// using Int256 = Int<256>;
// using Int512 = Int<512>;
// using Int1024 = Int<1024>;
// using Int2048 = Int<2048>;
