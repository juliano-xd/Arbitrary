#pragma once

#include <algorithm>
#include <cstring>
#include <immintrin.h>
#include <type_traits>
#include <utility>

namespace Multiplication {

    using u64 = unsigned long long;
    using u128 = unsigned __int128;

    // Base case: O(N^2) full multiplication using u128 arithmetic
    // Computes res[0..2n-1] = a[0..n-1] * b[0..n-1]
    //
    // @param res Output buffer of size 2*n
    // @param a Input array A of size n
    // @param b Input array B of size n
    // @param n Number of limbs
    inline void mul_schoolbook_full(u64 *res, const u64 *a, const u64 *b, size_t n) noexcept {
        memset(res, 0, 2 * n * sizeof(u64));  // Faster than std::fill_n
        for (size_t i = 0; i < n; ++i) {
            u64 y = a[i];
            if (y == 0)  // Keep branch - it's well predicted and saves work
                continue;

            // Prefetch next iteration data for better cache performance
            if (i + 1 < n) {
                __builtin_prefetch(&a[i + 1], 0, 3);
                __builtin_prefetch(&res[i + n + 1], 1, 3);
            }

            u128 carry = 0;
            for (size_t j = 0; j < n; ++j) {
                u128 temp = (u128)b[j] * y + res[i + j] + carry;
                res[i + j] = (u64)temp;
                carry = temp >> 64;
            }
            size_t k = i + n;
            while (carry) {
                u128 temp = (u128)res[k] + carry;
                res[k] = (u64)temp;
                carry = temp >> 64;
                k++;
            }
        }
    }

    // Template for fixed-size schoolbook multiplication
    template <size_t N>
    __attribute__((always_inline))
    inline void mul_schoolbook_fixed(u64 *res, const u64 *a, const u64 *b) noexcept {
        std::fill_n(res, 2 * N, 0);
        #pragma GCC unroll 16
        for (size_t i = 0; i < N; ++i) {
            u64 y = a[i];
            if (y == 0) continue;
            u128 carry = 0;
            #pragma GCC unroll 16
            for (size_t j = 0; j < N; ++j) {
                u128 temp = (u128)b[j] * y + res[i + j] + carry;
                res[i + j] = (u64)temp;
                carry = temp >> 64;
            }
            size_t k = i + N;
            while (carry && k < 2 * N) {
                u128 temp = (u128)res[k] + carry;
                res[k] = (u64)temp;
                carry = temp >> 64;
                k++;
            }
        }
    }

    // --- Specialized Kernels ---

    // Specialized 2x2 kernel
    __attribute__((always_inline))
    inline void mul_schoolbook_fixed_2x2(u64 * __restrict__ res, const u64 * __restrict__ a, const u64 * __restrict__ b) noexcept {
        u64 lo0, hi0, lo1, hi1;

        // Row 0: a[0] * b[0..1]
        lo0 = _mulx_u64(a[0], b[0], &hi0);
        lo1 = _mulx_u64(a[0], b[1], &hi1);
        res[0] = lo0;
        unsigned char c = 0;
        c = _addcarry_u64(c, hi0, lo1, &res[1]);
        c = _addcarry_u64(c, hi1, 0, &res[2]);

        // Row 1: a[1] * b[0..1]
        lo0 = _mulx_u64(a[1], b[0], &hi0);
        lo1 = _mulx_u64(a[1], b[1], &hi1);

        c = 0;
        c = _addcarry_u64(c, res[1], lo0, &res[1]);
        c = _addcarry_u64(c, res[2], lo1, &res[2]);
        u64 c_lo = c;

        c = 0;
        c = _addcarry_u64(c, res[2], hi0, &res[2]);
        u64 tmp;
        c = _addcarry_u64(c, hi1, c_lo, &tmp);
        res[3] = tmp;
    }

    // Specialized 3x3 kernel
    __attribute__((always_inline))
    inline void mul_schoolbook_fixed_3x3(u64 * __restrict__ res, const u64 * __restrict__ a, const u64 * __restrict__ b) noexcept {
        u64 lo0, hi0, lo1, hi1, lo2, hi2;

        // Row 0
        {
            lo0 = _mulx_u64(a[0], b[0], &hi0);
            lo1 = _mulx_u64(a[0], b[1], &hi1);
            lo2 = _mulx_u64(a[0], b[2], &hi2);

            res[0] = lo0;
            unsigned char c = 0;
            c = _addcarry_u64(c, hi0, lo1, &res[1]);
            c = _addcarry_u64(c, hi1, lo2, &res[2]);
            c = _addcarry_u64(c, hi2, 0, &res[3]);
        }

        // Row 1
        {
            lo0 = _mulx_u64(a[1], b[0], &hi0);
            lo1 = _mulx_u64(a[1], b[1], &hi1);
            lo2 = _mulx_u64(a[1], b[2], &hi2);

            unsigned char c = 0;
            c = _addcarry_u64(c, res[1], lo0, &res[1]);
            c = _addcarry_u64(c, res[2], lo1, &res[2]);
            c = _addcarry_u64(c, res[3], lo2, &res[3]);
            u64 c_lo = c;

            c = 0;
            c = _addcarry_u64(c, res[2], hi0, &res[2]);
            c = _addcarry_u64(c, res[3], hi1, &res[3]);
            u64 tmp;
            c = _addcarry_u64(c, hi2, c_lo, &tmp);
            res[4] = tmp;
        }

        // Row 2
        {
            lo0 = _mulx_u64(a[2], b[0], &hi0);
            lo1 = _mulx_u64(a[2], b[1], &hi1);
            lo2 = _mulx_u64(a[2], b[2], &hi2);

            unsigned char c = 0;
            c = _addcarry_u64(c, res[2], lo0, &res[2]);
            c = _addcarry_u64(c, res[3], lo1, &res[3]);
            c = _addcarry_u64(c, res[4], lo2, &res[4]);
            u64 c_lo = c;

            c = 0;
            c = _addcarry_u64(c, res[3], hi0, &res[3]);
            c = _addcarry_u64(c, res[4], hi1, &res[4]);
            u64 tmp;
            c = _addcarry_u64(c, hi2, c_lo, &tmp);
            res[5] = tmp;
        }
    }
    __attribute__((always_inline))
    inline void mul_schoolbook_fixed_4x4(u64 * __restrict__ res, const u64 * __restrict__ a, const u64 * __restrict__ b) noexcept {
        // Row 0
        {
            u64 lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
            lo0 = _mulx_u64(a[0], b[0], &hi0);
            lo1 = _mulx_u64(a[0], b[1], &hi1);
            lo2 = _mulx_u64(a[0], b[2], &hi2);
            lo3 = _mulx_u64(a[0], b[3], &hi3);

            res[0] = lo0;
            unsigned char c = 0;
            c = _addcarry_u64(c, hi0, lo1, &res[1]);
            c = _addcarry_u64(c, hi1, lo2, &res[2]);
            c = _addcarry_u64(c, hi2, lo3, &res[3]);
            c = _addcarry_u64(c, hi3, 0, &res[4]);
        }

        // Row 1
        {
            u64 lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
            lo0 = _mulx_u64(a[1], b[0], &hi0);
            lo1 = _mulx_u64(a[1], b[1], &hi1);
            lo2 = _mulx_u64(a[1], b[2], &hi2);
            lo3 = _mulx_u64(a[1], b[3], &hi3);

            unsigned char c = 0;
            c = _addcarry_u64(c, res[1], lo0, &res[1]);
            c = _addcarry_u64(c, res[2], lo1, &res[2]);
            c = _addcarry_u64(c, res[3], lo2, &res[3]);
            c = _addcarry_u64(c, res[4], lo3, &res[4]);
            u64 c_lo = c;

            c = 0;
            c = _addcarry_u64(c, res[2], hi0, &res[2]);
            c = _addcarry_u64(c, res[3], hi1, &res[3]);
            c = _addcarry_u64(c, res[4], hi2, &res[4]);
            u64 tmp;
            c = _addcarry_u64(c, hi3, c_lo, &tmp);
            res[5] = tmp;
        }

        // Row 2
        {
            u64 lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
            lo0 = _mulx_u64(a[2], b[0], &hi0);
            lo1 = _mulx_u64(a[2], b[1], &hi1);
            lo2 = _mulx_u64(a[2], b[2], &hi2);
            lo3 = _mulx_u64(a[2], b[3], &hi3);

            unsigned char c = 0;
            c = _addcarry_u64(c, res[2], lo0, &res[2]);
            c = _addcarry_u64(c, res[3], lo1, &res[3]);
            c = _addcarry_u64(c, res[4], lo2, &res[4]);
            c = _addcarry_u64(c, res[5], lo3, &res[5]);
            u64 c_lo = c;

            c = 0;
            c = _addcarry_u64(c, res[3], hi0, &res[3]);
            c = _addcarry_u64(c, res[4], hi1, &res[4]);
            c = _addcarry_u64(c, res[5], hi2, &res[5]);
            u64 tmp;
            c = _addcarry_u64(c, hi3, c_lo, &tmp);
            res[6] = tmp;
        }

        // Row 3
        {
            u64 lo0, hi0, lo1, hi1, lo2, hi2, lo3, hi3;
            lo0 = _mulx_u64(a[3], b[0], &hi0);
            lo1 = _mulx_u64(a[3], b[1], &hi1);
            lo2 = _mulx_u64(a[3], b[2], &hi2);
            lo3 = _mulx_u64(a[3], b[3], &hi3);

            unsigned char c = 0;
            c = _addcarry_u64(c, res[3], lo0, &res[3]);
            c = _addcarry_u64(c, res[4], lo1, &res[4]);
            c = _addcarry_u64(c, res[5], lo2, &res[5]);
            c = _addcarry_u64(c, res[6], lo3, &res[6]);
            u64 c_lo = c;

            c = 0;
            c = _addcarry_u64(c, res[4], hi0, &res[4]);
            c = _addcarry_u64(c, res[5], hi1, &res[5]);
            c = _addcarry_u64(c, res[6], hi2, &res[6]);
            u64 tmp;
            c = _addcarry_u64(c, hi3, c_lo, &tmp);
            res[7] = tmp;
        }
    }


    // --- Dispatchers ---

    // Dispatcher for full multiplication to leverage fixed-size templates
    inline void mul_schoolbook_dispatch(u64 *res, const u64 *a, const u64 *b, size_t n) noexcept {
        switch(n) {
            case 1: mul_schoolbook_fixed<1>(res, a, b); break;
            case 2: mul_schoolbook_fixed_2x2(res, a, b); break;
            case 3: mul_schoolbook_fixed_3x3(res, a, b); break;
            case 4: mul_schoolbook_fixed_4x4(res, a, b); break; // Use specialized kernel
            case 5: mul_schoolbook_fixed<5>(res, a, b); break;
            case 6: mul_schoolbook_fixed<6>(res, a, b); break;
            case 7: mul_schoolbook_fixed<7>(res, a, b); break;
            case 8: mul_schoolbook_fixed<8>(res, a, b); break;
            case 9: mul_schoolbook_fixed<9>(res, a, b); break;
            case 10: mul_schoolbook_fixed<10>(res, a, b); break;
            case 11: mul_schoolbook_fixed<11>(res, a, b); break;
            case 12: mul_schoolbook_fixed<12>(res, a, b); break;
            case 13: mul_schoolbook_fixed<13>(res, a, b); break;
            case 14: mul_schoolbook_fixed<14>(res, a, b); break;
            case 15: mul_schoolbook_fixed<15>(res, a, b); break;
            case 16: mul_schoolbook_fixed<16>(res, a, b); break;
            case 17: mul_schoolbook_fixed<17>(res, a, b); break;
            case 18: mul_schoolbook_fixed<18>(res, a, b); break;
            case 19: mul_schoolbook_fixed<19>(res, a, b); break;
            case 20: mul_schoolbook_fixed<20>(res, a, b); break;
            case 21: mul_schoolbook_fixed<21>(res, a, b); break;
            case 22: mul_schoolbook_fixed<22>(res, a, b); break;
            case 23: mul_schoolbook_fixed<23>(res, a, b); break;
            case 24: mul_schoolbook_fixed<24>(res, a, b); break;
            case 25: mul_schoolbook_fixed<25>(res, a, b); break;
            case 26: mul_schoolbook_fixed<26>(res, a, b); break;
            case 27: mul_schoolbook_fixed<27>(res, a, b); break;
            case 28: mul_schoolbook_fixed<28>(res, a, b); break;
            case 29: mul_schoolbook_fixed<29>(res, a, b); break;
            case 30: mul_schoolbook_fixed<30>(res, a, b); break;
            case 31: mul_schoolbook_fixed<31>(res, a, b); break;
            case 32: mul_schoolbook_fixed<32>(res, a, b); break;
            default: mul_schoolbook_full(res, a, b, n); break;
        }
    }

    // Template for fixed-size truncated multiplication
    template <size_t N>
    __attribute__((always_inline))
    inline void mul_schoolbook_truncated_fixed(u64 *res, const u64 *a, const u64 *b) {
        std::fill_n(res, N, 0);
        #pragma GCC unroll 16
        for (size_t i = 0; i < N; ++i) {
            u64 y = a[i];
            if (y == 0) continue;
            u128 carry = 0;
            #pragma GCC unroll 16
            for (size_t j = 0; j < N - i; ++j) {
                u128 temp = (u128)b[j] * y + res[i + j] + carry;
                res[i + j] = (u64)temp;
                carry = temp >> 64;
            }
        }
    }

    // Template for fixed-size truncated Comba multiplication (Product Scanning)
    // Significantly reduces register pressure compared to Schoolbook for small N.
    // Computes res[k] = sum(a[i]*b[j]) for i+j=k, only for k < N.
    template <size_t N>
    __attribute__((always_inline))
    inline void mul_comba_truncated_fixed(u64 *res, const u64 *a, const u64 *b) {
        // Use 3-limb accumulator for safe carry propagation
        u64 r0 = 0, r1 = 0, r2 = 0;

        // Compute columns k = 0 to N-1
        [&]<std::size_t... K>(std::index_sequence<K...>) {
            auto step_col = [&](auto k_const) {
                constexpr size_t k = k_const;

                // Inner loop: sum(a[i]*b[j])
                [&]<std::size_t... I>(std::index_sequence<I...>) {
                    auto step_row = [&](auto i_const) {
                         constexpr size_t i = i_const;
                         constexpr size_t j = k - i;

                         u64 lo, hi;
                         lo = _mulx_u64(a[i], b[j], &hi);

                         unsigned char c;
                         c = _addcarry_u64(0, r0, lo, &r0);
                         c = _addcarry_u64(c, r1, hi, &r1);
                         c = _addcarry_u64(c, r2, 0, &r2);
                    };
                    (step_row(std::integral_constant<std::size_t, I>{}), ...);
                }(std::make_index_sequence<k + 1>{}); // i goes 0 to k

                res[k] = r0;
                // Shift accumulators
                r0 = r1;
                r1 = r2;
                r2 = 0;
            };
            (step_col(std::integral_constant<std::size_t, K>{}), ...);
        }(std::make_index_sequence<N>{});
    }

    // Specialized truncated 2x2 (Register Optimized)
    __attribute__((always_inline))
    inline void mul_schoolbook_truncated_fixed_2x2(u64& r0, u64& r1, u64 a0, u64 a1, u64 b0, u64 b1) noexcept {
        u128 full_product = static_cast<u128>(a0) * b0;
        r0 = static_cast<u64>(full_product);       // Bits [0..63]
        r1 = static_cast<u64>(full_product >> 64); // Carry inicial para Bits [64..127]
        r1 += a0 * b1; // Otimizado!
        r1 += a1 * b0; // Otimizado!
    }
    // Specialized truncated 3x3
    __attribute__((always_inline))
    inline void mul_schoolbook_truncated_fixed_3x3(u64 * res, const u64 * a, const u64 * b) noexcept {
        u64 r0 = 0, r1 = 0, r2 = 0;

        // k=0: a0b0
        {
            u64 lo, hi;
            lo = _mulx_u64(a[0], b[0], &hi);
            r0 = lo;
            r1 = hi;
        }

        // k=1: a0b1 + a1b0
        {
            u64 lo, hi;
            unsigned char c = 0;

            lo = _mulx_u64(a[0], b[1], &hi);
            c = _addcarry_u64(c, r1, lo, &r1);
            c = _addcarry_u64(c, r2, hi, &r2);

            c = 0;
            lo = _mulx_u64(a[1], b[0], &hi);
            c = _addcarry_u64(c, r1, lo, &r1);
            c = _addcarry_u64(c, r2, hi, &r2);
        }
        res[0] = r0;

        // k=2: a0b2 + a1b1 + a2b0
        {
            u64 lo, hi;
            unsigned char c = 0;

            lo = _mulx_u64(a[0], b[2], &hi);
            c = _addcarry_u64(c, r2, lo, &r2);

            c = 0;
            lo = _mulx_u64(a[1], b[1], &hi);
            c = _addcarry_u64(c, r2, lo, &r2);

            c = 0;
            lo = _mulx_u64(a[2], b[0], &hi);
            c = _addcarry_u64(c, r2, lo, &r2);
        }
        res[1] = r1;
        res[2] = r2;
    }

    // Specialized truncated 4x4
    __attribute__((always_inline))
    inline void mul_schoolbook_truncated_fixed_4x4(u64 * __restrict__ res, const u64 * __restrict__ a, const u64 * __restrict__ b) noexcept {
        // Safe implementation avoiding u128 overflow
        u64 r0 = 0, r1 = 0, r2 = 0;

        // k=0: a0b0
        {
            u64 lo, hi;
            lo = _mulx_u64(a[0], b[0], &hi);
            unsigned char c = 0;
            c = _addcarry_u64(c, r0, lo, &r0);
            c = _addcarry_u64(c, r1, hi, &r1);
            c = _addcarry_u64(c, r2, 0, &r2);
            res[0] = r0;
            r0 = r1; r1 = r2; r2 = 0;
        }

        // k=1: a0b1 + a1b0
        {
            u64 lo, hi;
            unsigned char c = 0;

            lo = _mulx_u64(a[0], b[1], &hi);
            c = _addcarry_u64(c, r0, lo, &r0);
            c = _addcarry_u64(c, r1, hi, &r1);
            c = _addcarry_u64(c, r2, 0, &r2); // Accum carry to r2

            c = 0;
            lo = _mulx_u64(a[1], b[0], &hi);
            c = _addcarry_u64(c, r0, lo, &r0);
            c = _addcarry_u64(c, r1, hi, &r1);
            c = _addcarry_u64(c, r2, 0, &r2);

            res[1] = r0;
            r0 = r1; r1 = r2; r2 = 0;
        }

        // k=2: a0b2 + a1b1 + a2b0
        {
            u64 lo, hi;
            unsigned char c = 0;

            lo = _mulx_u64(a[0], b[2], &hi);
            c = _addcarry_u64(c, r0, lo, &r0);
            c = _addcarry_u64(c, r1, hi, &r1);
            c = _addcarry_u64(c, r2, 0, &r2);

            c = 0;
            lo = _mulx_u64(a[1], b[1], &hi);
            c = _addcarry_u64(c, r0, lo, &r0);
            c = _addcarry_u64(c, r1, hi, &r1);
            c = _addcarry_u64(c, r2, 0, &r2);

            c = 0;
            lo = _mulx_u64(a[2], b[0], &hi);
            c = _addcarry_u64(c, r0, lo, &r0);
            c = _addcarry_u64(c, r1, hi, &r1);
            c = _addcarry_u64(c, r2, 0, &r2);

            res[2] = r0;
            r0 = r1; r1 = r2; r2 = 0;
        }

        // k=3: sum(a[i]*b[j])
        // Parallelize multiplication and summation to maximize ILP.
        {
            u64 p0 = a[0] * b[3];
            u64 p1 = a[1] * b[2];
            u64 p2 = a[2] * b[1];
            u64 p3 = a[3] * b[0];

            // Tree reduction
            u64 s0 = p0 + p1;
            u64 s1 = p2 + p3;

            res[3] = r0 + s0 + s1;
        }
    }

    // Specialized Split-Block 8-limb Multiplication
    // Decomposes N=8 into 4x4 blocks to increase ILP.
    // A*B = (Al + Ah*2^256) * (Bl + Bh*2^256)
    //     = Al*Bl + (Al*Bh + Ah*Bl)*2^256 + ...
    // We need result mod 2^512.
    __attribute__((always_inline))
    inline void mul_split_truncated_fixed_8(u64 *res, const u64 *a, const u64 *b) {
        // 1. Compute Full Lower Product: Al * Bl -> res[0..7]
        mul_schoolbook_fixed_4x4(res, a, b);

        // 2. Compute Truncated Cross Products: (Al * Bh) and (Ah * Bl)
        u64 t1[4];
        mul_schoolbook_truncated_fixed_4x4(t1, a, b + 4);

        u64 t2[4];
        mul_schoolbook_truncated_fixed_4x4(t2, a + 4, b);

        // 3. Accumulate Cross Products into Upper Half: res[4..7]
        unsigned char c = 0;
        c = _addcarry_u64(c, res[4], t1[0], &res[4]);
        c = _addcarry_u64(c, res[5], t1[1], &res[5]);
        c = _addcarry_u64(c, res[6], t1[2], &res[6]);
        c = _addcarry_u64(c, res[7], t1[3], &res[7]);

        c = 0;
        c = _addcarry_u64(c, res[4], t2[0], &res[4]);
        c = _addcarry_u64(c, res[5], t2[1], &res[5]);
        c = _addcarry_u64(c, res[6], t2[2], &res[6]);
        c = _addcarry_u64(c, res[7], t2[3], &res[7]);
    }


    // Dispatcher for truncated schoolbook multiplication
    inline void mul_schoolbook_truncated_dispatch(u64 *res, const u64 *a, const u64 *b, size_t n) noexcept {
        switch(n) {
            case 4: mul_schoolbook_truncated_fixed_4x4(res, a, b); break;
            // Add other specializations here if needed
            default: {
                std::fill_n(res, n, 0);
                for (size_t i = 0; i < n; ++i) {
                    u64 y = a[i];
                    if (y == 0) continue;
                    u128 carry = 0;
                    for (size_t j = 0; j < n - i; ++j) {
                        u128 temp = (u128)b[j] * y + res[i + j] + carry;
                        res[i + j] = (u64)temp;
                        carry = temp >> 64;
                    }
                }
            } break;
        }
    }

    // Template for fixed-size truncated squaring
    // Computes lower N limbs of a^2
    template <size_t N>
    __attribute__((always_inline))
    inline void square_schoolbook_truncated_fixed(u64 *res, const u64 *a) {
        std::fill_n(res, N, 0);

        // 1. Cross products: sum(a[i]*a[j]) for j > i
        for (size_t i = 0; i < N - 1; ++i) {
            u64 y = a[i];
            if (y == 0) continue;
            u128 carry = 0;
            for (size_t j = i + 1; j < N - i; ++j) {
                u128 temp = (u128)a[j] * y + res[i + j] + carry;
                res[i + j] = (u64)temp;
                carry = temp >> 64;
            }
        }

        // 2. Shift left by 1 (multiply cross products by 2)
        u64 shift_carry = 0;
        for (size_t i = 0; i < N; ++i) {
            u64 next = res[i] >> 63;
            res[i] = (res[i] << 1) | shift_carry;
            shift_carry = next;
        }

        // 3. Add squares: a[i]^2
        for (size_t i = 0; i < (N + 1) / 2; ++i) {
            u128 sq = (u128)a[i] * a[i];
            u64 lo = (u64)sq;
            u64 hi = (u64)(sq >> 64);

            unsigned char c = _addcarry_u64(0, res[2*i], lo, &res[2*i]);
            if (2*i + 1 < N) {
                c = _addcarry_u64(c, res[2*i+1], hi, &res[2*i+1]);
                size_t k = 2*i + 2;
                while (c && k < N) {
                    c = _addcarry_u64(c, res[k], 0, &res[k]);
                    k++;
                }
            }
        }
    }

    // Full squaring using schoolbook method
    inline void square_schoolbook_full(u64 *res, const u64 *a, size_t n) noexcept {
        std::fill_n(res, 2 * n, 0);

        // 1. Cross products
        for (size_t i = 0; i < n - 1; ++i) {
            u64 y = a[i];
            if (y == 0) continue;
            u128 carry = 0;
            for (size_t j = i + 1; j < n; ++j) {
                u128 temp = (u128)a[j] * y + res[i + j] + carry;
                res[i + j] = (u64)temp;
                carry = temp >> 64;
            }
            size_t k = i + n;
            while (carry) {
                u128 temp = (u128)res[k] + carry;
                res[k] = (u64)temp;
                carry = temp >> 64;
                k++;
            }
        }

        // 2. Shift left by 1
        u64 shift_carry = 0;
        for (size_t i = 0; i < 2 * n; ++i) {
            u64 next = res[i] >> 63;
            res[i] = (res[i] << 1) | shift_carry;
            shift_carry = next;
        }

        // 3. Add squares
        for (size_t i = 0; i < n; ++i) {
            u128 sq = (u128)a[i] * a[i];
            u64 lo = (u64)sq;
            u64 hi = (u64)(sq >> 64);

            unsigned char c = _addcarry_u64(0, res[2*i], lo, &res[2*i]);
            c = _addcarry_u64(c, res[2*i+1], hi, &res[2*i+1]);
            size_t k = 2*i + 2;
            while (c && k < 2*n) {
                c = _addcarry_u64(c, res[k], 0, &res[k]);
                k++;
            }
        }
    }

    // Flattened Karatsuba for fixed size N
    // Calls mul_schoolbook_fixed for halves to avoid recursion overhead
    template <size_t N>
    __attribute__((always_inline))
    inline void mul_karatsuba_flat(u64 *res, const u64 *a, const u64 *b, u64 *tmp) noexcept {
        constexpr size_t M = N / 2;
        constexpr size_t M2 = N - M;

        const u64 *a0 = a;
        const u64 *a1 = a + M;
        const u64 *b0 = b;
        const u64 *b1 = b + M;

        u64 *z0 = res;
        u64 *z2 = res + 2 * M;
        u64 *sum_a = tmp;
        u64 *sum_b = sum_a + M2 + 1;
        u64 *z1 = sum_b + M2 + 1;

        mul_schoolbook_fixed<M>(z0, a0, b0);
        mul_schoolbook_fixed<M2>(z2, a1, b1);

        unsigned char c_a = 0;
        #pragma GCC unroll 8
        for (size_t i = 0; i < M; ++i) c_a = _addcarry_u64(c_a, a0[i], a1[i], &sum_a[i]);
        for (size_t i = M; i < M2; ++i) c_a = _addcarry_u64(c_a, 0, a1[i], &sum_a[i]);
        sum_a[M2] = c_a;

        unsigned char c_b = 0;
        #pragma GCC unroll 8
        for (size_t i = 0; i < M; ++i) c_b = _addcarry_u64(c_b, b0[i], b1[i], &sum_b[i]);
        for (size_t i = M; i < M2; ++i) c_b = _addcarry_u64(c_b, 0, b1[i], &sum_b[i]);
        sum_b[M2] = c_b;

        mul_schoolbook_fixed<M2 + 1>(z1, sum_a, sum_b);

        unsigned char b_sub = 0;
        #pragma GCC unroll 8
        for (size_t i = 0; i < 2 * M; ++i) b_sub = _subborrow_u64(b_sub, z1[i], z0[i], &z1[i]);
        #pragma GCC unroll 8
        for (size_t i = 2 * M; i < 2 * M2 + 2; ++i) b_sub = _subborrow_u64(b_sub, z1[i], 0, &z1[i]);

        b_sub = 0;
        #pragma GCC unroll 8
        for (size_t i = 0; i < 2 * M2; ++i) b_sub = _subborrow_u64(b_sub, z1[i], z2[i], &z1[i]);
        #pragma GCC unroll 8
        for (size_t i = 2 * M2; i < 2 * M2 + 2; ++i) b_sub = _subborrow_u64(b_sub, z1[i], 0, &z1[i]);

        unsigned char c_add = 0;
        #pragma GCC unroll 8
        for (size_t i = 0; i < 2 * M2 + 2; ++i) {
            if (M + i < 2 * N) c_add = _addcarry_u64(c_add, res[M + i], z1[i], &res[M + i]);
        }
    }

    // Recursive Karatsuba Full Multiplication
    inline void mul_karatsuba_full(u64 *res, const u64 *a, const u64 *b, size_t n,
                                   u64 *tmp) noexcept {
        if (n <= 32) [[likely]] {
            mul_schoolbook_dispatch(res, a, b, n);
            return;
        }

        size_t m = n / 2;
        size_t m2 = n - m;

        const u64 *a0 = a;
        const u64 *a1 = a + m;
        const u64 *b0 = b;
        const u64 *b1 = b + m;

        u64 *z0 = res;
        u64 *z2 = res + 2 * m;

        u64 *sum_a = tmp;
        u64 *sum_b = sum_a + m2 + 1;
        u64 *z1 = sum_b + m2 + 1;

        mul_karatsuba_full(z0, a0, b0, m, z1 + 2 * (m2 + 1));
        mul_karatsuba_full(z2, a1, b1, m2, z1 + 2 * (m2 + 1));

        unsigned char c_a = 0;
        size_t i = 0;
        for (; i + 4 <= m; i += 4) {
            c_a = _addcarry_u64(c_a, a0[i], a1[i], &sum_a[i]);
            c_a = _addcarry_u64(c_a, a0[i+1], a1[i+1], &sum_a[i+1]);
            c_a = _addcarry_u64(c_a, a0[i+2], a1[i+2], &sum_a[i+2]);
            c_a = _addcarry_u64(c_a, a0[i+3], a1[i+3], &sum_a[i+3]);
        }
        for (; i < m; ++i)
            c_a = _addcarry_u64(c_a, a0[i], a1[i], &sum_a[i]);
        for (; i < m2; ++i)
            c_a = _addcarry_u64(c_a, 0, a1[i], &sum_a[i]);
        sum_a[m2] = c_a;

        unsigned char c_b = 0;
        i = 0;
        for (; i + 4 <= m; i += 4) {
            c_b = _addcarry_u64(c_b, b0[i], b1[i], &sum_b[i]);
            c_b = _addcarry_u64(c_b, b0[i+1], b1[i+1], &sum_b[i+1]);
            c_b = _addcarry_u64(c_b, b0[i+2], b1[i+2], &sum_b[i+2]);
            c_b = _addcarry_u64(c_b, b0[i+3], b1[i+3], &sum_b[i+3]);
        }
        for (; i < m; ++i)
            c_b = _addcarry_u64(c_b, b0[i], b1[i], &sum_b[i]);
        for (; i < m2; ++i)
            c_b = _addcarry_u64(c_b, 0, b1[i], &sum_b[i]);
        sum_b[m2] = c_b;

        mul_karatsuba_full(z1, sum_a, sum_b, m2 + 1, z1 + 2 * (m2 + 1));

        unsigned char b_sub = 0;
        i = 0;
        for (; i + 4 <= 2 * m; i += 4) {
            b_sub = _subborrow_u64(b_sub, z1[i], z0[i], &z1[i]);
            b_sub = _subborrow_u64(b_sub, z1[i+1], z0[i+1], &z1[i+1]);
            b_sub = _subborrow_u64(b_sub, z1[i+2], z0[i+2], &z1[i+2]);
            b_sub = _subborrow_u64(b_sub, z1[i+3], z0[i+3], &z1[i+3]);
        }
        for (; i < 2 * m; ++i)
            b_sub = _subborrow_u64(b_sub, z1[i], z0[i], &z1[i]);
        for (; i < 2 * m2 + 2; ++i)
            b_sub = _subborrow_u64(b_sub, z1[i], 0, &z1[i]);

        b_sub = 0;
        i = 0;
        for (; i + 4 <= 2 * m2; i += 4) {
            b_sub = _subborrow_u64(b_sub, z1[i], z2[i], &z1[i]);
            b_sub = _subborrow_u64(b_sub, z1[i+1], z2[i+1], &z1[i+1]);
            b_sub = _subborrow_u64(b_sub, z1[i+2], z2[i+2], &z1[i+2]);
            b_sub = _subborrow_u64(b_sub, z1[i+3], z2[i+3], &z1[i+3]);
        }
        for (; i < 2 * m2; ++i)
            b_sub = _subborrow_u64(b_sub, z1[i], z2[i], &z1[i]);
        for (; i < 2 * m2 + 2; ++i)
            b_sub = _subborrow_u64(b_sub, z1[i], 0, &z1[i]);

        unsigned char c_add = 0;
        i = 0;
        for (; i + 4 <= 2 * m2 + 2; i += 4) {
            if (m + i + 3 < 2 * n) {
                c_add = _addcarry_u64(c_add, res[m + i], z1[i], &res[m + i]);
                c_add = _addcarry_u64(c_add, res[m + i+1], z1[i+1], &res[m + i+1]);
                c_add = _addcarry_u64(c_add, res[m + i+2], z1[i+2], &res[m + i+2]);
                c_add = _addcarry_u64(c_add, res[m + i+3], z1[i+3], &res[m + i+3]);
            } else {
                if (m + i < 2 * n) c_add = _addcarry_u64(c_add, res[m + i], z1[i], &res[m + i]);
                if (m + i+1 < 2 * n) c_add = _addcarry_u64(c_add, res[m + i+1], z1[i+1], &res[m + i+1]);
                if (m + i+2 < 2 * n) c_add = _addcarry_u64(c_add, res[m + i+2], z1[i+2], &res[m + i+2]);
                if (m + i+3 < 2 * n) c_add = _addcarry_u64(c_add, res[m + i+3], z1[i+3], &res[m + i+3]);
            }
        }
        for (; i < 2 * m2 + 2; ++i) {
            if (m + i < 2 * n) {
                c_add = _addcarry_u64(c_add, res[m + i], z1[i], &res[m + i]);
            }
        }
    }

    // Truncated Schoolbook Multiplication
    inline void mul_schoolbook_truncated(u64 *res, const u64 *a, const u64 *b,
                                         size_t n) noexcept {
        std::fill_n(res, n, 0);
        for (size_t i = 0; i < n; ++i) {
            u64 y = a[i];
            if (y == 0)
                continue;
            u128 carry = 0;
            for (size_t j = 0; j < n - i; ++j) {
                u128 temp = (u128)b[j] * y + res[i + j] + carry;
                res[i + j] = (u64)temp;
                carry = temp >> 64;
            }
        }
    }

    // Templated Truncated Multiplication Wrapper
    template <size_t N>
    __attribute__((always_inline))
    inline void mul_truncated_fixed(u64 *res, const u64 *a, const u64 *b, u64 *tmp) noexcept {
        if constexpr (N <= 32) {
            if constexpr (N == 4) {
                mul_schoolbook_truncated_fixed_4x4(res, a, b);
            } else {
                mul_schoolbook_truncated_fixed<N>(res, a, b);
            }
            return;
        }

        u64 *full_res = tmp;
        u64 *scratch = full_res + 2 * N;

        mul_karatsuba_full(full_res, a, b, N, scratch);

        for (size_t i = 0; i < N; ++i)
            res[i] = full_res[i];
    }

    // Recursive Full Squaring
    inline void square_karatsuba_full(u64 *res, const u64 *a, size_t n, u64 *tmp) noexcept {
        if (n <= 24) {
            square_schoolbook_full(res, a, n);
            return;
        }

        size_t m = n / 2;
        size_t m2 = n - m;

        const u64 *a0 = a;
        const u64 *a1 = a + m;

        u64 *z0 = res;
        u64 *z2 = res + 2*m;

        square_karatsuba_full(z0, a0, m, tmp);
        square_karatsuba_full(z2, a1, m2, tmp + 2*m);

        u64 *mid = tmp;
        u64 *scratch = mid + 2*m2;

        if (m == m2) {
            mul_karatsuba_full(mid, a0, a1, m, scratch);
        } else {
            u64 *a0_padded = scratch;
            std::copy_n(a0, m, a0_padded);
            std::fill_n(a0_padded + m, m2 - m, 0);
            mul_karatsuba_full(mid, a0_padded, a1, m2, scratch + m2);
        }

        u64 shift_carry = 0;
        for (size_t i = 0; i < 2 * m2; ++i) {
            u64 next = mid[i] >> 63;
            mid[i] = (mid[i] << 1) | shift_carry;
            shift_carry = next;
        }

        unsigned char c = 0;
        for (size_t i = 0; i < 2 * m2; ++i) {
            c = _addcarry_u64(c, res[m + i], mid[i], &res[m + i]);
        }
        size_t k = m + 2 * m2;
        u64 extra = shift_carry;
        c = _addcarry_u64(c, res[k], extra, &res[k]);
        k++;
        while (c && k < 2 * n) {
            c = _addcarry_u64(c, res[k], 0, &res[k]);
            k++;
        }
    }

    // Truncated Multiplication Wrapper
    inline void mul_truncated(u64 *res, const u64 *a, const u64 *b, size_t n,
                              u64 *tmp) noexcept {
        if (n <= 32) [[likely]] {
            mul_schoolbook_truncated(res, a, b, n);
            return;
        }

        u64 *full_res = tmp;
        u64 *scratch = full_res + 2 * n;

        mul_karatsuba_full(full_res, a, b, n, scratch);

        for (size_t i = 0; i < n; ++i)
            res[i] = full_res[i];
    }

    // Template for fixed-size truncated squaring
    template <size_t N>
    __attribute__((always_inline))
    inline void square_truncated_fixed(u64 *res, const u64 *a, u64 *tmp) noexcept {
        if constexpr (N <= 32) {
            square_schoolbook_truncated_fixed<N>(res, a);
            return;
        }

        constexpr size_t M = N / 2;
        constexpr size_t M2 = N - M;

        u64 *z0 = tmp;
        u64 *scratch = z0 + 2 * M;

        square_karatsuba_full(z0, a, M, scratch);

        for(size_t i=0; i<2*M; ++i) res[i] = z0[i];
        if constexpr (N > 2*M) res[2*M] = 0;

        u64 *mid = scratch;
        u64 *a0_padded = mid + M2;
        u64 *next_scratch = a0_padded + M2;

        std::copy_n(a, M, a0_padded);
        if (M < M2) std::fill_n(a0_padded + M, M2 - M, 0);

        mul_truncated(mid, a0_padded, a + M, M2, next_scratch);

        u64 shift_carry = 0;
        unsigned char c = 0;
        for (size_t i = 0; i < M2; ++i) {
            u64 val = mid[i];
            u64 next = val >> 63;
            val = (val << 1) | shift_carry;
            shift_carry = next;

            c = _addcarry_u64(c, res[M + i], val, &res[M + i]);
        }
    }

    } // namespace Multiplication
