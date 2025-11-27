#pragma once

#include <algorithm>
#include <cstdint>
#include <cstring>  // For memset
#include <immintrin.h>
#include <vector>

namespace Karatsuba {

using u64 = unsigned long long;
using u128 = unsigned __int128;

// Base case: O(N^2) full multiplication using u128 arithmetic
// Computes res[0..2n-1] = a[0..n-1] * b[0..n-1]
//
// @param res Output buffer of size 2*n
// @param a Input array A of size n
// @param b Input array B of size n
// @param n Number of limbs
inline void mul_schoolbook_full(u64 *res, const u64 *a, const u64 *b,
                                size_t n) noexcept {
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
  for (size_t i = 0; i < N; ++i) {
    u64 y = a[i];
    if (y == 0) continue;
    u128 carry = 0;
    for (size_t j = 0; j < N; ++j) {
      u128 temp = (u128)b[j] * y + res[i + j] + carry;
      res[i + j] = (u64)temp;
      carry = temp >> 64;
    }
    size_t k = i + N;
    while (carry) {
      u128 temp = (u128)res[k] + carry;
      res[k] = (u64)temp;
      carry = temp >> 64;
      k++;
    }
  }
}

// Template for fixed-size truncated multiplication
template <size_t N>
__attribute__((always_inline))
inline void mul_schoolbook_truncated_fixed(u64 *res, const u64 *a, const u64 *b) {
  std::fill_n(res, N, 0);
  for (size_t i = 0; i < N; ++i) {
    u64 y = a[i];
    if (y == 0) continue;
    u128 carry = 0;
    for (size_t j = 0; j < N - i; ++j) {
      u128 temp = (u128)b[j] * y + res[i + j] + carry;
      res[i + j] = (u64)temp;
      carry = temp >> 64;
    }
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
//
// @tparam N Size of the multiplication (must be fixed at compile time)
// @param res Output buffer of size 2*N
// @param a Input array A of size N
// @param b Input array B of size N
// @param tmp Temporary buffer of size ~4*N
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
// Computes res[0..2n-1] = a[0..n-1] * b[0..n-1]
// Requires temp buffer of size ~ 4n
//
// @param res Output buffer of size 2*n
// @param a Input array A of size n
// @param b Input array B of size n
// @param n Number of limbs (runtime variable)
// @param tmp Temporary buffer
inline void mul_karatsuba_full(u64 *res, const u64 *a, const u64 *b, size_t n,
                               u64 *tmp) noexcept {
  if (n <= 24) [[likely]] {
    mul_schoolbook_full(res, a, b, n);
    return;
  }

  size_t m = n / 2;
  size_t m2 = n - m;

  // Pointers for parts
  const u64 *a0 = a;
  const u64 *a1 = a + m;
  const u64 *b0 = b;
  const u64 *b1 = b + m;

  // Result parts
  u64 *z0 = res;         // Low part (2m)
  u64 *z2 = res + 2 * m; // High part (2m2)

  // Temp parts
  u64 *sum_a = tmp;            // m2 + 1
  u64 *sum_b = sum_a + m2 + 1; // m2 + 1
  u64 *z1 = sum_b + m2 + 1;    // 2(m2+1)

  // 1. Compute Z0 = A0 * B0
  mul_karatsuba_full(z0, a0, b0, m, z1 + 2 * (m2 + 1));

  // 2. Compute Z2 = A1 * B1
  mul_karatsuba_full(z2, a1, b1, m2, z1 + 2 * (m2 + 1));

  // 3. Compute Z1 = (A0 + A1) * (B0 + B1)
  // sum_a = A0 + A1
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

  // sum_b = B0 + B1
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

  // Z1 = sum_a * sum_b
  mul_karatsuba_full(z1, sum_a, sum_b, m2 + 1, z1 + 2 * (m2 + 1));

  // 4. Z1 = Z1 - Z0 - Z2
  // Subtract Z0
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

  // Subtract Z2
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

  // 5. Add Z1 to result at offset m
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
  // Propagate carry if needed (unlikely to overflow 2n if logic is correct, but
  // safe to ignore for fixed size buffers if we assume it fits)
}

// Truncated Schoolbook Multiplication
// Computes res[0..n-1] = (a[0..n-1] * b[0..n-1]) mod 2^N
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
    mul_schoolbook_truncated_fixed<N>(res, a, b);
    return;
  }

  u64 *full_res = tmp;
  u64 *scratch = full_res + 2 * N;

  mul_karatsuba_full(full_res, a, b, N, scratch);

  for (size_t i = 0; i < N; ++i)
    res[i] = full_res[i];
}

// Recursive Full Squaring
// Computes res = a^2
// res size 2n, a size n
inline void square_karatsuba_full(u64 *res, const u64 *a, size_t n, u64 *tmp) noexcept {
  if (n <= 24) {
    square_schoolbook_full(res, a, n);
    return;
  }

  size_t m = n / 2;
  size_t m2 = n - m;

  const u64 *a0 = a;
  const u64 *a1 = a + m;

  u64 *z0 = res;       // A0^2, size 2m
  u64 *z2 = res + 2*m; // A1^2, size 2m2

  // Recurse Squares
  square_karatsuba_full(z0, a0, m, tmp);
  square_karatsuba_full(z2, a1, m2, tmp + 2*m); // Reuse tmp? Need to check depth.
  // Standard Karatsuba scratch is 2n.
  // Here we need scratch for recursive calls.
  // tmp + 2*m is safe for z2 call if z0 call is done.
  
  // Compute Middle Term: 2 * A0 * A1
  // We use mul_karatsuba_full(a0, a1).
  // Handle size mismatch if m != m2.
  u64 *mid = tmp; // Size 2*m2
  u64 *scratch = mid + 2*m2;
  
  if (m == m2) {
      mul_karatsuba_full(mid, a0, a1, m, scratch);
  } else {
      // Pad a0 to size m2
      u64 *a0_padded = scratch;
      std::copy_n(a0, m, a0_padded);
      std::fill_n(a0_padded + m, m2 - m, 0);
      mul_karatsuba_full(mid, a0_padded, a1, m2, scratch + m2);
  }

  // Add 2 * mid to result
  // res = z0 + z2*2^2m + 2*mid*2^m
  // z0 is at res[0], z2 is at res[2m].
  // We need to add 2*mid starting at res[m].
  
  // Shift mid left by 1
  u64 shift_carry = 0;
  for (size_t i = 0; i < 2 * m2; ++i) {
    u64 next = mid[i] >> 63;
    mid[i] = (mid[i] << 1) | shift_carry;
    shift_carry = next;
  }
  // Add shift_carry to next limb?
  // 2*A0*A1 might have 2*m2 + 1 limbs.
  // We handle the carry during addition.
  
  unsigned char c = 0;
  for (size_t i = 0; i < 2 * m2; ++i) {
      c = _addcarry_u64(c, res[m + i], mid[i], &res[m + i]);
  }
  // Propagate carry and shift_carry
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
// Computes res[0..n-1] = (a[0..n-1] * b[0..n-1]) mod 2^N
// Requires temp buffer of size ~ 8n
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

  // Karatsuba Squaring for Truncated Result
  // N = 2M.
  // res[0..N-1] = (A0^2) + (2*A0*A1 * 2^M)
  // We need A0^2 (full 2M) and lower M limbs of 2*A0*A1.
  
  constexpr size_t M = N / 2;
  constexpr size_t M2 = N - M; // Should be M if N is even, but let's be generic
  
  u64 *z0 = tmp; // Size 2M
  u64 *scratch = z0 + 2 * M;
  
  // 1. Full Square A0
  square_karatsuba_full(z0, a, M, scratch);
  
  // Copy z0 to res
  for(size_t i=0; i<2*M; ++i) res[i] = z0[i];
  if constexpr (N > 2*M) res[2*M] = 0;
  
  // 2. Truncated Mul A0 * A1
  u64 *mid = scratch; // Reuse scratch
  u64 *a0_padded = mid + M2; // Scratch space
  u64 *next_scratch = a0_padded + M2;
  
  std::copy_n(a, M, a0_padded);
  if (M < M2) std::fill_n(a0_padded + M, M2 - M, 0);
  
  mul_truncated(mid, a0_padded, a + M, M2, next_scratch);
  
  // 3. Add 2 * mid to res starting at M
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

} // namespace Karatsuba
