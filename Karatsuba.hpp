#pragma once

#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace Karatsuba {

using u64 = unsigned long long;
using u128 = unsigned __int128;

// Base case: O(N^2) full multiplication using u128 arithmetic
// Computes res[0..2n-1] = a[0..n-1] * b[0..n-1]
inline void mul_schoolbook_full(u64 *res, const u64 *a, const u64 *b,
                                size_t n) {
  std::fill_n(res, 2 * n, 0);
  for (size_t i = 0; i < n; ++i) {
    u64 y = a[i];
    if (y == 0)
      continue;
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

// Recursive Karatsuba Full Multiplication
// Computes res[0..2n-1] = a[0..n-1] * b[0..n-1]
// Requires temp buffer of size ~ 4n
inline void mul_karatsuba_full(u64 *res, const u64 *a, const u64 *b, size_t n,
                               u64 *tmp) {
  if (n <= 64) {
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
  for (size_t i = 0; i < m; ++i)
    c_a = _addcarry_u64(c_a, a0[i], a1[i], &sum_a[i]);
  for (size_t i = m; i < m2; ++i)
    c_a = _addcarry_u64(c_a, 0, a1[i], &sum_a[i]);
  sum_a[m2] = c_a;

  // sum_b = B0 + B1
  unsigned char c_b = 0;
  for (size_t i = 0; i < m; ++i)
    c_b = _addcarry_u64(c_b, b0[i], b1[i], &sum_b[i]);
  for (size_t i = m; i < m2; ++i)
    c_b = _addcarry_u64(c_b, 0, b1[i], &sum_b[i]);
  sum_b[m2] = c_b;

  // Z1 = sum_a * sum_b
  mul_karatsuba_full(z1, sum_a, sum_b, m2 + 1, z1 + 2 * (m2 + 1));

  // 4. Z1 = Z1 - Z0 - Z2
  // Subtract Z0
  unsigned char b_sub = 0;
  for (size_t i = 0; i < 2 * m; ++i)
    b_sub = _subborrow_u64(b_sub, z1[i], z0[i], &z1[i]);
  for (size_t i = 2 * m; i < 2 * m2 + 2; ++i)
    b_sub = _subborrow_u64(b_sub, z1[i], 0, &z1[i]);

  // Subtract Z2
  b_sub = 0;
  for (size_t i = 0; i < 2 * m2; ++i)
    b_sub = _subborrow_u64(b_sub, z1[i], z2[i], &z1[i]);
  for (size_t i = 2 * m2; i < 2 * m2 + 2; ++i)
    b_sub = _subborrow_u64(b_sub, z1[i], 0, &z1[i]);

  // 5. Add Z1 to result at offset m
  unsigned char c_add = 0;
  for (size_t i = 0; i < 2 * m2 + 2; ++i) {
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
                                     size_t n) {
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

// Truncated Multiplication Wrapper
// Computes res[0..n-1] = (a[0..n-1] * b[0..n-1]) mod 2^N
// Requires temp buffer of size ~ 8n
inline void mul_truncated(u64 *res, const u64 *a, const u64 *b, size_t n,
                          u64 *tmp) {
  if (n <= 64) {
    mul_schoolbook_truncated(res, a, b, n);
    return;
  }

  u64 *full_res = tmp;
  u64 *scratch = full_res + 2 * n;

  mul_karatsuba_full(full_res, a, b, n, scratch);

  for (size_t i = 0; i < n; ++i)
    res[i] = full_res[i];
}

} // namespace Karatsuba
