#pragma once

#include "Karatsuba.hpp"
#include <algorithm>
#include <cstdint>
#include <immintrin.h>
#include <vector>

namespace Division {

using u64 = unsigned long long;
using u128 = unsigned __int128;

// Helper: Compare two N-block numbers
// Returns 1 if a > b, -1 if a < b, 0 if a == b
inline int cmp_n(const u64 *a, const u64 *b, size_t n) {
  for (size_t i = n; i-- > 0;) {
    if (a[i] > b[i])
      return 1;
    if (a[i] < b[i])
      return -1;
  }
  return 0;
}

// Helper: Add N-block numbers: res = a + b. Returns carry
inline u64 add_n(u64 *__restrict__ res, const u64 *__restrict__ a, const u64 *__restrict__ b, size_t n) {
  unsigned char carry = 0;
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    carry = _addcarry_u64(carry, a[i], b[i], &res[i]);
    carry = _addcarry_u64(carry, a[i+1], b[i+1], &res[i+1]);
    carry = _addcarry_u64(carry, a[i+2], b[i+2], &res[i+2]);
    carry = _addcarry_u64(carry, a[i+3], b[i+3], &res[i+3]);
  }
  for (; i < n; ++i) {
    carry = _addcarry_u64(carry, a[i], b[i], &res[i]);
  }
  return carry;
}

// Helper: Subtract N-block numbers: res = a - b
// Returns borrow
inline u64 sub_n(u64 *__restrict__ res, const u64 *__restrict__ a, const u64 *__restrict__ b, size_t n) {
  unsigned char borrow = 0;
  size_t i = 0;
  for (; i + 4 <= n; i += 4) {
    borrow = _subborrow_u64(borrow, a[i], b[i], &res[i]);
    borrow = _subborrow_u64(borrow, a[i+1], b[i+1], &res[i+1]);
    borrow = _subborrow_u64(borrow, a[i+2], b[i+2], &res[i+2]);
    borrow = _subborrow_u64(borrow, a[i+3], b[i+3], &res[i+3]);
  }
  for (; i < n; ++i) {
    borrow = _subborrow_u64(borrow, a[i], b[i], &res[i]);
  }
  return borrow;
}

// Helper: Shift Left N-block number by bits (0-63)
// res = a << shift
// Returns carry out
inline u64 shl_n(u64 *__restrict__ res, const u64 *__restrict__ a, size_t n, int shift) {
  if (shift == 0) {
    if (res != a)
      std::copy_n(a, n, res);
    return 0;
  }
  u64 carry = 0;
  for (size_t i = 0; i < n; ++i) {
    u64 next_carry = a[i] >> (64 - shift);
    res[i] = (a[i] << shift) | carry;
    carry = next_carry;
  }
  return carry;
}

// Helper: Shift Right N-block number by bits (0-63)
// res = a >> shift
// Returns carry out (bits shifted out)
inline u64 shr_n(u64 *__restrict__ res, const u64 *__restrict__ a, size_t n, int shift) {
  if (shift == 0) {
    if (res != a)
      std::copy_n(a, n, res);
    return 0;
  }
  u64 carry = 0;
  for (size_t i = n; i-- > 0;) {
    u64 next_carry = a[i] << (64 - shift);
    res[i] = (a[i] >> shift) | carry;
    carry = next_carry;
  }
  return carry;
}



// Forward declaration
void div_recursive(u64 *q, u64 *r, const u64 *a, const u64 *b, size_t n,
                   u64 *tmp);

// Base case: 2N / N using Knuth's Algorithm D
// This is used when N is small enough (e.g. <= 64 blocks)
inline void div_knuth_base(u64 *__restrict__ q, u64 *__restrict__ r, const u64 *__restrict__ a, size_t an, const u64 *__restrict__ b,
                    size_t bn) {
  // Knuth's Algorithm D (The Art of Computer Programming, Vol 2, 4.3.1)
  // Inputs: a (dividend) of size an, b (divisor) of size bn
  // Outputs: q (quotient) of size an-bn+1, r (remainder) of size bn

  if (bn == 0) {
      // Division by zero should be handled by caller or throw
      return; 
  }
  if (an < bn) {
      // Dividend smaller than divisor
      std::fill_n(q, an - bn + 1, 0);
      std::copy_n(a, an, r);
      std::fill_n(r + an, bn - an, 0);
      return;
  }

  // D1: Normalize - shift v left so its MSB is >= 2^63
  const int shift = __builtin_clzll(b[bn - 1]);
  
  // We need a temporary buffer for normalized u and v
  // Size needed: u: an+1, v: bn
  // We can use a small stack buffer if sizes are small, or heap/scratch.
  // Given this is "base", sizes are likely small (<= 64).
  // But to be safe and avoid stack overflow on larger sizes, we should use a scratch buffer if possible.
  // However, the signature doesn't provide scratch for this leaf function efficiently.
  // Let's assume for "base" usage, we can allocate on stack or use std::vector (slow).
  // Better: use a fixed size stack buffer since we know the limit for "base" is small (e.g. 64).
  
  constexpr size_t MAX_BASE_BLOCKS = 600; // Safe upper bound for base case (2*N <= 510)
  u64 u_norm[MAX_BASE_BLOCKS + 1];
  u64 v_norm_storage[MAX_BASE_BLOCKS];
  const u64* v_ptr;

  // Normalize divisor v
  if (shift > 0) {
      shl_n(v_norm_storage, b, bn, shift);
      v_ptr = v_norm_storage;
  } else {
      // Optimization: Avoid copy if no shift
      v_ptr = b;
  }

  // Normalize dividend u
  // u_norm needs to be an + 1 size to handle potential carry
  if (shift > 0) {
      u_norm[an] = shl_n(u_norm, a, an, shift);
  } else {
      std::copy_n(a, an, u_norm);
      u_norm[an] = 0;
  }

  const u64 v_high = v_ptr[bn - 1];
  const u64 v_next = (bn > 1) ? v_ptr[bn - 2] : 0;

  // D2-D7: Main loop - compute quotient digits
  // Loop from m = an - bn down to 0
  for (int j = an - bn; j >= 0; --j) {
      // D3: Calculate trial quotient digit
      u64 u_high = u_norm[j + bn];
      u64 u_mid = u_norm[j + bn - 1];
      u64 q_hat, r_hat;

      if (u_high == v_high) [[unlikely]] {
          q_hat = ~0ULL;
          r_hat = u_mid + v_high;
          if (r_hat < v_high) { 
             // Overflowed 64-bit
          } 
      } else {
          u128 dividend = ((u128)u_high << 64) | u_mid;
          q_hat = dividend / v_high;
          r_hat = dividend % v_high;
      }

      // D3 continued: Refine q_hat
      while (true) {
           u128 lhs = (u128)q_hat * v_next;
           u128 rhs = ((u128)r_hat << 64) | u_norm[j + bn - 2];
           if (lhs <= rhs) [[likely]] break;
           
           q_hat--;
           r_hat += v_high;
           if (r_hat < v_high) [[unlikely]] break; 
      }

      // D4: Multiply and subtract
      u64 mult_carry = 0;
      unsigned char sub_borrow = 0;
      
      size_t i = 0;
      for (; i + 4 <= bn; i += 4) {
          u64 hi, lo;
          
          // Unroll 0
          lo = _mulx_u64(q_hat, v_ptr[i], &hi);
          unsigned char c = _addcarry_u64(0, lo, mult_carry, &lo);
          mult_carry = hi + c;
          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i], lo, &u_norm[j + i]);

          // Unroll 1
          lo = _mulx_u64(q_hat, v_ptr[i+1], &hi);
          c = _addcarry_u64(0, lo, mult_carry, &lo);
          mult_carry = hi + c;
          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i+1], lo, &u_norm[j + i+1]);

          // Unroll 2
          lo = _mulx_u64(q_hat, v_ptr[i+2], &hi);
          c = _addcarry_u64(0, lo, mult_carry, &lo);
          mult_carry = hi + c;
          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i+2], lo, &u_norm[j + i+2]);

          // Unroll 3
          lo = _mulx_u64(q_hat, v_ptr[i+3], &hi);
          c = _addcarry_u64(0, lo, mult_carry, &lo);
          mult_carry = hi + c;
          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i+3], lo, &u_norm[j + i+3]);
      }

      for (; i < bn; ++i) {
          u64 hi, lo;
          lo = _mulx_u64(q_hat, v_ptr[i], &hi);
          
          unsigned char c = _addcarry_u64(0, lo, mult_carry, &lo);
          mult_carry = hi + c;

          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i], lo, &u_norm[j + i]);
      }
      
      u64 old_high = u_norm[j + bn];
      u64 diff = old_high - mult_carry;
      bool borrow1 = (old_high < mult_carry);
      u_norm[j + bn] = diff - sub_borrow;
      bool borrow2 = (diff < sub_borrow);
      sub_borrow = borrow1 || borrow2;

      // D5 & D6: Test and add back
      if (sub_borrow) [[unlikely]] {
          q_hat--;
          u64 add_carry = 0;
          // Use add_n helper? No, it's mixed with u_norm offset.
          // Just unroll manually or use loop.
          // This path is rare (prob ~ 2^-64), so keep it simple to save code size?
          // Or optimize it too? It's rare, so optimization matters less.
          for (size_t k = 0; k < bn; ++k) {
              add_carry = _addcarry_u64(add_carry, u_norm[j + k], v_ptr[k], &u_norm[j + k]);
          }
          u_norm[j + bn] += add_carry;
      }

      q[j] = q_hat;
  }

  // D8: Unnormalize remainder
  if (shift > 0) {
      shr_n(r, u_norm, bn, shift);
  } else {
      std::copy_n(u_norm, bn, r);
  }
}


// Helper for Knuth's Algorithm D
// Normalizes u and v, then performs division
// q: quotient (size u_len - v_len + 1)
// r: remainder (size v_len)
// u: dividend (size u_len)
// v: divisor (size v_len)
template <size_t MaxBlocks = 600>
inline void div_knuth_impl(u64 *q, u64 *r, const u64 *u, int u_len,
                           const u64 *v, int v_len) {
  // D1: Normalize
  int shift = __builtin_clzll(v[v_len - 1]);
  int an = u_len;
  int bn = v_len;
  
  // Stack buffers
  u64 u_norm[MaxBlocks + 1];
  u64 v_norm_storage[MaxBlocks];
  const u64* v_ptr;

  // Normalize divisor v
  if (shift > 0) {
      shl_n(v_norm_storage, v, bn, shift);
      v_ptr = v_norm_storage;
  } else {
      // Optimization: Avoid copy if no shift
      v_ptr = v;
  }

  // Normalize dividend u
  // u_norm needs to be an + 1 size to handle potential carry
  if (shift > 0) {
      u_norm[an] = shl_n(u_norm, u, an, shift);
  } else {
      std::copy_n(u, an, u_norm);
      u_norm[an] = 0;
  }

  const u64 v_high = v_ptr[bn - 1];
  const u64 v_next = (bn > 1) ? v_ptr[bn - 2] : 0;

  // D2-D7: Main loop - compute quotient digits
  // Loop from m = an - bn down to 0
  for (int j = an - bn; j >= 0; --j) {
      // D3: Calculate trial quotient digit
      u64 u_high = u_norm[j + bn];
      u64 u_mid = u_norm[j + bn - 1];
      u64 q_hat, r_hat;

      if (u_high == v_high) [[unlikely]] {
          q_hat = ~0ULL;
          r_hat = u_mid + v_high;
          if (r_hat < v_high) { 
             // Overflowed 64-bit
          } 
      } else {
          u128 dividend = ((u128)u_high << 64) | u_mid;
          q_hat = dividend / v_high;
          r_hat = dividend % v_high;
      }

      // D3 continued: Refine q_hat
      while (true) {
           u128 lhs = (u128)q_hat * v_next;
           u128 rhs = ((u128)r_hat << 64) | u_norm[j + bn - 2];
           if (lhs <= rhs) [[likely]] break;
           
           q_hat--;
           r_hat += v_high;
           if (r_hat < v_high) [[unlikely]] break; 
      }

      // D4: Multiply and subtract
      u64 mult_carry = 0;
      unsigned char sub_borrow = 0;
      
      size_t i = 0;
      for (; i + 4 <= bn; i += 4) {
          u64 hi, lo;
          
          // Unrolled multiply
          u128 prod0 = (u128)q_hat * v_ptr[i];
          u128 prod1 = (u128)q_hat * v_ptr[i+1];
          u128 prod2 = (u128)q_hat * v_ptr[i+2];
          u128 prod3 = (u128)q_hat * v_ptr[i+3];
          
          // Add carry to prod
          prod0 += mult_carry;
          lo = (u64)prod0; mult_carry = prod0 >> 64;
          
          prod1 += mult_carry;
          u64 lo1 = (u64)prod1; mult_carry = prod1 >> 64;
          
          prod2 += mult_carry;
          u64 lo2 = (u64)prod2; mult_carry = prod2 >> 64;
          
          prod3 += mult_carry;
          u64 lo3 = (u64)prod3; mult_carry = prod3 >> 64;
          
          // Subtract from u_norm
          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i], lo, &u_norm[j + i]);
          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i+1], lo1, &u_norm[j + i+1]);
          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i+2], lo2, &u_norm[j + i+2]);
          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i+3], lo3, &u_norm[j + i+3]);
      }
      
      for (; i < bn; ++i) {
          u128 prod = (u128)q_hat * v_ptr[i] + mult_carry;
          u64 lo = (u64)prod;
          mult_carry = prod >> 64;
          sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + i], lo, &u_norm[j + i]);
      }
      
      // Handle last borrow/carry
      sub_borrow = _subborrow_u64(sub_borrow, u_norm[j + bn], mult_carry, &u_norm[j + bn]);

      // D6: Add back if negative
      if (sub_borrow) {
          q_hat--;
          unsigned char add_carry = 0;
          for (size_t k = 0; k < bn; ++k) {
              add_carry = _addcarry_u64(add_carry, u_norm[j + k], v_ptr[k], &u_norm[j + k]);
          }
          u_norm[j + bn] += add_carry;
      }

      q[j] = q_hat;
  }

  // D8: Unnormalize remainder
  if (shift > 0) {
      shr_n(r, u_norm, bn, shift);
  } else {
      std::copy_n(u_norm, bn, r);
  }
}

// Legacy wrapper
inline void div_knuth_base(u64 *q, u64 *r, const u64 *u, int u_len,
                           const u64 *v, int v_len) {
    div_knuth_impl<600>(q, r, u, u_len, v, v_len);
}

inline void div_2n_1n_base(u64 *q, u64 *r, const u64 *a, const u64 *b,
                           size_t n) {
    div_knuth_base(q, r, a, 2 * n, b, n);
}

// Recursive Division: 3m / 2m
// Inputs: A (3m blocks), B (2m blocks)
// Outputs: Q (m blocks), R (2m blocks)
// Temp: Scratch space
void div_3n_2n(u64 *q, u64 *r, const u64 *a, const u64 *b, size_t m, u64 *tmp) {
  const u64 *a1 = a + 2 * m; // A[2m..3m-1]
  const u64 *a2 = a + m;     // A[m..2m-1]
  const u64 *a3 = a;         // A[0..m-1]
  const u64 *b1 = b + m;     // B[m..2m-1]
  const u64 *b2 = b;         // B[0..m-1]

  // 1. q_hat = [A1, A2] / B1
  // We need to construct [A1, A2] (size 2m)
  // div_recursive expects contiguous A.
  // Let's copy [A1, A2] to tmp.
  u64 *a12 = tmp; // tmp[0..2m-1]
  std::copy_n(a2, m, a12);
  std::copy_n(a1, m, a12 + m);

  u64 *q_hat = tmp + 2 * m; // tmp[2m..3m-1], size m
  u64 *r_hat = tmp + 3 * m; // tmp[3m..4m-1], size m

  // Recursive call: 2m / m
  // The tmp for this call starts at tmp + 4*m
  div_recursive(q_hat, r_hat, a12, b1, m, tmp + 4 * m);

  // 2. D = q_hat * B2
  // We need to multiply q_hat (m) * b2 (m) -> 2m blocks
  u64 *d = tmp + 4 * m; // tmp[4m..6m-1], size 2m
  Karatsuba::mul_karatsuba_full(d, q_hat, b2, m, tmp + 6 * m);

  // 3. R = ([R_hat, A3] - D)
  // Construct [R_hat, A3] in the output 'r' buffer.
  // R is output buffer (2m).
  // R = (R_hat << m) + A3 - D
  // Copy A3 to r[0..m-1]
  std::copy_n(a3, m, r);
  // Copy R_hat to r[m..2m-1]
  std::copy_n(r_hat, m, r + m);

  // Subtract D from R
  u64 borrow = sub_n(r, r, d, 2 * m);

  // 4. While R < 0 (borrow), Q--, R += B
  // If borrow is set, it means the result was effectively negative.
  // We need to correct by decrementing q_hat and adding B to R.
  while (borrow) {
    // q_hat--
    // Decrement q_hat (size m)
    unsigned char c = 1; // Initial borrow for decrement
    for (size_t i = 0; i < m; ++i) {
      c = _subborrow_u64(c, q_hat[i], 0, &q_hat[i]);
    }

    // R += B
    // B is 2m blocks.
    unsigned char carry = 0;
    for (size_t i = 0; i < 2 * m; ++i) {
      carry = _addcarry_u64(carry, r[i], b[i], &r[i]);
    }
    // If adding B resulted in a carry, it means R is now positive.
    // So the borrow from the subtraction is effectively cancelled.
    if (carry)
      borrow = 0;
  }

  // Output Q
  std::copy_n(q_hat, m, q);
}

// Recursive Division: 2n / n
// Inputs: A (2n blocks), B (n blocks)
// Outputs: Q (n blocks), R (n blocks)
// Temp: Scratch space
void div_2n_1n(u64 *q, u64 *r, const u64 *a, const u64 *b, size_t n, u64 *tmp) {
  if (n % 2 != 0 ||
      n <= 32) { // Base case threshold: 32 blocks
    div_knuth_base(q, r, a, 2 * n, b, n);
    return;
  }

  size_t m = n / 2; // n = 2m

  // Step 1: Divide top 3m blocks of A by B (2m blocks)
  // A_top is A[m..2n-1] = A[m..4m-1], which is 3m blocks.
  const u64 *a_top = a + m; // Points to A[m]

  u64 *q1 =
      q + m; // High part of Q, size m. Q is n blocks, so q+m is the upper half.
  u64 *r1 = tmp; // Remainder of first step, size 2m = n. Uses tmp[0..n-1].

  // Call div_3n_2n(A_top, B)
  // tmp for this call starts at tmp + n (since r1 used n blocks)
  div_3n_2n(q1, r1, a_top, b, m, tmp + n);

  // Step 2: Divide [R1, A_low_low] by B
  // A_low_low is A[0..m-1].
  // Construct input for second step: [R1, A[0..m-1]] -> size 3m.
  // This needs to be contiguous. Use tmp buffer starting after r1.
  u64 *a_bot = tmp + n;              // tmp[n..n+3m-1] = tmp[n..n+2n-1]
  std::copy_n(a, m, a_bot);          // Copy A[0..m-1] to a_bot[0..m-1]
  std::copy_n(r1, 2 * m, a_bot + m); // Copy R1 (2m blocks) to a_bot[m..3m-1]

  u64 *q2 = q; // Low part of Q, size m.
  u64 *r2 = r; // Final remainder, size n (2m).

  // Call div_3n_2n(a_bot, B)
  // tmp for this call starts at tmp + n + 3*m
  div_3n_2n(q2, r2, a_bot, b, m, tmp + n + 3 * m);
}

// Wrapper for recursion
void div_recursive(u64 *q, u64 *r, const u64 *a, const u64 *b, size_t n,
                   u64 *tmp) {
  div_2n_1n(q, r, a, b, n, tmp);
}

} // namespace Division
