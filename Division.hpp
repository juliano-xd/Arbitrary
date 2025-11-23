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

// Helper: Add N-block numbers: res = a + b
// Returns carry
inline u64 add_n(u64 *res, const u64 *a, const u64 *b, size_t n) {
  unsigned char carry = 0;
  for (size_t i = 0; i < n; ++i) {
    carry = _addcarry_u64(carry, a[i], b[i], &res[i]);
  }
  return carry;
}

// Helper: Subtract N-block numbers: res = a - b
// Returns borrow
inline u64 sub_n(u64 *res, const u64 *a, const u64 *b, size_t n) {
  unsigned char borrow = 0;
  for (size_t i = 0; i < n; ++i) {
    borrow = _subborrow_u64(borrow, a[i], b[i], &res[i]);
  }
  return borrow;
}

// Helper: Shift Left N-block number by bits (0-63)
// res = a << shift
// Returns carry out
inline u64 shl_n(u64 *res, const u64 *a, size_t n, int shift) {
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
inline u64 shr_n(u64 *res, const u64 *a, size_t n, int shift) {
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

// Base case: 2N / N using hardware division (if N is small) or simple
// schoolbook For N=1 (64 bits), we have 128 / 64 bit division. x86_64 'div'
// instruction does 128/64 -> 64 quotient, 64 remainder.
inline void div_2n_1n_base(u64 *q, u64 *r, const u64 *a, const u64 *b,
                           size_t n) {
  // Simple schoolbook division for small N
  // We assume q and r are pre-allocated (q: n, r: n)
  // a is 2n, b is n.
  // This is a placeholder. For high performance, we need a robust base case.
  // Let's use a simple bit-wise long division or Knuth's Algorithm D for the
  // base case. Since we are inside Burnikel-Ziegler, we expect N to be small
  // here (e.g. <= 32). We can just reuse the logic from UInt::div_mod but
  // adapted for pointers.

  // Implementing Knuth's Algorithm D on pointers:
  // Normalize first.
  // We assume 'a' and 'b' are normalized? BZ usually requires normalization.
  // Let's assume caller handles normalization or we do it here.
  // For simplicity in this step, let's assume we call back to a
  // "schoolbook_div" helper.

  // TODO: Implement proper base case.
}

// Placeholder for Knuth's Algorithm D base case
// This function needs to be properly implemented for the BZ algorithm to work.
void div_knuth_base(u64 *q, u64 *r, const u64 *a, size_t an, const u64 *b,
                    size_t bn) {
  // This is a placeholder. A proper implementation of Knuth's Algorithm D
  // or a similar schoolbook division for arbitrary precision numbers
  // should go here.
  // For now, we'll just zero out q and r to avoid uninitialized memory access.
  std::fill_n(q, an - bn + 1, 0); // Quotient size is an - bn + 1
  std::fill_n(r, bn, 0);          // Remainder size is bn
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
      n <= 64) { // Base case threshold, e.g., 64 blocks (4096 bits)
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
