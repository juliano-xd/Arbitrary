# Arbitrary — AGENTS.md

C++20 arbitrary-precision unsigned integer library. x86_64 only (BMI2 + ADX intrinsics required).

## Project structure

```
UInt.hpp               # Core type Arbitrary::UInt<N> + all arithmetic operators
Multiplication.hpp     # Multiplication algorithms (schoolbook, Comba, Karatsuba)
Division.hpp           # Division algorithms (Knuth D, recursive, DivisionFixed)
benchmark/             # Google Benchmark suite (CMake project, untracked)
docs/index.html        # Interactive docs (Tailwind + Chart.js)
```

Header-only. No dependencies beyond the C++20 standard library.

## Build & run benchmarks

```bash
cd benchmark && mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
./bench_arithmetic
```

There are **no tests** — only benchmarks. No CI, no linting, no formatting config.

## Using the library

```cpp
#include "UInt.hpp"
using Arbitrary::UInt;
UInt<4> a{"123456789"};   // 4 × 64-bit limbs
UInt<4> b{"987654321"};
auto c = a * b;           // dispatches optimal algorithm by N
```

Compiler flags: `-std=c++20 -O3 -march=native -mbmi2 -madx`

## Architecture notes

- `UInt<N>`: N limbs of `uint64_t`, aligned to 64 bytes. N is `uint8_t`, must be > 0.
- Arithmetic dispatch is template-based on N at compile time. Key paths:
  - **Mul**: N=1 direct, N=2 schoolbook 2×2, N=3 fixed 3×3, N=4–6 schoolbook/Comba, N=8 split-block Toom-style 4×4, N≤16 schoolbook, N>16 Karatsuba
  - **Div**: N=1 single-limb fast, N≤16 `DivisionFixed`, N≤64 Knuth D, larger recursive divide-and-conquer
- Heavy use of `_mulx_u64` (BMI2), `_addcarry_u64`/`_subborrow_u64` (ADX), inline asm with `.rept` unrolling for add/sub
- `operator*` is non-constexpr (calls `mul_runtime()` for asm dispatch). Compile-time
  multiplication available via `mul_consteval()` (public, constexpr schoolbook)

## Limitations

- No `float`/`double` / `int` conversion, no I/O streams, no bit manipulation beyond `& | << >> ~`
- No dynamic allocation at all — limb count is fixed at compile time
- License is **CC BY-NC 4.0** (non-commercial only)
- No `.gitignore` exists; take care not to commit `benchmark/build/` artifacts

## N=3 multiplication optimization

N=3 uses inline asm directly in the dispatch (register input operands, no `"memory"` clobber):

```
3 MULX (k=0,k=1) + 3 IMUL (k=2, non-widening) + 2 ADC + 2 ADD + 3 ADD (tree)
```

Key choices:
- k=1 products use MULX (widening) + ADC carry chain
- k=2 products use IMUL (non-widening) since hi goes to col 3 (truncated)
- k=2 operands preloaded during k=1 to hide load latency
- Tree reduction for k=2 sum: (p0+p1) and (p2+r2) computed in parallel
- Register-only constraints (`"rm"`) avoid the need for `"memory"` clobber
- Output written directly to `bits[]` (no temp buffer, no `copy_n`)

Performance: ~6.8 ns mul, ~4.4 ns square, ~4.0 ns per limb (mul), ~2.2 ns/limb (square)

## N=4 multiplication optimization

N=4 uses inline asm in `mul_runtime()` with `"+m"` constraints on `bits[]`:

```
6 MULX (widening) + 4 IMUL (k=3 non-widening) + ~20 ADD/ADC
```

### Register allocation

- **Accumulators**: r9 (col0), rax (col1), r10 (col2), r8 (col3) — all caller-saved,
  no callee-save push/pop overhead.
- **MULX src**: rdx loaded per-row from `bits[i]` via `"+m"` memory operand.
- **MULX dst**: rcx (lo), rsi (hi) — reused per product within each row.
- **b-operands**: memory (`"m"` constraints) — L1-cached stack loads, effectively same
  latency as register MULX because the load µop is dispatched early (out-of-order).

### Carry chain structure (row-scanning schoolbook)

Each row uses 3 ADD/ADC triples and 1 IMUL+ADD for column 3:

```
add  lo→r_col     ; CF = overflow to next column
adc  hi→r_col+1   ; r_col+1 += hi + CF; CF = overflow to next column
adc  $0→r_col+2   ; r_col+2 += CF (captured carry)
```

The `adc $0` is necessary for all but the first product in row 0 (where all accumulators
are zero, guaranteeing CF=0 throughout the first triple).

### Throughput bottleneck

At 4.386 GHz, **5.63 ns in-place** (`operator*=`) = ~24.7 cycles. The bottleneck is the
ADD/ADC serial carry chain through column registers (~15 cycles dominated by r3 len=9 ops).

| Component | Instructions | Cycles (min) |
|-----------|-------------|--------------|
| 6 MULX    | 6           | 6 (p1: 1/cy) |
| 4 IMUL    | 4           | 2 (p0: 2/cy) |
| ADD/ADC   | ~19         | ~10-15 (serial carry chains) |
| loads/stores | 8        | ~4 (L1 hits) |
| **Total** | **~37**     | **~25**      |

### Key design decision: `"+m"` vs `"=&r"` outputs

- **`"+m"`** (current): asm reads a-values from and writes results to `bits[]` directly.
  BM_MulEq_Pure<4>: **5.59 ns** (fastest in-place). BM_Mul_Pure<4>: 7.77 ns
  (regression: `+m` prevents GCC from redirecting the store to the NRVO return slot).
- **`"=&r"`** (previous): asm outputs to registers, GCC stores via SIMD merge.
  BM_MulEq_Pure<4>: 5.95 ns. BM_Mul_Pure<4>: 5.76 ns (copy elided by NRVO).

In-place (`operator*=`) is the primary path, so `"+m"` is preferred.

### Dead ends tried

| Approach | Result | Why it failed |
|----------|--------|---------------|
| b0-in-register (r12) | 6.05 ns | r12 push/pop overhead > memory-MULX load µop savings |
| Column-scanning Comba | 7.61 ns | Shift/store between columns adds latency |
| Reg-reg MULX (r12-r14) | 7.82 ns | Callee-save push/pop overhead dominated savings |
| Removed first `adc $0` | 5.81 ns | Single-instruction saving within noise |

### Theoretical minimum for N=4 truncated

- 6 MULX × 4c latency, 1/cycle = 6 cy minimum to issue
- ADD/ADC carry chain: ~10-15 cy through register values (fundamentally serial)
- Target 3.5-4.0 ns (~15-17 cy) unreachable without algorithmic change (e.g., SIMD
  with 32-bit limbs, or ADX dual-carries for larger N)
- ADX (ADCX/ADOX) doesn't help N=4 because the carry chain per µop group is too short
  (3-4 ADCs max per row) to benefit from splitting into two independent chains

### Dispatching note

N=4 is the only non-power-of-2 to use inline asm (not the C++ schoolbook kernel path).
N=5,6 fall through to `mul_schoolbook_truncated_fixed<N>`. N=7-8 also schoolbook.

## Verification note

Correctness comparisons MUST use a reference implementation that propagates carry per product
pair (standard schoolbook), NOT an outer-product accumulator (`u128 temp[2*N]` with column-wise
sum). The latter overflows `u128` for columns with 2+ products (any N ≥ 2), producing false positives.

## GCC 15 `if consteval` miscompilation (historical)

GCC 15 has a bug where `if consteval` is treated as always-true inside `constexpr FORCE_INLINE`
functions whose body is "simple enough" for the compiler to constant-fold. This caused the
compile-time schoolbook fallback to be selected at runtime.

**Resolution:** `operator*` is non-constexpr (calls `operator*=` → `mul_runtime()` with asm
dispatch). `mul_consteval()` is provided as a public method for users who need compile-time
multiplication. This avoids `if consteval` entirely, so the GCC bug cannot trigger.

## Historical bugs (mul dispatch aliasing)

All runtime dispatch paths in `operator*=` previously passed `&bits[0]` as both `res` and `a`
to kernel functions. This is **broken** for any kernel that zeroes `res` before reading `a`
(which includes `std::fill_n`-based schoolbook and squaring kernels).

**Affected paths (fixed 2026-05-28):**
- N=9-16: `mul_schoolbook_truncated_fixed<N>` / `square_schoolbook_truncated_fixed<N>`
- N=17-32: via `mul_truncated_fixed` / `square_truncated_fixed` internal fallback to schoolbook

**Fix pattern:**
```cpp
u64 buf[N];
kernel(buf, &bits[0], ...);
copy_n(buf, N, &bits[0]);
```

**Note on column carries in Comba-style kernels:**
In truncated Comba multiplication, the carry from adding a product's lo word to the
column-k accumulator overflows to column k+1 (truncated), NOT back into column k.
Resetting `c = 0` between products within the same column is intentional and
correct — don't "chain" carries across products within a column.
