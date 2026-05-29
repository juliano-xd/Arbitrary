#include <benchmark/benchmark.h>
#include "UInt.hpp"

using namespace Arbitrary;

template <u8 N>
static UInt<N> make_random(u64 seed) {
    return UInt<N>::random(seed);
}

#define ARITH_BENCHMARK(name, N, op)                                          \
    static void BM_##name##_##N(benchmark::State &state) {                    \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            auto b = make_random<N>(seed + 1000);                             \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::DoNotOptimize(b);                                      \
            auto c = a op b;                                                  \
            benchmark::DoNotOptimize(c);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_##name##_##N)

// ─── Addition ───────────────────────────────────────────────────────────────

ARITH_BENCHMARK(Add, 1, +);
ARITH_BENCHMARK(Add, 2, +);
ARITH_BENCHMARK(Add, 4, +);
ARITH_BENCHMARK(Add, 8, +);
ARITH_BENCHMARK(Add, 16, +);
ARITH_BENCHMARK(Add, 32, +);
ARITH_BENCHMARK(Add, 64, +);
ARITH_BENCHMARK(Add, 128, +);

// ─── Subtraction ────────────────────────────────────────────────────────────

#define SUB_BENCHMARK(N)                                                      \
    static void BM_Sub_##N(benchmark::State &state) {                         \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            auto b = make_random<N>(seed + 1000);                             \
            if (a < b) std::swap(a, b);                                       \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::DoNotOptimize(b);                                      \
            auto c = a - b;                                                   \
            benchmark::DoNotOptimize(c);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_Sub_##N)

SUB_BENCHMARK(1);
SUB_BENCHMARK(2);
SUB_BENCHMARK(4);
SUB_BENCHMARK(8);
SUB_BENCHMARK(16);
SUB_BENCHMARK(32);
SUB_BENCHMARK(64);
SUB_BENCHMARK(128);

// ─── Multiplication ─────────────────────────────────────────────────────────

ARITH_BENCHMARK(Mul, 1, *);
ARITH_BENCHMARK(Mul, 2, *);
ARITH_BENCHMARK(Mul, 3, *);
ARITH_BENCHMARK(Mul, 4, *);
ARITH_BENCHMARK(Mul, 5, *);
ARITH_BENCHMARK(Mul, 6, *);
ARITH_BENCHMARK(Mul, 8, *);
ARITH_BENCHMARK(Mul, 16, *);
ARITH_BENCHMARK(Mul, 32, *);
ARITH_BENCHMARK(Mul, 64, *);
ARITH_BENCHMARK(Mul, 128, *);

// ─── Multiplication by u64 ──────────────────────────────────────────────────

#define MULU64_BENCHMARK(N)                                                   \
    static void BM_MulU64_##N(benchmark::State &state) {                      \
        u64 seed = __LINE__;                                                  \
        u64 b = 0xDEADBEEFCAFEBABEull;                                        \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::DoNotOptimize(b);                                      \
            auto c = a * b;                                                   \
            benchmark::DoNotOptimize(c);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_MulU64_##N)

MULU64_BENCHMARK(4);
MULU64_BENCHMARK(16);
MULU64_BENCHMARK(64);

// ─── Division ───────────────────────────────────────────────────────────────

#define DIV_BENCHMARK(N)                                                      \
    static void BM_Div_##N(benchmark::State &state) {                         \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            auto b = make_random<N>(seed + 1000);                             \
            if (b.is_zero()) b.bits[0] = 1;                                   \
            if (a < b) std::swap(a, b);                                       \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::DoNotOptimize(b);                                      \
            auto c = a / b;                                                   \
            benchmark::DoNotOptimize(c);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_Div_##N)

DIV_BENCHMARK(1);
DIV_BENCHMARK(2);
DIV_BENCHMARK(3);
DIV_BENCHMARK(4);
DIV_BENCHMARK(5);
DIV_BENCHMARK(8);
DIV_BENCHMARK(16);
DIV_BENCHMARK(32);

// ─── Modulo ─────────────────────────────────────────────────────────────────

#define MOD_BENCHMARK(N)                                                      \
    static void BM_Mod_##N(benchmark::State &state) {                         \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            auto b = make_random<N>(seed + 1000);                             \
            if (b.is_zero()) b.bits[0] = 1;                                   \
            if (a < b) std::swap(a, b);                                       \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::DoNotOptimize(b);                                      \
            auto c = a % b;                                                   \
            benchmark::DoNotOptimize(c);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_Mod_##N)

MOD_BENCHMARK(1);
MOD_BENCHMARK(4);
MOD_BENCHMARK(16);
MOD_BENCHMARK(32);

// ─── divmod ─────────────────────────────────────────────────────────────────

#define DIVMOD_BENCHMARK(N)                                                   \
    static void BM_DivMod_##N(benchmark::State &state) {                      \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            auto b = make_random<N>(seed + 1000);                             \
            if (b.is_zero()) b.bits[0] = 1;                                   \
            if (a < b) std::swap(a, b);                                       \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::DoNotOptimize(b);                                      \
            auto [q, r] = UInt<N>::divmod(a, b);                              \
            benchmark::DoNotOptimize(q);                                      \
            benchmark::DoNotOptimize(r);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_DivMod_##N)

DIVMOD_BENCHMARK(1);
DIVMOD_BENCHMARK(4);
DIVMOD_BENCHMARK(16);
DIVMOD_BENCHMARK(32);

// ─── Squaring (a * a) ───────────────────────────────────────────────────────

#define SQUARE_BENCHMARK(N)                                                   \
    static void BM_Square_##N(benchmark::State &state) {                      \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            benchmark::DoNotOptimize(a);                                      \
            auto c = a * a;                                                   \
            benchmark::DoNotOptimize(c);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_Square_##N)

SQUARE_BENCHMARK(1);
SQUARE_BENCHMARK(4);
SQUARE_BENCHMARK(8);
SQUARE_BENCHMARK(16);
SQUARE_BENCHMARK(32);
SQUARE_BENCHMARK(64);
SQUARE_BENCHMARK(128);

// ─── String Conversion ──────────────────────────────────────────────────────

#define TO_STRING_BENCHMARK(N)                                                \
    static void BM_ToString_##N(benchmark::State &state) {                    \
        auto a = make_random<N>(__LINE__);                                    \
        for (auto _ : state) {                                                \
            auto x = a;                                                       \
            benchmark::DoNotOptimize(x);                                      \
            auto s = x.to_string();                                           \
            benchmark::DoNotOptimize(s);                                      \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_ToString_##N)

TO_STRING_BENCHMARK(1);
TO_STRING_BENCHMARK(4);
TO_STRING_BENCHMARK(16);
TO_STRING_BENCHMARK(64);

#define TO_HEX_BENCHMARK(N)                                                   \
    static void BM_ToHexString_##N(benchmark::State &state) {                 \
        auto a = make_random<N>(__LINE__);                                    \
        for (auto _ : state) {                                                \
            auto x = a;                                                       \
            benchmark::DoNotOptimize(x);                                      \
            auto s = x.to_hex_string();                                       \
            benchmark::DoNotOptimize(s);                                      \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_ToHexString_##N)

TO_HEX_BENCHMARK(1);
TO_HEX_BENCHMARK(4);
TO_HEX_BENCHMARK(16);
TO_HEX_BENCHMARK(64);

// ─── Construction from String ───────────────────────────────────────────────

#define FROM_STRING_BENCHMARK(N)                                              \
    static void BM_FromString_##N(benchmark::State &state) {                  \
        std::string hex;                                                      \
        for (u8 i = 0; i < N; ++i)                                           \
            hex += "DEADBEEFCAFEBABE";                                        \
        hex = "0x" + hex;                                                     \
        for (auto _ : state) {                                                \
            UInt<N> a(hex);                                                   \
            benchmark::DoNotOptimize(a);                                      \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_FromString_##N)

FROM_STRING_BENCHMARK(1);
FROM_STRING_BENCHMARK(4);
FROM_STRING_BENCHMARK(16);
FROM_STRING_BENCHMARK(64);

// ─── Bitwise AND ────────────────────────────────────────────────────────────

ARITH_BENCHMARK(BitwiseAnd, 4, &);
ARITH_BENCHMARK(BitwiseAnd, 64, &);
ARITH_BENCHMARK(BitwiseAnd, 128, &);

// ─── Shift Left ─────────────────────────────────────────────────────────────

#define SHL_BENCHMARK(N)                                                      \
    static void BM_ShiftLeft_##N(benchmark::State &state) {                   \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            benchmark::DoNotOptimize(a);                                      \
            auto c = a << 7;                                                  \
            benchmark::DoNotOptimize(c);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_ShiftLeft_##N)

SHL_BENCHMARK(4);
SHL_BENCHMARK(16);
SHL_BENCHMARK(64);

// ─── Shift Right ────────────────────────────────────────────────────────────

#define SHR_BENCHMARK(N)                                                      \
    static void BM_ShiftRight_##N(benchmark::State &state) {                  \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            benchmark::DoNotOptimize(a);                                      \
            auto c = a >> 7;                                                  \
            benchmark::DoNotOptimize(c);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_ShiftRight_##N)

SHR_BENCHMARK(4);
SHR_BENCHMARK(16);
SHR_BENCHMARK(64);

// ─── Comparison ─────────────────────────────────────────────────────────────

#define CMP_BENCHMARK(N)                                                      \
    static void BM_Compare_##N(benchmark::State &state) {                     \
        u64 seed = __LINE__;                                                  \
        bool r;                                                               \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            auto b = make_random<N>(seed + 1000);                             \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::DoNotOptimize(b);                                      \
            r = a < b;                                                        \
            benchmark::DoNotOptimize(r);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_Compare_##N)

CMP_BENCHMARK(4);
CMP_BENCHMARK(16);
CMP_BENCHMARK(64);
CMP_BENCHMARK(128);

// ─── Increment ──────────────────────────────────────────────────────────────

#define INC_BENCHMARK(N)                                                      \
    static void BM_Increment_##N(benchmark::State &state) {                   \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            benchmark::DoNotOptimize(a);                                      \
            ++a;                                                              \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_Increment_##N)

INC_BENCHMARK(4);
INC_BENCHMARK(64);

// ─── Decrement ──────────────────────────────────────────────────────────────

#define DEC_BENCHMARK(N)                                                      \
    static void BM_Decrement_##N(benchmark::State &state) {                   \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            benchmark::DoNotOptimize(a);                                      \
            --a;                                                              \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_Decrement_##N)

DEC_BENCHMARK(4);
DEC_BENCHMARK(64);

// ─── Compound Assignment ────────────────────────────────────────────────────

#define ADDEQ_BENCHMARK(N)                                                    \
    static void BM_AddEq_##N(benchmark::State &state) {                       \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            auto b = make_random<N>(seed + 1000);                             \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::DoNotOptimize(b);                                      \
            a += b;                                                           \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_AddEq_##N)

ADDEQ_BENCHMARK(4);
ADDEQ_BENCHMARK(64);

#define MULEQ_BENCHMARK(N)                                                    \
    static void BM_MulEq_##N(benchmark::State &state) {                       \
        u64 seed = __LINE__;                                                  \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            auto b = make_random<N>(seed + 1000);                             \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::DoNotOptimize(b);                                      \
            a *= b;                                                           \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_MulEq_##N)

MULEQ_BENCHMARK(4);
MULEQ_BENCHMARK(16);
MULEQ_BENCHMARK(32);

// ─── Random Generation ──────────────────────────────────────────────────────

#define RANDOM_BENCHMARK(N)                                                   \
    static void BM_Random_##N(benchmark::State &state) {                      \
        u64 seed = 12345;                                                     \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = UInt<N>::random(seed);                                   \
            benchmark::DoNotOptimize(a);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_Random_##N)

RANDOM_BENCHMARK(4);
RANDOM_BENCHMARK(16);
RANDOM_BENCHMARK(64);
RANDOM_BENCHMARK(128);

// ─── Bit Test ───────────────────────────────────────────────────────────────

#define BIT_TEST_BENCHMARK(N)                                                 \
    static void BM_BitTest_##N(benchmark::State &state) {                     \
        u64 seed = __LINE__;                                                  \
        bool r;                                                               \
        for (auto _ : state) {                                                \
            ++seed;                                                           \
            auto a = make_random<N>(seed);                                    \
            benchmark::DoNotOptimize(a);                                      \
            r = a.bt(42);                                                     \
            benchmark::DoNotOptimize(r);                                      \
            benchmark::ClobberMemory();                                       \
        }                                                                     \
    }                                                                         \
    BENCHMARK(BM_BitTest_##N)

BIT_TEST_BENCHMARK(4);
BIT_TEST_BENCHMARK(64);

BENCHMARK_MAIN();
