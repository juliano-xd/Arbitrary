// Comprehensive multiplication test + benchmark suite
// Part 1: Correctness tests (reference schoolbook vs dispatched)
// Part 2: Detailed Google Benchmarks covering all dispatch boundaries

#include <benchmark/benchmark.h>
#include <cstdio>
#include <cstring>
#include "UInt.hpp"

using namespace Arbitrary;

// ═══════════════════════════════════════════════════════════════
// Part 1 — Reference implementation (independent schoolbook)
// ═══════════════════════════════════════════════════════════════

template <u8 N>
static UInt<N> ref_mul(const UInt<N> &a, const UInt<N> &b) {
    UInt<N> r;
    r.bits.fill(0);
    for (u8 i = 0; i < N; ++i) {
        u64 y = a.bits[i];
        if (y == 0) continue;
        u128 carry = 0;
        for (u8 j = 0; j < N - i; ++j) {
            u128 t = (u128)b.bits[j] * y + r.bits[i + j] + carry;
            r.bits[i + j] = (u64)t;
            carry = t >> 64;
        }
    }
    return r;
}

// ═══════════════════════════════════════════════════════════════
// Correctness tests
// ═══════════════════════════════════════════════════════════════

static int g_fail = 0;

#define CHECK(cond, fmt, ...) do { \
    if (!(cond)) { \
        printf("  FAIL " fmt "\n", ##__VA_ARGS__); \
        ++g_fail; \
    } \
} while(0)

template <u8 N>
static void test_against_reference(int trials) {
    for (int t = 0; t < trials; ++t) {
        u64 s = 0x9E3779B97F4A7C15ull + t * 7919;
        auto a = UInt<N>::random(s);
        auto b = UInt<N>::random(s + 31337);

        auto expected = ref_mul(a, b);
        UInt<N> got = a;
        got *= b;

        CHECK(expected.bits == got.bits,
              "N=%d trial=%d ref≠disp", (int)N, t);
    }
}

template <u8 N>
static void test_commutativity(int trials) {
    for (int t = 0; t < trials; ++t) {
        u64 s = 0xC0FFEEull + t * 1337;
        auto a = UInt<N>::random(s);
        auto b = UInt<N>::random(s + 5555);

        UInt<N> ab = a; ab *= b;
        UInt<N> ba = b; ba *= a;

        CHECK(ab.bits == ba.bits,
              "N=%d trial=%d a*b ≠ b*a", (int)N, t);
    }
}

template <u8 N>
static void test_square_against_mul(int trials) {
    for (int t = 0; t < trials; ++t) {
        u64 s = 0x5EEDull + t * 777;
        auto a = UInt<N>::random(s);

        UInt<N> sq; sq = a; sq *= a;          // dispatched square path
        UInt<N> mul_ref = ref_mul(a, a);       // reference

        CHECK(sq.bits == mul_ref.bits,
              "N=%d trial=%d square≠ref", (int)N, t);
    }
}

template <u8 N>
static void test_mul_u64(int trials) {
    for (int t = 0; t < trials; ++t) {
        u64 s = 0xABADCAFEull + t * 999;
        auto a = UInt<N>::random(s);
        u64 v = UInt<1>::random(s + 4444).bits[0];

        UInt<N> expected; expected = a;
        // reference scalar mul: simple u128 loop (independent impl)
        u64 carry = 0;
        for (u8 i = 0; i < N; ++i) {
            u128 p = (u128)expected.bits[i] * v + carry;
            expected.bits[i] = (u64)p;
            carry = p >> 64;
        }

        UInt<N> got = a;
        got *= v;

        CHECK(expected.bits == got.bits,
              "N=%d trial=%d scalar mul mismatch", (int)N, t);
    }
}

static void test_edge_cases() {
    // Test specific edge values across all dispatch boundaries
    const u8 NS[] = {1,2,3,4,5,6,7,8,16,17,32,33,64,128};
    for (u8 n : NS) {
        // Zero
        { UInt<1> z{0}; UInt<sizeof(NS)/sizeof(NS[0])> zero; zero.bits.fill(0);
          // Actually let me use a simpler approach
        }
    }
    // Redo with template
    auto test_n = [&]<u8 N>() {
        UInt<N> zero; zero.bits.fill(0);
        UInt<N> one; one.bits.fill(0); one.bits[0] = 1;
        UInt<N> all1; all1.bits.fill(~0ull);
        UInt<N> msb; msb.bits.fill(0); msb.bits[N-1] = 1ull << 63;
        UInt<N> two_pow; two_pow.bits.fill(0); two_pow.bits[0] = 1ull << 63;

        auto a = UInt<N>::random(12345);

        // zero * anything = zero
        { UInt<N> r = zero; r *= a; CHECK(r.is_zero(), "N=%d 0*a ≠ 0", N); }
        { UInt<N> r = a; r *= zero; CHECK(r.is_zero(), "N=%d a*0 ≠ 0", N); }

        // one * anything = anything (mod 2^(64N))
        { UInt<N> r = one; r *= a; CHECK(r.bits == a.bits, "N=%d 1*a ≠ a", N); }
        { UInt<N> r = a; r *= one; CHECK(r.bits == a.bits, "N=%d a*1 ≠ a", N); }

        // self-assignment (aliasing): a *= a
        { UInt<N> r = a; r *= a; UInt<N> ref = ref_mul(a, a);
          CHECK(r.bits == ref.bits, "N=%d a*=a ≠ ref", N); }

        // commutativity for edge vectors
        { UInt<N> r1 = all1; r1 *= a; UInt<N> r2 = a; r2 *= all1;
          CHECK(r1.bits == r2.bits, "N=%d all1*a ≠ a*all1", N); }
    };
    // Instantiate for specific N
    auto run = [&]<u8... Ns>(std::integer_sequence<u8, Ns...>) {
        (test_n.template operator()<Ns>(), ...);
    };
    run(std::integer_sequence<u8, 1,2,3,4,5,6,7,8,16,17,32,33,64,128>{});
}

// ═══════════════════════════════════════════════════════════════
// Part 2 — Detailed Google Benchmarks
// ═══════════════════════════════════════════════════════════════

// Benchmark multiplication at a specific N
template <u8 N>
static void BM_Mul_N(benchmark::State &state) {
    u64 seed = 12345;
    for (auto _ : state) {
        ++seed;
        auto a = UInt<N>::random(seed);
        auto b = UInt<N>::random(seed + 1000);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        auto c = a * b;
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();
    }
}

// Benchmark squaring at a specific N
template <u8 N>
static void BM_Square_N(benchmark::State &state) {
    u64 seed = 12345;
    for (auto _ : state) {
        ++seed;
        auto a = UInt<N>::random(seed);
        benchmark::DoNotOptimize(a);
        auto c = a * a;
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();
    }
}

// Benchmark multiply by u64
template <u8 N>
static void BM_MulU64_N(benchmark::State &state) {
    u64 seed = 12345;
    u64 b = 0xDEADBEEFCAFEBABEull;
    for (auto _ : state) {
        ++seed;
        auto a = UInt<N>::random(seed);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        auto c = a * b;
        benchmark::DoNotOptimize(c);
        benchmark::ClobberMemory();
    }
}

// Benchmark operator*=
template <u8 N>
static void BM_MulEq_N(benchmark::State &state) {
    u64 seed = 12345;
    for (auto _ : state) {
        ++seed;
        auto a = UInt<N>::random(seed);
        auto b = UInt<N>::random(seed + 1000);
        benchmark::DoNotOptimize(a);
        benchmark::DoNotOptimize(b);
        a *= b;
        benchmark::DoNotOptimize(a);
        benchmark::ClobberMemory();
    }
}

// Register all benchmarks — every N from 1 to 32 (every dispatch point),
// then specific larger N
#define REGISTER_MUL(N) \
    BENCHMARK_TEMPLATE(BM_Mul_N, N);        \
    BENCHMARK_TEMPLATE(BM_Square_N, N);     \
    BENCHMARK_TEMPLATE(BM_MulU64_N, N);     \
    BENCHMARK_TEMPLATE(BM_MulEq_N, N)

// — Dispatch boundaries (1-32 every single N) —
REGISTER_MUL(1);
REGISTER_MUL(2);
REGISTER_MUL(3);
REGISTER_MUL(4);
REGISTER_MUL(5);
REGISTER_MUL(6);
REGISTER_MUL(7);
REGISTER_MUL(8);
REGISTER_MUL(9);
REGISTER_MUL(10);
REGISTER_MUL(11);
REGISTER_MUL(12);
REGISTER_MUL(13);
REGISTER_MUL(14);
REGISTER_MUL(15);
REGISTER_MUL(16);
REGISTER_MUL(17);
REGISTER_MUL(18);
REGISTER_MUL(19);
REGISTER_MUL(20);
REGISTER_MUL(21);
REGISTER_MUL(22);
REGISTER_MUL(23);
REGISTER_MUL(24);
REGISTER_MUL(25);
REGISTER_MUL(26);
REGISTER_MUL(27);
REGISTER_MUL(28);
REGISTER_MUL(29);
REGISTER_MUL(30);
REGISTER_MUL(31);
REGISTER_MUL(32);

// — Larger sizes at key boundaries —
REGISTER_MUL(33);   // Karatsuba starts here for mul_truncated_fixed
REGISTER_MUL(40);
REGISTER_MUL(48);
REGISTER_MUL(56);
REGISTER_MUL(64);
REGISTER_MUL(80);
REGISTER_MUL(96);
REGISTER_MUL(112);
REGISTER_MUL(128);
REGISTER_MUL(192);
REGISTER_MUL(255);  // max N

// ═══════════════════════════════════════════════════════════════
// Main: run correctness tests, then Google Benchmark
// ═══════════════════════════════════════════════════════════════

int main(int argc, char **argv) {
    printf("════════════════════════════════════════════════════\n");
    printf("  Multiplication Correctness Tests\n");
    printf("════════════════════════════════════════════════════\n\n");

    // Reference comparison for all dispatch boundaries
    printf("Reference comparison (randomized):\n");
    test_against_reference<1>(2000);
    test_against_reference<2>(2000);
    test_against_reference<3>(2000);
    test_against_reference<4>(2000);
    test_against_reference<5>(1000);
    test_against_reference<6>(1000);
    test_against_reference<7>(1000);
    test_against_reference<8>(1000);
    test_against_reference<16>(500);
    test_against_reference<17>(500);
    test_against_reference<32>(300);
    test_against_reference<33>(200);
    test_against_reference<64>(100);
    test_against_reference<128>(50);

    printf("Commutativity:\n");
    test_commutativity<1>(1000);
    test_commutativity<2>(1000);
    test_commutativity<3>(1000);
    test_commutativity<4>(1000);
    test_commutativity<8>(500);
    test_commutativity<16>(500);
    test_commutativity<32>(200);
    test_commutativity<64>(100);
    test_commutativity<128>(50);

    printf("Squaring (a*=a vs ref_mul):\n");
    test_square_against_mul<1>(1000);
    test_square_against_mul<2>(1000);
    test_square_against_mul<4>(1000);
    test_square_against_mul<8>(1000);
    test_square_against_mul<16>(500);
    test_square_against_mul<32>(200);
    test_square_against_mul<64>(100);
    test_square_against_mul<128>(50);

    printf("Scalar (a*=u64 vs reference):\n");
    test_mul_u64<4>(1000);
    test_mul_u64<16>(500);
    test_mul_u64<64>(200);
    test_mul_u64<128>(100);

    printf("Edge cases (zero, one, all-ones, aliasing):\n");
    test_edge_cases();

    if (g_fail == 0)
        printf("\n✅ All correctness tests PASSED\n\n");
    else
        printf("\n❌ %d test(s) FAILED\n\n", g_fail);

    printf("════════════════════════════════════════════════════\n");
    printf("  Multiplication Benchmarks\n");
    printf("════════════════════════════════════════════════════\n\n");

    ::benchmark::Initialize(&argc, argv);
    ::benchmark::RunSpecifiedBenchmarks();

    return g_fail;
}
