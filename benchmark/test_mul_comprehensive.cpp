#include <benchmark/benchmark.h>
#include "UInt.hpp"
#include <random>
#include <cstdio>
#include <cstdint>
using namespace Arbitrary;

// Reference multiplication using __int128 (independent of actual dispatch)
template <size_t N>
static UInt<N> ref_mul(const UInt<N> &a, const UInt<N> &b) {
    u128 temp[2*N] = {0};
    for (size_t i = 0; i < N; ++i)
        for (size_t j = 0; j < N; ++j)
            temp[i+j] += (u128)a.bits[i] * b.bits[j];
    UInt<N> out;
    u128 carry = 0;
    for (size_t i = 0; i < N; ++i) {
        u128 s = temp[i] + carry;
        out.bits[i] = (u64)s;
        carry = s >> 64;
    }
    return out;
}

// Helper: create UInt<N> from array
template <size_t N>
static void set_limbs(UInt<N> &x, const u64 *vals) {
    for (size_t i = 0; i < N; ++i) x.bits[i] = vals[i];
}

// Benchmark: random mul for given N
static void BM_Regression_AllN(benchmark::State &state) {
    size_t N = state.range(0);
    std::mt19937_64 rng(777);
    u64 av[64], bv[64];
    for (auto _ : state) {
        for (size_t i = 0; i < N; ++i) { av[i] = rng(); bv[i] = rng(); }
        switch (N) {
            #define CASE(n) case n: { UInt<n> a, b; set_limbs(a, av); set_limbs(b, bv); auto c = a*b; benchmark::DoNotOptimize(c); break; }
            CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
            #undef CASE
            default: break;
        }
    }
}
BENCHMARK(BM_Regression_AllN)->DenseRange(1, 8);

int main(int argc, char **argv) {
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) return 1;

    bool ok = true;
    std::mt19937_64 rng(42);

    // ===== N=3 exhaustive edge cases =====
    fprintf(stderr, "N=3: edge cases...\n");
    u64 tv[] = {
        0, 1, 2, 0xFF, 0xFFFF, 0xFFFFFFFF, 0xFFFFFFFFFFFFFFFFull,
        0x8000000000000000ull, 0x7FFFFFFFFFFFFFFFull,
        0xAAAAAAAAAAAAAAAAull, 0x5555555555555555ull,
        0xFFFFFFFF00000000ull, 0x00000000FFFFFFFFull
    };
    for (u64 ai = 0; ai < 13; ++ai) {
        for (u64 bi = 0; bi < 13; ++bi) {
            UInt<3> a{tv[ai], tv[(ai+1)%13], tv[(ai+2)%13]};
            UInt<3> b{tv[bi], tv[(bi+1)%13], tv[(bi+2)%13]};
            UInt<3> c = a * b;
            UInt<3> ref = ref_mul(a, b);
            if (c != ref) {
                fprintf(stderr, "FAIL edge a=[%016lx %016lx %016lx] b=[%016lx %016lx %016lx]\n"
                    "  got=[%016lx %016lx %016lx] exp=[%016lx %016lx %016lx]\n",
                    a.bits[2],a.bits[1],a.bits[0], b.bits[2],b.bits[1],b.bits[0],
                    c.bits[2],c.bits[1],c.bits[0], ref.bits[2],ref.bits[1],ref.bits[0]);
                ok = false;
            }
        }
    }

    // ===== N=3 random =====
    fprintf(stderr, "N=3: random 100000...\n");
    for (u64 t = 0; t < 100000; ++t) {
        UInt<3> a{rng(), rng(), rng()};
        UInt<3> b{rng(), rng(), rng()};
        UInt<3> c = a * b;
        UInt<3> ref = ref_mul(a, b);
        if (c != ref) {
            fprintf(stderr, "FAIL random trial %lu\n", t); ok = false; break;
        }
    }

    // ===== N=3 operator*= =====
    fprintf(stderr, "N=3: operator*= ...\n");
    for (u64 t = 0; t < 100000; ++t) {
        UInt<3> a{rng(), rng(), rng()};
        UInt<3> b{rng(), rng(), rng()};
        UInt<3> c = a * b;
        a *= b;
        if (a != c) { fprintf(stderr, "FAIL *= trial %lu\n", t); ok = false; break; }
    }

    // ===== N=3 aliasing =====
    fprintf(stderr, "N=3: aliasing...\n");
    for (u64 t = 0; t < 100000; ++t) {
        UInt<3> a{rng(), rng(), rng()};
        UInt<3> b = a;
        a *= a;
        b *= b;
        if (a != b) { fprintf(stderr, "FAIL aliasing trial %lu\n", t); ok = false; break; }
    }

    // ===== N=3 zero/one =====
    fprintf(stderr, "N=3: zero/one...\n");
    UInt<3> zero{0,0,0};
    UInt<3> one{1,0,0};
    UInt<3> max_3{~0ull,~0ull,~0ull};
    for (u64 t = 0; t < 1000; ++t) {
        UInt<3> a{rng(), rng(), rng()};
        if (zero * a != zero) { fprintf(stderr, "FAIL zero*a\n"); ok = false; }
        if (a * zero != zero) { fprintf(stderr, "FAIL a*zero\n"); ok = false; }
        if (one * a != a)     { fprintf(stderr, "FAIL one*a\n"); ok = false; }
        if (a * one != a)     { fprintf(stderr, "FAIL a*one\n"); ok = false; }
        if (max_3 * zero != zero) { fprintf(stderr, "FAIL max*zero\n"); ok = false; }
    }

    // ===== N=3 compile-time =====
    fprintf(stderr, "N=3: consteval...\n");
    {
        constexpr UInt<3> a{0xDEADBEEF, 0xCAFEBABE, 0x12345678};
        constexpr UInt<3> b{0x87654321, 0xFEDCBA98, 0x0F0F0F0F};
        constexpr UInt<3> c = a * b;
        UInt<3> ref = ref_mul(a, b);
        if (c != ref) { fprintf(stderr, "FAIL consteval\n"); ok = false; }
    }
    {
        constexpr UInt<3> a{~0ull, ~0ull, ~0ull};
        constexpr UInt<3> b{~0ull, ~0ull, ~0ull};
        constexpr UInt<3> c = a * b;
        UInt<3> ref = ref_mul(a, b);
        if (c != ref) { fprintf(stderr, "FAIL consteval max\n"); ok = false; }
    }

    // ===== N=3 squaring vs mul =====
    fprintf(stderr, "N=3: squaring...\n");
    for (u64 t = 0; t < 100000; ++t) {
        UInt<3> a{rng(), rng(), rng()};
        UInt<3> b = a;
        if (a * a != a * b) { fprintf(stderr, "FAIL square trial %lu\n", t); ok = false; break; }
    }

    // ===== Dispatch boundaries N=1..16 =====
    fprintf(stderr, "N=1..16: random 10000 each...\n");
    for (size_t N = 1; N <= 16; ++N) {
        u64 av[16], bv[16];
        for (u64 t = 0; t < 10000; ++t) {
            for (size_t i = 0; i < N; ++i) { av[i] = rng(); bv[i] = rng(); }
            #define CASE(n) case n: { UInt<n> a, b; set_limbs(a, av); set_limbs(b, bv); \
                if (a * b != ref_mul(a, b)) { fprintf(stderr, "FAIL N=%zu trial %lu\n", N, t); ok = false; goto fail1; } break; }
            switch (N) {
                CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8)
                CASE(9) CASE(10) CASE(11) CASE(12) CASE(13) CASE(14) CASE(15) CASE(16)
                default: break;
            }
            #undef CASE
        }
    }
    fail1:

    // ===== N=17..32 =====
    fprintf(stderr, "N=17..32: random 1000 each...\n");
    for (size_t N = 17; N <= 32; ++N) {
        u64 av[32], bv[32];
        for (u64 t = 0; t < 1000; ++t) {
            for (size_t i = 0; i < N; ++i) { av[i] = rng(); bv[i] = rng(); }
            #define CASE(n) case n: { UInt<n> a, b; set_limbs(a, av); set_limbs(b, bv); \
                if (a * b != ref_mul(a, b)) { fprintf(stderr, "FAIL N=%zu trial %lu\n", N, t); ok = false; goto fail2; } break; }
            switch (N) {
                CASE(17) CASE(18) CASE(19) CASE(20) CASE(21) CASE(22) CASE(23) CASE(24)
                CASE(25) CASE(26) CASE(27) CASE(28) CASE(29) CASE(30) CASE(31) CASE(32)
                default: break;
            }
            #undef CASE
        }
    }
    fail2:

    // ===== N=3 commutativity =====
    fprintf(stderr, "N=3: commutativity...\n");
    for (u64 t = 0; t < 100000; ++t) {
        UInt<3> a{rng(), rng(), rng()};
        UInt<3> b{rng(), rng(), rng()};
        if (a * b != b * a) { fprintf(stderr, "FAIL commutativity trial %lu\n", t); ok = false; break; }
    }

    // ===== N=3 overflow-specific: k=2 edge conditions =====
    fprintf(stderr, "N=3: k=2 overflow edges...\n");
    // Test the case where hi(a0*b1) + CF1 overflows
    // hi(a0*b1) = 0xFFFFFFFFFFFFFFFF, CF1 = 1 → r2 wraps to 0
    {
        UInt<3> a{0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull};
        UInt<3> b{0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull, 0xFFFFFFFFFFFFFFFFull};
        if (a * b != ref_mul(a, b)) { fprintf(stderr, "FAIL all-ones\n"); ok = false; }
    }
    // Single bit at top of each limb
    {
        UInt<3> a{0x8000000000000000ull, 0x8000000000000000ull, 0x8000000000000000ull};
        UInt<3> b{0x8000000000000000ull, 0x8000000000000000ull, 0x8000000000000000ull};
        if (a * b != ref_mul(a, b)) { fprintf(stderr, "FAIL top-bit\n"); ok = false; }
    }
    // Alternating bits
    {
        UInt<3> a{0xAAAAAAAAAAAAAAAAull, 0xAAAAAAAAAAAAAAAAull, 0xAAAAAAAAAAAAAAAAull};
        UInt<3> b{0x5555555555555555ull, 0x5555555555555555ull, 0x5555555555555555ull};
        if (a * b != ref_mul(a, b)) { fprintf(stderr, "FAIL alternating\n"); ok = false; }
    }

    fprintf(stderr, "\n*** %s ***\n", ok ? "ALL TESTS PASSED" : "SOME TESTS FAILED");
    if (!ok) return 1;
    ::benchmark::RunSpecifiedBenchmarks();
    return 0;
}
