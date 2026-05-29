<div align="center">
  <h1>Arbitrary</h1>
  <p><strong>Biblioteca C++20 de precisão arbitrária para inteiros sem sinal</strong></p>
  <p>Alta performance, sem alocação dinâmica, 100% header-only.</p>
  <p>
    <img src="https://img.shields.io/badge/C%2B%2B-20-blue.svg" alt="C++20">
    <img src="https://img.shields.io/badge/architecture-x86__64-blue.svg" alt="x86_64">
    <img src="https://img.shields.io/badge/license-CC--BY--NC--4.0-lightgrey.svg" alt="CC BY-NC 4.0">
  </p>
</div>

---

## Visão Geral

**Arbitrary** é uma biblioteca de inteiros sem sinal de precisão arbitrária projetada para cargas de trabalho criptográficas e computação pesada onde **alocação dinâmica de memória é inaceitável**. Todos os dados residem na pilha (_stack_), com tamanho fixo definido em tempo de compilação.

A biblioteca faz uso intenso de intrínsecos BMI2/ADX (`_mulx_u64`, `_addcarry_u64`, `_subborrow_u64`) e assembly _inline_ para entregar desempenho máximo em processadores x86_64 modernos.

---

## Características

- **Header-only** — basta incluir o arquivo e compilar.
- **Zero alocação dinâmica** — sem `new`/`delete`, sem `malloc`.
- **Tamanho fixo em tempo de compilação** — de 64 bits até milhares de bits, definido pelo parâmetro `N` (número de _limbs_ de 64 bits).
- **Múltiplos algoritmos de multiplicação** — _Schoolbook_, _Comba_, _Karatsuba_ e _Toom-style split_, selecionados automaticamente conforme o tamanho.
- **Divisão eficiente** — Algoritmo D de Knuth com variante recursiva para operandos grandes.
- **Intrínsecos BMI2/ADX** — `_mulx_u64`, `_addcarry_u64`, `_subborrow_u64`.
- **Assembly inline** — loops de adição/subtração com `addq`/`adcq` e `subq`/`sbbq`.
- **Avaliação em tempo de compilação** — rotas `consteval` com `unsigned __int128` para uso em contextos `constexpr`.
- **Alinhamento a 64 bytes** — otimização de linha de cache.
- **Sem dependências externas** — apenas a biblioteca padrão do C++20.

---

## Arquitetura

```
Arbitrary/
├── UInt.hpp                 # Tipo inteiro principal + operadores aritméticos
├── Multiplication.hpp       # Algoritmos de multiplicação (Schoolbook, Comba, Karatsuba)
├── Division.hpp             # Algoritmos de divisão (Knuth D, divisão recursiva)
├── benchmark/
│   ├── CMakeLists.txt       # Build dos benchmarks (usa Google Benchmark)
│   └── bench_arithmetic.cpp # Benchmarks de todas as operações aritméticas
├── docs/
│   └── index.html           # Documentação interativa
└── LICENCE                  # CC BY-NC 4.0
```

### UInt\<N\>

O tipo central é `Arbitrary::UInt<N>`, onde `N` é o número de _limbs_ de 64 bits (valor `uint8_t` > 0). O armazenamento interno é sempre `std::array<u64, N>`.

### Multiplicação

A seleção do algoritmo é feita automaticamente em tempo de compilação:

| `N` | Algoritmo |
|-----|-----------|
| 1   | Multiplicação direta `u64` |
| 2   | Schoolbook 2×2 com `__int128` |
| 3   | Schoolbook 3×3 otimizado |
| 4   | Schoolbook 4×4 |
| 5–6 | Comba (produto-varredura) |
| 8   | Split-block (Toom-style 4×4) |
| ≤16 | Schoolbook ou quadrático especializado |
| >16 | Karatsuba O(n^1.585) |

### Divisão

| Condição | Algoritmo |
|----------|-----------|
| Divisor de 1 limb | Loop simples com `__int128` |
| `N ≤ 16` | Knuth D especializado com desenrolamento |
| `N ≤ 64` | Knuth D genérico |
| `N > 64` com diferença pequena | Knuth D genérico |
| `N > 64` com diferença grande | Divisão recursiva divisão-e-conquista |

---

## Como usar

### Compilação

Por ser header-only, basta incluir o arquivo principal:

```cpp
#include "UInt.hpp"
```

Compile com:

```bash
g++ -std=c++20 -O3 -march=native -mbmi2 -madx main.cpp -o main
```

**Flags necessárias:**
- `-std=c++20` — padrão C++20.
- `-O3` — otimização obrigatória para intrínsecos e assembly inline.
- `-march=native` — ativa BMI2, ADX e outras extensões x86_64.
- `-mbmi2 -madx` — (opcional, explícito).

Compiladores suportados: **GCC** e **Clang**.

### Exemplo

```cpp
#include <iostream>
#include "UInt.hpp"

using namespace Arbitrary;

int main() {
    UInt<4> a("12345678901234567890");
    UInt<4> b("0xDEADBEEF");

    UInt<4> c = a + b;
    UInt<4> d = a * b;
    auto [q, r] = UInt<8>::divmod(a * b, a);

    std::cout << "a + b = " << c.to_string() << "\n";
    std::cout << "a * b = " << d.to_hex_string() << "\n";
    std::cout << "(a*b) / a = " << q.to_string() << "\n";

    // Geração de números aleatórios
    UInt<4> rnd = UInt<4>::random(12345);
    std::cout << "random = " << rnd.to_hex_string() << "\n";
}
```

### API principal

| Operação | Descrição |
|----------|-----------|
| `UInt<N>(string_view)` | Constrói a partir de decimal ou hexadecimal |
| `to_string()` / `to_hex_string()` | Converte para string decimal ou hexadecimal |
| `+`, `-`, `*`, `/`, `%` | Operadores aritméticos |
| `+=`, `-=`, `*=`, `/=`, `%=` | Operadores compostos |
| `&`, `\|`, `^`, `~`, `<<`, `>>` | Operadores bitwise |
| `==`, `!=`, `<`, `>`, `<=`, `>=` | Operadores de comparação |
| `++`, `--` | Incremento/decremento |
| `bt(i)` | Testa o i-ésimo bit |
| `bts(i)` | Ativa o i-ésimo bit |
| `is_zero()` | Verifica se é zero |
| `static random(seed)` | Gera número aleatório (xorshift64*) |
| `static divmod(a, b)` | Retorna par {quociente, resto} |

---

## Benchmarks

O diretório `benchmark/` contém benchmarks usando [Google Benchmark](https://github.com/google/benchmark) para todas as operações da biblioteca. Para compilar e executar:

```bash
cd benchmark
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)
./bench_arithmetic
```

### Resultados de desempenho (CPU x86_64 @ 4.4 GHz, GCC 15, C++20)

| Operação | N=1 | N=4 | N=16 | N=64 | N=128 |
|----------|-----|-----|------|------|-------|
| Adição | 9.7 ns | 7.5 ns | 24.4 ns | 132 ns | 275 ns |
| Subtração | 10.7 ns | 15.4 ns | 34.1 ns | 152 ns | 300 ns |
| Multiplicação | 1.95 ns | 16.2 ns | 23.6 ns | 2495 ns | 7369 ns |
| Multiplicação (por u64) | — | 5.38 ns | 22.3 ns | 103 ns | — |
| Divisão | 10.8 ns | 36.2 ns | 98.6 ns | — | — |
| divmod | 9.79 ns | 37.3 ns | 95.8 ns | — | — |
| Squaring | 1.19 ns | 14.0 ns | 12.2 ns | 2385 ns | 6980 ns |
| to_string | 168 ns | 436 ns | 2424 ns | 32451 ns | — |
| to_hex_string | 170 ns | 256 ns | 601 ns | 2101 ns | — |
| from_string (hex) | 34.6 ns | 128 ns | 503 ns | 2040 ns | — |
| Random | 3.83 ns | 11.9 ns | 52.5 ns | 114 ns |

- N = número de limbs de 64 bits (ex: N=64 → 4096 bits)
- Tempos em nanossegundos (ns)
- Compilado com `-O3 -march=native -mbmi2 -madx`

---

## Requisitos

- **CPU** x86_64 com suporte a **BMI2** e **ADX** (Intel Haswell+ / AMD Excavator+).
- **Compilador** GCC 11+ ou Clang 14+ com suporte a C++20.
- Nenhuma biblioteca externa é necessária.

---

## Licença

Este projeto está licenciado sob a **Creative Commons Attribution-NonCommercial 4.0 International**. Consulte o arquivo [LICENCE](LICENCE) para mais detalhes.

---

## Autor

Desenvolvido por [juliano-xd](https://github.com/juliano-xd).
