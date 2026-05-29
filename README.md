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
├── docs/
│   └── index.html           # Documentação interativa
└── LICENCE                  # CC BY-NC 4.0
```

### UInt\<N\>

O tipo central é `Arbitrary::UInt<N>`, onde `N` é o número de _limbs_ de 64 bits (valor `uint8_t` > 0).

| `N` | Armazenamento interno |
|-----|----------------------|
| 1   | `u64`                |
| 2   | `unsigned __int128`  |
| ≥3  | `std::array<u64, N>` |

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
