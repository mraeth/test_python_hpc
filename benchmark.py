#!/usr/bin/env python3
"""
Benchmark: Kokkos (pybind11) vs JAX vs PyKokkos - Dot Product Performance.

All implementations use the same mylib interface.
"""

import sys
import time

# Import libraries with the same interface
sys.path.insert(0, "test_binder/python")
import mylib as kokkos_lib

sys.path.insert(0, "test_jax")
import mylib as jax_lib

sys.path.insert(0, "test_pykokkos")
import mylib as pykokkos_lib

# Number of iterations per benchmark
ITERATIONS = 100
WARMUP = 5


def benchmark(lib, n: int) -> float:
    """Benchmark dot product for a given library, return time in ms."""
    a = lib.Array1D(n)
    b = lib.Array1D(n)
    a.from_list([1.0] * n)
    b.from_list([2.0] * n)

    # Warmup
    for _ in range(WARMUP):
        lib.dot_product(a, b)

    # Benchmark
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        result = lib.dot_product(a, b)
    elapsed = time.perf_counter() - start

    return (elapsed / ITERATIONS) * 1000  # ms per call


def main():
    # Initialize all libraries
    kokkos_lib.initialize()
    jax_lib.initialize()
    pykokkos_lib.initialize()

    libraries = [
        ("Kokkos", kokkos_lib),
        ("JAX", jax_lib),
        ("PyKokkos", pykokkos_lib),
    ]

    print("=" * 70)
    print("Benchmark: Dot Product Performance Comparison")
    print("=" * 70)
    print(f"Iterations per size: {ITERATIONS}")
    print()

    # Array sizes to test
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000]

    # Build header
    header = f"{'Size':>12}"
    for name, _ in libraries:
        header += f" {name + ' (ms)':>14}"
    print(header)
    print("-" * len(header))

    # Run benchmarks
    for n in sizes:
        row = f"{n:>12,}"
        for name, lib in libraries:
            t = benchmark(lib, n)
            row += f" {t:>14.4f}"
        print(row)

    print("-" * len(header))
    print()

    # Correctness check
    print("Correctness check:")
    test_a = [1.0, 2.0, 3.0, 4.0, 5.0]
    test_b = [5.0, 4.0, 3.0, 2.0, 1.0]

    for name, lib in libraries:
        a = lib.Array1D(5)
        b = lib.Array1D(5)
        a.from_list(test_a)
        b.from_list(test_b)
        result = lib.dot_product(a, b)
        print(f"  {name:10}: dot({test_a}, {test_b}) = {result}")


if __name__ == "__main__":
    main()
