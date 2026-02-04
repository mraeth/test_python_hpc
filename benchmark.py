#!/usr/bin/env python3
"""
Benchmark: Dot Product Performance Comparison.

Compares:
- Kokkos + pybind11 (C++ with Python bindings)
- JAX (XLA JIT compilation)
- PyKokkos (Python with Kokkos-style parallelism)
- NumPy (optimized C library)
- Pure Python (naive list iteration)
"""

import sys
import time

import numpy as np

# Import libraries with the same interface
sys.path.insert(0, "test_binder/python")
import mylib as kokkos_lib

sys.path.insert(0, "test_jax")
import mylib as jax_lib

sys.path.insert(0, "test_pykokkos")
import mylib as pykokkos_lib


# ============================================================================
# NumPy wrapper (same interface as other libraries)
# ============================================================================
class NumpyArray1D:
    def __init__(self, n: int):
        self.data = np.zeros(n, dtype=np.float64)

    def from_list(self, values: list):
        self.data = np.array(values, dtype=np.float64)

    def to_list(self) -> list:
        return self.data.tolist()


class NumpyLib:
    @staticmethod
    def initialize():
        pass  # NumPy needs no initialization

    @staticmethod
    def Array1D(n: int) -> NumpyArray1D:
        return NumpyArray1D(n)

    @staticmethod
    def dot_product(a: NumpyArray1D, b: NumpyArray1D) -> float:
        return float(np.dot(a.data, b.data))


numpy_lib = NumpyLib()


# ============================================================================
# Pure Python wrapper (naive list iteration)
# ============================================================================
class PythonArray1D:
    def __init__(self, n: int):
        self.data = [0.0] * n

    def from_list(self, values: list):
        self.data = list(values)

    def to_list(self) -> list:
        return self.data


class PythonLib:
    @staticmethod
    def initialize():
        pass  # Pure Python needs no initialization

    @staticmethod
    def Array1D(n: int) -> PythonArray1D:
        return PythonArray1D(n)

    @staticmethod
    def dot_product(a: PythonArray1D, b: PythonArray1D) -> float:
        result = 0.0
        for i in range(len(a.data)):
            result += a.data[i] * b.data[i]
        return result


python_lib = PythonLib()


# ============================================================================
# Benchmark
# ============================================================================
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
    numpy_lib.initialize()
    python_lib.initialize()

    libraries = [
        ("Kokkos", kokkos_lib),
        ("JAX", jax_lib),
        ("PyKokkos", pykokkos_lib),
        ("NumPy", numpy_lib),
        ("Python", python_lib),
    ]

    print("=" * 90)
    print("Benchmark: Dot Product Performance Comparison")
    print("=" * 90)
    print(f"Iterations per size: {ITERATIONS}")
    print()

    # Array sizes to test (smaller max for pure Python)
    sizes = [1_000, 10_000, 100_000, 1_000_000, 10_000_000, 100_000_000]

    # Build header
    header = f"{'Size':>12}"
    for name, _ in libraries:
        header += f" {name + ' (ms)':>12}"
    print(header)
    print("-" * len(header))

    # Run benchmarks
    for n in sizes:
        row = f"{n:>12,}"
        for name, lib in libraries:
            # Skip pure Python for very large arrays (too slow)
            if name == "Python" and n > 1_000_000:
                row += f" {'(skip)':>12}"
            else:
                t = benchmark(lib, n)
                row += f" {t:>12.4f}"
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
