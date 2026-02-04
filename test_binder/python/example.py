#!/usr/bin/env python3
"""Example: Kokkos dot product from Python."""

import mylib
import time

def main():
    mylib.initialize()
    print(f"Kokkos initialized: {mylib.is_initialized()}\n")

    # Create arrays from Python lists
    a = mylib.Array1D(5)
    b = mylib.Array1D(5)
    a.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
    b.from_list([5.0, 4.0, 3.0, 2.0, 1.0])

    # Compute dot product using Kokkos parallel_reduce
    result = mylib.dot_product(a, b)
    print(f"a = {a.to_list()}")
    print(f"b = {b.to_list()}")
    print(f"dot_product(a, b) = {result}")
    print()

    # Performance test
    n = 1_000_000
    large_a = mylib.Array1D(n)
    large_b = mylib.Array1D(n)

    # Fill using from_list (handles device sync internally)
    large_a.from_list([1.0] * n)
    large_b.from_list([2.0] * n)

    # Time 100 dot products
    start = time.perf_counter()
    for _ in range(100):
        result = mylib.dot_product(large_a, large_b)
    elapsed = time.perf_counter() - start

    print(f"100x dot_product({n} elements): {elapsed*1000:.2f} ms")
    print(f"Result: {result}")

if __name__ == "__main__":
    main()
