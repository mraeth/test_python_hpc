#!/usr/bin/env python3
"""
Example: JAX dot product - equivalent to the Kokkos example.

JAX compiles Python to optimized kernels via XLA (no C++ needed).

Install: pip install jax jaxlib
"""

import jax
import jax.numpy as jnp
import time

def main():
    print(f"JAX version: {jax.__version__}")
    print(f"Devices: {jax.devices()}")
    print()

    # Create arrays (JAX arrays are immutable, like NumPy)
    a = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
    b = jnp.array([5.0, 4.0, 3.0, 2.0, 1.0])

    # Dot product - JAX compiles this to optimized code
    result = jnp.dot(a, b)
    print(f"a = {a.tolist()}")
    print(f"b = {b.tolist()}")
    print(f"jnp.dot(a, b) = {result}")
    print()

    # =========================================================================
    # JIT compilation for custom kernels
    # =========================================================================

    # Define a custom dot product function
    @jax.jit  # JIT compile to XLA
    def dot_product(x, y):
        return jnp.sum(x * y)

    # First call compiles, subsequent calls are fast
    _ = dot_product(a, b)  # Warm-up / compile
    result = dot_product(a, b)
    print(f"Custom dot_product(a, b) = {result}")
    print()

    # =========================================================================
    # Performance test
    # =========================================================================
    n = 1_000_000
    large_a = jnp.ones(n)
    large_b = jnp.full(n, 2.0)

    # Warm-up (JIT compilation happens here)
    _ = dot_product(large_a, large_b).block_until_ready()

    # Time 100 dot products
    start = time.perf_counter()
    for _ in range(100):
        result = dot_product(large_a, large_b)
    result.block_until_ready()  # Ensure computation is done
    elapsed = time.perf_counter() - start

    print(f"100x dot_product({n} elements): {elapsed*1000:.2f} ms")
    print(f"Result: {result}")
    print()

    # =========================================================================
    # Bonus: Automatic differentiation (not possible with plain Kokkos)
    # =========================================================================

    @jax.jit
    def f(x):
        """Function whose gradient we want."""
        return jnp.sum(x ** 2)

    # Compute gradient automatically
    grad_f = jax.grad(f)

    x = jnp.array([1.0, 2.0, 3.0])
    print(f"f(x) = sum(x^2) where x = {x.tolist()}")
    print(f"f(x) = {f(x)}")
    print(f"grad(f)(x) = {grad_f(x).tolist()}")  # Should be 2*x

if __name__ == "__main__":
    main()
