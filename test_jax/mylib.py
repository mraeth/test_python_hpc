"""
JAX-based array library - mirrors the Kokkos mylib interface.

Install: pip install jax jaxlib
"""

import jax
import jax.numpy as jnp
from typing import List

# ============================================================================
# Initialization
# ============================================================================

_initialized = False


def initialize():
    """Initialize JAX runtime."""
    global _initialized
    _initialized = True


def is_initialized() -> bool:
    """Check if JAX is initialized."""
    return _initialized


# ============================================================================
# Array1D class - mirrors Kokkos Array1D interface
# ============================================================================

class Array1D:
    """1D array wrapper using JAX arrays internally."""

    def __init__(self, n: int):
        self._data = jnp.zeros(n)
        self._size = n

    def size(self) -> int:
        return self._size

    def get(self, i: int) -> float:
        return float(self._data[i])

    def set(self, i: int, val: float):
        self._data = self._data.at[i].set(val)

    def from_list(self, values: List[float]):
        if len(values) != self._size:
            raise RuntimeError("Size mismatch")
        self._data = jnp.array(values)

    def to_list(self) -> List[float]:
        return self._data.tolist()

    @property
    def data(self) -> jnp.ndarray:
        """Access underlying JAX array (for kernels)."""
        return self._data

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, i: int) -> float:
        return self.get(i)

    def __setitem__(self, i: int, val: float):
        self.set(i, val)


# ============================================================================
# Kernels (JIT compiled)
# ============================================================================

@jax.jit
def _dot_product_impl(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    """Internal JIT-compiled dot product using optimized BLAS."""
    # jnp.dot uses optimized BLAS (MKL on CPU, cuBLAS on GPU)
    return jnp.dot(a, b)


def dot_product(a: Array1D, b: Array1D) -> float:
    """Compute dot product of two arrays (parallel via XLA)."""
    if a.size() != b.size():
        raise RuntimeError("Size mismatch in dot_product")
    result = _dot_product_impl(a.data, b.data)
    # block_until_ready() ensures computation completes (important for GPU)
    result.block_until_ready()
    return float(result)


# Warmup function to pre-compile kernels
def warmup(n: int = 1000):
    """Pre-compile JIT kernels with dummy data."""
    a = Array1D(n)
    b = Array1D(n)
    a.from_list([1.0] * n)
    b.from_list([1.0] * n)
    _ = dot_product(a, b)
