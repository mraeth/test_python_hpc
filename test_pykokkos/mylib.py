"""
PyKokkos-based array library - mirrors the Kokkos mylib interface.

PyKokkos provides Kokkos-like parallel programming in Python with JIT compilation.

Install (requires conda or building from source):
    conda install -c conda-forge pykokkos
    # or
    pip install git+https://github.com/kokkos/pykokkos.git

Note: PyKokkos requires pykokkos-base which compiles Kokkos C++ code.
"""

import pykokkos as pk
from typing import List

# ============================================================================
# Initialization
# ============================================================================

_initialized = False


def initialize():
    """Initialize PyKokkos runtime."""
    global _initialized
    pk.set_default_space(pk.OpenMP)  # or pk.Cuda for GPU
    _initialized = True


def is_initialized() -> bool:
    """Check if PyKokkos is initialized."""
    return _initialized


# ============================================================================
# Array1D class - mirrors Kokkos Array1D interface
# ============================================================================

class Array1D:
    """1D array wrapper using PyKokkos View internally."""

    def __init__(self, n: int):
        self._size = n
        self._data = pk.View([n], dtype=pk.double)

    def size(self) -> int:
        return self._size

    def get(self, i: int) -> float:
        return float(self._data[i])

    def set(self, i: int, val: float):
        self._data[i] = val

    def from_list(self, values: List[float]):
        if len(values) != self._size:
            raise RuntimeError("Size mismatch")
        for i, v in enumerate(values):
            self._data[i] = v

    def to_list(self) -> List[float]:
        return [float(self._data[i]) for i in range(self._size)]

    @property
    def data(self):
        """Access underlying PyKokkos View (for kernels)."""
        return self._data

    def __len__(self) -> int:
        return self._size

    def __getitem__(self, i: int) -> float:
        return self.get(i)

    def __setitem__(self, i: int, val: float):
        self.set(i, val)


# ============================================================================
# Kernels
# ============================================================================

@pk.workunit
def _dot_kernel(i: int, acc: pk.Acc[pk.double], a: pk.View1D[pk.double], b: pk.View1D[pk.double]):
    """PyKokkos parallel_reduce kernel for dot product."""
    acc += a[i] * b[i]


def dot_product(a: Array1D, b: Array1D) -> float:
    """Compute dot product of two arrays (parallel via PyKokkos)."""
    if a.size() != b.size():
        raise RuntimeError("Size mismatch in dot_product")

    result = pk.parallel_reduce(a.size(), _dot_kernel, a=a.data, b=b.data)
    return float(result)


# Warmup function
def warmup(n: int = 1000):
    """Pre-compile kernels with dummy data."""
    a = Array1D(n)
    b = Array1D(n)
    a.from_list([1.0] * n)
    b.from_list([1.0] * n)
    _ = dot_product(a, b)
