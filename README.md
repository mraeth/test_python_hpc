# C++/Python Hybrid Computing Comparison

This project compares three approaches for implementing high-performance numerical kernels that are called from Python:

1. **Kokkos + pybind11** - C++ kernels with Python bindings
2. **JAX** - Pure Python with XLA JIT compilation
3. **PyKokkos** - Python with Kokkos-style parallel constructs

All three implementations share the same Python API, allowing direct performance comparison.

## Quick Start (CPU)

```bash
# 1. Install pybind11
pip install pybind11

# 2. Build Kokkos (only needed once)
./build_kokkos.sh

# 3. Build the Python module
cd test_binder
mkdir -p build && cd build
cmake ..
make -j4
cd ../..

# 4. Run the benchmark
OMP_PROC_BIND=spread OMP_PLACES=threads python benchmark.py
```

## Quick Start (GPU - NVIDIA CUDA)

```bash
# 1. Install pybind11 and JAX with CUDA
pip install pybind11
pip install --upgrade "jax[cuda12]"  # or cuda11 for older systems

# 2. Set environment variables for your GPU
export KOKKOS_ARCH=AMPERE80   # A100
# export KOKKOS_ARCH=VOLTA70  # V100
# export KOKKOS_ARCH=HOPPER90 # H100
# export KOKKOS_ARCH=ADA89    # RTX 4090, L40

# 3. Build Kokkos with CUDA (only needed once)
./build_kokkos.sh --clean --cuda

# 4. Build the Python module
cd test_binder
rm -rf build && mkdir build && cd build
cmake ..
make -j4
cd ../..

# 5. Run the benchmark
python benchmark.py
```

## Quick Start (GPU - AMD ROCm)

```bash
# 1. Install pybind11
pip install pybind11

# 2. Set environment variables for your GPU
export KOKKOS_ARCH=MI250X    # MI250X
# export KOKKOS_ARCH=VEGA90A # MI200 series
export ROCM_PATH=/opt/rocm   # default ROCm path

# 3. Build Kokkos with HIP (only needed once)
./build_kokkos.sh --clean --hip

# 4. Build the Python module
cd test_binder
rm -rf build && mkdir build && cd build
cmake ..
make -j4
cd ../..

# 5. Run the benchmark
python benchmark.py
```

### GPU Architecture Reference

| GPU | KOKKOS_ARCH |
|-----|-------------|
| NVIDIA T4 (GitHub CI) | `TURING75` |
| NVIDIA V100 | `VOLTA70` |
| NVIDIA A100 | `AMPERE80` |
| NVIDIA A40, RTX 3090 | `AMPERE86` |
| NVIDIA H100 | `HOPPER90` |
| NVIDIA L40, RTX 4090 | `ADA89` |
| AMD MI200 series | `VEGA90A` |
| AMD MI250X | `MI250X` |

## Project Structure

```
test_python_hpc/
├── README.md
├── benchmark.py              # Performance comparison script
├── build_kokkos.sh           # Build shared Kokkos installation
├── _deps/                    # Shared dependencies (generated)
│   └── kokkos-install/       # Pre-built Kokkos
├── test_binder/              # Kokkos + pybind11 implementation
│   ├── CMakeLists.txt        # Build configuration
│   ├── src/
│   │   ├── mylib.hpp         # C++ declarations
│   │   ├── mylib.cpp         # C++ implementations (Kokkos kernels)
│   │   └── bindings.cpp      # pybind11 bindings
│   └── python/               # Output directory (generated)
│       └── mylib.*.so        # Compiled Python module
├── test_jax/                 # JAX implementation
│   └── mylib.py
└── test_pykokkos/            # PyKokkos implementation
    └── mylib.py
```

## Prerequisites

### Required

- CMake >= 3.16
- C++17 compiler with OpenMP support
- Python >= 3.8
- pybind11: `pip install pybind11`

### For JAX implementation

- JAX (CPU): `pip install jax jaxlib`
- JAX (GPU): `pip install "jax[cuda12]"`

### For PyKokkos implementation (optional)

PyKokkos requires installation from source:

```bash
# Clone PyKokkos
git clone https://github.com/kokkos/pykokkos.git
cd pykokkos/

# Create conda environment
conda create -n pyk python=3.11 -y
conda env update -n pyk -f base/environment.yml
conda activate pyk

# Install (CPU with OpenMP)
python install_base.py install -- \
    -DENABLE_LAYOUTS=ON \
    -DENABLE_MEMORY_TRAITS=OFF \
    -DENABLE_VIEW_RANKS=3 \
    -DENABLE_CUDA=OFF \
    -DENABLE_OPENMP=ON

# Install (GPU with CUDA)
python install_base.py install -- \
    -DENABLE_LAYOUTS=ON \
    -DENABLE_MEMORY_TRAITS=OFF \
    -DENABLE_VIEW_RANKS=3 \
    -DENABLE_CUDA=ON \
    -DENABLE_OPENMP=ON
```

## Common API

All three implementations expose the same interface:

```python
import mylib

# Initialize runtime
mylib.initialize()

# Create arrays
a = mylib.Array1D(n)
b = mylib.Array1D(n)

# Initialize from Python lists
a.from_list([1.0, 2.0, 3.0, 4.0, 5.0])
b.from_list([5.0, 4.0, 3.0, 2.0, 1.0])

# Compute dot product (runs in parallel)
result = mylib.dot_product(a, b)

# Export back to Python
values = a.to_list()
```

## Building (Detailed)

### 1. Build Kokkos

Kokkos is built once and shared by all projects:

```bash
./build_kokkos.sh
```

Options:
- `--cuda` - Enable CUDA backend
- `--clean` - Clean rebuild

This creates `_deps/kokkos-install/`.

### 2. Build the Kokkos + pybind11 module

```bash
cd test_binder
mkdir -p build && cd build
cmake ..
make -j4
```

The compiled module is placed in `test_binder/python/`.

## Running the Benchmark

```bash
python benchmark.py
```

### CPU Results (OpenMP)

```
        Size  Kokkos (ms)     JAX (ms) PyKokkos (ms)   NumPy (ms)  Python (ms)
------------------------------------------------------------------------------
       1,000       0.0017       0.0025       0.0028       0.0007       0.0360
      10,000       0.0025       0.0024       0.0024       0.0018       0.3206
     100,000       0.0155       0.0073       0.0076       0.0031       3.4104
   1,000,000       0.0929       0.0886       0.0998       0.0300      32.3268
  10,000,000       3.2805       3.3127       3.2148       3.0832       (skip)
------------------------------------------------------------------------------
```

### GPU Results (NVIDIA A100)

```
        Size  Kokkos (ms)     JAX (ms) PyKokkos (ms)   NumPy (ms)  Python (ms)
------------------------------------------------------------------------------
       1,000       0.0211       0.0209       0.0211       0.0013       0.0611
      10,000       0.0211       0.0211       0.0210       0.0047       0.6312
     100,000       0.0225       0.0223       0.0224       0.0158       6.4613
   1,000,000       0.0268       0.0265       0.0266       0.1025      65.3419
  10,000,000       0.1652       0.1666       0.1652       1.2505       (skip)
------------------------------------------------------------------------------
```

### Note on OpenMP Initialization

When using the Kokkos module (compiled with `-O3`), it initializes the OpenMP runtime at import time. This initialization affects **all** OpenMP-based libraries in the Python process, including JAX and PyKokkos. Compiling the Kokkos module with `-O3` is critical for optimal performance - without it, all OpenMP-based libraries will run significantly slower (up to 7x) due to suboptimal OpenMP runtime configuration. The CMakeLists.txt includes `-O3` by default.

## Comparison

| Aspect | Kokkos + pybind11 | JAX | PyKokkos |
|--------|-------------------|-----|----------|
| Language | C++ | Python | Python |
| Bindings | Manual (pybind11) | None | None |
| Build | CMake + make | None | None |
| JIT | No | Yes (XLA) | Yes |
| GPU support | Yes (CUDA/HIP) | Yes (CUDA) | Yes (CUDA) |
| Autodiff | No | Yes | No |
| Best for | Existing C++ code | New Python projects | Kokkos-style in Python |

## Adding New Kernels

### Kokkos + pybind11

1. Add declaration to `test_binder/src/mylib.hpp`
2. Add implementation to `test_binder/src/mylib.cpp`
3. Add pybind11 bindings to `test_binder/src/bindings.cpp`
4. Run `make`

Example - adding a new function:

**mylib.hpp:**
```cpp
double sum_array(const Array1D& a);
```

**mylib.cpp:**
```cpp
double sum_array(const Array1D& a) {
    auto va = a.view();
    double result = 0.0;
    Kokkos::parallel_reduce("sum", a.size(),
        KOKKOS_LAMBDA(const int i, double& lsum) {
            lsum += va(i);
        },
        result
    );
    return result;
}
```

**bindings.cpp:**
```cpp
m.def("sum_array", &sum_array, py::arg("a"),
      "Compute sum of array elements");
```

### JAX

Add function to `test_jax/mylib.py`:

```python
@jax.jit
def _my_kernel_impl(a, b):
    return jnp.sum(a * b)

def my_kernel(a: Array1D, b: Array1D) -> float:
    return float(_my_kernel_impl(a.data, b.data))
```

### PyKokkos

Add kernel to `test_pykokkos/mylib.py`:

```python
@pk.workunit
def _my_kernel(i: int, acc: pk.Acc[pk.double], a: pk.View1D[pk.double], b: pk.View1D[pk.double]):
    acc += a[i] * b[i]

def my_kernel(a: Array1D, b: Array1D) -> float:
    return float(pk.parallel_reduce(a.size(), _my_kernel, a=a.data, b=b.data))
```

## License

MIT
