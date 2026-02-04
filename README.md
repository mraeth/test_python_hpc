# C++/Python Hybrid Computing Comparison

This project compares three approaches for implementing high-performance numerical kernels that are called from Python:

1. **Kokkos + pybind11** - C++ kernels with Python bindings
2. **JAX** - Pure Python with XLA JIT compilation
3. **PyKokkos** - Python with Kokkos-style parallel constructs

All three implementations share the same Python API, allowing direct performance comparison.

## Project Structure

```
cpp_python_test/
├── README.md
├── benchmark.py              # Performance comparison script
├── build_kokkos.sh           # Build shared Kokkos installation
├── _deps/                    # Shared dependencies
│   └── kokkos-install/       # Pre-built Kokkos (after running build_kokkos.sh)
├── test_binder/              # Kokkos + pybind11 implementation
│   ├── CMakeLists.txt        # Build config (includes Binder integration)
│   ├── binder.config         # Binder configuration
│   ├── src/
│   │   ├── mylib.hpp         # C++ declarations
│   │   ├── mylib.cpp         # C++ implementations (Kokkos kernels)
│   │   └── all_includes.hpp  # Master include for Binder
│   ├── generated/
│   │   └── mylib.cpp         # pybind11 bindings (auto-generated or manual)
│   └── python/
│       └── mylib.*.so        # Compiled Python module
├── test_jax/                 # JAX implementation
│   └── mylib.py
└── test_pykokkos/            # PyKokkos implementation
    └── mylib.py
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

## Prerequisites

- CMake >= 3.16
- C++17 compiler with OpenMP support
- Python >= 3.8
- pybind11: `pip install pybind11`
- JAX: `pip install jax jaxlib`
- PyKokkos: `pip install pykokkos` (optional, requires pykokkos-base)

## Building

### 1. Build Kokkos (once, shared by all projects)

```bash
./build_kokkos.sh
```

Options:
- `--cuda` - Enable CUDA backend
- `--clean` - Clean rebuild

### 2. Build the Kokkos + pybind11 module

```bash
cd test_binder
mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=/path/to/python
make -j4
```

The compiled module is placed in `test_binder/python/`.

## Running the Benchmark

```bash
OMP_PROC_BIND=spread OMP_PLACES=threads python benchmark.py
```

Example output:

```
======================================================================
Benchmark: Dot Product Performance Comparison
======================================================================
Iterations per size: 100

        Size    Kokkos (ms)       JAX (ms)  PyKokkos (ms)
---------------------------------------------------------
       1,000         0.006         0.006         0.140
      10,000         0.097         0.028         0.474
     100,000         2.804         1.589         1.178
   1,000,000         8.140         8.380         9.139
  10,000,000        30.192        29.268        29.723
---------------------------------------------------------

Correctness check:
  Kokkos    : dot([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]) = 35.0
  JAX       : dot([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]) = 35.0
  PyKokkos  : dot([1.0, 2.0, 3.0, 4.0, 5.0], [5.0, 4.0, 3.0, 2.0, 1.0]) = 35.0
```

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

## Automatic Binding Generation with Binder

Instead of manually writing pybind11 bindings, you can use [Binder](https://github.com/RosettaCommons/binder) to auto-generate them from C++ headers.

### Installing Binder

```bash
# Option 1: Conda
conda install -c conda-forge binder

# Option 2: Build from source
git clone https://github.com/RosettaCommons/binder.git
cd binder && mkdir build && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=/path/to/install
make -j4 && make install
```

### Project Setup for Binder

Create a master include file that includes all headers to bind:

```cpp
// test_binder/src/all_includes.hpp
#include "mylib.hpp"
```

Create a Binder configuration file:

```
# test_binder/binder.config
+include <mylib.hpp>
+namespace ::
+class Array1D
+function initialize_kokkos
+function is_kokkos_initialized
+function dot_product
```

### Generating Bindings

```bash
cd test_binder

binder \
    --root-module mylib \
    --prefix generated/ \
    --config binder.config \
    --single-file \
    src/all_includes.hpp \
    -- \
    -std=c++17 \
    -I src \
    -I /path/to/kokkos/include \
    -I /path/to/pybind11/include
```

This generates `generated/mylib.cpp` with all pybind11 bindings.

### Workflow

1. Write C++ code in `src/mylib.hpp` and `src/mylib.cpp`
2. Run Binder to regenerate `generated/mylib.cpp`
3. Build with CMake

For large projects, integrate Binder into your CMake build:

```cmake
# Add custom command to regenerate bindings
add_custom_target(generate_bindings
    COMMAND binder
        --root-module mylib
        --prefix ${CMAKE_SOURCE_DIR}/generated/
        --config ${CMAKE_SOURCE_DIR}/binder.config
        --single-file
        ${CMAKE_SOURCE_DIR}/src/all_includes.hpp
        --
        -std=c++17
        -I ${CMAKE_SOURCE_DIR}/src
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Generating pybind11 bindings with Binder"
)
```

Then run: `make generate_bindings`

## Adding New Kernels

### Kokkos + pybind11 (Manual)

1. Add declaration to `test_binder/src/mylib.hpp`
2. Add implementation to `test_binder/src/mylib.cpp`
3. Add binding to `test_binder/generated/mylib.cpp`
4. Rebuild with `make`

### Kokkos + pybind11 (With Binder)

1. Add declaration to `test_binder/src/mylib.hpp`
2. Add implementation to `test_binder/src/mylib.cpp`
3. Update `binder.config` if needed
4. Run `binder ...` or `make generate_bindings`
5. Rebuild with `make`

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
