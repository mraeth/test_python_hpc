# C++/Python Hybrid Computing Comparison

This project compares three approaches for implementing high-performance numerical kernels that are called from Python:

1. **Kokkos + pybind11** - C++ kernels with auto-generated Python bindings (via Binder)
2. **JAX** - Pure Python with XLA JIT compilation
3. **PyKokkos** - Python with Kokkos-style parallel constructs

All three implementations share the same Python API, allowing direct performance comparison.

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/mraeth/test_python_hpc.git
cd test_python_hpc

# 2. Build Kokkos (only needed once)
./build_kokkos.sh

# 3. Build the Kokkos + pybind11 module (includes Binder build)
cd test_binder
mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python)
make -j4
cd ../..

# 4. Run the benchmark
OMP_PROC_BIND=spread OMP_PLACES=threads python benchmark.py
```

**Note:** The first build will compile Binder from source, which takes some time. Subsequent builds are fast.

## Project Structure

```
test_python_hpc/
├── README.md
├── benchmark.py              # Performance comparison script
├── build_kokkos.sh           # Build shared Kokkos installation
├── _deps/                    # Shared dependencies (generated)
│   └── kokkos-install/       # Pre-built Kokkos
├── test_binder/              # Kokkos + pybind11 implementation
│   ├── CMakeLists.txt        # Build config (builds Binder, generates bindings)
│   ├── binder.config         # Binder configuration
│   ├── src/
│   │   ├── mylib.hpp         # C++ declarations
│   │   ├── mylib.cpp         # C++ implementations (Kokkos kernels)
│   │   └── all_includes.hpp  # Master include for Binder
│   ├── generated/
│   │   └── mylib.cpp         # pybind11 bindings (auto-generated, committed)
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
- **LLVM and Clang development libraries** (for building Binder)
- pybind11: `pip install pybind11`

### For JAX/PyKokkos implementations

- JAX: `pip install jax jaxlib`
- PyKokkos: `pip install pykokkos`

### Installing LLVM/Clang

Binder requires LLVM and Clang libraries to build. Install them for your platform:

**Ubuntu/Debian:**
```bash
sudo apt install llvm-dev libclang-dev clang
```

**Fedora/RHEL:**
```bash
sudo dnf install llvm-devel clang-devel clang
```

**macOS (Homebrew):**
```bash
brew install llvm
# Add to PATH and set environment variables:
export PATH="/opt/homebrew/opt/llvm/bin:$PATH"
export LLVM_DIR="/opt/homebrew/opt/llvm/lib/cmake/llvm"
export Clang_DIR="/opt/homebrew/opt/llvm/lib/cmake/clang"
```

**Conda:**
```bash
conda install -c conda-forge llvmdev clangdev
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
mkdir build && cd build
cmake .. -DPython3_EXECUTABLE=$(which python)
make -j4
```

The build process:
1. Finds LLVM/Clang on your system
2. Builds Binder from source (first time only)
3. Generates pybind11 bindings from C++ headers
4. Compiles the Python module

The compiled module is placed in `test_binder/python/`.

### 3. Regenerating bindings

Bindings are automatically regenerated when you modify:
- `src/mylib.hpp`
- `src/all_includes.hpp`
- `binder.config`

Just run `make` and the bindings will be updated if needed.

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
| Bindings | Auto-generated (Binder) | None | None |
| Build | CMake + make | None | None |
| JIT | No | Yes (XLA) | Yes |
| GPU support | Yes (CUDA/HIP) | Yes (CUDA) | Yes (CUDA) |
| Autodiff | No | Yes | No |
| Best for | Existing C++ code | New Python projects | Kokkos-style in Python |

## Automatic Binding Generation with Binder

The project uses [Binder](https://github.com/RosettaCommons/binder) to auto-generate pybind11 bindings from C++ headers. Binder is built from source as part of the CMake build process.

### How it works

1. CMake finds LLVM/Clang on your system
2. Binder is downloaded and built from source (cached in `build/`)
3. Bindings are generated from `src/all_includes.hpp` using `binder.config`
4. The generated bindings are written to `generated/mylib.cpp`
5. The Python module is compiled with the generated bindings

### Configuration Files

**`src/all_includes.hpp`** - Master include file:
```cpp
#include "mylib.hpp"
```

**`binder.config`** - Controls what gets bound:
```
+include <mylib.hpp>
+namespace ::
+class Array1D
+function initialize_kokkos
+function is_kokkos_initialized
+function dot_product
-namespace Kokkos
```

## Adding New Kernels

### Kokkos + pybind11

1. Add declaration to `test_binder/src/mylib.hpp`
2. Add implementation to `test_binder/src/mylib.cpp`
3. Update `binder.config` to include new functions
4. Run `make` (bindings regenerate automatically)
5. Commit the updated `generated/mylib.cpp`

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
