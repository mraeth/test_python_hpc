#!/bin/bash
#
# Build Kokkos in _deps directory for use by test_binder and test_pykokkos
#
# Usage: ./build_kokkos.sh [options]
#
# Options:
#   --cuda     Enable CUDA backend (requires nvcc)
#   --hip      Enable HIP backend (requires hipcc)
#   --clean    Clean and rebuild from scratch
#
# Environment Variables (for GPU builds):
#   KOKKOS_ARCH      - GPU architecture (e.g., AMPERE80, VOLTA70, HOPPER90, MI250X)
#   CUDA_ROOT        - CUDA installation path (auto-detected if nvcc is in PATH)
#   ROCM_PATH        - ROCm installation path for HIP (default: /opt/rocm)
#
# Common NVIDIA architectures:
#   VOLTA70   - V100
#   AMPERE80  - A100
#   AMPERE86  - RTX 3090, A40
#   HOPPER90  - H100
#   ADA89     - RTX 4090, L40
#
# Common AMD architectures:
#   VEGA90A   - MI200 series
#   MI250X    - MI250X
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$SCRIPT_DIR/_deps"
KOKKOS_SRC="$DEPS_DIR/kokkos"
KOKKOS_BUILD="$DEPS_DIR/kokkos-build"
KOKKOS_INSTALL="$DEPS_DIR/kokkos-install"

KOKKOS_VERSION="4.5.01"
ENABLE_CUDA=OFF
ENABLE_HIP=OFF

# Parse arguments
for arg in "$@"; do
    case $arg in
        --cuda)
            ENABLE_CUDA=ON
            shift
            ;;
        --hip)
            ENABLE_HIP=ON
            shift
            ;;
        --clean)
            echo "Cleaning previous build..."
            rm -rf "$KOKKOS_SRC" "$KOKKOS_BUILD" "$KOKKOS_INSTALL"
            shift
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

# Validate options
if [ "$ENABLE_CUDA" = "ON" ] && [ "$ENABLE_HIP" = "ON" ]; then
    echo "Error: Cannot enable both CUDA and HIP"
    exit 1
fi

echo "============================================================"
echo "Building Kokkos $KOKKOS_VERSION"
echo "============================================================"
echo "Install directory: $KOKKOS_INSTALL"
echo "OpenMP: ON"
echo "CUDA: $ENABLE_CUDA"
echo "HIP: $ENABLE_HIP"

# Build architecture flags
ARCH_FLAGS=""

if [ "$ENABLE_CUDA" = "ON" ]; then
    # Auto-detect CUDA if not set
    if [ -z "$CUDA_ROOT" ]; then
        if command -v nvcc &> /dev/null; then
            CUDA_ROOT=$(dirname $(dirname $(which nvcc)))
            echo "Auto-detected CUDA_ROOT: $CUDA_ROOT"
        else
            echo "Error: nvcc not found. Set CUDA_ROOT or add nvcc to PATH"
            exit 1
        fi
    fi
    echo "CUDA_ROOT: $CUDA_ROOT"

    # GPU architecture
    if [ -n "$KOKKOS_ARCH" ]; then
        ARCH_FLAGS="-DKokkos_ARCH_${KOKKOS_ARCH}=ON"
        echo "GPU Architecture: $KOKKOS_ARCH"
    else
        echo "Warning: KOKKOS_ARCH not set. Kokkos will use default architecture."
        echo "  Set KOKKOS_ARCH for optimal performance (e.g., AMPERE80, VOLTA70)"
    fi
fi

if [ "$ENABLE_HIP" = "ON" ]; then
    ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    if [ ! -d "$ROCM_PATH" ]; then
        echo "Error: ROCm not found at $ROCM_PATH. Set ROCM_PATH environment variable."
        exit 1
    fi
    echo "ROCM_PATH: $ROCM_PATH"

    if [ -n "$KOKKOS_ARCH" ]; then
        ARCH_FLAGS="-DKokkos_ARCH_${KOKKOS_ARCH}=ON"
        echo "GPU Architecture: $KOKKOS_ARCH"
    else
        echo "Warning: KOKKOS_ARCH not set. Set it for optimal performance (e.g., MI250X, VEGA90A)"
    fi
fi

echo ""

# Create directories
mkdir -p "$DEPS_DIR"

# Clone Kokkos if not present
if [ ! -d "$KOKKOS_SRC" ]; then
    echo "Cloning Kokkos..."
    git clone --depth 1 --branch "$KOKKOS_VERSION" \
        https://github.com/kokkos/kokkos.git "$KOKKOS_SRC"
else
    echo "Kokkos source already exists at $KOKKOS_SRC"
fi

# Configure
echo ""
echo "Configuring Kokkos..."
mkdir -p "$KOKKOS_BUILD"
cd "$KOKKOS_BUILD"

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$KOKKOS_INSTALL"
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DCMAKE_CXX_STANDARD=17
    -DKokkos_ENABLE_OPENMP=ON
    -DKokkos_ENABLE_SERIAL=ON
    -DKokkos_ENABLE_CUDA=$ENABLE_CUDA
    -DKokkos_ENABLE_HIP=$ENABLE_HIP
)

# Add CUDA-specific flags
if [ "$ENABLE_CUDA" = "ON" ]; then
    CMAKE_ARGS+=(
        -DCMAKE_CUDA_COMPILER="$CUDA_ROOT/bin/nvcc"
        -DKokkos_ENABLE_CUDA_LAMBDA=ON
    )
fi

# Add HIP-specific flags
if [ "$ENABLE_HIP" = "ON" ]; then
    CMAKE_ARGS+=(
        -DCMAKE_CXX_COMPILER="$ROCM_PATH/bin/hipcc"
    )
fi

# Add architecture flags
if [ -n "$ARCH_FLAGS" ]; then
    CMAKE_ARGS+=($ARCH_FLAGS)
fi

cmake "$KOKKOS_SRC" "${CMAKE_ARGS[@]}"

# Build
echo ""
echo "Building Kokkos..."
make -j$(nproc)

# Install
echo ""
echo "Installing Kokkos..."
make install

echo ""
echo "============================================================"
echo "Kokkos installed successfully!"
echo "============================================================"
echo ""
echo "To use with CMake, add:"
echo "  -DKokkos_ROOT=$KOKKOS_INSTALL"
echo ""
echo "Or set environment variable:"
echo "  export Kokkos_ROOT=$KOKKOS_INSTALL"
echo ""
