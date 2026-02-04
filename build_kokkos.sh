#!/bin/bash
#
# Build Kokkos in _deps directory for use by test_binder and test_pykokkos
#
# Usage: ./build_kokkos.sh [options]
#
# Options:
#   --cuda     Enable CUDA backend (requires nvcc)
#   --clean    Clean and rebuild from scratch
#

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPS_DIR="$SCRIPT_DIR/_deps"
KOKKOS_SRC="$DEPS_DIR/kokkos"
KOKKOS_BUILD="$DEPS_DIR/kokkos-build"
KOKKOS_INSTALL="$DEPS_DIR/kokkos-install"

KOKKOS_VERSION="4.5.01"
ENABLE_CUDA=OFF

# Parse arguments
for arg in "$@"; do
    case $arg in
        --cuda)
            ENABLE_CUDA=ON
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

echo "============================================================"
echo "Building Kokkos $KOKKOS_VERSION"
echo "============================================================"
echo "Install directory: $KOKKOS_INSTALL"
echo "OpenMP: ON"
echo "CUDA: $ENABLE_CUDA"
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

cmake "$KOKKOS_SRC" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$KOKKOS_INSTALL" \
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON \
    -DCMAKE_CXX_STANDARD=17 \
    -DKokkos_ENABLE_OPENMP=ON \
    -DKokkos_ENABLE_SERIAL=ON \
    -DKokkos_ENABLE_CUDA=$ENABLE_CUDA

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
