#!/bin/bash
# Script to generate pybind11 bindings using binder
#
# Prerequisites:
#   conda install -c conda-forge binder
# OR build from source:
#   https://github.com/RosettaCommons/binder

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SRC_DIR="$SCRIPT_DIR/src"
OUT_DIR="$SCRIPT_DIR/generated"

mkdir -p "$OUT_DIR"

# Run binder to generate bindings
binder \
    --root-module mylib \
    --prefix "$OUT_DIR/" \
    --config "$SCRIPT_DIR/binder_config.cfg" \
    --single-file \
    "$SRC_DIR/all_includes.hpp" \
    -- \
    -std=c++17 \
    -I"$SRC_DIR" \
    -DNDEBUG

echo "Bindings generated in $OUT_DIR/"
