#!/usr/bin/env bash
set -euo pipefail

TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${DREAMPLACE_SOURCE_ORIGIN:?Set DREAMPLACE_SOURCE_ORIGIN to an official DREAMPlace 4.1.0 checkout}"
: "${DREAMPLACE_DATA_ORIGIN:?Set DREAMPLACE_DATA_ORIGIN to the prepared DREAMPlace benchmark directory}"
SOURCE_ORIGIN="$DREAMPLACE_SOURCE_ORIGIN"
DATA_ORIGIN="$DREAMPLACE_DATA_ORIGIN"
SOURCE_COPY="$TASK_ROOT/third_party/DREAMPlace-4.1.0-source"
DATA_COPY="$TASK_ROOT/datasets/dreamplace-official"
BUILD_DIR="$TASK_ROOT/third_party/DREAMPlace-4.1.0-build"
INSTALL_DIR="$TASK_ROOT/third_party/DREAMPlace-4.1.0-install"
PYTHON_BIN="${DREAMPLACE_PYTHON:-python}"
NVCC_BIN="${DREAMPLACE_NVCC:-nvcc}"
TOOLCHAIN_BIN="${DREAMPLACE_TOOLCHAIN_BIN:-$(dirname "$(command -v "$PYTHON_BIN")")}"
RESULT_ROOT="${LINKPLACE_RESULT_ROOT:-$TASK_ROOT/outputs/formal}"
BOOTSTRAP_DIR="$RESULT_ROOT/bootstrap"
BUILD_PATH="$(dirname "$NVCC_BIN"):$PATH"

mkdir -p "$TASK_ROOT/third_party" "$TASK_ROOT/datasets" "$BOOTSTRAP_DIR"

if [[ ! -f "$SOURCE_COPY/.copy-complete" ]]; then
    mkdir -p "$SOURCE_COPY"
    cp -a "$SOURCE_ORIGIN"/. "$SOURCE_COPY"/
    touch "$SOURCE_COPY/.copy-complete"
fi

if [[ ! -f "$DATA_COPY/.copy-complete" ]]; then
    mkdir -p "$DATA_COPY"
    cp -a "$DATA_ORIGIN"/. "$DATA_COPY"/
    touch "$DATA_COPY/.copy-complete"
fi

if git -C "$SOURCE_COPY" submodule status --recursive | grep -q '^-'; then
    git -C "$SOURCE_COPY" submodule update --init --recursive
fi

EXPECTED_COMMIT="5d13c9001a3bc900dca1e108e633d5dd45b00701"
ACTUAL_COMMIT="$(git -C "$SOURCE_COPY" rev-parse HEAD)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
    echo "unexpected DREAMPlace revision: $ACTUAL_COMMIT" >&2
    exit 2
fi

if [[ ! -f "$INSTALL_DIR/dreamplace/Placer.py" ]]; then
    mkdir -p "$BUILD_DIR" "$INSTALL_DIR"
    cd "$BUILD_DIR"
    PATH="$BUILD_PATH" cmake "$SOURCE_COPY" \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DPython_EXECUTABLE="$PYTHON_BIN" \
        -DCMAKE_CUDA_COMPILER="$NVCC_BIN" \
        -DCMAKE_CUDA_FLAGS="-gencode=arch=compute_80,code=compute_80" \
        -DBISON_EXECUTABLE="$TOOLCHAIN_BIN/bison" \
        -DFLEX_EXECUTABLE="$TOOLCHAIN_BIN/flex" \
        -DTCL_TCLSH="$TOOLCHAIN_BIN/tclsh" \
        -DCMAKE_CXX_ABI=0
    PATH="$BUILD_PATH" make -j8
    PATH="$BUILD_PATH" make install
fi

PYTHONPATH="$INSTALL_DIR:$INSTALL_DIR/dreamplace" "$PYTHON_BIN" -c \
    'import Placer, PlaceDB, NonLinearPlace; import dreamplace.configure; from dreamplace.ops.hpwl import hpwl' \
    >/dev/null
printf '%s\n' "$ACTUAL_COMMIT" > "$BOOTSTRAP_DIR/dreamplace.commit"
touch "$BOOTSTRAP_DIR/dreamplace-ready"
echo "DREAMPlace 4.1.0 ready at $INSTALL_DIR"
