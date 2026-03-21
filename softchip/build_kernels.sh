#!/bin/bash
# Build the production ternary matmul kernel.
# This is the only kernel required for training with USE_GHOST=False.
#
# Output: softchip/ternary_matmul_v3.so
#
# Requires: gcc with AVX2+FMA support (all x86 CPUs since ~2013)
# Tested on: AMD Ryzen 5 PRO 5675U (Zen 3), gcc 12+

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Building ternary_matmul_v3.so..."
gcc -O3 -mavx2 -mfma -march=native -fopenmp -shared -fPIC \
    -o "${SCRIPT_DIR}/ternary_matmul_v3.so" \
    "${SCRIPT_DIR}/ternary_matmul_v3.c" \
    -lm
echo "  Done: ${SCRIPT_DIR}/ternary_matmul_v3.so"

# Optional: also build the standalone benchmark binary
if [ "${1}" = "--bench" ]; then
    echo "Building standalone benchmark..."
    gcc -O3 -mavx2 -mfma -march=native -fopenmp -DSTANDALONE_TEST \
        -o "${SCRIPT_DIR}/ternary_bench_v3" \
        "${SCRIPT_DIR}/ternary_matmul_v3.c" \
        -lm
    echo "  Done: ${SCRIPT_DIR}/ternary_bench_v3"
    echo "  Run: ${SCRIPT_DIR}/ternary_bench_v3"
fi

echo "Build complete. Verify with: python test_softchip_accuracy.py"
