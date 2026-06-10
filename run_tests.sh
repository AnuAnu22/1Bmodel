#!/usr/bin/env bash
# run_tests.sh — run the full test suite from the project root.
#
# Usage:
#   bash run_tests.sh          # all tests
#   bash run_tests.sh -x       # stop on first failure
#   bash run_tests.sh -k mhc   # run only mHC tests
#
# Requirements:
#   pip install -r requirements.txt   (with jax[cpu] for local CPU runs)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=== DeepSeek-1B Test Suite ==="
echo "Python: $(python --version)"
echo "JAX:    $(python -c 'import jax; print(jax.__version__)')"
echo ""

# Use CPU-only backend for tests (avoids CUDA setup on dev machines).
# Remove or override JAX_PLATFORMS for GPU runs.
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

# Disable JIT for the first run to get clear Python tracebacks.
# Set JAX_DISABLE_JIT=0 to run JIT-compiled (faster but harder to debug).
export JAX_DISABLE_JIT="${JAX_DISABLE_JIT:-0}"

python -m pytest tests/ \
  --tb=short \
  --no-header \
  -v \
  "$@"
