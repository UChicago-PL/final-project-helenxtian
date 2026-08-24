#!/usr/bin/env bash
set -euo pipefail

SIZE="${SIZE:-8192}"
ITERATIONS="${ITERATIONS:-5}"
SKIP_CPU="${SKIP_CPU:-1}"
export PATH="$HOME/.ghcup/bin:$PATH"

if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found; run scripts/setup-b200.sh first." >&2
  exit 1
fi

readonly CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
CUDA_LIB_DIR="${CUDA_LIB_DIR:-$CUDA_HOME/lib64}"
export LD_LIBRARY_PATH="$CUDA_LIB_DIR:${LD_LIBRARY_PATH:-}"
mkdir -p benchmark-results

CABAL_CUDA_ARGS=(
  -fcuda
  "--extra-include-dirs=$CUDA_HOME/include"
  "--extra-lib-dirs=$CUDA_LIB_DIR"
)

cabal update
cabal test all -f-cuda
cabal build bench:htensor-bench "${CABAL_CUDA_ARGS[@]}"
BENCHMARK_BIN="$(cabal list-bin bench:htensor-bench "${CABAL_CUDA_ARGS[@]}")"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_PATH="benchmark-results/b200-${SIZE}-${TIMESTAMP}.csv"

BENCHMARK_ARGS=(
  --size "$SIZE"
  --iterations "$ITERATIONS"
)
if [[ "$SKIP_CPU" == "1" ]]; then
  BENCHMARK_ARGS+=(--skip-cpu)
fi

"$BENCHMARK_BIN" "${BENCHMARK_ARGS[@]}" > "$RESULT_PATH"

cat "$RESULT_PATH"
echo "Saved benchmark results to $RESULT_PATH"