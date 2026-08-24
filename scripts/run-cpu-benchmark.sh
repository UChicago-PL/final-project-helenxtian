#!/usr/bin/env bash
set -euo pipefail

SIZE="${SIZE:-512}"
ITERATIONS="${ITERATIONS:-3}"
mkdir -p benchmark-results

cabal test all -f-cuda
cabal build bench:htensor-bench -f-cuda
BENCHMARK_BIN="$(cabal list-bin bench:htensor-bench -f-cuda)"
TIMESTAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULT_PATH="benchmark-results/cpu-${SIZE}-${TIMESTAMP}.csv"

"$BENCHMARK_BIN" \
  --size "$SIZE" \
  --iterations "$ITERATIONS" \
  > "$RESULT_PATH"

cat "$RESULT_PATH"
echo "Saved benchmark results to $RESULT_PATH"