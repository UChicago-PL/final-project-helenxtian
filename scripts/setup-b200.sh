#!/usr/bin/env bash
set -euo pipefail

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "setup-b200.sh must be run on Linux" >&2
  exit 1
fi

if ! command -v apt-get >/dev/null 2>&1; then
  echo "This setup script currently supports Debian/Ubuntu systems with apt-get." >&2
  exit 1
fi

sudo apt-get update
sudo apt-get install -y build-essential curl libffi-dev libgmp-dev libncurses-dev pkg-config

if ! command -v ghcup >/dev/null 2>&1 && [[ ! -x "$HOME/.ghcup/bin/ghcup" ]]; then
  curl --proto '=https' --tlsv1.2 -sSf https://get-ghcup.haskell.org |
    BOOTSTRAP_HASKELL_NONINTERACTIVE=1 \
    BOOTSTRAP_HASKELL_MINIMAL=1 \
    sh
fi

export PATH="$HOME/.ghcup/bin:$PATH"
ghcup install ghc recommended --set
ghcup install cabal recommended --set

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi was not found. Install an NVIDIA driver before running the GPU benchmark." >&2
  exit 1
fi
if ! command -v nvcc >/dev/null 2>&1; then
  echo "nvcc was not found. Install the CUDA toolkit before running the GPU benchmark." >&2
  exit 1
fi

readonly CUDA_HOME="${CUDA_HOME:-$(dirname "$(dirname "$(command -v nvcc)")")}"
if [[ ! -f "$CUDA_HOME/include/cublas_v2.h" ]]; then
  echo "cuBLAS headers were not found under $CUDA_HOME/include." >&2
  exit 1
fi
if [[ ! -d "$CUDA_HOME/lib64" ]]; then
  echo "CUDA libraries were not found under $CUDA_HOME/lib64." >&2
  exit 1
fi

echo "Haskell: $(ghc --numeric-version)"
echo "Cabal: $(cabal --numeric-version)"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "CUDA_HOME=$CUDA_HOME"
echo "Setup complete. Run scripts/run-b200-benchmark.sh from the repository root."