# H-Tensor Benchmarking

## What Is Measured

The benchmark compares the existing CPU C FFI matrix multiplication with an optional CUDA/cuBLAS implementation. It creates two square Float32 tensor files with known diagonal values, maps both files through `loadTensorFloatRO`, runs matrix multiplication, and verifies the first and last diagonal outputs after every iteration.

Generated tensor files are sparse except for their headers and diagonal values. This keeps setup practical for large matrices while still forcing the matrix kernels to process the full dimensions. The benchmark is intended to compare loading, transfer, and compute stages, not storage-device read throughput.

The data path is:

```text
tensor file -> mmap host address -> CUDA host-to-device copy -> B200 -> cuBLAS SGEMM
```

The tensor payload is not copied into a Haskell-managed array during loading. CUDA still copies the mapped host data into device memory. Do not describe the GPU path as end-to-end zero-copy.

## Clone and Prepare

Commit and push all project changes before cloning on the remote machine. Do not copy `dist-newstyle`, because Cabal must rebuild native objects for Linux and the remote CPU architecture.

```sh
git clone <repository-url>
cd final-project-helenxtian
scripts/setup-b200.sh
```

The setup script:

- Supports Debian or Ubuntu systems using `apt-get`.
- Installs build prerequisites and a current GHC/Cabal toolchain through GHCup.
- Verifies `nvidia-smi`, `nvcc`, CUDA headers, and cuBLAS headers.
- Assumes the NVIDIA driver and CUDA toolkit are already installed by the machine administrator.

If CUDA is installed outside the `nvcc` prefix, export both paths explicitly:

```sh
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_LIB_DIR="$CUDA_HOME/lib64"
```

## Run on B200

The default B200 run uses an $8192 \times 8192$ matrix, performs five iterations, and skips the scalar CPU kernel:

```sh
scripts/run-b200-benchmark.sh
```

Override settings with environment variables:

```sh
SIZE=16384 ITERATIONS=10 scripts/run-b200-benchmark.sh
```

For a direct CPU/GPU comparison, choose a smaller matrix and enable CPU execution:

```sh
SIZE=1024 ITERATIONS=5 SKIP_CPU=0 scripts/run-b200-benchmark.sh
```

The script builds with Cabal's `cuda` flag, supplies CUDA include/library directories, runs all regular tests, and writes a timestamped CSV under `benchmark-results/`.

## CPU-Only Use

Normal builds do not compile or link CUDA:

```sh
cabal test all
scripts/run-cpu-benchmark.sh
```

Custom CPU benchmark settings use the same environment variables:

```sh
SIZE=768 ITERATIONS=5 scripts/run-cpu-benchmark.sh
```

The explicit Cabal forms are:

```sh
# CUDA disabled
cabal build bench:htensor-bench -f-cuda

# CUDA enabled
cabal build bench:htensor-bench -fcuda \
  --extra-include-dirs="$CUDA_HOME/include" \
  --extra-lib-dirs="$CUDA_LIB_DIR"
```

## CSV Columns

| Column | Meaning |
| --- | --- |
| `backend` | `mmap`, `cpu`, or `gpu` |
| `size` | Rows and columns of each square matrix |
| `iteration` | Benchmark iteration; mmap setup uses zero |
| `mmap_ms` | Time to map and parse both tensor files |
| `h2d_ms` | CUDA-event time for both host-to-device copies |
| `kernel_ms` | CPU wall time or cuBLAS SGEMM CUDA-event time |
| `d2h_ms` | CUDA-event time for the result copy to host |
| `device_total_ms` | CUDA event span from first copy through result copy |
| `end_to_end_ms` | Full host-observed call, including GPU allocation and cuBLAS setup |

Lines beginning with `#` are metadata, including the detected CUDA device name.

## Interpretation

- Compare `mmap_ms` with payload size to discuss startup behavior, not full file-read bandwidth. `mmap` establishes mappings lazily.
- Compare `device_total_ms` with `end_to_end_ms` to expose allocation, context, and library setup overhead.
- Compare CPU and GPU rows only at the same size and with the same generated files.
- The CPU kernel is a portable scalar baseline. It is not BLAS-optimized, so the comparison measures this implementation against cuBLAS rather than peak CPU performance.
- The CUDA wrapper uses row-major Float32 inputs and transforms the cuBLAS call so its result matches row-major $C = AB$.

## Validation Status

The CPU-only build, tests, benchmark runner, CSV output, and result checks are validated on macOS with GHC 9.14.1. The CUDA source is isolated behind the `cuda` flag and cannot be compiled or executed on a machine without the CUDA toolkit and NVIDIA GPU. Run the B200 script after cloning to complete CUDA compile, link, device, and numerical validation.