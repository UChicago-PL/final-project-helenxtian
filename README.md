# final-project-helenxtian
# H-Tensor: Zero-Copy Memory-Mapped Tensors in Haskell

## Overview

Modern AI systems often spend more time loading model weights from disk than performing inference. Large language models can easily exceed 10–100GB, and traditional loading requires reading the entire file into memory before computation can begin. This introduces startup latencies of tens of seconds.

This project proposes H-Tensor, a small Haskell library that enables zero-copy memory-mapped tensors using mmap. Instead of copying model weights into memory, the library will map a binary file directly into the process address space and rely on the operating system’s paging mechanism to load data lazily on demand. The result is near-instant startup time even for very large models.

This addresses a critical pain point in modern AI infrastructure where models spend more time loading than performing inference.

## Milestones

### Easy: Basic Memory-Mapped Loading
Goal: Implement a minimal working memory-mapped tensor abstraction\
Success Criteria: Program starts instantly regardless of file size; can read individual elements correctly
- Implement an mmap wrapper using:
    - Foreign.Ptr
    - Foreign.ForeignPtr
    - bracket for safe resource management
- Define a simple binary format:
    - 128-byte header (shape and dtype metadata)
    - Raw contiguous tensor data
- Parse the header using a Haskell parsing library (binary or attoparsec)
- Define a basic tensor type
    ```haskell
    data Tensor a = Tensor
        { tensorShape :: (Int, Int)
        , tensorData  :: ForeignPtr a
        }
    ```
- Implement safe indexing
    ```haskell 
    (!) :: Storable a => Tensor a -> (Int, Int) -> a 
    ```

### Medium: Type-Safe GADT Interface
Goal: Encode tensor invariants in the type system\
Success Criteria: Code attempting Tensor Float + Tensor Int fails at compile time; malformed files return descriptive errors
- Define a GADT that captures dtype at the type level
    ```haskell
    data DType a where
    DFloat32 :: DType Float
    DFloat64 :: DType Double

    data Tensor a where
    MkTensor :: Storable a
            => DType a
            -> (Int, Int)
            -> ForeignPtr a
            -> Tensor a
    ```
- Introduce structured error handling for invalid headers, type mismatches, or shape mismatches
    ```haskell
    ExceptT TensorError IO
    ``` 
- Implement shape-aware matrix multiplication:
    ```haskell
    matmul :: Tensor a -> Tensor a -> ExceptT TensorError IO (Tensor a)
    ```
- Add QuickCheck property tests:
    - Header round-trip correctness
    - Indexing consistency
    - Shape invariants preserved by operations

### Hard: SIMD-Accelerated Validation
Goal: Integrate optimized native code while maintaining Haskell interface\
Success Criteria: Checksum runs at RAM bandwidth (~20GB/s); detects single-bit corruption
- Implement an AVX2-accelerated checksum function in C
- Bind it via Haskell FFI:
    ```haskell
    foreign import ccall unsafe "simd_checksum"
        simd_checksum :: Ptr Word8 -> CSize -> IO Word64
    ```
-  Provide a safe wrapper:
    ```haskell
    validateTensor :: Tensor a -> IO Bool
    ```
- Benchmark performance using criterion
- Compare:
    - Pure Haskell fold
    - Strict ByteString fold
    - SIMD-accelerated FFI implementation

## Additional Topics
From Course: Parser combinators (Week 7), ExceptT (Week 8), Storable vectors (Week 9), GADTs\
Self-Study: mmap syscalls, Haskell Foreign library, C FFI, SIMD intrinsics\
Resources: Real World Haskell Ch. 17, mmap package docs, Intel Intrinsics Guide