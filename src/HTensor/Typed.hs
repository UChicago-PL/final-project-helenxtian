{-# LANGUAGE DataKinds #-}
{-# LANGUAGE ForeignFunctionInterface #-}
{-# LANGUAGE GADTs #-}
{-# LANGUAGE KindSignatures #-}
{-# LANGUAGE ScopedTypeVariables #-}
{-# LANGUAGE TypeApplications #-}

module HTensor.Typed
  ( DType(..)
  , TypedTensor
  , SomeTypedTensor(..)
  , mkTypedTensorRO
  , loadTypedTensorFloatRO
  , typedTensorShape
  , indexTypedRO
  , matmulFloat
  )
where

import Data.Proxy (Proxy(..))
import Foreign.C.Types (CSize(..))
import Foreign.ForeignPtr (ForeignPtr, mallocForeignPtrArray, withForeignPtr)
import Foreign.Ptr (Ptr)
import Foreign.Storable (Storable, peekElemOff)
import GHC.TypeNats (KnownNat, Nat, SomeNat(..), natVal, someNatVal)
import HTensor.Types
  ( TensorError(..)
  , loadTensorFloatRO
  , tensorForeignPtr
  , tensorShape
  )

data DType a where
  Float32 :: DType Float
  Float64 :: DType Double

data TypedTensor (rows :: Nat) (cols :: Nat) a where
  TypedTensorRO
    :: (KnownNat rows, KnownNat cols, Storable a)
    => DType a
    -> ForeignPtr a
    -> TypedTensor rows cols a

data SomeTypedTensor a where
  SomeTypedTensor :: TypedTensor rows cols a -> SomeTypedTensor a

mkTypedTensorRO
  :: forall rows cols a
   . (KnownNat rows, KnownNat cols, Storable a)
  => DType a
  -> ForeignPtr a
  -> Either TensorError (TypedTensor rows cols a)
mkTypedTensorRO dtype fptr
  | rows <= 0 || cols <= 0 = Left (InvalidShape (rows, cols))
  | otherwise = Right (TypedTensorRO dtype fptr)
  where
    rows = fromIntegral (natVal (Proxy @rows))
    cols = fromIntegral (natVal (Proxy @cols))

loadTypedTensorFloatRO :: FilePath -> IO (Either TensorError (SomeTypedTensor Float))
loadTypedTensorFloatRO path = do
  loaded <- loadTensorFloatRO path
  pure $ do
    tensor <- loaded
    let (rows, cols) = tensorShape tensor
    case (someNatVal (fromIntegral rows), someNatVal (fromIntegral cols)) of
      (SomeNat (_ :: Proxy rows), SomeNat (_ :: Proxy cols)) ->
        SomeTypedTensor <$> mkTypedTensorRO @rows @cols Float32 (tensorForeignPtr tensor)

typedTensorShape :: forall rows cols a. TypedTensor rows cols a -> (Int, Int)
typedTensorShape (TypedTensorRO _ _) =
  ( fromIntegral (natVal (Proxy @rows))
  , fromIntegral (natVal (Proxy @cols))
  )

indexTypedRO
  :: forall rows cols a
   . TypedTensor rows cols a
  -> (Int, Int)
  -> IO (Either TensorError a)
indexTypedRO tensor@(TypedTensorRO _ fptr) index@(row, col)
  | row < 0 || col < 0 || row >= rows || col >= cols =
      pure (Left (IndexOutOfBounds (rows, cols) index))
  | otherwise = withForeignPtr fptr $ \ptr ->
      Right <$> peekElemOff ptr (row * cols + col)
  where
    (rows, cols) = typedTensorShape tensor

matmulFloat
  :: TypedTensor rows inner Float
  -> TypedTensor inner cols Float
  -> IO (TypedTensor rows cols Float)
matmulFloat leftTensor@(TypedTensorRO Float32 left) rightTensor@(TypedTensorRO Float32 right) = do
  result <- mallocForeignPtrArray (rows * cols)
  withForeignPtr left $ \leftPtr ->
    withForeignPtr right $ \rightPtr ->
      withForeignPtr result $ \resultPtr ->
        c_matmulFloat
          leftPtr
          rightPtr
          resultPtr
          (fromIntegral rows)
          (fromIntegral inner)
          (fromIntegral cols)
  pure (TypedTensorRO Float32 result)
  where
    (rows, inner) = typedTensorShape leftTensor
    (_, cols) = typedTensorShape rightTensor

foreign import ccall unsafe "htensor_matmul_f32"
  c_matmulFloat
    :: Ptr Float
    -> Ptr Float
    -> Ptr Float
    -> CSize
    -> CSize
    -> CSize
    -> IO ()