module HTensor.Types
  ( Shape
  , Tensor
  , TensorError(..)
  , mkTensorRO
  , tensorShape
  , indexRO
  )
where

import Foreign.ForeignPtr (ForeignPtr, withForeignPtr)
import Foreign.Storable (Storable, peekElemOff)

type Shape = (Int, Int)

data TensorError
  = InvalidShape Shape
  | IndexOutOfBounds Shape (Int, Int)
  deriving (Eq, Show)

data Tensor a = TensorRO Shape (ForeignPtr a)

mkTensorRO :: Storable a => Shape -> ForeignPtr a -> Either TensorError (Tensor a)
mkTensorRO shape@(rows, cols) fptr
  | rows <= 0 || cols <= 0 = Left (InvalidShape shape)
  | otherwise = Right (TensorRO shape fptr)

tensorShape :: Tensor a -> Shape
tensorShape (TensorRO shape _) = shape

indexRO :: Storable a => Tensor a -> (Int, Int) -> IO (Either TensorError a)
indexRO (TensorRO shape@(rows, cols) fptr) idx@(row, col)
  | row < 0 || col < 0 || row >= rows || col >= cols = pure (Left (IndexOutOfBounds shape idx))
  | otherwise = withForeignPtr fptr $ \ptr -> do
      let flatIx = row * cols + col
      val <- peekElemOff ptr flatIx
      pure (Right val)
