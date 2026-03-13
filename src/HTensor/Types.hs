module HTensor.Types
  ( Shape
  , Tensor
  , TensorError(..)
  , mkTensorRO
  , loadTensorFloatRO
  , withTensorFloatRO
  , tensorShape
  , indexRO
  , (!?)
  , (!)
  )
where

import Control.Exception (bracket)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import Data.Word (Word8)
import Foreign.ForeignPtr (ForeignPtr, plusForeignPtr, withForeignPtr)
import Foreign.Ptr (castPtr)
import Foreign.Storable (Storable, peekElemOff)
import HTensor.Header (Header(..), headerSize, parseHeader)
import System.IO.MMap (Mode(ReadOnly), mmapFileForeignPtr)

type Shape = (Int, Int)

data TensorError
  = InvalidShape Shape
  | IndexOutOfBounds Shape (Int, Int)
  | HeaderParseError String
  | FileTooSmall Int
  | UnsupportedDType Word8
  | DataSizeMismatch { expectedBytes :: Int, actualBytes :: Int }
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

(!?) :: Storable a => Tensor a -> (Int, Int) -> IO (Either TensorError a)
(!?) = indexRO

(!) :: Storable a => Tensor a -> (Int, Int) -> IO a
tensor ! idx = do
  result <- indexRO tensor idx
  case result of
    Left err -> ioError (userError ("index error: " ++ show err))
    Right val -> pure val

infixl 9 !?
infixl 9 !

loadTensorFloatRO :: FilePath -> IO (Either TensorError (Tensor Float))
loadTensorFloatRO path = do
  (rawFptr, _, fileBytes) <- mmapFileForeignPtr path ReadOnly Nothing
  if fileBytes < headerSize
    then pure (Left (FileTooSmall fileBytes))
    else do
      headerBytes <- readHeaderBytes rawFptr
      case parseHeader headerBytes of
        Left err -> pure (Left (HeaderParseError err))
        Right hdr ->
          if hDTypeTag hdr /= 1
            then pure (Left (UnsupportedDType (hDTypeTag hdr)))
            else do
              let rows = fromIntegral (hRows hdr)
                  cols = fromIntegral (hCols hdr)
                  expected = headerSize + (rows * cols * sizeOfFloat)
              if fileBytes < expected
                then pure (Left (DataSizeMismatch expected fileBytes))
                else do
                  let dataFptr = plusForeignPtrBytes rawFptr headerSize
                  pure (mkTensorRO (rows, cols) dataFptr)

withTensorFloatRO :: FilePath -> (Either TensorError (Tensor Float) -> IO b) -> IO b
withTensorFloatRO path = bracket (loadTensorFloatRO path) (const (pure ()))

sizeOfFloat :: Int
sizeOfFloat = 4

readHeaderBytes :: ForeignPtr a -> IO ByteString
readHeaderBytes fptr =
  withForeignPtr fptr $ \ptr ->
    BS.packCStringLen (castPtr ptr, headerSize)

plusForeignPtrBytes :: ForeignPtr a -> Int -> ForeignPtr b
plusForeignPtrBytes = plusForeignPtr
