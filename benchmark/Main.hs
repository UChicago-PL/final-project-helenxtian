{-# LANGUAGE CPP #-}
{-# LANGUAGE ForeignFunctionInterface #-}

module Main (main) where

import Control.Exception (bracket)
import Control.Monad (forM_, when)
import Data.Binary.Put (putByteString, putFloatle, putWord32le, putWord8, runPut)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Foreign.C.Types (CSize(..))
import Foreign.ForeignPtr (ForeignPtr, mallocForeignPtrArray, withForeignPtr)
import Foreign.Ptr (Ptr)
import Foreign.Storable (peekElemOff)
#ifdef CUDA_AVAILABLE
import Foreign.C.String (CString, peekCString)
import Foreign.C.Types (CChar, CInt(..))
import Foreign.Marshal.Alloc (alloca, allocaBytes)
import Foreign.Storable (peek)
#endif
import GHC.Clock (getMonotonicTimeNSec)
import HTensor.Header (headerSize)
import HTensor.Types (Tensor, loadTensorFloatRO, tensorForeignPtr)
import System.Directory (createDirectoryIfMissing, removeFile)
import System.Environment (getArgs)
import System.Exit (die)
import System.FilePath ((</>))
import System.IO (IOMode(WriteMode), SeekMode(AbsoluteSeek), hSeek, hSetFileSize, withBinaryFile)
import Text.Read (readMaybe)

data Config = Config
  { configSize :: Int
  , configIterations :: Int
  , configDirectory :: FilePath
  , configKeepFiles :: Bool
  , configSkipCpu :: Bool
  }

defaultConfig :: Config
defaultConfig = Config
  { configSize = 512
  , configIterations = 3
  , configDirectory = "benchmark-data"
  , configKeepFiles = False
  , configSkipCpu = False
  }

main :: IO ()
main = do
  config <- parseArgs defaultConfig =<< getArgs
  let size = configSize config
      leftPath = configDirectory config </> "left-" ++ show size ++ ".ht"
      rightPath = configDirectory config </> "right-" ++ show size ++ ".ht"
  createDirectoryIfMissing True (configDirectory config)
  ensureTensorFile size 1.0 leftPath
  ensureTensorFile size 2.0 rightPath
  bracket
    (pure ())
    (const $ when (not (configKeepFiles config)) $ mapM_ removeFile [leftPath, rightPath])
    (const $ runBenchmarks config leftPath rightPath)

runBenchmarks :: Config -> FilePath -> FilePath -> IO ()
runBenchmarks config leftPath rightPath = do
  (leftLoadMs, left) <- timed (loadOrFail leftPath)
  (rightLoadMs, right) <- timed (loadOrFail rightPath)
  putStrLn "backend,size,iteration,mmap_ms,h2d_ms,kernel_ms,d2h_ms,device_total_ms,end_to_end_ms"
  putRow "mmap" (configSize config) 0 (leftLoadMs + rightLoadMs) 0 0 0 0 (leftLoadMs + rightLoadMs)
  when (not (configSkipCpu config)) $
    forM_ [1 .. configIterations config] $ \iteration -> do
      elapsed <- benchmarkCpu (configSize config) left right
      putRow "cpu" (configSize config) iteration 0 0 elapsed 0 elapsed elapsed
#ifdef CUDA_AVAILABLE
  deviceName <- cudaDeviceName
  putStrLn ("# cuda_device=" ++ deviceName)
  forM_ [1 .. configIterations config] $ \iteration -> do
    (h2dMs, kernelMs, d2hMs, deviceTotalMs, endToEndMs) <-
      benchmarkGpu (configSize config) left right
    putRow "gpu" (configSize config) iteration 0 h2dMs kernelMs d2hMs deviceTotalMs endToEndMs
#endif

benchmarkCpu :: Int -> Tensor Float -> Tensor Float -> IO Double
benchmarkCpu size left right = do
  result <- mallocForeignPtrArray (size * size)
  (elapsed, ()) <- timed
    (withThreeForeignPtrs
      (tensorForeignPtr left)
      (tensorForeignPtr right)
      result
      (\leftPtr rightPtr resultPtr ->
        c_matmulFloat
          leftPtr
          rightPtr
          resultPtr
          (fromIntegral size)
          (fromIntegral size)
          (fromIntegral size)))
  validateResult size result
  pure elapsed

#ifdef CUDA_AVAILABLE
benchmarkGpu :: Int -> Tensor Float -> Tensor Float -> IO (Double, Double, Double, Double, Double)
benchmarkGpu size left right = do
  result <- mallocForeignPtrArray (size * size)
  alloca $ \h2dPtr ->
    alloca $ \kernelPtr ->
      alloca $ \d2hPtr ->
        alloca $ \deviceTotalPtr -> do
          (endToEndMs, status) <- timed
            (withThreeForeignPtrs
              (tensorForeignPtr left)
              (tensorForeignPtr right)
              result
              (\leftPtr rightPtr resultPtr ->
                c_cudaMatmulFloat
                  leftPtr
                  rightPtr
                  resultPtr
                  (fromIntegral size)
                  (fromIntegral size)
                  (fromIntegral size)
                  h2dPtr
                  kernelPtr
                  d2hPtr
                  deviceTotalPtr))
          checkCuda status
          validateResult size result
          h2dMs <- realToFrac <$> peek h2dPtr
          kernelMs <- realToFrac <$> peek kernelPtr
          d2hMs <- realToFrac <$> peek d2hPtr
          deviceTotalMs <- realToFrac <$> peek deviceTotalPtr
          pure (h2dMs, kernelMs, d2hMs, deviceTotalMs, endToEndMs)

cudaDeviceName :: IO String
cudaDeviceName =
  allocaBytes 256 $ \buffer -> do
    status <- c_cudaDeviceName buffer 256
    checkCuda status
    peekCString buffer

checkCuda :: CInt -> IO ()
checkCuda 0 = pure ()
checkCuda status = do
  message <- c_cudaLastError >>= peekCString
  die ("CUDA benchmark failed (status " ++ show status ++ "): " ++ message)
#endif

validateResult :: Int -> ForeignPtr Float -> IO ()
validateResult size result =
  withForeignPtr result $ \ptr -> do
    first <- peekElemOff ptr 0
    lastDiagonal <- peekElemOff ptr (size * size - 1)
    when (first /= 2.0 || lastDiagonal /= 2.0) $
      die ("matrix result validation failed: expected diagonal value 2.0, got " ++ show (first, lastDiagonal))

withThreeForeignPtrs
  :: ForeignPtr a
  -> ForeignPtr b
  -> ForeignPtr c
  -> (Ptr a -> Ptr b -> Ptr c -> IO value)
  -> IO value
withThreeForeignPtrs first second third action =
  withForeignPtr first $ \firstPtr ->
    withForeignPtr second $ \secondPtr ->
      withForeignPtr third $ \thirdPtr ->
        action firstPtr secondPtr thirdPtr

loadOrFail :: FilePath -> IO (Tensor Float)
loadOrFail path = do
  loaded <- loadTensorFloatRO path
  either (die . ("failed to load benchmark tensor: " ++) . show) pure loaded

ensureTensorFile :: Int -> Float -> FilePath -> IO ()
ensureTensorFile size diagonalValue path =
  withBinaryFile path WriteMode $ \handle -> do
    LBS.hPut handle $ runPut $ do
      putWord32le (fromIntegral size)
      putWord32le (fromIntegral size)
      putWord8 1
      putByteString (BS.replicate (headerSize - 9) 0)
    hSetFileSize handle (fromIntegral headerSize + fromIntegral size * fromIntegral size * 4)
    forM_ [0 .. size - 1] $ \index -> do
      let elementOffset = index * size + index
      hSeek handle AbsoluteSeek (fromIntegral headerSize + fromIntegral elementOffset * 4)
      LBS.hPut handle (runPut (putFloatle diagonalValue))

timed :: IO value -> IO (Double, value)
timed action = do
  start <- getMonotonicTimeNSec
  value <- action
  end <- getMonotonicTimeNSec
  pure (fromIntegral (end - start) / 1.0e6, value)

putRow :: String -> Int -> Int -> Double -> Double -> Double -> Double -> Double -> Double -> IO ()
putRow backend size iteration mmapMs h2dMs kernelMs d2hMs deviceTotalMs endToEndMs =
  putStrLn $ concatWithComma
    [ backend
    , show size
    , show iteration
    , show mmapMs
    , show h2dMs
    , show kernelMs
    , show d2hMs
    , show deviceTotalMs
    , show endToEndMs
    ]

concatWithComma :: [String] -> String
concatWithComma [] = ""
concatWithComma [value] = value
concatWithComma (value:values) = value ++ "," ++ concatWithComma values

parseArgs :: Config -> [String] -> IO Config
parseArgs config [] = pure config
parseArgs config ("--size":value:rest) =
  parsePositive "--size" value >>= \size -> parseArgs config { configSize = size } rest
parseArgs config ("--iterations":value:rest) =
  parsePositive "--iterations" value >>= \iterations ->
    parseArgs config { configIterations = iterations } rest
parseArgs config ("--directory":value:rest) =
  parseArgs config { configDirectory = value } rest
parseArgs config ("--keep-files":rest) =
  parseArgs config { configKeepFiles = True } rest
parseArgs config ("--skip-cpu":rest) =
  parseArgs config { configSkipCpu = True } rest
parseArgs _ (argument:_) = die ("unknown or incomplete argument: " ++ argument ++ usage)

parsePositive :: String -> String -> IO Int
parsePositive name value =
  case readMaybe value of
    Just parsed | parsed > 0 -> pure parsed
    _ -> die (name ++ " must be a positive integer")

usage :: String
usage = "\nusage: cabal bench htensor-bench --benchmark-options='--size N --iterations N [--directory PATH] [--keep-files] [--skip-cpu]'"

foreign import ccall unsafe "htensor_matmul_f32"
  c_matmulFloat
    :: Ptr Float
    -> Ptr Float
    -> Ptr Float
    -> CSize
    -> CSize
    -> CSize
    -> IO ()

#ifdef CUDA_AVAILABLE
foreign import ccall unsafe "htensor_cuda_matmul_f32"
  c_cudaMatmulFloat
    :: Ptr Float
    -> Ptr Float
    -> Ptr Float
    -> CSize
    -> CSize
    -> CSize
    -> Ptr Float
    -> Ptr Float
    -> Ptr Float
    -> Ptr Float
    -> IO CInt

foreign import ccall unsafe "htensor_cuda_device_name"
  c_cudaDeviceName :: Ptr CChar -> CSize -> IO CInt

foreign import ccall unsafe "htensor_cuda_last_error"
  c_cudaLastError :: IO CString
#endif