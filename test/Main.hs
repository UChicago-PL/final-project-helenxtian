module Main (main) where

import Data.Binary.Put (putByteString, putFloatle, putWord32le, putWord8, runPut)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Data.Word (Word8, Word32)
import Foreign.ForeignPtr (ForeignPtr, mallocForeignPtrArray, withForeignPtr)
import Foreign.Storable (pokeElemOff)
import System.Directory (getTemporaryDirectory, removeFile)
import System.Exit (exitFailure, exitSuccess)
import System.IO (hClose, openBinaryTempFile)
import HTensor.Header (Header(..), headerSize, parseHeader)
import HTensor.Types (TensorError(..), (!), (!?), indexRO, loadTensorFloatRO, mkTensorRO, withTensorFloatRO)

-- Paths to committed fixture files (relative to project root)
fixturePng :: FilePath
fixturePng = "test/fixtures/sample.png"

fixturePdf :: FilePath
fixturePdf = "test/fixtures/sample.pdf"

main :: IO ()
main = do
  ok1 <- testParseHeader
  ok2 <- testShortHeader
  ok3 <- testInvalidShape
  ok4 <- testIndexing
  ok5 <- testLoadFromFile
  ok6 <- testLoadFromImageAfterConversion
  ok7 <- testLoadFromPdfAfterConversion
  ok8 <- testIndexingOperators
  ok9 <- testWithTensorFloatRO

  if and [ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8, ok9]
    then do
      putStrLn "All tests passed."
      exitSuccess
    else exitFailure

testParseHeader :: IO Bool
testParseHeader = do
  let bs = mkHeaderBytes 3 5 1
      expected = Header 3 5 1
  case parseHeader bs of
    Right header | header == expected -> pass "parseHeader valid header"
    Right header -> failTest $ "parseHeader mismatch: got " ++ show header
    Left err -> failTest $ "parseHeader failed unexpectedly: " ++ err

testShortHeader :: IO Bool
testShortHeader = do
  let short = BS.replicate 8 0
  case parseHeader short of
    Left _ -> pass "parseHeader short header"
    Right h -> failTest $ "expected failure on short header, got " ++ show h

testInvalidShape :: IO Bool
testInvalidShape = do
  fptr <- (mallocForeignPtrArray 4 :: IO (ForeignPtr Float))
  case mkTensorRO (0, 2) fptr of
    Left (InvalidShape _) -> pass "mkTensorRO invalid shape"
    Left e -> failTest $ "expected InvalidShape, got " ++ show e
    Right _ -> failTest "expected invalid shape failure"

testIndexing :: IO Bool
testIndexing = do
  fptr <- (mallocForeignPtrArray 4 :: IO (ForeignPtr Float))
  withForeignPtr fptr $ \ptr -> do
    pokeElemOff ptr 0 10
    pokeElemOff ptr 1 20
    pokeElemOff ptr 2 30
    pokeElemOff ptr 3 40

  case mkTensorRO (2, 2) fptr of
    Left err -> failTest $ "mkTensorRO failed unexpectedly: " ++ show err
    Right tensor -> do
      inBounds <- indexRO tensor (1, 0)
      outBounds <- indexRO tensor (2, 0)
      case (inBounds, outBounds) of
        (Right v, Left (IndexOutOfBounds _ _)) | v == 30 -> pass "indexRO in/out bounds"
        _ -> failTest "indexRO did not return expected results"

testLoadFromFile :: IO Bool
testLoadFromFile = do
  tmpDir <- getTemporaryDirectory
  (tmpPath, handle) <- openBinaryTempFile tmpDir "htensor.bin"
  LBS.hPut handle (runPut $ do
    putWord32le 2
    putWord32le 2
    putWord8 1
    putByteString (BS.replicate (headerSize - 9) 0)
    putFloatle 1.0
    putFloatle 2.0
    putFloatle 3.0
    putFloatle 4.0
    )
  hClose handle
  loaded <- loadTensorFloatRO tmpPath
  removeFile tmpPath

  case loaded of
    Left err -> failTest $ "loadTensorFloatRO failed unexpectedly: " ++ show err
    Right tensor -> do
      result <- indexRO tensor (1, 1)
      case result of
        Right v | v == 4.0 -> pass "loadTensorFloatRO end-to-end"
        Right v -> failTest $ "expected 4.0 at (1,1), got " ++ show v
        Left e -> failTest $ "indexRO failed unexpectedly: " ++ show e

-- Reads the committed sample.png fixture (PNG magic bytes) and converts it
-- to a tensor, then checks that the first two values are 137.0 and 80.0.
testLoadFromImageAfterConversion :: IO Bool
testLoadFromImageAfterConversion = do
  tmpDir <- getTemporaryDirectory
  (tensorPath, tensorHandle) <- openBinaryTempFile tmpDir "fixture-image.ht"
  hClose tensorHandle
  writeTensorFromFileBytes fixturePng tensorPath

  loaded <- loadTensorFloatRO tensorPath
  removeFile tensorPath

  case loaded of
    Left err -> failTest $ "image conversion load failed: " ++ show err
    Right tensor -> do
      v0 <- indexRO tensor (0, 0)
      v1 <- indexRO tensor (0, 1)
      case (v0, v1) of
        (Right a, Right b) | a == 137.0 && b == 80.0 -> pass "image bytes -> tensor load"
        _ -> failTest "image conversion values mismatch"

-- Reads the committed sample.pdf fixture (%PDF-1.4 ... %%EOF) and converts
-- it to a tensor, then checks that the first byte value is 37.0 (ASCII '%').
testLoadFromPdfAfterConversion :: IO Bool
testLoadFromPdfAfterConversion = do
  tmpDir <- getTemporaryDirectory
  (tensorPath, tensorHandle) <- openBinaryTempFile tmpDir "fixture-pdf.ht"
  hClose tensorHandle
  writeTensorFromFileBytes fixturePdf tensorPath

  loaded <- loadTensorFloatRO tensorPath
  removeFile tensorPath

  case loaded of
    Left err -> failTest $ "pdf conversion load failed: " ++ show err
    Right tensor -> do
      first <- indexRO tensor (0, 0)
      case first of
        Right v | v == 37.0 -> pass "pdf bytes -> tensor load"
        Right v -> failTest $ "expected first byte 37.0 (%), got " ++ show v
        Left e -> failTest $ "indexRO failed unexpectedly: " ++ show e

testIndexingOperators :: IO Bool
testIndexingOperators = do
  fptr <- (mallocForeignPtrArray 4 :: IO (ForeignPtr Float))
  withForeignPtr fptr $ \ptr -> do
    pokeElemOff ptr 0 10
    pokeElemOff ptr 1 20
    pokeElemOff ptr 2 30
    pokeElemOff ptr 3 40

  case mkTensorRO (2, 2) fptr of
    Left err -> failTest $ "mkTensorRO failed unexpectedly: " ++ show err
    Right tensor -> do
      maybeVal <- tensor !? (0, 1)
      forcedVal <- tensor ! (1, 1)
      case maybeVal of
        Right v | v == 20.0 && forcedVal == 40.0 -> pass "index operators (!?) and (!)"
        _ -> failTest "index operators returned unexpected values"

testWithTensorFloatRO :: IO Bool
testWithTensorFloatRO = do
  tmpDir <- getTemporaryDirectory
  (tmpPath, handle) <- openBinaryTempFile tmpDir "htensor-with.bin"
  LBS.hPut handle (runPut $ do
    putWord32le 1
    putWord32le 2
    putWord8 1
    putByteString (BS.replicate (headerSize - 9) 0)
    putFloatle 9.0
    putFloatle 11.0
    )
  hClose handle

  ok <- withTensorFloatRO tmpPath $ \loaded ->
    case loaded of
      Left err -> failTest $ "withTensorFloatRO failed unexpectedly: " ++ show err
      Right tensor -> do
        result <- tensor !? (0, 1)
        case result of
          Right v | v == 11.0 -> pass "withTensorFloatRO bracket helper"
          Right v -> failTest $ "expected 11.0, got " ++ show v
          Left e -> failTest $ "index operator failed unexpectedly: " ++ show e

  removeFile tmpPath
  pure ok

writeTensorFromFileBytes :: FilePath -> FilePath -> IO ()
writeTensorFromFileBytes srcPath dstPath = do
  bytes <- BS.readFile srcPath
  let n = BS.length bytes
      body = runPut $ mapM_ (putFloatle . fromIntegral) (BS.unpack bytes)
      fileContent = runPut $ do
        putWord32le 1
        putWord32le (fromIntegral n)
        putWord8 1
        putByteString (BS.replicate (headerSize - 9) 0)
        putByteString (LBS.toStrict body)
  LBS.writeFile dstPath fileContent

mkHeaderBytes :: Word32 -> Word32 -> Word8 -> ByteString
mkHeaderBytes rows cols dtype =
  LBS.toStrict (runPut $ do
    putWord32le rows
    putWord32le cols
    putWord8 dtype
    putByteString (BS.replicate (headerSize - 9) 0)
  )

pass :: String -> IO Bool
pass name = do
  putStrLn ("[PASS] " ++ name)
  pure True

failTest :: String -> IO Bool
failTest msg = do
  putStrLn ("[FAIL] " ++ msg)
  pure False
  