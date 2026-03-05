module Main (main) where

import Data.Binary.Put (putByteString, putWord32le, putWord8, runPut)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Data.Word (Word8, Word32)
import Foreign.ForeignPtr (ForeignPtr, mallocForeignPtrArray, withForeignPtr)
import Foreign.Storable (pokeElemOff)
import System.Exit (exitFailure, exitSuccess)
import HTensor.Header (Header(..), headerSize, parseHeader)
import HTensor.Types (TensorError(..), indexRO, mkTensorRO)

main :: IO ()
main = do
  ok1 <- testParseHeader
  ok2 <- testShortHeader
  ok3 <- testInvalidShape
  ok4 <- testIndexing

  if and [ok1, ok2, ok3, ok4]
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
