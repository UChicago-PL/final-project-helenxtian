module HTensor.Header (Header(..), headerSize, parseHeader) where

import Data.Binary.Get (Get, getWord32le, getWord8, runGetOrFail)
import Data.ByteString (ByteString)
import qualified Data.ByteString as BS
import qualified Data.ByteString.Lazy as LBS
import Data.Word (Word32, Word8)

data Header = Header
  { hRows :: Word32
  , hCols :: Word32
  , hDTypeTag :: Word8
  } deriving (Eq, Show)

headerSize :: Int
headerSize = 128

parseHeader :: ByteString -> Either String Header
parseHeader bs
  | BS.length bs < headerSize = Left "header shorter than 128 bytes"
  | otherwise =
      case runGetOrFail parser (LBS.fromStrict (BS.take headerSize bs)) of
        Left (_, _, err) -> Left err
        Right (_, _, header) -> Right header

parser :: Get Header
parser = do
  rows <- getWord32le
  cols <- getWord32le
  dtype <- getWord8
  pure (Header rows cols dtype)
