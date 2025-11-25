
import sys
import pandas as pd
from binance.client import Client
from Streamer.mock_streamer import Streamer

def main():
   try:
      client = Client()
   except Exception as e:
      sys.exit(f"⚠ Binance istemcisi oluşturulamadı: {e}")
   sym = "DOGEUSDT"
   tf = "1m"
   limit = 5
   data = Streamer().fetch_kline(client, sym, tf, limit)[2]

   
   columns = [
      "open_time", "open", "high", "low", "close", "volume",
      "close_time", "quote_volume", "num_trades",
      "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
   ]

   df = pd.DataFrame(data, columns=columns)

   # Float dönüşümleri
   float_cols = ["open", "high", "low", "close", "volume", "quote_volume",
               "taker_buy_base_volume", "taker_buy_quote_volume"]
   df[float_cols] = df[float_cols].astype(float)
      
   o = df["open"].to_numpy(dtype=float)
   h = df["high"].to_numpy(dtype=float)
   l = df["low"].to_numpy(dtype=float)
   c = df["close"].to_numpy(dtype=float)
   v = df["volume"].to_numpy(dtype=float)

   print("\nO:", o)
   print("H:", h)
   print("L:", l)
   print("C:", c)
   print("V:", v)


main()
