import IStreamer

class Streamer(IStreamer):

   def fetch_kline(self, client, sym, tf, limit):
        try:
            kl = client.futures_klines(symbol=sym, interval=tf, limit=limit)
            return sym, tf, kl
        except Exception as e:
            print("%s | %s preload hata: %s", sym, tf, e)
            return sym, tf, None
         
   def start(self): pass
   
   def stop(self): pass

