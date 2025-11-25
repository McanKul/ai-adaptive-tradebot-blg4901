from abc import ABC, abstractmethod

#todo ileride asenkron olacak her biri
class IStreamer(ABC):

   @staticmethod #todo şimdilik nesne oluşturmadan kullanmak için
   def fetch_kline(client, sym, tf, limit):
      try:
         kl = client.futures_klines(symbol=sym, interval=tf, limit=limit)
         return sym, tf, kl
      except Exception as e:
         print("%s | %s preload hata: %s", sym, tf, e)
         return sym, tf, None
   def start(self): ...

   @abstractmethod
   def stop(self): ...
   
   

