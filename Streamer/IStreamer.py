from abc import ABC, abstractmethod

#todo ileride asenkron olacak her biri
class IStreamer(ABC):
   
   @abstractmethod
   def fetch_kline(self): ...
   
   @abstractmethod
   def start(self): ...

   @abstractmethod
   def stop(self): ...
   
   

