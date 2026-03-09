import asyncio
from typing import List
from datetime import datetime
from ddgs import DDGS

from Interfaces.INewsSource import INewsSource, NewsArticle
from utils.logger import setup_logger

log = setup_logger("DDGNewsSource")

class DDGNewsSource(INewsSource):
    """
    Fetches latest crypto news using DuckDuckGo's News API.
    Reliable, free, and no API keys required.
    """
    
    def __init__(self):
        self.ddgs = DDGS()

    async def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """
        Fetch news articles for a given symbol.
        """
        # Clean symbol: "BTCUSDT" -> "Bitcoin crypto"
        search_query = self._get_search_term(symbol)
        
        log.info(f"Searching DuckDuckGo News for: '{search_query}'...")

        try:
            # Run the blocking search in a separate thread
            # FIX IS HERE: Changed 'keywords' to 'query'
            results = await asyncio.to_thread(
                self.ddgs.news,
                query=search_query,     # <--- CHANGED FROM keywords=search_query
                region='wt-wt',
                safesearch='off',
                timelimit='d', 
                max_results=limit
            )

            articles = []
            if not results:
                log.warning(f"No news found for {symbol}. Trying broader search...")
                # Fallback: Try past week if today is quiet
                results = await asyncio.to_thread(
                    self.ddgs.news,
                    query=search_query, # <--- CHANGED HERE TOO
                    region='wt-wt',
                    safesearch='off',
                    timelimit='w', 
                    max_results=limit
                )

            for res in results:
                # DDGS news returns: {'date':, 'title':, 'body':, 'url':, 'source':}
                article = NewsArticle(
                    title=res.get('title', ''),
                    content=res.get('body', ''),
                    source=res.get('source', 'DuckDuckGo'),
                    published_at=res.get('date', datetime.now().isoformat()),
                    symbol=symbol.replace("USDT", "")
                )
                articles.append(article)

            log.info(f"Fetched {len(articles)} articles for {symbol}")
            return articles

        except Exception as e:
            log.error(f"Error fetching news: {e}")
            return []

    def _get_search_term(self, symbol: str) -> str:
        """Map trading pairs to meaningful search queries."""
        base = symbol.replace("USDT", "").replace("USD", "")
        mapping = {
            "BTC": "Bitcoin cryptocurrency news",
            "ETH": "Ethereum cryptocurrency news",
            "SOL": "Solana crypto news",
            "XRP": "Ripple XRP news",
            "DOGE": "Dogecoin news"
        }
        return mapping.get(base, f"{base} crypto news")