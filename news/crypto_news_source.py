import aiohttp
from typing import List
from datetime import datetime

from Interfaces.INewsSource import INewsSource, NewsArticle
from utils.logger import setup_logger

log = setup_logger("NewsSource")


class MockNewsSource(INewsSource):
    """Mock news source for testing purposes. Returns predefined news articles."""
    
    def __init__(self):
        self._mock_news = {
            "BTC": [
                NewsArticle(
                    title="Bitcoin Surges Past Key Resistance Level",
                    content="Bitcoin has broken through the $50,000 resistance level with strong volume, indicating bullish momentum.",
                    source="MockNews",
                    published_at=datetime.now().isoformat(),
                    symbol="BTC"
                ),
                NewsArticle(
                    title="Institutional Investors Increase Bitcoin Holdings",
                    content="Major financial institutions have announced increased Bitcoin allocations in their portfolios.",
                    source="MockNews",
                    published_at=datetime.now().isoformat(),
                    symbol="BTC"
                ),
            ],
            "ETH": [
                NewsArticle(
                    title="Ethereum Network Upgrade Successful",
                    content="The latest Ethereum network upgrade has been completed successfully, improving scalability.",
                    source="MockNews",
                    published_at=datetime.now().isoformat(),
                    symbol="ETH"
                ),
            ],
        }
    
    async def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """Return mock news for testing."""
        # Extract base symbol (e.g., "BTCUSDT" -> "BTC")
        base_symbol = symbol.replace("USDT", "").replace("USD", "")
        
        articles = self._mock_news.get(base_symbol, [])
        log.info("MockNewsSource fetched %d articles for %s", len(articles[:limit]), symbol)
        return articles[:limit]


class CryptoCompareNewsSource(INewsSource):
    """
    Fetches crypto news from CryptoCompare API.
    Free tier available at: https://min-api.cryptocompare.com/
    """
    
    BASE_URL = "https://min-api.cryptocompare.com/data/v2/news/"
    
    def __init__(self, api_key: str = None):
        """
        Initialize CryptoCompare news source.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
        self.headers = {"authorization": f"Apikey {api_key}"} if api_key else {}
    
    async def fetch_news(self, symbol: str, limit: int = 10) -> List[NewsArticle]:
        """Fetch news from CryptoCompare API."""
        # Extract base symbol
        base_symbol = symbol.replace("USDT", "").replace("USD", "")
        
        params = {
            "categories": base_symbol,
            "extraParams": "TradingBot"
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.BASE_URL,
                    params=params,
                    headers=self.headers,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status != 200:
                        log.warning("CryptoCompare API returned status %d", response.status)
                        return []
                    
                    data = await response.json()
                    
                    if data.get("Response") == "Error":
                        log.warning("CryptoCompare API error: %s", data.get("Message"))
                        return []
                    
                    articles = []
                    for item in data.get("Data", [])[:limit]:
                        article = NewsArticle(
                            title=item.get("title", ""),
                            content=item.get("body", ""),
                            source=item.get("source", "CryptoCompare"),
                            published_at=datetime.fromtimestamp(
                                item.get("published_on", 0)
                            ).isoformat(),
                            symbol=base_symbol
                        )
                        articles.append(article)
                    
                    log.info("CryptoCompare fetched %d articles for %s", len(articles), symbol)
                    return articles
                    
        except aiohttp.ClientError as e:
            log.error("Network error fetching news: %s", e)
            return []
        except Exception as e:
            log.error("Error fetching news from CryptoCompare: %s", e)
            return []
