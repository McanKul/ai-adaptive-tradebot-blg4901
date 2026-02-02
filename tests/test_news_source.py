import pytest
from news.crypto_news_source import MockNewsSource


class TestMockNewsSource:
    """Test cases for MockNewsSource."""
    
    @pytest.fixture
    def source(self):
        return MockNewsSource()
    
    @pytest.mark.asyncio
    async def test_fetch_btc_news(self, source):
        """Should return BTC news articles."""
        articles = await source.fetch_news("BTCUSDT")
        assert len(articles) > 0
        assert all(a.symbol == "BTC" for a in articles)
    
    @pytest.mark.asyncio
    async def test_fetch_eth_news(self, source):
        """Should return ETH news articles."""
        articles = await source.fetch_news("ETH")
        assert len(articles) > 0
        assert all(a.symbol == "ETH" for a in articles)
    
    @pytest.mark.asyncio
    async def test_fetch_unknown_symbol_returns_empty(self, source):
        """Should return empty list for unknown symbols."""
        articles = await source.fetch_news("UNKNOWNUSDT")
        assert articles == []
    
    @pytest.mark.asyncio
    async def test_limit_parameter(self, source):
        """Should respect the limit parameter."""
        articles = await source.fetch_news("BTCUSDT", limit=1)
        assert len(articles) <= 1
    
    @pytest.mark.asyncio
    async def test_article_has_required_fields(self, source):
        """Articles should have all required fields."""
        articles = await source.fetch_news("BTCUSDT")
        for article in articles:
            assert article.title
            assert article.content
            assert article.source
            assert article.published_at
            assert article.symbol
