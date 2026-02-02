import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from news.gemini_sentiment import GeminiSentimentAnalyzer


class TestGeminiSentimentAnalyzer:
    """Test cases for GeminiSentimentAnalyzer with mocked API."""
    
    @pytest.fixture
    def analyzer(self):
        with patch('news.gemini_sentiment.genai') as mock_genai:
            mock_model = MagicMock()
            mock_genai.GenerativeModel.return_value = mock_model
            analyzer = GeminiSentimentAnalyzer(api_key="test_key")
            analyzer.model = mock_model
            return analyzer
    
    @pytest.mark.asyncio
    async def test_empty_texts_returns_neutral(self):
        """Empty text list should return neutral sentiment."""
        with patch('news.gemini_sentiment.genai'):
            analyzer = GeminiSentimentAnalyzer(api_key="test_key")
            result = await analyzer.analyze([], "BTCUSDT")
            assert result == 0.5
    
    @pytest.mark.asyncio
    async def test_no_api_key_returns_neutral(self):
        """No API key should return neutral sentiment."""
        with patch('news.gemini_sentiment.genai'):
            with patch.dict('os.environ', {}, clear=True):
                analyzer = GeminiSentimentAnalyzer(api_key=None)
                analyzer.model = None
                result = await analyzer.analyze(["some text"], "BTCUSDT")
                assert result == 0.5
    
    @pytest.mark.asyncio
    async def test_successful_analysis(self, analyzer):
        """Successful API call should return parsed score."""
        mock_response = MagicMock()
        mock_response.text = "0.75"
        analyzer.model.generate_content_async = AsyncMock(return_value=mock_response)
        
        result = await analyzer.analyze(["Bitcoin hits new high"], "BTCUSDT")
        
        assert result == 0.75
        analyzer.model.generate_content_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_score_is_clamped_high(self, analyzer):
        """Scores above 1 should be clamped to 1.0."""
        mock_response = MagicMock()
        mock_response.text = "1.5"
        analyzer.model.generate_content_async = AsyncMock(return_value=mock_response)
        
        result = await analyzer.analyze(["text"], "BTCUSDT")
        assert result == 1.0
    
    @pytest.mark.asyncio
    async def test_score_is_clamped_low(self, analyzer):
        """Scores below 0 should be clamped to 0.0."""
        analyzer.clear_cache()
        mock_response = MagicMock()
        mock_response.text = "-0.3"
        analyzer.model.generate_content_async = AsyncMock(return_value=mock_response)
        
        result = await analyzer.analyze(["other text"], "BTCUSDT")
        assert result == 0.0
    
    @pytest.mark.asyncio
    async def test_invalid_response_returns_neutral(self, analyzer):
        """Invalid response should return neutral sentiment."""
        analyzer.clear_cache()
        mock_response = MagicMock()
        mock_response.text = "invalid text"
        analyzer.model.generate_content_async = AsyncMock(return_value=mock_response)
        
        result = await analyzer.analyze(["Bitcoin news"], "BTCUSDT")
        assert result == 0.5
    
    @pytest.mark.asyncio
    async def test_api_error_returns_neutral(self, analyzer):
        """API error should return neutral sentiment."""
        analyzer.clear_cache()
        analyzer.model.generate_content_async = AsyncMock(side_effect=Exception("API Error"))
        
        result = await analyzer.analyze(["Bitcoin news"], "BTCUSDT")
        assert result == 0.5
    
    @pytest.mark.asyncio
    async def test_caching_works(self, analyzer):
        """Same inputs should use cached result."""
        mock_response = MagicMock()
        mock_response.text = "0.8"
        analyzer.model.generate_content_async = AsyncMock(return_value=mock_response)
        
        # First call
        result1 = await analyzer.analyze(["Bitcoin news"], "BTCUSDT")
        # Second call with same inputs
        result2 = await analyzer.analyze(["Bitcoin news"], "BTCUSDT")
        
        assert result1 == result2 == 0.8
        # Should only be called once due to caching
        assert analyzer.model.generate_content_async.call_count == 1
    
    def test_clear_cache(self, analyzer):
        """Clear cache should empty the cache."""
        analyzer._cache["test_key"] = 0.7
        analyzer.clear_cache()
        assert len(analyzer._cache) == 0
