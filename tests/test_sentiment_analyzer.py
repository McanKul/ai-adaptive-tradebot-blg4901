"""
tests/test_sentiment_analyzer.py
=================================
Unit tests for GeminiSentimentAnalyzer.

The analyzer uses the modern google-genai SDK and calls
``client.aio.models.generate_content(...)`` (an async coroutine) which
returns an object with a ``.text`` attribute holding a JSON string like
``'{"score": 0.75}'``.  These tests mock that exact path.
"""
import json
import os
import sys

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from news.gemini_sentiment import GeminiSentimentAnalyzer

pytestmark = pytest.mark.requires_network


def _build_analyzer_with_mock(score_payload):
    """Return (analyzer, mock_generate) where the awaited call returns a
    fake response whose ``.text`` is the given payload (string).
    ``score_payload`` may already be a JSON string or a Python value
    that will be serialized."""
    if not isinstance(score_payload, str):
        score_payload = json.dumps(score_payload)

    with patch("news.gemini_sentiment.genai") as mock_genai:
        mock_client = MagicMock()
        # Path used in production: client.aio.models.generate_content(...)
        mock_response = MagicMock()
        mock_response.text = score_payload
        gen_mock = AsyncMock(return_value=mock_response)
        mock_client.aio.models.generate_content = gen_mock
        mock_genai.Client.return_value = mock_client

        analyzer = GeminiSentimentAnalyzer(api_key="test_key")
        # Ensure the analyzer ended up with our mocked client
        assert analyzer.client is mock_client
        return analyzer, gen_mock


class TestGeminiSentimentAnalyzer:

    @pytest.mark.asyncio
    async def test_empty_texts_returns_neutral(self):
        with patch("news.gemini_sentiment.genai"):
            analyzer = GeminiSentimentAnalyzer(api_key="test_key")
            assert await analyzer.analyze([], "BTCUSDT") == 0.5

    @pytest.mark.asyncio
    async def test_no_api_key_returns_neutral(self):
        with patch("news.gemini_sentiment.genai"):
            with patch.dict("os.environ", {}, clear=True):
                analyzer = GeminiSentimentAnalyzer(api_key=None)
                assert analyzer.client is None
                assert await analyzer.analyze(["x"], "BTCUSDT") == 0.5

    @pytest.mark.asyncio
    async def test_successful_analysis(self):
        analyzer, gen_mock = _build_analyzer_with_mock({"score": 0.75})
        result = await analyzer.analyze(["Bitcoin hits new high"], "BTCUSDT")
        assert result == 0.75
        gen_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_score_is_clamped_high(self):
        analyzer, _ = _build_analyzer_with_mock({"score": 1.5})
        assert await analyzer.analyze(["text"], "BTCUSDT") == 1.0

    @pytest.mark.asyncio
    async def test_score_is_clamped_low(self):
        analyzer, _ = _build_analyzer_with_mock({"score": -0.3})
        assert await analyzer.analyze(["other"], "BTCUSDT") == 0.0

    @pytest.mark.asyncio
    async def test_invalid_response_returns_neutral(self):
        # Non-JSON text triggers the json.loads exception → 0.5
        analyzer, _ = _build_analyzer_with_mock("not-a-json-string")
        assert await analyzer.analyze(["Bitcoin news"], "BTCUSDT") == 0.5

    @pytest.mark.asyncio
    async def test_api_error_returns_neutral(self):
        with patch("news.gemini_sentiment.genai") as mock_genai:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=Exception("API Error")
            )
            mock_genai.Client.return_value = mock_client
            analyzer = GeminiSentimentAnalyzer(api_key="test_key")
            assert await analyzer.analyze(["news"], "BTCUSDT") == 0.5

    @pytest.mark.asyncio
    async def test_caching_works(self):
        analyzer, gen_mock = _build_analyzer_with_mock({"score": 0.8})
        r1 = await analyzer.analyze(["Bitcoin news"], "BTCUSDT")
        r2 = await analyzer.analyze(["Bitcoin news"], "BTCUSDT")
        assert r1 == r2 == 0.8
        # Cache hit means only one network call
        assert gen_mock.call_count == 1

    def test_clear_cache(self):
        with patch("news.gemini_sentiment.genai"):
            analyzer = GeminiSentimentAnalyzer(api_key="test_key")
            analyzer._cache["k"] = 0.7
            analyzer.clear_cache()
            assert analyzer._cache == {}
