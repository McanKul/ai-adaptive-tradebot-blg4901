import os
import hashlib
from typing import List, Dict
import google.generativeai as genai

from Interfaces.ISentimentAnalyzer import ISentimentAnalyzer
from utils.logger import setup_logger

log = setup_logger("SentimentAnalyzer")


class GeminiSentimentAnalyzer(ISentimentAnalyzer):
    """
    Sentiment analyzer using Google's Gemini Pro model.
    Returns a score in [0, 1] where 0 = bearish, 1 = bullish.
    """
    
    SYSTEM_PROMPT = """You are a cryptocurrency market sentiment analyzer. 
Your task is to analyze news headlines and content to determine market sentiment.

Respond with ONLY a single decimal number between 0 and 1:
- 0.0-0.2: Very bearish (strong negative sentiment, likely price drop)
- 0.2-0.4: Bearish (negative sentiment)
- 0.4-0.6: Neutral (mixed or no clear sentiment)
- 0.6-0.8: Bullish (positive sentiment)
- 0.8-1.0: Very bullish (strong positive sentiment, likely price increase)

Consider factors like:
- Market trends mentioned
- Institutional adoption
- Regulatory news
- Technical developments
- Market fear or greed indicators

Respond with ONLY a number, no explanation."""

    def __init__(self, model: str = "gemini-1.5-flash", api_key: str = None):
        """
        Initialize Gemini sentiment analyzer.
        
        Args:
            model: Gemini model to use (default: gemini-1.5-flash for speed/cost)
                   Options: gemini-1.5-flash, gemini-1.5-pro, gemini-pro
            api_key: Google AI API key. If not provided, reads from GOOGLE_API_KEY env var.
        """
        self.model_name = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self._cache: Dict[str, float] = {}  # Simple in-memory cache
        
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=self.SYSTEM_PROMPT
            )
            log.info("Gemini sentiment analyzer initialized with model: %s", self.model_name)
        else:
            self.model = None
            log.warning("No Google API key provided. Set GOOGLE_API_KEY environment variable.")
    
    def _get_cache_key(self, texts: List[str], symbol: str) -> str:
        """Generate a cache key from the input texts."""
        content = f"{symbol}:{'|'.join(sorted(texts))}"
        return hashlib.md5(content.encode()).hexdigest()
    
    async def analyze(self, texts: List[str], symbol: str) -> float:
        """
        Analyze sentiment of news texts using Gemini.
        
        Args:
            texts: List of news headlines/content to analyze
            symbol: Trading symbol for context
            
        Returns:
            Sentiment score in [0, 1]
        """
        if not texts:
            log.info("No texts to analyze, returning neutral sentiment (0.5)")
            return 0.5
        
        if not self.model:
            log.warning("No API key available, returning neutral sentiment (0.5)")
            return 0.5
        
        # Check cache
        cache_key = self._get_cache_key(texts, symbol)
        if cache_key in self._cache:
            log.info("Using cached sentiment for %s: %.2f", symbol, self._cache[cache_key])
            return self._cache[cache_key]
        
        # Prepare the content for analysis
        news_content = "\n\n".join([
            f"Headline {i+1}: {text}" for i, text in enumerate(texts[:5])  # Limit to 5 items
        ])
        
        user_prompt = f"""Analyze the following news for {symbol} and provide a sentiment score:

{news_content}

Sentiment score (0.0 to 1.0):"""

        try:
            # Generate response using Gemini
            response = await self.model.generate_content_async(
                user_prompt,
                generation_config=genai.GenerationConfig(
                    temperature=0.1,  # Low temperature for consistent results
                    max_output_tokens=10
                )
            )
            
            # Parse the response
            result_text = response.text.strip()
            
            try:
                score = float(result_text)
                # Clamp to valid range
                score = max(0.0, min(1.0, score))
            except ValueError:
                log.warning("Could not parse sentiment score: %s. Using neutral.", result_text)
                score = 0.5
            
            # Cache the result
            self._cache[cache_key] = score
            
            log.info("Sentiment analysis for %s: %.2f (from %d texts)", symbol, score, len(texts))
            return score
            
        except Exception as e:
            log.error("Gemini API error: %s. Returning neutral sentiment.", e)
            return 0.5
    
    def clear_cache(self):
        """Clear the sentiment cache."""
        self._cache.clear()
        log.info("Sentiment cache cleared")
