"""
tools/run_mock_news_engine.py
==============================
Mock news sentiment test — uses MockNewsSource + GeminiSentimentAnalyzer.

Usage:
    python tools/run_mock_news_engine.py
"""



import asyncio

import os

import sys

from datetime import datetime

from news.gemini_sentiment import model_name



from dotenv import load_dotenv



# Load environment variables from .env

load_dotenv()



from news.crypto_news_source import MockNewsSource

from news.gemini_sentiment import GeminiSentimentAnalyzer

from news.news_engine import NewsEngine

from Interfaces.INewsSource import NewsArticle





def sentiment_label(score: float) -> str:

    """Convert a numeric sentiment score to a human-readable label."""

    if score <= 0.2:

        return "Very Bearish"

    elif score <= 0.4:

        return "Bearish"

    elif score <= 0.6:

        return "Neutral"

    elif score <= 0.8:

        return "Bullish"

    else:

        return "Very Bullish"





async def run():

    # ── Setup ──────────────────────────────────────────────────────────────

    google_api_key = os.getenv("GOOGLE_API_KEY")

    if not google_api_key:

        print("ERROR: GOOGLE_API_KEY not found in .env — cannot run sentiment analysis.")

        sys.exit(1)



    news_source = MockNewsSource()

    analyzer = GeminiSentimentAnalyzer(api_key=google_api_key)

    engine = NewsEngine(

        news_source=news_source,

        sentiment_analyzer=analyzer,

        refresh_interval=0,   # always fetch fresh for testing

        news_limit=10

    )



    symbols = ["BTCUSDT", "ETHUSDT"]

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")



    # ── Collect results ────────────────────────────────────────────────────

    lines: list[str] = []

    separator = "=" * 72



    lines.append(separator)

    lines.append("  NEWS ENGINE — MOCK TEST RESULTS")

    lines.append(separator)

    lines.append(f"  Run Date       : {run_timestamp}")

    lines.append(f"  News Source     : MockNewsSource")

    lines.append(f"  Analyzer        : GeminiSentimentAnalyzer ({model_name})")

    lines.append(f"  Symbols Tested  : {', '.join(symbols)}")

    lines.append(separator)

    lines.append("")



    for symbol in symbols:

        lines.append(f"{'─' * 72}")

        lines.append(f"  SYMBOL: {symbol}")

        lines.append(f"{'─' * 72}")



        # Fetch articles via the same path the engine uses

        articles = await news_source.fetch_news(symbol, limit=10)



        if not articles:

            lines.append("  (no mock articles available for this symbol)")

            lines.append("")

            continue



        lines.append(f"  Articles fetched: {len(articles)}")

        lines.append("")



        for idx, article in enumerate(articles, start=1):

            lines.append(f"  [{idx}] Title       : {article.title}")

            lines.append(f"      Content     : {article.content}")

            lines.append(f"      Source      : {article.source}")

            lines.append(f"      Published   : {article.published_at}")

            lines.append(f"      Symbol      : {article.symbol}")

            lines.append("")



        # Run sentiment through the engine

        score = await engine.get_sentiment(symbol, force_refresh=True)



        lines.append(f"  ► Sentiment Score : {score:.4f}")

        lines.append(f"  ► Sentiment Label : {sentiment_label(score)}")

        lines.append(f"  ► Analyzed By     : GeminiSentimentAnalyzer ({model_name})")

        lines.append(f"  ► Analysis Date   : {run_timestamp}")

        lines.append("")



    lines.append(separator)

    lines.append("  END OF REPORT")

    lines.append(separator)



    report = "\n".join(lines)



    # ── Write to file ──────────────────────────────────────────────────────

    output_path = os.path.join(os.path.dirname(__file__), "mock_results.txt")

    with open(output_path, "w", encoding="utf-8") as f:

        f.write(report)



    # Also print to terminal

    print(report)

    print(f"\n✔ Results written to {output_path}")





if __name__ == "__main__":

    asyncio.run(run())