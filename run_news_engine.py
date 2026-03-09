"""
Manual runner for the News Engine.
"""
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# --- IMPORT THE NEW SOURCE ---
from news.ddg_news_source import DDGNewsSource 
from news.gemini_sentiment import GeminiSentimentAnalyzer
from news.news_engine import NewsEngine

def sentiment_label(score: float) -> str:
    if score <= 0.2: return "Very Bearish"
    elif score <= 0.4: return "Bearish"
    elif score <= 0.6: return "Neutral"
    elif score <= 0.8: return "Bullish"
    else: return "Very Bullish"

async def run(symbols: list = None) -> dict:
    if symbols is None:
        symbols = ["BTCUSDT", "ETHUSDT"]

    google_api_key = os.getenv("GOOGLE_API_KEY")
    if not google_api_key:
        print("ERROR: GOOGLE_API_KEY not found in .env")
        sys.exit(1)

    # USE THE NEW SOURCE
    print("Initializing News Source (DuckDuckGo News)...")
    news_source = DDGNewsSource()
    
    # Use your gemini-2.0-flash model
    analyzer = GeminiSentimentAnalyzer(api_key=google_api_key)
    
    engine = NewsEngine(
        news_source=news_source,
        sentiment_analyzer=analyzer,
        refresh_interval=0,
        news_limit=10
    )

    run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    results = {}

    # ── Collect results ────────────────────────────────────────────────────
    lines = []
    separator = "=" * 72

    lines.append(separator)
    lines.append("  NEWS ENGINE — LIVE NEWS TEST")
    lines.append(separator)
    lines.append(f"  Run Date       : {run_timestamp}")
    lines.append(f"  News Source    : DDGNewsSource")
    lines.append(f"  Analyzer       : GeminiSentimentAnalyzer ({analyzer.model_name})")
    lines.append(separator)
    lines.append("")

    for symbol in symbols:
        lines.append(f"{'─' * 72}")
        lines.append(f"  SYMBOL: {symbol}")
        lines.append(f"{'─' * 72}")

        print(f"Fetching news for {symbol}...")
        articles = await news_source.fetch_news(symbol, limit=5)

        if not articles:
            lines.append("  (No articles found)")
            lines.append("")
            continue

        lines.append(f"  Articles fetched: {len(articles)}")
        lines.append("")

        for idx, article in enumerate(articles, start=1):
            lines.append(f"  [{idx}] Title : {article.title}")
            lines.append(f"      Source: {article.source}")
            lines.append(f"      Date  : {article.published_at}")
            lines.append("")

        # Run sentiment
        print(f"Analyzing sentiment for {symbol}...")
        score = await engine.get_sentiment(symbol, force_refresh=True)
        results[symbol] = score

        lines.append(f"  ► Sentiment Score : {score:.4f}")
        lines.append(f"  ► Sentiment Label : {sentiment_label(score)}")
        lines.append("")
        
        await asyncio.sleep(2)

    lines.append(separator)
    lines.append("  END OF REPORT")
    lines.append(separator)

    report = "\n".join(lines)
    output_path = os.path.join(os.path.dirname(__file__), "news_results.txt")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n✔ Results written to {output_path}")

    return results

if __name__ == "__main__":
    if len(sys.argv) > 1:
        symbols_arg = sys.argv[1:]
        asyncio.run(run(symbols_arg))
    else:
        asyncio.run(run())