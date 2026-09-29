"""
NLP sentiment pillar.

Source: live news via Google News RSS (per-company query) + optional
Economic Times / Moneycontrol general feeds.

Scoring uses FinBERT (ProsusAI/finbert) instead of VADER — VADER is
tuned for generic social-media English and misreads financial phrasing
("beats estimates" reads neutral-ish to VADER but is strongly positive
in a financial context).

Output: long-format DataFrame with columns
    [Ticker, Date, Source, Headline, Sentiment_Label, Sentiment_Score]
where Sentiment_Score is signed in [-1, 1] (positive - negative FinBERT
class probability), matching the sign convention of the old VADER
compound score so downstream code doesn't need to change.
"""
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import feedparser
import pandas as pd

from config.settings import NEWS_RSS_TEMPLATES
from config.tickers import NIFTY_50_TICKERS, TICKER_TO_COMPANY_NAME

# --- FinBERT scorer, loaded lazily (only when first needed) ---------------
_finbert_pipeline = None


def get_finbert_pipeline():
    global _finbert_pipeline
    if _finbert_pipeline is None:
        from transformers import pipeline
        from config.settings import SENTIMENT_MODEL_NAME
        _finbert_pipeline = pipeline("sentiment-analysis", model=SENTIMENT_MODEL_NAME)
    return _finbert_pipeline


def score_headlines_finbert(headlines: list[str]) -> list[dict]:
    """
    Returns a list of {label, score} where score is signed:
    +confidence for positive, -confidence for negative, 0 for neutral.
    """
    if not headlines:
        return []
    clf = get_finbert_pipeline()
    raw = clf(headlines, truncation=True, batch_size=16)

    results = []
    for r in raw:
        label = r["label"].lower()
        conf = r["score"]
        signed = conf if label == "positive" else (-conf if label == "negative" else 0.0)
        results.append({"label": label, "score": signed})
    return results


# --- News RSS ---------------------------------------------------------------
def fetch_news_for_ticker(ticker: str, days_back: int = 7) -> pd.DataFrame:
    """Fetch recent Google News RSS headlines for a company name."""
    company_name = TICKER_TO_COMPANY_NAME.get(ticker, ticker.replace(".NS", ""))
    query = quote_plus(company_name)
    rss_url = NEWS_RSS_TEMPLATES[0].format(query=query)

    feed = feedparser.parse(rss_url)
    cutoff = datetime.now() - timedelta(days=days_back)

    rows = []
    for entry in feed.entries:
        if not hasattr(entry, "published"):
            continue
        pub_date = pd.to_datetime(entry.published, utc=True).tz_localize(None)
        if pub_date < cutoff:
            continue
        rows.append({"Ticker": ticker, "Date": pub_date.date(), "Headline": entry.title, "Source": "News"})

    return pd.DataFrame(rows)


# --- Orchestration ------------------------------------------------------------
def build_sentiment_dataset(
    tickers: list[str] = NIFTY_50_TICKERS,
    days_back: int = 30,
) -> pd.DataFrame:
    """
    Fetch news headlines for all tickers, score with FinBERT, aggregate
    to one row per (Ticker, Date) with a mean sentiment score.

    Google News RSS is intended for recent news, so days_back should stay
    reasonably small. Larger historical windows need a news archive/API.
    """
    all_headlines = []
    for ticker in tickers:
        news_df = fetch_news_for_ticker(ticker, days_back=days_back)
        if not news_df.empty:
            all_headlines.append(news_df)

    if not all_headlines:
        return pd.DataFrame(columns=["Ticker", "Date", "sentiment_score"])

    headlines_df = pd.concat(all_headlines, ignore_index=True)

    scored = score_headlines_finbert(headlines_df["Headline"].tolist())
    headlines_df["Sentiment_Label"] = [s["label"] for s in scored]
    headlines_df["Sentiment_Score"] = [s["score"] for s in scored]

    daily = (
        headlines_df.groupby(["Ticker", "Date"])["Sentiment_Score"]
        .mean()
        .reset_index()
        .rename(columns={"Sentiment_Score": "sentiment_score"})
    )
    return daily


if __name__ == "__main__":
    df = build_sentiment_dataset(tickers=NIFTY_50_TICKERS, days_back=30)
    print(df.shape)
    print(df.head())
