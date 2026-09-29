"""
Fundamental analysis pillar.

Fundamentals change slowly (quarterly), so this is designed to run on a
separate, much lower-frequency schedule than technical/sentiment (e.g.
weekly), and be joined onto the daily feature table by ticker.

Output: one row per ticker with columns
    [Ticker, PE_Ratio, PB_Ratio, ROE, Debt_To_Equity, EPS_Growth_YoY,
     Revenue_Growth_YoY, Promoter_Holding_Pct, As_Of_Date]

NOTE: yfinance's `.info` dict is inconsistently populated for NSE
tickers and can be stale. For production use, cross-check against
screener.in exports or a paid data vendor (Tickertape, Groww API,
Refinitiv, etc). This module is written so that source can be swapped
without touching the rest of the pipeline.
"""
from datetime import date

import pandas as pd
import yfinance as yf

from config.tickers import NIFTY_50_TICKERS


def fetch_fundamentals_yfinance(ticker: str) -> dict:
    """Best-effort fundamental snapshot from yfinance. Fields may be None."""
    info = yf.Ticker(ticker).info

    return {
        "Ticker": ticker,
        "PE_Ratio": info.get("trailingPE"),
        "Forward_PE": info.get("forwardPE"),
        "PB_Ratio": info.get("priceToBook"),
        "ROE": info.get("returnOnEquity"),
        "Debt_To_Equity": info.get("debtToEquity"),
        "Profit_Margin": info.get("profitMargins"),
        "Revenue_Growth_YoY": info.get("revenueGrowth"),
        "Earnings_Growth_YoY": info.get("earningsGrowth"),
        "Market_Cap": info.get("marketCap"),
        "As_Of_Date": date.today(),
    }


def fetch_fundamentals_screener(ticker: str) -> dict:
    """
    TODO: implement screener.in-based fetch for fields yfinance misses,
    especially promoter holding % and pledge %, which matter a lot for
    Indian equities and aren't in yfinance's info dict at all.

    screener.in has no official API; scraping their public company pages
    is the common workaround. Respect robots.txt / rate limits.
    """
    raise NotImplementedError


def build_fundamental_dataset(tickers: list[str] = NIFTY_50_TICKERS) -> pd.DataFrame:
    rows = []
    for ticker in tickers:
        try:
            rows.append(fetch_fundamentals_yfinance(ticker))
        except Exception as e:
            print(f"[fundamental] failed for {ticker}: {e}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = build_fundamental_dataset()
    print(df.shape)
    print(df.head())
