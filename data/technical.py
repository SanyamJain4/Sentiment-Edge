"""
Technical analysis pillar.

Fetches OHLCV data for a list of tickers and computes indicators.
Output: one long-format DataFrame with columns
    [Ticker, Date, Open, High, Low, Close, Volume, SMA_50, EMA_50,
     RSI, MACD, MACD_Signal, Upper_Band, Lower_Band, Rel_Strength_vs_Index]
"""
import numpy as np
import pandas as pd
import talib
import yfinance as yf

from config.settings import YEARS_OF_HISTORY
from config.tickers import NIFTY_50_TICKERS

NIFTY_INDEX_TICKER = "^NSEI"  # Nifty 50 index itself, for relative strength


def fetch_ohlcv(ticker: str, years: int = YEARS_OF_HISTORY) -> pd.DataFrame:
    """Download OHLCV history for a single ticker."""
    from datetime import datetime, timedelta

    start = (datetime.today() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    end = datetime.today().strftime("%Y-%m-%d")

    df = yf.download(ticker, start=start, end=end, auto_adjust=False, progress=False)
    if df.empty:
        return df
    # yfinance may return a MultiIndex even when downloading one ticker.
    # Reduce it to the ordinary OHLCV columns expected below.
    if isinstance(df.columns, pd.MultiIndex):
        for level in range(df.columns.nlevels):
            if ticker in df.columns.get_level_values(level):
                df = df.xs(ticker, axis=1, level=level)
                break
    df.dropna(inplace=True)
    df.reset_index(inplace=True)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df["Ticker"] = ticker
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add TA-Lib indicators to a single-ticker OHLCV DataFrame."""
    close = np.asarray(df["Close"]).astype(float).flatten()

    df["SMA_50"] = df["Close"].rolling(window=50).mean()
    df["EMA_50"] = df["Close"].ewm(span=50, adjust=False).mean()
    df["RSI"] = talib.RSI(close, timeperiod=14)
    df["MACD"], df["MACD_Signal"], _ = talib.MACD(close)
    df["Upper_Band"], _, df["Lower_Band"] = talib.BBANDS(close, timeperiod=20)

    # Label for supervised learning: NEXT trading day's direction,
    # not same-day (avoids the look-ahead leak in the original notebook).
    df["Future_Close"] = df["Close"].shift(-1)
    df["Price_Change_Fwd"] = (df["Future_Close"] - df["Close"]) / df["Close"]
    df["Stock_Movement"] = (df["Price_Change_Fwd"] > 0).astype("Int64")
    df.loc[df["Future_Close"].isna(), "Stock_Movement"] = pd.NA

    return df


def add_relative_strength(df: pd.DataFrame, index_df: pd.DataFrame) -> pd.DataFrame:
    """Add each stock's return relative to the Nifty 50 index return, same day."""
    idx = index_df[["Date", "Close"]].rename(columns={"Close": "Index_Close"})
    idx["Index_Return"] = idx["Index_Close"].pct_change()

    merged = df.merge(idx[["Date", "Index_Return"]], on="Date", how="left")
    merged["Stock_Return"] = merged["Close"].pct_change()
    merged["Rel_Strength_vs_Index"] = merged["Stock_Return"] - merged["Index_Return"]
    return merged


def build_technical_dataset(tickers: list[str] = NIFTY_50_TICKERS) -> pd.DataFrame:
    """Fetch + compute technical indicators for every ticker. Returns long-format df."""
    index_df = fetch_ohlcv(NIFTY_INDEX_TICKER)

    all_rows = []
    for ticker in tickers:
        raw = fetch_ohlcv(ticker)
        if raw.empty:
            print(f"[technical] no data for {ticker}, skipping")
            continue
        enriched = add_technical_indicators(raw)
        enriched = add_relative_strength(enriched, index_df)
        all_rows.append(enriched)

    return pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()


if __name__ == "__main__":
    df = build_technical_dataset()
    print(df.shape)
    print(df.head())
