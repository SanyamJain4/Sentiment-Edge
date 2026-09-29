"""
Merges the three pillars (technical, fundamental, sentiment) into a single
feature table keyed by (Ticker, Date), ready for modeling.

Alignment logic:
  - Technical: exact match on (Ticker, Date) — this is the daily backbone.
  - Sentiment: exact match on (Ticker, Date); if missing, look back up to
    N days for the most recent score for that ticker (sentiment doesn't
    arrive every trading day).
  - Fundamental: as-of join — since fundamentals update quarterly, each
    stock's Latest fundamental snapshot is broadcast across all its
    daily rows (simplification; a stricter version would only use
    fundamentals with As_Of_Date <= Date, to avoid leaking future
    quarterly results into past predictions).
"""
import pandas as pd

SENTIMENT_LOOKBACK_DAYS = 6


def align_sentiment(technical_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Attach a sentiment_score column to technical_df, per ticker with lookback."""
    if sentiment_df.empty:
        technical_df["sentiment_score"] = pd.NA
        return technical_df

    out_rows = []
    for ticker, group in technical_df.groupby("Ticker"):
        sent_t = sentiment_df[sentiment_df["Ticker"] == ticker].sort_values("Date")

        scores = []
        for d in group["Date"]:
            exact = sent_t[sent_t["Date"] == d]
            if not exact.empty:
                scores.append(exact["sentiment_score"].values[0])
                continue
            window_start = d - pd.Timedelta(days=SENTIMENT_LOOKBACK_DAYS)
            recent = sent_t[(sent_t["Date"] >= window_start) & (sent_t["Date"] < d)]
            scores.append(recent["sentiment_score"].iloc[-1] if not recent.empty else None)

        group = group.copy()
        group["sentiment_score"] = scores
        out_rows.append(group)

    return pd.concat(out_rows, ignore_index=True)


def attach_fundamentals(df: pd.DataFrame, fundamental_df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast each ticker's latest fundamental snapshot across all its rows."""
    if fundamental_df.empty:
        return df

    fundamental_cols = [c for c in fundamental_df.columns if c not in ("Ticker", "As_Of_Date")]
    return df.merge(fundamental_df[["Ticker"] + fundamental_cols], on="Ticker", how="left")


FEATURE_COLUMNS = [
    "sentiment_score", "Close", "SMA_50", "EMA_50", "RSI", "MACD", "MACD_Signal",
    "Upper_Band", "Lower_Band", "Rel_Strength_vs_Index",
    "PE_Ratio", "PB_Ratio", "ROE", "Debt_To_Equity", "Profit_Margin",
    "Revenue_Growth_YoY", "Earnings_Growth_YoY",
]
TARGET_COLUMN = "Stock_Movement"
ID_COLUMNS = ["Ticker", "Date"]


def build_master_feature_table(technical_df, fundamental_df, sentiment_df) -> pd.DataFrame:
    df = align_sentiment(technical_df, sentiment_df)
    df = attach_fundamentals(df, fundamental_df)

    keep = ID_COLUMNS + [c for c in FEATURE_COLUMNS if c in df.columns] + [TARGET_COLUMN]
    df = df[keep].dropna(subset=[TARGET_COLUMN])
    return df


if __name__ == "__main__":
    from data.technical import build_technical_dataset
    from data.fundamental import build_fundamental_dataset
    from data.sentiment import build_sentiment_dataset
    from config.tickers import NIFTY_50_TICKERS

    sample_tickers = NIFTY_50_TICKERS[:3]
    tech = build_technical_dataset(sample_tickers)
    fund = build_fundamental_dataset(sample_tickers)
    sent = build_sentiment_dataset(sample_tickers)

    master = build_master_feature_table(tech, fund, sent)
    print(master.shape)
    print(master.head())
