"""
Smoke test: exercises the real pipeline code (technical indicators,
feature merging, chronological split, model training/scoring, and
dashboard visualizations) using synthetic data instead of live
yfinance/News calls, since this sandbox has no network access to those
hosts.

This does NOT test the actual data-fetching functions (fetch_ohlcv,
fetch_news_for_ticker) — those need real network access to verify.
It tests everything downstream of them, including every chart function
used by streamlit_app.py.
"""
import numpy as np
import pandas as pd

np.random.seed(42)

TICKERS = ["RELIANCE.NS", "TCS.NS", "INFY.NS"]
N_DAYS = 300


def make_synthetic_ohlcv(ticker: str, n_days: int = N_DAYS) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days).date
    price = 1000 + np.cumsum(np.random.randn(n_days) * 5)
    price = np.maximum(price, 50)  # keep positive

    df = pd.DataFrame({
        "Date": dates,
        "Open": price + np.random.randn(n_days),
        "High": price + abs(np.random.randn(n_days)) * 2,
        "Low": price - abs(np.random.randn(n_days)) * 2,
        "Close": price,
        "Volume": np.random.randint(1_000_000, 5_000_000, n_days),
    })
    df["Ticker"] = ticker
    return df


def make_synthetic_sentiment(ticker: str, n_days: int = N_DAYS) -> pd.DataFrame:
    # Sparse: sentiment doesn't arrive every day, like real news
    dates = pd.bdate_range(end=pd.Timestamp.today(), periods=n_days).date
    sample_dates = np.random.choice(dates, size=int(n_days * 0.4), replace=False)
    scores = np.random.uniform(-1, 1, len(sample_dates))
    return pd.DataFrame({"Ticker": ticker, "Date": sample_dates, "sentiment_score": scores})


def make_synthetic_fundamentals(ticker: str) -> dict:
    return {
        "Ticker": ticker,
        "PE_Ratio": np.random.uniform(15, 40),
        "Forward_PE": np.random.uniform(15, 40),
        "PB_Ratio": np.random.uniform(2, 10),
        "ROE": np.random.uniform(0.1, 0.3),
        "Debt_To_Equity": np.random.uniform(0, 1.5),
        "Profit_Margin": np.random.uniform(0.05, 0.25),
        "Revenue_Growth_YoY": np.random.uniform(-0.05, 0.2),
        "Earnings_Growth_YoY": np.random.uniform(-0.1, 0.3),
        "Market_Cap": np.random.uniform(5e11, 2e13),
        "As_Of_Date": pd.Timestamp.today().date(),
    }


def run_smoke_test():
    from data.technical import add_technical_indicators, add_relative_strength
    from features.build_features import build_master_feature_table
    from models.train import train_model
    from models.predict import load_model

    print("=== Step 1: synthetic technical data ===")
    index_df = make_synthetic_ohlcv("^NSEI")
    technical_rows = []
    for ticker in TICKERS:
        raw = make_synthetic_ohlcv(ticker)
        enriched = add_technical_indicators(raw)
        enriched = add_relative_strength(enriched, index_df)
        technical_rows.append(enriched)
    technical_df = pd.concat(technical_rows, ignore_index=True)
    print(technical_df.shape, "columns:", list(technical_df.columns))
    print(technical_df[["Ticker", "Date", "Close", "RSI", "MACD", "Stock_Movement"]].tail(3))

    print("\n=== Step 2: synthetic fundamental data ===")
    fundamental_df = pd.DataFrame([make_synthetic_fundamentals(t) for t in TICKERS])
    print(fundamental_df)

    print("\n=== Step 3: synthetic sentiment data ===")
    sentiment_df = pd.concat([make_synthetic_sentiment(t) for t in TICKERS], ignore_index=True)
    print(sentiment_df.shape, "sample:")
    print(sentiment_df.head())

    print("\n=== Step 4: merge into master feature table ===")
    master_df = build_master_feature_table(technical_df, fundamental_df, sentiment_df)
    print(master_df.shape)
    print(master_df.head())
    print("\nNulls per column:\n", master_df.isnull().sum())

    print("\n=== Step 5: train model (chronological split) ===")
    model, feature_cols, eval_artifacts = train_model(master_df)

    print("\n=== Step 6: score latest day ===")
    from models.train import save_model, save_eval_artifacts
    save_model(model, feature_cols)
    save_eval_artifacts(eval_artifacts)

    from models.predict import score_latest
    ranked = score_latest(master_df)
    print(ranked)

    print("\n=== Step 7: generate all dashboard visualizations ===")
    from viz.visualization import (
        plot_price_with_signals, plot_macd_with_signals, plot_rsi,
        plot_sentiment_timeline, plot_confusion_matrix,
        plot_feature_importance, plot_prediction_confidence,
    )

    sample_ticker = TICKERS[0]
    sample_df = master_df[master_df["Ticker"] == sample_ticker]

    figs = {
        "price_with_signals": plot_price_with_signals(sample_df, sample_ticker),
        "macd_with_signals": plot_macd_with_signals(sample_df, sample_ticker),
        "rsi": plot_rsi(sample_df, sample_ticker),
        "sentiment_timeline": plot_sentiment_timeline(sample_df, sample_ticker),
        "confusion_matrix": plot_confusion_matrix(eval_artifacts["confusion_matrix"]),
        "feature_importance": plot_feature_importance(eval_artifacts["feature_importances"]),
    }
    if sample_ticker in eval_artifacts["predictions_df"]["Ticker"].unique():
        figs["prediction_confidence"] = plot_prediction_confidence(eval_artifacts["predictions_df"], sample_ticker)

    for name, fig in figs.items():
        assert fig is not None and len(fig.data) > 0, f"{name} produced an empty figure"
        print(f"  ✓ {name}: {len(fig.data)} trace(s)")

    print("\n✅ Smoke test completed — pipeline + visualization logic runs end to end.")


if __name__ == "__main__":
    run_smoke_test()
