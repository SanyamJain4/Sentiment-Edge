"""
End-to-end daily pipeline:
  1. Pull technical data for all Nifty 50 tickers
  2. Pull fundamental data (cheap to refresh; safe to run daily even
     though the underlying numbers only change quarterly)
  3. Pull live news sentiment, score with FinBERT
  4. Merge into master feature table, save to disk
  5. Score latest day with the trained model, print/save ranked list

Run: python -m pipeline.run_daily
"""
import pandas as pd

from config.settings import FEATURES_DIR
from config.tickers import NIFTY_50_TICKERS
from data.technical import build_technical_dataset
from data.fundamental import build_fundamental_dataset
from data.sentiment import build_sentiment_dataset
from features.build_features import build_master_feature_table
from models.predict import score_latest


def main(tickers=NIFTY_50_TICKERS):
    print(f"Running pipeline for {len(tickers)} tickers...")

    print("[1/4] Technical data...")
    technical_df = build_technical_dataset(tickers)

    print("[2/4] Fundamental data...")
    fundamental_df = build_fundamental_dataset(tickers)

    print("[3/4] Sentiment data...")
    sentiment_df = build_sentiment_dataset(tickers)

    print("[4/4] Merging into master feature table...")
    master_df = build_master_feature_table(technical_df, fundamental_df, sentiment_df)

    out_path = FEATURES_DIR / "master_features.parquet"
    master_df.to_parquet(out_path, index=False)
    print(f"Saved master feature table ({master_df.shape}) to {out_path}")

    try:
        ranked = score_latest(master_df)
        print("\nToday's ranked predictions (top 10):")
        print(ranked.head(10).to_string(index=False))
    except FileNotFoundError:
        print("\nNo trained model found yet — run models/train.py first.")


if __name__ == "__main__":
    main()
