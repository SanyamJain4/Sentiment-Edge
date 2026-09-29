"""
Loads the trained model and scores the latest feature row per ticker,
producing a ranked list of Nifty 50 stocks by predicted upward-move
probability.
"""
import joblib
import pandas as pd

from config.settings import MODELS_DIR


def load_model(path=None):
    path = path or (MODELS_DIR / "nifty50_direction_model.joblib")
    bundle = joblib.load(path)
    return bundle["model"], bundle["feature_cols"]


def score_latest(master_df: pd.DataFrame) -> pd.DataFrame:
    """Takes the master feature table, scores the most recent date's
    row for every ticker, returns a ranked DataFrame."""
    model, feature_cols = load_model()

    latest = master_df.sort_values("Date").groupby("Ticker").tail(1).copy()
    latest["Ticker"] = latest["Ticker"].astype("category")

    latest["Predicted_Up_Probability"] = model.predict_proba(latest[feature_cols])[:, 1]
    ranked = latest[["Ticker", "Date", "Predicted_Up_Probability"]].sort_values(
        "Predicted_Up_Probability", ascending=False
    )
    return ranked.reset_index(drop=True)


if __name__ == "__main__":
    master_df = pd.read_parquet(MODELS_DIR.parent / "features" / "master_features.parquet")
    ranked = score_latest(master_df)
    print(ranked.head(10))
