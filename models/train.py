"""
Trains a classifier to predict next-day stock direction from the master
feature table.

Key differences from the original notebook:
  - Chronological train/test split (last N% of DATES held out), not a
    random 80/20 split. A random split lets the model "see the future"
    during training, which inflates test accuracy in a way that won't
    hold up live.
  - LightGBM instead of RandomForest: handles missing values natively
    (no need to drop every row with a NaN indicator), trains faster
    across 50 tickers x 2 years of daily data, and tends to outperform
    RandomForest on tabular financial features.
  - Ticker is included as a categorical feature so one model learns
    shared patterns across all 50 stocks instead of training 50
    separate models (start here; per-stock models are a reasonable
    follow-up once you have a working baseline).
  - Evaluation artifacts (confusion matrix, per-class report, feature
    importances, and every test-set prediction with its probability)
    are saved alongside the model, so the Streamlit dashboard can show
    real performance charts without needing to retrain.
"""
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from config.settings import MODELS_DIR, RANDOM_SEED, TEST_SET_FRACTION
from features.build_features import FEATURE_COLUMNS, TARGET_COLUMN


def chronological_split(df: pd.DataFrame, test_fraction: float = TEST_SET_FRACTION):
    """Split by DATE, not randomly — everything before the cutoff date is
    train, everything after is test. Prevents future data leaking into
    training."""
    dates_sorted = sorted(df["Date"].unique())
    cutoff_idx = int(len(dates_sorted) * (1 - test_fraction))
    cutoff_date = dates_sorted[cutoff_idx]

    train_df = df[df["Date"] < cutoff_date]
    test_df = df[df["Date"] >= cutoff_date]
    return train_df, test_df, cutoff_date


def train_model(master_df: pd.DataFrame):
    df = master_df.copy()
    df["Ticker"] = df["Ticker"].astype("category")

    train_df, test_df, cutoff_date = chronological_split(df)
    print(f"Train: {len(train_df)} rows, Test: {len(test_df)} rows, cutoff date: {cutoff_date}")

    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns] + ["Ticker"]

    X_train, y_train = train_df[feature_cols], train_df[TARGET_COLUMN].astype(int)
    X_test, y_test = test_df[feature_cols], test_df[TARGET_COLUMN].astype(int)

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=-1,
        learning_rate=0.05,
        random_state=RANDOM_SEED,
    )
    model.fit(X_train, y_train, categorical_feature=["Ticker"])

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")
    print(classification_report(y_test, y_pred))
    print(cm)

    # Per-row predictions with actual outcome, for signal charts / drill-down
    predictions_df = test_df[["Ticker", "Date", TARGET_COLUMN]].copy()
    predictions_df = predictions_df.rename(columns={TARGET_COLUMN: "Actual"})
    predictions_df["Predicted"] = y_pred
    predictions_df["Predicted_Up_Probability"] = y_proba

    # Feature importances (LightGBM's built-in gain-based importance)
    importances = pd.Series(
        model.feature_importances_, index=[c for c in feature_cols if c != "Ticker"] + ["Ticker"]
    ).sort_values(ascending=False)

    eval_artifacts = {
        "accuracy": acc,
        "classification_report": report,
        "confusion_matrix": cm,
        "feature_importances": importances,
        "predictions_df": predictions_df,
        "cutoff_date": cutoff_date,
    }

    return model, feature_cols, eval_artifacts


def save_model(model, feature_cols, path: Path = None):
    path = path or (MODELS_DIR / "nifty50_direction_model.joblib")
    joblib.dump({"model": model, "feature_cols": feature_cols}, path)
    print(f"Saved model to {path}")


def save_eval_artifacts(eval_artifacts: dict, path: Path = None):
    path = path or (MODELS_DIR / "eval_artifacts.joblib")
    joblib.dump(eval_artifacts, path)
    print(f"Saved evaluation artifacts to {path}")


if __name__ == "__main__":
    # Expects a pre-built master feature table; see features/build_features.py
    master_df = pd.read_parquet(MODELS_DIR.parent / "features" / "master_features.parquet")
    model, feature_cols, eval_artifacts = train_model(master_df)
    save_model(model, feature_cols)
    save_eval_artifacts(eval_artifacts)
