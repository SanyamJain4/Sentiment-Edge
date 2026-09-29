# Nifty 50 Multi-Factor Stock Direction Model

Predicts next-day up/down movement for Nifty 50 constituents by combining
three signal pillars: **technical**, **fundamental**, and **NLP sentiment**
(FinBERT on live news).

## Structure

```
config/
  tickers.py       Nifty 50 ticker list + company-name lookup (for news search)
  settings.py       Paths, model/date settings

data/
  technical.py       OHLCV + TA-Lib indicators, relative strength vs Nifty index
  fundamental.py      Valuation/profitability ratios (yfinance, screener.in TODO)
  sentiment.py         Live news (RSS), scored with FinBERT

features/
  build_features.py   Merges the three pillars into one (Ticker, Date) table

models/
  train.py           Chronological train/test split, LightGBM classifier,
                       saves eval_artifacts.joblib (confusion matrix, feature
                       importances, per-row test predictions) for the dashboard
  predict.py           Scores the latest day, ranks stocks by up-probability

viz/
  visualization.py    Plotly chart builders: price+signal markers, MACD
                       buy/sell crossovers, RSI, sentiment timeline,
                       confusion matrix, feature importance, prediction confidence

pipeline/
  run_daily.py         Orchestrates steps 1-4 end to end

streamlit_app.py      Interactive dashboard — pick a stock, see all charts
smoke_test.py          Runs the whole pipeline + all charts on synthetic data
requirements.txt
```

---

## Complete steps to run this

### 1. Unzip and enter the project folder

```bash
unzip nifty50_project.zip
cd nifty50_project
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
```

Keeps these dependencies isolated from your system Python.

### 3. Install the TA-Lib C library (needed before the Python package will install)

- **macOS:** `brew install ta-lib`
- **Ubuntu/Debian:**
  ```bash
  sudo apt-get update && sudo apt-get install -y build-essential wget
  wget https://sourceforge.net/projects/ta-lib/files/ta-lib/0.4.0/ta-lib-0.4.0-src.tar.gz
  tar -xzf ta-lib-0.4.0-src.tar.gz && cd ta-lib
  ./configure --prefix=/usr && make && sudo make install
  cd ..
  ```
- **Windows:** compiling from source is painful — easiest is installing an unofficial prebuilt `TA-Lib` wheel matching your Python version (search "TA-Lib Christoph Gohlke wheel"), then `pip install <the_downloaded_wheel.whl>`.

### 4. Install Python dependencies

```bash
pip install -r requirements.txt
```

This pulls in `yfinance`, `TA-Lib`, `pandas`, `lightgbm`, `feedparser`, `transformers`, `torch`, `pyarrow`. The `transformers`/`torch` install is the slowest part — a few minutes and a few GB of disk on first install (needed for FinBERT sentiment scoring).

### 5. Fill in the company-name lookup

Open `config/tickers.py` and complete `TICKER_TO_COMPANY_NAME` for all 50 tickers (only 4 are filled in as examples). This name drives the Google News search query per stock, so accuracy here matters directly for sentiment quality. You can bulk-generate it once with:

```python
import yfinance as yf
name = yf.Ticker("RELIANCE.NS").info["longName"]
```

Cache the result somewhere so you're not re-fetching all 50 names every run.

### 6. Smoke-test with synthetic data first (optional but recommended)

```bash
python smoke_test.py
```

This exercises the full pipeline logic — technical indicators, feature merging, chronological split, model training, scoring — using generated fake data, so you can confirm everything runs end to end *before* burning time/API calls on live data. No network access required for this step.

### 7. Test each live data source individually

Before running the full 50-ticker pipeline, test each pillar alone so failures are easy to isolate:

```bash
python -m data.technical      # prints OHLCV + indicators for all 50 tickers
python -m data.fundamental    # prints fundamental ratios for all 50 tickers
python -m data.sentiment      # edit its __main__ block to test 2-3 tickers first
```

`data/sentiment.py`'s `__main__` block fetches all 50 tickers from the previous 30 days. FinBERT downloads ~440MB on first run, and Google News RSS calls for 50 tickers can take a few minutes, so you can temporarily use `NIFTY_50_TICKERS[:3]` while confirming the setup.

### 8. Run the full pipeline once to build the feature table

```bash
python -m pipeline.run_daily
```

This fetches all three pillars for all 50 tickers, merges them, and saves `data_cache/features/master_features.parquet`. It will print **"No trained model found yet"** at the end — that's expected on the first run, since nothing has been trained.

### 9. Train the model

```bash
python -m models.train
```

Trains LightGBM on the saved feature table using a chronological (time-based) split, prints accuracy/classification report/confusion matrix, and saves the model to `data_cache/models/nifty50_direction_model.joblib`.

### 10. Re-run the pipeline to get live predictions

```bash
python -m pipeline.run_daily
```

This time it finds the trained model and prints a ranked list of all 50 Nifty stocks by predicted next-day up-probability.

### 11. Launch the interactive dashboard

```bash
streamlit run streamlit_app.py
```

Opens in your browser at `http://localhost:8501`. Pick any Nifty 50 stock from the sidebar to see:
- **Technical & Signals tab:** price chart with SMA/EMA/Bollinger overlay and green/red up-down direction markers, MACD with buy/sell crossover markers, RSI with overbought/oversold zones
- **Sentiment tab:** daily FinBERT sentiment score over time
- **Fundamentals tab:** latest P/E, P/B, ROE, Debt/Equity, and growth ratios
- **Model Performance tab:** test-set accuracy, confusion matrix, feature importance ranking, and predicted-probability-vs-actual-outcome for that specific stock

The dashboard reads whatever's already in `data_cache/` — it doesn't retrain or refetch anything itself, so re-run steps 8-10 whenever you want it showing fresher data.

---

## Things to know before your first live run

- **First full run will be slow.** 50 tickers × 3 data sources, plus FinBERT scoring hundreds of headlines on CPU. Consider testing on `NIFTY_50_TICKERS[:5]` in `config/tickers.py` before committing to all 50.
- **yfinance `.info` coverage for NSE tickers is genuinely spotty.** Expect some `None` fundamentals for smaller-cap Nifty constituents — that's a data-source limitation, not a bug.
- **Re-running trains on whatever's in `master_features.parquet`.** If you want a fresh daily retrain, re-run steps 8 and 9 each day (or wire that into a scheduler/cron job — not included here).

## What's stubbed vs. implemented

| Pillar | Status |
|---|---|
| Technical | Fully implemented: SMA/EMA/RSI/MACD/Bollinger + relative strength vs Nifty index, chronological labeling fixed |
| Fundamental | yfinance-based ratios implemented; screener.in scraper for promoter holding % is a TODO stub |
| Sentiment | News RSS fetch implemented; FinBERT scoring implemented (replaces VADER) |
| Modeling | LightGBM with proper time-based split implemented |

## Known design decisions worth revisiting

- **Fundamentals are broadcast across all daily rows for a ticker** rather
  than strictly as-of-joined by filing date. For a first pass this is fine;
  tightening it avoids leaking future-quarter fundamentals into past dates.
- **One shared model across all 50 tickers** (ticker as a categorical
  feature) rather than 50 separate models. Simpler to start with; revisit
  if certain stocks behave very differently (e.g. PSU banks vs IT services).
- **Sentiment lookback window is 6 days** when no same-day news mention
  exists. News volume per stock (news-only, no Reddit) is thin, so this
  window may need widening — see `SENTIMENT_LOOKBACK_DAYS` in
  `features/build_features.py`.
