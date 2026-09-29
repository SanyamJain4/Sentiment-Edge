"""
Global settings and paths.
"""
from pathlib import Path

# --- Paths -------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data_cache"
DATA_DIR.mkdir(exist_ok=True)

RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
MODELS_DIR = DATA_DIR / "models"
for d in (RAW_DIR, FEATURES_DIR, MODELS_DIR):
    d.mkdir(exist_ok=True)

# --- Date range ----------------------------------------------------------
YEARS_OF_HISTORY = 2

# --- News RSS sources (India-focused) ------------------------------------
NEWS_RSS_TEMPLATES = [
    "https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en",
    "https://www.moneycontrol.com/rss/buzzingstocks.xml",   # not query-able, general feed
    "https://economictimes.indiatimes.com/markets/stocks/rssfeeds/2146842.cms",  # general markets feed
]

# --- Sentiment model -------------------------------------------------------
# FinBERT is fine-tuned on financial text and handles domain language
# ("beats estimates", "misses guidance") far better than VADER.
SENTIMENT_MODEL_NAME = "ProsusAI/finbert"

# --- Modeling --------------------------------------------------------------
PREDICTION_HORIZON_DAYS = 1     # predict next N trading days' direction
TEST_SET_FRACTION = 0.2         # last 20% of dates, NOT a random split
RANDOM_SEED = 42
