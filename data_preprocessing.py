import pandas as pd
import talib
import yfinance as yf

# Download historical stock data for Tesla (TSLA)
stock_data = yf.download('TSLA', start='2022-01-01', end='2024-01-01')

# Calculate technical indicators
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()  # 50-day simple moving average
stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()  # 50-day exponential moving average

# Relative Strength Index (RSI)
stock_data['RSI'] = talib.RSI(stock_data['Close'].to_numpy().ravel(), timeperiod=14)

# Moving Average Convergence Divergence (MACD)
stock_data['MACD'], stock_data['MACD_Signal'], _ = talib.MACD(stock_data['Close'].to_numpy().ravel(), fastperiod=12, slowperiod=26, signalperiod=9)

# Bollinger Bands
stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = talib.BBANDS(stock_data['Close'].to_numpy().ravel(), timeperiod=20)

# Calculate daily stock price change (movement)
stock_data['Price Change'] = stock_data['Close'].pct_change()

# Classify stock movement (up or down)
stock_data['Stock Movement'] = stock_data['Price Change'].apply(lambda x: 1 if x > 0 else 0)  # 1 for upward movement, 0 for downward

# Load sentiment data from Reddit and news
df_posts = pd.read_csv("reddit_posts.csv")
df_news = pd.read_csv("scraped_news.csv")

# Convert 'date' column in df_posts to datetime
df_posts['date'] = pd.to_datetime(df_posts['date']).dt.date

# Aggregate sentiment by day (average sentiment score per day)
df_sentiment_daily = pd.concat([
    df_posts.groupby('date')['sentiment_score'].mean(),
    df_news.groupby('date')['sentiment_score'].mean()
]).reset_index()

# Manually align the sentiment data and stock data by matching dates
aligned_data = []

for date in stock_data['Date']:
    if date in df_sentiment_daily['date'].values:
        sentiment_score = df_sentiment_daily[df_sentiment_daily['date'] == date]['sentiment_score'].values[0]
        stock_movement = stock_data[stock_data['Date'] == date]['Stock Movement'].values[0]
        # Add technical indicators as features
        sma_50 = stock_data[stock_data['Date'] == date]['SMA_50'].values[0]
        ema_50 = stock_data[stock_data['Date'] == date]['EMA_50'].values[0]
        rsi = stock_data[stock_data['Date'] == date]['RSI'].values[0]
        macd = stock_data[stock_data['Date'] == date]['MACD'].values[0]
        upper_band = stock_data[stock_data['Date'] == date]['Upper_Band'].values[0]
        lower_band = stock_data[stock_data['Date'] == date]['Lower_Band'].values[0]

        aligned_data.append([date, sentiment_score, stock_movement, sma_50, ema_50, rsi, macd, upper_band, lower_band])

# Create a new DataFrame from the aligned data
aligned_df = pd.DataFrame(aligned_data, columns=['Date', 'sentiment_score', 'Stock Movement', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band'])

# Save the aligned data for later use
aligned_df.to_csv("aligned_data.csv", index=False)

print(aligned_df.head())
