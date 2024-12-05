!pip install praw
# Download TA-Lib source code
!wget https://github.com/TA-Lib/ta-lib/releases/download/v0.4.0/ta-lib-0.4.0-src.tar.gz

# Extract the source code
!tar -xvzf ta-lib-0.4.0-src.tar.gz

# Change to the source directory
%cd ta-lib
# Configure and install TA-Lib
!./configure --prefix=/usr/local
!make
!make install
# Install the Python package after installing the library
!pip install ta-lib
import talib
print(talib.__version__)
!pip install  pandas nltk yfinance scikit-learn beautifulsoup4 requests 
# Import required libraries
import praw
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download
import yfinance as yf
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from bs4 import BeautifulSoup
import requests
  # For technical indicators

# Download the VADER lexicon for sentiment analysis
download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Set up the Reddit API client (replace with your credentials)
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                     client_secret='YOU_CLIENT_SECRET',
                     user_agent='YOUR_PROJECT_NAME')

# Define subreddit and keyword for scraping
subreddit = reddit.subreddit("stocks")
keyword = "Tesla"

# Collect data from Reddit
posts = []
for post in subreddit.search(keyword, limit=100):  # Adjust the limit as needed
    posts.append({
        'title': post.title,
        'subreddit': post.subreddit.display_name,
        'score': post.score,
        'url': post.url,
        'date': pd.to_datetime(post.created_utc, unit='s')  # Convert UTC timestamp to datetime
    })

# Create a DataFrame
df_posts = pd.DataFrame(posts)

# Perform sentiment analysis on the titles of the posts
df_posts['sentiment_score'] = df_posts['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Save the data for future analysis
df_posts.to_csv("reddit_posts.csv", index=False)

# Clean the text data (removing special characters, etc.)
df_posts['cleaned_title'] = df_posts['title'].apply(lambda x: ' '.join(e.lower() for e in x.split() if e.isalpha()))

# Handle missing values (optional)
df_posts.dropna(subset=['title'], inplace=True)

# Web scraping using BeautifulSoup (Scraping stock-related news articles)
def scrape_stock_news(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # Assuming articles are in <article> tags (you should adjust for real websites)
    articles = soup.find_all('article')
    news = []
    for article in articles:
        title = article.find('h2').text if article.find('h2') else "No Title"
        news.append(title)

    return news

# Example usage of web scraping function
url = "https://www.bbc.com/news/business"  
stock_news = scrape_stock_news(url)

# Convert news into a DataFrame
df_news = pd.DataFrame(stock_news, columns=['title'])

# Perform sentiment analysis on the scraped news titles
df_news['sentiment_score'] = df_news['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Combine the Reddit and scraped news sentiment data for stock prediction
df_combined = pd.concat([df_posts[['date', 'sentiment_score']], df_news], ignore_index=True)

# Download historical stock data for Tesla (TSLA)
stock_data = yf.download('TSLA', start='2022-01-01', end='2024-01-01')

# Calculate technical indicators

# Moving Averages (SMA, EMA)
stock_data['SMA_50'] = stock_data['Close'].rolling(window=50).mean()  # 50-day simple moving average
stock_data['EMA_50'] = stock_data['Close'].ewm(span=50, adjust=False).mean()  # 50-day exponential moving average

# Relative Strength Index (RSI)
# Convert the 'Close' column to a 1-dimensional numpy array


# Calculate technical indicators

# Convert 'Close' to a 1D numpy array and calculate RSI
stock_data['RSI'] = talib.RSI(stock_data['Close'].to_numpy().ravel(), timeperiod=14)

# Moving Average Convergence Divergence (MACD)
stock_data['MACD'], stock_data['MACD_Signal'], _ = talib.MACD(stock_data['Close'].to_numpy().ravel(), fastperiod=12, slowperiod=26, signalperiod=9)

# Bollinger Bands
stock_data['Upper_Band'], stock_data['Middle_Band'], stock_data['Lower_Band'] = talib.BBANDS(stock_data['Close'].to_numpy().ravel(), timeperiod=20)


# Moving Average Convergence Divergence (MACD)

# Bollinger Bands


# Calculate daily stock price change (movement)
stock_data['Price Change'] = stock_data['Close'].pct_change()

# Classify stock movement (up or down)
stock_data['Stock Movement'] = stock_data['Price Change'].apply(lambda x: 1 if x > 0 else 0)  # 1 for upward movement, 0 for downward

# Convert 'date' column in df_combined to datetime and extract the date only (remove time)
df_combined['date'] = pd.to_datetime(df_combined['date'], errors='coerce')  # Convert to datetime if not already
df_combined['date'] = df_combined['date'].dt.date  # Now extract the date only (remove time)

# Ensure the stock data is flattened and its date is in the correct format
stock_data.reset_index(inplace=True)  # Flatten the index to make 'Date' a column
stock_data['Date'] = stock_data['Date'].dt.date  # Convert 'Date' to date only

# Aggregate sentiment by day (average sentiment score per day)
df_sentiment_daily = df_combined.groupby('date')['sentiment_score'].mean().reset_index()

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

# Display the aligned data
print(aligned_df.head())

# Prepare feature and target variables
X = aligned_df[['sentiment_score', 'SMA_50', 'EMA_50', 'RSI', 'MACD', 'Upper_Band', 'Lower_Band']]  # Features
y = aligned_df['Stock Movement']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Use GridSearchCV to tune hyperparameters for the model
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5, 10],
}
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3)
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
best_model = grid_search.best_estimator_

# Make predictions on the test set using the best model
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))



