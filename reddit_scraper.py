import praw
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk import download

# Download the VADER lexicon for sentiment analysis
download('vader_lexicon')

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

# Set up the Reddit API client (replace with your credentials)
reddit = praw.Reddit(client_id='YOUR_CLIENT_ID',
                     client_secret='YOUR_CLIENT_SECRET',
                     user_agent='YOUR_USER_AGENT')

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

print(df_posts.head())
