import requests
from bs4 import BeautifulSoup
import pandas as pd
from nltk.sentiment import SentimentIntensityAnalyzer

# Initialize SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()

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
url = "https://www.bbc.com/news/business"  # Replace with a relevant stock news page
stock_news = scrape_stock_news(url)

# Convert news into a DataFrame
df_news = pd.DataFrame(stock_news, columns=['title'])

# Perform sentiment analysis on the scraped news titles
df_news['sentiment_score'] = df_news['title'].apply(lambda x: sia.polarity_scores(x)['compound'])

# Save the news data
df_news.to_csv("scraped_news.csv", index=False)

print(df_news.head())
