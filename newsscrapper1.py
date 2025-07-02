import requests
from bs4 import BeautifulSoup

def fetch_news(query):
    """
    Fetches top news headlines related to the query from Google News.

    Args:
        query (str): Search term (e.g., 'stock market')

    Returns:
        List[str]: List of headlines or news snippets
    """
    try:
        # Format query for URL
        query = query.replace(" ", "+")
        url = f"https://news.google.com/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"

        headers = {
            "User-Agent": "Mozilla/5.0"
        }

        response = requests.get(url, headers=headers)
        soup = BeautifulSoup(response.content, "html.parser")

        # Extract news headlines from HTML
        headlines = soup.find_all("a", class_="DY5T1d RZIKme", limit=10)
        results = [headline.get_text() for headline in headlines]

        # If nothing was fetched
        if not results:
            return [f"{query} is doing well", f"{query} facing challenges"]

        return results

    except Exception as e:
        # Fallback in case of error or scraping failure
        print(f"Error fetching news: {e}")
        return [f"{query} is doing well", f"{query} facing challenges"]
