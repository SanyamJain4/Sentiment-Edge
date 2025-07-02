import streamlit as st
import pandas as pd
import numpy as np
from model1 import predict_sentiment
from newsscrapper1 import fetch_news
from reddit_scraper import fetch_reddit_posts

# Page configuration
st.set_page_config(page_title="Sentiment Edge", layout="centered")

# Title and description
st.title("📈 Sentiment Edge")
st.write("Analyze sentiment from **News** and **Reddit** on your query.")

# Sidebar for user input
st.sidebar.header("User Input")
query = st.sidebar.text_input("Enter a topic to analyze:", value="Stock Market")

source = st.sidebar.radio("Choose data source:", ["News", "Reddit"])

if st.sidebar.button("Analyze"):
    st.subheader(f"Sentiment Analysis Results for: `{query}`")
    
    # Fetch data from the selected source
    if source == "News":
        texts = fetch_news(query)
    else:
        texts = fetch_reddit_posts(query)

    if not texts:
        st.warning("No data found for this query.")
    else:
        # Display raw texts (optional)
        with st.expander("🔍 Show Raw Data"):
            for text in texts:
                st.write(f"- {text}")
        
        # Predict sentiment using model
        results = predict_sentiment(texts)
        sentiments = pd.Series(results).value_counts()

        # Display results
        st.write("### 📊 Sentiment Distribution")
        st.bar_chart(sentiments)

        st.write("### ✅ Summary")
        for label, count in sentiments.items():
            st.write(f"**{label}**: {count} articles/posts")

# Footer
st.markdown("---")
st.markdown("Built by [Sanyam Jain](https://github.com/SanyamJain4)")

