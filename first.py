import pandas as pd
import re

# Load CSVs
sentiment_df = pd.read_csv('/home/nikhil/Desktop/finalproj/stock_data_sentiments_cleaned.csv')
ticker_df = pd.read_csv('/home/nikhil/Desktop/finalproj/Yahoo-Finance-Ticker-Symbols.csv')

# Clean ticker data
ticker_df['Ticker'] = ticker_df['Ticker'].astype(str).str.strip().str.upper()
valid_tickers = set(ticker_df['Ticker'].unique())

# Optional: Stopword filter
stopwords = {'FOR', 'TO', 'ON', 'MY', 'SEE', 'ARE', 'AND', 'WITH', 'THE'}

# Function to extract tickers
def extract_tickers(text):
    text = str(text).upper()
    words = re.findall(r'\b[A-Z]{2,5}\b', text)
    return [word for word in words if word in valid_tickers and word not in stopwords]

# Apply and filter
sentiment_df['Tickers'] = sentiment_df['Text'].apply(extract_tickers)
sentiment_df = sentiment_df[sentiment_df['Tickers'].map(len) > 0].reset_index(drop=True)

# Show cleaned output
print(sentiment_df[['Text', 'Sentiment', 'Tickers']].head())
