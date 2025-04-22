import pandas as pd
import re

# === Load CSVs ===
sentiment_df = pd.read_csv('/home/nikhil/Desktop/finalproj/stock_data_sentiments_cleaned.csv')
ticker_df = pd.read_csv('/home/nikhil/Desktop/finalproj/Yahoo-Finance-Ticker-Symbols.csv')

# === Clean ticker symbols ===
ticker_df['Ticker'] = ticker_df['Ticker'].astype(str).str.strip().str.upper()
valid_tickers = set(ticker_df['Ticker'].unique())

# === Extract tickers from text ===
def extract_tickers(text):
    text = str(text).upper()  # Convert text to uppercase to match tickers
    words = re.findall(r'\b[A-Z]{2,5}\b', text)  # Extract all words with 2 to 5 uppercase letters
    return [word for word in words if word in valid_tickers]  # Remove stopwords filter

# Apply the function to the Text column and create a new column 'Tickers'
sentiment_df['Tickers'] = sentiment_df['Text'].apply(extract_tickers)

# Filter out rows where no valid tickers were found
sentiment_df = sentiment_df[sentiment_df['Tickers'].map(len) > 0].reset_index(drop=True)

# === Explode tickers so each row is (Text, Sentiment, Ticker) ===
exploded_df = sentiment_df.explode('Tickers').rename(columns={'Tickers': 'Ticker'}).reset_index(drop=True)

# === Aggregate sentiment data per ticker ===
agg_df = exploded_df.groupby('Ticker').agg(
    mention_count=('Sentiment', 'count'),
    avg_sentiment=('Sentiment', 'mean'),
    total_sentiment=('Sentiment', 'sum')
).sort_values(by='mention_count', ascending=False).reset_index()

# === Print and Save Outputs ===
print("\n📄 Sample: Cleaned & Exploded Data")
print(exploded_df.head())

print("\n📊 Sample: Aggregated Sentiment per Ticker")
print(agg_df.head())

# Optional: Save for submission
exploded_df.to_csv('/home/nikhil/Desktop/finalproj/cleaned_sentiment_exploded.csv', index=False)
agg_df.to_csv('/home/nikhil/Desktop/finalproj/aggregated_sentiment_per_ticker.csv', index=False)