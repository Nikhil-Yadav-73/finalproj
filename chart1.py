import matplotlib.pyplot as plt
import pandas as pd
import re

# === Load CSVs ===
sentiment_df = pd.read_csv('/home/nikhil/Desktop/finalproj/stock_data_sentiments_cleaned.csv')
ticker_df = pd.read_csv('/home/nikhil/Desktop/finalproj/Yahoo-Finance-Ticker-Symbols.csv')

# === Clean ticker symbols ===
ticker_df['Ticker'] = ticker_df['Ticker'].astype(str).str.strip().str.upper()
valid_tickers = set(ticker_df['Ticker'].unique())

# Optional stopwords to filter out common false matches
stopwords = {'FOR', 'TO', 'ON', 'MY', 'SEE', 'ARE', 'AND', 'WITH', 'THE'}

# === Extract tickers from text ===
def extract_tickers(text):
    text = str(text).upper()
    words = re.findall(r'\b[A-Z]{2,5}\b', text)
    return [word for word in words if word in valid_tickers and word not in stopwords]

sentiment_df['Tickers'] = sentiment_df['Text'].apply(extract_tickers)
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

# Plot: Average sentiment per ticker (top 10)
agg_df.head(10).plot(kind='bar', x='Ticker', y='avg_sentiment', legend=False, title="Average Sentiment per Ticker", color='skyblue')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
