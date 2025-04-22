import matplotlib.pyplot as plt
import pandas as pd
import re
import os

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

# === Create the observations folder if it doesn't exist ===
observations_folder = '/home/nikhil/Desktop/finalproj/observations/'
if not os.path.exists(observations_folder):
    os.makedirs(observations_folder)

# === Plot: Total Mentions per Ticker (top 10) ===
agg_df.head(10).plot(kind='bar', x='Ticker', y='mention_count', legend=False, title="Total Mentions per Ticker", color='lightcoral')
plt.ylabel('Mentions Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(observations_folder, 'total_mentions_per_ticker.png'))
plt.clf()

# === Plot: Average Sentiment per Ticker (top 10) ===
agg_df.head(10).plot(kind='bar', x='Ticker', y='avg_sentiment', legend=False, title="Average Sentiment per Ticker", color='skyblue')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(observations_folder, 'average_sentiment_per_ticker.png'))
plt.clf()

# === Plot: Total Sentiment per Ticker (top 10) ===
agg_df.head(10).plot(kind='bar', x='Ticker', y='total_sentiment', legend=False, title="Total Sentiment per Ticker", color='lightgreen')
plt.ylabel('Total Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(observations_folder, 'total_sentiment_per_ticker.png'))
plt.clf()

# === Plot: Distribution of Sentiment Scores ===
sentiment_df['Sentiment'].plot(kind='hist', bins=30, color='skyblue', title="Distribution of Sentiment Scores")
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(observations_folder, 'sentiment_distribution.png'))
plt.clf()

# === Plot: Top 10 Tickers with Maximum Total Sentiment ===
agg_df.nlargest(10, 'total_sentiment').plot(kind='bar', x='Ticker', y='total_sentiment', legend=False, title="Top 10 Tickers with Maximum Total Sentiment", color='lightcoral')
plt.ylabel('Total Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(observations_folder, 'top_10_tickers_max_total_sentiment.png'))
plt.clf()

# === Plot: Top 10 Tickers with Minimum Average Sentiment ===
agg_df.nsmallest(10, 'avg_sentiment').plot(kind='bar', x='Ticker', y='avg_sentiment', legend=False, title="Top 10 Tickers with Minimum Average Sentiment", color='lightgreen')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(observations_folder, 'top_10_tickers_min_avg_sentiment.png'))
plt.clf()

# === Plot: Sentiment vs Mentions Scatter Plot ===
plt.scatter(agg_df['mention_count'], agg_df['avg_sentiment'], alpha=0.5, color='skyblue')
plt.title('Sentiment vs Mentions')
plt.xlabel('Mentions Count')
plt.ylabel('Average Sentiment')
plt.tight_layout()
plt.savefig(os.path.join(observations_folder, 'sentiment_vs_mentions.png'))
plt.clf()

# === Optional: Sentiment Distribution for Specific Ticker (e.g., IBM) ===
specific_ticker = 'IBM'
specific_df = exploded_df[exploded_df['Ticker'] == specific_ticker]
specific_df['Sentiment'].plot(kind='hist', bins=30, color='lightcoral', title=f"Sentiment Distribution for {specific_ticker}")
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig(os.path.join(observations_folder, f'sentiment_distribution_{specific_ticker}.png'))
plt.clf()

# === Print and Save Outputs ===
print("\n📄 Sample: Cleaned & Exploded Data")
print(exploded_df.head())

print("\n📊 Sample: Aggregated Sentiment per Ticker")
print(agg_df.head())

# Save Outputs
exploded_df.to_csv('/home/nikhil/Desktop/finalproj/cleaned_sentiment_exploded.csv', index=False)
agg_df.to_csv('/home/nikhil/Desktop/finalproj/aggregated_sentiment_per_ticker.csv', index=False)
