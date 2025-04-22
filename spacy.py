import pandas as pd
import spacy
import re

# === Load CSVs ===
sentiment_df = pd.read_csv('/home/nikhil/Desktop/finalproj/stock_data_sentiments_cleaned.csv')
ticker_df = pd.read_csv('/home/nikhil/Desktop/finalproj/Yahoo-Finance-Ticker-Symbols.csv')

# === Clean ticker symbols ===
ticker_df['Ticker'] = ticker_df['Ticker'].astype(str).str.strip().str.upper()
valid_tickers = set(ticker_df['Ticker'].unique())

# === Load spaCy model ===
nlp = spacy.load("en_core_web_sm")

# === Function to extract tickers using spaCy ===
def extract_tickers_spacy(text):
    doc = nlp(text)
    tickers = [ent.text.upper() for ent in doc.ents if ent.label_ == "ORG"]
    # Filter out invalid tickers and stopwords
    return [ticker for ticker in tickers if ticker in valid_tickers]

# === Apply spaCy-based extraction ===
sentiment_df['Tickers'] = sentiment_df['Text'].apply(extract_tickers_spacy)

# === Filter Rows with Valid Tickers ===
sentiment_df = sentiment_df[sentiment_df['Tickers'].map(len) > 0].reset_index(drop=True)

# === Explode Tickers ===
exploded_df = sentiment_df.explode('Tickers').rename(columns={'Tickers': 'Ticker'}).reset_index(drop=True)

# === Aggregate Sentiment Data ===
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
