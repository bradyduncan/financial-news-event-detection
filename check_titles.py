# debug_daily.py
import pandas as pd

df = pd.read_csv("data/stock_prediction/results/sentiment_price_daily_returns.csv")
print(f"Total daily rows: {len(df)}")
print(f"Tickers: {sorted(df['ticker'].unique())}")
print()
print("Rows per ticker:")
print(df["ticker"].value_counts().sort_values(ascending=False))