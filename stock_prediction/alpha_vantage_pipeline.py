from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from preprocessing.config import DEFAULT_SEED, DEFAULT_SUBSET
from preprocessing.load_data import load_phrasebank
from preprocessing.sentiment_score import sentiment_score_from_proba
from pipelines.finbert_pipeline.classifiers import train_logistic_regression
from pipelines.finbert_pipeline.embeddings import (
    MODEL_NAME,
    compute_embeddings,
    load_or_create_embeddings,
)
from stock_prediction.evaluator import evaluate_sentiment_signal, save_metrics


try:
    from transformers import AutoModel, AutoTokenizer
except Exception as exc:
    raise ImportError(
        "transformers is required for FinBERT embeddings. "
        "Install it with: pip install transformers"
    ) from exc


DEFAULT_TICKERS = ["AAPL", "AMZN", "MSFT"]
DEFAULT_OUTPUT_DIR = Path("data/alpha_vantage")
DEFAULT_CACHE_DIR = Path("data/finbert_embeddings")


def load_news(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing news file: {path}. Run alpha_vantage_fetch.py first to create it."
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def parse_time_published(value: str) -> datetime:
    return datetime.strptime(value, "%Y%m%dT%H%M%S")


def news_to_rows(news: dict, tickers: Iterable[str]) -> pd.DataFrame:
    tickers_set = {t.upper() for t in tickers}
    rows = []
    for item in news.get("feed", []):
        time_published = parse_time_published(item["time_published"])
        title = item.get("title", "")
        summary = item.get("summary", "")
        url = item.get("url", "")
        full_text = f"{title}. {summary}".strip()

        for ts in item.get("ticker_sentiment", []):
            ticker = ts.get("ticker", "").upper()
            if ticker in tickers_set:
                rows.append(
                    {
                        "ticker": ticker,
                        "published_time": time_published,
                        "title": title,
                        "summary": summary,
                        "text": full_text,
                        "url": url,
                    }
                )
    return pd.DataFrame(rows)


def load_prices(output_dir: Path, tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    prices = {}
    for ticker in tickers:
        csv_path = output_dir / f"prices_{ticker}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
        else:
            json_path = output_dir / f"prices_{ticker}.json"
            if not json_path.exists():
                raise FileNotFoundError(
                    f"Missing price CSV and JSON for {ticker}: {csv_path} / {json_path}"
                )
            with json_path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            from stock_prediction.alpha_vantage_fetch import daily_prices_to_df

            df = daily_prices_to_df(data)
        prices[ticker] = df.sort_values("date").reset_index(drop=True)
    return prices


def compute_next_day_return(prices: pd.DataFrame, date: pd.Timestamp) -> float:
    prior = prices.loc[prices["date"] <= date]
    future = prices.loc[prices["date"] > date]
    if prior.empty or future.empty:
        return float("nan")
    prior_close = prior.iloc[-1]["adjusted_close"]
    future_close = future.iloc[0]["adjusted_close"]
    if pd.isna(prior_close) or pd.isna(future_close):
        return float("nan")
    return float((future_close - prior_close) / prior_close)


def train_phrasebank_model(
    subset: str,
    seed: int,
    batch_size: int,
    max_length: int,
    cache_dir: Path,
):
    texts, labels, label_names = load_phrasebank(subset)
    if isinstance(labels[0], str):
        from sklearn.preprocessing import LabelEncoder

        encoder = LabelEncoder()
        labels = encoder.fit_transform(labels).tolist()
        if label_names is None:
            label_names = list(encoder.classes_)

    if label_names is None:
        label_names = [str(i) for i in sorted(set(labels))]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    model.eval()

    cache_path = cache_dir / subset / "phrasebank_embeddings.npy"
    embeddings = load_or_create_embeddings(
        texts,
        cache_path=cache_path,
        tokenizer=tokenizer,
        model=model,
        batch_size=batch_size,
        max_length=max_length,
        expected_len=len(texts),
    )

    lr_model, _, _ = train_logistic_regression(embeddings, labels, seed)
    return lr_model, label_names, tokenizer, model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on Phrasebank and test on Alpha Vantage news."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--news-path", default=None)
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    news_path = Path(args.news_path) if args.news_path else output_dir / "news.json"
    cache_dir = Path(args.cache_dir)

    news = load_news(news_path)
    articles = news_to_rows(news, args.tickers)
    if articles.empty:
        raise ValueError("No articles found for the provided tickers.")

    prices = load_prices(output_dir, args.tickers)

    lr_model, label_names, tokenizer, model = train_phrasebank_model(
        subset=args.subset,
        seed=args.seed,
        batch_size=args.batch_size,
        max_length=args.max_length,
        cache_dir=cache_dir,
    )

    news_embeddings = compute_embeddings(
        articles["text"].tolist(),
        tokenizer=tokenizer,
        model=model,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    proba = lr_model.predict_proba(news_embeddings)
    articles["sentiment_score"] = sentiment_score_from_proba(proba, label_names)

    returns = []
    for _, row in articles.iterrows():
        ticker = row["ticker"]
        published_date = pd.to_datetime(row["published_time"]).normalize()
        returns.append(compute_next_day_return(prices[ticker], published_date))
    articles["next_day_return"] = returns

    out_path = output_dir / "alpha_vantage_sentiment_returns.csv"
    articles.to_csv(out_path, index=False)
    print(f"Saved dataset to {out_path}")

    metrics = evaluate_sentiment_signal(articles, "sentiment_score", "next_day_return")
    save_metrics(metrics, output_dir / "alpha_vantage_eval.json")
    save_metrics(metrics, output_dir / "alpha_vantage_eval.csv")

    # Aggregate by day to avoid constant returns when many articles share the same date.
    daily = (
        articles.assign(published_date=pd.to_datetime(articles["published_time"]).dt.normalize())
        .groupby(["ticker", "published_date"], as_index=False)
        .agg(
            sentiment_score=("sentiment_score", "mean"),
            article_count=("sentiment_score", "size"),
        )
    )
    daily_returns = []
    for _, row in daily.iterrows():
        ticker = row["ticker"]
        daily_returns.append(compute_next_day_return(prices[ticker], row["published_date"]))
    daily["next_day_return"] = daily_returns

    daily_out = output_dir / "alpha_vantage_daily_sentiment_returns.csv"
    daily.to_csv(daily_out, index=False)
    print(f"Saved daily dataset to {daily_out}")

    daily_metrics = evaluate_sentiment_signal(
        daily, "sentiment_score", "next_day_return"
    )
    save_metrics(daily_metrics, output_dir / "alpha_vantage_daily_eval.json")
    save_metrics(daily_metrics, output_dir / "alpha_vantage_daily_eval.csv")


if __name__ == "__main__":
    main()
