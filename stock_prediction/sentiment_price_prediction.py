from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import time

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from preprocessing.config import DEFAULT_SEED, DEFAULT_SUBSET
from preprocessing.load_data import load_phrasebank
from preprocessing.sentiment_score import sentiment_score_from_proba
from pipelines.finbert_pipeline.classifiers import train_xgboost
from pipelines.finbert_pipeline.embeddings import (
    MODEL_NAME,
    compute_embeddings,
    load_or_create_embeddings,
)
from stock_prediction.evaluator import evaluate_sentiment_signal, save_metrics
from stock_prediction.sentiment_price_regressor import run_regressor
from transformers import AutoModel, AutoTokenizer
import requests
import trafilatura

# Global vars
DEFAULT_TICKERS_FILE = Path("data/stock_prediction/marketstack/tickers.txt")
DEFAULT_OUTPUT_DIR = Path("data/stock_prediction/results")
DEFAULT_CACHE_DIR = Path("data/finbert_embeddings")
DEFAULT_TEXT_CACHE = Path("data/stock_prediction/gdelt/article_text_cache.json")

# Possible references for each ticker
TICKER_ALIASES = {
    "AAPL": ["apple"],
    "MSFT": ["microsoft"],
    "AMZN": ["amazon"],
    "GOOGL": ["google", "alphabet"],
    "META": ["meta", "facebook"],
    "NVDA": ["nvidia"],
    "JPM": ["jpmorgan", "jp morgan", "chase"],
    "JNJ": ["johnson & johnson", "johnson and johnson"],
    "TSLA": ["tesla"],
    "AMD": ["amd", "advanced micro devices"],
    "QCOM": ["qualcomm"],
    "SHOP": ["shopify"],
    "ROKU": ["roku"],
    "ETSY": ["etsy"],
    "UPST": ["upstart"],
    "FSLY": ["fastly"],
    "RBLX": ["roblox"],
    "SPY": ["s&p 500", "s&p500"],
}


def load_news(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing news file: {path}"
        )
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

# Tickers are the company IDs for market prices
def load_tickers_from_file(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Missing tickers file: {path}")
    tickers = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            value = line.strip().upper()
            if value:
                tickers.append(value)
    if not tickers:
        raise ValueError(f"No tickers found in {path}")
    return tickers


def load_text_cache(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_text_cache(path: Path, cache: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def fetch_article_text(url: str, timeout_seconds: float = 20.0) -> str:
    if requests is None or trafilatura is None:
        raise ImportError(
            "trafilatura and requests are required for article text extraction. "
            "Install with: pip install trafilatura requests"
        )
    downloaded = trafilatura.fetch_url(url)
    if not downloaded:
        # Fallback to requests for sites that block trafilatura's downloader.
        headers = {"User-Agent": "sentiment-price/1.0"}
        resp = requests.get(url, timeout=timeout_seconds, headers=headers)
        resp.raise_for_status()
        downloaded = resp.text
    text = trafilatura.extract(
        downloaded,
        url=url,
        include_comments=False,
        include_tables=False,
        favor_recall=True,
    )
    if not text:
        return ""
    return " ".join(text.split())


def enrich_articles_with_text(
    articles: pd.DataFrame,
    cache_path: Path,
    max_articles: int,
    sleep_seconds: float,
    min_text_chars: int,
) -> pd.DataFrame:
    cache = load_text_cache(cache_path)
    urls = articles["url"].dropna().unique().tolist()
    fetched = 0
    updated_cache = False

    for url in urls:
        if not url or url in cache:
            continue
        if fetched >= max_articles:
            break
        try:
            text = fetch_article_text(url)
        except Exception:
            text = ""
        if text and len(text) >= min_text_chars:
            cache[url] = text
            updated_cache = True
        fetched += 1
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if updated_cache:
        save_text_cache(cache_path, cache)

    def choose_text(row: pd.Series) -> str:
        cached = cache.get(row.get("url", ""), "")
        return cached if cached else row.get("text", "")

    out = articles.copy()
    out["text"] = out.apply(choose_text, axis=1)
    return out


def parse_gdelt_time(value: str) -> datetime:
    value = value.strip()
    for fmt in ("%Y%m%d%H%M%S", "%Y%m%dT%H%M%SZ"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise ValueError(f"Unsupported GDELT seendate format: {value}")


def passes_keyword_filter(text: str, keywords: list[str]) -> bool:
    if not keywords:
        return True
    lower = text.lower()
    return any(kw.lower() in lower for kw in keywords)


def news_to_rows(
    news: dict, tickers: Iterable[str], keywords: list[str] | None = None
) -> pd.DataFrame:
    """Parse GDELT feed, requiring ticker symbol appears in the article title."""
    tickers_set = {t.upper() for t in tickers}
    rows = []
    feed = news.get("feed", [])
    if not feed:
        return pd.DataFrame(rows)
    keywords = keywords or []

    for item in feed:
        ticker = item.get("ticker", "").upper()
        if ticker not in tickers_set:
            continue
        seendate = item.get("seendate", "")
        if not seendate:
            continue
        title = item.get("title", "")
        # Require ticker or company name in the title as a relevance gate
        title_lower = title.lower()
        aliases = TICKER_ALIASES.get(ticker, [])
        if ticker.lower() not in title_lower and not any(a in title_lower for a in aliases):
            continue
        time_published = parse_gdelt_time(seendate)
        url = item.get("url", "")
        full_text = title
        if not passes_keyword_filter(full_text, keywords):
            continue
        rows.append(
            {
                "ticker": ticker,
                "published_time": time_published,
                "title": title,
                "summary": "",
                "text": full_text,
                "url": url,
            }
        )
    return pd.DataFrame(rows)


def load_prices(prices_dir: Path, tickers: Iterable[str]) -> dict[str, pd.DataFrame]:
    prices = {}
    for ticker in tickers:
        csv_path = prices_dir / f"prices_{ticker}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df["date"] = pd.to_datetime(df["date"])
        else:
            json_path = prices_dir / f"prices_{ticker}.json"
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


def load_benchmark_if_available(prices_dir: Path, ticker: str) -> pd.DataFrame | None:
    ticker = ticker.upper()
    csv_path = prices_dir / f"prices_{ticker}.csv"
    if not csv_path.exists():
        return None
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def compute_forward_return(
    prices: pd.DataFrame, date: pd.Timestamp, horizon_days: int
) -> float:
    # Entry is the first trading day strictly after publication
    future = prices.loc[prices["date"] > date]
    if future.empty or horizon_days <= 0 or len(future) <= horizon_days:
        return float("nan")
    entry_close = future.iloc[0]["adjusted_close"]
    exit_close = future.iloc[horizon_days]["adjusted_close"]
    if pd.isna(entry_close) or pd.isna(exit_close):
        return float("nan")
    return float((exit_close - entry_close) / entry_close)


def build_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = prices.sort_values("date").copy()
    df["daily_return"] = df["adjusted_close"].pct_change()
    df["lag_return_1"] = df["daily_return"].shift(1)
    df["lag_return_3"] = df["daily_return"].shift(3)
    df["rolling_return_mean_5"] = df["daily_return"].rolling(5).mean()
    df["rolling_return_mean_10"] = df["daily_return"].rolling(10).mean()
    df["rolling_vol_5"] = df["daily_return"].rolling(5).std()
    df["rolling_vol_10"] = df["daily_return"].rolling(10).std()
    return df[
        [
            "date",
            "daily_return",
            "lag_return_1",
            "lag_return_3",
            "rolling_return_mean_5",
            "rolling_return_mean_10",
            "rolling_vol_5",
            "rolling_vol_10",
        ]
    ]


def attach_price_features(
    daily: pd.DataFrame, prices_by_ticker: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    parts = []
    for ticker, group in daily.groupby("ticker"):
        price_features = build_price_features(prices_by_ticker[ticker])
        group_sorted = group.sort_values("published_date")
        merged = pd.merge_asof(
            group_sorted,
            price_features.sort_values("date"),
            left_on="published_date",
            right_on="date",
            direction="backward",
            tolerance=pd.Timedelta("3 days"),  # avoid matching stale prices
        ).drop(columns=["date"])
        parts.append(merged)
    return pd.concat(parts, ignore_index=True)


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

    xgb_model, _, _ = train_xgboost(embeddings, labels, seed)

    return xgb_model, label_names, tokenizer, model


def build_daily_features(articles: pd.DataFrame) -> pd.DataFrame:
    """Aggregate articles to daily per-ticker rows with richer features."""
    articles = articles.assign(
        published_date=pd.to_datetime(articles["published_time"]).dt.normalize()
    )
    daily = (
        articles.groupby(["ticker", "published_date"], as_index=False)
        .agg(
            sentiment_score=("sentiment_score", "median"),
            sentiment_std=("sentiment_score", "std"),
            max_abs_sentiment=("sentiment_score", lambda s: s.abs().max()),
            article_count=("sentiment_score", "size"),
        )
    )
    # Fill std with 0 for single-article days
    daily["sentiment_std"] = daily["sentiment_std"].fillna(0.0)
    daily = daily.sort_values(["ticker", "published_date"]).reset_index(drop=True)
    daily["day_of_week"] = daily["published_date"].dt.dayofweek

    # Rolling sentiment windows
    daily["rolling_sentiment_3"] = (
        daily.groupby("ticker")["sentiment_score"]
        .transform(lambda s: s.rolling(3).mean())
    )
    daily["rolling_sentiment_5"] = (
        daily.groupby("ticker")["sentiment_score"]
        .transform(lambda s: s.rolling(5).mean())
    )

    # Day-over-day change and acceleration
    daily["sentiment_change"] = (
        daily.groupby("ticker")["sentiment_score"]
        .transform(lambda s: s.diff())
    )
    daily["sentiment_acceleration"] = (
        daily.groupby("ticker")["sentiment_change"]
        .transform(lambda s: s.diff())
    )

    return daily


def attach_forward_returns(
    daily: pd.DataFrame,
    prices: dict[str, pd.DataFrame],
    benchmark_prices: pd.DataFrame | None,
    horizon_days: int,
) -> pd.DataFrame:
    """Vectorized forward-return computation via merge instead of iterrows."""
    all_rows = []
    for ticker, group in daily.groupby("ticker"):
        ticker_prices = prices[ticker].sort_values("date")
        group = group.sort_values("published_date").copy()

        # Find the first trading day strictly after each pub date
        returns = []
        bench = []
        for pub_date in group["published_date"]:
            returns.append(
                compute_forward_return(ticker_prices, pub_date, horizon_days)
            )
            if benchmark_prices is not None:
                bench.append(
                    compute_forward_return(benchmark_prices, pub_date, horizon_days)
                )
        group["next_day_return"] = returns
        if benchmark_prices is not None:
            group["benchmark_return"] = bench
            group["excess_return"] = group["next_day_return"] - group["benchmark_return"]
        all_rows.append(group)

    return pd.concat(all_rows, ignore_index=True)


def run_classifier(
    daily: pd.DataFrame,
    output_dir: Path,
    seed: int,
    target_col: str,
    use_weighted_sentiment: bool = False,
) -> None:
    """Binary classification: predict whether excess/forward return is positive."""
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, f1_score
    import xgboost as xgb

    daily = daily.dropna(subset=[target_col]).copy()
    # Binary target: 1 if return > 0, else 0
    daily["direction"] = (daily[target_col] > 0).astype(int)

    sentiment_col = "weighted_sentiment_score" if use_weighted_sentiment else "sentiment_score"
    feature_cols = [
        sentiment_col, "sentiment_std", "max_abs_sentiment", "article_count",
        "day_of_week", "rolling_sentiment_3", "rolling_sentiment_5",
        "sentiment_change", "sentiment_acceleration",
        "daily_return", "lag_return_1", "lag_return_3",
        "rolling_return_mean_5", "rolling_return_mean_10",
        "rolling_vol_5", "rolling_vol_10",
    ]
    feature_cols = [c for c in feature_cols if c in daily.columns]
    daily = daily.dropna(subset=feature_cols)

    X = daily[feature_cols].values
    y = daily["direction"].values

    if len(X) < 10:
        print("Too few samples for classification. Skipping.")
        return

    # Class weight as scale_pos_weight
    n_neg = (y == 0).sum()
    n_pos = (y == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    tscv = TimeSeriesSplit(n_splits=5) # time series split to avoid leakage
    accuracies, f1s = [], []
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            random_state=seed,
            use_label_encoder=False,
            eval_metric="logloss",
        )
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
        f1s.append(f1_score(y_test, preds))

    clf_metrics = {
        "mean_accuracy": float(np.mean(accuracies)),
        "mean_f1": float(np.mean(f1s)),
        "per_fold_accuracy": [float(a) for a in accuracies],
        "per_fold_f1": [float(f) for f in f1s],
    }
    print(
        f"Classification - Accuracy: {clf_metrics['mean_accuracy']:.4f}, "
        f"F1: {clf_metrics['mean_f1']:.4f}"
    )
    save_metrics(clf_metrics, output_dir / "classification_eval.json")
    save_metrics(clf_metrics, output_dir / "classification_eval.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train on Phrasebank and test on GDELT news with Marketstack prices."
    )
    parser.add_argument(
        "--tickers", nargs="+", default=None,
        help="Explicit tickers list (overrides --tickers-file).",
    )
    parser.add_argument(
        "--tickers-file", default=str(DEFAULT_TICKERS_FILE),
        help="Path to tickers.txt used when --tickers is not provided.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--subset", default=DEFAULT_SUBSET)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--news-path", default=None)
    parser.add_argument("--prices-dir", default="data/stock_prediction/marketstack")
    parser.add_argument(
        "--benchmark-ticker", default="SPY",
        help="Benchmark ticker for excess returns (requires prices_<TICKER>.csv).",
    )
    parser.add_argument(
        "--horizon-days", type=int, default=2,
        help="Trading-day horizon for forward returns (e.g., 2 = 2 trading days ahead).",
    )
    parser.add_argument(
        "--keywords", nargs="*", default=[],
        help="Optional keyword filter; only keep articles containing any keyword.",
    )
    parser.add_argument(
        "--weight-by-article-count", action="store_true",
        help="Scale daily sentiment by log(1 + article_count) before evaluation.",
    )
    parser.add_argument(
        "--min-articles", type=int, default=0,
        help="Minimum articles per ticker-day to keep.",
    )
    parser.add_argument(
        "--min-abs-sentiment", type=float, default=0.0,
        help="Minimum absolute sentiment score to keep.",
    )
    parser.add_argument(
        "--classify", action="store_true",
        help="Run binary direction classifier in addition to regression.",
    )
    parser.add_argument(
        "--no-article-text",
        action="store_true",
        help="Skip URL text extraction and use titles only.",
    )
    parser.add_argument(
        "--text-cache",
        default=str(DEFAULT_TEXT_CACHE),
        help="Path to JSON cache for extracted article text.",
    )
    parser.add_argument(
        "--max-articles",
        type=int,
        default=500,
        help="Max number of new URLs to scrape per run.",
    )
    parser.add_argument(
        "--scrape-sleep",
        type=float,
        default=0.5,
        help="Seconds to sleep between URL fetches.",
    )
    parser.add_argument(
        "--min-text-chars",
        type=int,
        default=200,
        help="Minimum extracted text length to cache.",
    )
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    news_path = Path(args.news_path) if args.news_path else Path("data/stock_prediction/gdelt/news.json")
    prices_dir = Path(args.prices_dir)
    cache_dir = Path(args.cache_dir)
    tickers = args.tickers or load_tickers_from_file(Path(args.tickers_file))

    news = load_news(news_path)
    articles = news_to_rows(news, tickers, keywords=args.keywords)
    if articles.empty:
        raise ValueError("No articles found for the provided tickers.")

    if not args.no_article_text:
        articles = enrich_articles_with_text(
            articles=articles,
            cache_path=Path(args.text_cache),
            max_articles=args.max_articles,
            sleep_seconds=args.scrape_sleep,
            min_text_chars=args.min_text_chars,
        )

    prices = load_prices(prices_dir, tickers)
    benchmark_prices = load_benchmark_if_available(prices_dir, args.benchmark_ticker)

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

    # Save article-level dataset 
    article_out = output_dir / "sentiment_articles.csv"
    articles.to_csv(article_out, index=False)
    print(f"Saved article-level sentiments to {article_out}")

    # Aggregate daily features
    daily = build_daily_features(articles)
    daily = attach_price_features(daily, prices)

    if args.weight_by_article_count:
        daily["weighted_sentiment_score"] = daily["sentiment_score"] * (
            daily["article_count"].astype(float).add(1.0).apply(np.log)
        )
    if args.min_articles > 0:
        daily = daily[daily["article_count"] >= args.min_articles]
    if args.min_abs_sentiment > 0:
        daily = daily[daily["sentiment_score"].abs() >= args.min_abs_sentiment]

    # Forward returns - daily level
    daily = attach_forward_returns(daily, prices, benchmark_prices, args.horizon_days)

    daily_out = output_dir / "sentiment_price_daily_returns.csv"
    daily.to_csv(daily_out, index=False)
    print(f"Saved daily dataset to {daily_out}")

    target_col = "excess_return" if benchmark_prices is not None else "next_day_return"
    daily_score_col = (
        "weighted_sentiment_score"
        if args.weight_by_article_count
        else "sentiment_score"
    )
    daily_metrics = evaluate_sentiment_signal(daily, daily_score_col, target_col)
    save_metrics(daily_metrics, output_dir / "sentiment_price_daily_eval.json")
    save_metrics(daily_metrics, output_dir / "sentiment_price_daily_eval.csv")

    # Regression for forward returns
    run_regressor(
        daily=daily,
        output_dir=output_dir,
        seed=args.seed,
        use_weighted_sentiment=args.weight_by_article_count,
        target_col=target_col,
    )

    # Classification
    if args.classify:
        run_classifier(
            daily=daily,
            output_dir=output_dir,
            seed=args.seed,
            target_col=target_col,
            use_weighted_sentiment=args.weight_by_article_count,
        )


if __name__ == "__main__":
    main()
