from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import urlopen
import time

import pandas as pd


BASE_URL = "https://www.alphavantage.co/query"
DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN"]
DEFAULT_OUTPUT_DIR = Path("data/alpha_vantage")


def _require_api_key() -> str:
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing ALPHAVANTAGE_API_KEY environment variable. "
            "Set it before running this script."
        )
    return api_key


def _fetch_json(params: dict) -> dict:
    url = f"{BASE_URL}?{urlencode(params)}"
    with urlopen(url) as resp:  # nosec - URL is constructed from trusted base + params
        return json.loads(resp.read().decode("utf-8"))


def _raise_if_error(data: dict, context: str) -> None:
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error for {context}: {data['Error Message']}")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit note for {context}: {data['Note']}")
    if "Information" in data:
        raise RuntimeError(
            f"Alpha Vantage information for {context}: {data['Information']}"
        )


def _has_rate_limit_or_error(data: dict) -> bool:
    return any(key in data for key in ("Error Message", "Note", "Information"))


def fetch_news(
    tickers: Iterable[str],
    time_from: str,
    time_to: str,
    limit: int,
    api_key: str,
) -> dict:
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ",".join(tickers),
        "time_from": time_from,
        "time_to": time_to,
        "limit": limit,
        "apikey": api_key,
    }
    data = _fetch_json(params)
    _raise_if_error(data, "NEWS_SENTIMENT")
    if "feed" not in data:
        raise RuntimeError("NEWS_SENTIMENT response missing 'feed' field.")
    return data


def fetch_news_windowed(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    window_days: int,
    limit: int,
    api_key: str,
    sleep_seconds: float,
) -> dict:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d")
    if window_days <= 0:
        raise ValueError("--news-window-days must be >= 1")

    combined = {
        "items": "0",
        "sentiment_score_definition": "",
        "relevance_score_definition": "",
        "feed": [],
    }
    seen = set()

    cur = start_dt
    while cur <= end_dt:
        window_end = min(cur + timedelta(days=window_days - 1), end_dt)
        time_from = cur.strftime("%Y%m%dT0000")
        time_to = window_end.strftime("%Y%m%dT2359")
        time.sleep(sleep_seconds)
        data = fetch_news(tickers, time_from, time_to, limit, api_key)

        if not combined["sentiment_score_definition"]:
            combined["sentiment_score_definition"] = data.get(
                "sentiment_score_definition", ""
            )
        if not combined["relevance_score_definition"]:
            combined["relevance_score_definition"] = data.get(
                "relevance_score_definition", ""
            )

        for item in data.get("feed", []):
            key = (item.get("url", ""), item.get("time_published", ""))
            if key in seen:
                continue
            seen.add(key)
            combined["feed"].append(item)

        cur = window_end + timedelta(days=1)

    combined["items"] = str(len(combined["feed"]))
    return combined


def fetch_daily_prices(ticker: str, api_key: str, outputsize: str = "compact") -> dict:
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": ticker,
        "outputsize": outputsize,
        "apikey": api_key,
    }
    data = _fetch_json(params)
    _raise_if_error(data, f"TIME_SERIES_DAILY {ticker}")
    if "Time Series (Daily)" not in data:
        raise RuntimeError(
            f"TIME_SERIES_DAILY response missing 'Time Series (Daily)' for {ticker}"
        )
    return data


def save_json(data: dict, path: Path) -> None:
    if path.exists() and _has_rate_limit_or_error(data):
        raise RuntimeError(
            f"Refusing to overwrite {path} with an Alpha Vantage error/limit response."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def daily_prices_to_df(data: dict) -> pd.DataFrame:
    series = data.get("Time Series (Daily)")
    if not series:
        raise ValueError("Missing Time Series (Daily) in price response.")

    rows = []
    for date_str, values in series.items():
        rows.append(
            {
                "date": pd.to_datetime(date_str),
                "open": float(values.get("1. open", "nan")),
                "high": float(values.get("2. high", "nan")),
                "low": float(values.get("3. low", "nan")),
                "close": float(values.get("4. close", "nan")),
                "adjusted_close": float(values.get("4. close", "nan")),
                "volume": float(values.get("5. volume", "nan")),
            }
        )
    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df


def save_prices_csv(
    data: dict,
    path: Path,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> None:
    df = daily_prices_to_df(data)
    if start_date is not None:
        df = df[df["date"] >= start_date]
    if end_date is not None:
        df = df[df["date"] <= end_date]
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def parse_date_arg(date_str: str) -> str:
    dt = datetime.strptime(date_str, "%Y-%m-%d")
    return dt.strftime("%Y%m%dT0000")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch Alpha Vantage daily prices.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start-date", default=None)
    parser.add_argument("--end-date", default=None)
    parser.add_argument(
        "--fetch-news",
        action="store_true",
        help="Fetch Alpha Vantage NEWS_SENTIMENT and save news.json.",
    )
    parser.add_argument(
        "--news-limit",
        type=int,
        default=200,
        help="Max number of news items to request (used with --fetch-news).",
    )
    parser.add_argument(
        "--news-window-days",
        type=int,
        default=7,
        help="Window size in days for paginated news fetch (used with --fetch-news).",
    )
    parser.add_argument(
        "--outputsize",
        choices=["compact", "full"],
        default="compact",
        help="Alpha Vantage output size. 'full' may require a premium plan.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sleep-seconds", type=float, default=15.0)
    args = parser.parse_args()

    api_key = _require_api_key()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.start_date is None or args.end_date is None:
        raise ValueError("Both --start-date and --end-date are required (YYYY-MM-DD).")

    if args.fetch_news:
        news = fetch_news_windowed(
            tickers=args.tickers,
            start_date=args.start_date,
            end_date=args.end_date,
            window_days=args.news_window_days,
            limit=args.news_limit,
            api_key=api_key,
            sleep_seconds=args.sleep_seconds,
        )
        save_json(news, output_dir / "news.json")

    start_dt = datetime.strptime(args.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end_date, "%Y-%m-%d")

    for ticker in args.tickers:
        time.sleep(args.sleep_seconds)
        prices = fetch_daily_prices(ticker, api_key, outputsize=args.outputsize)
        save_json(prices, output_dir / f"prices_{ticker}.json")
        save_prices_csv(prices, output_dir / f"prices_{ticker}.csv", start_dt, end_dt)

    if args.fetch_news:
        print(f"Saved news to {output_dir / 'news.json'}")
    print(f"Saved prices to {output_dir}")


if __name__ == "__main__":
    main()
