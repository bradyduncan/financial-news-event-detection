from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


BASE_URL = "https://api.marketstack.com/v1/eod"
DEFAULT_TICKERS = ["AAPL", "AMZN", "MSFT", "ROKU", "ETSY", "UPST"]
DEFAULT_OUTPUT_DIR = Path("data/stock_prediction/marketstack")
MAX_REQUESTS = 88


def _require_api_key() -> str:
    api_key = os.getenv("MARKETSTACK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Missing MARKETSTACK_API_KEY environment variable. "
            "Set it before running this script."
        )
    return api_key


def _fetch_json(params: dict, timeout_seconds: float) -> dict:
    url = f"{BASE_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "marketstack-fetch/1.0"})
    with urlopen(req, timeout=timeout_seconds) as resp:  # nosec - URL is constructed from trusted base + params
        return json.loads(resp.read().decode("utf-8"))


def fetch_eod_prices(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    api_key: str,
    limit: int,
    timeout_seconds: float,
) -> list[dict]:
    data = []
    offset = 0
    symbols = ",".join(tickers)
    while True:
        params = {
            "access_key": api_key,
            "symbols": symbols,
            "date_from": start_date,
            "date_to": end_date,
            "limit": limit,
            "offset": offset,
        }
        payload = _fetch_json(params, timeout_seconds=timeout_seconds)
        batch = payload.get("data", [])
        data.extend(batch)
        if len(batch) < limit:
            break
        offset += limit
    return data


def estimate_requests(
    start_date: str, end_date: str, num_tickers: int, limit: int
) -> int:
    start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
    end_dt = datetime.strptime(end_date, "%Y-%m-%d").date()
    if end_dt < start_dt:
        raise ValueError("--end-date must be >= --start-date")

    # Conservative estimate: count Mon-Fri business days.
    day = start_dt
    business_days = 0
    while day <= end_dt:
        if day.weekday() < 5:
            business_days += 1
        day = day.fromordinal(day.toordinal() + 1)

    total_rows = business_days * max(1, num_tickers)
    pages = (total_rows + limit - 1) // limit
    return int(pages * max(1, num_tickers))


def save_prices(prices: list[dict], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    by_symbol: dict[str, list[dict]] = {}
    for row in prices:
        symbol = row.get("symbol", "").upper()
        if not symbol:
            continue
        by_symbol.setdefault(symbol, []).append(row)

    for symbol, rows in by_symbol.items():
        df = pd.DataFrame(rows)
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date.astype(str)
        df = df.sort_values("date").reset_index(drop=True)

        out_csv = output_dir / f"prices_{symbol}.csv"

        # Normalize to the pipeline's expected columns.
        normalized = pd.DataFrame(
            {
                "date": df.get("date"),
                "open": df.get("open"),
                "high": df.get("high"),
                "low": df.get("low"),
                "close": df.get("close"),
                "adjusted_close": df.get("close"),
                "volume": df.get("volume"),
            }
        )
        normalized.to_csv(out_csv, index=False)



def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch Marketstack EOD prices for tickers."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    args = parser.parse_args()

    # Basic validation
    datetime.strptime(args.start_date, "%Y-%m-%d")
    datetime.strptime(args.end_date, "%Y-%m-%d")

    api_key = _require_api_key()
    output_dir = Path(args.output_dir)

    estimated = estimate_requests(
        start_date=args.start_date,
        end_date=args.end_date,
        num_tickers=len(args.tickers),
        limit=args.limit,
    )
    if estimated > MAX_REQUESTS:
        raise RuntimeError(
            f"Estimated {estimated} requests exceeds max {MAX_REQUESTS}. "
            "Reduce tickers, shorten date range, or increase --limit."
        )

    prices = fetch_eod_prices(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        api_key=api_key,
        limit=args.limit,
        timeout_seconds=args.timeout_seconds,
    )
    save_prices(prices, output_dir)
    print(f"Saved Marketstack prices to {output_dir}")


if __name__ == "__main__":
    main()
