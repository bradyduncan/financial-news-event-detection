from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError


BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA",
    "JPM", "JNJ", "TSLA", "AMD", "QCOM", "SHOP",
    "ROKU", "ETSY", "UPST", "FSLY", "RBLX",
]
DEFAULT_OUTPUT_DIR = Path("data/gdelt")
PROGRESS_FILENAME = "gdelt_progress.json"

# Simple name variants to catch common headline phrasing.
TICKER_SEARCH_TERMS = {
    "AAPL": ["AAPL", "Apple"],
    "MSFT": ["MSFT", "Microsoft"],
    "AMZN": ["AMZN", "Amazon"],
    "GOOGL": ["GOOGL", "Google", "Alphabet"],
    "META": ["META", "Meta Platforms", "Facebook"],
    "NVDA": ["NVDA", "Nvidia"],
    "JPM": ["JPM", "JPMorgan", "JP Morgan"],
    "JNJ": ["JNJ", "Johnson & Johnson"],
    "TSLA": ["TSLA", "Tesla"],
    "AMD": ["AMD", "Advanced Micro Devices"],
    "QCOM": ["QCOM", "Qualcomm"],
    "SHOP": ["SHOP", "Shopify"],
    "ROKU": ["ROKU", "Roku"],
    "ETSY": ["ETSY", "Etsy"],
    "UPST": ["UPST", "Upstart"],
    "FSLY": ["FSLY", "Fastly"],
    "RBLX": ["RBLX", "Roblox"],
}


def fetch_json(params: dict, timeout_seconds: float) -> dict:
    url = f"{BASE_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": "gdelt-fetch/1.0"})
    with urlopen(req, timeout=timeout_seconds) as resp:  # nosec
        raw = resp.read().decode("utf-8")
        if not raw.strip():
            raise ValueError("Empty response body")
        return json.loads(raw)


def default_terms_for_ticker(ticker: str) -> list[str]:
    return TICKER_SEARCH_TERMS.get(ticker.upper(), [ticker.upper()])


def build_query(terms: Iterable[str]) -> str:
    parts = []
    for term in terms:
        term = term.strip()
        if not term:
            continue
        if " " in term:
            parts.append(f"\"{term}\"")
        else:
            parts.append(term)
    if not parts:
        raise ValueError("No valid query terms provided.")
    return "(" + " OR ".join(parts) + ")"


def parse_date(date_str: str) -> datetime:
    return datetime.strptime(date_str, "%Y-%m-%d")


def fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y%m%d%H%M%S")


def fetch_gdelt_articles(
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    maxrecords: int,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
) -> list[dict]:
    params = {
        "query": query,
        "mode": "artlist",
        "format": "json",
        "maxrecords": maxrecords,
        "startdatetime": fmt_dt(start_dt),
        "enddatetime": fmt_dt(end_dt),
    }
    attempt = 0
    while True:
        try:
            data = fetch_json(params, timeout_seconds=timeout_seconds)
            return data.get("articles", [])
        except (URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            attempt += 1
            if attempt > retries:
                raise RuntimeError(
                    f"GDELT request failed after {retries} retries: {exc}"
                ) from exc
            time.sleep(backoff_seconds * attempt)


def matches_terms(text: str, terms: Iterable[str]) -> bool:
    lower = text.lower()
    for term in terms:
        if term.lower() in lower:
            return True
    return False


def fetch_windowed_news(
    tickers: Iterable[str],
    start_date: str,
    end_date: str,
    window_days: int,
    maxrecords: int,
    sleep_seconds: float,
    timeout_seconds: float,
    retries: int,
    backoff_seconds: float,
    output_dir: Path,
) -> dict:
    start_dt = parse_date(start_date)
    end_dt = parse_date(end_date)
    if window_days <= 0:
        raise ValueError("--window-days must be >= 1")

    output_dir.mkdir(parents=True, exist_ok=True)
    news_path = output_dir / "news.json"
    # Resume from the last saved window if a prior run was interrupted.
    progress = load_progress(output_dir)

    if news_path.exists():
        with news_path.open("r", encoding="utf-8") as f:
            combined = json.load(f)
        if "feed" not in combined:
            combined = {"items": "0", "feed": []}
    else:
        combined = {"items": "0", "feed": []}

    seen = set()
    for item in combined.get("feed", []):
        key = (item.get("url", ""), item.get("seendate", ""))
        seen.add(key)

    resume_date = None
    resume_index = 0
    if progress:
        resume_date = progress.get("current_date")
        resume_index = int(progress.get("next_ticker_index", 0))

    cur = parse_date(resume_date) if resume_date else start_dt
    if cur < start_dt:
        cur = start_dt
        resume_date = None
        resume_index = 0
    if cur > end_dt:
        combined["items"] = str(len(combined["feed"]))
        save_json(combined, news_path)
        clear_progress(output_dir)
        return combined

    while cur <= end_dt:
        window_end = min(cur + timedelta(days=window_days - 1), end_dt)
        tickers_list = list(tickers)
        skip_until = resume_index if resume_date and cur.strftime("%Y-%m-%d") == resume_date else 0
        for idx, ticker in enumerate(tickers_list):
            if idx < skip_until:
                continue
            terms = default_terms_for_ticker(ticker)
            query = build_query(terms)
            try:
                articles = fetch_gdelt_articles(
                    query,
                    cur,
                    window_end,
                    maxrecords,
                    timeout_seconds,
                    retries,
                    backoff_seconds,
                )
            except RuntimeError as exc:
                # Keep partial results so a later run can pick up where it stopped.
                combined["items"] = str(len(combined["feed"]))
                save_json(combined, news_path)
                save_progress(output_dir, cur.strftime("%Y-%m-%d"), idx)
                return combined

            for item in articles:
                title = item.get("title", "")
                url = item.get("url", "")
                if not title and not url:
                    continue
                if not matches_terms(f"{title} {url}", terms):
                    continue

                key = (url, item.get("seendate", ""))
                if key in seen:
                    continue
                seen.add(key)
                combined["feed"].append(
                    {
                        "ticker": ticker.upper(),
                        "title": title,
                        "url": url,
                        "seendate": item.get("seendate", ""),
                        "domain": item.get("domain", ""),
                        "language": item.get("language", ""),
                        "sourcecountry": item.get("sourcecountry", ""),
                        "socialimage": item.get("socialimage", ""),
                    }
                )

            time.sleep(sleep_seconds)
            combined["items"] = str(len(combined["feed"]))
            save_json(combined, news_path)
            save_progress(output_dir, cur.strftime("%Y-%m-%d"), idx + 1)
        save_progress(output_dir, (window_end + timedelta(days=1)).strftime("%Y-%m-%d"), 0)
        cur = window_end + timedelta(days=1)

    combined["items"] = str(len(combined["feed"]))
    save_json(combined, news_path)
    clear_progress(output_dir)
    return combined


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_progress(output_dir: Path) -> dict | None:
    progress_path = output_dir / PROGRESS_FILENAME
    if not progress_path.exists():
        return None
    with progress_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(output_dir: Path, current_date: str, next_ticker_index: int) -> None:
    progress_path = output_dir / PROGRESS_FILENAME
    payload = {
        "current_date": current_date,
        "next_ticker_index": next_ticker_index,
    }
    save_json(payload, progress_path)


def clear_progress(output_dir: Path) -> None:
    progress_path = output_dir / PROGRESS_FILENAME
    if progress_path.exists():
        progress_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch GDELT DOC 2.0 news for tickers using windowed queries."
    )
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--sleep-seconds", type=float, default=1.0)
    parser.add_argument("--maxrecords", type=int, default=100)
    parser.add_argument("--window-days", type=int, default=7)
    parser.add_argument("--timeout-seconds", type=float, default=20.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--backoff-seconds", type=float, default=5.0)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    news = fetch_windowed_news(
        tickers=args.tickers,
        start_date=args.start_date,
        end_date=args.end_date,
        window_days=args.window_days,
        maxrecords=args.maxrecords,
        sleep_seconds=args.sleep_seconds,
        timeout_seconds=args.timeout_seconds,
        retries=args.retries,
        backoff_seconds=args.backoff_seconds,
        output_dir=output_dir,
    )
    out_path = output_dir / "news.json"
    save_json(news, out_path)
    print(f"Saved GDELT news to {out_path}")


if __name__ == "__main__":
    main()
