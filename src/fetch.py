"""Fetch gold news and prices from Alpha Vantage + RSS."""
import os
import requests
import pandas as pd
import feedparser
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("ALPHAVANTAGE_API_KEY")
BASE_URL = "https://www.alphavantage.co/query"
DATA_DIR = "data"
RSS_URL = "https://news.google.com/rss/search?q=gold+price&hl=en-GB&gl=GB&ceid=GB:en"


def fetch_gold_news(limit=200):
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": "GLD",
        "sort": "LATEST",
        "limit": limit,
        "apikey": API_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()
    if "feed" not in raw:
        raise RuntimeError(f"Unexpected news response (rate limit?): {raw}")

    rows = []
    for a in raw["feed"]:
        rows.append({
            "title": a.get("title"),
            "summary": a.get("summary"),
            "url": a.get("url"),
            "published_at": a.get("time_published"),
            "source": a.get("source"),
            "av_sentiment_score": float(a.get("overall_sentiment_score", 0)),
            "av_sentiment_label": a.get("overall_sentiment_label"),
        })
    df = pd.DataFrame(rows)
    df["published_at"] = (pd.to_datetime(df["published_at"],
                                         format="%Y%m%dT%H%M%S")
                          .dt.tz_localize("US/Eastern")
                          .dt.tz_convert("UTC"))
    return df


def fetch_rss(url=RSS_URL):
    feed = feedparser.parse(url, agent="Mozilla/5.0")
    if not feed.entries:
        print(f"WARNING: RSS returned 0 entries (status={feed.get('status')})")
    rows = []
    for e in feed.entries:
        rows.append({
            "title": e.get("title"),
            "summary": e.get("summary", ""),
            "url": e.get("link"),
            "published_at": e.get("published", None),
            "source": "rss",
            "av_sentiment_score": None,
            "av_sentiment_label": None,
        })
    cols = ["title", "summary", "url", "published_at", "source",
            "av_sentiment_score", "av_sentiment_label"]
    df = pd.DataFrame(rows, columns=cols)
    if len(df):
        df["published_at"] = pd.to_datetime(df["published_at"],
                                            errors="coerce", utc=True)
    return df


def fetch_gold_price():
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": "GLD",
        "outputsize": "compact",
        "apikey": API_KEY,
    }
    r = requests.get(BASE_URL, params=params, timeout=30)
    r.raise_for_status()
    raw = r.json()
    key = "Time Series (Daily)"
    if key not in raw:
        raise RuntimeError(f"Unexpected price response (rate limit?): {raw}")

    df = pd.DataFrame.from_dict(raw[key], orient="index")
    df.columns = ["open", "high", "low", "close", "volume"]
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    return df.sort_index()


def update_news_store(path=f"{DATA_DIR}/news_raw.csv"):
    """Fetch API + RSS, merge with saved history, dedupe by URL.
    Returns (full_df, n_new)."""
    os.makedirs(DATA_DIR, exist_ok=True)
    fresh = pd.concat([fetch_gold_news(), fetch_rss()], ignore_index=True)

    if os.path.exists(path):
        old = pd.read_csv(path, parse_dates=["published_at"])
        known_urls = set(old["url"])
        combined = pd.concat([old, fresh], ignore_index=True)
    else:
        known_urls = set()
        combined = fresh

    combined = combined.drop_duplicates(subset="url", keep="first")
    combined = combined.sort_values("published_at",
                                    ascending=False).reset_index(drop=True)
    combined.to_csv(path, index=False)
    n_new = len(set(fresh["url"]) - known_urls)
    return combined, n_new