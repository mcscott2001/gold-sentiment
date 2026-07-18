"""Sentiment scoring (VADER + FinBERT) and aggregation into a verdict."""
import re
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()
_finbert = None  # loaded lazily - it's slow to import, so only on first use

GOOD_THRESHOLD = 0.15
BAD_THRESHOLD = -0.15


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+", "", text)        # strip URLs
    text = re.sub(r"\$[A-Z]{1,5}", "", text)   # strip $TICKER tags
    return text.strip()


def vader_score(text):
    """Compound score in [-1, 1]."""
    return _vader.polarity_scores(clean_text(text))["compound"]


def _load_finbert():
    global _finbert
    if _finbert is None:
        from transformers import pipeline
        _finbert = pipeline("text-classification",
                            model="ProsusAI/finbert",
                            top_k=None, truncation=True)
    return _finbert


def finbert_score(text):
    """Signed score in [-1, 1]: P(positive) - P(negative)."""
    text = clean_text(text)
    if not text:
        return 0.0
    results = _load_finbert()(text[:512])[0]
    probs = {r["label"]: r["score"] for r in results}
    return probs.get("positive", 0) - probs.get("negative", 0)


def score_articles(df):
    """Add vader_score and finbert_score columns for rows missing them."""
    df = df.copy()
    for col in ("vader_score", "finbert_score"):
        if col not in df.columns:
            df[col] = None
        df[col] = pd.to_numeric(df[col], errors="coerce")

    todo = df["finbert_score"].isna()
    if todo.any():
        texts = df.loc[todo, "title"].fillna("")
        df.loc[todo, "vader_score"] = texts.apply(vader_score).astype(float)
        df.loc[todo, "finbert_score"] = texts.apply(finbert_score).astype(float)
    return df


def aggregate_sentiment(df, hours=24):
    """Recency-weighted mean finbert score over the last `hours`.
    Returns (score, n_articles). Newer articles count more (weight
    decays linearly from 1.0 now to 0.5 at the window edge)."""
    now = pd.Timestamp.now(tz="UTC")
    df = df.copy()
    df["published_at"] = pd.to_datetime(df["published_at"],
                                        utc=True, errors="coerce")
    window = df[df["published_at"] >= now - pd.Timedelta(hours=hours)].copy()
    window = window.dropna(subset=["finbert_score"])
    if len(window) == 0:
        return 0.0, 0

    age_hrs = (now - window["published_at"]).dt.total_seconds() / 3600
    weights = 1.0 - 0.5 * (age_hrs / hours)
    score = (window["finbert_score"] * weights).sum() / weights.sum()
    return round(float(score), 4), len(window)


def verdict(score):
    if score >= GOOD_THRESHOLD:
        return "GOOD"
    if score <= BAD_THRESHOLD:
        return "BAD"
    return "NEUTRAL"