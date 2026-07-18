"""Orchestrates fetch -> analyze -> notify. State lives in data/state.json."""
import json
import os
from datetime import datetime

import pandas as pd

from src.fetch import update_news_store, fetch_gold_price
from src.analyze import score_articles, aggregate_sentiment, verdict
from src.notify import notify_all

DATA_DIR = "data"
SCORED_PATH = f"{DATA_DIR}/news_scored.csv"
STATE_PATH = f"{DATA_DIR}/state.json"
STRONG = 0.30  # |score| above this always notifies, even without a flip


def load_state():
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            return json.load(f)
    return {"last_verdict": None}


def save_state(state):
    with open(STATE_PATH, "w") as f:
        json.dump(state, f, indent=2)


def log(msg):
    line = f"{datetime.now():%Y-%m-%d %H:%M:%S}  {msg}"
    print(line)
    with open(f"{DATA_DIR}/run.log", "a") as f:
        f.write(line + "\n")


def run_pipeline():
    os.makedirs(DATA_DIR, exist_ok=True)
    log("--- pipeline start ---")

    # 1. Fetch news (1 API call + free RSS)
    news, n_new = update_news_store()
    log(f"fetched: {n_new} new articles, {len(news)} total in store")

    # 2. Score - merge in previous scores so only new rows hit FinBERT
    if os.path.exists(SCORED_PATH):
        prev = pd.read_csv(SCORED_PATH,
                           usecols=["url", "vader_score", "finbert_score"])
        news = news.merge(prev, on="url", how="left")
    news = score_articles(news)
    news.to_csv(SCORED_PATH, index=False)

    # 3. Aggregate into a verdict
    score, n_articles = aggregate_sentiment(news, hours=24)
    current = verdict(score)
    log(f"sentiment: {current} (score={score}, n={n_articles})")

    # 4. Gold price for context (1 API call). Non-fatal if it fails.
    price_line = ""
    try:
        price = fetch_gold_price()
        latest, prev_close = price["close"].iloc[-1], price["close"].iloc[-2]
        pct = 100 * (latest - prev_close) / prev_close
        price_line = f"GLD: ${latest:.2f} ({pct:+.2f}% vs prev close)"
        log(price_line)
    except Exception as e:
        log(f"price fetch failed: {e}")

    # 5. Decide whether to notify
    state = load_state()
    flipped = state["last_verdict"] is not None and current != state["last_verdict"]
    strong = abs(score) >= STRONG

    if flipped or strong:
        reason = "sentiment changed" if flipped else "strong reading"
        title = f"Gold sentiment: {current}"
        body = (f"{reason}: {state['last_verdict']} -> {current}\n"
                f"score {score} across {n_articles} articles (24h)\n"
                f"{price_line}")
        notify_all(title, body)
        log(f"NOTIFIED ({reason})")
    else:
        log("no notification (no change, not strong)")

    state.update({"last_verdict": current, "last_score": score,
                  "last_run": datetime.now().isoformat()})
    save_state(state)
    log("--- pipeline end ---")
    return current, score


if __name__ == "__main__":
    run_pipeline()