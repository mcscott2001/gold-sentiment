"""Notification channels: Telegram and desktop popup."""
import os
import requests
from dotenv import load_dotenv

load_dotenv()
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram(message):
    if not BOT_TOKEN or not CHAT_ID:
        print("Telegram not configured - skipping")
        return False
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    r = requests.post(url, json={"chat_id": CHAT_ID, "text": message},
                      timeout=15)
    if not r.ok:
        print(f"Telegram send failed: {r.text}")
    return r.ok


def send_desktop(title, message):
    try:
        from plyer import notification
        notification.notify(title=title, message=message[:250], timeout=10)
        return True
    except Exception as e:
        print(f"Desktop notification failed: {e}")
        return False


def notify_all(title, message):
    send_desktop(title, message)
    send_telegram(f"{title}\n\n{message}")