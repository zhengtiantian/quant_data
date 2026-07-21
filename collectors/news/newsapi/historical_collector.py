import os
import requests
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("NEWS_API_KEY")

def fetch_news_range(query="Tesla", start="2020-01-01", end="2020-01-05"):
    url = (
        f"https://newsapi.org/v2/everything?"
        f"q={query}&from={start}&to={end}&"
        f"sortBy=publishedAt&language=en&apiKey={API_KEY}"
    )

    print(f"🛰️ Fetching news for [{query}] from {start} to {end}")

    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json()

        if not data.get("articles"):
            print("⚠️ No articles found.")
            return

        for i, a in enumerate(data["articles"][:10], 1):
            print(f"\n[{i}] {a['publishedAt']} - {a['title']}")
            print(a["description"] or "")
            print(a["url"])
            print("-" * 100)

    except Exception as e:
        print("❌ Fetch failed:", e)


if __name__ == "__main__":
    fetch_news_range("AAPL", "2020-01-01", "2020-01-05")