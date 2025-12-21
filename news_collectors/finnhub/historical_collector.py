import os, requests
from datetime import date
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("FINNHUB_API_KEY")

def fetch_finnhub_news(symbol="AAPL", start="2020-01-01", end="2020-11-10"):
    url = f"https://finnhub.io/api/v1/company-news?symbol={symbol}&from={start}&to={end}&token={API_KEY}"
    print(f"🛰️ Fetching news for {symbol} from {start} to {end}")
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    data = r.json()
    if not data:
        print("⚠️ No news found")
        return
    for i, item in enumerate(data[:10], 1):
        print(f"\n[{i}] {item['datetime']} - {item['headline']}")
        print(item.get("summary", ""))
        print(item["url"])
        print("-" * 100)

if __name__ == "__main__":
    fetch_finnhub_news("NVDA", "2020-01-01", "2020-01-05")