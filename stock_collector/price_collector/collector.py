import os
import requests
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# =========================================================
# 1. Load env
# =========================================================

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"
FINNHUB_ENV = ROOT / "news_collectors" / "finnhub" / ".env"

load_dotenv(GLOBAL_ENV, override=False)
load_dotenv(FINNHUB_ENV, override=False)

FINNHUB_API_KEY = (
    os.getenv("FINNHUB_API_KEY")
    or os.getenv("FINNHUB_TOKEN")
    or os.getenv("FINNHUB_KEY")
)
MONGO_URI = os.getenv("MONGO_URI")

if not FINNHUB_API_KEY:
    raise RuntimeError("FINNHUB API key missing (checked FINNHUB_API_KEY/FINNHUB_TOKEN/FINNHUB_KEY)")
if not MONGO_URI:
    raise RuntimeError("MONGO_URI missing")

# =========================================================
# 2. Load stock universe (SP500 + NASDAQ100 + DOW30)
# =========================================================

def load_stock_universe():
    """
    Load stock list from MongoDB (previously stored in stock_universe collection).
    """
    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["stock_universe"]

    stocks = list(col.find({}, {"symbol": 1, "_id": 0}))
    return [s["symbol"] for s in stocks]


# =========================================================
# 3. Fetch single stock quote from Finnhub
# =========================================================

def fetch_quote(symbol):
    """
    Finnhub Quote API — free plan available
    Returns:
        c: Current price
        h: High price of the day
        l: Low price of the day
        pc: Previous close
        t: Timestamp (sec)
    """
    url = "https://finnhub.io/api/v1/quote"
    params = {"symbol": symbol, "token": FINNHUB_API_KEY}

    r = requests.get(url, params=params)
    if r.status_code != 200:
        print(f"Failed: {symbol} -> HTTP {r.status_code}")
        return None

    data = r.json()
    if data.get("c", 0) == 0:  # no price available
        return None

    return {
        "symbol": symbol,
        "price": data.get("c"),
        "high": data.get("h"),
        "low": data.get("l"),
        "prev_close": data.get("pc"),
        "timestamp": datetime.fromtimestamp(data.get("t"), UTC).isoformat(),
        "collectedAt": datetime.now(UTC).isoformat(),
        "source": "finnhub"
    }


# =========================================================
# 4. Save to MongoDB
# =========================================================

def save_quotes(quotes):
    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["stock_prices"]

    if quotes:
        col.insert_many(quotes)
        print(f"Inserted {len(quotes)} price records")


# =========================================================
# 5. Main Collector
# =========================================================

def collect_all_prices():
    print("\n=== Loading stock universe ===")
    symbols = load_stock_universe()
    print(f"Total stocks: {len(symbols)}")

    results = []

    for s in symbols:
        q = fetch_quote(s)
        if q:
            results.append(q)

    save_quotes(results)

    print(f"Done. Collected {len(results)} stock quotes.")


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    collect_all_prices()
