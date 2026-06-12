import os
import yfinance as yf
from datetime import datetime, UTC
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient, errors
import pandas as pd
import time

# =========================================================
# 1. Load env
# =========================================================

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[2]
GLOBAL_ENV = ROOT / ".env"

load_dotenv(GLOBAL_ENV, override=False)
print(f"Loaded GLOBAL .env: {GLOBAL_ENV}")

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise RuntimeError("Missing MONGO_URI")


# =========================================================
# 2. Load stock universe
# =========================================================

def load_stock_universe():
    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["stock_universe"]
    symbols = [s["symbol"] for s in col.find({}, {"symbol": 1, "_id": 0})]
    # Optional filter: SYMBOLS_FILTER=AAPL,MSFT,... runs only those symbols
    sym_filter = os.getenv("SYMBOLS_FILTER", "")
    if sym_filter:
        allowed = {s.strip().upper() for s in sym_filter.split(",")}
        symbols = [s for s in symbols if s in allowed]
    print(f"Loaded {len(symbols)} stock symbols")
    return symbols


# =========================================================
# 3. FIXED: Safe float extractor (handles MultiIndex)
# =========================================================

def val(x):
    """Return scalar float/int regardless of MultiIndex."""
    if isinstance(x, pd.Series):
        return float(x.values[0])
    return float(x)


# =========================================================
# 4. Fetch 10Y 1D history safely
# =========================================================

def fetch_daily_history(symbol, period="10y", interval="1d"):
    try:
        df = yf.download(
            symbol,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False
        )
    except Exception as e:
        print(f"❌ {symbol} download failed: {e}")
        return []

    if df.empty:
        print(f"⚠️ No data for {symbol}")
        return []

    df = df.dropna()

    # Fix MultiIndex columns (some stocks return ("Open", ""), etc.)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    records = []
    for ts, row in df.iterrows():
        try:
            records.append({
                "symbol": symbol,
                "timestamp": ts.to_pydatetime().replace(tzinfo=UTC).isoformat(),
                "open": val(row["Open"]),
                "high": val(row["High"]),
                "low": val(row["Low"]),
                "close": val(row["Close"]),
                "volume": int(row["Volume"]),
                "interval": interval,
                "source": "yahoo",
                "collectedAt": datetime.now(UTC).isoformat()
            })
        except Exception as e:
            print(f"⚠️ Row parse error for {symbol}: {e}")
            continue

    return records


# =========================================================
# 5. Save to MongoDB
# =========================================================

def save_history(records):
    if not records:
        return 0

    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["stock_prices_history"]

    try:
        col.create_index(
            [("symbol", 1), ("timestamp", 1)],
            unique=True,
            sparse=True
        )
    except:
        pass

    inserted = 0

    for rec in records:
        try:
            col.insert_one(rec)
            inserted += 1
        except errors.DuplicateKeyError:
            continue

    return inserted


# =========================================================
# 6. Main collector
# =========================================================

def collect_history_all():
    symbols = load_stock_universe()

    total = 0

    # Default to an incremental window ("1mo") so the daily job只补最近几天;
    # 设 PRICE_HISTORY_PERIOD=10y 可做一次性全量回填。
    period = os.getenv("PRICE_HISTORY_PERIOD", "1mo")
    sleep_s = float(os.getenv("PRICE_HISTORY_SLEEP", "0.8"))

    print(f"\n=== Task: daily 1d history (period={period}) ===")

    for sym in symbols:
        print(f"\nFetching 1D {sym} (period={period}) ...")
        data = fetch_daily_history(sym, period=period)

        count = save_history(data)
        total += count

        print(f"Inserted {count} records for {sym}")
        time.sleep(sleep_s)

    print(f"\n=== DONE: Inserted {total} records ===")


# =========================================================
# Run
# =========================================================

if __name__ == "__main__":
    collect_history_all()