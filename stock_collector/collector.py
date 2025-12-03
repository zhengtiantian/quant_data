import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from dotenv import load_dotenv
from pymongo import MongoClient

# Load env
CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
GLOBAL_ENV = ROOT / ".env"
load_dotenv(GLOBAL_ENV, override=True)
print("Loaded:", GLOBAL_ENV)

MONGO_URI = os.getenv("MONGO_URI")
URL_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
URL_NASDAQ100 = "https://en.wikipedia.org/wiki/Nasdaq-100"
URL_DOW30 = "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"


def fetch_table(url, required_headers):
    print(f"\nFetching {url} ...")
    resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")

    tables = soup.find_all("table", class_="wikitable")

    for table in tables:
        rows = table.find_all("tr")
        header_cells = [th.get_text(strip=True) for th in rows[0].find_all(["th", "td"])]

        # Check if table contains all required headers
        if all(h in header_cells for h in required_headers):
            print("Matched table:", header_cells[:6])

            results = []
            for row in rows[1:]:
                cells = [td.get_text(strip=True) for td in row.find_all("td")]
                if len(cells) < len(required_headers):
                    continue

                results.append(cells)

            return results

    raise RuntimeError("ERROR: No matching table found")


def parse_sp500():
    rows = fetch_table(URL_SP500, ["Symbol", "Security"])
    parsed = []
    for c in rows:
        parsed.append({
            "symbol": c[0],
            "name": c[1],
            "sector": c[3],
            "sub_industry": c[4],
            "location": c[5],
            "cik": c[7],
            "index": "SP500"
        })
    return parsed


def parse_nasdaq100():
    rows = fetch_table(URL_NASDAQ100, ["Ticker", "Company"])
    parsed = []
    for c in rows:
        parsed.append({
            "symbol": c[0],
            "name": c[1],
            "sector": None,
            "sub_industry": None,
            "location": None,
            "cik": None,
            "index": "NASDAQ100"
        })
    return parsed


def parse_dow30():
    rows = fetch_table(URL_DOW30, ["Symbol", "Company"])
    parsed = []
    for c in rows:
        parsed.append({
            "symbol": c[0],
            "name": c[1],
            "sector": None,
            "sub_industry": None,
            "location": None,
            "cik": None,
            "index": "DOW30"
        })
    return parsed


def save_all(stocks):
    client = MongoClient(MONGO_URI)
    col = client["quant_data"]["stock_universe"]

    # wipe old table
    print("\nDropping old stock_universe ...")
    col.drop()

    col.insert_many(stocks)
    print(f"Inserted {len(stocks)} stocks into MongoDB")


if __name__ == "__main__":
    all_stocks = []

    all_stocks += parse_sp500()
    all_stocks += parse_nasdaq100()
    all_stocks += parse_dow30()

    print(f"\nTotal stocks collected: {len(all_stocks)}")
    save_all(all_stocks)