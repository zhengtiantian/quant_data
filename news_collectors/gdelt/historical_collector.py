import requests, zipfile, io, pandas as pd, time
from datetime import datetime, timedelta
from trafilatura import fetch_url, extract
import re

def fetch_gdelt_with_articles(keyword="Apple", start="2020-01-01", end="2020-11-23"):
    base = "http://data.gdeltproject.org/gkg"
    start_dt = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end, "%Y-%m-%d")
    current = start_dt
    keyword_pattern = re.compile(keyword, re.IGNORECASE)

    print(f"🛰️ Fetching GDELT + Articles for {keyword} ({start} → {end})")
    found_articles = []

    while current <= end_dt:
        ts = current.strftime("%Y%m%d%H%M00")
        url = f"{base}/{ts}.gkg.csv.zip"
        try:
            r = requests.get(url, timeout=10)
            if r.status_code != 200:
                current += timedelta(minutes=15)
                continue

            z = zipfile.ZipFile(io.BytesIO(r.content))
            with z.open(z.namelist()[0]) as f:
                df = pd.read_csv(f, sep="\t", header=None, usecols=[1, 7, 9, 15])
                df.columns = ["Date", "Themes", "Tone", "URL"]
                df["Date"] = pd.to_datetime(df["Date"].astype(str), format="%Y%m%d%H%M%S", errors="coerce")

                # 过滤空URL
                df = df[df["URL"].notna()]

                # 抽取 Apple 相关新闻 URL
                apple_rows = df[df["Themes"].str.contains("APPLE", na=False, case=False)]
                for _, row in apple_rows.iterrows():
                    page = fetch_url(row["URL"])
                    if not page:
                        continue
                    content = extract(page)
                    if not content:
                        continue
                    title_line = content.strip().split("\n")[0][:120]
                    if not keyword_pattern.search(content):
                        continue

                    found_articles.append({
                        "date": row["Date"],
                        "tone": row["Tone"],
                        "url": row["URL"],
                        "title": title_line,
                        "text": content[:400]
                    })
                    print(f"\n✅ {row['Date']} | {title_line}")
                    print(f"Tone={row['Tone']}")
                    print(row['URL'])
                    print(content[:300].replace("\n", " ") + "...")
                    print("-" * 80)

        except Exception as e:
            print("⚠️ skip", url, e)

        current += timedelta(minutes=15)
        time.sleep(0.3)

    print(f"\n📊 Total Apple-related news: {len(found_articles)}")
    return found_articles


if __name__ == "__main__":
    fetch_gdelt_with_articles(keyword="Apple", start="2020-01-01", end="2020-01-02")