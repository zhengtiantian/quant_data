from google.cloud import bigquery
from newspaper import Article
from langdetect import detect
import pandas as pd
import time
import os

# ======================
# 参数配置
# ======================
COMPANY = "Apple"
START = "2020-03-01"
END = "2020-03-03"
MAX_NEWS = 15  # 控制数量以免太慢

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(
    os.path.dirname(__file__), "gdelt.json"
)

# ======================
# BigQuery 查询
# ======================
client = bigquery.Client()

query = f"""
SELECT
  DocumentIdentifier,
  V2Tone,
  V2Themes,
  V2Organizations,
  DATE
FROM
  `gdelt-bq.gdeltv2.gkg`
WHERE
  DATE BETWEEN 20200301000000 AND 20200303235959
  AND LOWER(V2Organizations) LIKE '%{COMPANY.lower()}%'
LIMIT {MAX_NEWS}
"""

print(f"🛰️ Fetching {COMPANY} news URLs from GDELT BigQuery...")
rows = list(client.query(query))

if not rows:
    print("❌ No records found in GDELT for given range.")
    exit(0)

results = []

# ======================
# 抓取新闻正文
# ======================
for i, row in enumerate(rows, 1):
    url = row.DocumentIdentifier
    print(f"\n[{i}] {url}")
    try:
        article = Article(url)
        article.download()
        article.parse()

        title = article.title.strip()
        text = article.text.strip()

        # 跳过空内容
        if not text or not title:
            print("⚠️ Empty article, skip.")
            continue

        # 自动语言检测，只保留英文
        try:
            lang = detect(text[:500])
            if lang != "en":
                print(f"⚠️ Non-English ({lang}), skip.")
                continue
        except:
            print("⚠️ Language detection failed, skip.")
            continue

        # 清理
        text_clean = text.replace("\n", " ").replace("  ", " ")

        results.append({
            "date": str(row.DATE),
            "title": title,
            "tone": row.V2Tone,
            "url": url,
            "content_sample": text_clean[:600]
        })

        print(f"✅ {title[:80]}... (Tone={row.V2Tone:.2f})")

    except Exception as e:
        print(f"❌ Failed to parse {url}: {e}")

    time.sleep(1.5)  # 防止封锁

# ======================
# 输出结果
# ======================
if not results:
    print("\n❌ No valid English articles found.")
else:
    df = pd.DataFrame(results)
    print("\n=== Sample News Extracts ===")
    for _, row in df.iterrows():
        print(f"\n🗞️ [{row['date']}] {row['title']}")
        print(f"Tone: {row['tone']}")
        print(f"URL: {row['url']}")
        print(row['content_sample'])
        print("=" * 100)

    # 你也可以保存结果
    # df.to_csv(f"{COMPANY.lower()}_gdelt_news.csv", index=False)