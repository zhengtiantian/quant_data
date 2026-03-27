#!/usr/bin/env python3
"""详细分析数据质量和匹配程度。"""

from collections import Counter, defaultdict
from pymongo import MongoClient

client = MongoClient('mongodb://root:root@localhost:37018/')
db = client['quant_data']
col = db['news_articles']
AMBIGUOUS_SYMBOLS = ["ARM", "META", "MU", "AMD", "DDOG", "AAPL", "MSFT", "AMZN", "GOOGL"]

print('='*70)
print('数据质量详细分析报告')
print('='*70)

# 1. 总体统计
print('\n📊 1. 总体数据统计')
print('-'*70)
total = col.count_documents({})
full_count = col.count_documents({"data_quality": "full"})
title_only_count = col.count_documents({"data_quality": "title_only"})
url_only_count = col.count_documents({"data_quality": "url_only"})

print(f'总文章数: {total:,}')
print(f'\n数据质量分布:')
print(f'  full (有完整正文):        {full_count:,} 条 ({full_count/total*100:.1f}%)')
print(f'  title_only (只有标题):    {title_only_count:,} 条 ({title_only_count/total*100:.1f}%)')
print(f'  url_only (只有URL):       {url_only_count:,} 条 ({url_only_count/total*100:.1f}%)')

# 2. title_only 文章详细分析
print('\n📰 2. Title_Only 文章详细分析')
print('-'*70)

if title_only_count > 0:
    # 按公司统计
    print(f'\n按公司统计 (Top 10):')
    title_only_by_company = col.aggregate([
        {"$match": {"data_quality": "title_only"}},
        {"$group": {
            "_id": {"symbol": "$symbol", "name": "$name"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ])
    
    print(f'{"公司代码":<10} {"公司名称":<40} {"Title_Only数":>12}')
    print('-'*70)
    for stat in title_only_by_company:
        symbol = stat['_id']['symbol']
        name = stat['_id']['name'][:38] if stat['_id']['name'] else 'N/A'
        count = stat['count']
        print(f'{symbol:<10} {name:<40} {count:>12,}')
    
    # 查看一些样本
    print(f'\n样本 Title_Only 文章 (前10条):')
    samples = col.find(
        {"data_quality": "title_only"},
        {"title": 1, "url": 1, "symbol": 1, "note": 1}
    ).limit(10)
    
    for i, doc in enumerate(samples, 1):
        print(f'\n  {i}. [{doc.get("symbol", "N/A")}] {doc.get("title", "N/A")[:60]}...')
        print(f'     URL: {doc.get("url", "N/A")[:65]}...')
        print(f'     原因: {doc.get("note", "N/A")}')
else:
    print('  没有 title_only 文章')

# 3. Full 文章分析
print('\n📄 3. Full 文章分析')
print('-'*70)

if full_count > 0:
    # 内容长度统计
    content_stats = col.aggregate([
        {"$match": {"data_quality": "full"}},
        {"$group": {
            "_id": None,
            "avg_length": {"$avg": "$content_length"},
            "min_length": {"$min": "$content_length"},
            "max_length": {"$max": "$content_length"}
        }}
    ])
    
    for stat in content_stats:
        print(f'内容长度统计:')
        print(f'  平均长度: {stat["avg_length"]:.0f} 字符')
        print(f'  最短: {stat["min_length"]:,} 字符')
        print(f'  最长: {stat["max_length"]:,} 字符')

# 4. 按公司对比 full vs title_only
print('\n📊 4. 各公司数据质量对比 (Top 10)')
print('-'*70)

company_quality = col.aggregate([
    {"$group": {
        "_id": {
            "symbol": "$symbol",
            "name": "$name",
            "quality": "$data_quality"
        },
        "count": {"$sum": 1}
    }},
    {"$sort": {"_id.symbol": 1, "_id.quality": 1}}
])

# 重组数据
from collections import defaultdict
company_data = defaultdict(lambda: {"full": 0, "title_only": 0, "url_only": 0, "name": ""})

for item in company_quality:
    symbol = item['_id']['symbol']
    quality = item['_id']['quality']
    count = item['count']
    name = item['_id']['name']
    
    company_data[symbol][quality] = count
    company_data[symbol]['name'] = name

# 按总数排序并显示Top 10
sorted_companies = sorted(
    company_data.items(),
    key=lambda x: x[1]['full'] + x[1]['title_only'],
    reverse=True
)[:10]

print(f'{"公司":<8} {"名称":<30} {"Full":>8} {"Title":>8} {"Total":>8} {"Title%":>8}')
print('-'*70)
for symbol, data in sorted_companies:
    name = data['name'][:28] if data['name'] else 'N/A'
    full = data['full']
    title = data['title_only']
    total = full + title
    title_pct = (title / total * 100) if total > 0 else 0
    print(f'{symbol:<8} {name:<30} {full:>8,} {title:>8,} {total:>8,} {title_pct:>7.1f}%')

# 5. Title_Only 原因分布
print('\n🧪 5. Title_Only 原因分布')
print('-'*70)

if title_only_count > 0:
    reason_stats = col.aggregate([
        {"$match": {"data_quality": "title_only"}},
        {"$group": {
            "_id": "$note",
            "count": {"$sum": 1}
        }},
        {"$sort": {"count": -1}},
        {"$limit": 10}
    ])

    print(f'{"原因":<52} {"数量":>12} {"占Title%":>10}')
    print('-'*70)
    for stat in reason_stats:
        reason = (stat["_id"] or "N/A").replace("\n", " ")[:50]
        count = stat["count"]
        pct = count / title_only_count * 100 if title_only_count else 0
        print(f'{reason:<52} {count:>12,} {pct:>9.1f}%')
else:
    print('  没有 title_only 文章')

# 6. 歧义公司专项复核
print('\n🎯 6. 歧义公司专项复核')
print('-'*70)

ambiguous_quality = col.aggregate([
    {"$match": {"symbol": {"$in": AMBIGUOUS_SYMBOLS}}},
    {"$group": {
        "_id": {
            "symbol": "$symbol",
            "name": "$name",
            "quality": "$data_quality"
        },
        "count": {"$sum": 1}
    }},
    {"$sort": {"_id.symbol": 1, "_id.quality": 1}}
])

ambiguous_data = defaultdict(lambda: {"full": 0, "title_only": 0, "url_only": 0, "name": ""})
for item in ambiguous_quality:
    symbol = item["_id"]["symbol"]
    quality = item["_id"]["quality"]
    ambiguous_data[symbol][quality] = item["count"]
    ambiguous_data[symbol]["name"] = item["_id"]["name"] or ""

print(f'{"公司":<8} {"名称":<24} {"Full":>8} {"Title":>8} {"URL":>8} {"Title%":>8}')
print('-'*70)
for symbol in AMBIGUOUS_SYMBOLS:
    data = ambiguous_data[symbol]
    name = data["name"][:22] if data["name"] else "N/A"
    total = data["full"] + data["title_only"] + data["url_only"]
    title_pct = (data["title_only"] / total * 100) if total > 0 else 0
    print(
        f'{symbol:<8} {name:<24} {data["full"]:>8,} {data["title_only"]:>8,} '
        f'{data["url_only"]:>8,} {title_pct:>7.1f}%'
    )

print('\n歧义公司样本复核:')
for symbol in AMBIGUOUS_SYMBOLS[:5]:
    samples = list(col.find(
        {"symbol": symbol},
        {"title": 1, "data_quality": 1, "url": 1}
    ).limit(3))
    if not samples:
        continue
    print(f'\n[{symbol}]')
    for i, doc in enumerate(samples, 1):
        title = (doc.get("title") or "N/A")[:70]
        quality = doc.get("data_quality", "N/A")
        url = (doc.get("url") or "N/A")[:72]
        print(f'  {i}. ({quality}) {title}')
        print(f'     {url}')

# 7. 时间分布
print('\n📅 7. 数据收集进度')
print('-'*70)

import os
PROGRESS_FILE = "/mnt/data24t/docker-volumes/gdelt_cache/progress.txt"
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        current_batch = f.read().strip()
    print(f'当前已完成批次: {current_batch}')
    print(f'平均每批文章数: {total/int(current_batch):.0f}' if int(current_batch) > 0 else '')
else:
    print('进度文件不存在')

print('\n' + '='*70)
print('✅ 分析完成')
print('='*70)
