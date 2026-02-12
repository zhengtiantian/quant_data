#!/usr/bin/env python3
"""
展示所有股票的数据质量分析
"""

from pymongo import MongoClient
from collections import defaultdict

client = MongoClient('mongodb://root:root@localhost:37018/')
db = client['quant_data']
col = db['news_articles']

print('='*80)
print('所有股票数据质量完整报告')
print('='*80)

# 1. 总体统计
total = col.count_documents({})
full_count = col.count_documents({"data_quality": "full"})
title_only_count = col.count_documents({"data_quality": "title_only"})

print(f'\n📊 总体数据统计')
print('-'*80)
print(f'总文章数: {total:,}')
print(f'  Full (有完整正文):     {full_count:,} 条 ({full_count/total*100:.1f}%)')
print(f'  Title_Only (只有标题): {title_only_count:,} 条 ({title_only_count/total*100:.1f}%)')

# 2. 按公司统计
print(f'\n📈 所有公司数据质量对比')
print('-'*80)

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
company_data = defaultdict(lambda: {"full": 0, "title_only": 0, "url_only": 0, "name": ""})

for item in company_quality:
    symbol = item['_id']['symbol']
    quality = item['_id']['quality']
    count = item['count']
    name = item['_id']['name']
    
    company_data[symbol][quality] = count
    company_data[symbol]['name'] = name

# 按总数排序
sorted_companies = sorted(
    company_data.items(),
    key=lambda x: x[1]['full'] + x[1]['title_only'],
    reverse=True
)

print(f'{"排名":<4} {"公司":<8} {"名称":<35} {"Full":>8} {"Title":>8} {"Total":>8} {"Title%":>8}')
print('-'*80)

for rank, (symbol, data) in enumerate(sorted_companies, 1):
    name = data['name'][:33] if data['name'] else 'N/A'
    full = data['full']
    title = data['title_only']
    total_count = full + title
    title_pct = (title / total_count * 100) if total_count > 0 else 0
    
    # 根据title_only比例标记
    if title_pct > 10:
        marker = '🔴'
    elif title_pct > 5:
        marker = '⚠️'
    else:
        marker = '✅'
    
    print(f'{rank:<4} {symbol:<8} {name:<35} {full:>8,} {title:>8,} {total_count:>8,} {title_pct:>7.1f}% {marker}')

# 3. 统计信息
print(f'\n📊 统计摘要')
print('-'*80)
print(f'总公司数: {len(sorted_companies)}')

# 计算title_only比例分布
high_title_pct = sum(1 for _, data in sorted_companies 
                     if (data['title_only'] / (data['full'] + data['title_only']) * 100) > 10)
medium_title_pct = sum(1 for _, data in sorted_companies 
                       if 5 < (data['title_only'] / (data['full'] + data['title_only']) * 100) <= 10)
low_title_pct = sum(1 for _, data in sorted_companies 
                    if (data['title_only'] / (data['full'] + data['title_only']) * 100) <= 5)

print(f'\nTitle_Only比例分布:')
print(f'  🔴 高比例 (>10%):  {high_title_pct} 家公司')
print(f'  ⚠️ 中比例 (5-10%): {medium_title_pct} 家公司')
print(f'  ✅ 低比例 (≤5%):   {low_title_pct} 家公司')

# 4. Top 5 和 Bottom 5
print(f'\n🏆 文章数最多的5家公司:')
for rank, (symbol, data) in enumerate(sorted_companies[:5], 1):
    total_count = data['full'] + data['title_only']
    print(f'  {rank}. {symbol}: {total_count:,} 篇')

print(f'\n📉 文章数最少的5家公司:')
for rank, (symbol, data) in enumerate(sorted_companies[-5:], 1):
    total_count = data['full'] + data['title_only']
    print(f'  {rank}. {symbol}: {total_count:,} 篇')

print('\n' + '='*80)
print('✅ 报告完成')
print('='*80)
