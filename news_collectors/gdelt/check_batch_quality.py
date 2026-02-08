#!/usr/bin/env python3
"""
检查第一批次数据的质量
1. 数据总量
2. 文章匹配程度（每个公司的文章数）
3. date字段校验（是否还有000000的情况）
4. 数据质量分布
"""

from pymongo import MongoClient
from collections import Counter
from datetime import datetime

# 连接数据库
client = MongoClient('mongodb://root:root@localhost:37018/')
db = client['quant_data']
col = db['news_articles']

print('='*70)
print('第一批次数据质量检查报告')
print('='*70)

# 1. 数据总量
print('\n📊 1. 数据总量统计')
print('-'*70)
total_count = col.count_documents({})
print(f'总文章数: {total_count:,} 条')

# 按来源统计
sources = col.aggregate([
    {"$group": {"_id": "$source.platform", "count": {"$sum": 1}}},
    {"$sort": {"count": -1}}
])
print('\n按来源统计:')
for src in sources:
    print(f'  {src["_id"]}: {src["count"]:,} 条')

# 2. 文章匹配程度 - 每个公司的文章数
print('\n📈 2. 文章匹配程度（Top 20公司）')
print('-'*70)
company_stats = col.aggregate([
    {"$group": {
        "_id": {"symbol": "$symbol", "name": "$name"},
        "count": {"$sum": 1}
    }},
    {"$sort": {"count": -1}},
    {"$limit": 20}
])

print(f'{"公司代码":<10} {"公司名称":<40} {"文章数":>10}')
print('-'*70)
for stat in company_stats:
    symbol = stat['_id']['symbol']
    name = stat['_id']['name'][:38] if stat['_id']['name'] else 'N/A'
    count = stat['count']
    print(f'{symbol:<10} {name:<40} {count:>10,}')

# 3. date字段校验
print('\n🔍 3. Date字段校验')
print('-'*70)

# 3.1 检查null的date
null_dates = col.count_documents({
    "$or": [
        {"date": {"$exists": False}},
        {"date": None},
        {"date": ""}
    ]
})
print(f'date为null的记录: {null_dates} 条')

# 3.2 检查时分秒为000000的记录
zero_time = col.count_documents({
    "date": {"$regex": r"^\d{8}000000$"}
})
print(f'时分秒为000000的记录: {zero_time} 条')

# 3.3 检查date格式
print('\ndate格式分布:')
date_length_stats = col.aggregate([
    {"$project": {
        "date_length": {"$strLenCP": {"$toString": "$date"}}
    }},
    {"$group": {
        "_id": "$date_length",
        "count": {"$sum": 1}
    }},
    {"$sort": {"_id": 1}}
])

for stat in date_length_stats:
    length = stat['_id']
    count = stat['count']
    print(f'  长度 {length}: {count:,} 条')

# 3.4 显示一些样本date
print('\n样本date (前10条):')
samples = col.find({}, {"date": 1, "timestamp": 1, "publishedAt": 1, "url": 1}).limit(10)
for i, doc in enumerate(samples, 1):
    print(f'\n  {i}. date: {doc.get("date", "N/A")}')
    print(f'     timestamp: {doc.get("timestamp", "N/A")}')
    print(f'     publishedAt: {doc.get("publishedAt", "N/A")}')
    print(f'     url: {doc.get("url", "N/A")[:60]}...')

# 4. 数据质量分布
print('\n📋 4. 数据质量分布')
print('-'*70)
quality_stats = col.aggregate([
    {"$group": {
        "_id": "$data_quality",
        "count": {"$sum": 1}
    }},
    {"$sort": {"count": -1}}
])

total = col.count_documents({})
for stat in quality_stats:
    quality = stat['_id'] or 'unknown'
    count = stat['count']
    percentage = (count * 100) / total if total > 0 else 0
    print(f'  {quality}: {count:,} 条 ({percentage:.1f}%)')

# 5. 时间范围
print('\n📅 5. 数据时间范围')
print('-'*70)

# 找出最早和最晚的文章
earliest = col.find_one(
    {"date": {"$exists": True, "$ne": None}},
    sort=[("date", 1)]
)
latest = col.find_one(
    {"date": {"$exists": True, "$ne": None}},
    sort=[("date", -1)]
)

if earliest:
    print(f'最早文章: {earliest.get("date", "N/A")}')
    print(f'  URL: {earliest.get("url", "N/A")[:60]}...')
    
if latest:
    print(f'最晚文章: {latest.get("date", "N/A")}')
    print(f'  URL: {latest.get("url", "N/A")[:60]}...')

# 6. 按日期分组统计
print('\n📊 6. 按日期分组统计（前20天）')
print('-'*70)
date_stats = col.aggregate([
    {"$match": {"date": {"$exists": True, "$ne": None}}},
    {"$project": {
        "date_only": {"$substr": ["$date", 0, 8]}
    }},
    {"$group": {
        "_id": "$date_only",
        "count": {"$sum": 1}
    }},
    {"$sort": {"_id": 1}},
    {"$limit": 20}
])

print(f'{"日期":<12} {"文章数":>10}')
print('-'*70)
for stat in date_stats:
    date_str = stat['_id']
    count = stat['count']
    print(f'{date_str:<12} {count:>10,}')

# 7. 检查进度文件
print('\n🔄 7. 批次进度')
print('-'*70)
import os
PROGRESS_FILE = "/mnt/data24t/docker-volumes/gdelt_cache/progress.txt"
if os.path.exists(PROGRESS_FILE):
    with open(PROGRESS_FILE, 'r') as f:
        current_batch = f.read().strip()
    print(f'当前已完成批次: {current_batch}')
else:
    print('进度文件不存在')

print('\n' + '='*70)
print('✅ 检查完成')
print('='*70)
