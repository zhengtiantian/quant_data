import os
import json
import re
from pymongo import MongoClient

# 配置
MONGO_URI = "mongodb://root:root@127.0.0.1:37018/"
DB_NAME = "quant_data"
COLLECTION_NAME = "news_articles_test"
RULES_DIR = "/home/xiz/quant_trading/quant_data/news_collectors/gdelt/company_rules"

def load_rules():
    rules = {}
    for filename in os.listdir(RULES_DIR):
        if filename.endswith(".json"):
            symbol = filename.replace(".json", "")
            with open(os.path.join(RULES_DIR, filename), "r") as f:
                rules[symbol] = json.load(f)
    return rules

def analyze_matches():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    collection = db[COLLECTION_NAME]
    
    rules = load_rules()
    
    print(f"{'Symbol':<8} | {'Match Type':<15} | {'Keywords Found':<30} | {'Title'}")
    print("-" * 100)
    
    cursor = collection.find().sort("symbol", 1)
    
    stats = {}

    for doc in cursor:
        symbol = doc.get("symbol")
        title = doc.get("title", "")
        content = doc.get("content", "")
        full_text = f"{title} {content}".lower()
        
        if symbol not in rules:
            continue
            
        rule = rules[symbol]
        expansion_keywords = rule.get("expansion_keywords", [])
        primary_keywords = rule.get("primary_keywords", [])
        
        # 简化版排除词判断（此处仅为分析，不代表过滤逻辑）
        matches = []
        match_type = "Weak (Name)"
        
        # 1. 检查扩展词（强匹配特征）
        for kw in expansion_keywords:
            if len(kw) < 3: continue
            if re.search(r'\b' + re.escape(kw.lower()) + r'\b', full_text):
                matches.append(kw)
                match_type = "Strong (Product)"
        
        # 2. 如果没匹配到扩展词，检查主关键词
        if not matches:
            for kw in primary_keywords:
                if re.search(r'\b' + re.escape(kw.lower()) + r'\b', full_text):
                    matches.append(kw)
        
        # 统计
        if symbol not in stats:
            stats[symbol] = {"total": 0, "strong": 0, "weak": 0}
        stats[symbol]["total"] += 1
        if match_type == "Strong (Product)":
            stats[symbol]["strong"] += 1
        else:
            stats[symbol]["weak"] += 1

        # 打印部分示例
        if stats[symbol]["total"] <= 5: # 每个公司展示前5条
            kw_str = ", ".join(matches[:3]) + ("..." if len(matches) > 3 else "")
            print(f"{symbol:<8} | {match_type:<15} | {kw_str:<30} | {title[:60]}")

    print("\n" + "="*50)
    print("📊 匹配深度统计报告")
    print("="*50)
    print(f"{'Symbol':<8} | {'Total':<8} | {'Strong %':<10} | {'Weak %':<10}")
    print("-" * 50)
    for sym, data in stats.items():
        strong_pc = (data["strong"] / data["total"]) * 100
        weak_pc = (data["weak"] / data["total"]) * 100
        print(f"{sym:<8} | {data['total']:<8} | {strong_pc:>8.1f}% | {weak_pc:>8.1f}%")

if __name__ == "__main__":
    analyze_matches()
