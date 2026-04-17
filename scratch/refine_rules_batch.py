import json
import os

RULES_DIR = "/Users/xiz/Quant_trade/quant_data/news_collectors/gdelt/company_rules"

# 定义 26 只科技股的“硬核探测词” (必须包含其中之一才算数)
STRICT_ANCHORS = {
    "TSM": ["Taiwan Semiconductor", "TSMC", "Fab "],
    "CRM": ["Salesforce", "Benioff", "SaaS"],
    "ARM": ["Arm Holdings", "Arm CPU", "SoftBank"],
    "MU": ["Micron", "DRAM", "memory chip"],
    "NOW": ["ServiceNow", "Bill McDermott", "ITSM"],
    "AAPL": ["Apple Inc", "iPhone", "MacBook", "Tim Cook"],
    "MSFT": ["Microsoft", "Nadella", "Azure", "Windows"],
    "NVDA": ["NVIDIA", "Jensen Huang", "GPU", "GeForce"],
    "AMD": ["Advanced Micro Devices", "Lisa Su", "Ryzen"],
    "IBM": ["International Business Machines", "Watson", "Arvind Krishna"],
    "NFLX": ["Netflix", "Streaming", "Hastings"],
    "UBER": ["Uber Technologies", "Dara Khosrowshahi", "Ride-hailing"],
    "DELL": ["Dell Technologies", "Michael Dell"],
    "ORCL": ["Oracle Corp", "Larry Ellison", "Oracle Cloud"],
    "INTU": ["Intuit", "QuickBooks", "TurboTax"],
    "ADBE": ["Adobe", "Photoshop", "Creative Cloud"],
    "SNOW": ["Snowflake Inc", "Sridhar Ramaswamy", "Data Cloud"],
    "PLTR": ["Palantir", "Alex Karp", "Gotham", "Foundry"],
    "ASML": ["ASML Holding", "Lithography", "EUV"],
    "PANW": ["Palo Alto Networks", "Cybersecurity", "Nikesh Arora"],
    "CRWD": ["CrowdStrike", "George Kurtz", "Falcon"],
    "FTNT": ["Fortinet", "Ken Xie", "FortiGate"],
    "SMCI": ["Super Micro Computer", "Charles Liang", "Server"],
    "AMAT": ["Applied Materials", "Gary Dickerson"],
    "LRCX": ["Lam Research", "Tim Archer"],
    "KLAC": ["KLA Corp", "KLA-Tencor"],
    "TXN": ["Texas Instruments", "Rich Templeton"],
}

def refine_all_rules():
    for symbol, anchors in STRICT_ANCHORS.items():
        file_path = os.path.join(RULES_DIR, f"{symbol}.json")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # 1. 注入强制核验关键词 (这会触发生产环境代码中的 AmbiguousNameRule 强校验)
        data.setdefault("special_rules", {}).setdefault("ambiguous", {})
        data["special_rules"]["ambiguous"]["strict_validation_keywords"] = anchors
        
        # 2. 针对 TSM 等典型的噪音进行负向过滤
        if symbol == "TSM":
            data.setdefault("negative_keywords", []).extend(["scotsman", "talkingpointsmemo"])
        
        # 3. 提高短 Ticker 的匹配门槛 (非全称命中时需要更多证据)
        if len(symbol) <= 3:
            data["special_rules"]["ambiguous"]["min_matches"] = 2
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
            print(f"✅ Refined {symbol}.json")

if __name__ == "__main__":
    refine_all_rules()
