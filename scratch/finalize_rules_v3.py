import json
import os

RULES_DIR = "/Users/xiz/Quant_trade/quant_data/news_collectors/gdelt/company_rules"

# 26 只目标标的的根目录锚点配置
STRICT_ANCHORS = {
    "TSM": ["Taiwan Semiconductor", "TSMC"],
    "ASML": ["ASML Holding", "Lithography"],
    "CRM": ["Salesforce", "Benioff"],
    "PLTR": ["Palantir", "Alex Karp"],
    "NOW": ["ServiceNow", "Workflow"],
    "ADBE": ["Adobe", "Photoshop"],
    "NFLX": ["Netflix", "Streaming"],
    "UBER": ["Uber Technologies", "Dara Khosrowshahi"],
    "SNOW": ["Snowflake Inc", "Data Cloud"],
    "MDB": ["MongoDB", "Dev Ittycheria"],
    "PANW": ["Palo Alto Networks", "Cybersecurity"],
    "CRWD": ["CrowdStrike", "George Kurtz"],
    "SMCI": ["Super Micro Computer", "Charles Liang"],
    "AMAT": ["Applied Materials", "Gary Dickerson"],
    "LRCX": ["Lam Research", "Tim Archer"],
    "KLAC": ["KLA Corp"],
    "TXN": ["Texas Instruments"],
    "ADI": ["Analog Devices"],
    "MCHP": ["Microchip Technology"],
    "ORCL": ["Oracle Corp", "Larry Ellison"],
    "FTNT": ["Fortinet", "Ken Xie"],
    "ABNB": ["Airbnb", "Brian Chesky"],
    "CSCO": ["Cisco Systems", "Chuck Robbins"],
    "IBM": ["International Business Machines", "Watson"],
    "DELL": ["Dell Technologies", "Michael Dell"],
    "INTU": ["Intuit", "QuickBooks"]
}

def finalize_rules_v3():
    for symbol, anchors in STRICT_ANCHORS.items():
        file_path = os.path.join(RULES_DIR, f"{symbol}.json")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # 🐛 BUG FIX: 必须把这些字段放在 ROOT 级别，AmbiguousNameRule 才能读到
        data["strict_validation_keywords"] = anchors
        
        # 确保 ambiguous 规则块存在且设置了必需词
        amb_cfg = data.setdefault("special_rules", {}).setdefault("ambiguous", {})
        amb_cfg["required_keywords"] = [symbol]
        amb_cfg["min_matches"] = 1
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
            print(f"🚀 FINALIZED Rule for {symbol}.json (Root-Anchors synced)")

if __name__ == "__main__":
    finalize_rules_v3()
