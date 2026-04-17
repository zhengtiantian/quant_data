import json
import os

RULES_DIR = "/Users/xiz/Quant_trade/quant_data/news_collectors/gdelt/company_rules"

# 严格限定为这 26 只新加入的标的
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

def fix_all_rules_strict():
    for symbol, anchors in STRICT_ANCHORS.items():
        file_path = os.path.join(RULES_DIR, f"{symbol}.json")
        if not os.path.exists(file_path):
            continue
            
        with open(file_path, 'r') as f:
            data = json.load(f)
            
        # 核心：确保只对这 26 只股票激活 ambiguous 二次校验位
        amb_cfg = data.setdefault("special_rules", {}).setdefault("ambiguous", {})
        
        # 绑定必需词：确保 GDELT 提取逻辑会触发校验
        amb_cfg["required_keywords"] = [symbol, data.get("name", symbol)]
        amb_cfg["min_matches"] = 1
        
        # 绑定锚点词：必须出现这些业务强词才能通关
        amb_cfg["strict_validation_keywords"] = anchors
        
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
            print(f"🎯 Hard-Filter ACTIVATED for target: {symbol}.json")

if __name__ == "__main__":
    fix_all_rules_strict()
