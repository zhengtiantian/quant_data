# Special Rules for Stock News Collection

## 需要特殊处理的公司

### 1. META (Facebook → Meta, 2021-10-28)
**问题**: 公司改名
**规则**: 
- 2021-10-28之前: 使用 "Facebook", "Facebook Inc."
- 2021-10-28之后: 使用 "Meta", "Meta Platforms"

### 2. GOOGL (Google → Alphabet, 2015-08-10)
**问题**: 公司重组
**规则**:
- 2015-08-10之前: 使用 "Google", "Google Inc."
- 2015-08-10之后: 使用 "Alphabet", "Alphabet Inc."

### 3. CRM (Salesforce 收购历史)
**问题**: 多次重大收购
**规则**:
- 2018-05之后: 添加 "MuleSoft"
- 2019-08之后: 添加 "Tableau"
- 2021-07之后: 添加 "Slack"

### 4. AVGO (Broadcom 收购历史)
**问题**: 多次重大收购
**规则**:
- 2018-11之后: 添加 "CA Technologies"
- 2019-11之后: 添加 "Symantec"
- 2023-11之后: 添加 "VMware"

### 5. ARM (私有化 → IPO, 2016-09 → 2023-09)
**问题**: 私有化期间新闻较少
**规则**:
- 2016-09 到 2023-09: 添加 "SoftBank ARM"
- 2023-09之后: 正常使用 "ARM", "Arm Holdings"

### 6. TSLA (Tesla)
**问题**: 名字可能匹配到 Nikola Tesla
**规则**:
- 必须包含 "Tesla Inc" 或 "Tesla Motors" 或产品名
- 排除 "Nikola Tesla" 相关历史文章

### 7. AMZN (Amazon)
**问题**: 可能匹配到亚马逊雨林
**规则**:
- 必须包含公司相关词: "Amazon.com", "AWS", "Bezos", "Jassy"
- 或产品名: "Alexa", "Prime", "Kindle"

### 8. META (Meta)
**问题**: 可能匹配到 metadata, meta标签
**规则**:
- 必须包含: "Meta Platforms", "Facebook", "Instagram", "WhatsApp", "Zuckerberg"
- 排除纯技术文章中的 "meta"

## 文件结构

```
gdelt/
├── historical_collector.py          # 主收集器
├── special_rules/                   # 特殊规则目录
│   ├── README.md                   # 说明文档
│   ├── __init__.py                 # Python包初始化
│   ├── base_rule.py                # 基础规则类
│   ├── renamed_companies.py        # 改名公司规则
│   ├── acquired_companies.py       # 收购公司规则
│   ├── ambiguous_names.py          # 歧义名称规则
│   └── rule_config.json            # 规则配置文件
```
