#!/usr/bin/env python3
"""
Modified RuleManager to load from individual company JSON files.
"""

import json
import os
import glob
from datetime import datetime
from .base_rule import BaseRule, NoRule
from .renamed_companies import RenamedCompanyRule
from .acquired_companies import AcquiredCompanyRule
from .ambiguous_names import AmbiguousNameRule
from .composite_rule import CompositeRule

class RuleManager:
    """规则管理器 - 从 company_rules/ 目录加载独立的 JSON 文件"""
    
    def __init__(self, rules_dir=None):
        if rules_dir is None:
            # 默认路径指向项目根目录下的 company_rules
            # 当前文件位于 collectors/news/gdelt/special_rules/__init__.py
            # 目标目录是 collectors/news/gdelt/company_rules/
            rules_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "company_rules")
        
        self.rules_dir = rules_dir
        self.company_configs = {}
        self.rules = {}
        self._load_all_rules()
    
    def _load_all_rules(self):
        """扫描目录并加载所有 .json 文件"""
        if not os.path.exists(self.rules_dir):
            print(f"⚠️ Rules directory not found: {self.rules_dir}")
            return

        json_files = glob.glob(os.path.join(self.rules_dir, "*.json"))
        print(f"📂 Loading rules from {len(json_files)} company files...")
        
        for file_path in json_files:
            try:
                with open(file_path, "r") as f:
                    config = json.load(f)
                    symbol = config.get("symbol")
                    if symbol:
                        self.company_configs[symbol] = config
                        self._initialize_rule(symbol, config)
            except Exception as e:
                print(f"⚠️ Failed to load rule at {file_path}: {e}")

    def _initialize_rule(self, symbol, config):
        """根据配置初始化具体的规则类（支持组合多个规则）"""
        special = config.get("special_rules", {})
        active_rules = []
        
        # 1. 改名规则
        if "renamed" in special:
            active_rules.append(RenamedCompanyRule(symbol, special["renamed"]))
        elif "rename_date" in special: # 兼容旧格式
            active_rules.append(RenamedCompanyRule(symbol, special))
            
        # 2. 歧义规则
        if "ambiguous" in special:
            company_name = config.get("name", symbol)
            # 如果是字典（详细配置）
            if isinstance(special["ambiguous"], dict):
                active_rules.append(AmbiguousNameRule(symbol, special["ambiguous"], company_name=company_name, full_config=config))
            # 如果仅仅是 True（直接使用旧配置）
            elif special["ambiguous"] is True:
                active_rules.append(AmbiguousNameRule(symbol, special, company_name=company_name, full_config=config))
                
        # 3. 收购规则
        if "acquisitions" in special:
            active_rules.append(AcquiredCompanyRule(symbol, special))
            
        # 4. 特殊期间 (如 ARM 私有化)
        if "special_periods" in special:
            # 这里暂时可以作为扩展关键词处理，或者根据需要添加特定类
            pass

        if len(active_rules) == 0:
            self.rules[symbol] = NoRule(symbol, config)
        elif len(active_rules) == 1:
            self.rules[symbol] = active_rules[0]
        else:
            self.rules[symbol] = CompositeRule(symbol, active_rules)

    def get_keywords(self, symbol, article_date):
        """获取该股票的所有有效关键词（基础关键词 + 扩展词 + 特殊日期关键词）"""
        config = self.company_configs.get(symbol, {})
        primary = config.get("primary_keywords", [])
        expansion = config.get("expansion_keywords", [])
        
        # 处理特殊期间的关键词 (如 ARM)
        special_period_keywords = []
        special = config.get("special_rules", {})
        if "special_periods" in special:
            periods = special["special_periods"]
            for p_name, p_cfg in periods.items():
                start = datetime.strptime(p_cfg["start_date"], "%Y-%m-%d")
                end = datetime.strptime(p_cfg["end_date"], "%Y-%m-%d")
                if start <= article_date <= end:
                    special_period_keywords.extend(p_cfg.get("additional_keywords", []))

        # 从特殊规则中获取针对日期的额外关键词 (如 Facebook/Meta 切换)
        rule = self.rules.get(symbol)
        special_rule_keywords = rule.get_keywords(article_date) if rule else []
        
        # 合并去重
        return list(set(primary + expansion + special_period_keywords + special_rule_keywords))

    def should_include(self, symbol, article):
        """应用二次过滤规则"""
        rule = self.rules.get(symbol)
        if rule:
            return rule.should_include(article)
        return True

    def get_all_symbols(self):
        return list(self.company_configs.keys())
