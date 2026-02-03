#!/usr/bin/env python3
"""
Composite Rule for Special Company Handling
Allows multiple rules to be combined for a single company.
"""

from .base_rule import BaseRule

class CompositeRule(BaseRule):
    """组合规则类 - 允许一个公司同时拥有多种规则（如改名+歧义过滤）"""
    
    def __init__(self, symbol, rules):
        super().__init__(symbol, {})
        self.rules = rules
    
    def get_keywords(self, article_date):
        """合并所有子规则的关键词"""
        keywords = []
        for rule in self.rules:
            keywords.extend(rule.get_keywords(article_date))
        return list(set(keywords))
    
    def should_include(self, article):
        """所有子规则都必须通过"""
        for rule in self.rules:
            if not rule.should_include(article):
                return False
        return True
