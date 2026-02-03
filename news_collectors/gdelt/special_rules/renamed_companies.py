#!/usr/bin/env python3
"""
Renamed Companies Rule Handler
处理公司改名的特殊规则
"""

from datetime import datetime
from .base_rule import BaseRule


class RenamedCompanyRule(BaseRule):
    """改名公司规则"""
    
    def __init__(self, symbol, config):
        super().__init__(symbol, config)
        self.rename_date = self.parse_date(config['rename_date'])
        self.old_keywords = config['old_keywords']
        self.new_keywords = config['new_keywords']
    
    def get_keywords(self, article_date):
        """
        根据文章日期返回应该使用的关键词
        
        改名前: 使用旧关键词
        改名后: 使用新关键词
        """
        if article_date < self.rename_date:
            return self.old_keywords
        else:
            return self.new_keywords
    
    def should_include(self, article):
        """
        判断文章是否应该被包含
        
        对于改名公司，检查文章中是否包含对应时期的关键词
        """
        article_date = article.get('date')
        if not article_date:
            return True  # 无日期信息，保守包含
        
        article_date = self.parse_date(article_date)
        if not article_date:
            return True
        
        # 获取该时期应该使用的关键词
        keywords = self.get_keywords(article_date)
        
        # 检查文章内容
        content = (article.get('title', '') + ' ' + article.get('content', '')).lower()
        
        # 至少匹配一个关键词
        for keyword in keywords:
            if keyword.lower() in content:
                return True
        
        return False


# 示例使用
if __name__ == "__main__":
    # META示例
    meta_config = {
        "rename_date": "2021-10-28",
        "old_name": "Facebook",
        "old_keywords": ["Facebook", "Facebook Inc."],
        "new_keywords": ["Meta", "Meta Platforms"]
    }
    
    meta_rule = RenamedCompanyRule("META", meta_config)
    
    # 测试2020年的文章（改名前）
    old_article = {
        "date": "2020-01-01",
        "title": "Facebook announces new privacy policy",
        "content": "Facebook CEO Mark Zuckerberg..."
    }
    print(f"2020年文章应该包含: {meta_rule.should_include(old_article)}")
    print(f"2020年关键词: {meta_rule.get_keywords(datetime(2020, 1, 1))}")
    
    # 测试2022年的文章（改名后）
    new_article = {
        "date": "2022-01-01",
        "title": "Meta Platforms reports Q4 earnings",
        "content": "Meta CEO Mark Zuckerberg..."
    }
    print(f"2022年文章应该包含: {meta_rule.should_include(new_article)}")
    print(f"2022年关键词: {meta_rule.get_keywords(datetime(2022, 1, 1))}")
