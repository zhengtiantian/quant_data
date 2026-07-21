#!/usr/bin/env python3
"""
Acquired Companies Rule Handler
处理公司收购的特殊规则
"""

from datetime import datetime
from .base_rule import BaseRule


class AcquiredCompanyRule(BaseRule):
    """收购公司规则"""
    
    def __init__(self, symbol, config):
        super().__init__(symbol, config)
        self.acquisitions = []
        
        # 解析收购信息
        for acq in config['acquisitions']:
            self.acquisitions.append({
                'date': self.parse_date(acq['date']),
                'company': acq['company'],
                'keywords': acq['keywords']
            })
        
        # 按日期排序
        self.acquisitions.sort(key=lambda x: x['date'])
    
    def get_keywords(self, article_date):
        """
        根据文章日期返回应该添加的关键词
        
        返回所有在该日期之后完成的收购公司的关键词
        """
        additional_keywords = []
        
        for acq in self.acquisitions:
            if article_date >= acq['date']:
                additional_keywords.extend(acq['keywords'])
        
        return additional_keywords
    
    def should_include(self, article):
        """
        判断文章是否应该被包含
        
        对于收购公司，如果文章提到被收购公司，也应该包含
        """
        article_date = article.get('date')
        if not article_date:
            return True
        
        article_date = self.parse_date(article_date)
        if not article_date:
            return True
        
        # 获取该时期的收购关键词
        keywords = self.get_keywords(article_date)
        
        if not keywords:
            return True  # 没有收购发生，使用默认规则
        
        # 检查文章内容
        content = (article.get('title', '') + ' ' + article.get('content', '')).lower()
        
        # 如果提到被收购公司，应该包含
        for keyword in keywords:
            if keyword.lower() in content:
                return True
        
        return True  # 即使没提到被收购公司，也可能是母公司新闻


# 示例使用
if __name__ == "__main__":
    # Salesforce示例
    crm_config = {
        "acquisitions": [
            {
                "date": "2018-05-01",
                "company": "MuleSoft",
                "keywords": ["MuleSoft", "Mule ESB"]
            },
            {
                "date": "2019-08-01",
                "company": "Tableau",
                "keywords": ["Tableau", "Tableau Software"]
            },
            {
                "date": "2021-07-21",
                "company": "Slack",
                "keywords": ["Slack", "Slack Technologies"]
            }
        ]
    }
    
    crm_rule = AcquiredCompanyRule("CRM", crm_config)
    
    # 测试2017年的文章（无收购）
    print(f"2017年额外关键词: {crm_rule.get_keywords(datetime(2017, 1, 1))}")
    
    # 测试2020年的文章（有MuleSoft和Tableau）
    print(f"2020年额外关键词: {crm_rule.get_keywords(datetime(2020, 1, 1))}")
    
    # 测试2022年的文章（有所有收购）
    print(f"2022年额外关键词: {crm_rule.get_keywords(datetime(2022, 1, 1))}")
    
    # 测试Slack相关文章
    slack_article = {
        "date": "2022-01-01",
        "title": "Salesforce integrates Slack with CRM",
        "content": "Slack collaboration tools now available..."
    }
    print(f"Slack文章应该包含: {crm_rule.should_include(slack_article)}")
