#!/usr/bin/env python3
"""
Base Rule Class for Special Company Handling
"""

from datetime import datetime
from abc import ABC, abstractmethod


class BaseRule(ABC):
    """基础规则类"""
    
    def __init__(self, symbol, config):
        self.symbol = symbol
        self.config = config
    
    @abstractmethod
    def get_keywords(self, article_date):
        """
        根据文章日期获取应该使用的关键词
        
        Args:
            article_date: datetime对象，文章发布日期
            
        Returns:
            list: 关键词列表
        """
        pass
    
    @abstractmethod
    def should_include(self, article):
        """
        判断文章是否应该被包含
        
        Args:
            article: dict, 包含title, content, date等字段
            
        Returns:
            bool: True表示应该包含，False表示应该排除
        """
        pass
    
    def parse_date(self, date_str):
        """解析日期字符串"""
        if isinstance(date_str, datetime):
            return date_str
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except:
            return None


class NoRule(BaseRule):
    """无特殊规则的默认类"""
    
    def get_keywords(self, article_date):
        return []
    
    def should_include(self, article):
        return True
