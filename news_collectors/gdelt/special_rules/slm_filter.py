#!/usr/bin/env python3
"""
SLM-based Article Relevance Filter
使用 Qwen 小语言模型进行文章相关性判断（通过 quant_langchain API）
"""

import requests
import os
from typing import Dict, Optional


class SLMFilter:
    """使用 SLM 判断文章是否真正讨论某个公司"""
    
    def __init__(self, api_url: str = None, enabled: bool = True):
        # 自动检测 API 地址
        if api_url is None:
            api_url = os.getenv("QUANT_LANGCHAIN_API", "http://localhost:18000")
        
        self.api_url = api_url.rstrip('/')
        self.enabled = enabled
        self.cache = {}  # 简单的缓存避免重复调用
        
        # 测试连接
        self._test_connection()
        
    def _test_connection(self):
        """测试 API 连接"""
        try:
            resp = requests.get(f"{self.api_url}/health", timeout=2)
            if resp.status_code == 200:
                print(f"✅ SLM Filter: Connected to {self.api_url}")
            else:
                print(f"⚠️ SLM Filter: API returned {resp.status_code}")
                self.enabled = False
        except Exception as e:
            print(f"⚠️ SLM Filter: Cannot connect to API ({e}), filter disabled")
            self.enabled = False
    
    def is_relevant(self, symbol: str, company_name: str, title: str, content: str, trigger_keywords: str = "") -> bool:
        """
        判断文章是否真正讨论该公司
        
        Args:
            symbol: 股票代码 (如 INTC)
            company_name: 公司名称 (如 Intel)
            title: 文章标题
            content: 文章正文
            trigger_keywords: 触发匹配的关键词上下文
        """
        if not self.enabled:
            return True
        
        # 缓存键
        cache_key = f"{symbol}:{hash(title)}"
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # 构建 prompt
        prompt = self._build_prompt(symbol, company_name, title, content, trigger_keywords)
        
        try:
            # 调用 quant_langchain API
            response = requests.post(
                f"{self.api_url}/api/ask",
                json={"question": prompt},
                timeout=10
            )
            
            if response.status_code == 200:
                answer = response.json().get("answer", "").strip()
                # 更robust的判断：检查答案开头是否是 YES
                result = answer.upper().startswith("YES")
                
                # 缓存结果
                self.cache[cache_key] = result
                return result
            else:
                print(f"⚠️ SLM API error: {response.status_code}")
                return True  # 失败时保守处理
                
        except Exception as e:
            print(f"⚠️ SLM 调用失败: {e}，默认允许通过")
            return True  # 失败时保守处理，允许通过
    
    def _build_prompt(self, symbol: str, company_name: str, title: str, content: str, trigger_keywords: str = "") -> str:
        """构建 prompt"""
        # 简化名称，便于模型理解
        simple_names = {
            "Alphabet Inc.(Class A)": "Google",
            "Meta Platforms": "Facebook",
            "Apple Inc.": "Apple",
            "Microsoft": "Microsoft",
            "Amazon.com Inc.": "Amazon",
            "Tesla, Inc.": "Tesla",
            "NVIDIA Corporation": "NVIDIA"
        }
        display_name = simple_names.get(company_name, company_name)
        
        # 只使用前 500 字符的正文
        content_preview = content[:500] if content else ""
        
        # 增加对登录墙/订阅墙的启发式检测，提示模型
        paywall_hints = ["login", "subscribe", "etprime", "sign in", "exclusive for members", "register to read"]
        is_possibly_paywalled = any(hint in content_preview.lower() for hint in paywall_hints)
        
        paywall_instruction = ""
        if is_possibly_paywalled:
            paywall_instruction = "\nIMPORTANT: The content snippet looks like a paywall or login prompt. Please rely HEAVILY on the Title to make your decision."

        context_str = f"\nContext: Flagged keywords: {trigger_keywords}" if trigger_keywords else ""
        
        prompt = f"""Task: Decide if this news is RELEVANT to "{display_name}" (Ticker: {symbol}).

Article Title: {title}
Article Content Snippet: {content_preview}{paywall_instruction}{context_str}

Positive Criteria (Answer YES):
- Direct company news: Financials, earnings, stock movements, M&A, partnerships.
- Corporate Governance: Board changes, diversity, lawsuits, INVESTOR PRESSURE, layoffs.
- Product news: Launches, reviews, software updates (iOS, Android, Windows, chips, etc.), and NEW FEATURE ANNOUNCEMENTS.
- Legal, Regulatory & Public Policy: Lawsuits, government conflicts, court orders, privacy debates, regulatory hearings, and antitrust actions.
- Public Statements & Stance: Official manifestos, open letters, or public stances taken by the CEO/Founder on major societal, political, or legal issues.
- Sales performance & Consumer Trends: Reports on best-selling items, holiday sales records, or market share changes.
- Brand Sentiment & Market Positioning: Trust surveys, industry rankings, or reputational reports, even when comparing multiple competitors.
- Specific Platform Changes: Even moderate updates to user interface or functionality of major platforms (e.g., "YouTube's new progress bar", "Instagram's new navigation").
- Ecosystem Relevance (especially for ARM): News about major products using {display_name}'s architecture or licensing (e.g. Apple A/M-series chips, Qualcomm Snapdragon, Samsung Exynos, MediaTek Dimensity) is YES.
- Even if the content is short or blocked by a paywall, if the Title is clearly about {display_name}, answer YES.

Negative Criteria (Answer NO):
- Mentioning {display_name} only incidentally (e.g. "Former Apple employees started a new car company"). Note: Industry-wide comparison reports including {display_name} are RELEVANT, not incidental.
- Pure product sales/listings (e.g. "Refurbished Intel Laptop for sale on eBay").
- Generic technical terms (e.g. "A meta-analysis of results" when looking for Meta).
- Celebrity Gossip & Personal Social Media Updates: News about celebrities' personal lives, fashion, dating, or social media posts (e.g., "Kylie Jenner's new photo on Instagram", "Drake follows someone on Instagram").
- Non-business Lifestyle/Entertainment: Routine movie reviews, holiday photos, or personal recipes unless they directly impact the company's business model.

Answer ONLY "YES" or "NO".
Answer:"""
        
        return prompt


# 示例使用
if __name__ == "__main__":
    filter = SLMFilter()
    
    # 测试案例 1: 真实的 Intel 新闻
    result1 = filter.is_relevant(
        "INTC",
        "Intel",
        "Intel launches new 16nm processor",
        "Intel Corporation announced today a new processor..."
    )
    print(f"Test 1 (Intel news): {result1}")  # 应该是 True
    
    # 测试案例 2: Beauty intel
    result2 = filter.is_relevant(
        "INTC",
        "Intel",
        "Beauty intel: MTG Refa, KATE TOKYO",
        "KNEAD THIS NOW. Flaunt a youthful appearance..."
    )
    print(f"Test 2 (Beauty intel): {result2}")  # 应该是 False
    
    # 测试案例 3: Intelligence Committee
    result3 = filter.is_relevant(
        "INTC",
        "Intel",
        "Intel Chairman: Somalia Plane Explosion",
        "The chairman of the Senate Intelligence Committee warned..."
    )
    print(f"Test 3 (Intelligence Committee): {result3}")  # 应该是 False
