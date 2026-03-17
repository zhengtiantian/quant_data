#!/usr/bin/env python3
"""
SLM-based Article Relevance Filter
使用 Qwen 小语言模型进行文章相关性判断（通过 Ollama API）
"""

import hashlib
import os
import threading
from typing import Dict, Optional

import requests

# 适度保守的全局并发锁：默认 2，可通过环境变量进一步调整。
# 对于本地 Mac + 4B 量化模型，2 是一个比较稳妥的折中点。
_OLLAMA_MAX_CONCURRENCY = max(1, int(os.getenv("OLLAMA_MAX_CONCURRENCY", "2")))
_ollama_semaphore = threading.Semaphore(_OLLAMA_MAX_CONCURRENCY)

class SLMFilter:
    """使用 SLM 判断文章是否真正讨论某个公司"""

    def __init__(self, api_url: str = None, model: str = None, enabled: bool = True):
        if api_url is None:
            api_url = os.getenv("OLLAMA_API", "http://localhost:11434")
        if model is None:
            model = os.getenv("OLLAMA_MODEL", "qwen3:4b-q4_K_M")

        self.api_url = api_url.rstrip('/')
        self.model = model
        self.enabled = enabled
        self.cache = {}
        self._cache_lock = threading.Lock()

        self._test_connection()

    def _test_connection(self):
        """测试 Ollama 连接"""
        try:
            resp = requests.get(f"{self.api_url}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m["name"] for m in resp.json().get("models", [])]
                print(
                    f"✅ SLM Filter: Connected to Ollama, "
                    f"concurrency={_OLLAMA_MAX_CONCURRENCY}, models={models}"
                )
            else:
                print(f"⚠️ SLM Filter: Ollama returned {resp.status_code}")
                self.enabled = False
        except Exception as e:
            print(f"⚠️ SLM Filter: Cannot connect to Ollama ({e}), filter disabled")
            self.enabled = False

    def is_relevant(self, symbol: str, company_name: str, title: str, content: str, trigger_keywords: str = "") -> bool:
        """
        判断文章是否真正讨论该公司
        """
        if not self.enabled:
            return True

        title_key = title.strip().lower()
        content_key = content[:300].strip().lower()
        digest = hashlib.sha1(f"{symbol}|{title_key}|{content_key}".encode("utf-8")).hexdigest()
        cache_key = f"{symbol}:{digest}"
        with self._cache_lock:
            cached = self.cache.get(cache_key)
        if cached is not None:
            return cached

        prompt = self._build_prompt(symbol, company_name, title, content, trigger_keywords)

        try:
            # 获取凭证，必须排队，让显卡喘口气
            with _ollama_semaphore:
                response = requests.post(
                    f"{self.api_url}/api/generate",
                    json={
                        "model": self.model,
                        "prompt": prompt,
                        "stream": False,
                        "options": {
                            "num_predict": 3,
                            "temperature": 0.1,
                            "top_p": 0.9,
                        }
                    },
                    timeout=45
                )

            if response.status_code == 200:
                answer = response.json()["response"].strip()
                result = answer.upper().startswith("YES")
                with self._cache_lock:
                    self.cache[cache_key] = result
                return result
            else:
                print(f"⚠️ Ollama API error: {response.status_code}，默认拒绝")
                return False

        except Exception as e:
            print(f"⚠️ Ollama 调用失败/超时: {e}，默认拒绝")
            return False

    def _build_prompt(self, symbol: str, company_name: str, title: str, content: str, trigger_keywords: str = "") -> str:
        """构建 prompt"""
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

        content_preview = content[:500] if content else ""

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
