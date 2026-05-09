#!/usr/bin/env python3
"""
SLM-based Article Relevance Filter
使用本地小语言模型进行文章相关性判断（默认通过 LM Studio OpenAI 兼容 API）
"""

import hashlib
import os
import re
import threading
import time
from collections import defaultdict
from typing import Dict, List, Optional

import requests

from .slm_skills import SkillContext, get_skill

# 适度保守的全局并发锁：默认 2，可通过环境变量进一步调整。
_SLM_MAX_CONCURRENCY = max(1, int(os.getenv("SLM_MAX_CONCURRENCY", os.getenv("OLLAMA_MAX_CONCURRENCY", "2"))))
_slm_semaphore = threading.Semaphore(_SLM_MAX_CONCURRENCY)
_slm_stats_lock = threading.Lock()
_slm_stats = defaultdict(lambda: defaultdict(int))

# 多端点加权轮询
# SLM_ENDPOINTS="url|model|weight,..." 按权重分发到多个推理服务
# 例：SLM_ENDPOINTS="http://127.0.0.1:1234/v1|qwen3.5-4b|1,http://192.168.31.226:1234/v1|qwen3.5-4b|3,http://192.168.31.226:1234/v1|qwen3.5-4b:2|3"
# 若未设置，退回到 SLM_MODELS 单 URL 模式
def _build_endpoint_pool(default_url: str, default_model: str) -> List[tuple]:
    raw = os.getenv("SLM_ENDPOINTS", "")
    if raw:
        pool = []
        for entry in raw.split(","):
            parts = entry.strip().split("|")
            if len(parts) == 3:
                url, model, weight = parts[0].rstrip("/"), parts[1], int(parts[2])
            elif len(parts) == 2:
                url, model, weight = parts[0].rstrip("/"), parts[1], 1
            else:
                continue
            pool.extend([(url, model)] * weight)
        if pool:
            return pool
    # 退回单 URL + SLM_MODELS 轮询
    models_raw = os.getenv("SLM_MODELS", "")
    models = [m.strip() for m in models_raw.split(",") if m.strip()] if models_raw else [default_model]
    return [(default_url, m) for m in models]

_endpoint_pool_lock = threading.Lock()

def _next_endpoint(pool: List[tuple], counter_holder: list) -> tuple:
    with _endpoint_pool_lock:
        idx = counter_holder[0] % len(pool)
        counter_holder[0] += 1
    return pool[idx]

# 兼容旧接口
def _build_model_pool(default_model: str) -> List[str]:
    raw = os.getenv("SLM_MODELS", "")
    if raw:
        return [m.strip() for m in raw.split(",") if m.strip()]
    return [default_model]


def _thread_name(thread_name: Optional[str] = None) -> str:
    thread_name = thread_name or threading.current_thread().name
    return thread_name


def _owner_name(thread_name: Optional[str] = None) -> str:
    thread_name = _thread_name(thread_name)
    return thread_name.split("-scan_")[0] if "-scan_" in thread_name else thread_name


def track_slm_stat(metric: str, amount: int = 1, owner: Optional[str] = None) -> None:
    thread_name = _thread_name(owner)
    owner_name = _owner_name(thread_name)
    with _slm_stats_lock:
        _slm_stats[thread_name][metric] += amount
        if owner_name != thread_name:
            _slm_stats[owner_name][metric] += amount


def get_slm_stats(owner: Optional[str] = None) -> Dict[str, int]:
    current_owner = owner or _thread_name()
    with _slm_stats_lock:
        stats = dict(_slm_stats.get(current_owner, {}))
    return {
        "requests": stats.get("requests", 0),
        "cache_hits": stats.get("cache_hits", 0),
        "yes": stats.get("yes", 0),
        "no": stats.get("no", 0),
        "errors": stats.get("errors", 0),
        "elapsed_ms": stats.get("elapsed_ms", 0),
    }


def format_slm_stats(owner: Optional[str] = None) -> str:
    current_name = owner or _thread_name()
    stats = get_slm_stats(current_name)
    if not any(stats.values()):
        fallback_owner = _owner_name(current_name)
        if fallback_owner != current_name:
            stats = get_slm_stats(fallback_owner)
    return (
        f"slm(req={stats['requests']}, cache={stats['cache_hits']}, "
        f"yes={stats['yes']}, no={stats['no']}, err={stats['errors']})"
    )

class SLMFilter:
    """使用 SLM 判断文章是否真正讨论某个公司"""

    def __init__(self, api_url: str = None, model: str = None, enabled: bool = True):
        self.provider = os.getenv("SLM_PROVIDER", "lmstudio").lower()
        self.api_mode = os.getenv("SLM_API_MODE", "openai").lower()
        if api_url is None:
            if self.provider == "lmstudio":
                api_url = os.getenv("SLM_API_URL", os.getenv("LMSTUDIO_API", "http://127.0.0.1:1234/v1"))
            else:
                api_url = os.getenv("SLM_API_URL", os.getenv("OLLAMA_API", "http://127.0.0.1:11434"))
        if model is None:
            if self.provider == "lmstudio":
                model = os.getenv("SLM_MODEL", os.getenv("LMSTUDIO_MODEL", "qwen3.5-4b"))
            else:
                model = os.getenv("SLM_MODEL", os.getenv("OLLAMA_MODEL", "qwen3:4b-q4_K_M"))

        self.api_url = api_url.rstrip('/')
        self.model = model
        self.model_pool = _build_model_pool(model)
        self._endpoint_pool = _build_endpoint_pool(self.api_url, model)
        self._model_counter = [0]  # round-robin counter
        self.enabled = enabled
        self.skill_name = os.getenv("SLM_SKILL", "company_match_v1")
        self.skill = get_skill(self.skill_name)
        self.cache = {}
        self._cache_lock = threading.Lock()
        self._test_connection()

    def _test_connection(self):
        """测试所有唯一端点连接，仅打印结果，不阻塞也不 disable filter"""
        unique_urls = dict.fromkeys(url for url, _ in self._endpoint_pool)
        for base_url in unique_urls:
            url = f"{base_url}/models" if self.provider == "lmstudio" else f"{base_url}/api/tags"
            try:
                resp = requests.get(url, timeout=10)
                if resp.status_code == 200:
                    if self.provider == "lmstudio":
                        models = [m["id"] for m in resp.json().get("data", [])]
                        provider_name = "LM Studio"
                    else:
                        models = [m["name"] for m in resp.json().get("models", [])]
                        provider_name = "Ollama"
                    print(
                        f"✅ SLM Filter: {base_url} ({provider_name}), "
                        f"concurrency={_SLM_MAX_CONCURRENCY}, models={models}"
                    )
                else:
                    print(f"⚠️ SLM Filter: {base_url} returned {resp.status_code}, will retry on first request")
            except Exception as e:
                print(f"⚠️ SLM Filter: Cannot connect to {base_url} ({e}), will retry on first request")

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
            track_slm_stat("cache_hits")
            return cached

        prompt = self.skill.build_prompt(
            SkillContext(
                symbol=symbol,
                company_name=company_name,
                title=title,
                content=content,
                trigger_keywords=trigger_keywords,
            )
        )

        try:
            _t_slm = time.time()
            with _slm_semaphore:
                track_slm_stat("requests")
                api_url, model = _next_endpoint(self._endpoint_pool, self._model_counter)

                def _do_request():
                    if self.provider == "lmstudio":
                        if self.api_mode == "lmstudio_rest":
                            return requests.post(
                                f"{api_url}/chat",
                                json={
                                    "model": model,
                                    "messages": [{"role": "user", "content": f"/no_think\n{prompt}"}],
                                    "max_tokens": 4,
                                    "temperature": 0,
                                },
                                timeout=(20, 60),
                            )
                        else:
                            return requests.post(
                                f"{api_url}/completions",
                                json={
                                    "model": model,
                                    "prompt": prompt,
                                    "max_tokens": 4,
                                    "temperature": 0,
                                },
                                timeout=(20, 60),
                            )
                    else:
                        return requests.post(
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
                            timeout=(20, 60),
                        )

                try:
                    response = _do_request()
                except requests.exceptions.ConnectTimeout:
                    time.sleep(2)
                    response = _do_request()

            track_slm_stat("elapsed_ms", int((time.time() - _t_slm) * 1000))
            if response.status_code == 200:
                answer = self._extract_answer(response.json())
                if answer not in {"YES", "NO"}:
                    track_slm_stat("errors")
                    with self._cache_lock:
                        self.cache[cache_key] = False
                    return False
                result = answer == "YES"
                track_slm_stat("yes" if result else "no")
                with self._cache_lock:
                    self.cache[cache_key] = result
                return result
            else:
                track_slm_stat("errors")
                try:
                    err_body = response.text[:300]
                except Exception:
                    err_body = "(unreadable)"
                print(f"⚠️ {self.provider} API error: {response.status_code} — {err_body}")
                return False

        except Exception as e:
            track_slm_stat("elapsed_ms", int((time.time() - _t_slm) * 1000))
            track_slm_stat("errors")
            print(f"⚠️ {self.provider} 调用失败/超时: {e}，默认拒绝")
            return False

    def _extract_answer(self, data: Dict) -> str:
        """统一提取 YES/NO 答案。"""
        def normalize(candidate: str) -> str:
            text = (candidate or "").strip()
            if not text:
                return ""
            upper = text.upper()
            if upper in {"YES", "NO"}:
                return upper
            match = re.search(r"\b(YES|NO)\b", upper)
            return match.group(1) if match else ""

        if self.provider == "lmstudio":
            if self.api_mode == "lmstudio_rest":
                choices = data.get("choices", [])
                if choices:
                    message = choices[0].get("message", {})
                    answer = normalize(message.get("content") or "")
                    if answer:
                        return answer
                prediction = data.get("prediction") or {}
                candidate = prediction.get("content") or prediction.get("text") or data.get("text") or ""
                return normalize(str(candidate))
            else:
                choices = data.get("choices", [])
                if not choices:
                    return ""
                # /completions returns text; /chat/completions returns message.content
                text = choices[0].get("text")
                if text is not None:
                    return normalize(text)
                message = choices[0].get("message", {})
                answer = normalize(message.get("content") or "")
                if answer:
                    return answer
                return normalize(message.get("reasoning_content") or "")
        return normalize(data.get("response") or "")

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
