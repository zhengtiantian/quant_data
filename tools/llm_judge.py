"""
LLM Judge for Rule Optimization Agent (F.9)
Evaluates whether confirmed articles are true or false positives.
Supports Claude API (preferred) and local SLM via LM Studio.
"""

import json
import os
import re
import time
from typing import Optional

import requests

# ── Model routing ──────────────────────────────────────────────────────────
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
ANTHROPIC_MODEL   = os.getenv("JUDGE_CLAUDE_MODEL", "claude-haiku-4-5-20251001")
LOCAL_API_URL     = os.getenv("SLM_API_URL", os.getenv("LMSTUDIO_API", "http://192.168.31.226:1234/v1"))
LOCAL_JUDGE_MODEL = os.getenv("JUDGE_LOCAL_MODEL", os.getenv("SLM_MODEL", "qwen3.5-4b"))

FP_TYPES = [
    "INCIDENTAL_MENTION",  # company is cited as data source, not subject
    "ANALYST_CITE",        # another stock gets rated; this company is the analyst/bank
    "WRONG_COMPANY",       # different entity with similar name
    "MARKET_ROUNDUP",      # company is one item in a stocks/sectors list
    "PRODUCT_PAGE",        # product listing, review, buy-now page
    "PERIPHERY",           # mentioned for context/comparison; article is about something else
]

_JUDGE_PROMPT = """\
You are a financial news relevance judge.

Task: evaluate whether this article is PRIMARILY about {company} ({symbol}).

Title: {title}
Content (first 700 chars): {content_preview}

VERDICT choices:
- TP (True Positive): Article's main subject is {company}'s own business — its earnings, products, \
strategy, executives, legal/regulatory issues, financial results, or direct business actions.
- FP (False Positive): The company appears only incidentally. FP types:
  * INCIDENTAL_MENTION — company is cited as a survey/data source \
(e.g. "Morgan Stanley survey finds iPhone loyalty 92%"; "Goldman Sachs says markets will...")
  * ANALYST_CITE — another company's stock gets a rating from {company} analysts; \
{company} is the bank/analyst, not the subject
  * WRONG_COMPANY — different entity/person/thing with a similar name
  * MARKET_ROUNDUP — {company} is one item in a list of stocks, sectors, or companies
  * PRODUCT_PAGE — product listing, buy-now page, accessory review
  * PERIPHERY — {company} mentioned for context or comparison; article is really about something else

Respond with JSON only (no markdown, no explanation outside the object):
{{"verdict": "TP" or "FP", "fp_type": null or one of {fp_types}, \
"confidence": 0.7-1.0, "reason": "one sentence"}}"""

_PROPOSE_PROMPT = """\
You are a regex pattern engineer for a financial news relevance filter written in Python (re.IGNORECASE applied).

Symbol: {symbol} ({company_name})
FP type being fixed: {fp_type}
False positive examples — each is a title + one-sentence reason it's a FP:
{fp_examples}

Write ONE Python regex pattern that would REJECT these false positive articles.
Requirements:
- Use \\b for word boundaries
- Be specific: avoid matching true-positive headlines about {company}
- Pattern will be compiled with re.IGNORECASE
- Target either STATIC_KILL_PATTERNS (always reject regardless of other hits) \
or CONTEXTUAL_REJECT_PATTERNS (reject unless a keep-pattern fires)
- Rate risk: low = very unlikely to kill valid articles; medium = possible edge cases; \
high = could block valid content

Respond with JSON only:
{{"pattern": "regex_string", \
"target_dict": "STATIC_KILL_PATTERNS" or "CONTEXTUAL_REJECT_PATTERNS", \
"rationale": "why pattern catches FPs without harming TPs", \
"risk": "low" or "medium" or "high"}}"""


class LLMJudge:
    def __init__(self, model: str = "auto"):
        """
        model: "claude" | "local" | "auto"
        auto → Claude if ANTHROPIC_API_KEY is set, else local SLM
        """
        if model == "auto":
            self.backend = "claude" if ANTHROPIC_API_KEY else "local"
        else:
            self.backend = model
        print(f"✅ LLMJudge backend: {self.backend} "
              f"({'claude-haiku' if self.backend == 'claude' else LOCAL_JUDGE_MODEL})")

    # ── Public API ──────────────────────────────────────────────────────────

    def evaluate(self, article: dict, symbol: str, company_name: str) -> Optional[dict]:
        """
        Evaluate one confirmed article.
        Returns: {"verdict": "TP"|"FP", "fp_type": str|None, "confidence": float, "reason": str}
        or None if judge fails.
        """
        title   = (article.get("title") or "").strip()
        content = (article.get("content") or "").strip()
        content_preview = " ".join(content.split())[:700]

        prompt = _JUDGE_PROMPT.format(
            company=company_name,
            symbol=symbol,
            title=title,
            content_preview=content_preview,
            fp_types=str(FP_TYPES),
        )
        raw = self._call(prompt)
        if raw is None:
            return None
        return self._parse_verdict(raw)

    def propose_pattern(
        self,
        fp_examples: list[dict],
        symbol: str,
        company_name: str,
        fp_type: str,
    ) -> Optional[dict]:
        """
        Given FP examples for one (symbol, fp_type), propose a new rejection pattern.
        Returns: {"pattern": str, "target_dict": str, "rationale": str, "risk": str}
        """
        examples_text = "\n".join(
            f"- Title: {e.get('title', '')[:120]}\n  Reason: {e.get('reason', '')}"
            for e in fp_examples[:8]
        )
        prompt = _PROPOSE_PROMPT.format(
            symbol=symbol,
            company_name=company_name,
            fp_type=fp_type,
            fp_examples=examples_text,
        )
        raw = self._call(prompt)
        if raw is None:
            return None
        return self._parse_json(raw)

    # ── Backend dispatch ────────────────────────────────────────────────────

    def _call(self, prompt: str, retries: int = 2) -> Optional[str]:
        for attempt in range(retries + 1):
            try:
                if self.backend == "claude":
                    return self._call_claude(prompt)
                else:
                    return self._call_local(prompt)
            except Exception as e:
                if attempt < retries:
                    time.sleep(2 ** attempt)
                else:
                    print(f"  ⚠️  Judge call failed: {e}")
                    return None
        return None

    def _call_claude(self, prompt: str) -> str:
        resp = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": ANTHROPIC_API_KEY,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json={
                "model": ANTHROPIC_MODEL,
                "max_tokens": 200,
                "messages": [{"role": "user", "content": prompt}],
            },
            timeout=(10, 30),
        )
        resp.raise_for_status()
        return resp.json()["content"][0]["text"]

    def _call_local(self, prompt: str) -> str:
        resp = requests.post(
            f"{LOCAL_API_URL}/completions",
            json={
                "model": LOCAL_JUDGE_MODEL,
                "prompt": f"/no_think\n{prompt}",
                "max_tokens": 120,
                "temperature": 0,
            },
            timeout=(20, 60),
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"]

    # ── Response parsing ────────────────────────────────────────────────────

    def _parse_verdict(self, text: str) -> Optional[dict]:
        result = self._parse_json(text)
        if result and "verdict" in result:
            verdict = result.get("verdict", "").upper()
            if verdict not in ("TP", "FP"):
                return None
            fp_type = result.get("fp_type")
            if fp_type and fp_type.upper() not in FP_TYPES:
                fp_type = "PERIPHERY"
            return {
                "verdict": verdict,
                "fp_type": fp_type.upper() if fp_type else None,
                "confidence": float(result.get("confidence", 0.7)),
                "reason": str(result.get("reason", "")),
            }
        # Fallback: scan raw text for TP/FP keywords
        upper = text.upper()
        if '"VERDICT": "TP"' in upper or '"TP"' in upper[:40]:
            return {"verdict": "TP", "fp_type": None, "confidence": 0.6, "reason": "(fallback parse)"}
        if '"VERDICT": "FP"' in upper or '"FP"' in upper[:40]:
            fp_type = next((t for t in FP_TYPES if t in upper), "PERIPHERY")
            return {"verdict": "FP", "fp_type": fp_type, "confidence": 0.6, "reason": "(fallback parse)"}
        return None

    @staticmethod
    def _parse_json(text: str) -> Optional[dict]:
        text = text.strip()
        # Strip markdown code fences if present
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
        # Try whole text first
        try:
            return json.loads(text)
        except Exception:
            pass
        # Try to extract a JSON object
        match = re.search(r"\{[\s\S]+\}", text)
        if match:
            try:
                return json.loads(match.group())
            except Exception:
                pass
        return None
