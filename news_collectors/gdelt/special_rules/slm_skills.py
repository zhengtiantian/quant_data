#!/usr/bin/env python3
"""Reusable SLM prompt skills for article evaluation tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SkillContext:
    symbol: str
    company_name: str
    title: str
    content: str
    trigger_keywords: str = ""


class BaseSLMSkill:
    """Base interface for reusable SLM prompt skills."""

    name: str = "base"

    def build_prompt(self, context: SkillContext) -> str:
        raise NotImplementedError


class CompanyMatchSkill(BaseSLMSkill):
    """Binary company/article relevance classification skill."""

    name = "company_match_v1"

    SIMPLE_NAMES = {
        "Alphabet Inc.(Class A)": "Google",
        "Meta Platforms": "Facebook",
        "Apple Inc.": "Apple",
        "Microsoft": "Microsoft",
        "Amazon.com Inc.": "Amazon",
        "Tesla, Inc.": "Tesla",
        "NVIDIA Corporation": "NVIDIA",
    }

    def build_prompt(self, context: SkillContext) -> str:
        display_name = self.SIMPLE_NAMES.get(context.company_name, context.company_name)
        content_preview = " ".join((context.content or "").split())[:320]
        trigger_line = f"Trigger Keywords: {context.trigger_keywords}\n" if context.trigger_keywords else ""

        return (
            "Binary classification task.\n"
            f"Company: {display_name} ({context.symbol})\n"
            f"Title: {context.title}\n"
            f"Body: {content_preview}\n"
            f"{trigger_line}"
            "Answer YES only if the article is primarily about the company, its business, products, policies, earnings, major product updates, or company-controlled platforms.\n"
            "Answer NO if the company/platform is only incidental, if this is a product listing/accessory page, a generic post on a platform, or unrelated content.\n"
            "Output exactly one token: YES or NO.\n"
            "Answer:"
        )


SKILL_REGISTRY: Dict[str, BaseSLMSkill] = {
    CompanyMatchSkill.name: CompanyMatchSkill(),
}


def get_skill(name: str) -> BaseSLMSkill:
    skill = SKILL_REGISTRY.get(name)
    if skill is None:
        raise KeyError(f"Unknown SLM skill: {name}")
    return skill
