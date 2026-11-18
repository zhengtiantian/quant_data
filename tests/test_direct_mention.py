"""The v2 direct-mention gate: keywords (any case) or the short company name as written."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "labeling"))

import slm_company_match_v2 as m2  # noqa: E402


def test_short_name_strips_legal_suffixes():
    assert m2._short_name("Intel Corporation") == "Intel"
    assert m2._short_name("KLA Corporation") == "KLA"
    assert m2._short_name("Meta Platforms") == "Meta"
    assert m2._short_name("Snowflake Inc.") == "Snowflake"
    assert m2._short_name("SK hynix") == "SK hynix"


def test_bare_name_counts_as_a_mention():
    assert m2._has_direct_mention("INTC", "Intel Advises Customers Not To Download Spectre Patch", "")
    assert m2._has_direct_mention("ORCL", "Oracle shares jump on cloud backlog", "")
    assert m2._has_direct_mention("INTC", "Chip stocks slide", "Shares of Intel fell 3% on Tuesday.")


def test_name_is_case_sensitive_but_keywords_are_not():
    assert not m2._has_direct_mention("INTC", "Military intel suggests a new offensive", "")
    assert m2._has_direct_mention("INTC", "INTEL CORPORATION files 10-Q", "")


def test_unnamed_article_still_fails():
    assert not m2._has_direct_mention("INTC", "Chip stocks slide on export rules", "Semiconductor shares fell broadly.")
