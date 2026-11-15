"""The distilled MLX students that replace the LM Studio teachers in production (2026-11-10).

    relevance  Qwen3-1.7B + LoRA   "is this article materially about this company?"  -> YES / NO
               teacher qwen3.5-9b with the plan-B prompt; 20,000 labels; 92.5% agreement with it,
               ~21 articles/s (rule_check/relevance_2026-11-10/report_rel.md)
    label_a    Gemma-4-E2B + LoRA  pass A labels  -> event|sentiment|strength   (teacher Gemma-4-E4B)
    label_b    Qwen3-1.7B + LoRA   pass B labels  -> event|sentiment|strength   (teacher qwen3.5-9b)
               both on the balanced 50,000 set (distill_fulltext_2026-11-06/final50k_2026-11-09)

The students see a compact input with no instructions (the behaviour is learnt), so the input text
built here must stay exactly what they were trained on. They run in this process with mlx_lm, in
batches; the script must run under tools/lm_session.py `local`, which holds the same lock as the
LM Studio tasks so the GPU has one user at a time. Needs the Python with mlx_lm (STUDENT_PYTHON in
airflow/dags/_host_common.py); the adapters live outside the repository in QUANT_STUDENTS_DIR.
"""
from __future__ import annotations

import os
import re
from pathlib import Path

STUDENTS_DIR = Path(os.getenv("QUANT_STUDENTS_DIR", "/Volumes/Data4T/models/students"))
MODELS_DIR = Path(os.getenv("MLX_MODELS_DIR", str(Path.home() / ".lmstudio" / "models" / "mlx-community")))

SPECS = {
    # relevance_q17_v4_22k (2026-11-10 evening): taught by the 9B teacher with prompt v4 (a single flight or
    # store incident is not about the company; a lawsuit, a fine, a large disruption is) on 22,660 rows.
    # Against the blind gold 82.3% where relevance_q17_20k (prompt v2, the first student) had 78.2%, wrong YES
    # 70 -> 58 and wrong NO 31 -> 24, same speed. The earlier adapter stays in its directory.
    "relevance": {"dir": "relevance_q17_v4_22k", "base": "Qwen3-1.7B-bf16", "chat": "qwen", "max_tokens": 2, "batch": 32},
    "label_a": {"dir": "label_a_e2b_50k", "base": "gemma-4-e2b-it-bf16", "chat": "gemma", "max_tokens": 16, "batch": 16},
    "label_b": {"dir": "label_b_q17_50k", "base": "Qwen3-1.7B-bf16", "chat": "qwen", "max_tokens": 16, "batch": 16},
    # GD1 students (2026-11-13, N.9): taught by Gemini Decision 1 on 96,816 articles, trained on Colab G4.
    # Relevance: blind gold 86.1% (current relevance student 80.8%). Labels: blind accuracy ~85%.
    # Shadow only for now: student_gd1_shadow.py writes them to the student_gd1 field, nothing reads it yet.
    "gd1_relevance": {"dir": "gd1_rel_q17", "base": "Qwen3-1.7B-bf16", "chat": "qwen", "max_tokens": 2, "batch": 32},
    "gd1_label": {"dir": "gd1_label_q17", "base": "Qwen3-1.7B-bf16", "chat": "qwen", "max_tokens": 16, "batch": 16},
    # N.6 label student (2026-11-15, Colab G9): GD1 v2 sentiment / strength plus GD1 v3 event (with leadership),
    # 14 themes and the effect on up to five other covered companies, in one answer (label_v3_input / parse_label_v3).
    "gd1_label_v3": {"dir": "gd1_label_v3_q17", "base": "Qwen3-1.7B-bf16", "chat": "qwen", "max_tokens": 64, "batch": 16},
}

# --- N.6 v3 labels (2026-11-15) ------------------------------------------------------------------------------
# The teacher (Colab g8/label_v3.py) was asked about up to five other covered companies: those the title or
# the first 500 characters name (key_people.json aliases), then the reporter's top peers (research/rl
# build_peers.py, PEERS_PARQUET). The student sees the same list, so it is built here the same way.
V3_THEMES = ("ai_capex", "ai_policy", "memory_cycle", "export_controls", "tariffs_trade", "rates_inflation",
             "elections_policy", "geopolitics_war", "energy_oil", "antitrust", "supply_chain", "consumer_demand",
             "health_policy", "crypto")
V3_MAX_OTHERS, V3_MAX_PEERS = 5, 4
PEERS_PARQUET = Path(os.getenv("PEERS_PARQUET", "/Volumes/Data4T/earnings_policy/rl/peers.parquet"))
_v3: dict = {}


def _v3_tables() -> dict:
    if not _v3:
        import json
        kp = json.loads(Path(__file__).with_name("key_people.json").read_text())
        names = {s: e["aliases"] for s, e in kp.items() if not s.startswith("_")}
        _v3["names"] = names
        _v3["re"] = {s: re.compile(r"(?<![\w-])(?:" + "|".join(re.escape(a) for a in al) + r")(?![\w-])") for s, al in names.items()}
        peers: dict[str, list[str]] = {}
        if PEERS_PARQUET.exists():
            import pandas as pd
            p = pd.read_parquet(PEERS_PARQUET)
            for rep, g in p.groupby("reporter"):
                peers[rep] = list(g.sort_values("score", ascending=False).peer[:V3_MAX_PEERS])
        _v3["peers"] = peers
    return _v3


def v3_others(symbol: str, title: str, content: str) -> list[str]:
    """The other companies the v3 student is asked about, in the teacher's order."""
    t = _v3_tables()
    text = f"{title or ''} {(content or '')[:LABEL_CONTENT_CHARS]}"
    out: list[str] = []
    for s in [s for s, p in t["re"].items() if s != symbol and p.search(text)] + t["peers"].get(symbol, []):
        if s not in out and s != symbol:
            out.append(s)
    return out[:V3_MAX_OTHERS]


def label_v3_input(symbol: str, name: str, title: str, content: str, others: list[str]) -> str:
    """As colab_students/g9/build_v3.user: title and the first 500 characters as stored (not flattened)."""
    u = f"{symbol} ({name})\nTitle: {title or ''}\nContent: {(content or '')[:LABEL_CONTENT_CHARS]}"
    if others:
        names = _v3_tables()["names"]
        u += "\nOthers: " + ", ".join(f"{c} ({names[c][0]})" for c in others)
    return u


_LABEL_V3_RE = re.compile(r"([A-Za-z]+)\|(-?[0-9.]+)\|([a-z]+)\|([a-z_,\-]*)\|([0-9+\-,]*)")


def parse_label_v3(text: str, others: list[str]) -> dict | None:
    """{event_type, sentiment, signal_strength, themes, impacts {symbol: -2..2}} as written, or None.
    Unknown themes are dropped; impacts are kept only when there is one per listed company."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = _LABEL_V3_RE.search(t)
    if not m:
        return None
    try:
        sentiment = float(m.group(2))
        levels = [] if m.group(5) in ("", "-") else [int(x) for x in m.group(5).split(",")]
    except ValueError:
        return None
    themes = [] if m.group(4) in ("", "-") else [x for x in m.group(4).split(",") if x in V3_THEMES]
    impacts = {s: max(-2, min(2, v)) for s, v in zip(others, levels)} if len(levels) == len(others) else None
    return {"event_type": m.group(1), "sentiment": sentiment, "signal_strength": m.group(3), "themes": themes, "impacts": impacts}

REL_TITLE_CHARS, REL_BODY_CHARS = 200, 600
LABEL_TITLE_CHARS, LABEL_CONTENT_CHARS = 200, 500


def _flat(text: str | None) -> str:
    return " ".join((text or "").split())


def relevance_input(symbol: str, name: str, title: str, content: str) -> str:
    """As relevance_2026-11-10/student_io.user: the title stripped, the body flattened."""
    return (f"Company: {name} ({symbol})\nTitle: {(title or '').strip()[:REL_TITLE_CHARS]}\n"
            f"Body: {_flat(content)[:REL_BODY_CHARS]}")


def parse_relevance(text: str) -> bool | None:
    t = (text or "").strip().upper()
    return True if t.startswith("YES") else False if t.startswith("NO") else None


def label_input(symbol: str, name: str, title: str, content: str) -> str:
    """As final50k_2026-11-09/build_final.py: title and content flattened, 200 / 500 characters."""
    return f"{symbol} ({name})\nTitle: {_flat(title)[:LABEL_TITLE_CHARS]}\nContent: {_flat(content)[:LABEL_CONTENT_CHARS]}"


_LABEL_RE = re.compile(r"([A-Za-z]+)\|(-?[0-9.]+)\|([a-z]+)")


def parse_label(text: str) -> tuple[str, float, str] | None:
    """(event_type as written, sentiment, signal_strength), before the caller's normalisation."""
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL).strip()
    m = _LABEL_RE.search(t)
    if not m:
        return None
    try:
        return m.group(1), float(m.group(2)), m.group(3)
    except ValueError:
        return None


class Student:
    def __init__(self, name: str):
        from mlx_lm import load                      # imported here: only the student tasks have mlx
        spec = SPECS[name]
        self.name, self.spec = name, spec
        self.adapter = STUDENTS_DIR / spec["dir"]
        if not (self.adapter / "adapters.safetensors").exists():
            raise FileNotFoundError(f"student adapter missing: {self.adapter}")
        self.model, self.tok = load(str(MODELS_DIR / spec["base"]), adapter_path=str(self.adapter))
        self.id = f"student:{spec['dir']}"
        self.batch = int(os.getenv("STUDENT_BATCH", "0")) or spec["batch"]

    def _ids(self, user: str) -> list[int]:
        if self.spec["chat"] == "gemma":
            return self.tok.encode(f"<bos><|turn>user\n{user}<turn|>\n<|turn>model\n", add_special_tokens=False)
        return self.tok.encode(f"<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")

    def generate(self, users: list[str]) -> list[str]:
        """One answer per input, in input order. Inputs are batched by length (padding is waste)."""
        from mlx_lm import batch_generate
        ids = [self._ids(u) for u in users]
        order = sorted(range(len(ids)), key=lambda i: len(ids[i]))
        out = [""] * len(ids)
        for i in range(0, len(order), self.batch):
            chunk = order[i:i + self.batch]
            texts = batch_generate(self.model, self.tok, [ids[j] for j in chunk],
                                   max_tokens=self.spec["max_tokens"], verbose=False).texts
            for j, t in zip(chunk, texts):
                out[j] = t
        return out
