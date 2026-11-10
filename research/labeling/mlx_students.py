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
    "relevance": {"dir": "relevance_q17_20k", "base": "Qwen3-1.7B-bf16", "chat": "qwen", "max_tokens": 2, "batch": 32},
    "label_a": {"dir": "label_a_e2b_50k", "base": "gemma-4-e2b-it-bf16", "chat": "gemma", "max_tokens": 16, "batch": 16},
    "label_b": {"dir": "label_b_q17_50k", "base": "Qwen3-1.7B-bf16", "chat": "qwen", "max_tokens": 16, "batch": 16},
}

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
