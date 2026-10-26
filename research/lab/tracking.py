"""MLflow for the lab: experiment `earnings_pulse` on the platform's MLflow (127.0.0.1:15050)."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

EXPERIMENT = os.getenv("LAB_EXPERIMENT", "earnings_pulse")
URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:15050")


def _mlflow():
    import mlflow
    mlflow.set_tracking_uri(URI)
    mlflow.set_experiment(EXPERIMENT)
    return mlflow


def flat(prefix: str, d: Dict[str, Any]) -> Dict[str, float]:
    return {f"{prefix}_{k}": float(v) for k, v in d.items() if isinstance(v, (int, float)) and v == v}


def log_run(name: str, params: Dict[str, Any], verdict: Dict[str, Any], tags: Optional[Dict[str, str]] = None,
            parent_id: Optional[str] = None, artifacts: Optional[Dict[str, Any]] = None, extra: Optional[Dict[str, float]] = None) -> str:
    mlflow = _mlflow()
    with mlflow.start_run(run_name=name, nested=parent_id is not None,
                          parent_run_id=parent_id, tags={**(tags or {}), **({"mlflow.parentRunId": parent_id} if parent_id else {})}) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        m = {}
        for part in ("all", "select", "report", "amc", "bmo"):
            m.update(flat(part, verdict.get(part, {})))
        m.update({f"year_{y}": v for y, v in verdict.get("years", {}).items()})
        m.update(extra or {})
        mlflow.log_metrics(m)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "verdict.json"
            p.write_text(json.dumps(verdict, indent=1, default=float))
            mlflow.log_artifact(str(p))
            for fname, obj in (artifacts or {}).items():
                f = Path(d) / fname
                obj.to_parquet(f) if hasattr(obj, "to_parquet") else f.write_text(str(obj))
                mlflow.log_artifact(str(f))
        return run.info.run_id


def parent(name: str, params: Dict[str, Any], tags: Optional[Dict[str, str]] = None) -> str:
    mlflow = _mlflow()
    with mlflow.start_run(run_name=name, tags=tags or {}) as run:
        mlflow.log_params({k: str(v) for k, v in params.items()})
        return run.info.run_id


def log_parent_metrics(run_id: str, metrics: Dict[str, float]) -> None:
    mlflow = _mlflow()
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)
