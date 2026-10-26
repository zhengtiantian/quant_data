#!/usr/bin/env python3
"""Lab entry point (PROJECT_PLAN Q.6).

    python research/lab/zoo.py judge --model lightgbm --preds bakeoff_preds
    python research/lab/zoo.py judge --model cnn --preds cnn_preds
    python research/lab/zoo.py judge --model avg_lgbm_cnn --preds bakeoff_preds,cnn_preds

judge: every policy variant of the grid on the model's out-of-sample predictions, one MLflow child
run each, under a parent run for the model; the variant chosen on 2019-2022 by the fixed rule
(highest total within the risk limits) is tagged `chosen`, and its deflated Sharpe ratio is computed
against every trial logged in the experiment so far, all models included.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import evaluate as ev  # noqa: E402
import tracking as tr  # noqa: E402

KEYS = ["eid", "place", "level", "buyback"]


def preds_for(spec: str) -> pd.DataFrame:
    parts = [ev.load_preds(p) for p in spec.split(",")]
    if len(parts) == 1:
        return parts[0]
    base = parts[0][KEYS + ["symbol", "year", "gain", "pred", "q10"]]
    for i, p in enumerate(parts[1:], 1):
        base = base.merge(p[KEYS + ["pred", "q10"]], on=KEYS, suffixes=("", f"_{i}"))
    pc = [c for c in base if c.startswith("pred")]
    qc = [c for c in base if c.startswith("q10")]
    return base.assign(pred=base[pc].mean(1), q10=base[qc].mean(1))[KEYS + ["symbol", "year", "gain", "pred", "q10"]]


def all_trial_sharpes() -> list:
    import mlflow
    mlflow.set_tracking_uri(tr.URI)
    runs = mlflow.search_runs(experiment_names=[tr.EXPERIMENT], filter_string="tags.kind = 'variant'")
    return [v for v in runs.get("metrics.all_sharpe_trade", pd.Series(dtype=float)).dropna().tolist()]


def judge(model: str, spec: str, note: str = "") -> None:
    preds = preds_for(spec)
    sess = ev.sessions()
    pid = tr.parent(model, {"preds": spec, "rows": len(preds), "note": note}, tags={"kind": "model", "model": model})
    best, rows = None, []
    for i, v in enumerate(ev.variants(), 1):
        r = ev.apply_policy(preds, **v)
        vd = ev.verdict(r, sess)
        sel = vd["select"]
        ok = ev.passes(sel)
        rid = tr.log_run(f"{model} {v}", {**v, "model": model}, vd, tags={"kind": "variant", "model": model,
                         "passes_risk": str(ok)}, parent_id=pid)
        rows.append({**v, "run_id": rid, "passes": ok, **{f"select_{k}": x for k, x in sel.items()},
                     **{f"report_{k}": x for k, x in vd["report"].items()}})
        if ok and (best is None or sel["sum"] > best[1]["select"]["sum"]):
            best = (v, vd, rid, r)
        print(f"PROGRESS judge {i} {len(ev.variants())} {model}", flush=True)
    table = pd.DataFrame(rows)
    if best is None:
        tr.log_parent_metrics(pid, {"variants": len(rows), "variants_passing": 0})
        print(f"{model}: no variant passes the 2019-{ev.SELECT_UNTIL} risk limits")
        return
    v, vd, rid, r = best
    dsr = ev.deflated_sharpe(r.loc[r["acted"], "gain"].values, all_trial_sharpes())
    import mlflow
    mlflow.set_tracking_uri(tr.URI)
    mlflow.MlflowClient().set_tag(rid, "chosen", "true")
    tr.log_parent_metrics(pid, {"variants": len(rows), "variants_passing": int(table["passes"].sum()),
                                **{f"chosen_{k}": x for k, x in vd["report"].items() if isinstance(x, (int, float))},
                                **{f"chosen_dsr_{k}": x for k, x in dsr.items() if isinstance(x, (int, float))}})
    print(f"{model}: chosen {v} | 2019-{ev.SELECT_UNTIL} {vd['select']['acted']} acted {vd['select']['mean']:+.2f}% | "
          f"after {vd['report']['acted']} acted {vd['report']['mean']:+.2f}% p10 {vd['report']['p10']:+.2f} worst {vd['report']['worst']:+.1f} | "
          f"all {vd['all']['acted']} {vd['all']['mean']:+.2f}% | DSR {dsr['dsr']:.3f} (SR/trade {dsr['sharpe_trade']:.3f} vs bar {dsr['sr0']:.3f}, "
          f"{dsr['trials']} trials)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["judge"])
    ap.add_argument("--model", required=True)
    ap.add_argument("--preds", required=True)
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    judge(a.model, a.preds, a.note)
    return 0


if __name__ == "__main__":
    sys.exit(main())
