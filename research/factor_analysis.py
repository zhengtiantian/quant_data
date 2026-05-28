#!/usr/bin/env python3
"""C.9 Factor Analysis Report: IC Decay, SHAP Feature Importance, Long-short Portfolio."""

from __future__ import annotations

import argparse
import math
import os
import sys

import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning

warnings.filterwarnings("ignore", category=ConstantInputWarning)
from dotenv import load_dotenv
from pathlib import Path

CURRENT = Path(__file__).resolve()
ROOT = CURRENT.parents[1]
load_dotenv(ROOT / ".env", override=False)

sys.path.insert(0, str(CURRENT.parent))
from backtest_news_factor import load_feature_frame

HORIZONS = [5, 10, 15, 20, 30, 45, 60]

SINGLE_FACTORS = [
    "avg_sentiment_5d",
    "avg_sentiment_3d",
    "sentiment_shift_5d",
    "high_signal_count_3d",
    "negative_event_count_5d",
    "quality_score",
    "full_ratio",
    "news_burst_20d",
    "article_count",
    "past_ret_20d",
    "volatility_20d",
    "surprise_pct_last",
    "earnings_recency_weight",
    "disagreement_avg_5d",
]

MODEL_FEATURES = [
    "full_ratio", "quality_score", "article_count", "news_burst_20d",
    "past_ret_20d", "past_ret_60d", "volatility_20d", "volatility_60d", "volume_shock_20d",
    "days_to_earnings", "days_since_earnings", "is_earnings_window_5d",
    "is_post_earnings_window_20d", "days_to_earnings_bucket", "days_since_earnings_bucket",
    "is_pre_earnings_10d", "is_post_earnings_5d", "is_post_earnings_10d",
    "eps_estimate_last", "reported_eps_last", "surprise_pct_last",
    "is_positive_surprise", "is_negative_surprise",
    "is_post_positive_surprise_20d", "is_post_negative_surprise_20d",
    "surprise_bucket",
    "full_ratio_x_post_positive", "quality_score_x_post_negative",
    "earnings_recency_weight",
    "avg_sentiment_3d", "avg_sentiment_5d", "sentiment_shift_5d",
    "high_signal_count_3d", "has_regulatory_risk_5d",
    "earnings_beat_signal", "earnings_miss_signal",
    "disagreement_avg_5d", "negative_event_count_5d",
    "full_ratio_sector_rel", "quality_score_sector_rel", "news_burst_20d_sector_rel",
]


# ─── helpers ──────────────────────────────────────────────────────────────────

def rank_ic(series_a: pd.Series, series_b: pd.Series) -> float:
    valid = pd.concat([series_a, series_b], axis=1).dropna()
    if len(valid) < 5:
        return math.nan
    ic, _ = spearmanr(valid.iloc[:, 0], valid.iloc[:, 1])
    return float(ic)


def add_sector_relative(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["full_ratio", "quality_score", "news_burst_20d"]:
        if col in df.columns:
            df[f"{col}_sector_rel"] = (
                df.groupby(["trade_date", "sector"])[col]
                .transform(lambda x: pd.to_numeric(x, errors="coerce").rank(pct=True) - 0.5)
            )
    return df


# ─── Part 1: IC Decay ─────────────────────────────────────────────────────────

def ic_decay_table(df: pd.DataFrame, factors: list[str]) -> None:
    print("\n" + "=" * 80)
    print("PART 1: IC Decay — Single Factor vs. Return Horizons")
    print("=" * 80)

    available = [f for f in factors if f in df.columns]
    ret_cols = {h: f"excess_ret_{h}d" for h in HORIZONS if f"excess_ret_{h}d" in df.columns}
    if not ret_cols:
        print("No return columns found.")
        return

    header = f"{'factor':<32}" + "".join(f"{h:>8}d" for h in ret_cols)
    print(header)
    print("-" * len(header))

    rows = []
    for factor in available:
        ics = {}
        for h, ret_col in ret_cols.items():
            daily_ics = []
            for _, grp in df.groupby("trade_date"):
                ic = rank_ic(grp[factor], grp[ret_col])
                if not math.isnan(ic):
                    daily_ics.append(ic)
            ics[h] = float(np.mean(daily_ics)) if daily_ics else math.nan
        rows.append((factor, ics))
        line = f"{factor:<32}" + "".join(
            f"{ics[h]:>9.4f}" if not math.isnan(ics[h]) else f"{'n/a':>9}"
            for h in ret_cols
        )
        print(line)

    # IC stability: t-stat for 20d IC
    print()
    print(f"{'factor':<32} {'IC_20d':>9} {'IC_60d':>9} {'peak_horizon':>14} {'IC@peak':>9}")
    print("-" * 75)
    for factor, ics in rows:
        ic20 = ics.get(20, math.nan)
        ic60 = ics.get(60, math.nan)
        valid = {h: v for h, v in ics.items() if not math.isnan(v)}
        if valid:
            peak_h = max(valid, key=lambda h: abs(valid[h]))
            peak_ic = valid[peak_h]
        else:
            peak_h, peak_ic = "n/a", math.nan
        ic20_s = f"{ic20:.4f}" if not math.isnan(ic20) else "n/a"
        ic60_s = f"{ic60:.4f}" if not math.isnan(ic60) else "n/a"
        peak_s = f"{peak_ic:.4f}" if not math.isnan(peak_ic) else "n/a"
        print(f"{factor:<32} {ic20_s:>9} {ic60_s:>9} {str(peak_h)+'d':>14} {peak_s:>9}")


def ic_by_year(df: pd.DataFrame, factor: str = "avg_sentiment_5d",
               horizon: int = 60) -> None:
    ret_col = f"excess_ret_{horizon}d"
    if factor not in df.columns or ret_col not in df.columns:
        return
    print(f"\nIC by Year: {factor} vs excess_ret_{horizon}d")
    print(f"{'year':>6} {'IC':>9} {'N_days':>8}")
    print("-" * 30)
    df = df.copy()
    df["year"] = df["trade_date"].dt.year
    for year, ydf in df.groupby("year"):
        daily_ics = []
        for _, grp in ydf.groupby("trade_date"):
            ic = rank_ic(grp[factor], grp[ret_col])
            if not math.isnan(ic):
                daily_ics.append(ic)
        mean_ic = float(np.mean(daily_ics)) if daily_ics else math.nan
        ic_s = f"{mean_ic:.4f}" if not math.isnan(mean_ic) else "n/a"
        print(f"{int(year):>6} {ic_s:>9} {len(daily_ics):>8}")


# ─── Part 2: SHAP Feature Importance ─────────────────────────────────────────

def shap_importance(df: pd.DataFrame, features: list[str],
                    target: str = "excess_ret_60d", top_n: int = 15) -> None:
    try:
        import shap
        import lightgbm as lgb
    except ImportError as e:
        print(f"Skipping SHAP: {e}")
        return

    print("\n" + "=" * 80)
    print(f"PART 2: SHAP Feature Importance  (target={target})")
    print("=" * 80)

    avail_features = [f for f in features if f in df.columns]
    sub = df.dropna(subset=[target]).copy()
    if len(sub) < 500:
        print("Insufficient data for SHAP analysis.")
        return

    num_cols = [c for c in avail_features if c != "sector"]
    for col in num_cols:
        sub[col] = pd.to_numeric(sub[col], errors="coerce").fillna(sub[col].median())
    if "sector" in avail_features:
        sub["sector"] = sub["sector"].fillna("unknown").astype("category")

    X = sub[avail_features]
    y = sub[target].astype(float)

    cat_cols = [c for c in avail_features if c == "sector"]
    model = lgb.LGBMRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        num_leaves=20, min_child_samples=10, verbose=-1, n_jobs=-1,
    )
    model.fit(X, y, categorical_feature=cat_cols)

    explainer = shap.TreeExplainer(model)
    sample = X.sample(min(3000, len(X)), random_state=42)
    shap_values = explainer.shap_values(sample)

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = (
        pd.DataFrame({"feature": avail_features, "mean_abs_shap": mean_abs})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    print(f"\nTop {top_n} features by mean |SHAP|:")
    print(f"{'rank':>5} {'feature':<36} {'mean_|SHAP|':>14} {'% of total':>12}")
    print("-" * 72)
    total = importance["mean_abs_shap"].sum()
    for i, row in importance.iterrows():
        pct = row["mean_abs_shap"] / mean_abs.sum() * 100
        bar = "█" * int(pct / 2)
        print(f"{i+1:>5} {row['feature']:<36} {row['mean_abs_shap']:>14.6f} {pct:>11.1f}%  {bar}")

    # Group: LLM sentiment vs. traditional
    llm_cols = {"avg_sentiment_3d", "avg_sentiment_5d", "sentiment_shift_5d",
                "high_signal_count_3d", "has_regulatory_risk_5d", "earnings_beat_signal",
                "earnings_miss_signal", "disagreement_avg_5d", "negative_event_count_5d"}
    llm_shap = sum(mean_abs[i] for i, f in enumerate(avail_features) if f in llm_cols)
    trad_shap = mean_abs.sum() - llm_shap
    print(f"\nGroup contribution:")
    print(f"  LLM sentiment features : {llm_shap / mean_abs.sum() * 100:.1f}%")
    print(f"  Traditional features   : {trad_shap / mean_abs.sum() * 100:.1f}%")


# ─── Part 3: Long-short Portfolio ────────────────────────────────────────────

def long_short_portfolio(df: pd.DataFrame, features: list[str],
                         target: str = "excess_ret_60d",
                         top_pct: float = 0.2) -> None:
    try:
        import lightgbm as lgb
    except ImportError:
        print("Skipping Long-short: lightgbm not installed.")
        return

    print("\n" + "=" * 80)
    print(f"PART 3: Long-short Portfolio  (target={target}, top/bottom {top_pct:.0%})")
    print("=" * 80)

    avail_features = [f for f in features if f in df.columns]
    sub = df.dropna(subset=[target]).copy()
    sub["year"] = sub["trade_date"].dt.year
    years = sorted(sub["year"].unique())

    cat_cols = [c for c in avail_features if c == "sector"]
    num_cols = [c for c in avail_features if c != "sector"]

    all_preds = []
    for test_year in years:
        if test_year <= years[0] + 1:
            continue
        train = sub[sub["year"] < test_year].copy()
        test = sub[sub["year"] == test_year].copy()
        if len(train) < 500 or len(test) < 100:
            continue

        for col in num_cols:
            med = pd.to_numeric(train[col], errors="coerce").median()
            train[col] = pd.to_numeric(train[col], errors="coerce").fillna(med)
            test[col] = pd.to_numeric(test[col], errors="coerce").fillna(med)
        for col in cat_cols:
            train[col] = train[col].fillna("unknown").astype("category")
            test[col] = test[col].fillna("unknown").astype("category")

        model = lgb.LGBMRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05,
            num_leaves=15, min_child_samples=5, verbose=-1, n_jobs=-1,
        )
        model.fit(train[avail_features], train[target],
                  categorical_feature=cat_cols)
        test = test.copy()
        test["pred"] = model.predict(test[avail_features])
        all_preds.append(test)

    if not all_preds:
        print("Insufficient walk-forward data.")
        return

    full = pd.concat(all_preds).sort_values("trade_date")

    # Per-date long/short selection
    long_rets, short_rets, dates = [], [], []
    for dt, grp in full.groupby("trade_date"):
        n = max(1, int(len(grp) * top_pct))
        ranked = grp.sort_values("pred", ascending=False)
        long_ret = float(ranked.head(n)[target].mean())
        short_ret = float(ranked.tail(n)[target].mean())
        long_rets.append(long_ret)
        short_rets.append(short_ret)
        dates.append(dt)

    ls_series = pd.Series(
        [l - s for l, s in zip(long_rets, short_rets)], index=dates
    )
    long_series = pd.Series(long_rets, index=dates)
    short_series = pd.Series(short_rets, index=dates)

    def annualized_sharpe(s: pd.Series, horizon: int) -> float:
        if s.std() == 0:
            return math.nan
        return float(s.mean() / s.std() * math.sqrt(252 / horizon))

    def max_dd(s: pd.Series) -> float:
        cum = (1 + s).cumprod()
        return float(((cum - cum.cummax()) / cum.cummax()).min())

    h = int(target.split("_")[-1][:-1])

    print(f"\n{'Metric':<30} {'Long':>12} {'Short':>12} {'L-S Spread':>12}")
    print("-" * 70)
    print(f"{'Mean period return':<30} {long_series.mean():>11.2%} {short_series.mean():>11.2%} {ls_series.mean():>11.2%}")
    ann_l = (1 + long_series.mean()) ** (252 / h) - 1
    ann_s = (1 + short_series.mean()) ** (252 / h) - 1
    ann_ls = (1 + ls_series.mean()) ** (252 / h) - 1
    print(f"{'Annualized return':<30} {ann_l:>11.2%} {ann_s:>11.2%} {ann_ls:>11.2%}")
    print(f"{'Sharpe (ann.)':<30} {annualized_sharpe(long_series, h):>11.2f} "
          f"{annualized_sharpe(short_series, h):>11.2f} "
          f"{annualized_sharpe(ls_series, h):>11.2f}")
    print(f"{'Max Drawdown':<30} {max_dd(long_series):>11.2%} {max_dd(short_series):>11.2%} {max_dd(ls_series):>11.2%}")
    print(f"{'Hit Rate (L-S > 0)':<30} {(ls_series > 0).mean():>11.2%}")
    print(f"{'N periods':<30} {len(ls_series):>12}")

    # Yearly breakdown
    full_ls = ls_series.to_frame("ls").assign(year=lambda x: x.index.year)
    print(f"\nYearly L-S performance:")
    print(f"{'year':>6} {'mean_ls':>10} {'ann_ls':>10} {'sharpe':>8} {'max_dd':>9} {'hit_rate':>10}")
    print("-" * 60)
    for yr, ydf in full_ls.groupby("year"):
        s = ydf["ls"]
        ann = (1 + s.mean()) ** (252 / h) - 1
        sh = annualized_sharpe(s, h)
        mdd = max_dd(s)
        hr = (s > 0).mean()
        sh_s = f"{sh:.2f}" if not math.isnan(sh) else "n/a"
        print(f"{int(yr):>6} {s.mean():>10.2%} {ann:>10.2%} {sh_s:>8} {mdd:>9.2%} {hr:>10.2%}")


# ─── main ─────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--collection", default="daily_symbol_features_company_matched_v2")
    p.add_argument("--db", default="quant_data")
    p.add_argument("--min-trade-date", default="2015-12-04")
    p.add_argument("--target", default="excess_ret_60d")
    p.add_argument("--parts", nargs="+", default=["ic", "shap", "ls"],
                   choices=["ic", "shap", "ls"])
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading features from {args.collection}...")
    df = load_feature_frame(
        db_name=args.db,
        collection_name=args.collection,
        min_trade_date=args.min_trade_date,
        max_trade_date=None,
        symbols=None,
    )
    if df.empty:
        print("No data loaded.")
        return
    print(f"Loaded {len(df):,} rows | {df['symbol'].nunique()} symbols | "
          f"{df['trade_date'].min().date()} → {df['trade_date'].max().date()}")

    df = add_sector_relative(df)

    if "ic" in args.parts:
        ic_decay_table(df, SINGLE_FACTORS)
        ic_by_year(df, factor="avg_sentiment_5d", horizon=60)

    if "shap" in args.parts:
        shap_importance(df, MODEL_FEATURES, target=args.target)

    if "ls" in args.parts:
        long_short_portfolio(df, MODEL_FEATURES, target=args.target)


if __name__ == "__main__":
    main()
