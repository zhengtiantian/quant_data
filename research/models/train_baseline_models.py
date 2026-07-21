#!/usr/bin/env python3
"""Train baseline models on daily symbol features with walk-forward validation."""

from __future__ import annotations

import argparse
import math
import os

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None  # type: ignore

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from dotenv import load_dotenv

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from research.backtest.backtest_news_factor import load_feature_frame

SECTOR_REL_COLS = [
    "full_ratio", "quality_score", "news_burst_20d", "earnings_recency_weight",
    # D-series cross-sectional features worth sector-normalising
    "analyst_buy_ratio", "analyst_buy_ratio_chg_1m", "inst_holding_pct_chg",
]


def add_sector_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add sector-relative rank features (within trade_date × sector)."""
    df = df.copy()
    for col in SECTOR_REL_COLS:
        if col not in df.columns:
            continue
        df[f"{col}_sector_rel"] = (
            df.groupby(["trade_date", "sector"])[col]
            .transform(lambda x: pd.to_numeric(x, errors="coerce").rank(pct=True) - 0.5)
        )
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="daily_symbol_features")
    parser.add_argument("--db", default="quant_data")
    parser.add_argument("--min-trade-date", default="2015-12-04")
    parser.add_argument("--target", default="all",
                        choices=["excess_ret_10d", "excess_ret_15d", "excess_ret_20d",
                                 "excess_ret_30d", "excess_ret_45d", "excess_ret_60d", "all"])
    parser.add_argument("--mlflow-uri", default=os.getenv("MLFLOW_TRACKING_URI", ""),
                        help="MLflow tracking URI (empty = disabled)")
    return parser.parse_args()


def evaluate_predictions(df: pd.DataFrame, target: str, pred_col: str) -> dict:
    ics = []
    top_returns = []
    
    for dt, group in df.groupby("trade_date"):
        if len(group) < 5:
            continue
        valid = group.dropna(subset=[target, pred_col])
        if len(valid) < 5:
            continue
            
        ic, _ = spearmanr(valid[pred_col], valid[target])
        if not math.isnan(ic):
            ics.append(ic)
        
        # Top 5 mean excess return
        top_N = min(5, len(valid))
        top_portfolio = valid.sort_values(pred_col, ascending=False).head(top_N)
        top_returns.append(top_portfolio[target].mean())
        
    return {
        "Rank IC": np.mean(ics) if ics else np.nan,
        "Top N Mean Ex Ret": np.mean(top_returns) if top_returns else np.nan
    }


def walk_forward_rank_eval(
    df: pd.DataFrame, features: list[str], target: str,
    collection: str = "unknown", use_mlflow: bool = False,
) -> None:
    """Walk-forward evaluation using LightGBM lambdarank objective."""
    try:
        import lightgbm as lgb
    except ImportError:
        print(f"  [skip] lightgbm not installed — skipping LightGBM Ranker for {target}")
        return

    df = df.dropna(subset=[target]).copy()
    if df.empty:
        return

    available = [f for f in features if f in df.columns]
    dropped = [f for f in features if f not in df.columns]
    if dropped:
        print(f"  [info] {len(dropped)} features not in collection, skipped: {dropped[:6]}{'...' if len(dropped)>6 else ''}")
    features = available

    df["year"] = df["trade_date"].dt.year
    years = sorted(df["year"].unique())

    categorical_features = ["sector"]
    cat_cols = [c for c in categorical_features if c in features]
    num_cols = [c for c in features if c not in cat_cols]

    params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5],
        "learning_rate": 0.05,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "verbosity": -1,
        "n_jobs": -1,
    }

    print(f"\n{'='*70}")
    print(f"Target: {target}")
    print(f"\nModel: LightGBM Ranker (lambdarank)")
    print(f"{'Year':>6} | {'Train Obs':>10} | {'Test Obs':>10} | {'Rank IC':>10} | {'Top 5 Ex Ret':>15}")
    print("-" * 65)

    all_preds = []
    for test_year in years:
        if test_year <= years[0] + 1:
            continue

        train = df[df["year"] < test_year].sort_values("trade_date").reset_index(drop=True)
        test = df[df["year"] == test_year].sort_values("trade_date").reset_index(drop=True)

        if len(train) < 500 or len(test) < 100:
            continue

        # Quintile rank labels (0-4) within each trade_date
        def make_labels(sub: pd.DataFrame) -> pd.Series:
            return sub.groupby("trade_date")[target].transform(
                lambda x: pd.qcut(x.rank(method="first"), q=min(5, len(x)),
                                  labels=False, duplicates="drop")
                if len(x) >= 2 else pd.Series(0, index=x.index)
            ).fillna(0).astype(int)

        train_labels = make_labels(train)
        train_groups = train.groupby("trade_date", sort=False).size().values

        # Prepare feature matrices
        X_train = train[features].copy()
        X_test = test[features].copy()
        for col in num_cols:
            med = X_train[col].median()
            X_train[col] = X_train[col].fillna(med)
            X_test[col] = X_test[col].fillna(med)
        for col in cat_cols:
            X_train[col] = X_train[col].fillna("unknown").astype("category")
            X_test[col] = X_test[col].fillna("unknown").astype("category")

        train_ds = lgb.Dataset(
            X_train, label=train_labels, group=train_groups,
            categorical_feature=cat_cols, free_raw_data=False
        )
        model = lgb.train(params, train_ds, num_boost_round=100,
                          callbacks=[lgb.log_evaluation(period=-1)])

        preds = model.predict(X_test)
        test_copy = test.copy()
        test_copy["pred"] = preds
        all_preds.append(test_copy)

        res = evaluate_predictions(test_copy, target, "pred")
        ic = res["Rank IC"]
        topN = res["Top N Mean Ex Ret"]
        ic_str = f"{ic:.4f}" if pd.notna(ic) else "n/a"
        top_str = f"{topN:.2%}" if pd.notna(topN) else "n/a"
        print(f"{test_year:>6} | {len(train):>10} | {len(test):>10} | {ic_str:>10} | {top_str:>15}")

    if all_preds:
        total = pd.concat(all_preds)
        res = evaluate_predictions(total, target, "pred")
        ic = res["Rank IC"]
        topN = res["Top N Mean Ex Ret"]
        ic_str = f"{ic:.4f}" if pd.notna(ic) else "n/a"
        top_str = f"{topN:.2%}" if pd.notna(topN) else "n/a"
        print("-" * 65)
        print(f"{'ALL':>6} | {'-':>10} | {len(total):>10} | {ic_str:>10} | {top_str:>15}")

        if use_mlflow:
            with mlflow.start_run(run_name=f"LightGBM Ranker | {target}"):
                mlflow.log_params({
                    "model": "LightGBM Ranker", "target": target,
                    "collection": collection, "n_features": len(features),
                    "objective": "lambdarank",
                })
                if pd.notna(ic):
                    mlflow.log_metric("rank_ic_overall", float(ic))
                if pd.notna(topN):
                    mlflow.log_metric("top5_overall", float(topN))


def walk_forward_train_eval(
    df: pd.DataFrame, features: list[str], target: str,
    collection: str = "unknown", use_mlflow: bool = False,
) -> None:
    df = df.dropna(subset=[target]).copy()
    if df.empty:
        print("Data is empty after dropping missing values.")
        return

    # Drop features not present in the loaded dataframe (e.g. D-series on old collections)
    available = [f for f in features if f in df.columns]
    dropped = [f for f in features if f not in df.columns]
    if dropped:
        print(f"  [info] {len(dropped)} features not in collection, skipped: {dropped[:6]}{'...' if len(dropped)>6 else ''}")
    features = available

    df["year"] = df["trade_date"].dt.year
    years = sorted(df["year"].unique())

    categorical_features = ["sector"]
    numeric_features = [f for f in features if f != "sector"]

    models = {
        "Ridge": Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ])),
            ("ridge", Ridge(alpha=10.0))
        ]),
        "LightGBM (HistGB)": Pipeline([
            ("preprocessor", ColumnTransformer([
                ("num", SimpleImputer(strategy="median"), numeric_features),
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
            ])),
            ("hgb", HistGradientBoostingRegressor(max_iter=100, max_depth=3, learning_rate=0.05, verbose=0))
        ])
    }
    
    print("\n" + "="*70)
    print(f"Target: {target}")
    print(f"Features: {features}")
    
    # Collect per-year predictions from each model for ensemble
    model_year_preds: dict[str, dict[int, pd.DataFrame]] = {m: {} for m in models}

    for model_name, model in models.items():
        print(f"\nModel: {model_name}")
        print(f"{'Year':>6} | {'Train Obs':>10} | {'Test Obs':>10} | {'Rank IC':>10} | {'Top 5 Ex Ret':>15}")
        print("-" * 65)

        all_preds = []
        year_metrics: dict[int, dict] = {}
        for test_year in years:
            if test_year <= years[0] + 1:
                continue

            train = df[df["year"] < test_year]
            test = df[df["year"] == test_year]

            if len(train) < 500 or len(test) < 100:
                continue

            model.fit(train[features], train[target])
            preds = model.predict(test[features])

            test_copy = test.copy()
            test_copy["pred"] = preds
            all_preds.append(test_copy)
            model_year_preds[model_name][test_year] = test_copy

            res = evaluate_predictions(test_copy, target, "pred")
            ic = res["Rank IC"]
            topN = res["Top N Mean Ex Ret"]
            year_metrics[test_year] = res

            ic_str = f"{ic:.4f}" if pd.notna(ic) else "n/a"
            top_str = f"{topN:.2%}" if pd.notna(topN) else "n/a"
            print(f"{test_year:>6} | {len(train):>10} | {len(test):>10} | {ic_str:>10} | {top_str:>15}")

        if all_preds:
            total_test = pd.concat(all_preds)
            res = evaluate_predictions(total_test, target, "pred")
            ic = res["Rank IC"]
            topN = res["Top N Mean Ex Ret"]
            ic_str = f"{ic:.4f}" if pd.notna(ic) else "n/a"
            top_str = f"{topN:.2%}" if pd.notna(topN) else "n/a"
            print("-" * 65)
            print(f"{'ALL':>6} | {'-':>10} | {len(total_test):>10} | {ic_str:>10} | {top_str:>15}")

            if use_mlflow:
                with mlflow.start_run(run_name=f"{model_name} | {target}"):
                    mlflow.log_params({
                        "model": model_name, "target": target,
                        "collection": collection, "n_features": len(features),
                    })
                    for yr, yr_res in year_metrics.items():
                        if pd.notna(yr_res["Rank IC"]):
                            mlflow.log_metric("rank_ic", yr_res["Rank IC"], step=yr)
                        if pd.notna(yr_res["Top N Mean Ex Ret"]):
                            mlflow.log_metric("top5_excess_ret", yr_res["Top N Mean Ex Ret"], step=yr)
                    if pd.notna(ic):
                        mlflow.log_metric("rank_ic_overall", float(ic))
                    if pd.notna(topN):
                        mlflow.log_metric("top5_overall", float(topN))

    # Ensemble: average predictions from Ridge and LightGBM
    model_names = list(models.keys())
    common_years = set(model_year_preds[model_names[0]]) & set(model_year_preds[model_names[1]])
    if common_years:
        print(f"\nModel: Ensemble (Ridge + LightGBM avg)")
        print(f"{'Year':>6} | {'Train Obs':>10} | {'Test Obs':>10} | {'Rank IC':>10} | {'Top 5 Ex Ret':>15}")
        print("-" * 65)
        ensemble_all = []
        for test_year in sorted(common_years):
            df_a = model_year_preds[model_names[0]][test_year]
            df_b = model_year_preds[model_names[1]][test_year]
            merged = df_a.copy()
            merged["pred"] = (df_a["pred"].values + df_b["pred"].values) / 2.0
            ensemble_all.append(merged)
            res = evaluate_predictions(merged, target, "pred")
            ic = res["Rank IC"]
            topN = res["Top N Mean Ex Ret"]
            train_size = len(df[df["year"] < test_year])
            ic_str = f"{ic:.4f}" if pd.notna(ic) else "n/a"
            top_str = f"{topN:.2%}" if pd.notna(topN) else "n/a"
            print(f"{test_year:>6} | {train_size:>10} | {len(merged):>10} | {ic_str:>10} | {top_str:>15}")
        total = pd.concat(ensemble_all)
        res = evaluate_predictions(total, target, "pred")
        ic = res["Rank IC"]
        topN = res["Top N Mean Ex Ret"]
        ic_str = f"{ic:.4f}" if pd.notna(ic) else "n/a"
        top_str = f"{topN:.2%}" if pd.notna(topN) else "n/a"
        print("-" * 65)
        print(f"{'ALL':>6} | {'-':>10} | {len(total):>10} | {ic_str:>10} | {top_str:>15}")

        if use_mlflow:
            with mlflow.start_run(run_name=f"Ensemble | {target}"):
                mlflow.log_params({
                    "model": "Ensemble", "target": target,
                    "collection": collection, "n_features": len(features),
                })
                for yr_data in ensemble_all:
                    yr = yr_data["year"].iloc[0]
                    yr_res = evaluate_predictions(yr_data, target, "pred")
                    if pd.notna(yr_res["Rank IC"]):
                        mlflow.log_metric("rank_ic", yr_res["Rank IC"], step=yr)
                    if pd.notna(yr_res["Top N Mean Ex Ret"]):
                        mlflow.log_metric("top5_excess_ret", yr_res["Top N Mean Ex Ret"], step=yr)
                if pd.notna(ic):
                    mlflow.log_metric("rank_ic_overall", float(ic))
                if pd.notna(topN):
                    mlflow.log_metric("top5_overall", float(topN))


def main():
    args = parse_args()
    print(f"Loading data from {args.collection}...")
    df = load_feature_frame(
        db_name=args.db,
        collection_name=args.collection,
        min_trade_date=args.min_trade_date,
        max_trade_date=None,
        symbols=None
    )
    
    if df.empty:
        print("No valid rows returned.")
        return

    print(f"Data loaded: {len(df):,} rows.")
    df = add_sector_relative_features(df)
    print(f"Added sector-relative features: {[c+'_sector_rel' for c in SECTOR_REL_COLS if c in df.columns]}")

    use_mlflow = bool(args.mlflow_uri) and MLFLOW_AVAILABLE
    if args.mlflow_uri and not MLFLOW_AVAILABLE:
        print("MLFLOW_TRACKING_URI set but mlflow not installed — skipping experiment tracking")
    if use_mlflow:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment("quant_baseline_models")
        print(f"MLflow tracking: {args.mlflow_uri}")
    
    # Check if necessary new features exist (we added past_ret and volatility)
    required = [
        "past_ret_20d", "volatility_20d",
        "days_to_earnings", "is_earnings_window_5d",
        "surprise_pct_last", "is_positive_surprise",
        "days_to_earnings_bucket", "is_post_positive_surprise_20d",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Dataset is missing features: {missing}. Have you rebuilt daily symbol features?")
        return

    # Baseline features specified by the project plan
    features = [
        # news quality
        "full_ratio",
        "quality_score",
        "article_count",
        "news_burst_20d",
        # price momentum / risk
        "past_ret_20d",
        "past_ret_60d",
        "volatility_20d",
        "volatility_60d",
        "volume_shock_20d",
        # earnings timing
        "days_to_earnings",
        "days_since_earnings",
        "is_earnings_window_5d",
        "is_post_earnings_window_20d",
        "days_to_earnings_bucket",
        "days_since_earnings_bucket",
        "is_pre_earnings_10d",
        "is_post_earnings_5d",
        "is_post_earnings_10d",
        # earnings surprise
        "eps_estimate_last",
        "reported_eps_last",
        "surprise_pct_last",
        "is_positive_surprise",
        "is_negative_surprise",
        # drift + surprise
        "is_post_positive_surprise_20d",
        "is_post_negative_surprise_20d",
        # layer 3: surprise magnitude
        "surprise_bucket",
        # layer 3: news quality × earnings interaction
        "full_ratio_x_post_positive",
        "quality_score_x_post_negative",
        # layer 3: earnings recency decay
        "earnings_recency_weight",
        # LLM sentiment features (Stage 3.5.2)
        "avg_sentiment_3d",
        "avg_sentiment_5d",
        "sentiment_shift_5d",
        "high_signal_count_3d",
        "has_regulatory_risk_5d",
        "earnings_beat_signal",
        "earnings_miss_signal",
        "disagreement_avg_5d",
        "negative_event_count_5d",
        # sector-relative features
        "full_ratio_sector_rel",
        "quality_score_sector_rel",
        "news_burst_20d_sector_rel",
        # D.8 — pre/after-market gaps
        "pm_gap",
        "ah_gap",
        "ah_volume_ratio",
        # D.4 — analyst consensus
        "analyst_buy_ratio",
        "analyst_consensus_score",
        "analyst_buy_ratio_chg_1m",
        "analyst_buy_ratio_sector_rel",
        "analyst_buy_ratio_chg_1m_sector_rel",
        # D.7 — institutional 13F
        "inst_holding_pct",
        "inst_holding_pct_chg",
        "inst_holding_pct_chg_sector_rel",
        # D.1 — macro regime
        "macro_vix",
        "macro_vix_pctile_252d",
        "macro_vix_change_5d",
        "macro_tnx",
        "macro_tnx_change_20d",
        "macro_spy_ret_20d",
        "macro_spy_above_200ma",
        "macro_risk_on",
        "macro_is_high_vol",
        "macro_dxy_change_20d",
        # D.2 — retail sentiment (sparse but directional)
        "retail_bull_ratio",
        "retail_sent_score",
        # sector
        "sector",
    ]
    
    horizons = [10, 15, 20, 30, 45, 60] if args.target == "all" else [int(args.target.split("_")[-1][:-1])]

    for h in horizons:
        target = f"excess_ret_{h}d"
        print(f"\nRunning Walk-forward Evaluation for {h}d Target...")
        walk_forward_train_eval(df, features, target=target,
                                collection=args.collection, use_mlflow=use_mlflow)

    for h in horizons:
        target = f"excess_ret_{h}d"
        print(f"\nRunning Ranking Model Evaluation for {h}d Target...")
        walk_forward_rank_eval(df, features, target=target,
                               collection=args.collection, use_mlflow=use_mlflow)


if __name__ == "__main__":
    main()
