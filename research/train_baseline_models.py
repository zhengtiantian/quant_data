#!/usr/bin/env python3
"""Train baseline models on daily symbol features with walk-forward validation."""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path

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

from backtest_news_factor import load_feature_frame

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--collection", default="daily_symbol_features_company_matched_v1")
    parser.add_argument("--db", default="quant_data")
    parser.add_argument("--min-trade-date", default="2015-12-04")
    parser.add_argument("--target", default="excess_ret_20d", choices=["excess_ret_20d", "excess_ret_60d"])
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


def walk_forward_train_eval(df: pd.DataFrame, features: list[str], target: str) -> None:
    df = df.dropna(subset=[target] + features).copy()
    if df.empty:
        print("Data is empty after dropping missing values.")
        return
    
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
    
    for model_name, model in models.items():
        print(f"\nModel: {model_name}")
        print(f"{'Year':>6} | {'Train Obs':>10} | {'Test Obs':>10} | {'Rank IC':>10} | {'Top 5 Ex Ret':>15}")
        print("-" * 65)
        
        all_preds = []
        for test_year in years:
            if test_year <= years[0] + 1:
                # Need at least 2 years of training data to start
                continue 
            
            # Walk-forward: train on everything before test_year
            train = df[df["year"] < test_year]
            test = df[df["year"] == test_year]
            
            if len(train) < 500 or len(test) < 100:
                continue
                
            model.fit(train[features], train[target])
            preds = model.predict(test[features])
            
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
            total_test = pd.concat(all_preds)
            res = evaluate_predictions(total_test, target, "pred")
            ic = res["Rank IC"]
            topN = res["Top N Mean Ex Ret"]
            ic_str = f"{ic:.4f}" if pd.notna(ic) else "n/a"
            top_str = f"{topN:.2%}" if pd.notna(topN) else "n/a"
            print("-" * 65)
            print(f"{'ALL':>6} | {'-':>10} | {len(total_test):>10} | {ic_str:>10} | {top_str:>15}")


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
    
    # Check if necessary new features exist (we added past_ret and volatility)
    required = ["past_ret_20d", "volatility_20d"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"Dataset is missing features: {missing}. Have you rebuilt daily symbol features?")
        return

    # Baseline features specified by the project plan
    features = [
        "full_ratio",
        "quality_score",
        "article_count",
        "news_burst_20d",
        "past_ret_20d",
        "past_ret_60d",
        "volatility_20d",
        "volatility_60d",
        "volume_shock_20d",
        "sector"
    ]
    
    print("\nRunning Walk-forward Evaluation for 20d Target...")
    walk_forward_train_eval(df, features, target="excess_ret_20d")
    
    print("\nRunning Walk-forward Evaluation for 60d Target...")
    walk_forward_train_eval(df, features, target="excess_ret_60d")


if __name__ == "__main__":
    main()
