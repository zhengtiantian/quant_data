# quant_data — Research & ML Pipeline

[![Tests](https://github.com/zhengtiantian/quant_data/actions/workflows/test.yml/badge.svg)](https://github.com/zhengtiantian/quant_data/actions/workflows/test.yml)

The data and machine-learning half of the [AI-Driven Equity Signal
Platform](https://github.com/zhengtiantian/ai-equity-signal-platform): it collects financial
news, labels it with a dual-LLM pipeline, engineers features, trains the ranking models, and
writes the daily signals that the rest of the platform serves.

Serving (REST API, dashboard, AI assistant, MCP) lives in the sibling repos — see the
[platform README](https://github.com/zhengtiantian/ai-equity-signal-platform) for the whole
system and the cross-repo roadmap.

## Key Results

| Metric | Value |
|--------|-------|
| News articles processed | 845K+ (from 13TB raw GDELT data) |
| Stock universe | 103 equities (100 US + HXSCL OTC) |
| LLM agreement rate | 77.3% (Gemma + Qwen) |
| Portfolio backtest Sharpe (20d, net of cost) | **0.77** (gross 0.92) vs SPY 0.54 |
| Portfolio backtest Sharpe (60d, net of cost) | **0.73** (gross 0.77) vs SPY 0.47 |
| Best single-factor cross-sectional IC | **+0.227** (`ah_gap`, 5d, after-hours price gap) |
| Strongest 60d-horizon factor | **+0.198** (`inst_holding_pct_chg`, institutional 13F QoQ change) |
| 2026 holdout model IC (LightGBM, all features) | **0.73** vs ~0.05 historical baseline |
| Feature store | 189K+ rows · 103 symbols × 7 return horizons |
| Airflow DAGs owned by this repo | 10 (7 scheduled + 3 manual) |

*Net Sharpe includes a transaction cost model: 5bps commission + 10–30bps liquidity-tiered slippage round-trip, and excludes symbols with <$5M 20d avg dollar volume. See `research/backtest/backtest_portfolio.py`.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION LAYER                         │
│  GDELT (13TB) │ Finnhub │ NewsAPI │ Yahoo Finance │ FMP             │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ raw news articles
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      NLP / LLM LABELING LAYER                        │
│                                                                       │
│  company_match (SLM)  ──▶  Pass A: Gemma 3B                         │
│        │                        │                                    │
│        │                   Pass B: Qwen 4B                           │
│        │                        │                                    │
│        └──────────────▶  Snorkel Dawid-Skene                        │
│                          (label aggregation)                          │
│                               │                                       │
│                    llm_sentiment_final / event_type                  │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ 845K labeled articles
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      FEATURE ENGINEERING LAYER                       │
│                                                                       │
│  News features      LLM sentiment features   Earnings features       │
│  article_count      avg_sentiment_3d/5d       surprise_pct_last      │
│  news_burst_20d     sentiment_shift_5d        days_to_earnings        │
│  quality_score      high_signal_count_3d      earnings_recency        │
│  full_ratio         negative_event_count      is_earnings_window      │
│                                                                       │
│  + Price features: past_ret_20d/60d, volatility_20d/60d             │
│  + Sector-relative ranks                                             │
│                                                                       │
│  Alt-data factors (D-series, added 2026-06):                        │
│  macro_*        VIX/10Y/DXY/SPY regime           (D.1, urllib+yf)   │
│  retail_*       StockTwits crowd sentiment        (D.2, urllib)     │
│  analyst_*      Finnhub consensus + 1m momentum   (D.4, urllib)     │
│  inst_holding_* SEC EDGAR 13F QoQ change          (D.7, edgartools) │
│  ah_gap/pm_gap  yfinance 1m extended-hours gap    (D.8, yfinance)   │
│                                                                       │
│  → daily_symbol_features (189K+ rows, 103 symbols × 7 horizons)     │
└──────────────────────────┬──────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         ML MODEL LAYER                               │
│                                                                       │
│  Walk-forward validation (expanding window by year)                  │
│                                                                       │
│  Ridge Regression ──┐                                                │
│                     ├──▶ Ensemble (avg) ──▶ IC=0.059, Sharpe=0.85  │
│  LightGBM Ranker ───┘                                                │
│                                                                       │
│  Factor Analysis: IC decay (5-60d) │ SHAP importance │ Long-short   │
│  MLflow: experiment tracking & model versioning (8 runs logged)      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ daily signals
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│   HANDOFF — this repo ends here; serving lives in sibling repos      │
│                                                                       │
│   daily_signals + daily_symbol_features (MongoDB)                    │
│        └──▶ quant_api (REST + Kafka) ──▶ quant_ui / quant_ai / MCP   │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION LAYER                            │
│                                                                       │
│  Apache Airflow, host-based scheduler (launchd-managed) — 10 DAGs   │
│                                                                       │
│  Scheduled (7):                                                      │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ */30 min   quant_intraday_news    (Finnhub+NewsAPI+Yahoo)│        │
│  │ 06:30 d.   price_history_backfill                        │        │
│  │ 07:30 d.   daily_signal_pipeline  (price→feat→signal→DQ) │        │
│  │ 20:30 d.   quant_retail_sentiment (StockTwits)            │        │
│  │ 04:00 Sun  gdelt_batch_verify     (batch self-heal check)│        │
│  │ 06:00 Sun  weekly_inst13f_holdings                        │        │
│  │ 07:00 Sun  weekly_model_training  (Ridge+LightGBM CV)     │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                       │
│  Manual / on-demand (3) — full-history news backfill in two         │
│  chained stages + a quality-audit tool:                             │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ backfill_1_collect_and_match                              │        │
│  │   GDELT collect ⟶ SLM company match                       │        │
│  │ backfill_2_enrich_and_features                            │        │
│  │   LLM pass A ⟶ pass B ⟶ label merge ⟶ feature rebuild     │        │
│  │ quant_news_validation (relevance/quality audit report)   │        │
│  └─────────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Data & ML (Python)
| Component | Technology |
|-----------|-----------|
| News collection | GDELT, Finnhub, NewsAPI, Yahoo Finance, FMP |
| LLM labeling | Gemma 3B + Qwen 4B (via LM Studio) |
| Label aggregation | Snorkel (Dawid-Skene) |
| ML models | LightGBM, Ridge Regression, Ensemble |
| Feature store | MongoDB (189K rows, 100 symbols) |
| Experiment tracking | MLflow |

### Infrastructure
| Component | Technology |
|-----------|-----------|
| Orchestration | Apache Airflow 2.10, host-based scheduler (launchd-managed) — 10 production DAGs |
| Database | MongoDB 6.0 (articles, features, signals), MySQL 8.0 (GDELT batch queue) |
| Distributed work | MySQL-backed task queue — multi-machine GDELT workers, crash-safe retry, idempotent upserts |
| Containerization | Docker |

> Serving-side technology (Spring Boot API, React dashboard, Keycloak, Kafka, quant_ai, MCP)
> belongs to the sibling repos — see the
> [platform README](https://github.com/zhengtiantian/ai-equity-signal-platform).

---

## Project Structure

```
quant_data/
├── collectors/news/          # Multi-source news ingestion
│   ├── gdelt/                # GDELT GKG pipeline (13TB)
│   ├── finnhub/              # Finnhub API collector
│   ├── newsapi/              # NewsAPI collector
│   └── yahoo/                # Yahoo Finance news
│
├── collectors/macro/           # D.1 — VIX/rates/dollar/SPY regime (yfinance)
├── collectors/retail/          # D.2 — StockTwits retail sentiment (urllib)
├── collectors/analyst/         # D.4 — Finnhub analyst consensus (urllib)
├── collectors/inst_13f/        # D.7 — SEC EDGAR 13F holdings (edgartools)
├── collectors/premarket/       # D.8 — pre/after-market 1m gaps (yfinance)
│
├── research/                 # Core research pipeline
│   ├── daily_symbol_features.py     # Feature engineering (60+ features incl. D-series)
│   ├── llm_enrich_articles.py       # Dual LLM labeling (Gemma + Qwen via LM Studio)
│   ├── train_baseline_models.py     # Walk-forward model training
│   ├── backtest_news_factor.py      # Single-factor backtest + risk metrics
│   ├── backtest_event_driven.py     # Event-driven backtest
│   ├── backtest_portfolio.py        # Top-N portfolio backtest + cost model
│   ├── factor_analysis.py           # IC decay, SHAP, Long-short portfolio
│   ├── score_daily_signals.py       # H.2: 4-regime weight switching (RISK_ON/NEUTRAL/STRESSED/RISK_OFF)
│   ├── track_positions.py           # H.3: vol-adaptive stop-loss (2×vol_20d) + OOS IC monitor
│   └── data_quality_check.py        # Pipeline health checks
│
├── tests/                    # Unit test suite (90 tests, CI-enforced)
│   ├── conftest.py           # sys.path setup for research/ imports
│   ├── test_feature_build.py # News aggregation, LLM sentiment, quality score
│   ├── test_regime_scoring.py# H.2: classify_regime, compute_score, _safe_float
│   ├── test_positions.py     # H.3: _stop_pct, compute_daily_vols, _spearman_ic, stop-loss trigger
│   └── test_earnings_regime.py # Earnings features, macro regime (D-series)
│
├── airflow/dags/             # Airflow orchestration — 14 production DAGs (7 scheduled, 7 manual)
├── collectors/stock/          # Stock price & universe management
├── tools/                    # GDELT import, index building utilities
├── scheduler/                # task.py — legacy launchd scheduler (superseded by Airflow, kept disabled)
└── pytest.ini                # Test config + coverage settings
```

---

## Research Results

### IC Decay (Single Factor vs Return Horizons)

| Factor | 20d IC | 60d IC | Peak |
|--------|--------|--------|------|
| `surprise_pct_last` | -0.039 | -0.064 | Strongest mean-reversion |
| `avg_sentiment_5d` | +0.026 | +0.023 | 45d horizon |
| `article_count` | +0.015 | +0.034 | 45d horizon |
| `volatility_20d` | +0.009 | +0.036 | 60d horizon |
| `negative_event_count_5d` | +0.005 | +0.029 | 60d horizon |

### Long-short Portfolio (Walk-forward, 60d target, top/bottom 20%)

| Metric | Long | Short | L-S Spread |
|--------|------|-------|------------|
| Annualized return | +16.3% | -4.7% | **+21.7%** |
| Sharpe ratio | 0.86 | -0.36 | **0.85** |
| Hit rate (L-S > 0) | — | — | 63.6% |

### Walk-forward Model (LightGBM Ensemble, 60d)

| Year | Rank IC | Top-5 Excess Return |
|------|---------|-------------------|
| 2020 | 0.058 | +4.21% |
| 2021 | 0.071 | +5.33% |
| 2022 | 0.041 | +3.12% |
| 2023 | 0.089 | +7.84% |
| 2024 | 0.055 | +4.90% |
| **ALL** | **0.059** | **+5.89%** |

### Alt-Data Factor IC (D-series, cross-sectional Spearman, 2026 data)

| Factor | Source | CS-IC (5d) | CS-IC (20d) | CS-IC (60d) |
|--------|--------|-----------|-------------|-------------|
| `ah_gap` | yfinance 1m after-hours | **+0.227** | — | — |
| `analyst_buy_ratio_chg_1m` | Finnhub | +0.155 | +0.151 | — |
| `inst_holding_pct_chg` | SEC EDGAR 13F | +0.045 | +0.092 | **+0.198** |
| `macro_spy_ret_20d` | yfinance | +0.100 | +0.285 | — |
| `avg_sentiment_5d` (baseline) | LLM | +0.064 | +0.176 | +0.145 |

LightGBM trained on 2016–2025, evaluated on 2026 holdout (all features incl. D-series): **IC = 0.73** vs ~0.05 on a comparable model without D-series features — driven mainly by `inst_holding_pct_chg` and `macro_tnx`. Sample size for the holdout is still small (~525 rows); D-series features only have ~6 months of history and need more data before the lift is fully trusted in production.

---

## Getting Started

### Prerequisites
- Python 3.11+
- MongoDB and MySQL reachable — start them from the platform repo (`cd quant_docker && docker compose up -d mongodb mysql`)
- [LM Studio](https://lmstudio.ai/) with the labeling models loaded:
  - `gemma-4-e4b-it-mlx` — news labeling pass A
  - `qwen3.5-9b-mlx` — news labeling pass B
  - `qwen3.5-4b-mlx` — SLM company match / relevance filter

Bringing up the whole platform (API, dashboard, Airflow, MLflow, Kafka) is covered in the
[platform README](https://github.com/zhengtiantian/ai-equity-signal-platform).

### Run Research Pipeline

```bash
cd quant_data
source .venv311/bin/activate

# Build daily features
python research/features/daily_symbol_features.py

# Train models
python research/models/train_baseline_models.py --target all

# Run factor analysis
python research/backtest/factor_analysis.py --parts ic shap ls

# Run single-factor backtest
python research/backtest/backtest_news_factor.py \
  --collection daily_symbol_features_company_matched_v2 \
  --factors avg_sentiment_5d quality_score \
  --horizons 20 60 --strategies top long_short

# Run portfolio backtest (Top-N, net of transaction cost)
python research/backtest/backtest_portfolio.py   # BACKTEST_HOLD_DAYS=20|60 env var
```

### Run Tests

```bash
cd quant_data
.venv311/bin/python -m pytest tests/ -q
```

---

## Roadmap

The roadmap is maintained once, at the platform level, because most items span several repos.
See the [platform README](https://github.com/zhengtiantian/ai-equity-signal-platform#roadmap)
for the item list and
[PROJECT_PLAN.md](https://github.com/zhengtiantian/ai-equity-signal-platform/blob/main/PROJECT_PLAN.md)
for the detailed spec and effort estimate behind each ID.
