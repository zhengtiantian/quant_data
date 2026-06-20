# AI-Driven Equity Signal Platform

An end-to-end quantitative research platform that processes financial news through LLM pipelines to generate daily equity trading signals across 100 US stocks.

## Key Results

| Metric | Value |
|--------|-------|
| News articles processed | 840K+ (from 8TB+ raw GDELT data) |
| Stock universe | 100 US equities |
| LLM agreement rate | 77.3% (Gemma + Qwen) |
| Portfolio backtest Sharpe (20d, net of cost) | **0.77** (gross 0.92) vs SPY 0.54 |
| Portfolio backtest Sharpe (60d, net of cost) | **0.73** (gross 0.77) vs SPY 0.47 |
| Best single-factor cross-sectional IC | **+0.227** (`ah_gap`, 5d, after-hours price gap) |
| Strongest 60d-horizon factor | **+0.198** (`inst_holding_pct_chg`, institutional 13F QoQ change) |
| 2026 holdout model IC (LightGBM, all features) | **0.73** vs ~0.05 historical baseline |
| Platform services | 22 Docker microservices |

*Net Sharpe includes a transaction cost model: 5bps commission + 10-30bps
liquidity-tiered slippage round-trip, and excludes symbols with <$5M 20d avg
dollar volume. See `research/backtest_portfolio.py`.*

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA COLLECTION LAYER                         │
│  GDELT (8TB+) │ Finnhub │ NewsAPI │ Yahoo Finance │ FMP             │
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
                           │ 840K labeled articles
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
│  → daily_symbol_features (189K+ rows, 100 symbols × 7 horizons)     │
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
│  MLflow: experiment tracking & model versioning                      │
└──────────────────────────┬──────────────────────────────────────────┘
                           │ daily signals
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        SERVING / PLATFORM LAYER                      │
│                                                                       │
│  Kafka ──▶ Signal Distribution ──▶ Alert / Position Tracking        │
│                                                                       │
│  Spring Boot REST API (Keycloak JWT auth)                            │
│       │                                                               │
│       ▼                                                               │
│  React UI Dashboard                                                  │
│  (signal scores │ portfolio tracking │ trade alerts)                │
│                                                                       │
│  LangChain Agent + Qdrant RAG ──▶ "Analyze AAPL news" → advice      │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION LAYER                            │
│                                                                       │
│  Production schedule (launchd, scheduler/task.py) — currently live: │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ 05:15  gdelt_backfill        07:45  premarket_signals    │        │
│  │ 06:00  inst_13f (Sun)        07:48  analyst_consensus    │        │
│  │ 07:30  daily_price           07:50  macro_indicators     │        │
│  │ 08:00  daily_symbol_features 08:30  score_daily_signals  │        │
│  │ 08:40  track_positions       09:00  data_quality_check   │        │
│  │ 20:30  retail_sentiment                                  │        │
│  └─────────────────────────────────────────────────────────┘        │
│                                                                       │
│  Airflow DAGs are defined (`airflow/dags/`) but not yet verified    │
│  to run this schedule end-to-end — Stage 7 migration item.          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Tech Stack

### Data & ML (Python)
| Component | Technology |
|-----------|-----------|
| News collection | GDELT, Finnhub, NewsAPI, Yahoo Finance |
| LLM labeling | Gemma 3B + Qwen 4B (via Ollama) |
| Label aggregation | Snorkel (Dawid-Skene) |
| ML models | LightGBM, Ridge Regression, Ensemble |
| Feature store | MongoDB (134K rows, 100 symbols) |
| Experiment tracking | MLflow |
| Vector search | Qdrant |
| AI agent | LangChain + RAG |

### Infrastructure
| Component | Technology |
|-----------|-----------|
| Orchestration | Apache Airflow 2.10 |
| Message queue | Apache Kafka 3.7 |
| Database | MongoDB 6.0, MySQL 8.0 |
| Auth | Keycloak |
| Containerization | Docker (22 services) |

### Backend / Frontend
| Component | Technology |
|-----------|-----------|
| REST API | Spring Boot 3, Java 17 |
| Auth | Keycloak JWT + RBAC |
| Frontend | React + Redux Toolkit |
| AI agent service | Python FastAPI + LangChain |

---

## Project Structure

```
quant_data/
├── news_collectors/          # Multi-source news ingestion
│   ├── gdelt/                # GDELT GKG pipeline (8TB+)
│   ├── finnhub/              # Finnhub API collector
│   ├── newsapi/              # NewsAPI collector
│   └── yahoo/                # Yahoo Finance news
│
├── macro_collector/           # D.1 — VIX/rates/dollar/SPY regime (yfinance)
├── retail_collector/          # D.2 — StockTwits retail sentiment (urllib)
├── analyst_collector/         # D.4 — Finnhub analyst consensus (urllib)
├── inst_13f_collector/        # D.7 — SEC EDGAR 13F holdings (edgartools)
├── premarket_collector/       # D.8 — pre/after-market 1m gaps (yfinance)
│
├── research/                 # Core research pipeline
│   ├── daily_symbol_features.py     # Feature engineering (60+ features incl. D-series)
│   ├── llm_enrich_articles.py       # Dual LLM labeling (Gemma + Qwen)
│   ├── snorkel_label_merge.py       # Label aggregation
│   ├── train_baseline_models.py     # Walk-forward model training
│   ├── backtest_news_factor.py      # Single-factor backtest + risk metrics
│   ├── backtest_event_driven.py     # Event-driven backtest
│   ├── backtest_portfolio.py        # Top-N portfolio backtest + cost model (H.1)
│   ├── factor_analysis.py           # IC decay, SHAP, Long-short portfolio
│   ├── score_daily_signals.py       # Daily signal scoring + macro regime multiplier
│   ├── track_positions.py           # Paper-trading position tracker + exit alerts
│   └── data_quality_check.py        # Pipeline health checks (C.7)
│
├── airflow/dags/             # Airflow orchestration (defined, not yet verified e2e)
│   ├── daily_pipeline.py            # Daily ETL (05:00)
│   ├── llm_enrichment.py            # LLM labeling pipeline (09:00)
│   ├── model_training.py            # Weekly model retraining
│   └── price_history_backfill.py    # Historical price backfill
│
├── stock_collector/          # Stock price & universe management
├── tools/                    # GDELT import, index building utilities
├── scheduler/                # task.py — actual production scheduler (launchd)
└── main.py                   # FastAPI service entry point
```

---

## Research Results

### IC Decay (Single Factor vs Return Horizons)

| Factor | 20d IC | 60d IC | Peak |
|--------|--------|--------|------|
| `surprise_pct_last` | -0.039 | -0.064 | Strongest (mean-reversion) |
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
| `avg_sentiment_5d` (existing baseline) | LLM | +0.064 | +0.176 | +0.145 |

LightGBM trained on 2016-2025, evaluated on 2026 holdout (all features incl.
D-series): **IC = 0.73** vs ~0.05 on a comparable model without D-series
features — driven mainly by `inst_holding_pct_chg` and `macro_tnx`. Sample
size for the holdout is still small (525 rows); D-series features only have
~6 months of history so far and need more data before the lift is fully
trusted in production.

---

## Getting Started

### Prerequisites
- Docker Desktop
- Python 3.11+
- Ollama (for local LLM inference)

### Start Platform

```bash
cd quant_docker
docker compose up -d
```

Services available at:
- **quant_ui**: http://localhost:18080
- **quant_api**: http://localhost:18081
- **Airflow**: http://localhost:15060
- **MLflow**: http://localhost:15050
- **Kafka UI**: http://localhost:15070

### Run Research Pipeline

```bash
cd quant_data
source .venv311/bin/activate

# Build daily features
python research/daily_symbol_features.py

# Train models
python research/train_baseline_models.py --target all

# Run factor analysis
python research/factor_analysis.py --parts ic shap ls

# Run single-factor backtest
python research/backtest_news_factor.py \
  --collection daily_symbol_features_company_matched_v2 \
  --factors avg_sentiment_5d quality_score \
  --horizons 20 60 --strategies top long_short

# Run portfolio backtest (Top-N, net of transaction cost)
python research/backtest_portfolio.py   # BACKTEST_HOLD_DAYS=20|60 env var
```

---

## Platform Overview

### Signal Generation Flow
```
Daily news (GDELT + APIs)
    → LLM sentiment labeling (Gemma + Qwen)
    → Feature engineering (40+ factors)
    → LightGBM ensemble scoring
    → Composite signal per symbol
    → Kafka topic: quant.daily_signals
    → UI dashboard + alerts
```

### LangChain RAG Agent
Ask natural language questions about stocks:
> "Has AAPL had any negative regulatory news in the last 30 days?"
> "What's the sentiment trend for NVDA this month?"

Agent retrieves relevant articles from Qdrant, analyzes with LLM, returns structured answer.

---

## Roadmap

- [x] **D.1/D.2/D.4/D.7/D.8** Alt-data research layer (macro, retail, analyst, 13F, premarket)
- [x] **H.1** Transaction cost model (commission + liquidity-tiered slippage)
- [x] **C.1** Daily signal automation (via launchd, not yet migrated to Airflow)
- [x] **C.4/C.5** Paper-trading position tracker + exit alerts (incl. analyst-downgrade, inst-outflow triggers)
- [x] **C.7/C.9** Data quality checks + factor analysis report (IC decay, SHAP)
- [ ] **H.2** Dynamic factor re-weighting by regime (basic risk-on/off multiplier already live in scoring)
- [ ] **H.3** Paper trading stop-loss + rolling OOS-IC monitor
- [ ] **Stage 7** Airflow/Kafka/MLflow verified end-to-end (currently: defined but not proven live)
- [ ] **F.4** LangGraph multi-agent research assistant
- [ ] **F.5** FinBERT fine-tuning (200x inference speedup)

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full roadmap and per-item status.
