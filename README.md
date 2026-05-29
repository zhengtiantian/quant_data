# AI-Driven Equity Signal Platform

An end-to-end quantitative research platform that processes financial news through LLM pipelines to generate daily equity trading signals across 100 US stocks.

## Key Results

| Metric | Value |
|--------|-------|
| News articles processed | 840K+ (from 8TB+ raw GDELT data) |
| Stock universe | 100 US equities |
| LLM agreement rate | 77.3% (Gemma + Qwen) |
| Walk-forward Long-short annualized return | **+21.7%** |
| Long-short Sharpe ratio | **0.85** |
| Hit rate | 63.6% |
| Best single-factor IC (60d) | 0.064 (`surprise_pct_last`) |
| Platform services | 22 Docker microservices |

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
│  → daily_symbol_features (134K rows, 100 symbols × 7 horizons)      │
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
│  Airflow DAGs (daily schedule)                                       │
│  ┌─────────────────────────────────────────┐                        │
│  │ 05:00  gdelt_backfill ─┐                │                        │
│  │        finnhub_news   ─┼─▶ price_update ─▶ feature_build        │
│  │        newsapi_news   ─┤                │                        │
│  │        yahoo_news     ─┘                │                        │
│  │ 09:00  llm_enrich_pass_a ─┐            │                        │
│  │        llm_enrich_pass_b ─┼─▶ snorkel ─▶ feature_rebuild        │
│  │ weekly model_training     │            │                        │
│  └─────────────────────────────────────────┘                        │
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
├── research/                 # Core research pipeline
│   ├── daily_symbol_features.py     # Feature engineering (40+ features)
│   ├── llm_enrich_articles.py       # Dual LLM labeling (Gemma + Qwen)
│   ├── snorkel_label_merge.py       # Label aggregation
│   ├── train_baseline_models.py     # Walk-forward model training
│   ├── backtest_news_factor.py      # Single-factor backtest + risk metrics
│   ├── backtest_event_driven.py     # Event-driven backtest
│   ├── factor_analysis.py           # IC decay, SHAP, Long-short portfolio
│   └── score_daily_signals.py       # Daily signal scoring
│
├── airflow/dags/             # Airflow orchestration
│   ├── daily_pipeline.py            # Daily ETL (05:00)
│   ├── llm_enrichment.py            # LLM labeling pipeline (09:00)
│   ├── model_training.py            # Weekly model retraining
│   └── price_history_backfill.py    # Historical price backfill
│
├── stock_collector/          # Stock price & universe management
├── tools/                    # GDELT import, index building utilities
├── scheduler/                # Legacy scheduler (replaced by Airflow)
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

# Run backtest
python research/backtest_news_factor.py \
  --collection daily_symbol_features_company_matched_v2 \
  --factors avg_sentiment_5d quality_score \
  --horizons 20 60 --strategies top long_short
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

- [ ] **H.1** Transaction cost model (commission + slippage)
- [ ] **H.2** Market regime detection (VIX-based, dynamic factor weights)
- [ ] **H.3** Paper trading engine with OOS validation
- [ ] **F.4** LangGraph multi-agent research assistant
- [ ] **F.5** FinBERT fine-tuning (200x inference speedup)
- [ ] **C.1** Daily signal automation via Airflow

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for full roadmap.
