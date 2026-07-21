# AI-Driven Equity Signal Platform

[![Tests](https://github.com/zhengtiantian/quant_data/actions/workflows/test.yml/badge.svg)](https://github.com/zhengtiantian/quant_data/actions/workflows/test.yml)

An end-to-end quantitative research platform that processes financial news through LLM pipelines to generate daily equity trading signals across 103 stocks.

## Key Results

| Metric | Value |
|--------|-------|
| News articles processed | 840K+ (from 8TB+ raw GDELT data) |
| Stock universe | 103 equities (100 US + HXSCL OTC) |
| LLM agreement rate | 77.3% (Gemma + Qwen) |
| Portfolio backtest Sharpe (20d, net of cost) | **0.77** (gross 0.92) vs SPY 0.54 |
| Portfolio backtest Sharpe (60d, net of cost) | **0.73** (gross 0.77) vs SPY 0.47 |
| Best single-factor cross-sectional IC | **+0.227** (`ah_gap`, 5d, after-hours price gap) |
| Strongest 60d-horizon factor | **+0.198** (`inst_holding_pct_chg`, institutional 13F QoQ change) |
| 2026 holdout model IC (LightGBM, all features) | **0.73** vs ~0.05 historical baseline |
| Platform services | 9 Docker microservices |

*Net Sharpe includes a transaction cost model: 5bps commission + 10–30bps liquidity-tiered slippage round-trip, and excludes symbols with <$5M 20d avg dollar volume. See `research/backtest/backtest_portfolio.py`.*

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
│  quant_ai (RAG + Local LLM) ──▶ natural language stock Q&A          │
└─────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       ORCHESTRATION LAYER                            │
│                                                                       │
│  Apache Airflow, host-based scheduler (launchd-managed) — 14 DAGs   │
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
│  Manual / on-demand (7) — full-history news backfill, split into    │
│  independently-triggerable steps + a quality-audit tool:            │
│  ┌─────────────────────────────────────────────────────────┐        │
│  │ backfill_1_gdelt_collect → backfill_2_company_match →     │        │
│  │   backfill_3/4_llm_enrich_a/b → backfill_5_snorkel_merge  │        │
│  │   → backfill_6_feature_rebuild                            │        │
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
| Orchestration | Apache Airflow 2.10, host-based scheduler (launchd-managed) — 14 production DAGs |
| Message queue | Apache Kafka 3.7 |
| Database | MongoDB 6.0, MySQL 8.0 |
| Auth | Keycloak |
| Containerization | Docker |

### Backend / Frontend
| Component | Technology |
|-----------|-----------|
| REST API | Spring Boot 3, Java 21 |
| Auth | Keycloak JWT + RBAC |
| Frontend | React + TypeScript + Vite |
| AI assistant | Python FastAPI + RAG + LM Studio |

---

## Project Structure

```
quant_data/
├── collectors/news/          # Multi-source news ingestion
│   ├── gdelt/                # GDELT GKG pipeline (8TB+)
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
- Docker Desktop (48GB+ RAM recommended — all LLMs run via LM Studio)
- Python 3.11+
- [LM Studio](https://lmstudio.ai/) with the following models loaded:
  - `gemma3:4b` — news labeling Pass A
  - `qwen3:4b` — news labeling Pass B
  - `qwen3.5-9b-mlx` — quant_ai chat assistant
  - `nomic-embed-text` — quant_ai RAG embeddings

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

### Completed
- [x] **D.1/D.2/D.4/D.7/D.8** Alt-data research layer (macro, retail, analyst, 13F, premarket)
- [x] **H.1** Transaction cost model (commission + liquidity-tiered slippage)
- [x] **H.2** Dynamic 4-regime weight switching (RISK_ON / NEUTRAL / STRESSED / RISK_OFF)
- [x] **H.3** Volatility-adaptive stop-loss (2×vol_20d, clamped 4–12%) + rolling OOS IC monitor
- [x] **C.1** Daily signal automation (launchd production scheduler)
- [x] **C.4/C.5** Paper-trading position tracker + exit alerts
- [x] **C.7/C.9** Data quality checks + factor analysis report (IC decay, SHAP)
- [x] **C.8** ETL unit tests — 90 tests, CI-enforced via GitHub Actions
- [x] **E.2** CI/CD GitHub Actions — auto-run tests on every push
- [x] **E.7** Root README + Mermaid architecture diagram

### Signal & Quant Research
- [ ] **H.4** Rolling signal quality monitor — dashboard view of OOS IC trend over time
- [ ] **B.1** Long-short portfolio enhancement — beta neutralization, sector exposure limits
- [ ] **Stage 7** Airflow + Kafka end-to-end verified — replace launchd with Airflow DAGs in production

### Live Trading
- [ ] **G.1** Broker API integration (Alpaca) — connect existing signals to real order execution with pre-trade risk guardrails (max position 5%, daily loss kill-switch, fill reconciliation); Stage 1: paper account → Stage 2: live with small capital

### AI Engineering — LLM / RAG
- [ ] **F.2** RAG news search (Qdrant) — replace MongoDB full-scan with vector similarity search for quant_ai
- [ ] **F.5** FinBERT fine-tuning — replace dual-LLM labeling with a single fine-tuned model (~200× inference speedup)
- [ ] **F.10** Strategy Studio → backtest execution — wire the existing natural-language strategy UI to `backtest_portfolio.py` so generated strategies produce real Sharpe / drawdown results
- [ ] **F.11** News pre-filter SLM — lightweight binary classifier (distilbert) before dual-LLM pass; eliminates ~70% irrelevant GDELT articles
- [ ] **F.12** Signal explanation generation — SLM generates a 2-sentence "why this stock scored high" explanation for each top signal; displayed inline in SignalsPanel
- [ ] **F.13** Morning briefing agent — 07:00 daily pre-market summary for held positions: overnight news, regime, exit warnings
- [ ] **F.14** Earnings surprise prediction — in the 10-day pre-earnings window, LLM aggregates news sentiment + analyst consensus → beat/miss probability as a new factor
- [ ] **F.15** SEC EDGAR + earnings transcript RAG — 10-K/10-Q risk sections and earnings call transcripts embedded in Qdrant; natural language queries on filing content
- [ ] **F.19** LLM factor hypothesis generator — prompt LLM with current IC table + failure modes → suggests new factor ideas for human review

### AI Engineering — Agents
- [ ] **F.4** LangGraph multi-agent research assistant — 4-node graph: data_agent → analysis_agent → strategy_agent → risk_agent
- [ ] **F.8** Active learning agent — surface low-confidence LLM labels for human review; close the annotation feedback loop
- [ ] **F.9** Rule optimization agent — iterative self-improving loop: sample → LLM judge → diagnose FP/FN → modify rules (🟡 code written, not yet tested)
- [ ] **F.16** Real-time news monitoring agent — 30-minute polling of NewsAPI for held positions; instant alert on sentiment spike or negative event cluster
- [ ] **F.17** Portfolio Manager Agent — LangGraph 2-node agent reads daily signals + positions + regime → structured add/reduce/hold recommendation
- [ ] **F.18** Backtest reflection agent — auto-diagnoses weak-year IC failures (2022/2024) and generates a hypothesis report
- [ ] **F.6** Rule validator ReAct agent — LLM-powered interactive rule debugging loop
- [ ] **F.7** Airflow adaptive scheduling agent — dynamically adjust collection windows based on data quality metrics

### MCP Integration
- [ ] **I.1** quant_mcp_server — expose signals, news, positions, factor IC, regime, and backtest trigger as MCP tools; any MCP-compatible client can query live platform data
- [ ] **I.2** Claude Desktop integration — register quant_mcp_server in Claude Desktop; natural language trading queries with zero custom integration code
- [ ] **I.3** Alpaca order execution via MCP — extend MCP server with order tools so LLM agents can place/cancel orders through the same interface (pre-trade guardrails enforced server-side)
- [ ] **I.4** External data MCP tools — wrap Finnhub, SEC EDGAR, yfinance as MCP tools so agents autonomously decide what data to fetch
- [ ] **I.5** MCP inter-service communication — replace quant_ai → quant_api REST calls with MCP protocol for dynamic tool discovery

### Platform & Infrastructure
- [ ] **E.6** WebSocket real-time push — stream live signal scores to the React dashboard without polling
- [ ] **E.9** UI intraday price chart — TradingView Lightweight Charts + Alpaca bars API; entry/stop-loss overlay on each position; no hourly data stored in own DB
- [ ] **E.4** Kubernetes configuration — replace Docker Compose with K8s manifests for production deployment

### Stock Universe
- [ ] **G.2** Phase 2 expansion — energy/materials (XOM, CVX, NEE, LIN, APD); Phase 3 international ADRs (BABA, JD, PDD, SE); Phase 4 REITs/financials

See [PROJECT_PLAN.md](PROJECT_PLAN.md) for detailed specs and effort estimates per item.
