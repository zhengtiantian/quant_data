# Quant Data Project Plan

## Goal
Build a research-ready, then production-ready, mid-frequency stock prediction system for holding periods of:

- 1 week
- 2-4 weeks
- 2-3 months

The immediate objective is not live trading. The immediate objective is:

- reliable data pipelines
- clean research datasets
- factor validation
- backtesting
- model-ready feature tables

## Current Status

What already exists:

- historical and incremental news collection
- stock daily price collection
- company-match cleaning pipeline for news
- cleaned matched-news collection:
  - `news_articles_company_matched_v1`
- daily feature build pipeline
- cleaned feature collection:
  - `daily_symbol_features_company_matched_v1`
- first-pass single-factor backtest script

What is still missing:

- benchmark-relative labels and evaluation
- stronger research-quality filtering layer
- richer factor library
- grouped / long-short backtests
- model training pipeline
- portfolio construction layer
- production signal generation layer

## System Scope

The system should support:

- company/news relevance filtering
- daily feature generation
- mid-horizon return prediction
- factor and model research
- portfolio simulation
- scheduled data refresh
- production deployment later

## Required Data

### 1. News Data
Already partially done.

Needed:

- raw news store
- company-matched clean news store
- source metadata
- title
- body/content
- data quality tags
- company relevance result

Current collections:

- `news_articles`
- `news_articles_company_matched_v1`

### 2. Price Data
Needed:

- daily OHLCV
- adjusted close if possible
- volume
- trading date alignment

Current collection:

- `stock_prices_history`

### 3. Benchmark and Market Data
Still needed / should be added:

- SPY / QQQ / sector ETF daily prices
- benchmark forward returns
- optional factor benchmarks:
  - momentum
  - size
  - quality
  - volatility

### 4. Fundamental / Event Data
Recommended next:

- earnings dates
- EPS / revenue / guidance
- valuation metrics
- analyst estimate revision data if available
- corporate actions

### 5. Metadata
Needed:

- sector / industry
- market cap bucket
- symbol universe definition
- active/inactive date ranges

## Core Collections / Tables

### Raw Layer

- `news_articles`
- `stock_prices_history`

### Clean Layer

- `news_articles_company_matched_v1`

### Feature Layer

- `daily_symbol_features_company_matched_v1`

### Planned Next Collections

- `daily_symbol_features_research_v1`
  - optional refined feature table after additional research-quality filtering
- `backtest_results`
  - optional persistent experiment tracking
- `benchmark_prices`
  - benchmark ETF / index data
- `earnings_events`
  - earnings calendar / event layer

## Current Feature Set

Already generated in daily features:

- `article_count`
- `full_count`
- `title_only_count`
- `url_only_count`
- `full_ratio`
- `title_only_ratio`
- `avg_content_length`
- `max_content_length`
- `unique_url_count`
- `unique_source_count`
- `unique_platform_count`
- `extraction_failed_count`
- `timeout_fallback_count`
- `news_count_3d`
- `news_count_5d`
- `news_count_20d`
- `full_count_5d`
- `avg_full_ratio_5d`
- `news_burst_20d`
- `trade_date`
- `close`
- `future_ret_5d`
- `future_ret_20d`
- `future_ret_60d`

## Prediction Targets

The system should support multiple target definitions.

### Absolute Return Targets

- `future_ret_5d`
- `future_ret_20d`
- `future_ret_60d`

### Direction Targets

- `future_ret_5d > 0`
- `future_ret_20d > 0`
- `future_ret_60d > 0`

### Recommended Future Targets

- benchmark-relative return:
  - stock return minus SPY / QQQ / sector ETF return
- ranking target:
  - cross-sectional rank by future return

For this project, ranking and excess-return targets are likely more useful than pure up/down classification.

## Research Workflow

### Phase 1. Data Integrity

Done or mostly done:

- collect raw news
- clean company relevance
- build daily features
- align with price labels

Still needed:

- benchmark alignment
- event/fundamental enrichment

### Phase 2. Factor Research

Current next step.

Need to run:

- top/bottom bucket tests
- long-short tests
- top 3 / top 5 / top 10 portfolio comparisons
- stability by year
- stability by horizon

First candidate factors:

- `full_ratio`
- `news_burst_20d`
- `article_count`
- `unique_source_count`
- `avg_content_length`
- `extraction_failed_count` as a reverse-quality feature

### Phase 3. Multi-Factor Research

After single-factor validation:

- combine quality factors
- combine quantity + quality + recency factors
- compare weighted score vs single-factor selection

### Phase 4. Modeling

Only after factor validation.

Recommended order:

1. linear / logistic models
2. tree models:
   - LightGBM
   - XGBoost
3. ranking models

Do not start with large text models for price prediction.

### Phase 5. Portfolio Construction

Need explicit rules for:

- rebalance frequency
- top N holdings
- equal weight vs score weight
- sector cap
- single-name cap
- turnover limit
- transaction cost assumptions

### Phase 6. Production

Later phase:

- daily incremental collection
- feature refresh
- model scoring
- signal generation
- order routing
- monitoring
- logging
- retry / failure handling

## Research Principles

### 1. Prefer simple baselines first

Always test:

- single factor
- grouped returns
- naive top-N strategy

before training larger models.

### 2. Use safe date windows

Backtests should use:

- `trade_date >= 2015-12-04`

because current price data coverage starts there.

### 3. Use clean collection by default

Research should prefer:

- `news_articles_company_matched_v1`
- `daily_symbol_features_company_matched_v1`

instead of raw collections.

### 4. Keep raw data immutable

Do not delete raw source collections.

Use separate clean / derived layers.

## Engineering Plan

## Stage 1. Immediate Next Tasks

1. Extend first-pass backtest
- top / bottom buckets
- long-short
- top 3 / top 5 / top 10

2. Add benchmark comparison
- SPY / QQQ / sector baseline

3. Build quality factor basket
- combine `full_ratio`, `unique_source_count`, `avg_content_length`, etc.

4. Validate by horizon
- 5d
- 20d
- 60d

## Stage 2. Near-Term Data Expansion

Status: substantially complete.

### Completed items

- benchmark/index price collection:
  - done (`SPY`, `QQQ` daily prices, `benchmark_ret_5d/20d/60d`, `excess_ret_*`)
- earnings event Layer 1 (timing only):
  - done
  - features: `days_to_earnings`, `days_since_earnings`,
    `is_earnings_window_5d`, `is_post_earnings_window_20d`
- earnings event Layer 2 (surprise + richer timing):
  - done
  - features: `eps_estimate_last`, `reported_eps_last`, `surprise_pct_last`,
    `is_positive_surprise`, `is_negative_surprise`,
    `days_to_earnings_bucket`, `days_since_earnings_bucket`,
    `is_pre_earnings_10d`, `is_post_earnings_5d`, `is_post_earnings_10d`,
    `is_post_positive_surprise_20d`, `is_post_negative_surprise_20d`
  - result: Ridge `60d` IC improved from `0.073` to `0.121` (+66%)
- sector mapping: basic version integrated

### Next: earnings event Layer 3

Build on Layer 2 with deeper surprise structure and interaction features.

1. surprise magnitude buckets
   - split `surprise_pct_last` into ordinal buckets:
     - large miss (< -5%), small miss (-5% to 0), small beat (0 to +5%),
       large beat (> +5%)
   - rationale: the market reacts non-linearly to surprise size

2. news quality × earnings interaction features
   - `full_ratio_x_post_positive`: `full_ratio` × `is_post_positive_surprise_20d`
   - `quality_score_x_post_negative`: `quality_score` × `is_post_negative_surprise_20d`
   - rationale: a positive surprise with high-quality news coverage is a
     stronger combined signal than either alone

3. earnings recency decay
   - `earnings_recency_weight`: exponential decay of `days_since_earnings`
     (weight = exp(-days_since / 20))
   - rationale: the post-earnings drift effect weakens as time passes

4. rerun backtest and model after Layer 3
   - compare against the Layer 2 baseline (Ridge `60d` IC = `0.121`)

### Most important next step: universe expansion

The current universe is 14 symbols.
This is the single biggest constraint on signal reliability.

Why this matters:

- at 14 symbols, one bad year (e.g. 2022, 2024) can make the IC negative
  due to sampling noise alone
- cross-sectional ranking needs depth: `top 3 / top 5` out of 14 is not
  robust; `top 5 / top 10` out of 50-100 is meaningfully more stable
- the current signals may be real but cannot be confirmed at this universe size

Target: expand to 50-100 symbols.

What is required:

1. extend `stock_universe` collection with new symbols
   - suggested scope: S&P 500 tech/growth names, or a defined sector basket
   - keep sector distribution balanced

2. collect historical news for new symbols
   - extend GDELT / news collector coverage
   - run `company_match_rescore` on new articles

3. collect price history for new symbols
   - extend `stock_prices_history`

4. rebuild feature table
   - `FEATURE_REBUILD_ALL=true` after data is ready

5. rerun all backtests and models on expanded universe
   - compare single-factor IC at 50-symbol vs 14-symbol universe
   - a good signal should survive and become more stable

## Stage 3. Modeling

Status: first-pass baselines done.

Current baseline results (14-symbol universe):

- `Ridge` `60d` Rank IC: `0.121` / Top 5 excess return: `+6.58%`
- `HistGB` `60d` Rank IC: `0.118` / Top 5 excess return: `+6.32%`

Known issue: year-by-year variance is high.

- strong years: 2020 (IC = 0.35), 2023 (IC = 0.19), 2025 (IC = 0.24)
- weak years: 2022 (IC = -0.06), 2024 (IC = -0.04)
- likely cause: macro regime shifts (2022: rate hikes + war; 2024: election
  + rate pivot) break the news-quality thesis temporarily

Next modeling steps (do only after universe expansion):

1. year stability analysis
   - run `--group-by-year` on factor backtest with full feature set
   - identify which features drive the 2022/2024 failures
   - test whether regime-aware features (e.g. VIX level, market vol) help

2. simple Ridge + HistGB ensemble
   - average predictions from both models
   - rationale: the two models now have comparable IC and different error
     patterns; blending should reduce variance

3. sector-relative feature normalization
   - normalize `full_ratio`, `quality_score` within sector on each day
   - reduces bias from sectors that structurally attract more/less news

4. ranking model
   - convert to a pairwise ranking objective (LambdaRank or similar)
   - more directly aligned with the top-N portfolio construction goal

## Stage 3.5 Event-Driven Strategy Research

Goal: evolve the fixed-horizon model into a news event-driven strategy where
signals trigger dynamic entry, hold, and exit decisions rather than fixed
holding periods.

This stage must be completed before Stage 5 engineering work begins.

### 3.5.1 Article-level event tagging (LLM batch annotation) ✅ Done 2026-05-22

All 840,212 articles in `news_articles_company_matched_v2` tagged with:

- Pass A (Gemma): `llm_sentiment_a`, `llm_event_type_a`, `llm_signal_strength_a` — 100%
- Pass B (Qwen): `llm_sentiment_b`, `llm_event_type_b`, `llm_signal_strength_b` — 100%
- Snorkel merge: `llm_sentiment_final`, `llm_disagreement`, `llm_label_model_probs` — 100%
- Inter-model agreement rate 77.3%; mean sentiment +0.296 (overall positive bias)

Deliverable: `research/llm_enrich_articles.py`, `research/snorkel_label_merge.py`

Resume framing:
*"Enriched 840K financial news articles with two-pass LLM ensemble (Gemma +
Qwen), aggregated via Snorkel Label Model achieving 77.3% inter-model
agreement on sentiment, event type, and signal strength labels."*

### 3.5.2 Event-level daily features ✅ Done 2026-05-22

Rebuilt `daily_symbol_features_company_matched_v2` (134,642 rows, 100 symbols)
with full LLM sentiment feature coverage (100%):

| Feature | Coverage |
|---|---|
| `avg_sentiment_3d/5d` | 100% |
| `sentiment_shift_5d` | 99.6% |
| `high_signal_count_3d` | 100% |
| `negative_event_count_5d` | 100% |
| `disagreement_avg_5d` | 100% |
| `has_regulatory_risk_5d` | 100% |
| `earnings_beat_signal` / `earnings_miss_signal` | 100% |

Multi-horizon model results with LLM features (walk-forward, 100 symbols):

| Horizon | Ridge IC | Ensemble IC | Ensemble Top5 Excess Return |
|---|---|---|---|
| 20d | 0.036 | 0.031 | +1.83% |
| 45d | 0.044 | 0.043 | +4.38% |
| 60d | 0.056 | 0.059 | +6.59% |

IC increases monotonically with holding period; the 60d signal is strongest.

Deliverable: `research/daily_symbol_features.py`, `research/train_baseline_models.py`

### 3.5.3 Dynamic holding period backtest ✅ Done 2026-05-22

Event-driven framework results (100 symbols, 2018–2026):

Best config: `min_hold=20d`, `max_hold=60d`, `sentiment_shift_exit=-0.35`

| Metric | Event-Driven | Fixed 20d Baseline | Fixed 60d Baseline |
|---|---|---|---|
| Avg Hold | 13.7d | 20d | 60d |
| Excess Return | +1.40% | +1.22% | +3.80% |
| Win Rate | 52.7% | 58.0% | 63.1% |
| Trades | 390 | 2,060 | 2,020 |

Event-driven outperforms the fixed 20d baseline within 13.7 days (+1.40% vs +1.22%) with lower turnover.
Exit reasons: score_below_exit 69%, sentiment_reversal 17% (reasonable).
2022/2024 remain weak years (macroeconomic regime shifts).

Deliverable: `research/backtest_event_driven.py`

### 3.5.4 Multi-horizon label expansion ✅ Done 2026-05-22

Added to `daily_symbol_features_company_matched_v2`:
- `future_ret_10d/15d/30d/45d` — 87-88% coverage
- `excess_ret_10d/15d/30d/45d` — 87-88% coverage
- Also fixed `load_feature_frame` projection in `backtest_news_factor.py`

Deliverable: `research/daily_symbol_features.py`, `research/backtest_news_factor.py`

### 3.5.5 Fine-tune small model for sentiment scoring

After the two-pass LLM enrichment (3.5.1) completes, replace the slow LLM
inference pipeline with a purpose-fine-tuned small model. This eliminates the
need to keep Gemma + Qwen + a judge model running and reduces inference from
days to minutes.

**Training data**

Use articles where Pass A (Gemma) and Pass B (Qwen) agree as high-confidence
pseudo-labels:
- `llm_sentiment_a` and `llm_sentiment_b` same sign AND |diff| ≤ 0.2
- Expected: ~60-70% of 586K articles → ~350K+ clean training samples

**Model**

Fine-tune `ProsusAI/finbert` (110M params, already pre-trained on financial
text) with three output heads:
- sentiment: regression head, output -1.0 to +1.0
- event_type: 7-class softmax (earnings / product / MA / regulation / macro /
  competition / other)
- signal_strength: 3-class softmax (high / medium / low)

Training setup:
- LoRA or full fine-tune (BERT is small enough for full fine-tune)
- GPU: Windows machine (same as LM Studio)
- Estimated training time: 1-2 hours on 350K samples
- Framework: HuggingFace Transformers + PyTorch

**Inference speed after fine-tuning**

| Setup | Speed |
|---|---|
| Current LLM (Gemma via LM Studio) | ~3.9 art/s |
| Fine-tuned FinBERT (GPU) | ~1000-2000 art/s |
| Fine-tuned FinBERT (CPU only) | ~200 art/s |

586K articles: current ~41h → fine-tuned ~5-10 min

**Deliverables**

- `research/finetune_sentiment.py` — training script
- `research/infer_sentiment.py` — batch inference on remaining articles
- benchmark report: fine-tuned vs LLM scores on held-out set

**Resume framing**

*"Fine-tuned FinBERT on 350K+ pseudo-labeled financial news articles
(distilled from Gemma + Qwen ensemble), achieving 200x inference speedup
while maintaining comparable sentiment, event classification, and signal
strength accuracy."*

### Execution order

1. 3.5.4 Multi-horizon labels ✅
2. 3.5.1 Article tagging ✅
3. 3.5.2 Event features ✅
4. 3.5.3 Dynamic backtest ✅
5. 3.5.5 Fine-tune small model — Pending development

---

## Stage 4. Production Readiness

1. schedule feature build from clean news
2. track model versions
3. persist experiment results
4. build daily signal job
5. add monitoring

## File / Module Roadmap

### Existing

- `research/daily_symbol_features.py`
- `research/company_match_rescore.py`
- `research/backtest_news_factor.py`
- `research/load_earnings_events.py`
- `research/load_benchmark_prices.py`
- `research/train_baseline_models.py`
- `news_collectors/gdelt/special_rules/slm_filter.py`
- `news_collectors/gdelt/special_rules/slm_skills.py`

### Suggested Next Files

- `research/backtest_year_stability.py`
  - year-by-year IC breakdown with the full feature set
- `research/build_interaction_features.py`
  - earnings × news quality interaction features (Layer 3)
- `research/expand_universe.py`
  - helper to onboard new symbols into stock_universe and trigger collection

## Recommended Success Criteria

The project should be considered healthy if:

- clean company-matched news can be built reproducibly
- feature table refresh is stable
- at least one factor shows persistent positive excess return
- results are stable across multiple years
- simple baselines are competitive before modeling

## Current Practical Recommendation

Do next (in priority order, updated 2026-06-20):

1. **H.1 Backtest with transaction costs + liquidity filter** (2 days) — current Sharpe 0.70-0.84 is pre-cost; quantitative interviews always ask this, highest ROI
2. **Stage 7 validation** — Airflow DAG end-to-end run, Kafka producer/consumer actual operation, execution log API live (still the biggest gap for interview storytelling; C-series has used launchd as a replacement for daily scheduling, but not Airflow)
3. **E.7 README + architecture diagram** (1 day) — required for all interviews, currently missing, fastest result
4. **H.2.2 Dynamic factor weights** (regime-aware) — regime_mult base multiplier already exists; needs extension to auto-switch factor weight sets in volatile market
5. **H.3.2/H.3.3 Paper trading improvements** — position tracking and exit triggers already exist; missing OOS IC monitoring and -5% stop-loss rule
6. **Stage 3.5.5 FinBERT fine-tuning** — train on Gemma+Qwen consensus labels to replace LLM inference pipeline

Completed:

- universe expansion to 100 symbols ✅
- year stability analysis ✅
- earnings Layer 1 / 2 / 3 ✅
- Ridge + LightGBM + Ensemble baseline (60d IC = 0.10, Top5 = +6.7%) ✅
- Stage 3.5.1 LLM article tagging (840K articles, Gemma + Qwen + Snorkel merge) ✅
- Stage 3.5.2 event-level daily features (134K rows, 100% LLM coverage) ✅
- Stage 3.5.3 event-driven backtest (min20d, +1.40% vs fixed 20d +1.22%) ✅
- Stage 3.5.4 multi-horizon labels (10/15/30/45/60d) ✅
- **D.1/D.2/D.4/D.7/D.8 research layer extensions all completed** (2026-06-16), D.3/D.5/D.6 skipped ✅
- **C.1-C.7, C.9 all completed (baseline versions)** (daily signal automation + UI + position tracking + exit alerts + risk metrics + data quality checks + factor analysis report, 2026-06-16~20) ✅
- **H.2.1 regime multiplier baseline** (score_daily_signals.py integrated macro_risk_on / macro_vix_pctile_252d to adjust signal strength) ✅
- Ops: removed duplicate launchd scheduler job to prevent all collection tasks from running twice daily ✅

Do not do next:

- live trading automation (live trading is still H.1-H.5 fully complete + 3-6 months of paper trading away from being truly usable)
- Stage 5 engineering before Stage 7 validation is complete

## Notes

The clean matched-news dataset is research-usable at the current scale.

The two-layer earnings feature set has produced a meaningful model
improvement (Ridge `60d` IC: `0.073` → `0.121`).

Signal IC at 60d = 0.059 on the 100-symbol universe is more credible than the 0.121 at 14 symbols (a larger universe dilutes overfitting). The weak 2022/2024 years are driven by macroeconomic regime shifts; the signal itself is credible.

---

# Interview Roadmap

## A. Data Engineer Interview Roadmap

### Currently Available (can discuss directly)
- MongoDB 840K+ articles + 675M GKG inverted index (large-scale document storage + full-text search)
- Python ETL pipeline: news collection → company matching → feature build (multiple data sources, incremental/full load)
- LLM batch inference pipeline (840K articles, two passes + Snorkel Label Model)
- Docker Compose multi-service orchestration (MongoDB / MySQL / Kafka / Airflow / MLflow / Qdrant)
- Airflow DAG definitions (DAG structure, task dependencies, SLA)

### Critical Gaps — Must Fill (will be asked in interviews)

| What is missing | Why it matters | Stage | Effort |
|---|---|---|---|
| Kafka has no actual producer/consumer | DE must-ask; "deployed but unused" is unconvincing | Stage 7 / 5.1.1 | 3 days |
| Airflow DAG has not actually run end-to-end | "wrote a DAG" ≠ "it ran" | Stage 7 | 1 week |
| MLflow has no actual recorded runs | Can use the tool but has no output; can't answer interview questions about run results | Stage 7 / 5.2.3 | 1 day |
| No ETL unit tests | DE interviews always ask how pipeline reliability is guaranteed | New | 3 days |
| No data quality checks | Standard for production-grade pipelines; absence raises reliability doubts | New | 2 days |
| No idempotency design doc | DE must-know: will re-running the pipeline produce duplicates? | New (add doc) | 1 day |

### Bonus Items — Stronger with, not required to interview

| Item | Value | Stage | Effort |
|---|---|---|---|
| Flink streaming pipeline | High-demand DE skill at large companies, most differentiating | 5.2.2 | 2 weeks |
| dbt data lineage | Data modeling best practice, common in FinTech DE | 5.2.4 | 1 week |
| Prometheus + Grafana monitoring | Observability, also useful for SRE interviews | 5.4 | 3 days |
| Schema Registry (Kafka) | Avro schema evolution, large-scale data governance | 5.1.1 | 2 days |
| CI/CD for data pipeline | GitHub Actions running tests + lint | New | 2 days |
| Data lineage diagram | Which tables depend on which source; can use OpenLineage | New | 3 days |

### Complete DE interview story (after filling gaps)
*"Built a financial news processing system covering 100 stocks: established full-text index on GDELT raw data (675M records); Python ETL cleaned and matched 840K articles with idempotency design and data quality checks; Airflow scheduled 5 DAGs for daily incremental updates; Kafka publishes daily trading signals (complete producer/consumer pipeline); MLflow tracked 100+ model experiments; entire system deployable with a single Docker Compose command."*

---

## B. FinTech / Quantitative Finance Interview Roadmap

### Currently Available (can discuss directly)
- Walk-forward validation (not backtest overfitting), IC=0.059 (60d), 100 symbol universe
- Multi-factor model: news quality + momentum + earnings events + LLM sentiment
- Event-driven backtest framework (min_hold=20d, +1.40% excess return, outperforming fixed holding baseline)
- LLM dual-model ensemble labeling (Gemma + Qwen + Snorkel, 77.3% agreement rate)
- 840K articles with three-dimensional labels: sentiment / event type / signal strength

### Critical Gaps — Must Fill (quantitative interview essentials)

| What is missing | Why it matters | Effort |
|---|---|---|
| Sharpe ratio / max drawdown / Sortino | Must-ask for quant roles; no numbers = incomplete strategy | 2 days |
| Turnover + transaction cost model (0.05% assumption) | Is the strategy still profitable after costs? | 1 day |
| Annualized return vs SPY buy-and-hold comparison | Does the strategy beat passive holding? | 1 day |
| Factor IC decay curve (autocorrelation) | How many days until the signal expires, determines rebalance frequency | 1 day |
| Information Ratio (IC / std(IC)) | Signal stability; IC=0.059 is usable but IR must be shown | 0.5 days |
| Long-short portfolio (not just Top-N) | Standard for quant funds; long-only = incomplete | 3 days |
| Factor correlation matrix (VIF / redundancy check) | Avoid multicollinearity, model interpretability | 1 day |
| Year-by-year Sharpe + drawdown chart | Quantify risk in weak years (2022/2024) | 1 day |

### Bonus Items — Stronger with

| Item | Value | Stage | Effort |
|---|---|---|---|
| FinBERT fine-tuning (200x inference speedup) | AI+Finance crossover, unique highlight | 3.5.5 | 1 week |
| Factor attribution analysis (SHAP feature importance) | Interpretability, can demo live in interview | New | 1 day |
| Beta neutralization / market-neutral portfolio | Institutional quant standard | New | 2 days |
| Volatility-weighted position sizing (non-equal weight) | More sophisticated portfolio construction than equal weight | New | 1 day |
| Liquidity filter (market cap threshold) | Avoid small-cap slippage, real-world executability | New | 0.5 days |
| RAG news semantic search | "Semantic search + finance" highlight | 5.3.1 | 1 week |
| Multi-agent research assistant | AI engineering depth showcase | 5.3.2 | 2 weeks |
| Paper trading live verification record | Out-of-sample real-world performance, most convincing in interviews | C.6 | Ongoing |

### Complete quantitative interview story (after filling gaps)
*"Built a news-driven multi-factor model on a 100-stock tech universe: 840K articles labeled by Gemma+Qwen dual-model LLM ensemble (77.3% agreement); walk-forward validation 60d Rank IC=0.059, Information Ratio=X.X; event-driven holding framework achieved +1.40% excess return in an average of 13.7 days (after transaction costs +X.X%), annualized Sharpe=X.X, max drawdown X%, long-short annualized excess X%."*

---

## C. Project Productionization Roadmap (News Events → Buy/Sell/Hold Decisions)

### Minimum Viable Loop (do first)

#### C.1 Daily signal generation automation ✅ Done (integrated with launchd scheduler, not Airflow)
- `score_daily_signals.py` integrated into `scheduler/task.py`, runs automatically at 08:30 daily
  (`daily_symbol_features` 08:00 → `score_daily_signals` 08:30 → `track_positions` 08:40)
- Scores 100 stocks, writes results to `daily_signals` collection
- Fields: `symbol`, `trade_date`, `composite_score`, `signal_rank`, `signal_type` (LONG/NEUTRAL),
  `regime_mult`, and D-series context fields (`ah_gap`, `analyst_buy_ratio_chg_1m`,
  `inst_holding_pct_chg`, `retail_sent_score`, `macro_risk_on`, `macro_vix`, etc.)
- Note: still using launchd, not Airflow (Stage 7 / 5.2.1 Airflow migration still pending)

#### C.2 Risk metrics ✅ Done (`research/backtest_portfolio.py`)
- Available: Sharpe ratio (annualized, 4% risk-free rate), max drawdown, win rate, annualized return vs SPY
- 2026-06-16 validation results (full history 2015-2026, reusing `score_daily_signals.compute_score`):
  - 20d holding: Sharpe 0.84, annualized 25.8%, vs SPY Sharpe 0.54/12.1%
  - 60d holding: Sharpe 0.70, annualized 24.3%, vs SPY Sharpe 0.46/11.6%
- Still needed: Sortino ratio, turnover / transaction cost assumptions (see H.1; costs not yet deducted, numbers are optimistic)

#### C.3 Signal UI page ✅ Done (`SignalsPanel.tsx`)
- Displays all signals for the day, sorted by `signal_rank`, LONG/NEUTRAL labels color-coded
- Risk-on/Risk-off badge at top showing `regime_mult` (macro multiplier)
- Columns: Score / Quality / News burst / Earnings / AH Gap / Analyst Δ1m / Inst Δ
- 2026-06-16: added 3 D-series columns + regime badge (`quant_api` DailySignalEvent fields expanded in sync)
- Still needed: separate grouped display for watch list / avoid list (currently a single sorted table)

### Advanced (second priority)

#### C.4 Position tracking ✅ Done (`research/track_positions.py`)
- `positions` collection records simulated holdings: `symbol`, `entry_date`, `entry_price`, `entry_score`,
  `entry_rank`, `days_held`, `current_return`/`exit_return`, `exit_trigger`
- Idempotent: each run fully rebuilds state from `daily_signals` + `stock_prices_history`
- `PositionsPanel.tsx` displays current position status + unrealized P&L (already in quant_ui)
- 2026-06-20 validation: 19 positions (16 open/3 closed), PANW +62.6% / LRCX +50.5% leading

#### C.5 Exit alerts ✅ Done + extended 2026-06-16
- Existing triggers: `max_hold` (60 days), `score_below_exit`, `sentiment_reversal`, `earnings_miss`
- **Two new D-series triggers added**: `analyst_downgrade` (`analyst_buy_ratio_chg_1m` < -10%), `inst_outflow` (`inst_holding_pct_chg` < -1% QoQ)
- Alerts written to `alerts` collection; UI red dot / email push still pending (currently script print + collection query only)

#### C.6 Paper trading record ✅ Baseline done (i.e. C.4 `track_positions.py`)
- Daily `daily_signals` Top-N used to simulate entries; unrealized P&L updated daily with real prices
- Total running days are still short; 3-6 month out-of-sample performance is accumulating (ongoing task, not one-time development)

#### C.7 ETL data quality checks ✅ Baseline done (`research/data_quality_check.py`)
- Existing checks: news volume, price freshness, feature freshness, key field NULL rate thresholds
  (`quality_score`/`full_ratio`/`close`/`past_ret_20d`), signal freshness
- Integrated with scheduler to auto-run at 09:00 daily (`ENABLE_DATA_QUALITY_JOB`)
- Still needed: model training IC anomaly detection (2-sigma alert), write results to `quant_api` instead of script output only

#### C.8 ETL unit tests ✅ Partially done (`tests/test_feature_build.py`)
- Covered: `aggregate_news_features` (counts/ratios/rolling windows),
  `aggregate_llm_sentiment_features` (weighted sentiment/earnings beat signal/empty input edge case),
  `quality_score` ranking, date parsing/bucketing utility functions
- Still needed: `attach_price_labels` (forward-return calculation) dedicated tests,
  `compute_score` pipeline tests, macro factor derivation (macro_vix_pctile / macro_risk_on) tests,
  D-series `attach_*_features` function tests, coverage reporting (target >70%)

#### C.9 Factor analysis report ✅ Done (`research/factor_analysis.py`)
- Implemented: IC decay table (multi-horizon), year-by-year IC breakdown, SHAP feature importance
  (including LLM vs traditional factor contribution split)
- Updated: long-short portfolio + year-by-year Sharpe/drawdown chart implemented in `factor_analysis.py` walk-forward section ✅
- Still needed (optional enhancement): factor correlation matrix (VIF redundancy check)

### Full productionization path (time estimates)

```
Stage 7 services running end-to-end (1 week)
  → C.2 risk metrics + C.9 factor report (3 days)  ← can discuss in interview immediately
  → C.8 ETL unit tests (3 days)                    ← proves reliability in DE interviews
  → C.7 data quality checks (2 days)               ← production-grade credibility
  → C.1 daily signal automation (3 days)           ← project starts "moving"
  → C.3 signal UI page (3 days)                    ← visualized decisions
  → C.4 position tracking (1 week)                 ← real decision support
  → C.5 exit alerts (3 days)                       ← loop closed
  → C.6 paper trading (ongoing)                    ← accumulate real-world performance
  → project productionized + interview-ready ✓
```

---

## D. Research Layer Extensions (improve model quality)

### D.1 Macro Regime features ✅ Done 2026-06-16
- VIX level/percentile, 10Y interest rate, US dollar index, SPY 20-day trend
- Fields: `macro_vix`, `macro_vix_pctile_252d`, `macro_vix_change_5d`, `macro_tnx`,
  `macro_tnx_change_20d`, `macro_spy_ret_20d`, `macro_spy_above_200ma`,
  `macro_risk_on`, `macro_is_high_vol`, `macro_dxy_change_20d`
- In 2026 holdout, `macro_tnx` is the #2 ranked feature in LightGBM feature importance
- `macro_risk_on`/`macro_vix_pctile_252d` integrated into `score_daily_signals.py` as
  regime multiplier (risk-on ×1.20, high VIX ×0.85) — H.2.1 baseline implemented
- Deliverable: `macro_collector/collector.py`

### D.2 Alternative Data — Retail Sentiment ✅ Done 2026-06-16
- Integrated StockTwits public API (no Reddit developer approval required, bypassing the registration rejection issue)
- New factors: `retail_msg_count`, `retail_bull_ratio`, `retail_sent_score`,
  `retail_sentiment_divergence` (= retail_sent_score − avg_sentiment_3d)
- Coverage starts from 2025-12 only, fill rate low (~3-4%), requires ongoing accumulation;
  2026-06-20 review: IC reversed from +0.09 to -0.10 (N still small, no conclusion yet, **weights not adjusted**)
- Deliverable: `retail_collector/collector.py`

### D.3 Earnings filing text mining (10-K/10-Q) — Skipped
- Reason: requires LLM to parse large volumes of long text; engineering effort is high, lower priority than other D items
- Status: [ ] Not developing for now

### D.4 Analyst rating change factor ✅ Done 2026-06-16
- Finnhub `/api/v1/stock/recommendation`, accessed via `urllib` (under VPN, `requests`/`curl_cffi` failed; only `urllib.request.urlopen` works)
- New factors: `analyst_buy_ratio`, `analyst_sell_ratio`, `analyst_consensus_score`,
  `analyst_buy_ratio_chg_1m`
- Cross-sectional IC: `analyst_buy_ratio_chg_1m` 5d=+0.155 / 20d=+0.151, one of the cleanest signals in the D series
- Deliverable: `analyst_collector/collector.py`

### D.5 Options market signals — Skipped
- Reason: CBOE PCR free historical data only goes back to 2019, cannot cover recent backtest window
- Status: [ ] Not developing for now

### D.6 Short Interest (short squeeze pressure) — Skipped
- Reason: FINRA website returns bot protection page under VPN, unable to scrape
- Status: [ ] Not developing for now

### D.7 Institutional holdings change (13F) ✅ Done 2026-06-16
- `edgartools`, tracking Vanguard/StateStreet/Fidelity holdings for the most recent 2 quarters
- New factors: `inst_holding_pct` (aggregate holding percentage across institutions), `inst_holding_pct_chg` (QoQ,
  computed separately for each institution's most recent two periods then averaged, avoiding misalignment from different institutions' disclosure cycles)
- Cross-sectional IC: `inst_holding_pct_chg` 60d=**+0.198**, strongest single factor in the D series
- In 2026 holdout, the single feature contributing most to LightGBM regression IC (IC=+0.337)
- Integrated as `inst_outflow` exit trigger in `track_positions.py`
- Deliverable: `inst_13f_collector/collector.py`

### D.8 Pre-market / after-hours price signals ✅ Done 2026-06-16
- yfinance 1-minute data (`prepost=True`), chunked into 7-day windows to avoid "8-day limit" error
- New factors: `pm_gap`, `ah_gap`, `ah_volume_ratio` (`pm_volume_ratio` removed — yfinance pre-market volume field is always 0, cannot compute a valid ratio)
- Cross-sectional IC: `ah_gap` 5d=**+0.227**, the strongest short-term signal in the D series
- yfinance 1-minute history capped at 30 days; scheduler accumulates daily — as of 2026-06-20 there are 23 trading days
- Deliverable: `premarket_collector/collector.py`

### D-series end-to-end integration (2026-06-16 ~ 2026-06-20)
All completed items are wired through the full pipeline:
`daily_symbol_features.py` (features) → `train_baseline_models.py` / `backtest_news_factor.py`
(training; 60d target significantly outperforms 20d; 2026 holdout IC=0.73 vs historical baseline 0.04-0.05) →
`score_daily_signals.py` (scoring weights + macro regime multiplier) → `track_positions.py`
(`analyst_downgrade`/`inst_outflow` new exit triggers) → `quant_api`/`quant_ui`
(DailySignalEvent expanded with 8 new D-series fields; SignalsPanel shows regime badge + 3 new columns).

`backtest_portfolio.py` full-history backtest (2015-2026, reusing `score_daily_signals.compute_score`):
60d Sharpe=0.70, 20d Sharpe=0.84, both outperforming SPY (Sharpe 0.46-0.54) — but these results are primarily driven by the existing LLM/earnings factors; D-series data windows are short, so their contribution to historical backtest is not yet fully reflected; true incremental impact requires attribution analysis after data accumulates.

**Ops fix**: On 2026-06-20 discovered `com.quant.scheduler` (old) and `com.quant.pipeline-scheduler`
(new) were both running as launchd jobs, causing all collection tasks to execute twice daily. The old job has been unloaded (plist moved to `~/Quant_trade_backups/`).

---

## E. Engineering Layer Completion (interview bonus / project completeness)

### E.1 REST API documentation (Swagger / OpenAPI)
- Add Swagger annotations to all `quant_api` endpoints; generate interactive documentation
- Open `/swagger-ui.html` during interview demo to directly showcase API design
- Status: [ ] Pending development (1 day)

### E.2 CI/CD Pipeline (GitHub Actions)
- Auto-run on every push: Python lint (ruff) + unit tests + Docker build
- main branch protection: PRs must pass CI before merging
- Status: [ ] Pending development (2 days)

### E.3 Docker image version management
- Tag quant_api / quant_ui / quant_data images with git commit hash
- docker-compose pins versions, no `latest` tag
- Support one-command rollback to previous stable version
- Status: [ ] Pending development (1 day)

### E.4 K8s deployment configuration
- Convert docker-compose to Kubernetes YAML (Deployment / Service / ConfigMap)
- No need to actually run K8s; having the config files is enough for interviews
- Bonus for DE / Platform engineer interviews at large companies
- Status: [ ] Pending development (3 days)

### E.5 Data lineage diagram (OpenLineage)
- Use OpenLineage or hand-written lineage JSON to describe:
  `news_articles` → `company_match` → `news_articles_company_matched_v2`
  → `daily_symbol_features_company_matched_v2` → `daily_signals`
- Visualize data flow; a common DE interview topic
- Status: [ ] Pending development (2 days)

### E.6 WebSocket real-time push
- When a position triggers an exit condition, UI pops an alert in real time (no page refresh needed)
- Spring Boot WebSocket + quant_ui frontend subscribe
- Corresponds to the WebSocket infrastructure for 5.1.2 Backtest orchestration API
- Status: [ ] Pending development

### E.7 Project README + architecture diagram
- Write a complete README: project background, architecture diagram, quick start, key metrics
- Architecture diagram: data flow → feature engineering → model → signals → UI
- Required before any interview; first impression on GitHub
- Status: [ ] Pending development (1 day)

### E.8 Demo video / screenshots
- Record a 2-3 minute demo video: UI signal page + backtest results + factor analysis charts
- Put at the top of README as a GIF or YouTube link
- Status: [ ] Pending development

---

## F. AI Engineering Layer (AI Engineer / MLE interview bonus)

### F.1 Prompt Engineering evaluation framework
- Compare labeling accuracy of different prompt templates (use 100 manually-annotated articles as ground truth)
- Output evaluation report: precision / recall / confusion matrix (event_type classification)
- Proves the LLM pipeline is "validated", not just run arbitrarily
- Status: [ ] Pending development (2 days)

### F.2 Vector database news semantic search (RAG)
- Qdrant already deployed; add embeddings for `news_articles_company_matched_v2`
- Endpoint: `/search?q=NVIDIA earnings beat Q3` → returns relevant articles + LLM summary
- Embedding model: `nomic-embed-text` (LM Studio local)
- Add search box to quant_ui
- Status: [ ] Pending development (corresponds to 5.3.1; prototype in 2-3 days)

### F.3 Model interpretability report (SHAP)
- Run SHAP analysis on LightGBM model
- Output: factor contribution ranking, per-stock prediction explanation (why AAPL gets a high score)
- Interview demo killer: not just "IC=0.059" but also "driven primarily by avg_sentiment_5d and earnings_recency_weight"
- Status: [ ] Pending development (1 day)

### F.4 Multi-Agent research assistant (LangGraph)
Extend the existing `quant_langchain` container from LLMChain to a LangGraph stateful agent graph.

**Agent architecture (4-node graph):**
```
User input: "Analyze recent NVDA news and give a trading recommendation"
    ↓
data_agent      → calls quant_api to fetch news + price + sentiment factors
    ↓
analysis_agent  → reads IC ranking + sentiment distribution, LLM interprets recent events
    ↓
strategy_agent  → generates entry/exit rules (combining LLM scores + quantitative signals)
    ↓
risk_agent      → position sizing, stop-loss conditions, final output: position recommendation JSON
```

**Current state (existing foundation):**
- `quant_langchain/main.py`: placeholder endpoints `/api/workflow/generate-spec` and `/api/workflow/generate-tasks` already exist, but only do prompt → JSON with no execution capability
- `quant_api/StrategyService.java`: already calls langchain-agent; can directly serve as tools server
- Missing: LangGraph agent loop, tool registration, state persistence

**Development steps:**
1. Install langgraph, refactor `quant_langchain/main.py` (2 days)
2. Register tools: `search_news(symbol, days)` → calls quant_api; `get_features(symbol)` → MongoDB; `run_backtest(rules)` → Python
3. Implement 4-node graph + state passing (2 days)
4. Add ReAct loop: analysis_agent can decide whether it needs more data (1 day)
5. Integrate Qdrant RAG (after F.2 is done): data_agent can semantically search relevant news (1 day)

**Demo flow (for interviews):**
- Input: "Has AAPL had any negative news in the last 30 days? Should I hold a position now?"
- data_agent fetches news + sentiment distribution
- analysis_agent finds `avg_sentiment_5d = -0.3`, identifies regulatory risk event
- strategy_agent recommends "wait and watch; build position after sentiment recovers"
- risk_agent adds: if holding, stop-loss at -3%, max position size 5%

**Interview value:** covers ReAct reasoning + tool use + state graph — core competency for MLE / AI Engineer roles
- Status: [ ] Pending development (2 weeks)

### F.6 Upgrade rule_validator to a real Agent (ReAct loop)
**Current state:** `quant_data/tools/rule_validator_agent.py` has a prototype of "LLM judges → tool call → re-judges", but the sequence is hardcoded with no reasoning loop.

**Upgrade direction:**
- Validation failure → agent autonomously decides: fetch more articles / modify prompt / mark as ambiguous
- When handling ambiguous names from `ambiguous_names.py`, agent proactively searches for company background to decide
- Change from "pipeline" to "ReAct agent with memory"

**Development steps:**
1. Rewrite `rule_validator_agent.py` with LangGraph; add tools: `fetch_article(url)`, `search_web(company)`, `label_ambiguous(name, reason)`
2. Agent decides on its own whether more evidence is needed, up to N steps
3. Output trace: which steps each rule validation went through and the final reasoning basis

- Status: [ ] Pending development (3 days)

### F.7 Airflow adaptive scheduling Agent
**Current state:** All 4 DAGs in `airflow/dags/` are static time-triggered, with no adaptive capability.

**Upgrade direction:**
- Add `quality_monitor_dag`: after the pipeline runs each day, agent analyzes:
  - Is news volume abnormally low? (data source down?)
  - Is LLM label disagreement rate increasing? (model degradation?)
  - Has signal IC been declining for 5 consecutive days? (needs retraining?)
- Agent outputs decision: trigger re-run / switch to stronger model / send alert / auto-submit retraining job

**Development steps:**
1. Create `quality_monitor_dag.py`; BranchPythonOperator decides based on metrics (1 day)
2. Add LLM decision node: send anomaly metric summary to LLM, output recommended action (1 day)
3. Integrate Kafka: agent decision → publish to topic → downstream consumes (1 day)

- Status: [ ] Pending development (3 days)

### F.8 LLM annotation active learning Agent (disagreement sample handling)
**Current state:** Gemma + Qwen disagreement rate is 22.7%; merged blindly via Snorkel Dawid-Skene voting.

**Upgrade direction:**
- For high-uncertainty samples where label_model_probs < 0.7, agent actively intervenes:
  - Sends the original text + both model outputs together to a stronger LLM (Qwen3-27B / GPT-4o)
  - Agent produces the final label + explanation ("model A mislabeled because the article is about earnings, not a company event")
  - Added as hard negative examples to the FinBERT training set
- Expected: disagreement sample accuracy improved from ~60% to ~85%

**Development steps:**
1. Filter samples where `llm_disagreement=1 AND llm_label_model_probs < 0.7` (~190K records)
2. Write `active_learning_agent.py`: batch-submit to stronger model, parse output, write back to MongoDB
3. Update `llm_sentiment_final` field, rerun feature build
4. Compare IC before and after

- Status: [ ] Pending development (4 days)

### F.5 FinBERT fine-tuning (3.5.5)
- Train on Gemma+Qwen consensus labels (~650K samples)
- Three heads: sentiment regression + event_type 7-class + signal_strength 3-class
- Inference speed: ~3.9 art/s → ~1000+ art/s (200x speedup)
- Status: [ ] Pending development (1-2 weeks)

### F.9 Rule Optimization Agent (Iterative Self-Improving Loop)

**Goal**: After news collection, automatically evaluate collection quality by stratified sampling across years, computing precision/recall with an LLM judge, diagnosing failure modes, modifying rules, and repeating — converging on optimal rules without manual tuning.

This directly addresses the known SLM Phase 3 "incidental mention" false positive problem (Morgan Stanley survey cited in Apple articles, da Vinci art in ISRG articles, etc.) and eliminates the need for manual rule maintenance.

**Agent architecture (4-phase loop):**

```
Round N:
  ┌─ Sampler    → stratified sample: 10 articles × 10 years × target symbols
  ├─ Evaluator  → LLM judge (Claude/GPT-4o) labels each: TP / FP / FN
  ├─ Diagnoser  → groups errors by failure type:
  │               - incidental mention FP
  │               - wrong company FP (similar name)
  │               - relevant article FN (missed match)
  │               - rule too aggressive (over-filtered)
  └─ Modifier   → proposes rule changes; applies them; logs diff

Repeat up to 10 rounds; stop when F1 improvement < 1% between rounds.
```

**What can be modified per round:**
| Rule layer | Modification type |
|---|---|
| SLM Phase 3 prompt (`slm_filter.py`) | Strengthen "primary subject" constraint; add incidental-mention examples |
| `ambiguous_names.py` entries | Add/remove disambiguation entries for specific symbols |
| Relevance threshold | Adjust score cutoff per symbol type (large-cap vs small-cap) |
| Special per-symbol rules | Add blocklist keywords that generate systematic FPs |

**Ground truth strategy:**
- Primary judge: Claude Opus (silver standard; low cost per article, fully automated)
- Escalation: articles where judge confidence < 0.7 → flag for human review
- Batch size: 100 articles per round (10/year × 10 years); full evaluation costs ~$0.50/round

**Sampling strategy:**
```python
# Stratified sample: balanced across years and symbols
years = range(2015, 2025)          # 10 years
articles_per_year = 10             # 10 per year = 100 total per round
symbols_sampled = 5                # rotate symbols each round to avoid overfitting
# → 100 articles judged per round × 10 rounds = 1000 total labels
```

**Output per round:**
- Accuracy report: precision / recall / F1, broken down by year and by error type
- Rule diff: unified diff of what changed vs previous round
- Convergence tracker: F1 trend across rounds (stop condition: ΔF1 < 1%)

**Stored in MongoDB:**
```json
{
  "round": 3,
  "timestamp": "2026-07-15T10:00:00Z",
  "metrics": { "precision": 0.87, "recall": 0.91, "f1": 0.89 },
  "error_breakdown": { "incidental_mention_fp": 6, "wrong_company_fp": 2, "fn_missed": 4 },
  "rules_changed": ["slm_filter.py prompt v3", "ambiguous_names: +MSFT_survey_blocklist"],
  "sample_ids": ["article_id_1", "..."]
}
```

**Development tasks:**
1. `tools/rule_optimizer_agent.py` — main agent loop (LangGraph or simple Python loop)
2. `tools/news_sampler.py` — stratified sampling from `news_articles_company_matched_v2`
3. `tools/llm_judge.py` — call Claude API, parse TP/FP/FN verdict + reason
4. `tools/rule_modifier.py` — apply proposed changes to SLM prompt / ambiguous_names.py, write diff to MongoDB
5. `tools/accuracy_reporter.py` — compute metrics, print round summary, decide continue/stop
6. Integrate into `scheduler/task.py` as a weekly or on-demand job

**Execution modes:**
- `--mode eval` (evaluation only; no rule changes) — runs before any deploy to get baseline
- `--mode optimize` (full loop; modifies rules) — run after adding new symbols or when FP rate spikes
- `--mode audit --round 3` (inspect a specific round's samples and verdicts)

**Interview framing:**
*"Built a self-improving news validation agent that iteratively samples historical articles across 10 years, evaluates relevance with an LLM judge, diagnoses failure modes (incidental mention FP, wrong-company FP, missed matches), and modifies SLM prompts and disambiguation rules — converging from F1=0.XX to F1=0.XX in 10 rounds."*

**Relates to:** F.6 (rule_validator ReAct), F.8 (active learning), project_slm_optimization (incidental mention FP)

- Status: 🟡 Developed, not yet tested (2026-07-13)
  - `tools/llm_judge.py` — LLM judge (Claude API + local SLM fallback, TP/FP + fp_type + regex proposal)
  - `tools/rule_optimizer.py` — main loop (sample → judge → diagnose → propose → patch → repeat)
  - `news_collectors/gdelt/special_rules/ambiguous_names.py` — patch loader added (merges `tools/rule_optimizer_patches.json` at init)

---

## Consolidated Priority Table (all pending items)

| Priority | Item | Interview Value | Practical Value | Effort | Status |
|---|---|---|---|---|---|
| ⭐⭐⭐ | **H.1 Backtest with transaction costs + liquidity filter** | Quant essential | 🔴 Real returns | 2 days | [ ] Pending |
| ⭐⭐⭐ | H.2 Market Regime detection (VIX filter) | Quant essential | 🔴 IC stability | 4 days | 🟡 Baseline done (regime_mult); dynamic weight switching pending |
| ⭐⭐⭐ | H.3 Paper Trading engine + stop-loss | Quant essential | 🔴 OOS validation | 4 days | 🟡 Engine + partial exit triggers done; -5% stop-loss / OOS IC monitoring pending |
| ⭐⭐⭐ | Stage 7 Airflow + Kafka end-to-end | DE critical | High | 1 week | [ ] Pending (daily scheduling runs via launchd, not Airflow) |
| ⭐⭐⭐ | C.2 Risk metrics (Sharpe / drawdown) | Quant essential | High | - | ✅ Done |
| ⭐⭐⭐ | C.9 Factor analysis report (IC/IR/SHAP) | Quant essential | Medium | - | ✅ Done |
| ⭐⭐⭐ | C.8 ETL unit tests | DE essential | Medium | 3 days | 🟡 Partially done; earnings/D-series tests + coverage reporting still needed |
| ⭐⭐⭐ | E.7 README + architecture diagram | All interviews | Medium | 1 day | ✅ Done (2026-07-15) |
| ⭐⭐⭐ | Stage 7 MLflow actual runs | DE/MLE | Medium | 1 day | ✅ Done (2026-07-15) — 8 runs logged (Ridge/LightGBM/Ensemble × 20d+60d) |
| ⭐⭐ | H.4 Signal quality monitoring (rolling IC) | Quant strong | 🟡 Signal health | 2 days | [ ] Pending |
| ⭐⭐ | C.1 Daily signal automation | Medium | Extremely high | 3 days | ✅ Done (launchd, not Airflow) |
| ⭐⭐ | C.3 Signal UI page | Medium | Extremely high | 3 days | ✅ Done |
| ⭐⭐ | F.2 RAG news search (Qdrant) | AI essential | High | 3 days | [ ] Pending |
| ⭐⭐ | F.3 SHAP interpretability | MLE strong | Medium | 1 day | ✅ Done (factor_analysis.py) |
| ⭐⭐ | F.4 LangGraph multi-agent research assistant | AI Engineer must-have | High | 2 weeks | [ ] Pending |
| ⭐⭐ | F.8 Active learning Agent (disagreement samples) | MLE+AI | High | 4 days | [ ] Pending |
| ⭐⭐ | **F.9 Rule Optimization Agent (iterative eval→modify loop)** | AI Engineer strong | 🔴 Fixes FP/FN in rule layer | 5-7 days | 🟡 Developed, not tested |
| ⭐⭐ | E.2 CI/CD GitHub Actions | DE strong | Medium | 2 days | [ ] Pending |
| ⭐⭐ | C.7 Data quality checks | DE strong | High | 2 days | ✅ Done (data_quality_check.py) |
| ⭐⭐ | B quant bonus: Long-short portfolio | Quant strong | Medium | 3 days | [ ] Pending |
| ⭐⭐ | B quant bonus: Beta neutralization | Quant strong | Medium | 2 days | [ ] Pending |
| ⭐⭐ | F.5 FinBERT fine-tuning | MLE strong | High | 1-2 weeks | [ ] Pending |
| ⭐ | F.6 rule_validator ReAct Agent | AI bonus | Medium | 3 days | [ ] Pending |
| ⭐ | F.7 Airflow adaptive scheduling Agent | DE+AI | Medium | 3 days | [ ] Pending |
| ⭐⭐⭐ | D.1 Macro Regime features (merged into H.2) | Quant essential | 🔴 Real utility | 4 days | ✅ Done |
| ⭐ | E.4 K8s configuration | DE bonus | Low | 3 days | [ ] Pending |
| ⭐ | E.5 Data lineage diagram | DE bonus | Low | 2 days | [ ] Pending |
| ⭐ | E.6 WebSocket real-time push | Backend bonus | High | 3 days | [ ] Pending |
| ⭐ | D.2 Retail sentiment | Quant bonus | Medium | 3 days | ✅ Done (StockTwits, not Reddit) |
| ⭐ | F.1 Prompt evaluation framework | MLE bonus | Medium | 2 days | [ ] Pending |
| ⭐ | E.8 Demo video | All bonus | High | 0.5 days | [ ] Pending |
| — | D.4 Analyst rating changes | Quant bonus | Medium | 3 days | ✅ Done |
| — | D.7 Institutional 13F holdings change | Quant bonus | High | 3 days | ✅ Done (strongest single factor, 60d IC=+0.20) |
| — | D.8 Pre-market / after-hours price signals | Quant bonus | High | 1 day | ✅ Done (strongest short-term factor, 5d IC=+0.23) |

---

# Stage 5. Engineering Extensions (post-research)

Goal: after the research stages above produce a validated signal on a
50+ symbol universe, harden the system into a production-quality full-stack
showcase. These items are intentionally scoped to exercise three skill
areas — **Java backend**, **data engineering**, and **AI/ML engineering** —
so the project becomes a multi-role interview artifact.

Do not start these until:
- universe is expanded to 50+ symbols
- year-stability analysis is complete
- at least one signal has persistent positive excess return

## 5.1 Java Backend Track

### 5.1.1 Event-driven signal distribution (Kafka)
- Spring Boot microservice publishes daily signals to a Kafka topic
- Subscribers: UI push, alerting, risk control, audit log
- Requirements: idempotent consumers, dead-letter queue, schema registry
- Deliverables:
  - `quant_signal_publisher` service
  - Kafka broker + schema registry in docker-compose
  - consumer reference implementations
- Resume framing:
  *"Designed event-driven signal distribution system using Kafka,
  supporting 40+ symbols with sub-second latency and idempotent consumers."*

### 5.1.2 Backtest orchestration API
- REST endpoint to submit backtest jobs → async worker → WebSocket push
  results
- Resilience4j circuit breakers, Bucket4j rate limiting, Redis result cache
- Keycloak JWT for auth, role-based access (researcher vs admin)
- Deliverables:
  - new endpoints under `quant_api`
  - async job queue (RabbitMQ, already present)
  - WebSocket progress stream
- Resume framing:
  *"Built async backtest orchestration service with circuit breakers,
  rate limiting, and real-time result streaming over WebSocket."*

### 5.1.3 Admin dashboard backend
- CRUD for: stock universe, job schedules, backtest experiments,
  model versions
- Audit log of every mutation
- Expand `quant_ui` with admin views
- Resume framing:
  *"Implemented admin control plane with auditable CRUD for universe,
  scheduling, and model version management."*

## 5.2 Data Engineering Track

### 5.2.1 Airflow orchestration (replace `scheduler/task.py`)
- Migrate the current monolithic scheduler to Airflow DAGs
- One DAG per domain: news ingest, price history, feature build,
  backtest, model refresh
- Backfill support, SLA monitoring, retry policies
- Deliverables:
  - `airflow/dags/` directory
  - docker-compose service for Airflow webserver + scheduler
- Resume framing:
  *"Migrated monolithic scheduler to Airflow DAGs orchestrating 15+
  daily ETL jobs with backfill, SLA, and retry policies."*

### 5.2.2 Streaming news pipeline (Kafka + Flink)
- Replace batch GDELT collector with streaming ingest
- Flink jobs: dedup → content extraction → SLM relevance → fan-out to
  MongoDB (raw) and Qdrant (embeddings)
- Exactly-once semantics where possible
- Deliverables:
  - Flink job modules
  - Kafka topic design doc
- Resume framing:
  *"Built streaming news ingestion pipeline with Kafka + Flink
  processing 1M+ articles with exactly-once semantics."*

### 5.2.3 MLflow experiment tracking
- Every backtest + model training run logged to MLflow
- Parameters, metrics, artifacts, model registry
- Deliverables:
  - MLflow server in docker-compose (Postgres backend)
  - wrapper in `research/` scripts to log runs
- Resume framing:
  *"Implemented experiment tracking with MLflow, managing 100+ backtest
  runs and model versions across Ridge / HistGB / ensemble baselines."*

### 5.2.4 dbt transformations
- Model the MySQL / MongoDB feature derivations as dbt models
- Lineage graphs, schema tests, documentation
- Deliverables:
  - `dbt_project/` with staging / intermediate / mart layers
- Resume framing:
  *"Introduced dbt for data transformations, providing lineage graphs,
  automated tests, and self-documenting feature logic."*

## 5.3 AI / ML Engineering Track

### 5.3.1 RAG news search system
- Qdrant already deployed for Dify — add a second collection for
  news article embeddings
- Service: `/search?q=NVIDIA earnings beat` → semantic recall + LLM summary
- Embedding model: `text-embedding-nomic-embed-text-v1.5` (already in
  LM Studio)
- Deliverables:
  - embedding batch job for news_articles_company_matched_v2
  - FastAPI / Spring search service
  - simple UI search page
- Resume framing:
  *"Built RAG system over 1M+ financial news articles using Qdrant
  and local LLM, supporting semantic search and summarization."*

### 5.3.2 Multi-agent research assistant (LangGraph)
- Expand the existing `langchain-agent` container into a multi-agent graph:
  - data agent: pulls relevant news / prices / earnings
  - analysis agent: extracts events, generates hypotheses
  - strategy agent: proposes factor / portfolio adjustments
  - risk agent: validates constraints
- Orchestrated with LangGraph (stateful, branching)
- Deliverables:
  - `quant_langchain/` extension with agent graph
  - demo UI or CLI
- Resume framing:
  *"Designed multi-agent financial research system using LangGraph with
  specialized agents for news analysis, signal generation, and risk
  assessment."*

### 5.3.3 SLM fine-tuning for company-match
- Use the v2 matched dataset as labeled training data
- Fine-tune Qwen3.5-4B on the company-relevance task (LoRA)
- Benchmark: base vs fine-tuned on held-out set
- Target: +5% accuracy or -30% latency at equal accuracy
- Deliverables:
  - `research/finetune_slm.py` training script
  - before/after benchmark report
- Resume framing:
  *"Fine-tuned Qwen3.5-4B with LoRA on 500K+ article classification
  dataset, improving accuracy from X% to Y% and reducing inference
  latency by Z%."*

### 5.3.4 LLM-based feature generation
- Use a larger LLM to generate enrichment features per article:
  - sentiment (-1..+1)
  - event type (earnings / product / M&A / regulation / macro)
  - entity list (companies / people)
- Store as additional fields in v2
- Feed into feature pipeline as new daily aggregates
- Deliverables:
  - batch enrichment job
  - 5-10 new daily features
- Resume framing:
  *"Added LLM-based feature extraction (sentiment, event type, entities)
  to enrich 1M+ articles, contributing to a +X% lift in model IC."*

## 5.4 Integration: End-to-End MLOps Showcase

Single flagship deliverable that ties the three tracks together:

```
Kafka (news) → Flink (clean + SLM) → Feature Store
  → Model Inference API (Spring Boot) → Kafka (signals)
  → UI + Alerts + Audit
  → Prometheus / Grafana / OpenTelemetry (monitoring)
```

This touches: event streaming, Java backend, ML inference, observability,
cloud-native deployment — interview-ready for backend, data engineer,
and AI / ML engineer roles simultaneously.

Deliverables:
- architecture diagram (`docs/architecture.md`)
- all pieces deployable via a single `docker-compose up`
- README walkthrough of the full signal lifecycle

## 5.5 Execution Order (Stage 5)

Recommended build order, each item ~1-2 weeks:

1. **5.3.1 RAG news search** — quickest AI win, reuses existing Qdrant
2. **5.2.1 Airflow migration** — immediate data-engineering credential
3. **5.2.3 MLflow tracking** — low effort, high resume value
4. **5.1.1 Kafka signal publisher** — core event-driven piece
5. **5.3.2 Multi-agent system** — deepest AI showcase
6. **5.2.2 Flink streaming** — headline data-engineering item
7. **5.3.3 SLM fine-tuning** — after enough v2 data accumulates
8. **5.4 End-to-end integration** — final polish

## 5.6 Resume Skill Coverage Matrix

| Skill                       | Covered by              |
|-----------------------------|-------------------------|
| Spring Boot microservices   | 5.1.1, 5.1.2, 5.1.3     |
| Kafka / event-driven        | 5.1.1, 5.2.2, 5.4       |
| Resilience patterns         | 5.1.2                   |
| WebSocket / real-time       | 5.1.2                   |
| Airflow / orchestration     | 5.2.1                   |
| Flink / stream processing   | 5.2.2                   |
| MLflow / experiment tracking| 5.2.3                   |
| dbt / data modeling         | 5.2.4                   |
| RAG / vector search         | 5.3.1                   |
| LangGraph / multi-agent     | 5.3.2                   |
| LLM fine-tuning (LoRA)      | 5.3.3                   |
| LLM inference at scale      | 5.3.4, existing pipeline|
| Observability / monitoring  | 5.4                     |
| Docker / K8s deployment     | 5.4                     |

---

## Stage 6 — GDELT GKG Full Index Import into MongoDB (Pending development)

### 6.1 Background and Goal

Matching historical GDELT data for new stocks currently requires scanning ~350,000 CSV files in batches, taking about 13 hours.
Goal: import 7 key columns from the CSVs into MongoDB and build a `$text` full-text index, reducing new-stock keyword matching from **13 hours → seconds**.

### 6.2 GDELT CSV column structure (7 columns retained)

| Col # | Field Name | MongoDB Field | Purpose |
|------|-----------|-----------------|------------------------|
| 1    | DATE      | `date`          | Publication date (YYYYMMDDHHMMSS) |
| 4    | URL       | `url`           | Article URL |
| 7    | V1Themes  | `themes`        | Theme keywords (primary matching) |
| 11   | V1Persons | `persons`       | Person names |
| 13   | V1Orgs    | `orgs`          | Organizations / company names (secondary matching) |
| 15   | V1Tone    | `tone`          | GCAM sentiment score (auxiliary feature) |
| 23   | AllNames  | `all_names`     | All entity names |

In the original 27-column CSV, column 17 (GCAM) accounts for 72% of the size; retaining 7 columns reduces data from 7TB to approximately **160 GB** (Parquet estimate), and MongoDB BSON + WiredTiger compression brings it to roughly **300–600 GB** (including `$text` index).

### 6.3 Development tasks

#### 6.3.1 Import script (`tools/gdelt_import_to_mongo.py`)

- Multi-process parallel read of `.csv` files (~350,000) from Data24T + Data6T
- Extract 7 columns per row; construct MongoDB document
- Write to collection `quant_data.gkg_index` in batches (batch_size=5000)
- Supports resume from checkpoint: imported files tracked in `gkg_import_progress` collection
- Deduplication: URL as unique key (`url` unique index)

```python
# Example document structure
{
    "date": "20200314120000",
    "url": "https://example.com/article",
    "themes": "ECON_BANKRUPTCY;ENV_SOLAR;...",
    "persons": "Tim Cook;Elon Musk;...",
    "orgs": "Apple Inc;Tesla;...",
    "tone": "2.5,-1.2,3.7,...",
    "all_names": "Apple-COMPANY;Tesla-COMPANY;..."
}
```

#### 6.3.2 Index creation

```javascript
// Compound text index (primary matching)
db.gkg_index.createIndex(
    { themes: "text", persons: "text", orgs: "text", all_names: "text" },
    { name: "gkg_text_idx", weights: { orgs: 10, themes: 5, all_names: 3, persons: 1 } }
)

// Auxiliary query indexes
db.gkg_index.createIndex({ date: 1 })
db.gkg_index.createIndex({ url: 1 }, { unique: true })
```

#### 6.3.3 New matching pipeline

Replace the current CSV batch scan in `historical_collector.py` with:

1. **Step 1**: `db.gkg_index.find({ $text: { $search: "keyword" } })` → returns candidate URL list in seconds
2. **Step 2**: Query `news_articles` collection; reuse already-fetched body text
3. **Step 3**: Fetch missing URLs (many old URLs from 2016-2020 may be dead; handle 404 gracefully)
4. **Step 4**: Write to `news_articles_company_matched_v2`

#### 6.3.4 Validation

- Pick 10 stocks from the existing 60; re-match using the new pipeline
- Compare URL hit rate against the current CSV approach to verify consistency
- Check `$text` index recall rate (any missed matches?)

### 6.4 Storage estimate

| Item | Estimated size |
|-----------------------------|------------|
| Raw CSV (7TB) → 7 columns extracted | ~160 GB |
| MongoDB BSON (uncompressed) | ~200 GB |
| After WiredTiger compression (~3.5x) | ~60 GB |
| `$text` index | ~200–400 GB |
| **Total (including index)** | **~300–600 GB** |

Once import is validated, the original CSVs can be deleted, saving **6TB+** of disk space.

### 6.5 Notes

- MongoDB server needs at least **600 GB** of free space (including temporary space for index build)
- Building the `$text` index will take **several hours** (570M records); recommended to create it all at once after import is complete
- V1Tone (`tone` field) format is comma-separated multiple scores; use the first value (overall sentiment score)
- Many old URLs (2016–2020) are likely dead; the URL fetching stage must handle 404/timeout gracefully and log the dead-link rate
- Import script must support parallel reads from multiple disks (Data24T + Data6T) to avoid single-disk I/O bottleneck

### 6.6 Prerequisites

- [ ] MongoDB server confirmed to have 600 GB+ free space
- [ ] Data24T and Data6T are mounted and readable
- [ ] After import, run regression tests on 10 sample stocks; delete original CSVs only after passing

---

# Stage 7 — New Mac Validation + Scheduled Task Verification + Execution Records (Pending development)

## 7.1 Background

A 48GB RAM Mac has been purchased and data/code migration is complete. A comprehensive validation of all services and scheduled tasks on the new machine is needed, along with adding observability for execution results.

## 7.2 Service validation

### 7.2.1 Docker service verification
- [ ] Verify all containers start correctly: mongo6 / quant_api / quant_ui / quant_data /
      airflow-webserver / airflow-scheduler / kafka / mlflow, etc.
- [ ] Check inter-service network connectivity (project-net internal communication)
- [ ] Verify external volume mounts are correct (MongoDB, MySQL, Airflow logs, etc. data intact)
- [ ] Confirm `.env` connection addresses are valid on the new machine

### 7.2.2 Airflow scheduled tasks running end-to-end (not just validation, must actually run)
- [ ] Check scheduling times and dependencies for all DAGs in `airflow/dags/`
- [ ] Manually trigger each DAG; confirm full end-to-end execution succeeds (not just "defined")
- [ ] Confirm Docker socket mount path is correct (`/var/run/docker.sock`)
- [ ] Wait for at least one automatic schedule trigger; confirm timed execution works
- [ ] Check DAG execution logs; fix any failing tasks
- [ ] Goal: news collection / feature build / model training — all three main DAGs running stably

### 7.2.3 Kafka actually running (not just deployed, must have real data flow)
- [ ] Confirm kafka container is healthy and reachable at `kafka:9092`
- [ ] Create required topics (e.g. `quant.signals`, `quant.news`)
- [ ] Implement at least one producer: push to `quant.signals` topic after daily signal generation
- [ ] Implement at least one consumer: consume signals and write to `daily_signals` collection or trigger alert
- [ ] Verify message flow through kafka-ui (port 15070)
- [ ] Goal: signal generation → Kafka → consume and write to DB — full pipeline running

### 7.2.4 MLflow actual run recording
- [ ] Confirm mlflow container is healthy (port 15050)
- [ ] Enable `--mlflow-uri` in `train_baseline_models.py`; run one training and record
- [ ] Verify parameters, metrics, and model files are visible in MLflow UI
- [ ] Goal: every model training run is recorded; IC / Top5 excess return is traceable

### 7.2.5 LLM inference validation
- [ ] Verify LM Studio / Ollama is available on the new Mac and models are loaded
- [ ] Verify `SLM_API_URL` environment variable points to the correct endpoint
- [ ] Run a small batch of `llm_enrich_articles.py` to confirm inference is working

## 7.3 Execution result recording (UI / API)

Current problem: execution results for Airflow and Python scripts (success/failure, rows processed, duration, key metrics) have no unified visibility entry point, making troubleshooting difficult.

### 7.3.1 quant_api execution log endpoint
- [ ] Add `pipeline_runs` collection (MongoDB) or table (MySQL) to record each task execution:
  ```json
  {
    "task_name": "feature_build",
    "started_at": "2026-05-22T21:00:00Z",
    "finished_at": "2026-05-22T21:03:00Z",
    "status": "success",
    "rows_processed": 134642,
    "error_message": null,
    "metadata": { "collection": "daily_symbol_features_company_matched_v2" }
  }
  ```
- [ ] quant_api provides REST endpoints: `POST /api/pipeline-runs` (write) and
      `GET /api/pipeline-runs` (query most recent N records)
- [ ] Each Python script (daily_symbol_features.py, llm_enrich_articles.py, etc.)
      calls the endpoint to report results after completion

### 7.3.2 quant_ui execution history page
- [ ] Add a "Task Execution Records" page in the UI showing:
  - Task name, status (success/failure), start/end time, rows processed
  - Execution trend chart for the last 30 days
  - Failed tasks highlighted + error message display
- [ ] Support filtering by task name and time range

### 7.3.3 Model training result persistence
- [ ] After each `train_baseline_models.py` run, write per-horizon IC / Top5 excess return
      to `model_results` collection with timestamp and feature set version tag
- [ ] quant_ui displays model historical result trends to facilitate comparison across different feature versions

## 7.4 Priority

1. Docker baseline service validation (blocks all subsequent work)
2. **Airflow DAG actually running** (not just defined; must run stably — key interview proof)
3. **Kafka producer/consumer actually running** (signal → topic → consume and write to DB full pipeline)
4. **MLflow actual run recording** (every training run is traceable)
5. quant_api execution log endpoint (pipeline_runs write to DB)
6. quant_ui execution history page
7. Model result persistence

---

## G. Reference Resources (compiled 2026-05-22)

### G.1 LLM sentiment factor / alpha mining papers

| Paper | Key points |
|------|------|
| [Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets (arXiv 2508.07408)](https://arxiv.org/pdf/2508.07408) | Interpretable LLM quantitative framework, event-aware sentiment factor construction; useful to compare against this project's LLM tagging approach |
| [Interpretable ML for Macro Alpha: News Sentiment Case Study (arXiv 2505.16136)](https://arxiv.org/pdf/2505.16136) | FinBERT + GDELT → FX/treasury strategy, OOS Sharpe >4; validates the GDELT + LLM approach |
| [AlphaAgent: LLM-Driven Alpha Mining (arXiv 2502.16789)](https://arxiv.org/html/2502.16789v2) | LLM-automated alpha factor mining including anti-decay mechanism |
| [Automate Strategy Finding with LLM in Quant Investment (arXiv 2409.06289)](https://arxiv.org/html/2409.06289v1) | LLM-driven strategy discovery pipeline; useful prompt design reference |

### G.2 CBOE options data (free download)

- **PCR historical CSV** (for D.5 Put/Call Ratio factor):
  `https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv`
- **CBOE historical data main page**:
  `https://www.cboe.com/us/options/market_statistics/historical_data/`
- Usage: load daily PCR data directly with `pd.read_csv(url)`; no registration required

### G.3 FinBERT fine-tuning resources (for F.5 / Stage 3.5.5)

| Resource | Key points |
|------|------|
| [Fine-Tuning FinBERT for Sector-Specific Financial News (MDPI Electronics 2025)](https://www.mdpi.com/2079-9292/14/23/4680) | zero-shot F1=0.555 → after fine-tuning F1=**0.707**; includes sector-specific training approach, directly applicable as training script reference |
| [ProsusAI/finbert on HuggingFace](https://huggingface.co/ProsusAI/finbert) | Official FinBERT pre-trained weights, ready to load |
| [Efficient FinBERT via Quantization (ACL FinNLP 2025)](https://aclanthology.org/2025.finnlp-2.6.pdf) | INT8/INT4 quantization compression; runnable locally on Mac M-series chips |

### G.4 SEC EDGAR 13F Python tools (for D.7)

| Library | Description |
|----|------|
| [edgartools](https://github.com/dgunning/edgartools) | **Preferred**, free and open-source; parses 13F into structured Python objects; data goes back to 2005; `pip install edgartools` |
| [sec-api-python](https://github.com/janlukasschroeder/sec-api-python) | Paid SDK with free trial quota; suitable for bulk production scraping |

---

## H. Live Trading System Roadmap (from research → truly usable)

### Background: why live trading is not yet possible

Backtest results (long-short annualized +21.7%, Sharpe 0.85) look promising, but the following issues would cause real money to lose:

| Problem | Specific manifestation | Risk level |
|------|---------|---------|
| IC year-over-year instability | IC was negative in 2022/2024/2025, signal reversed | 🔴 Fatal |
| No transaction costs | Backtest does not deduct commissions + slippage; real returns reduced by 30-40% | 🔴 Fatal |
| Signal lag | LLM processing takes several hours after news is published | 🟡 Serious |
| No position sizing | Equal-weight allocation; single-stock risk is unconstrained | 🟡 Serious |
| No out-of-sample validation | All data used for backtest; overfitting risk | 🟡 Serious |
| Only 100 stocks | Concentration too high; liquidity risk | 🟠 Moderate |

---

### H.1 Fix backtest realism (🔴 must do first)

**H.1.1 Add transaction cost model**
- One-way commission: 0.05% (standard broker level)
- Slippage model: 0.3% for small-cap, 0.1% for large-cap (tiered by market cap)
- Modify `backtest_news_factor.py` and `backtest_event_driven.py`:
  ```python
  net_return = gross_return - commission * 2 - slippage * 2
  ```
- Goal: determine how much of the long-short annualized return remains after costs
- Status: [ ] Pending development (1 day)

**H.1.2 Add liquidity filter**
- Exclude stocks with average daily volume < $5M (avoid execution failure)
- Add `avg_daily_volume` feature; filter at position selection
- Status: [ ] Pending development (0.5 days)

**H.1.3 Add position limit constraints**
- Max position per stock: 5% (avoid over-concentration)
- Max position per sector: 25%
- Modify `score_daily_signals.py` to add position constraint logic
- Status: [ ] Pending development (0.5 days)

---

### H.2 Resolve IC instability (🔴 must do first)

**H.2.1 Market Regime detection**

Year-by-year IC analysis reveals the signal is only effective in "trending markets":
```
2022 (bear market / high VIX): IC = -0.035 → disable signal
2023 (recovery / low VIX):     IC = +0.163 → enable signal  
2024 (choppy):                  IC = -0.022 → disable signal
2025 (choppy):                  IC = -0.020 → disable signal
```

Implementation:
- Data source: `yfinance` for daily VIX data
- Regime definitions:
  - **Trending market**: VIX < 20 AND SPY 20-day moving average trending up → enable LLM sentiment factor weighting
  - **Choppy / panic**: VIX > 25 → reduce position to 50%, switch to mean-reversion factors
- New field: `market_regime` (trend / volatile / crisis)
- Adjust composite_score weights by regime in `score_daily_signals.py`
- Status: [ ] Pending development (2 days)

**H.2.2 Dynamic factor weights**
- Different factor weights for different regimes:
  - Trending market: LLM sentiment weight 40%, momentum weight 30%, earnings weight 30%
  - Choppy market: LLM sentiment weight 10%, mean-reversion weight 50%, earnings weight 40%
- Add `market_regime` as a feature during model training, or train separate models per regime
- Status: [ ] Pending development (2 days)

---

### H.3 Paper Trading validation (🔴 must do, run for at least 3-6 months)

**Goal**: validate signals in a real market environment and accumulate out-of-sample data

**H.3.1 Paper Trading engine**
- Execute automatically after market close each day:
  1. Read the top 10 stocks by composite_score from that day's `daily_signals`
  2. Check regime: if VIX > 25, skip or halve the position
  3. Simulate order entry; record entry price (use next day's open price)
  4. Write to `paper_positions` collection
- Fields: `symbol`, `entry_date`, `entry_price`, `size`, `score_at_entry`, `regime_at_entry`
- Status: [ ] Pending development (2 days)

**H.3.2 Paper Trading performance tracking**
- Update unrealized P&L daily
- Compute real out-of-sample IC (each day's actual holdings: predicted vs actual performance)
- Target metrics:
  - OOS IC > 0.02 (signal is effective in real markets)
  - OOS Sharpe > 0.5 (strategy is executable)
  - 3 consecutive months of positive excess return (signal is stable)
- Status: [ ] Pending development (1 day)

**H.3.3 Stop-loss and exit logic**
- Per-stock stop-loss: auto-exit if the position drops -5% from entry
- Sentiment reversal stop-loss: if `avg_sentiment_5d` turns negative during holding, actively exit
- Time stop-loss: auto-exit if holding exceeds `max_hold_days` (default 45 days)
- Status: [ ] Pending development (1 day)

---

### H.4 Signal quality monitoring (🟡 important)

**H.4.1 Real-time IC monitoring**
- Compute rolling IC over the most recent 20 trading days every day
- If IC stays below 0 for 5 consecutive days → trigger alert, pause new position entries
- Write to `signal_quality` collection; UI displays IC trend chart
- Status: [ ] Pending development (1 day)

**H.4.2 Model drift detection**
- Weekly: compare the latest feature distribution against the training-time distribution (KL divergence)
- If distribution shift exceeds threshold → trigger retraining reminder
- Status: [ ] Pending development (1 day)

---

### H.5 Live trading readiness checklist

All of the following must be satisfied before considering real capital:

- [ ] Paper trading has run for **at least 3 months**, OOS IC > 0.02
- [ ] Backtest Sharpe **remains > 0.5** after including transaction costs
- [ ] Regime detection works correctly; automatically reduces position when VIX > 25
- [ ] Stop-loss logic tested (simulated extreme market conditions)
- [ ] Max position per stock ≤ 5%; overall leverage = 1x (no leverage)
- [ ] Broker API connected (Interactive Brokers / Alpaca)
- [ ] First live-trading capital ≤ 10% of total capital (testing phase)

---

### H. Roadmap overview

```
Now (research validation complete)
    ↓
H.1 Fix backtest realism (3 days)        ← know the real returns
H.2 Regime detection (4 days)            ← know when to use the signal
H.3 Paper Trading (1 month to get going) ← real-market validation
    ↓ continue running 3-6 months
H.4 Signal monitoring (2 days)           ← keep signal effective
    ↓ paper trading metrics pass
H.5 Live trading checklist all ✓
    ↓
Small live position (≤10% of total capital)
    ↓ 6 months of stable excess return
Truly usable quantitative trading system ✓
```

---

## G. Strategy Generation Complete Implementation Plan (quant_langchain)

> Current state: `quant_langchain/main.py` already has the endpoint framework for `/api/workflow/generate-spec` and `/api/workflow/generate-tasks`, but all three core features are stub implementations that need to be completed.

---

### G.1 RAG — Replace keyword matching with vector retrieval (3 days)

**Current problems:**
- `retrieve_context()` scores by keyword token overlap, not semantic retrieval
- The knowledge directory has only 1 file (26 lines); RAG has almost no real effect
- Low hit rate: user input "RSI mean reversion" cannot match if phrased differently

**Goal:** Replace keyword matching with Qdrant (already running in docker-compose) for true vector semantic retrieval

**Development steps:**

1. **Add knowledge base documents** (1 day)
   - `knowledge/strategies/` — strategy templates: RSI, MACD, momentum, mean-reversion examples
   - `knowledge/factors/` — factor descriptions: IC per factor, use cases, parameter ranges
   - `knowledge/risk/` — risk control rules: stop-loss, position, max drawdown defaults
   - `knowledge/modules/` — detailed parameter descriptions for each module in MODULE_CATALOG

2. **Integrate Qdrant vector retrieval** (2 days)
   - Install: `langchain-qdrant`, `sentence-transformers`
   - Embed knowledge documents with `all-MiniLM-L6-v2`; write to Qdrant
   - Replace `retrieve_context()` with `qdrant_client.search(query_vector, top_k=4)`
   - Build index asynchronously at startup; support incremental updates

**Done when:** Input "RSI buy when oversold" retrieves the RSI strategy template document

---

### G.2 MCP — Tools actually executable (4 days)

**Current problems:**
- Module paths in `MODULE_CATALOG` (e.g. `quant_langchain.features.momentum`) do not exist
- LLM only "sees" the tool list in the prompt; it cannot actually call them
- No tool call loop: module names in the LLM-generated spec are hallucinated and never executed

**Goal:** Wrap each tool in MODULE_CATALOG as a callable function; close the loop of LLM → tool call → result returned

**Development steps:**

1. **Implement tool functions** (2 days)

```python
# quant_langchain/tools/market_data.py
def fetch_market_data(symbols: list, timeframe: str, lookback_days: int) -> dict:
    """Call quant_api /api/signals to fetch historical signals and price data"""
    resp = requests.get(f"{QUANT_API}/api/signals", params={"symbols": symbols})
    return resp.json()

# quant_langchain/tools/feature_builder.py
def build_features(indicators: list, window: int) -> dict:
    """Call quant_api /api/features to get feature data"""
    ...

# quant_langchain/tools/backtest.py
def backtest_strategy(spec: dict, initial_cash: int, fee_bps: int) -> dict:
    """Generate backtest summary based on spec (calls quant_data research module)"""
    ...
```

2. **Register tools with LangChain** (1 day)
   - Register each function with the `@tool` decorator
   - Change the `generate-spec` endpoint to use `AgentExecutor` so the LLM actually calls tools
   - Collect results from each tool call and assemble into the final spec

3. **MCP protocol wrapping (optional upgrade)** (1 day)
   - Add JSON schema description for each tool
   - Output format complies with MCP tool result spec
   - Prepares for future Claude MCP integration

**Done when:** Spec generation logs show `[Tool Call] fetch_market_data(AAPL, daily, 60)` → real data returned

---

### G.3 API Key routing — proactive tiering by task complexity (1 day)

**Current problems:**
- Uses `OpenAI` (text completion); should switch to `ChatOpenAI` (chat model)
- Routing logic only falls back when Ollama is down; no proactive model selection by task type
- `LLMChain` is a deprecated API; should migrate to LCEL (LangChain Expression Language)

**Goal:** Local small model handles simple tasks; cloud large model handles complex reasoning

**Routing rules:**

| Task | Model | Reason |
|------|------|------|
| generate-spec (simple strategy) | Qwen3-8B local | MODULE_CATALOG constrains output; structure is fixed |
| generate-spec (complex multi-factor) | Claude Sonnet / GPT-4o | Factor combination reasoning; small models hallucinate |
| generate-tasks (code generation) | Claude Sonnet / GPT-4o | High code quality required |
| chat (strategy Q&A) | Qwen3-8B local | Many conversation turns; cost-sensitive |

**Development steps:**

```python
# Switch to ChatOpenAI + LCEL
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate

def get_llm(task_type: str = "simple"):
    # Proactively use cloud model for complex tasks
    if task_type == "code_generation" and ANTHROPIC_API_KEY:
        return ChatAnthropic(model="claude-sonnet-4-6", api_key=ANTHROPIC_API_KEY)
    if task_type == "complex_spec" and OPENAI_API_KEY:
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
    # Default to local
    try:
        # health check Ollama...
        return ChatOllama(model=LOCAL_MODEL_NAME)
    except:
        return ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
```

**Done when:** generate-tasks endpoint logs show `[Router] task=code_generation → claude-sonnet-4-6`

---

### G.4 Priority and time estimate

| Priority | Feature | Effort | Value |
|--------|------|--------|------|
| 🔴 Highest | G.2 MCP tools actually executable | 4 days | Strategy generation changes from "hallucinated output" to "real execution" |
| 🟠 High | G.3 API Key routing refactor | 1 day | Code quality improvement; deprecated API removed |
| 🟡 Medium | G.1 RAG vector retrieval | 3 days | Retrieval precision improved; interview demo highlight |

**Total timeline: approximately 8 days**

Recommended order: G.3 (1 day, base layer) → G.2 (4 days, tool layer) → G.1 (3 days, knowledge layer)

**Estimated time from now to "truly usable": approximately 6-9 months** (of which 3-6 months is waiting for paper trading data to accumulate)
