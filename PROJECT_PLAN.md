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

Do next (in priority order):

1. **universe expansion** — most important; expand to 50+ symbols before
   further feature or model work; current 14-symbol results cannot be
   reliably confirmed
2. **year stability analysis** — run group-by-year on current full feature
   set; understand what drives 2022/2024 IC failures before adding more
   features
3. **earnings Layer 3** — surprise magnitude buckets and news × earnings
   interaction features; do after stability analysis confirms value

Do not do next:

- live trading automation
- complex deep learning or LLM-based modeling
- adding more data sources before validating signals on expanded universe
- switching local SLM model without accuracy benchmarking

## Notes

The clean matched-news dataset is research-usable at the current scale.

The two-layer earnings feature set has produced a meaningful model
improvement (Ridge `60d` IC: `0.073` → `0.121`).

The signal looks real but the universe is too small to confirm it with
confidence. Universe expansion is the gating step before any further
research investment makes sense.

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
