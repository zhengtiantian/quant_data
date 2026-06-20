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

### 3.5.1 Article-level event tagging (LLM batch annotation) ✅ 完成 2026-05-22

All 840,212 articles in `news_articles_company_matched_v2` tagged with:

- Pass A (Gemma): `llm_sentiment_a`, `llm_event_type_a`, `llm_signal_strength_a` — 100%
- Pass B (Qwen): `llm_sentiment_b`, `llm_event_type_b`, `llm_signal_strength_b` — 100%
- Snorkel merge: `llm_sentiment_final`, `llm_disagreement`, `llm_label_model_probs` — 100%
- 两模型一致率 77.3%；情绪均值 +0.296（整体偏正面）

Deliverable: `research/llm_enrich_articles.py`, `research/snorkel_label_merge.py`

Resume framing:
*"Enriched 840K financial news articles with two-pass LLM ensemble (Gemma +
Qwen), aggregated via Snorkel Label Model achieving 77.3% inter-model
agreement on sentiment, event type, and signal strength labels."*

### 3.5.2 Event-level daily features ✅ 完成 2026-05-22

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

| Horizon | Ridge IC | Ensemble IC | Ensemble Top5 超额 |
|---|---|---|---|
| 20d | 0.036 | 0.031 | +1.83% |
| 45d | 0.044 | 0.043 | +4.38% |
| 60d | 0.056 | 0.059 | +6.59% |

IC 随持仓期单调递增，60d 信号最强。

Deliverable: `research/daily_symbol_features.py`, `research/train_baseline_models.py`

### 3.5.3 Dynamic holding period backtest ✅ 完成 2026-05-22

Event-driven framework results (100 symbols, 2018–2026):

Best config: `min_hold=20d`, `max_hold=60d`, `sentiment_shift_exit=-0.35`

| 指标 | 事件驱动 | 固定20d基线 | 固定60d基线 |
|---|---|---|---|
| 平均持仓 | 13.7d | 20d | 60d |
| 超额收益 | +1.40% | +1.22% | +3.80% |
| 胜率 | 52.7% | 58.0% | 63.1% |
| 交易次数 | 390 | 2,060 | 2,020 |

事件驱动在 13.7 天内超过固定 20d 基线（+1.40% vs +1.22%），换手率更低。
退出原因：score_below_exit 69%，sentiment_reversal 17%（已合理）。
2022/2024 仍为弱年（宏观制度变化）。

Deliverable: `research/backtest_event_driven.py`

### 3.5.4 Multi-horizon label expansion ✅ 完成 2026-05-22

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
5. 3.5.5 Fine-tune small model — 待开发

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

1. **H.1 回测加交易成本+流动性过滤**（2天）— 当前 Sharpe 0.70-0.84 未扣成本，
   量化面试必问，性价比最高
2. **Stage 7 校验** — Airflow DAG 跑通、Kafka producer/consumer 实际运行、
   执行日志接口上线（仍是面试讲故事的最大 gap，C 系列已用 launchd 代替跑通了
   日常调度，但不是 Airflow）
3. **E.7 README + 架构图**（1天）— 全部面试都需要，目前缺失，最快出成果
4. **H.2.2 动态因子权重**（regime-aware）— 已有 regime_mult 基础乘数，
   需扩展为震荡市自动切换因子权重集
5. **H.3.2/H.3.3 Paper trading 完善** — 已有持仓追踪和退出触发，缺 OOS IC
   监控和 -5% 止损规则
6. **Stage 3.5.5 FinBERT 微调** — 用 Gemma+Qwen 一致标签训练，替代 LLM 推理管道

Completed:

- universe expansion to 100 symbols ✅
- year stability analysis ✅
- earnings Layer 1 / 2 / 3 ✅
- Ridge + LightGBM + Ensemble baseline (60d IC = 0.10, Top5 = +6.7%) ✅
- Stage 3.5.1 LLM article tagging (840K articles, Gemma + Qwen + Snorkel merge) ✅
- Stage 3.5.2 event-level daily features (134K rows, 100% LLM 覆盖) ✅
- Stage 3.5.3 event-driven backtest (min20d, +1.40% vs 固定20d +1.22%) ✅
- Stage 3.5.4 multi-horizon labels (10/15/30/45/60d) ✅
- **D.1/D.2/D.4/D.7/D.8 研究层扩展全部完成**（2026-06-16），D.3/D.5/D.6 跳过 ✅
- **C.1-C.7, C.9 全部完成基础版**（每日信号自动化 + UI + 仓位跟踪 + 退出提醒 +
  风险指标 + 数据质量检查 + 因子分析报告，2026-06-16~20）✅
- **H.2.1 regime multiplier 基础版**（score_daily_signals.py 已接入 macro_risk_on/
  macro_vix_pctile_252d 调整信号强度）✅
- 运维：清除重复 launchd scheduler job，避免采集任务每日重复执行两次 ✅

Do not do next:

- live trading automation（实盘，距离真正可用还差 H.1-H.5 全部完成 + 3-6个月 paper trading）
- Stage 5 engineering before Stage 7 校验完成

## Notes

The clean matched-news dataset is research-usable at the current scale.

The two-layer earnings feature set has produced a meaningful model
improvement (Ridge `60d` IC: `0.073` → `0.121`).

Signal IC 在 100-symbol universe 下 60d = 0.059，比 14-symbol 时的 0.121
更可信（更大 universe 稀释了过拟合）。2022/2024 弱年由宏观制度变化驱动，
信号本身可信。

---

# 面试路线图

## A. Data Engineer 面试路线图

### 当前已有（可以直接讲的）
- MongoDB 840K+ 文章 + 675M GKG 倒排索引（大规模文档存储 + 全文检索）
- Python ETL pipeline：新闻采集 → 公司匹配 → feature build（多数据源、增量/全量）
- LLM 批量推理管道（840K 条，两 Pass + Snorkel Label Model）
- Docker Compose 多服务编排（MongoDB / MySQL / Kafka / Airflow / MLflow / Qdrant）
- Airflow DAG 定义（DAG 结构、任务依赖、SLA）

### 关键 gap — 必须补（面试会被追问）

| 缺什么 | 为什么重要 | 对应 Stage | 工作量 |
|---|---|---|---|
| Kafka 没有实际 producer/consumer | DE 必问，"部署了但没用"说服力不足 | Stage 7 / 5.1.1 | 3天 |
| Airflow DAG 没有真正跑通 | "写了 DAG" ≠ "跑通了" | Stage 7 | 1周 |
| MLflow 没有实际记录 run | 工具会用但没产出，面试问 run 结果答不上 | Stage 7 / 5.2.3 | 1天 |
| 没有 ETL 单元测试 | DE 面试必问 pipeline 可靠性保障 | 新增 | 3天 |
| 没有数据质量检查 | 生产级 pipeline 标配，无 → 被质疑可靠性 | 新增 | 2天 |
| 没有幂等性设计文档 | DE 必考：重跑 pipeline 数据会不会重复？ | 新增（补文档） | 1天 |

### 加分项 — 做了更强，不做也能面

| 项目 | 价值 | 对应 Stage | 工作量 |
|---|---|---|---|
| Flink 流处理管道 | 大厂 DE 高频技能，最具区分度 | 5.2.2 | 2周 |
| dbt 数据血缘 | 数据建模规范，FinTech DE 常考 | 5.2.4 | 1周 |
| Prometheus + Grafana 监控 | 可观测性，SRE 面也能用 | 5.4 | 3天 |
| Schema Registry（Kafka） | Avro schema 演进，大规模数据治理 | 5.1.1 | 2天 |
| CI/CD for data pipeline | GitHub Actions 跑测试 + lint | 新增 | 2天 |
| 数据血缘图 | 哪张表依赖哪个 source，可用 OpenLineage | 新增 | 3天 |

### DE 面试能讲的完整故事（补完 gap 后）
*"构建了覆盖 100 支股票的金融新闻处理系统：GDELT 原始数据（675M 条）建立
全文索引；Python ETL 清洗匹配 840K 篇文章，含幂等性设计和数据质量检查；
Airflow 调度 5 条 DAG 每日增量更新；Kafka 发布每日交易信号（producer/consumer
完整链路）；MLflow 追踪 100+ 次模型实验；整套系统 Docker Compose 一键部署。"*

---

## B. FinTech / 量化金融面试路线图

### 当前已有（可以直接讲的）
- Walk-forward 验证（非回测过拟合），IC=0.059（60d），100 symbol universe
- 多因子模型：新闻质量 + 动量 + 财报事件 + LLM 情感
- 事件驱动回测框架（min_hold=20d，+1.40% 超额，优于固定持仓基线）
- LLM 双模型集成打标（Gemma + Qwen + Snorkel，77.3% 一致率）
- 840K 篇文章，情感 / 事件类型 / 信号强度三维标签

### 关键 gap — 必须补（量化面试必问）

| 缺什么 | 为什么重要 | 工作量 |
|---|---|---|
| Sharpe ratio / 最大回撤 / Sortino | 量化岗必问，没有数字 = 策略不完整 | 2天 |
| 换手率 + 交易成本模型（万5假设） | 策略扣成本后是否仍盈利 | 1天 |
| 年化收益 vs SPY Buy-and-Hold 对比 | 策略有没有超过被动持有 | 1天 |
| 因子 IC 衰减曲线（Autocorrelation） | 信号多少天后失效，决定换仓频率 | 1天 |
| Information Ratio（IC / std(IC)） | 信号稳定性，IC=0.059 够用但要展示 IR | 0.5天 |
| Long-short 组合（不只是 Top-N） | 量化基金标配，只做 long = 不完整 | 3天 |
| 因子相关性矩阵（VIF/冗余检测） | 避免多重共线性，模型可解释性 | 1天 |
| 逐年 Sharpe + 回撤图 | 2022/2024 弱年的风险量化说明 | 1天 |

### 加分项 — 做了更强

| 项目 | 价值 | 对应 Stage | 工作量 |
|---|---|---|---|
| FinBERT 微调（200x 推理加速） | AI+金融交叉，独特亮点 | 3.5.5 | 1周 |
| 因子归因分析（SHAP feature importance） | 可解释性，面试现场能展示 | 新增 | 1天 |
| Beta 中性化 / 市场中性组合 | 机构级量化标配 | 新增 | 2天 |
| 波动率加权仓位（非等权） | 比等权更精细的组合构建 | 新增 | 1天 |
| 流动性过滤（市值下限） | 避免小票滑点，真实可执行性 | 新增 | 0.5天 |
| RAG 新闻语义搜索 | "语义搜索+金融"亮点 | 5.3.1 | 1周 |
| 多 agent 研究助手 | AI工程深度展示 | 5.3.2 | 2周 |
| Paper trading 实盘验证记录 | Out-of-sample 真实表现，面试最有说服力 | C.6 | 持续 |

### 量化面试能讲的完整故事（补完 gap 后）
*"在 100 支科技股 universe 上构建新闻驱动多因子模型：840K 篇文章经
Gemma+Qwen 双模型 LLM 集成打标（77.3% 一致率）；Walk-forward 验证
60d Rank IC=0.059，Information Ratio=X.X；事件驱动持仓框架平均 13.7 天
实现 +1.40% 超额（扣交易成本后 +X.X%），年化 Sharpe=X.X，最大回撤 X%，
Long-short 年化超额 X%。"*

---

## C. 项目实用化路线图（新闻事件 → 买卖 Hold 决策）

### 最小可用闭环（优先做）

#### C.1 每日信号生成自动化 ✅ 完成（接入 launchd scheduler，非 Airflow）
- `score_daily_signals.py` 接入 `scheduler/task.py`，每日 08:30 自动跑
  （`daily_symbol_features` 08:00 → `score_daily_signals` 08:30 → `track_positions` 08:40）
- 对 100 个股票打分，写入 `daily_signals` 集合
- 字段：`symbol`, `trade_date`, `composite_score`, `signal_rank`, `signal_type`(LONG/NEUTRAL),
  `regime_mult`，以及 D 系列上下文字段（`ah_gap`, `analyst_buy_ratio_chg_1m`,
  `inst_holding_pct_chg`, `retail_sent_score`, `macro_risk_on`, `macro_vix` 等）
- 注：仍用 launchd，非 Airflow（Stage 7 / 5.2.1 迁移 Airflow 仍待开发）

#### C.2 风险指标完善 ✅ 完成（`research/backtest_portfolio.py`）
- 已有：Sharpe ratio（年化，无风险利率 4%）、最大回撤、胜率、年化收益 vs SPY
- 2026-06-16 验证结果（全历史 2015-2026，复用 `score_daily_signals.compute_score`）：
  - 20d 持仓：Sharpe 0.84，年化 25.8%，vs SPY Sharpe 0.54/12.1%
  - 60d 持仓：Sharpe 0.70，年化 24.3%，vs SPY Sharpe 0.46/11.6%
- 待补：Sortino ratio、换手率 / 交易成本假设（见 H.1，仍未扣成本，数字偏乐观）

#### C.3 信号 UI 页面 ✅ 完成（`SignalsPanel.tsx`）
- 显示当日全部信号，按 `signal_rank` 排序，LONG/NEUTRAL 标签着色
- 顶部 Risk-on/Risk-off 徽章显示 `regime_mult`（宏观乘数）
- 列：Score / Quality / News burst / Earnings / AH Gap / Analyst Δ1m / Inst Δ
- 2026-06-16 新增 D 系列 3 列 + regime badge（`quant_api` DailySignalEvent 同步扩展字段）
- 待补：watch list / avoid list 的独立分组展示（目前是单一排序表格）

### 进阶可用（第二优先）

#### C.4 仓位跟踪 ✅ 完成（`research/track_positions.py`）
- `positions` 集合记录模拟持仓：`symbol`, `entry_date`, `entry_price`, `entry_score`,
  `entry_rank`, `days_held`, `current_return`/`exit_return`, `exit_trigger`
- 幂等：每次运行从 `daily_signals` + `stock_prices_history` 全量重建状态
- `PositionsPanel.tsx` 展示当前持仓状态 + 浮盈浮亏（已存在于 quant_ui）
- 2026-06-20 验证：19 个持仓（16 开/3 平），PANW +62.6%／LRCX +50.5% 领涨

#### C.5 退出提醒 ✅ 完成 + 2026-06-16 扩展
- 已有触发：`max_hold`（60天）、`score_below_exit`、`sentiment_reversal`、`earnings_miss`
- **新增两个 D 系列触发**：`analyst_downgrade`（`analyst_buy_ratio_chg_1m` < -10%）、
  `inst_outflow`（`inst_holding_pct_chg` < -1% QoQ）
- 提醒写入 `alerts` 集合；UI 红点/邮件推送仍待开发（目前仅脚本打印 + 集合查询）

#### C.6 Paper trading 记录 ✅ 基础版完成（即 C.4 `track_positions.py`）
- 已按每日 `daily_signals` Top-N 模拟开仓，逐日用真实价格更新浮盈浮亏
- 累计运行天数尚短，3-6 个月 out-of-sample 表现仍在积累中（持续任务，非一次性开发）

#### C.7 ETL 数据质量检查 ✅ 基础版完成（`research/data_quality_check.py`）
- 已有检查：新闻量、价格新鲜度、特征新鲜度、关键字段 NULL 率阈值
  (`quality_score`/`full_ratio`/`close`/`past_ret_20d`)、信号新鲜度
- 已接入 scheduler 每日 09:00 自动跑（`ENABLE_DATA_QUALITY_JOB`）
- 待补：模型训练 IC 异常检测（2个标准差告警）、写入 `quant_api` 而非仅脚本输出

#### C.8 ETL 单元测试 ✅ 部分完成（`tests/test_feature_build.py`）
- 已覆盖：`aggregate_news_features`（计数/比率/滚动窗口）、
  `aggregate_llm_sentiment_features`（加权情感/earnings beat 信号/空输入边界）、
  `quality_score` 排名、日期解析/分桶工具函数
- 待补：`attach_earnings_event_features`（days_to/since 计算）专项测试、
  D 系列 `attach_*_features` 函数测试、覆盖率统计（目标 >70%）

#### C.9 因子分析报告 ✅ 完成（`research/factor_analysis.py`）
- 已实现：IC 衰减表（多 horizon）、按年 IC 分解、SHAP feature importance
  （含 LLM vs 传统因子贡献占比拆分）
- 待补：因子相关性矩阵（VIF 冗余检测）、Long-short 组合 + 逐年 Sharpe/回撤图
  （目前 Sharpe/回撤已在 `backtest_portfolio.py` 里，但是 Top-N long-only，非 long-short）

### 完整可用路线（时间估算）

```
Stage 7 服务跑通（1周）
  → C.2 风险指标 + C.9 因子报告（3天） ← 面试立刻能讲
  → C.8 ETL 单元测试（3天）            ← DE 面试证明可靠性
  → C.7 数据质量检查（2天）            ← 生产级可信度
  → C.1 每日信号自动化（3天）          ← 项目开始"动起来"
  → C.3 信号 UI 页面（3天）            ← 可视化决策
  → C.4 仓位跟踪（1周）               ← 真正辅助决策
  → C.5 退出提醒（3天）               ← 闭环完成
  → C.6 Paper trading（持续）          ← 积累真实表现
  → 项目实用化 + 面试两用 ✓
```

---

## D. 研究层扩展（提升模型质量）

### D.1 宏观 Regime 特征 ✅ 完成 2026-06-16
- VIX 水平/百分位、10Y 利率、美元指数、SPY 20日趋势
- 字段：`macro_vix`, `macro_vix_pctile_252d`, `macro_vix_change_5d`, `macro_tnx`,
  `macro_tnx_change_20d`, `macro_spy_ret_20d`, `macro_spy_above_200ma`,
  `macro_risk_on`, `macro_is_high_vol`, `macro_dxy_change_20d`
- 2026 holdout 中 `macro_tnx` 是 LightGBM 特征重要性排名第 2 的特征
- `macro_risk_on`/`macro_vix_pctile_252d` 已接入 `score_daily_signals.py` 作为
  regime multiplier（risk-on ×1.20，高 VIX ×0.85）—— H.2.1 的基础版已实现
- Deliverable: `macro_collector/collector.py`

### D.2 Alternative Data — 散户情绪 ✅ 完成 2026-06-16
- 接入 StockTwits 公开 API（无需 Reddit 开发者审核，绕开了注册被拒的问题）
- 新因子：`retail_msg_count`, `retail_bull_ratio`, `retail_sent_score`,
  `retail_sentiment_divergence`（= retail_sent_score − avg_sentiment_3d）
- 覆盖仅 2025-12 起，fill 率低(~3-4%)，需持续积累；
  2026-06-20 复核：IC 由 +0.09 反转为 -0.10（N 仍偏小，未下结论，**未调整权重**）
- Deliverable: `retail_collector/collector.py`

### D.3 财报原文挖掘（10-K/10-Q）— 跳过
- 原因：需要 LLM 解析大量长文本，工程量大，优先级低于其他 D 项
- 状态：[ ] 暂不开发

### D.4 分析师评级变化因子 ✅ 完成 2026-06-16
- Finnhub `/api/v1/stock/recommendation`，通过 `urllib`（VPN 下 `requests`/`curl_cffi` 失败，
  仅 `urllib.request.urlopen` 可用）
- 新因子：`analyst_buy_ratio`, `analyst_sell_ratio`, `analyst_consensus_score`,
  `analyst_buy_ratio_chg_1m`
- CS-IC（截面）：`analyst_buy_ratio_chg_1m` 5d=+0.155 / 20d=+0.151，全部 D 系列中最干净的信号之一
- Deliverable: `analyst_collector/collector.py`

### D.5 期权市场信号 — 跳过
- 原因：CBOE PCR 免费历史数据仅到 2019 年，无法覆盖近期回测窗口
- 状态：[ ] 暂不开发

### D.6 Short Interest（做空压力）— 跳过
- 原因：VPN 下 FINRA 网站返回 bot 防护页面，无法抓取
- 状态：[ ] 暂不开发

### D.7 机构持仓变化（13F） ✅ 完成 2026-06-16
- `edgartools`，跟踪 Vanguard/StateStreet/Fidelity 最近 2 个季度持仓
- 新因子：`inst_holding_pct`（跨机构汇总持仓比例）、`inst_holding_pct_chg`（QoQ，
  按机构各自最近两期分别计算后取平均，规避了不同机构披露周期错位的问题）
- CS-IC：`inst_holding_pct_chg` 60d=**+0.198**，全部 D 系列单因子中最强
- 2026 holdout 中是 LightGBM regression IC 贡献最大的单特征（IC=+0.337）
- 已接入 `track_positions.py` 的 `inst_outflow` 退出触发
- Deliverable: `inst_13f_collector/collector.py`

### D.8 盘前盘后价格信号 ✅ 完成 2026-06-16
- yfinance 1m 数据（`prepost=True`），7天窗口分块抓取规避"8天上限"报错
- 新因子：`pm_gap`, `ah_gap`, `ah_volume_ratio`（`pm_volume_ratio` 已移除——yfinance
  盘前 volume 字段恒为 0，无法计算有效比值）
- CS-IC：`ah_gap` 5d=**+0.227**，全部 D 系列单因子中最强短线信号
- yfinance 1m 历史上限 30 天，scheduler 每日累积，截至 2026-06-20 已有 23 个交易日
- Deliverable: `premarket_collector/collector.py`

### D 系列接入闭环（2026-06-16 ~ 2026-06-20）
全部完成项已贯穿整条 pipeline：
`daily_symbol_features.py`（特征）→ `train_baseline_models.py` / `backtest_news_factor.py`
（训练，60d 目标显著优于 20d，2026 holdout IC=0.73 vs 历史基线 0.04-0.05）→
`score_daily_signals.py`（打分权重 + macro regime 乘数）→ `track_positions.py`
（`analyst_downgrade`/`inst_outflow` 新退出触发）→ `quant_api`/`quant_ui`
（DailySignalEvent 新增 8 个 D 系列字段，SignalsPanel 展示 regime badge + 3 新列）。

`backtest_portfolio.py` 全历史回测（2015-2026，复用 `score_daily_signals.compute_score`）：
60d Sharpe=0.70，20d Sharpe=0.84，均跑赢 SPY（Sharpe 0.46-0.54）——但此结果主要由
既有 LLM/earnings 因子驱动，D 系列因数据窗口短，对历史回测贡献尚未充分体现，
真实增量需等待数据积累后再做 attribution。

**运维修复**：2026-06-20 发现 `com.quant.scheduler`（旧）与 `com.quant.pipeline-scheduler`
（新）两个 launchd job 同时运行，导致所有采集任务每天重复执行两次。已卸载旧 job
（plist 移至 `~/Quant_trade_backups/`）。

---

## E. 工程层补全（面试加分 / 项目完整性）

### E.1 REST API 文档（Swagger / OpenAPI）
- `quant_api` 所有接口加 Swagger 注解，生成可交互文档
- 面试演示时打开 `/swagger-ui.html` 直接展示 API 设计
- 状态：[ ] 待开发（1天）

### E.2 CI/CD Pipeline（GitHub Actions）
- 每次 push 自动跑：Python lint（ruff）+ 单元测试 + Docker build
- main branch 保护：PR 必须通过 CI 才能合并
- 状态：[ ] 待开发（2天）

### E.3 Docker 镜像版本管理
- quant_api / quant_ui / quant_data 镜像打 git commit hash tag
- docker-compose 固定版本，不用 `latest`
- 支持一键回滚到上一个稳定版本
- 状态：[ ] 待开发（1天）

### E.4 K8s 部署配置
- 把 docker-compose 转成 Kubernetes YAML（Deployment / Service / ConfigMap）
- 不需要真正跑 K8s，有配置文件面试就能讲
- 大厂 DE / Platform 面试加分
- 状态：[ ] 待开发（3天）

### E.5 数据血缘图（OpenLineage）
- 用 OpenLineage 或手写 lineage JSON 描述：
  `news_articles` → `company_match` → `news_articles_company_matched_v2`
  → `daily_symbol_features_company_matched_v2` → `daily_signals`
- 可视化数据流向，DE 面试高频考点
- 状态：[ ] 待开发（2天）

### E.6 WebSocket 实时推送
- 持仓触发退出条件时，UI 实时弹出提醒（不需要刷页面）
- Spring Boot WebSocket + quant_ui 前端 subscribe
- 对应 5.1.2 Backtest orchestration API 的 WebSocket 基础设施
- 状态：[ ] 待开发

### E.7 项目 README + 架构图
- 写完整 README：项目背景、架构图、快速启动、核心指标展示
- 架构图展示：数据流 → 特征工程 → 模型 → 信号 → UI
- 面试前必备，GitHub 展示第一印象
- 状态：[ ] 待开发（1天）

### E.8 Demo 视频 / 截图
- 录制 2-3 分钟演示视频：UI 信号页面 + 回测结果 + 因子分析图
- 放在 README 顶部 GIF 或 YouTube 链接
- 状态：[ ] 待开发

---

## F. AI 工程层（AI Engineer / MLE 面试加分）

### F.1 Prompt Engineering 评测框架
- 对比不同 prompt 模板的打标准确率（用人工标注的 100 篇作为 ground truth）
- 输出评测报告：precision / recall / confusion matrix（event_type 分类）
- 证明 LLM pipeline 是"经过验证的"，不是随便跑的
- 状态：[ ] 待开发（2天）

### F.2 向量数据库新闻语义搜索（RAG）
- Qdrant 已部署，对 `news_articles_company_matched_v2` 做 embedding
- 接口：`/search?q=NVIDIA earnings beat Q3` → 返回相关文章 + LLM 摘要
- Embedding 模型：`nomic-embed-text`（LM Studio 本地）
- quant_ui 加搜索框
- 状态：[ ] 待开发（对应 5.3.1，2-3天可出原型）

### F.3 模型可解释性报告（SHAP）
- 对 LightGBM 模型跑 SHAP analysis
- 输出：各因子贡献度排名、个股预测可解释（为什么给 AAPL 高分）
- 面试演示杀手锏：不只是"IC=0.059"，还能说"主要由 avg_sentiment_5d 和
  earnings_recency_weight 贡献"
- 状态：[ ] 待开发（1天）

### F.4 多 Agent 研究助手（LangGraph）
扩展现有 `quant_langchain` 容器，从 LLMChain 升级为 LangGraph 有状态 Agent 图。

**Agent 架构（4 节点图）：**
```
用户输入: "分析 NVDA 最近新闻，给出交易建议"
    ↓
data_agent      → 调 quant_api 拉新闻 + 价格 + 情感因子
    ↓
analysis_agent  → 读 IC 排名 + 情绪分布，LLM 解读近期事件
    ↓
strategy_agent  → 生成 entry/exit 规则（结合 LLM 评分 + 量化信号）
    ↓
risk_agent      → 仓位控制、止损条件、最终输出持仓建议 JSON
```

**当前状态（现有基础）：**
- `quant_langchain/main.py`：已有 `/api/workflow/generate-spec`、`/api/workflow/generate-tasks` 占位接口，但只做 prompt → JSON，无执行能力
- `quant_api/StrategyService.java`：已调用 langchain-agent，可直接作为 tools server
- 缺：LangGraph agent loop、工具注册、状态持久化

**开发步骤：**
1. 安装 langgraph，重构 `quant_langchain/main.py`（2天）
2. 注册工具：`search_news(symbol, days)` → 调 quant_api；`get_features(symbol)` → MongoDB；`run_backtest(rules)` → Python
3. 实现 4 节点图 + 状态传递（2天）
4. 加入 ReAct 循环：analysis_agent 可以决定是否需要更多数据（1天）
5. 接入 Qdrant RAG（F.2 完成后）：data_agent 可语义搜索相关新闻（1天）

**demo 演示流程（面试用）：**
- 输入："AAPL 最近 30 天有没有负面消息，现在该持仓吗？"
- data_agent 拉新闻 + 情感分布
- analysis_agent 发现 `avg_sentiment_5d = -0.3`，识别监管风险事件
- strategy_agent 给出"观望，等情绪回升再建仓"
- risk_agent 补充：如持仓，止损 -3%，最大仓位 5%

**面试价值：** 覆盖 ReAct 推理 + tool use + 状态图，是 MLE / AI Engineer 岗核心考点
- 状态：[ ] 待开发（2周）

### F.6 rule_validator 升级为真 Agent（ReAct 循环）
**现状：** `quant_data/tools/rule_validator_agent.py` 已有"LLM判断→工具调用→再判断"雏形，但是硬编码顺序，无推理循环。

**升级方向：**
- 验证失败 → agent 自主决定：抓更多文章 / 修改 prompt / 标记 ambiguous
- 处理 `ambiguous_names.py` 歧义词时，agent 主动搜索公司背景判断
- 从"流水线"改为"带 memory 的 ReAct agent"

**开发步骤：**
1. 用 LangGraph 改写 `rule_validator_agent.py`，加工具：`fetch_article(url)`、`search_web(company)`、`label_ambiguous(name, reason)`
2. Agent 自己决定是否需要更多证据，最多 N 步
3. 输出 trace：每条规则验证了哪些步骤、最终判断依据

- 状态：[ ] 待开发（3天）

### F.7 Airflow 自适应调度 Agent
**现状：** `airflow/dags/` 4 个 DAG 全是静态时间触发，无自适应能力。

**升级方向：**
- 新增 `quality_monitor_dag`：每天 pipeline 跑完后，agent 分析：
  - 新闻量是否异常低？（数据源挂了？）
  - LLM 标签分歧率是否升高？（模型性能退化？）
  - 信号 IC 是否连续 5 天下降？（需要重训？）
- Agent 输出决策：触发补跑 / 换用更强模型 / 发告警 / 自动提交 retraining job

**开发步骤：**
1. 建 `quality_monitor_dag.py`，BranchPythonOperator 根据指标决策（1天）
2. 加 LLM 决策节点：把异常指标摘要发给 LLM，输出建议动作（1天）
3. 接入 Kafka：agent 决策 → 发 topic → downstream 消费（1天）

- 状态：[ ] 待开发（3天）

### F.8 LLM 标注主动学习 Agent（分歧样本处理）
**现状：** Gemma + Qwen 两模型分歧率 22.7%，靠 Snorkel Dawid-Skene 盲目投票合并。

**升级方向：**
- 对 label_model_probs < 0.7 的高不确定样本，agent 主动介入：
  - 把原文 + 两个模型输出一起发给更强 LLM（Qwen3-27B / GPT-4o）
  - Agent 给出最终标签 + 解释原因（"model A 误判，因为文章主体是财报，非公司事件"）
  - 作为 hard negative examples 加入 FinBERT 训练集
- 预期：分歧样本准确率从 ~60% 提升至 ~85%

**开发步骤：**
1. 筛选 `llm_disagreement=1 AND llm_label_model_probs < 0.7` 样本（~190K条）
2. 写 `active_learning_agent.py`：批量发给强模型，解析输出，写回 MongoDB
3. 更新 `llm_sentiment_final` 字段，重跑 feature build
4. 对比前后 IC 变化

- 状态：[ ] 待开发（4天）

### F.5 FinBERT 微调（3.5.5）
- 用 Gemma+Qwen 一致标签（~650K 样本）训练
- 三个 head：sentiment regression + event_type 7分类 + signal_strength 3分类
- 推理速度：~3.9 art/s → ~1000+ art/s（200x 加速）
- 状态：[ ] 待开发（1-2周）

---

## 综合优先级表（所有待开发项）

| 优先级 | 项目 | 面试价值 | 实用价值 | 工作量 | 状态 |
|---|---|---|---|---|---|
| ⭐⭐⭐ | **H.1 回测加交易成本+流动性过滤** | 量化必需 | 🔴 真实收益 | 2天 | [ ] 待开发 |
| ⭐⭐⭐ | H.2 市场 Regime 检测（VIX过滤） | 量化必需 | 🔴 IC稳定性 | 4天 | 🟡 基础版完成（regime_mult），动态权重切换待开发 |
| ⭐⭐⭐ | H.3 Paper Trading 引擎+止损 | 量化必需 | 🔴 OOS验证 | 4天 | 🟡 引擎+部分退出触发已完成，-5%止损/OOS IC监控待开发 |
| ⭐⭐⭐ | Stage 7 Airflow + Kafka 跑通 | DE 关键 | 高 | 1周 | [ ] 待开发（日常调度用 launchd 跑通，非 Airflow） |
| ⭐⭐⭐ | C.2 风险指标（Sharpe/回撤） | 量化必需 | 高 | - | ✅ 已完成 |
| ⭐⭐⭐ | C.9 因子分析报告（IC/IR/SHAP） | 量化必需 | 中 | - | ✅ 已完成 |
| ⭐⭐⭐ | C.8 ETL 单元测试 | DE 必需 | 中 | 3天 | 🟡 部分完成，待补 earnings/D系列测试+覆盖率 |
| ⭐⭐⭐ | E.7 README + 架构图 | 全部面试 | 中 | 1天 | [ ] 待开发 |
| ⭐⭐⭐ | Stage 7 MLflow 实际跑 | DE/MLE | 中 | 1天 | [ ] 待开发 |
| ⭐⭐ | H.4 信号质量监控（rolling IC）| 量化强 | 🟡 信号健康 | 2天 | [ ] 待开发 |
| ⭐⭐ | C.1 每日信号自动化 | 中 | 极高 | 3天 | ✅ 已完成（launchd，非 Airflow） |
| ⭐⭐ | C.3 信号 UI 页面 | 中 | 极高 | 3天 | ✅ 已完成 |
| ⭐⭐ | F.2 RAG 新闻搜索（Qdrant） | AI必需 | 高 | 3天 | [ ] 待开发 |
| ⭐⭐ | F.3 SHAP 可解释性 | MLE强 | 中 | 1天 | ✅ 已完成（factor_analysis.py） |
| ⭐⭐ | F.4 LangGraph 多 Agent 研究助手 | AI Engineer必杀 | 高 | 2周 | [ ] 待开发 |
| ⭐⭐ | F.8 主动学习 Agent（分歧样本） | MLE+AI | 高 | 4天 | [ ] 待开发 |
| ⭐⭐ | E.2 CI/CD GitHub Actions | DE强 | 中 | 2天 | [ ] 待开发 |
| ⭐⭐ | C.7 数据质量检查 | DE强 | 高 | 2天 | ✅ 已完成（data_quality_check.py） |
| ⭐⭐ | B 量化加分：Long-short 组合 | 量化强 | 中 | 3天 | [ ] 待开发 |
| ⭐⭐ | B 量化加分：Beta 中性化 | 量化强 | 中 | 2天 | [ ] 待开发 |
| ⭐⭐ | F.5 FinBERT 微调 | MLE强 | 高 | 1-2周 | [ ] 待开发 |
| ⭐ | F.6 rule_validator ReAct Agent | AI加分 | 中 | 3天 | [ ] 待开发 |
| ⭐ | F.7 Airflow 自适应调度 Agent | DE+AI | 中 | 3天 | [ ] 待开发 |
| ⭐⭐⭐ | D.1 宏观 Regime 特征（已并入 H.2）| 量化必需 | 🔴 真实可用 | 4天 | ✅ 已完成 |
| ⭐ | E.4 K8s 配置 | DE加分 | 低 | 3天 | [ ] 待开发 |
| ⭐ | E.5 数据血缘图 | DE加分 | 低 | 2天 | [ ] 待开发 |
| ⭐ | E.6 WebSocket 实时推送 | 后端加分 | 高 | 3天 | [ ] 待开发 |
| ⭐ | D.2 散户情绪 | 量化加分 | 中 | 3天 | ✅ 已完成（StockTwits，非 Reddit） |
| ⭐ | F.1 Prompt 评测框架 | MLE加分 | 中 | 2天 | [ ] 待开发 |
| ⭐ | E.8 Demo 视频 | 全部加分 | 高 | 0.5天 | [ ] 待开发 |
| — | D.4 分析师评级变化 | 量化加分 | 中 | 3天 | ✅ 已完成 |
| — | D.7 机构 13F 持仓变化 | 量化加分 | 高 | 3天 | ✅ 已完成（最强单因子，60d IC=+0.20） |
| — | D.8 盘前盘后价格信号 | 量化加分 | 高 | 1天 | ✅ 已完成（最强短线因子，5d IC=+0.23） |

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

## Stage 6 — GDELT GKG 全量索引导入 MongoDB（待开发）

### 6.1 背景与目标

当前匹配新股票历史 GDELT 数据需要逐批扫描 ~35 万个 CSV 文件，耗时约 13 小时。
目标：将 CSV 中 7 个关键列导入 MongoDB，建立 `$text` 全文索引，使新股票关键词匹配从 **13 小时 → 秒级**。

### 6.2 GDELT CSV 列结构（保留 7 列）

| 列号 | 字段名     | MongoDB 字段    | 用途                    |
|------|-----------|-----------------|------------------------|
| 1    | DATE      | `date`          | 发布日期（YYYYMMDDHHMMSS）|
| 4    | URL       | `url`           | 文章链接                 |
| 7    | V1Themes  | `themes`        | 主题关键词（匹配主力）     |
| 11   | V1Persons | `persons`       | 人名                    |
| 13   | V1Orgs    | `orgs`          | 机构/公司名（匹配辅助）   |
| 15   | V1Tone    | `tone`          | GCAM 情绪分数（辅助特征）|
| 23   | AllNames  | `all_names`     | 所有实体名               |

原始 27 列 CSV 中，列 17（GCAM）占 72% 体积；保留 7 列后，数据量从 7TB 降至约 **160 GB**（Parquet 估算），MongoDB BSON + WiredTiger 压缩后约 **300–600 GB**（含 `$text` 索引）。

### 6.3 开发任务

#### 6.3.1 导入脚本（`tools/gdelt_import_to_mongo.py`）

- 多进程并行读取 Data24T + Data6T 上的 `.csv` 文件（约 35 万个）
- 每行提取 7 列，构造 MongoDB 文档
- 按批次（batch_size=5000）写入集合 `quant_data.gkg_index`
- 支持断点续跑：已导入的文件记录进度集合 `gkg_import_progress`
- 去重：以 URL 为唯一键（`url` 建 unique index）

```python
# 文档结构示例
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

#### 6.3.2 索引建立

```javascript
// 复合文本索引（匹配主力）
db.gkg_index.createIndex(
    { themes: "text", persons: "text", orgs: "text", all_names: "text" },
    { name: "gkg_text_idx", weights: { orgs: 10, themes: 5, all_names: 3, persons: 1 } }
)

// 辅助查询索引
db.gkg_index.createIndex({ date: 1 })
db.gkg_index.createIndex({ url: 1 }, { unique: true })
```

#### 6.3.3 新匹配流程改造

替换当前 `historical_collector.py` 中的 CSV 批次扫描，改为：

1. **Step 1**：`db.gkg_index.find({ $text: { $search: "keyword" } })` → 秒级返回候选 URL 列表
2. **Step 2**：查 `news_articles` 集合，复用已抓取的正文
3. **Step 3**：对缺失 URL 发起抓取（旧 URL 2016-2020 可能大量失效，需处理 404）
4. **Step 4**：写入 `news_articles_company_matched_v2`

#### 6.3.4 验证

- 对已有 60 只股票中抽取 10 只，用新流程重新匹配
- 对比 URL 命中率与当前 CSV 方案是否一致
- 检查 `$text` 索引召回率（有无漏匹配）

### 6.4 存储预估

| 项目                        | 估算大小    |
|-----------------------------|------------|
| 原始 CSV（7TB）→ 提取 7 列  | ~160 GB    |
| MongoDB BSON（未压缩）       | ~200 GB    |
| WiredTiger 压缩后（~3.5x）  | ~60 GB     |
| `$text` 索引                | ~200–400 GB|
| **总计（含索引）**           | **~300–600 GB** |

导入验证通过后，原始 CSV 可删除，节省 **6TB+** 硬盘空间。

### 6.5 注意事项

- MongoDB 服务器需预留至少 **600 GB** 可用空间（含索引构建临时空间）
- `$text` 索引构建约需 **数小时**（570M 条记录），建议在导入完成后一次性创建
- V1Tone（`tone` 字段）格式为逗号分隔的多个分数，使用时取第一个值（整体情绪分）
- 旧 URL（2016–2020）大量已失效，URL 抓取阶段需做 404/timeout 容错并记录失效率
- 导入脚本须支持多磁盘（Data24T + Data6T）并行读取，避免单盘 I/O 成为瓶颈

### 6.6 前置条件

- [ ] MongoDB 服务器确认有 600 GB+ 可用空间
- [ ] Data24T 和 Data6T 已挂载并可读
- [ ] 导入完成后对 10 只样本股票做回归测试，通过后再删除原始 CSV

---

# Stage 7 — 新 Mac 校验 + 定时任务验证 + 执行记录（待开发）

## 7.1 背景

已购置 48GB 内存 Mac 并完成数据和代码的复制，需要全面校验各服务和定时任务
在新机器上正常运行，并补充执行结果的可观测性。

## 7.2 服务校验

### 7.2.1 Docker 服务验证
- [ ] 验证所有容器正常启动：mongo6 / quant_api / quant_ui / quant_data /
      airflow-webserver / airflow-scheduler / kafka / mlflow 等
- [ ] 检查各服务间网络连通性（project-net 内部通信）
- [ ] 验证外部卷挂载正确（MongoDB、MySQL、Airflow logs 等数据完整）
- [ ] 确认 `.env` 中连接地址在新机器上有效

### 7.2.2 Airflow 定时任务跑通（不只是校验，要真正运行）
- [ ] 逐一检查 `airflow/dags/` 中所有 DAG 的调度时间和依赖关系
- [ ] 手动触发每条 DAG，确认全链路端到端执行成功（不只是"定义了"）
- [ ] 确认 Docker socket 挂载路径正确（`/var/run/docker.sock`）
- [ ] 等待至少一次自动调度触发，确认定时执行正常
- [ ] 检查 DAG 执行日志，修复任何失败的 task
- [ ] 目标：news 采集 / feature build / model training 三条主 DAG 稳定运行

### 7.2.3 Kafka 实际运行（不只是部署，要有真实数据流）
- [ ] 确认 kafka 容器正常，可连接 `kafka:9092`
- [ ] 创建必要的 topic（如 `quant.signals`, `quant.news`）
- [ ] 实现至少一个 producer：每日信号生成后推送到 `quant.signals` topic
- [ ] 实现至少一个 consumer：消费信号写入 `daily_signals` 集合或触发告警
- [ ] 通过 kafka-ui（端口 15070）验证消息正常流转
- [ ] 目标：信号生成 → Kafka → 消费写库，完整链路跑通

### 7.2.4 MLflow 实际记录 run
- [ ] 确认 mlflow 容器正常（端口 15050）
- [ ] 在 `train_baseline_models.py` 中启用 `--mlflow-uri`，跑一次训练并记录
- [ ] 验证 MLflow UI 中可看到参数、指标、模型文件
- [ ] 目标：每次模型训练都有 run 记录，IC / Top5 超额可追溯

### 7.2.5 LLM 推理校验
- [ ] 在新 Mac 上验证 LM Studio / Ollama 可用，模型已加载
- [ ] 验证 `SLM_API_URL` 环境变量指向正确端点
- [ ] 跑一小批 `llm_enrich_articles.py` 确认推理正常

## 7.3 执行结果记录（UI / API）

当前问题：Airflow 和 Python 脚本的执行结果（成功/失败、处理条数、耗时、关键指标）
没有统一的可见性入口，排查问题困难。

### 7.3.1 quant_api 执行日志接口
- [ ] 新增 `pipeline_runs` 集合（MongoDB）或表（MySQL），记录每次任务执行：
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
- [ ] quant_api 提供 REST 接口：`POST /api/pipeline-runs`（写入）、
      `GET /api/pipeline-runs`（查询最近 N 条）
- [ ] 在各 Python 脚本（daily_symbol_features.py、llm_enrich_articles.py 等）
      执行完成后调用接口上报结果

### 7.3.2 quant_ui 执行历史页面
- [ ] 在 UI 中新增"任务执行记录"页面，展示：
  - 任务名称、状态（成功/失败）、开始/结束时间、处理条数
  - 最近 30 天执行趋势图
  - 失败任务高亮 + 错误信息展示
- [ ] 支持按任务名称筛选和时间范围查询

### 7.3.3 模型训练结果持久化
- [ ] 每次 `train_baseline_models.py` 运行后，将各 horizon 的 IC / Top5 超额收益
      写入 `model_results` 集合，带时间戳和 feature set 版本标记
- [ ] quant_ui 展示模型历史结果趋势，方便对比不同 feature 版本的效果

## 7.4 优先级

1. Docker 基础服务验证（阻塞后续所有工作）
2. **Airflow DAG 真正跑通**（不只是定义，要稳定运行，面试关键证明）
3. **Kafka producer/consumer 真正运行**（信号→topic→消费写库完整链路）
4. **MLflow 实际记录 run**（每次训练可追溯）
5. quant_api 执行日志接口（pipeline_runs 写库）
6. quant_ui 执行历史页面
7. 模型结果持久化

---

## G. 参考资源（2026-05-22 整理）

### G.1 LLM 情绪因子 / Alpha 挖掘论文

| 论文 | 要点 |
|------|------|
| [Event-Aware Sentiment Factors from LLM-Augmented Financial Tweets (arXiv 2508.07408)](https://arxiv.org/pdf/2508.07408) | 可解释 LLM 量化框架，事件感知情绪因子构建，适合和本项目 LLM tagging 方案对比 |
| [Interpretable ML for Macro Alpha: News Sentiment Case Study (arXiv 2505.16136)](https://arxiv.org/pdf/2505.16136) | FinBERT + GDELT → 外汇/国债策略，OOS Sharpe >4；验证了 GDELT + LLM 路线可行性 |
| [AlphaAgent: LLM-Driven Alpha Mining (arXiv 2502.16789)](https://arxiv.org/html/2502.16789v2) | LLM 自动化挖掘 alpha 因子，含对抗 alpha 衰减机制 |
| [Automate Strategy Finding with LLM in Quant Investment (arXiv 2409.06289)](https://arxiv.org/html/2409.06289v1) | LLM 驱动的策略自动寻找流程，可参考 prompt 设计 |

### G.2 CBOE 期权数据（免费下载）

- **PCR 历史 CSV**（对应 D.5 Put/Call Ratio 因子）：
  `https://cdn.cboe.com/resources/options/volume_and_call_put_ratios/indexpcarchive.csv`
- **CBOE 历史数据总页面**：
  `https://www.cboe.com/us/options/market_statistics/historical_data/`
- 用法：直接 `pd.read_csv(url)` 即可加载日频 PCR 数据，无需注册

### G.3 FinBERT 微调资源（对应 F.5 / Stage 3.5.5）

| 资源 | 要点 |
|------|------|
| [Fine-Tuning FinBERT for Sector-Specific Financial News (MDPI Electronics 2025)](https://www.mdpi.com/2079-9292/14/23/4680) | zero-shot F1=0.555 → 微调后 F1=**0.707**；含分行业训练方案，可直接参考训练脚本设计 |
| [ProsusAI/finbert on HuggingFace](https://huggingface.co/ProsusAI/finbert) | 官方 FinBERT 预训练权重，直接加载使用 |
| [Efficient FinBERT via Quantization (ACL FinNLP 2025)](https://aclanthology.org/2025.finnlp-2.6.pdf) | INT8/INT4 量化压缩方案，Mac M 芯片可本地运行 |

### G.4 SEC EDGAR 13F Python 工具（对应 D.7）

| 库 | 说明 |
|----|------|
| [edgartools](https://github.com/dgunning/edgartools) | **首选**，免费开源，将 13F 解析为结构化 Python 对象，数据回溯到 2005 年，`pip install edgartools` |
| [sec-api-python](https://github.com/janlukasschroeder/sec-api-python) | 付费 SDK，有免费试用额度，适合生产环境批量抓取 |

---

## H. 真实交易系统路线图（从研究 → 真正能用）

### 背景：为什么现在还不能上实盘

回测结果（Long-short 年化 +21.7%，Sharpe 0.85）看起来不错，但以下问题会让真实交易亏钱：

| 问题 | 具体表现 | 风险等级 |
|------|---------|---------|
| IC 年度不稳定 | 2022/2024/2025 年 IC 为负，信号反向 | 🔴 致命 |
| 无交易成本 | 回测未扣佣金+滑点，实际收益减少 30-40% | 🔴 致命 |
| 信号滞后 | 新闻公开后 LLM 处理需要数小时 | 🟡 严重 |
| 无仓位控制 | 等权重配置，单票风险未限制 | 🟡 严重 |
| 无 out-of-sample 验证 | 所有数据都用于回测，过拟合风险 | 🟡 严重 |
| 仅 100 只股票 | 集中度太高，流动性风险 | 🟠 中等 |

---

### H.1 修复回测真实性（🔴 必须先做）

**H.1.1 加入交易成本模型**
- 单边佣金：0.05%（万5，券商一般水平）
- 滑点模型：小票 0.3%，大票 0.1%（按市值分级）
- 修改 `backtest_news_factor.py` 和 `backtest_event_driven.py`：
  ```python
  net_return = gross_return - commission * 2 - slippage * 2
  ```
- 目标：看扣成本后 Long-short 年化还剩多少
- 状态：[ ] 待开发（1天）

**H.1.2 加入流动性过滤**
- 剔除日均成交量 < 500万美元的股票（避免无法执行）
- 加入 `avg_daily_volume` 特征，在选仓时过滤
- 状态：[ ] 待开发（0.5天）

**H.1.3 加入仓位上限约束**
- 单票最大仓位：5%（避免过度集中）
- 单行业最大仓位：25%
- 修改 `score_daily_signals.py` 加入仓位约束逻辑
- 状态：[ ] 待开发（0.5天）

---

### H.2 解决 IC 不稳定问题（🔴 必须先做）

**H.2.1 市场 Regime 检测**

IC 年度分析揭示信号只在"趋势市场"有效：
```
2022（熊市/高 VIX）: IC = -0.035 → 关闭信号
2023（反弹/低 VIX）: IC = +0.163 → 开启信号  
2024（震荡）       : IC = -0.022 → 关闭信号
2025（震荡）       : IC = -0.020 → 关闭信号
```

实现方案：
- 数据源：`yfinance` 拉取 VIX 日频数据
- Regime 定义：
  - **趋势市场**：VIX < 20 且 SPY 20日均线向上 → 开启 LLM 情绪因子权重
  - **震荡/恐慌**：VIX > 25 → 降低仓位至 50%，切换均值回归因子
- 新字段：`market_regime`（trend / volatile / crisis）
- 在 `score_daily_signals.py` 中按 regime 调整 composite_score 权重
- 状态：[ ] 待开发（2天）

**H.2.2 动态因子权重**
- 不同 regime 下各因子权重不同：
  - 趋势市场：LLM 情绪权重 40%，动量权重 30%，earnings 权重 30%
  - 震荡市场：LLM 情绪权重 10%，均值回归权重 50%，earnings 权重 40%
- 在模型训练时加入 `market_regime` 作为特征，或分 regime 单独训练
- 状态：[ ] 待开发（2天）

---

### H.3 Paper Trading 验证（🔴 必须做，至少跑 3-6 个月）

**目标**：在真实市场环境中验证信号，积累 out-of-sample 数据

**H.3.1 Paper Trading 引擎**
- 每日盘后自动执行：
  1. 读取当日 `daily_signals` 中 composite_score 前 10 只
  2. 检查 regime：如果 VIX > 25，跳过或减半仓位
  3. 模拟下单，记录入场价（用次日开盘价）
  4. 写入 `paper_positions` 集合
- 字段：`symbol`, `entry_date`, `entry_price`, `size`, `score_at_entry`, `regime_at_entry`
- 状态：[ ] 待开发（2天）

**H.3.2 Paper Trading 绩效追踪**
- 每日更新持仓浮盈浮亏
- 计算真实 out-of-sample IC（每天实际持仓的预测 vs 实际表现）
- 目标指标：
  - OOS IC > 0.02（说明信号在真实市场有效）
  - OOS Sharpe > 0.5（说明策略可执行）
  - 连续 3 个月正超额收益（说明稳定）
- 状态：[ ] 待开发（1天）

**H.3.3 止损和退出逻辑**
- 单票止损：入场后跌 -5% 自动退出
- 情感逆转止损：持仓期间 `avg_sentiment_5d` 变负，主动退出
- 时间止损：持仓超过 `max_hold_days`（默认 45 天）自动退出
- 状态：[ ] 待开发（1天）

---

### H.4 信号质量监控（🟡 重要）

**H.4.1 实时 IC 监控**
- 每天计算最近 20 个交易日的 rolling IC
- IC 连续 5 天低于 0 → 触发告警，暂停新建仓
- 写入 `signal_quality` 集合，UI 显示 IC 趋势图
- 状态：[ ] 待开发（1天）

**H.4.2 模型漂移检测**
- 每周对比最新特征分布和训练时分布（KL 散度）
- 分布偏移超过阈值 → 触发重训提醒
- 状态：[ ] 待开发（1天）

---

### H.5 实盘准备检查清单

在考虑用真实资金之前，必须全部满足：

- [ ] Paper trading 运行 **至少 3 个月**，OOS IC > 0.02
- [ ] 加入交易成本后回测 Sharpe **仍然 > 0.5**
- [ ] Regime 检测正常工作，VIX > 25 时自动降仓
- [ ] 止损逻辑测试通过（模拟极端行情）
- [ ] 单票最大仓位 ≤ 5%，整体杠杆 = 1x（不加杠杆）
- [ ] 已接入 broker API（Interactive Brokers / Alpaca）
- [ ] 第一次实盘资金 ≤ 总资金的 10%（测试阶段）

---

### H. 路线总览

```
现在（研究验证完成）
    ↓
H.1 修复回测真实性（3天）    ← 知道真实收益
H.2 Regime 检测（4天）       ← 知道什么时候用
H.3 Paper Trading（1个月跑通）← 真实市场验证
    ↓ 持续跑 3-6 个月
H.4 信号监控（2天）          ← 保持信号有效性
    ↓ Paper trading 指标达标
H.5 实盘准备清单全部 ✓
    ↓
小仓位实盘（总资金 10% 以内）
    ↓ 6个月稳定超额
真正可用的量化交易系统 ✓
```

**预计从现在到"真正可用"的时间：约 6-9 个月**（其中 3-6 个月是等 paper trading 数据积累）
