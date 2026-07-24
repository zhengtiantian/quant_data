# Research Summary

## Purpose
This document records the latest formal research results for the current
`quant_data` workflow, so the project has a stable written conclusion instead
of relying on ad hoc terminal output or memory.

This file should be read together with:

- [PROJECT_PLAN.md](https://github.com/zhengtiantian/ai-equity-signal-platform/blob/main/PROJECT_PLAN.md)
- [RESEARCH_FACTORS.md](/Users/xiz/Quant_trade/quant_data/RESEARCH_FACTORS.md)

## Research Scope
The latest run focused on the current clean daily feature collection:

- collection: `daily_symbol_features_company_matched_v1`
- rows: `40,009`
- symbols: `14`
- trade date range: `2015-12-04 -> 2026-03-23`

The research question was:

- which news factors currently work best
- whether `20d` or `60d` is the more stable holding horizon
- which bucket size fits the current 14-name stock universe
- whether a simple baseline model can beat a benchmark-relative target

## Scripts Used
- [research/backtest/backtest_news_factor.py](/Users/xiz/Quant_trade/quant_data/research/backtest/backtest_news_factor.py)
- [research/models/train_baseline_models.py](/Users/xiz/Quant_trade/quant_data/research/models/train_baseline_models.py)

## Factor Backtest Setup
The latest factor run used:

- factors:
  - `article_count`
  - `news_burst_20d`
  - `full_ratio`
  - `quality_score`
- horizons:
  - `20d`
  - `60d`
- portfolio sizes:
  - `top 3`
  - `top 5`
  - `top 10`
- strategies:
  - `top`
  - `bottom`
  - `long_short`
- excess return mode:
  - `benchmark`

## Factor Results

### 1. `full_ratio` is the strongest factor
This remains the best single factor in the current system.

Key results:

- `20d top 3`
  - mean excess return: `+0.56%`
- `20d long_short top3-bottom3`
  - mean excess return: `+0.64%`
- `60d top 3`
  - mean excess return: `+1.32%`
- `60d long_short top3-bottom3`
  - mean excess return: `+2.00%`

Interpretation:

- stocks with a higher share of full-content, readable articles tend to
  outperform over medium horizons
- the quality signal becomes stronger over `60d` than over `20d`

### 2. `quality_score` is the next-best factor
The composite quality factor is also useful and behaves similarly to
`full_ratio`.

Key results:

- `20d top 3`
  - mean excess return: `+0.45%`
- `20d long_short top3-bottom3`
  - mean excess return: `+0.76%`
- `60d top 3`
  - mean excess return: `+0.89%`
- `60d long_short top3-bottom3`
  - mean excess return: `+1.87%`

Interpretation:

- the broader quality basket works
- it is slightly less direct than `full_ratio`, but still clearly stronger
  than raw news volume factors

### 3. `news_burst_20d` is weak to neutral
This factor does not currently show a strong or stable edge.

Observed pattern:

- some `60d` buckets are slightly positive
- most results stay near zero
- the factor is not strong enough to be a primary signal on its own

Interpretation:

- unusual news volume alone is not enough
- an attention spike without quality confirmation is weak

### 4. `article_count` is weak and often misleading
Raw article volume is not a good signal in the current setup.

Observed pattern:

- `top` buckets are often negative on excess-return basis
- `bottom` buckets sometimes do better than `top`

Interpretation:

- more articles does not mean better information
- noisy, duplicated, low-value news flow can inflate count-based factors

## Horizon Conclusion
`60d` is clearly more stable than `20d`.

Why:

- the strongest factors, `full_ratio` and `quality_score`, both improve
  materially at `60d`
- long-short spreads are stronger at `60d`
- the baseline model also performs better on `excess_ret_60d`

Working conclusion:

- the current news-quality thesis is better suited to medium-horizon
  positioning than short-medium `20d` positioning

## Bucket Size Conclusion
For the current `14`-name universe:

- `top 3` is best
- `top 5` is acceptable
- `top 10` is too diluted

Why:

- with only 14 names, `top 10` is already too close to holding most of the
  universe
- the strongest alpha appears in the tightest buckets

Working conclusion:

- use `top 3` as the main research lens
- use `top 5` as a robustness check

## Baseline Model Setup
The latest model run used:

- target:
  - `excess_ret_20d`
  - `excess_ret_60d`
- features:
  - `full_ratio`
  - `quality_score`
  - `article_count`
  - `news_burst_20d`
  - `past_ret_20d`
  - `past_ret_60d`
  - `volatility_20d`
  - `volatility_60d`
  - `volume_shock_20d`
  - `sector`
- evaluation:
  - walk-forward by year
  - metrics:
    - Rank IC
    - Top 5 mean excess return

Models tested:

- `Ridge`
- `HistGradientBoostingRegressor`

## Baseline Model Results

### 1. `Ridge` beats the tree baseline
The linear model is currently the stronger baseline.

#### `excess_ret_20d`
- `Ridge`
  - all-sample Rank IC: `0.0207`
  - Top 5 mean excess return: `+2.00%`
- `HistGB`
  - all-sample Rank IC: `-0.0008`
  - Top 5 mean excess return: `+1.52%`

Interpretation:

- `20d` has some signal
- but it is not yet especially strong or stable

#### `excess_ret_60d`
- `Ridge`
  - all-sample Rank IC: `0.0762`
  - Top 5 mean excess return: `+6.15%`
- `HistGB`
  - all-sample Rank IC: `0.0095`
  - Top 5 mean excess return: `+4.21%`

Interpretation:

- `60d` is the stronger prediction target
- the current feature set already supports a useful first-pass model
- more complex nonlinear modeling is not yet necessary

## Practical Research Conclusion
The current system supports the following thesis:

- news quality matters more than news quantity
- the signal is more medium-horizon than short-horizon
- a concentrated `top 3` ranking framework is the best fit for the current
  universe size
- a simple linear baseline is enough for the current stage

This means the best current working setup is:

- primary factors:
  - `full_ratio`
  - `quality_score`
- primary target:
  - `excess_ret_60d`
- primary portfolio lens:
  - `top 3`
- baseline model:
  - `Ridge`

## What This Means For Next Development
The next stage should not focus on collecting more generic news first.

The next stage should focus on:

- making the research output reproducible
- expanding the factor library around quality, diversity, and event structure
- testing whether benchmark-relative medium-horizon signal survives when the
  universe expands
- only then moving to more advanced modeling and portfolio rules

## Recommended Next Step
The single most reasonable next step is:

- build a richer research feature set around:
  - event structure
  - source diversity
  - recency / persistence
  - sector-relative context

Then rerun the same:

- `60d`
- `top 3`
- `Ridge`

and compare against this document as the baseline.

## Earnings Event Layer — Layer 1 (Timing Only)
An initial earnings-event layer was added using timing features only.

Implemented components:

- earnings source:
  - `yfinance`
- earnings collection:
  - `earnings_events`
- loader script:
  - [research/features/load_earnings_events.py](/Users/xiz/Quant_trade/quant_data/research/features/load_earnings_events.py)
- feature integration:
  - [research/features/daily_symbol_features.py](/Users/xiz/Quant_trade/quant_data/research/features/daily_symbol_features.py)

Layer 1 features:

- `days_to_earnings`
  - trading days until the next scheduled earnings date
- `days_since_earnings`
  - trading days since the most recent past earnings date
- `is_earnings_window_5d`
  - 1 if earnings is within the next 5 trading days
- `is_post_earnings_window_20d`
  - 1 if the most recent earnings was within the past 20 trading days

Operational result:

- earnings events loaded:
  - `1,112`
- feature table rebuilt:
  - `40,009` rows in `daily_symbol_features_company_matched_v1`

Layer 1 model result (`Ridge`):

- `20d` Rank IC: `0.0285` / Top 5 excess return: `+2.02%`
- `60d` Rank IC: `0.0730` / Top 5 excess return: `+6.20%`

Interpretation:

- timing alone is useful context but not a decisive new alpha source
- the improvement over the news-only baseline was real but modest

## Earnings Event Layer — Layer 2 (Surprise + Richer Timing)
A second earnings-event layer was added with EPS surprise and richer
timing structure.

### New features in Layer 2

**Surprise features** (from the most recent past earnings event):

- `surprise_pct_last`
  - EPS surprise percentage at the most recent past earnings:
    `(reported_eps - eps_estimate) / abs(eps_estimate) * 100`
  - positive means the company beat estimates
  - negative means the company missed
- `eps_estimate_last`
  - analyst EPS estimate for the most recent past earnings
- `reported_eps_last`
  - actual reported EPS for the most recent past earnings
- `is_positive_surprise`
  - 1 if `surprise_pct_last > 0`
- `is_negative_surprise`
  - 1 if `surprise_pct_last < 0`

**Timing bucket features**:

- `days_to_earnings_bucket`
  - ordinal bucket of `days_to_earnings`:
    0 = 0–5d (imminent), 1 = 6–15d (near), 2 = 16–30d (upcoming), 3 = 31+d
- `days_since_earnings_bucket`
  - ordinal bucket of `days_since_earnings` using the same scale

**Tighter pre/post windows**:

- `is_pre_earnings_10d`
  - 1 if earnings is within the next 10 trading days
- `is_post_earnings_5d`
  - 1 if the most recent earnings was within the past 5 trading days
- `is_post_earnings_10d`
  - 1 if the most recent earnings was within the past 10 trading days

**Post-earnings drift combined with surprise sign**:

- `is_post_positive_surprise_20d`
  - 1 if within 20-day post-earnings window AND most recent surprise was positive
- `is_post_negative_surprise_20d`
  - 1 if within 20-day post-earnings window AND most recent surprise was negative

Operational result:

- earnings events loaded:
  - `1,100`
- feature table rebuilt:
  - `40,009` rows in `daily_symbol_features_company_matched_v1`

### `surprise_pct_last` as a standalone factor

`surprise_pct_last` was tested as a single cross-sectional ranking factor.

Key results (`top 3`, benchmark excess return):

- `5d top 3` mean excess return: `+0.16%`
- `20d top 3` mean excess return: `+0.40%`
- `60d top 3` mean excess return: `+1.05%`
- `60d long_short top3-bottom3` mean excess return: `-0.20%`

Interpretation:

- has a small short-horizon edge (`5d`) slightly stronger than `full_ratio`
- at `20d` and `60d`, weaker than `full_ratio` and `quality_score`
- the `60d` long-short spread turns negative, meaning EPS surprise direction
  does not provide stable medium-horizon directional alpha on its own
- the EPS surprise signal is most useful as a model feature, not as a
  standalone ranking factor

## Layer 2 Model Results

### `20d` target

| Model | Rank IC | Top 5 mean excess return |
|---|---|---|
| `Ridge` (Layer 1 baseline) | `0.0285` | `+2.02%` |
| `Ridge` (Layer 2) | **`0.0555`** | **`+2.05%`** |
| `HistGB` (Layer 1 baseline) | `-0.0008` | `+1.52%` |
| `HistGB` (Layer 2) | **`0.0384`** | **`+2.07%`** |

### `60d` target

| Model | Rank IC | Top 5 mean excess return |
|---|---|---|
| `Ridge` (Layer 1 baseline) | `0.0730` | `+6.20%` |
| `Ridge` (Layer 2) | **`0.1214`** | **`+6.58%`** |
| `HistGB` (Layer 1 baseline) | `0.0095` | `+4.21%` |
| `HistGB` (Layer 2) | **`0.1181`** | **`+6.32%`** |

Interpretation:

- `Ridge` IC on `60d` improved from `0.073` to `0.121`, a 66% improvement
- `HistGB` IC on `60d` improved from near zero to `0.118`, now comparable
  to `Ridge`
- `20d` IC roughly doubled from `0.029` to `0.056`
- the Layer 2 improvement is substantial, not marginal
- the two models are now close in `60d` performance, suggesting the feature
  set is doing more of the work than the model architecture

## Updated Working Conclusion
After adding the Layer 2 earnings-event feature set:

- news quality (`full_ratio`, `quality_score`) remains the core ranking signal
- EPS surprise features provide a meaningful second layer of information,
  especially for the `60d` prediction target
- `surprise_pct_last` is not a strong standalone factor but is a valuable
  model input when combined with quality and timing features
- `HistGB` is now competitive with `Ridge` at `60d`, suggesting the feature
  set is richer and nonlinear relationships are being captured
- the primary working setup is updated to:
  - primary factors: `full_ratio`, `quality_score`
  - primary target: `excess_ret_60d`
  - primary portfolio lens: `top 3`
  - baseline model: `Ridge` or `HistGB` (now comparable)

## What This Means For Next Development
The next stage should focus on:

- testing whether the event features are stable year-by-year
  (the yearly walk-forward results show high variance in 2022 and 2024)
- adding richer post-earnings drift structure:
  - surprise magnitude buckets
  - interaction between surprise sign and news quality
- expanding the universe beyond 14 names to test whether signals survive
- only after universe expansion, consider more advanced modeling
