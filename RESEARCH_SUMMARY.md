# Research Summary

## Purpose
This document records the latest formal research results for the current
`quant_data` workflow, so the project has a stable written conclusion instead
of relying on ad hoc terminal output or memory.

This file should be read together with:

- [PROJECT_PLAN.md](/Users/xiz/Quant_trade/quant_data/PROJECT_PLAN.md)
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
- [research/backtest_news_factor.py](/Users/xiz/Quant_trade/quant_data/research/backtest_news_factor.py)
- [research/train_baseline_models.py](/Users/xiz/Quant_trade/quant_data/research/train_baseline_models.py)

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

## Earnings Event Layer Update
An initial earnings-event layer has now been added to the research pipeline.

Implemented components:

- earnings source:
  - `yfinance`
- earnings collection:
  - `earnings_events`
- loader script:
  - [research/load_earnings_events.py](/Users/xiz/Quant_trade/quant_data/research/load_earnings_events.py)
- feature integration:
  - [research/daily_symbol_features.py](/Users/xiz/Quant_trade/quant_data/research/daily_symbol_features.py)

The first event features added are:

- `days_to_earnings`
- `days_since_earnings`
- `is_earnings_window_5d`
- `is_post_earnings_window_20d`

Operational result:

- earnings events loaded:
  - `1,112`
- feature table rebuilt:
  - `40,009` rows in `daily_symbol_features_company_matched_v1`

## Earnings Event Result
The first earnings-event layer is useful, but not yet a major standalone edge.

### `20d` baseline with earnings-event features
Compared with the previous baseline:

- previous `Ridge`
  - Rank IC: `0.0207`
  - Top 5 mean excess return: `+2.00%`
- current `Ridge`
  - Rank IC: `0.0285`
  - Top 5 mean excess return: `+2.02%`

Interpretation:

- `20d` sees a small improvement
- earnings timing provides some useful short-horizon context
- the improvement is real but modest

### `60d` baseline with earnings-event features
Compared with the previous baseline:

- previous `Ridge`
  - Rank IC: `0.0762`
  - Top 5 mean excess return: `+6.15%`
- current `Ridge`
  - Rank IC: `0.0730`
  - Top 5 mean excess return: `+6.20%`

Interpretation:

- `Top 5` excess return improves slightly
- overall rank correlation is slightly lower
- this means the current earnings-event layer is helpful as context, but not
  yet a decisive new alpha source

## Updated Working Conclusion
After adding the first earnings-event layer:

- news quality is still the core signal
- earnings timing adds useful context, especially for `20d`
- the current earnings features should be treated as a support layer, not a
  replacement for quality factors
- the next event research step should focus on richer earnings surprise and
  post-earnings structure features
