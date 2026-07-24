# Research Factors Guide

## Purpose
This document explains the main research factors and labels used in the
`quant_data` project, how to interpret them, and why they matter for the
current mid-horizon stock research workflow.

This file is intended to complement:

- [PROJECT_PLAN.md](https://github.com/zhengtiantian/ai-equity-signal-platform/blob/main/PROJECT_PLAN.md)

`PROJECT_PLAN.md` explains where the project is going.

This document explains what the current factors and labels actually mean.

## Research Context
The current system is built for:

- daily-frequency research
- holding periods around:
  - 1 week
  - 2-4 weeks
  - 2-3 months

The current workflow is:

1. collect raw news
2. clean company relevance
3. build daily symbol features
4. align to daily price labels
5. run factor backtests

The goal is not to predict intraday moves.

The goal is to understand whether the information environment around a stock
contains medium-horizon signal.

## Core Collections

- `news_articles`
  - raw news articles

- `news_articles_company_matched_v1`
  - company-matched clean news

- `stock_prices_history`
  - daily OHLCV price data

- `daily_symbol_features_company_matched_v1`
  - daily symbol-level feature table used for research

## Main Feature Categories
The current feature set can be grouped into four families:

1. quantity
2. quality
3. diversity / confirmation
4. labels

### 1. Quantity Features
These describe how much news exists around a stock.

- `article_count`
  - total number of articles for a symbol on a given day

- `news_count_3d`
  - rolling 3-day article count

- `news_count_5d`
  - rolling 5-day article count

- `news_count_20d`
  - rolling 20-day article count

- `news_burst_20d`
  - how unusual today’s article count is relative to prior 20-day average

#### How to interpret quantity
These factors answer:

- Is the stock getting attention?
- Is today’s coverage unusually high?

Useful for:

- event detection
- abnormal-news days

Weakness:

- a lot of low-quality duplicate news can inflate counts
- high volume alone does not imply useful signal

### 2. Quality Features
These describe how readable and complete the news is.

- `full_count`
  - number of articles with full matched content

- `title_only_count`
  - number of articles where only the title could be used

- `url_only_count`
  - number of articles with almost no usable extracted content

- `full_ratio`
  - `full_count / article_count`

- `title_only_ratio`
  - `title_only_count / article_count`

- `avg_content_length`
  - average extracted content length for the day

- `max_content_length`
  - longest extracted content length for the day

- `extraction_failed_count`
  - count of extraction failures or unavailable content

- `timeout_fallback_count`
  - count of timeout/fallback extraction cases

#### Why quality matters
These features try to distinguish:

- real, readable coverage
- thin, noisy, low-information coverage

In current research, quality-based factors are more promising than raw volume.

### 3. Diversity / Confirmation Features
These describe whether multiple distinct sources are talking about the stock.

- `unique_url_count`
  - number of distinct URLs

- `unique_source_count`
  - number of distinct sources

- `unique_platform_count`
  - number of distinct platforms

#### Why diversity matters
These features try to measure:

- is this a single-source event?
- or a broader information consensus?

This matters because:

- multiple-source coverage is often more credible than one isolated mention

### 4. Price Labels
These align news features to later market outcomes.

- `trade_date`
  - first matching trading day on or after the news date

- `close`
  - close price on that trade date

- `future_ret_5d`
  - stock return over the next 5 trading days

- `future_ret_20d`
  - stock return over the next 20 trading days

- `future_ret_60d`
  - stock return over the next 60 trading days

These are absolute stock returns, not benchmark-relative by themselves.

## What `full_ratio` Means
`full_ratio` is the simplest important quality factor in the current project.

Formula:

```text
full_ratio = full_count / article_count
```

### Example
If a stock has:

- `article_count = 10`
- `full_count = 7`

then:

```text
full_ratio = 0.7
```

This means:

- 70% of that day’s news items were articles with proper content

### Interpretation
High `full_ratio` means:

- a larger share of the day’s news is readable and content-rich

Low `full_ratio` means:

- much of the day’s coverage is thin, title-only, or low-utility

### Research role
`full_ratio` acts like:

- a single-dimension news quality factor

In current backtests it is one of the strongest factors.

## What `quality_score` Means
`quality_score` is a composite factor built from multiple quality-related fields.

It is not a raw source field. It is a constructed research factor.

### Current inputs
The current implementation combines:

- `full_ratio`
- `unique_source_count`
- `avg_content_length`
- `extraction_failed_count` as a negative input
- `news_burst_20d` as a weak positive helper

### Construction logic
The score is built cross-sectionally by date:

1. each component is ranked across symbols on the same day
2. ranks are centered
3. positive signals add to score
4. negative signals subtract from score
5. components are averaged into a final score

### Interpretation
High `quality_score` suggests:

- more complete articles
- broader source confirmation
- better content depth
- fewer extraction failures
- some degree of abnormal attention

Low `quality_score` suggests:

- weaker information quality
- thinner or noisier coverage

### Research role
`quality_score` is a more balanced version of the quality thesis.

In current backtests:

- `full_ratio` is often sharper
- `quality_score` is often more stable

## Benchmark Concepts
The project now also supports benchmark-relative labels.

### Why benchmarks matter
A stock can go up and still be weak.

Example:

- stock return over 20d = `+8%`
- `QQQ` return over 20d = `+10%`

Then:

- the stock underperformed by `-2%`

So the project now distinguishes between:

- absolute return
- excess return

### Benchmarks currently used
- `SPY`
  - broad US large-cap benchmark

- `QQQ`
  - Nasdaq-100 benchmark
  - better aligned with the current symbol universe

### Current benchmark fields
- `spy_ret_5d`
- `spy_ret_20d`
- `spy_ret_60d`

- `qqq_ret_5d`
- `qqq_ret_20d`
- `qqq_ret_60d`

- `benchmark_symbol`
- `benchmark_ret_5d`
- `benchmark_ret_20d`
- `benchmark_ret_60d`

- `excess_ret_5d`
- `excess_ret_20d`
- `excess_ret_60d`

Current default benchmark:

- `QQQ`

So by default:

```text
excess_ret_20d = future_ret_20d - qqq_ret_20d
```

## What Current Backtests Suggest
Current results show:

- quantity-only signals are weaker
- quality signals are stronger
- `full_ratio` is strong
- `quality_score` is also useful
- smaller portfolios (`top 3`, `top 5`) work better than `top 10`
- `20d / 60d` horizons are stronger than very short horizons
- quality factors still retain signal even when measured relative to `QQQ`

This suggests the research direction is currently:

- not “more news is better”
- but rather:
  - higher-quality news environments are more predictive than raw volume

## How to Think About These Factors
These factors are not magic predictions by themselves.

They are ways to quantify:

- how much information exists
- how complete it is
- how broadly confirmed it is
- whether the information environment looks useful or noisy

The current working hypothesis is:

- high-quality, well-covered, content-rich news days may carry more medium-horizon signal than simple article counts

## Current Best-Use Factors
At the current stage of the project, the most important factors are:

1. `full_ratio`
2. `quality_score`
3. `news_burst_20d` as supporting context

## Recommended Next Research Steps
The next research steps after this document are:

1. keep using `daily_symbol_features_company_matched_v1`
2. evaluate factors in benchmark-relative mode
3. test stability by year
4. move to baseline models only after factor behavior is stable

## Summary
If you only remember a few things:

- `full_ratio` = complete-article share
- `quality_score` = composite news quality factor
- `QQQ` is the main benchmark for current research
- `excess_ret_*` measures stock performance relative to the benchmark
- current evidence suggests quality matters more than raw news volume
