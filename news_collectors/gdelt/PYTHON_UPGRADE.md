# Python Upgrade Plan

## Target

- Stop using macOS CommandLineTools Python `3.9.6`
- Build a dedicated Homebrew Python `3.11` virtualenv for `quant_data`

## Why

- `lxml/newspaper3k` is running on an old system Python build
- native extension stability is materially better on a current Python toolchain
- this collector now uses a process pool for HTML parsing; keeping the runtime modern reduces ABI and packaging risk

## One-time setup

Run:

```bash
cd /Users/xiz/Quant_trade
bash quant_data/scripts/setup_python311_env.sh
```

This creates:

- virtualenv: `/Users/xiz/Quant_trade/quant_data/.venv311`

## Daily use

```bash
cd /Users/xiz/Quant_trade/quant_data/news_collectors/gdelt
source /Users/xiz/Quant_trade/quant_data/.venv311/bin/activate
python historical_collector.py
```

## Recommended runtime

```bash
HOST_ID=mac
BATCH_WORKERS=5
FETCH_WORKERS=3
PARSE_WORKERS=2
FETCH_TASK_TIMEOUT=45
ARTICLE_REQUEST_TIMEOUT=10
```

Notes:

- `FETCH_WORKERS` controls how many article jobs are kept in flight per batch worker
- `PARSE_WORKERS` is the hard cap for child-process HTML parsing concurrency
- if `lxml/newspaper3k` crashes inside a child process, the parent collector survives and falls back to `title_only`

## Cutover

1. Stop the old collector.
2. Activate `.venv311`.
3. Start one collector instance and watch the first few batches.
4. If stable, retire the old `.venv`.
