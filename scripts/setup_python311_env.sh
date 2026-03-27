#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
QUANT_DATA_DIR="$REPO_ROOT/quant_data"
VENV_DIR="$QUANT_DATA_DIR/.venv311"

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is required." >&2
  exit 1
fi

brew install python@3.11

PY311_BIN="$(brew --prefix python@3.11)/bin/python3.11"
if [[ ! -x "$PY311_BIN" ]]; then
  echo "python3.11 not found at $PY311_BIN" >&2
  exit 1
fi

"$PY311_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
pip install -r "$QUANT_DATA_DIR/requirements.txt"

cat <<EOF
Python 3.11 environment is ready.

Activate with:
  source $VENV_DIR/bin/activate

Collector example:
  cd $QUANT_DATA_DIR/news_collectors/gdelt
  source $VENV_DIR/bin/activate
  python historical_collector.py
EOF
