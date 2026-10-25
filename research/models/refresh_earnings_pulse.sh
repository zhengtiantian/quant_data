#!/bin/bash
# Weekly refresh of the earnings-pulse model behind quant_ai's live watch (PROJECT_PLAN Q.5 v4):
# the week's reports join the data, the walk-forward is rerun, and the live model is replaced only
# if its out-of-sample record is still positive (earnings_pulse.py final). Run by weekly_model_training.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv311/bin/python
EP_REFRESH=1 $PY research/models/earnings_policy.py fetch       # daily bars and report dates (yfinance)
$PY research/models/earnings_policy.py minutes                  # 5-minute bars of the new reports (Alpaca)
$PY research/models/earnings_policy.py build                    # events and their features
$PY research/models/earnings_releases.py fetch                  # SEC acceptance times of the new releases
$PY research/models/earnings_pulse.py fetch                     # 1-minute bars of the new after-the-close evenings
export PULSE_TAG=_strict PULSE_COST_PCT=0.25 PULSE_FILL_THROUGH_PCT=0.05
$PY research/models/earnings_pulse.py build
PULSE_DROP=vol_at,surprise,beat,fy_outlook,margin_outlook,tone,guide_seq,rev_vs_guide $PY research/models/earnings_pulse.py model
$PY research/models/earnings_pulse.py final
