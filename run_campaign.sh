#!/usr/bin/env bash
# Full campaign: tests, four per-architecture trainings, six-curve evaluation.
# Usage: PY=/path/to/python WORKERS=16 OUT=/path/to/outputs bash run_campaign.sh
# Re-running the IDENTICAL command resumes from completed chunks.
set -euo pipefail
PY="${PY:-python3}"
WORKERS="${WORKERS:-8}"
OUT="${OUT:-results}"
cd "$(dirname "$0")"
mkdir -p "$OUT"
echo "[$(date -u +%FT%TZ)] python=$PY workers=$WORKERS out=$OUT"
"$PY" -B -c "import numpy, scipy, matplotlib; print('numpy', numpy.__version__, 'scipy', scipy.__version__, 'matplotlib', matplotlib.__version__)"
"$PY" -B -m unittest discover -s tests
# 1400 m forward propagation for every curve; real feedback = serialization/airtime +
# 1400 m reverse propagation (no processing, no turnaround); ideal feedback costs nothing.
# 10 Mbit/s dedicated link; declared main channel scenario: 10-RF-packet weather epoch.
TIMING=(--feedback-rate-mbps 10 --forward-propagation-us 4.669897332774129
        --feedback-propagation-us 4.669897332774129 --feedback-processing-us 0
        --epoch-rf-packets 10)
for arch in hd_real hd_ideal ded_real ded_ideal; do
    echo "[$(date -u +%FT%TZ)] training $arch"
    "$PY" -B python/optimize_window.py --architecture "$arch" "${TIMING[@]}" \
        --n-train 2000 --n-validation 4000 --workers "$WORKERS" --bootstrap 2000 \
        --seed 1210000000 --output "$OUT/optimization_$arch"
done
echo "[$(date -u +%FT%TZ)] six-curve evaluation"
"$PY" -B python/run_comparison.py \
    --opt-hd-real "$OUT/optimization_hd_real/optimization.json" \
    --opt-hd-ideal "$OUT/optimization_hd_ideal/optimization.json" \
    --opt-ded-real "$OUT/optimization_ded_real/optimization.json" \
    --opt-ded-ideal "$OUT/optimization_ded_ideal/optimization.json" \
    --nmc 10000 --nmc-outage 1000000 --workers "$WORKERS" --bootstrap 2000 \
    --seed 1300000000 --output "$OUT/publication"
echo "[$(date -u +%FT%TZ)] CAMPAIGN COMPLETE"
