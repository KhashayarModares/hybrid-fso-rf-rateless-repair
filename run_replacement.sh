#!/usr/bin/env bash
# Re-run the two REAL feedback curves of the finished campaign with the agreed feedback costs:
#   both:        serialization of 13+W bits at 200 Mbit/s (RF link rate), 1400 m reverse
#                propagation, independent report loss LOSS = RF outage probability of the
#                channel model (rf_erasure, identical in clear and haze), no processing delay;
#   half duplex: direction-change gap TURN_US = DMG aSIFSTime 3 us (IEEE Std 802.11-2024,
#                Table 20-30), twice per report; forward RF silent during the exchange.
# Pooling, open loop and ideal feedback are reused from OUT/publication on identical blocks.
# Usage: PY=/path/to/python WORKERS=6 OUT=/path/to/outputs bash run_replacement.sh
# Re-running the identical command resumes completed chunks.
set -euo pipefail
PY="${PY:-python3}"
WORKERS="${WORKERS:-6}"
OUT="${OUT:-results}"
LOSS="${LOSS:-0.04868957933020315}"
TURN_US="${TURN_US:-3}"
FB_RATE_MBPS="${FB_RATE_MBPS:-200}"
TAG="${TAG:-widegrid}"
cd "$(dirname "$0")"
echo "[$(date -u +%FT%TZ)] python=$PY workers=$WORKERS out=$OUT loss=$LOSS sifs_us=$TURN_US feedback_rate_mbps=$FB_RATE_MBPS tag=$TAG"
test -f "$OUT/publication/completed.json" || { echo "No finished campaign in $OUT/publication"; exit 1; }
"$PY" -B -m unittest discover -s tests
COMMON=(--feedback-rate-mbps "$FB_RATE_MBPS" --forward-propagation-us 4.669897332774129
        --feedback-propagation-us 4.669897332774129 --feedback-processing-us 0
        --feedback-loss-probability "$LOSS" --epoch-rf-packets 10
        --n-train 2000 --n-validation 4000 --workers "$WORKERS" --bootstrap 2000 --seed 1210000000)
echo "[$(date -u +%FT%TZ)] training hd_real (loss $LOSS, SIFS $TURN_US us per direction change)"
"$PY" -B python/optimize_window.py --architecture hd_real "${COMMON[@]}" --turnaround-us "$TURN_US" \
    --output "$OUT/optimization_hd_$TAG"
echo "[$(date -u +%FT%TZ)] training ded_real (loss $LOSS)"
"$PY" -B python/optimize_window.py --architecture ded_real "${COMMON[@]}" \
    --output "$OUT/optimization_ded_$TAG"
echo "[$(date -u +%FT%TZ)] replacement evaluation on the campaign blocks"
"$PY" -B python/run_replacement.py --campaign "$OUT/publication" \
    --opt-hd-lossy "$OUT/optimization_hd_$TAG/optimization.json" \
    --opt-ded-lossy "$OUT/optimization_ded_$TAG/optimization.json" \
    --workers "$WORKERS" --output "$OUT/publication_$TAG"
echo "[$(date -u +%FT%TZ)] REPLACEMENT COMPLETE"
