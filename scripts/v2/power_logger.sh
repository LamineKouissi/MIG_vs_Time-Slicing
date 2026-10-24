#!/usr/bin/env bash
# Logs whole-GPU board power (W) with epoch timestamps until killed.
# Works in MIG mode too: power is reported for the physical GPU, not per slice.
#
# Usage:  power_logger.sh <gpu-index> <outfile> [interval_s]
set -uo pipefail

GPU="$1"
OUTFILE="$2"
INTERVAL="${3:-0.5}"

echo "epoch_s,power_w" >"$OUTFILE"
while true; do
    watts=$(nvidia-smi -i "$GPU" --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1 || true)
    echo "$(date +%s.%N),${watts// /}" >>"$OUTFILE"
    sleep "$INTERVAL"
done
