#!/bin/bash
set -euo pipefail

OUTFILE="${1:-gpu_util.csv}"
# Ensure parent directory exists
mkdir -p "$(dirname "$OUTFILE")"

echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib" > "$OUTFILE"

echo "[$(date -Iseconds)] GPU utilization logger started -> $OUTFILE (kill with 'kill $$')"

while true; do
    TIMESTAMP=$(date -Iseconds)
    DATA=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
    if [ -n "$DATA" ]; then
        echo "${TIMESTAMP},${DATA}" >> "$OUTFILE"
    fi
    sleep 1
done
