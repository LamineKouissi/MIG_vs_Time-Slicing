#!/bin/bash
set -euo pipefail

PHASE="$1"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/results"
mkdir -p "$RESULTS_DIR"

echo "============================================"
echo "  Phase: $PHASE"
echo "  Starting at: $(date -Iseconds)"
echo "============================================"

echo "[+] Starting steady load on instance B..."
"$SCRIPT_DIR/run_bench.sh" "${PHASE}_instanceB_steady" 8001 200 2 &
PID_B=$!

echo "[+] Waiting 30s for steady load to warm up..."
sleep 30

echo "[+] Spiking instance A with 500 requests at 50 req/s..."
"$SCRIPT_DIR/run_bench.sh" "${PHASE}_instanceA_SPIKE" 8000 500 50 &
PID_A=$!

echo "[+] Waiting for both benchmarks to complete..."
wait $PID_B $PID_A

echo "============================================"
echo "  Phase $PHASE COMPLETE"
echo "  Results in: $RESULTS_DIR"
echo "============================================"
