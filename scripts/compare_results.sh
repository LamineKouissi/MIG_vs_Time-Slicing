#!/bin/bash
set -euo pipefail

RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"

echo ""
echo "================================================================="
echo "  MIG ISOLATION EXPERIMENT — RESULTS COMPARISON"
echo "================================================================="
echo ""

for phase in baseline mig; do
    echo "--- Phase: ${phase^^} ---"
    echo ""
    printf "%-35s %12s %12s %12s %12s %12s %12s\n" \
        "Label" "p50 TTFT" "p99 TTFT" "p50 TPOT" "p99 TPOT" "TPOT StdDev" "Throughput"
    printf "%-35s %12s %12s %12s %12s %12s %12s\n" \
        "-----" "--------" "--------" "--------" "--------" "-----------" "----------"

    # Use nullglob so pattern expands to empty if no match
    shopt -s nullglob
    for f in "$RESULTS_DIR"/${phase}_*.json; do
        [ -f "$f" ] || continue
        label=$(basename "$f" .json)

        p50_ttft=$(jq -r '(.median_ttft_ms // "N/A") | if type == "number" then (. * 10 | round / 10 | tostring) + "ms" else . end' "$f" 2>/dev/null)
        p99_ttft=$(jq -r '(.p99_ttft_ms // "N/A") | if type == "number" then (. * 10 | round / 10 | tostring) + "ms" else . end' "$f" 2>/dev/null)
        p50_tpot=$(jq -r '(.median_tpot_ms // "N/A") | if type == "number" then (. * 10 | round / 10 | tostring) + "ms" else . end' "$f" 2>/dev/null)
        p99_tpot=$(jq -r '(.p99_tpot_ms // "N/A") | if type == "number" then (. * 10 | round / 10 | tostring) + "ms" else . end' "$f" 2>/dev/null)
        std_tpot=$(jq -r '(.std_tpot_ms // "N/A") | if type == "number" then (. * 100 | round / 100 | tostring) + "ms" else . end' "$f" 2>/dev/null)
        throughput=$(jq -r '(.request_throughput // "N/A") | if type == "number" then (. * 10 | round / 10 | tostring) + " rps" else . end' "$f" 2>/dev/null)

        printf "%-35s %12s %12s %12s %12s %12s %12s\n" \
            "$label" "$p50_ttft" "$p99_ttft" "$p50_tpot" "$p99_tpot" "$std_tpot" "$throughput"
    done
    shopt -u nullglob
    echo ""
done

echo "================================================================="
echo "Key: compare instance B's (STEADY) metrics between phases."
echo "  Baseline: p99 TTFT should be DEGRADED during instance A's spike."
echo "  MIG:      p99 TTFT should stay FLAT, and TPOT StdDev should collapse."
echo "================================================================="
