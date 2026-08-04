#!/bin/bash
set -euo pipefail

LABEL="$1"
PORT="$2"
NUM_PROMPTS="${3:-200}"
REQUEST_RATE="${4:-2}"
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$RESULTS_DIR"

OUTFILE="${RESULTS_DIR}/${LABEL}.json"

echo "[$(date -Iseconds)] Starting benchmark: label=$LABEL port=$PORT prompts=$NUM_PROMPTS rate=$REQUEST_RATE"

vllm bench serve \
    --backend openai \
    --base-url "http://localhost:${PORT}" \
    --endpoint /v1/completions \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --dataset-name random \
    --random-input-len 128 \
    --random-output-len 128 \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --save-result \
    --result-filename "$OUTFILE" \
    2>&1 | tee "${RESULTS_DIR}/${LABEL}.log"

echo "[$(date -Iseconds)] Benchmark complete: $OUTFILE"
