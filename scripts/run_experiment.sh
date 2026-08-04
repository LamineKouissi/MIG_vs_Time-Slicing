#!/bin/bash
# Master script to run the full MIG vs Time-Slice experiment
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="$BASE_DIR/results"

# Setup logging function
log() {
    echo "[$(date -Iseconds)] $1"
}

log "============================================"
log " Starting Time-Slice vs MIG Experiment"
log "============================================"

log "Step 1: Installing prerequisites..."
sudo apt update && sudo apt install -y jq bc curl python3-pip python3-venv

mkdir -p "$RESULTS_DIR"
cd "$BASE_DIR"

log "Setting up Python virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install vllm huggingface_hub

log "Pre-downloading the model..."
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"

log "--- PHASE 1: BASELINE (Time-Sliced) ---"
log "Starting Baseline Docker containers..."
sudo docker run -d --name vllm-a --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.45 \
    --max-model-len 4096

sudo docker run -d --name vllm-b --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8001:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.45 \
    --max-model-len 4096

log "Waiting for Baseline models to load (this can take a few minutes)..."
for i in {1..36}; do
    if curl -s "http://localhost:8000/v1/models" | grep -q "Qwen/Qwen2.5-7B-Instruct" 2>/dev/null && \
       curl -s "http://localhost:8001/v1/models" | grep -q "Qwen/Qwen2.5-7B-Instruct" 2>/dev/null; then
        log "Baseline models are ready!"
        break
    fi
    sleep 5
    if [ $i -eq 36 ]; then
        log "ERROR: Baseline models failed to start within timeout."
        exit 1
    fi
done

log "Starting GPU utilization logger for Baseline..."
"$SCRIPT_DIR/gpu_util_logger.sh" "$RESULTS_DIR/baseline_gpu_util.csv" & 
GPU_LOG_PID=$!

log "Running Baseline Test Pattern..."
"$SCRIPT_DIR/run_test_pattern.sh" baseline

log "Stopping GPU utilization logger..."
kill $GPU_LOG_PID || true

log "Cleaning up Baseline containers..."
sudo docker stop vllm-a vllm-b || true
sudo docker rm vllm-a vllm-b || true

log "--- PHASE 2: MIG (Hardware Isolation) ---"
log "Enabling MIG Mode..."
sudo nvidia-smi -mig 1

log "Creating 2x 3g.40gb GPU instances..."
sudo nvidia-smi mig -cgi 3g.40gb,3g.40gb -C

log "Retrieving MIG device UUIDs..."
MIG_UUIDS=$(sudo nvidia-smi -L | grep "MIG 3g.40gb" | awk -F'UUID: ' '{print $2}' | awk -F')' '{print $1}')
MIG_A=$(echo "$MIG_UUIDS" | sed -n '1p')
MIG_B=$(echo "$MIG_UUIDS" | sed -n '2p')

if [[ -z "$MIG_A" || -z "$MIG_B" ]]; then
    log "ERROR: Could not retrieve exactly 2 MIG UUIDs."
    exit 1
fi

log "MIG Device A: $MIG_A"
log "MIG Device B: $MIG_B"

log "Starting MIG Docker containers..."
sudo docker run -d --name vllm-mig-a \
    --gpus "\"device=${MIG_A}\"" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096

sudo docker run -d --name vllm-mig-b \
    --gpus "\"device=${MIG_B}\"" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8001:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096

log "Waiting for MIG models to load (this can take a few minutes)..."
for i in {1..36}; do
    if curl -s "http://localhost:8000/v1/models" | grep -q "Qwen/Qwen2.5-7B-Instruct" 2>/dev/null && \
       curl -s "http://localhost:8001/v1/models" | grep -q "Qwen/Qwen2.5-7B-Instruct" 2>/dev/null; then
        log "MIG models are ready!"
        break
    fi
    sleep 5
    if [ $i -eq 36 ]; then
        log "ERROR: MIG models failed to start within timeout."
        exit 1
    fi
done

log "Starting GPU utilization logger for MIG..."
"$SCRIPT_DIR/gpu_util_logger.sh" "$RESULTS_DIR/mig_gpu_util.csv" & 
GPU_LOG_PID=$!

log "Running MIG Test Pattern..."
"$SCRIPT_DIR/run_test_pattern.sh" mig

log "Stopping GPU utilization logger..."
kill $GPU_LOG_PID || true

log "Cleaning up MIG containers..."
sudo docker stop vllm-mig-a vllm-mig-b || true
sudo docker rm vllm-mig-a vllm-mig-b || true

log "--- TEARDOWN ---"
log "Disabling MIG mode and destroying instances..."
sudo nvidia-smi mig -dci || true
sudo nvidia-smi mig -dgi || true
sudo nvidia-smi -mig 0 || true

log "--- RESULTS COMPARISON ---"
"$SCRIPT_DIR/compare_results.sh"

log "============================================"
log " Experiment fully complete! Results are in $RESULTS_DIR"
log "============================================"
