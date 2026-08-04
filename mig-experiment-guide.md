# MIG vs Time-Slice Experiment: The Complete Guide

This guide walks you through running a benchmark that compares **NVIDIA Multi-Instance GPU (MIG) hardware isolation** against **standard time-slicing** for LLM inference using vLLM. The experiment measures whether a "noisy neighbor" workload on one GPU partition degrades the latency of a second, co-located workload.

> **What you'll need:** A machine with an **NVIDIA A100 or H100** GPU (MIG-capable), Docker with the NVIDIA container runtime, and `sudo` access.

---

## Table of Contents

1. [After Cloning: What Do I Do?](#1--after-cloning-what-do-i-do)
2. [How the Scripts Relate to Each Other](#2--how-the-scripts-relate-to-each-other)
3. [Option A: Run the Entire Experiment Automatically](#3--option-a-run-the-entire-experiment-automatically)
4. [Option B: Run Each Step Manually](#4--option-b-run-each-step-manually)
   - [Step 0 — Verify GPU Support](#step-0--verify-gpu-support-for-mig)
   - [Step 1 — Install Prerequisites](#step-1--install-prerequisites)
   - [Step 2 — Phase 1: Baseline (Time-Sliced)](#step-2--phase-1-baseline-time-sliced)
   - [Step 3 — Phase 2: MIG (Hardware Isolation)](#step-3--phase-2-mig-hardware-isolation)
   - [Step 4 — Compare Results](#step-4--compare-results)
   - [Step 5 — Teardown](#step-5--teardown)
5. [Understanding Every Script in Detail](#5--understanding-every-script-in-detail)
6. [Troubleshooting & FAQ](#6--troubleshooting--faq)

---

## 1 — After Cloning: What Do I Do?

Once you've cloned the repository onto your GPU machine, here is the **complete checklist** to go from zero to results:

```bash
# 1. Clone the repo (if you haven't already)
git clone https://github.com/LamineKouissi/MIG_vs_Time-Slicing.git
cd MIG_vs_Time-Slicing

# 2. Make all scripts executable
chmod +x scripts/*.sh

# 3. Choose ONE of the two options below:
```

### Option A — Fully Automated (recommended for first-time users)

Run a single command. It handles everything: installing dependencies, downloading the model, running both benchmark phases, and printing the results.

```bash
./scripts/run_experiment.sh
```

You'll be prompted for your `sudo` password. The full run takes approximately **30–60 minutes** depending on your hardware and network speed.

### Option B — Manual, Step-by-Step

If you want to understand what's happening at each stage, skip to [Section 4](#4--option-b-run-each-step-manually) and follow each step individually.

---

## 2 — How the Scripts Relate to Each Other

There are **5 scripts** in the `scripts/` directory. They form a hierarchy — the master script calls the others in order. Here's how they connect:

```
run_experiment.sh          ← Master orchestrator. The only script you need to call directly.
│
├── 1. Installs prerequisites (apt, pip, model download)
│
├── 2. PHASE 1 — BASELINE (Time-Sliced)
│   ├── Starts 2 Docker containers (vllm-a, vllm-b) sharing the full GPU
│   ├── gpu_util_logger.sh ← Runs in the background, logs GPU metrics every 1s
│   └── run_test_pattern.sh "baseline"
│       ├── run_bench.sh "baseline_instanceB_steady" 8001 200 2    ← Steady load
│       │                                                            (background)
│       ├── sleep 30 (let steady load warm up)
│       └── run_bench.sh "baseline_instanceA_SPIKE"  8000 500 50   ← Spike load
│   └── Stops & removes containers
│
├── 3. PHASE 2 — MIG (Hardware Isolation)
│   ├── Enables MIG mode, creates 2× 3g.40gb partitions
│   ├── Starts 2 Docker containers, each pinned to one MIG slice
│   ├── gpu_util_logger.sh ← Same logger, different output file
│   └── run_test_pattern.sh "mig"
│       ├── run_bench.sh "mig_instanceB_steady" 8001 200 2         ← Steady load
│       ├── sleep 30
│       └── run_bench.sh "mig_instanceA_SPIKE"  8000 500 50        ← Spike load
│   └── Stops & removes containers
│
├── 4. TEARDOWN — Destroys MIG partitions, disables MIG mode
│
└── 5. compare_results.sh  ← Reads the JSON files from both phases, prints a table
```

### Script-by-Script Summary

| Script | Purpose | Called by | You run it directly? |
|--------|---------|-----------|---------------------|
| `run_experiment.sh` | Master orchestrator — runs the full experiment end-to-end | You (or CI) | **Yes** (Option A) |
| `run_test_pattern.sh` | Simulates the "noisy neighbor" scenario for one phase | `run_experiment.sh` | Only in Option B |
| `run_bench.sh` | Fires a specific load (N prompts at R req/s) at one vLLM instance | `run_test_pattern.sh` | Only in Option B |
| `gpu_util_logger.sh` | Logs GPU utilization to CSV every second | `run_experiment.sh` | Only in Option B |
| `compare_results.sh` | Reads JSON results from `results/` and prints a comparison table | `run_experiment.sh` | Only in Option B |

---

## 3 — Option A: Run the Entire Experiment Automatically

This is the simplest path. One command does everything.

```bash
# From the repo root:
chmod +x scripts/*.sh
./scripts/run_experiment.sh
```

**What happens behind the scenes** (in order):

1. Installs system packages (`jq`, `bc`, `curl`, `python3-venv`)
2. Creates a Python virtual environment and installs `vllm`
3. Pre-downloads the `Qwen/Qwen2.5-7B-Instruct` model (~14 GB)
4. **Phase 1 (Baseline):** Launches 2 vLLM containers sharing the full GPU via time-slicing, runs the noisy-neighbor test, collects results
5. **Phase 2 (MIG):** Enables MIG mode, creates 2 hardware-isolated GPU partitions, launches 2 containers (one per partition), runs the same test, collects results
6. Tears down MIG partitions and disables MIG mode
7. Runs `compare_results.sh` to print the final comparison table

**What to expect in your terminal:**

```
[2026-08-04T20:00:00+00:00] ============================================
[2026-08-04T20:00:00+00:00]  Starting Time-Slice vs MIG Experiment
[2026-08-04T20:00:00+00:00] ============================================
[2026-08-04T20:00:00+00:00] Step 1: Installing prerequisites...
...
[2026-08-04T20:00:45+00:00] --- PHASE 1: BASELINE (Time-Sliced) ---
...
[2026-08-04T20:15:00+00:00] --- PHASE 2: MIG (Hardware Isolation) ---
...
[2026-08-04T20:30:00+00:00] --- RESULTS COMPARISON ---

=================================================================
  MIG ISOLATION EXPERIMENT — RESULTS COMPARISON
=================================================================

--- Phase: BASELINE ---
Label                              p50 TTFT     p99 TTFT   ...
baseline_instanceA_SPIKE            12.3ms      245.7ms    ...
baseline_instanceB_steady           15.1ms      312.4ms    ...

--- Phase: MIG ---
Label                              p50 TTFT     p99 TTFT   ...
mig_instanceA_SPIKE                 10.2ms       48.3ms    ...
mig_instanceB_steady                11.8ms       22.1ms    ...
```

> **Key result to look for:** Instance B's p99 TTFT in the BASELINE phase should be much worse than in the MIG phase. This proves that MIG's hardware isolation prevents noisy-neighbor interference.

**When the script finishes**, your results are saved in:

```
results/
├── baseline_instanceA_SPIKE.json    # Raw benchmark data
├── baseline_instanceA_SPIKE.log     # vLLM bench console output
├── baseline_instanceB_steady.json
├── baseline_instanceB_steady.log
├── baseline_gpu_util.csv            # GPU utilization timeline
├── mig_instanceA_SPIKE.json
├── mig_instanceA_SPIKE.log
├── mig_instanceB_steady.json
├── mig_instanceB_steady.log
└── mig_gpu_util.csv
```

---

## 4 — Option B: Run Each Step Manually

Use this path if you want to understand, tweak, or debug each stage. Every command below is what `run_experiment.sh` does internally.

### Step 0 — Verify GPU Support for MIG

Before starting, confirm your GPU supports MIG and check its current state.

```bash
# Check if MIG mode is currently enabled or disabled
nvidia-smi --query-gpu=mig.mode.current --format=csv

# List all MIG profiles this GPU supports (e.g., 1g.10gb, 2g.20gb, 3g.40gb, 7g.80gb)
# The profile format is: <Compute_Slices>g.<Memory_Size>gb
nvidia-smi mig -lgip
```

---

### Step 1 — Install Prerequisites

```bash
# Install system utilities
sudo apt update && sudo apt install -y jq bc curl python3-pip python3-venv

# Create a working directory for results
mkdir -p results

# Set up an isolated Python environment
python3 -m venv .venv
source .venv/bin/activate

# Install vLLM (the inference engine we benchmark) and huggingface_hub (for model download)
pip install vllm huggingface_hub

# Verify Docker + NVIDIA runtime are available
sudo docker info | grep -i runtime
# You should see "nvidia" in the output. If not, install nvidia-container-toolkit.

# Pre-download the model so benchmark timing isn't skewed by network speed
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"
```

---

### Step 2 — Phase 1: Baseline (Time-Sliced)

In this phase, two vLLM containers share the **same physical GPU** via time-slicing. There is no hardware isolation — the NVIDIA driver rapidly switches the GPU between both workloads.

#### 2a. Launch 2 vLLM Containers on the Full GPU

```bash
# Instance A — will receive the traffic spike (port 8000)
sudo docker run -d --name vllm-a --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.45 \
    --max-model-len 4096

# Instance B — will receive steady load; we monitor if Instance A's spike degrades it (port 8001)
sudo docker run -d --name vllm-b --gpus all \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8001:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.45 \
    --max-model-len 4096
```

> **Why `--gpu-memory-utilization 0.45`?** Both containers see the full GPU memory. If each tries to use >50%, the second one will crash with an Out-Of-Memory error. We cap at 45% to leave headroom.

> **Why `--ipc=host`?** vLLM uses shared memory for optimized tensor operations. This flag allows it.

#### 2b. Wait for Models to Load

The models take 1–3 minutes to load into GPU memory. Poll the health endpoint:

```bash
# Wait until both instances respond with the model name
while true; do
    A=$(curl -s http://localhost:8000/v1/models 2>/dev/null)
    B=$(curl -s http://localhost:8001/v1/models 2>/dev/null)
    if echo "$A" | grep -q "Qwen" && echo "$B" | grep -q "Qwen"; then
        echo "Both instances are ready!"
        break
    fi
    echo "Waiting for models to load..."
    sleep 5
done
```

#### 2c. Start the GPU Utilization Logger

```bash
# This runs in the background, writing GPU stats every second to a CSV
./scripts/gpu_util_logger.sh results/baseline_gpu_util.csv &
GPU_LOG_PID=$!
```

#### 2d. Run the Noisy-Neighbor Test

```bash
# This orchestrates the "noisy neighbor" scenario:
#   1. Starts a STEADY load on Instance B (200 prompts at 2 req/s)
#   2. Waits 30 seconds for Instance B to warm up
#   3. Fires a SPIKE on Instance A (500 prompts at 50 req/s)
#   4. Waits for both to finish and saves JSON results
./scripts/run_test_pattern.sh baseline
```

> **What's happening:** Instance B is humming along at a gentle 2 requests/second. Then Instance A gets slammed with 50 requests/second. Because they share the GPU via time-slicing, Instance A's spike should **steal GPU cycles** from Instance B, causing its latency (TTFT, TPOT) to degrade.

#### 2e. Stop the Logger and Clean Up

```bash
# Stop the background GPU logger
kill $GPU_LOG_PID

# Remove the baseline containers
sudo docker stop vllm-a vllm-b
sudo docker rm vllm-a vllm-b
```

---

### Step 3 — Phase 2: MIG (Hardware Isolation)

Now we carve the physical GPU into two **hardware-isolated partitions** using MIG. Each partition gets its own dedicated compute units (SMs) and memory bandwidth.

#### 3a. Enable MIG Mode and Create Partitions

```bash
# Put the GPU into MIG mode
sudo nvidia-smi -mig 1

# Create 2 GPU Instances with the 3g.40gb profile (3 compute slices, ~40GB memory each)
# The -C flag also creates the Compute Instances inside them automatically
sudo nvidia-smi mig -cgi 3g.40gb,3g.40gb -C
```

#### 3b. Find the MIG Device UUIDs

Each MIG partition gets a unique UUID. We need these to pin Docker containers to specific partitions.

```bash
# List all GPUs including MIG devices — look for the "MIG 3g.40gb" lines
sudo nvidia-smi -L

# Example output:
# GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-xxxxx)
#   MIG 3g.40gb  Device 0: (UUID: MIG-aaaaa-bbbbb-ccccc-ddddd-eeeee)
#   MIG 3g.40gb  Device 1: (UUID: MIG-fffff-ggggg-hhhhh-iiiii-jjjjj)

# Extract just the UUIDs programmatically:
MIG_UUIDS=$(sudo nvidia-smi -L | grep "MIG 3g.40gb" | awk -F'UUID: ' '{print $2}' | awk -F')' '{print $1}')
MIG_A=$(echo "$MIG_UUIDS" | sed -n '1p')
MIG_B=$(echo "$MIG_UUIDS" | sed -n '2p')

echo "MIG Device A: $MIG_A"
echo "MIG Device B: $MIG_B"
```

#### 3c. Launch 2 vLLM Containers, Each Pinned to One MIG Slice

```bash
# Instance A — pinned to MIG partition 0 (port 8000)
sudo docker run -d --name vllm-mig-a \
    --gpus "\"device=${MIG_A}\"" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8000:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096

# Instance B — pinned to MIG partition 1 (port 8001)
sudo docker run -d --name vllm-mig-b \
    --gpus "\"device=${MIG_B}\"" \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -p 8001:8000 --ipc=host \
    vllm/vllm-openai:latest \
    --model Qwen/Qwen2.5-7B-Instruct \
    --gpu-memory-utilization 0.90 \
    --max-model-len 4096
```

> **Why `--gpu-memory-utilization 0.90` now?** Each container has its own isolated memory partition. There's no sharing, so we can safely use 90% of the available memory.

> **Why `--gpus "\"device=${MIG_A}\""` ?** This tells Docker to expose only this specific MIG partition to the container. The container sees it as a full, independent GPU.

#### 3d. Wait for Models to Load

```bash
while true; do
    A=$(curl -s http://localhost:8000/v1/models 2>/dev/null)
    B=$(curl -s http://localhost:8001/v1/models 2>/dev/null)
    if echo "$A" | grep -q "Qwen" && echo "$B" | grep -q "Qwen"; then
        echo "Both MIG instances are ready!"
        break
    fi
    echo "Waiting for MIG models to load..."
    sleep 5
done
```

#### 3e. Start Logger and Run the Same Test

```bash
# Start GPU logger for the MIG phase
./scripts/gpu_util_logger.sh results/mig_gpu_util.csv &
GPU_LOG_PID=$!

# Run the EXACT same noisy-neighbor test
./scripts/run_test_pattern.sh mig

# Stop the logger
kill $GPU_LOG_PID
```

> **Expected behavior:** The spike on Instance A will NOT degrade Instance B's latency, because the hardware partitions are fully isolated.

#### 3f. Clean Up MIG Containers

```bash
sudo docker stop vllm-mig-a vllm-mig-b
sudo docker rm vllm-mig-a vllm-mig-b
```

---

### Step 4 — Compare Results

```bash
./scripts/compare_results.sh
```

This reads all `results/*.json` files and prints a comparison table. The key columns to focus on:

| Metric | What it means | What to compare |
|--------|---------------|-----------------|
| **p99 TTFT** | 99th-percentile Time To First Token — worst-case startup latency | Instance B's p99 TTFT: Baseline vs MIG |
| **p99 TPOT** | 99th-percentile Time Per Output Token — worst-case generation speed | Instance B's p99 TPOT: Baseline vs MIG |
| **TPOT StdDev** | Standard deviation of token generation time — measures jitter/stability | Lower = more stable. MIG should be much lower |

> **What proves MIG works:** If Instance B's p99 TTFT in the BASELINE phase is significantly worse than in the MIG phase, it means that time-slicing allowed Instance A's spike to interfere with Instance B. Under MIG, the numbers should be nearly identical between the spike and no-spike scenarios.

---

### Step 5 — Teardown

Return the GPU to its normal (non-MIG) state:

```bash
# Destroy Compute Instances
sudo nvidia-smi mig -dci
# Destroy GPU Instances
sudo nvidia-smi mig -dgi
# Disable MIG mode
sudo nvidia-smi -mig 0
```

---

## 5 — Understanding Every Script in Detail

### `run_bench.sh` — The Load Generator

This is the lowest-level script. It fires a controlled number of requests at a single vLLM instance and saves the results.

```bash
#!/bin/bash
set -euo pipefail

LABEL="$1"          # A name for this run (e.g., "baseline_instanceA_SPIKE")
PORT="$2"           # Which port/instance to hit (8000 or 8001)
NUM_PROMPTS="${3:-200}"    # How many requests to send (default: 200)
REQUEST_RATE="${4:-2}"     # Requests per second (default: 2)
RESULTS_DIR="$(cd "$(dirname "$0")/.." && pwd)/results"
mkdir -p "$RESULTS_DIR"

OUTFILE="${RESULTS_DIR}/${LABEL}.json"

vllm bench serve \
    --backend openai \                       # Use OpenAI-compatible API format
    --base-url "http://localhost:${PORT}" \   # Target this vLLM instance
    --endpoint /v1/completions \             # Use the completions endpoint
    --model "Qwen/Qwen2.5-7B-Instruct" \    # Must match the loaded model
    --dataset-name random \                  # Generate random prompts (not a real dataset)
    --random-input-len 128 \                 # Each prompt is 128 tokens
    --random-output-len 128 \                # Request 128 output tokens per prompt
    --num-prompts "$NUM_PROMPTS" \            # Total number of prompts to send
    --request-rate "$REQUEST_RATE" \          # Arrival rate in requests/second
    --save-result \                          # Save detailed metrics to JSON
    --result-filename "$OUTFILE"             # Output path for the JSON results
```

**Usage:** `./scripts/run_bench.sh <label> <port> [num_prompts] [request_rate]`

---

### `run_test_pattern.sh` — The Noisy-Neighbor Scenario

This script orchestrates the "noisy neighbor" experiment for one phase. It calls `run_bench.sh` twice — once for steady load and once for the spike.

```bash
#!/bin/bash
set -euo pipefail

PHASE="$1"   # "baseline" or "mig"

# 1. Start steady load on Instance B (port 8001) — 200 prompts at 2 req/s
#    This runs in the BACKGROUND so we can launch the spike simultaneously
./scripts/run_bench.sh "${PHASE}_instanceB_steady" 8001 200 2 &
PID_B=$!

# 2. Wait 30 seconds for the steady load to warm up and establish a baseline
sleep 30

# 3. Fire a massive spike at Instance A (port 8000) — 500 prompts at 50 req/s
#    This is the "noisy neighbor" that should (or shouldn't) degrade Instance B
./scripts/run_bench.sh "${PHASE}_instanceA_SPIKE" 8000 500 50 &
PID_A=$!

# 4. Wait for both benchmarks to finish
wait $PID_B $PID_A
```

**The key question this answers:** Does the spike on Instance A cause Instance B's latency to degrade?
- **Baseline (time-sliced):** YES — they share GPU compute, so the spike steals cycles.
- **MIG (hardware-isolated):** NO — each instance has its own dedicated hardware.

---

### `gpu_util_logger.sh` — GPU Metrics Collector

Runs in the background during each phase, polling `nvidia-smi` every second.

```bash
#!/bin/bash
set -euo pipefail

OUTFILE="${1:-gpu_util.csv}"

# Write CSV header
echo "timestamp,gpu_util_pct,mem_used_mib,mem_total_mib" > "$OUTFILE"

# Loop forever (killed by the parent script when the phase ends)
while true; do
    TIMESTAMP=$(date -Iseconds)
    DATA=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total \
           --format=csv,noheader,nounits 2>/dev/null | head -1)
    echo "${TIMESTAMP},${DATA}" >> "$OUTFILE"
    sleep 1
done
```

**Output:** A CSV file like `results/baseline_gpu_util.csv` that you can plot to visualize GPU utilization over time.

---

### `compare_results.sh` — Results Formatter

Reads all the JSON result files and prints a formatted comparison table.

It extracts these metrics using `jq`:
- **p50/p99 TTFT (ms):** Time To First Token — how long before the first token is generated
- **p50/p99 TPOT (ms):** Time Per Output Token — how fast tokens are generated
- **TPOT StdDev (ms):** Standard deviation of TPOT — measures generation stability/jitter
- **Throughput (req/s):** Overall request processing rate

**Usage:** `./scripts/compare_results.sh` (reads from `results/` automatically)

---

### `run_experiment.sh` — The Master Script

This is the top-level orchestrator. It calls all the other scripts in the correct order and handles all the Docker and MIG lifecycle management. You never need to call the other scripts individually unless you're doing a manual run (Option B).

**Execution flow:**

```
1. Install prerequisites (apt, pip)
2. Download model
3. PHASE 1 — BASELINE
   a. Start 2 Docker containers (time-sliced, --gpu-memory-utilization 0.45)
   b. Wait for health checks
   c. Start gpu_util_logger.sh (background)
   d. Run run_test_pattern.sh "baseline"
   e. Kill logger, stop & remove containers
4. PHASE 2 — MIG
   a. Enable MIG mode (nvidia-smi -mig 1)
   b. Create 2× 3g.40gb partitions
   c. Extract MIG UUIDs
   d. Start 2 Docker containers (one per MIG slice, --gpu-memory-utilization 0.90)
   e. Wait for health checks
   f. Start gpu_util_logger.sh (background)
   g. Run run_test_pattern.sh "mig"
   h. Kill logger, stop & remove containers
5. TEARDOWN — Destroy MIG partitions, disable MIG mode
6. Run compare_results.sh — Print final comparison
```

---

## 6 — Troubleshooting & FAQ

### "My GPU doesn't support MIG"

MIG is only available on **A100, A30, H100, and H200** GPUs. Consumer GPUs (RTX 3090, 4090, etc.) do not support MIG. Check with:
```bash
nvidia-smi --query-gpu=name,mig.mode.current --format=csv
```

### "Docker container crashes with OOM"

In the baseline phase, both containers share the full GPU memory. If `--gpu-memory-utilization` is set too high (e.g., 0.50+), the second container can't allocate enough VRAM. The scripts use `0.45` for safety. You can lower this value if you're running a larger model.

### "MIG creation fails with 'insufficient resources'"

This means the GPU doesn't have enough free resources for the requested profile. Possible causes:
- MIG instances from a previous run still exist. Clean up with:
  ```bash
  sudo nvidia-smi mig -dci && sudo nvidia-smi mig -dgi
  ```
- The profile is too large. Run `nvidia-smi mig -lgip` to see available profiles.

### "curl: connection refused when health-checking"

The vLLM containers take 1–3 minutes to load the model into GPU memory. Wait and retry. Check container logs:
```bash
sudo docker logs vllm-a
```

### "The script fails mid-run. How do I clean up?"

```bash
# Stop and remove any leftover containers
sudo docker stop vllm-a vllm-b vllm-mig-a vllm-mig-b 2>/dev/null
sudo docker rm vllm-a vllm-b vllm-mig-a vllm-mig-b 2>/dev/null

# Destroy any leftover MIG partitions
sudo nvidia-smi mig -dci 2>/dev/null
sudo nvidia-smi mig -dgi 2>/dev/null
sudo nvidia-smi -mig 0 2>/dev/null
```

### "Can I use a different model?"

Yes. Replace `Qwen/Qwen2.5-7B-Instruct` everywhere it appears (in `run_bench.sh` and in the Docker `--model` flag). Make sure the model fits within the MIG partition's memory. For `3g.40gb` partitions, models up to ~15B parameters (in FP16) typically work.

### "Can I run this on a multi-GPU machine?"

Yes, but you'll need to target a specific GPU. Add `--gpus '"device=0"'` to the Docker commands for the baseline phase to ensure both containers share the same physical GPU. For MIG, the UUID already pins to a specific partition on a specific GPU.