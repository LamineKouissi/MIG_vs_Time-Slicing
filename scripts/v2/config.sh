# shellcheck shell=bash disable=SC2034
# Benchmark v2 settings. Override any value from the environment, e.g.
#   REPS=3 MODES="timeslice mig" ./scripts/v2/run_v2.sh

# --- Software (pinned so runs are reproducible) ---
VLLM_VERSION="${VLLM_VERSION:-0.29.0}"          # server image tag and client pip version
MODEL="${MODEL:-Qwen/Qwen2.5-7B-Instruct}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"

# --- Hardware ---
GPU="${GPU:-0}"                                  # physical GPU index
MIG_PROFILE="${MIG_PROFILE:-3g.40gb}"            # must exist on your GPU: nvidia-smi mig -lgip
MEM_SHARED="${MEM_SHARED:-0.45}"                 # per-instance memory fraction on the shared GPU
MEM_MIG="${MEM_MIG:-0.90}"                       # per-instance memory fraction inside a MIG slice
MPS_PCT="${MPS_PCT:-}"                           # optional CUDA_MPS_ACTIVE_THREAD_PERCENTAGE per instance (empty = no cap)
POWER_INTERVAL_S="${POWER_INTERVAL_S:-0.5}"      # GPU power sampling interval, seconds

# --- Workload ---
IN_LEN="${IN_LEN:-128}"                          # prompt tokens per request
OUT_LEN="${OUT_LEN:-128}"                        # generated tokens per request (EOS ignored)
B_RATE="${B_RATE:-2}"                            # quiet tenant B, req/s
A_RATE="${A_RATE:-50}"                           # noisy tenant A during its spike, req/s
WINDOW_S="${WINDOW_S:-30}"                       # length of each measured window, seconds
B_PROMPTS=$(( B_RATE * WINDOW_S ))
A_PROMPTS=$(( A_RATE * WINDOW_S ))
WARMUP_PROMPTS="${WARMUP_PROMPTS:-30}"
COOLDOWN_S="${COOLDOWN_S:-10}"

# --- Experiment ---
MODES="${MODES:-timeslice mps mig}"
REPS="${REPS:-5}"
PORT_A="${PORT_A:-8000}"
PORT_B="${PORT_B:-8001}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
