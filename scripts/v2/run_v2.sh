#!/usr/bin/env bash
# Noisy-neighbour benchmark v2: time-slicing vs MPS vs MIG on one GPU.
#
# Two vLLM instances serve the same model on one GPU. For each sharing mode,
# and for REPS repetitions, three windows of WINDOW_S seconds are measured:
#   idle        quiet tenant B at B_RATE req/s, noisy tenant A idle   -> what the mode costs B
#   spikealone  noisy tenant A at A_RATE req/s, B idle                -> A's capacity in the mode
#   contended   A spiking and B steady over the same window           -> interference
#
# Usage:  ./scripts/v2/run_v2.sh            (settings in scripts/v2/config.sh)
#         REPS=3 MODES="timeslice mig" ./scripts/v2/run_v2.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BASE_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
# shellcheck source-path=SCRIPTDIR source=config.sh
source "$SCRIPT_DIR/config.sh"

IMAGE="vllm/vllm-openai:v${VLLM_VERSION}"
VENV="$BASE_DIR/.venv-v2"
RUN_ID="$(date +%Y%m%d-%H%M%S)"
OUT="$BASE_DIR/results/v2/$RUN_ID"
MPS_PIPE=/tmp/nvidia-mps
MPS_LOG=/tmp/nvidia-mps-log
LOGGER_PID=""
POWER_PID=""
SUDO_KEEPALIVE_PID=""
mkdir -p "$OUT"

log() { echo "[$(date -Iseconds)] $*" | tee -a "$OUT/run.log"; }
die() { log "ERROR: $*"; exit 1; }

# ---------------------------------------------------------------- setup ----

keep_sudo_alive() {
    sudo -v
    while true; do sudo -n true; sleep 60; done 2>/dev/null &
    SUDO_KEEPALIVE_PID=$!
}

setup_env() {
    log "Installing prerequisites"
    sudo apt-get update -qq >/dev/null && sudo apt-get install -y -qq curl python3-venv python3-pip >/dev/null
    command -v docker >/dev/null || die "Docker is not installed"
    if ! sudo docker info 2>/dev/null | grep -i nvidia >/dev/null; then
        log "WARNING: 'docker info' lists no NVIDIA runtime; GPU containers may fail to start"
    fi

    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV"
    set +u   # some venv activate scripts read unset variables
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    set -u
    local have
    have=$(python -c 'import importlib.metadata as m; print(m.version("vllm"))' 2>/dev/null || true)
    if [ "$have" != "$VLLM_VERSION" ]; then
        log "Installing vllm==$VLLM_VERSION as the benchmark client (takes a few minutes)"
        pip install -q "vllm==$VLLM_VERSION"
    fi
    python -c 'import matplotlib, numpy, huggingface_hub' 2>/dev/null || pip install -q matplotlib numpy huggingface_hub

    log "Pulling $IMAGE"
    sudo docker pull -q "$IMAGE" >/dev/null
    log "Downloading $MODEL into $HF_CACHE"
    HF_HOME="$HF_CACHE" python -c "from huggingface_hub import snapshot_download; snapshot_download('$MODEL')" >/dev/null
}

record_env() {
    local gpu cuda client power_limit
    gpu=$(nvidia-smi -i "$GPU" --query-gpu=name,driver_version,memory.total --format=csv,noheader || true)
    power_limit=$(nvidia-smi -i "$GPU" --query-gpu=power.limit --format=csv,noheader || true)
    cuda=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9.]*\).*/\1/p' || true)
    client=$(python -c 'import importlib.metadata as m; print(m.version("vllm"))' || true)
    cat > "$OUT/env.txt" <<EOF
date: $(date -Iseconds)
gpu: $gpu
cuda: $cuda
vllm_image: $IMAGE
vllm_client: $client
model: $MODEL
max_model_len: $MAX_MODEL_LEN
tokens_in_out: $IN_LEN/$OUT_LEN
tenant_B: $B_RATE req/s, $B_PROMPTS prompts per window
tenant_A: $A_RATE req/s, $A_PROMPTS prompts per window
window_s: $WINDOW_S
reps: $REPS
modes: $MODES
mig_profile: $MIG_PROFILE
mem_shared: $MEM_SHARED
mem_mig: $MEM_MIG
mps_thread_pct: ${MPS_PCT:-none}
power_limit: $power_limit
power_interval_s: $POWER_INTERVAL_S
EOF
    log "Environment: $gpu | CUDA $cuda | vLLM $client"
}

# -------------------------------------------------------------- servers ----

wait_ready() { # name port
    local name=$1 port=$2
    for _ in $(seq 1 180); do
        if curl -sf "http://localhost:$port/v1/models" >/dev/null 2>&1; then
            log "$name ready on port $port"
            return 0
        fi
        if [ "$(sudo docker inspect -f '{{.State.Running}}' "$name" 2>/dev/null || echo false)" != "true" ]; then
            # shellcheck disable=SC2024  # the log file belongs to the user, not root
            sudo docker logs --tail 80 "$name" >>"$OUT/run.log" 2>&1 || true
            die "$name stopped during startup; its last log lines are in $OUT/run.log"
        fi
        sleep 5
    done
    die "$name was not ready after 15 minutes"
}

start_server() { # name port gpus mem_fraction [extra docker args...]
    local name=$1 port=$2 gpus=$3 mem=$4
    shift 4
    sudo docker run -d --name "$name" --gpus "$gpus" \
        -v "$HF_CACHE:/root/.cache/huggingface" -p "$port:8000" --ipc=host "$@" \
        "$IMAGE" --model "$MODEL" --gpu-memory-utilization "$mem" --max-model-len "$MAX_MODEL_LEN" \
        >/dev/null
    wait_ready "$name" "$port"   # start one at a time so memory profiling doesn't race
}

stop_servers() {
    sudo docker rm -f vllm-a vllm-b >/dev/null 2>&1 || true
}

# ---------------------------------------------------------------- modes ----

setup_timeslice() {
    start_server vllm-a "$PORT_A" all "$MEM_SHARED"
    start_server vllm-b "$PORT_B" all "$MEM_SHARED"
}
teardown_timeslice() { :; }

setup_mps() {
    # Exclusive-process mode makes a non-MPS client fail loudly instead of silently time-slicing.
    sudo nvidia-smi -i "$GPU" -c EXCLUSIVE_PROCESS >/dev/null
    sudo mkdir -p "$MPS_PIPE" "$MPS_LOG"
    sudo env CUDA_VISIBLE_DEVICES="$GPU" CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" \
        CUDA_MPS_LOG_DIRECTORY="$MPS_LOG" nvidia-cuda-mps-control -d
    local args=(-v "$MPS_PIPE:$MPS_PIPE" -e "CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE" --pid=host)
    if [ -n "$MPS_PCT" ]; then
        args+=(-e "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=$MPS_PCT")
    fi
    start_server vllm-a "$PORT_A" all "$MEM_SHARED" "${args[@]}"
    start_server vllm-b "$PORT_B" all "$MEM_SHARED" "${args[@]}"
    if ! nvidia-smi | grep nvidia-cuda-mps-server >/dev/null; then
        die "No MPS server process on the GPU; the instances are not sharing through MPS"
    fi
}
teardown_mps() {
    echo quit | sudo env CUDA_MPS_PIPE_DIRECTORY="$MPS_PIPE" nvidia-cuda-mps-control >/dev/null 2>&1 || true
    sleep 2
    sudo nvidia-smi -i "$GPU" -c DEFAULT >/dev/null 2>&1 || true
}

mig_current() {
    nvidia-smi -i "$GPU" --query-gpu=mig.mode.current --format=csv,noheader 2>/dev/null | tr -d ' ' || true
}

setup_mig() {
    sudo nvidia-smi -i "$GPU" -mig 1 >/dev/null 2>&1 || true
    if [ "$(mig_current)" != "Enabled" ]; then
        log "MIG is pending; trying a GPU reset"
        sudo nvidia-smi -i "$GPU" -r >/dev/null 2>&1 || true
    fi
    [ "$(mig_current)" = "Enabled" ] || die "Could not enable MIG on GPU $GPU (the host may need a reboot)"

    local profiles
    profiles=$(sudo nvidia-smi mig -i "$GPU" -lgip 2>&1 || true)
    echo "$profiles" >"$OUT/mig/profiles.txt"
    if ! echo "$profiles" | grep -F "$MIG_PROFILE" >/dev/null; then
        die "This GPU does not offer MIG profile $MIG_PROFILE; pick one from $OUT/mig/profiles.txt and set MIG_PROFILE"
    fi
    sudo nvidia-smi mig -i "$GPU" -cgi "$MIG_PROFILE,$MIG_PROFILE" -C >/dev/null

    local uuids=()
    mapfile -t uuids < <(nvidia-smi -L | grep -F "MIG $MIG_PROFILE" | sed -E 's/.*UUID: ([^)]+)\).*/\1/')
    [ "${#uuids[@]}" -ge 2 ] || die "Expected 2 MIG devices, found ${#uuids[@]}"
    log "MIG devices: A=${uuids[0]} B=${uuids[1]}"
    start_server vllm-a "$PORT_A" "\"device=${uuids[0]}\"" "$MEM_MIG"
    start_server vllm-b "$PORT_B" "\"device=${uuids[1]}\"" "$MEM_MIG"
}
teardown_mig() {
    sudo nvidia-smi mig -i "$GPU" -dci >/dev/null 2>&1 || true
    sudo nvidia-smi mig -i "$GPU" -dgi >/dev/null 2>&1 || true
    sudo nvidia-smi -i "$GPU" -mig 0 >/dev/null 2>&1 || true
}

# ------------------------------------------------------------ measuring ----

bench() { # port prompts rate seed outfile
    local port=$1 n=$2 rate=$3 seed=$4 file=$5
    vllm bench serve --backend openai --base-url "http://localhost:$port" --endpoint /v1/completions \
        --model "$MODEL" --dataset-name random --random-input-len "$IN_LEN" --random-output-len "$OUT_LEN" \
        --ignore-eos --num-prompts "$n" --request-rate "$rate" --seed "$seed" \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,90,99 \
        --save-result --save-detailed --result-filename "$file" --disable-tqdm \
        >"${file%.json}.log" 2>&1 || die "benchmark failed; see ${file%.json}.log"
}

warmup() { # mode
    log "[$1] warm-up"
    bench "$PORT_A" "$WARMUP_PROMPTS" 5 7 "$OUT/$1/warmup_A.json"
    bench "$PORT_B" "$WARMUP_PROMPTS" 5 7 "$OUT/$1/warmup_B.json"
}

now() { date +%s.%N; }

mark_window() { # mode rep window start_s   (end = now); the analysis matches power samples to windows
    echo "$1,$2,$3,$4,$(now)" >>"$OUT/windows.csv"
}

run_rep() { # mode rep
    local mode=$1 rep=$2 d="$OUT/$1/rep$2" s=$(( $2 * 100 )) t0
    mkdir -p "$d"

    log "[$mode rep $rep/$REPS] B alone at $B_RATE req/s (A idle)"
    t0=$(now)
    bench "$PORT_B" "$B_PROMPTS" "$B_RATE" $(( s + 1 )) "$d/idle_B.json"
    mark_window "$mode" "$rep" idle "$t0"
    sleep "$COOLDOWN_S"

    log "[$mode rep $rep/$REPS] A spike alone at $A_RATE req/s (B idle)"
    t0=$(now)
    bench "$PORT_A" "$A_PROMPTS" "$A_RATE" $(( s + 2 )) "$d/spikealone_A.json"
    mark_window "$mode" "$rep" spikealone "$t0"
    sleep "$COOLDOWN_S"

    log "[$mode rep $rep/$REPS] A spiking and B steady over the same window"
    t0=$(now)
    bench "$PORT_A" "$A_PROMPTS" "$A_RATE" $(( s + 3 )) "$d/contended_A.json" &
    local pid_a=$!
    bench "$PORT_B" "$B_PROMPTS" "$B_RATE" $(( s + 4 )) "$d/contended_B.json" &
    local pid_b=$!
    wait "$pid_a"
    wait "$pid_b"
    mark_window "$mode" "$rep" contended "$t0"
    sleep "$COOLDOWN_S"
}

start_gpu_logger() { # mode
    # DCGM profiling counters (SM and DRAM activity) when available, otherwise nvidia-smi.
    if command -v dcgmi >/dev/null 2>&1 && dcgmi discovery -l >/dev/null 2>&1; then
        dcgmi dmon -e 1001,1002,1005 -d 1000 >"$OUT/$1/dcgm.txt" 2>&1 &
    else
        "$BASE_DIR/scripts/gpu_util_logger.sh" "$OUT/$1/nvidia_smi.csv" >/dev/null 2>&1 &
    fi
    LOGGER_PID=$!
    # Board power for the whole GPU, in every mode (MIG included), for energy per token.
    "$SCRIPT_DIR/power_logger.sh" "$GPU" "$OUT/$1/power.csv" "$POWER_INTERVAL_S" >/dev/null 2>&1 &
    POWER_PID=$!
}
stop_gpu_logger() {
    local pid
    for pid in $LOGGER_PID $POWER_PID; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    LOGGER_PID=""
    POWER_PID=""
}

# ------------------------------------------------------------- lifecycle ----

cleanup() {
    local code=$?
    stop_gpu_logger
    stop_servers
    teardown_mps
    teardown_mig
    if [ -n "$SUDO_KEEPALIVE_PID" ]; then kill "$SUDO_KEEPALIVE_PID" 2>/dev/null || true; fi
    if [ "$code" -ne 0 ]; then
        log "Stopped early (exit $code). The GPU is back in normal mode. Partial results: $OUT"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT TERM

main() {
    log "Run $RUN_ID | modes: $MODES | reps: $REPS | results: $OUT"
    keep_sudo_alive
    stop_servers; teardown_mps; teardown_mig     # start from a clean GPU
    setup_env
    record_env
    echo "mode,rep,window,start_s,end_s" >"$OUT/windows.csv"

    for mode in $MODES; do
        case "$mode" in
            timeslice | mps | mig) ;;
            *) die "Unknown mode '$mode' (use timeslice, mps or mig)" ;;
        esac
        log "=== Mode: $mode ==="
        mkdir -p "$OUT/$mode"
        "setup_$mode"
        nvidia-smi >"$OUT/$mode/nvidia-smi.txt" 2>&1 || true
        warmup "$mode"
        start_gpu_logger "$mode"
        for rep in $(seq 1 "$REPS"); do
            run_rep "$mode" "$rep"
        done
        stop_gpu_logger
        stop_servers
        "teardown_$mode"
    done

    log "All modes finished; analysing"
    python "$SCRIPT_DIR/analyze_v2.py" "$OUT" | tee -a "$OUT/run.log"
    log "Done. Summary: $OUT/summary.md | Figure: $OUT/v2_summary.png"
}

main "$@"
