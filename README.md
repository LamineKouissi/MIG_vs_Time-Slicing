# MIG vs Time-Slice Experiment

This repository benchmarks **NVIDIA Multi-Instance GPU (MIG) hardware isolation** against **standard GPU time-slicing** for LLM inference using [vLLM](https://github.com/vllm-project/vllm).

The experiment proves that when two workloads share a GPU via time-slicing, a traffic spike on one workload degrades the other's latency ("noisy neighbor" problem). MIG eliminates this by giving each workload its own dedicated hardware partition.

## Requirements

- **GPU:** NVIDIA A100, A30, H100, or H200 (MIG-capable)
- **Software:** Docker with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), `sudo` access
- **Disk:** ~30 GB free (for the model + Docker images)
- **Time:** ~30–60 minutes for the full experiment

## Quick Start

```bash
git clone https://github.com/LamineKouissi/MIG_vs_Time-Slicing.git
cd MIG_vs_Time-Slicing
chmod +x scripts/*.sh
./scripts/run_experiment.sh
```

That's it. The script handles everything: installing dependencies, downloading the model, running both benchmark phases, and printing the results. You'll be prompted for your `sudo` password.

> **Want to understand what each step does?** Read the [Experiment Guide](mig-experiment-guide.md) — it explains every command, every script, and how they connect to each other.

## Project Structure

```
MIG_vs_Time-Slicing/
├── mig-experiment-guide.md           # Detailed educational guide (start here to learn)
├── README.md                         # This file
├── scripts/
│   ├── run_experiment.sh             # Master script — runs the full experiment
│   ├── run_test_pattern.sh           # Orchestrates the noisy-neighbor scenario
│   ├── run_bench.sh                  # Fires load at a single vLLM instance
│   ├── gpu_util_logger.sh            # Logs GPU utilization to CSV every second
│   └── compare_results.sh           # Prints the final results comparison table
└── results/                          # Created at runtime — all JSON, CSV, and log outputs
```

### How the scripts call each other

```
run_experiment.sh            ← You run this one
├── gpu_util_logger.sh       ← Background: logs GPU stats
├── run_test_pattern.sh      ← Orchestrates the noisy-neighbor test
│   └── run_bench.sh         ← Fires N requests at R req/s at one instance
└── compare_results.sh       ← Reads results/ and prints comparison table
```

## Results

After the experiment completes, `compare_results.sh` prints a table comparing Baseline (Time-Sliced) vs MIG across these metrics:

| Metric | What it means |
|--------|---------------|
| **p99 TTFT** | Worst-case Time To First Token (startup latency) |
| **p99 TPOT** | Worst-case Time Per Output Token (generation speed) |
| **TPOT StdDev** | Generation stability — lower = smoother output |

**What to look for:** Instance B's p99 TTFT should be significantly worse in the Baseline phase than in the MIG phase, proving that hardware isolation prevents noisy-neighbor interference.

## License

See [LICENSE](LICENSE).
