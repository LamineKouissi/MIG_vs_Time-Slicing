# MIG vs MPS vs Time-Slicing for LLM Inference

Two [vLLM](https://github.com/vllm-project/vllm) instances serving Qwen2.5-7B share one GPU. This benchmark measures how much a traffic spike on one instance (the noisy tenant, **A**) hurts the other (the quiet tenant, **B**), and what each way of sharing the GPU costs when nothing is contending:

- **Time-slicing:** the default when two processes share a GPU; their kernels take turns.
- **MPS** (CUDA Multi-Process Service): kernels from both processes run at the same time, without hard isolation.
- **MIG** (Multi-Instance GPU): each instance gets its own hardware partition (default: two `3g.40gb` slices).

## Requirements

- **GPU:** a MIG-capable GPU that offers the profile you choose. The default `3g.40gb` exists on 80 GB A100 and H100 cards. Other GPUs use other profile names (for example `3g.20gb` on an A100 40GB). Check with `nvidia-smi mig -lgip` and set `MIG_PROFILE`.
- **Software:** Docker with the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html), `sudo` access.
- **Disk:** about 40 GB (vLLM client, server image, model).
- **Time:** about 1 hour with the defaults (3 modes × 5 runs).

## Quick start (v2)

```bash
git clone https://github.com/LamineKouissi/MIG_vs_Time-Slicing.git
cd MIG_vs_Time-Slicing
chmod +x scripts/*.sh scripts/v2/*.sh
./scripts/v2/run_v2.sh
```

Results land in `results/v2/<run-id>/`:

- `summary.md` and `summary.csv`: every metric for every mode and window, plus headline numbers
- `energy.csv`: mean GPU power, energy and energy per output token for every mode and window (also in `summary.md`)
- `v2_summary.png` / `.svg`: the figure
- `env.txt`: GPU, driver, CUDA and vLLM versions and every setting used
- `<mode>/rep<N>/*.json`: raw per-request results from `vllm bench serve --save-detailed`
- `<mode>/power.csv` and `windows.csv`: raw GPU power samples and the start and end time of every window

Settings live in [`scripts/v2/config.sh`](scripts/v2/config.sh) and can be overridden from the environment:

```bash
REPS=3 MODES="timeslice mig" ./scripts/v2/run_v2.sh
```

| Variable | Default | Meaning |
| --- | --- | --- |
| `VLLM_VERSION` | `0.29.0` | Server image tag and client version (pinned) |
| `MODES` | `timeslice mps mig` | Sharing modes to test |
| `REPS` | `5` | Runs per mode |
| `MIG_PROFILE` | `3g.40gb` | MIG slice for each instance |
| `B_RATE` / `A_RATE` | `2` / `50` | Request rates of the quiet and noisy tenant (req/s, integers) |
| `WINDOW_S` | `30` | Length of each measured window (s) |
| `MPS_PCT` | empty | Optional `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` cap per instance |
| `POWER_INTERVAL_S` | `0.5` | GPU power sampling interval (s) |

## What v2 measures

For each mode and each run, three windows of `WINDOW_S` seconds:

| Window | Load | What it answers |
| --- | --- | --- |
| `idle` | B at 2 req/s, A idle | What the mode costs B when nothing contends |
| `spikealone` | A at 50 req/s, B idle | A's capacity in this mode |
| `contended` | A at 50 req/s and B at 2 req/s, same window | How much A's spike hurts B, and B's presence hurts A |

Every request has 128 random prompt tokens and generates exactly 128 tokens (EOS ignored). The table pools every request across runs. The figure shows the median of the runs with min–max whiskers, so run-to-run spread stays visible.

Notes:

- On exit (including Ctrl-C and failures) the script stops the containers, quits the MPS daemon, resets the compute mode, destroys the MIG instances and disables MIG.
- MPS mode switches the GPU to exclusive-process compute mode, so an instance that fails to attach to MPS fails loudly instead of silently time-slicing.
- If a run stops early, analyse what it collected: `python3 scripts/v2/analyze_v2.py results/v2/<run-id>`.
- GPU counters come from DCGM (SM and DRAM activity) when `dcgmi` is installed, otherwise from `nvidia-smi`, which reports little under MIG.
- GPU power is logged in every mode with `nvidia-smi` (`power.draw`). It is board power for the whole GPU, so under MIG it covers both slices. Energy per output token divides a window's energy by all tokens generated in it.

## Results (v2, 24 Oct 2026)

- **Hardware and software:** NVIDIA A100 80GB PCIe on Azure (`Standard_NC24ads_A100_v4`), driver 580.178.04, CUDA 13.0, vLLM 0.29.0 (server image and client), `Qwen/Qwen2.5-7B-Instruct`, 128 tokens in and out.
- **Runs:** 5 per mode, 30 s windows, 0 failed requests. MIG mode needed a VM reboot on Azure, so it ran as a second invocation on the same VM with the same settings and was merged into this run (its `env.txt` is in `mig/env_mig_run.txt`).
- **Full table:** [`results/v2/20261024-190042/summary.md`](results/v2/20261024-190042/summary.md). Raw per-request JSON (about 150 MB) is not committed.
- v1 ran on an H100 PCIe 80 GB, so v1 and v2 numbers come from different GPUs.

![v2 results](results/v2/20261024-190042/v2_summary.png)

| Metric | Time-slicing | MPS | MIG 3g.40gb |
| --- | --- | --- | --- |
| Quiet tenant B, p99 TTFT, A idle / A spiking (ms) | 50.8 / 137.8 | 50.6 / 216.3 | 105.2 / 100.0 |
| Quiet tenant B, median TPOT, A idle / A spiking (ms) | 10.8 / 28.6 | 11.0 / 41.5 | 20.5 / 21.0 |
| Noisy tenant A, throughput, alone / next to B (req/s) | 37.6 / 29.7 | 37.5 / 33.6 | 23.1 / 22.3 |
| Energy per output token, A spiking and B steady (J) | 0.082 | 0.074 | 0.104 |

## v1: the original single run

The first version is kept in `scripts/` (`./scripts/run_experiment.sh`) and explained step by step in the [Experiment Guide](mig-experiment-guide.md). It ran each phase once, compared time-slicing with MIG only, and reported percentiles over the quiet tenant's whole run rather than the spike window. v2 fixes those limits.

## Project structure

```
MIG_vs_Time-Slicing/
├── README.md
├── mig-experiment-guide.md      # Step-by-step guide to v1
├── scripts/
│   ├── v2/
│   │   ├── run_v2.sh            # v2: time-slicing vs MPS vs MIG, repeated runs
│   │   ├── config.sh            # v2 settings
│   │   ├── power_logger.sh      # v2 GPU power logger
│   │   └── analyze_v2.py        # v2 summary table and figure
│   ├── run_experiment.sh        # v1 master script
│   ├── run_test_pattern.sh      # v1 noisy-neighbour pattern
│   ├── run_bench.sh             # v1 load generator
│   ├── gpu_util_logger.sh       # nvidia-smi logger (used by both)
│   └── compare_results.sh       # v1 results table
└── results/                     # Created at runtime
```

## License

See [LICENSE](LICENSE).
