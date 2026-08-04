# MIG vs Time-Slice Experiment

This repository automates the benchmark experiment comparing NVIDIA Multi-Instance GPU (MIG) hardware isolation against standard time-slicing using vLLM.

## Project Structure

- `mig-experiment-guide.md`: A detailed, step-by-step educational guide. If you want to understand exactly what each command and script does under the hood rather than just running the automated script, read this file!
- `scripts/`: Contains all the bash scripts needed to run the benchmark.
  - `run_experiment.sh`: The master script that automates the entire experiment from setup to teardown.
  - `run_bench.sh`: Wrapper for `vllm bench serve` load generator.
  - `run_test_pattern.sh`: Orchestrates steady load and spikes across instances.
  - `gpu_util_logger.sh`: Logs GPU utilization metrics periodically.
  - `compare_results.sh`: Analyzes and formats the JSON outputs from vLLM.
- `results/`: Directory where all generated logs, CSVs, and JSON test results will be stored.

## How to Run

You can run the entire experiment automatically with a single command. The script will handle dependencies, Docker containers, MIG configurations, benchmarking, and cleanup.

```bash
chmod +x scripts/*.sh
./scripts/run_experiment.sh
```

**Note**: You may be prompted for your `sudo` password as it requires administrative privileges to install prerequisites, run Docker containers, and configure NVIDIA MIG settings.

## Results

After the run is complete, `compare_results.sh` will output a comparison table of Baseline (Time-Sliced) vs MIG performance, specifically looking at P99 Time to First Token (TTFT) and Time Per Output Token (TPOT) stability.
