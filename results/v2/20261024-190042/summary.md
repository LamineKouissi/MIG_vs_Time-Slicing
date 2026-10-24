# Benchmark v2 results (2026-10-24T19:00:48+00:00)

GPU: NVIDIA A100 80GB PCIe, 580.178.04, 81920 MiB | vLLM 0.29.0 | Qwen/Qwen2.5-7B-Instruct | 128/128 tokens in/out | 5 runs per case

Latencies pool every request across runs; throughput is the median of the runs (min-max range in summary.csv).

| Mode | Case | Runs | p50 TTFT (ms) | p99 TTFT (ms) | p50 TPOT (ms) | p99 TPOT (ms) | TPOT std (ms) | p99 ITL (ms) | Throughput (req/s) | Failed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Time-slicing | Quiet tenant B, neighbour idle | 5 | 38.2 | 50.8 | 10.8 | 11.3 | 0.18 | 17.2 | 1.91 | 0 |
| Time-slicing | Quiet tenant B, neighbour spiking | 5 | 99.2 | 137.8 | 28.6 | 30.4 | 2.19 | 54.3 | 1.79 | 0 |
| Time-slicing | Noisy tenant A, spiking alone | 5 | 3133.0 | 7850.0 | 50.8 | 52.0 | 7.35 | 99.9 | 37.63 | 0 |
| Time-slicing | Noisy tenant A, spiking next to B | 5 | 13209.7 | 18831.4 | 60.4 | 88.8 | 21.07 | 264.5 | 29.71 | 0 |
| MPS | Quiet tenant B, neighbour idle | 5 | 38.3 | 50.6 | 11.0 | 11.5 | 0.17 | 17.2 | 1.91 | 0 |
| MPS | Quiet tenant B, neighbour spiking | 5 | 127.8 | 216.3 | 41.5 | 46.2 | 4.37 | 81.1 | 1.72 | 0 |
| MPS | Noisy tenant A, spiking alone | 5 | 3173.7 | 7839.1 | 51.1 | 52.0 | 7.35 | 98.2 | 37.49 | 0 |
| MPS | Noisy tenant A, spiking next to B | 5 | 5814.3 | 12887.6 | 59.0 | 62.5 | 9.89 | 119.4 | 33.64 | 0 |
| MIG 3g.40gb | Quiet tenant B, neighbour idle | 5 | 70.6 | 105.2 | 20.5 | 21.6 | 0.47 | 35.4 | 1.84 | 0 |
| MIG 3g.40gb | Quiet tenant B, neighbour spiking | 5 | 73.9 | 100.0 | 21.0 | 22.1 | 0.50 | 38.6 | 1.84 | 0 |
| MIG 3g.40gb | Noisy tenant A, spiking alone | 5 | 12605.7 | 30657.5 | 82.5 | 85.6 | 9.34 | 290.5 | 23.06 | 0 |
| MIG 3g.40gb | Noisy tenant A, spiking next to B | 5 | 14740.7 | 33313.9 | 84.3 | 93.8 | 10.79 | 317.9 | 22.30 | 0 |

## GPU power and energy

Whole-GPU board power from `nvidia-smi` (`power.draw`), sampled every 0.5 s over each measured window. Median of the runs (min-max range in energy.csv).

| Mode | Window | Runs | Mean power (W) | Energy (kJ) | Output tokens | Energy per output token (J) |
| --- | --- | --- | --- | --- | --- | --- |
| Time-slicing | B alone, A idle | 5 | 226 | 9.8 | 7680 | 1.276 |
| Time-slicing | A spiking alone | 5 | 246 | 13.1 | 192000 | 0.068 |
| Time-slicing | A spiking, B steady | 5 | 257 | 16.3 | 199680 | 0.082 |
| MPS | B alone, A idle | 5 | 232 | 10.0 | 7680 | 1.299 |
| MPS | A spiking alone | 5 | 248 | 13.2 | 192000 | 0.069 |
| MPS | A spiking, B steady | 5 | 254 | 14.7 | 199680 | 0.074 |
| MIG 3g.40gb | B alone, A idle | 5 | 170 | 7.5 | 7680 | 0.979 |
| MIG 3g.40gb | A spiking alone | 5 | 246 | 19.1 | 192000 | 0.100 |
| MIG 3g.40gb | A spiking, B steady | 5 | 259 | 20.9 | 199680 | 0.104 |

## Headline numbers

- Time-slicing: when A spikes, B's p99 TTFT goes 50.8 -> 137.8 ms (2.71x) and its median TPOT 10.8 -> 28.6 ms.
- Time-slicing: A's throughput is 37.6 req/s alone and 29.7 req/s next to B (-21%).
- MPS: when A spikes, B's p99 TTFT goes 50.6 -> 216.3 ms (4.28x) and its median TPOT 11.0 -> 41.5 ms.
- MPS: A's throughput is 37.5 req/s alone and 33.6 req/s next to B (-10%).
- MIG 3g.40gb: when A spikes, B's p99 TTFT goes 105.2 -> 100.0 ms (0.95x) and its median TPOT 20.5 -> 21.0 ms.
- MIG 3g.40gb: A's throughput is 23.1 req/s alone and 22.3 req/s next to B (-3%).
- Cost of isolation with the neighbour idle: B's median TPOT is 10.8 ms on the shared GPU and 20.5 ms on its MIG slice (1.89x).
- A's capacity alone: 37.6 req/s on the whole GPU vs 23.1 req/s on its MIG slice.
- Energy per output token with A spiking and B steady (whole-GPU board power): Time-slicing 0.082 J, MPS 0.074 J, MIG 3g.40gb 0.104 J.
