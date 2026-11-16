# Support Ticket Routing with vLLM and SGLang

A single-GPU inference project that turns a support ticket into a queue assignment and a short explanation. vLLM and SGLang serve the same Qwen3-4B model behind the same OpenAI-compatible API, so the client can switch engines without changing the request.

[Run the demo](#run-the-demo) · [How it works](#how-it-works) · [Benchmark methodology](#benchmark-methodology) · [RTX 5090 results](#rtx-5090-results) · [Troubleshooting](#troubleshooting)

## Run the demo

### 1. Set up the environments

The tested configuration is an NVIDIA RTX 5090 with Ubuntu 26.04 on WSL2. Run the commands from an Ubuntu shell with a working `nvidia-smi`, Git, `curl`, Python 3, and the `gcc` and `g++` toolchain. From an existing checkout in PowerShell, `wsl.exe -d Ubuntu-26.04` opens the corresponding project directory in WSL.

```bash
nvidia-smi
git clone https://github.com/amitb-gpu/vllm-sglang.git
cd vllm-sglang
bash setup.sh vllm
bash setup.sh sglang
```

Skip the clone and directory change when you already have the checkout open. Each setup command installs a pinned engine version in its own uv-managed Python 3.12 environment. vLLM and SGLang can then use their respective PyTorch and CUDA packages while sharing one pinned model snapshot. The full installation used about 24 GiB on the WSL filesystem, including roughly 7.5 GiB of model weights. [Qwen3-4B-Instruct-2507 model card](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)

### 2. Start vLLM and route a ticket

In the first terminal, start vLLM. The launcher selects the environment and keeps the server in the foreground.

```bash
bash serve.sh vllm
```

In a second Ubuntu terminal in the same project directory, check that the server is ready and send a ticket. The Python client uses only the standard library.

```bash
curl -fsS http://127.0.0.1:8000/health
curl -fsS http://127.0.0.1:8000/v1/models
python3 triage.py "Our invoice shows the same charge twice."
```

The health check returns HTTP 200 when ready and can have an empty body. `/v1/models` lists the shared model alias, `support-triage`. A ticket response looks like this.

```json
{
  "queue": "billing",
  "reason": "The ticket involves a duplicate charge on an invoice, which is a billing issue."
}
```

### 3. Switch to SGLang

Stop vLLM with Ctrl+C, then start SGLang in the first terminal. Run one engine at a time on the GPU.

```bash
bash serve.sh sglang
```

Repeat the health check and the same ticket command in the second terminal.

```bash
curl -fsS http://127.0.0.1:8000/health
python3 triage.py "Our invoice shows the same charge twice."
```

## How it works

```text
Ticket text → triage.py → POST /v1/chat/completions → vLLM or SGLang → Qwen3-4B
                 ↑              streamed response with usage              │
                 └───────────────────────────────────────────────────────────┘
                 ↓
          Validated queue + reason
```

1. The client sends a routing policy, the ticket text, and a JSON schema. The policy gives security and privacy cases priority and asks the model to route the root issue.
2. The engine applies the model's chat template and processes the input during **prefill**. During **decode**, it generates the answer incrementally and constrains the output to the schema.
3. The client reads the server-sent events, measures when the first text arrives, and collects completion token usage. It checks the end of the stream and validates the assembled JSON before displaying the answer.

| Queue | Example ticket |
| --- | --- |
| `billing` | A duplicate subscription charge |
| `account_access` | A lost MFA device |
| `technical` | A CSV export that fails |
| `human_review` | Suspected account takeover, a privacy request, or an unclear report |

The schema constrains the queue to one of these four values. The labeled dataset checks whether the chosen value fits the ticket. It includes overlapping issues, a privacy escalation, and a ticket that tries to override the routing instructions.

### Serving configuration

Both launchers bind to loopback, serve BF16 weights, cap the context at 4,096 tokens, and allow 16 active sequences. vLLM uses FlashAttention 2 and SGLang uses Triton attention. Those backends worked around a FlashInfer CUDA compiler/header mismatch on the test host.

| Setting | Default | How to use it |
| --- | --- | --- |
| `PORT` | `8000` | Set it when starting the server and pass the matching `--url http://127.0.0.1:PORT/v1` to the client and benchmark. |
| `TICKET_INFERENCE_HOME` | `~/.cache/ticket-inference` | Use the same absolute Linux path for setup and serving. It holds the environments, caches, and weights. |
| `GPU_MEMORY_FRACTION` | `0.60` | Sets vLLM's GPU memory target or SGLang's static memory fraction, according to the active engine. |

Both engines expose Prometheus metrics. During a run, inspect `/metrics` for engine-level request and scheduling data, or use the benchmark report for client-observed timings.

```bash
curl -fsS http://127.0.0.1:8000/metrics
```

## Benchmark methodology

Run each command against the matching active server, switching engines between commands. The benchmark prints one summary per concurrency level and saves per-ticket predictions, timings, and errors to JSON.

```bash
python3 benchmark.py vllm --output /tmp/vllm-tickets.json
python3 benchmark.py sglang --output /tmp/sglang-tickets.json
```

The load generator keeps up to 1, 4, or 8 requests in flight. Each worker submits its next ticket as soon as the previous request finishes. Before each measured level, the benchmark sends a warm-up request per worker. The timed workload then covers 24 synthetic tickets twice in a fixed shuffled order, for 48 requests per level and 144 per engine. The same schema, greedy decoding, and 96-token output cap apply to both engines. Both compute prompt KV entries afresh for each request.

| Metric | Definition |
| --- | --- |
| p95 time to first token | Client time from sending the request to receiving the first streamed chunk containing text |
| p95 end-to-end latency | Client time from sending the request to completion of the stream |
| Requests and output tokens per second | Successful requests and engine-reported completion tokens divided by wall time |
| Routing accuracy | Correct queues divided by all attempted tickets |
| Errors | HTTP, streaming, or response validation failures |

Latency percentiles use the nearest rank among successful requests. A valid response with the wrong queue contributes to serving throughput and counts as incorrect for routing accuracy. The report records the dataset's SHA-256 hash and the shuffle seed so two runs can be matched to the same inputs.

## RTX 5090 results

Measured November 10, 2026 on one GeForce RTX 5090 under Ubuntu 26.04.1 on WSL2. vLLM 0.33.0 used FlashAttention 2 and PyTorch 2.14.0+cu132. SGLang 0.5.23 used Triton attention and PyTorch 2.14.1. Each engine ran once through the full sweep on the shared workstation. [vLLM raw results](results/vllm.json) · [SGLang raw results](results/sglang.json)

| Engine | Concurrent clients | Requests/s | Output tokens/s | p95 first token | p95 end to end |
| --- | ---: | ---: | ---: | ---: | ---: |
| vLLM | 1 | 4.31 | 141.80 | 43.44 ms | 272.38 ms |
| vLLM | 4 | 14.09 | 464.87 | 53.51 ms | 323.23 ms |
| vLLM | 8 | 22.37 | 736.46 | 74.99 ms | 414.35 ms |
| SGLang | 1 | 4.94 | 164.47 | 19.71 ms | 234.00 ms |
| SGLang | 4 | 14.62 | 477.39 | 38.63 ms | 330.59 ms |
| SGLang | 8 | 24.26 | 793.05 | 59.01 ms | 391.36 ms |

All 288 measured requests returned valid responses. Each engine routed 22 of the 24 distinct tickets correctly at every concurrency level. Both sent a broken export mentioned in a refund review to billing and inferred a login issue from a vague account complaint that called for human review.

Going from one to eight clients increased request throughput by roughly five times for both engines while raising p95 full-response latency. Continuous batching lets the engine combine work from multiple requests on the GPU. The numbers describe one sweep of short prompts on this host, with each engine's selected attention backend.

## Troubleshooting

| Symptom | What to check |
| --- | --- |
| `nvidia-smi` fails in Ubuntu | Check GPU visibility in the named WSL distro before installing the engines. |
| Connection refused during startup | Keep the first terminal open and wait for model loading and CUDA graph warm-up to finish. The first start can take several minutes. |
| Port 8000 is occupied | Start the server with `PORT=8001 bash serve.sh vllm` and pass `--url http://127.0.0.1:8001/v1` to the client. |
| Server reports a GPU memory error | Check available GPU memory with `nvidia-smi` and confirm the other inference engine has stopped. Adjust `GPU_MEMORY_FRACTION` if the GPU is shared. |
| Smoke test works but the benchmark fails | Inspect the benchmark's reported error or the server log in the first terminal. The report records failures that occur after warm-up. |

## Repository map

| Path | Role |
| --- | --- |
| [`setup.sh`](setup.sh) | Creates each isolated engine environment and caches the shared model. |
| [`settings.sh`](settings.sh) | Pins engine versions and the model revision. |
| [`serve.sh`](serve.sh) | Starts the selected engine with the local serving configuration. |
| [`triage.py`](triage.py) | Defines the routing policy and schema, reads the streaming API, and validates responses. |
| [`benchmark.py`](benchmark.py) | Runs the concurrency sweep and emits request-level results and aggregate metrics. |
| [`tickets.jsonl`](tickets.jsonl) | Contains the 24 synthetic tickets and their expected queues. |
| [`results/`](results/) | Contains the measured RTX 5090 runs for both engines. |

## Phase 2: KV cache reuse and offload

When a model reads a request, it saves notes about the words it has processed. Those notes are its KV cache. We timed how soon an answer started when the model could reuse those notes. Both engines began answering much sooner when the notes for a shared policy were already on the graphics card. Reloading notes from system memory or a local drive also paid off for longer passages.

### Methodology

The [committed plan](results/phase2/PLAN.md) defines the workloads, warm-ups, trial counts, and acceptance rules. All timing uses the client clock and the metric definitions above. The Phase 2 launcher defaults to cache off and binds to loopback. Model, engine versions, command-line flags, memory settings, and machine details are in the JSON accompanying each run.

```bash
PREFIX_CACHE=on bash serve-phase2.sh vllm
# Stop vLLM before switching engines.
PREFIX_CACHE=on bash serve-phase2.sh sglang
```

The short workload uses the baseline tickets unchanged. The policy workload places the same synthetic support document before every ticket. The pinned tokenizer counted **2,000 document tokens**, before the chat template and ticket text. Each concurrency level warms the server before measurement. For cache-on policy runs, the warm-up also seeds the shared prefix. The two policy conditions use identical prompts. The short cache-off rows come from the earlier published baseline, so their differences can include changes in machine conditions between sessions.

### Native prefix cache

Short tickets, published baseline versus cache on. [vLLM raw run](results/phase2/prefix-vllm-on-tickets.json) · [SGLang raw run](results/phase2/prefix-sglang-on-tickets.json)

| Engine | Clients | Cache | p95 first text (ms) | p95 end to end (ms) | Requests/s | Output tokens/s | Accuracy | Errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM | 1 | Off | 43.44 | 272.38 | 4.31 | 141.80 | 91.67% | 0 |
| vLLM | 1 | On | 27.77 | 254.95 | 4.63 | 153.55 | 91.67% | 0 |
| vLLM | 4 | Off | 53.51 | 323.23 | 14.09 | 464.87 | 91.67% | 0 |
| vLLM | 4 | On | 23.99 | 276.29 | 16.04 | 534.72 | 91.67% | 0 |
| vLLM | 8 | Off | 74.99 | 414.35 | 22.37 | 736.46 | 91.67% | 0 |
| vLLM | 8 | On | 49.59 | 314.85 | 27.95 | 925.26 | 91.67% | 0 |
| SGLang | 1 | Off | 19.71 | 234.00 | 4.94 | 164.47 | 91.67% | 0 |
| SGLang | 1 | On | 28.84 | 248.25 | 4.80 | 159.51 | 91.67% | 0 |
| SGLang | 4 | Off | 38.63 | 330.59 | 14.62 | 477.39 | 91.67% | 0 |
| SGLang | 4 | On | 43.90 | 332.43 | 14.39 | 477.71 | 91.67% | 0 |
| SGLang | 8 | Off | 59.01 | 391.36 | 24.26 | 793.05 | 91.67% | 0 |
| SGLang | 8 | On | 45.37 | 448.50 | 23.06 | 761.44 | 91.67% | 0 |

Shared policy, fresh cache-off and cache-on runs. [vLLM off](results/phase2/prefix-vllm-off-policy.json) · [vLLM on](results/phase2/prefix-vllm-on-policy.json) · [SGLang off](results/phase2/prefix-sglang-off-policy.json) · [SGLang on](results/phase2/prefix-sglang-on-policy.json)

| Engine | Clients | Cache | p95 first text (ms) | p95 end to end (ms) | Requests/s | Output tokens/s | Accuracy | Errors |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vLLM | 1 | Off | 106.35 | 354.17 | 3.12 | 110.95 | 100% | 0 |
| vLLM | 1 | On | 28.50 | 284.35 | 4.05 | 145.09 | 100% | 0 |
| vLLM | 4 | Off | 269.60 | 818.64 | 6.23 | 221.44 | 100% | 0 |
| vLLM | 4 | On | 38.19 | 342.95 | 13.52 | 482.29 | 100% | 0 |
| vLLM | 8 | Off | 623.97 | 1,309.57 | 7.79 | 276.74 | 100% | 0 |
| vLLM | 8 | On | 44.68 | 397.97 | 22.39 | 804.18 | 100% | 0 |
| SGLang | 1 | Off | 110.42 | 370.87 | 3.02 | 108.15 | 100% | 0 |
| SGLang | 1 | On | 30.00 | 275.66 | 4.12 | 145.66 | 100% | 0 |
| SGLang | 4 | Off | 293.46 | 866.26 | 6.13 | 217.44 | 100% | 0 |
| SGLang | 4 | On | 46.21 | 353.05 | 12.99 | 460.70 | 100% | 0 |
| SGLang | 8 | Off | 616.42 | 1,437.45 | 7.25 | 259.33 | 100% | 0 |
| SGLang | 8 | On | 54.53 | 451.52 | 20.80 | 739.22 | 100% | 0 |

The long policy reduced p95 time to first text at every tested concurrency when the cache was enabled. Short-ticket results were mixed, especially for SGLang. The long policy also clarified the routing rules for cases missed by the short prompt. Accuracy was identical between cache settings within the policy workload.

The following cache-on counters are deltas across the timed cell. vLLM exposes queried and matched tokens through `vllm:prefix_cache_queries_total` and `vllm:prefix_cache_hits_total`. SGLang exposes device-cached prompt tokens through `sglang:cached_tokens_total`. Its `sglang:cache_hit_rate` is a gauge, so the raw scrapes preserve its readings without treating them as a whole-cell rate.

| Workload | Clients | vLLM hit tokens | vLLM queried tokens | SGLang device-cached tokens |
| --- | ---: | ---: | ---: | ---: |
| Short | 1 | 5,936 | 6,938 | 6,243 |
| Short | 4 | 6,496 | 6,938 | 6,890 |
| Short | 8 | 6,496 | 6,938 | 6,890 |
| Policy | 1 | 102,368 | 103,466 | 102,771 |
| Policy | 4 | 102,944 | 103,466 | 103,418 |
| Policy | 8 | 102,944 | 103,466 | 103,418 |

With caching off, vLLM reported zero prefix hits and queries. SGLang omitted the device-cached token series. SGLang creates the series on its first hit, so a new counter starts at zero for the delta. The results preserve both scrapes, including any unavailable series.

To reproduce the native sweeps, run the commands below one at a time with the manual server stopped. The runner uses its own loopback port and exits after the sweep. Move existing Phase 2 result files to another location before writing new runs with the same names.

```bash
python="${TICKET_INFERENCE_HOME:-$HOME/.cache/ticket-inference}/vllm/bin/python"
"$python" bench_phase2.py vllm on
"$python" bench_phase2.py vllm off
"$python" bench_phase2.py sglang on
"$python" bench_phase2.py sglang off
```

### LMCache reload versus recompute

LMCache `0.5.7.post1` ran in its own uv environment with vLLM `0.33.0` and the release's dynamic connector. The built-in vLLM connector points to an older module layout. The installation and import errors, the selected connector, and successful backend gates are recorded in the compatibility report. [Compatibility checks](results/phase2/lmcache-compatibility.json)

All modes used the same model, BF16, a context cap of **8,704 tokens**, and a GPU memory setting of **0.60**. Native GPU prefix reuse was disabled. The cold control ran in the same environment with no connector. Each warm trial primed a distinct prefix with one ticket and timed a request with a different suffix. The runner checked the store and retrieve logs against vLLM's external-hit counter. For disk, it also verified files on the WSL ext4 volume. LMCache used CPU memory as a staging buffer on the disk path, with its CPU cache tier disabled.

Each cell uses ten measured requests after two excluded warm-ups. All timed CPU and disk requests reloaded the full document length, and all timed requests returned valid results. One excluded disk warm-up sampled a partially written file before the reload, which still reported a full hit. The runner now waits for the full expected byte count. [Cold raw trials](results/phase2/offload-cold.json) · [CPU raw trials](results/phase2/offload-cpu.json) · [Disk raw trials](results/phase2/offload-disk.json)

| Shared document tokens | Cold median (ms) | Cold p95 (ms) | CPU median (ms) | CPU p95 (ms) | Disk median (ms) | Disk p95 (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 59.70 | 62.45 | 56.14 | 66.89 | 55.23 | 66.58 |
| 2,048 | 110.53 | 114.28 | 76.64 | 89.93 | 87.76 | 100.20 |
| 4,096 | 209.91 | 217.46 | 124.52 | 129.31 | 148.21 | 158.15 |
| 8,192 | 451.49 | 464.34 | 222.32 | 241.22 | 273.22 | 292.36 |

**Crossover.** Both reload paths first beat recompute on median TTFT at **1,024 tokens** (CPU by **3.56 ms**, disk by **4.47 ms**), with slower p95s at that length, and both first beat recompute on median and p95 at **2,048 tokens** (CPU median **33.89 ms** faster, disk median **22.77 ms** faster).

Warm disk requests immediately followed priming. Linux could serve reads from its page cache, so this is an application-level reload measurement with a local file backend. With ten observations per cell, nearest-rank p95 is the slowest measured request.

```bash
bash setup-lmcache.sh
python="${TICKET_INFERENCE_HOME:-$HOME/.cache/ticket-inference}/vllm/bin/python"
"$python" bench_offload.py cpu --gate
"$python" bench_offload.py disk --gate
"$python" bench_offload.py cold
"$python" bench_offload.py cpu
"$python" bench_offload.py disk
```

For a manual server, select `KV_OFFLOAD=cpu` or `KV_OFFLOAD=disk` with `bash serve-phase2.sh vllm`. Set `MAX_MODEL_LEN` to the desired context cap and use a dedicated Linux filesystem directory for `LMCACHE_DISK_DIR` when running disk trials.

### Go client parity

The standard-library-only Go client uses the same ticket file, routing schema, and streaming protocol. It reproduces the Python shuffle order from the same seed and writes the same JSON report shape. The parity run used a single cache-off vLLM server in Python, Go, Go, Python order. The table pools both passes for each client. [Comparison, environment, and links to the raw passes](results/phase2/go-parity.json)

| Clients | Python median first text (ms) | Go median first text (ms) | Python median end to end (ms) | Go median end to end (ms) |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 23.977 | 24.451 | 220.143 | 222.494 |
| 4 | 27.935 | 27.790 | 253.963 | 256.593 |
| 8 | 33.348 | 34.316 | 297.069 | 299.549 |

All six timing comparisons passed the tolerance of the greater of 10 ms or 10% of the Python median. The clients matched the input hash, ticket order, report schema, and routing counts, with zero response errors. Go opened a fresh connection per request to match Python's HTTP behavior. The small timing differences reflect the combined effects of client parsing, scheduling, and server variation.

Build with Go `1.26.11` from the repository root. With either Phase 2 server already running, pass its engine label to the client.

```bash
(cd go && GOTOOLCHAIN=go1.26.11 go test ./... && GOTOOLCHAIN=go1.26.11 go build -o /tmp/ticket-bench .)
/tmp/ticket-bench vllm --output /tmp/go-tickets.json
```

To reproduce the same-server comparison, stop the manual server first, move the existing parity JSON files out of the results directory, and run `"$python" bench_go_parity.py /tmp/ticket-bench` using the Python path shown above.

### What this does not show

- Multi-GPU scaling or behavior on native Linux. The runs used one RTX 5090 under WSL2.
- Generalization to other models or production traffic. The workloads used one model and synthetic tickets and policies.
- Networked-storage latency or uncached physical disk latency. The disk backend used local ext4 with the OS page cache available.

## How to explain it

> I served the same model on one RTX 5090 using vLLM and SGLang. Both expose the same API, so I kept the application and workload fixed. I compared first-token latency, full response latency, throughput, and routing accuracy as I increased the number of concurrent clients.

> I then measured where the model's saved prompt state should live. I compared native GPU reuse on both engines, then disabled it in vLLM to isolate LMCache reloads from system memory and local storage. I verified hits with counters and connector logs before comparing latency against recompute. A Go client reproduced the Python workload and median timings within the declared tolerance.

## References

- [WSL agent bridge](https://github.com/amitb-gpu/wsl-agent-bridge)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [SGLang installation](https://docs.sglang.io/docs/get-started/install)
