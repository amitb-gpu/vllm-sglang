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

## How to explain it

> I served the same model on one RTX 5090 using vLLM and SGLang. Both expose the same API, so I kept the application and workload fixed. I compared first-token latency, full response latency, throughput, and routing accuracy as I increased the number of concurrent clients.

## References

- [WSL agent bridge](https://github.com/amitb-gpu/wsl-agent-bridge)
- [vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- [vLLM structured outputs](https://docs.vllm.ai/en/latest/features/structured_outputs/)
- [SGLang installation](https://docs.sglang.io/docs/get-started/install)
