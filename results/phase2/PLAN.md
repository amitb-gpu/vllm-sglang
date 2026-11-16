# Phase 2 measurement plan

## Controls and metrics

Use one RTX 5090, Ubuntu 26.04 on WSL2, and the Qwen3-4B-Instruct-2507 snapshot pinned in `settings.sh`. Bind to loopback and run one engine at a time. Keep vLLM 0.33.0 with FlashAttention 2 and SGLang 0.5.23 with Triton attention, including the FlashInfer workaround. Leave the published short-ticket runs and their source files in place.

Measure time to first token from request dispatch to the first streamed chunk containing text. Measure end-to-end latency from dispatch to completion of the stream. Compute p95 by nearest rank among successful requests. Throughput is successful requests or engine-reported output tokens from successful requests divided by total wall time. Accuracy is correct queues divided by all attempts. Errors include HTTP, incomplete streaming, and response validation failures. Keep request-level observations, server flags, model revision, software versions, GPU memory settings, context limits, and any LMCache config in committed JSON. Record failed attempts with diagnostic evidence. Exclude startup, setup, and warm-up requests from timed cells.

## GPU prefix reuse

- Run the existing 24 labeled tickets twice in the same Python seed-5090 order at each of 1, 4, and 8 in-flight requests, with matching routing policy, schema, greedy decoding, and 96-token cap. Use one warm-up request per worker per level. Compare cache-on with the committed cache-off baseline and call out that it was measured in a separate session.
- Repeat with a deterministic synthetic support policy prepended before the variable ticket, targeting 2,000 tokens by the pinned tokenizer. Record actual token counts and document hash. For each engine run one cache-off and one cache-on sweep using identical inputs and settings apart from the cache flag. Warm up each cell with a separate ticket that shares the policy. Record /metrics counter samples immediately before and after each measured cell, including query and hit denominators when exposed. Preserve the metric names and units. Report absent counters as unavailable.
- Use BF16, context length 4,096, GPU memory setting 0.60, and 16 active sequences. Record routing accuracy, errors, p95 first-token and end-to-end latency, requests/s and output tokens/s for every cell. Report paired differences within each engine and the observed cache hits.

## vLLM LMCache offload

- Screen released LMCache versions against the exact vLLM/runtime ABI and connector, then validate an isolated uv environment. Gate measurements on connector startup and a proven store/retrieve cycle. If no released build works, stop the offload experiment at the failed gate and record the error and release evidence. Keep vLLM pinned.
- Use one client and identical prompts across cold recompute, CPU reload, and local disk reload. Use approximate shared-prefix targets of 1,024, 2,048, 4,096, and 8,192 tokenizer tokens. Allow room for the ticket suffix and output, choose the smallest common context cap covering the longest tested prompt, and record the cap and actual token counts. Document any unsupported length or backend.
- Disable native GPU prefix reuse. Use no connector for the cold recompute control. For each warm trial, prime a previously unused prefix, confirm the store completed, then time a second request with a different ticket suffix. Confirm reloads using connector hit tokens and storage backend telemetry. Enable only CPU storage for CPU runs and disable the LMCache CPU tier for disk runs. Document the exact storage path type and that the OS page cache can serve local disk reads.
- Collect two excluded warm-ups and ten timed observations per length and mode. Keep misses and failed attempts visible. Compute median and nearest-rank p95 TTFT, and report the first tested length at which the warm median is lower than cold, with absolute and relative differences. If p95 disagrees or a cell lacks verified hits, qualify the result.

## Go client parity

- Pin the Go compiler. Use one cache-off vLLM server for both clients. Run two full passes per client, with Python/Go followed by Go/Python. Use the same dataset bytes, SHA-256, seed, shuffled ticket IDs, schema, concurrency, and per-level warm-up plus 48 measured requests. Keep the original benchmark JSON shape and write run metadata separately.
- Pool successful outcomes by client and concurrency to compare median TTFT and end-to-end latency. Require the exact same workload order, compatible outcome schema, zero errors, matching routing counts, and a median difference of at most the greater of 10% of the Python median or 10 ms for each timing metric. Report all medians, observed differences, and any cells that miss the tolerance.
