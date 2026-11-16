#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

engine=${1:-}
prefix=${PREFIX_CACHE:-off}
offload=none
case "$engine:$prefix:$offload" in
    vllm:on:none|vllm:off:none|sglang:on:none|sglang:off:none) ;;
    *) echo "Use {vllm|sglang}, PREFIX_CACHE={off|on},." >&2; exit 2 ;;
esac

venv="$runtime/$engine"
export PATH="$venv/bin:$PATH"
export XDG_CACHE_HOME="$runtime/$engine-cache"
export CUDA_VISIBLE_DEVICES=0
export HF_HUB_OFFLINE=1
port=${PORT:-8000}
memory_fraction=${GPU_MEMORY_FRACTION:-0.60}
context=${MAX_MODEL_LEN:-4096}
common=(--host 127.0.0.1 --port "$port" --revision "$revision" --dtype bfloat16
        --served-model-name support-triage)
cache_args=()
if [[ "$engine" == vllm ]]; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
    if [[ "$prefix" == on ]]; then
        cache_args+=(--enable-prefix-caching)
    else
        cache_args+=(--disable-prefix-caching)
    fi
    command=(vllm serve "$model" "${common[@]}" --max-model-len "$context" --max-num-seqs 16
             --gpu-memory-utilization "$memory_fraction" "${cache_args[@]}"
             --attention-backend FLASH_ATTN --generation-config vllm)
else
    export SGLANG_CACHE_DIR="$XDG_CACHE_HOME"
    export SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit"
    if [[ -z "${CUDA_HOME:-}" ]]; then
        export CUDA_HOME="$venv/lib/python3.12/site-packages/nvidia/cu13"
        linker_dir="$SGLANG_CACHE_DIR/linker"
        mkdir -p "$linker_dir"
        ln -sfn "$CUDA_HOME/lib/libcudart.so.13" "$linker_dir/libcudart.so"
        export LIBRARY_PATH="$linker_dir${LIBRARY_PATH:+:$LIBRARY_PATH}"
    fi
    if [[ "$prefix" == off ]]; then cache_args+=(--disable-radix-cache); fi
    command=(python -m sglang.launch_server --model-path "$model" "${common[@]}"
             --context-length "$context" --max-running-requests 16 --cuda-graph-max-bs-decode 16
             --mem-fraction-static "$memory_fraction" "${cache_args[@]}" --enable-metrics
             --attention-backend triton --sampling-backend pytorch --sampling-defaults openai)
fi

if [[ -n "${PHASE2_CONFIG_OUT:-}" ]]; then
    "$venv/bin/python" - "$PHASE2_CONFIG_OUT" "$engine" "$prefix" "$offload" "$context" "$memory_fraction" "$runtime" "${command[@]}" <<'PY'
import importlib.metadata as metadata
import json
import os
import platform
import sys
from pathlib import Path

out, engine, prefix, offload, context, fraction, runtime, *command = sys.argv[1:]
env_keys = ("CUDA_VISIBLE_DEVICES", "VLLM_USE_FLASHINFER_SAMPLER", "VLLM_USE_V2_MODEL_RUNNER",
            "HF_HUB_OFFLINE", "LMCACHE_USE_EXPERIMENTAL", "LMCACHE_TRACK_USAGE", "CUDA_HOME")
versions = {p: metadata.version(p) for p in (engine, "torch", "transformers", "triton")}
Path(out).write_text(json.dumps({
    "engine": engine, "prefix_cache": prefix, "kv_offload": offload,
    "max_model_len": int(context), "gpu_memory_fraction": float(fraction),
    "flags": command, "versions": versions, "python": platform.python_version(),
    "environment": {k: os.environ[k].replace(runtime, "$TICKET_INFERENCE_HOME")
                    for k in env_keys if k in os.environ},
    "lmcache_config": None,
}, indent=2), encoding="utf-8")
PY
fi
exec "${command[@]}"
