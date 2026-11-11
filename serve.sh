#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

engine=${1:-}
case "$engine" in
    vllm|sglang) ;;
    *) echo "Usage: bash serve.sh {vllm|sglang}" >&2; exit 2 ;;
esac

venv="$runtime/$engine"
if [[ ! -x "$venv/bin/python" ]]; then
    echo "Run bash setup.sh $engine first." >&2
    exit 1
fi

port=${PORT:-8000}
memory_fraction=${GPU_MEMORY_FRACTION:-0.60}
export XDG_CACHE_HOME="$runtime/$engine-cache"
export CUDA_VISIBLE_DEVICES=0
export PATH="$venv/bin:$PATH"

common=(--host 127.0.0.1 --port "$port" --revision "$revision" --dtype bfloat16
        --served-model-name support-triage)
# Use backends that work with the packaged CUDA stack on the WSL host.
if [[ "$engine" == vllm ]]; then
    export VLLM_USE_FLASHINFER_SAMPLER=0
    exec vllm serve "$model" "${common[@]}" --max-model-len 4096 --max-num-seqs 16 \
        --gpu-memory-utilization "$memory_fraction" --disable-prefix-caching \
        --attention-backend FLASH_ATTN --generation-config vllm
else
    export SGLANG_CACHE_DIR="$XDG_CACHE_HOME"
    export SGLANG_JIT_CACHE_DIR="$SGLANG_CACHE_DIR/jit"
    if [[ -z "${CUDA_HOME:-}" ]]; then
        export CUDA_HOME="$venv/lib/python3.12/site-packages/nvidia/cu13"
        # The CUDA wheel provides the versioned runtime. JIT linking uses -lcudart.
        linker_dir="$SGLANG_CACHE_DIR/linker"
        mkdir -p "$linker_dir"
        ln -sfn "$CUDA_HOME/lib/libcudart.so.13" "$linker_dir/libcudart.so"
        export LIBRARY_PATH="$linker_dir${LIBRARY_PATH:+:$LIBRARY_PATH}"
    fi
    exec python -m sglang.launch_server --model-path "$model" "${common[@]}" \
        --context-length 4096 --max-running-requests 16 --cuda-graph-max-bs-decode 16 \
        --mem-fraction-static "$memory_fraction" --disable-radix-cache --enable-metrics \
        --attention-backend triton --sampling-backend pytorch --sampling-defaults openai
fi
