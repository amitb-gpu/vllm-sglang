#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"
export UV_CACHE_DIR="$runtime/uv-cache" UV_PYTHON_INSTALL_DIR="$runtime/python"
uv="$runtime/bin/uv"
venv="$runtime/vllm-lmcache"
if [[ ! -x "$uv" ]]; then
    echo "Run bash setup.sh vllm to install the project toolchain and model." >&2
    exit 1
fi
if [[ ! -x "$venv/bin/python" ]]; then
    "$uv" venv --managed-python --python 3.12 "$venv"
fi
"$uv" pip install --python "$venv/bin/python" --torch-backend=auto \
    "vllm==$vllm_version" "lmcache==$lmcache_version"
"$uv" pip check --python "$venv/bin/python"
"$venv/bin/python" -c 'import torch, lmcache.cuda_ops; from lmcache.integration.vllm.lmcache_connector_v1 import LMCacheConnectorV1Dynamic; print(torch.__version__, torch.version.cuda, LMCacheConnectorV1Dynamic.__name__)'
