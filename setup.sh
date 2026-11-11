#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/settings.sh"

engine=${1:-}
case "$engine" in
    vllm|sglang) ;;
    *) echo "Usage: bash setup.sh {vllm|sglang}" >&2; exit 2 ;;
esac

export UV_CACHE_DIR="$runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$runtime/python"
uv="$runtime/bin/uv"

if [[ ! -x "$uv" ]]; then
    mkdir -p "$runtime/bin"
    curl -fLsS --retry 2 https://astral.sh/uv/0.13.7/install.sh -o "$runtime/uv-install.sh"
    UV_UNMANAGED_INSTALL="$runtime/bin" sh "$runtime/uv-install.sh"
fi

python="$runtime/$engine/bin/python"
if [[ ! -x "$python" ]]; then
    "$uv" venv --python 3.12 --managed-python "$runtime/$engine"
fi

if [[ "$engine" == vllm ]]; then
    "$uv" pip install --python "$python" --torch-backend=auto "vllm==$vllm_version"
else
    "$uv" pip install --python "$python" --prerelease=allow "sglang==$sglang_version"
fi
"$uv" pip check --python "$python"

# Download once for both engines. The hub reuses completed files on subsequent calls.
"$python" - "$model" "$revision" <<'PY'
import sys

from huggingface_hub import snapshot_download

print(snapshot_download(
    repo_id=sys.argv[1],
    revision=sys.argv[2],
    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja"],
))
PY
