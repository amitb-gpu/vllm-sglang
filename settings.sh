# Sourced by setup.sh and serve.sh.
runtime="${TICKET_INFERENCE_HOME:-$HOME/.cache/ticket-inference}"
model=Qwen/Qwen3-4B-Instruct-2507
revision=cdbee75f17c01a7cc42f958dc650907174af0554
vllm_version=0.33.0
sglang_version=0.5.23

export HF_HOME="$runtime/huggingface"
export HF_HUB_DISABLE_TELEMETRY=1
