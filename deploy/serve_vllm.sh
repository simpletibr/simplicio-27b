#!/usr/bin/env bash
# Simplicio 27B: the only vLLM serve command (README, Colab, self-hosting).
# Serves the merged BF16 checkpoint with the model's own chat template and
# vLLM's built-in Qwen parsers. Requires vllm==0.31.0.
# Usage: ./deploy/serve_vllm.sh [MODEL_ID] [PORT]
# Env:   HOST (default 127.0.0.1), VLLM_API_KEY (vLLM then requires it on /v1/*), GPU_MEM (default 0.92), VLLM_BIN (default vllm)
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
# shellcheck disable=SC1091
. "${ROOT}/deploy/context.env"
set +a

MODEL_ID="${1:-wesleysimplicio/Simplicio-27B}"
PORT="${2:-8000}"
HOST="${HOST:-127.0.0.1}"
GPU_MEM="${GPU_MEM:-0.92}"
VLLM_BIN="${VLLM_BIN:-vllm}"

echo "=== vLLM ${MODEL_ID} on ${HOST}:${PORT} (max-model-len ${MAX_MODEL_LEN}) ==="

exec "${VLLM_BIN}" serve "${MODEL_ID}" \
    --host "${HOST}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM}" \
    --served-model-name simplicio-27b simpleti/simplicio-27b \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --sse-keep-alive-interval 15 \
    --enable-force-include-usage
