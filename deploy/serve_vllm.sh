#!/usr/bin/env bash
# ==============================================================================
# Servidor de Produção vLLM (OpenAI-Compatible API) para Simplicio 27B
# Compatível com OpenRouter, OpenCode, Aider, Continue.dev e Cursor
# Closes: simpletibr/simplicio-27b#2 simpletibr/simplicio-27b#3
# ==============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
set -a
. "${ROOT}/deploy/context.env"
set +a
MODEL_ID=${1:-"wesleysimplicio/Simplicio-27B"}
PORT=${2:-8000}
HOST="${HOST:-0.0.0.0}"
PLUGIN="${ROOT}/deploy/simplicio_tool_parser.py"
TEMPLATE="${ROOT}/deploy/chat_template_chatml.jinja"

echo "=== Iniciando vLLM OpenAI-Compatible Server para $MODEL_ID na porta $PORT (max-model-len ${MAX_MODEL_LEN}) ==="

exec vllm serve "$MODEL_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization 0.95 \
    --chat-template "$TEMPLATE" \
    --enable-auto-tool-choice \
    --tool-parser-plugin "$PLUGIN" \
    --tool-call-parser simplicio \
    --reasoning-parser qwen3 \
    --served-model-name simplicio-27b simpleti/simplicio-27b \
    --stop '<|im_end|>' \
    --stop '<|endoftext|>' \
    --stop '</deliver>' \
    --stop $'</tool>\n<tool>'
