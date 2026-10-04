#!/usr/bin/env bash
# ==============================================================================
# Servidor de Produção vLLM (OpenAI-Compatible API) para Simplicio 27B
# Compatível com OpenRouter, OpenCode, Aider, Continue.dev e Cursor
# ==============================================================================

MODEL_ID=${1:-"wesleysimplicio/Simplicio-27B"}
PORT=${2:-8000}
HOST="0.0.0.0"

echo "=== Iniciando vLLM OpenAI-Compatible Server para $MODEL_ID na porta $PORT ==="

vllm serve "$MODEL_ID" \
    --host "$HOST" \
    --port "$PORT" \
    --tensor-parallel-size 1 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.95 \
    --chat-template-preset chatml \
    --served-model-name "simplicio-27b"
