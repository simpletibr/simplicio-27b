#!/usr/bin/env bash
# ==============================================================================
# Pipeline de Conversão e Quantização GGUF para Ollama (Simplicio 27B)
# ==============================================================================
set -e

MODEL_DIR=${1:-"./checkpoint-final"}
OUTPUT_NAME=${2:-"simplicio-27b"}

echo "=== 1. Clonando llama.cpp (se necessário) ==="
if [ ! -d "llama.cpp" ]; then
    git clone https://github.com/ggerganov/llama.cpp
    cd llama.cpp
    cmake -B build
    cmake --build build --config Release -j
    pip install -r requirements.txt
    cd ..
fi

echo "=== 2. Convertendo Hugging Face para GGUF (FP16 / BF16) ==="
python3 llama.cpp/convert_hf_to_gguf.py "$MODEL_DIR" \
    --outfile "${OUTPUT_NAME}-bf16.gguf" \
    --outtype bf16

echo "=== 3. Gerando Quantização Recomendada (Q4_K_M ~16.8 GB) ==="
./llama.cpp/build/bin/llama-quantize \
    "${OUTPUT_NAME}-bf16.gguf" \
    "${OUTPUT_NAME}-Q4_K_M.gguf" \
    Q4_K_M

echo "=== 4. Gerando Quantização Alta Fidelidade (Q8_0 ~28.5 GB) ==="
./llama.cpp/build/bin/llama-quantize \
    "${OUTPUT_NAME}-bf16.gguf" \
    "${OUTPUT_NAME}-Q8_0.gguf" \
    Q8_0

echo "=== 5. Criando Modelo no Ollama Local ==="
ollama create wesleysimplicio/simplicio-27b -f Modelfile

echo "=== Concluído! Para testar execute: ==="
echo "ollama run wesleysimplicio/simplicio-27b"
