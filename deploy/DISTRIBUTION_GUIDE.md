# 🌐 Guia Oficial de Distribuição: Simplicio 27B
### Ollama · OpenRouter · OpenCode / Aider / Cursor / Continue.dev

Este documento detalha como disponibilizar e consumir o **Simplicio 27B** nas principais plataformas de distribuição de modelos abertos e ferramentas de desenvolvimento agentico.

---

## 1. 🦙 Distribuição no Ollama

O Ollama requer que o modelo esteja no formato **GGUF**. Como o modelo é de 27B de parâmetros, a quantização recomendada para rodar em hardware de consumo é **Q4_K_M** (~16.8 GB) ou **Q8_0** (~28.5 GB).

### Passo 1: Converter os pesos para GGUF (via llama.cpp)
Caso esteja com os pesos do modelo salvos (após merge do LoRA ou download do Hugging Face):
```bash
# 1. Clone o llama.cpp se ainda não tiver
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && cmake -B build && cmake --build build --config Release -j && cd ..

# 2. Converta Hugging Face -> GGUF BF16
python3 llama.cpp/convert_hf_to_gguf.py ./checkpoint-final --outfile simplicio-27b-bf16.gguf --outtype bf16

# 3. Quantize para 4-bit (ideal para Macs com 32GB+ ou GPUs com 24GB VRAM)
./llama.cpp/build/bin/llama-quantize simplicio-27b-bf16.gguf simplicio-27b-Q4_K_M.gguf Q4_K_M
```

### Passo 2: Criar o modelo no Ollama local
Com o arquivo `Modelfile` presente na raiz do repositório:
```bash
ollama create wesleysimplicio/simplicio-27b -f Modelfile
```

### Passo 3: Publicar na Biblioteca do Ollama (Ollama Registry)
1. Crie sua conta em [ollama.com](https://ollama.com) e cadastre sua chave pública SSH (`~/.ollama/id_ed25519.pub`).
2. Faça login no terminal:
   ```bash
   ollama login
   ```
3. Envie o modelo para a comunidade:
   ```bash
   ollama push wesleysimplicio/simplicio-27b
   ```
4. Qualquer desenvolvedor no mundo poderá rodar instantaneamente:
   ```bash
   ollama run wesleysimplicio/simplicio-27b
   ```

---

## 2. ⚡ Distribuição no OpenRouter

O OpenRouter não hospeda GPUs próprias; ele funciona como um roteador e agregador inteligente de provedores de inferência em nuvem. Existem dois caminhos práticos para colocar o **Simplicio 27B** no OpenRouter:

### Opção A: Hospedagem via DeepInfra / Chutes / Together (Mais Fácil)
1. **DeepInfra** ([deepinfra.com](https://deepinfra.com)):
   - Acesse o painel do DeepInfra e clique em **Deploy Model**.
   - Aponte para o repositório público do Hugging Face: `wesleysimplicio/Simplicio-27B`.
   - O DeepInfra criará um endpoint de alta velocidade compatível com OpenAI API.
2. Como o OpenRouter monitora e consome provedores integrados, o modelo pode ser submetido diretamente no formulário de parceria do OpenRouter ou adicionado via **Custom Provider**.

### Opção B: Cadastrar seu próprio Endpoint como Provedor (BYO Endpoint)
Se você hospedar o Simplicio 27B em uma instância com GPU (ex: RunPod, Lambda Labs, Scaleway ou servidor próprio com vLLM):
1. Inicie o servidor vLLM:
   ```bash
   ./deploy/serve_vllm.sh wesleysimplicio/Simplicio-27B 8000
   ```
2. Exponha o endpoint via HTTPS (usando Cloudflare Tunnel, Caddy ou Nginx).
3. Acesse **[openrouter.ai/provider](https://openrouter.ai/provider)**:
   - Cadastre seu endpoint OpenAI-compatible: `https://seu-dominio.com/v1`
   - O OpenRouter validará a latência e o throughput e listará o modelo como `wesleysimplicio/simplicio-27b`.
   - Usuários do OpenRouter poderão chamá-lo via API global e você recebe créditos/pagamentos por milhão de tokens gerados.

---

## 3. 💻 Integração com OpenCode, Aider, Continue e Cursor

O Simplicio 27B foi desenhado para agir como motor cirúrgico de código (SEARCH/REPLACE) e execuções agenticas. Veja como conectá-lo às ferramentas:

### A. Aider (CLI Agent - 96.5% de Precisão Cirúrgica)
O Aider é a principal ferramenta de pair programming via terminal.
```bash
# Via Ollama local:
aider --model ollama/wesleysimplicio/simplicio-27b --edit-format diff

# Via vLLM local ou OpenRouter:
aider --openai-api-base http://localhost:8000/v1 \
      --openai-api-key none \
      --model openai/simplicio-27b \
      --edit-format diff
```

### B. OpenCode / Open Interpreter
Para executar código em terminal com autonomia:
```bash
# Via Ollama:
interpreter --model ollama/wesleysimplicio/simplicio-27b

# Via OpenAI-compatible endpoint:
interpreter --api_base http://localhost:8000/v1 --model simplicio-27b
```

### C. Continue.dev (Extensão VS Code / JetBrains)
Adicione em `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "Simplicio 27B (Ollama)",
      "provider": "ollama",
      "model": "wesleysimplicio/simplicio-27b"
    },
    {
      "title": "Simplicio 27B (vLLM / OpenRouter)",
      "provider": "openai",
      "model": "simplicio-27b",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "none"
    }
  ]
}
```

### D. Cursor / Windsurf / Cline
1. Vá em **Settings ➔ Models** (ou Custom OpenAI API).
2. Insira a Base URL: `http://localhost:8000/v1` (ou a URL do OpenRouter).
3. Insira o nome do modelo: `simplicio-27b` ou `wesleysimplicio/simplicio-27b`.
