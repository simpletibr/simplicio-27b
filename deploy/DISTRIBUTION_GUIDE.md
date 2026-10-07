# Simplicio 27B distribution guide
### Ollama · OpenRouter · OpenCode / Aider / Continue.dev / Cursor

How to publish **Simplicio 27B** on model platforms and connect it to coding tools.

---

## 1. Ollama

The tag `wesleysimplicio/simplicio-27b:latest` is in the Ollama library.

```bash
ollama run wesleysimplicio/simplicio-27b
```

It combines two files published in `wesleysimplicio/Simplicio-27B`:

| File | Role | Size |
|---|---|---|
| `Qwen3.8-27B.Q4_K_M.gguf` | Q4_K_M weights | 16810715584 bytes |
| `Qwen3.8-27B.BF16-mmproj.gguf` | vision projector | 931145952 bytes |

The `Modelfile` at the repository root uses these file names with `temperature` 0.2, `top_p` 0.95, `top_k` 40, `repeat_penalty` 1.1 and `num_ctx` 40960, the value in `deploy/context.env`. The published tag still has `num_ctx` 32768; recreate and push it to bring it in line. The Ollama page shows the architecture's maximum window, 256K.

To recreate the tag, download both files into this folder and run:

```bash
ollama create wesleysimplicio/simplicio-27b -f Modelfile
ollama push wesleysimplicio/simplicio-27b
```

## 2. OpenRouter

Simplicio 27B is not listed on OpenRouter yet. OpenRouter does not host models; it routes requests to inference providers. There are two ways to get listed.

### Option A: a hosted inference provider
1. Deploy the public Hugging Face repository `wesleysimplicio/Simplicio-27B` on a provider such as [DeepInfra](https://deepinfra.com), which gives you an OpenAI-compatible endpoint.
2. Ask for the model to be added through that provider's OpenRouter integration or OpenRouter's provider form.

### Option B: your own endpoint
If you host the model on your own GPU (RunPod, Lambda Labs, Scaleway or your own server):
1. Install `vllm==0.31.0` and start the server. The script serves the merged BF16 weights with `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`, thinking off by default and model names `simplicio-27b` and `simpleti/simplicio-27b`:
   ```bash
   pip install vllm==0.31.0
   ./deploy/serve_vllm.sh wesleysimplicio/Simplicio-27B 8000
   ```
2. Expose the endpoint over HTTPS (Cloudflare Tunnel, Caddy or Nginx).
3. Register the OpenAI-compatible endpoint (`https://your-domain.com/v1`) at [openrouter.ai/provider](https://openrouter.ai/provider).

---

## 3. Coding tools

Tool calls need the vLLM server above. The Ollama template does not declare tools, so tools that rely on tool calls only work against vLLM.

### A. OpenCode
Add the vLLM server to `opencode.json` as an OpenAI-compatible provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "simplicio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Simplicio 27B",
      "options": { "baseURL": "http://localhost:8000/v1" },
      "models": { "simplicio-27b": { "name": "Simplicio 27B" } }
    }
  }
}
```

```bash
opencode -m simplicio/simplicio-27b
```

### B. Aider (not tested)
The model writes four-character markers (`<<<< SEARCH`, `====`, `>>>> REPLACE`); Aider's diff edit format uses seven. These commands connect Aider, but edits have not been tested:

```bash
# Local Ollama:
aider --model ollama_chat/wesleysimplicio/simplicio-27b --edit-format diff

# Local vLLM:
aider --openai-api-base http://localhost:8000/v1 \
      --openai-api-key none \
      --model openai/simplicio-27b \
      --edit-format diff
```

### C. Open Interpreter
```bash
# Ollama:
interpreter --model ollama/wesleysimplicio/simplicio-27b

# OpenAI-compatible endpoint:
interpreter --api_base http://localhost:8000/v1 --model simplicio-27b
```

### D. Continue.dev (VS Code / JetBrains extension)
Add to `~/.continue/config.json`:
```json
{
  "models": [
    {
      "title": "Simplicio 27B (Ollama)",
      "provider": "ollama",
      "model": "wesleysimplicio/simplicio-27b"
    },
    {
      "title": "Simplicio 27B (vLLM)",
      "provider": "openai",
      "model": "simplicio-27b",
      "apiBase": "http://localhost:8000/v1",
      "apiKey": "none"
    }
  ]
}
```

### E. Cursor / Windsurf / Cline
1. Open **Settings → Models** (or Custom OpenAI API).
2. Set the base URL to `http://localhost:8000/v1`.
3. Set the model name to `simplicio-27b`.
