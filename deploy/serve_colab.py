# ==============================================================================
# SimpleTI: Simplicio 27B · 1-Click vLLM Serving with Cloudflare Tunnel
# Zero Fixed GPU Cost · High Throughput OpenAI-Compatible Endpoint
# https://simpleti.com.br/simplicio-27b
# ==============================================================================

import subprocess
import time
import re
import os

print("🚀 [1/3] Instalando dependências (vLLM e Cloudflare Tunnel)...")
subprocess.run(["pip", "install", "-q", "vllm"], check=False)

# Baixa cloudflared se não existir
if not os.path.exists("cloudflared"):
    subprocess.run(["wget", "-q", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "cloudflared"])
    subprocess.run(["chmod", "+x", "cloudflared"])

print("⚡ [2/3] Iniciando vLLM Server na porta 8000...")
vllm_cmd = [
    "vllm", "serve", "wesleysimplicio/Simplicio-27B",
    "--host", "0.0.0.0",
    "--port", "8000",
    "--tensor-parallel-size", "1",
    "--max-model-len", "16384",
    "--gpu-memory-utilization", "0.92",
    "--chat-template-preset", "chatml",
    "--served-model-name", "simplicio-27b"
]
vllm_proc = subprocess.Popen(vllm_cmd)

print("🌐 [3/3] Abrindo túnel público seguro HTTPS via Cloudflare...")
tunnel_cmd = ["./cloudflared", "tunnel", "--url", "http://localhost:8000"]
tunnel_proc = subprocess.Popen(tunnel_cmd, stderr=subprocess.PIPE, text=True)

# Captura URL pública do tunnel
public_url = None
start = time.time()
while time.time() - start < 30:
    line = tunnel_proc.stderr.readline()
    match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
    if match:
        public_url = match.group(0)
        break

if public_url:
    print("\n==================================================================")
    print(f"🎉 ENDPOINT ATIVO COM SUCESSO (CUSTO ZERO DE GPU FIXA):")
    print(f"👉 Base URL: {public_url}/v1")
    print(f"👉 Modelo:   simplicio-27b")
    print("==================================================================")
    print("\nComo conectar no OpenCode:")
    print(f"  OPENAI_BASE_URL={public_url}/v1 opencode -m openai/simplicio-27b\n")
    print("Como conectar no Aider:")
    print(f"  aider --openai-api-base {public_url}/v1 --model openai/simplicio-27b\n")
else:
    print("Tunnel iniciado na porta 8000. Verifique logs do cloudflared.")

# Mantém processo rodando
try:
    vllm_proc.wait()
except KeyboardInterrupt:
    vllm_proc.terminate()
    tunnel_proc.terminate()
