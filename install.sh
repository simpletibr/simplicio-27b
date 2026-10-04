#!/usr/bin/env bash
# ==============================================================================
# SimpleTI: Simplicio 27B Fast Installer & Runner
# Frontier Autonomous Software Engineering & Atomic Surgical Code Synthesis Model
# https://simpleti.com.br/simplicio-27b
# ==============================================================================
set -e

BOLD="\033[1m"
BLUE="\033[38;2;39;107;241m"
GREEN="\033[32m"
RESET="\033[0m"

echo -e ""
echo "  ██████╗ ██╗███╗   ███╗██████╗ ██╗     ██╗ ██████╗██╗ ██████╗     ██████╗ ███████╗██████╗ "
echo "  ██╔════╝ ██║████╗ ████║██╔══██╗██║     ██║██╔════╝██║██╔═══██╗    ╚════██╗╚════██║██╔══██╗"
echo "  ███████╗ ██║██╔████╔██║██████╔╝██║     ██║██║     ██║██║   ██║     █████╔╝    ██╔╝██████╔╝"
echo "  ╚════██║ ██║██║╚██╔╝██║██╔═══╝ ██║     ██║██║     ██║██║   ██║    ██╔═══╝    ██╔╝ ██╔══██╗"
echo "  ███████║ ██║██║ ╚═╝ ██║██║     ███████╗██║╚██████╗██║╚██████╔╝    ███████╗   ██║  ██████╔╝"
echo "  ╚══════╝ ╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚═╝ ╚═════╝╚═╝ ╚═════╝     ╚══════╝   ╚═╝  ╚═════╝ "
echo -e ""
echo -e "⚡ Simplicio 27B · Instalador e Inicializador Rápido"
echo "🌐 SimpleTI: https://simpleti.com.br/simplicio-27b"
echo "------------------------------------------------------------------"

# 1. Verificar ou instalar Ollama
if command -v ollama >/dev/null 2>&1; then
    echo -e "✅ Ollama detectado: ollama version is 0.31.1"
else
    echo -e "📦 Instalando Ollama no sistema..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 2. Verificar OpenCode
if command -v opencode >/dev/null 2>&1; then
    echo -e "✅ OpenCode CLI detectado!"
    echo "   Dica: Para rodar no OpenCode execute:"
    echo "   opencode -m openrouter/simpleti/simplicio-27b"
fi

# 3. Verificar Aider
if command -v aider >/dev/null 2>&1; then
    echo -e "✅ Aider CLI detectado!"
    echo "   Dica: Para rodar no Aider execute:"
    echo "   aider --model ollama_chat/wesleysimplicio/simplicio-27b:latest"
fi

# 4. Iniciar Simplicio 27B
echo ""
echo -e "🚀 Baixando e iniciando o Simplicio 27B no Ollama..."
echo "------------------------------------------------------------------"
exec ollama run wesleysimplicio/simplicio-27b
