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
echo -e "⚡ Simplicio 27B · Installer and runner"
echo "🌐 SimpleTI: https://simpleti.com.br/simplicio-27b/"
echo "------------------------------------------------------------------"

# 1. Check for Ollama, install it if missing
if command -v ollama >/dev/null 2>&1; then
    echo -e "✅ Ollama found: $(ollama --version)"
else
    echo -e "📦 Installing Ollama..."
    curl -fsSL https://ollama.com/install.sh | sh
fi

# 2. Check for OpenCode
if command -v opencode >/dev/null 2>&1; then
    echo -e "✅ OpenCode CLI found."
    echo "   OpenCode needs tool calls, which the vLLM server provides. Setup:"
    echo "   https://github.com/simpletibr/simplicio-27b#opencode"
fi

# 3. Start Simplicio 27B
echo ""
echo -e "🚀 Downloading and starting Simplicio 27B in Ollama..."
echo "------------------------------------------------------------------"
exec ollama run wesleysimplicio/simplicio-27b
