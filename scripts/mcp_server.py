#!/usr/bin/env python3
"""Servidor MCP (Model Context Protocol) nativo do Simplicio 27B.

Expõe ferramentas cirúrgicas para assistentes e agentes de código (Claude Code,
Cursor, Windsurf, OpenCode, Antigravity) via transporte stdio padrão JSON-RPC 2.0.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

SEARCH_RE = re.compile(r"^<{4,7} SEARCH\s*$")
DIVIDER_RE = re.compile(r"^={4,7}\s*$")
REPLACE_RE = re.compile(r"^>{4,7} REPLACE\s*$")


def parse_blocks(text: str) -> list[tuple[str, str]]:
    """Extrai blocos SEARCH/REPLACE (suporta marcadores de 4 a 7 caracteres)."""
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    blocks = []
    state = "out"
    search: list[str] = []
    replace: list[str] = []
    for line in text.splitlines():
        if state == "out":
            if SEARCH_RE.match(line):
                state = "search"
                search, replace = [], []
        elif state == "search":
            if DIVIDER_RE.match(line):
                state = "replace"
            else:
                search.append(line)
        else:
            if REPLACE_RE.match(line):
                blocks.append(("\n".join(search), "\n".join(replace)))
                state = "out"
            else:
                replace.append(line)
    return blocks


def apply_patch_to_text(source: str, blocks: list[tuple[str, str]]) -> str:
    """Aplica blocos SEARCH/REPLACE ao texto original de forma determinística."""
    if not blocks:
        raise ValueError("Nenhum bloco SEARCH/REPLACE encontrado no patch.")
    for i, (search, replace) in enumerate(blocks):
        if not search:
            raise ValueError(f"Bloco {i+1}: o trecho SEARCH está vazio.")
        if search not in source:
            raise ValueError(f"Bloco {i+1}: trecho SEARCH não encontrado no arquivo:\n---\n{search}\n---")
        # Substitui apenas a primeira ocorrência
        source = source.replace(search, replace, 1)
    return source


def tool_apply_diff(arguments: dict[str, Any]) -> str:
    path_str = arguments.get("path")
    patch = arguments.get("patch", "")
    if not path_str:
        return "Erro: parâmetro path é obrigatório."
    p = Path(path_str).resolve()
    if not p.is_file():
        return f"Erro: arquivo não encontrado em {p}"

    try:
        source = p.read_text(encoding="utf-8")
        blocks = parse_blocks(patch)
        updated = apply_patch_to_text(source, blocks)
        p.write_text(updated, encoding="utf-8")
        return f"Sucesso: {len(blocks)} bloco(s) SEARCH/REPLACE aplicado(s) cirurgicamente em {p.name}."
    except Exception as exc:
        return f"Falha na aplicação do patch: {exc}"


def tool_validate_patch(arguments: dict[str, Any]) -> str:
    patch = arguments.get("patch", "")
    blocks = parse_blocks(patch)
    if not blocks:
        return "Inválido: nenhum bloco SEARCH/REPLACE reconhecido."
    summary = []
    for i, (s, r) in enumerate(blocks, 1):
        s_lines = len(s.splitlines())
        r_lines = len(r.splitlines())
        summary.append(f"Bloco {i}: SEARCH ({s_lines} linhas) -> REPLACE ({r_lines} linhas)")
    return "Válido. " + "; ".join(summary)


def tool_format_prompt(arguments: dict[str, Any]) -> str:
    context = arguments.get("context", "")
    task = arguments.get("task", "")
    system = ("Voce e o Simplicio 27B, treinado para executar tarefas de desenvolvimento "
              "seguindo rigorosamente os 50 pontos do Simplicio-Loop: Orientacao, Planejamento, "
              "Edicao Cirurgica por Diff, Validacao e Entrega Verificada sem alucinacao.")
    user = f"Contexto:\n{context}\n\nTarefa:\n{task}\n"
    return json.dumps({"system": system, "user": user}, indent=2, ensure_ascii=False)


TOOLS = [
    {
        "name": "apply_surgical_diff",
        "description": "Aplica blocos SEARCH/REPLACE cirúrgicos a um arquivo especificado.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Caminho absoluto ou relativo para o arquivo."},
                "patch": {"type": "string", "description": "Texto contendo um ou mais blocos SEARCH/REPLACE (4 ou 7 caracteres)."}
            },
            "required": ["path", "patch"]
        }
    },
    {
        "name": "validate_patch",
        "description": "Verifica se um patch contém blocos SEARCH/REPLACE sintaticamente válidos.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "patch": {"type": "string", "description": "Texto do patch a ser validado."}
            },
            "required": ["patch"]
        }
    },
    {
        "name": "format_prompt",
        "description": "Formata uma instrução e contexto no padrão oficial do Simplicio-Loop.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "Arquivos e detalhes do repositório."},
                "task": {"type": "string", "description": "Alteração desejada."}
            },
            "required": ["context", "task"]
        }
    }
]


def handle_request(req: dict[str, Any]) -> dict[str, Any] | None:
    method = req.get("method")
    msg_id = req.get("id")

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {
                    "name": "simplicio-27b",
                    "version": "0.2.0"
                },
                "capabilities": {
                    "tools": {}
                }
            }
        }
    elif method == "notifications/initialized":
        return None
    elif method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "tools": TOOLS
            }
        }
    elif method == "tools/call":
        params = req.get("params", {})
        tool_name = params.get("name")
        arguments = params.get("arguments", {})

        if tool_name == "apply_surgical_diff":
            text = tool_apply_diff(arguments)
        elif tool_name == "validate_patch":
            text = tool_validate_patch(arguments)
        elif tool_name == "format_prompt":
            text = tool_format_prompt(arguments)
        else:
            text = f"Erro: ferramenta desconhecida {tool_name!r}"

        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "result": {
                "content": [
                    {
                        "type": "text",
                        "text": text
                    }
                ]
            }
        }
    elif method == "ping":
        return {"jsonrpc": "2.0", "id": msg_id, "result": {}}

    if msg_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": msg_id,
            "error": {
                "code": -32601,
                "message": f"Método não encontrado: {method}"
            }
        }
    return None


def main() -> None:
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            resp = handle_request(req)
            if resp is not None:
                sys.stdout.write(json.dumps(resp, ensure_ascii=False) + "\n")
                sys.stdout.flush()
        except Exception as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"}
            }
            sys.stdout.write(json.dumps(err_resp, ensure_ascii=False) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
