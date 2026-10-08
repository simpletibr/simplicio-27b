# Recibos de aceite ao vivo

`scripts/live_acceptance.py` roda os critérios de aceite das issues #2 a #5 contra o gateway que você apontar (em produção, `https://simpleti.com.br/v1`). Cada execução grava aqui um `acceptance-<UTC>.json`. O script usa só a biblioteca padrão do Python e não traz nenhuma URL de produção: a URL-base vem de `--base-url` ou de `SIMPLETI_BASE_URL`.

```bash
read -rs SIMPLETI_API_KEY && export SIMPLETI_API_KEY   # chave de API de cliente, nunca a chave admin
python3 scripts/live_acceptance.py --base-url https://simpleti.com.br/v1               # fase up: 13 checagens, com o Colab no ar
python3 scripts/live_acceptance.py --base-url https://simpleti.com.br/v1 --phase down  # fase down: Colab desligado há 120 s
```

A chave entra só por `SIMPLETI_API_KEY` (nunca por argumento ou URL) e é removida de tudo que o script imprime ou grava. A URL-base não pode ter credenciais, query nem fragmento.

O script imprime `PASS <id>` ou `FAIL <id>: <motivo>` por checagem e, no fim, `recibo: <caminho>`. Uma checagem que não consegue rodar (rede, 403, 404, 401, exceção) é `FAIL` com o motivo, nunca `PASS`.

| saída | significado |
|---|---|
| 0 | todas as checagens passaram |
| 1 | alguma checagem falhou |
| 2 | falta configuração (`SIMPLETI_API_KEY`, URL-base) ou o recibo não pôde ser gravado; nada é criado |

Opções: `--phase up|down` (padrão `up`), `--out-dir DIR` (padrão `benchmarks/live`), `--out ARQUIVO` (caminho exato; `-` imprime o JSON no stdout e manda os `PASS`/`FAIL` ao stderr), `--timeout SEGUNDOS` (por request, padrão 300).

## Recibo (`simplicio.live-acceptance/v1`)

No recibo, `passed` no topo é o veredito. As chaves do topo são fixas, nesta ordem:

| chave | conteúdo |
|---|---|
| `schema` | `simplicio.live-acceptance/v1` |
| `base_url` | URL-base usada, sem credenciais |
| `phase` | `up` ou `down` |
| `git_commit` | `git rev-parse HEAD` do repo que rodou o script, ou `null` |
| `started_at`, `finished_at` | UTC, `AAAA-MM-DDTHH:MM:SSZ` |
| `context` | `deploy/context.env` (`MAX_MODEL_LEN`, `OUTPUT_BUDGET`, `MEASURED_AGENT_PROMPT`) |
| `passed` | `true` só se todas as checagens passaram |
| `summary` | `total`, `passed`, `failed`, `failed_ids` |
| `checks` | uma entrada por checagem |

Cada item de `checks` tem `id`, `issue`, `criterion` (o bullet de `## Aceite` da issue), `expected`, `passed`, `problems` (motivos, vazio se passou), `actual` e `exchanges`. Cada troca de `exchanges` tem `request` (sem a chave), `status`, `latency_ms`, `content_type`, `retry_after` e `body`. Prompts com mais de 2000 caracteres aparecem como `<N caracteres omitidos>`.

"200 válido" nas colunas `expected` quer dizer: status 200, `choices[0].finish_reason` em `stop`, `length` ou `tool_calls`, e `usage.prompt_tokens` e `usage.completion_tokens` inteiros.

## Checagens

| id | issue | pedido |
|---|---|---|
| `i2-id-simplicio-27b`, `i2-id-simpleti-simplicio-27b` | #2 | os dois ids respondem 200 |
| `i2-tool-auto` | #2 | `webfetch` com `tool_choice: "auto"` volta `tool_calls`, não XML no `content` |
| `i2-tool-none-oi` | #2 | `tool_choice: "none"`, `oi`: `stop`, sem `<think>` no `content` |
| `i3-pong` | #3 | `pong` com `stop` em `max_tokens: 64` |
| `i3-agente-oi-system`, `i3-agente-oi-tools` | #3 | prompt de agente + `oi`: `stop` ou uma tool call, menos de 512 tokens |
| `i3-segundo-turno` | #3 | turno depois de um `webfetch` não reescreve o resultado da tool |
| `i4-prompt-31692` | #4 | prompt calibrado para o tamanho medido do OpenCode responde 200 |
| `i4-acima-do-teto` | #4 | prompt acima do teto responde 4xx com o `error.message` do vLLM |
| `i5-stream-include-usage`, `i5-stream-sem-options` | #5 | stream com `finish_reason`, `[DONE]` e `usage` no último chunk |
| `i5-todo-200-valido` | #5 | toda troca 200 da execução tem `finish_reason` e `usage` |
| `i5-upstream-fora` (só `--phase down`) | #5 | com o upstream desligado, o request seguinte é 5xx com corpo, não 200 vazio |

Ficam fora do script: #2 4º e #4 1º (cobertos por `tests/test_serve_vllm.py` e `tests/test_context_len.py`) e #5 3º (loop do OpenCode).
