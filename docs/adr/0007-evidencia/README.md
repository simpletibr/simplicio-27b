# Evidência da ADR 0007: estudo de ensemble entre famílias (2026-11-10)

Esta pasta guarda os scripts e os dois relatórios do estudo que a [ADR 0007](../0007-duas-partes-entre-familias.md) cita. Tudo rodou em CPU, sem token do Hugging Face e sem rodar nenhum modelo: o estudo só lê probabilidades que os autores publicaram.

Regra de evidência: os números dos relatórios são medidas. O que esta pasta **não** mede está dito no fim.

## Arquivos

| arquivo | o que é |
|---|---|
| `relatorio_30_benchmarks.json` | o estudo principal: 30 dos 37 benchmarks do índice público, 94.173 pedidos, pontuados pelas funções do próprio kit |
| `relatorio_6_benchmarks.json` | o primeiro recorte (6 benchmarks, 26.419 pedidos), com intervalos de confiança por bootstrap pareado |
| `build_gold.py`, `build_added_rows.py` | gabarito e linhas dos benchmarks "added" (MMLU-Pro, BBH, When2Call, New Yorker, RAGTruth, PhishNChips), com os adaptadores do kit |
| `rebuild_subset.py` | reconstrói um subconjunto dos benchmarks da base com `acquire`, `normalize` e `freeze` do kit |
| `prepare_rows.py` | aplica o filtro da edição 0.3 e a lista de exclusões; escreve as linhas sem o `state` e um índice |
| `extract.py`, `extract_ext.py` | tiram de um `results.jsonl(.gz)` publicado as probabilidades de cada pedido, conferindo o `payload_sha256` |
| `study.py` | estudo dos 6 benchmarks (pares, peso ajustado em metades, bootstrap) |
| `study_ext.py` | estudo dos 30 benchmarks (singles, 66 pares, metades, trios), pontuado por `benchmark_summary`, `added.report` e `index02` do kit |

## Fontes

- **Placar:** `data/index.json` do Space `multimodalart/jev-decision-index`, gerado em 2026-11-06T09:05:08Z, 274 modelos, 3.724.017 bytes, sha256 `66d257218ef2f6d472fb65304b5afbe2ef7d5e3d5d26364478d56a0709b9be5c`.
- **Kit:** `apolinario/decision-index` no commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` (o mesmo de `decide/config.yaml`). As fontes de cada benchmark e os sha256 fixados estão no próprio kit (`decision_index/suite/build/acquire.py` e `adapters_added.py`).
- **Resultados por linha** (datasets públicos do Hugging Face, lidos na revisão indicada):

| nome no estudo | dataset | run | revisão |
|---|---|---|---|
| `glide` | `fastino/GLiDE-V1.2.1-Loop-decision-index` | `glide-v1.2.1-loop` | `aa491d17b524a3329a93b4392c635a12e15d1a15` |
| `d3-ultra` | `vllm-sr/d3-ultra-decision-index` | `d3-ultra` | `de1289830e68ff45cfb4be71aa3ccfc7020ce605` |
| `nimble` | `bespokelabs/nimble-v4-max-decision-index` | `nimble-v4.1-max` | `293b569eb9f46a3e6175e45302abf5fdbe4212dc` |
| `d3-turbo` | `vllm-sr/d3-turbo-decision-index` | `d3-turbo-v1.1` | `1431a76c44a0679e4f588cc4a6fe7d1976d28140` |
| `d3` | `vllm-sr/d3-decision-index` | `d3` | `0cfbfd61699fd4c614658d0112616cd5fea11a89` |
| `knowline` | `PelaAI/KnowLine-31B-Gen1.5-decision-index` | `KnowLine-31B-Gen1.5` | `d22d773d8ba3a02bb94047871d67ec1e7a743306` |
| `flexdm` | `1000luca/FlexDM-LoRA-v2.1-decision-index` | `flexdm-l2-sub3` | `0236738c480632075b187fb9dac4de10181e8a4b` |
| `sieve` | `sthanika-ai/Sieve-31B-v3-decision-index-results` | `Sieve-31B-v3` | `62fe0997d176f73267923661dee73dde9c899e94` |
| `turion` | `turion-ai/turion-jev-31b-decision-index-results` | `turion-jev-31b` | `cd7b034e9d249ea5556565286234b1393de2fb0c` |
| `h2o` | `h2oai/h2o-lightning-31b-decision-index` | `h2o-lightning-31b` | `9383d920fc9adc0fe71be7c9e001fe156b7d9002` |
| `jeffan` | `tattrongvu/Jeffan-31b-1710-decision-index` | `Jeffan-31b-1710` | `8eb60531212bf16b92be6f1274f06f4649f6f97f` |
| `liquamen` | `plantwaterer/liquamen-decision-index` | `liquamen` | `01e270beb292e728469e4128bd0de3fd7c4934b3` |
| `reax` | `decisionmodels/REAX-1.3-decision-index` | `reax13-public-0.3-20261103` | `ed690ece351a37b0c77dad648852b826947a24d3` |

O `pplx-decider-v1.2` não entrou: não há dataset público com os resultados por linha dele.

## O que foi conferido

1. **Mesmo pedido.** O `payload_sha256` de cada linha reconstruída é igual ao do resultado publicado: 111.754 de 111.754 linhas, nos 13 runs. Dessas, 94.173 são de benchmarks do índice; o resto é MMLU e ARC, que o índice não conta.
2. **Mesma nota.** Para os 13 runs e os 30 benchmarks, o score e o skill que o kit calcula sobre as linhas reconstruídas são iguais aos do placar em 4 casas (diferença máxima 0,0000).
3. **GSM8K 0.3.** O arquivo reconstruído é idêntico ao do lab byte a byte (sha256 `759334894858a37d6bd7bd07298eac8e55da236e31dad327ccd06a9979f4738a`, o `gsm8k_sha256` da edição 0.3 no kit).
4. **Ordem.** O índice parcial (a fórmula do índice público só nos 30 benchmarks) ordena os 13 modelos quase como o índice público do placar: Spearman 0,989.

Ficaram de fora 7 benchmarks do índice: HLE (fechado, pede login), HoVer (banco de 2,2 GB), Amazon ESCI, BRIGHT, ToolRet, API-Bank e ChessBench (fontes grandes). A reconstrução completa pede cerca de 7 GB de downloads e 17 GB de trabalho (README do kit); a máquina tinha 20 GB livres.

## Como repetir

Em uma pasta temporária, com `KIT` apontando para o kit extraído no commit acima e `WORK` para uma pasta vazia. As versões dos pacotes são as que o estudo usou, com Python 3.12.13:

```sh
uv venv --python 3.12 venv
uv pip install --python=venv/bin/python pyarrow==26.0.1 numpy==2.5.4 huggingface_hub==1.33.0 httpx==0.28.1 scipy==1.18.1 mido==1.3.3 python-chess==1.999 chess==1.11.2 tiktoken==0.14.0

# 1. linhas e gabarito (os seis "added" primeiro: ponha as fontes fixadas onde adapters_added.py as espera, dentro de WORK)
venv/bin/python -I build_gold.py "$KIT" "$WORK" gold.jsonl
venv/bin/python -I build_added_rows.py "$KIT" "$WORK" rows/added-rows.jsonl
venv/bin/python -I rebuild_subset.py "$KIT" "$WORK" 24,26,27,28,29 12 4 5 32 33 11,42 21,25 41,44 40 30,43 38 39 9 1 22 23 50 20
# GSM8K da edição 0.3 (imprime o sha256 do arquivo reconstruído)
venv/bin/python -I -c "import sys; sys.path.insert(0, sys.argv[1]); from decision_index.suite.build import gsm8k_v3; from decision_index.suite.build.layout import Layout; l = Layout(sys.argv[2]); print(gsm8k_v3.main(l, l.suite / 'release-v1-rebuilt' / 'selected-rows.jsonl', log=lambda m: None)['gsm8k_sha256'])" "$KIT" "$WORK"
venv/bin/python -I prepare_rows.py "$KIT" rows "$WORK/artifacts/benchmark-suite/release-v1-rebuilt/selected-rows.jsonl" \
    "$WORK/artifacts/benchmark-suite/release-v3-rebuilt/gsm8k-rows.jsonl" rows/added-rows.jsonl

# 2. resultados publicados: results/<nome>/results.jsonl.gz e results/<nome>/benchmark-summary.json, na revisão da tabela
venv/bin/python -I extract.py gold.jsonl results/glide/results.jsonl.gz ext/glide.json glide          # um por modelo
venv/bin/python -I extract_ext.py rows/index.jsonl results/glide/results.jsonl.gz ext2/glide.json glide

# 3. os dois estudos
venv/bin/python -I study.py gold.jsonl ext results index.json relatorio_6_benchmarks.json
venv/bin/python -I study_ext.py "$KIT" rows/slim-rows.jsonl ext2 index.json relatorio_30_benchmarks.json
```

O estudo dos 30 benchmarks levou 861 s numa CPU.

Armadilhas encontradas:

- O `http_fetch` do kit (urllib) devolveu quatro arquivos truncados pelo proxy da máquina (CLINC150, Humicroedit, cfcolor, GPQA). Baixados com `curl` no destino esperado, os sha256 bateram com os fixados e o `acquire` os aceitou.
- Rode o `git` com `GIT_CONFIG_GLOBAL=/dev/null` e o Hub com um `HF_HOME` vazio: nenhuma fonte usada aqui pede credencial.
- `rebuild_subset.py` trata em conjunto os benchmarks que um mesmo construtor do kit faz (por isso `24,26,27,28,29`).

## O que esta pasta não mede

- O Full (a parte privada do placar) de nenhuma combinação. Tudo aqui é a suíte pública.
- Memória e latência de duas partes numa placa: nada rodou numa GPU.
- O efeito de quantizar as partes em FP8.
- Os 7 benchmarks que ficaram de fora e o `pplx-decider-v1.2`.
- Intervalo de confiança no estudo dos 30 benchmarks: a checagem foi repetir cada par nas duas metades dos pedidos. O bootstrap está só no estudo dos 6.
