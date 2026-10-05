<p align="center">
  <img src="https://raw.githubusercontent.com/simpletibr/simplicio-27b/main/assets/simplicio-logo.png" width="130" alt="SimpleTI">
</p>

<h1 align="center">Simplicio 27B</h1>
<p align="center">Modelo aberto de 27B para edição cirúrgica de código. Os gráficos abaixo separam o que este repositório mediu do que a Artificial Analysis publicou para outros modelos.</p>

<p align="center">
  <a href="https://huggingface.co/wesleysimplicio/Simplicio-27B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Simplicio--27B-yellow.svg" alt="Hugging Face"></a>
  <a href="https://ollama.com/wesleysimplicio/simplicio-27b"><img src="https://img.shields.io/badge/Ollama-simplicio--27b-black" alt="Ollama"></a>
  <a href="https://github.com/simpletibr/simplicio-27b"><img src="https://img.shields.io/badge/GitHub-simpletibr%2Fsimplicio--27b-blue?logo=github" alt="GitHub"></a>
</p>

## Como executar

Os pesos ficam num único repositório: [wesleysimplicio/Simplicio-27B](https://huggingface.co/wesleysimplicio/Simplicio-27B). A tag do Ollama é [wesleysimplicio/simplicio-27b](https://ollama.com/wesleysimplicio/simplicio-27b).

```bash
ollama run wesleysimplicio/simplicio-27b
```

```bash
aider --model ollama_chat/wesleysimplicio/simplicio-27b:latest --edit-format diff
```

| Artefato | Arquivo | Tamanho medido |
|---|---|---|
| Adapter LoRA | `adapter_model.safetensors` | 637610624 bytes |
| Merge 16-bit | `model-00001-of-00018.safetensors` … `model-00018-of-00018.safetensors` | 18 shards |
| GGUF Q4_K_M | `Qwen3.8-27B.Q4_K_M.gguf` | 16810715584 bytes |
| Projetor de visão | `Qwen3.8-27B.BF16-mmproj.gguf` | 931145952 bytes |

A tag `latest` do Ollama junta o Q4_K_M e o projetor. A página mostra a janela máxima da arquitetura, 256K. O contexto padrão gravado na tag é `num_ctx` 32768.

## Como ler os números

Há dois conjuntos, e eles não entram no mesmo eixo.

1. **Conjunto próprio.** 120 tarefas não vistas, pareadas com Qwen3.8-27B base. Uma tentativa por tarefa, temperature 0. O desfecho é binário: o patch passa no teste da tarefa ou não passa.
2. **Artificial Analysis Coding Agent Index v1.5.** DeepSWE v1.1, Terminal-Bench 4.0 e SWE-Atlas-QnA, com peso igual. Três tentativas por tarefa. Opus 5.5, Sonnet 5.5 e GPT-6.1 Sol aparecem aqui com os números publicados pela Artificial Analysis em 5 de outubro de 2026. Simplicio 27B não foi executado nesse índice.

## Conjunto próprio

Amostragem: 30 diffs cirúrgicos, 30 casos funcionais, 30 armadilhas de API e 30 tarefas adversariais. Temperature 0, `do_sample` desligado, `max_new_tokens` 1536. O mesmo prompt vai para o Simplicio 27B e para a base. Três repetições a temperature 0 produziram o mesmo hash.

<p align="center">
  <img src="https://raw.githubusercontent.com/simpletibr/simplicio-27b/main/assets/aa_pass_rate.png" alt="Taxa de aprovação, AST válido e ausência de API fantasma no conjunto de 120 tarefas" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/simpletibr/simplicio-27b/main/assets/aa_tokens.png" alt="Tokens de saída por tarefa no conjunto de 120" width="100%">
</p>

| Métrica | Simplicio 27B | Qwen3.8-27B base | Intervalo de Wilson 95% |
|---|---:|---:|---|
| Aprovação | 96,67% (116/120) | 35,00% (42/120) | Simplicio [91,74%, 98,70%]; base [27,05%, 43,88%] |
| AST válido | 100% (120/120) | 88,33% (106/120) | Simplicio [96,90%, 100%]; base [81,37%, 92,92%] |
| Sem API fantasma | 100% (120/120) | 83,33% (100/120) | Simplicio [96,90%, 100%]; base [75,65%, 88,94%] |
| Tokens de saída / tarefa | 480,5 | 835,0 | redução de 42,46% |

A tabela 2×2 do arquivo `benchmarks/statistical_proof_n120.json` fecha em 120: 41 os dois passam, 75 só o Simplicio passa, 1 só a base passa, 3 os dois falham. Dela saem 116/120 e 42/120. O teste exato de McNemar nesses pares discordantes dá p = 2,04×10⁻²¹. O campo de proporção da base no mesmo JSON diz 34/120 e não fecha com essa tabela. Os gráficos usam a tabela 2×2.

Uma checagem de hardware à parte, em 3 tarefas no NVIDIA A100-SXM4-40GB, registrou as três como aprovação e latência de cerca de 107 s por tarefa. Esse N=3 não é a taxa de 116/120.

Arquivos: `benchmarks/statistical_proof_n120.json`, `benchmarks/empirical_a100_results.json`, `data/unseen_eval_120.json`.

## Opus 5.5, Sonnet 5.5 e GPT-6.1 Sol

O cartão segue o formato de comparação por linha. A coluna do Simplicio 27B só traz número onde o conjunto próprio mediu esse modelo. Nas linhas da Artificial Analysis a coluna fica vazia de propósito.

| Avaliação | Amostragem | Simplicio 27B | Opus 5.5 (max) | Sonnet 5.5 (max) | GPT-6.1 Sol (xhigh) |
|---|---|---:|---:|---:|---:|
| Aprovação no conjunto próprio | 120 tarefas, 1 tentativa, T=0 | 96,67% (116/120) | não medido | não medido | não medido |
| Tokens de saída no conjunto próprio | média, max 1536 | 480,5 | não medido | não medido | não medido |
| Coding Agent Index v1.5 | média de 3 benchmarks, 3 tentativas | não avaliado | 66 | 68 | 63 |
| DeepSWE v1.1 | 113 tarefas | não avaliado | 68% | 72% | 73% |
| Terminal-Bench 4.0 | 66 tarefas | não avaliado | 63% | 66% | 55% |
| SWE-Atlas-QnA | 124 tarefas | não avaliado | 66% | 67% | 61% |
| Custo por tarefa | API pay-per-token | sem preço de API medido | US$ 13,04 | US$ 14,19 | US$ 1,04 |
| Tempo por tarefa | wall time do agente | não avaliado | 1,1 h | 1,5 h | 15,5 min |
| Tokens por tarefa | total do agente | não avaliado | 15,6 M | 27,7 M | 3,2 M |

Harness publicado: Opus 5.5 e Sonnet 5.5 em Claude Code, GPT-6.1 Sol em Codex. Fonte: [Claude Code vs Codex](https://artificialanalysis.ai/agents/coding-agents/comparisons/claude-code-vs-codex), consulta em 5 de outubro de 2026. O post com o gráfico de índice contra custo está em [Artificial Analysis, 2 out 2026](https://x.com/ArtificialAnlys/status/2105814318294114720).

<p align="center">
  <img src="https://raw.githubusercontent.com/simpletibr/simplicio-27b/main/assets/aa_index.png" alt="Coding Agent Index de Sonnet 5.5, Opus 5.5 e GPT-6.1 Sol" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/simpletibr/simplicio-27b/main/assets/aa_components.png" alt="DeepSWE, Terminal-Bench 4.0 e SWE-Atlas-QnA para os três modelos" width="100%">
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/simpletibr/simplicio-27b/main/assets/aa_index_vs_cost.png" alt="Índice contra custo por tarefa, com Simplicio 27B fora do eixo" width="100%">
</p>

480,5 tokens de saída no conjunto próprio e 3,2 milhões de tokens de agente no índice da Artificial Analysis medem coisas diferentes. O gráfico de custo não coloca o Simplicio 27B num ponto inventado.

## Treino

- 101 trajetórias: Python 45%, TypeScript 25%, Rust 10%, Go 10%, SQL 10%.
- 10 épocas, batch efetivo 8, 120 passos, LoRA nas camadas 48–63, camadas 0–47 congeladas.
- Notebook: `Simplicio_27B_Training_Colab.ipynb`. Hardware registrado: um NVIDIA A100-SXM4 40GB.
- Base: Qwen3.8-27B. Treino com Unsloth, NF4.

Servir o merge com vLLM: `./deploy/serve_vllm.sh wesleysimplicio/Simplicio-27B 8000`. O restante da distribuição está em `deploy/DISTRIBUTION_GUIDE.md`.

## Citação

```bibtex
@software{simplicio_27b_2026,
  author = {Wesley Simplicio},
  title = {Simplicio 27B},
  year = {2026},
  publisher = {SimpleTI},
  url = {https://github.com/simpletibr/simplicio-27b},
  howpublished = {\url{https://huggingface.co/wesleysimplicio/Simplicio-27B}}
}
```
