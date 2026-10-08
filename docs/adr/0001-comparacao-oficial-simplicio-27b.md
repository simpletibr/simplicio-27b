# ADR 0001: Comparação oficial do Simplicio 27B

- **Status:** Aceita
- **Data:** 2026-10-08
- **Relacionadas:** #27 (execução no Colab), #25 (harness), #26 (tarefas), #33 (benchmarks públicos, DPO e GRPO)

## Contexto

O README diz que o ganho do fine-tuning ainda não está medido (ver #27). Antes de rodar qualquer comparação, a lista de benchmarks e de comparadores precisa estar fixa, para que os números que favorecem o modelo não sejam escolhidos depois.

Há duas fontes de comparadores, e elas não são equivalentes:

- **Medidos por nós:** o Simplicio 27B e o modelo base Qwen3.8-27B, servidos pelo mesmo `deploy/serve_vllm.sh` e avaliados pelo mesmo harness.
- **Reportados por terceiros:** Claude Haiku 5.5, Claude Haiku 4.5, GPT-6 Luna e Sonnet 5.5 (números de divulgação da Anthropic), e os índices do Artificial Analysis (AA). Não temos acesso aos modelos fechados, e cada fornecedor escolhe seu esforço de raciocínio, seu prompt e seu juiz. Esses números são referência de mercado, não medição nossa.

Os números de referência abaixo vêm das imagens de divulgação entregues pelo dono do projeto em 2026-10-08, transcritas à mão. Conferir contra a fonte oficial antes de citar publicamente.

## Decisão

A comparação oficial tem dois níveis. Só o nível 1 gera resultado do Simplicio 27B.

### Nível 1: medido por nós

| Configuração | Modelo | Thinking |
|---|---|---|
| A | `Qwen/Qwen3.8-27B` (controle base) | desligado |
| B | `Qwen/Qwen3.8-27B` (controle base) | ligado |
| C | `wesleysimplicio/Simplicio-27B` | desligado |
| D | `wesleysimplicio/Simplicio-27B` | ligado |

Regras do nível 1:

- Pesos BF16, servidos por `deploy/serve_vllm.sh`.
- Temperatura 0, fixada pelo harness. O mesmo system prompt e o mesmo template de usuário valem para todas as configurações.
- Thinking desligado e ligado são linhas separadas. Nunca se misturam numa mesma média.
- Uma execução por configuração. Se a diferença ficar dentro do intervalo de confiança, o resultado é empate.

Benchmarks do nível 1:

| Benchmark | Status | Motivo |
|---|---|---|
| Harness próprio: 40 tarefas de edição SEARCH/REPLACE (`benchmarks/harness/tasks/`) | **Obrigatório. Decide esta ADR.** | É o único benchmark do objetivo do projeto: um patch que passa nos testes ocultos. |
| Humanity's Last Exam, sem ferramentas | Obrigatório, depois de fixar o juiz | Exige um modelo juiz. O juiz é escolhido e fixado numa issue própria antes da execução. |
| Humanity's Last Exam, com ferramentas | Fora de escopo agora | Exige um loop de ferramentas que o harness não tem. |

Métricas gravadas em cada run:

- pass rate e intervalo de Wilson 95% (já produzidos pelo harness);
- tokens de entrada e de saída por tarefa;
- latência por tarefa;
- `finish_reason` e contagem de erros;
- CU consumidas e tempo de parede no Colab;
- `simplicio.execution-report/v1`, com CPU e RAM só quando medidos. Nenhum número é estimado.

### Nível 2: reportado por terceiros

Estes benchmarks entram no relatório como referência, com fonte e data, na coluna "reportado". Nenhum deles é executado por nós nesta fase.

| Benchmark | Status | Motivo |
|---|---|---|
| GDPval-AA v2.1 (trabalho de conhecimento, Elo) | Reportado | Depende da avaliação do AA. |
| AA-Briefcase v1.1 (trabalho de conhecimento, Elo) | Reportado | Idem. |
| AA Intelligence Index v4.3.2 (10 avaliações) | Reportado | Índice composto. As sub-avaliações públicas podem virar issue própria. |
| OSWorld 2.1, subconjunto offline (uso de computador) | Fora de escopo agora | Exige desktop virtual e loop de agente. O Simplicio é modelo de código em texto. |
| Terminal-Bench 4.0 (codificação agêntica em terminal) | Fora de escopo agora | Exige container com shell e loop de agente. |
| FrontierCode 1.1, principal (codificação agêntica) | Reportado | Benchmark do fornecedor. |
| Chartography, sem ferramentas (raciocínio visual) | Fora de escopo | O Simplicio é só texto. |
| Tokens de saída por tarefa (gráfico do AA) | Reportado | Nosso equivalente é o item de tokens do nível 1. |

Preço de API não entra na comparação. O custo medido por nós é em CU e tempo de Colab.

### Referências reportadas

**Anthropic: card de divulgação do Claude Haiku 5.5** (Sonnet 5.5 aparece só como referência)

| Benchmark | Haiku 5.5 | Haiku 4.5 | GPT-6 Luna | Sonnet 5.5 |
|---|---|---|---|---|
| GDPval-AA v2.1 (Elo) | 1620 | 735 | 1437 | 1840 |
| AA-Briefcase v1.1 (Elo) | 1578 | 614 | 1336 | 1824 |
| OSWorld 2.1, offline | 72.4% | 15.7% | 48.9% | 83.9% |
| Humanity's Last Exam, sem ferramentas | 45.9% | 10.2% | — | 56.9% |
| Humanity's Last Exam, com ferramentas | 57.4% | 18.7% | — | 64.5% |
| Terminal-Bench 4.0 | 39.2% | 0.0% | 16.4% | 70.6% |
| FrontierCode 1.1, principal | 46.4% | — | 42.4% | 52.1% (xhigh) |
| Chartography, sem ferramentas | 46.4% | 6.4% | 29.1% | 61.6% |

**Artificial Analysis (AA): gráficos do Intelligence Index v4.3.2, AA-Briefcase Elo e tokens de saída por tarefa**

| Modelo | AA Intelligence Index | AA-Briefcase Elo | Tokens de saída por tarefa |
|---|---|---|---|
| **Qwen3.8 27B (xhigh)** | **34** | **1397** | 67k |
| Claude Haiku 5.5 max (com fallback) | 43 | 1578 | 162k |
| Claude Haiku 5.5 xhigh (com fallback) | 41 | 1533 | 89k |
| Claude Haiku 5.5 high (com fallback) | 38 | 1443 | 55k |
| Claude Haiku 5.5 medium (com fallback) | 34 | 1372 | 33k |
| Claude Haiku 5.5 low (com fallback) | 29 | 1112 | 17k |
| GPT-6 Luna max | 38 | 1336 | 50k |
| GPT-6 Luna xhigh | 35 | 1237 | 27k |
| GPT-6 Luna high | 33 | 1186 | 20k |
| GPT-6 Luna medium | 30 | 1099 | 11k |
| GPT-6 Luna low | 22 | 741 | 2k |
| Claude 4.5 Haiku | 17 | 614 | 18k |

O Qwen3.8 27B (xhigh) é o modelo base do Simplicio 27B. A linha dele é a referência de partida, reportada pelo AA. Nosso número de base vem só do nível 1.

## Critério de decisão (fixado antes da execução)

O Simplicio 27B só é declarado melhor que o base se, no harness de 40 tarefas e com o mesmo modo de thinking:

1. o pass rate do Simplicio ficar acima do pass rate do base, e os intervalos de Wilson 95% não se sobrepuserem; e
2. a taxa de `finish_reason` válido e a taxa de erros não piorarem.

Se os intervalos se sobrepuserem, o resultado é empate, e o relatório diz isso.

Nenhuma frase do tipo "supera o Haiku" entra no README sem uma tabela do nível 1 que a sustente. Os modelos Claude e GPT-6 Luna aparecem só como números reportados.

## Regras de publicação

- Número do nível 2 nunca aparece como resultado do Simplicio.
- Cada número do nível 1 cita o commit, o `prompt_sha256` e o summary do run, todos gerados pelo harness.
- Nenhuma publicação no Hugging Face e nenhuma distribuição saem desta ADR. Cada uma exige decisão própria.

## Consequências

- O README só pode trocar "o ganho do fine-tuning não está medido" depois que existir a tabela do nível 1 gerada sob esta ADR.
- A #27 executa o nível 1. A #33 (DPO e GRPO) é comparada pelo mesmo protocolo e pelo mesmo critério de decisão.
- Os benchmarks marcados como "fora de escopo agora" ficam documentados aqui com o motivo. Eles só são reabertos por issue própria, se o motivo mudar.
