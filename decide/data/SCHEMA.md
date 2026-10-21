# Formatos de dados do Simplicio-Decide-27B

Contrato da Squad B (dados, treino, dia alugado). A Squad A (kit, Modal, notebooks) lê e escreve estes formatos.
O código que valida tudo isto está em `decide/data/schema.py`; os testes de contrato são `tests/test_contract_67.py`.
Regra de evidência: só o que foi medido é fato; o resto leva **UNVERIFIED**.

## 1. Fluxo

```
fontes (HF, JSONL local, gerador opcional, traces do Simplicio-Loop)
   -> build_corpus.py    corpus/            (parquet, colunas da seção 3)
   -> decontaminate.py   corpus-limpo/      (mesmo formato; rejeitados + relatório)
   -> Modal (Squad A)    labels/<teacher>/  (parquet, colunas da seção 4)
   -> decide/train       combinador (seção 5) -> alvos -> treino -> checkpoint
   -> proxy_private.py   privado/           (fontes excluídas do treino) + validação G1
```

Todo conjunto de dados é **um arquivo parquet ou uma pasta de partes `*.parquet`** com as mesmas colunas, mais um
`manifest.json` ao lado das partes (contagens e sha256 por parte). Quem escreve grava a parte num nome temporário e
renomeia; quem retoma pula as partes que o manifest já lista.

## 2. Pedido (request)

É o esquema de entrada do pplx-decider v1.2 (Apache-2.0), sem imagens:

```json
{
  "state": "texto ou objeto JSON",
  "questions": {
    "q1": {"type": "choice", "instructions": "…", "criteria": {"A": "opção 1", "B": "opção 2"}},
    "q2": {"type": "noul",   "instructions": "…", "criteria": {"true": "sim", "false": "não"}},
    "q3": {"type": "score",  "instructions": "…", "criteria": ["nível 0", "nível 1", "nível 2"]}
  }
}
```

- `state` é obrigatório (pode ser `{}`). `questions` tem 1 ou mais perguntas; o id é uma string não vazia.
- `choice`: `criteria` é um objeto chave -> descrição, de 1 a 255 opções, na ordem em que aparecem no prompt. A chave é o
  nome devolvido na resposta; o **código** (A, B, …) é posicional e vem da lista de 255 códigos do checkpoint
  (`decision_config.json`), que pula alguns códigos de duas letras que não são um token só.
- `noul`: `criteria` é opcional (`true`/`false`). As duas opções são sempre, nesta ordem, `false` e `true`.
- `score`: `criteria` é uma lista de 2 ou mais níveis ordenados; as chaves são `"0"`, `"1"`, …
- As chaves de opção de uma pergunta `choice` aparecem no texto do prompt (`código: chave: descrição`).

## 3. Corpus (parquet)

| coluna | tipo | conteúdo |
|---|---|---|
| `example_id` | string | único no corpus; `<fonte>:<id nativo ou índice>` |
| `source` | string | chave da fonte em `decide/data/sources.yaml` (ou `loop`, `gen:<provedor>:<domínio>`) |
| `license` | string | licença da fonte no momento da construção (ex.: `cc-by-4.0`, `mit`, `owner`, `provider-terms`) |
| `split` | string | `train` ou `dev` (dev = validação tirada das mesmas fontes, por hash do `example_id`) |
| `lang` | string | `en`, `pt`, … ou `mul` |
| `request` | string | JSON canônico do pedido (seção 2) |
| `gold` | string | JSON `{question_id: chave de opção ou null}`; `null` = sem rótulo ouro (só professor) |

`gold` referencia a **chave** da opção como `options()` do pplx a devolve: a chave do `criteria` (choice), `"false"` ou
`"true"` (noul), `"0"`… (score). Não é o código posicional.

JSON canônico: `json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))`.

## 4. Rótulos dos professores (parquet)

Escritos pelo app Modal da Squad A, lidos pelo treino. Uma linha por (exemplo, pergunta, professor).

| coluna | tipo | conteúdo |
|---|---|---|
| `example_id` | string | o do corpus |
| `question_id` | string | o id da pergunta no `request` |
| `teacher` | string | nome estável do professor (ex.: `pplx-decider-v1.2`, `d3`) |
| `codes` | list<string> | códigos das opções válidas, na ordem **original** das opções do corpus |
| `probs` | list<float> | probabilidades na mesma ordem; soma 1 (tolerância 1e-3, o professor roda em bf16) |
| `temperature` | double | temperatura **já aplicada** às `probs` (`p = softmax(z / T)`) |

Com `temperature` guardada dá para recalibrar sem rodar o professor de novo: `p' ∝ p^(T / T')`.
Chave de junção com o corpus: `(example_id, question_id)`.

## 5. Combinador de alvos (JSON)

O alvo do aluno **não** é fixo em "média dos professores" (fase 0A: a média de pesos iguais não superou o melhor
membro). O JSON é produzido a montante (calibração dos combinadores) e lido por `decide/train/targets.py`:

```json
{
  "version": 1,
  "method": "single | mean | calibrated_mean | logpool",
  "teacher": "pplx-decider-v1.2",
  "teachers": {
    "pplx-decider-v1.2": {"weight": 0.6, "temperature": 1.38},
    "d3":                {"weight": 0.4, "temperature": 1.10}
  },
  "by_question_type": {
    "choice": {"teachers": {"pplx-decider-v1.2": {"weight": 0.7}, "d3": {"weight": 0.3}}},
    "noul":   {"method": "single", "teacher": "pplx-decider-v1.2"}
  }
}
```

- `single`: usa só `teacher`.
- `mean`: média das probabilidades, com `weight` (default 1 para todos).
- `calibrated_mean`: leva cada professor à `temperature` indicada (`p^(T_guardada / T)`) e faz a média ponderada.
- `logpool`: pool log-linear ponderado, `p ∝ Π p_i^{w_i}`, renormalizado.
- `by_question_type` sobrescreve o método e os pesos por tipo de pergunta (`choice`, `noul`, `score`).
- Sem combinador, o treino só aceita um único professor presente; com 2 ou mais ele recusa e lista os nomes.

O alvo final é `alpha · ouro + (1 − alpha) · combinado`; sem ouro, só o combinado.

## 6. Resultados por linha (entrada do privado caseiro)

Parquet com uma linha por (pedido, pergunta): `example_id` (aceita também `request_id`, `run_id` ou `id`),
`question_id`, `codes`, `probs`. As probabilidades estão na ordem original das opções. O formato exato que o
`run_requests` da Squad A grava ainda precisa ser confirmado por ela (**UNVERIFIED**); o leitor é tolerante nesses nomes.

## 7. Privado caseiro (proxy)

- Feito só de fontes marcadas `role: heldout` em `sources.yaml`; elas **nunca** entram no corpus de treino
  (`assert_disjoint` falha o build se houver interseção de fonte, de hash exato ou de n-grama).
- `manifest.json` do proxy: fontes, contagens por fonte/idioma/tipo, sha256 do arquivo, parâmetros.
- Pontuação: acurácia corrigida pelo acaso, `(acc − chance) / (1 − chance)`, média simples entre fontes, mais NLL e
  Brier. Não é a métrica do Jev (cada benchmark do Jev tem a sua); o G1 diz se a ordem é parecida.
- G1: Spearman ≥ 0,8 entre a pontuação do proxy e o Full do board para ≥ 6 modelos abertos
  (`validate_against_board`). Também se reporta o Spearman contra `new_domains` e `same_skills`.

## 8. Licenças e lista negra

- `sources.yaml` é negar por padrão. Uma fonte só treina com `decision: allow`, licença na lista permissiva
  (MIT, Apache-2.0, BSD, CC0, CC BY 3.0/4.0) e `evidence` preenchida (URL e data da conferência).
- `review` (CC BY-SA, licença sem metadado, conteúdo derivado de fonte restrita): só com `--include-review`, decisão do dono.
- Lista negra dura, sem exceção: BBH/BIG-bench (canary), GPQA, HLE, ANLI (CC BY-NC), RAGTruth e qualquer fonte cuja
  licença seja NC/ND. O corpus também recusa qualquer texto com o canary do BIG-bench.
- A suíte pública do Jev é dado de avaliação: nunca entra no treino; só serve de referência da descontaminação.

## 9. Descontaminação

Referências: as linhas da suíte pública (`selected-rows`, `added-rows`, `gsm8k-rows` do kit) e, quando existir, o proxy.
Um exemplo é rejeitado se (1) o pedido inteiro ou uma pergunta (estado + instruções + opções, normalizados) tem o mesmo
hash; (2) uma **linha** com ≥ 5 palavras é idêntica a uma linha de referência; ou (3) uma janela de 13 palavras
(dentro de uma linha) aparece na referência. Unidades que aparecem em mais de 20 pedidos de referência são tratadas
como modelo de texto (boilerplate) e ignoradas. Normalização: NFKC, caixa baixa, só palavras (`\w+`).
O relatório (`decontamination.json`) lista parâmetros, contagens por motivo e por fonte e o sha256 das entradas.
