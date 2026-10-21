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

**Quem lê só lê o que o manifest lista.** `schema.corpus_files`/`read_corpus` (e, por eles, a descontaminação, o treino
e o privado) devolvem exatamente as partes de `parts` (ou o `file` do privado), na ordem do manifest, cada uma conferida
pelo sha256; parte listada que falta, mudou ou sai da pasta é `ManifestError`. Um `*.parquet` ao lado que o manifest não
lista (uma rodada antiga com `--include-review`, shards que um cap menor deixou) **não é dado** e é ignorado.
**`read_corpus` nunca faz glob:** pasta sem manifest é build interrompido (ou pasta montada à mão) e dá `ManifestError`;
o que se pode passar sem manifest é um arquivo `*.parquet` que o chamador nomeia. `read_teacher` em si ainda lê por glob
(uma pasta onde o aluguel ou o Modal despejou as partes), mas a construção dos alvos exige o manifest do corpus, dos rótulos
e dos traços, e o treino o dos alvos (`require_manifest`); `schema.seal_directory(pasta, kind="labels")` (ou `"traces"`)
grava o manifest de uma pasta assim (`parts` com `name`, `rows`, `sha256`). `build_corpus` e `decontaminate` apagam o manifest ao começar,
escrevem outro ao terminar e apagam partes, marcadores e rejeitados que a rodada não lista (`removed_stale`):
**uma pasta `--out` por corpus**, e `--only` gera um corpus parcial.

**Quem escreve só apaga e só sobrescreve o que escreveu.** Os manifests de `build_corpus`, `decontaminate`, do privado,
de `build_targets` e de `seal_directory` levam o carimbo `"tool"` (`build_corpus`, `decontaminate`, `proxy_private`,
`build_targets`, `seal_directory`; os relatórios `decontamination.json` e `report.json` também): um `manifest.json` de outra
ferramenta, feito à mão ou ilegível não é apagado nem trocado, a rodada falha com `ManifestError` (`TargetError` em
`build_targets`) e a pasta fica como estava. `decontaminate` e `build_targets` recusam `--out` que seja um arquivo, que seja
uma entrada (corpus, rótulos, traços), que esteja dentro dela ou que a contenha (uma entrada de um arquivo só,
`--corpus x/c.parquet --out x`, também; `--out` é julgado por onde ele realmente está: symlink e `..` são resolvidos antes), e só
apagam o que as suas próprias marcações `<parte>.done.json` e o manifest da última rodada concluída nomeiam, mais os `.tmp`
de nomes que elas mesmas escrevem. O manifest só pode nomear arquivos simples da própria pasta (um nome com `/`, `..` ou
caminho absoluto é de manifest forjado: `build_targets` recusa antes de apagar qualquer coisa). Qualquer outro arquivo em
`--out` fica onde está (e, fora do manifest, nunca é lido): um arquivo com o nome de uma parte que a ferramenta não escreveu
(sem marcador e fora do manifest anterior) faz a rodada falhar em vez de ser sobrescrito.

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
  (`decision_config.json`), que pula alguns códigos de duas letras que não são um token só. O JSON canônico ordena as
  chaves como texto, então **as chaves têm que ordenar na ordem das opções**: letras `A`…`Z` até 26 opções e
  `option_000`…`option_254` (três dígitos) acima disso. `ingest.convert_source` recusa um adaptador cujas chaves não
  ordenam.
- Tipos como no `types.py` do pplx-decider: `state` e toda descrição são `Content` (texto, objeto ou lista), `instructions`
  é `Content` ou ausente, a descrição de uma opção `choice` pode ser `null`, `criteria` de `noul` é ausente ou um objeto só
  com `true`/`false`, os níveis de `score` são `Content`.
- `noul`: `criteria` é opcional (`true`/`false`). As duas opções são sempre, nesta ordem, `false` e `true`.
- `score`: `criteria` é uma lista de 2 ou mais níveis ordenados; as chaves são `"0"`, `"1"`, …
- As chaves de opção de uma pergunta `choice` aparecem no texto do prompt (`código: chave: descrição`).

## 3. Corpus (parquet)

| coluna | tipo | conteúdo |
|---|---|---|
| `example_id` | string | único no corpus (conferido por `ingest` em cada fonte e por `build_corpus` entre fontes e extras); `<fonte>:<id nativo ou índice>`. O `id:` do `sources.yaml` tem que ser uma coluna única: id nativo repetido para pedidos diferentes derruba o build |
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
  (`assert_disjoint` falha o build se houver interseção de fonte; com `--corpus`, se algum pedido do corpus for um
  vazamento pelas regras da seção 9). O build exige `--corpus-manifest` (fontes) e/ou `--corpus` (pedidos), ou
  `--skip-leak-check` dito de propósito; o manifest do privado registra o que foi checado (`sources_only`, `skipped`).
  Ordem usual: (1) corpus bruto; (2) privado com `--corpus-manifest` do bruto; (3) descontaminar o corpus contra a suíte
  **e** o privado; (4) rodar o build do privado de novo com `--corpus` = corpus limpo (determinístico, serve de portão).
  O `manifest.json` do `build_corpus` e o do `decontaminate` listam `sources` (fonte -> linhas); um manifest sem `sources`
  é recusado, porque a checagem de fontes passaria sem provar nada.
- `manifest.json` do proxy: fontes, contagens por fonte/idioma/tipo, sha256 do arquivo, parâmetros.
- Pontuação: acurácia corrigida pelo acaso, `(acc − chance) / (1 − chance)`, média simples entre fontes, mais NLL e
  Brier. Não é a métrica do Jev (cada benchmark do Jev tem a sua); o G1 diz se a ordem é parecida.
- G1: Spearman ≥ 0,8 entre a pontuação do proxy e o Full do board para ≥ 6 modelos abertos
  (`validate_against_board`). Também se reporta o Spearman contra `new_domains` e `same_skills`. Cada chave de
  `--scores` é um `engine` do board (o id canônico) ou um nome de exibição que pertence a um modelo só; o `engine` vence
  o nome. O mesmo modelo por duas chaves (engine e nome) é recusado (`duplicates`, `passed: false`), nome de dois modelos
  não casa (`ambiguous`) e o `n` conta modelos distintos.

## 8. Licenças e lista negra

- `sources.yaml` é negar por padrão. Uma fonte só treina com `decision: allow`, licença na lista permissiva
  (MIT, Apache-2.0, BSD, CC0, CC BY 3.0/4.0) e `evidence` preenchida (URL e data da conferência).
- `review` (CC BY-SA, licença sem metadado, conteúdo derivado de fonte restrita): só com `--include-review`, decisão do dono.
- Lista negra dura, sem exceção: BBH/BIG-bench (canary), GPQA, HLE, ANLI (CC BY-NC), RAGTruth, HellaSwag (contextos do
  WikiHow, CC BY-NC-SA, e do ActivityNet, apesar do MIT do card) e qualquer fonte cuja licença seja NC/ND. O corpus também
  recusa qualquer texto com o canary do BIG-bench, mesmo partido em linhas ou em textos diferentes.
- `revision` (sha de 40 hex do commit em que a licença e os campos foram conferidos) é obrigatório para fonte do Hub: o leitor
  baixa só nesse commit e recusa fonte sem pin antes de qualquer chamada de rede (`--reader hub-latest` explora uma fonte
  nova). Repositórios sem parquet na main leem a branch `refs/convert/parquet`, e `convert_revision` fixa esse commit.
  Se o sha sumir do Hub o build falha, não cai para a main: refaça a conferência de licença e atualize os dois campos.
- `fields_verified` é a palavra do YAML; a prova é `tests/fixtures/decide_data/columns.json`, as colunas medidas em cada
  `revision` por `python -m decide.data.build_corpus --dump-columns <arquivo> --reader hub`. Um teste offline confere que todo
  campo citado em `args` existe nessa lista e que os pins do fixture e do YAML são os mesmos: ao subir uma `revision`, rode o
  comando de novo e releia o diff.
- A suíte pública do Jev é dado de avaliação: nunca entra no treino; só serve de referência da descontaminação.

### 8.1 Gerador de domínios novos (opcional)

Desligado por padrão, com duas chaves: `data.generator.enabled: true` em `decide/config.data.yaml` **e** `--generate` em
`python -m decide.data.build_corpus`. O provedor (`command`: programa que lê o prompt no stdin; `openai-compatible`:
`https://` ou `http://localhost`, chave na variável de ambiente `api_key_env`) e as tarefas
(`{domain, lang, qtype, description}`) vêm do config. As linhas saem com `source = gen:<provedor>:<domínio>` e licença
`provider-terms` (valem os termos do provedor escolhido), passam pela mesma validação, por `example_id` único e pela
descontaminação; ficam em `<out>/generated.json` para uma rodada retomada não pagar o provedor de novo.

## 9. Descontaminação

Referências: as linhas da suíte pública (`selected-rows`, `added-rows`, `gsm8k-rows` do kit; linhas com `state`/`questions`
no topo ou com um objeto `request`) e, quando existir, o proxy. Os pedidos são canonicalizados (chaves ordenadas em todos os
níveis) antes de indexar e de checar, então a ordem das chaves em qualquer arquivo não importa. Um exemplo é rejeitado se
(1) o pedido inteiro ou uma pergunta (estado + instruções + opções, normalizados) tem o mesmo hash (`exact_request`,
`exact_question`); (2) uma **linha** com ≥ 5 palavras é idêntica a uma linha de referência (`line`); (3) uma linha **curta**
(menos de 5 palavras, com pelo menos uma letra) do estado ou de uma instrução é idêntica a uma linha curta do estado ou de
uma instrução da referência (`short_line`: o item de teste pode ser só "thanks so much!"; opções não contam); ou (4) uma
janela de 13 palavras (dentro de uma linha) aparece na referência (`ngram`). Unidades que aparecem em mais de 20 pedidos
de referência são tratadas como modelo de texto (boilerplate) e ignoradas; um mesmo estado vale 1 não importa quantas
perguntas o compartilhem, então uma passagem com 30 perguntas de referência continua protegida. Normalização: NFKC, caixa
baixa, só palavras (`\w+`). O canary é procurado no texto das folhas unidas por quebra de linha (`canary`).

**Linhas de marcação de formato.** Uma linha que só carrega formato não é um enunciado curto: o `>>>> REPLACE` de um pedido de
edição de código não diz nada sobre o item, e rejeitava 213 de 320 traços do loop. A lista é explícita e versionada
(`text.MARKER_LINES`, versão 1; a versão entra na impressão digital da descontaminação e no relatório, `markers_version`).
Vale só para a regra `short_line`: a linha inteira, sem espaços nas pontas, tem de casar (caixa não importa), e nem o
índice nem a checagem a contam como unidade. Qualquer outra linha curta, inclusive código (`@classmethod`, `Ok(())`,
`SELECT`), continua valendo como enunciado curto. Mudar a lista exige subir a versão e uma entrada nesta tabela.

| exemplo | casa com | por quê |
|---|---|---|
| `<<<<<<< SEARCH` | três ou mais `<` e `SEARCH` | abre um bloco de edição SEARCH/REPLACE |
| `=======` | três ou mais `=` | separa as duas metades do bloco (sem letras, nunca contou; listada por completude) |
| `>>>>>>> REPLACE` | três ou mais `>` e `REPLACE` | fecha o bloco; os traços do loop escrevem com quatro `>` |
| `` ```python `` | três ou mais crases ou til, com etiqueta de linguagem opcional | cerca de código |

Uma referência que não rende nenhum pedido, ou cujas linhas não lidas passam de 1 %, derruba a rodada (`ReferenceDataError`):
a triagem contra uma referência não lida deixaria tudo passar. O relatório (`decontamination.json`) lista parâmetros,
contagens por motivo e por fonte, linhas lidas/puladas por referência e o sha256 das entradas.
Limite conhecido: um trecho longo repetido nas **instruções** (não no estado) de mais de 20 pedidos de referência é tratado
como modelo de texto.

## 10. Bloco de raciocínio (opcional)

Plano v2 ([ADR 0004](../../docs/adr/0004-pensar-sob-demanda.md)): um modelo pensador (Qwen3.8-27B com thinking) escreve
um raciocínio e o decisor lê **estado + raciocínio**, ainda numa passada só. O bloco é opcional; sem ele, tudo funciona
como antes.

**No pedido.** Chave opcional de topo `reasoning`: uma string, não vazia depois de `strip`, com o texto do raciocínio sem
as marcas `<think>` e `</think>`. Vale para o pedido inteiro (todas as perguntas), não para uma pergunta só.

```json
{"state": "…", "questions": {"q1": {"type": "choice", "criteria": {"A": "…", "B": "…"}}}, "reasoning": "O estado diz … logo B."}
```

**No prompt.** O raciocínio entra **dobrado no estado**, depois do estado e antes da primeira pergunta. As partes continuam
separadas por uma linha em branco:

```
State:
<estado>

Raciocínio:
<raciocínio>

Question 0:
…
```

O rótulo é exatamente `Raciocínio:` (com acento) e faz parte do contrato entre o treino e a inferência. O texto passa
pela mesma neutralização de tokens especiais do estado. Um pedido sem `reasoning` gera o prompt do v1.2 sem mudar um byte.

**Uma só definição: `decide.train.protocol.fold_reasoning(pedido, max_chars=8000)`.** Devolve o pedido sem a chave
`reasoning` e com o estado dobrado, `describe(estado).rstrip() + "\n\nRaciocínio:\n" + bloco` (estado vazio: só o rótulo e o
bloco). O `bloco` é o raciocínio depois de:

1. fins de linha CRLF e CR viram LF;
2. espaços e tabs no fim de cada linha saem (`rstrip(" \t")`);
3. uma linha que seja o próprio rótulo (Unicode NFC, espaços nas pontas ignorados) vira `Raciocínio :`, mantendo o recuo,
   para que o estado dobrado tenha uma única linha de rótulo, a que abre o bloco;
4. três ou mais linhas em branco seguidas viram uma só, e as quebras de linha das pontas saem;
5. corte em **caracteres**: no máximo 8000 (`REASONING_MAX_CHARS`). Acima disso fica o **fim** do texto (é lá que está a
   conclusão), a partir de uma quebra de linha quando há uma, atrás de uma linha `[... N caracteres omitidos ...]`.
   Um texto que já cabe (um já cortado, por exemplo) não é cortado de novo.

Um estado que já tem uma linha de rótulo é recusado (`RequestError`), nunca sobrescrito; `build_targets` não anexa o traço a
uma linha assim (`reasoning_state_has_block` no relatório) e o treino e a avaliação a usam sem raciocínio. `messages()` monta o
prompt a partir do pedido dobrado, então as duas formas geram o mesmo prompt **por construção**. O lado do pensador
(`decide/think/trace.py`) implementa a mesma função; o contrato entre os dois lados é o arquivo dourado
`tests/fold_reasoning_cases.json` (10 casos: estado com espaço no fim, rótulo dentro do raciocínio, CRLF, três ou mais
linhas em branco, estado vazio, estado como objeto, corte), que os dois lados têm de reproduzir. **Isto tem de estar resolvido
antes de qualquer treino com `--reasoning`:** o `Encoder` do pplx-decider v1.2 ignora a chave `reasoning` em silêncio, então
um motor que só conhece `state` (o `api.run_requests` do kit) chama `fold_reasoning` antes de mandar o pedido, ou recusa a
chave. O pedido também recusa `images` (`RequestError`): o protocolo é só texto e o v1.2 leria o pedido como texto sem avisar.

**Forma dobrada.** Para um motor que só conhece `state`, `fold_reasoning(pedido)` é o pedido a mandar. O corpus (seção 3)
nunca guarda `reasoning` no `request`: o raciocínio fica nos traços abaixo e entra no pedido só na hora do treino ou da
decisão.

**Traços do pensador (parquet).** O passo "gerar traços do pensador" do dia alugado grava uma pasta de partes
`traces/<pensador>/part-NNNN.parquet` (mais `manifest.json`, como na seção 1). Uma linha por exemplo e por pensador:

| coluna | tipo | conteúdo |
|---|---|---|
| `example_id` | string | o do corpus |
| `thinker` | string | nome estável do pensador (ex.: `Qwen3.8-27B`) |
| `reasoning` | string | texto do raciocínio, não vazio; quem gera não grava linha sem raciocínio |
| `tokens` | int32 | tokens de pensamento gerados (≥ 0) |
| `finish_reason` | string | `stop` (fechou o pensamento dentro do orçamento) ou `length` (o orçamento cortou) |
| `request_sha256` | string | `schema.request_digest` do `request` do corpus para o qual o raciocínio foi escrito (64 hex minúsculos) |

Um raciocínio cortado (`length`) é mantido: na hora de decidir ele é cortado pelo mesmo orçamento (item 5 acima).
`decide.data.traces` valida, escreve e lê esse formato; `decide.train.build_targets --traces` junta pelo `example_id` e
**recusa** o traço cujo `request_sha256` não é o do pedido do corpus (`TargetError` com o `example_id`): um raciocínio nunca é
anexado a um pedido para o qual não foi escrito. A pasta de traços tem `manifest.json` (`seal_directory(pasta, kind="traces")`).
Limite conhecido (ADR 0004, item 6): a descontaminação por n-grama roda sobre o `request` do corpus, não sobre a coluna
`reasoning`; os traços de exemplos que a descontaminação rejeitou não entram porque a junção parte do corpus limpo, mas um
trecho de suíte pública que o pensador cite no próprio raciocínio não é filtrado. Fica como pendência (issue de acompanhamento).

## 11. Dataset de alvos (saída de `decide.train.build_targets`)

É a única entrada de dados do treino. Pasta de partes `*.parquet` (uma por parte do corpus, mesmo nome), mais
`manifest.json` e `report.json` (contagens). O manifest é como o do corpus (seção 1): `kind: "targets"`, `columns`, `rows` e
`parts`, uma lista de `{name, rows, sha256, fingerprint, stats}` (a impressão digital das entradas da parte, para retomar).
Cada parte pronta deixa um marcador `<parte>.done.json` com a mesma entrada. O trainer lê **só** as partes do manifest,
com o sha256 conferido (`ManifestError` se uma mudou ou sumiu). `build_targets` apaga o manifest e o relatório ao
começar e escreve os novos ao terminar, apaga partes e marcadores que a rodada não lista (`removed_stale`) e exige que
`--corpus` seja a saída da descontaminação: **uma pasta `--out` por conjunto de alvos**.

**De onde vêm as linhas.** O `manifest.json` do corpus tem de dizer `tool: "decontaminate"` e nomear `decontamination.json`,
que leva o mesmo carimbo. Um corpus bruto (`build_corpus`), uma pasta selada à mão (`seal_directory`), um parquet solto e um
carimbo sem o relatório são recusados (`TargetError` que diz o que fazer): o manifest prova que a lista de partes está
inteira, não que as linhas foram triadas contra as suítes públicas e os privados. A saída explícita é
`--allow-undecontaminated`. De um jeito ou de outro o manifest e o `report.json` dos alvos levam
`corpus: {tool, manifest_sha256, decontamination_sha256, allow_undecontaminated}` (`null` no que não existe;
`allow_undecontaminated` é o que a rodada recebeu), a proveniência entra na impressão digital de cada parte (mudá-la refaz as
partes), e o treino a copia para `training.corpus` do `decision_config.json` do checkpoint e para o evento `training_started`.
Rótulos e traços também são lidos pelo manifest da pasta.

| coluna | tipo | conteúdo |
|---|---|---|
| `example_id` | string | o do corpus |
| `source` | string | a do corpus |
| `split` | string | `train` ou `dev` |
| `lang` | string | o do corpus |
| `request` | string | JSON canônico do pedido (seção 2), sem `reasoning` |
| `gold` | string | JSON `{question_id: chave de opção ou null}`, como no corpus |
| `combined` | string | JSON `{question_id: [prob, …] ou null}`: a distribuição dos professores já combinada (seção 5), normalizada, na ordem original das opções; `null` = nenhum professor rotulou a pergunta |
| `teachers` | string | JSON lista dos professores que entraram no exemplo |
| `reasoning` | string | texto do raciocínio (seção 10) ou `""` |

Uma linha só sai se alguma pergunta tem alvo (ouro ou professores). Uma pergunta sem alvo dentro de uma linha que sai
fica com peso 0 no treino. Política de professor faltando (`--missing`): `drop` (padrão: a pergunta fica só com o ouro),
`renormalize` (combina os que existem) ou `error` (recusa também a pergunta que nenhum professor rotulou, nomeando-a).

O `report.json` conta `questions` (perguntas das linhas lidas) e `unlabeled_questions` (as que ficaram sem distribuição de
professor, seja porque ninguém as rotulou, seja porque `drop` descartou uma combinação incompleta): a conta do gate F3
("pelo menos 95 % rotulado") é `1 - unlabeled_questions / questions`. Os demais contadores: `dropped_unlabeled` (linhas sem
nenhum alvo), `reasoning_rows`, `reasoning_truncated`, `reasoning_state_has_block`, `traces_unmatched`, `missing_teacher`.
