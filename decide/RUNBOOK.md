# RUNBOOK do Simplicio-Decide-27B: sessão por sessão

Para o dono (Wesley). Versão 2 · 2026-10-21 (revalidada contra a `main` com #73, #74, #77, #78, #79, #80, #81 e #82; os avisos, a tabela e o que a sessão 8 e a seção 13 dizem do raciocínio (fold, `--gt-report`) foram revistos contra a `main` em `c32c417`, com #86, #87, #88, #91, #94, #95 e #96; veja os avisos logo abaixo). O plano está em [`PLAN.md`](PLAN.md), os formatos em [`data/SCHEMA.md`](data/SCHEMA.md), as decisões em [ADR 0003](../docs/adr/0003-simplicio-decide-27b.md) e [ADR 0004](../docs/adr/0004-pensar-sob-demanda.md). Épico [#64](https://github.com/simpletibr/simplicio-27b/issues/64).

Regra de evidência: só número medido é fato. Tudo que está marcado **UNVERIFIED** é estimativa, conta ou leitura de documentação. **Nada deste projeto rodou numa GPU.** O código foi testado em CPU, com modelos minúsculos e dublês (`make check`). A primeira vez que cada sessão rodar de verdade é um teste: comece pelo ensaio pequeno que cada uma descreve.

## Avisos de hoje (leia antes de qualquer sessão)

Dois avisos. O segundo é o resultado de conferir cada comando abaixo contra o código que está na `main` agora (`c32c417`). O aviso sobre o núcleo do kit **saiu**: os 3 bloqueantes que ele citava (falta de memória por lote, revisão do Hub sem fixar e doador sem cobertura, da revisão independente do PR [#80](https://github.com/simpletibr/simplicio-27b/pull/80)) foram corrigidos no PR [#86](https://github.com/simpletibr/simplicio-27b/pull/86), que está na `main`. Isso não quer dizer que o kit rodou numa GPU: **nunca rodou, e nenhum modelo de tamanho real passou por ele (UNVERIFIED).**

1. **O gate GT oficial só vale se sair do notebook do Kaggle** (`notebooks/decide/think_gt_kaggle.ipynb`, backend `transformers`). Os PRs [#87](https://github.com/simpletibr/simplicio-27b/pull/87) e [#90](https://github.com/simpletibr/simplicio-27b/pull/90) estão na `main`: uma linha que vem só com a resposta (pelo menos 100 tokens e nenhum raciocínio visível, `reasoning_hidden`) não vira mais `FAIL` oficial; o veredito sai `EXPLORATORY`, e `INCONCLUSIVE` se as linhas que esconderam o raciocínio ou pararam sem fechar o pensamento (no código, `reasoning_hidden + no_close`) passarem de 10% das linhas que responderam (`decide/think/compare.py`). **Mas o gate hospedado (endpoint de API, `decide.think.hosted`, Modal) não tem evidência operacional:** os limites de 100 tokens e de 10% foram escolhidos, não medidos num provedor real (o próprio #87 diz isso), e nenhuma execução contra um provedor real foi registrada. Até existir uma, esses caminhos servem para a sonda de 3 linhas, **não para o veredito** (sessão 3).
2. **O que a `main` roda hoje.** Conferi as opções de cada comando com `--help` e a existência de cada caminho; nada disso rodou numa GPU.

   | Sessão | Roda a partir da `main`? | O que falta |
   |---|---|---|
   | 0 preparação | sim | |
   | F0 kit (`decide/sd`, [#86](https://github.com/simpletibr/simplicio-27b/pull/86)) | o código está na `main`, com os 3 bloqueantes do #80 corrigidos (OOM por pedido, revisão do Hub de 40 hex, doador com cobertura) e `dir:CAMINHO` para uma pasta desta máquina; os testes o exercitam numa CPU com modelos minúsculos; o fold do raciocínio na avaliação ([#95](https://github.com/simpletibr/simplicio-27b/pull/95)) é opt-in e não tem flag de linha de comando (seção 13) | nenhum modelo de tamanho real passou por ele e nada rodou numa GPU (UNVERIFIED); os notebooks `kaggle_0N_*.ipynb`, `decide/modal` e `notebooks/decide/docs/` (parte 2, Squad A) não estão na `main` |
   | 1 F2 dados | passos 1 a 3, 5 e 6 (`decide.data`) e, do passo 7, `proxy_eval requests` e `score`; o `proxy_eval run` (F6, [#91](https://github.com/simpletibr/simplicio-27b/pull/91)) está na `main`, mas nunca carregou um modelo de tamanho real. Não o passo 4 | `decide.modal.suite_requests` (Squad A) não existe na `main`; o `run` do G1 carrega modelos pelo kit, que nunca carregou um modelo de tamanho real (UNVERIFIED) |
   | 2 F1 fidelidade | não | os notebooks `kaggle_0N_*.ipynb` e `notebooks/decide/docs/` (parte 2 do kit, Squad A) não estão na `main` |
   | 3 F2b GT | sim, mas o veredito só pelo Kaggle | aviso 1 |
   | 4 F3 rotulagem | não | `decide/modal` (Squad A) não existe na `main` |
   | 5 alvos | sim | os traços do pensador só nascem no passo `traces` do dia alugado (F6, na `main`), e hoje o dia fica só-decisor (sessão 8); veja a sessão 5 |
   | 6 F4 ensaio 4B ([#79](https://github.com/simpletibr/simplicio-27b/pull/79), [#88](https://github.com/simpletibr/simplicio-27b/pull/88); o #94 só acrescenta um teste) | passos 1 a 3 (treino, `decide.train`), só ensaiados em CPU com um modelo minúsculo; o passo 4 (G2): o `proxy_eval run` está na `main`, mas nunca carregou um modelo de tamanho real | o treino nunca rodou numa GPU e o G2 nunca foi medido (UNVERIFIED) |
   | 7 F5 merge | não | os notebooks do kit (parte 2, Squad A) não estão na `main` |
   | 8 F6 dia alugado ([#91](https://github.com/simpletibr/simplicio-27b/pull/91)) | o plano (sem `--execute`) e o ensaio em CPU, com um modelo minúsculo, uma pasta no lugar do Hub e o `decide.sd` de verdade até o G4; o dia de verdade não | nenhuma fase rodou numa GPU nem contra o Hugging Face de verdade; as fases `g4` e `public_round` carregam o modelo pelo kit, que nunca carregou um modelo de tamanho real (UNVERIFIED) |
   | 9 F7 submissão | à mão | depende da sessão 8 |

   Onde um comando abaixo depende de um caminho que ainda não existe, o texto diz.

## 0. Como ler

- Cada sessão diz **onde roda**, **o que precisa antes**, **os comandos**, **o que escreve**, **como saber que deu certo**, **se cair** e **o que vem depois**.
- Os comandos partem da raiz do repositório clonado. `work/` é uma pasta local qualquer (fora do git); troque os nomes à vontade, mas use os mesmos entre sessões.
- **Segredos.** O token do Hugging Face (`HF_TOKEN`) e qualquer chave de API entram só como variável de ambiente ou em *Secrets* da plataforma. Nunca numa célula, num arquivo do repositório, numa mensagem de commit. Nenhum comando abaixo imprime o valor.
- **Tudo retoma.** Todo passo longo grava em partes e pula o que já terminou. Se a sessão cair, abra a mesma e rode o mesmo comando. A seção 12 lista o que fazer em cada falha.
- **Tudo nasce privado.** Os repositórios do Hugging Face que o projeto escreve são privados. Tornar algo público é decisão sua, depois de ler o cartão (sessão 9).
- Quem escreveu o quê: `decide/data` (F2), `decide/train` (F4) e `decide/rental` (F6) são desta squad; `decide/sd`, `decide/modal` e `notebooks/decide/` (kit e Modal) são da Squad A; `decide/think` e os notebooks `think_gt_*` (pensar) são da Squad C. **Já estão na `main`:** `decide/data` (#74), `decide/think` (#78), `decide/train` (#79), o núcleo do kit `decide/sd` (#80) e `decide/rental` com `decide/config.rental.yaml` (F6, #71: o `proxy_eval` das sessões 1 e 6 e o `run` da sessão 8). **Ainda não estão:** `decide/modal`, os notebooks `kaggle_0N_*.ipynb` e `notebooks/decide/docs/` (parte 2 do kit, Squad A). Um caminho dessas listas só vale depois do merge da PR que o traz.

## 1. Mapa

| # | Fase (issue) | Onde | Depende de | Escreve | Gate (o que decide seguir) |
|---|---|---|---|---|---|
| 0 | preparação | seu computador | nada | contas, tokens, repositórios privados | `make check` verde |
| 1 | F2 dados ([#67](https://github.com/simpletibr/simplicio-27b/issues/67)) | CPU: VPS ou Kaggle sem GPU | 0 | `work/corpus-limpo`, `work/proxy` | G1: o privado caseiro ordena ≥ 6 modelos abertos como o board (Spearman ≥ 0,8) |
| 2 | F1 fidelidade ([#66](https://github.com/simpletibr/simplicio-27b/issues/66)) | Kaggle 1 a 2 T4 | 0 | `state.json` no dataset de estado | G0: argmax igual ≥ 98% (4B) e ≥ 95% (27B 4-bit) em ≥ 500 linhas |
| 3 | F2b pensar ([#75](https://github.com/simpletibr/simplicio-27b/issues/75)) | Kaggle T4, API, GPU de 80 GB ou Modal | 0 | `report.json` do GT | GT: pensador ≥ +10 pontos sobre o melhor decisor, IC 95% acima de 0 |
| 4 | F3 rotulagem ([#68](https://github.com/simpletibr/simplicio-27b/issues/68)) | Modal | 1 | `labels/` no dataset de estado | ≥ 95% do corpus rotulado |
| 5 | alvos | CPU | 1, 4 (e 3 se passou) | `work/targets` | contagem de linhas bate com o corpus |
| 6 | F4 ensaio 4B ([#69](https://github.com/simpletibr/simplicio-27b/issues/69)) | Kaggle T4 (ou crédito da Modal) | 1, 4, 5 | `runs/4b-*` | G2: ≥ +1,0 sobre o pplx-flash no privado caseiro |
| 7 | F5 merge, opcional ([#70](https://github.com/simpletibr/simplicio-27b/issues/70)) | Kaggle 2 T4 | 2 | `eval:*`, `compare:*` | G3: melhor merge ≥ pplx v1.2 |
| 8 | F6 dia alugado ([#71](https://github.com/simpletibr/simplicio-27b/issues/71)) | GPU alugada | 1, 4, 5 (3 se passou) | `runs/`, `soup/`, `final/`, `g4/`, `public/` | G4: ≥ pplx v1.2 + 1,5 no privado caseiro |
| 9 | F7 submissão ([#72](https://github.com/simpletibr/simplicio-27b/issues/72)) | HF e Space | 8 | pesos, resultados, cartão, PR | G5: concordância do lab ≥ 98,5% |

Ordem sugerida: 0, depois 1 e 2 e 3 em paralelo (são independentes), depois 4, 5, 6, 7, 8, 9. A sessão 7 não bloqueia nada: se nenhum merge ganhar, o ponto de partida é o `pplx-decider-v1.2` puro.

**Nomes de gate.** O plano chama o ensaio 4B de G2 e o merge de G3. O kit (`notebooks/decide/docs/02_gates_e_evidencia.md`) chama o merge de `G2-merge` e a escolha do ponto de partida de `G3-start`. São o mesmo experimento; ao ler os relatórios do kit, troque os números de cabeça.

## 2. Sessão 0: preparação (uma vez)

**Onde:** seu computador. **Tempo:** não medido.

1. Contas, uma por plataforma (somar cota com várias contas fica fora do plano e dos termos das plataformas):
   - Kaggle com o telefone verificado (GPU e Internet dependem disso, segundo a documentação do Kaggle, UNVERIFIED);
   - Hugging Face, com um token do tipo **Write** (*Settings → Access Tokens*);
   - Modal, com o crédito mensal gratuito (US$ 30/mês no plano; confira a página de preços, UNVERIFIED);
   - um provedor de GPU alugada para a sessão 8 (o orquestrador roda dentro da máquina alugada e não chama a API do provedor).
2. Baixe o repositório numa revisão fixa e confira que o código está bom no seu computador:

   ```
   git ls-remote https://github.com/simpletibr/simplicio-27b refs/heads/main     # o primeiro campo é o COMMIT de 40 caracteres
   git init simplicio-27b && cd simplicio-27b
   git fetch --depth 1 https://github.com/simpletibr/simplicio-27b COMMIT
   git checkout --detach FETCH_HEAD
   make check
   ```

   (Em vez de `clone` direto: uma regra de teste do repositório exige que todo download nomeie um commit de 40 caracteres, e este RUNBOOK segue a mesma regra. Use o COMMIT da `main` do dia, ou o que o planejador indicar.)

   `make check` roda os testes, o `ruff`, o `shellcheck` e `php -l`. Ele não precisa de GPU. A verificação `scripts/check_vllm_args.py` precisa do vLLM instalado e pode ser pulada com `SIMPLICIO_SKIP_VLLM_ARGS=1`.
3. Crie no Hugging Face (todos **privados**):

   | Repositório | Tipo | Quem cria | Para quê |
   |---|---|---|---|
   | `wesleysimplicio/simplicio-decide-state` | dataset | o kit, na primeira sessão | estado das etapas, amostra, resultados por linha, rótulos |
   | `wesleysimplicio/simplicio-decide-data` | dataset | você | corpus limpo, rótulos e privado caseiro para o dia alugado |
   | `wesleysimplicio/simplicio-decide-day` | modelo | o orquestrador | cópia de cada seed e da média, para retomar o dia |
   | `wesleysimplicio/Simplicio-Decide-27B` | modelo | o orquestrador ou o notebook 6 do kit | o modelo final |

   Os nomes vêm de `decide/config.yaml` e de `decide/config.rental.yaml` (da F6); mude lá, não aqui.
4. Guarde o token onde cada plataforma o guarda: *Add-ons → Secrets* no Kaggle (nome `HF_TOKEN`, marcado como anexado ao notebook), *Secrets* no Colab, `modal secret create huggingface HF_TOKEN=...` na Modal, `export HF_TOKEN=...` no terminal da máquina alugada.

**Certo quando:** `make check` termina com `OK`.

## 3. Sessão 1: F2, corpus, descontaminação e privado caseiro

**Onde:** CPU com 32 GB de RAM e Internet (uma VPS ou o Kaggle sem acelerador). **Precisa:** `HF_TOKEN` exportado, a suíte pública 0.3 baixada pelo kit (veja o passo 3) e os traços do Simplicio-Loop que já estão em `data/`. **Pacotes:** `pip install -r decide/data/requirements.txt`

Formatos e regras: [`data/SCHEMA.md`](data/SCHEMA.md). A ordem importa, porque o privado e o corpus não podem compartilhar fonte nem pedido:

1. **Ensaio pequeno** (minutos): confira que as fontes convertem.

   ```
   python -m decide.data.build_corpus --out work/corpus-fumaca --reader hub --scale 0.01 --only clinc_oos,gsm8k
   python -m decide.data.build_corpus --inspect clinc_oos --reader hub-latest
   ```

2. **Corpus bruto.** Acrescente os traços do Simplicio-Loop com `--loop` (arquivo repetível). Os domínios novos gerados por um provedor de LLM ficam **desligados**; só ligue (`data.generator.enabled: true` em `decide/config.data.yaml` **e** `--generate`) se aceitar os termos do provedor escolhido sobre o texto gerado.

   ```
   python -m decide.data.build_corpus --out work/corpus --reader hub \
       --loop data/simplicio_loop_50pts_train.jsonl --loop data/v2/train.jsonl
   ```

3. **Privado caseiro, primeira passada** (só fontes marcadas `role: heldout`, que nunca treinam):

   ```
   python -m decide.data.proxy_private build --out work/proxy --reader hub --corpus-manifest work/corpus/manifest.json
   ```

4. **A suíte pública 0.3**, que a descontaminação usa como referência. Quem a baixa é o kit: o notebook 1 (`kaggle_01_preparar.ipynb`) ou, na sua máquina, `python -m decide.modal.suite_requests --out requests/`, que deixa a suíte em `requests/suite-0.3/` (precisa de `HF_TOKEN`; cerca de 75 MB). **Nenhum dos dois está na `main` hoje** (parte 2 do kit, Squad A): este passo, e os passos 5 e 6 que dependem dele, esperam o merge dessa PR. Os arquivos de linhas (`selected-rows.jsonl.gz`, `added-rows.jsonl.gz`, `gsm8k-rows.jsonl.gz`) são a referência.
5. **Descontaminação** contra a suíte e o privado:

   ```
   python -m decide.data.decontaminate --corpus work/corpus --out work/corpus-limpo \
       --reference requests/suite-0.3 --reference work/proxy
   ```

6. **Privado caseiro, segunda passada**, agora com o corpus limpo. É determinístico e serve de portão: se algum pedido do corpus vazar um do privado, o comando falha.

   ```
   python -m decide.data.proxy_private build --out work/proxy --reader hub \
       --corpus-manifest work/corpus/manifest.json --corpus work/corpus-limpo
   ```

7. **G1.** Rode o mesmo privado em pelo menos 6 modelos abertos que têm nota Full no board e compare as ordens. O ensaio de cada modelo usa os comandos de `proxy_eval` (seção 8, passo 4): `requests` uma vez, `run` por modelo (a pasta `--out` de cada um), depois `score` para todos, que escreve o `scores.json` do tipo `{"modelo": habilidade}`. **O nome de cada modelo tem de ser o `engine` do board** (o id canônico), senão a validação não o reconhece.

   ```
   python -m decide.rental.proxy_eval requests --proxy work/proxy --out work/proxy-requests.jsonl
   python -m decide.rental.proxy_eval run --requests work/proxy-requests.jsonl --model ENGINE=hub:ORG/REPO@COMMIT --out work/g1/ENGINE
   python -m decide.rental.proxy_eval score --proxy work/proxy --results ENGINE=work/g1/ENGINE OUTRO=work/g1/OUTRO --out work/g1/scores.json
   python -m decide.data.proxy_private validate --scores work/g1/scores.json --board work/space/data/index.json
   ```

   **O `proxy_eval` (F6) está na `main`, mas o `run` nunca carregou um modelo do tamanho real: os testes usam um substituto do kit (UNVERIFIED).** O `run` carrega os modelos pelo kit (o #86, com os bloqueantes do #80 corrigidos, está na `main`; nenhum modelo de tamanho real passou por ele, UNVERIFIED); o 27B em 4 bits precisa de `--quant nf4` e de duas T4. Os modelos que o kit carrega são os de layout pplx (`pplx-decider-v1.2`, `pplx-decider-flash-v1.2`, `d3`, `d3-turbo`, `d3-mini`); o Amos tem cabeça incompatível e o Lychee é CC BY-NC (fora do projeto). **Se faltar modelo carregável para chegar a 6, o G1 não fecha**: diga isso na issue, não invente a ordem. (`work/space/data/index.json` é o `index.json` do board; baixe-o do Space do Jev. Revisão exata de cada modelo: copie o commit da página do modelo no Hub.)

**Escreve:** `work/corpus-limpo/` (partes `*.parquet`, `manifest.json`, `decontamination.json`), `work/proxy/` (`manifest.json`). **Certo quando:** o relatório da descontaminação lista as contagens por motivo, a segunda passada do privado termina sem `LeakError` e o G1 diz `passed: true`.

**Se cair:** rode o mesmo comando. Cada fonte e cada parte é marcada ao terminar. Use **uma pasta `--out` por corpus**. Uma pasta que outra ferramenta escreveu nunca é apagada: o comando falha e a deixa como estava.

**Antes de sair (sessão 4 e 8 leem daqui):** suba para o repositório de dados (privado):

```
hf upload wesleysimplicio/simplicio-decide-data work/corpus-limpo corpus-limpo --repo-type dataset
hf upload wesleysimplicio/simplicio-decide-data work/proxy proxy --repo-type dataset
```

(O comando `hf` vem do `huggingface_hub`; confira a sintaxe com `hf upload --help`, UNVERIFIED.) Suba só a pasta pronta: o leitor do Modal lê **todo** `*.parquet` da pasta, não só os que o `manifest.json` lista.

## 4. Sessão 2: F1, fidelidade do carregador (Kaggle)

**Não rode agora** (aviso 2): os notebooks e os guias abaixo são da parte 2 do kit e não estão na `main`.

**Onde:** Kaggle, 1 a 2 T4. **Passo a passo completo:** `notebooks/decide/docs/01_comecar_no_kaggle.md` (kit). Em resumo:

1. Importe os sete notebooks `kaggle_0N_*.ipynb`, anexe o segredo `HF_TOKEN` e ligue a Internet.
2. Rode `kaggle_01_preparar` (CPU, sorteia a amostra da suíte), depois `kaggle_02_ensaio_4b` (1 T4: d3-mini). **Se a fidelidade do 4B não chegar a 98%, pare:** o carregador tem problema e gastar horas do 27B seria desperdício.
3. Rode `kaggle_03_fidelidade_27b` (2 T4: d3 em 4 bits contra o resultado publicado). O limiar de 95% é provisório e será recalibrado com a primeira medida.

**Certo quando:** `kaggle_07_relatorio` mostra `G0-4b` e `G0-27b-nf4` como `pass`. **Se cair:** abra o mesmo notebook e rode tudo; o passo que não deu tempo diz **PAUSADO** e não registra evidência pela metade. A cota gratuita é de cerca de 30 h de GPU por semana (documentação, UNVERIFIED).

## 5. Sessão 3: F2b, o gate GT (pensar vale a pena?)

**Veredito oficial só pelo Kaggle (aviso 1).** Os quatro lugares abaixo rodam o mesmo código, mas **só o notebook do Kaggle (`notebooks/decide/think_gt_kaggle.ipynb`, backend `transformers`) produz um `PASS` ou `FAIL` que vale como gate, porque o gate hospedado não tem evidência operacional** (os limites de 100 tokens e de 10% das linhas que responderam, do #87, não foram medidos num provedor real). Num endpoint de API (o comando do passo 2 usa `--backend openai`), no `decide.think.hosted` ou na Modal, faça a sonda de 3 linhas e pare: uma linha que vem só com a resposta sai `reasoning_hidden` e o veredito vira `EXPLORATORY`, ou `INCONCLUSIVE` quando as linhas que esconderam o raciocínio ou pararam sem fechar o pensamento passam de 10% das linhas que responderam, e não `FAIL`; mas isso ainda não foi visto num provedor real. Se o seu resultado de API não for `PASS`, repita no Kaggle antes de registrar na issue.

**Onde:** um de quatro (escolha pelo que você tem): Kaggle T4, uma API com o Qwen3.8-27B pensando, uma GPU de 80 GB ou mais, ou a Modal. **Guia completo:** `decide/think/README.md` (Squad C, PR [#78](https://github.com/simpletibr/simplicio-27b/pull/78), já na `main`). Os notebooks fixam o código num commit (`SIMPLICIO_SHA`); use os da `main` mais nova.

1. Confira o pacote privado `pack.jsonl.gz` (400 linhas de 10 provas; guarde fora do clone):

   ```
   python -m decide.think.sample verify --manifest decide/think/gt-sample-manifest.json --pack pack.jsonl.gz
   ```

2. **Sonda de 3 linhas** (`--limit 3`), depois a rodada inteira (retomável), depois o veredito:

   ```
   python -m decide.think.thinker --backend openai --base-url https://SEU-ENDPOINT/v1 --model Qwen/Qwen3.8-27B-FP8 \
       --pack pack.jsonl.gz --manifest decide/think/gt-sample-manifest.json --out work/results.jsonl --limit 3
   python -m decide.think.compare --pack pack.jsonl.gz --thinker work/results.jsonl --out work/report
   ```

   A última linha impressa é `GT VERDICT: ...`. O relatório (`work/report/report.json`) só tem contagens e ids, e o orquestrador do dia alugado o lê.

**Decisão:** `PASS` liga o passo dos traços do pensador na sessão 8 (e o treino que lê raciocínio). `FAIL` ou `PARTIAL` deixa o dia no modo só-decisor, que é a entrega mínima do plano. O resultado de `FAIL` vale: registre-o na issue [#75](https://github.com/simpletibr/simplicio-27b/issues/75).

## 6. Sessão 4: F3, rotulagem pelos professores (Modal)

**Não rode agora** (aviso 2): `decide/modal` e o guia abaixo são da parte 2 do kit e não estão na `main` (a rodada gasta o crédito da Modal).

**Onde:** Modal, no crédito gratuito. **Guia:** `notebooks/decide/docs/04_rodada_paga.md` (kit; a seção 3 ensina a sonda). **Pacotes:** `pip install modal==1.6.2 && modal setup`.

O professor é o próprio `perplexity-ai/pplx-decider-v1.2` (autodestilação). Sem destilação de ensemble: o experimento da Fase 0A-bis não achou ganho nela.

1. Rode a **sonda** primeiro (sem `--yes` só imprime o plano; com `--yes` roda). `N` é o número de linhas de `work/corpus-limpo/manifest.json` (campo `rows`):

   ```
   modal run -m decide.modal.app::labels --run labels-1 --corpus hf:wesleysimplicio/simplicio-decide-data:corpus-limpo \
       --teacher pplx-decider-v1.2 --model-repo perplexity-ai/pplx-decider-v1.2 --total 100 --shard-requests 100
   ```

2. Veja a vazão (`::status`) e refaça o preço da rodada inteira (`python -m decide.modal.cost ...`, tudo **ESTIMATIVA**). A rodada inteira:

   ```
   modal run -m decide.modal.app::labels --run labels-1 --corpus hf:wesleysimplicio/simplicio-decide-data:corpus-limpo \
       --teacher pplx-decider-v1.2 --model-repo perplexity-ai/pplx-decider-v1.2 --total N --yes
   modal run -m decide.modal.app::export --run labels-1 --total N
   ```

   Para fechar o computador no meio, use `modal deploy` e `--deployed --no-wait` (o `--detach` sozinho não basta; veja o documento 04 do kit).
3. Baixe os rótulos e suba para o repositório de dados (a pasta `labels/pplx-decider-v1.2` é o que o dia alugado espera): os rótulos ficam no dataset de estado em `labels/labels-1/`; copie para `work/labels/pplx-decider-v1.2/`, confira o `manifest.json` e suba:

   ```
   hf download wesleysimplicio/simplicio-decide-state --repo-type dataset --include "labels/labels-1/*" --local-dir work/state
   hf upload wesleysimplicio/simplicio-decide-data work/state/labels/labels-1 labels/pplx-decider-v1.2 --repo-type dataset
   ```

   (Sintaxe do `hf` não conferida, UNVERIFIED.) Se a pasta vier sem `manifest.json`, grave um: `python -c "from decide.data import schema as S; S.seal_directory('work/state/labels/labels-1', kind='labels')"`.

**Certo quando:** `::export` imprime `complete=True` e ≥ 95% das linhas do corpus têm rótulo (a sessão 5 mede e recusa abaixo disso). **Se cair:** repita o mesmo comando; as partes prontas são puladas e uma parte interrompida recomeça do último checkpoint. Se o crédito acabar, a rodada espera o crédito do mês seguinte.

## 7. Sessão 5: alvos (CPU)

**Onde:** CPU, qualquer uma (para só montar os alvos, só o `pyarrow` é usado). **Lê:** `corpus-limpo/` e `labels/pplx-decider-v1.2/`. **Pacotes:** `pip install -r decide/train/requirements.txt`

```
python -m decide.train.build_targets --corpus work/corpus-limpo --labels work/labels/pplx-decider-v1.2 --out work/targets
```

- Com mais de um professor, passe um combinador (`--combiner combiner.json`; formato na seção 5 do `SCHEMA.md`). Sem ele, exige exatamente um professor.
- `--missing drop` (padrão) descarta o que o professor não rotulou e não tem rótulo ouro; `renormalize` e `error` estão no `--help`.
- **`--corpus` tem de ser a pasta que a descontaminação deixou** (`manifest.json` com `tool: "decontaminate"` e o `decontamination.json` ao lado). Corpus bruto, pasta selada à mão e parquet solto são recusados com um `TargetError` que diz o que fazer. A saída explícita é `--allow-undecontaminated`, uma decisão sua: ela fica gravada nos alvos e no checkpoint (`corpus.allow_undecontaminated`). Não a use no dia que vale.
- **`--labels` (pasta) e `--traces` (pasta) precisam de `manifest.json`**; o `build_targets` lê só as partes que ele lista e confere o sha256. Rótulos copiados da Modal podem vir sem ele: `python -c "from decide.data import schema as S; S.seal_directory('work/labels/pplx-decider-v1.2', kind='labels')"` (e `kind='traces'` para os traços). Sele de novo se a pasta mudou.
- **Traços do pensador (`--traces work/traces/Qwen3.8-27B`, acrescenta o raciocínio ao alvo): não faça isto à mão e não treine com raciocínio ainda.** Hoje nenhum comando da `main` gera essa pasta; quem a gera é o passo `traces` do dia alugado (F6, na `main`, mas o dia fica só-decisor hoje: sessão 8). Cada linha de traço leva o `request_sha256` do pedido para o qual foi escrita, e o `build_targets` recusa o traço de outro pedido. Antes de qualquer treino com `--reasoning` falta uma coisa: a descontaminação por n-grama cobrir a coluna `reasoning` (issue [#83](https://github.com/simpletibr/simplicio-27b/issues/83)). O motor do kit já sabe dobrar o raciocínio na avaliação com a regra do treino (`fold_reasoning`, `data/SCHEMA.md`, seção 10), mas só quando pedido: veja a seção 13.

**Escreve:** `work/targets/` (`*.parquet`, `manifest.json`, `report.json`). **Certo quando:** `report.json` mostra `rows_out` perto de `rows_in` e a cobertura de **professor**, `1 - unlabeled_questions / questions` (seção 11 do `SCHEMA.md`), é ≥ 0,95; abaixo disso volte à sessão 4. Atenção: `rows_out` também conta linhas que só têm rótulo ouro, por isso a conta certa é a das perguntas. O dia alugado repete essa medida na fase `targets`, antes de qualquer hora de GPU. **Se cair:** rode de novo; as partes terminadas são puladas e as corrompidas são refeitas.

## 8. Sessão 6: F4, o ensaio 4B (G2)

**Onde:** Kaggle T4 (ou o crédito da Modal). **Importante (UNVERIFIED):** a T4 não tem bf16 nativo; o treino em bf16 pode cair em emulação e ficar muito lento. O padrão `--dtype auto` usa bf16 numa GPU e fp32 na CPU. O treino **recusa `--dtype fp16`**: não há escala de perda, e um gradiente que zera por underflow é um número finito que nenhuma checagem vê (o modelo aprenderia menos do que a curva de perda mostra). Na T4 sobram o bf16 emulado (o padrão) e o fp32, que provavelmente não cabe para o 4B (conta: 4 bilhões × 4 bytes ≈ 16 GB contra 15 GB de memória, UNVERIFIED). Nada disso foi medido numa GPU; o pilot do passo 2 é o teste. Se ele mostrar emulação lenta, faça este ensaio numa GPU com bf16 (a Modal não tem app de treino na `main`; seria preciso escrevê-lo). **TODO (Squad A):** o kit não tem um notebook de treino do 4B; rode os comandos numa célula ou no terminal. **Pacotes:** `pip install -r decide/train/requirements.txt`

Ponto de partida: `perplexity-ai/pplx-decider-flash-v1.2` (4B). Baixe-o numa pasta local (`hf download perplexity-ai/pplx-decider-flash-v1.2 --local-dir work/flash`).

1. **Plano sem carregar nada:**

   ```
   python -m decide.train.train --targets work/targets --checkpoint work/flash --out runs/4b-seed1 --mode lora --lora-rank 128 --dry-run
   ```

2. **Pilot** (poucos passos; olhe a perda, a memória e os tokens por segundo):

   ```
   python -m decide.train.train --targets work/targets --checkpoint work/flash --out runs/4b-pilot --mode lora --lora-rank 128 \
       --max-train 2000 --max-steps 20 --eval-every 10 --max-dev 200
   ```

   Olhe também `runs/4b-pilot/summary.json`: `dropped_too_long` conta os pedidos que passaram de `--max-length` e ficaram fora do treino. Se for muita coisa, o modelo não aprende os pedidos longos; aumente `--max-length` (se couber na memória) ou revise o corpus.

3. **Treino**, 1 época, uma semente primeiro. O treino já calibra a temperatura no fim (`final/decision_config.json`), não precisa rodar mais nada para um checkpoint só:

   ```
   python -m decide.train.train --targets work/targets --checkpoint work/flash --out runs/4b-seed1 --mode lora --lora-rank 128
   ```

   Se o G2 ficar perto da margem, acrescente sementes (`--seed 20261022`, `--seed 20261023`, cada uma em `--out` próprio) e faça a média. A média guarda uma temperatura **provisória** e **tem de ser recalibrada**, senão o kit a recusa:

   ```
   python -m decide.train.soup --inputs runs/4b-seed1/final runs/4b-seed2/final runs/4b-seed3/final --out runs/4b-soup
   python -m decide.train.temperature --checkpoint runs/4b-soup --targets work/targets --write
   ```

4. **G2: medir no privado caseiro**, o novo contra o ponto de partida, nas mesmas linhas. **O `run` nunca carregou um modelo de tamanho real (UNVERIFIED):** `decide.rental.proxy_eval` (F6) e o kit corrigido (#86) estão na `main`, mas os testes usam modelos minúsculos numa CPU; a primeira execução de verdade é o teste. Os comandos abaixo conferem com o `--help` e com esses testes; não são uma receita rodada numa GPU. Cada `run` retoma de onde parou e precisa do kit (a primeira linha do bloco o instala):

   ```
   pip install -r notebooks/decide/requirements.txt
   python -m decide.rental.proxy_eval requests --proxy work/proxy --out work/proxy-requests.jsonl
   python -m decide.rental.proxy_eval run --requests work/proxy-requests.jsonl --model flash=dir:work/flash --out work/g2/flash
   python -m decide.rental.proxy_eval run --requests work/proxy-requests.jsonl --model novo=dir:runs/4b-seed1/final --out work/g2/novo
   python -m decide.rental.proxy_eval compare --proxy work/proxy --new work/g2/novo --baseline work/g2/flash --margin 1.0
   ```

   `compare` imprime `PASS`, `FAIL` ou `INVALID` (linhas faltando ou inválidas), a diferença em pontos do privado (100 × habilidade média corrigida pelo acaso) e o intervalo de confiança do bootstrap pareado. **Sai com código 3 quando o gate não passa.** `--margin` não tem valor padrão de propósito. `--min-rows` (padrão **100**, o mesmo piso do G4) é o número de linhas **com gold** abaixo do qual o veredito é `INVALID` (`too_few_rows`), por maior que seja o ganho: um ganho medido em poucas linhas não é ganho. O 100 é uma **escolha**, não um cálculo de poder estatístico (UNVERIFIED); o privado caseiro pede `per_source: 300` por fonte (`decide/config.data.yaml`). Um proxy menor que o piso precisa de `--min-rows` menor, escrito no comando de propósito.

**Decisão:** G2 passou: siga para a sessão 8 e considere o perfil completo (veja o aviso da seção 9). G2 falhou: não gaste o dia alugado com a receita como está; veja os logs de perda (`runs/*/events.jsonl`), mude uma coisa por vez (`--alpha`, `--lr`, `--epochs`) e repita o ensaio. O modo seguro continua sendo o `pplx-decider-v1.2` puro.

**Se cair:** `--resume` continua uma execução parada (a pasta `--out` é a mesma; os argumentos têm de ser os mesmos, senão ele recusa e diz o que mudou). Uma pasta com uma execução terminada recusa recomeçar.

## 9. Sessão 7: F5, merge de backbones (opcional)

**Não rode agora** (aviso 2): os notebooks do kit não estão na `main`.

**Onde:** Kaggle, 2 T4, notebooks `kaggle_04_varredura` e `kaggle_05_calibrar` do kit. É um experimento barato e de baixa expectativa: a Fase 0A mostrou que média de pesos de modelos de topo não supera o melhor membro. "Nenhum merge ganhou" é um resultado válido; o ponto de partida do dia alugado fica o `pplx-decider-v1.2`. Se um merge ganhar (G3), troque `rental.start` em `decide/config.rental.yaml` pelo repositório do vencedor, com a revisão exata.

## 10. Sessão 8: F6, o dia alugado

**Está na `main`** (F6, issue [#71](https://github.com/simpletibr/simplicio-27b/issues/71)): `decide/rental` e `decide/config.rental.yaml`. Os testes (`tests/test_flow_71.py`, `tests/test_contract_71.py`) ensaiam o dia inteiro em CPU, com um modelo minúsculo e uma pasta no lugar do Hugging Face; **nenhuma fase rodou numa GPU nem contra o Hugging Face de verdade** (UNVERIFIED). As fases `g4` e `public_round` carregam o modelo pelo kit (#86, na `main`), que nunca carregou um modelo de tamanho real: a primeira execução de verdade é o teste (UNVERIFIED).

**Onde:** uma máquina Linux alugada com a GPU do perfil. **Perfis** (`decide/rental/profiles/`): `econ_b200` (1× B200, LoRA r128, 1 semente: o PLAN diz ~14 h e ~US$ 90; o estimador do comando abaixo imprime 12,75 h e US$ 79,70 até o veredito do G4, e 17,51 h e US$ 109,42 com a rodada pública e o envio, **UNVERIFIED**) e `full_8xh100` (8× H100, ajuste completo, 2 sementes: o PLAN diz ~12 a 16 h e ~US$ 380 a 510; o estimador imprime 17,49 h e US$ 552,52 até o G4, **UNVERIFIED**). O econômico é o padrão. O completo só entra se o G2 mostrou ganho ≥ +1,0.

> **Aviso do perfil completo.** Com DDP cada GPU guarda os pesos e os gradientes em bf16 (~104 GB) e uma H100 tem 80 GB, então `full_8xh100` **não cabe em 8× H100** (conta, UNVERIFIED). O estimador avisa (`ddp_does_not_fit`) e a checagem prévia do orquestrador **recusa** o perfil; só `--ignore-warning ddp_does_not_fit`, uma decisão sua, a contorna. Use H200 ou B200, ou escreva FSDP em `decide/train` antes de alugar.

**Antes de alugar (no seu computador):**

1. Veja o plano, o preço estimado e o veredito da checagem prévia. **Nada é baixado, treinado nem enviado sem `--execute`.** Pacotes: `pip install -r decide/rental/requirements.txt`


   ```
   python -m decide.rental.run --profile econ_b200
   ```

2. O ensaio dos testes (`tests/test_flow_71.py`) não usa os dados reais, então **a primeira execução de verdade é o teste**. Comece pela fase `preflight` e pelo `--until targets` (CPU, sem treino; `--allow-cpu` dispensa a GPU do perfil nesse ensaio): `python -m decide.rental.run --profile econ_b200 --execute --allow-cpu --until targets --workdir work/day`.
3. Confira que o repositório de dados (`simplicio-decide-data`) tem `corpus-limpo/`, `labels/pplx-decider-v1.2/`, `proxy/` e, se o GT passou, `traces/`.

**Na máquina alugada:**

```
git init simplicio-27b && cd simplicio-27b
git fetch --depth 1 https://github.com/simpletibr/simplicio-27b COMMIT    # o mesmo COMMIT da sessão 0
git checkout --detach FETCH_HEAD
pip install -r decide/train/requirements.txt -r decide/rental/requirements.txt -r notebooks/decide/requirements.txt
export HF_TOKEN=...                      # só na sessão do terminal
python -m decide.rental.run --profile econ_b200 --execute --workdir /workspace/day \
    --gt-report work/report/report.json --public-requests requests/requests.jsonl.gz
```

- Sem `--gt-report` (ou com um GT que não liga os traços) o dia é **só-decisor**. Sem `--public-requests` a rodada pública é pulada. Só liga os traços um `report.json` do formato `think-gt-report/2` (o que o `decide.think.compare` escreve) com `verdict.status` igual a `PASS`, cobertura completa (`coverage.complete`), o modelo do pensador (ou a versão `-FP8` dele) igual ao de `config.rental.yaml`, uma revisão registrada (e igual à da config, quando a config fixa uma) e o pensamento ligado. Um veredito `EXPLORATORY` (há linhas que vieram só com a resposta, `reasoning_hidden`) ou `INCONCLUSIVE` não liga os traços, e o plano impresso diz `thinker traces: OFF` com os motivos (`decide/rental/gates.py::gt_evidence`, lido do código; o dia com um relatório real do Kaggle não foi rodado: UNVERIFIED).
- Uma fase pulada por falta de entrada não conta como pronta: rode o mesmo comando **com** `--public-requests` e a rodada pública é feita, e o envio é refeito para levar os resultados dela. Com `--gt-report` os traços só valem se o treino ainda não rodou; depois dele o orquestrador recusa e manda usar outro `--workdir` (ou seguir sem `--gt-report`).
- **Hoje, não passe `--gt-report`.** Os traços do pensador entram no treino como o campo `reasoning`, e esse campo ainda não passa pela descontaminação por n-gramas (issue [#83](https://github.com/simpletibr/simplicio-27b/issues/83)). Rode o dia só-decisor até essa correção entrar na `main` (seção 13). O fold do kit (#95) não muda isso: o dia não o liga (o G4 tira o `reasoning` dos pedidos, e o `SdKit` usa o padrão `"refuse"`), e a inferência híbrida (pensador + decisor) não está implementada.
- O dia tem 11 fases: `preflight, fetch, targets, traces, targets_traces, train, soup, calibrate, g4, public_round, publish`. Cada uma é pulada se já terminou, suas saídas não mudaram **e o que ela usou continua o mesmo** (o estado guarda as impressões digitais das saídas em que cada fase se apoiou). `--only` e `--until` rodam parte delas, mas **`--only` não contorna gate**: cada fase confere o que ela precisa (a tabela `NEEDS` de `decide/rental/run.py`) e recusa rodar, sem escrever nada, se isso não terminou, mudou no disco depois do seu gate ou ainda deve ser decidido de novo. Por exemplo, `--only train soup calibrate g4 publish` depois de um dia que parou na cobertura de rótulos recusa já na `train` (as fases `fetch`, `targets` e as demais da lista têm de estar prontas antes; rode-as na ordem, uma a uma, se quiser).
- **Gates param o dia, não só avisam.** Cobertura de rótulos (F3) antes de qualquer hora de GPU; G4 (novo ≥ ponto de partida + 1,5 pontos do privado) antes da rodada pública e do envio. O G4 compara o novo com o ponto de partida **nas mesmas linhas** e imprime o intervalo de confiança do bootstrap pareado (`ci_includes_zero` é aviso, não condição).
- **O G4 fica preso aos bytes que ele mediu.** O estado e o `g4/verdict.json` (campo `measured`) guardam o digest de **todos** os bytes de `final/` (`final_digest`), a impressão digital do ponto de partida e a identidade do baseline. A rodada pública e o envio recusam um `final/` que o G4 não mediu; um novo treino, uma nova calibração ou uma edição à mão depois do G4 faz o G4 rodar de novo antes de qualquer coisa ser enviada, e o `g4/new` antigo vai para `g4/new.stale-N` (não é apagado, e o kit não o retoma como se fosse do checkpoint novo). O resultado do baseline é mantido se o ponto de partida não mudou. O digest lê o checkpoint inteiro uma vez por fase que o pede: custo de um passe sobre ~50 GB de pesos, **tempo não medido (UNVERIFIED)**, pequeno perto das horas que ele protege.
- **Um orquestrador por `--workdir`.** `state/day.lock` (um `flock`): um segundo comando no mesmo `--workdir` é recusado com o PID e a máquina do primeiro; o lock de um processo que morreu é retomado sozinho. Só em Linux/macOS; em outro sistema não há trava, e o dia é feito para uma máquina Linux alugada.
- **O que volta do repositório de trabalho é conferido antes de ser usado.** O `state/day.json` vem de fora (outra máquina, uma edição à mão): antes de mover um byte, a restauração recusa um estado de outro perfil ou configuração, uma saída com caminho fora do `--workdir` (absoluto, com `..`, com barra invertida ou letra de unidade, ou que sai por um link) e um nome de saída que não seja um nome simples. Se recusar, nada é escrito. O `day.json` é posto no lugar por último, então uma restauração interrompida é terminada pelo comando seguinte.
- **Não há teto de gasto e o orquestrador não desliga a máquina.** O relógio do aluguel corre até **você** parar a instância, também depois de um gate ter parado o dia (código 3) e depois de `complete`. Pelo estimador, o perfil econômico custa cerca de US$ 6,25 por hora (US$ 79,70 em 12,75 h, **UNVERIFIED**, conta do estimador, não preço lido do provedor); pare a máquina assim que o relatório do dia disser `complete` ou o dia parar, e suba os `final/` e `g4/` que precisar guardar antes, se o gate parou.
- Códigos de saída: `0` o dia terminou (ou o plano foi impresso); `1` erro (o estado diz onde; rode de novo); `2` uso ou configuração; `3` um gate parou o dia.
- Cada fase pronta é copiada para o repositório privado `simplicio-decide-day`. Numa máquina nova, com **o mesmo `--workdir` (caminho absoluto)**, o estado e as saídas voltam antes de qualquer cálculo. O que não é copiado: o `resume.pt` de uma semente no meio do treino (uma máquina perdida no meio de uma semente perde o progresso desde o último `resume.pt` **do disco**).
- **O que a fase `fetch` confere antes de qualquer hora de GPU** (e o `fetch` baixa o checkpoint de 52 GB só depois): o corpus saiu do `decontaminate`; o proxy tem as linhas do `manifest.json` iguais às das partes e pelo menos `gates.g4_min_rows` (100 no padrão) linhas com gold; o `manifest.json` do proxy diz que a segunda passada (`proxy_private build --corpus <corpus limpo>`) rodou e achou 0 vazamentos (`corpus_leak_check`); o proxy e o corpus não têm fonte em comum; e o `decontamination.json` do corpus lista uma referência com tantos pedidos quanto o proxy tem de linhas. **Isso não prova** que essa referência era ESTE proxy (o digest que o `decontaminate` guarda é o da pasta da primeira passada, e a segunda passada reescreve o `manifest.json`, então ele não pode ser comparado com a pasta que chega ao dia) nem que o corpus foi triado contra a amostra do GT (a suíte pública dada ao `decontaminate` cobre qualquer amostra dela, se foi dada: passo 5 da sessão 1). Se o proxy for refeito depois do passo 5, refaça os passos 5 e 6.
- **A checagem prévia recusa** `gates.g4_margin_points` ≤ 0 (um G4 que nada exige), `gates.label_coverage` abaixo de 0,9 ou acima de 1 (o gate F3 pede 0,95), `gates.g4_min_rows` < 1 e dois papéis no mesmo repositório (`work_repo`, `weights_repo`, `data_repo`), **sem diferenciar maiúsculas** (o Hub trata `Org/Repo` e `org/repo` como o mesmo repositório): o backup do dia leva o proxy privado com o gold (`g4/requests.jsonl`) e não pode cair no repositório dos pesos.
- O envio recusa, pelo nome, arquivos que parecem segredo (`.env`, `*token*`, `apikey*`, `auth.json`, `.netrc`, `*.pem`, `id_rsa`, e agora também `openai_key.txt`, `gcp-key.json`, `bearer.txt` e `service-account.json`: `key` e `bearer` como palavra do nome, e `service-account` em qualquer parte); a lista é de nomes, não de conteúdo: um segredo num arquivo chamado `notas.txt` passa, e um nome como `primary_key_map.json` é recusado sem ser segredo (renomeie).
- Processos filhos (o treino) não recebem o token do Hugging Face e rodam offline.

**Escreve (em `--workdir`):** `state/day.json`, `logs/`, `data/`, `targets/`, `runs/seed*/`, `soup/`, `final/`, `g4/verdict.json`, `public/`, e o relatório do dia (`report.md` e `report.json`). O modelo final sobe, privado, para `Simplicio-Decide-27B`.

**Certo quando:** o relatório do dia diz `complete`, `g4/verdict.json` tem `"status": "PASS"` e o modelo está no repositório de pesos.

**Se cair:** rode o mesmo comando. Se um gate parou o dia (código 3), leia o motivo no relatório: não existe parâmetro para pular gate de propósito (nem o `--only`: veja acima). Um G4 `INVALID` quer dizer linhas faltando ou inválidas; rode de novo para que o kit as complete.

**Não implementado:** a inferência híbrida (o pensador raciocina e o decisor lê) na avaliação do G4 e na rodada pública; o dia mede o **decisor sozinho**, sem raciocínio. A variante da Modal para o dia (`modal_day.py`) também não existe.

## 11. Sessão 9: F7, submissão

**Onde:** Hugging Face e o Space. **Guia:** `notebooks/decide/docs/04_rodada_paga.md`, seções 5 e 6 (kit).

1. **Rodada pública completa em bf16.** O dia alugado a faz com uma ordem das opções e a temperatura do próprio checkpoint (`public/`). A da Modal pode usar a receita do modelo (duas ordens, temperaturas por tipo). Use **uma só** e escreva no cartão qual foi. Pela Modal: `modal run -m decide.modal.app::final ... --recipe hf:wesleysimplicio/Simplicio-Decide-27B --yes`, depois `::assemble`. Comece pela sonda de 100 requisições. O número do PR é o índice público do `results.jsonl` publicado, descrito como **ajustado por seleção**; não é a nota Full do Jev.
2. **Cartão do modelo.** `report/README.preview.md` (kit) é honesto por construção. Preencha à mão o que o kit não sabe: a licença e os dados do LoRA `wesleysimplicio/Simplicio-27B`; os professores (`pplx-decider-v1.2` rotulando o corpus); as fontes do corpus com a licença de cada uma; o relatório de descontaminação (`decontamination.json`); o resultado do G4 com o intervalo de confiança; e, se o pensador entrou, o relatório do GT. O Lychee (CC BY-NC) e o Amos (cabeça incompatível) ficam de fora, e o cartão tem de continuar dizendo isso.
3. **Tornar público** `Simplicio-Decide-27B` e o dataset de resultados, só depois de ler o cartão. Um repositório privado não pode ser reproduzido pelos revisores.
4. **PR no Space** (`multimodalart/jev-decision-index`): o formato dos arquivos, o destino do PR e as regras de revisão **não foram verificados**; leia as instruções atuais do Space antes. Gate G5: concordância do laboratório ≥ 98,5% (medida por eles, não por nós).

## 12. Se algo falhar

| Sintoma | Onde | O que fazer |
|---|---|---|
| a sessão do Kaggle ou Colab caiu | qualquer notebook | abra o mesmo notebook e rode tudo de novo; o passo continua do último bloco |
| `PAUSADO` no notebook | kit | nada se perdeu; rode na próxima sessão |
| preempção ou queda na Modal | rotulagem | repita o mesmo comando; partes prontas são puladas |
| `ManifestError` ao ler corpus, rótulos, traços ou alvos | dados | a pasta é um build interrompido ou mexido à mão; refaça o passo que a gerou (não apague só uma parte). Rótulos ou traços copiados de fora, sem `manifest.json` (`... has no manifest.json listing its parts`): sele a pasta, como na sessão 5 |
| `TargetError: ... not by decontaminate` ou `... is a parquet file, not the folder decontaminate leaves` | `build_targets` | o `--corpus` não é a saída da descontaminação; aponte para a pasta de `decide.data.decontaminate --out` (sessão 1, passo 5). `--allow-undecontaminated` é a saída explícita e fica gravada nos alvos |
| `TargetError: ... the trace was made for another request` | `build_targets` | os traços foram escritos para outro corpus (o `request_sha256` não confere); gere os traços de novo a partir do corpus limpo atual |
| `TargetError: ... --out` | `build_targets` | `--out` é uma entrada, está dentro de uma, ou foi feito por outra ferramenta: use uma pasta nova |
| `--resume: the saved run differs in ...` | treino | você mudou um argumento (ou os dados, ou o checkpoint) entre a parada e a retomada; volte ao valor anterior ou comece uma pasta nova |
| o treino recusa recomeçar | treino | a pasta já tem uma execução terminada; use outra `--out` |
| perda ou gradiente não finito, e a execução para | treino | repita com `--dtype fp32` (numa T4 provavelmente não cabe para o 4B, conta na sessão 6) ou numa GPU com bf16. Não foi medido numa GPU (UNVERIFIED) |
| `--dtype fp16 is refused` | treino | de propósito: sem escala de perda um gradiente que zera por underflow passa calado. Use o padrão (bf16 numa GPU, emulado e lento numa T4, UNVERIFIED) ou fp32 |
| `--resume: ... already exists, so the run is finished` | treino | a execução terminou (existe `final/`); o `resume.pt` é apagado nesse ponto. Use outra `--out` para uma nova |
| `the temperature ... is a placeholder` | kit ou dia | rode `python -m decide.train.temperature --checkpoint <pasta> --targets <alvos> --write` |
| falta de memória na GPU | treino | diminua `--token-budget` ou `--batch-rows` (o `--effective-batch` não muda, só quantas linhas cabem por passada) |
| G4 parou o dia | dia alugado | veja `g4/verdict.json`; o novo não superou o ponto de partida por 1,5 pontos; não suba o modelo nem faça a rodada pública |
| o token vazou | qualquer | revogue em *Settings → Access Tokens* e crie outro; não basta apagar a mensagem |

## 13. Em aberto

- **O que ainda segura o aviso 1:** o gate GT hospedado não tem evidência operacional (#87, Squad C: os limites de 100 tokens e de 10% não foram medidos num provedor real). Os bloqueantes do núcleo do kit (#80) foram corrigidos no #86 e o aviso deles saiu.
- **Antes de qualquer treino com `--reasoning`:** a descontaminação cobrir a coluna `reasoning` dos traços (issue [#83](https://github.com/simpletibr/simplicio-27b/issues/83), aberta). O fold na avaliação já está na `main` (próximo item).
- **Raciocínio dobrado (fold) na avaliação** (PR [#95](https://github.com/simpletibr/simplicio-27b/pull/95), na `main` em `c32c417`). Os fatos vêm do corpo do #95 e da revisão independente dele; "medi" marca o que eu rodei.
  - **Onde:** o parâmetro `reasoning` de `decide.sd.evaluate.run_requests` (`decide/sd/evaluate.py`); `decide.sd.api.run_requests(..., reasoning=...)` repassa. **Não existe flag de linha de comando para o fold:** nenhum comando deste RUNBOOK o liga (o teste `RunbookFactsTest` falha no dia em que uma flag aparecer). Valores: `"refuse"` (padrão) e `"fold"`; qualquer outro é `EvaluateError` antes de qualquer trabalho.
  - **`"refuse"`:** um pedido com `reasoning` chega ao motor, que o recusa: linha final `unsupported`, e o erro diz como pedir o fold. Nada é dobrado sem ser pedido.
  - **`reasoning="fold"`:** o `reasoning` sai do pedido e entra no fim do texto do estado: uma linha em branco, a linha-rótulo `Raciocínio:` e o texto limpo e cortado em 8000 caracteres (`REASONING_MAX_CHARS`). Quem define o bloco é só `fold_reasoning` (`decide/train/protocol.py`; casos dourados em `tests/fold_reasoning_cases.json`), a mesma função do treino com `--reasoning`: o prompt da avaliação é o do treino.
  - **O corte** guarda o fim do texto, a partir de uma quebra de linha. Se o corte cai no meio de uma linha e há linhas depois, a linha cortada sai inteira. Medi: um raciocínio com 20 000 caracteres numa linha, seguido de uma linha `so B`, num estado `s`, deixa o estado dobrado com 55 caracteres (`[... 20001 caracteres omitidos ...]` e `so B`); uma linha única de 20 000 caracteres guarda os últimos ~8000 (o estado inteiro ficou com 8015 caracteres, com rótulo e aviso). A regra é a mesma no treino (mesma função), então não há descompasso de prompt; só convém saber que um raciocínio de uma linha enorme some quase todo.
  - **O que o fold não faz:** um pedido cujo estado já tem o bloco `Raciocínio:`, ou com `reasoning` vazio, não é dobrado nem chega ao motor: linha `unsupported` com o motivo. Uma linha sem `reasoning` é igual nos dois modos. Imagens o kit recusa de qualquer forma (o motor lê só o texto).
  - **O que a saída registra, e a retomada:** as linhas dobradas têm `reasoning_mode: "fold"` e o `environment.json` guarda o modo da rodada. Linhas que o padrão recusou são pedidas de novo quando a retomada pede `"fold"`; uma pasta com linhas dobradas não é retomada com `"refuse"` (`EvaluateError`: uma pasta responde um prompt só; comece outra ou use `resume=False`).
  - **O fold precisa do repositório inteiro, no commit do treino.** Ao dobrar, o kit importa `decide/train/protocol.py` (que usa `decide/train/codes.py`; só biblioteca padrão, nada a instalar). Copiar só `decide/sd` para a máquina faz o fold falhar logo no começo com `ModuleNotFoundError`, sem gravar nada, e a mensagem não explica (achado N1 da revisão independente do #95; não reproduzi). Um notebook do kit que avalie com o fold (Squad A, ainda fora da `main`) tem de clonar o repositório inteiro no commit fixado. E a regra do bloco vem do checkout do kit, não do checkpoint: avalie um modelo treinado com raciocínio num commit cujo `tests/fold_reasoning_cases.json` e `decide/train/protocol.py` sejam iguais aos do commit em que ele foi treinado (`git diff COMMIT_DO_TREINO COMMIT_DO_KIT -- tests/fold_reasoning_cases.json decide/train/protocol.py` sem saída).
  - **O orçamento de lote não conta o raciocínio dobrado** (issue [#97](https://github.com/simpletibr/simplicio-27b/issues/97), aberta): `Item.tokens()` usa o `proxy_tokens` da linha, que não inclui o texto dobrado (até 8000 caracteres a mais por linha). O preço do primeiro bloco do orçamento e a elegibilidade por `max_proxy_tokens` ficam otimistas para as linhas com `reasoning`; se o prompt dobrado passar do `max_length` do motor, a linha vira `unsupported` (nunca é cortada), então nada fica errado em silêncio. Só importa em lote grande com orçamento apertado.
  - **O dia alugado não liga o fold:** o G4 tira o `reasoning` dos pedidos (o veredito grava `mode: decider only: no thinker reasoning is shown to either model`) e o `SdKit` chama o padrão `"refuse"`. Inferência híbrida: veja "Não implementado" abaixo.
  - **UNVERIFIED:** o fold foi exercitado só com motores de teste e com o motor do kit sobre um modelo minúsculo, em CPU (relato do #95); que um modelo treinado com `--reasoning mixed` responda melhor ao prompt dobrado não foi medido (o #95 só garante que o prompt é o mesmo do treino).
- **UNVERIFIED, tudo:** o treino em GPU (a T4 sem bf16, a B200, o DDP em 8 cartas, o LoRA r128 em 27B, a memória e a vazão), o carregamento do 27B real pelo kit, o `hf` e a API do Hub com a sua conta, os preços e os tempos dos perfis, a Modal, as regras do Space.
- **TODO (Squad A):** notebook de treino do 4B; confirmar o caminho `requests/suite-0.3/`; confirmar o formato dos rótulos exportados pela Modal (`labels/<run>/`) contra o leitor `read_teacher`.
- **Squad C:** o orquestrador lê o `report.json` do GT no formato `think-gt-report/2`; o teste `RealGtReportContract` (`tests/test_contract_71.py`) monta o relatório com o `decide.think.compare.build_report` real e confere o nome do formato e os campos que o dia lê (o veredito, a cobertura e a configuração do pensador). **TODO (Squad C):** o `pack.jsonl.gz` e o script de reconstrução da suíte ainda não estão versionados.
- **Não implementado:** inferência híbrida (pensador + decisor) no G4 e na rodada pública; FSDP; o dia na Modal.
- **Plano B (bases alternativas):** só vale se o gate GT falhar **e** a receita do Qwen3.8 estagnar abaixo de 66 no privado caseiro depois do dia econômico. Está na ADR 0005 e na seção 10 do `PLAN.md` (PR [#81](https://github.com/simpletibr/simplicio-27b/pull/81), na `main`); nada neste RUNBOOK depende dele.
