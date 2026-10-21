# decide/think: o kit do gate GT (F2b)

O Qwen3.8-27B pensando (thinking) acerta pelo menos **+10 pontos** a mais que o melhor decisor de um passe, com o piso do
IC 95% pareado **acima de 0**? Este diretório tem o que falta para responder, numa amostra pública de 400 linhas (10 provas
de conhecimento e raciocínio, 40 por prova). Issue #75, épico #64, [ADR 0004](../../docs/adr/0004-pensar-sob-demanda.md),
[plano v2](../PLAN.md).

> **Estado desta entrega.** Nada aqui rodou numa GPU. Rodou na CPU: os testes (com um modelo aleatório pequeno e um
> servidor HTTP falso), o pacote de 400 linhas reais dos 8 decisores e um ensaio do relatório com um pensador **sintético**.
> O que cada backend faz na máquina de verdade, e quanto demora, é **UNVERIFIED** até alguém rodar a sonda de 3 linhas
> descrita abaixo. Segue a regra da ADR 0004: só número medido é fato; estimativa leva a marca e não decide nada sozinha.

## Comece por aqui

1. **Pacote.** Você precisa do `pack.jsonl.gz` (privado, 0,6 MB). Veja "Amostra e pacote". Confira com `sample verify`.
2. **Onde rodar.** Escolha um backend em "Backends":

   | você tem | rode | custo |
   |---|---|---|
   | só o Kaggle grátis | `notebooks/decide/think_gt_kaggle.ipynb` (T4 x2, 4-bit) | grátis; o tempo é incerto e provavelmente longo (veja "Tempo e custo") |
   | uma chave de API com o Qwen3.8-27B pensando | `notebooks/decide/think_gt_api.ipynb`, `MODE = "endpoint"` | tokens de saída |
   | uma GPU de 80 GB ou mais (Colab G4, A100, H100) | `notebooks/decide/think_gt_api.ipynb`, `MODE = "local-vllm"`, ou `python -m decide.think.hosted` | horas de GPU |
   | uma conta Modal | `modal run -m decide.think.modal_app` | horas de GPU |

3. **Sonda.** Rode **3 linhas** primeiro (`--limit 3`; os notebooks já começam assim). Ela mostra se roda, quantos tokens
   o modelo gasta por linha e a velocidade. Com **10 linhas** (uma de cada prova, porque o pacote vem em rodízio) dá para
   estimar o total.
4. **Rodada inteira.** Tire o limite. É retomável: rode de novo com os mesmos argumentos e ele continua de onde parou.
5. **Veredito.** A última linha impressa é `GT VERDICT: ...`. O relatório completo fica em `report/report.md` e
   `report/report.json`.

Os três comandos que tudo isso envolve (os notebooks, `hosted` e a Modal só os encadeiam):

```
python -m decide.think.sample verify --manifest decide/think/gt-sample-manifest.json --pack pack.jsonl.gz
python -m decide.think.thinker --backend openai --base-url https://SEU-ENDPOINT/v1 --model Qwen/Qwen3.8-27B-FP8 --pack pack.jsonl.gz --manifest decide/think/gt-sample-manifest.json --out work/results.jsonl --limit 3
python -m decide.think.compare --pack pack.jsonl.gz --thinker work/results.jsonl --out work/report
```

## A regra do gate

- **Pergunta.** O Qwen3.8-27B com thinking bate o melhor decisor de um passe por **pelo menos +10,0 pontos** de acurácia, com o
  **limite inferior do IC 95% acima de 0**?
- **Unidade do veredito.** A média macro sobre as provas da amostra (cada prova pesa igual, como no índice do kit): acurácia
  macro do pensador menos a do **melhor decisor nestas linhas** (o melhor dos 8 por acurácia macro; empate desempata pelo nome).
- **IC.** Bootstrap percentil pareado e estratificado: 10.000 reamostras, semente 20261021, linhas reamostradas **dentro de
  cada prova**, pensador e decisor nas mesmas linhas. `ci_reselected` repete a conta escolhendo o melhor decisor de novo a cada
  reamostragem (mais exigente).
- **PASS** se delta >= +10,0 **e** piso do IC > 0. **FAIL** caso contrário, com o motivo (`delta_below_min`,
  `ci_low_not_above_zero`). `can_still_pass` aparece quando o topo do IC ainda alcança +10. **PARTIAL** quando faltam linhas.
- **Como cada lado responde.** O decisor responde com o argmax das probabilidades publicadas (primeiro máximo na ordem das
  opções). Uma linha só conta como acerto se **todas** as perguntas dela estão certas (SATA-Bench é exato, como no kit); no
  ChessBench vale qualquer lance do conjunto de empates. O pensador responde com a última linha `ANSWER <pergunta>: <rótulo>`
  **depois** do último `</think>`. Resposta ilegível, orçamento de tokens esgotado ou falha terminal conta como **erro** e é
  contada no relatório. Linha com erro transitório (rede, falta de memória) conta como **ausente**, não como errada: é
  listada e o veredito sai `PARTIAL` até a próxima rodada refazê-la.
- A habilidade corrigida pelo acaso (`skill`, a fórmula do kit) é impressa por prova, mas o veredito é em acurácia.

### O que a amostra já diz antes de rodar o pensador (medido nas 400 linhas reais)

O melhor decisor nestas linhas é o **startlux-27b**, com **70,25** de acurácia macro. Então o pensador precisa de **80,25** de
acurácia macro **nas mesmas 400 linhas**. O espaço até 100 em cada prova mostra de onde o ganho pode vir:

| prova | acaso | startlux-27b (referência) | espaço até 100 |
|---|---:|---:|---:|
| GPQA Diamond | 25,0 | 50,0 | 50,0 |
| WinoGrande | 50,0 | 100,0 | 0,0 |
| GSM8K | 17,5 | 80,0 | 20,0 |
| ChessBench | 8,2 | 17,5 | 82,5 |
| MuSR | 37,1 | 70,0 | 30,0 |
| SATA-Bench | 1,3 | 50,0 | 50,0 |
| CRUXEval | 37,0 | 87,5 | 12,5 |
| CLadder | 50,0 | 90,0 | 10,0 |
| MMLU-Pro | 11,1 | 62,5 | 37,5 |
| BBH | 31,0 | 95,0 | 5,0 |
| **macro** | | **70,25** | **29,75** |

Onde a referência já está perto de 100 (WinoGrande, BBH, CLadder, CRUXEval) o pensador só pode empatar ou perder; o +10
tem que vir de GPQA, ChessBench, MuSR, MMLU-Pro e SATA-Bench. Como várias provas têm linhas com mais de uma pergunta, o
macro por linha cai mais rápido do que a acurácia por pergunta: no ensaio, um pensador **sintético** que acerta cada pergunta
com probabilidade 0,8 fica em 73,0 de macro e o relatório dá `FAIL (delta_below_min, ci_low_not_above_zero)`, delta +2,7 e IC
[-3,0, +8,5]. Isso é um ensaio da ferramenta, **não um resultado sobre o Qwen**. Nele a meia-largura do IC foi de uns 5,8
pontos, então com 400 linhas o critério que manda é o +10; o piso acima de 0 vem junto quando o delta passa de uns 6.

Contra os modelos Apache (d3, d3-turbo, Amos) use `--models d3,d3-turbo,amos` no `compare`: o veredito oficial do GT é contra
o melhor dos 8, mesmo que ele tenha licença não comercial (StartLux e Lychee são CC BY-NC, ADR 0004).

## Amostra e pacote

- **Sorteio determinístico.** Semente 20261021; cada caso recebe o rank `sha256("<semente>|case|<prova>|<id do caso>")` e a
  amostra de tamanho n são os n menores ranks, então a de 40 é prefixo da de 100 e a ordem das entradas não muda nada. Uma
  requisição por caso (o GSM8K v0.3 tem 4 versões do mesmo problema; o bootstrap precisa de unidades independentes). O pacote
  vem em rodízio entre as provas: qualquer prefixo de uma rodada ainda é uma amostra estratificada.
- **400 linhas, não 440.** O HLE (catálogo 45) é gated no Hub e a suíte reconstruída não o tem (`missing_benchmarks: [45]`).
  Ficam 10 provas x 40. **O gate não testa o HLE**, onde está a afirmação do cartão do modelo (30,8 contra 15,8 a 19,4 dos
  decisores no board); a comparação do cartão com o board já é **UNVERIFIED** na própria ADR (ambientes diferentes). Se você
  tiver acesso ao HLE e reconstruir a suíte com ele, o mesmo `sample build` sorteia as linhas dele, mas aí o manifesto é outro e
  isso já não é o gate registrado.
- **Manifesto** (`gt-sample-manifest.json`, versionado): semente, ids e hashes das linhas, sha256 do conteúdo do pacote
  (`1f5f5631f239631f703e6388c9920114ac99c828cdbf99df2c4aa8303cef6d1f`) e sha256 de cada arquivo de entrada. Sem texto de prova.
- **Pacote** (`pack.jsonl.gz`, **privado**): estado, perguntas, ouro e as probabilidades dos 8 decisores, uma linha por
  requisição. Tem GPQA e BBH (os termos pedem para não publicar em texto claro). **Nunca** vai para git, para dataset público
  nem para um notebook com saída. O `.gitignore` desta pasta ignora `*.jsonl`, `*.jsonl.gz`, `pack*/`, `work/` e `report/`,
  mas guarde o pacote fora do clone.
- **Relatório** (`report.md`, `report.json`): só contagens, acurácias e ids de linha, sem texto de prova; pode ir para a
  issue. `results.jsonl` **não**: tem o raciocínio do modelo sobre as provas.

### Reconstruir o pacote

Se você já tem o `pack.jsonl.gz`, só rode o `sample verify` (saída esperada: `[verify] OK: the pack is the manifest's sample`).
Para reconstruí-lo você precisa de duas entradas:

1. **Os 4 arquivos da suíte pública 0.3**: `selected-rows.jsonl.gz`, `added-rows.jsonl.gz`, `gsm8k-rows.jsonl.gz` e
   `excluded-questions.json` (sha256 de cada um em `inputs.suite_files` do manifesto). A suíte **não é redistribuída**: foi
   reconstruída na Fase 0A com os builders do kit (`suite/build/*`, kit no commit 25c6ec7) a partir das fontes fixadas. O script
   dessa reconstrução é do planejador e **ainda não está versionado neste repositório** (a própria ADR 0004 registra isso para o
   relatório da Fase 0A). Sem esses 4 arquivos o `sample build` não roda.
2. **Os 8 arquivos de resultados por linha**, públicos no Hub, nas revisões abaixo (cerca de 0,9 GB; o do linne-oak é um
   `.jsonl` sem compressão de 617 MB). Cole o bloco de sha256 num arquivo `SHA256SUMS` na pasta onde os baixou (um
   subdiretório por modelo) e rode `sha256sum -c SHA256SUMS`.

| modelo | repositório do Hub (dataset) | revisão | arquivo |
|---|---|---|---|
| startlux-27b | `startlux-models/decision-index-results` | `c8c1a581e280e158f7ebd120d509722c25611703` | `runs/StartLux-Decision-v1.1-27B-Instant/results.jsonl.gz` |
| d3-turbo | `vllm-sr/d3-turbo-decision-index` | `1431a76c44a0679e4f588cc4a6fe7d1976d28140` | `runs/d3-turbo/results.jsonl.gz` |
| drex-max | `nace-ai/drex-max-v1-decision-index-results` | `f59dcd6d60cbbab4a3292f12bbbdcd8a2ee021fb` | `runs/drex-max-v1/results.jsonl.gz` |
| ajev-lm6q | `andyzhang232/ajev-gemma4-31b-lm6q-decision-index` | `64cca1e8bcba847953f3684929f971f4fd752009` | `runs/ajev-lm6q-l70/results.compact.jsonl.gz` |
| d3 | `vllm-sr/d3-decision-index` | `0cfbfd61699fd4c614658d0112616cd5fea11a89` | `runs/d3/results.jsonl.gz` |
| amos | `mukaddamzaid/amos-1-decision-index-results` | `8aec0e2d21cb33a48d747e15d05aa7fe65a2aa48` | `runs/amos-1-large-0.3/results.jsonl.gz` |
| linne-oak | `PNPM244/linne-oak-1-0-decision-index` | `1225e146b543f28cba49e3db92aa143982555e54` | `runs/linne-oak-1-0/results.jsonl` |
| lychee | `LycheeAI/lychee-27b-decision-index` | `cc17da8a52d258f591d492c6850a00797bf963e7` | `runs/lychee-27b/results.jsonl.gz` |

```
862b68e358d694bfb05a04cf4ddb718d08fca60712d9883ebfba3d7c6e7c3ae0  startlux-27b/results.jsonl.gz
8791ecba4ed4e6eabd7b6708baacb0e2791adc326e0b9483ce6b44c7b505b7db  d3-turbo/results.jsonl.gz
a760bed8c5f717455d7e8179e4cbfa2e181916f055c743a8c98443e996a91813  drex-max/results.jsonl.gz
c40e2dd147d61f340d1e2fb3c3ff54f6154c4cb63d09a5e756ca768c492c7281  ajev-lm6q/results.compact.jsonl.gz
63f50bc247460128bab9b66d31035e6e0f874074068d661a58b9810922d4f59e  d3/results.jsonl.gz
b9bbf2a44f96d3411453f081fec696c1fc5bc5a2bbeca0fe4978496d83fbf894  amos/results.jsonl.gz
992e350153a51f8019234bba80e2ef431aa19d5db62386509992e0283123631f  linne-oak/results.jsonl
3dc81e62fa65c33e73de25911d6be9e1a5f7c679039d43fbfaff3d0e42689e2d  lychee/results.jsonl.gz
```

```
python -m decide.think.sample build --suite suite-0.3 --out pack-out --results startlux-27b=startlux-27b/results.jsonl.gz --results d3-turbo=d3-turbo/results.jsonl.gz --results drex-max=drex-max/results.jsonl.gz --results ajev-lm6q=ajev-lm6q/results.compact.jsonl.gz --results d3=d3/results.jsonl.gz --results amos=amos/results.jsonl.gz --results linne-oak=linne-oak/results.jsonl --results lychee=lychee/results.jsonl.gz
python -m decide.think.sample verify --manifest decide/think/gt-sample-manifest.json --pack pack-out/pack.jsonl.gz
```

O `build` escreve `pack-out/pack.jsonl.gz` e um `pack-out/manifest.json` **idêntico** ao `gt-sample-manifest.json`
(foi assim que o manifesto foi gerado; com os arquivos já no disco, o `build` levou 32 s). Cada um dos 8 arquivos tem as
34.612 requisições candidatas das 10 provas.

## Backends

Todos rodam o mesmo `thinker.py` com o mesmo protocolo (prompt, amostragem do modo thinking: temperatura 1,0, top_p 0,95,
top_k 20, min_p 0, `enable_thinking` ligado, orçamento de 16.384 tokens por pergunta). O que muda é onde o modelo roda.
Modelo e revisão são fixados por commit: `Qwen/Qwen3.8-27B` em `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (bf16, 51,7 GiB,
18 partes, Apache-2.0) e `Qwen/Qwen3.8-27B-FP8` em `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` (28,7 GiB). O relatório diz
qual par rodou e avisa se não for um dos dois.

| backend | pesos | o que foi verificado aqui | o que **não** foi |
|---|---|---|---|
| transformers + NF4 (Kaggle T4 x2) | bf16 baixado (51,7 GiB), 4 bits na carga | o executor, o lote, a divisão do lote por falta de memória e a retomada, com modelos aleatórios minúsculos na CPU (um Qwen2 e a classe híbrida do Qwen3.5); os comandos do notebook contra os parsers reais | rodar numa T4; que as versões se resolvem com o torch do Kaggle; que o modelo cabe em 2 x 15 GiB; velocidade; que o fp16 não estoura |
| endpoint OpenAI-compatível | do provedor | o cliente contra um servidor HTTP falso (JSON, SSE, 400, 401, 429 com `Retry-After`, 5xx, `reasoning_content` e `reasoning`) | qualquer provedor de verdade |
| vLLM em GPU de 80 GB ou mais (`hosted`) | FP8, 28,7 GiB | o fluxo subir, esperar, pensar, comparar e parar o servidor, com um processo falso; os flags de `vllm serve` contra a lista de flags lida **do código-fonte** do vLLM 0.31.0 | subir o vLLM de verdade |
| Modal (`modal_app.py`) | FP8, 28,7 GiB | o app importa e registra `gate` e `main` com o cliente `modal==1.6.2`; `gate` roda o `hosted` num volume, com um cliente falso | rodar na Modal: imagem, GPU, custo |

### Kaggle T4 x2, grátis: `notebooks/decide/think_gt_kaggle.ipynb`

1. Settings > Accelerator: **GPU T4 x2**; Internet: **On**. Os limites de sessão e de cota semanal do Kaggle mudam: confira os da
   sua conta (**UNVERIFIED** aqui). `TIME_BUDGET_H` (padrão 8,0) faz o notebook parar de começar linhas antes do limite.
2. Crie um **dataset privado** `think-gt-pack` com o `pack.jsonl.gz` e adicione-o como Input. Nunca o torne público.
3. Rode com `LIMIT = "3"`, leia tokens/s e ETA, e então ponha `LIMIT = ""`. O notebook fixa o código por commit
   (`SIMPLICIO_SHA`), confere a GPU, o disco e o pacote antes de gastar GPU, e termina no `GT VERDICT`.
4. Se a sessão acabar, use **Save Version**, adicione a saída como Input e aponte `RESUME_FROM` para o `results.jsonl` dela: o
   notebook o copia (só se ainda não houver um) e continua. O orçamento de tokens só pode **subir** entre rodadas.

Por que não vLLM aqui. Lendo o código-fonte do vLLM 0.31.0 (não executado): ele não registra quantização bitsandbytes, a
plataforma CUDA dele recusa bf16 abaixo da capacidade 8.0 (a T4 é 7.5) e o checkpoint FP8 (28,7 GiB) não cabe em 2 x 15 GiB.
Daí `transformers==5.5.0` com `bitsandbytes==0.50.2` (NF4, cálculo em fp16), as mesmas versões do Colab de treino do repositório
(`accelerate==1.15.0`, `huggingface_hub==1.33.0`). Duas notas lidas no código, não executadas: o `transformers` 5.5.0 cai numa
implementação lenta em PyTorch puro nas camadas Gated DeltaNet do modelo quando `flash-linear-attention` e `causal-conv1d`
não estão instalados (o notebook não os instala), e o disco precisa de mais de 58 GiB livres para o checkpoint bf16 (a célula
de conferências confere; `/kaggle/working` é limitado, por isso o cache vai para `/kaggle/temp`, **UNVERIFIED** que lá haja espaço).

**É um teste de um lado só.** Um PASS em 4 bits vale (quantizar não costuma melhorar o modelo, **UNVERIFIED**); um FAIL em
4 bits **não prova** que o modelo em bf16 ou FP8 falharia. O relatório avisa disso; confirme pelo endpoint, pelo `hosted` ou pela
Modal antes de descartar o pensador.

Dá para mudar o cálculo do NF4 com `--dtype` (`float16` é o padrão, `bfloat16` é o dtype do modelo e roda emulado numa T4, mais
lento e **não testado**). O dtype entra no protocolo e no cabeçalho dos resultados: outro dtype exige outro `--out`.

### Endpoint OpenAI-compatível: `notebooks/decide/think_gt_api.ipynb`, `MODE = "endpoint"`

Precisa de `BASE_URL`, do id do modelo que o endpoint conhece (`MODEL`) e da chave como segredo do Colab ou do Kaggle, ou na
variável de ambiente `KEY_NAME`. A chave nunca é impressa, nunca vai na linha de comando e não é escrita nos resultados. Num
terminal, leia-a sem deixar no histórico:

```
read -rs THINK_API_KEY && export THINK_API_KEY
python -m decide.think.thinker --backend openai --base-url https://SEU-ENDPOINT/v1 --model Qwen/Qwen3.8-27B-FP8 --revision 017b9c7af6b5689d5dd426a76e0bc077eb5ca20a --concurrency 8 --max-tokens 16384 --pack pack.jsonl.gz --manifest decide/think/gt-sample-manifest.json --out work/results.jsonl --limit 3
```

- O **pré-voo** faz uma chamada de 1 token e diz se o raciocínio volta separado da resposta. O `--revision` aqui só é
  **declarado** (vai para o relatório): quem escolhe o que roda é o servidor.
- **Confirme que o endpoint pensa.** Provedores diferentes ligam o thinking de jeitos diferentes, e vários ignoram
  `chat_template_kwargs`. O sinal de que não pensa é o número de tokens por linha na casa das dezenas: a sonda imprime os
  tokens de cada linha, o fim do `thinker` avisa (`WARNING: ... the model barely thought`) quando a média fica abaixo de 100
  e o relatório avisa (`AVISO: a mediana é de ... tokens por linha`). Uma rodada assim não é o gate.
- Se o endpoint responder HTTP 400 a `top_k`, `min_p`, `repetition_penalty` ou `chat_template_kwargs`, use `--standard-only` e
  ligue o thinking pelo campo do próprio provedor em `--extra-body '{"campo": valor}'`. **Não ponha chave em `--extra-body`**: o
  JSON vai para o cabeçalho dos resultados e para o relatório. O `thinker` recusa chaves com cara de segredo, mas é só uma rede
  de segurança.
- **Custo** = tokens de saída da amostra inteira x preço por token. Os tokens por linha só aparecem na sonda. Não há preço
  nenhum estimado aqui.

### vLLM numa GPU de 80 GB ou mais: `python -m decide.think.hosted` (ou `MODE = "local-vllm"` no notebook da API)

Para uma A100 80 GB, uma H100 ou uma RTX PRO 6000 de 96 GB, alugada ou sua. Sobe `vllm serve` com o checkpoint FP8 fixado,
thinking ligado e o parser `qwen3` (o raciocínio chega separado em `reasoning_content`), **só em loopback**, espera ele
responder, roda o `thinker`, escreve o relatório e **para o servidor**, inclusive se algo der errado.

```
pip install vllm==0.31.0
python -m decide.think.hosted --pack pack.jsonl.gz --work work --limit 3
python -m decide.think.hosted --pack pack.jsonl.gz --work work
python -m decide.think.serve --print
```

- `--work` guarda `results.jsonl`, `report/` e `vllm.log`. Rodar de novo com os mesmos argumentos retoma.
- Códigos de saída: **0** terminou; **2** o pacote não é o do manifesto, ou o `thinker` recusou o protocolo (outro modelo, outro
  prompt, orçamento menor no mesmo arquivo); **3** o `thinker` parou após erros seguidos (o relatório sai mesmo assim);
  **4** o servidor não subiu (as últimas linhas do `vllm.log` são impressas).
- `serve --print` só mostra o comando, para quem sobe o servidor por outro meio. O contexto do servidor é
  `max(32768, orçamento + 8192)`.
- Memória, por conta a partir do `config.json` do modelo (calculada, **não medida**): só 16 das 64 camadas guardam KV (uma de
  cada 4, com 4 cabeças de KV de dimensão 256), cerca de 64 KiB por token em 16 bits; uma sequência de 21 mil tokens ocupa uns
  1,3 GiB, mais uns 0,15 GiB de estado das camadas lineares. Com 16 sequências em paralelo isso é da ordem de 10 a 23 GiB além
  dos 28,7 GiB do FP8. Numa GPU de 40 GB vai apertado, e a velocidade cai porque o cache cabe em poucas sequências.

### Modal: `modal run -m decide.think.modal_app`

```
pip install modal==1.6.2 && modal setup
modal run -m decide.think.modal_app --pack pack.jsonl.gz --limit 3
modal run -m decide.think.modal_app --pack pack.jsonl.gz
modal volume get simplicio-think-gt-state gt/results.jsonl .
```

Rode **em modo módulo, a partir da raiz do repositório** (em modo script o contêiner não acha o pacote `decide`). O pacote é
conferido contra o manifesto **na sua máquina, antes de alugar qualquer coisa**. O checkpoint FP8 é baixado uma vez para um Volume
e reaproveitado; os resultados são gravados no Volume a cada minuto, então um contêiner que a Modal troque retoma de onde
parou. A GPU padrão é H100; troque com `SIMPLICIO_THINK_GPU=A100-80GB modal run ...`. O pensador para de começar linhas 1 hora
antes do limite de 12 h do contêiner. O app **nunca rodou na Modal** (**UNVERIFIED**): trate a primeira execução, com
`--limit 3`, como teste. O custo é a tarifa da GPU escolhida (página de preços da Modal) vezes as horas que a sonda indicar;
não foi estimado.

## Tempo e custo

**Medido.** O pacote real tem 400 prompts com 750.567 caracteres no total (mediana 1.075; máximo 14.658, uma linha do
SATA-Bench; MuSR e SATA-Bench são as provas de prompt longo, em torno de 5.000 caracteres, e as outras ficam entre 700 e
1.300 na média). Com 3 a 4 caracteres por token (o tokenizador de verdade não rodou, **UNVERIFIED**) são uns 190 a 250 mil
tokens de entrada no total e menos de 5 mil na maior linha. A entrada pesa pouco; o custo é a saída. O orçamento é de 16.384
tokens por pergunta, então o teto aritmético é de 400 x 16.384 = 6.553.600 tokens gerados.

**Não medido, em nenhum backend:** tokens por linha do pensador de verdade, tokens por segundo, horas e dinheiro. O Qwen3.8-27B
com amostragem a temperatura 1,0 vai gastar muito mais tokens em GPQA, MuSR e xadrez do que em WinoGrande; a sonda de 3 linhas
(GPQA, WinoGrande, GSM8K, as três primeiras do rodízio) diz se roda, a de 10 (uma por prova) dá uma média melhor.

**Como estimar com a sonda.** A cada linha o log imprime `tokens, segundos | tok/s ETA`. Horas totais = 400 x (tokens por linha)
/ (tokens por segundo agregados) / 3.600. Dinheiro = horas x tarifa da GPU, ou tokens totais x preço do provedor.

**O que a física permite (teto, não expectativa; UNVERIFIED).** A decodificação gasta, por token gerado, uma leitura de todos os
pesos da memória da GPU. Na T4 (320 GB/s de folha de dados) com uns 15 GiB lidos por token no 4-bit, o teto é da ordem de 20
tokens/s **por sequência**, o que dá no mínimo `tokens por linha / 180` horas para as 400 linhas com uma sequência por vez
(3.000 tokens por linha: 17 horas, no melhor caso). Lotes maiores podem ajudar, mas o kernel de 4 bits do bitsandbytes e a
implementação lenta das camadas DeltaNet tendem a deixar o real **bem abaixo** do teto. Numa H100 (3,35 TB/s, 30,8 GB de pesos
FP8) o teto de memória passa de 1.000 tokens/s com 16 sequências em paralelo (1,2 milhão de tokens, 400 linhas x 3.000, em uns
12 minutos); a 10% do teto seriam duas horas. Em resumo: a T4 grátis provavelmente leva várias sessões, uma GPU grande leva de minutos a poucas
horas. Só a sonda diz.

## Como ler o relatório

A última linha impressa é `GT VERDICT: PASS|FAIL|INCONCLUSIVE (motivos) | thinker X% vs <referência> Y% | delta +D pts, 95% CI
[a, b] | rule: delta >= +10.0 and CI low > 0 | N of 400 rows` (resumida aqui: depois do par de acurácias vem
`(best decision model, macro over 10 benchmarks)`, e um FAIL que ainda pode virar PASS acrescenta `the CI still holds +10.0:
more rows could pass`; sem linha nenhuma pontuada a linha é só `GT VERDICT: INCONCLUSIVE | <motivo> (0 of 400 rows)`; se
faltar linha, a contagem final ganha `, PARTIAL: ...`). O `report.md` tem, em ordem:

1. `Thinker: ...`, qual modelo, revisão, backend, quantização, orçamento e se o thinking estava ligado, com `AVISO` quando a
   rodada não é o gate (outro modelo ou revisão, quantizado, thinking desligado).
2. Acurácia por prova (e a diferença com IC por prova; com 40 linhas o IC por prova é largo e não decide nada), habilidade
   corrigida pelo acaso, e o pensador contra cada decisor.
3. Linha a linha contra a referência: só o pensador acerta, só ela acerta, ambos, nenhum, e o teto de um roteador perfeito
   entre os dois (um teto, não uma medida).
4. **Gate offline (exploratório).** Manda ao pensador só as linhas em que a referência tem confiança abaixo de tau. O tau é lido
   nas mesmas linhas em que é medido, então o meio da curva é **otimista**: é o teto do que um gate por confiança poderia dar com
   esses dois sistemas, não uma previsão. A coluna "cabe na regra de latência" marca os tau cuja fração enviada fica dentro da
   folga abaixo.
5. **Custo e falhas do thinker.** Linhas, ilegíveis, orçamento esgotado, falhas; tokens e segundos (total, mediana, p80, máximo).
   Avisos: mais de 10% das linhas bateram no orçamento (o FAIL pode ser do orçamento, não do modelo: refaça com `--max-tokens`
   maior, só as cortadas são refeitas), saída ilegível, mediana de tokens baixa demais, execução parcial.
6. **Folga de latência.** A regra do kit é latência média, mediana e p80 de até 1.000 ms por requisição numa RTX PRO 6000, uma
   requisição por vez (ADR 0004, fato 4). Se o pensador atende uma fração f das requisições com tempo médio t, a média passa de
   1.000 ms quando f > 1.000 ms / t (ignorando o tempo do decisor). Como a média entra na regra, uma fração pequena com raciocínio
   longo já estoura. O t vem do backend que rodou o pensador, não da RTX PRO 6000: é um limite superior e **UNVERIFIED** para o board.

O que fazer com cada veredito:

| veredito | o que significa | próximo passo |
|---|---|---|
| **PASS** com thinking ligado e o par modelo/revisão fixado | o pensador entra no plano (ADR 0004, decisão 4) | guarde `report.md` e `report.json`, abra a issue da fase seguinte (o híbrido, abaixo) |
| **PASS** em 4 bits | vale, o teste é de um lado só | idem |
| **FAIL** com `can still pass` | o IC ainda comporta +10 | veja os avisos (orçamento, ruído, thinking desligado) e refaça o que eles pedem; mais linhas exigem outro pacote e outra issue |
| **FAIL** sem isso, em bf16/FP8 | o pensador não paga | ADR 0004: a F3 só faz autodestilação, o F6 treina o decisor sem bloco de raciocínio e o plano segue só-decisor |
| **FAIL** em 4 bits | não prova nada sobre bf16/FP8 | confirme por endpoint, `hosted` ou Modal |
| **PARTIAL** | faltam linhas | rode de novo para terminar; o veredito parcial só mostra a direção |

O que o veredito **não** diz: (a) o bootstrap reamostra linhas, não a aleatoriedade da geração (temperatura 1,0): duas rodadas
podem dar números diferentes; (b) escolher o melhor de 8 decisores nas mesmas linhas favorece a referência, o que é o lado
cauteloso para um gate; (c) a suíte pública não é o Full privado do board; (d) o HLE não está na amostra.

## Se a saída for ruído

O relatório avisa `AVISO: N linhas com saída ilegível` quando, nos primeiros 2.000 caracteres do raciocínio ou da resposta, um
mesmo caractere passa de 40% ou as letras e dígitos ficam abaixo de 40%. Essas linhas **não dizem nada sobre o modelo**, e um FAIL
feito delas não é um veredito. Na T4 a causa provável é o fp16 estourando (**UNVERIFIED**: não há T4 aqui): o modelo é
publicado em bf16, que tem um alcance bem maior que o do fp16, e um estouro leva o modelo a repetir um token.

O que fazer:

1. Não use esse FAIL. Guarde o `results.jsonl` (é privado) e o relatório.
2. Refaça a rodada em outro numérico, com **outro** `--out`, porque o dtype faz parte do protocolo e o `thinker` recusa misturar
   (o kit não refaz só as linhas ruidosas: as 400 rodam de novo). Na T4, `--dtype bfloat16` (emulado, mais lento, não testado).
   O caminho que resolve de verdade é rodar o mesmo pacote num backend com bf16 ou FP8 nativo: endpoint, `hosted` ou Modal.
3. Se o ruído vier de um endpoint hospedado, o problema é do provedor: troque de endpoint ou de modelo servido.

## Desenho do híbrido (ADR 0004, esboço)

Isto é um esboço do que o PASS destrava, não parte do gate.

- **Gate.** O decisor de um passe sempre responde. Um gate decide quando o pensador também roda, pela **confiança do decisor**
  (a maior probabilidade da pergunta, abaixo de um tau) e pelo **tipo de pergunta** (raciocínio ou não). O pensador raciocina e
  o decisor lê estado + raciocínio e devolve as probabilidades. A curva "Gate offline" do relatório é o teto do gate só por
  confiança nestas 400 linhas; o tipo de pergunta ainda não foi medido (**UNVERIFIED**).
- **Bloco `Raciocínio:`.** O formato está na seção 10 de `decide/data/SCHEMA.md` (issue #69; estava na branch `decide/69-treino`,
  ainda não na `main`, quando isto foi escrito): chave opcional `reasoning` no pedido, mostrada como uma parte `Raciocínio:`
  entre o estado e a primeira pergunta, e a forma dobrada `estado + "\n\nRaciocínio:\n" + raciocínio` para quem só conhece
  `state`. O módulo `trace.py` é o lado do pensador: `from_record(linha)` tira o texto de uma linha do `results.jsonl`
  (`mode` `reasoning`, `final` ou `both`), limpa, neutraliza linhas que seriam lidas como o marcador e, se passar do limite,
  mantém o **fim** (onde está a conclusão) atrás de um aviso de quantos caracteres saíram; `attach_trace` e `split_trace`
  escrevem e lêem a forma dobrada. Onde o `trace.py` e o SCHEMA.md divergirem, vale o SCHEMA.md. O SCHEMA.md **mantém** um
  raciocínio cortado pelo orçamento (`finish_reason: length`); o `from_record` o recusa por padrão, então passe
  `allow_truncated=True` ao gravar traços do corpus. Para gerar traços do corpus de treino da F2 falta um leitor de corpus
  que este kit não tem: ele lê só o pacote do GT.
- **Memória numa GPU de 96 GB (ADR 0004).** Pensador em FP8 (28,7 GiB) mais decisor em bf16 (uns 52 GB, segundo a ADR)
  somam uns 80 GiB de pesos; o KV do pensador de uma requisição de 21 mil tokens é da ordem de 1,5 GiB (conta acima).
  Sobra algo como 14 GiB para o decisor, a ativação e os contextos CUDA. É uma conta, **não uma medida** (**UNVERIFIED**), e
  ainda não conta o KV do raciocínio de várias requisições.
- **Latência.** A "Folga de latência" do relatório é o primeiro número do risco da ADR: com thinking de milhares de tokens por
  linha, a fração de requisições que podem pensar sem estourar a média de 1.000 ms é pequena. Onde estiver o Reasoning board do
  kit (as regras não estão publicadas, **UNVERIFIED**), o modo com raciocínio pode ter lugar próprio.

## Dados de avaliação, nunca de treino

A amostra do GT e a suíte pública são dado de avaliação. Os pedidos dela e os traços do pensador sobre eles **nunca** entram num
corpus de treino (ADR 0004, item 6). O aviso está no manifesto, no cabeçalho dos resultados, no relatório e nos notebooks.

## Verificação

Tudo na CPU, sem GPU e sem rede (exceto onde indicado):

```
python3 -m unittest tests.test_contract_75 tests.test_flow_75
python -m decide.think.mutation_check
```

- `tests/test_contract_75.py`: os contratos do pensador, da comparação, do bootstrap, do veredito, da retomada, do `serve`, do
  `hosted`, do app da Modal, dos dois notebooks (os comandos das células são lidos com os parsers de verdade) e deste README.
- `tests/test_flow_75.py`: o fluxo ponta a ponta contra um servidor HTTP falso, um Qwen2 aleatório de 2 camadas (precisa de
  `torch`, `transformers` e `tokenizers`; sem eles esses testes são pulados), as células dos notebooks executadas com
  dublês e o app da Modal com um `modal` falso (e com o cliente de verdade, se estiver instalado).
- **Mutação.** `python -m decide.think.mutation_check` planta um bug de uma linha numa **cópia** do kit (leitura da resposta,
  bootstrap, veredito, pontuação, retomada, amostra, `serve`, `hosted`, relatório, notebooks, README) e exige que algum teste
  falhe. Um mutante que ninguém percebe é um teste que falta e faz o comando sair com 1. Imprime `N of N mutants killed`.
- **Linhas reais.** O pacote de 400 linhas foi reconstruído a partir dos 8 resultados públicos e confere com o manifesto; o
  prompt e a leitura da resposta fecham o ciclo nas 400 (a resposta de um modelo perfeito, escrita no formato pedido, volta com
  as chaves do ouro, 0 falhas); o `compare` roda sobre elas (resultado do ensaio acima).

## O que não foi verificado

Marcado **UNVERIFIED** acima; em um lugar só:

1. Qualquer execução numa GPU (T4, A100, H100, RTX PRO 6000): nenhum tempo, nenhuma memória, nenhum token por segundo.
2. Que as versões fixadas (`transformers==5.5.0`, `accelerate==1.15.0`, `bitsandbytes==0.50.2`, `huggingface_hub==1.33.0`) se
   resolvem com o torch do Kaggle, e que o NF4 do Qwen3.8-27B carrega e gera em 2 x T4.
3. A leitura do código do vLLM 0.31.0 (sem bitsandbytes, sem bf16 abaixo da capacidade 8.0) e a lista de flags de `vllm serve`:
   foram lidas, não executadas. O vLLM nunca serviu o checkpoint FP8 daqui.
4. A Modal (imagem, GPU, custo) e qualquer provedor hospedado.
5. Que o fp16 seja a causa do ruído na T4.
6. O custo em dinheiro de qualquer backend e a latência no board (RTX PRO 6000, uma requisição por vez).
7. O formato do bloco `Raciocínio:` contra o SCHEMA.md da issue #69 (lido, não executado) e a conta de memória do híbrido.
8. A comparação do cartão do modelo (GPQA 89,2, HLE 30,8) com o board (52,0 a 61,7 e 15,8 a 19,4): ambientes diferentes.

## Arquivos

| arquivo | o que é |
|---|---|
| `gt-sample-manifest.json` | o manifesto da amostra (ids e hashes, sem texto) |
| `common.py` | constantes (semente, fixações, regra do gate, acaso por prova) e utilitários de JSONL |
| `sample.py` | sorteio determinístico, pacote e manifesto; `sample build` e `sample verify` |
| `prompt.py`, `answers.py` | o que o pensador recebe e como a resposta dele é lida (rótulos A-Z; só depois do último `</think>`) |
| `thinker.py` | o executor: backends `openai` e `transformers`, retomada por linha, orçamento, erros |
| `compare.py` | acurácia, habilidade, bootstrap pareado, veredito e relatório |
| `serve.py`, `hosted.py`, `modal_app.py` | `vllm serve` fixado, o fluxo inteiro numa GPU grande e o app da Modal (o único que importa `modal`) |
| `trace.py` | o bloco `Raciocínio:` (lado do pensador) |
| `mutation_check.py` | o teste de mutação |
| `../../notebooks/decide/think_gt_kaggle.ipynb`, `../../notebooks/decide/think_gt_api.ipynb` | os dois notebooks |

Os notebooks ficam em `notebooks/decide/` porque o README da raiz (que o teste `tests/test_contract_30.py` exige que cite
todo `notebooks/*.ipynb`) não é deste diretório; o `tests/test_flow_30.py` aceita `notebooks/decide/<nome>.ipynb`. Depois do
merge, troque `SIMPLICIO_SHA` nos dois notebooks pelo commit de merge em `main`.
