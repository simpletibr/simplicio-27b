# ADR 0007: Rota 2, duas partes de famílias diferentes e ponto de partida revisto

- **Status:** Proposta. Nenhum resultado do modelo foi medido numa GPU. O que está medido é o placar de 2026-11-06 e um estudo em CPU sobre resultados públicos (contexto 4). Cada passo só avança quando o gate dele for medido. As decisões que cabiam ao dono foram tomadas no mesmo dia, por delegação dele (decisão 10).
- **Data:** 2026-11-10
- **Plano:** [decide/PLAN.md](../../decide/PLAN.md), versão 2.2, seção 11.
- **Relacionadas:** #64 (épico), #66 (F1), #67 (F2), #70 (F5), #71 (F6), #72 (F7), #75 (F2b); [ADR 0003](0003-simplicio-decide-27b.md), [ADR 0004](0004-pensar-sob-demanda.md), [ADR 0005](0005-plano-b-base-alternativa.md).
- **Escopo:** emenda os itens 2 (ponto de partida) e 4 (pensar sob demanda como aposta principal) da decisão da ADR 0004 e o item 1 da decisão da ADR 0003 (as referências numéricas da meta; o alvo "#1" não muda), e acrescenta ao desenho uma segunda parte de outra família. Não muda a regra de evidência, a regra de licença (só pesos Apache-2.0, só dados com licença comercial), o protocolo da parte Qwen nem os gates G0 a G5. Não dispara o Plano B1 da ADR 0005 (treinar em Gemma-4-31B): usa um decisor Gemma público como está.
- **Evidência:** [0007-evidencia/](0007-evidencia/README.md), com os scripts, os relatórios e as revisões de cada fonte.

Regra de evidência: só número medido entra como fato, com a fonte ao lado. Projeção, conta ou leitura de cartão não reproduzida leva a marca **UNVERIFIED** e não decide nada sozinha.

## Contexto

Fatos medidos em 2026-11-10, com a fonte de cada um.

1. **O placar andou.** `index.json` do Space, gerado em 2026-11-06T09:05:08Z ("Decision Index 0.4.1"), 274 modelos; eram 208 em 2026-10-20 (ADR 0003).

   | # | modelo | Full | mesmas habilidades | domínios novos | índice público | base | tipo | licença no Hub |
   |---|---|---|---|---|---|---|---|---|
   | 1 | REAX 1.3 | 73,98 | 69,29 | 63,75 | 73,71 | duas partes: Gemma-4-31B e Qwen 27B | técnica de inferência | Apache-2.0 |
   | 2 | GLiDE 27B | 73,37 | 70,07 | 61,57 | 72,49 | Qwen3.8-27B | ajuste completo | Fastino Community |
   | 3 | KnowLine 31B Gen1.5 | 71,91 | 67,07 | 62,30 | 70,50 | Gemma-4-31B | LoRA | Apache-2.0 |
   | 4 | FlexDM-LoRA 2.1 | 71,61 | 66,36 | 62,57 | 68,99 | Gemma-4-31B | LoRA | CC BY-SA 4.0 |
   | 5 | Sieve 31B v3 | 71,10 | 68,53 | 58,86 | 71,68 | Gemma-4-31B | ajuste completo | Apache-2.0 |
   | 6 | Turion Jev 31B | 70,47 | 65,34 | 61,50 | 66,16 | Gemma-4-31B | LoRA | Apache-2.0 |
   | 7 | d3-ultra | 70,18 | 66,77 | 59,16 | 71,57 | Qwen3.8-27B | ajuste completo | Apache-2.0 |
   | 13 | mammoth-1 | 68,44 | 66,26 | 56,27 | 75,34 | Qwen3.5-27B | ajuste completo | não conferida |
   | 17 | pplx-decider-v1.2 | 67,98 | 64,30 | 57,76 | 67,74 | Qwen3.8-27B | ajuste completo | Apache-2.0 |

   - O mammoth-1 era o #1 em 2026-10-20 e o pplx-decider-v1.2, o #2 (ADR 0003). No arquivo de hoje o líder está 5,54 pontos acima do mammoth-1 e 6,00 acima do pplx.
   - No top 10 há 6 modelos com base Gemma-4-31B, 3 com Qwen3.8-27B e 1 de duas partes.
   - O campo `rank_tie` dá o mesmo grupo (1) ao REAX 1.3 e ao GLiDE 27B. A página de metodologia do Space diz que modelos a até 0,9 ponto do primeiro do grupo dividem o posto (contexto 9). Os scripts que montam o placar não estão no Space nem no kit e não foram lidos.
   - O gate G4 pede o pplx + 1,5 no privado caseiro. Se esse ganho passasse inteiro para o Full (**UNVERIFIED**), 69,48 ficaria entre o #10 (69,75) e o #11 (69,14).

2. **Como os líderes chegaram lá.** Lido nos cartões e nos arquivos de configuração dos repositórios, nas revisões do apêndice. Nada disto foi reproduzido por nós.
   - **REAX 1.3:** duas partes que rodam em paralelo numa RTX PRO 6000, as duas em FP8, com pico de 88,5 GiB; cada pedido vai às duas e a resposta é a média, meio a meio, das probabilidades. A parte G é um ajuste da linha Gemma misturado com o KnowLine-31B-Gen1; a parte Q é um ajuste da linha Qwen misturado com o d3-ultra. O cartão diz "blended"; que a mistura é nos pesos é leitura nossa, pela contagem de parâmetros abaixo (um modelo por família). Latência medida pelos autores: 207 / 655 / 741 ms (mediana, média, p80). O `served_params` do placar, 57.358.419.036, é a soma exata de um decisor Qwen3.8-27B (26.085.330.160) e de um Gemma-4-31B (31.273.088.876) do mesmo arquivo.
   - **GLiDE 27B:** um passe por pergunta. O `decision_config.json` traz `midloop`: os blocos de camadas 24 a 35 e 36 a 47 rodam 2 vezes dentro do mesmo passe, sem parâmetro novo e sem gerar token.
   - **KnowLine 31B:** LoRA de posto 32 (alpha 64), 1 época. Pedido de até 3.000 tokens é lido duas vezes; acima disso, a Gen1.5 põe as perguntas antes do estado. A Gen1.5 tem os mesmos pesos da Gen1 e subiu de 70,13 para 70,50 no índice público sem treino (número dos autores).
   - **Sieve 31B v3:** ajuste completo com 709.604 registros (957.775 perguntas), 5.544 passos, cerca de 97 horas de GPU em 2 B300. Pesos finais = 0,3 × base + 0,7 × treinado. Perda de entropia cruzada + 0,3 × Brier, opções embaralhadas, opção "nenhuma das anteriores" e distratores acrescentados no treino, duas ordens de opções na inferência, 22 temperaturas por categoria de pergunta.
   - **Turion Jev 31B:** LoRA de posto 64 (alpha 128), 450 passos, uma B300 por cerca de 7,3 horas. 157.106 linhas de treino, das quais 106.680 vêm de 439 fontes do `tasksource/tasksource-jev-typed-decisions` (só as marcadas para uso comercial).

3. **Latência.** O README do kit (commit `25c6ec7`) deixa fora do placar o modelo cuja mediana, média ou p80 passa de 1.000 ms por pedido numa RTX PRO 6000. No `index.json`, mediana e média em ms: pplx-decider-v1.2 144,4 / 415,1; GLiDE 122,4 / 387,6; d3-ultra 200,5 / 423,7; KnowLine 202,4 / 604,8; Sieve 231,8 / 663,0; Turion 143,8 / 299,2; REAX 1.3 245,9 / 790,4.

4. **Estudo de ensemble entre famílias (CPU, só resultados públicos).** Método e conferências em [0007-evidencia/README.md](0007-evidencia/README.md).
   - **O que entrou:** 12 modelos individuais do topo (5 da família Qwen 27B, 7 da família Gemma-4-31B) e o REAX 1.3, em 30 dos 37 benchmarks do índice público, 94.173 pedidos. O gabarito foi reconstruído com os adaptadores do kit a partir das fontes fixadas.
   - **Conferência:** o `payload_sha256` de cada pedido reconstruído é igual ao que cada modelo respondeu, e o score e o skill calculados pelo kit são iguais aos do placar em 4 casas, nos 13 runs e nos 30 benchmarks.
   - **Régua:** "índice parcial" é a fórmula do índice público 0.3 só nesses 30 benchmarks. Ele ordena os 13 modelos quase como o índice público do placar (Spearman 0,989). Ganho = par, com média meio a meio das probabilidades, menos o melhor membro.

   | tipo de par | todos os pares | membros a até 2 pontos um do outro | só membros Apache-2.0 |
   |---|---|---|---|
   | Gemma + Qwen | +0,63 (35 pares; 77% positivos; de −1,56 a +2,32) | +1,52 (13 pares; 13 positivos; de +0,62 a +2,32) | +0,72 (15 pares) |
   | Gemma + Gemma | +0,31 (21 pares; 62% positivos) | +0,99 (7 pares; 6 positivos) | +0,48 (10 pares) |
   | Qwen + Qwen | −0,48 (10 pares; 50% positivos) | +0,48 (6 pares; 5 positivos) | −1,50 (3 pares; nenhum positivo) |

   | par | índice parcial | ganho | melhor / pior que o melhor membro, por benchmark | soma das latências médias |
   |---|---|---|---|---|
   | KnowLine Gen1.5 + d3-ultra (Apache) | 80,77 | +2,32 | 23 / 6 | 1.028 ms |
   | Sieve v3 + d3-ultra (Apache) | 81,00 | +1,84 | 24 / 5 | 1.087 ms |
   | KnowLine Gen1.5 + d3-turbo (Apache) | 81,30 | +1,57 | 22 / 6 | 947 ms |
   | Turion + d3-ultra (Apache) | 79,62 | +1,17 | 18 / 10 | 723 ms |
   | GLiDE + KnowLine Gen1.5 | 80,91 | +0,73 | 20 / 10 | 992 ms |
   | GLiDE + d3-ultra (mesma família) | 80,43 | +0,25 | 20 / 9 | 811 ms |
   | d3-ultra + d3-turbo (mesma família) | 79,59 | −0,14 | 22 / 7 | 766 ms |
   | REAX 1.3 (referência) | 80,97 | | | 790 ms (média do placar) |

   - **Pares de modelos Apache públicos, sem treino nenhum, chegam ao nível público do líder:** KnowLine Gen1.5 + d3-ultra faz 80,77, Sieve v3 + d3-ultra 81,00 e KnowLine Gen1.5 + d3-turbo 81,30, contra 80,97 do REAX 1.3 nos mesmos benchmarks.
   - **Meio a meio basta:** com um peso ajustado numa metade dos pedidos e medido na outra, o peso ficou entre 0,5 e 0,6 em 8 pares testados. KnowLine + d3-ultra ganhou +2,27 numa metade e +2,39 na outra.
   - **Três partes rendem pouco a mais:** de +0,2 a +0,8 sobre o melhor par formado pelos mesmos membros (4 trios testados).
   - **Um benchmark pode inverter um recorte pequeno:** no primeiro recorte, de 6 benchmarks, o melhor membro era o KnowLine e o PhishNChips pesava 20%; KnowLine + d3-ultra ficou −1,03 [−1,85; −0,22] abaixo dele, por perder 18,0 pontos de skill nesse benchmark. Nos 30 benchmarks o PhishNChips pesa cerca de 6%, o melhor membro passa a ser o d3-ultra e o mesmo par fica +2,32 acima. O intervalo é de bootstrap pareado, 2.000 reamostragens, e só existe para o recorte de 6.

5. **Índice público e Full não andam juntos no topo.** Entre os 13 modelos do estudo, a correlação de postos entre o índice público do placar e o Full é 0,41. Jeffan-31b-1710 (público 73,13, Full 69,75), Nimble v4.1 Max (72,74 e 70,03) e d3-turbo v1.1 (71,62 e 67,31) vão bem no público e caem no Full; KnowLine (70,50 e 71,91) e Turion (66,16 e 70,47) fazem o contrário.

6. **O único sinal do lado privado é indireto.** No placar, o Full do REAX 1.3 fica 2,07 acima do melhor modelo público de que ele descende (KnowLine Gen1.5): +2,22 nas mesmas habilidades e +1,45 nos domínios novos. No índice público ele fica 1,63 abaixo do melhor deles (mammoth-1). As partes do REAX também levam ajustes de terceiros que não estão no placar, então isso não isola o efeito do ensemble.

7. **O corpus de hoje mira o que o ponto de partida já sabe.** O `decide/data/sources.yaml` libera 7 fontes de treino, com teto somado de 144.007 linhas. Três são os conjuntos de treino de benchmarks do próprio índice, nos quais o pplx-decider-v1.2 já tem skill público de 0,94 (CLINC150), 0,91 (BANKING77) e 0,81 (GSM8K). Na mesma base, o GLiDE passa o pplx sobretudo em SATA-Bench (+0,230), PhishNChips (+0,289), ACOS (+0,238) e MuSR (+0,157) no skill público, e em MMLU-Pro (+0,356) e GPQA (+0,140) no equivalente privado que o placar mostra. O PLAN (F2) previa "tasksource com licença comercial"; nenhuma fonte de treino do tasksource está no `sources.yaml`.

8. **O repositório.** Na `main` em `b46c20b`: nenhum gate medido (G0 a G5, GT), nada rodou numa GPU (RUNBOOK), as sessões 2, 4 e 7 do RUNBOOK não rodam da `main`, e o último commit é de 2026-10-23.

9. **Regras de submissão e preço da placa, lidos em 2026-11-10.** Fontes e revisões no apêndice.
   - **Como se submete** (README do kit, commit `25c6ec7`): rodada pública 0.3 completa, enviada a um dataset do Hub, e um PR com o nome do modelo, o repositório dele no Hub, o dataset de resultados, o motor e o commit, o hardware e as configurações de inferência. Os mantenedores rodam o modelo eles mesmos nas partes privadas: os pesos têm de estar no Hub, e o código de inferência usado (ou um servidor `/v1/systemone`) tem de estar disponível.
   - **Latência:** medida pelos mantenedores numa RTX PRO 6000, em processo único, um pedido por vez, pelo caminho mais rápido que o código do modelo oferece, numa amostra privada.
   - **Regras duras** (sete, em `data/methodology.json` do Space), entre elas: sem truncar, sem filtrar opções, sem ajustar prompt (cada participante roda com os próprios auxiliares de inferência, prompts e regras de leitura), sem repetir um pedido porque a resposta saiu errada, e pedido sem resposta conta como erro.
   - **Categorias:** "inference technique" é o participante que não publicou pesos próprios e serve um checkpoint de fábrica de um fornecedor de base (Qwen, Google e outros) com código próprio de decodificação ou pontuação. No `index.json` de 2026-11-06 há 28 participantes com esse `meta.kind`; o REAX 1.3 é o de maior Full entre eles. Um modelo aparece uma vez: trocar o checkpoint aposenta a rodada anterior.
   - **Não há regra escrita** sobre submissão de duas partes nem sobre servir ajustes publicados por terceiros. Quem aceita é a revisão do PR.
   - **Empate:** modelos a até 0,9 ponto do primeiro do grupo dividem o posto (página de metodologia).
   - **Reasoning board:** a página inicial do placar o anuncia como "coming soon", para modelos que podem pensar antes de responder.
   - **Placa:** o `hf-job` do kit roda por padrão no Hugging Face Jobs, no sabor `rtx-pro-6000` (uma RTX PRO 6000 de 96 GB), com limite de 72 horas; o `run` manda 8 pedidos ao mesmo tempo por padrão. A API de hardware do Hugging Face Jobs dá US$ 0,045833 por minuto para esse sabor, ou US$ 2,75 por hora.

## Decisão

1. **Duas partes de famílias diferentes.** O modelo de submissão passa a poder ter uma parte Gemma-4-31B (parte G) e uma parte Qwen3.8-27B (parte Q), as duas na mesma placa de 96 GB, com média meio a meio das probabilidades de cada pergunta. Só vale se o gate GE (item 8) passar. A entrega mínima continua sendo um modelo só.
2. **As partes são escolhidas pelo Full do placar, não pelo índice público** (contexto 5). Candidatas Apache-2.0 hoje: para a parte G, KnowLine 31B Gen1.5 (71,91), Sieve 31B v3 (71,10) e Turion Jev 31B (70,47); para a parte Q, d3-ultra (70,18) e pplx-decider-v1.2 (67,98).
3. **Ponto de partida do treino (parte Q).** Passa a ser o candidato Apache de maior Full que o carregador do kit reproduzir (G0). O candidato é o d3-ultra; o pplx-decider-v1.2 continua sendo o que o kit carrega e o que o `decide/config.rental.yaml` aponta. O arquivo de configuração só muda depois do G0 do d3-ultra.
4. **Dados.** A F2 passa a buscar diversidade de tarefas: as fontes do `tasksource/tasksource-jev-typed-decisions` entram uma a uma pelo `sources.yaml`, cada uma com a sua licença conferida, e pela descontaminação. Nenhuma fonte entra sem os dois.
5. **Pensar sob demanda sai do caminho crítico.** O kit do GT (`decide/think`) fica como está, guardado e congelado até a primeira publicação (decisão 10). Motivo: a regra de latência inclui a média; com a média do pplx em 415 ms sobram 585 ms, e um raciocínio de 2.048 tokens (o orçamento do `config.rental.yaml`) a 40 a 80 tokens por segundo levaria 26 a 51 s, o que caberia em 1% a 2% dos pedidos (**UNVERIFIED**: a velocidade é suposta). Os dois primeiros do placar ganham profundidade sem gerar token (contexto 2).
6. **Alavancas sem treino, a medir na primeira sessão de GPU:** duas ordens de opções (já existe em `decide/sd/tta.py`; o dia alugado usa uma), perguntas antes do estado ou leitura dupla, e os blocos do meio em dois passes. Cada uma é hipótese para um modelo que não foi treinado com ela.
7. **Autodestilação ou interpolação.** O ensaio 4B compara a receita atual (rótulo ouro + autodestilação, que depende da F3) com rótulo ouro + interpolação dos pesos com o ponto de partida (`decide.train.soup --weights`). A F3 só sai do plano se o ensaio mostrar que a troca não perde.
8. **Gate novo, GE (duas partes numa placa de 96 GB).** Os quatro limiares são escolhas do planejador, não medidas:
   - cada parte reproduz o argmax do seu resultado publicado em ≥ 95% de ≥ 500 linhas (provisório, a recalibrar como o G0 do 27B em 4 bits);
   - pico de memória abaixo de 96 GB com as duas partes carregadas;
   - mediana, média e p80 de latência ≤ 1.000 ms numa amostra de ≥ 750 pedidos da suíte pública, um pedido por vez;
   - índice parcial do par ≥ o do melhor membro + 1,0, nas mesmas linhas.
9. **Metas.** A meta "#1" não muda. As duas referências numéricas do PLAN (69,5 e 72,5) foram escritas contra o placar de 2026-10-20 e ficaram para trás (contexto 1). As novas se medem contra o placar do dia da submissão. Mínima: Full acima do melhor Full entre os modelos públicos de que a submissão parte, ou seja, o trabalho tem de acrescentar algo ao que já se baixa pronto. Alvo: Full ≥ o do líder menos 0,9, a distância dentro da qual o placar dá o mesmo posto (contexto 9); no arquivo de 2026-11-06 isso é 73,08.
10. **Decisões do E0, tomadas em 2026-11-10 por delegação do dono.** O texto operacional está na seção 11 do PLAN.
    - **Primeira publicação:** o par sem treino, como ensaio declarado (v0); o modelo com treino próprio vem em seguida (v1). O v0 exercita o caminho inteiro (motor, GE, rodada pública, cartão, Space) e traz o Full de um par, que só o lab mede. Não é o que põe o modelo à frente (consequências).
    - **Partes do v0:** KnowLine 31B Gen1.5 e d3-ultra, pela decisão 2. Se a latência do GE não fechar, a parte G passa a ser o Turion Jev 31B (contexto 3). Cada parte roda com o código de inferência dos seus autores, que é como o placar roda cada participante (contexto 9).
    - **Nome e lugar:** `wesleysimplicio/Simplicio-Decide-27B` para a parte Q treinada por nós; `wesleysimplicio/Simplicio-Decide-Duo` para o modelo de duas partes. O Simplicio-27B de conversa fica como está e ganha um link para o decisor.
    - **O v0 não se apresenta como modelo treinado.** Antes da rodada pública completa, os mantenedores são consultados na discussão do Space, porque nenhuma regra escrita trata de um par de modelos de terceiros sem treino (contexto 9). Se não aceitarem, o v0 não vai ao placar.
    - **Licença:** "só Apache-2.0" fica, com o custo medido nas consequências.
    - **Kit do pensador:** guardado e congelado; reabre por ADR nova quando o Reasoning board existir.
    - **GPU e orçamento:** Hugging Face Jobs, `rtx-pro-6000`, a US$ 2,75 por hora (contexto 9). Até 12 horas para o GE (US$ 33,00) e até 40 horas para a rodada pública completa do v0 (US$ 110,00). O tempo de cada etapa não foi medido (**UNVERIFIED**).
    - **Continua com o dono:** pedir acesso ao HLE no Hub, confirmar cada compra de GPU e aprovar o texto de tudo que sai em nome dele.

## Consequências

- **Nada muda no código por esta ADR.** O motor de duas partes não existe; o carregador do kit só conhece o layout do pplx e do d3; não há carregador para a família Gemma. O trabalho novo está listado na seção 11 do PLAN.
- **O RUNBOOK continua valendo** para a trilha de treino de uma parte, que é a da parte Q.
- **A regra de licença tem um custo medido.** Ficam de fora o GLiDE 27B (licença Fastino Community), o FlexDM-LoRA (CC BY-SA 4.0), o Nimble (CC BY-NC 4.0) e o H2O Lightning (licença `other`). A melhor parte Q em Apache tem Full 70,18, contra 73,37 do GLiDE.
- **Qualquer um pode montar o mesmo par.** Duas partes públicas chegam ao nível do líder no índice público parcial; ficar à frente depende de melhorar pelo menos uma parte com treino próprio.
- **O cartão tem de dizer de quem é cada parte.** Uma submissão de duas partes públicas sem treino é uma técnica de inferência sobre pesos de terceiros; o cartão, o `NOTICE` e as licenças de cada parte precisam dizer isso.
- **O nome.** Duas partes somam 57 bilhões de parâmetros servidos; "Simplicio-Decide-27B" descreve só a parte Q. Por isso o modelo de duas partes se chama `Simplicio-Decide-Duo` (decisão 10).
- **O decisor não substitui o Simplicio-27B de conversa.** Um modelo de decisão devolve probabilidades e não gera texto: não serve pelo Ollama, pelo GGUF nem pelo `deploy/serve_vllm.sh`. Ele vai para repositório próprio (`wesleysimplicio/Simplicio-Decide-27B`, que já é o `hub.winner_repo` do `decide/config.yaml`), e o cartão do Simplicio-27B ganha um link quando esse repositório existir.
- **A primeira entrada no placar pode não ser um modelo nosso.** O v0 é um par de modelos de terceiros; se os mantenedores o aceitarem, ele aparece com o nome Simplicio e o cartão é o que diz de quem são os pesos. Pela regra de uma linha por modelo, a rodada do v1 aposenta a do v0 (contexto 9).
- **Há gasto novo fora do dia alugado:** até US$ 143,00 no caminho do v0, somando o GE e a rodada pública completa, se os dois tetos forem atingidos.
- Esta ADR só muda por outra ADR. Os números do plano mudam por nova versão do `decide/PLAN.md`, com a data e o que foi medido.

## O que não foi medido (UNVERIFIED)

- O Full de qualquer combinação, e se o ganho do índice público passa para a parte privada.
- Memória e latência de duas partes em FP8 numa placa: os números são do cartão do REAX e do placar.
- O efeito do FP8 nas respostas de cada parte.
- Se o carregador do kit reproduz o d3-ultra, e quanto custa um carregador para a família Gemma.
- Se as alavancas da decisão 6 ajudam um modelo que não foi treinado com elas.
- Os 7 benchmarks do índice que ficaram de fora do estudo (HLE, HoVer, Amazon ESCI, BRIGHT, ToolRet, API-Bank, ChessBench) e o pplx-decider-v1.2, que não publica resultados por linha.
- Se os mantenedores aceitam uma submissão de duas partes com pesos de terceiros e sem treino, e como a classificam: nenhuma regra escrita trata do caso (contexto 9).
- O tempo da sessão de GPU do GE e da rodada pública de duas partes. O preço por hora foi lido (contexto 9); os tetos da decisão 10 são escolhas.
- Se a conta do dono no Hugging Face tem acesso ao Jobs e ao HLE.
- Se o FP8 de cada parte pode ser feito na carga, sem publicar cópia dos pesos.

## Apêndice: revisões e fontes lidas

Repositórios de modelo no Hugging Face, na revisão vista em 2026-11-10.

| repositório | revisão | licença no Hub |
|---|---|---|
| `decisionmodels/REAX-1.3` | `fa4bf7ee3bb56a8bc421dad090631be1c5845c2d` | apache-2.0 |
| `fastino/GLiDE-27B` | `4d32743ae96b74f0116f050d37742bbd43c04e7d` | other (Fastino Community) |
| `PelaAI/KnowLine-31B-Gen1.5` | `8e56aa1b236a6dd1064dcfd22ba1e5e360e4c140` | apache-2.0 |
| `sthanika-ai/Sieve-31B-v3` | `8fcee858f7f20a50bfa3c8f984c22adb3cbb46a2` | apache-2.0 |
| `turion-ai/turion-jev-31b` | `866c0fc6e040cba23fcc72113a80e5c99d5ebab1` | apache-2.0 |
| `vllm-sr/d3-ultra` | `ccdcfa611649a59e095e0b5e1732a15cedafadd6` | apache-2.0 |
| `tasksource/tasksource-jev-typed-decisions` (dataset) | `8df2e5c9a33f19a2c2b8febf291c3acb0a91a8a8` | other (o cartão do dataset não foi lido; o do Turion fala em fontes marcadas para uso comercial) |

As revisões dos datasets de resultados por linha e o sha256 do `index.json` estão em [0007-evidencia/README.md](0007-evidencia/README.md).

Regras e preço lidos em 2026-11-10 (contexto 9), sem credencial.

| fonte | o que foi lido | versão |
|---|---|---|
| README do kit `apolinario/decision-index` | como se submete, regra de latência, `hf-job`, `--workers` | commit `25c6ec7b6b5f44a216c63c3a700de3a9c6d6699d` |
| Space `multimodalart/jev-decision-index`, `data/methodology.json` | regras duras, categorias de participante, regra de uma linha por modelo | revisão `e05cd04cad2b4f8d6c34aaa486ff88f1a9584942` do Space; sha256 `bf299b4961972eae6c8c6924c494e934205e72eab1aa062b9b3457eebcbc1bdc`; 294.537 bytes |
| mesmo Space, `methodology.html` | regra de empate (0,9 ponto) | mesma revisão; sha256 `d909efacc573b174d00fc999674cb311e27d64bbe3bc66be16a652f839432174` |
| mesmo Space, `index.html` | Reasoning board como "coming soon" | mesma revisão; sha256 `2e65add99dedeabd8b4f28160006d988e5665e651f07b4c51de5ac93806cfc1e` |
| `https://huggingface.co/api/jobs/hardware` | sabor `rtx-pro-6000`: 96 GB de VRAM, 23 vCPU, 256 GB de RAM, 475 GB de disco, US$ 0,045833 por minuto | resposta de 2026-11-10 |

O `data/methodology.json` traz `generated_utc` de 2026-09-28, lista 70 participantes e não inclui o REAX 1.3; por isso a contagem de participantes por categoria do contexto 9 vem do `index.json` de 2026-11-06.
