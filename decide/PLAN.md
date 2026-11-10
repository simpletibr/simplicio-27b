# Simplicio-Decide-27B — plano para o #1 do Jev Decision Index

Versão 2.2 · 2026-11-10 (placar de 2026-11-06 e estudo de ensemble entre famílias; seção 11 acrescentada, com a rota 2, o caminho até publicar e as decisões do E0; metas da seção 1 revistas) · dono: Wesley · planejamento: Claude (Opus até a 2.1, Fable na 2.2) · execução: squads.
Regra de evidência: só número medido. Projeção ou estimativa leva a marca **UNVERIFIED**.
Épico: [#64](https://github.com/simpletibr/simplicio-27b/issues/64). Uma issue por fase, na tabela da seção 5.
Rota atual: a da seção 11 ([ADR 0007](../docs/adr/0007-duas-partes-entre-familias.md)). As seções 3 a 7 descrevem a trilha de treino de uma parte, que continua valendo para a parte Qwen.

## 1. Objetivo
Transformar o Simplicio-27B num modelo de decisão no estilo Jev: recebe um estado e perguntas tipadas e devolve uma probabilidade por opção, numa passada só. Base Qwen3.8-27B, licença Apache-2.0.

| meta | critério | referência (placar de 2026-11-06) |
|---|---|---|
| mínima | Full acima do melhor Full entre os modelos públicos de que a submissão parte | 71,91 quando o KnowLine 31B Gen1.5 é a parte G; 70,18 para a parte Q sozinha, se partir do d3-ultra (67,98 se partir do pplx-decider-v1.2) |
| alvo | #1: Full ≥ o do líder menos 0,9 no dia da submissão | 73,08 (REAX 1.3 = 73,98); o placar dá o mesmo posto a quem fica a até 0,9 ponto do primeiro do grupo |

**Placar de 2026-11-06 (medido; `index.json` gerado em 2026-11-06T09:05:08Z, 274 modelos):** o #1 é o REAX 1.3 (73,98) e o #2, o GLiDE 27B (73,37); o pplx-decider-v1.2 caiu para #17 (67,98) e o mammoth-1 para #13 (68,44). As referências da versão 2.1 (mínima 69,5 e alvo 72,5, contra o placar de 2026-10-20) ficaram para trás: 69,5 hoje ficaria entre o #10 (69,75) e o #11 (69,14), e 72,5 fica abaixo do #2. As metas da tabela foram decididas em 2026-11-10 (seção 11) e se medem contra o placar do dia da submissão, não contra um número fixo.

## 2. Ponto de partida (medido)
- O Simplicio-27B atual é Qwen3.8-27B + LoRA de código (r32, alpha 32, 120 steps, 101 exemplos). Não é modelo de decisão; entra só como ajuste pequeno no merge.
- O top 10 tem 26–31B, base densa; 7 deles em Qwen3.8-27B.
- Receita comum do top: head de 255 códigos, atenção não-causal nas camadas de full-attention, soft cross-entropy, temperatura de calibração e média de pesos.
- Fase 0B: pplx-decider-v1.2, d3 e d3-turbo têm o mesmo layout de backbone (rev 1d4bf0f2). Dá para fundir só o backbone; head e prompt ficam os do pplx. Amos é incompatível; Lychee é CC BY-NC e fica fora.
- A rodada pública completa tem ~292 milhões de tokens de entrada (contagem do Jev no board).
- Fase 0A (média de pesos iguais, 8 modelos do top 10; o rescore do kit bateu com o board em até 0,004 nos 8): a média de {d3, d3-turbo, Amos} fica 2,77 pontos abaixo do d3-turbo. Média de pesos iguais não supera o melhor membro.
- Fase 0A-bis (combinadores calibrados, ajustados numa metade das linhas públicas e medidos na outra): nenhum (temperatura + média, pool log-linear, pesos por tipo de pergunta, roteamento por confiança) supera o melhor membro Apache; o pool colapsa no d3-turbo (pesos 0,85 / 0 / 0,18). Só o teto com os 8 modelos, que inclui modelos CC BY-NC, ganha +0,55 [+0,22, +0,89].
- Notas brutas públicas do board: os melhores decisores (StartLux, pplx v1.2, d3-turbo, mammoth) têm GPQA Diamond de 52,0 a 61,7 e HLE de 15,8 a 19,4. O card do Qwen3.8-27B (thinking, reportado pelo fornecedor) tem GPQA Diamond 89,2 e HLE 30,8. O ambiente de avaliação é outro, então a comparação é **UNVERIFIED**: o F2b mede.
- Placar de 2026-11-06: no top 10 há 6 modelos com base Gemma-4-31B, 3 com Qwen3.8-27B e 1 de duas partes (REAX 1.3: uma parte Gemma e uma Qwen em FP8 na mesma placa, com média meio a meio das probabilidades, segundo o cartão). O melhor Full entre os Apache de uma parte é o do KnowLine 31B Gen1.5 (71,91); na família Qwen, o do d3-ultra (70,18).
- Estudo de ensemble entre famílias (2026-11-10, CPU, [ADR 0007](../docs/adr/0007-duas-partes-entre-familias.md)): 13 modelos do topo em 30 dos 37 benchmarks do índice público (94.173 pedidos), com score e skill iguais aos do placar em 4 casas. Par Gemma + Qwen com membros a até 2 pontos um do outro: +1,52 no índice parcial sobre o melhor membro (13 de 13 pares positivos). Par Qwen + Qwen só com modelos Apache: −1,50 (3 pares), o que confirma a Fase 0A-bis dentro da família. KnowLine Gen1.5 + d3-ultra, meio a meio: 80,77, contra 78,45 do melhor membro e 80,97 do REAX 1.3 nos mesmos benchmarks.
- Entre esses 13 modelos, o índice público e o Full têm correlação de postos de 0,41: o índice público não serve para escolher partes nem ponto de partida.

## 3. Desenho do modelo
- **Protocolo:** o do pplx-decider-v1.2 — prompt conjunto, leitura nos marcadores `<|fim_suffix|>`, head [255, 5120], full-attention não-causal, DeltaNet causal, temperatura por tipo de pergunta.
- **Ponto de partida (parte Qwen):** o candidato Apache de maior Full que o carregador do kit reproduzir (G0). O pplx-decider-v1.2 é o que o kit carrega e o que o `config.rental.yaml` aponta, mas deixou de ser o melhor Full entre os Apache (67,98; o d3-ultra tem 70,18). A troca para o d3-ultra depende do G0 dele (ADR 0007, decisão 3). O merge segue experimento opcional e barato.
- **Treino:** fine-tune contínuo no corpus F2 com rótulo ouro + autodestilação do próprio pplx v1.2 para não esquecer. Sem destilação de ensemble (Fase 0A-bis). Opções embaralhadas por exemplo, 1 época, média de seeds. O ensaio 4B compara a autodestilação com a interpolação dos pesos com o ponto de partida (ADR 0007, decisão 7).
- **Duas partes (ADR 0007, condicionada ao gate GE):** uma parte Gemma-4-31B e uma Qwen3.8-27B na mesma placa de 96 GB, com média meio a meio das probabilidades. As partes são escolhidas pelo Full do placar. O motor de duas partes não existe na `main`. Sem o GE, a entrega é um modelo só.
- **Pensar sob demanda (fora do caminho crítico desde a ADR 0007; segue condicionado ao gate GT):** nas perguntas de raciocínio, o Qwen3.8-27B pensa (thinking) e o decisor lê estado + raciocínio e devolve as probabilidades; um gate decide quando pensar (confiança do decisor + tipo de pergunta). Cabe na GPU do lab (96 GB): pensador em FP8 (~27 GB) + decisor em bf16 (~52 GB) (**UNVERIFIED**: a conta não inclui o KV cache do raciocínio). Sem o GT, a entrega é o modo só-decisor. Ver [ADR 0004](../docs/adr/0004-pensar-sob-demanda.md). Saiu do caminho crítico porque a regra de latência inclui a média (ADR 0007, decisão 5). O kit (`decide/think`) fica guardado e congelado até a primeira publicação (seção 11).

## 4. Recursos
Uma conta por plataforma. Várias contas para somar cota ficam fora: o FAQ do Colab proíbe usar múltiplas contas para contornar limites de recurso.

| recurso | uso no plano |
|---|---|
| Kaggle (T4×2 16 GB, ~30 h/semana, sessão de 12 h) | smoke, fidelidade em 4-bit, ensaio 4B, varredura do merge |
| Colab free (T4, só interativo) | apoio ao ensaio 4B e à calibração |
| Kaggle CPU / VPS (10 vCPU, 32 GB) | corpus, descontaminação, privado caseiro |
| Modal (US$ 30/mês) | rotulagem pelos professores 27B em bf16, em partes que retomam |
| Hugging Face Hub | estado dos estágios, datasets, pesos |
| 1 dia de GPU alugada | treino do 27B, calibração e rodada final |
| Vast.ai, uma RTX PRO 6000 de 96 GB (o modelo de placa em que o lab mede a latência) em host verificado; de US$ 1,47 a US$ 2,56 por hora com 250 GB de disco, lido em 2026-11-10 | gate GE (E4) e rodada pública completa da rota 2 |
| Hugging Face Jobs, `rtx-pro-6000` (a mesma placa; US$ 2,75 por hora, lido em 2026-11-10) | reserva do E4 e da rodada pública |
| Salad (sob demanda, até 32 GB por placa; RTX 4090 de 24 GB de US$ 0,16 a US$ 0,44 por hora, lido em 2026-11-10) | reserva barata para o que couber em 24 a 32 GB, se a cota do Kaggle faltar |

## 5. Fases e gates
| fase | issue | onde | entrega | gate (medido) |
|---|---|---|---|---|
| F0 setup | [#65](https://github.com/simpletibr/simplicio-27b/issues/65) | CPU | repo, kit, issues, ADR | `make check` verde |
| F1 fidelidade | [#66](https://github.com/simpletibr/simplicio-27b/issues/66) | Kaggle T4 | d3-mini (4B) e d3 (27B) em 4-bit comparados com os resultados bf16 publicados | G0: argmax igual em ≥ 98% (4B) e ≥ 95% (27B 4-bit) em ≥ 500 linhas; limiar do 27B a recalibrar |
| F2 dados | [#67](https://github.com/simpletibr/simplicio-27b/issues/67) | CPU | corpus: splits oficiais descontaminados + tasksource com licença comercial + domínios novos PT-BR/EN + traces do Simplicio-Loop; privado caseiro | G1: o privado caseiro ordena ≥ 6 modelos abertos como o Full do board (Spearman ≥ 0,8) |
| F2b pensar | [#75](https://github.com/simpletibr/simplicio-27b/issues/75) | Kaggle T4, Modal ou API | Qwen3.8-27B com thinking numa amostra das provas de conhecimento, comparado linha a linha com d3-turbo e StartLux | GT: acerto ≥ +10 pontos sobre o melhor decisor na amostra, IC 95% acima de 0 |
| F3 rotulagem | [#68](https://github.com/simpletibr/simplicio-27b/issues/68) | Modal free | autodestilação: logits do pplx v1.2 no corpus (+ traços do pensador se o GT passar) | ≥ 95% do corpus rotulado |
| F4 ensaio 4B | [#69](https://github.com/simpletibr/simplicio-27b/issues/69) | Kaggle T4 | a mesma receita em 4B, partindo do pplx-decider-flash-v1.2 | G2: ≥ +1,0 sobre o pplx-flash no privado caseiro |
| F5 merge 27B (opcional) | [#70](https://github.com/simpletibr/simplicio-27b/issues/70) | Kaggle T4 (4-bit) | varredura de λ | G3: melhor merge ≥ pplx v1.2 no privado caseiro |
| F6 dia alugado | [#71](https://github.com/simpletibr/simplicio-27b/issues/71) | GPU alugada | treino do decisor que lê raciocínio + gate, média de seeds, temperatura, rodada pública completa | G4: ≥ pplx v1.2 + 1,5 no privado caseiro antes da rodada final |
| F7 submissão | [#72](https://github.com/simpletibr/simplicio-27b/issues/72) | HF + Space | pesos, resultados, PR no Space, card honesto | G5: concordância do lab ≥ 98,5% |

A rota 2 (seção 11) acrescenta o gate GE e os passos E0 a E6. As fases acima continuam valendo para a parte Qwen; a F2b (pensar) sai do caminho crítico.

## 6. Dia alugado (estimativas, UNVERIFIED)
| perfil | GPU | carga | tempo | custo no Modal (preemptível) |
|---|---|---|---|---|
| econômico (padrão) | 1× B200 | ~250 mil exemplos, LoRA r128, 1 seed | ~14 h | ~US$ 90 |
| completo | 8× H100 | ~600 mil exemplos, full FT, 2 seeds | ~12–16 h | ~US$ 380–510 |

- O perfil completo só entra se o G2 mostrar ganho ≥ +1,0 no ensaio 4B.
- Conta usada: FLOPs ≈ 2·P·tokens na inferência e 6·P·tokens no treino, com 40% de aproveitamento da GPU; preços da página do Modal em 2026-10-21.
- No econômico, a rotulagem roda antes, no crédito free do Modal (F3). Se o dia acabar antes da rodada final, ela vai para o crédito free do mês seguinte.

## 7. Calendário (UNVERIFIED)
- Semana 1: F0, F1, F2, F2b; F3 começa no Modal.
- Semana 2: F4 e F5.
- Semana 3: F6 (dia alugado) e F7.

## 8. Riscos
| risco | mitigação |
|---|---|
| T4 (sm75) sem kernels do DeltaNet | caminho em torch puro, ou F1/F4 no crédito do Modal |
| 4-bit distorce o ranking | o G0 mede antes de qualquer decisão |
| privado caseiro não correlaciona com o Full | decidir pela amostra pública + teto de ensemble (Fase 0A) |
| vazamento de teste | descontaminação por n-grama contra os testes públicos; nada que imite testes privados; tudo documentado no card |
| corte sobe antes da submissão | submeter assim que o G4 passar; v2 depois |
| licença | só pesos Apache; só dados com licença comercial |
| preempção no Modal | tudo em partes, com checkpoint no HF |
| o board limitar raciocínio/latência | manter o modo só-decisor como entrega mínima |
| testes privados com thinking demoram mais no lab | gate conservador e orçamento de tokens por pergunta |
| a média de latência de duas partes passa de 1.000 ms (KnowLine 605 ms + d3-ultra 424 ms em sequência) | rodar as partes em paralelo em FP8, como o REAX 1.3 (790 ms no placar), ou trocar a parte G por uma mais rápida (Turion, 299 ms); o GE mede |
| o índice público engana na escolha das partes (Spearman 0,41 com o Full entre os 13 modelos do estudo) | escolher pelo Full do placar; no treino, decidir pelo privado caseiro (G1 e G4) |
| qualquer um monta o mesmo par de modelos públicos | o par chega ao nível do líder no índice público parcial; ficar à frente depende de treinar pelo menos uma parte |
| a máquina alugada no Vast.ai é de terceiros e vê o token do Hugging Face e a cópia privada da suíte | só host verificado; token de escopo fino, criado para a sessão e revogado depois; apagar a instância ao fim; se isso não bastar, datacenter verificado (US$ 2,36 a US$ 2,56 por hora) ou Hugging Face Jobs (US$ 2,75) |
| nenhuma regra escrita do placar trata de uma submissão de duas partes com pesos de terceiros (README do kit e metodologia do Space, lidos em 2026-11-10); quem aceita é a revisão do PR | perguntar aos mantenedores na discussão do Space antes de pagar a rodada pública completa; se não aceitarem, o v0 não vai ao placar (seção 11) |

## 9. Organização
- Planejador (Opus): plano, contratos e revisão. Não executa.
- Squads (1 coordenador Sonnet + até 5 Haiku): kit (`decide/sd`, `decide/modal`, `notebooks/decide`) e dados/treino/aluguel (`decide/data`, `decide/train`, `decide/rental`, docs).
- Merge: um coordenador Sonnet, em série, só PR com "APROVADO PELO SQUAD".
- Issues: 1 épico e 1 issue por fase, com o gate como critério de aceite.

## 10. Plano B (bases alternativas)
O plano acima **não muda**. Esta seção só registra o que fazer se ele falhar. Evidência e fontes: [ADR 0005](../docs/adr/0005-plano-b-base-alternativa.md); o que não foi medido leva **UNVERIFIED**.
- **Pedido do dono:** avaliar o InternScience/Sage-A1.5 (arXiv 2610.28578) como decisor, mantendo o plano atual.
- **Sage-A1.5 (medido no Hub, 2026-10-21):** MoE `qwen3_5_moe`, 40 camadas (10 de atenção completa, 30 de DeltaNet), largura 2048, 256 especialistas (8 por token), 35.951.822.704 parâmetros, 71,9 GB em BF16, Apache-2.0; 35B no total e 3B ativos (arXiv 2610.28578, legenda da Fig. 9). Base: o card diz Qwen3.6-35B-A3B, o artigo diz "built on Agents-A1" (§1); a base direta do Agents-A1 não consta no texto lido: **UNVERIFIED**. O artigo só avalia com ferramentas (HLE w/ tools 55,1) e não traz GPQA, MMLU-Pro nem HLE sem ferramentas, então nenhuma nota dele é comparável ao gate GT; nada nele trata de decisão em uma passada.
- **Placar (`index.json`, 2026-10-20):** os 8 modelos de decisão com base de 30 a 35B-A3B vão de 46,50 a 62,28 (melhor: KnowLine 35B-A3B Gen1, LoRA, #27); os três com a mesma base do Sage ficaram em 54,61, 54,49 e 50,85. O décimo do board tinha 67,48 e a nossa meta mínima, na versão 2.1 deste plano, era 69,5. As receitas diferem, então isso não prova teto.
- **Incompatível com o plano atual:** largura 2048 contra 5120, então nada do merge de backbones, da cabeça do pplx (255 × 5120), do delta do Simplicio ou do kit de fidelidade transfere; pediria cabeça nova (255 × 2048), atenção não-causal só nas 10 camadas de atenção completa e LoRA em atenção e especialistas compartilhados (LoRA nos roteados não está resolvido). BF16 de 71,9 GB não cabe no Kaggle; o NF4 numa T4 é **UNVERIFIED** e provavelmente inviável.
- **Plano B1 (preferido): Gemma-4-31B com LoRA.** Dois no top 10, ambos Apache-2.0: AJev lm6q, 69,05, #5, LoRA (r 64, alpha 89,6, λ = 0,7, segundo o card, que reporta 65,08 no público, auto-reportado); linne-oak 1.0, 68,55, #7, ajuste completo. Não reaproveita o Simplicio nem o merge Qwen; pede cabeça nova (largura 5376), protocolo para o tokenizador do Gemma, LoRA e dados novos. **Gatilho:** o gate GT falhar **e** a receita do Qwen3.8 estagnar abaixo de 66 no privado caseiro depois do dia econômico de aluguel. A rota 2 (seção 11) usa um decisor Gemma público como parte G sem treiná-lo; treinar em Gemma continua sendo este Plano B1.
- **Plano B2 (baixa prioridade): Sage-A1.5 como decisor.** Só por pedido explícito do dono. Teste mínimo: mesma receita, mesmo subconjunto (~20 mil linhas) e mesma semente no Sage e no Qwen3.8, medidos no privado caseiro (1.500 linhas); o Sage morre se ficar mais de 2 pontos abaixo. Custo **UNVERIFIED**: 1 a 2 h de B200 por modelo, a cerca de US$ 6,25 por hora (US$ 0,001736 por segundo; gpuperhour.com, lida em 2026-09-21). Releitura em 2026-10-21: US$ 5,36 (média de 30 dias) a US$ 7,99 (mais barato sob demanda) por hora, ou cerca de US$ 5 a US$ 16 por modelo. Detalhe na ADR.
- **Uso útil sem virar decisor:** o Sage pode ser um pensador alternativo (`--model` do kit do GT) para gerar traços baratos. Só depois de a PR #78 entrar na `main`, e só vale no treino se o GT medir esse pensador. O thinking padrão dele e o interruptor do template não constam no card. Não medido. B2 continua em baixa prioridade: a mesma família 35B-A3B tem 62,28 como melhor no placar.
- **Plano C (opcional, não é base alternativa): pesquisa autônoma da receita no 4B.** [ADR 0006](../docs/adr/0006-plano-c-pesquisa-autonoma.md), a partir do artigo Athanor (arXiv 2610.21055). **Não substitui nada e não mexe em gate.** É só a busca de receita guiada por agente sobre o `pplx-decider-flash-v1.2` (Full 54,29 no board de 2026-10-21), medida no G2, e só **depois de G0 a G3 medidos** e se o dono aprovar por escrito um orçamento além do dia alugado. Não vale para o 27B. O artigo parte do Qwen3-4B-Base com 4 agentes por 104 h (de 15,9 para 67,3 numa suíte de desenvolvimento e de 9,2 para 48,2 numa separada) e gasta cerca de US$ 11.800; o nosso C1 enxuto, cerca de 190 horas de GPU (~US$ 700), é extrapolação do coordenador, **UNVERIFIED**, assim como a disponibilidade e a licença do código e a transferência para um decisor já treinado. Exige uma fatia selada do privado caseiro que o agente nunca vê e o vencedor re-medido em semente nova.

## 11. Rota 2: duas partes e o caminho até publicar
Evidência, decisão e o que não foi medido: [ADR 0007](../docs/adr/0007-duas-partes-entre-familias.md). Esta seção vale onde contradiz as seções 3 a 7.

**O que muda**
- O modelo de submissão pode ter duas partes, uma Gemma-4-31B (parte G) e uma Qwen3.8-27B (parte Q), com média meio a meio das probabilidades. Candidatas Apache-2.0 pelo Full de 2026-11-06: para a parte G, KnowLine 31B Gen1.5 (71,91), Sieve 31B v3 (71,10) ou Turion Jev 31B (70,47); para a parte Q, d3-ultra (70,18) ou pplx-decider-v1.2 (67,98).
- O treino continua sendo o das fases F2 a F6, na parte Q, com corpus mais diverso (E1) e ponto de partida decidido pelo G0. A parte G entra como decisor público, sem treino; treinar em Gemma continua sendo o Plano B1 (seção 10).
- Pensar sob demanda (F2b) sai do caminho crítico.
- Duas partes Apache públicas, sem treino, chegam ao nível do líder no índice público parcial: KnowLine Gen1.5 + d3-ultra faz 80,77 e Sieve v3 + d3-ultra 81,00, contra 80,97 do REAX 1.3. Ficar à frente depende de treinar pelo menos uma parte. O Full de qualquer par é **UNVERIFIED**.

**Gate GE (duas partes numa placa de 96 GB).** Limiares escolhidos, não medidos: cada parte reproduz o argmax do seu resultado publicado em ≥ 95% de ≥ 500 linhas; pico de memória abaixo de 96 GB; mediana, média e p80 de latência ≤ 1.000 ms em ≥ 750 pedidos; índice parcial do par ≥ o do melhor membro + 1,0 nas mesmas linhas. A sessão roda numa RTX PRO 6000 de 96 GB, o modelo de placa em que o lab mede a latência (decisão 9 do E0), e o resultado registra a variante da placa e o limite de potência.

**Caminho até publicar**

| passo | o que é | onde | estado em 2026-11-10 |
|---|---|---|---|
| E0 decisões | onde e com que nome o decisor é publicado; uma ou duas partes na primeira publicação; orçamento de GPU | dono | decidido em 2026-11-10, por delegação do dono (lista no fim desta seção) |
| E1 dados diversos (F2, [#67](https://github.com/simpletibr/simplicio-27b/issues/67)) | fontes do `tasksource/tasksource-jev-typed-decisions` no `sources.yaml`, com a licença conferida fonte a fonte, e descontaminação | CPU | o código de ingestão existe (`decide/data`); as fontes não estão no registro |
| E2 suíte pública local | reconstruir a suíte com o kit, para a descontaminação e para medir | CPU | 30 de 37 benchmarks reconstruídos em 2026-11-10 com menos de 1,5 GB de fontes, o que basta para o E4; a completa, que a rodada pública exige, pede cerca de 17 GB de trabalho e acesso ao HLE (fechado), e não cabe na máquina do estudo (20 GB livres); o `decide.modal.suite_requests` que o RUNBOOK cita não está na `main` |
| E3 motor de duas partes | carregar as duas partes em FP8, tirar a média das probabilidades e servir no formato do kit | código novo | não existe. Cada parte tem código de inferência publicado pelos autores e o motor do REAX é público (licença MIT, segundo o cartão); o reuso não foi avaliado (**UNVERIFIED**) |
| E4 primeira sessão de GPU | G0 de cada parte, GE e as alavancas sem treino da ADR 0007 | uma RTX PRO 6000 de 96 GB, alugada no Vast.ai | nada rodou numa GPU; teto de 12 horas e de US$ 24,00; o tempo que a sessão leva é **UNVERIFIED** |
| E5 treino da parte Q (F4 e F6) | ensaio 4B (G2) e dia alugado (G4) com o corpus do E1 | Kaggle e GPU alugada | o código existe (`decide/train`, `decide/rental`) e nunca rodou numa GPU |
| E6 publicação (F7, [#72](https://github.com/simpletibr/simplicio-27b/issues/72)) | rodada pública completa (140.178 pedidos pontuáveis: 109.759 da base e 30.419 dos benchmarks acrescentados, pelo kit), pesos e resultados no Hugging Face, cartão com a licença de cada parte, PR no Space | GPU, Hugging Face e Space | a fase `publish` do `decide.rental.run` existe e nunca rodou contra o Hub de verdade; o gerador de cartão que o RUNBOOK cita (`report/README.preview.md`) não está na `main`; nenhuma regra escrita do placar trata de duas partes com pesos de terceiros, então a pergunta aos mantenedores vem antes da rodada completa |

**Duas formas de primeira publicação**
- **Sem treino (E2, E3, E4, E6):** duas partes públicas, meio a meio. É uma técnica de inferência sobre pesos de terceiros, e o cartão tem de dizer isso. O placar lista o REAX 1.3 como "inference technique", categoria que a metodologia define para quem não publica pesos próprios e serve um checkpoint de fábrica com código próprio; como um par de ajustes de terceiros seria classificado é decisão dos mantenedores. Dá o primeiro número medido e exercita o caminho inteiro até o Space.
- **Com treino próprio (E1, E2, E5, E6; mais E3 e E4 se for de duas partes):** a parte Q treinada no corpus diverso, sozinha ou ao lado de uma parte G pública. É o que pode pôr o modelo à frente.

**Decisões do E0 (2026-11-10)**

O dono delegou estas decisões ao planejador em 2026-11-10. Elas valem até outra versão deste plano; o que nelas é estimativa leva a marca **UNVERIFIED**. As regras do placar e o preço da placa foram lidos no mesmo dia (ADR 0007, contexto 9).

1. **Primeira publicação: o par sem treino, como ensaio declarado (v0). O modelo com treino próprio vem em seguida (v1).** Ordem do v0: E3; E4 com as linhas dos 30 benchmarks já reconstruídos; pergunta aos mantenedores (item 5); E2 completo; rodada pública; E6. O E1 corre em paralelo, porque só o treino pode pôr o modelo à frente. Se o GE falhar com os dois pares do item 2, a primeira publicação passa a ser a parte Q treinada, sozinha (F2 a F7).
2. **Partes do v0:** KnowLine 31B Gen1.5 (parte G) e d3-ultra (parte Q), as de maior Full entre as Apache-2.0 de cada família. Se a latência do GE não fechar, a parte G passa a ser o Turion Jev 31B, que tem média de 299 ms no placar, contra 605 ms do KnowLine. Cada parte roda com o código de inferência dos seus autores, na revisão fixada.
3. **Onde e com que nome.** A parte Q treinada por nós vai para `wesleysimplicio/Simplicio-Decide-27B` (o `hub.winner_repo` do `config.yaml`). O modelo de duas partes vai para `wesleysimplicio/Simplicio-Decide-Duo`: "27B" descreveria só a parte Q, e o cartão abre dizendo quais são as duas partes e que elas somam 57 bilhões de parâmetros servidos. O decisor devolve probabilidades e não gera texto, então o Simplicio-27B de conversa fica como está no Ollama, no GGUF e no vLLM, e ganha um link para o decisor quando o repositório do decisor existir.
4. **O v0 não se apresenta como modelo treinado.** O cartão diz de quem é cada parte, a licença do adaptador e da base de cada uma, a revisão usada e que não há treino nosso. O repositório leva o motor e a configuração e aponta para os repositórios dos autores. Só publica cópia de pesos se o FP8 não puder ser feito na carga (**UNVERIFIED**: depende do E3), e nesse caso com a licença e a origem de cada cópia.
5. **Antes de pagar a rodada pública completa do v0, perguntar aos mantenedores** na discussão do Space se aceitam um par de modelos de terceiros sem treino e como o classificam. Nenhuma regra escrita trata do caso. Se não aceitarem, o v0 fica como receita e resultados no Hub, sem entrada no placar.
6. **Metas:** as da seção 1. Mínima, Full acima do melhor Full entre os modelos públicos de que a submissão parte. Alvo, Full ≥ o do líder menos 0,9 no dia da submissão (73,08 em 2026-11-06).
7. **Licença:** "só Apache-2.0" fica. Deixa de fora o GLiDE 27B (Full 73,37; licença Fastino Community) e o FlexDM-LoRA 2.1 (71,61; CC BY-SA 4.0).
8. **Kit do pensador (`decide/think`):** guardado e congelado. Fica na árvore com os testes e não recebe trabalho novo nem GPU antes da primeira publicação. Reabre por ADR nova quando o Reasoning board existir; em 2026-11-06 a página do placar o anuncia como "coming soon".
9. **GPU do E4 e da rodada pública:** uma RTX PRO 6000 de 96 GB, o modelo de placa em que o lab mede a latência, alugada no Vast.ai. O dono apontou em 2026-11-10 que o Vast.ai e o Salad saem mais baratos que o Hugging Face, e os preços lidos no mesmo dia confirmam isso para o Vast.ai: host verificado, com confiabilidade ≥ 0,98, download ≥ 500 Mbps e 250 GB de disco, sai de US$ 1,47 a US$ 2,56 por hora nas variantes S e WS (14 ofertas, mediana US$ 1,85), contra US$ 2,75 no Hugging Face Jobs.
   - **Regra de escolha:** variante S ou WS, host verificado com esses mínimos, até US$ 2,00 por hora. A variante Max-Q e os hosts com limite de potência declarado abaixo de 500 W ficam de fora, porque distorcem o critério de latência. Das 14 ofertas lidas, 9 custavam até US$ 2,00 e 6 dessas não tinham esse limite de potência.
   - **Reserva:** o Hugging Face Jobs (`rtx-pro-6000`, o padrão do `hf-job` do kit), se não houver oferta assim.
   - **Salad:** não entra nestes passos. Sob demanda ele vai até 32 GB por placa (RTX 5090), e a RTX PRO 6000 só existe por contrato. Fica como opção barata para o que couber em 24 a 32 GB.
   - **Dia de treino (E5):** continua no Modal. No Vast.ai a B200 parte de US$ 7,50 por hora, acima do que a seção 6 supõe (cerca de US$ 90 por 14 horas). A comparação de preços se repete antes de cada compra.
10. **Orçamento, além do dia alugado:** até 12 horas para o E4 e até 40 horas para a rodada pública completa do v0, com o preço limitado a US$ 2,00 por hora: no máximo US$ 24,00 e US$ 80,00. Na reserva do Hugging Face Jobs seriam US$ 33,00 e US$ 110,00. O teto de 40 horas vem de 140.178 pedidos a 1.000 ms, um por vez (38,9 horas); o `run` do kit manda 8 pedidos ao mesmo tempo por padrão, então o tempo real deve ser menor (**UNVERIFIED**). Sessão que chega ao teto para, e o que foi medido até ali é registrado.

**O que continua com o dono,** porque só ele pode fazer: pedir acesso ao HLE no Hub (a suíte completa depende dele); criar um token do Hugging Face de escopo fino para a máquina alugada e revogá-lo depois; confirmar cada compra de GPU com item e total; e aprovar o texto de tudo que sai em nome dele (a pergunta aos mantenedores, o cartão e o PR no Space).
