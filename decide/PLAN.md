# Simplicio-Decide-27B — plano para o #1 do Jev Decision Index

Versão 2.1 · 2026-10-21 (após Fase 0A e 0A-bis; seção 10, Plano B, acrescentada) · dono: Wesley · planejamento: Claude (Opus) · execução: squads.
Regra de evidência: só número medido. Projeção ou estimativa leva a marca **UNVERIFIED**.
Épico: [#64](https://github.com/simpletibr/simplicio-27b/issues/64). Uma issue por fase, na tabela da seção 5.

## 1. Objetivo
Transformar o Simplicio-27B num modelo de decisão no estilo Jev: recebe um estado e perguntas tipadas e devolve uma probabilidade por opção, numa passada só. Base Qwen3.8-27B, licença Apache-2.0.

| meta | critério | referência (board de 2026-10-20) |
|---|---|---|
| mínima | Full ≥ 69,5 | passa o pplx-decider-v1.2 (#2, 69,42) |
| alvo | #1 | mammoth-1 = 71,24; mirar ≥ 72,5 porque o corte sobe toda semana (UNVERIFIED) |

## 2. Ponto de partida (medido)
- O Simplicio-27B atual é Qwen3.8-27B + LoRA de código (r32, alpha 32, 120 steps, 101 exemplos). Não é modelo de decisão; entra só como ajuste pequeno no merge.
- O top 10 tem 26–31B, base densa; 7 deles em Qwen3.8-27B.
- Receita comum do top: head de 255 códigos, atenção não-causal nas camadas de full-attention, soft cross-entropy, temperatura de calibração e média de pesos.
- Fase 0B: pplx-decider-v1.2, d3 e d3-turbo têm o mesmo layout de backbone (rev 1d4bf0f2). Dá para fundir só o backbone; head e prompt ficam os do pplx. Amos é incompatível; Lychee é CC BY-NC e fica fora.
- A rodada pública completa tem ~292 milhões de tokens de entrada (contagem do Jev no board).
- Fase 0A (média de pesos iguais, 8 modelos do top 10; o rescore do kit bateu com o board em até 0,004 nos 8): a média de {d3, d3-turbo, Amos} fica 2,77 pontos abaixo do d3-turbo. Média de pesos iguais não supera o melhor membro.
- Fase 0A-bis (combinadores calibrados, ajustados numa metade das linhas públicas e medidos na outra): nenhum (temperatura + média, pool log-linear, pesos por tipo de pergunta, roteamento por confiança) supera o melhor membro Apache; o pool colapsa no d3-turbo (pesos 0,85 / 0 / 0,18). Só o teto com os 8 modelos, que inclui modelos CC BY-NC, ganha +0,55 [+0,22, +0,89].
- Notas brutas públicas do board: os melhores decisores (StartLux, pplx v1.2, d3-turbo, mammoth) têm GPQA Diamond de 52,0 a 61,7 e HLE de 15,8 a 19,4. O card do Qwen3.8-27B (thinking, reportado pelo fornecedor) tem GPQA Diamond 89,2 e HLE 30,8. O ambiente de avaliação é outro, então a comparação é **UNVERIFIED**: o F2b mede.

## 3. Desenho do modelo
- **Protocolo:** o do pplx-decider-v1.2 — prompt conjunto, leitura nos marcadores `<|fim_suffix|>`, head [255, 5120], full-attention não-causal, DeltaNet causal, temperatura por tipo de pergunta.
- **Ponto de partida:** pplx-decider-v1.2 (melhor Full entre os modelos Apache). O merge vira experimento opcional e barato.
- **Treino:** fine-tune contínuo no corpus F2 com rótulo ouro + autodestilação do próprio pplx v1.2 para não esquecer. Sem destilação de ensemble (Fase 0A-bis). Opções embaralhadas por exemplo, 1 época, média de seeds.
- **Pensar sob demanda (aposta principal, condicionada ao gate GT):** nas perguntas de raciocínio, o Qwen3.8-27B pensa (thinking) e o decisor lê estado + raciocínio e devolve as probabilidades; um gate decide quando pensar (confiança do decisor + tipo de pergunta). Cabe na GPU do lab (96 GB): pensador em FP8 (~27 GB) + decisor em bf16 (~52 GB) (**UNVERIFIED**: a conta não inclui o KV cache do raciocínio). Sem o GT, a entrega é o modo só-decisor. Ver [ADR 0004](../docs/adr/0004-pensar-sob-demanda.md).

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

## 9. Organização
- Planejador (Opus): plano, contratos e revisão. Não executa.
- Squads (1 coordenador Sonnet + até 5 Haiku): kit (`decide/sd`, `decide/modal`, `notebooks/decide`) e dados/treino/aluguel (`decide/data`, `decide/train`, `decide/rental`, docs).
- Merge: um coordenador Sonnet, em série, só PR com "APROVADO PELO SQUAD".
- Issues: 1 épico e 1 issue por fase, com o gate como critério de aceite.

## 10. Plano B (bases alternativas)
O plano acima **não muda**. Esta seção só registra o que fazer se ele falhar. Evidência e fontes: [ADR 0005](../docs/adr/0005-plano-b-base-alternativa.md); o que não foi medido leva **UNVERIFIED**.
- **Pedido do dono:** avaliar o InternScience/Sage-A1.5 (arXiv 2610.28578) como decisor, mantendo o plano atual.
- **Sage-A1.5 (medido no Hub, 2026-10-21):** MoE `qwen3_5_moe`, 40 camadas (10 de atenção completa, 30 de DeltaNet), largura 2048, 256 especialistas (8 por token), 35.951.822.704 parâmetros, 71,9 GB em BF16, Apache-2.0; 35B no total e 3B ativos (arXiv 2610.28578, legenda da Fig. 9). Base: o card diz Qwen3.6-35B-A3B, o artigo diz "built on Agents-A1" (§1); a base direta do Agents-A1 não consta no texto lido: **UNVERIFIED**. O artigo só avalia com ferramentas (HLE w/ tools 55,1) e não traz GPQA, MMLU-Pro nem HLE sem ferramentas, então nenhuma nota dele é comparável ao gate GT; nada nele trata de decisão em uma passada.
- **Placar (`index.json`, 2026-10-20):** os 8 modelos de decisão com base de 30 a 35B-A3B vão de 46,50 a 62,28 (melhor: KnowLine 35B-A3B Gen1, LoRA, #27); os três com a mesma base do Sage ficaram em 54,61, 54,49 e 50,85. O décimo do board tem 67,48 e a nossa meta mínima é 69,5. As receitas diferem, então isso não prova teto.
- **Incompatível com o plano atual:** largura 2048 contra 5120, então nada do merge de backbones, da cabeça do pplx (255 × 5120), do delta do Simplicio ou do kit de fidelidade transfere; pediria cabeça nova (255 × 2048), atenção não-causal só nas 10 camadas de atenção completa e LoRA em atenção e especialistas compartilhados (LoRA nos roteados não está resolvido). BF16 de 71,9 GB não cabe no Kaggle; o NF4 numa T4 é **UNVERIFIED** e provavelmente inviável.
- **Plano B1 (preferido): Gemma-4-31B com LoRA.** Dois no top 10, ambos Apache-2.0: AJev lm6q, 69,05, #5, LoRA (r 64, alpha 89,6, λ = 0,7, segundo o card, que reporta 65,08 no público, auto-reportado); linne-oak 1.0, 68,55, #7, ajuste completo. Não reaproveita o Simplicio nem o merge Qwen; pede cabeça nova (largura 5376), protocolo para o tokenizador do Gemma, LoRA e dados novos. **Gatilho:** o gate GT falhar **e** a receita do Qwen3.8 estagnar abaixo de 66 no privado caseiro depois do dia econômico de aluguel.
- **Plano B2 (baixa prioridade): Sage-A1.5 como decisor.** Só por pedido explícito do dono. Teste mínimo: mesma receita, mesmo subconjunto (~20 mil linhas) e mesma semente no Sage e no Qwen3.8, medidos no privado caseiro (1.500 linhas); o Sage morre se ficar mais de 2 pontos abaixo. Custo **UNVERIFIED**: 1 a 2 h de B200 por modelo, a cerca de US$ 6,25 por hora (US$ 0,001736 por segundo; gpuperhour.com, lida em 2026-09-21). Releitura em 2026-10-21: US$ 5,36 (média de 30 dias) a US$ 7,99 (mais barato sob demanda) por hora, ou cerca de US$ 5 a US$ 16 por modelo. Detalhe na ADR.
- **Uso útil sem virar decisor:** o Sage pode ser um pensador alternativo (`--model` do kit do GT) para gerar traços baratos. Só depois de a PR #78 entrar na `main`, e só vale no treino se o GT medir esse pensador. O thinking padrão dele e o interruptor do template não constam no card. Não medido. B2 continua em baixa prioridade: a mesma família 35B-A3B tem 62,28 como melhor no placar.
- **Plano C (opcional, não é base alternativa): pesquisa autônoma da receita no 4B.** [ADR 0006](../docs/adr/0006-plano-c-pesquisa-autonoma.md), a partir do artigo Athanor (arXiv 2610.21055). **Não substitui nada e não mexe em gate.** É só a busca de receita guiada por agente sobre o `pplx-decider-flash-v1.2` (Full 54,29 no board de 2026-10-21), medida no G2, e só **depois de G0 a G3 medidos** e se o dono aprovar por escrito um orçamento além do dia alugado. Não vale para o 27B. O artigo parte do Qwen3-4B-Base com 4 agentes por 104 h (de 15,9 para 67,3 numa suíte de desenvolvimento e de 9,2 para 48,2 numa separada) e gasta cerca de US$ 11.800; o nosso C1 enxuto, cerca de 190 horas de GPU (~US$ 700), é extrapolação do coordenador, **UNVERIFIED**, assim como a disponibilidade e a licença do código e a transferência para um decisor já treinado. Exige uma fatia selada do privado caseiro que o agente nunca vê e o vencedor re-medido em semente nova.
