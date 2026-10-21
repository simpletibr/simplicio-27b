# Simplicio-Decide-27B — plano para o #1 do Jev Decision Index

Versão 1 · 2026-10-21 · dono: Wesley · planejamento: Claude (Opus) · execução: squads.
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

## 3. Desenho do modelo
- **Protocolo:** o do pplx-decider-v1.2 — prompt conjunto, leitura nos marcadores `<|fim_suffix|>`, head [255, 5120], full-attention não-causal, DeltaNet causal, temperatura por tipo de pergunta.
- **Pesos iniciais:** merge de backbones (pplx v1.2 como âncora + d3 [+ d3-turbo] + delta do LoRA do Simplicio, λ escolhido por varredura). Se o merge não passar o G3, o ponto de partida é o pplx v1.2 puro.
- **Treino:** destilação de ensemble. Os professores (pplx v1.2 e d3) rotulam o corpus e o modelo aprende a média das probabilidades, misturada com o rótulo ouro quando existir. Opções embaralhadas por exemplo, 1 época, média de seeds.

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
| F3 rotulagem | [#68](https://github.com/simpletibr/simplicio-27b/issues/68) | Modal free | probabilidades dos professores em bf16 | ≥ 95% do corpus rotulado |
| F4 ensaio 4B | [#69](https://github.com/simpletibr/simplicio-27b/issues/69) | Kaggle T4 | a mesma receita em 4B, partindo do pplx-decider-flash-v1.2 | G2: ≥ +1,0 sobre o pplx-flash no privado caseiro |
| F5 merge 27B | [#70](https://github.com/simpletibr/simplicio-27b/issues/70) | Kaggle T4 (4-bit) | varredura de λ | G3: melhor merge ≥ pplx v1.2 no privado caseiro |
| F6 dia alugado | [#71](https://github.com/simpletibr/simplicio-27b/issues/71) | GPU alugada | treino, média de seeds, temperatura, rodada pública completa | G4: ≥ pplx v1.2 + 1,5 no privado caseiro antes da rodada final |
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
- Semana 1: F0, F1, F2; F3 começa no Modal.
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

## 9. Organização
- Planejador (Opus): plano, contratos e revisão. Não executa.
- Squads (1 coordenador Sonnet + até 5 Haiku): kit (`decide/sd`, `decide/modal`, `notebooks/decide`) e dados/treino/aluguel (`decide/data`, `decide/train`, `decide/rental`, docs).
- Merge: um coordenador Sonnet, em série, só PR com "APROVADO PELO SQUAD".
- Issues: 1 épico e 1 issue por fase, com o gate como critério de aceite.
