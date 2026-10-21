# ADR 0005: Plano B, bases alternativas ao Qwen3.8-27B (Gemma-4-31B como B1, Sage-A1.5 como B2)

- **Status:** Proposta. Registro de uma avaliação, não de uma troca. Nenhum treino foi feito em outra base e nenhum modelo alternativo foi medido por nós.
- **Data:** 2026-10-21
- **Plano:** [decide/PLAN.md](../../decide/PLAN.md), seção 10.
- **Relacionadas:** #64 (épico), #75 (F2b, gate GT), [ADR 0003](0003-simplicio-decide-27b.md), [ADR 0004](0004-pensar-sob-demanda.md).
- **Escopo:** o plano atual (base Qwen3.8-27B, ponto de partida pplx-decider-v1.2, gates G0 a G5) **continua como está**. Esta ADR só diz o que fazer, e quando, se ele falhar, e o que o Sage-A1.5 poderia ser sem virar decisor.

Regra de evidência: só número medido entra como fato, com a fonte ao lado. Projeção ou estimativa leva a marca **UNVERIFIED** e não decide nada sozinha. Todas as leituras abaixo foram feitas em 2026-10-21 (o planejador as fez primeiro; a squad de dados e treino as repetiu na mesma data, e onde a releitura diferiu a diferença está dita).

## Contexto

1. **Pedido do dono.** Avaliar o InternScience/Sage-A1.5 (arXiv 2610.28578) como Decider, mantendo o plano atual.

2. **O que o Sage-A1.5 é (medido).** Fontes: `config.json`, `model.safetensors.index.json`, `README.md` (o card) e a API do Hugging Face (`/api/models/InternScience/Sage-A1.5`), todos na revisão `ce23c7123fdd6e0ce55a9f8d067fbe3a86145c50`.

   | item | valor | fonte |
   |---|---|---|
   | arquitetura | `qwen3_5_moe` (`Qwen3_5MoeForConditionalGeneration`) | `config.json` |
   | camadas | 40: 30 de DeltaNet (`linear_attention`) e 10 de atenção completa (`full_attention`, uma a cada 4) | `config.json`, `layer_types` |
   | largura | `hidden_size` 2048 | `config.json` |
   | especialistas | 256, 8 por token; `moe_intermediate_size` 512 e `shared_expert_intermediate_size` 512 | `config.json` |
   | vocabulário | 248.320 | `config.json` |
   | tamanho dos pesos | `total_size` 71.903.645.408 bytes, em BF16 | `model.safetensors.index.json` |
   | parâmetros | 35.951.822.704 (todos BF16) | API, campo `safetensors.total` |
   | licença | Apache-2.0 | API, `cardData.license` |
   | base | o card diz "based on **Qwen3.6-35B-A3B**" e "Building upon **A1**" | `README.md`, parágrafo de abertura |

   O total de parâmetros é o mesmo, byte a byte, do Qwen3.5-35B-A3B e do Qwen3.6-35B-A3B (API, mesmo campo, mesma data): 35.951.822.704. **Divergência de leitura:** a primeira leitura do planejador registrou a base como "não informada"; o card, relido pela squad de dados e treino, traz o nome da base no primeiro parágrafo. Vale o card.

   **Não medido (UNVERIFIED):** os ~3 bilhões de parâmetros ativos (vêm do "A3B" do nome da base, não de um número do card); o artigo completo (o HTML do arXiv recusou a leitura com 429, e a squad de dados e treino não o leu); GPQA Diamond e HLE **sem ferramentas**, que são o que o board mede (o card só traz "HLE w/ tools", 55,1, com ferramentas e fora do ambiente do board); o comportamento do modelo como decisor de uma passada.

3. **O que o placar diz sobre bases MoE (medido).** Fonte: `index.json` do board, gerado em 2026-10-20T14:13:56Z (campos `models[].v03.v`, o Full, e `models[].meta.base_model`), baixado de novo em 2026-10-21. Os oito modelos de decisão com base de 30 a 35B-A3B:

   | Full | modelo | colocação | base declarada no board | tipo |
   |---|---|---|---|---|
   | 62,28 | KnowLine 35B-A3B Gen1 | #27 | Qwen3.5-35B-A3B | LoRA |
   | 58,96 | JEV-35B | #44 | Qwen3.5-35B-A3B | LoRA + head |
   | 54,85 | JPT-35B-A3B | #60 | Qwen3.5-35B-A3B | LoRA |
   | 54,61 | Nace.AI Drex v2.0 (35B-A3B) | #61 | **Qwen3.6-35B-A3B** | ajuste completo |
   | 54,49 | Xor 1.2 | #62 | **Qwen3.6-35B-A3B** | LoRA |
   | 50,85 | decisio 0.4.0 (Qwen3.6-35B-A3B) | #67 | **Qwen3.6-35B-A3B** | técnica de inferência |
   | 47,21 | Decider 35B-A3B (NVFP4) | #77 | Qwen3.5-35B-A3B-Base | ajuste completo |
   | 46,50 | Cloudflare clef-omni (30B-A3B) | #81 | Qwen3-Omni-30B-A3B-Instruct | ajuste completo |

   (Colocação entre os 208 modelos do `index.json`, pelo Full; tipo é o `meta.kind` do board.) O melhor de todos é 62,28. Os três que partem da base do Sage (Qwen3.6-35B-A3B) ficaram em 54,61, 54,49 e 50,85. O top 10 do board tem só modelos densos de 26 a 31B; o décimo (Amos 1 Large) tem Full 67,48 e a nossa meta mínima é 69,5 (PLAN.md, seção 1). O Sage precisaria de **+5,2 sobre o melhor MoE do placar** só para entrar no top 10 (67,48 − 62,28) e de **+7,2** para a nossa meta mínima.

   **Ressalva:** as receitas são diferentes (dados, cabeça, tipo de ajuste, protocolo), então "o melhor MoE do placar é 62,28" descreve o que foi submetido, não um teto da arquitetura.

4. **O que o placar diz sobre o Gemma-4-31B (medido).** Mesma fonte. Oito modelos com base `google/gemma-4-31B-it` estão entre os 20 primeiros; dois estão no top 10: **AJev lm6q, Full 69,05, #5, LoRA** (a API mostra as etiquetas `peft` e `lora`; o rank do LoRA não está no board nem na API, o "r64" que circula é do planejador, **UNVERIFIED**) e **linne-oak 1.0, Full 68,55, #7, ajuste completo** (`meta.kind` do board). Os dois repositórios e o `google/gemma-4-31B-it` têm licença Apache-2.0 na API. O `config.json` do Gemma-4-31B-it dá `hidden_size` 5376, 60 camadas e vocabulário de 262.144 (arquitetura `gemma4`).

## Decisão

1. **O plano atual não muda.** O Qwen3.8-27B com o pplx-decider-v1.2 como ponto de partida segue sendo a rota; nenhuma hora de GPU é gasta em outra base antes de um gatilho abaixo.
2. **Plano B1 (preferido): família Gemma-4-31B**, com LoRA, como os dois modelos do top 10 acima. É uma segunda opção **com evidência de placar** (69,05 e 68,55 contra 67,48 do décimo). Não reaproveita o delta do Simplicio, o merge de backbones da Fase 0B, a cabeça do pplx (255 × 5120 contra 5376 de largura) nem o kit de fidelidade, que são do Qwen3.8-27B: pede cabeça nova, protocolo reescrito para o tokenizador do Gemma (**UNVERIFIED**), LoRA e dados novos.
3. **Plano B2 (baixa prioridade): Sage-A1.5 como decisor.** Só se o dono pedir explicitamente. Não há número do modelo como decisor, e as bases MoE do placar não passaram de 62,28 (contexto 3).
4. **Gatilhos.** Decididos pelo planejador, não medidos:
   - **B1** só se as duas condições valerem: (a) o gate GT (F2b, #75) falhar e (b) a receita do Qwen3.8-27B estagnar abaixo de 66 no privado caseiro (G1) depois do dia econômico de aluguel (F6).
   - **B2** só por pedido explícito do dono. Teste mínimo: a mesma receita, o mesmo subconjunto (~20 mil linhas) e a mesma semente no Sage e no Qwen3.8, avaliados no `proxy_private` (1.500 linhas). O Sage morre se ficar **mais de 2 pontos** abaixo do Qwen3.8 nesse teste.
5. **O Sage-A1.5 como pensador, não como decisor (uso útil, não medido).** Com ~3B ativos (**UNVERIFIED**) ele pode ser um gerador barato dos traços `Raciocínio:` do gate GT. Só depois de a PR #78 (kit do GT) estar na `main`, e como um `--model` alternativo do `decide.think.thinker`. Atenção: o interruptor de thinking do template do Qwen3.8 pode diferir do dele (o card não diz qual é o thinking padrão; o servidor do card usa `--reasoning-parser qwen3`); o "modo de verificação" do Sage exige uma mensagem de sistema não vazia (`chat_template_kwargs: {"verification": true}`, `README.md`). Nada disso foi medido. Os traços de qualquer pensador só valem no treino se o GT medir esse pensador.

## O que não transfere do plano atual para o Sage-A1.5

| item do plano | por que não transfere |
|---|---|
| merge de backbones (Fase 0B, F5) | a largura é 2048, não 5120, e o backbone é MoE: nenhum tensor se alinha com o d3, o d3-turbo ou o pplx |
| cabeça do pplx (255 × 5120) e o delta do Simplicio | formas diferentes; precisaria de cabeça nova, 255 × 2048, treinada do zero |
| atenção não-causal do protocolo | só nas 10 camadas de atenção completa; as 30 de DeltaNet continuam causais, como no Qwen3.8 |
| LoRA | em atenção e nos especialistas compartilhados é o caminho conhecido; LoRA nos especialistas roteados não está resolvido (**UNVERIFIED**) |
| cota gratuita | 71,9 GB em BF16 (medido) não cabe nas T4 do Kaggle; o NF4 com especialistas fundidos numa T4 é, na melhor hipótese, **UNVERIFIED** e provavelmente inviável. Não prometer o Kaggle |
| kit de fidelidade (F1) | o carregador do kit é do layout do pplx/d3; um MoE precisa de outro caminho |

## Consequências

- O custo de testar o Sage como decisor (B2) é uma estimativa, **UNVERIFIED**: da ordem de 1 a 2 h de B200 por modelo no teste mínimo, a US$ 0,001736 por segundo (preço do B200 informado pelo planejador, da página de preços da Modal em 2026-10-21; não relido por esta squad), ou seja, cerca de US$ 6,25 por hora de GPU (0,001736 × 3600). O PLAN.md seção 6 usa os preços da mesma página, mas no modo preemptível e sem este número por segundo. Não inclui o tempo para escrever a cabeça e o carregador novos.
- O B1 tem mais evidência de placar e menos reaproveitamento de código; o B2 tem menos evidência e o mesmo problema de reaproveitamento. A diferença está dita para que a escolha, se vier, seja feita sabendo disso.
- Nenhuma fase, gate ou issue do plano atual muda. Os gatilhos acima são a única coisa nova, e eles dependem de medidas que o plano atual já produz (GT e privado caseiro).
- Esta ADR só muda por outra ADR. Os números do plano mudam por nova versão do `decide/PLAN.md`, com a data e o que foi medido.
