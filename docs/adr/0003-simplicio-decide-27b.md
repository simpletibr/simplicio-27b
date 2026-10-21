# ADR 0003: Simplicio-Decide-27B, um modelo de decisão para o Jev Decision Index

- **Status:** Proposta. Nenhum resultado do modelo foi medido. Cada passo do plano só avança quando o gate dele (G0 a G5) for medido.
- **Data:** 2026-10-21
- **Plano:** [decide/PLAN.md](../../decide/PLAN.md), versão 1.
- **Relacionadas:** #64 (épico) e #65 a #72 (fases F0 a F7).
- **Escopo:** esta ADR não altera as ADRs 0001 e 0002, que tratam da comparação de código do Simplicio-27B.

Regra de evidência: só número medido entra como fato. Projeção ou estimativa leva a marca **UNVERIFIED** e não decide nada sozinha.

## Contexto

Fatos medidos, com a fonte de cada um.

1. **O board.** O `index.json` do Space, gerado em 2026-10-20T14:13:56Z, lista 208 modelos, todos com Full score. O #1 é o mammoth-1, com 71,24. O #2 é o pplx-decider-v1.2, com 69,42. O #3 é o StartLux-Decision v1.1 27B, com 69,29. O Full score é 0,60 × testes privados das mesmas habilidades + 0,40 × tarefas privadas de domínios novos, as duas partes postas na escala pública com modelos stock sem ajuste como régua. A rodada pública completa é obrigatória para submeter, mas não entra no Full.
2. **O top 10.** Tem de 26,09 a 31,27 bilhões de parâmetros servidos. Sete usam Qwen3.8-27B como base, um usa Qwen3.5-27B (o #1) e dois usam Gemma 4 31B.
3. **O Simplicio-27B atual** é o Qwen3.8-27B com um LoRA de código: r 32, alpha 32, 120 steps, 101 exemplos em português (README do repositório). Não é um modelo de decisão e não tem nenhuma medida no índice.
4. **Compatibilidade de pesos (Fase 0B, cabeçalhos dos safetensors, sem baixar pesos).** pplx-decider-v1.2, d3, d3-turbo e Lychee compartilham o layout de tensores do backbone (1183 tensores, nomes e formas idênticos) e a revisão base do Qwen3.8-27B (1d4bf0f2). Heads, posição de leitura, prompts e máscara de atenção diferem entre eles. O Amos tem layout próprio e é incompatível. O Lychee é CC BY-NC. O adapter do Simplicio (LoRA r 32, alpha 32, 256 módulos) é compatível com o bf16 do Qwen3.8-27B: nos dois tensores amostrados, o bf16 publicado do Simplicio é igual a bf16(base + B·A). Não houve teste de interferência nem de acurácia de nenhum merge.
5. **O protocolo do pplx-decider-v1.2.** Licença Apache-2.0 e base `Qwen/Qwen3.8-27B` (API do Hugging Face, 2026-10-21). Template `joint-question-inline-readout-v1`, leitura nos marcadores `<|fim_suffix|>`, head de 255 códigos × 5120, atenção não-causal nas camadas de full-attention e DeltaNet causal.
6. **O tamanho da rodada pública.** 110.201 requisições, 109.759 pontuáveis (`suite` do `index.json`). O plano registra cerca de 292 milhões de tokens de entrada, pela contagem do Jev no board; esse número não foi recontado aqui.
7. **Regras do kit de reprodução (README do kit, edição 0.4).** Modelo com latência mediana, média ou p80 acima de 1.000 ms por requisição numa RTX PRO 6000 não entra no board. Se os testes do lab acharem sinal de que o treino do modelo se sobrepôs a um teste privado, aquele teste não conta para o modelo. A suíte pública é dado de avaliação: o kit pede para não treinar nela nem republicá-la. O BBH carrega o canary do BIG-bench e não pode entrar em corpus de treino.
8. **Recursos.** O plano usa uma conta por plataforma. O FAQ do Colab proíbe usar várias contas para contornar limites de recurso.

## Decisão

1. **Meta.** Mínima: Full ≥ 69,5, que passa o pplx-decider-v1.2. Alvo: #1; mirar ≥ 72,5 porque o corte sobe toda semana (**UNVERIFIED**).
2. **Protocolo.** O do pplx-decider-v1.2: prompt conjunto, leitura nos marcadores `<|fim_suffix|>`, head [255, 5120], full-attention não-causal, DeltaNet causal, temperatura por tipo de pergunta.
3. **Pesos iniciais.** Merge só de backbones: pplx v1.2 como âncora, mais d3 (e d3-turbo, opcional), mais o delta do LoRA do Simplicio, com λ escolhido por varredura (F5). Head e prompt ficam os do pplx. Se o melhor merge não passar o G3, o ponto de partida é o pplx v1.2 puro.
4. **Treino.** Destilação de ensemble. Os professores (pplx v1.2 e d3) rotulam o corpus em bf16 (F3). O aluno aprende a média das probabilidades deles, misturada com o rótulo ouro quando existir. Opções embaralhadas por exemplo, 1 época, média de seeds.
5. **Dados (F2).** Splits oficiais de treino dos benchmarks públicos, descontaminados contra os testes públicos por hash exato e por n-grama; datasets do tasksource só com licença comercial confirmada dataset a dataset; domínios novos em PT-BR e EN; traces do Simplicio-Loop. Nada que imite os testes privados. O "privado caseiro" é feito só de fontes excluídas do treino.
6. **Recursos.** Kaggle, Colab free, Kaggle CPU ou VPS, crédito free do Modal e Hugging Face Hub para as fases F0 a F5; um dia de GPU alugada para o treino do 27B (F6). Perfil econômico (1× B200) como padrão; o perfil completo (8× H100) só entra se o G2 mostrar ganho ≥ +1,0 no ensaio 4B. Tempos e custos do dia alugado são estimativas (**UNVERIFIED**).
7. **Gates.** G0 fidelidade em 4-bit (F1), G1 privado caseiro × Full do board (F2), G2 ensaio 4B (F4), G3 merge (F5), G4 antes da rodada final (F6), G5 concordância do lab (F7). Os limiares estão na tabela da seção 5 do plano e são o critério de aceite de cada issue.
8. **Licença e proveniência.** Só pesos Apache-2.0 e só dados com licença comercial. Lychee (CC BY-NC) e Amos (incompatível) ficam de fora. O código de treino vendorizado do autojev (Apache-2.0) mantém LICENSE, NOTICE e o aviso das mudanças.

## Consequências

- O modelo final deriva do pplx-decider-v1.2 (Apache-2.0). O card precisa dizer isso, listar os dados usados, o método de descontaminação e o que não foi medido.
- Nenhum número do board é atribuído ao Simplicio-Decide-27B antes da rodada pública completa e da validação do lab. Até lá, qualquer afirmação de ranking é **UNVERIFIED**.
- O privado caseiro é um proxy. O G1 mede se ele ordena modelos abertos como o Full do board (Spearman ≥ 0,8 em pelo menos 6 modelos). Passar o G1 não prova que o ganho no proxy aparece no Full privado. Se o proxy não correlacionar, a decisão passa a usar a amostra pública e o teto de ensemble (Fase 0A).
- A descontaminação por hash e n-grama diminui o risco de vazamento de teste, mas não o elimina. O kit desconta do modelo um teste privado se achar sobreposição, então o risco é medido pelo lab, não por nós.
- A rotulagem pelos professores 27B é a etapa que mais consome tokens. Ela roda em partes que retomam, com estado no Hugging Face, porque a execução no Modal pode ser preemptada e o crédito é limitado (US$ 30 por mês, seção 4 do plano).
- O G4 bloqueia a rodada pública completa. Se o G4 falhar, o dia alugado não vira submissão e o próximo passo é uma v2. Se o dia acabar antes da rodada final, ela vai para o crédito free do mês seguinte.
- Risco do plano: a T4 (sm75) fica sem os kernels do DeltaNet. F1 e F4 usam o caminho em torch puro ou o crédito do Modal, com custo de tempo ainda não medido (**UNVERIFIED**).
- Esta ADR só muda por outra ADR. Os números do plano mudam por nova versão do `decide/PLAN.md`, com a data e o que foi medido.
