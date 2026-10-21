# ADR 0004: Pensar sob demanda, sem destilação de ensemble

- **Status:** Proposta. Nenhum resultado do modelo foi medido. O próximo gate é o GT (F2b).
- **Data:** 2026-10-21
- **Plano:** [decide/PLAN.md](../../decide/PLAN.md), versão 2.
- **Relacionadas:** #64 (épico), #67 (F2), #76 (F2b). Esta ADR substitui os itens 3 (pesos iniciais) e 4 (treino) da decisão da [ADR 0003](0003-simplicio-decide-27b.md) e a consequência dela sobre a rotulagem pelos professores 27B. Os outros itens da ADR 0003 continuam valendo.
- **Escopo:** não muda a meta, o protocolo, as regras de dados e de licença nem os gates G0 a G5.

Regra de evidência: só número medido entra como fato. Projeção ou estimativa leva a marca **UNVERIFIED** e não decide nada sozinha.

## Contexto

Fatos medidos, com a fonte de cada um. Os itens 1 e 2 vêm do relatório da Fase 0A e 0A-bis, feito pelo planejador com as funções do próprio kit e com os resultados por linha publicados no Hugging Face (o relatório ainda não está versionado neste repositório).

1. **Média de pesos iguais (Fase 0A).** Oito modelos do top 10 têm resultados por linha publicados e foram reavaliados com o kit; o rescore bateu com o board em até 0,004 ponto nos oito. O mammoth-1 é gated (acesso negado) e nenhum dataset de resultados por linha do pplx-decider-v1.2 foi encontrado, então os dois ficaram de fora. A média de pesos iguais de {d3, d3-turbo, Amos}, os três modelos Apache sobre Qwen3.8-27B, ficou 2,77 pontos abaixo do d3-turbo (IC 95% de [−3,16, −2,35], bootstrap pareado por caso).
2. **Combinadores calibrados (Fase 0A-bis).** Ajustados numa metade das linhas públicas e medidos na outra (e a troca): temperatura por modelo + média, pool log-linear com um peso por modelo, o mesmo pool com pesos por tipo de pergunta, e roteamento por confiança. Nenhum superou o melhor membro Apache com o IC 95% acima de zero. O pool por tipo de pergunta ficou em −0,02 [−0,25, +0,17] contra o d3-turbo, e os pesos aprendidos colapsam nele (d3-turbo 0,85, d3 0, Amos 0,18). Só o teto com os oito modelos, que inclui StartLux e Lychee (CC BY-NC), ganha: +0,55 [+0,22, +0,89].
3. **Notas brutas do board contra o card do Qwen3.8-27B.** No `index.json` do board (gerado em 2026-10-20T14:13:56Z), o campo `raw` de GPQA Diamond e de HLE é, na ordem StartLux, pplx-decider-v1.2, d3-turbo e mammoth-1: GPQA Diamond 52,0, 55,1, 57,7 e 61,7; HLE 17,6, 16,0, 15,8 e 19,4. O card do Qwen3.8-27B (thinking, reportado pelo fornecedor) traz GPQA Diamond 89,2 e HLE 30,8. Os ambientes são outros: no board o GPQA tem quatro opções e o HLE é de múltipla escolha com 5 a 26 opções, só texto, respondido numa passada sem raciocínio; no card o HLE é julgado pelo GPT-4o. A comparação direta é **UNVERIFIED**.
4. **Regras do kit (README do kit).** Modelo cuja latência mediana, média ou p80 passa de 1.000 ms por requisição, medida pelos mantenedores numa RTX PRO 6000 (um processo, uma requisição por vez), não entra no board. O mesmo README anuncia um "Reasoning board" (0.3.x) para modelos que podem pensar antes de responder às perguntas difíceis ("thinking takes longer, so they get a board of their own"). As regras desse board não estão publicadas (**UNVERIFIED**).
5. **O ponto de partida.** O pplx-decider-v1.2 é Apache-2.0, está em #2 no board com Full 69,42 e é o de maior Full entre os modelos Apache; o #1, mammoth-1, tem licença `other` na API do Hugging Face.

## Decisão

1. **Sem destilação de ensemble.** O aluno não aprende a média de professores. Nenhum combinador calibrado superou o melhor membro Apache (fatos 1 e 2).
2. **Ponto de partida: pplx-decider-v1.2.** O merge de backbones (F5) vira experimento opcional e barato; se rodar, o G3 continua sendo o critério.
3. **Treino.** Fine-tune contínuo no corpus da F2 com rótulo ouro, mais autodestilação do próprio pplx v1.2 (os logits dele no mesmo corpus) para o modelo não esquecer o que já sabe. Opções embaralhadas por exemplo, 1 época, média de seeds.
4. **Pensar sob demanda (aposta principal, condicionada ao gate GT).** Nas perguntas de raciocínio, o Qwen3.8-27B pensa (thinking) e o decisor lê o estado e o raciocínio e devolve as probabilidades. Um gate decide quando pensar, pela confiança do decisor e pelo tipo de pergunta. O pensador só entra no plano se a F2b passar o GT: acerto ≥ +10 pontos sobre o melhor decisor na amostra, com IC 95% acima de zero.
5. **Entrega mínima.** O modo só-decisor continua sendo a entrega mínima. É o que vale se o GT falhar, se o board não aceitar a latência ou se o lab não aceitar o raciocínio.
6. **Dados do pensador.** O raciocínio entra no pedido como um bloco opcional `Raciocínio:` depois do estado (formato em `decide/data/SCHEMA.md`). Os traços do pensador vêm só do corpus de treino da F2. A amostra do GT e a suíte pública são dado de avaliação: nem os pedidos nem os traços delas entram no treino.

## Consequências

- A F2b vira o próximo gate, antes de qualquer treino que dependa de raciocínio. Se o GT falhar, a F3 só faz autodestilação, o F6 treina o decisor sem bloco de raciocínio e o plano segue só-decisor.
- O custo de inferência sobe quando o pensador roda, porque o Qwen3.8-27B gera os tokens de raciocínio antes de o decisor ler. Nos pesos, cabe na GPU do lab (96 GB): pensador em FP8 (~27 GB) mais decisor em bf16 (~52 GB). A conta não inclui o KV cache do raciocínio (**UNVERIFIED**).
- Risco de regra: a mediana, a média e o p80 de latência têm de ficar em até 1.000 ms (fato 4). Como a média entra na regra, uma fração pequena de requisições com raciocínio longo já pode estourá-la. A fração máxima de requisições em que o pensador pode rodar e o orçamento de tokens por pergunta não foram medidos (**UNVERIFIED**). Os testes privados do lab também rodam com o raciocínio e demoram mais. O gate de pensar tem de ser conservador, e o Reasoning board pode ser o lugar certo do modo com raciocínio.
- A rotulagem da F3 fica mais barata: um professor só (o pplx v1.2) em vez de dois. Os traços do pensador (vLLM, thinking ligado, orçamento de tokens por pergunta) só são gerados se o GT passar.
- O teto de +0,55 do fato 2 depende de modelos CC BY-NC. Com pesos Apache não há ganho medido por combinação, então o ganho esperado do plano vem do treino e do pensador, e os dois são hipóteses até o G2 e o GT medirem.
- Os demais riscos e consequências da ADR 0003 continuam valendo, inclusive o de o privado caseiro não correlacionar com o Full.
- Esta ADR só muda por outra ADR. Os números do plano mudam por nova versão do `decide/PLAN.md`, com a data e o que foi medido.
