# ADR 0006: Plano C, pesquisa autônoma da receita no 4B (opcional, depois de G0 a G3)

- **Status:** Proposta. Registro de uma avaliação, não de uma troca. Nenhum agente de pesquisa foi rodado e nenhum número do artigo foi reproduzido por nós.
- **Data:** 2026-10-21
- **Plano:** [decide/PLAN.md](../../decide/PLAN.md), seção 10.
- **Relacionadas:** #64 (épico), #69 (F4, ensaio 4B), #71 (F6, dia alugado), [ADR 0003](0003-simplicio-decide-27b.md), [ADR 0004](0004-pensar-sob-demanda.md), [ADR 0005](0005-plano-b-base-alternativa.md).
- **Escopo:** o plano atual (base Qwen3.8-27B, ponto de partida pplx-decider-v1.2, gates G0 a G5) **continua como está**. O Plano C **não substitui nada e não mexe em nenhum gate**. Ele só diz quando, e com que limites, vale a pena usar agentes de pesquisa para procurar a receita do ensaio 4B (G2).

Regra de evidência: só número medido entra como fato, com a fonte ao lado. Projeção ou estimativa leva a marca **UNVERIFIED** e não decide nada sozinha. As citações do artigo foram conferidas no texto extraído do PDF (as linhas citadas são desse texto, não do PDF paginado).

## Contexto

1. **O artigo.** Athanor: An Agent-Native System for Autonomous Post-Training Research (Zhou et al., arXiv 2610.21055v1, 15 de outubro de 2026). Um sistema em que agentes de IA conduzem o pós-treino de ponta a ponta.

2. **O que o artigo mediu.** Fonte: o texto do artigo; linhas do texto extraído entre parênteses.

   | item | valor | linhas |
   |---|---|---|
   | equipe | 4 agentes (Claude Opus 5.5 no Claude Code), cada um num nó de 8 GPUs | 316-318 |
   | ponto de partida | Qwen3-4B-Base, dados públicos, sem chamar modelo professor, sem conselho de pesquisa humano | 317-318 |
   | duração | 104 horas | 318, 529 |
   | resultado | de 15,9 para 67,3 numa suíte de desenvolvimento de seis tarefas e de 9,2 para 48,2 numa suíte separada que os agentes nunca sabem que existe; o modelo de instrução do fornecedor marca 66,5 e 47,9 | 318-321, 353-355 |
   | registro | 500 jobs e 255 avaliações completas pedidas | 804-805 |
   | GPUs | os jobs seguraram 3.086 das 3.315 horas de GPU disponíveis | 737-738 |
   | custo | a US$ 3,49 por hora de GPU (mediana de 41 provedores): cerca de US$ 10.800 de GPU e US$ 11.800 no total, dos quais a inferência dos agentes é 9% | 747-748, 787-790 |
   | curva de custo | a fronteira chega a cerca de 50 depois de US$ 1.100 e a cerca de 60 depois de US$ 2.400; os últimos seis pontos custam cerca de US$ 8.000 | 790-792 |
   | viés de semente | a primeira semente é em média 0,4 ponto maior que as seguintes, e no máximo 1,1; o resultado reportado é a média de várias sementes | 605-609 |
   | convergência | os quatro agentes escolhem, na primeira hora, o mesmo corpus público de raciocínios longos destilados de um modelo mais forte (AM-Thinking-v1-Distilled) | 1010-1014 |

3. **O que o próprio artigo diz que não sabe.** Uma campanha, um modelo base e um modelo de agente; se a receita passa para modelos maiores ou outras famílias "é não testado" (1221-1228). O modelo não chega ao teto do fornecedor (os dois lançamentos com raciocínio marcam mais), e boa parte dos dados públicos foi destilada de modelos mais fortes pelos editores, então o resultado mostra o que agentes constroem de recursos públicos, não da base sozinha (1230-1235). Correção dos dados foi checada menos que formato: um agente deixou uma verdade-terreno suspeita para um filtro que não rodou, e um conversor marcou soluções como verificadas sem compará-las com a esperada (2187-2191).

4. **O nosso ponto de comparação (medido).** O ensaio 4B parte do `pplx-decider-flash-v1.2` (base Qwen3.5-4B), que tem Full **54,29** no placar (`index.json` do board gerado em 2026-10-21T15:08:18Z e baixado em 2026-10-21, campo `models[].v03.v`; `rank_v` 67 entre os 219 modelos do arquivo, que o board regenera, então o número muda; um arquivo anterior, usado na ADR 0005, tinha outro valor). O gate do ensaio é o G2: o novo tem de ficar pelo menos **+1,0** acima do ponto de partida no privado caseiro, nas mesmas linhas (PLAN, seção 5). O placar é só a referência do ponto de partida; o gate é medido no privado caseiro.

5. **Por que isto não é o mesmo problema.** O Athanor parte de um modelo **base** e produz um generalista de conversa, medido em tarefas de capacidade geral. Nós partimos de um decisor já treinado, com uma cabeça de 255 códigos, e medimos habilidade corrigida pelo acaso em probabilidades. Que o ganho do artigo passe para essa tarefa é **UNVERIFIED**: o artigo não mede decisão em uma passada.

## Decisão

1. **O plano atual não muda.** Nenhuma fase, gate ou issue muda por esta ADR, e nenhuma hora de GPU nem de agente é gasta com o Plano C antes do gatilho abaixo.
2. **Plano C1, o único que registramos: busca de receita guiada por agente, no 4B.** Um ou mais agentes propõem e rodam variações da receita do `decide.train` (mistura de fontes, `--lr`, `--lora-rank`, `--alpha`, `--epochs`, combinação de sementes) sobre o `pplx-decider-flash-v1.2`, e cada variação é medida no G2. Nada é treinado em 27B.
3. **Gatilho (decidido pelo planejador, não medido).** Só depois de **G0, G1, G2 e G3 terem sido medidos** (o carregador é fiel, o privado caseiro ordena os modelos como o board, o ensaio 4B rodou uma vez com a receita manual e o merge teve o seu resultado) **e** só se o dono aprovar, por escrito, um orçamento **além do dia alugado**. Sem os dois, o Plano C não começa.
4. **Não vale para o 27B.** O artigo teve 500 jobs (804) e o próprio texto diz que o efeito em modelos maiores é não testado (1226-1228). Um único job do dia econômico já é cerca de 14 horas de uma B200 (PLAN, seção 6, UNVERIFIED), então uma busca com centenas de jobs não cabe no 27B por ordens de grandeza.
5. **Limites que o Plano C herda e que não se negociam.**
   - **Uma fatia selada.** Os agentes do artigo nunca sabem que a suíte separada existe (320-321). O nosso privado caseiro é um conjunto só; o C1 exigiria separar dele uma fatia que o agente nunca vê nem pontua, e medir o vencedor nela. Isso reduz o poder do G1 (quantas linhas sobram é **UNVERIFIED**) e é uma decisão a tomar antes, não depois.
   - **O vencedor é re-medido.** O artigo mostra viés de seleção de 0,4 ponto em média entre a primeira semente e as seguintes (605-609), e a nossa margem é de +1,0. Um ganho do C1 só conta depois de semente nova na fatia selada, com o intervalo de confiança do bootstrap pareado do `proxy_eval compare`.
   - **Sem chamar modelo professor de fora** e **sem fonte nova que não passe por `decide/data/sources.yaml` e pela descontaminação.** Os agentes do artigo convergem num corpus destilado de modelo mais forte (1010-1014); a licença e os termos dele **não foram lidos por nós**, e o texto avisa que os editores destilaram boa parte dos dados públicos (1233-1235). No nosso projeto, texto gerado por um provedor de LLM está desligado por padrão (SCHEMA, seção 8.1) e a lista de fontes é negada por padrão.
   - **Nada de suíte pública nem de linha do privado no contexto do agente.** O agente vê notas agregadas do G2, nunca linhas. É o mesmo princípio do ADR 0004, item 6, que já proíbe dado de avaliação no treino; aqui a regra vale também para o que o agente lê.
   - **Verificação de dados antes de treinar.** O artigo mostra agentes que não conferiram a correção do que converteram (2187-2191); no nosso caso valem as checagens que já existem (`decontaminate`, `request_sha256` dos traços, manifests selados) e nenhuma é pulada pelo agente.
6. **O que aproveitamos agora, sem rodar nada.** Média de sementes para reportar (já na F4 e na F6); registro de cada job e avaliação (o `events.jsonl` do treino); e a regra de uma fatia que ninguém otimiza. As três já estão no plano ou estão aqui como requisito do C1.

## Custo (UNVERIFIED)

A ordem de grandeza que o coordenador extrapolou do orçamento do artigo para um C1 enxuto no 4B é de **cerca de 190 horas de GPU, cerca de US$ 700**. A conta não está versionada; o preço do artigo (US$ 3,49 por hora, mediana de 41 provedores, 747-748, nota 2 em 817) é de uma classe de GPU que o texto lido não nomeia e não é o da B200 do nosso perfil (PLAN, seção 6); e a inferência dos agentes (9% do total no artigo, 790) entra ou não conforme a conta: **tratar como UNVERIFIED**. Uma comparação medida no artigo: US$ 700 fica **abaixo** do primeiro ponto da curva do artigo (cerca de 50 depois de US$ 1.100, 790-791), então o que o C1 renderia a esse gasto não é algo que o artigo meça.

## O que não sabemos (UNVERIFIED)

- Se o **código do Athanor** está disponível e sob qual licença: o texto lido não traz endereço de repositório nem licença para ele.
- Se um agente consegue operar o `decide.train` e o `proxy_eval` sem ajuda (a F6 não foi rodada numa GPU).
- Se o ganho passa de um modelo base para um decisor já treinado (contexto 5).
- O custo real do C1 (seção anterior) e quantas linhas da fatia selada o G1 tolera perder.
- Os termos de uso de rodar agentes de terceiros contra o nosso repositório e o nosso privado.

## Consequências

- Nenhuma fase, gate ou issue do plano atual muda. O gatilho depende de medidas que o plano atual já produz (G0 a G3) e de uma decisão do dono.
- O Plano C não tem trabalho agendado. Se o dono aprovar o orçamento, a abertura é uma ADR nova com: o orçamento em GPU e em inferência de agente, a fatia selada, a regra de parada, a lista de fontes permitidas e o que o agente pode e não pode ler.
- Esta ADR só muda por outra ADR. Os números do plano mudam por nova versão do `decide/PLAN.md`, com a data e o que foi medido.
