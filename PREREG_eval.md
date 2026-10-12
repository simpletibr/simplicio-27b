# PREREG_eval: Pré-Registro da Avaliação Oficial do Simplicio 27B

- **Data do Pré-Registro:** 2026-10-12
- **Protocolo Base:** ADR 0001 (Comparação oficial do Simplicio 27B)
- **Metodologia de Referência:** Padrão Ekbasis (OpenInterpretability)

---

## 1. Objetivo e Hipótese

Avaliar se o fine-tuning LoRA do **Simplicio 27B** (base Qwen3.8-27B) produz ganhos mensuráveis e reproduzíveis na tarefa de geração de correções de código cirúrgicas (SEARCH/REPLACE patches) em relação ao modelo base sob o mesmo protocolo exato de inferência.

**Hipótese Nula ($H_0$):** O pass rate do Simplicio 27B em testes funcionais ocultos é indistinguível do modelo base Qwen3.8-27B (intervalos de confiança de Wilson a 95% se sobrepõem).
**Hipótese Alternativa ($H_1$):** O pass rate do Simplicio 27B supera o do modelo base com significância estatística (intervalos de confiança de Wilson a 95% disjuntos).

---

## 2. Conjunto de Tarefas (Ground Truth)

- **Harness:** `benchmarks/harness/harness.py`
- **Tarefas:** As 40 tarefas isoladas em `benchmarks/harness/tasks/`
- **Critério de Sucesso de Cada Tarefa:**
  1. O modelo produz um patch SEARCH/REPLACE sintaticamente válido.
  2. O patch é aplicado no repositório temporário isolado.
  3. A suíte oculta de testes `pytest` executa no ambiente isolado e termina com código de saída 0 (todos os testes passam).
  4. Nenhuma linha fora do escopo da modificação é corrompida.

---

## 3. Matriz de Avaliação (Nível 1 da ADR 0001)

Cada modelo é avaliado nas 40 tarefas sob duas condições de raciocínio:

| ID | Modelo | Pesos | Thinking | Temperatura | Decoding |
|---|---|---|---|---|---|
| A | `Qwen/Qwen3.8-27B` | BF16 (vLLM) | Desligado | 0.0 | Greedy |
| B | `Qwen/Qwen3.8-27B` | BF16 (vLLM) | Ligado | 0.0 | Greedy |
| C | `wesleysimplicio/Simplicio-27B` | BF16 (vLLM) | Desligado | 0.0 | Greedy |
| D | `wesleysimplicio/Simplicio-27B` | BF16 (vLLM) | Ligado | 0.0 | Greedy |

Regras fixadas antes da execução:
1. **System Prompt e Template:** Idênticos em todas as configurações (definidos no harness).
2. **Contexto:** Limitado ao mesmo teto (`MAX_MODEL_LEN = 40960`).
3. **Isolamento de Thinking:** Resultados com thinking ligado e desligado NUNCA são agregados ou misturados em médias compostas.

---

## 4. Métricas Gravadas

Para cada execução, são registrados:
- **Pass Rate Funcional ($p$):** Proporção de tarefas que passam nos testes ocultos.
- **Intervalo de Wilson a 95%:** Calculado rigorosamente como:
  $$w = \frac{p + \frac{z^2}{2n} \pm z \sqrt{\frac{p(1-p)}{n} + \frac{z^2}{4n^2}}}{1 + \frac{z^2}{n}}$$
  onde $z = 1.96$ e $n = 40$.
- **Taxa de Diff Válido:** Percentual de patches que o parser conseguiu aplicar.
- **Tokens de Entrada e Saída:** Contagem média e máxima de tokens consumidos.
- **Latência de Execução:** Tempo em segundos por tarefa.
- **Finish Reason:** Taxa de terminações válidas (`stop`) vs truncamentos (`length`).

---

## 5. Critério de Decisão (Regra de Aceite)

1. O Simplicio 27B só é declarado **superior** ao base se:
   - Seu pass rate for superior E os intervalos de Wilson 95% não apresentarem sobreposição.
   - A taxa de erros (`finish_reason != stop` ou exceções de servidor) for igual ou menor à do base.
2. Se houver sobreposição nos intervalos de Wilson, o resultado é **obrigatoriamente declarado como EMPATE**, e o relatório final registrará que a diferença não possui significância estatística para o tamanho amostral avaliado.
3. Se o pass rate for inferior, o resultado é declarado como **regressão**.

Nenhum resultado será publicado no README sem os artefatos de commit, prompt hash e JSON de saída gerados pelo harness.
