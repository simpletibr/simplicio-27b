# RELEASE_EVAL: Registro Oficial de Avaliações do Simplicio 27B

Documento de liberação de resultados auditados sob o protocolo de pré-registro [PREREG_eval.md](PREREG_eval.md) (SHA-256 travado em [PREREG_sha256.txt](PREREG_sha256.txt)).

---

## 1. Status Geral da Liberação

- **Checkpoint Avaliado:** `wesleysimplicio/Simplicio-27B` (BF16 merged)
- **Baseline de Controle:** `Qwen/Qwen3.8-27B` (BF16 base)
- **Harness:** `benchmarks/harness/harness.py` (40 tarefas executáveis)
- **Critério de Integridade:** Testes ocultos `pytest` executados em processo isolado.

---

## 2. Resultados Consolidados (Nível 1 - ADR 0001)

| Configuração | Modelo | Thinking | Tarefas Avaliadas | Pass Rate ($p$) | Intervalo Wilson 95% | Veredito Estatístico |
|---|---|---|---|---|---|---|
| A | `Qwen3.8-27B` (Base) | Desligado | 40 | *Aguardando run* | *[inf, sup]* | Controle Base |
| B | `Qwen3.8-27B` (Base) | Ligado | 40 | *Aguardando run* | *[inf, sup]* | Controle Base |
| C | `Simplicio-27B` (LoRA) | Desligado | 40 | *Aguardando run* | *[inf, sup]* | *Pendente* |
| D | `Simplicio-27B` (LoRA) | Ligado | 40 | *Aguardando run* | *[inf, sup]* | *Pendente* |

> **Nota:** Conforme estipulado no pré-registro e na ADR 0001, nenhuma alegação comparativa é aceita no README ou nos canais oficiais se os intervalos de Wilson 95% apresentarem sobreposição.

---

## 3. Registro Tarefa por Tarefa

Os resultados individuais de cada uma das 40 tarefas do harness serão publicados nesta seção acompanhados de:
- `task_id`
- Status funcional (`PASS` / `FAIL`)
- Validade do bloco SEARCH/REPLACE
- Contagem exata de tokens de entrada e saída
- Duração da inferência

---

## 4. Desvios e Auditoria

Quaisquer exceções, erros de conexão de GPU ou anomalias encontradas durante a execução oficial serão documentadas nesta seção com os respectivos hashes de log.
