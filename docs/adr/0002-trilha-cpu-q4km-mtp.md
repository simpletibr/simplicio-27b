# ADR 0002: Trilha CPU com GGUF Q4_K_M e decodificação especulativa MTP

- **Status:** Rascunho. Não aceita até os resultados da trilha estarem medidos.
- **Data:** 2026-10-08
- **Emenda:** [ADR 0001](0001-comparacao-oficial-simplicio-27b.md), no ponto "pesos BF16, servidos por vLLM".
- **Relacionadas:** #27 (execução), #60 (issue da comparação oficial), #61 (limpeza do README).

## Contexto

A ADR 0001 define a comparação oficial com pesos BF16 servidos por vLLM, o que exige GPU. Hoje não há GPU disponível: a conta de Colab usada não tem compute units, e a VPS é um Xeon E5-2699 v4 com 10 vCPU, AVX2, sem AVX-512 e sem GPU, com 31 GB de RAM.

A única trilha executável nesse hardware é o GGUF Q4_K_M com llama.cpp. Ela não substitui a trilha BF16: os números dela não são comparáveis com os da trilha de GPU, e precisam ser reportados separadamente.

## Decisão proposta

1. **Trilha CPU.** O servidor é o `llama-server` do llama.cpp, commit `a11f57b` (build `0.6.0-dev`), compilado com `GGML_NATIVE=ON` e `LLAMA_CURL=OFF`.
2. **Arquivo do Simplicio.** `Qwen3.8-27B.Q4_K_M.gguf` de `wesleysimplicio/Simplicio-27B` no Hugging Face, 16.810.715.584 bytes, SHA-256 `25371da44b53f5b3b6d8beb4208a2b39d545e557842beff22269bdc2e3bcd311`, conferido após o download.
3. **Comparadores na mesma trilha.** O base precisa ter o mesmo quant e o mesmo conversor. Cada run registra a origem e o SHA-256 de cada arquivo. Um base Q4_K_M de outro conversor (por exemplo, o `UD-Q4_K_M` do Unsloth, que é um quant dinâmico) é reportado como outra linha, e não como o controle.
4. **Decodificação.** Greedy (temperatura 0), igual à trilha BF16 e ao harness.
5. **Speculative decoding por MTP** (`--spec-type draft-mtp`). Fica permitido somente se a saída for idêntica à da decodificação sem especulação, tarefa por tarefa, no mesmo run. A igualdade é verificada, não assumida. A evidência atual é de um prompt, com saída idêntica (166 caracteres) e 40 de 54 tokens de draft aceitos. Isso é insuficiente para uma afirmação de velocidade.
6. **Métricas por run:** pass rate com intervalo de Wilson 95%, tokens de entrada e de saída, latência por tarefa, velocidade de prefill e de geração (tok/s), e o estado do cache de páginas (frio ou quente) no início do run.
7. **Critério de decisão:** o mesmo da ADR 0001.

## Limites

- Velocidade é propriedade do hardware e do estado do sistema, não do modelo. Um número de tok/s só vale junto com CPU, RAM, carga e estado do cache.
- O ganho do MTP está medido em um prompt. Não é um resultado geral.
- O `simplicio-local` foi avaliado e descartado para esta trilha: o engine dele é o colibri, que implementa só GLM e OLMoE, e o adapter de Qwen é um stub. O caminho real de Qwen3.8 nele é o próprio `llama-server`.
- Qualquer download de arquivo grande durante um run pode expulsar as páginas do modelo da RAM e degradar a velocidade. Por isso downloads não são feitos durante um run.

## Consequências

- Nenhum número de tok/s entra no README sem o hardware, a carga e o estado do cache registrados junto.
- Os resultados da trilha CPU e da trilha BF16 aparecem em tabelas separadas, cada uma com sua coluna de origem.
