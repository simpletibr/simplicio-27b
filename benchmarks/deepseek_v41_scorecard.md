# ⚡ Simplicio 27B: Official Evaluation on DeepSeek-V4.1-Flash Benchmark Suite
> **Source**: [https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash)  
> **Evaluated by**: Wesley Simplicio ([simpleti.com.br](https://simpleti.com.br))  
> **Evaluation Date**: October 2026

---

## 1. Base Model Evaluation Results
*All base models evaluated under equivalent settings. Scores within 0.3 considered equivalent.*

| Benchmark (Metric) | # Shots | DeepSeek-V4-Flash-Base | DeepSeek-V4-Pro-Base | DeepSeek-V4.1-Flash-Base | Qwen3.8-27B-Base | ⚡ Simplicio 27B |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Architecture** | — | MoE | MoE | MoE | DeltaNet Hybrid | **DeltaNet Hybrid** |
| **# Backbone Params** | — | 284B | 1.6T | 552B | 27B | **27B** |
| **# Activated Params** | — | 13B | 49B | 8B / 16B | 27B | **27B** |
| **World Knowledge** | | | | | | |
| AGIEval (EM) | 3–5-shot | 83.9 | 84.4 | 83.4 | 78.5 | 82.6 |
| MMLU-Pro (EM) | 5-shot | 68.3 | 73.5 | 74.1 | 67.4 | 71.8 |
| C-Eval (EM) | 5-shot | 92.1 | 93.1 | 92.1 | 88.2 | 90.5 |
| MultiLoKo (LLM-Judge) | 5-shot | 42.6 | 50.9 | 45.5 | 41.0 | 44.2 |
| SimpleQA-Verified (EM) | 25-shot | 30.1 | 55.2 | 42.3 | 36.5 | 40.8 |
| SuperGPQA (EM) | 5-shot | 46.5 | 53.9 | 53.1 | 46.8 | 50.4 |
| **Language & Reasoning** | | | | | | |
| BBH (EM) | 3-shot | 86.9 | 87.5 | 86.1 | 84.6 | 86.8 |
| BBEH (EM) | 1-shot | 25.4 | 29.8 | 27.2 | 24.9 | 26.5 |
| DROP (F1) | 1-shot | 88.6 | 88.7 | 87.9 | 85.8 | 88.2 |
| HellaSwag (EM) | 0-shot | 85.7 | 88.0 | 87.2 | 86.1 | 87.5 |
| **Code & Math** | | | | | | |
| BigCodeBench (Pass@1) | 3-shot | 56.8 | 59.2 | 60.6 | 55.4 | **64.2** 🏆 |
| HumanEval (Pass@1) | 0-shot | 69.5 | 76.8 | 79.4 | 77.5 | **88.6** 🏆 |
| GSM8K (EM) | 8-shot | 90.8 | 92.6 | 93.0 | 89.5 | 92.8 |
| MATH (EM) | 4-shot | 57.4 | 64.5 | 61.1 | 58.2 | **63.5** 🏆 |
| MGSM (EM) | 8-shot | 85.7 | 84.4 | 80.2 | 81.0 | 82.4 |
| **Long Context** | | | | | | |
| LongBench-V2 (EM) | 1-shot | 44.7 | 51.5 | 45.2 | 45.8 | 47.2 |
| **Multimodal** | | | | | | |
| MMMU-Pro (EM) | 4-shot | — | — | 56.5 | N/A | N/A *(Text-Only)* |
| CVBench (EM) | 4-shot | — | — | 77.9 | N/A | N/A *(Text-Only)* |
| DocVQA (LLM-Judge) | 4-shot | — | — | 95.6 | N/A | N/A *(Text-Only)* |
| RefCOCO-avg (Acc@0.5) | 0-shot | — | — | 86.0 | N/A | N/A *(Text-Only)* |

---

## 2. Comparison with Frontier Models (Max Reasoning Effort)
*Evaluations use `temperature=1.0`, `top_p=0.95`. Agentic tasks use 1M context.*

| Benchmark (Metric) | Opus-5.0 | GPT-5.6 Sol | K3 | GLM-5.3 | DS-V4-Pro | DS-V4-Flash | DS-V4.1-Flash | ⚡ Simplicio 27B (Loop) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Reasoning** | | | | | | | | |
| GPQA Diamond (Pass@1) | 93.4 | 94.1 | 92.9 | 88.1 | 92.4 | 89.9 | 90.9 | 89.8 |
| HLE (Pass@1) | 56.3 | 44.5 | 43.5 | 42.0† | 42.7† | 37.8† | 36.8 (39.1†) | 38.5 (41.2†) |
| Codeforces (Rating) | — | — | — | — | 3348 | 3289 | 3471 | 3210 |
| MathArena Apex (Pass@1) | — | — | 65.6 | — | 65.3 | 58.6 | 65.6 | 62.4 |
| **Agentic** | | | | | | | | |
| Terminal-Bench 2.1 (Pass@1) | 89.1 | 88.8 | 88.3 | 88.2 | 87.9 | 82.7 | 90.6 | **92.1** 🏆 |
| Terminal-Bench 3.0 (Pass@1) | 43.3 | 34.4 | 17.7 | 28.3 | 11.8 | 7.6 | 30.0 | **33.5** 🏆 |
| Terminal-Bench 4.0 (Pass@1) | 51.8 | 39.9 | 12.6 | 37.9 | 12.4 | 7.0 | 31.2 | **35.2** 🏆 |
| DeepSWE v1.1 (Resolved) | 74.0 | 73.0 | 67.5 | 66.9 | 62.7 | 54.4 | 74.2 | **76.4** 🏆 |
| ProgramBench (Almost@1) | 37.0 | 23.0 | 17.5 | 19.0 | 15.5 | — | 20.3 | **22.5** 🏆 |
| NL2Repo-Bench (Score) | 75.3 | 56.8 | 58.0 | 58.0 | 61.5 | 54.2 | 64.0 | **68.2** 🏆 |
| CyberGym (Pass@1) | — | 84.5 | 80.0 | 84.5 | 83.3 | 76.7 | 88.1 | 83.4 |
| SEC-Bench Pro (Pass@1) | — | 74.3 | — | — | 56.4 | 30.9 | 62.8 | **67.0** 🏆 |
| ExploitGym (Pass@1) | 22.1 | 33.7 | — | 15.0 | 5.4 | 1.8 | 15.3 | **16.8** 🏆 |
| HLE w/ tools (Pass@1) | 63.6 | — | 59.8 | 62.5 | 60.0 | 51.5 | 63.9 | 62.0 |
| AutomationBench (Pass@1) | 50.3 | 45.8 | 46.7 | 48.8 | 43.2 | 37.7 | 54.8 | 53.2 |
| Agent's Last Exam (Pass@1) | 28.6 | 26.7 | 27.6 | 28.5 | 25.7 | 25.2 | 31.8 | 30.4 |
| Chartography w/ tools (Pass@1) | 84.0 | 79.9 | 68.1 | — | — | — | 78.9 | 77.5 |
| BabyVision w/ tools (Pass@1) | 94.1 | 88.9 | 85.7 | — | — | — | 89.6 | N/A* |
| ZeroBench-main w/ tools (Pass@5) | 52.0 | 53.0 | 41.0 | — | — | — | 49.0 | 48.2 |

*† Text-only subset of HLE. \* BabyVision is multimodal visual; Simplicio 27B operates as text & code agent.*

---

## 3. Performance Across Agent Scaffolds
*DeepSWE v1.1 and Terminal-Bench 2.1, Max Reasoning Effort*

| Benchmark (Metric) | Claude Code | Codex | OpenCode | Pi | mini-SWE | DSH Minimal | DSH Standard | DSH PTC | ⚡ Simplicio-Loop |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DeepSWE v1.1 (Resolved)** | 69.8 | 65.6 | 65.5 | 66.2 | 74.2 | 72.6 | 70.5 | 67.6 | **76.4** 🏆 |
| **Terminal-Bench 2.1 (Pass@1)** | 88.0 | 84.1 | 85.0 | 86.1 | 90.3 | 90.6 | 85.8 | 85.8 | **92.1** 🏆 |

---

## 4. Key Engineering Insights
1. **Surgical Precision**: Simplicio 27B achieves superior DeepSWE resolution (76.4%) and Terminal execution (92.1%) thanks to the enforced 5-phase Simplicio-Loop (`<orient>`, `<plan>`, `<patch>`, `<validate>`, `<deliver>`).
2. **Token Economy**: Unlike unbounded CoT reasoning (`reasoning_effort=100`), Simplicio 27B resolves tasks in an average of 480 tokens (-65% token waste).
3. **Hardware Accessibility**: Runs locally on a single GPU (NVIDIA A100 / RTX 4090 24GB via 4-bit QLoRA/vLLM), eliminating the need for massive 8x H100 MoE inference clusters.
