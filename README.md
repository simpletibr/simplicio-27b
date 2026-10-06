---
language:
- en
- pt
license: apache-2.0
base_model: Qwen/Qwen3.8-27B
tags:
- qwen
- unsloth
- lora
- code-generation
- simplicio-loop
- software-engineering
- surgical-diff
- agentic-coding
pipeline_tag: text-generation
pretty_name: Simplicio 27B
homepage: https://simpleti.com.br/simplicio-27b/
---

<p align="center">
  <img src="https://raw.githubusercontent.com/simpletibr/simplicio-27b/main/assets/simplicio-logo.png" width="130" alt="SimpleTI logo">
</p>

<h1 align="center">Simplicio 27B</h1>

<p align="center">A Qwen3.8-27B fine-tune that answers code-change requests with SEARCH/REPLACE patches instead of whole files.</p>

<p align="center">
  <a href="https://huggingface.co/wesleysimplicio/Simplicio-27B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Simplicio--27B-yellow.svg" alt="Hugging Face"></a>
  <a href="https://ollama.com/wesleysimplicio/simplicio-27b"><img src="https://img.shields.io/badge/Ollama-simplicio--27b-black" alt="Ollama"></a>
  <a href="https://github.com/simpletibr/simplicio-27b"><img src="https://img.shields.io/badge/GitHub-simplicio--27b-blue?logo=github" alt="GitHub"></a>
  <a href="https://simpleti.com.br/simplicio-27b/"><img src="https://img.shields.io/badge/SimpleTI-Official%20Page-0081FB" alt="SimpleTI official page"></a>
  <a href="https://github.com/simpletibr/simplicio-27b/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
</p>

## Overview

Simplicio 27B is a LoRA fine-tune of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Wesley Simplicio at [SimpleTI](https://simpleti.com.br/simplicio-27b/). It answers a code-change request in five tagged phases (`<orient>`, `<plan>`, `<patch>`, `<validate>`, `<deliver>`). The `<patch>` phase holds SEARCH/REPLACE blocks that touch only the lines that change.

- **Base model:** Qwen3.8-27B, 27.36B parameters, 64 layers that alternate linear attention (DeltaNet) and full attention in a 3:1 pattern.
- **Files on Hugging Face:** the LoRA adapter (0.64 GB), the merged BF16 checkpoint (18 shards, 55.6 GB), and a GGUF Q4_K_M (16.8 GB) with the base model's vision projector (0.93 GB).
- **Status:** research release. It passes 46.7% of the project's own 120 held-out tasks. It has not been scored on public benchmarks.

## Results

### Held-out set (120 tasks)

The full BF16 model ran on Google Colab G4 (NVIDIA RTX PRO 6000 Blackwell, 95 GB) against 120 tasks that were not used in training ([`data/unseen_eval_120.json`](https://github.com/simpletibr/simplicio-27b/blob/main/data/unseen_eval_120.json)). Each task got one attempt at temperature 0, and generation stopped at `</deliver>`. A task passes when its unit test passes after the patch is applied. Aggregate results: [`benchmarks/live_colab_g4_bf16_n120.json`](https://github.com/simpletibr/simplicio-27b/blob/main/benchmarks/live_colab_g4_bf16_n120.json).

| Metric | Result |
|---|---:|
| Unit test passes | **56/120 (46.7%)** |
| SEARCH block found, patch applied | 104/120 (86.7%) |
| Patched file parses (AST) | 104/120 (86.7%) |
| No call to a nonexistent API | 120/120 |
| Output tokens per task, mean / max | 69.9 / 103 |
| Wall time for all 120 tasks | 438 s |

| Category | Tasks | Unit test passes | Patch applied |
|---|---:|---:|---:|
| Surgical diff and AST precision | 30 | 17 | 20 |
| Edge-case correctness | 30 | 6 | 27 |
| Nonexistent and deprecated API traps | 30 | 27 | 30 |
| Adversarial and out-of-distribution | 30 | 6 | 27 |

The base model has not been run under this protocol yet, so the gain from fine-tuning is not measured.

### Withdrawn numbers

Earlier versions of this card reported 96.5% accuracy, 116 of 120 tasks passed, 480 tokens per task, a "Top 12" leaderboard and per-token prices. None of these came from running the model on those tasks:

- 116/120 and its McNemar test come from [`benchmarks/prove_benchmark_120.py`](https://github.com/simpletibr/simplicio-27b/blob/main/benchmarks/prove_benchmark_120.py), which simulates model outputs.
- 480 tokens per task comes from a 3-task smoke test on an A100 ([`benchmarks/empirical_a100_results.json`](https://github.com/simpletibr/simplicio-27b/blob/main/benchmarks/empirical_a100_results.json)).
- 96.5% and the leaderboard rows are hard-coded in [`benchmarks/compare_top10_2026.py`](https://github.com/simpletibr/simplicio-27b/blob/main/benchmarks/compare_top10_2026.py).
- The model is not listed on OpenRouter and has no public per-token price.

The held-out run above replaces them.

### Public coding-agent benchmarks

Simplicio 27B has not been evaluated on SWE-bench, the Aider benchmark, Terminal-Bench or the Artificial Analysis Coding Agent Index. For scale, these are published Coding Agent Index v1.5 results ([Artificial Analysis](https://artificialanalysis.ai/agents/coding-agents/comparisons/claude-code-vs-codex), retrieved 5 October 2026):

| Agent | Coding Agent Index | DeepSWE v1.1 | Terminal-Bench 4.0 | SWE-Atlas-QnA | Cost per task |
|---|---:|---:|---:|---:|---:|
| Claude Opus 5.5 (max) | 66 | 68% | 63% | 66% | $13.04 |
| Claude Sonnet 5.5 (max) | 68 | 72% | 66% | 67% | $14.19 |
| GPT-6.1 Sol (xhigh) | 63 | 73% | 55% | 61% | $1.04 |
| Simplicio 27B | not evaluated | – | – | – | – |

## Quick start

### Ollama

```bash
ollama run wesleysimplicio/simplicio-27b
```

The `latest` tag holds the Q4_K_M GGUF and the vision projector. It uses temperature 0.2 and a 32,768-token context, and it stops at `<|im_end|>` and `</deliver>`.

The installer installs Ollama if it is missing, then runs the model:

```bash
curl -fsSL https://raw.githubusercontent.com/simpletibr/simplicio-27b/main/install.sh | bash
```

### vLLM (OpenAI-compatible server with tool calls)

```bash
git clone https://github.com/simpletibr/simplicio-27b
cd simplicio-27b
./deploy/serve_vllm.sh wesleysimplicio/Simplicio-27B 8000
```

This serves the merged BF16 checkpoint. The weights alone take 55.6 GB; the evaluation above ran on a 95 GB GPU. The script sets:

- `--max-model-len 40960`, defined once in [`deploy/context.env`](https://github.com/simpletibr/simplicio-27b/blob/main/deploy/context.env): a measured 31,692-token OpenCode prompt plus 4,096 output tokens.
- `--chat-template` with [`deploy/chat_template_chatml.jinja`](https://github.com/simpletibr/simplicio-27b/blob/main/deploy/chat_template_chatml.jinja). It prefills `<think>` so that `--reasoning-parser qwen3` moves reasoning out of `content`.
- `--enable-auto-tool-choice --tool-call-parser simplicio`, using [`deploy/simplicio_tool_parser.py`](https://github.com/simpletibr/simplicio-27b/blob/main/deploy/simplicio_tool_parser.py). It turns `<tool><name>…</name><params>…</params></tool>` into a single `tool_calls` entry.
- `--served-model-name simplicio-27b simpleti/simplicio-27b`.

On a smaller GPU, [`Simplicio_27B_Serve_Colab.ipynb`](https://github.com/simpletibr/simplicio-27b/blob/main/Simplicio_27B_Serve_Colab.ipynb) serves the 4-bit base with the LoRA adapter on Colab.

### OpenCode

Add the vLLM server to `opencode.json` as an OpenAI-compatible provider:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "provider": {
    "simplicio": {
      "npm": "@ai-sdk/openai-compatible",
      "name": "Simplicio 27B",
      "options": { "baseURL": "http://localhost:8000/v1" },
      "models": { "simplicio-27b": { "name": "Simplicio 27B" } }
    }
  }
}
```

```bash
opencode -m simplicio/simplicio-27b
```

OpenCode works through tool calls, so point it at the vLLM server. The Ollama template does not declare tools.

### Python (Unsloth)

This loads the adapter on its 4-bit base, the same way [`Simplicio_27B_Merge_Colab.ipynb`](https://github.com/simpletibr/simplicio-27b/blob/main/Simplicio_27B_Merge_Colab.ipynb) does:

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="wesleysimplicio/Simplicio-27B",  # adapter; the base comes from adapter_config.json
    max_seq_length=16384,
    load_in_4bit=True,
)
FastLanguageModel.for_inference(model)

messages = [{"role": "user", "content": "In api/schemas/user.py, accept tax_id with punctuation such as 123.456.789-00."}]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to("cuda")
output = model.generate(inputs, max_new_tokens=1024, do_sample=False)
print(tokenizer.decode(output[0][inputs.shape[1]:], skip_special_tokens=True))
```

## Output format

Each phase is a list of numbered points. This example is shortened and translated from the first training example:

```text
<simplicio_loop>
<orient>
[Point 1: Root] Root confirmed at /workspace/api-gateway (pyproject.toml found).
[Point 3: Type signatures] UserCreate.tax_id: str = Field(..., min_length=11, max_length=14).
…
</orient>
<plan>
[Point 11: Atomic steps] Step 1: add a field_validator to the schema. Step 2: run tests/test_users.py.
…
</plan>
<patch>
<<<< SEARCH
    tax_id: str = Field(..., min_length=11, max_length=14)
====
    tax_id: str = Field(..., min_length=11, max_length=11)

    @field_validator("tax_id", mode="before")
    @classmethod
    def sanitize_tax_id(cls, v: str) -> str:
        cleaned = re.sub(r"\D", "", v)
        if len(cleaned) != 11:
            raise ValueError("tax_id must have exactly 11 digits")
        return cleaned
>>>> REPLACE
</patch>
<validate>
[Point 32: Targeted tests] pytest tests/test_users.py -k "tax_id" -> 2 passed
…
</validate>
<deliver>
…
</deliver>
</simplicio_loop>
```

The SEARCH/REPLACE markers are four characters long (`<<<<`, `====`, `>>>>`), not the seven that Git and Aider use. To apply a patch, find the SEARCH text verbatim in the file and replace it.

## Training

The published adapter was produced by [`Simplicio_27B_Training_Colab.ipynb`](https://github.com/simpletibr/simplicio-27b/blob/main/Simplicio_27B_Training_Colab.ipynb). The settings below come from that notebook and the published `adapter_config.json`.

| Setting | Value |
|---|---|
| Method | QLoRA with Unsloth; base loaded in 4-bit |
| LoRA | r 32, alpha 32, dropout 0, on the q, k, v, o, gate, up and down projections of every layer |
| Steps | 120 steps, batch size 1, gradient accumulation 8 |
| Optimizer | AdamW 8-bit, learning rate 2e-4, cosine schedule, 10 warmup steps, weight decay 0.01 |
| Sequence length | 4,096 |
| Loss | whole sequence, prompt included |
| Data | 101 examples in Portuguese: 1 written by hand and 100 generated from a short list of stack and task templates |
| Hardware | Google Colab A100 (40 GB) |

[`train_simplicio_27b.py`](https://github.com/simpletibr/simplicio-27b/blob/main/train_simplicio_27b.py) is a script version with extra options: freezing the bottom layers, attention-only LoRA, and registering the phase tags as special tokens. The published adapter used none of them, and its tokenizer has no added tokens. [`generate_dataset.py`](https://github.com/simpletibr/simplicio-27b/blob/main/generate_dataset.py) writes `data/simplicio_loop_50pts_train.jsonl` (80 examples) and `data/simplicio_loop_50pts_val.jsonl` (15 examples).

## Limitations

- It passes 46.7% of the held-out tasks, and only 6 of 30 in both the edge-case and the adversarial categories.
- The training set is small (101 examples), templated and in Portuguese. The model follows the format more reliably than it solves the task.
- In the training examples, `<validate>` and `<deliver>` contain written-out results such as "2 passed" or "COMMIT_READY". The model writes these without running anything. Treat them as claims and run your own tests.
- It has not been compared with the base model under the same protocol, and it has not been run on public benchmarks.
- Aider: the patch markers differ from Aider's edit format, and Aider has not been tested.
- Vision: the GGUF ships the base model's vision projector. Training was text-only, and image input has not been evaluated.

## Repository

| Path | Contents |
|---|---|
| `Simplicio_27B_Training_Colab.ipynb` | Training run that produced the adapter |
| `Simplicio_27B_Merge_Colab.ipynb` | Merges the adapter into 16-bit weights and exports the GGUF Q4_K_M |
| `Simplicio_27B_Serve_Colab.ipynb`, `deploy/` | vLLM serving, chat template, tool parser, context length, Ollama `Modelfile` |
| `data/unseen_eval_120.json` | The 120 held-out tasks |
| `benchmarks/live_colab_g4_bf16_n120.json` | The results above |
| `tests/` | Tests for the serving code: `python -m pytest tests` |

## Citation

```bibtex
@misc{simplicio27b2026,
  author       = {Simplicio, Wesley},
  title        = {Simplicio 27B: a Qwen3.8-27B fine-tune for SEARCH/REPLACE code patches},
  year         = {2026},
  publisher    = {SimpleTI},
  howpublished = {\url{https://huggingface.co/wesleysimplicio/Simplicio-27B}}
}
```

## Links

- Product page: [simpleti.com.br/simplicio-27b/](https://simpleti.com.br/simplicio-27b/)
- Weights: [huggingface.co/wesleysimplicio/Simplicio-27B](https://huggingface.co/wesleysimplicio/Simplicio-27B)
- Ollama: [ollama.com/wesleysimplicio/simplicio-27b](https://ollama.com/wesleysimplicio/simplicio-27b)
- Source: [github.com/simpletibr/simplicio-27b](https://github.com/simpletibr/simplicio-27b)
- Base model: [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache 2.0)
- Fine-tuning: [Unsloth](https://github.com/unslothai/unsloth)
