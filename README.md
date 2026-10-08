---
language:
- en
- pt
license: apache-2.0
library_name: transformers
base_model: Qwen/Qwen3.8-27B
base_model_relation: finetune
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

<p align="center">A Qwen3.8-27B fine-tune trained to answer code-change requests with SEARCH/REPLACE patches instead of whole files.</p>

<p align="center">
  <a href="https://huggingface.co/wesleysimplicio/Simplicio-27B"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Simplicio--27B-yellow.svg" alt="Hugging Face"></a>
  <a href="https://ollama.com/wesleysimplicio/simplicio-27b"><img src="https://img.shields.io/badge/Ollama-simplicio--27b-black" alt="Ollama"></a>
  <a href="https://github.com/simpletibr/simplicio-27b"><img src="https://img.shields.io/badge/GitHub-simplicio--27b-blue?logo=github" alt="GitHub"></a>
  <a href="https://simpleti.com.br/simplicio-27b/"><img src="https://img.shields.io/badge/SimpleTI-Official%20Page-0081FB" alt="SimpleTI official page"></a>
  <a href="https://github.com/simpletibr/simplicio-27b/blob/main/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-green.svg" alt="License: Apache 2.0"></a>
</p>

## Overview

Simplicio 27B is a LoRA fine-tune of [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) by Wesley Simplicio at [SimpleTI](https://simpleti.com.br/simplicio-27b/). It was trained to answer a code-change request in five tagged phases (`<orient>`, `<plan>`, `<patch>`, `<validate>`, `<deliver>`). The `<patch>` phase holds SEARCH/REPLACE blocks that touch only the lines that change.

- **Base model:** Qwen3.8-27B, 27.36B parameters, 64 layers that alternate linear attention (DeltaNet) and full attention in a 3:1 pattern.
- **Files on Hugging Face:** the LoRA adapter in `lora/` (0.64 GB), the merged BF16 checkpoint (18 shards, 55.6 GB), and a GGUF Q4_K_M (16.8 GB) with the base model's vision projector (0.93 GB). Image input is inherited from the base model and has not been evaluated.
- **Status:** research release. It has not been scored on a reproducible benchmark yet. See [Benchmarks](#benchmarks).

## Benchmarks

No benchmark result is published in this README yet. Numbers that were not produced by a reproducible run have been removed.

The benchmark harness is in [`benchmarks/harness/`](https://github.com/simpletibr/simplicio-27b/tree/main/benchmarks/harness). It runs 40 SEARCH/REPLACE tasks against any OpenAI-compatible endpoint. A task passes only when the hidden pytest tests pass on the patched code. A result will be published here only together with its commit, the model file and its checksum, the prompt hash, the decoding settings, and the per-task output.

Not evaluated: SWE-bench, Aider, Terminal-Bench, the Artificial Analysis indices, and image input.

## Quick start

### Ollama

```bash
ollama run wesleysimplicio/simplicio-27b --think=false
```

If Ollama is not installed, get it from [ollama.com/download](https://ollama.com/download). The `latest` tag is built from the root [`Modelfile`](https://github.com/simpletibr/simplicio-27b/blob/main/Modelfile) and needs Ollama 0.30.0 or newer. The Modelfile uses the Q4_K_M GGUF and leaves out the base model's vision projector. The published tag's manifest lists a projector layer too, so the two do not match yet. Image input has not been evaluated. It sets:

- `RENDERER qwen3.5` and `PARSER qwen3.5`: Ollama's built-in Qwen 3.5 prompt format. Tool calls come back in `message.tool_calls` and reasoning in `message.thinking`.
- `num_ctx 40960`: a 40,960-token context, the same as vLLM ([`deploy/context.env`](https://github.com/simpletibr/simplicio-27b/blob/main/deploy/context.env)).
- Qwen's recommended sampling for Qwen3.8-27B in non-thinking mode: `temperature 0.7`, `top_p 0.8`, `top_k 20`, `min_p 0`, `presence_penalty 1.5`, `repeat_penalty 1`. The benchmark harness uses greedy decoding, so to match it send `"options": {"temperature": 0}`. The tag stops only at `<|im_end|>`; add `"stop": ["</deliver>"]` inside `options` to also stop at the end of the answer.
- The system prompt used in training (see "Prompt format" below).

**Thinking.** The training data has no `<think>` blocks, so run with thinking off. Ollama turns thinking on by default for models that support it, so turn it off on every request:

| Client | Thinking off |
|---|---|
| `ollama run` | `--think=false`, or `/set nothink` inside a session |
| `/api/chat` | `"think": false` |
| `/v1/chat/completions` | `"reasoning_effort": "none"` |

With thinking on, the model reasons first and the reasoning goes to `message.thinking`. That mode has not been evaluated.

**Prompt format.** Training used this system prompt and user message. Ollama adds the system prompt by itself; with vLLM or transformers, send it as the `system` message:

```text
system: Voce e o Simplicio 27B, treinado para executar tarefas de desenvolvimento seguindo rigorosamente os 50 pontos do Simplicio-Loop: Orientacao, Planejamento, Edicao Cirurgica por Diff, Validacao e Entrega Verificada sem alucinacao.
user:   Contexto: <files, stack and bug report>
        Tarefa: <the change to make>
```

**Not evaluated.** The Q4_K_M file has not been scored. Image input has not been evaluated. Training was text-only. The training data has no tool calls, so tool calling is whatever the base model does.

### vLLM (OpenAI-compatible server with tool calls)

```bash
pip install vllm==0.31.0
git clone https://github.com/simpletibr/simplicio-27b
cd simplicio-27b
./deploy/serve_vllm.sh wesleysimplicio/Simplicio-27B 8000
```

This serves the merged BF16 checkpoint. The weights alone take 55.6 GB. [`deploy/serve_vllm.sh`](https://github.com/simpletibr/simplicio-27b/blob/main/deploy/serve_vllm.sh) is the only serve command, and the Colab notebook runs it too. It sets:

- `--max-model-len 40960`, defined once in [`deploy/context.env`](https://github.com/simpletibr/simplicio-27b/blob/main/deploy/context.env): the total context window.
- No `--chat-template`: vLLM uses the model's own `chat_template.jinja`.
- `--default-chat-template-kwargs '{"enable_thinking": false}'`: thinking is off by default because the training data has no `<think>` blocks. A request can turn it on with `"chat_template_kwargs": {"enable_thinking": true}`.
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`: vLLM's built-in parsers for Qwen's native `<tool_call><function=…>` format.
- `--enable-force-include-usage`: every response carries `usage`.
- `--served-model-name simplicio-27b simpleti/simplicio-27b`: both ids reach the same fine-tuned weights.
- Listens on `127.0.0.1` by default. To serve other machines, put an HTTPS proxy in front that forwards only `GET /v1/models` and `POST /v1/chat/completions`, and set `VLLM_API_KEY=<random token>` so vLLM requires `Authorization: Bearer <token>` on `/v1/*`.

`vllm serve` has no stop flag. `<|im_end|>` already ends generation through `generation_config.json`; to end at `</deliver>`, send `"stop": ["</deliver>"]` in the request.

[`notebooks/Simplicio_27B_Serve_Colab.ipynb`](https://github.com/simpletibr/simplicio-27b/blob/main/notebooks/Simplicio_27B_Serve_Colab.ipynb) runs the same script on a Colab G4 (RTX PRO 6000, 96 GB) behind the completion gate and a Cloudflare tunnel.

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

OpenCode works through tool calls, so point it at the vLLM server. OpenCode has not been tested against the Ollama tag.

### Python (Transformers)

The Hugging Face repository root is a plain Transformers checkpoint: the merged BF16 weights (55.6 GB). `do_sample=False` is greedy decoding, the setting the benchmark harness uses; without it, `generate` uses the sampling defaults in `generation_config.json`, the same values as the Ollama `Modelfile`.

```python
import torch
from transformers import AutoModelForImageTextToText, AutoTokenizer

repo = "wesleysimplicio/Simplicio-27B"
tokenizer = AutoTokenizer.from_pretrained(repo)
model = AutoModelForImageTextToText.from_pretrained(repo, dtype=torch.bfloat16, device_map="auto")

messages = [{"role": "user", "content": "In api/schemas/user.py, accept tax_id with punctuation such as 123.456.789-00."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, enable_thinking=False, return_tensors="pt"
).to(model.device)
output = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

The LoRA adapter is in [`lora/`](https://huggingface.co/wesleysimplicio/Simplicio-27B/tree/main/lora). To apply it to your own copy of the base model, load the base with `AutoModelForImageTextToText`, the class whose module names (`model.language_model.layers.*`) match the adapter:

```python
import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

base = AutoModelForImageTextToText.from_pretrained("Qwen/Qwen3.8-27B", dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(base, "wesleysimplicio/Simplicio-27B", subfolder="lora")
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

The published adapter was produced by [`notebooks/Simplicio_27B_Training_Colab.ipynb`](https://github.com/simpletibr/simplicio-27b/blob/main/notebooks/Simplicio_27B_Training_Colab.ipynb). The settings below come from that notebook and the published `lora/adapter_config.json`.

| Setting | Value |
|---|---|
| Method | QLoRA with Unsloth; base loaded in 4-bit |
| LoRA | r 32, alpha 32, dropout 0. q, k, v, o projections on the 16 full-attention layers (3, 7, …, 63); gate, up, down projections on all 64 layers. The `linear_attn` projections of the 48 DeltaNet layers and the vision tower were not adapted (512 LoRA tensors in the published adapter) |
| Steps | 120 steps, batch size 1, gradient accumulation 8 |
| Optimizer | AdamW 8-bit, learning rate 2e-4, cosine schedule, 10 warmup steps, weight decay 0.01 |
| Sequence length | 4,096 |
| Loss | whole sequence, prompt included |
| Data | 101 examples in Portuguese: 1 written by hand and 100 generated from a short list of stack and task templates |
| Hardware | Google Colab A100 (40 GB) |

An earlier script, [`train_simplicio_27b.py`](https://github.com/simpletibr/simplicio-27b/blob/b8567df/train_simplicio_27b.py), was removed because its defaults do not reproduce the published adapter: 150 steps, the bottom 48 layers frozen, and the 10 phase tags added as special tokens. The published adapter used none of these, and its tokenizer has no added tokens. [`generate_dataset.py`](https://github.com/simpletibr/simplicio-27b/blob/main/generate_dataset.py) writes `data/simplicio_loop_50pts_train.jsonl` (80 examples) and `data/simplicio_loop_50pts_val.jsonl` (15 examples).

## Limitations

- No functional correctness of the patches has been measured yet.
- The training set is small (101 examples), templated and in Portuguese. Whether outputs follow the five-phase format was not measured.
- In the training examples, `<validate>` and `<deliver>` contain written-out results such as "2 passed" or "COMMIT_READY". The model writes these without running anything. Treat them as claims and run your own tests.
- It has not been compared with the base model under the same protocol, and it has not been run on public benchmarks.
- No checkpoint has been evaluated yet. The BF16 weights, the Q4_K_M GGUF, the Ollama template and parameters, and the 4-bit base + LoRA path are all unmeasured.
- Decoding: the benchmark harness is greedy (temperature 0). The sampling defaults in the Hugging Face `generation_config.json` and in the Ollama tag are not greedy and have not been measured.
- Aider: the patch markers differ from Aider's edit format, and Aider has not been tested.
- Vision: the Hugging Face repository ships the base model's vision projector as a separate file. The Modelfile leaves it out, but the published Ollama manifest lists it. Training was text-only, and image input has not been evaluated.

## Repository

| Path | Contents |
|---|---|
| `scripts/hf_publish.py` | Syncs the Hugging Face repo with this one: file allowlist, adapter in `lora/`, one commit that only adds and updates files, never deletes on the Hub; dry-run by default, `--publish` needs `HF_TOKEN` |
| `lora/` | Where the LoRA adapter goes (`adapter_config.json`, `adapter_model.safetensors`); the weights are not in git |
| `notebooks/Simplicio_27B_Training_Colab.ipynb` | Training run that produced the adapter |
| `notebooks/Simplicio_27B_Merge_Colab.ipynb` | Merges the adapter into 16-bit weights and exports the GGUF Q4_K_M |
| `notebooks/Simplicio_27B_Serve_Colab.ipynb`, `deploy/` | vLLM serve script, Colab launcher, completion gate, context length |
| `gateway/` | Gateway code kept in the repo: `set_upstream.php`, `upstream_lease.php` (heartbeat lease: HTTP 503 when the Colab stops renewing), `status.php` (`GET /v1/status`), and `build_catalog.py`, which writes `models.json` (`GET /v1/models`) |
| `Modelfile` | The Ollama tag `wesleysimplicio/simplicio-27b` |
| `data/unseen_eval_120.json` | The 120 rows of the held-out set (40 unique tasks) |
| `benchmarks/harness/` | Benchmark harness: 40 SEARCH/REPLACE tasks with hidden tests |
| `tests/` | Tests for the serving code: `python3 -m unittest discover -s tests` |

## Development

`make check` needs `python3`, `ruff`, `shellcheck`, `php` and `make` on the `PATH`.

```bash
make check                           # unittest, ruff F/E9, shellcheck, php -l, vLLM args
git config core.hooksPath .githooks  # once per clone: run make check before every push
```

The last step parses the exact `vllm serve` argv from `deploy/serve_vllm.sh` with vLLM's own CLI parser, so it needs `vllm==0.31.0` importable. Without vLLM installed it fails with exit code 1 on purpose: a flag that vLLM does not accept must not pass the gate unnoticed. On a machine without vLLM, export `SIMPLICIO_SKIP_VLLM_ARGS=1` to skip only that check (it prints `SKIP: vllm not importable; argv extracted, not validated`); the other four steps still run. There is no CI: all checks run locally. Commit subjects are one plain sentence, without a `type:` prefix. Colab notebooks live in `notebooks/`; `tests/test_layout.py` fails on a notebook at the repository root or on a `blob/main` link to a missing file.

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
