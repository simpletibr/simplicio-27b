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
- **Files on Hugging Face:** the LoRA adapter (0.64 GB), the merged BF16 checkpoint (18 shards, 55.6 GB), and a GGUF Q4_K_M (16.8 GB) with the base model's vision projector (0.93 GB). Image input is inherited from the base model and has not been evaluated.
- **Status:** research release. On the project's own held-out set (40 short Python tasks, each run 3 times), 56 of 120 runs passed a text-matching check. It has not been scored on public benchmarks.

## Results

### Held-out set (40 tasks, each run 3 times)

The full BF16 checkpoint ran on Google Colab G4 (NVIDIA RTX PRO 6000 Blackwell, 95 GB) against [`data/unseen_eval_120.json`](https://github.com/simpletibr/simplicio-27b/blob/b8567df/data/unseen_eval_120.json), whose tasks were not used in training. The file has 120 rows, but they are 40 unique Python tasks, each listed 3 times with only `id` and `difficulty` changed; the original code averages 2.8 lines (at most 6). Each row got one attempt at temperature 0, and generation stopped at `</deliver>`. Aggregate results: [`benchmarks/live_colab_g4_bf16_n120.json`](https://github.com/simpletibr/simplicio-27b/blob/main/benchmarks/live_colab_g4_bf16_n120.json). Per-row outputs were not saved in the repository.

The harness that produced these numbers is not in the repository either. Its output fields (`diff_matched`, `ast_valid`, `zero_ghost_apis`, `unit_test_passed`) match `apply_and_verify_patch` in [`benchmarks/prove_benchmark_120.py`](https://github.com/simpletibr/simplicio-27b/blob/b8567df/benchmarks/prove_benchmark_120.py#L113-L172), which runs each task's assertion with only the patched text in scope and never executes the patched code. 25 of the 40 assertions only check whether given text appears (or does not appear) in the patched file. The other 15 call a function or read a variable that only the patched code would define, so they cannot pass under that scorer: the edge-case category's 6 passes equal its 2 text-only tasks × 3 runs. Read the pass counts as text matching, not as unit tests.

| Metric | Result |
|---|---:|
| Assertion passes (text matching) | **56/120 runs (46.7%)** |
| SEARCH block found, patch applied | 104/120 (86.7%) |
| Patched file parses (AST) | 104/120 (86.7%) |
| Output tokens per run, mean / max | 69.9 / 103 |
| Wall time for all 120 runs | 438 s |

| Category | Runs (unique tasks) | Assertion passes | Patch applied |
|---|---:|---:|---:|
| Surgical diff and AST precision | 30 (10) | 17 | 20 |
| Edge-case correctness | 30 (10) | 6 | 27 |
| Nonexistent and deprecated API traps | 30 (10) | 27 | 30 |
| Adversarial and out-of-distribution | 30 (10) | 6 | 27 |

The Wilson 95% interval for 56/120 is 38.0%–55.6%, but the 120 runs are 40 tasks repeated, so they are not independent. Counted over 40 tasks, the same 46.7% has a 95% interval of 32.2%–61.7%. Only the 25 text-only tasks can pass, and the category totals put the number of unique tasks that passed at least once between 19 and 25. In the first category, 17 passes and 20 applied patches are not multiples of 3, so copies of the same task got different results at temperature 0.

An earlier version of this table also reported that no run called a nonexistent API. In that scorer the check only runs on the 30 trap runs (the only rows with `forbidden_symbols`) and counts a run as clean even when its patch was not applied, so the row was removed.

Not evaluated: whether outputs follow the five-phase format, the base model under this protocol (so the gain from fine-tuning is not measured), the Q4_K_M GGUF used by Ollama, the 4-bit base + LoRA path, and image input.

### Withdrawn numbers

Earlier versions of this card reported 96.5% accuracy, 116 of 120 tasks passed, 480 tokens per task, a "Top 12" leaderboard and per-token prices. None of these came from running the model on those tasks:

- 116/120 and its McNemar test come from [`benchmarks/prove_benchmark_120.py`](https://github.com/simpletibr/simplicio-27b/blob/b8567df/benchmarks/prove_benchmark_120.py), which simulates model outputs.
- 480 tokens per task comes from a 3-task smoke test on an A100 ([`benchmarks/empirical_a100_results.json`](https://github.com/simpletibr/simplicio-27b/blob/b8567df/benchmarks/empirical_a100_results.json)). All three runs are recorded with exactly 480 output tokens, the `max_new_tokens=480` cap of every `model.generate` call in [`Simplicio_27B_2026_Benchmarks_Colab.ipynb`](https://github.com/simpletibr/simplicio-27b/blob/b8567df/Simplicio_27B_2026_Benchmarks_Colab.ipynb). No saved output backs these runs: that notebook and the one linked in the file's `notebook_gist` field have no cell outputs, and neither contains the run `py_safe_dict_get`.
- 96.5% and the leaderboard rows are hard-coded in [`benchmarks/compare_top10_2026.py`](https://github.com/simpletibr/simplicio-27b/blob/b8567df/benchmarks/compare_top10_2026.py).
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

The `latest` tag holds the Q4_K_M GGUF and the base model's vision projector. This build has not been evaluated: every result on this card comes from the BF16 checkpoint, and image input has not been evaluated.

If Ollama is not installed, get it from [ollama.com/download](https://ollama.com/download).

### vLLM (OpenAI-compatible server with tool calls)

```bash
pip install vllm==0.31.0
git clone https://github.com/simpletibr/simplicio-27b
cd simplicio-27b
./deploy/serve_vllm.sh wesleysimplicio/Simplicio-27B 8000
```

This serves the merged BF16 checkpoint, the same weights that scored 56/120 above. The weights alone take 55.6 GB; the evaluation above ran on a 95 GB GPU. [`deploy/serve_vllm.sh`](https://github.com/simpletibr/simplicio-27b/blob/main/deploy/serve_vllm.sh) is the only serve command, and the Colab notebook runs it too. It sets:

- `--max-model-len 40960`, defined once in [`deploy/context.env`](https://github.com/simpletibr/simplicio-27b/blob/main/deploy/context.env): a measured 31,692-token OpenCode prompt plus 4,096 output tokens.
- No `--chat-template`: vLLM uses the model's own `chat_template.jinja`.
- `--default-chat-template-kwargs '{"enable_thinking": false}'`: thinking is off by default because the training data has no `<think>` blocks. A request can turn it on with `"chat_template_kwargs": {"enable_thinking": true}`.
- `--enable-auto-tool-choice --tool-call-parser qwen3_coder --reasoning-parser qwen3`: vLLM's built-in parsers for Qwen's native `<tool_call><function=…>` format.
- `--enable-force-include-usage`: every response carries `usage`.
- `--served-model-name simplicio-27b simpleti/simplicio-27b`: both ids reach the same fine-tuned weights.

`vllm serve` has no stop flag. `<|im_end|>` already ends generation through `generation_config.json`; to end at `</deliver>`, send `"stop": ["</deliver>"]` in the request.

[`Simplicio_27B_Serve_Colab.ipynb`](https://github.com/simpletibr/simplicio-27b/blob/main/Simplicio_27B_Serve_Colab.ipynb) runs the same script on a Colab G4 (RTX PRO 6000, 96 GB) behind the completion gate and a Cloudflare tunnel.

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
| LoRA | r 32, alpha 32, dropout 0. q, k, v, o projections on the 16 full-attention layers (3, 7, …, 63); gate, up, down projections on all 64 layers. The `linear_attn` projections of the 48 DeltaNet layers and the vision tower were not adapted (512 LoRA tensors in the published adapter) |
| Steps | 120 steps, batch size 1, gradient accumulation 8 |
| Optimizer | AdamW 8-bit, learning rate 2e-4, cosine schedule, 10 warmup steps, weight decay 0.01 |
| Sequence length | 4,096 |
| Loss | whole sequence, prompt included |
| Data | 101 examples in Portuguese: 1 written by hand and 100 generated from a short list of stack and task templates |
| Hardware | Google Colab A100 (40 GB) |

An earlier script, [`train_simplicio_27b.py`](https://github.com/simpletibr/simplicio-27b/blob/b8567df/train_simplicio_27b.py), was removed because its defaults do not reproduce the published adapter: 150 steps, the bottom 48 layers frozen, and the 10 phase tags added as special tokens. The published adapter used none of these, and its tokenizer has no added tokens. [`generate_dataset.py`](https://github.com/simpletibr/simplicio-27b/blob/main/generate_dataset.py) writes `data/simplicio_loop_50pts_train.jsonl` (80 examples) and `data/simplicio_loop_50pts_val.jsonl` (15 examples).

## Limitations

- On the held-out set, 56 of 120 runs (40 unique tasks) passed a text-matching check. Functional correctness of the patches was not measured.
- The training set is small (101 examples), templated and in Portuguese. Whether outputs follow the five-phase format was not measured.
- In the training examples, `<validate>` and `<deliver>` contain written-out results such as "2 passed" or "COMMIT_READY". The model writes these without running anything. Treat them as claims and run your own tests.
- It has not been compared with the base model under the same protocol, and it has not been run on public benchmarks.
- Only the BF16 checkpoint was evaluated. The Q4_K_M GGUF, the Ollama template and parameters, and the 4-bit base + LoRA path were not.
- Decoding: the evaluation used temperature 0 (greedy). The sampling defaults in the Hugging Face `generation_config.json` and in the Ollama tag are not greedy and were not evaluated.
- Aider: the patch markers differ from Aider's edit format, and Aider has not been tested.
- Vision: the GGUF ships the base model's vision projector, so image input is inherited from Qwen3.8-27B. Training was text-only, and image input has not been evaluated.

## Repository

| Path | Contents |
|---|---|
| `Simplicio_27B_Training_Colab.ipynb` | Training run that produced the adapter |
| `Simplicio_27B_Merge_Colab.ipynb` | Merges the adapter into 16-bit weights and exports the GGUF Q4_K_M |
| `Simplicio_27B_Serve_Colab.ipynb`, `deploy/` | vLLM serve script, Colab launcher, completion gate, context length, Ollama `Modelfile` |
| `data/unseen_eval_120.json` | The 120 held-out tasks |
| `benchmarks/live_colab_g4_bf16_n120.json` | The results above |
| `tests/` | Tests for the serving code: `python3 -m unittest discover -s tests` |

## Development

`make check` needs `python3`, `ruff`, `shellcheck`, `php` and `make` on the `PATH`.

```bash
make check                           # unittest, ruff F/E9, shellcheck, php -l, vLLM args
git config core.hooksPath .githooks  # once per clone: run make check before every push
```

The last step parses the exact `vllm serve` argv from `deploy/serve_vllm.sh` with vLLM's own CLI parser, so it needs `vllm==0.31.0` importable. Without vLLM installed it fails with exit code 1 on purpose: a flag that vLLM does not accept must not pass the gate unnoticed. On a machine without vLLM, export `SIMPLICIO_SKIP_VLLM_ARGS=1` to skip only that check (it prints `SKIP: vllm not importable; argv extracted, not validated`); the other four steps still run. There is no CI: all checks run locally. Commit subjects are one plain sentence, without a `type:` prefix.

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
