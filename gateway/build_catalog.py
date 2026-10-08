#!/usr/bin/env python3
"""Gera gateway/models.json (GET /v1/models) de deploy/hf_instruct.json, deploy/context.env e gateway/pricing.json."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "deploy"))

from serve_colab import load_context  # noqa: E402

OUT = ROOT / "gateway" / "models.json"
PRICING = ROOT / "gateway" / "pricing.json"
CREATED = 1791136800
IDS = ("simpleti/simplicio-27b", "simplicio-27b")
DESCRIPTION = (
    "Research release. LoRA fine-tune of Qwen3.8-27B that answers code-change requests in five tagged "
    "phases and edits code with SEARCH/REPLACE blocks instead of rewriting whole files. On the project's "
    "internal check, 56 of 120 runs (46.7%) passed a text-matching check: 40 unique short Python tasks, each "
    "run 3 times, one attempt at temperature 0. Functional correctness of the patches was not measured. It "
    "has not been evaluated on public benchmarks or compared with the base model under the same protocol. "
    "Details and limits: https://huggingface.co/wesleysimplicio/Simplicio-27B"
)


def tokens(value: int) -> dict:
    return {"value": value, "unit": "token"}


def cost(kind: str, usd: str) -> dict:
    return {"type": kind, "unit": "token", "cost_usd": usd}


def render() -> str:
    instruct = json.loads((ROOT / "deploy" / "hf_instruct.json").read_text(encoding="utf-8"))
    if sorted(IDS) != sorted(instruct["served_model_names"]):
        raise SystemExit("IDS difere de served_model_names em deploy/hf_instruct.json")
    ctx = load_context()
    price = json.loads(PRICING.read_text(encoding="utf-8"))
    entry = {
        "name": "SimpleTI: Simplicio 27B",
        "hugging_face_id": "wesleysimplicio/Simplicio-27B",
        "created": CREATED,
        "description": DESCRIPTION,
        "context_length": ctx["MAX_MODEL_LEN"],
        "architecture": {
            "modality": "text->text",
            "tokenizer": "Qwen",
            "instruct_type": instruct["instruct_type"],
        },
    }
    inp = {"type": "text", "supported_inputs": {"max_context_length": tokens(ctx["MAX_MODEL_LEN"])}}
    out = {"type": "text", "supported_outputs": {"max_output_tokens": tokens(ctx["OUTPUT_BUDGET"])}}
    if price:
        entry["pricing"] = {"prompt": price["prompt"], "completion": price["completion"], "request": "0", "image": "0"}
        inp["pricing"] = [cost("prompt", price["prompt"]), cost("cached_prompt", price["cached_prompt"])]
        out["pricing"] = [cost("completion", price["completion"])]
    entry["input_modalities"] = [inp]
    entry["output_modalities"] = [out]
    data = [{"id": model_id, **entry} for model_id in IDS]
    return json.dumps({"data": data}, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str]) -> int:
    text = render()
    if argv == ["--check"]:
        if OUT.is_file() and OUT.read_text(encoding="utf-8") == text:
            print("gateway/models.json ok")
            return 0
        print("gateway/models.json desatualizado: rode python3 gateway/build_catalog.py", file=sys.stderr)
        return 1
    OUT.write_text(text, encoding="utf-8")
    print("escrito gateway/models.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
