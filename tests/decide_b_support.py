"""Shared helpers for the decide/data, decide/train and decide/rental tests (Squad B).

Not a test module (unittest discovery only loads test*.py).  The standard library is enough for most tests;
pyarrow, numpy, torch and transformers are only touched by the helpers that say so, and the tests that need them
skip cleanly when they are missing.
"""
from __future__ import annotations

import gzip
import importlib.util
import json
import random
import sys
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def have(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


HAVE_PYARROW = have("pyarrow")
HAVE_YAML = have("yaml")
HAVE_NUMPY = have("numpy")
HAVE_TORCH = have("torch")
HAVE_TRANSFORMERS = HAVE_TORCH and have("transformers")
HAVE_PEFT = HAVE_TORCH and have("peft")

needs_pyarrow = unittest.skipUnless(HAVE_PYARROW, "pyarrow is not installed")
needs_yaml = unittest.skipUnless(HAVE_YAML, "PyYAML is not installed")
needs_numpy = unittest.skipUnless(HAVE_NUMPY, "numpy is not installed")
needs_torch = unittest.skipUnless(HAVE_TORCH, "torch is not installed")
needs_transformers = unittest.skipUnless(HAVE_TRANSFORMERS, "torch + transformers are not installed")
needs_peft = unittest.skipUnless(HAVE_PEFT, "peft is not installed")

_VOCAB = (
    "river stone market lantern copper violin harbor winter orchid signal paper engine garden bridge "
    "falcon thunder marble canvas pepper silver window carpet meadow rocket saddle ladder hammer whisper "
    "cabinet compass glacier harvest isolate journey kettle lizard mirror nugget oyster pillow quartz "
    "ribbon sketch timber umbrella velvet walnut yonder zephyr anchor basalt cinder dagger ember fossil"
).split()


def lorem(seed: int, count: int) -> str:
    """Deterministic pseudo-text of `count` words; different seeds share (almost) no 13-grams."""
    rng = random.Random(seed)
    return " ".join(rng.choice(_VOCAB) + (str(rng.randrange(1000)) if rng.random() < 0.5 else "") for _ in range(count))


def choice_request(
    instructions: str = "Which city is the capital of France?",
    options: Sequence[str] = ("Paris", "Rome", "Berlin", "Madrid"),
    state: Any = None,
    keys: Optional[Sequence[str]] = None,
    qid: str = "q1",
) -> Dict[str, Any]:
    keys = list(keys) if keys is not None else [chr(65 + i) for i in range(len(options))]
    return {
        "state": {} if state is None else state,
        "questions": {qid: {"type": "choice", "instructions": instructions, "criteria": dict(zip(keys, options))}},
    }


def noul_request(instructions: str = "Is the sky blue?", state: Any = None) -> Dict[str, Any]:
    return {"state": {} if state is None else state,
            "questions": {"q1": {"type": "noul", "instructions": instructions, "criteria": {"true": "yes", "false": "no"}}}}


def score_request(levels: Sequence[str] = ("bad", "ok", "good"), state: Any = "a movie review") -> Dict[str, Any]:
    return {"state": state, "questions": {"q1": {"type": "score", "instructions": "Rate it.", "criteria": list(levels)}}}


def multi_request() -> Dict[str, Any]:
    req = choice_request()
    req["questions"]["q2"] = {"type": "noul", "instructions": "Is it in Europe?"}
    req["questions"]["q3"] = {"type": "score", "instructions": "How big?", "criteria": ["small", "medium", "large"]}
    return req


def kit_row(row_id: str, request: Dict[str, Any], expected: Optional[Dict[str, str]] = None, dataset: str = "Toy") -> Dict[str, Any]:
    """A row in the decision-index suite format (state, questions, expected, _evaluation)."""
    return {
        "id": row_id, "family": dataset, "split": "test", "state": request["state"], "questions": request["questions"],
        "expected": expected or {}, "metadata": {}, "scoring": {}, "_evaluation": {"dataset": dataset, "run_id": row_id},
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as fh:  # type: ignore[call-overload]
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    return path


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8") as fh:  # type: ignore[call-overload]
        return [json.loads(line) for line in fh if line.strip()]


def board_index(scores: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    """A minimal board index.json: engine -> {v, same_skills, new_domains, pub}."""
    return {"models": [{"engine": engine, "name": engine, "meta": {"kind": "full fine-tune"}, "v03": dict(v, engine=engine)}
                       for engine, v in scores.items()]}


# --------------------------------------------------------------------------- tokenizer stand-in (decide/train)

class FakeTokenizer:
    """Just enough of a Hugging Face tokenizer for decide/train tests, with no files and no network.

    ChatML-style template like the Qwen one (`<think>` block closed because thinking is disabled), special tokens
    kept whole, every answer code a single token, every other word a hashed id.  Not the real vocabulary: the real
    one is exercised by `DECIDE_PPLX_REFERENCE` (see test_contract_69.ReferenceParity).
    """

    SPECIALS = ("<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|fim_suffix|>")
    WORD_BUCKETS = 300

    def __init__(self, codes: Optional[Sequence[str]] = None) -> None:
        from decide.train.codes import ANSWER_CODES  # imported lazily: the module under test may not exist yet

        self.codes = list(codes or ANSWER_CODES)
        self.special_ids = {tok: i for i, tok in enumerate(self.SPECIALS)}
        self.code_ids = {code: len(self.SPECIALS) + i for i, code in enumerate(self.codes)}
        self.newline_id = len(self.SPECIALS) + len(self.codes)
        self.first_word_id = self.newline_id + 1
        self.vocab_size = self.first_word_id + self.WORD_BUCKETS
        self.pad_token, self.pad_token_id = "<|endoftext|>", 0
        self.padding_side = "left"
        self.all_special_tokens = ["<|im_end|>", "<|endoftext|>", "<|im_start|>"]
        self._split = __import__("re").compile(r"(<\|[a-z_]+\|>)|(\n)|([^\W_]+|[^\w\s])|(\s+)")

    # -- chat template
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, enable_thinking=True, **_):
        assert tokenize is False
        out = []
        for message in messages:
            content = message["content"]
            if not isinstance(content, str):
                content = "".join(part["text"] for part in content if part.get("type") == "text")
            out.append(f"<|im_start|>{message['role']}\n{content}<|im_end|>\n")
        if add_generation_prompt:
            out.append("<|im_start|>assistant\n" + ("" if enable_thinking else "<think>\n\n</think>\n\n"))
        return "".join(out)

    # -- tokenization
    def _id(self, piece: str) -> int:
        if piece in self.special_ids:
            return self.special_ids[piece]
        if piece in self.code_ids:
            return self.code_ids[piece]
        if piece == "\n":
            return self.newline_id
        import hashlib

        return self.first_word_id + int.from_bytes(hashlib.blake2b(piece.encode(), digest_size=4).digest(), "big") % self.WORD_BUCKETS

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        assert add_special_tokens is False
        ids = []
        for match in self._split.finditer(text):
            piece = match.group(0)
            if piece.strip() == "" and piece != "\n":
                continue  # spaces carry no token in this stand-in
            ids.append(self._id(piece))
        return ids

    def __call__(self, text, add_special_tokens: bool = False, **_):
        if isinstance(text, str):
            return {"input_ids": self.encode(text, add_special_tokens)}
        return {"input_ids": [self.encode(t, add_special_tokens) for t in text]}

    def convert_ids_to_tokens(self, ids):
        names = {v: k for k, v in {**self.special_ids, **self.code_ids}.items()}
        single = isinstance(ids, int)
        out = [names.get(i, f"<{i}>") for i in ([ids] if single else ids)]
        return out[0] if single else out


# --------------------------------------------------------------------------- tiny model + targets for the training flow tests

TINY_VOCAB = 600


def make_tiny_checkpoint(directory: Path, tokenizer: "FakeTokenizer", seed: int = 0, temperature: float = 1.0) -> Path:
    """A random 4-layer Qwen3.5 backbone with the files of a v1.2 joint checkpoint (needs torch + transformers).

    Same layout as perplexity-ai/pplx-decider-v1.2 (backbone + ``readout.safetensors`` + ``decision_config.json`` +
    ``joint_config.json``) and a tokenizer-sized vocabulary, so the real code paths run on a CPU in a fraction of a second.
    """
    import torch
    from safetensors.torch import save_file
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    from decide.train.codes import ANSWER_CODES
    from decide.train.protocol import PROTOCOL

    text = dict(
        vocab_size=TINY_VOCAB, hidden_size=64, intermediate_size=96, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_conv_kernel_dim=4, linear_key_head_dim=16, linear_num_key_heads=2, linear_num_value_heads=4, linear_value_head_dim=16,
        full_attention_interval=4, layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        max_position_embeddings=2048, rms_norm_eps=1e-6, partial_rotary_factor=0.25, attn_output_gate=True,
        rope_parameters={"mrope_interleaved": True, "mrope_section": [1, 1, 2], "partial_rotary_factor": 0.25, "rope_theta": 10000.0, "rope_type": "default"},
        tie_word_embeddings=False, pad_token_id=0, bos_token_id=1, eos_token_id=1,
    )
    vision = dict(depth=1, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64, patch_size=16, spatial_merge_size=2,
                  temporal_patch_size=2, num_position_embeddings=16, in_channels=3)
    config = Qwen3_5Config(text_config=text, vision_config=vision, image_token_id=590, video_token_id=591, vision_start_token_id=592,
                           vision_end_token_id=593, tie_word_embeddings=False)
    torch.manual_seed(seed)
    model = Qwen3_5Model(config).to(torch.float32)
    directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(directory))
    readout = torch.nn.Linear(64, 255, bias=False)
    save_file({"weight": readout.weight.detach().contiguous()}, str(directory / "readout.safetensors"))
    codes = list(ANSWER_CODES)
    (directory / "decision_config.json").write_text(json.dumps({
        "format_version": 2, "base_model": "tiny/qwen3_5", "revision": "0" * 40, "attention_mode": "noncausal_full_attention", "pooling": "last",
        "temperature": temperature, "max_pixels": 1, "codes": codes, "token_ids": [tokenizer.code_ids[c] for c in codes]}, indent=2), encoding="utf-8")
    (directory / "joint_config.json").write_text(json.dumps(PROTOCOL, indent=2), encoding="utf-8")
    (directory / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    return directory


def make_targets_dataset(root: Path, *, train: int = 8, dev: int = 6, traces_for: Sequence[int] = ()) -> Path:
    """A small corpus with one teacher and gold labels, built into a targets dataset by ``decide.train.build_targets``.

    Every request is a 3-option choice question whose right option is the one named in the state, so a model can learn it.
    Returns the targets folder.  ``traces_for`` lists the indices of the training examples that also get a thinker trace.
    """
    from decide.data import schema as S
    from decide.data import traces as TR
    from decide.train import build_targets as BT

    rows, teacher_rows = [], []
    names = ["river", "stone", "market", "lantern", "copper", "violin", "harbor", "winter", "orchid", "signal", "paper", "engine", "garden", "bridge"]
    for i in range(train + dev):
        right = i % 3
        options = [names[(i + k) % len(names)] for k in range(3)]
        request = choice_request(f"Which one is the odd one number {i}?", options, state=f"The answer is {options[right]} for case {i}.", keys=["x", "y", "z"])
        example_id = f"toy:{i}"
        rows.append(S.corpus_row(example_id=example_id, source="toy", license="mit", split="train" if i < train else "dev", lang="en", request=request,
                                 gold={"q1": ["x", "y", "z"][right]}))
        probs = [0.1, 0.1, 0.1]
        probs[right] = 0.8
        teacher_rows.append({"example_id": example_id, "question_id": "q1", "teacher": "toy-teacher", "codes": ["A", "B", "C"], "probs": probs, "temperature": 1.0})
    S.write_corpus_part(root / "corpus" / "toy-0000.parquet", rows)
    S.seal_directory(root / "corpus")
    S.write_teacher_part(root / "labels" / "toy-teacher" / "part-0000.parquet", teacher_rows)
    argv = ["--corpus", str(root / "corpus"), "--labels", str(root / "labels"), "--out", str(root / "targets")]
    if traces_for:
        TR.write_trace_part(root / "traces" / "toy-thinker" / "part-0000.parquet", [
            TR.trace_row(example_id=f"toy:{i}", thinker="toy-thinker", reasoning=f"The state names option {i % 3}, so the answer is {['x', 'y', 'z'][i % 3]}.",
                         tokens=12, finish_reason="stop") for i in traces_for])
        argv += ["--traces", str(root / "traces")]
    import contextlib
    import io

    with contextlib.redirect_stdout(io.StringIO()):
        BT.main(argv)
    return root / "targets"
