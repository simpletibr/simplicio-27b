"""Weight-space average ("model soup") of checkpoints of the same architecture, one tensor at a time.

Every tensor, and the readout, is the weighted mean of the inputs, computed in fp32 and stored in the dtype of the
inputs.  The output keeps the shard layout of the first input, copies its tokenizer and config files, and writes a
``decision_config.json`` that records the weights and the inputs.  The temperature is the mean of the inputs' unless
one is given: it is a placeholder, a soup must be re-calibrated (``python -m decide.train.temperature``).

Memory: one output shard (bf16, about 5 GB for the 27B) plus a few fp32 copies of the largest tensor.  Nothing is
written to the final path until every check passed and every shard was written; an interrupted run leaves only a
``<out>.souping`` folder, which the next run deletes.

    python -m decide.train.soup --inputs run1/final run2/final --out soup
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Top-level files that describe one specific input and must not be copied into the average.
NOT_COPIED = {"decision_config.json", "model.safetensors.index.json", "readout.safetensors", "release-manifest.json",
              "README.md", "LICENSE", "NOTICE"}


#: Keys of an input's ``decision_config.json`` that describe that one run (its calibration and training record); a soup drops them.
INHERITED_FROM_ONE_RUN = ("temperature_by_type", "temperature_fallback", "temperature_with_reasoning", "temperature_is_placeholder", "calibration", "training")


class SoupError(ValueError):
    """The inputs cannot be averaged (different tensors, shapes or dtypes, bad weights, ...)."""


def _safe_open():
    try:
        from safetensors import safe_open
    except ImportError as exc:  # pragma: no cover - message only
        raise SoupError("safetensors is required: install the versions in decide/train/requirements.txt") from exc
    return safe_open


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _layout(directory: Path) -> Dict[str, str]:
    """Tensor name -> shard file name, from the index or from the single ``model.safetensors``."""
    safe_open = _safe_open()
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise SoupError(f"{index}: no weight_map")
    else:
        single = directory / "model.safetensors"
        if not single.is_file():
            raise SoupError(f"{directory}: neither model.safetensors.index.json nor model.safetensors")
        with safe_open(str(single), "pt") as handle:
            weight_map = {name: single.name for name in handle.keys()}
    for name, shard in weight_map.items():
        if not isinstance(shard, str) or Path(shard).name != shard or not shard.endswith(".safetensors"):
            raise SoupError(f"{directory}: tensor {name!r} points to {shard!r}, which is not a local .safetensors file")
        if not (directory / shard).is_file():
            raise SoupError(f"{directory}: shard {shard} is missing")
    return dict(weight_map)


def _headers(directory: Path, layout: Dict[str, str]) -> Dict[str, Tuple[Tuple[int, ...], str]]:
    """Tensor name -> (shape, dtype) read from the safetensors headers (no tensor is loaded)."""
    safe_open = _safe_open()
    headers: Dict[str, Tuple[Tuple[int, ...], str]] = {}
    for shard in sorted(set(layout.values())):
        with safe_open(str(directory / shard), "pt") as handle:
            for name in handle.keys():
                piece = handle.get_slice(name)
                headers[name] = (tuple(piece.get_shape()), str(piece.get_dtype()))
    missing = sorted(set(layout) - set(headers))
    if missing:
        raise SoupError(f"{directory}: the index lists tensors that no shard holds, e.g. {missing[0]}")
    return {name: headers[name] for name in layout}


def _check_same(reference: Dict[str, Tuple[Tuple[int, ...], str]], other: Dict[str, Tuple[Tuple[int, ...], str]],
                first: Path, second: Path, what: str) -> None:
    only_first, only_second = sorted(set(reference) - set(other)), sorted(set(other) - set(reference))
    if only_first or only_second:
        raise SoupError(f"{what}: different tensors in {first.name} and {second.name} "
                        f"(only in the first: {only_first[:3]}; only in the second: {only_second[:3]})")
    for name in reference:
        (shape_a, dtype_a), (shape_b, dtype_b) = reference[name], other[name]
        if shape_a != shape_b:
            raise SoupError(f"{what}: {name} has shape {shape_a} in {first.name} and {shape_b} in {second.name}")
        if dtype_a != dtype_b:
            raise SoupError(f"{what}: {name} has dtype {dtype_a} in {first.name} and {dtype_b} in {second.name}")


def _normalized(weights: Optional[Sequence[float]], count: int) -> Tuple[List[float], List[float]]:
    """(raw weights, weights normalised to sum 1)."""
    raw = [1.0] * count if weights is None else [float(w) for w in weights]
    if len(raw) != count:
        raise SoupError(f"{len(raw)} weights for {count} inputs")
    if any(not math.isfinite(w) or w <= 0 for w in raw):
        raise SoupError("weights must be positive and finite")
    total = sum(raw)
    return raw, [w / total for w in raw]


def _temperature_of(directory: Path) -> Tuple[Dict[str, Any], float]:
    path = directory / "decision_config.json"
    if not path.is_file():
        raise SoupError(f"{directory}: decision_config.json is missing")
    config = json.loads(path.read_text(encoding="utf-8"))
    value = config.get("temperature")
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise SoupError(f"{path}: temperature must be a positive finite number")
    return config, float(value)


def average_checkpoints(inputs: Sequence[os.PathLike], out: os.PathLike, weights: Optional[Sequence[float]] = None,
                        temperature: Optional[float] = None) -> Dict[str, Any]:
    """Average the checkpoints in ``inputs`` into the new directory ``out``; returns a report."""
    import torch
    from safetensors.torch import save_file

    safe_open = _safe_open()
    directories = [Path(p) for p in inputs]
    if len(directories) < 2:
        raise SoupError("a soup needs at least two checkpoints")
    destination = Path(out)
    if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
        raise FileExistsError(f"Refusing to overwrite: {destination}")
    raw, normalized = _normalized(weights, len(directories))
    total = sum(raw)

    # ---- checks on headers only, before anything is written
    layouts = [_layout(d) for d in directories]
    headers = [_headers(d, layout) for d, layout in zip(directories, layouts)]
    for other, other_headers in zip(directories[1:], headers[1:]):
        _check_same(headers[0], other_headers, directories[0], other, "backbone")
    readouts = []
    for directory in directories:
        path = directory / "readout.safetensors"
        if not path.is_file():
            raise SoupError(f"{directory}: readout.safetensors is missing")
        with safe_open(str(path), "pt") as handle:
            readouts.append({name: (tuple(handle.get_slice(name).get_shape()), str(handle.get_slice(name).get_dtype()))
                             for name in handle.keys()})
    for other, other_headers in zip(directories[1:], readouts[1:]):
        _check_same(readouts[0], other_headers, directories[0], other, "readout")
    configs_and_temperatures = [_temperature_of(d) for d in directories]
    mean_temperature = sum(t for _, t in configs_and_temperatures) / len(directories)
    if temperature is not None and (not math.isfinite(temperature) or temperature <= 0):
        raise SoupError("temperature must be positive and finite")

    # ---- write into a sibling folder, move into place at the end
    work = destination.with_name(destination.name + ".souping")
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    handles: Dict[Tuple[int, str], Any] = {}

    def fetch(index: int, shard: str, name: str):
        key = (index, shard)
        if key not in handles:
            handles[key] = safe_open(str(directories[index] / shard), "pt").__enter__()
        return handles[key].get_tensor(name)

    def average(name: str, locate) -> "torch.Tensor":
        first = locate(0)
        if not first.is_floating_point():
            for index in range(1, len(directories)):
                if not torch.equal(first, locate(index)):
                    raise SoupError(f"{name}: a non-floating tensor differs between the inputs and cannot be averaged")
            return first
        accumulator = first.to(torch.float32) * raw[0]
        for index in range(1, len(directories)):
            accumulator.add_(locate(index).to(torch.float32), alpha=raw[index])
        return (accumulator / total).to(first.dtype)

    tensors = 0
    try:
        reference = layouts[0]
        by_shard: Dict[str, List[str]] = {}
        for name, shard in reference.items():
            by_shard.setdefault(shard, []).append(name)
        for shard, names in by_shard.items():
            produced = {}
            for name in names:
                produced[name] = average(name, lambda i, n=name: fetch(i, layouts[i][n], n)).contiguous()
                tensors += 1
            save_file(produced, str(work / shard), metadata={"format": "pt"})
            del produced
        index_path = directories[0] / "model.safetensors.index.json"
        if index_path.is_file():
            shutil.copyfile(index_path, work / index_path.name)
        readout_tensors = {}
        for name in readouts[0]:
            readout_tensors[name] = average(
                name, lambda i, n=name: _read_readout(safe_open, directories[i] / "readout.safetensors", n)).contiguous()
            tensors += 1
        save_file(readout_tensors, str(work / "readout.safetensors"), metadata={"format": "pt"})
    finally:
        for handle in handles.values():
            handle.__exit__(None, None, None)

    for item in sorted(directories[0].iterdir()):
        if item.is_file() and item.suffix != ".safetensors" and item.name not in NOT_COPIED:
            shutil.copyfile(item, work / item.name)
    config = dict(configs_and_temperatures[0][0])
    for key in INHERITED_FROM_ONE_RUN:
        config.pop(key, None)       # the averaged weights are nobody's run: the first input's fit and record would lie
    config["temperature"] = float(temperature) if temperature is not None else mean_temperature
    config["soup"] = {
        "method": "weighted mean in fp32, stored in the dtype of the inputs",
        "weights": normalized,
        "inputs": [{"name": d.name, "temperature": t,
                    "decision_config_sha256": _sha256(d / "decision_config.json"),
                    "readout_sha256": _sha256(d / "readout.safetensors")}
                   for d, (_, t) in zip(directories, configs_and_temperatures)],
        "temperature_is_placeholder": temperature is None,
    }
    (work / "decision_config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    if destination.exists():
        destination.rmdir()
    os.replace(work, destination)
    return {"out": str(destination), "inputs": [str(d) for d in directories], "weights": normalized, "tensors": tensors,
            "shards": len(by_shard), "temperature": config["temperature"]}


def _read_readout(safe_open, path: Path, name: str):
    with safe_open(str(path), "pt") as handle:
        return handle.get_tensor(name)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decide.train.soup", description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", nargs="+", required=True, help="checkpoint folders to average (two or more)")
    parser.add_argument("--out", required=True, help="new folder for the soup (must not exist or be empty)")
    parser.add_argument("--weights", nargs="+", type=float, help="one positive weight per input (default: all equal)")
    parser.add_argument("--temperature", type=float,
                        help="temperature to store (default: mean of the inputs'; re-calibrate afterwards)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    report = average_checkpoints(args.inputs, args.out, weights=args.weights, temperature=args.temperature)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
