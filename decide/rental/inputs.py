"""Checks on what the day downloads before anything is trained on it.

* the start checkpoint is the one the plan counted on: the parameter count and the bytes of ``model.safetensors.index.json`` are
  the numbers of ``pplx-decider-v1.2`` (``hardware.PARAMS`` / ``WEIGHT_BYTES``), every shard it lists is there, and so are the
  readout head and the decision config;
* a dataset folder is read through its ``manifest.json``: every part is checked against its sha256 (``decide.data.schema``).
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from decide.data import schema as S

from . import hardware as HW

EXPECTED_START: Tuple[int, int] = (HW.PARAMS, HW.WEIGHT_BYTES)
INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"


class InputsError(RuntimeError):
    """An input of the day is not what the plan counted on, with what was found."""


def _single_file_size(path: Path) -> Tuple[int, int]:
    """``(parameters, tensor bytes)`` of one safetensors file, from its header."""
    with open(path, "rb") as handle:
        (length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(length).decode("utf-8"))
    header.pop("__metadata__", None)
    parameters = size = 0
    for entry in header.values():
        count = 1
        for dimension in entry["shape"]:
            count *= dimension
        begin, end = entry["data_offsets"]
        parameters += count
        size += end - begin
    return parameters, size


def check_start_checkpoint(folder: "str | os.PathLike[str]", *, expected: Optional[Tuple[int, Optional[int]]] = EXPECTED_START) -> Dict[str, Any]:
    root = Path(folder)
    index = root / INDEX
    if index.is_file():
        data = json.loads(index.read_text(encoding="utf-8"))
        parameters, size = data["metadata"]["total_parameters"], data["metadata"]["total_size"]
        shards = sorted(set(data["weight_map"].values()))
        for shard in shards:
            if not (root / shard).is_file():
                raise InputsError(f"{root}: the index lists {shard} but the file is not there (an interrupted download?)")
    elif (root / SINGLE).is_file():
        parameters, size = _single_file_size(root / SINGLE)
        shards = [SINGLE]
    else:
        raise InputsError(f"{root}: neither {INDEX} nor {SINGLE} is there, so this is not a joint checkpoint")
    for name in ("readout.safetensors", "decision_config.json"):
        if not (root / name).is_file():
            raise InputsError(f"{root}: {name} is missing; pplx-decider-v1.2 has it, so this is not the checkpoint the day starts from")
    if expected is not None:
        want_parameters, want_size = expected
        if parameters != want_parameters or (want_size is not None and size != want_size):
            raise InputsError(f"{root}: {parameters} parameters and {size} bytes, the plan counted on {want_parameters} and {want_size}; "
                              "this is not pplx-decider-v1.2 (or the download is incomplete)")
    return {"total_parameters": parameters, "total_size": size, "shards": len(shards)}


def verify_dataset(path: "str | os.PathLike[str]", *, name: str, manifest_required: bool) -> Dict[str, Any]:
    root = Path(path)
    if not root.is_dir():
        raise InputsError(f"{name}: {root} does not exist; was it uploaded to the data repo under that name?")
    files = S.corpus_files(root, require_manifest=manifest_required)          # raises ManifestError for a part that changed or is missing
    facts: Dict[str, Any] = {"name": name, "parts": len(files), "rows": None}
    manifest = root / S.MANIFEST_NAME
    if manifest.is_file():
        facts["rows"] = json.loads(manifest.read_text(encoding="utf-8")).get("rows")
    else:
        facts["warning"] = f"{name}: no {S.MANIFEST_NAME}, so its parts are not checked against a sha256 (they are read by name)"
    return facts
