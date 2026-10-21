"""Checks on what the day downloads before anything is trained on it.

* the start checkpoint is the one the plan counted on: the parameter count and the bytes of ``model.safetensors.index.json`` are
  the numbers of ``pplx-decider-v1.2`` (``hardware.PARAMS`` / ``WEIGHT_BYTES``), every shard it lists is there, and so are the
  readout head and the decision config;
* a dataset folder is read through its ``manifest.json``: every part is checked against its sha256 (``decide.data.schema``);
* the corpus is the output of ``decide.data.decontaminate`` (SCHEMA.md section 11), which is what ``build_targets`` demands: the day
  says so at the fetch, before the 52 GB checkpoint is downloaded and before any GPU hour, not in the middle of the targets phase;
* the proxy G4 will score on is the one that corpus was screened against (``check_proxy_screened``), and it has enough questions
  (``proxy_rows``).  A G4 on a proxy that leaked into the training data measures memory, not the checkpoint.
"""
from __future__ import annotations

import json
import os
import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from decide.data import schema as S
from decide.train import build_targets as BT
from decide.train.targets import TargetError

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


def _json_object(path: Path) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def proxy_rows(proxy: "str | os.PathLike[str]") -> Dict[str, Optional[int]]:
    """The rows the manifest of the proxy declares, the rows its parts hold, and how many of those have a gold answer (the ones G4 scores)."""
    declared = _json_object(Path(proxy) / S.MANIFEST_NAME).get("rows")
    read = with_gold = 0
    for row in S.read_corpus(proxy):
        read += 1
        _, gold = S.parse_row(row)
        with_gold += 1 if any(key is not None for key in gold.values()) else 0
    return {"declared": declared if isinstance(declared, int) else None, "read": read, "with_gold": with_gold}


def check_proxy_screened(proxy: "str | os.PathLike[str]", corpus: "str | os.PathLike[str]") -> Dict[str, Any]:
    """The proxy was built and checked against THIS clean corpus, as far as the files say; ``InputsError`` otherwise.

    * its ``corpus_leak_check`` says the second pass of ``proxy_private build --corpus <clean corpus>`` ran and found no leak;
    * it shares no source with the corpus (``sources`` of the two manifests);
    * ``decontamination.json`` of the corpus lists a reference with as many requests as the proxy has rows, which is what it has when
      ``decontaminate`` was given the proxy.  (The reference digest it also records is over the proxy of the FIRST pass, before the
      second pass rewrote ``manifest.json``, so it cannot be compared with the folder that comes to the day; the count can.)

    What this cannot know: that the reference with that many requests WAS this proxy, and that the corpus was also screened against the
    sample of the GT gate (the public suite given to ``decontaminate`` covers it, if it was given).  Both stay with the RUNBOOK."""
    proxy_root, corpus_root = Path(proxy), Path(corpus)
    proxy_manifest = _json_object(proxy_root / S.MANIFEST_NAME)
    corpus_manifest = _json_object(corpus_root / S.MANIFEST_NAME)
    report = _json_object(corpus_root / "decontamination.json")
    rows = proxy_manifest.get("rows")
    problems = []
    leak = proxy_manifest.get("corpus_leak_check")
    if not (isinstance(leak, dict) and leak.get("checked") is True and leak.get("leaks") == 0):
        problems.append(f"its manifest does not say that the second pass found no corpus request in it (corpus_leak_check is {leak!r}): "
                        "build the proxy again with --corpus <the clean corpus> (RUNBOOK, session 1, step 6)")
    shared = sorted(set(corpus_manifest.get("sources") or {}) & set(proxy_manifest.get("sources") or {}))
    if shared:
        problems.append(f"{', '.join(shared)} are sources of both the corpus and the proxy")
    references = [ref for ref in report.get("references") or [] if isinstance(ref, dict)]
    if not isinstance(rows, int) or not any(ref.get("requests") == rows for ref in references):
        problems.append(f"decontamination.json of the corpus lists no reference with {rows} requests, the size of this proxy "
                        f"(it lists {[ref.get('requests') for ref in references]}): decontaminate was run without --reference <the proxy> (RUNBOOK, session 1, step 5)")
    if problems:
        raise InputsError("proxy: " + "; ".join(problems))
    return {"rows": rows, "references": [ref.get("path") for ref in references]}


def check_corpus_screened(path: "str | os.PathLike[str]", *, name: str = "corpus") -> Dict[str, Any]:
    """The provenance of a corpus folder that came out of ``decontaminate`` (``tool``, ``manifest_sha256``, ``decontamination_sha256``).

    Raises ``InputsError`` for anything else: a raw ``build_corpus`` folder, a folder sealed by hand, a parquet file.  It is the check
    ``build_targets`` makes (the same function), moved to the door of the day."""
    try:
        facts = BT.corpus_provenance(path, allow_undecontaminated=False)
    except TargetError as exc:
        raise InputsError(f"{name}: {exc}") from None
    return dict(facts)
