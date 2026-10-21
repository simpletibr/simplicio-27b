"""The adapter between the rented day and the kit (``decide/sd``, Squad A): run a model on a requests file, resumably.

The day only needs one thing from the kit, "answer these requests with this model", and it asks through
``decide.sd.api`` (``load_engine`` and ``run_requests``).  Two kinds of model:

* a Hub model (the baseline of G4: ``perplexity-ai/pplx-decider-v1.2`` at its pinned commit), loaded as the kit loads any model;
* a checkpoint folder on this machine (the new decider).  The kit loads weights through its own ``HubRepo`` over HTTP Range, so
  the folder is served to it by :func:`local_repo`: the kit's ``HubRepo`` over a transport that reads files instead of the
  network.  The temperatures come from the checkpoint's own ``decision_config.json`` (the fit ``decide.train.temperature`` wrote),
  passed to the engine explicitly; a checkpoint whose temperature is still a placeholder is refused.

UNVERIFIED: written against ``decide.sd.api`` / ``decide.sd.remote`` as read in Squad A's branch (``decide/65-kit``), tested with a
stand-in for the api and, when the kit is importable, with the kit's real ``HubRepo`` reading a tiny checkpoint.  No model of the
real size has been loaded through it.
"""
from __future__ import annotations

import gc
import importlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

LOCAL_REPO = "local/checkpoint"
_ENDPOINT = "local://checkpoint"
_REVISION = re.compile(r"[0-9a-f]{40}")


class KitError(RuntimeError):
    """The kit cannot run the model, with the reason."""


@dataclass(frozen=True)
class ModelRef:
    """A model to run: a checkpoint folder on this machine, or a Hub repo at a pinned commit."""

    name: str
    path: Optional[Path] = None
    repo: Optional[str] = None
    revision: Optional[str] = None

    def __post_init__(self) -> None:
        if (self.path is None) == (self.repo is None):
            raise ValueError("a model is either a checkpoint folder (path) or a Hub repo (repo + revision), not both and not neither")
        if self.repo is not None and not (isinstance(self.revision, str) and _REVISION.fullmatch(self.revision)):
            raise ValueError(f"{self.repo}: a Hub model needs a revision pinned to a 40-character commit, got {self.revision!r}")


def local_repo(checkpoint: "str | os.PathLike[str]") -> Any:
    """The kit's ``HubRepo`` reading ``checkpoint`` (a folder) through a transport that never touches the network."""
    remote = _import("decide.sd.remote")
    root = Path(checkpoint).resolve()
    marker = "/resolve/main/"

    class FolderTransport:
        """The two calls of the kit's transports (``fetch`` a byte range, ``get`` a whole file) over files in one folder."""

        def _file(self, url: str) -> Path:
            relative = url.split(marker, 1)[1] if marker in url else ""
            candidate = (root / relative).resolve()
            try:
                candidate.relative_to(root)
            except ValueError:
                raise remote.RemoteError(f"404: {relative} is outside the checkpoint folder") from None
            if not candidate.is_file():
                raise remote.RemoteError(f"404: {relative} is not in the checkpoint folder")
            return candidate

        def fetch(self, url: str, start: int, end: int) -> bytearray:
            with open(self._file(url), "rb") as handle:
                handle.seek(start)
                return bytearray(handle.read(end - start + 1))

        def get(self, url: str) -> bytes:
            return self._file(url).read_bytes()

    return remote.HubRepo(LOCAL_REPO, None, endpoint=_ENDPOINT, transport=FolderTransport())


def _import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except ImportError as exc:
        raise KitError(f"the kit (decide/sd, Squad A) cannot be imported: {exc}. It is not in this checkout yet, or its packages are not installed "
                       "(see notebooks/decide/requirements.txt)") from exc


def _temperatures(path: Path) -> Dict[str, float]:
    config_path = path / "decision_config.json"
    if not config_path.is_file():
        raise KitError(f"{config_path} does not exist: {path} is not a joint checkpoint")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("temperature_is_placeholder"):
        raise KitError(f"{path}: the temperature in decision_config.json is a placeholder (a soup that was not re-calibrated); "
                       "run python -m decide.train.temperature --write on it first")
    temperatures = {str(kind): float(value) for kind, value in (config.get("temperature_by_type") or {}).items()}
    temperatures["default"] = float(config["temperature"])
    return temperatures


class SdKit:
    def __init__(self, *, api: Any = None, module: str = "decide.sd.api", quant: str = "bf16", device: Any = "auto", tta_rotations: int = 1) -> None:
        self._api, self._module = api, module
        self.quant, self.device, self.tta_rotations = quant, device, tta_rotations

    def _resolve(self) -> Any:
        return self._api if self._api is not None else _import(self._module)

    def run(self, model: ModelRef, requests: "str | os.PathLike[str]", out: "str | os.PathLike[str]", *, budget: Any = None) -> Dict[str, Any]:
        """Answer the requests that have no result yet (the kit resumes); returns the kit's summary (``rows``, ``tokens``, ``seconds``, ``status``)."""
        options: Dict[str, Any] = {}
        if model.path is not None:
            path = Path(model.path)
            if not path.is_dir():
                raise KitError(f"{path} is not a folder: there is no checkpoint to run")
            options["temperatures"] = _temperatures(path)
            spec = {"anchor": LOCAL_REPO, "anchor_revision": None, "name": model.name}

            def processor(repo: str, revision: Optional[str] = None) -> Any:
                from transformers import AutoProcessor

                return AutoProcessor.from_pretrained(str(path))

            options["factories"] = {"repo": lambda repo, revision=None: local_repo(path), "processor": processor}
        else:
            spec = {"anchor": model.repo, "anchor_revision": model.revision, "name": model.name}
        api = self._resolve()
        engine = api.load_engine(spec, quant=self.quant, device=self.device, tta_rotations=self.tta_rotations, name=model.name, **options)
        try:
            return dict(api.run_requests(engine, requests, out, budget=budget, resume=True))
        finally:
            del engine
            gc.collect()
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass
