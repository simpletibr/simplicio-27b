"""Where the rented day gets its inputs and puts its outputs: the Hugging Face Hub, or a folder that stands in for it.

Two implementations of the same small interface:

* :class:`HfHub` wraps ``huggingface_hub``.  The token is read from an environment variable whose NAME comes from the config
  (``rental.hub.token_env``); it is never stored where ``repr`` can show it and never written to a log.  UNVERIFIED: the calls
  were checked against the signatures of the installed ``huggingface_hub`` (a test), not against the real service.
* :class:`LocalHub` keeps repos as folders.  It is how the whole day is rehearsed on a laptop, and how the tests run.

Rules both follow: every repo the day writes to is private (a public one is refused before any upload; making a model public
is the owner's decision, outside this tool), and a file whose name looks like a secret is never uploaded, whatever the patterns say.
"""
from __future__ import annotations

import contextlib
import fnmatch
import hashlib
import json
import os
import re
import shutil
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Sequence

#: What leaves the machine from a checkpoint folder: the weights and the small configs, not the training state
#: (``resume.pt``, optimizer shards, ``events.jsonl``, ``*.tmp``).
CHECKPOINT_PATTERNS = ("*.safetensors", "*.json", "*.jinja", "*.txt", "tokenizer*", "*.model", "vocab*", "merges*")
KINDS = ("model", "dataset")
#: A tokenizer's ``added_tokens.json`` and ``special_tokens_map.json`` are not credentials; ``token.json`` and ``hf_token.txt`` are.  A name with ``key``
#: or ``bearer`` as a word of its own (``openai_key.txt``, ``gcp-key.json``, ``bearer.txt``) or ``service-account`` in it is refused; ``keyword_args.json`` is not.
#: It is a list of NAMES: a secret in a file called ``notes.txt`` goes through (the RUNBOOK says so).
_SECRET = re.compile(r"(^\.env)|((^|[_-])(hf|access|api|auth|bearer)[_-]?tokens?([._-]|$))|(^tokens?(\.[a-z0-9]+)?$)|secret|credential"
                     r"|(^|[_-])api[_-]?keys?([._-]|$)|(^|[_-])o?auth([._-]|$)|passw(or)?d|^\.(netrc|npmrc|pypirc|pgpass)$"
                     r"|(^|[_-])keys?([._-]|$)|(^|[_-])bearer([._-]|$)|service[_-]?account"
                     r"|^id_(rsa|dsa|ecdsa|ed25519)|\.(pem|key|p12|pfx)$", re.IGNORECASE)


class HubError(RuntimeError):
    """The Hub (or its stand-in) cannot do what the day asks, with the reason."""


def _kind(kind: str) -> str:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    return kind


def _selected(root: Path, patterns: Optional[Sequence[str]]) -> List[Path]:
    files = sorted((f for f in root.rglob("*") if f.is_file() and not f.name.endswith(".tmp")), key=lambda f: f.relative_to(root).as_posix())
    if not patterns:
        return files
    return [f for f in files if any(fnmatch.fnmatch(f.relative_to(root).as_posix(), pattern) or fnmatch.fnmatch(f.name, pattern) for pattern in patterns)]


def check_upload_names(folder: "str | os.PathLike[str]", patterns: Optional[Sequence[str]]) -> None:
    """Refuse when a file that would be uploaded has a name that looks like a secret (``.env``, ``*token*``, ``*.pem``, ...)."""
    root = Path(folder)
    for file in _selected(root, patterns):
        if _SECRET.search(file.name):
            raise HubError(f"{file.relative_to(root).as_posix()} looks like a secret by its name, so nothing was uploaded; move it away or rename it")


class Hub:
    """What the day needs from a Hub."""

    def ensure_repo(self, repo: str, *, kind: str = "model") -> None:
        raise NotImplementedError

    def upload(self, repo: str, folder: "str | os.PathLike[str]", *, kind: str = "model", path_in_repo: str = "",
               patterns: Optional[Sequence[str]] = None, message: str = "") -> str:
        raise NotImplementedError

    def download(self, repo: str, dest: "str | os.PathLike[str]", *, kind: str = "model", revision: Optional[str] = None,
                 patterns: Optional[Sequence[str]] = None) -> Path:
        raise NotImplementedError

    def files(self, repo: str, *, kind: str = "model", revision: Optional[str] = None, prefix: str = "") -> List[str]:
        raise NotImplementedError


def _under(prefix: str, names: Sequence[str]) -> List[str]:
    prefix = prefix.strip("/")
    return sorted(name for name in names if not prefix or name == prefix or name.startswith(prefix + "/"))


# --------------------------------------------------------------------------- a folder as a Hub

class LocalHub(Hub):
    """Repos are folders under ``root/<kind>/<owner>/<name>``; revisions are not kept (any revision reads the current files)."""

    def __init__(self, root: "str | os.PathLike[str]") -> None:
        self.root = Path(root)

    def _dir(self, repo: str, kind: str) -> Path:
        return self.root / _kind(kind) / repo

    def _meta(self, repo: str, kind: str) -> Path:
        return self.root / ".meta" / kind / (repo.replace("/", "__") + ".json")

    def _exists(self, repo: str, kind: str) -> Path:
        directory = self._dir(repo, kind)
        if not directory.is_dir():
            raise HubError(f"repo {repo} ({kind}) does not exist in {self.root}")
        return directory

    def is_private(self, repo: str, *, kind: str = "model") -> bool:
        self._exists(repo, kind)
        return bool(json.loads(self._meta(repo, kind).read_text(encoding="utf-8"))["private"])

    def make_public_for_test(self, repo: str, *, kind: str = "model") -> None:
        self._exists(repo, kind)
        self._meta(repo, kind).write_text(json.dumps({"private": False}), encoding="utf-8")

    def ensure_repo(self, repo: str, *, kind: str = "model") -> None:
        directory = self._dir(repo, kind)
        if not directory.is_dir():
            directory.mkdir(parents=True)
            self._meta(repo, kind).parent.mkdir(parents=True, exist_ok=True)
            self._meta(repo, kind).write_text(json.dumps({"private": True}), encoding="utf-8")
        if not self.is_private(repo, kind=kind):
            raise HubError(f"repo {repo} is public; the day only writes to private repos (make it private, or use another repo)")

    def upload(self, repo: str, folder: "str | os.PathLike[str]", *, kind: str = "model", path_in_repo: str = "",
               patterns: Optional[Sequence[str]] = None, message: str = "") -> str:
        directory = self._exists(repo, kind)
        if not self.is_private(repo, kind=kind):
            raise HubError(f"repo {repo} is public; nothing was uploaded")
        source = Path(folder)
        check_upload_names(source, patterns)
        for file in _selected(source, patterns):
            target = directory / path_in_repo.strip("/") / file.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_name(target.name + ".tmp")
            shutil.copyfile(file, pending)
            os.replace(pending, target)
        listing = [[name, (directory / name).stat().st_size] for name in self.files(repo, kind=kind)]
        return hashlib.sha256(json.dumps(listing).encode("utf-8")).hexdigest()[:40]

    def files(self, repo: str, *, kind: str = "model", revision: Optional[str] = None, prefix: str = "") -> List[str]:
        directory = self._exists(repo, kind)
        return _under(prefix, [f.relative_to(directory).as_posix() for f in directory.rglob("*") if f.is_file()])

    def download(self, repo: str, dest: "str | os.PathLike[str]", *, kind: str = "model", revision: Optional[str] = None,
                 patterns: Optional[Sequence[str]] = None) -> Path:
        directory = self._exists(repo, kind)
        out = Path(dest)
        for file in _selected(directory, patterns):
            target = out / file.relative_to(directory)
            target.parent.mkdir(parents=True, exist_ok=True)
            pending = target.with_name(target.name + ".tmp")
            shutil.copyfile(file, pending)
            os.replace(pending, target)
        out.mkdir(parents=True, exist_ok=True)
        return out


# --------------------------------------------------------------------------- huggingface_hub

#: What the commonest failures of the Hub usually mean, by the name of the exception class (shown next to the repo that failed).
_HINTS = {
    "RepositoryNotFoundError": "the repo does not exist, or this token cannot see it: a private repo looks like a missing one",
    "GatedRepoError": "a gated repo: accept its terms on the Hub with the account the token belongs to",
    "RevisionNotFoundError": "no such revision in this repo",
    "EntryNotFoundError": "no such file in this repo",
    "LocalEntryNotFoundError": "the file is not on this machine and the Hub could not be reached",
    "HfHubHTTPError": "the Hub refused the request: check the token's permissions (write access for uploads)",
}


class HfHub(Hub):
    def __init__(self, api: Any, *, snapshot: Callable[..., Any], scrub: Callable[[str], str] = lambda text: text) -> None:
        self._api = api
        self._snapshot = snapshot
        self._scrub = scrub                    # a function, not the secret: nothing here can show the token

    @contextlib.contextmanager
    def _guard(self, action: str, repo: str) -> Iterator[None]:
        """Whatever the Hub client raises becomes a :class:`HubError` that names the action and the repo, with the token scrubbed and
        the original traceback not chained (it can carry request details).  A missing package stays an ``ImportError``."""
        try:
            yield
        except (HubError, ImportError):
            raise
        except Exception as exc:               # noqa: BLE001 - the client raises its own classes, httpx's and OSError's
            name = type(exc).__name__
            first = self._scrub(str(exc).strip().splitlines()[0]) if str(exc).strip() else ""
            hint = _HINTS.get(name) or next((text for klass, text in _HINTS.items() if klass in {c.__name__ for c in type(exc).__mro__}), "")
            raise HubError(f"{action} {repo}: {name}" + (f": {first}" if first else "") + (f" ({hint})" if hint else "")) from None

    def __repr__(self) -> str:
        return "HfHub()"

    @classmethod
    def from_env(cls, token_env: str, *, endpoint: str = "https://huggingface.co") -> "HfHub":
        token = os.environ.get(token_env)
        if not token:
            raise HubError(f"the environment variable {token_env} is not set; export your Hugging Face access token in it (the token itself is never written anywhere)")
        try:
            from huggingface_hub import HfApi, snapshot_download
        except ImportError as exc:
            raise ImportError("huggingface_hub is required (install the pinned packages of decide/rental/requirements.txt)") from exc

        def snapshot(**kwargs: Any) -> Any:
            return snapshot_download(token=token, endpoint=endpoint, **kwargs)

        return cls(HfApi(token=token, endpoint=endpoint), snapshot=snapshot, scrub=lambda text: text.replace(token, "***"))

    def _private(self, repo: str, kind: str) -> None:
        if not self._api.repo_info(repo, repo_type=_kind(kind)).private:
            raise HubError(f"repo {repo} is public; the day only writes to private repos (make it private, or use another repo)")

    def ensure_repo(self, repo: str, *, kind: str = "model") -> None:
        with self._guard("creating", repo):
            self._api.create_repo(repo, private=True, repo_type=_kind(kind), exist_ok=True)
            self._private(repo, kind)

    def upload(self, repo: str, folder: "str | os.PathLike[str]", *, kind: str = "model", path_in_repo: str = "",
               patterns: Optional[Sequence[str]] = None, message: str = "") -> str:
        with self._guard("uploading to", repo):
            self._private(repo, kind)
            check_upload_names(folder, patterns)
            info = self._api.upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=path_in_repo, repo_type=kind,
                                           allow_patterns=list(patterns) if patterns else None, commit_message=message or "upload from the rented day")
            return str(info.oid)

    def download(self, repo: str, dest: "str | os.PathLike[str]", *, kind: str = "model", revision: Optional[str] = None,
                 patterns: Optional[Sequence[str]] = None) -> Path:
        with self._guard("downloading", repo):
            self._snapshot(repo_id=repo, repo_type=_kind(kind), revision=revision, local_dir=str(dest), allow_patterns=list(patterns) if patterns else None)
        return Path(dest)

    def files(self, repo: str, *, kind: str = "model", revision: Optional[str] = None, prefix: str = "") -> List[str]:
        with self._guard("listing", repo):
            return _under(prefix, list(self._api.list_repo_files(repo, revision=revision, repo_type=_kind(kind))))
