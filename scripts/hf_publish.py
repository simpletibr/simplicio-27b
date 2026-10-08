#!/usr/bin/env python3
"""Sincroniza huggingface.co/wesleysimplicio/Simplicio-27B com este repo em UM commit (simulação por padrão).

Só biblioteca padrão. O cliente do Hugging Face é injetável (`main(..., client_factory=...)`); o real
(`HubClient`, que usa huggingface_hub) só é criado com --publish e só importa a biblioteca nessa hora.

Simulação (padrão): não lê o token, não cria cliente, não faz rede. O plano sai do diretório de origem
(--source, padrão: a raiz do repo) filtrado pela allowlist abaixo. Para ver também o que já está igual no Hub
(UPDATE em vez de ADD) e os arquivos do Hub que a allowlist não cobre, passe --remote-snapshot com o listing:
a resposta de GET https://huggingface.co/api/models/wesleysimplicio/Simplicio-27B/tree/main?recursive=true
salva em remote.json (o repo é público, não precisa de token).
  python3 scripts/hf_publish.py --remote-snapshot remote.json

Publicação: só com --publish (alias --apply), com HF_TOKEN no ambiente (nunca em argumento nem em arquivo),
com a árvore git limpa e HEAD igual ao branch remoto (a URL do commit na descrição tem de existir no GitHub).
Sem HF_TOKEN o script recusa antes de ler qualquer arquivo ou criar o cliente. O commit só ADICIONA e
ATUALIZA arquivos da allowlist: o script não tem operação que remova ou copie coisa alguma no Hub.

Allowlist, lado local (o que pode subir): README.md (o card), LICENSE, Modelfile, os assets/ que o README
cita, lora/adapter_config.json e lora/adapter_model.safetensors, e o generation_config.json gerado a partir
dos PARAMETER de amostragem do Modelfile. Tudo o mais é recusado ou ignorado: nomes de credencial (.env,
*token*, *.pem, ...), links simbólicos, caminhos que saem do diretório de origem, adapter solto na raiz,
pesos fora de lora/ e *.gguf (só com --allow-gguf, e só o que o Modelfile cita em FROM ./...).
Lado remoto: REMOTE_KEEP lista o que o Hub deve ter além da allowlist (config, shards, tokenizer, GGUF, lora/).
Qualquer outro arquivo do Hub vira um AVISO no plano ("órfão no Hub, não removido"); apagar é decisão do dono,
à mão. Isso inclui o adapter que hoje está na raiz do Hub: baixe-o para lora/ (ver lora/README.md), deixe o
--publish subi-lo e só então apague os dois arquivos da raiz.

Saída: 0 se ok, 2 se o script recusou (o motivo vai para o stderr).
"""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ID = "wesleysimplicio/Simplicio-27B"
GITHUB_COMMIT_URL = "https://github.com/simpletibr/simplicio-27b/commit/"
COMMIT_MESSAGE = "Sync from GitHub: allowlist, adapter in lora/, card metadata"
ROOT = Path(__file__).resolve().parent.parent
TOKEN_ENV = "HF_TOKEN"
BENCHMARK = "benchmarks/live_colab_g4_bf16_n120.json"
MIRRORED = ("README.md", "LICENSE", "Modelfile")
ADAPTER_DIR = "lora"
ADAPTER = ("adapter_config.json", "adapter_model.safetensors")
REMOTE_KEEP = (
    ".gitattributes", "config.json", "generation_config.json", "model.safetensors.index.json",
    "model-*.safetensors", "tokenizer*.json", "chat_template.jinja", "*processor_config.json",
    "*.gguf", "lora/*",
)
SAMPLING = {
    "temperature": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
    "repeat_penalty": "repetition_penalty", "presence_penalty": "presence_penalty",
}
SECRET_NAMES = (
    ".env*", "*.env", "*token*", "*secret*", "*credential*", "*password*", "*.pem", "*.key", "*.p12",
    "*.pfx", "id_rsa*", "id_ed25519*", ".netrc", "*.kdbx",
)
SKIP_DIRS = frozenset({".git", "__pycache__", ".ruff_cache", ".pytest_cache", ".venv", "venv", "node_modules",
                       ".ipynb_checkpoints", "scratchpad"})
WEIGHT_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt", ".h5", ".onnx")
SECRET_RE = re.compile(
    r"hf_[A-Za-z0-9]{30,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|AKIA[0-9A-Z]{16}"
    r"|Bearer\s+[A-Za-z0-9._~+/=-]{20,}|sk-[A-Za-z0-9]{20,}"
)
ASSET_RE = re.compile(r"assets/[A-Za-z0-9._-]+")
FROM_RE = re.compile(r"(?m)^FROM\s+\./(\S+)\s*$")
PARAMETER_RE = re.compile(r"(?m)^PARAMETER\s+(\w+)\s+(\S+)\s*$")
TEXT_SCAN_LIMIT = 1 << 20
WITHDRAWN_HEADING = "### Withdrawn numbers"
WITHDRAWN_FIGURES = ("96.5%", "Top 12")
# Metadados do card (front matter do README.md). Se o README divergir, o plano é recusado.
CARD_META = {
    "pretty_name": "Simplicio 27B",
    "license": "apache-2.0",
    "base_model": "Qwen/Qwen3.8-27B",
    "base_model_relation": "finetune",
    "library_name": "transformers",
    "pipeline_tag": "text-generation",
    "homepage": "https://simpleti.com.br/simplicio-27b/",
}
CARD_TAGS = ("qwen", "unsloth", "lora", "code-generation", "simplicio-loop", "software-engineering",
             "surgical-diff", "agentic-coding")


class Refused(Exception):
    """O script se recusa a continuar. A mensagem diz o motivo e nada foi enviado."""


@dataclass(frozen=True)
class Entry:
    """Um arquivo que vai para o Hub: lido de `source` ou gerado em memória (`data`)."""
    path: str
    source: Path | None = None
    data: bytes | None = None

    @property
    def size(self) -> int:
        return len(self.data) if self.data is not None else self.source.stat().st_size

    @property
    def payload(self) -> bytes | str:
        return self.data if self.data is not None else str(self.source)

    def chunks(self) -> Iterator[bytes]:
        if self.data is not None:
            yield self.data
            return
        with open(self.source, "rb") as handle:
            while chunk := handle.read(1 << 20):
                yield chunk

    def sha256(self) -> str:
        digest = hashlib.sha256()
        for chunk in self.chunks():
            digest.update(chunk)
        return digest.hexdigest()

    def blob_sha1(self) -> str:
        digest = hashlib.sha1(b"blob %d\0" % self.size)
        for chunk in self.chunks():
            digest.update(chunk)
        return digest.hexdigest()


@dataclass(frozen=True)
class RemoteFile:
    path: str
    blob_id: str
    lfs_sha256: str | None = None


@dataclass(frozen=True)
class Add:
    path: str
    payload: bytes | str
    size: int


@dataclass
class Scan:
    entries: dict[str, Entry]
    refused: list[tuple[str, str]]
    skipped: int
    readme: str
    modelfile: str


@dataclass
class Plan:
    ops: list[Add] = field(default_factory=list)
    lines: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def same_content(entry: Entry, remote: RemoteFile) -> bool:
    if remote.lfs_sha256:
        return entry.sha256() == remote.lfs_sha256
    return entry.blob_sha1() == remote.blob_id


def is_text_secret(entry: Entry) -> bool:
    """True se um arquivo de texto pequeno contém algo com cara de credencial."""
    if entry.size > TEXT_SCAN_LIMIT:
        return False
    try:
        text = b"".join(entry.chunks()).decode("utf-8")
    except UnicodeDecodeError:
        return False
    return SECRET_RE.search(text) is not None


def judge(rel: str, full: Path, root: Path, allowed: set[str], ggufs: set[str]) -> tuple[str, str]:
    """('ok'|'refuse'|'skip', motivo). 'skip' é fora da allowlist sem alarde; 'refuse' aparece no plano."""
    if full.is_symlink():
        inside = full.resolve().is_relative_to(root)
        return "refuse", "link simbólico" + ("" if inside else " que sai do diretório de origem")
    name = rel.rsplit("/", 1)[-1].lower()
    if any(fnmatch.fnmatch(name, pattern) for pattern in SECRET_NAMES):
        return "refuse", "nome de credencial ou segredo"
    if rel in ADAPTER:
        return "refuse", f"adapter na raiz: o lugar é {ADAPTER_DIR}/"
    # Pesos primeiro: nenhum caminho da allowlist (nem um asset citado no README) faz um peso subir.
    if name.endswith(WEIGHT_SUFFIXES) and rel not in {f"{ADAPTER_DIR}/adapter_model.safetensors", *ggufs}:
        if name.endswith(".gguf"):
            return "refuse", "GGUF grande: só com --allow-gguf e se o Modelfile tiver FROM ./este-arquivo"
        return "refuse", f"pesos só entram em {ADAPTER_DIR}/ e só o adapter"
    if not full.resolve().is_relative_to(root):
        return "refuse", "fora do diretório de origem"
    if rel in allowed or rel in ggufs:
        return "ok", ""
    return "skip", ""


def scan_source(source: Path, allow_gguf: bool = False) -> Scan:
    """Percorre `source` e separa o que a allowlist deixa subir do que é recusado ou ignorado."""
    root = source.resolve()
    if not root.is_dir():
        raise Refused(f"diretório de origem inexistente: {source}")
    for name in MIRRORED:
        if (root / name).is_symlink() or not (root / name).is_file():
            raise Refused(f"{name} ausente (ou é link simbólico) em {root}")
    try:
        readme = (root / "README.md").read_bytes().decode("utf-8")
        modelfile = (root / "Modelfile").read_bytes().decode("utf-8")
    except UnicodeDecodeError as exc:
        raise Refused(f"README.md e Modelfile precisam ser UTF-8: {exc}") from exc
    ggufs = set(FROM_RE.findall(modelfile)) if allow_gguf else set()
    allowed = {*MIRRORED, *(f"{ADAPTER_DIR}/{name}" for name in ADAPTER), *ASSET_RE.findall(readme)}
    entries: dict[str, Entry] = {}
    refused: list[tuple[str, str]] = []
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        keep = []
        for name in sorted(dirnames):
            if name in SKIP_DIRS:
                continue
            if (base / name).is_symlink():
                rel = (base / name).relative_to(root).as_posix()
                refused.append((rel, judge(rel, base / name, root, allowed, ggufs)[1]))
                continue
            keep.append(name)
        dirnames[:] = keep
        for name in sorted(filenames):
            full = base / name
            rel = full.relative_to(root).as_posix()
            verdict, reason = judge(rel, full, root, allowed, ggufs)
            if verdict == "refuse":
                refused.append((rel, reason))
            elif verdict == "skip" or not full.is_file():
                skipped += 1
            else:
                entries[rel] = Entry(rel, source=full)
    for rel, entry in entries.items():
        if is_text_secret(entry):
            raise Refused(f"{rel} contém algo que parece credencial: recuso publicar")
    return Scan(entries, refused, skipped, readme, modelfile)


def sampling_params(modelfile: str) -> dict[str, int | float]:
    params: dict[str, int | float] = {}
    for name, value in PARAMETER_RE.findall(modelfile):
        if name in SAMPLING:
            try:
                params[SAMPLING[name]] = int(value) if name == "top_k" else float(value)
            except ValueError as exc:
                raise Refused(f"Modelfile: PARAMETER {name} {value} não é número") from exc
    if "temperature" not in params:
        raise Refused("Modelfile sem PARAMETER temperature: não sei gerar generation_config.json")
    return params


def generation_config(remote_json: bytes, modelfile: str) -> bytes:
    """generation_config.json do Hub com a amostragem trocada pela do Modelfile (e só ela)."""
    params = sampling_params(modelfile)
    try:
        current = json.loads(remote_json)
    except ValueError as exc:
        raise Refused(f"generation_config.json do Hub não é JSON: {exc}") from exc
    cfg = {key: value for key, value in current.items() if key not in SAMPLING.values()}
    cfg["do_sample"] = True
    cfg.update(params)
    return (json.dumps(cfg, indent=2) + "\n").encode()


def front_matter(text: str) -> dict[str, Any]:
    """Chaves do front matter YAML do card: valor simples ('k: v') ou lista ('k:' e linhas '- item')."""
    match = re.match(r"---\n(.*?)\n---\n", text, re.S)
    meta: dict[str, Any] = {}
    key = None
    for line in (match.group(1).splitlines() if match else []):
        if line.startswith("- ") and key is not None and isinstance(meta[key], list):
            meta[key].append(line[2:].strip())
            continue
        name, colon, value = line.partition(":")
        if colon and not line.startswith(" "):
            key = name.strip()
            meta[key] = value.strip() or []
    return meta


def strip_section(text: str, heading: str) -> str:
    start = text.find(heading)
    if start < 0:
        return text
    following = re.search(r"\n#{1,3} ", text[start + len(heading):])
    end = start + len(heading) + following.start() if following else len(text)
    return text[:start] + text[end:]


def benchmark_claims(bench: dict[str, Any]) -> tuple[list[str], set[int]]:
    """Frases de resultado que o card precisa trazer, e os numeradores de N/120 que o card pode citar."""
    n = bench["n"]
    passed, matched = bench["functional_pass"], bench["diff_matched"]
    claims = [
        f"{passed}/{n} runs ({passed / n:.1%})",
        f"{matched}/{n} ({matched / n:.1%})",
        f"{bench['mean_tokens']} / {bench['max_tokens']}",
        f"{round(bench['seconds'])} s",
    ]
    numerators = {n, *(bench[key] for key in ("functional_pass", "diff_matched", "ast_valid",
                                              "zero_ghost_apis", "unit_test_passed"))}
    return claims, numerators


def card_problems(readme: str, bench: dict[str, Any]) -> list[str]:
    """O que falta ou sobra no card em relação aos metadados versionados e a benchmarks/."""
    problems = []
    meta = front_matter(readme)
    for key, expected in CARD_META.items():
        if meta.get(key) != expected:
            problems.append(f"front matter: {key} deveria ser {expected!r}, está {meta.get(key)!r}")
    tags = meta.get("tags") if isinstance(meta.get("tags"), list) else []
    if missing := [tag for tag in CARD_TAGS if tag not in tags]:
        problems.append(f"front matter: faltam tags {missing}")
    try:
        claims, numerators = benchmark_claims(bench)
        if abs(bench["functional_rate"] - bench["functional_pass"] / bench["n"]) > 5e-5:
            problems.append(f"{BENCHMARK}: functional_rate não bate com functional_pass/n")
    except (KeyError, TypeError, ZeroDivisionError) as exc:
        return [*problems, f"{BENCHMARK}: campo ausente ou inválido ({exc!r})"]
    problems += [f"o card não traz {claim!r}, que é o que {BENCHMARK} registra"
                 for claim in claims if claim not in readme]
    body = strip_section(readme, WITHDRAWN_HEADING)
    cited = {int(num) for num in re.findall(r"\b(\d+)(?:/| of )%d\b" % bench["n"], body)}
    if extra := sorted(cited - numerators):
        problems.append(f"o card cita {extra} de {bench['n']} execuções, que {BENCHMARK} não registra")
    problems += [f"o card repete o número retirado {figure!r} fora de '{WITHDRAWN_HEADING[4:]}'"
                 for figure in WITHDRAWN_FIGURES if figure in body]
    return problems


def build_card(readme: str, bench: dict[str, Any]) -> str:
    """O card é o README.md, sem acréscimo: só sobe se os metadados e os números baterem com o repo."""
    if problems := card_problems(readme, bench):
        raise Refused("card inválido: " + "; ".join(problems))
    return readme


def load_benchmark(root: Path) -> dict[str, Any]:
    path = root / BENCHMARK
    try:
        bench = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise Refused(f"{BENCHMARK} ilegível ({exc}): não publico números sem fonte") from exc
    if not isinstance(bench, dict):
        raise Refused(f"{BENCHMARK} deveria ser um objeto JSON")
    return bench


def load_snapshot(path: Path) -> dict[str, RemoteFile]:
    """Listing do Hub salvo em arquivo: resposta de /api/models/<repo>/tree/main?recursive=true, ou
    lista de {path, blob_id, lfs_sha256}."""
    try:
        raw = json.loads(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise Refused(f"snapshot ilegível: {path} ({exc})") from exc
    if not isinstance(raw, list):
        raise Refused("snapshot deve ser uma lista JSON de arquivos")
    files: dict[str, RemoteFile] = {}
    for item in raw:
        if not isinstance(item, dict) or item.get("type") == "directory":
            continue
        lfs = item.get("lfs")
        blob_id = item.get("blob_id") or item.get("oid")
        if not isinstance(item.get("path"), str) or not isinstance(blob_id, str):
            raise Refused("snapshot: item sem path ou oid")
        sha = item.get("lfs_sha256") or (lfs.get("oid") if isinstance(lfs, dict) else None)
        files[item["path"]] = RemoteFile(item["path"], blob_id, sha)
    return files


def remote_allowed(path: str, entries: dict[str, Entry]) -> bool:
    return path in entries or any(fnmatch.fnmatch(path, pattern) for pattern in REMOTE_KEEP)


def build_plan(entries: dict[str, Entry], modelfile: str, remote: dict[str, RemoteFile] | None = None,
               remote_gen: bytes | None = None) -> Plan:
    """Operações para o commit. Sem `remote` (simulação offline) só há ADD do que a allowlist deixa subir."""
    plan = Plan()
    params = sampling_params(modelfile)
    entries = dict(entries)
    if remote_gen is not None:
        entries["generation_config.json"] = Entry("generation_config.json",
                                                  data=generation_config(remote_gen, modelfile))
    else:
        plan.notes.append("MERGE generation_config.json: " + ", ".join(f"{k}={v}" for k, v in params.items())
                          + " (o arquivo do Hub só é lido no --publish)")
    if remote is None:
        for path, entry in sorted(entries.items()):
            plan.ops.append(Add(path, entry.payload, entry.size))
            plan.lines.append(f"ADD {path} ({entry.size} bytes)")
        plan.notes.append("Hub não consultado: o que já está igual lá (UPDATE) e os órfãos do Hub só aparecem "
                          "com --remote-snapshot ou --publish")
        return plan
    final = set(remote)
    for path, entry in sorted(entries.items()):
        if path not in remote or not same_content(entry, remote[path]):
            plan.ops.append(Add(path, entry.payload, entry.size))
            plan.lines.append(f"{'UPDATE' if path in remote else 'ADD'} {path} ({entry.size} bytes)")
            final.add(path)
    for path in sorted(remote):
        if not remote_allowed(path, entries):
            hint = f": mova para {ADAPTER_DIR}/ e apague à mão depois de conferir" if path in ADAPTER else ""
            plan.warnings.append(f"órfão no Hub, não removido: {path}{hint}")
    required = ["config.json", "model.safetensors.index.json", *(f"{ADAPTER_DIR}/{n}" for n in ADAPTER),
                *FROM_RE.findall(modelfile)]
    if missing := [path for path in required if path not in final]:
        hint = (f"; o card cita {ADAPTER_DIR}/: baixe o adapter para {ADAPTER_DIR}/ (ver {ADAPTER_DIR}/README.md)"
                if any(path.startswith(f"{ADAPTER_DIR}/") for path in missing) else "")
        raise Refused(f"plano inválido: faltam no Hub {missing}{hint}")
    return plan


class HubClient:
    """Cliente real. Só é criado com --publish e HF_TOKEN; huggingface_hub é importado aqui, não no topo."""

    def __init__(self, token: str) -> None:
        try:
            import huggingface_hub as hub
        except ImportError:
            raise Refused("huggingface_hub não instalado (versão fixada pelo repo: huggingface_hub==1.33.0); "
                          "só o --publish precisa dele") from None
        self._hub = hub
        self._api = hub.HfApi(token=token)

    def head_sha(self, repo_id: str) -> str:
        return self._api.model_info(repo_id).sha

    def list_files(self, repo_id: str, revision: str) -> dict[str, RemoteFile]:
        files = {}
        for item in self._api.list_repo_tree(repo_id, recursive=True, revision=revision):
            if hasattr(item, "blob_id"):  # pastas não têm blob_id
                files[item.path] = RemoteFile(item.path, item.blob_id, item.lfs.sha256 if item.lfs else None)
        return files

    def read_file(self, repo_id: str, path: str, revision: str) -> bytes:
        return Path(self._api.hf_hub_download(repo_id=repo_id, filename=path, revision=revision)).read_bytes()

    def create_commit(self, repo_id: str, ops: list[Add], parent: str, message: str, description: str) -> str:
        operations = [self._hub.CommitOperationAdd(path_in_repo=op.path, path_or_fileobj=op.payload)
                      for op in ops]
        info = self._api.create_commit(repo_id, operations=operations, parent_commit=parent,
                                       commit_message=message, commit_description=description)
        return info.commit_url


def default_git(source: Path) -> Callable[..., str]:
    def run(*args: str) -> str:
        try:
            done = subprocess.run(["git", *args], cwd=source, capture_output=True, text=True, check=True)
        except (OSError, subprocess.CalledProcessError) as exc:
            raise Refused(f"git {' '.join(args)} falhou em {source}") from exc
        return done.stdout.strip()
    return run


def run(args: argparse.Namespace, env: Any, client_factory: Callable[[str], Any],
        git: Callable[..., str] | None) -> int:
    source = Path(args.source).resolve()
    token = upstream = ""
    if args.publish:
        if args.remote_snapshot:
            raise Refused("--remote-snapshot é da simulação; o --publish lê o Hub sozinho")
        token = str(env.get(TOKEN_ENV, "")).strip()
        if not token:
            raise Refused(f"--publish exige {TOKEN_ENV} no ambiente (nunca em argumento ou arquivo); "
                          "nada foi lido nem enviado")
        git = git or default_git(source)
        if git("status", "--porcelain"):
            raise Refused("working tree com mudanças: faça commit antes de --publish")
        head = git("rev-parse", "HEAD")
        try:
            upstream = git("rev-parse", "@{upstream}")
        except Refused:
            raise Refused("o branch atual não tem upstream no GitHub: faça push -u antes de --publish") from None
        if head != upstream:
            raise Refused(f"HEAD ({head[:7]}) difere do branch remoto ({upstream[:7]}): faça push ou pull antes de "
                          "--publish, para a URL do commit na descrição existir no GitHub")
    scan = scan_source(source, allow_gguf=args.allow_gguf)
    entries = dict(scan.entries)
    card = build_card(scan.readme, load_benchmark(source))
    entries["README.md"] = Entry("README.md", data=card.encode("utf-8"))
    client = revision = remote = remote_gen = None
    if args.publish:
        client = client_factory(token)
        revision = client.head_sha(args.repo_id)
        remote = client.list_files(args.repo_id, revision)
        remote_gen = client.read_file(args.repo_id, "generation_config.json", revision)
    elif args.remote_snapshot:
        remote = load_snapshot(Path(args.remote_snapshot))
    plan = build_plan(entries, scan.modelfile, remote, remote_gen)
    where = revision or (f"snapshot {args.remote_snapshot}" if args.remote_snapshot else "não consultado (simulação offline)")
    print(f"{args.repo_id} @ {where}")
    print(f"origem: {source}")
    for line in plan.lines:
        print(line)
    for rel, reason in scan.refused:
        print(f"REFUSE {rel} ({reason})")
    if scan.skipped:
        print(f"SKIP {scan.skipped} arquivo(s) fora da allowlist")
    for note in plan.notes:
        print(f"NOTA {note}")
    for warning in plan.warnings:
        print(f"AVISO {warning}")
    print(f"{len(plan.ops)} operações")
    if not plan.ops:
        return 0
    if not args.publish:
        print(f"dry-run: nada foi enviado. Revise o plano e rode de novo com --publish (exige {TOKEN_ENV}).")
        return 0
    description = f"{GITHUB_COMMIT_URL}{upstream}"
    print(client.create_commit(args.repo_id, plan.ops, revision, COMMIT_MESSAGE, description))
    return 0


def main(argv: list[str] | None = None, *, env: Any = None,
         client_factory: Callable[[str], Any] | None = None, git: Callable[..., str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--publish", "--apply", dest="publish", action="store_true",
                        help=f"cria o commit no Hugging Face (exige {TOKEN_ENV} no ambiente)")
    parser.add_argument("--source", default=str(ROOT), help="diretório de origem (padrão: a raiz do repo)")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--remote-snapshot", metavar="ARQUIVO",
                        help="listing do Hub salvo em JSON, para a simulação comparar com o Hub")
    parser.add_argument("--allow-gguf", action="store_true",
                        help="deixa subir o *.gguf que o Modelfile cita em FROM ./...")
    args = parser.parse_args(argv)
    try:
        return run(args, os.environ if env is None else env, client_factory or HubClient, git)
    except Refused as exc:
        print(f"recusado: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
