"""Flow test for issue #23: scripts/hf_publish.py end to end, with a fake Hub client and no network.

  * Dry run (the default) over a temporary source directory that holds allowed and forbidden files:
    only the allowlist reaches the plan, everything else is REFUSEd or SKIPped, and the Hub client is
    never even created.
  * `--publish` without HF_TOKEN is refused before any file is read or client is built. With a token, a
    clean tree and HEAD pushed it builds one commit that only ADDS and UPDATES allowlisted files
    (generation_config.json rewritten from the Modelfile sampling). Files the Hub has outside the allowlist,
    weights included, are reported as orphans and are never deleted or copied.
  * The real client (HubClient) is exercised against a stand-in `huggingface_hub` module.
Nothing here talks to huggingface.co and no real token is used.
"""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import hf_publish  # noqa: E402

FAKE_TOKEN = "hf_" + "fakefake" * 4  # built at run time: no literal token in the repository
GEN = b'{"bos_token_id": 1, "do_sample": true, "temperature": 1.0, "top_k": 20, "top_p": 0.95}'
COPIED = ("README.md", "LICENSE", "Modelfile", "assets/simplicio-logo.png", hf_publish.BENCHMARK)
ALLOWED_ADDS = {
    "LICENSE", "Modelfile", "README.md", "assets/simplicio-logo.png",
    "lora/adapter_config.json", "lora/adapter_model.safetensors",
}
# Files the Hub has today that the allowlist does not cover: reported, never removed.
ORPHANS = {"adapter_config.json", "adapter_model.safetensors", "assets/leaderboard_top10.svg",
           "deploy/serve_vllm.sh", "data/x.jsonl", "model.safetensors", "vocab.json", "merges.txt",
           "special_tokens_map.json", "added_tokens.json", "training_args.bin"}
MODELFILE = (ROOT / "Modelfile").read_text(encoding="utf-8")


def regular(path: str, data: bytes) -> hf_publish.RemoteFile:
    return hf_publish.RemoteFile(path, hf_publish.Entry(path, data=data).blob_sha1())


def lfs(path: str, data: bytes) -> hf_publish.RemoteFile:
    return hf_publish.RemoteFile(path, hashlib.sha1(data).hexdigest(), hashlib.sha256(data).hexdigest())


def stale_remote() -> dict[str, hf_publish.RemoteFile]:
    ggufs = hf_publish.FROM_RE.findall(MODELFILE)
    files = [regular(".gitattributes", b"a"), regular("config.json", b"{}"),
             regular("model.safetensors.index.json", b"{}"), lfs("model-00001-of-00018.safetensors", b"w"),
             regular("generation_config.json", GEN), regular("tokenizer.json", b"t"),
             regular("processor_config.json", b"p"), regular("chat_template.jinja", b"c"),
             regular("adapter_config.json", b"{}"), lfs("adapter_model.safetensors", b"a"),
             lfs("assets/leaderboard_top10.svg", b"s"), regular("deploy/serve_vllm.sh", b"old"),
             regular("data/x.jsonl", b"d"), lfs("model.safetensors", b"m"), regular("vocab.json", b"v"),
             regular("merges.txt", b"m"), regular("special_tokens_map.json", b"s"),
             regular("added_tokens.json", b"a"), lfs("training_args.bin", b"t"),
             *(lfs(name, b"g") for name in ggufs)]
    return {f.path: f for f in files}


class FakeClient:
    """Stands in for HubClient: records every call, never touches the network."""

    def __init__(self, files: dict[str, hf_publish.RemoteFile], gen: bytes = GEN) -> None:
        self.files, self.gen, self.calls, self.commits = files, gen, [], []

    def head_sha(self, repo_id):
        self.calls.append("head_sha")
        return "abc123"

    def list_files(self, repo_id, revision):
        self.calls.append("list_files")
        return dict(self.files)

    def read_file(self, repo_id, path, revision):
        self.calls.append(f"read_file {path}")
        return self.gen

    def create_commit(self, repo_id, ops, parent, message, description):
        self.calls.append("create_commit")
        self.commits.append({"repo": repo_id, "ops": ops, "parent": parent, "message": message,
                             "description": description})
        return "https://huggingface.co/x/commit/1"

    def __getattr__(self, name):  # any other call (delete_file, copy, ...) is a bug in the script
        raise AssertionError(f"chamada inesperada ao cliente do Hub: {name}")


def write_snapshot(path: Path) -> None:
    """The Hub listing in the shape of GET /api/models/<repo>/tree/main?recursive=true."""
    tree = [{"type": "directory", "path": "assets", "oid": "d" * 40}]
    for name, remote in stale_remote().items():
        item = {"type": "file", "path": name, "oid": remote.blob_id, "size": 1}
        if remote.lfs_sha256:
            item["lfs"] = {"oid": remote.lfs_sha256, "size": 1}
        tree.append(item)
    path.write_text(json.dumps(tree))


def forbidden_factory(token):
    raise AssertionError("o cliente do Hub não pode ser criado nesta execução")


def run_main(*argv, env=None, factory=forbidden_factory, git=None):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = hf_publish.main(list(argv), env={} if env is None else env, client_factory=factory, git=git)
    return code, out.getvalue(), err.getvalue()


def make_source(tmp: Path, adapter: bool = True) -> Path:
    src = tmp / "src"
    for name in COPIED:
        (src / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, src / name)
    if adapter:
        (src / "lora").mkdir(exist_ok=True)
        (src / "lora" / "adapter_config.json").write_bytes(b'{"r": 32}')
        (src / "lora" / "adapter_model.safetensors").write_bytes(b"not real weights")
    return src


def add_forbidden(src: Path) -> dict[str, str]:
    """Create files the allowlist must not let through; returns {path: reason fragment}."""
    files = {
        ".env": "credencial", "hf_token.txt": "credencial", "keys/id_rsa": "credencial",
        "server.pem": "credencial", "lora/token.json": "credencial",
        "model.Q4_K_M.gguf": "GGUF", "adapter_config.json": "adapter na raiz",
        "weights/extra.safetensors": "pesos",
    }
    for rel, _ in files.items():
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(b"x")
    for rel in ("notes.txt", "deploy/serve.sh"):
        (src / rel).parent.mkdir(parents=True, exist_ok=True)
        (src / rel).write_bytes(b"x")
    outside = src.parent / "outside"
    outside.mkdir()
    (outside / "secret.txt").write_bytes(b"x")
    try:
        os.symlink(outside / "secret.txt", src / "escape.txt")
        os.symlink(outside, src / "linkdir")
    except OSError:
        return files
    files.update({"escape.txt": "link simbólico", "linkdir": "link simbólico"})
    return files


class TempCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)


class DryRunTest(TempCase):
    def test_plan_has_only_allowed_files(self):
        src = make_source(self.tmp)
        forbidden = add_forbidden(src)
        code, out, err = run_main("--source", str(src))
        self.assertEqual((code, err), (0, ""))
        added = set(re.findall(r"(?m)^ADD (\S+) \(\d+ bytes\)$", out))
        self.assertEqual(added, ALLOWED_ADDS)
        refused = dict(re.findall(r"(?m)^REFUSE (\S+) \((.*)\)$", out))
        self.assertEqual(set(refused), set(forbidden))
        for path, reason in forbidden.items():
            self.assertIn(reason, refused[path])
        self.assertNotIn("notes.txt", out)
        self.assertNotIn("serve.sh", out)
        self.assertRegex(out, r"SKIP \d+ arquivo\(s\) fora da allowlist")

    def test_no_client_call_without_publish(self):
        src = make_source(self.tmp)
        snapshot = self.tmp / "remote.json"
        write_snapshot(snapshot)
        calls = []
        for extra in ([], ["--remote-snapshot", str(snapshot)]):
            with self.subTest(extra=extra):
                code, out, _ = run_main("--source", str(src), *extra,
                                        env={"HF_TOKEN": FAKE_TOKEN},
                                        factory=lambda token: calls.append(token))
                self.assertEqual(code, 0)
                self.assertIn("dry-run: nada foi enviado", out)
        self.assertEqual(calls, [])

    def test_default_source_is_this_repo(self):
        code, out, err = run_main()
        self.assertEqual((code, err), (0, ""))
        self.assertIn("ADD README.md", out)
        self.assertNotRegex(out, r"(?m)^ADD (tests|scripts|deploy|data|benchmarks)/")

    def test_gguf_needs_explicit_permission(self):
        src = make_source(self.tmp)
        named = hf_publish.FROM_RE.findall(MODELFILE)[0]
        (src / named).write_bytes(b"gguf")
        (src / "other.gguf").write_bytes(b"gguf")
        _, out, _ = run_main("--source", str(src))
        self.assertNotIn(f"ADD {named}", out)
        self.assertIn(f"REFUSE {named} (GGUF", out)
        _, out, _ = run_main("--source", str(src), "--allow-gguf")
        self.assertIn(f"ADD {named} (4 bytes)", out)
        self.assertIn("REFUSE other.gguf (GGUF", out)

    def test_weights_cited_by_the_readme_are_refused(self):
        src = make_source(self.tmp)
        cited = [f"assets/x{suffix}" for suffix in (".safetensors", ".bin", ".pt", ".pth", ".gguf", ".ckpt",
                                                    ".h5", ".onnx", ".SAFETENSORS")]
        with open(src / "README.md", "a", encoding="utf-8") as handle:
            handle.writelines(f"\n![w]({name})\n" for name in cited)
        for name in cited:
            (src / name).write_bytes(b"weights")
        (src / "lora" / "extra.safetensors").write_bytes(b"weights")
        for extra in ([], ["--allow-gguf"]):
            with self.subTest(extra=extra):
                code, out, err = run_main("--source", str(src), *extra)
                self.assertEqual((code, err), (0, ""))
                self.assertEqual(set(re.findall(r"(?m)^ADD (\S+) \(\d+ bytes\)$", out)), ALLOWED_ADDS)
                refused = set(re.findall(r"(?m)^REFUSE (\S+) \(", out))
                self.assertEqual(refused, {*cited, "lora/extra.safetensors"})
                self.assertRegex(out, r"(?m)^REFUSE assets/x\.safetensors \(pesos só entram em lora/")
                self.assertRegex(out, r"(?m)^REFUSE assets/x\.gguf \(GGUF")

    def test_only_the_adapter_and_the_modelfile_gguf_may_be_weights(self):
        src = make_source(self.tmp)
        named = hf_publish.FROM_RE.findall(MODELFILE)[0]
        (src / named).write_bytes(b"gguf")
        _, out, _ = run_main("--source", str(src), "--allow-gguf")
        added = set(re.findall(r"(?m)^ADD (\S+) \(\d+ bytes\)$", out))
        self.assertEqual(added, ALLOWED_ADDS | {named})

    def test_secret_inside_an_allowed_file_stops_the_run(self):
        src = make_source(self.tmp)
        with open(src / "Modelfile", "a", encoding="utf-8") as handle:
            handle.write(f"# {FAKE_TOKEN}\n")
        code, out, err = run_main("--source", str(src))
        self.assertEqual(code, 2)
        self.assertIn("Modelfile", err)
        self.assertNotIn(FAKE_TOKEN, out + err)
        self.assertNotIn("ADD", out)

    def test_card_with_a_number_not_in_benchmarks_is_refused(self):
        src = make_source(self.tmp)
        readme = (src / "README.md").read_text(encoding="utf-8")
        (src / "README.md").write_text(readme.replace("56/120 runs (46.7%)", "57/120 runs (47.5%)"),
                                       encoding="utf-8")
        code, out, err = run_main("--source", str(src))
        self.assertEqual((code, out), (2, ""))
        self.assertIn("benchmarks/live_colab_g4_bf16_n120.json", err)

    def test_missing_source_files_are_refused(self):
        src = make_source(self.tmp)
        (src / "LICENSE").unlink()
        code, _, err = run_main("--source", str(src))
        self.assertEqual(code, 2)
        self.assertIn("LICENSE", err)
        code, _, err = run_main("--source", str(self.tmp / "nope"))
        self.assertEqual(code, 2)
        self.assertIn("inexistente", err)


class SnapshotTest(TempCase):
    def test_snapshot_shows_orphans_and_never_delete_or_copy(self):
        src = make_source(self.tmp)
        snapshot = self.tmp / "remote.json"
        write_snapshot(snapshot)
        code, out, err = run_main("--source", str(src), "--remote-snapshot", str(snapshot))
        self.assertEqual((code, err), (0, ""))
        warned = set(re.findall(r"(?m)^AVISO órfão no Hub, não removido: (\S+?)(?::.*)?$", out))
        self.assertEqual(warned, ORPHANS)
        self.assertIn("ADD lora/adapter_config.json", out)
        self.assertIn("ADD Modelfile", out)
        self.assertNotRegex(out, r"(?m)^(DELETE|COPY|REMOVE|MOVE)\b")
        self.assertIn("MERGE generation_config.json", out)

    def test_bad_snapshot_is_refused(self):
        src = make_source(self.tmp)
        snapshot = self.tmp / "remote.json"
        snapshot.write_text('{"not": "a list"}')
        code, _, err = run_main("--source", str(src), "--remote-snapshot", str(snapshot))
        self.assertEqual(code, 2)
        self.assertIn("lista JSON", err)


class PublishTest(TempCase):
    def publish(self, remote=None, adapter=True, **kw):
        src = make_source(self.tmp, adapter=adapter)
        client = FakeClient(stale_remote() if remote is None else remote)
        tokens = []

        def factory(token):
            tokens.append(token)
            return client

        git = kw.pop("git", lambda *a: "c0ffee" if a[0] == "rev-parse" else "")
        code, out, err = run_main("--source", str(src), "--publish", env={"HF_TOKEN": FAKE_TOKEN},
                                  factory=factory, git=git, **kw)
        return code, out, err, client, tokens

    def test_refuses_without_token(self):
        src = make_source(self.tmp)
        git_calls = []
        for env in ({}, {"HF_TOKEN": ""}, {"HF_TOKEN": "   "}, {"OTHER": "x"}):
            with self.subTest(env=env):
                code, out, err = run_main("--source", str(src), "--publish", env=env,
                                          git=lambda *a: git_calls.append(a) or "")
                self.assertEqual((code, out), (2, ""))
                self.assertIn("HF_TOKEN", err)
        self.assertEqual(git_calls, [])

    def test_apply_is_an_alias_of_publish(self):
        src = make_source(self.tmp)
        code, out, err = run_main("--source", str(src), "--apply", env={})
        self.assertEqual((code, out), (2, ""))
        self.assertIn("HF_TOKEN", err)

    def test_one_commit_with_token(self):
        code, out, err, client, tokens = self.publish()
        self.assertEqual((code, err), (0, ""))
        self.assertEqual(tokens, [FAKE_TOKEN])
        self.assertEqual(client.calls, ["head_sha", "list_files", "read_file generation_config.json",
                                        "create_commit"])
        self.assertNotIn(FAKE_TOKEN, out + err)
        (commit,) = client.commits
        self.assertEqual((commit["parent"], commit["repo"]), ("abc123", hf_publish.REPO_ID))
        self.assertEqual(commit["description"], "https://github.com/simpletibr/simplicio-27b/commit/c0ffee")
        self.assertIn("https://huggingface.co/x/commit/1", out)
        ops = commit["ops"]
        self.assertEqual({type(op) for op in ops}, {hf_publish.Add})
        self.assertEqual({op.path for op in ops},
                         ALLOWED_ADDS | {"generation_config.json"})
        self.assertNotRegex(out, r"(?m)^(DELETE|COPY)\b")

    def test_publish_never_deletes_what_the_hub_has_outside_the_allowlist(self):
        remote = stale_remote()
        self.assertLessEqual({"model.safetensors", "vocab.json", "merges.txt", "special_tokens_map.json",
                              "added_tokens.json", "training_args.bin"}, set(remote))
        before = dict(remote)
        code, out, err, client, _ = self.publish(remote=remote)
        self.assertEqual((code, err), (0, ""))
        (commit,) = client.commits
        sent = {op.path for op in commit["ops"]}
        self.assertEqual({type(op) for op in commit["ops"]}, {hf_publish.Add})
        self.assertEqual(sent & ORPHANS, set())
        self.assertEqual(client.files, before)  # the fake Hub's tree was not touched
        warned = set(re.findall(r"(?m)^AVISO órfão no Hub, não removido: (\S+?)(?::.*)?$", out))
        self.assertEqual(warned, ORPHANS)
        self.assertNotIn("DELETE", out)
        self.assertNotIn("DELETE", " ".join(repr(op) for op in commit["ops"]))
        self.assertEqual([c for c in client.calls if "delete" in c.lower() or "copy" in c.lower()], [])

    def test_generation_config_takes_the_modelfile_sampling(self):
        _, _, _, client, _ = self.publish()
        gen = next(op for op in client.commits[0]["ops"]
                   if isinstance(op, hf_publish.Add) and op.path == "generation_config.json")
        self.assertEqual(json.loads(gen.payload), {
            "bos_token_id": 1, "do_sample": True, "temperature": 0.7, "top_p": 0.8, "top_k": 20,
            "min_p": 0.0, "presence_penalty": 1.5, "repetition_penalty": 1.0,
        })

    def test_local_adapter_goes_to_lora_and_the_root_copy_is_only_warned(self):
        _, out, _, client, _ = self.publish()
        added = {op.path: op for op in client.commits[0]["ops"]}
        self.assertTrue(added["lora/adapter_config.json"].payload.endswith("adapter_config.json"))
        self.assertIn("AVISO órfão no Hub, não removido: adapter_config.json: mova para lora/", out)
        self.assertIn("AVISO órfão no Hub, não removido: adapter_model.safetensors: mova para lora/", out)

    def test_no_commit_when_in_sync(self):
        src = make_source(self.tmp)
        scan = hf_publish.scan_source(src)
        gen = hf_publish.generation_config(GEN, scan.modelfile)
        remote = {p: f for p, f in stale_remote().items() if p not in ORPHANS and p != "generation_config.json"}
        remote["generation_config.json"] = regular("generation_config.json", gen)
        for path, entry in scan.entries.items():
            remote[path] = hf_publish.RemoteFile(path, entry.blob_sha1())
        remote["lora/adapter_model.safetensors"] = hf_publish.RemoteFile(
            "lora/adapter_model.safetensors", "0" * 40, scan.entries["lora/adapter_model.safetensors"].sha256())
        code, out, _, client, _ = self.publish(remote=remote, adapter=True)
        self.assertEqual(code, 0)
        self.assertEqual(out.strip().splitlines()[-1], "0 operações")
        self.assertNotIn("create_commit", client.calls)

    def test_dirty_tree_is_refused_before_the_client_exists(self):
        src = make_source(self.tmp)
        code, out, err = run_main("--source", str(src), "--publish", env={"HF_TOKEN": FAKE_TOKEN},
                                  git=lambda *a: " M README.md")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("working tree", err)

    def test_not_a_git_checkout_is_refused(self):
        src = make_source(self.tmp)
        with mock.patch.dict(os.environ, {"GIT_CEILING_DIRECTORIES": str(self.tmp)}):
            code, out, err = run_main("--source", str(src), "--publish", env={"HF_TOKEN": FAKE_TOKEN})
        self.assertEqual((code, out), (2, ""))
        self.assertIn("git status", err)

    def test_head_must_be_the_pushed_commit(self):
        src = make_source(self.tmp)
        git = lambda *a: {"status": "", "rev-parse": "c0ffee" if a[1] == "HEAD" else "bad1dea"}[a[0]]  # noqa: E731
        code, out, err = run_main("--source", str(src), "--publish", env={"HF_TOKEN": FAKE_TOKEN}, git=git)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("difere do branch remoto", err)

    def test_branch_without_upstream_is_refused(self):
        src = make_source(self.tmp)

        def git(*a):
            if a == ("rev-parse", "@{upstream}"):
                raise hf_publish.Refused("sem upstream")
            return "c0ffee" if a[0] == "rev-parse" else ""

        code, out, err = run_main("--source", str(src), "--publish", env={"HF_TOKEN": FAKE_TOKEN}, git=git)
        self.assertEqual((code, out), (2, ""))
        self.assertIn("upstream", err)

    def test_commit_url_comes_from_the_remote_branch(self):
        calls = []

        def git(*a):
            calls.append(a)
            return "feed123" if a[0] == "rev-parse" else ""

        _, _, _, client, _ = self.publish(git=git)
        self.assertIn(("rev-parse", "@{upstream}"), calls)
        self.assertTrue(client.commits[0]["description"].endswith("/commit/feed123"))

    def test_snapshot_and_publish_do_not_mix(self):
        src = make_source(self.tmp)
        code, _, err = run_main("--source", str(src), "--publish", "--remote-snapshot", "x.json",
                                env={"HF_TOKEN": FAKE_TOKEN})
        self.assertEqual(code, 2)
        self.assertIn("--remote-snapshot", err)

    def test_modelfile_gguf_must_exist_on_the_hub(self):
        remote = {p: f for p, f in stale_remote().items() if not p.endswith(".gguf")}
        code, _, err, client, _ = self.publish(remote=remote)
        self.assertEqual(code, 2)
        self.assertIn("plano inválido", err)
        self.assertNotIn("create_commit", client.calls)

    def test_adapter_missing_from_lora_is_refused_without_copying_it(self):
        code, out, err, client, _ = self.publish(adapter=False)  # the Hub has it only at the root
        self.assertEqual((code, out), (2, ""))
        self.assertIn("plano inválido", err)
        self.assertIn("lora/adapter_config.json", err)
        self.assertIn("baixe o adapter para lora/", err)
        self.assertNotIn("create_commit", client.calls)


class SamplingTest(unittest.TestCase):
    def test_sampling_comes_from_the_modelfile(self):
        modelfile = ("PARAMETER temperature 0.7\nPARAMETER top_p 0.8\nPARAMETER top_k 20\n"
                     "PARAMETER repeat_penalty 1.0\nPARAMETER num_ctx 40960\n")
        self.assertEqual(json.loads(hf_publish.generation_config(GEN, modelfile)),
                         {"bos_token_id": 1, "do_sample": True, "temperature": 0.7, "top_p": 0.8,
                          "top_k": 20, "repetition_penalty": 1.0})

    def test_modelfile_without_temperature_is_refused(self):
        with self.assertRaises(hf_publish.Refused):
            hf_publish.generation_config(GEN, "PARAMETER num_ctx 40960\n")

    def test_real_modelfile_has_sampling(self):
        cfg = json.loads(hf_publish.generation_config(GEN, MODELFILE))
        self.assertLessEqual({"temperature", "top_p", "top_k"}, set(cfg))
        self.assertNotEqual(cfg["temperature"], 1.0)  # the thinking-mode value the Hub has today


def fake_hub_module(files=None):
    """Stand-in for huggingface_hub. It has no delete or copy operation, and HfApi traps any call besides
    the four the script is allowed to make."""
    hub = types.ModuleType("huggingface_hub")
    hub.seen = {}

    class Op:
        def __init__(self, **kw):
            self.kw = kw

    class CommitOperationAdd(Op):
        pass

    class HfApi:
        def __init__(self, token=None):
            hub.seen["token"] = token

        def __getattr__(self, name):
            raise AssertionError(f"chamada inesperada ao Hub: {name}")

        def model_info(self, repo_id):
            return SimpleNamespace(sha="abc123")

        def list_repo_tree(self, repo_id, recursive=False, revision=None):
            if files is not None:
                return [SimpleNamespace(path=f.path, blob_id=f.blob_id,
                                        lfs=SimpleNamespace(sha256=f.lfs_sha256) if f.lfs_sha256 else None)
                        for f in files.values()]
            return [SimpleNamespace(path="a.txt", blob_id="b" * 40, lfs=None),
                    SimpleNamespace(path="w.bin", blob_id="c" * 40, lfs=SimpleNamespace(sha256="d" * 64)),
                    SimpleNamespace(path="lora", tree_id="t")]  # a folder: no blob_id

        def hf_hub_download(self, repo_id, filename, revision=None):
            path = Path(tempfile.mkdtemp()) / filename
            path.write_bytes(GEN if files is not None else b"{}")
            return str(path)

        def create_commit(self, repo_id, operations, parent_commit, commit_message, commit_description):
            hub.seen.update(operations=operations, parent=parent_commit, message=commit_message,
                            description=commit_description)
            return SimpleNamespace(commit_url="https://huggingface.co/x/commit/9")

    hub.CommitOperationAdd, hub.HfApi = CommitOperationAdd, HfApi
    return hub


class HubClientTest(unittest.TestCase):
    def test_hub_client_maps_the_plan_to_hub_operations(self):
        hub = fake_hub_module()
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}):
            client = hf_publish.HubClient(FAKE_TOKEN)
            self.assertEqual(hub.seen["token"], FAKE_TOKEN)
            self.assertEqual(client.head_sha("x/y"), "abc123")
            self.assertEqual(client.list_files("x/y", "abc123"), {
                "a.txt": hf_publish.RemoteFile("a.txt", "b" * 40, None),
                "w.bin": hf_publish.RemoteFile("w.bin", "c" * 40, "d" * 64),
            })
            self.assertEqual(client.read_file("x/y", "generation_config.json", "abc123"), b"{}")
            url = client.create_commit("x/y", [hf_publish.Add("README.md", b"card", 4)], "abc123", "msg", "desc")
        self.assertEqual(url, "https://huggingface.co/x/commit/9")
        (add,) = hub.seen["operations"]
        self.assertEqual(add.kw, {"path_in_repo": "README.md", "path_or_fileobj": b"card"})
        self.assertEqual((hub.seen["parent"], hub.seen["message"], hub.seen["description"]),
                         ("abc123", "msg", "desc"))

    def test_publish_through_the_hub_client_only_adds(self):
        hub = fake_hub_module(stale_remote())
        src = make_source(Path(tempfile.mkdtemp()))
        self.addCleanup(shutil.rmtree, src.parent, ignore_errors=True)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = hf_publish.main(["--source", str(src), "--publish"], env={"HF_TOKEN": FAKE_TOKEN},
                                   git=lambda *a: "c0ffee" if a[0] == "rev-parse" else "")
        self.assertEqual((code, err.getvalue()), (0, ""))
        operations = hub.seen["operations"]
        self.assertEqual({type(op).__name__ for op in operations}, {"CommitOperationAdd"})
        self.assertEqual({op.kw["path_in_repo"] for op in operations} & ORPHANS, set())
        self.assertIn("model.safetensors", out.getvalue())  # reported as an orphan
        self.assertNotIn("DELETE", out.getvalue())
        self.assertNotIn(FAKE_TOKEN, out.getvalue())

    def test_missing_library_is_a_clean_refusal(self):
        with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
            with self.assertRaises(hf_publish.Refused) as ctx:
                hf_publish.HubClient(FAKE_TOKEN)
        self.assertNotIn(FAKE_TOKEN, str(ctx.exception))

    def test_default_factory_is_the_hub_client(self):
        hub = fake_hub_module()
        src = make_source(Path(tempfile.mkdtemp()))
        self.addCleanup(shutil.rmtree, src.parent, ignore_errors=True)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = hf_publish.main(["--source", str(src), "--publish"], env={"HF_TOKEN": FAKE_TOKEN},
                                   git=lambda *a: "")
        # The stand-in Hub has no config.json: the plan is invalid, so nothing is committed.
        self.assertEqual(code, 2)
        self.assertEqual(hub.seen["token"], FAKE_TOKEN)
        self.assertNotIn("operations", hub.seen)
        self.assertNotIn(FAKE_TOKEN, out.getvalue() + err.getvalue())


if __name__ == "__main__":
    unittest.main()
