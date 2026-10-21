"""The orchestrator of the rented day:  ``python -m decide.rental.run``.

By default it only PRINTS: the commands of the day, the estimate (UNVERIFIED) and the preflight verdict.  Nothing is downloaded,
trained or uploaded without ``--execute``.

    python -m decide.rental.run --profile econ_b200                                  # the plan and the estimate
    python -m decide.rental.run --profile econ_b200 --execute --gt-report report.json --public-requests suite.jsonl

The day is eleven phases (``state.PHASES``).  Each one is skipped when it finished and its outputs are unchanged, so the command
is simply run again after a crash, a preemption or a stop at a gate.  Every finished phase is copied to the private work repo
(``rental.hub.work_repo``), so on a new machine with the same ``--workdir`` the state and the outputs come back before anything
is recomputed.  What the work repo does NOT keep is the training state inside a seed (``resume.pt``): a machine lost in the middle of
a seed loses at most that seed's progress since the last ``resume.pt`` of the DISK, not of the Hub.

Gates stop the day, they never warn: label coverage (F3) before any GPU time is spent, and G4 before the public round.  The
thinker-trace step needs a GT report that passed.  Every upload goes to a private repo.

A gate cannot be walked around.  Every phase names what it needs (``NEEDS``): that must be finished, unchanged on the disk, and
still the thing the phase was built on.  So ``--only train ...`` after a day that stopped at the label gate is refused, a
``targets`` folder edited after its gate is not trusted, and what G4 measured is tied to the BYTES of the checkpoint (the digest
of every byte, ``state.digest_tree``): a retrain or a hand edit after G4 makes G4 run again, on results set aside, before the
public round or the upload.  One orchestrator at a time works in a work folder (``state/day.lock``).

Exit codes: 0 the day is complete (or the dry run printed); 1 an error (the state says where; run again); 2 a usage or
configuration error; 3 a gate stopped the day.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from decide.data import proxy_private as PP
from decide.data import schema as S

from . import estimate as E
from . import gates as G
from . import hub as H
from . import inputs as IN
from . import kit as K
from . import plan as PL
from . import preflight as PF
from . import report as RP
from . import traces as TR
from .config import DEFAULT_PATH, load_rental_config
from .profile import Profile, load_profile
from . import state as ST
from .state import PHASES, DayState, StateError

try:                                         # POSIX; the rented day runs on a Linux box (torch.distributed.run), a laptop rehearsal on another OS has no lock
    import fcntl
except ImportError:                          # pragma: no cover - not on Linux
    fcntl = None                             # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[2]
EXIT_OK, EXIT_ERROR, EXIT_USAGE, EXIT_STOPPED = 0, 1, 2, 3
#: The phases whose outputs are copied to the work repo.  ``fetch`` is not: its outputs are downloads that the Hub still has.
NOT_BACKED_UP = ("preflight", "fetch", "publish")
#: What a phase needs before it may run: the phases that must be settled (finished or skipped, their outputs unchanged on the disk,
#: and not due to be decided again), each optionally followed by ``:name,name`` for the outputs it is built on.  The fingerprints of
#: those outputs are written to the state when the phase finishes (``built_on``); if they differ later, the phase is stale.
NEEDS: Dict[str, Tuple[str, ...]] = {
    "preflight": (),
    "fetch": (),
    "targets": ("fetch:corpus,labels",),
    "traces": ("fetch:corpus", "targets"),
    "targets_traces": ("fetch:corpus,labels", "targets", "traces"),
    "train": ("fetch:start", "targets", "traces", "targets_traces"),
    "soup": ("train",),
    "calibrate": ("targets", "targets_traces", "train", "soup"),
    "g4": ("fetch:proxy,start", "calibrate:final"),
    "public_round": ("g4",),
    "publish": ("g4", "public_round"),
}
#: The only phases that may be settled by being skipped.
MAY_BE_SKIPPED = ("traces", "targets_traces", "soup", "public_round")
HUB_TOKEN_NAMES = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN")


class DayError(RuntimeError):
    """A phase failed (the state records which and why) or the day cannot start."""


class _Stop(Exception):
    """A gate said no."""


class _Skip(Exception):
    """A phase does not apply."""


@dataclass(frozen=True)
class DayResult:
    status: str                       # complete | partial | stopped
    stopped_at: Optional[str]
    reason: Optional[str]
    phases: Dict[str, str]


# --------------------------------------------------------------------------- child processes

def child_env(cfg: Mapping[str, Any], environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The environment of a step: no Hub token, no thinker key (the trainer needs neither), offline, the repo on the path."""
    env = dict(os.environ if environ is None else environ)
    for name in (cfg["hub"]["token_env"], cfg["thinker"]["api_key_env"], *HUB_TOKEN_NAMES):
        env.pop(name, None)
    env["PYTHONUNBUFFERED"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["PYTHONPATH"] = str(ROOT) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    return env


class SubprocessExecutor:
    """Runs a step as a child process in the repo, its output going to the step's log and to ``log``."""

    def __init__(self, log: Callable[[str], None] = print) -> None:
        self.log = log

    def run(self, argv: Sequence[str], *, log_path: "str | os.PathLike[str]", env: Mapping[str, str]) -> int:
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "ab") as handle:
            try:
                child = subprocess.Popen(list(argv), cwd=ROOT, env=dict(env), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except OSError as exc:
                handle.write(f"could not start {argv[0]}: {exc}\n".encode("utf-8"))
                return 127
            assert child.stdout is not None
            with child.stdout:
                for raw in child.stdout:
                    handle.write(raw)
                    handle.flush()
                    self.log(raw.decode("utf-8", "replace").rstrip())
            return child.wait()


# --------------------------------------------------------------------------- the day

def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")).hexdigest()


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_name(path.name + ".tmp")
    pending.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(pending, path)


class Day:
    def __init__(self, *, profile: Profile, cfg: Mapping[str, Any], workdir: "str | os.PathLike[str]", hub: H.Hub, kit: Any, executor: Any,
                 thinker: Any = None, gt_report: "str | os.PathLike[str] | None" = None, public_requests: "str | os.PathLike[str] | None" = None,
                 probes: Optional[PF.Probes] = None, allow_cpu: bool = False, ignore_warnings: Sequence[str] = (),
                 expected_start: Optional[Tuple[int, Optional[int]]] = IN.EXPECTED_START, log: Callable[[str], None] = print,
                 clock: Callable[[], float] = time.time, python: Optional[str] = None, thinker_tps: Optional[float] = None) -> None:
        self.profile, self.cfg, self.hub, self.kit, self.executor = profile, dict(cfg), hub, kit, executor
        self.thinker, self.gt_report, self.public_requests = thinker, gt_report, public_requests
        self.probes = probes or PF.system_probes()
        self.allow_cpu, self.ignore_warnings, self.expected_start = allow_cpu, tuple(ignore_warnings), expected_start
        self.log, self.clock, self.python, self.thinker_tps = log, clock, python, thinker_tps
        self.paths = PL.layout(workdir, self.cfg)
        self.identity = {"profile": profile.digest, "config": _digest({k: v for k, v in self.cfg.items() if k not in ("workdir", "public")})}
        self._state: Optional[DayState] = None
        self._preflight: Dict[str, Any] = {}
        self._backup_warnings: List[str] = []
        self._digests: Dict[str, str] = {}

    # ------------------------------------------------------------------ small things

    @property
    def state(self) -> DayState:
        if self._state is None:
            self._state = DayState(self.paths.state, identity=self.identity, clock=self.clock, root=self.paths.workdir)
        return self._state

    @contextlib.contextmanager
    def _exclusive(self):
        """One orchestrator at a time per work folder: two of them would train into the same folders and race on the state.

        A ``flock`` on ``state/day.lock``: the kernel lets go of it when the process dies, so a lock left behind by a crash is simply taken over."""
        lock = self.paths.state / "day.lock"
        lock.parent.mkdir(parents=True, exist_ok=True)
        handle = None
        for _ in range(3):
            handle = os.open(lock, os.O_RDWR | os.O_CREAT, 0o644)
            if fcntl is not None:
                try:
                    fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except OSError:
                    try:
                        held = os.read(handle, 4096).decode("utf-8", "replace").strip() or "unknown"
                    finally:
                        os.close(handle)
                    raise DayError(f"another orchestrator is working in {self.paths.workdir} ({held}); two of them would train into the same folders. "
                                   f"Wait for it to end, or stop it, and run again (lock: {lock})") from None
            try:
                same_file = os.fstat(handle).st_ino == os.stat(lock).st_ino          # the holder before us may have removed it while we were opening it
            except FileNotFoundError:
                same_file = False
            if same_file:
                break
            os.close(handle)
            handle = None
        if handle is None:
            raise DayError(f"could not take the lock {lock}; run again")
        try:
            os.ftruncate(handle, 0)
            os.write(handle, json.dumps({"pid": os.getpid(), "host": socket.gethostname(), "started": self.clock()}).encode("utf-8"))
            yield
        finally:
            with contextlib.suppress(OSError):
                lock.unlink()
            os.close(handle)

    def _say(self, text: str) -> None:
        self.paths.logs.mkdir(parents=True, exist_ok=True)
        with open(self.paths.logs / "day.log", "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {text}\n")
        self.log(text)

    def _execute(self, name: str, argv: List[str]) -> None:
        self._say(f"run {name}: {' '.join(argv[:6])} ...")
        code = self.executor.run(argv, log_path=self.paths.logs / f"{name}.log", env=child_env(self.cfg))
        if code != 0:
            raise DayError(f"{name} failed with exit code {code}; see {self.paths.logs / (name + '.log')}")

    def _evidence(self) -> Dict[str, Any]:
        return G.gt_evidence(self.gt_report, thinker=self.cfg["thinker"])

    def _public_requests(self) -> Optional[Path]:
        given = self.public_requests or self.cfg["public"].get("requests")
        return Path(given) if given else None

    # ------------------------------------------------------------------ the printed plan

    def plan_text(self) -> str:
        evidence = self._evidence()
        thinker = evidence["model"].split("/")[-1] if evidence["enabled"] else None
        check = PF.preflight(self.profile, self.cfg, self.paths.workdir, probes=self.probes, allow_cpu=self.allow_cpu, ignore_warnings=self.ignore_warnings,
                             public_requests=self._public_requests())
        lines = [f"rented day, profile {self.profile.name} ({self.profile.title}), workdir {self.paths.workdir}", "",
                 *PL.describe(self.profile, self.cfg, self.paths, thinker=thinker, python=self.python), "",
                 f"thinker traces: {'ON, ' + evidence['model'] + ' at revision ' + evidence['revision'][:7] if evidence['enabled'] else 'OFF (' + '; '.join(evidence['reasons']) + ')'}", "",
                 E.render_text(E.estimate(self.profile, traces=evidence["enabled"], thinker_tokens_per_second=self.thinker_tps)), "",
                 f"preflight on this machine: {'OK' if check['ok'] else str(len(check['problems'])) + ' problem(s)'}"]
        lines += [f"  problem: {text}" for text in check["problems"]] + [f"  warning: {text}" for text in check["warnings"]]
        lines.append("this was a dry run: nothing was downloaded, trained or uploaded (--execute does it)")
        return "\n".join(lines)

    # ------------------------------------------------------------------ the run

    def run(self, *, only: Optional[Sequence[str]] = None, until: Optional[str] = None) -> DayResult:
        for name in (*(only or ()), *([until] if until else [])):
            if name not in PHASES:
                raise ValueError(f"unknown phase {name!r}; the phases are {', '.join(PHASES)}")
        with self._exclusive():
            return self._run(only=only, until=until)

    def _run(self, *, only: Optional[Sequence[str]], until: Optional[str]) -> DayResult:
        check = PF.preflight(self.profile, self.cfg, self.paths.workdir, probes=self.probes, allow_cpu=self.allow_cpu, ignore_warnings=self.ignore_warnings,
                             public_requests=self._public_requests())
        if not check["ok"]:
            raise DayError("preflight failed, nothing was downloaded or started: " + "; ".join(check["problems"]))
        self._preflight = check
        self._digests = {}
        self._restore()
        state = self.state
        stop: Optional[_Stop] = None
        stopped_at: Optional[str] = None
        try:
            for phase in PHASES:
                if only and phase not in only:
                    continue
                stale = self._stale(phase) if phase != "preflight" else None
                if phase != "preflight" and state.reusable(phase) and stale is None:
                    self._say(f"{phase}: {state.status(phase)} already, skipped")
                else:
                    if stale is not None:
                        self._say(f"{phase}: {state.status(phase)} before, but {stale}")
                    try:
                        self._phase(phase)
                    except _Stop as halted:
                        stop, stopped_at = halted, phase
                        break
                if until == phase:
                    break
        finally:
            self._write_report()
        statuses = {phase: state.status(phase) for phase in PHASES}
        if stop is not None:
            return DayResult("stopped", stopped_at, str(stop), statuses)
        finished = all(status in ("done", "skipped") for status in statuses.values())
        return DayResult("complete" if finished else "partial", None, None, statuses)

    def _stale(self, phase: str) -> Optional[str]:
        """Why a phase that the state calls settled has to be decided again: the owner has since given the day what it lacked (``None``: it stays).

        A phase that was skipped for a missing input is not a finished phase.  The public round (no requests file) and the thinker traces
        (no passing GT report) are decided again as soon as the input exists; the upload is done again when the public round finished
        after it.  The traces cannot be made after the model trained without them: that day is over, and the message says so."""
        state, status = self.state, self.state.status(phase)
        if status == "done" and NEEDS.get(phase):
            recorded, now = state.built_on(phase), self._inputs(phase)
            if recorded != now:
                changed = sorted(name for name in {*recorded, *now} if recorded.get(name) != now.get(name))
                return (f"what it was built on changed ({', '.join(changed)})" if recorded
                        else "the state has no record of what it was built on, so it cannot be told that the inputs are the same")
        if phase == "g4" and status == "done":
            problem = self._measured_problem()
            if problem:
                return problem
        if phase == "public_round" and status == "skipped" and self._public_requests() is not None:
            return "the public requests file is given now"
        if phase == "traces" and status == "skipped" and self._evidence()["enabled"]:
            late = [name for name in ("train", "soup", "calibrate", "g4") if state.status(name) != "pending"]
            if late:
                raise DayError(f"the GT report passes now, but the traces were skipped in an earlier session and {', '.join(late)} already ran without them: "
                               "the model that exists was trained without the thinker's reasoning.  Use a new --workdir to train with them, "
                               "or run without --gt-report to go on with this model")
            return "the GT report passes now"
        if phase == "targets_traces" and status == "skipped" and state.status("traces") == "done":
            return "the traces were made in this session"
        if phase == "publish" and status == "done" and state.status("public_round") == "done":
            if "public round" not in (state.record("publish").get("detail") or {}).get("uploaded", []):
                return "the public round finished after the upload, and its results were not sent"
        return None

    # ------------------------------------------------------------------ what a phase needs

    def _inputs(self, phase: str) -> Dict[str, Any]:
        """The fingerprints, as the state recorded them, of the outputs the phase is built on (``{dep: status}`` for a dep that has none to give)."""
        found: Dict[str, Any] = {}
        for need in NEEDS.get(phase, ()):
            dep, _, names = need.partition(":")
            status = self.state.status(dep)
            if status != "done":
                found[dep] = status
                continue
            outputs = self.state.record(dep).get("outputs", {})
            wanted = names.split(",") if names else sorted(outputs)
            found[dep] = {name: outputs[name]["fingerprint"] for name in wanted if name in outputs}
        return found

    def _unsettled(self, dep: str) -> Optional[str]:
        """Why ``dep`` cannot be built on now (``None``: it is finished or legitimately skipped, unchanged on the disk, and not due to be decided again)."""
        status = self.state.status(dep)
        if status == "skipped" and dep in MAY_BE_SKIPPED:
            pass
        elif status != "done":
            reason = self.state.record(dep).get("reason")
            return f"{dep} is {status}" + (f" ({reason})" if reason else "") + ": it has to finish first"
        why = self.state.why_not_reusable(dep)
        if why:
            return f"{why}, so {dep} cannot be trusted"
        stale = self._stale(dep)
        if stale:
            return f"{dep} has to be decided again first: {stale}"
        return None

    def _require(self, phase: str) -> Dict[str, Any]:
        """Refuse a phase whose inputs are not settled; the inputs it will be built on otherwise.  Nothing is written when it refuses.

        This is what keeps ``--only`` from going around a gate: the phases it leaves out are checked, not assumed."""
        for need in NEEDS.get(phase, ()):
            problem = self._unsettled(need.partition(":")[0])
            if problem:
                raise DayError(f"{phase} cannot run now: {problem}.  Run the earlier phases first (leave out --only, or name them in it); the gates are asked again, never skipped")
        return self._inputs(phase)

    def _final_digest(self) -> Optional[str]:
        """The digest of every byte of the checkpoint to publish (``None``: it is not there).  One pass over the files, once per phase."""
        key = str(self.paths.final)
        if key not in self._digests:
            if not self.paths.final.is_dir():
                return None
            self._say(f"reading every byte of {self.paths.final} to tie G4 and the upload to it (one pass over the checkpoint; the time is UNVERIFIED)")
            self._digests[key] = ST.digest_tree(self.paths.final)
        return self._digests[key]

    def _start_fingerprint(self) -> Optional[str]:
        return ST.fingerprint(self.paths.start) if self.paths.start.exists() else None

    def _baseline_identity(self) -> str:
        """What the baseline run was made of: the bytes of the start checkpoint when it is the baseline, else the pinned repo revision."""
        baseline = self.cfg["gates"]["baseline"]
        if baseline == self.cfg["start"]:
            return f"start:{self._start_fingerprint()}"
        return f"hub:{baseline['repo']}@{baseline['revision']}"

    def _measured_problem(self) -> Optional[str]:
        """Why what is on the disk is not what G4 measured (``None``: it is).  The public round and the upload ask this, so they never run on another checkpoint."""
        record = self.state.record("g4")
        if record.get("status") != "done":
            return "G4 has not passed"
        detail = record.get("detail") or {}
        now = self._final_digest()
        if now is None:
            return f"{self.paths.final} is gone, so it cannot be told that it is the checkpoint G4 measured"
        if detail.get("final_digest") != now:
            return (f"the checkpoint in {self.paths.final} is not the one G4 measured (its digest is {now[:12]}, G4 measured {str(detail.get('final_digest'))[:12]}): "
                    "it was trained, calibrated or edited after G4")
        if detail.get("start_fingerprint") != self._start_fingerprint():
            return f"the start checkpoint in {self.paths.start} is not the one the baseline of G4 was measured on"
        return None

    def _results_for(self, out: Path, marks: Path, key: str, identity: str) -> None:
        """The kit resumes a run from its output folder; it must only find one that was made for exactly these bytes, else that folder is set aside (never deleted)."""
        known = json.loads(marks.read_text(encoding="utf-8")) if marks.is_file() else {}
        if out.exists() and known.get(key) != identity:
            number = 1
            while out.with_name(f"{out.name}.stale-{number}").exists():
                number += 1
            aside = out.with_name(f"{out.name}.stale-{number}")
            out.rename(aside)
            self._say(f"the results in {out} were not made for these bytes; they are set aside as {aside.name} and the run starts clean")
        known[key] = identity
        _write_json(marks, known)

    def _phase(self, phase: str) -> None:
        state = self.state
        built_on = self._require(phase)
        state.begin(phase)
        self._say(f"{phase}: begin")
        try:
            outputs, detail = getattr(self, f"_do_{phase}")()
        except _Skip as skipped:
            state.skip(phase, str(skipped))
            self._say(f"{phase}: skipped ({skipped})")
            return
        except _Stop as halted:
            state.stop(phase, str(halted))
            self._say(f"{phase}: STOPPED ({halted})")
            raise
        except KeyboardInterrupt:
            state.fail(phase, "interrupted")
            raise
        except Exception as exc:                  # noqa: BLE001 - the state keeps the reason, the caller gets one error that names the phase
            state.fail(phase, f"{type(exc).__name__}: {exc}")
            self._say(f"{phase}: FAILED ({exc})")
            raise DayError(f"{phase} failed: {exc}") from exc
        state.done(phase, outputs=outputs, detail=detail, built_on=built_on)
        self._digests = {}                        # a phase may have changed the bytes (calibrate does)
        self._say(f"{phase}: done in {state.seconds(phase):.1f} s")
        self._backup(phase)

    # ------------------------------------------------------------------ phases

    def _do_preflight(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        for text in self._preflight.get("warnings", []):
            self._say(f"preflight warning: {text}")
        return {}, {"facts": self._preflight.get("facts"), "warnings": self._preflight.get("warnings")}

    def _do_fetch(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        cfg, paths = self.cfg, self.paths
        inputs = cfg["inputs"]
        self.hub.download(cfg["hub"]["data_repo"], paths.data, kind="dataset", patterns=[f"{inputs[key]}/*" for key in ("corpus", "labels", "proxy")])
        facts: Dict[str, Any] = {}
        for key in ("corpus", "proxy", "labels"):
            try:
                facts[key] = IN.verify_dataset(getattr(paths, key), name=key, manifest_required=True)
            except S.ManifestError as exc:
                raise DayError(f"{key}: {exc}. A folder the day reads needs a manifest.json: seal it before the upload, e.g. "
                               f"`python -c \"from decide.data import schema as S; S.seal_directory('<folder>', kind='{key}')\"` (RUNBOOK, session 4)") from None
        facts["corpus"]["screened"] = IN.check_corpus_screened(paths.corpus)
        facts["proxy"]["rows_checked"] = rows = IN.proxy_rows(paths.proxy)
        if rows["declared"] != rows["read"]:
            raise DayError(f"the manifest of the proxy declares {rows['declared']} rows and its parts hold {rows['read']}: G4 scores the whole proxy, so it has to be the one that was sealed")
        facts["proxy"]["screened"] = IN.check_proxy_screened(paths.proxy, paths.corpus)
        if rows["with_gold"] < cfg["gates"]["g4_min_rows"]:
            raise DayError(f"the proxy has {rows['with_gold']} rows with a gold answer and G4 needs at least {cfg['gates']['g4_min_rows']} "
                           "(gates.g4_min_rows): the day would train for hours and then end INVALID at G4.  Build a bigger proxy, or lower the floor in the config if you decide a smaller proxy is enough")
        for item in facts.values():
            if item.get("warning"):
                self._say(f"warning: {item['warning']}")
        self.hub.download(cfg["start"]["repo"], paths.start, kind="model", revision=cfg["start"]["revision"])
        facts["start"] = IN.check_start_checkpoint(paths.start, expected=self.expected_start)
        return {"corpus": paths.corpus, "labels": paths.labels, "proxy": paths.proxy, "start": paths.start}, facts

    def _build_targets(self, thinker: Optional[str]) -> Dict[str, Any]:
        self._execute("targets" if thinker is None else "targets-traces", PL.targets_command(self.paths, thinker=thinker, python=self.python))
        out = self.paths.targets if thinker is None else self.paths.targets_traces
        return json.loads((out / "report.json").read_text(encoding="utf-8"))

    def _do_targets(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        report = self._build_targets(None)
        minimum = self.cfg["gates"]["label_coverage"]
        counts = G.teacher_coverage(self.paths.targets, report)
        coverage = G.label_coverage(counts, minimum=minimum)
        if not coverage["passed"]:
            raise _Stop(f"label coverage {coverage['coverage']:.3f} is below {minimum}: only {counts['rows_out']} of {counts['rows_in']} corpus rows are labeled "
                        "by a teacher; finish the labeling (F3) before paying for GPU time")
        return {"targets": self.paths.targets}, {"coverage": coverage["coverage"], "minimum": minimum, "rows_in": counts["rows_in"], "rows_labeled": counts["rows_out"],
                                                  "rows_with_a_target": report["rows_out"], "teachers": report.get("teachers_seen")}

    def _thinker_client(self, model: str) -> Any:
        if self.thinker is not None:
            return self.thinker
        settings = self.cfg["thinker"]
        if not settings.get("base_url"):
            raise DayError("thinker.base_url is not set: the GT passed, so the traces are wanted, but there is no thinker endpoint "
                           "(serve the model with vLLM on this machine and set it, or set it to a hosted endpoint)")
        key = os.environ.get(settings["api_key_env"]) if settings.get("api_key_env") else None
        return TR.ThinkerClient(settings["base_url"], model, api_key=key)

    def _do_traces(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        evidence = self._evidence()
        if not evidence["enabled"]:
            raise _Skip("; ".join(evidence["reasons"]))
        settings = self.cfg["thinker"]
        name = evidence["model"].split("/")[-1]
        report = TR.generate_traces(self.paths.corpus, self.paths.traces, client=self._thinker_client(evidence["model"]), thinker=name,
                                    budget_per_question=settings["budget_per_question"], max_tokens_cap=settings["max_tokens_cap"],
                                    concurrency=settings["concurrency"], log=self._say)
        if not report["complete"]:
            raise DayError(f"the traces are incomplete ({report['errors']} calls failed); run again, the finished examples are kept")
        if report["written"] + report["already_done"] == 0:
            raise DayError("the thinker produced no trace at all (no example came back with thinking); check the endpoint and that thinking is on")
        return {"traces": self.paths.traces}, {**report, "thinker": name, "evidence": evidence}

    def _do_targets_traces(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        state = self.state
        if state.status("traces") != "done":
            raise _Skip("no thinker traces: the traces phase did not run")
        report = self._build_targets(state.record("traces")["detail"]["thinker"])
        if not report.get("reasoning_rows"):
            raise DayError("no training example got a reasoning block from the traces")
        return {"targets_traces": self.paths.targets_traces}, {key: report.get(key) for key in ("rows_in", "rows_out", "reasoning_rows", "reasoning_truncated", "traces_unmatched", "thinker")}

    @staticmethod
    def _small_json(path: Path, limit: int = 20_000) -> Any:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        return data if len(text) <= limit else {"keys": sorted(data) if isinstance(data, dict) else None, "bytes": len(text)}

    def _do_train(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        with_traces = self.state.status("targets_traces") == "done"
        outputs: Dict[str, Path] = {}
        runs = []
        for index, seed in enumerate(PL.seeds(self.profile), 1):
            out = self.paths.seed_dir(index)
            final, summary = out / "final", out / "summary.json"
            if final.is_dir() and summary.is_file():
                self._say(f"train seed {seed}: finished earlier, kept")
            else:
                if final.exists():
                    raise DayError(f"{final} exists but {summary} does not: a save was interrupted. Move {final} away and run again "
                                   "(the run continues from its resume.pt)")
                self._execute(f"train-seed{index}", PL.train_command(self.profile, self.paths, index, with_traces=with_traces,
                                                                     resume=(out / "resume.pt").is_file(), python=self.python))
                if not (final.is_dir() and summary.is_file()):
                    raise DayError(f"the trainer ended without {final} and {summary}")
            outputs[f"seed{index}"] = final
            runs.append({"seed": seed, "summary": self._small_json(summary)})
        return outputs, {"seeds": runs, "with_traces": with_traces}

    def _do_soup(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        if self.profile.seeds == 1:
            raise _Skip("one seed: nothing to average")
        self._execute("soup", PL.soup_command(self.paths, self.profile.seeds, python=self.python))
        return {"soup": self.paths.soup}, {"inputs": self.profile.seeds}

    def _do_calibrate(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        paths = self.paths
        PL.stage(PL.candidate(paths, self.profile.seeds), paths.final)
        paths.calibrate.mkdir(parents=True, exist_ok=True)
        targets = paths.targets_traces if self.state.status("targets_traces") == "done" else paths.targets
        self._execute("calibrate", PL.temperature_command(paths, targets=targets, python=self.python))
        report = json.loads((paths.calibrate / "report.json").read_text(encoding="utf-8"))
        fields = report["fields"]
        return {"final": paths.final, "calibrate": paths.calibrate}, {"previous_temperature": report["previous_temperature"], "temperature": fields["temperature"],
                                                                      "by_type": fields.get("temperature_by_type"), "questions": fields["calibration"]["questions"]}

    def _baseline(self) -> K.ModelRef:
        baseline, start = self.cfg["gates"]["baseline"], self.cfg["start"]
        if baseline == start:
            return K.ModelRef(name="baseline", path=self.paths.start)          # the same bytes, already on this disk
        return K.ModelRef(name="baseline", repo=baseline["repo"], revision=baseline["revision"])

    def _do_g4(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        g4, gates = self.paths.g4, self.cfg["gates"]
        measured = {"final_digest": self._final_digest(), "start_fingerprint": self._start_fingerprint(), "baseline": self._baseline_identity()}
        if measured["final_digest"] is None:
            raise DayError(f"{self.paths.final} is not there: calibrate makes it, and G4 measures it")
        g4.mkdir(parents=True, exist_ok=True)
        requests = g4 / "requests.jsonl"
        G.proxy_to_requests(self.paths.proxy, requests)
        outs = {"baseline": g4 / "baseline", "new": g4 / "new"}
        identities = {"baseline": measured["baseline"], "new": measured["final_digest"]}
        for model in (self._baseline(), K.ModelRef(name="new", path=self.paths.final)):
            self._results_for(outs[model.name], g4 / "made-for.json", model.name, identities[model.name])
            info = self.kit.run(model, requests, outs[model.name])
            if info.get("status") != "complete":
                raise DayError(f"the kit ended the {model.name} run with status {info.get('status')!r}; run again, it resumes")
        tables = {name: out / "results.parquet" for name, out in outs.items()}
        for name, table in tables.items():
            if not table.is_file():
                raise DayError(f"the kit wrote no probability table for the {name} run ({table})")
        verdict = G.g4_verdict(G.proxy_requests(self.paths.proxy), list(PP.read_results(tables["new"])), list(PP.read_results(tables["baseline"])),
                               margin_points=gates["g4_margin_points"], scale=gates["proxy_scale"], resamples=gates["bootstrap"]["resamples"],
                               seed=gates["bootstrap"]["seed"], min_rows=gates["g4_min_rows"],
                               expected_rows=IN.proxy_rows(self.paths.proxy)["declared"])
        verdict["mode"] = "decider only: no thinker reasoning is shown to either model"
        verdict["measured"] = measured
        _write_json(g4 / "verdict.json", verdict)
        summary = {key: verdict[key] for key in ("status", "reasons", "notes", "delta_points", "ci95", "ci_above_zero", "margin_points", "questions", "scope")}
        summary["new_points"], summary["baseline_points"] = verdict["new"]["proxy_points"], verdict["baseline"]["proxy_points"]
        summary["final_digest"], summary["start_fingerprint"] = measured["final_digest"], measured["start_fingerprint"]
        if verdict["status"] != "PASS":
            raise _Stop(f"G4 {verdict['status']} ({', '.join(verdict['reasons'])}): the new checkpoint is {verdict['delta_points']} points over the baseline on the home proxy, "
                        f"the gate asks for {gates['g4_margin_points']}; the public round and the upload were not done")
        return {"g4": g4}, summary

    def _do_public_round(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        requests = self._public_requests()
        if requests is None:
            raise _Skip("no public requests file: pass --public-requests with the kit's requests file of the full public suite")
        if self.state.status("g4") != "done":
            raise DayError("the public round needs G4 passed first")
        problem = self._measured_problem()
        if problem:
            raise DayError(f"the public round needs G4 passed on this very checkpoint: {problem}")
        self._results_for(self.paths.public / "run", self.paths.public / "made-for.json", "run", self._final_digest() or "")
        info = self.kit.run(K.ModelRef(name="final", path=self.paths.final), requests, self.paths.public / "run")
        if info.get("status") != "complete":
            raise DayError(f"the kit ended the public round with status {info.get('status')!r}; run again, it resumes")
        seconds = info.get("seconds") or 0
        return {"public": self.paths.public}, {"rows": info.get("rows"), "tokens": info.get("tokens"), "seconds": seconds,
                                               "tokens_per_second": (info.get("tokens") or 0) / seconds if seconds else None}

    def _do_publish(self) -> Tuple[Dict[str, Path], Dict[str, Any]]:
        if self.state.status("g4") != "done":
            raise DayError("nothing is published before G4 passes")
        problem = self._measured_problem()
        if problem:
            raise DayError(f"nothing is published that G4 did not measure: {problem}")
        repo = self.cfg["hub"]["weights_repo"]
        self.hub.ensure_repo(repo, kind="model")
        revision = self.hub.upload(repo, self.paths.final, kind="model", patterns=H.CHECKPOINT_PATTERNS, message="weights from the rented day")
        uploaded = ["weights"]
        if self.state.status("public_round") == "done":
            self.hub.upload(repo, self.paths.public, kind="model", path_in_repo="public-round", patterns=["results.*", "*.json"], message="results of the public round")
            uploaded.append("public round")
        self._write_report(extra_phase_running="publish")
        self.hub.upload(repo, self.paths.report, kind="model", path_in_repo="day", patterns=["day-report.json"], message="report of the day")
        uploaded.append("report")
        return {}, {"repo": repo, "weights_revision": revision, "uploaded": uploaded, "final_digest": self._final_digest()}

    # ------------------------------------------------------------------ backup, restore, report

    def _backup(self, phase: str) -> None:
        if phase in NOT_BACKED_UP:
            return
        repo = self.cfg["hub"]["work_repo"]
        record = self.state.record(phase)
        try:
            self.hub.ensure_repo(repo, kind="model")
            for name, entry in record.get("outputs", {}).items():
                self.hub.upload(repo, entry["path"], kind="model", path_in_repo=f"outputs/{phase}/{name}", message=f"{phase}/{name}")
            self.hub.upload(repo, self.paths.state, kind="model", path_in_repo="state", patterns=["day.json"], message=f"state after {phase}")
        except Exception as exc:                  # noqa: BLE001 - a backup that fails is a risk to report, not a reason to waste the day
            text = f"backup of {phase} to {repo} failed ({type(exc).__name__}: {exc}); the day goes on, but a lost machine would lose this phase"
            self._backup_warnings.append(text)
            self._say(f"WARNING: {text}")

    def _restore(self) -> None:
        """Bring the state and the outputs of an earlier session back from the work repo.

        The state is read as INPUT: it is parsed and every path it names is checked (strictly inside the work folder, no ``..``, no
        backslash or drive letter, no link out, an output name that is one plain part) and its identity compared BEFORE a byte is
        moved.  The state file itself is put in place last, so a restore that stops halfway is finished by the next run."""
        paths, repo = self.paths, self.cfg["hub"]["work_repo"]
        if (paths.state / "day.json").exists():
            return
        try:
            if "state/day.json" not in self.hub.files(repo, kind="model", prefix="state"):
                return
        except H.HubError:
            return
        self._say(f"restoring the state and the outputs of an earlier session from {repo}")
        scratch = paths.workdir / ".restore"
        shutil.rmtree(scratch, ignore_errors=True)                # nothing left over from a stopped restore is trusted
        try:
            self.hub.download(repo, scratch, kind="model", patterns=["state/day.json"])
            downloaded = scratch / "state" / "day.json"
            try:
                text = downloaded.read_text(encoding="utf-8")
            except (OSError, ValueError) as exc:
                raise StateError(f"the state in {repo} cannot be read ({exc}); nothing was restored") from exc
            earlier = ST.parse_state(text, identity=self.identity, root=paths.workdir, label=f"the state in {repo}")
            for phase, record in earlier.items():
                if record.get("status") != "done" or phase in NOT_BACKED_UP:
                    continue
                for name, entry in record.get("outputs", {}).items():
                    destination = Path(entry["path"])
                    if destination.exists():
                        continue
                    self.hub.download(repo, scratch, kind="model", patterns=[f"outputs/{phase}/{name}/*"])
                    source = scratch / "outputs" / phase / name
                    if source.is_dir():
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        shutil.move(str(source), str(destination))
            paths.state.mkdir(parents=True, exist_ok=True)
            os.replace(downloaded, paths.state / "day.json")
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def _gates_summary(self) -> Dict[str, Any]:
        state, out = self.state, {}
        if state.status("targets") in ("done", "stopped"):
            record = state.record("targets")
            out["labels"] = record.get("detail") or {"stopped": record.get("reason")}
        traces = state.record("traces")
        out["gt"] = {"status": traces["status"], **({"evidence": (traces.get("detail") or {}).get("evidence")} if traces["status"] == "done" else {"reason": traces.get("reason")})}
        g4 = state.record("g4")
        if g4["status"] == "done":
            out["g4"] = g4["detail"]
        elif g4["status"] == "stopped":
            verdict = Path(self.paths.g4 / "verdict.json")
            out["g4"] = json.loads(verdict.read_text(encoding="utf-8")) if verdict.is_file() else {"status": "STOPPED", "reason": g4.get("reason")}
            out["g4"] = {key: out["g4"].get(key) for key in ("status", "reasons", "delta_points", "ci95", "margin_points")}
        return out

    def _write_report(self, extra_phase_running: Optional[str] = None) -> None:
        if self._state is None:
            return
        notes = list(self._backup_warnings) + ["every estimate is UNVERIFIED; the measured seconds and rows are from this machine"]
        report = RP.day_report(self.state, E.estimate(self.profile, traces=self.state.status("traces") == "done"), profile=self.profile,
                               gates=self._gates_summary(), notes=notes)
        _write_json(self.paths.report / "day-report.json", report)
        self._say("\n" + RP.render(report))


# --------------------------------------------------------------------------- command line

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m decide.rental.run", description="The rented GPU day (PLAN.md section 6). Prints the plan by default; --execute runs it.",
                                     epilog=__doc__.split("\n\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    add = parser.add_argument
    add("--config", default=str(DEFAULT_PATH), help="the rental config (default: decide/config.rental.yaml)")
    add("--profile", help="a shipped profile (econ_b200, full_8xh100) or a YAML path (default: the config's)")
    add("--workdir", help="where the day writes (default: the config's); use the same one to resume")
    add("--gt-report", help="report.json of decide.think.compare; with a PASS the thinker traces are made")
    add("--public-requests", help="the kit's requests file of the full public suite (without it the public round is skipped)")
    add("--hub", help="local:<folder> rehearses the day against a folder instead of Hugging Face")
    add("--execute", action="store_true", help="actually download, train, run the kit and upload (without it: print the plan and stop)")
    add("--only", nargs="+", choices=PHASES, help="run only these phases")
    add("--until", choices=PHASES, help="stop after this phase")
    add("--allow-cpu", action="store_true", help="do not require the profile's GPUs (rehearsals)")
    add("--ignore-warning", nargs="+", default=[], choices=PF.BLOCKING, help="run although the estimator says the profile will not fit; the owner's call, by name")
    add("--thinker-tps", type=float, help="measured tokens per second of the thinker, to estimate the trace step")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_rental_config(Path(args.config))["rental"]
        profile = load_profile(args.profile or cfg["profile"])
    except Exception as exc:                  # noqa: BLE001 - a bad config or profile is a usage error with the reason
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    workdir = args.workdir or cfg["workdir"]
    try:
        if args.hub and args.hub.startswith("local:"):
            hub: H.Hub = H.LocalHub(args.hub[len("local:"):])
        elif args.execute:
            hub = H.HfHub.from_env(cfg["hub"]["token_env"], endpoint=cfg["hub"]["endpoint"])
        else:
            hub = H.LocalHub(Path(workdir) / ".no-hub")          # a dry run never opens the Hub
        day = Day(profile=profile, cfg=cfg, workdir=workdir, hub=hub, kit=K.SdKit(), executor=SubprocessExecutor(), gt_report=args.gt_report,
                  public_requests=args.public_requests, allow_cpu=args.allow_cpu, ignore_warnings=args.ignore_warning, thinker_tps=args.thinker_tps)
        if not args.execute:
            print(day.plan_text())
            return EXIT_OK
        result = day.run(only=args.only, until=args.until)
    except (H.HubError, DayError, StateError, ValueError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_ERROR
    if result.status == "stopped":
        print(f"STOPPED at {result.stopped_at}: {result.reason}", file=sys.stderr)
        return EXIT_STOPPED
    print(f"day {result.status}")
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
